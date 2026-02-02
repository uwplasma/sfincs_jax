from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict

import h5py
import numpy as np

from .boozer_bc import read_boozer_bc_bracketing_surfaces
from .boozer_bc import read_boozer_bc_header
from .diagnostics import fsab_hat2 as fsab_hat2_jax
from .diagnostics import u_hat_np
from .diagnostics import vprime_hat as vprime_hat_jax
from .namelist import Namelist, read_sfincs_input
from .paths import resolve_existing_path
from .vmec_wout import _set_scale_factor, psi_a_hat_from_wout, read_vmec_wout, vmec_interpolation
from .v3 import V3Grids, geometry_from_namelist, grids_from_namelist


def _decode_if_bytes(x: Any) -> Any:
    if isinstance(x, (bytes, np.bytes_)):
        return x.decode("utf-8", errors="replace")
    if isinstance(x, np.ndarray) and x.dtype.kind in {"S", "O"}:
        # Common case in SFINCS: 1-element byte-string array.
        if x.size == 1:
            item = x.reshape(-1)[0]
            return _decode_if_bytes(item)
    return x


def read_sfincs_h5(path: Path) -> Dict[str, Any]:
    """Read a SFINCS `sfincsOutput.h5` file into memory.

    This is intended for small-to-moderate outputs used in tests and examples.
    """
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(str(path))

    out: Dict[str, Any] = {}
    with h5py.File(path, "r") as f:
        def visit(name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Dataset):
                v = obj[...]
                v = _decode_if_bytes(v)
                out[name] = v

        f.visititems(visit)
    return out


def _to_numpy_for_h5(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x
    # Handle JAX arrays without importing jax as a hard dependency at import time.
    if hasattr(x, "__array__"):
        return np.asarray(x)
    return x


def _fortran_h5_layout(x: Any) -> Any:
    """Mimic the layout of arrays written by v3's Fortran HDF5 output.

    SFINCS v3 writes arrays from Fortran (column-major). When the resulting HDF5 datasets
    are read back in Python (row-major), multi-dimensional arrays appear with axes
    effectively reversed compared to the (itheta, izeta, ...) indexing used in the Fortran
    source.

    To make `sfincs_jax` outputs comparable to Fortran `sfincsOutput.h5` *as read by Python*,
    we transpose arrays by reversing axes before writing.
    """
    arr = _to_numpy_for_h5(x)
    if not isinstance(arr, np.ndarray):
        return arr
    if arr.ndim <= 1:
        return arr
    axes = tuple(reversed(range(arr.ndim)))
    return np.ascontiguousarray(np.transpose(arr, axes=axes))


def write_sfincs_h5(
    *,
    path: Path,
    data: Dict[str, Any],
    fortran_layout: bool = True,
    overwrite: bool = True,
) -> None:
    """Write a minimal SFINCS-style HDF5 file (flat datasets at root)."""
    path = path.resolve()
    if path.exists() and not overwrite:
        raise FileExistsError(str(path))
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as f:
        for k, v in data.items():
            if v is None:
                continue
            vv = _to_numpy_for_h5(v)
            if fortran_layout:
                vv = _fortran_h5_layout(vv)
            f.create_dataset(k, data=vv)


def _as_1d_float(group: dict, key: str, *, default: float | None = None) -> np.ndarray:
    k = key.upper()
    if k not in group:
        if default is None:
            raise KeyError(key)
        return np.atleast_1d(np.asarray([default], dtype=np.float64))
    v = group[k]
    return np.atleast_1d(np.asarray(v, dtype=np.float64))


def _get_float(group: dict, key: str, default: float) -> float:
    v = group.get(key.upper(), default)
    if isinstance(v, list):
        v = v[0] if v else default
    return float(v)


def _get_int(group: dict, key: str, default: int) -> int:
    v = group.get(key.upper(), default)
    if isinstance(v, list):
        v = v[0] if v else default
    return int(v)


def _dphi_hat_dpsi_hat_from_er_geometry_scheme4(er: float) -> float:
    """Compute dPhiHat/dpsiHat from Er for geometryScheme=4 (v3 defaults).

    Matches `sfincs_jax.v3_fblock._dphi_hat_dpsi_hat_from_er_scheme4`, and v3's defaults:
    `inputRadialCoordinateForGradients=4` with rN forced to 0.5.
    """
    psi_a_hat = -0.384935
    a_hat = 0.5109
    psi_n = 0.25
    ddrhat2ddpsihat = a_hat / (2.0 * psi_a_hat * np.sqrt(psi_n))
    return float(ddrhat2ddpsihat * (-er))


def _fortran_logical(x: bool) -> np.int32:
    """Match v3's common logical representation in `sfincsOutput.h5`.

    Many v3 HDF5 outputs store logicals as int32 with `-1` for false and `+1` for true.
    """
    return np.int32(1 if bool(x) else -1)


def _resolve_equilibrium_file_from_namelist(*, nml: Namelist) -> Path:
    geom_params = nml.group("geometryParameters")
    equilibrium_file = geom_params.get("EQUILIBRIUMFILE", None)
    if equilibrium_file is None:
        raise ValueError("Missing geometryParameters.equilibriumFile")
    base_dir = nml.source_path.parent if nml.source_path is not None else None
    repo_root = Path(__file__).resolve().parents[1]
    extra = (repo_root / "tests" / "ref", repo_root / "sfincs_jax" / "data" / "equilibria")
    return resolve_existing_path(str(equilibrium_file), base_dir=base_dir, extra_search_dirs=extra).path


def _scheme4_radial_constants() -> tuple[float, float]:
    # Matches v3's built-in geometryScheme=4 W7-X simplified model.
    psi_a_hat = -0.384935
    a_hat = 0.5109
    return psi_a_hat, a_hat


def _set_input_radial_coordinate_wish(
    *,
    input_radial_coordinate: int,
    psi_a_hat: float,
    a_hat: float,
    psi_hat_wish_in: float,
    psi_n_wish_in: float,
    r_hat_wish_in: float,
    r_n_wish_in: float,
) -> tuple[float, float, float, float]:
    """Replicate v3 `radialCoordinates.setInputRadialCoordinateWish` for the wish coordinates."""
    if input_radial_coordinate == 0:
        psi_hat_wish = float(psi_hat_wish_in)
    elif input_radial_coordinate == 1:
        psi_hat_wish = float(psi_n_wish_in) * float(psi_a_hat)
    elif input_radial_coordinate == 2:
        psi_hat_wish = float(psi_a_hat) * float(r_hat_wish_in) * float(r_hat_wish_in) / (float(a_hat) * float(a_hat))
    elif input_radial_coordinate == 3:
        psi_hat_wish = float(r_n_wish_in) * float(r_n_wish_in) * float(psi_a_hat)
    else:
        raise ValueError(f"Invalid inputRadialCoordinate={input_radial_coordinate}")

    psi_n_wish = float(psi_hat_wish) / float(psi_a_hat)
    r_hat_wish = float(np.sqrt(float(a_hat) * float(a_hat) * float(psi_hat_wish) / float(psi_a_hat)))
    r_n_wish = float(np.sqrt(float(psi_hat_wish) / float(psi_a_hat)))
    return psi_hat_wish, psi_n_wish, r_hat_wish, r_n_wish


def _conversion_factors_to_from_dpsi_hat(*, psi_a_hat: float, a_hat: float, r_n: float) -> dict[str, float]:
    """Replicate v3 `radialCoordinates.setInputRadialCoordinate` derivative conversion factors."""
    psi_n = float(r_n) * float(r_n)
    root = float(np.sqrt(psi_n))
    ddpsi_n_to_ddpsi_hat = 1.0 / float(psi_a_hat)
    ddr_hat_to_ddpsi_hat = float(a_hat) / (2.0 * float(psi_a_hat) * root)
    ddr_n_to_ddpsi_hat = 1.0 / (2.0 * float(psi_a_hat) * root)

    ddpsi_hat_to_ddpsi_n = float(psi_a_hat)
    ddpsi_hat_to_ddr_hat = (2.0 * float(psi_a_hat) * root) / float(a_hat)
    ddpsi_hat_to_ddr_n = (2.0 * float(psi_a_hat) * root)

    return {
        "ddpsiN2ddpsiHat": ddpsi_n_to_ddpsi_hat,
        "ddrHat2ddpsiHat": ddr_hat_to_ddpsi_hat,
        "ddrN2ddpsiHat": ddr_n_to_ddpsi_hat,
        "ddpsiHat2ddpsiN": ddpsi_hat_to_ddpsi_n,
        "ddpsiHat2ddrHat": ddpsi_hat_to_ddr_hat,
        "ddpsiHat2ddrN": ddpsi_hat_to_ddr_n,
    }


def _evaluate_boozer_rzd_and_derivatives(
    *,
    theta: np.ndarray,
    zeta: np.ndarray,
    n_periods: int,
    m: np.ndarray,
    n: np.ndarray,
    parity: np.ndarray,
    r0: float,
    r_amp: np.ndarray,
    z_amp: np.ndarray,
    dz_amp: np.ndarray,
    dz_scale: float,
    chunk: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate (RHat, ZHat, Dz) and their (theta,zeta) derivatives on a Boozer grid.

    This mirrors the Fourier evaluation in v3 `geometry.F90` for geometryScheme 11/12,
    including Nyquist exclusions and the parity-dependent sine/cosine choices for ZHat and Dz.
    """
    theta1 = theta[None, :, None]  # (1,T,1)
    zeta1 = zeta[None, None, :]  # (1,1,Z)

    ntheta = int(theta.shape[0])
    nzeta = int(zeta.shape[0])
    m_max_grid = int(ntheta / 2.0)
    n_max_grid = int(nzeta / 2.0)

    if nzeta == 1:
        include = np.ones((int(m.shape[0]),), dtype=bool)
    else:
        include = (np.abs(n) <= n_max_grid) & (m <= m_max_grid)

    # Additional Nyquist exclusions for sine components (same logic as v3 `computeBHat`).
    is_sin = ~parity.astype(bool)
    if nzeta != 1 and np.any(is_sin):
        at_m_nyq = (m == 0) | (m.astype(np.float64) == (ntheta / 2.0))
        at_n_nyq = (n == 0) | (np.abs(n.astype(np.float64)) == (nzeta / 2.0))
        include = include & ~(is_sin & at_m_nyq & at_n_nyq)

    m = m[include].astype(np.float64)
    n = n[include].astype(np.float64)
    parity = parity[include].astype(bool)
    r_amp = r_amp[include].astype(np.float64)
    z_amp = z_amp[include].astype(np.float64)
    dz_amp = dz_amp[include].astype(np.float64) * float(dz_scale)

    r = np.full((ntheta, nzeta), float(r0), dtype=np.float64)
    dr_dtheta = np.zeros_like(r)
    dr_dzeta = np.zeros_like(r)

    z = np.zeros_like(r)
    dz_dtheta = np.zeros_like(r)
    dz_dzeta = np.zeros_like(r)

    dzeta = np.zeros_like(r)  # Dz field
    ddz_dtheta = np.zeros_like(r)
    ddz_dzeta = np.zeros_like(r)

    h = int(m.shape[0])
    for i0 in range(0, h, chunk):
        i1 = min(h, i0 + chunk)
        mc = m[i0:i1][:, None, None]
        nc = n[i0:i1][:, None, None]
        rc = r_amp[i0:i1][:, None, None]
        zc = z_amp[i0:i1][:, None, None]
        dzc = dz_amp[i0:i1][:, None, None]
        pc = parity[i0:i1][:, None, None]

        angle = mc * theta1 - float(n_periods) * nc * zeta1
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        # R uses the same basis as BHat (cos for parity=True, sin for parity=False).
        basis_r = np.where(pc, cos_a, sin_a)
        r = r + np.sum(rc * basis_r, axis=0)

        dtheta_basis_r = np.where(pc, -mc * sin_a, mc * cos_a)
        dr_dtheta = dr_dtheta + np.sum(rc * dtheta_basis_r, axis=0)

        dzeta_factor = float(n_periods) * nc
        dzeta_basis_r = np.where(pc, dzeta_factor * sin_a, -dzeta_factor * cos_a)
        dr_dzeta = dr_dzeta + np.sum(rc * dzeta_basis_r, axis=0)

        # Z and Dz use the opposite basis (sin for parity=True, cos for parity=False).
        basis_z = np.where(pc, sin_a, cos_a)
        z = z + np.sum(zc * basis_z, axis=0)
        dzeta = dzeta + np.sum(dzc * basis_z, axis=0)

        dtheta_basis_z = np.where(pc, mc * cos_a, -mc * sin_a)
        dz_dtheta = dz_dtheta + np.sum(zc * dtheta_basis_z, axis=0)
        ddz_dtheta = ddz_dtheta + np.sum(dzc * dtheta_basis_z, axis=0)

        dzeta_basis_z = np.where(pc, -dzeta_factor * cos_a, dzeta_factor * sin_a)
        dz_dzeta = dz_dzeta + np.sum(zc * dzeta_basis_z, axis=0)
        ddz_dzeta = ddz_dzeta + np.sum(dzc * dzeta_basis_z, axis=0)

    return r, dr_dtheta, dr_dzeta, z, dz_dtheta, dz_dzeta, dzeta, ddz_dtheta, ddz_dzeta


def _gpsipsi_from_bc_file(
    *,
    nml: Namelist,
    grids: V3Grids,
    geom,
    r_n_wish: float,
    vmecradial_option: int,
    geometry_scheme: int,
) -> np.ndarray:
    """Compute `gpsipsi` (written as `gpsiHatpsiHat`) for geometryScheme 11/12.

    This replicates the `nearbyRadiiGiven` branch used in v3's `geometry.F90` for
    Sugama magnetic drift support, but only computes `gpsipsi` (not curvature).
    """
    p = _resolve_equilibrium_file_from_namelist(nml=nml)
    header, surf_old, surf_new = read_boozer_bc_bracketing_surfaces(
        path=p, geometry_scheme=int(geometry_scheme), r_n_wish=float(r_n_wish)
    )

    r_old = float(surf_old.r_n)
    r_new = float(surf_new.r_n)
    if r_new == r_old:
        radial_weight = 1.0
    else:
        if int(vmecradial_option) == 1:
            radial_weight = 1.0 if abs(r_old - float(r_n_wish)) < abs(r_new - float(r_n_wish)) else 0.0
        else:
            radial_weight = (r_new * r_new - float(r_n_wish) * float(r_n_wish)) / (r_new * r_new - r_old * r_old)

    theta = np.asarray(grids.theta, dtype=np.float64)
    zeta = np.asarray(grids.zeta, dtype=np.float64)

    # Toroidal direction sign switch: n -> -n.
    n_old = -np.asarray(surf_old.n, dtype=np.int32)
    n_new = -np.asarray(surf_new.n, dtype=np.int32)

    # Toroidal direction sign switch for Dz: multiply coefficients by -1.
    dz_scale = float(2.0 * np.pi / float(header.n_periods)) * (-1.0)

    ro, dro_dt, dro_dz, zo, dzo_dt, dzo_dz, dzo, ddzo_dt, ddzo_dz = _evaluate_boozer_rzd_and_derivatives(
        theta=theta,
        zeta=zeta,
        n_periods=int(header.n_periods),
        m=np.asarray(surf_old.m, dtype=np.int32),
        n=n_old,
        parity=np.asarray(surf_old.parity, dtype=bool),
        r0=float(surf_old.r0),
        r_amp=np.asarray(surf_old.r_amp, dtype=np.float64),
        z_amp=np.asarray(surf_old.z_amp, dtype=np.float64),
        dz_amp=np.asarray(surf_old.dz_amp, dtype=np.float64),
        dz_scale=dz_scale,
    )
    rn, drn_dt, drn_dz, zn, dzn_dt, dzn_dz, dzn, ddzn_dt, ddzn_dz = _evaluate_boozer_rzd_and_derivatives(
        theta=theta,
        zeta=zeta,
        n_periods=int(header.n_periods),
        m=np.asarray(surf_new.m, dtype=np.int32),
        n=n_new,
        parity=np.asarray(surf_new.parity, dtype=bool),
        r0=float(surf_new.r0),
        r_amp=np.asarray(surf_new.r_amp, dtype=np.float64),
        z_amp=np.asarray(surf_new.z_amp, dtype=np.float64),
        dz_amp=np.asarray(surf_new.dz_amp, dtype=np.float64),
        dz_scale=dz_scale,
    )

    r = ro * radial_weight + rn * (1.0 - radial_weight)
    dr_dt = dro_dt * radial_weight + drn_dt * (1.0 - radial_weight)
    dr_dz = dro_dz * radial_weight + drn_dz * (1.0 - radial_weight)
    z = zo * radial_weight + zn * (1.0 - radial_weight)
    dz_dt = dzo_dt * radial_weight + dzn_dt * (1.0 - radial_weight)
    dz_dz = dzo_dz * radial_weight + dzn_dz * (1.0 - radial_weight)
    dz_field = dzo * radial_weight + dzn * (1.0 - radial_weight)
    ddz_dt = ddzo_dt * radial_weight + ddzn_dt * (1.0 - radial_weight)
    ddz_dz = ddzo_dz * radial_weight + ddzn_dz * (1.0 - radial_weight)

    # geometric toroidal angle: geomang = Dz - zeta
    geomang = dz_field - zeta[None, :]
    dgeomang_dtheta = ddz_dt
    dgeomang_dzeta = ddz_dz - 1.0

    cosg = np.cos(geomang)
    sing = np.sin(geomang)

    dX_dtheta = dr_dt * cosg - r * dgeomang_dtheta * sing
    dX_dzeta = dr_dz * cosg - r * dgeomang_dzeta * sing
    dY_dtheta = dr_dt * sing + r * dgeomang_dtheta * cosg
    dY_dzeta = dr_dz * sing + r * dgeomang_dzeta * cosg

    # Z is already the cylindrical vertical coordinate.
    dZ_dtheta = dz_dt
    dZ_dzeta = dz_dz

    d_hat = np.asarray(geom.d_hat, dtype=np.float64)
    gradpsiX = d_hat * (dY_dtheta * dZ_dzeta - dZ_dtheta * dY_dzeta)
    gradpsiY = d_hat * (dZ_dtheta * dX_dzeta - dX_dtheta * dZ_dzeta)
    gradpsiZ = d_hat * (dX_dtheta * dY_dzeta - dY_dtheta * dX_dzeta)
    gpsipsi = gradpsiX * gradpsiX + gradpsiY * gradpsiY + gradpsiZ * gradpsiZ
    return gpsipsi


def _gpsipsi_from_wout_file(
    *,
    nml: Namelist,
    grids: V3Grids,
    psi_n_wish: float,
    vmec_radial_option: int,
) -> np.ndarray:
    """Compute `gpsipsi` (written as `gpsiHatpsiHat`) for geometryScheme=5 (VMEC wout).

    This mirrors the metric-based expression used in v3 `geometry.F90::computeBHat_VMEC`.
    """
    geom_params = nml.group("geometryParameters")
    wout_path = _resolve_equilibrium_file_from_namelist(nml=nml)
    w = read_vmec_wout(wout_path)

    interp = vmec_interpolation(w=w, psi_n_wish=float(psi_n_wish), vmec_radial_option=int(vmec_radial_option))
    (i_full0, i_full1) = interp.index_full
    (w_full0, w_full1) = interp.weight_full
    (i_half0, i_half1) = interp.index_half
    (w_half0, w_half1) = interp.weight_half

    theta = np.asarray(grids.theta, dtype=np.float64)
    zeta = np.asarray(grids.zeta, dtype=np.float64)
    theta1 = theta[None, :, None]
    zeta1 = zeta[None, None, :]

    ntheta = int(theta.shape[0])
    nzeta = int(zeta.shape[0])

    # Reproduce v3's mode-inclusion logic (same as used for BHat, etc).
    n_periods = int(w.nfp)
    xm_nyq = np.asarray(w.xm_nyq, dtype=np.float64)
    xn_nyq = np.asarray(w.xn_nyq, dtype=np.float64)
    b00 = float(w.bmnc[0, i_half0] * w_half0 + w.bmnc[0, i_half1] * w_half1)
    if b00 == 0.0:
        raise ValueError("VMEC bmnc(0,0) is zero; cannot apply min_Bmn_to_load filter.")

    min_bmn_to_load = float(_get_float(geom_params, "min_Bmn_to_load", 0.0))
    ripple_scale = float(_get_float(geom_params, "rippleScale", 1.0))
    helicity_n = int(_get_int(geom_params, "helicity_n", 0))
    helicity_l = int(_get_int(geom_params, "helicity_l", 0))
    vmec_nyquist_option = int(
        _get_int(geom_params, "VMEC_Nyquist_option", _get_int(geom_params, "VMEC_NYQUIST_OPTION", 1))
    )
    if vmec_nyquist_option == 0:
        vmec_nyquist_option = 1
    if vmec_nyquist_option not in {1, 2}:
        raise ValueError("VMEC_Nyquist_option must be 1 (skip Nyquist) or 2 (include Nyquist).")

    # v3 applies the scale factor *before* checking `min_Bmn_to_load`.
    scale_all = np.array(
        [
            _set_scale_factor(
                n=int(round(float(xn_nyq[k]) / float(n_periods))),
                m=int(round(float(xm_nyq[k]))),
                helicity_n=helicity_n,
                helicity_l=helicity_l,
                ripple_scale=ripple_scale,
            )
            for k in range(int(xm_nyq.shape[0]))
        ],
        dtype=np.float64,
    )
    b_mode = (w.bmnc[:, i_half0] * w_half0 + w.bmnc[:, i_half1] * w_half1) * scale_all
    include = np.abs(b_mode / float(b00)) >= float(min_bmn_to_load)
    if int(vmec_nyquist_option) == 1:
        n_eff = xn_nyq / float(n_periods)
        include = include & (np.abs(xm_nyq) < float(w.mpol)) & (np.abs(n_eff) <= float(w.ntor))

    idx = np.nonzero(include)[0].astype(np.int32)
    if idx.size == 0:
        raise ValueError("No VMEC modes were included (min_Bmn_to_load too large?).")

    # Map (m,n) in the non-Nyquist mode table to indices (for rmnc/zmns).
    mode_to_index: dict[tuple[int, int], int] = {
        (int(w.xm[k]), int(w.xn[k])): int(k) for k in range(int(w.xm.shape[0]))
    }

    # VMEC spacing in psiHat (v3): dpsi = phi(2)/(2*pi).
    dpsi = float(w.phi[1]) / (2.0 * math.pi)

    rmnc = np.asarray(w.rmnc, dtype=np.float64)
    zmns = np.asarray(w.zmns, dtype=np.float64)
    d_rmnc_dpsi_hat = np.zeros_like(rmnc)
    d_zmns_dpsi_hat = np.zeros_like(zmns)
    d_rmnc_dpsi_hat[:, 1:] = (rmnc[:, 1:] - rmnc[:, :-1]) / float(dpsi)
    d_zmns_dpsi_hat[:, 1:] = (zmns[:, 1:] - zmns[:, :-1]) / float(dpsi)

    r = np.zeros((ntheta, nzeta), dtype=np.float64)
    dr_dtheta = np.zeros_like(r)
    dr_dzeta = np.zeros_like(r)
    dr_dpsi_hat = np.zeros_like(r)
    dz_dtheta = np.zeros_like(r)
    dz_dzeta = np.zeros_like(r)
    dz_dpsi_hat = np.zeros_like(r)

    chunk = 256
    for i0 in range(0, int(idx.size), chunk):
        sel_nyq = idx[i0 : min(int(idx.size), i0 + chunk)]
        non_sel = np.array(
            [mode_to_index.get((int(w.xm_nyq[k]), int(w.xn_nyq[k])), -1) for k in sel_nyq.tolist()],
            dtype=np.int32,
        )
        mask = non_sel >= 0
        if not np.any(mask):
            continue
        non_sel = non_sel[mask]
        m = np.asarray(w.xm[non_sel], dtype=np.float64)[:, None, None]
        n_nyq = np.asarray(w.xn[non_sel], dtype=np.float64)[:, None, None]

        scale = np.array(
            [
                _set_scale_factor(
                    n=int(round(float(w.xn[k]) / float(n_periods))),
                    m=int(round(float(w.xm[k]))),
                    helicity_n=helicity_n,
                    helicity_l=helicity_l,
                    ripple_scale=ripple_scale,
                )
                for k in non_sel.tolist()
            ],
            dtype=np.float64,
        )[:, None, None]

        angle = m * theta1 - n_nyq * zeta1
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        # R and Z live on the full mesh.
        r_coef = (rmnc[non_sel, i_full0] * w_full0 + rmnc[non_sel, i_full1] * w_full1)[:, None, None] * scale
        z_coef = (zmns[non_sel, i_full0] * w_full0 + zmns[non_sel, i_full1] * w_full1)[:, None, None] * scale

        # d/dpsiHat coefficients live on the half mesh.
        dr_dpsi_coef = (
            d_rmnc_dpsi_hat[non_sel, i_half0] * w_half0 + d_rmnc_dpsi_hat[non_sel, i_half1] * w_half1
        )[:, None, None] * scale
        dz_dpsi_coef = (
            d_zmns_dpsi_hat[non_sel, i_half0] * w_half0 + d_zmns_dpsi_hat[non_sel, i_half1] * w_half1
        )[:, None, None] * scale

        r += np.sum(r_coef * cos_a, axis=0)
        dr_dtheta += np.sum(-m * r_coef * sin_a, axis=0)
        dr_dzeta += np.sum(n_nyq * r_coef * sin_a, axis=0)
        dr_dpsi_hat += np.sum(dr_dpsi_coef * cos_a, axis=0)

        dz_dtheta += np.sum(m * z_coef * cos_a, axis=0)
        dz_dzeta += np.sum(-n_nyq * z_coef * cos_a, axis=0)
        dz_dpsi_hat += np.sum(dz_dpsi_coef * sin_a, axis=0)

    cosz = np.cos(zeta)[None, :]
    sinz = np.sin(zeta)[None, :]

    dX_dtheta = dr_dtheta * cosz
    dX_dzeta = dr_dzeta * cosz - r * sinz
    dX_dpsi = dr_dpsi_hat * cosz

    dY_dtheta = dr_dtheta * sinz
    dY_dzeta = dr_dzeta * sinz + r * cosz
    dY_dpsi = dr_dpsi_hat * sinz

    dZ_dtheta = dz_dtheta
    dZ_dzeta = dz_dzeta
    dZ_dpsi = dz_dpsi_hat

    g_tt = dX_dtheta * dX_dtheta + dY_dtheta * dY_dtheta + dZ_dtheta * dZ_dtheta
    g_tz = dX_dtheta * dX_dzeta + dY_dtheta * dY_dzeta + dZ_dtheta * dZ_dzeta
    g_zz = dX_dzeta * dX_dzeta + dY_dzeta * dY_dzeta + dZ_dzeta * dZ_dzeta
    g_pt = dX_dpsi * dX_dtheta + dY_dpsi * dY_dtheta + dZ_dpsi * dZ_dtheta
    g_pz = dX_dpsi * dX_dzeta + dY_dpsi * dY_dzeta + dZ_dpsi * dZ_dzeta
    g_pp = dX_dpsi * dX_dpsi + dY_dpsi * dY_dpsi + dZ_dpsi * dZ_dpsi

    denom = g_tt * g_zz - g_tz * g_tz
    gpsipsi = 1.0 / (
        g_pp
        + (g_pt * (g_tz * g_pz - g_pt * g_zz) + g_pz * (g_pt * g_tz - g_tt * g_pz)) / denom
    )
    return gpsipsi


def sfincs_jax_output_dict(*, nml: Namelist, grids: V3Grids) -> Dict[str, Any]:
    """Build a dictionary of `sfincsOutput.h5` datasets supported by `sfincs_jax`."""
    geom_params = nml.group("geometryParameters")
    phys = nml.group("physicsParameters")
    species = nml.group("speciesParameters")
    other = nml.group("otherNumericalParameters")
    resolution = nml.group("resolutionParameters")
    export_f = nml.group("export_f")
    precond = nml.group("preconditionerOptions")
    general = nml.group("general")

    geometry_scheme = _get_int(geom_params, "geometryScheme", -1)
    if geometry_scheme not in {1, 2, 4, 5, 11, 12}:
        raise NotImplementedError(
            "sfincs_jax sfincsOutput writing currently supports geometryScheme in {1,2,4,5,11,12} only."
        )

    geom = geometry_from_namelist(nml=nml, grids=grids)

    w_vmec = None
    if geometry_scheme == 4:
        psi_a_hat, a_hat = _scheme4_radial_constants()
    elif geometry_scheme == 1:
        # v3 defaults are in `globalVariables.F90`; allow the namelist to override them.
        a_hat = _get_float(geom_params, "aHat", 0.5585)
        psi_a_hat = _get_float(geom_params, "psiAHat", 0.15596)
    elif geometry_scheme == 2:
        # v3 ignores *_wish and uses the fixed LHD model with aHat=0.5585 and psiAHat=aHat^2/2.
        a_hat = 0.5585
        psi_a_hat = (a_hat * a_hat) / 2.0
    elif geometry_scheme in {11, 12}:
        bc_path = _resolve_equilibrium_file_from_namelist(nml=nml)
        header = read_boozer_bc_header(path=bc_path, geometry_scheme=int(geometry_scheme))
        psi_a_hat = float(header.psi_a_hat)
        a_hat = float(header.a_hat)
    else:
        wout_path = _resolve_equilibrium_file_from_namelist(nml=nml)
        w_vmec = read_vmec_wout(wout_path)
        psi_a_hat = float(psi_a_hat_from_wout(w_vmec))
        a_hat = float(w_vmec.aminor_p)

    input_radial_coordinate = _get_int(geom_params, "inputRadialCoordinate", 3)
    psi_hat_wish_in = _get_float(geom_params, "psiHat_wish", -1.0)
    psi_n_wish_in = _get_float(geom_params, "psiN_wish", 0.25)
    r_hat_wish_in = _get_float(geom_params, "rHat_wish", -1.0)
    r_n_wish_in = _get_float(geom_params, "rN_wish", 0.5)
    psi_hat_wish, psi_n_wish, r_hat_wish, r_n_wish = _set_input_radial_coordinate_wish(
        input_radial_coordinate=input_radial_coordinate,
        psi_a_hat=psi_a_hat,
        a_hat=a_hat,
        psi_hat_wish_in=psi_hat_wish_in,
        psi_n_wish_in=psi_n_wish_in,
        r_hat_wish_in=r_hat_wish_in,
        r_n_wish_in=r_n_wish_in,
    )

    if geometry_scheme == 5:
        if w_vmec is None:
            raise RuntimeError("Internal error: missing VMEC wout handle for geometryScheme=5.")
        vmec_radial_option = _get_int(geom_params, "VMECRadialOption", _get_int(geom_params, "VMECRADIALOPTION", 0))
        interp = vmec_interpolation(w=w_vmec, psi_n_wish=float(psi_n_wish), vmec_radial_option=int(vmec_radial_option))
        psi_n = float(interp.psi_n)
        r_n = float(math.sqrt(float(psi_n)))
    else:
        # For the Boozer schemes supported here, v3 effectively uses rN = rN_wish.
        r_n = float(r_n_wish)
        psi_n = float(r_n) * float(r_n)
    psi_hat = float(psi_a_hat) * float(psi_n)
    r_hat = float(a_hat) * float(r_n)

    # Scalars / sizes:
    # In monoenergetic runs (RHSMode=3) the upstream examples may omit speciesParameters.
    # Use v3's default Z=1 as a fallback so the output file can still be written.
    z_s = _as_1d_float(species, "Zs", default=1.0)
    n_species = int(z_s.shape[0])

    out: Dict[str, Any] = {}
    out["Nspecies"] = np.asarray(n_species, dtype=np.int32)
    out["Ntheta"] = np.asarray(int(grids.theta.shape[0]), dtype=np.int32)
    out["Nzeta"] = np.asarray(int(grids.zeta.shape[0]), dtype=np.int32)
    out["Nxi"] = np.asarray(int(grids.n_xi), dtype=np.int32)
    out["NL"] = np.asarray(int(grids.n_l), dtype=np.int32)
    out["Nx"] = np.asarray(int(grids.x.shape[0]), dtype=np.int32)
    out["theta"] = np.asarray(grids.theta, dtype=np.float64)
    out["zeta"] = np.asarray(grids.zeta, dtype=np.float64)
    out["x"] = np.asarray(grids.x, dtype=np.float64)
    out["Nxi_for_x"] = np.asarray(grids.n_xi_for_x, dtype=np.int32)

    # Numerical scheme settings (subset):
    out["geometryScheme"] = np.asarray(geometry_scheme, dtype=np.int32)
    out["thetaDerivativeScheme"] = np.asarray(_get_int(other, "thetaDerivativeScheme", 2), dtype=np.int32)
    out["zetaDerivativeScheme"] = np.asarray(_get_int(other, "zetaDerivativeScheme", 2), dtype=np.int32)
    out["ExBDerivativeSchemeTheta"] = np.asarray(_get_int(other, "ExBDerivativeSchemeTheta", 0), dtype=np.int32)
    out["ExBDerivativeSchemeZeta"] = np.asarray(_get_int(other, "ExBDerivativeSchemeZeta", 0), dtype=np.int32)
    out["magneticDriftDerivativeScheme"] = np.asarray(_get_int(other, "magneticDriftDerivativeScheme", 3), dtype=np.int32)
    out["xGridScheme"] = np.asarray(_get_int(other, "xGridScheme", 5), dtype=np.int32)
    out["Nxi_for_x_option"] = np.asarray(_get_int(other, "Nxi_for_x_option", 0), dtype=np.int32)
    out["solverTolerance"] = np.asarray(_get_float(resolution, "solverTolerance", 1e-8), dtype=np.float64)

    # Physics parameters (subset):
    # Defaults match v3 `globalVariables.F90`, which upstream examples sometimes rely on.
    out["Delta"] = np.asarray(_get_float(phys, "Delta", 4.5694e-3), dtype=np.float64)
    out["alpha"] = np.asarray(_get_float(phys, "alpha", 1.0), dtype=np.float64)
    out["nu_n"] = np.asarray(_get_float(phys, "nu_n", 8.330e-3), dtype=np.float64)
    out["Er"] = np.asarray(_get_float(phys, "Er", 0.0), dtype=np.float64)
    out["collisionOperator"] = np.asarray(_get_int(phys, "collisionOperator", 0), dtype=np.int32)
    out["magneticDriftScheme"] = np.asarray(_get_int(phys, "magneticDriftScheme", 0), dtype=np.int32)
    out["includeXDotTerm"] = _fortran_logical(bool(phys.get("INCLUDEXDOTTERM", False)))
    out["includeElectricFieldTermInXiDot"] = _fortran_logical(bool(phys.get("INCLUDEELECTRICFIELDTERMINXIDOT", False)))
    out["useDKESExBDrift"] = _fortran_logical(bool(phys.get("USEDKESEXBDRIFT", False)))

    # Radial-coordinate bookkeeping and conversions.
    out["psiAHat"] = np.asarray(float(psi_a_hat), dtype=np.float64)
    out["aHat"] = np.asarray(float(a_hat), dtype=np.float64)
    out["psiN"] = np.asarray(float(psi_n), dtype=np.float64)
    out["psiHat"] = np.asarray(float(psi_hat), dtype=np.float64)
    out["rN"] = np.asarray(float(r_n), dtype=np.float64)
    out["rHat"] = np.asarray(float(r_hat), dtype=np.float64)
    out["inputRadialCoordinate"] = np.asarray(int(input_radial_coordinate), dtype=np.int32)

    input_radial_grad = _get_int(geom_params, "inputRadialCoordinateForGradients", 4)
    out["inputRadialCoordinateForGradients"] = np.asarray(int(input_radial_grad), dtype=np.int32)

    conv = _conversion_factors_to_from_dpsi_hat(psi_a_hat=psi_a_hat, a_hat=a_hat, r_n=r_n)
    rhs_mode = int(_get_int(general, "RHSMode", 1))
    dphi_dpsihat_in = _get_float(phys, "dPhiHatdpsiHat", 0.0)
    dphi_dpsin_in = _get_float(phys, "dPhiHatdpsiN", 0.0)
    dphi_drhat_in = _get_float(phys, "dPhiHatdrHat", 0.0)
    dphi_drn_in = _get_float(phys, "dPhiHatdrN", 0.0)
    er_in = float(out["Er"])

    if rhs_mode == 3:
        # Monoenergetic coefficient computation (v3 `sfincs_main.F90` overwrites nu_n and dPhiHatdpsiHat).
        nu_prime = _get_float(phys, "nuPrime", 1.0)
        e_star = _get_float(phys, "EStar", 0.0)
        denom = float(geom.g_hat) + float(geom.iota) * float(geom.i_hat)
        out["nuPrime"] = np.asarray(float(nu_prime), dtype=np.float64)
        out["EStar"] = np.asarray(float(e_star), dtype=np.float64)
        out["nu_n"] = np.asarray(float(nu_prime) * float(geom.b0_over_bbar) / denom, dtype=np.float64)
        dphi_dpsihat = (
            2.0
            / (float(out["alpha"]) * float(out["Delta"]))
            * float(e_star)
            * float(geom.iota)
            * float(geom.b0_over_bbar)
            / float(geom.g_hat)
        )
    else:
        if int(input_radial_grad) == 0:
            dphi_dpsihat = float(dphi_dpsihat_in)
        elif int(input_radial_grad) == 1:
            dphi_dpsihat = float(conv["ddpsiN2ddpsiHat"]) * float(dphi_dpsin_in)
        elif int(input_radial_grad) == 2:
            dphi_dpsihat = float(conv["ddrHat2ddpsiHat"]) * float(dphi_drhat_in)
        elif int(input_radial_grad) == 3:
            dphi_dpsihat = float(conv["ddrN2ddpsiHat"]) * float(dphi_drn_in)
        elif int(input_radial_grad) == 4:
            dphi_dpsihat = float(conv["ddrHat2ddpsiHat"]) * (-float(er_in))
        else:
            raise NotImplementedError(f"Unsupported inputRadialCoordinateForGradients={input_radial_grad}")

    # Convert from d/dpsiHat to all other coordinates (matches v3).
    out["dPhiHatdpsiHat"] = np.asarray(dphi_dpsihat, dtype=np.float64)
    out["dPhiHatdpsiN"] = np.asarray(float(conv["ddpsiHat2ddpsiN"]) * dphi_dpsihat, dtype=np.float64)
    out["dPhiHatdrHat"] = np.asarray(float(conv["ddpsiHat2ddrHat"]) * dphi_dpsihat, dtype=np.float64)
    out["dPhiHatdrN"] = np.asarray(float(conv["ddpsiHat2ddrN"]) * dphi_dpsihat, dtype=np.float64)
    out["Er"] = np.asarray(-float(out["dPhiHatdrHat"]), dtype=np.float64)

    out["EParallelHat"] = np.asarray(_get_float(phys, "EParallelHat", 0.0), dtype=np.float64)
    out["rippleScale"] = np.asarray(_get_float(geom_params, "rippleScale", 1.0), dtype=np.float64)
    out["coordinateSystem"] = np.asarray(2 if geometry_scheme == 5 else 1, dtype=np.int32)

    out["integerToRepresentFalse"] = np.asarray(-1, dtype=np.int32)
    out["integerToRepresentTrue"] = np.asarray(1, dtype=np.int32)

    out["useIterativeLinearSolver"] = _fortran_logical(True)
    out["RHSMode"] = np.asarray(int(rhs_mode), dtype=np.int32)
    # In v3, `NIterations` is written as 0 during initializeOutputFile(), and is later
    # overwritten by updateOutputFile(iterationNum, ...). For the small output parity
    # fixtures, geometryScheme=4 typically retains 0, while 11/12 generally writes 1.
    out["NIterations"] = np.asarray(1 if geometry_scheme in {11, 12} else 0, dtype=np.int32)
    out["finished"] = _fortran_logical(True)

    out["xMax"] = np.asarray(_get_float(other, "xMax", 5.0), dtype=np.float64)
    out["xGrid_k"] = np.asarray(_get_float(other, "xGrid_k", 0.0), dtype=np.float64)
    out["xPotentialsGridScheme"] = np.asarray(_get_int(other, "xPotentialsGridScheme", 2), dtype=np.int32)
    out["NxPotentialsPerVth"] = np.asarray(_get_float(other, "NxPotentialsPerVth", 40.0), dtype=np.float64)

    x_grid_scheme = _get_int(other, "xGridScheme", 5)
    point_at_x0 = x_grid_scheme in {2, 6}
    out["pointAtX0"] = _fortran_logical(point_at_x0)

    out["export_full_f"] = _fortran_logical(bool(export_f.get("EXPORT_FULL_F", False)))
    out["export_delta_f"] = _fortran_logical(bool(export_f.get("EXPORT_DELTA_F", False)))

    out["force0RadialCurrentInEquilibrium"] = _fortran_logical(True)
    out["includePhi1"] = _fortran_logical(bool(phys.get("INCLUDEPHI1", False)))
    out["includePhi1InCollisionOperator"] = _fortran_logical(bool(phys.get("INCLUDEPHI1INCOLLISIONOPERATOR", False)))
    # v3 has additional internal logic for these flags; for the current parity fixtures,
    # this key is `true` even when includePhi1 is `false`.
    out["includePhi1InKineticEquation"] = _fortran_logical(True)
    out["includeTemperatureEquilibrationTerm"] = _fortran_logical(bool(phys.get("INCLUDETEMPERATUREEQUILIBRATIONTERM", False)))
    out["include_fDivVE_Term"] = _fortran_logical(bool(phys.get("INCLUDE_FDIVVE_TERM", False)))
    out["withAdiabatic"] = _fortran_logical(bool(phys.get("WITHADIABATIC", False)))
    out["withNBIspec"] = _fortran_logical(bool(phys.get("WITHNBISPEC", False)))

    out["classicalParticleFluxNoPhi1_psiHat"] = np.zeros((n_species,), dtype=np.float64)
    out["classicalParticleFluxNoPhi1_psiN"] = np.zeros((n_species,), dtype=np.float64)
    out["classicalParticleFluxNoPhi1_rHat"] = np.zeros((n_species,), dtype=np.float64)
    out["classicalParticleFluxNoPhi1_rN"] = np.zeros((n_species,), dtype=np.float64)
    out["classicalHeatFluxNoPhi1_psiHat"] = np.zeros((n_species,), dtype=np.float64)
    out["classicalHeatFluxNoPhi1_psiN"] = np.zeros((n_species,), dtype=np.float64)
    out["classicalHeatFluxNoPhi1_rHat"] = np.zeros((n_species,), dtype=np.float64)
    out["classicalHeatFluxNoPhi1_rN"] = np.zeros((n_species,), dtype=np.float64)

    # Preconditioner / constraints: mirror common v3 defaults for the fixture set.
    out["reusePreconditioner"] = _fortran_logical(bool(precond.get("REUSEPRECONDITIONER", True)))
    out["preconditioner_species"] = np.asarray(_get_int(precond, "preconditioner_species", 1), dtype=np.int32)
    out["preconditioner_x"] = np.asarray(_get_int(precond, "preconditioner_x", 1), dtype=np.int32)
    out["preconditioner_x_min_L"] = np.asarray(_get_int(precond, "preconditioner_x_min_L", 0), dtype=np.int32)
    out["preconditioner_xi"] = np.asarray(_get_int(precond, "preconditioner_xi", 1), dtype=np.int32)
    out["preconditioner_theta"] = np.asarray(_get_int(precond, "preconditioner_theta", 0), dtype=np.int32)
    out["preconditioner_zeta"] = np.asarray(_get_int(precond, "preconditioner_zeta", 0), dtype=np.int32)
    out["preconditioner_magnetic_drifts_max_L"] = np.asarray(
        _get_int(precond, "preconditioner_magnetic_drifts_max_L", 2), dtype=np.int32
    )

    # In v3, `constraintScheme` is read from physicsParameters and finalized in createGrids.F90.
    constraint_scheme = _get_int(phys, "constraintScheme", -1)
    if constraint_scheme < 0:
        # v3 sets `constraintScheme` during createGrids():
        #   - `collisionOperator = 0` (full FP)  -> constraintScheme = 1
        #   - `collisionOperator = 1` (PAS)      -> constraintScheme = 2
        # See `sfincs/fortran/version3/createGrids.F90`.
        constraint_scheme = 1 if int(out["collisionOperator"]) == 0 else 2
    out["constraintScheme"] = np.asarray(int(constraint_scheme), dtype=np.int32)

    # Species arrays:
    out["Zs"] = np.asarray(z_s, dtype=np.float64)
    out["mHats"] = np.asarray(_as_1d_float(species, "mHats", default=1.0), dtype=np.float64)
    out["THats"] = np.asarray(_as_1d_float(species, "THats", default=1.0), dtype=np.float64)
    out["nHats"] = np.asarray(_as_1d_float(species, "nHats", default=1.0), dtype=np.float64)
    dn_dpsihat_in = np.asarray(_as_1d_float(species, "dNHatdpsiHats", default=0.0), dtype=np.float64)
    dn_dpsin_in = np.asarray(_as_1d_float(species, "dNHatdpsiNs", default=0.0), dtype=np.float64)
    dn_drhat_in = np.asarray(_as_1d_float(species, "dNHatdrHats", default=0.0), dtype=np.float64)
    dn_drn_in = np.asarray(_as_1d_float(species, "dNHatdrNs", default=0.0), dtype=np.float64)

    dt_dpsihat_in = np.asarray(_as_1d_float(species, "dTHatdpsiHats", default=0.0), dtype=np.float64)
    dt_dpsin_in = np.asarray(_as_1d_float(species, "dTHatdpsiNs", default=0.0), dtype=np.float64)
    dt_drhat_in = np.asarray(_as_1d_float(species, "dTHatdrHats", default=0.0), dtype=np.float64)
    dt_drn_in = np.asarray(_as_1d_float(species, "dTHatdrNs", default=0.0), dtype=np.float64)

    if int(input_radial_grad) == 0:
        dn_dpsihat = dn_dpsihat_in
        dt_dpsihat = dt_dpsihat_in
    elif int(input_radial_grad) == 1:
        dn_dpsihat = float(conv["ddpsiN2ddpsiHat"]) * dn_dpsin_in
        dt_dpsihat = float(conv["ddpsiN2ddpsiHat"]) * dt_dpsin_in
    elif int(input_radial_grad) == 2:
        dn_dpsihat = float(conv["ddrHat2ddpsiHat"]) * dn_drhat_in
        dt_dpsihat = float(conv["ddrHat2ddpsiHat"]) * dt_drhat_in
    elif int(input_radial_grad) == 3:
        dn_dpsihat = float(conv["ddrN2ddpsiHat"]) * dn_drn_in
        dt_dpsihat = float(conv["ddrN2ddpsiHat"]) * dt_drn_in
    elif int(input_radial_grad) == 4:
        dn_dpsihat = float(conv["ddrHat2ddpsiHat"]) * dn_drhat_in
        dt_dpsihat = float(conv["ddrHat2ddpsiHat"]) * dt_drhat_in
    else:
        raise NotImplementedError(f"Unsupported inputRadialCoordinateForGradients={input_radial_grad}")

    out["dnHatdpsiHat"] = np.asarray(dn_dpsihat, dtype=np.float64)
    out["dnHatdpsiN"] = np.asarray(float(conv["ddpsiHat2ddpsiN"]) * dn_dpsihat, dtype=np.float64)
    out["dnHatdrHat"] = np.asarray(float(conv["ddpsiHat2ddrHat"]) * dn_dpsihat, dtype=np.float64)
    out["dnHatdrN"] = np.asarray(float(conv["ddpsiHat2ddrN"]) * dn_dpsihat, dtype=np.float64)

    out["dTHatdpsiHat"] = np.asarray(dt_dpsihat, dtype=np.float64)
    out["dTHatdpsiN"] = np.asarray(float(conv["ddpsiHat2ddpsiN"]) * dt_dpsihat, dtype=np.float64)
    out["dTHatdrHat"] = np.asarray(float(conv["ddpsiHat2ddrHat"]) * dt_dpsihat, dtype=np.float64)
    out["dTHatdrN"] = np.asarray(float(conv["ddpsiHat2ddrN"]) * dt_dpsihat, dtype=np.float64)

    # Geometry arrays (subset):
    out["NPeriods"] = np.asarray(int(geom.n_periods), dtype=np.int32)
    if geometry_scheme == 5:
        tw = np.asarray(grids.theta_weights, dtype=np.float64)[:, None]
        zw = np.asarray(grids.zeta_weights, dtype=np.float64)[None, :]
        wgt = tw * zw
        bh = np.asarray(geom.b_hat, dtype=np.float64)
        jac = 1.0 / np.asarray(geom.d_hat, dtype=np.float64)
        vprime = float(np.sum(wgt * jac))
        fsab_b2 = float(np.sum(wgt * (bh * bh) * jac)) / vprime
        b0 = float(np.sum(wgt * (bh**3) * jac)) / (vprime * fsab_b2)
        g_hat = float(np.sum(wgt * np.asarray(geom.b_hat_sub_zeta, dtype=np.float64))) / (4.0 * np.pi * np.pi)
        i_hat = float(np.sum(wgt * np.asarray(geom.b_hat_sub_theta, dtype=np.float64))) / (4.0 * np.pi * np.pi)
        out["B0OverBBar"] = np.asarray(b0, dtype=np.float64)
        out["iota"] = np.asarray(float(geom.iota), dtype=np.float64)
        out["GHat"] = np.asarray(g_hat, dtype=np.float64)
        out["IHat"] = np.asarray(i_hat, dtype=np.float64)
    else:
        out["B0OverBBar"] = np.asarray(float(geom.b0_over_bbar), dtype=np.float64)
        out["iota"] = np.asarray(float(geom.iota), dtype=np.float64)
        out["GHat"] = np.asarray(float(geom.g_hat), dtype=np.float64)
        out["IHat"] = np.asarray(float(geom.i_hat), dtype=np.float64)
    out["VPrimeHat"] = np.asarray(float(np.asarray(vprime_hat_jax(grids=grids, geom=geom), dtype=np.float64)), dtype=np.float64)
    out["FSABHat2"] = np.asarray(float(np.asarray(fsab_hat2_jax(grids=grids, geom=geom), dtype=np.float64)), dtype=np.float64)
    if geometry_scheme in {11, 12}:
        r_n_wish = float(r_n_wish)
        vmecradial_option = _get_int(geom_params, "VMECRadialOption", _get_int(geom_params, "VMECRADIALOPTION", 0))
        out["gpsiHatpsiHat"] = _gpsipsi_from_bc_file(
            nml=nml,
            grids=grids,
            geom=geom,
            r_n_wish=r_n_wish,
            vmecradial_option=int(vmecradial_option),
            geometry_scheme=int(geometry_scheme),
        )
    elif geometry_scheme == 5:
        vmec_radial_option = _get_int(geom_params, "VMECRadialOption", _get_int(geom_params, "VMECRADIALOPTION", 0))
        out["gpsiHatpsiHat"] = _gpsipsi_from_wout_file(
            nml=nml,
            grids=grids,
            psi_n_wish=float(psi_n_wish),
            vmec_radial_option=int(vmec_radial_option),
        )
    else:
        out["gpsiHatpsiHat"] = np.zeros_like(np.asarray(geom.b_hat, dtype=np.float64))

    bdotcurlb = (
        np.asarray(geom.d_hat, dtype=np.float64)
        * (
            np.asarray(geom.b_hat_sub_theta, dtype=np.float64) * np.asarray(geom.db_hat_sub_psi_dzeta, dtype=np.float64)
            - np.asarray(geom.b_hat_sub_theta, dtype=np.float64) * np.asarray(geom.db_hat_sub_zeta_dpsi_hat, dtype=np.float64)
            + np.asarray(geom.b_hat_sub_zeta, dtype=np.float64) * np.asarray(geom.db_hat_sub_theta_dpsi_hat, dtype=np.float64)
            - np.asarray(geom.b_hat_sub_zeta, dtype=np.float64) * np.asarray(geom.db_hat_sub_psi_dtheta, dtype=np.float64)
        )
    )
    out["BDotCurlB"] = np.asarray(bdotcurlb, dtype=np.float64)

    out["DHat"] = np.asarray(geom.d_hat, dtype=np.float64)
    out["BHat"] = np.asarray(geom.b_hat, dtype=np.float64)
    out["dBHatdpsiHat"] = np.asarray(geom.db_hat_dpsi_hat, dtype=np.float64)
    out["dBHatdtheta"] = np.asarray(geom.db_hat_dtheta, dtype=np.float64)
    out["dBHatdzeta"] = np.asarray(geom.db_hat_dzeta, dtype=np.float64)

    out["BHat_sub_psi"] = np.asarray(geom.b_hat_sub_psi, dtype=np.float64)
    out["dBHat_sub_psi_dtheta"] = np.asarray(geom.db_hat_sub_psi_dtheta, dtype=np.float64)
    out["dBHat_sub_psi_dzeta"] = np.asarray(geom.db_hat_sub_psi_dzeta, dtype=np.float64)
    out["BHat_sub_theta"] = np.asarray(geom.b_hat_sub_theta, dtype=np.float64)
    out["dBHat_sub_theta_dpsiHat"] = np.asarray(geom.db_hat_sub_theta_dpsi_hat, dtype=np.float64)
    out["dBHat_sub_theta_dzeta"] = np.asarray(geom.db_hat_sub_theta_dzeta, dtype=np.float64)
    out["BHat_sub_zeta"] = np.asarray(geom.b_hat_sub_zeta, dtype=np.float64)
    out["dBHat_sub_zeta_dpsiHat"] = np.asarray(geom.db_hat_sub_zeta_dpsi_hat, dtype=np.float64)
    out["dBHat_sub_zeta_dtheta"] = np.asarray(geom.db_hat_sub_zeta_dtheta, dtype=np.float64)
    out["BHat_sup_theta"] = np.asarray(geom.b_hat_sup_theta, dtype=np.float64)
    out["dBHat_sup_theta_dpsiHat"] = np.asarray(geom.db_hat_sup_theta_dpsi_hat, dtype=np.float64)
    out["dBHat_sup_theta_dzeta"] = np.asarray(geom.db_hat_sup_theta_dzeta, dtype=np.float64)
    out["BHat_sup_zeta"] = np.asarray(geom.b_hat_sup_zeta, dtype=np.float64)
    out["dBHat_sup_zeta_dpsiHat"] = np.asarray(geom.db_hat_sup_zeta_dpsi_hat, dtype=np.float64)
    out["dBHat_sup_zeta_dtheta"] = np.asarray(geom.db_hat_sup_zeta_dtheta, dtype=np.float64)
    diotadpsi_hat = 0.0
    if geometry_scheme in {11, 12}:
        # Compute diotadpsiHat from the bracketing surfaces (v3 uses nearby radii for 11/12).
        p = _resolve_equilibrium_file_from_namelist(nml=nml)
        header, surf_old, surf_new = read_boozer_bc_bracketing_surfaces(
            path=p, geometry_scheme=int(geometry_scheme), r_n_wish=float(r_n_wish)
        )
        delta_psi_hat = float(header.psi_a_hat) * (
            float(surf_new.r_n) * float(surf_new.r_n) - float(surf_old.r_n) * float(surf_old.r_n)
        )
        # Toroidal direction sign switch: iota -> -iota, matching v3.
        diotadpsi_hat = (-(float(surf_new.iota)) - (-(float(surf_old.iota)))) / float(delta_psi_hat)
    elif geometry_scheme == 5:
        if w_vmec is None:
            raise RuntimeError("Internal error: missing VMEC wout handle for geometryScheme=5.")
        vmec_radial_option = _get_int(geom_params, "VMECRadialOption", _get_int(geom_params, "VMECRADIALOPTION", 0))
        interp = vmec_interpolation(w=w_vmec, psi_n_wish=float(psi_n_wish), vmec_radial_option=int(vmec_radial_option))
        j0, j1 = interp.index_half
        dpsi_n = float(interp.psi_n_half[j1 - 1] - interp.psi_n_half[j0 - 1])
        if dpsi_n != 0.0:
            diotadpsi_hat = float(w_vmec.iotas[j1] - w_vmec.iotas[j0]) / dpsi_n / float(psi_a_hat)
    out["diotadpsiHat"] = np.asarray(float(diotadpsi_hat), dtype=np.float64)

    out["uHat"] = np.asarray(u_hat_np(grids=grids, geom=geom), dtype=np.float64)

    return out


def write_sfincs_jax_output_h5(
    *,
    input_namelist: Path,
    output_path: Path,
    fortran_layout: bool = True,
    overwrite: bool = True,
    compute_transport_matrix: bool = False,
) -> Path:
    """Create a SFINCS-style `sfincsOutput.h5` file from `sfincs_jax` for supported modes."""
    nml = read_sfincs_input(input_namelist)
    grids = grids_from_namelist(nml)
    data = sfincs_jax_output_dict(nml=nml, grids=grids)

    if bool(compute_transport_matrix):
        rhs_mode = int(nml.group("general").get("RHSMODE", 1))
        if rhs_mode in {2, 3}:
            # Import lazily to keep geometry-only use-cases lightweight.
            from .v3_driver import solve_v3_transport_matrix_linear_gmres

            tm = solve_v3_transport_matrix_linear_gmres(nml=nml).transport_matrix
            data["transportMatrix"] = np.asarray(tm, dtype=np.float64)

    data["input.namelist"] = input_namelist.read_text()
    write_sfincs_h5(path=output_path, data=data, fortran_layout=fortran_layout, overwrite=overwrite)
    return output_path.resolve()
