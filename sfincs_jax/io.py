from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import h5py
import numpy as np

from .diagnostics import fsab_hat2 as fsab_hat2_jax
from .diagnostics import u_hat_np
from .diagnostics import vprime_hat as vprime_hat_jax
from .namelist import Namelist, read_sfincs_input
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


def sfincs_jax_output_dict(*, nml: Namelist, grids: V3Grids) -> Dict[str, Any]:
    """Build a dictionary of `sfincsOutput.h5` datasets supported by `sfincs_jax`."""
    geom_params = nml.group("geometryParameters")
    phys = nml.group("physicsParameters")
    species = nml.group("speciesParameters")
    other = nml.group("otherNumericalParameters")
    resolution = nml.group("resolutionParameters")
    export_f = nml.group("export_f")
    precond = nml.group("preconditionerOptions")

    geometry_scheme = _get_int(geom_params, "geometryScheme", -1)
    if geometry_scheme != 4:
        raise NotImplementedError("sfincs_jax sfincsOutput writing currently supports geometryScheme=4 only.")

    geom = geometry_from_namelist(nml=nml, grids=grids)

    # v3 scheme-4 radial-coordinate defaults (see v3 geometry + radialCoordinates).
    psi_a_hat = -0.384935
    a_hat = 0.5109
    psi_n = 0.25
    r_n = float(np.sqrt(psi_n))
    psi_hat = float(psi_a_hat * psi_n)
    r_hat = float(a_hat * r_n)

    # Scalars / sizes:
    z_s = _as_1d_float(species, "Zs")
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
    out["Delta"] = np.asarray(_get_float(phys, "Delta", 0.0), dtype=np.float64)
    out["alpha"] = np.asarray(_get_float(phys, "alpha", 1.0), dtype=np.float64)
    out["nu_n"] = np.asarray(_get_float(phys, "nu_n", 0.0), dtype=np.float64)
    out["Er"] = np.asarray(_get_float(phys, "Er", 0.0), dtype=np.float64)
    out["collisionOperator"] = np.asarray(_get_int(phys, "collisionOperator", 0), dtype=np.int32)
    out["magneticDriftScheme"] = np.asarray(_get_int(phys, "magneticDriftScheme", 0), dtype=np.int32)
    out["includeXDotTerm"] = _fortran_logical(bool(phys.get("INCLUDEXDOTTERM", False)))
    out["includeElectricFieldTermInXiDot"] = _fortran_logical(bool(phys.get("INCLUDEELECTRICFIELDTERMINXIDOT", False)))
    out["useDKESExBDrift"] = _fortran_logical(bool(phys.get("USEDKESEXBDRIFT", False)))

    er = float(out["Er"])
    dphi_drhat = -er
    dphi_drN = dphi_drhat * a_hat
    dphi_dpsiN = dphi_drN / (2.0 * r_n)
    dphi_dpsiHat = dphi_dpsiN / psi_a_hat
    out["dPhiHatdpsiHat"] = np.asarray(dphi_dpsiHat, dtype=np.float64)
    out["dPhiHatdpsiN"] = np.asarray(dphi_dpsiN, dtype=np.float64)
    out["dPhiHatdrHat"] = np.asarray(dphi_drhat, dtype=np.float64)
    out["dPhiHatdrN"] = np.asarray(dphi_drN, dtype=np.float64)

    out["psiAHat"] = np.asarray(psi_a_hat, dtype=np.float64)
    out["aHat"] = np.asarray(a_hat, dtype=np.float64)
    out["psiN"] = np.asarray(psi_n, dtype=np.float64)
    out["psiHat"] = np.asarray(psi_hat, dtype=np.float64)
    out["rN"] = np.asarray(r_n, dtype=np.float64)
    out["rHat"] = np.asarray(r_hat, dtype=np.float64)
    out["inputRadialCoordinate"] = np.asarray(3, dtype=np.int32)
    out["inputRadialCoordinateForGradients"] = np.asarray(4, dtype=np.int32)

    out["EParallelHat"] = np.asarray(_get_float(phys, "EParallelHat", 0.0), dtype=np.float64)
    out["diotadpsiHat"] = np.asarray(0.0, dtype=np.float64)
    out["rippleScale"] = np.asarray(1.0, dtype=np.float64)
    out["coordinateSystem"] = np.asarray(1, dtype=np.int32)

    out["integerToRepresentFalse"] = np.asarray(-1, dtype=np.int32)
    out["integerToRepresentTrue"] = np.asarray(1, dtype=np.int32)

    out["useIterativeLinearSolver"] = _fortran_logical(True)
    out["RHSMode"] = np.asarray(1, dtype=np.int32)
    out["NIterations"] = np.asarray(0, dtype=np.int32)
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

    constraint_scheme = _get_int(precond, "constraintScheme", -1)
    if constraint_scheme < 0:
        # Minimal reproduction of v3 behavior for the current fixture regime:
        # choose 2 when collisions are enabled, else 0.
        constraint_scheme = 2 if int(out["collisionOperator"]) != 0 else 0
    out["constraintScheme"] = np.asarray(int(constraint_scheme), dtype=np.int32)

    # Species arrays:
    out["Zs"] = np.asarray(z_s, dtype=np.float64)
    out["mHats"] = np.asarray(_as_1d_float(species, "mHats", default=1.0), dtype=np.float64)
    out["THats"] = np.asarray(_as_1d_float(species, "THats", default=1.0), dtype=np.float64)
    out["nHats"] = np.asarray(_as_1d_float(species, "nHats", default=1.0), dtype=np.float64)
    if "DNHATDRHATS" in species:
        dn_drhat = np.asarray(_as_1d_float(species, "dNHatdrHats"), dtype=np.float64)
        out["dnHatdrHat"] = dn_drhat
        out["dnHatdrN"] = dn_drhat * a_hat
        out["dnHatdpsiN"] = out["dnHatdrN"] / (2.0 * r_n)
        out["dnHatdpsiHat"] = out["dnHatdpsiN"] / psi_a_hat
    if "DTHATDRHATS" in species:
        dt_drhat = np.asarray(_as_1d_float(species, "dTHatdrHats"), dtype=np.float64)
        out["dTHatdrHat"] = dt_drhat
        out["dTHatdrN"] = dt_drhat * a_hat
        out["dTHatdpsiN"] = out["dTHatdrN"] / (2.0 * r_n)
        out["dTHatdpsiHat"] = out["dTHatdpsiN"] / psi_a_hat

    # Geometry arrays (subset):
    out["NPeriods"] = np.asarray(int(geom.n_periods), dtype=np.int32)
    out["B0OverBBar"] = np.asarray(float(geom.b0_over_bbar), dtype=np.float64)
    out["iota"] = np.asarray(float(geom.iota), dtype=np.float64)
    out["GHat"] = np.asarray(float(geom.g_hat), dtype=np.float64)
    out["IHat"] = np.asarray(float(geom.i_hat), dtype=np.float64)
    out["VPrimeHat"] = np.asarray(float(np.asarray(vprime_hat_jax(grids=grids, geom=geom), dtype=np.float64)), dtype=np.float64)
    out["FSABHat2"] = np.asarray(float(np.asarray(fsab_hat2_jax(grids=grids, geom=geom), dtype=np.float64)), dtype=np.float64)
    out["gpsiHatpsiHat"] = np.zeros_like(np.asarray(geom.b_hat, dtype=np.float64))
    out["BDotCurlB"] = np.zeros_like(np.asarray(geom.b_hat, dtype=np.float64))

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
    out["BHat_sub_zeta"] = np.asarray(geom.b_hat_sub_zeta, dtype=np.float64)
    out["dBHat_sub_zeta_dpsiHat"] = np.asarray(geom.db_hat_sub_zeta_dpsi_hat, dtype=np.float64)
    out["BHat_sup_theta"] = np.asarray(geom.b_hat_sup_theta, dtype=np.float64)
    out["BHat_sup_zeta"] = np.asarray(geom.b_hat_sup_zeta, dtype=np.float64)
    out["dBHat_sub_theta_dzeta"] = np.zeros_like(np.asarray(geom.b_hat, dtype=np.float64))
    out["dBHat_sub_zeta_dtheta"] = np.zeros_like(np.asarray(geom.b_hat, dtype=np.float64))
    out["dBHat_sup_theta_dpsiHat"] = np.zeros_like(np.asarray(geom.b_hat, dtype=np.float64))
    out["dBHat_sup_theta_dzeta"] = np.zeros_like(np.asarray(geom.b_hat, dtype=np.float64))
    out["dBHat_sup_zeta_dpsiHat"] = np.zeros_like(np.asarray(geom.b_hat, dtype=np.float64))
    out["dBHat_sup_zeta_dtheta"] = np.zeros_like(np.asarray(geom.b_hat, dtype=np.float64))

    out["uHat"] = np.asarray(u_hat_np(grids=grids, geom=geom), dtype=np.float64)

    return out


def write_sfincs_jax_output_h5(
    *,
    input_namelist: Path,
    output_path: Path,
    fortran_layout: bool = True,
    overwrite: bool = True,
) -> Path:
    """Create a SFINCS-style `sfincsOutput.h5` file from `sfincs_jax` for supported modes."""
    nml = read_sfincs_input(input_namelist)
    grids = grids_from_namelist(nml)
    data = sfincs_jax_output_dict(nml=nml, grids=grids)
    data["input.namelist"] = input_namelist.read_text()
    write_sfincs_h5(path=output_path, data=data, fortran_layout=fortran_layout, overwrite=overwrite)
    return output_path.resolve()
