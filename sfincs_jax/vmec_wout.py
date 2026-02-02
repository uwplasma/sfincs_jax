from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import netcdf_file


@dataclass(frozen=True)
class VmecWout:
    path: Path
    nfp: int
    ns: int
    mpol: int
    ntor: int
    mnmax: int
    mnmax_nyq: int

    lasym: bool

    # Flux / geometry scalars:
    aminor_p: float
    phi: np.ndarray  # (ns,)

    # Mode tables:
    xm: np.ndarray  # (mnmax,)
    xn: np.ndarray  # (mnmax,)
    xm_nyq: np.ndarray  # (mnmax_nyq,)
    xn_nyq: np.ndarray  # (mnmax_nyq,)

    # Fourier coefficients (mode, radius):
    bmnc: np.ndarray  # (mnmax_nyq, ns)
    gmnc: np.ndarray  # (mnmax_nyq, ns)
    bsubumnc: np.ndarray  # (mnmax_nyq, ns)
    bsubvmnc: np.ndarray  # (mnmax_nyq, ns)
    bsubsmns: np.ndarray  # (mnmax_nyq, ns)
    bsupumnc: np.ndarray  # (mnmax_nyq, ns)
    bsupvmnc: np.ndarray  # (mnmax_nyq, ns)

    # (mnmax, ns) arrays for R,Z (full mesh):
    rmnc: np.ndarray  # (mnmax, ns)
    zmns: np.ndarray  # (mnmax, ns)
    lmns: np.ndarray  # (mnmax, ns-1) in file, but we keep as (mnmax, ns-1)

    # Flux functions (half mesh):
    iotas: np.ndarray  # (ns,)
    presf: np.ndarray  # (ns,)


def _read_var(f, name: str) -> np.ndarray:
    if name not in f.variables:
        raise KeyError(f"Missing variable {name!r} in wout file.")
    return np.array(f.variables[name].data)


def read_vmec_wout(path: str | Path) -> VmecWout:
    """Read a VMEC `wout_*.nc` file with SciPy's netCDF reader.

    Notes
    -----
    - This reader is intended to support a subset of SFINCS v3 `geometryScheme=5` workflows.
    - VMEC files are not differentiable inputs; after reading, downstream code can use JAX arrays.
    """
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(str(p))
    if p.suffix.lower() in {".txt", ".dat"}:
        # Many upstream distributions provide both an ASCII and a netCDF wout file. Prefer netCDF if present.
        p_nc = p.with_suffix(".nc")
        if p_nc.exists():
            p = p_nc
        else:
            raise NotImplementedError(
                "VMEC ASCII `wout_*.txt` files are not supported in sfincs_jax yet, "
                "and no `.nc` sibling was found."
            )

    with netcdf_file(p, "r", mmap=False) as f:
        nfp = int(_read_var(f, "nfp"))
        ns = int(_read_var(f, "ns"))
        mpol = int(_read_var(f, "mpol"))
        ntor = int(_read_var(f, "ntor"))
        mnmax = int(_read_var(f, "mnmax"))
        mnmax_nyq = int(_read_var(f, "mnmax_nyq"))

        lasym_raw = _read_var(f, "lasym__logical__")
        lasym = bool(int(np.asarray(lasym_raw).reshape(())))

        aminor_p = float(np.asarray(_read_var(f, "Aminor_p")).reshape(()))
        phi = _read_var(f, "phi").astype(np.float64)

        xm = _read_var(f, "xm").astype(np.int32)
        xn = _read_var(f, "xn").astype(np.int32)
        xm_nyq = _read_var(f, "xm_nyq").astype(np.int32)
        xn_nyq = _read_var(f, "xn_nyq").astype(np.int32)

        # Coefficients are stored in the file as (radius, mode); transpose to (mode, radius)
        # to match SFINCS v3 conventions used in `geometry.F90`.
        bmnc = _read_var(f, "bmnc").astype(np.float64).T
        gmnc = _read_var(f, "gmnc").astype(np.float64).T
        bsubumnc = _read_var(f, "bsubumnc").astype(np.float64).T
        bsubvmnc = _read_var(f, "bsubvmnc").astype(np.float64).T
        bsubsmns = _read_var(f, "bsubsmns").astype(np.float64).T
        bsupumnc = _read_var(f, "bsupumnc").astype(np.float64).T
        bsupvmnc = _read_var(f, "bsupvmnc").astype(np.float64).T

        rmnc = _read_var(f, "rmnc").astype(np.float64).T
        zmns = _read_var(f, "zmns").astype(np.float64).T
        lmns = _read_var(f, "lmns").astype(np.float64).T

        iotas = _read_var(f, "iotas").astype(np.float64)
        presf = _read_var(f, "presf").astype(np.float64)

    if lasym:
        # TODO: support non-stellarator-symmetric VMEC equilibria (bmns/gmns/etc).
        raise NotImplementedError("VMEC lasym=true is not yet supported in sfincs_jax.")

    # Basic sanity checks from v3:
    if xm[0] != 0 or xn[0] != 0:
        raise ValueError("Expected first (xm,xn) mode to be (0,0).")
    if xm_nyq[0] != 0 or xn_nyq[0] != 0:
        raise ValueError("Expected first (xm_nyq,xn_nyq) mode to be (0,0).")

    return VmecWout(
        path=p,
        nfp=nfp,
        ns=ns,
        mpol=mpol,
        ntor=ntor,
        mnmax=mnmax,
        mnmax_nyq=mnmax_nyq,
        lasym=lasym,
        aminor_p=aminor_p,
        phi=phi,
        xm=xm,
        xn=xn,
        xm_nyq=xm_nyq,
        xn_nyq=xn_nyq,
        bmnc=bmnc,
        gmnc=gmnc,
        bsubumnc=bsubumnc,
        bsubvmnc=bsubvmnc,
        bsubsmns=bsubsmns,
        bsupumnc=bsupumnc,
        bsupvmnc=bsupvmnc,
        rmnc=rmnc,
        zmns=zmns,
        lmns=lmns,
        iotas=iotas,
        presf=presf,
    )


def psi_a_hat_from_wout(w: VmecWout) -> float:
    """Compute psiAHat = phi(ns)/(2*pi) as in v3."""
    return float(w.phi[-1]) / (2.0 * math.pi)


def _set_scale_factor(*, n: int, m: int, helicity_n: int, helicity_l: int, ripple_scale: float) -> float:
    # Mirrors v3 `setScaleFactor()` in geometry.F90.
    scale = 1.0
    if helicity_n == 0 and n != 0:
        scale = float(ripple_scale)
    elif (n != 0) and (n * helicity_l) != (m * helicity_n):
        scale = float(ripple_scale)
    elif helicity_n != 0 and n == 0:
        scale = float(ripple_scale)
    return float(scale)


@dataclass(frozen=True)
class VmecInterpolation:
    index_full: tuple[int, int]  # 0-based indices
    weight_full: tuple[float, float]
    # 0-based indices into VMEC "half grid" arrays. In the wout file, half-grid arrays
    # have length `ns` with a dummy 0 at index 0, so valid values are typically in
    # [1, ns-1].
    index_half: tuple[int, int]
    weight_half: tuple[float, float]
    psi_n: float
    psi_n_full: np.ndarray  # (ns,)
    psi_n_half: np.ndarray  # (ns-1,)


def vmec_interpolation(*, w: VmecWout, psi_n_wish: float, vmec_radial_option: int) -> VmecInterpolation:
    """Replicate v3's radius selection and interpolation index/weight logic (geometryScheme=5)."""
    # psiN_full = phi / (2*pi*psiAHat) = phi / phi(ns)
    psi_n_full = np.asarray(w.phi, dtype=np.float64) / float(w.phi[-1])
    psi_n_half = 0.5 * (psi_n_full[:-1] + psi_n_full[1:])  # (ns-1,)

    if not (0.0 <= float(psi_n_wish) <= 1.0):
        raise ValueError("psiN_wish must be in [0,1].")

    # Choose "actual" radius to use based on VMEC radial option:
    if int(vmec_radial_option) == 0:
        psi_n = float(psi_n_wish)
    elif int(vmec_radial_option) == 1:
        # Nearest value on the VMEC half grid.
        j = int(np.argmin((psi_n_half - float(psi_n_wish)) ** 2))
        psi_n = float(psi_n_half[j])
    elif int(vmec_radial_option) == 2:
        j = int(np.argmin((psi_n_full - float(psi_n_wish)) ** 2))
        psi_n = float(psi_n_full[j])
    else:
        raise ValueError(f"Invalid VMECRadialOption={vmec_radial_option}")

    # Full-mesh indices/weights:
    ns = int(w.ns)
    if psi_n == 1.0:
        i0 = ns - 2
        i1 = ns - 1
        w0 = 0.0
    else:
        i0 = int(math.floor(psi_n * (ns - 1)))
        i1 = i0 + 1
        w0 = float(i0 + 1) - float(psi_n) * float(ns - 1)
    w1 = 1.0 - w0

    # Half-mesh indices/weights. Note: psi_n_half has length ns-1 and corresponds to half-grid points
    # between full-grid points 0..ns-2. VMEC half-mesh arrays in the wout file have length ns with a
    # dummy 0 at index 0 (Fortran index 1), so the "real" half indices start at python index 1
    # (Fortran index 2).
    if float(psi_n) < float(psi_n_half[0]):
        j0 = 1
        j1 = 2
        wh0 = (float(psi_n_half[1]) - float(psi_n)) / (float(psi_n_half[1]) - float(psi_n_half[0]))
    elif float(psi_n) > float(psi_n_half[-1]):
        j0 = ns - 2
        j1 = ns - 1
        wh0 = (float(psi_n_half[-1]) - float(psi_n)) / (float(psi_n_half[-1]) - float(psi_n_half[-2]))
    elif float(psi_n) == float(psi_n_half[-1]):
        j0 = ns - 2
        j1 = ns - 1
        wh0 = 0.0
    else:
        # v3 uses (1-based):
        #   j_for = floor(psiN*(ns-1) + 0.5) + 1
        # with a minimum of 2 since the first half-grid array element is always 0.
        j_for = int(math.floor(float(psi_n) * float(ns - 1) + 0.5)) + 1
        if j_for < 2:
            j_for = 2
        # Convert to python indices:
        j0 = j_for - 1
        j1 = j0 + 1
        wh0 = float(j_for) - float(psi_n) * float(ns - 1) - 0.5
    wh1 = 1.0 - wh0

    return VmecInterpolation(
        index_full=(i0, i1),
        weight_full=(w0, w1),
        index_half=(j0, j1),
        weight_half=(float(wh0), float(wh1)),
        psi_n=float(psi_n),
        psi_n_full=psi_n_full,
        psi_n_half=psi_n_half,
    )
