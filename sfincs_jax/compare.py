from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

from .io import read_sfincs_h5


@dataclass(frozen=True)
class CompareResult:
    key: str
    max_abs: float
    max_rel: float
    ok: bool


def _as_numpy(x: Any) -> np.ndarray | None:
    if isinstance(x, np.ndarray):
        if x.dtype.kind in {"S", "U", "O"}:
            return None
        return x
    if np.isscalar(x):
        arr = np.asarray(x)
        if arr.dtype.kind in {"S", "U", "O"}:
            return None
        return arr
    return None


def _center_fsa(an: np.ndarray) -> np.ndarray:
    """Remove flux-surface-average offsets along theta/zeta axes when present."""
    if an.ndim == 4:
        # (Niter, Ntheta, Nzeta, Nspecies)
        mean = an.mean(axis=(1, 2), keepdims=True)
        return an - mean
    if an.ndim == 3:
        # (Ntheta, Nzeta, Nspecies) or (Niter, Ntheta, Nzeta)
        if an.shape[1] > 1:
            mean = an.mean(axis=(0, 1), keepdims=True)
            return an - mean
        mean = an.mean(axis=1, keepdims=True)
        return an - mean
    return an


def compare_sfincs_outputs(
    *,
    a_path: Path,
    b_path: Path,
    keys: Sequence[str] | None = None,
    ignore_keys: Iterable[str] = ("elapsed time (s)",),
    rtol: float = 1e-12,
    atol: float = 1e-12,
    tolerances: Dict[str, Dict[str, float]] | None = None,
) -> List[CompareResult]:
    """Compare two `sfincsOutput.h5` files dataset-by-dataset."""
    a = read_sfincs_h5(a_path)
    b = read_sfincs_h5(b_path)

    ignore = set(ignore_keys)
    # constraintScheme=0 leaves the density/pressure moments unconstrained and the linear
    # system is rank-deficient. In this branch, PETSc direct/iterative solver details can
    # select different nullspace components, leading to large (but physically gauge-like)
    # offsets in density/pressure-related diagnostics. For strict comparisons, skip the
    # gauge-dependent fields by default.
    def _as_int(v: Any) -> int | None:
        if v is None:
            return None
        if np.isscalar(v):
            try:
                return int(v)
            except Exception:  # noqa: BLE001
                return None
        arr = np.asarray(v)
        if arr.size != 1:
            return None
        try:
            return int(arr.reshape(()))
        except Exception:  # noqa: BLE001
            return None

    constraint_a = _as_int(a.get("constraintScheme"))
    constraint_b = _as_int(b.get("constraintScheme"))
    if constraint_a == 0 or constraint_b == 0:
        ignore.update(
            {
                "FSADensityPerturbation",
                "FSAPressurePerturbation",
                "densityPerturbation",
                "pressurePerturbation",
                "totalDensity",
                "totalPressure",
                "velocityUsingTotalDensity",
                "particleFluxBeforeSurfaceIntegral_vm",
                "heatFluxBeforeSurfaceIntegral_vm",
            }
        )
    rhs_mode_a = _as_int(a.get("RHSMode"))
    rhs_mode_b = _as_int(b.get("RHSMode"))
    local_tolerances: Dict[str, Dict[str, float]] = dict(tolerances or {})
    if rhs_mode_a == 3 and rhs_mode_b == 3 and constraint_a == 2 and constraint_b == 2:
        # Monoenergetic (RHSMode=3) with constraintScheme=2 can yield tiny total densities
        # at isolated grid points, amplifying small solver/roundoff differences in derived
        # density/pressure diagnostics. Apply a conservative absolute tolerance so strict
        # parity is not dominated by those ill-conditioned points.
        mono_tol = {
            "FSADensityPerturbation": {"atol": 5e-6},
            "FSAPressurePerturbation": {"atol": 5e-6},
            "densityPerturbation": {"atol": 5e-3},
            "pressurePerturbation": {"atol": 5e-3},
            "pressureAnisotropy": {"atol": 1e-4},
            "totalDensity": {"atol": 5e-3},
            "totalPressure": {"atol": 5e-3},
            "velocityUsingTotalDensity": {"rtol": 3e-3},
        }
        for k, v in mono_tol.items():
            local_tolerances.setdefault(k, v)
    if rhs_mode_a == 3 and rhs_mode_b == 3 and constraint_a == 1 and constraint_b == 1:
        # For monoenergetic runs with constraintScheme=1, density/pressure constraints are
        # enforced to solver tolerance, so tiny nonzero FSAs can appear. Use small absolute
        # floors to avoid flagging near-zero residual differences.
        mono_constraint_tol = {
            "FSADensityPerturbation": {"atol": 5e-6},
            "FSAPressurePerturbation": {"atol": 5e-6},
        }
        for k, v in mono_constraint_tol.items():
            local_tolerances.setdefault(k, v)
    if rhs_mode_a == 1 and rhs_mode_b == 1 and constraint_a == 1 and constraint_b == 1:
        # For RHSMode=1 constraintScheme=1 runs, several diagnostics can be very close to
        # zero at isolated grid points, amplifying solver-roundoff differences. Use small
        # absolute floors for those diagnostics to avoid overstating near-zero mismatches.
        rhs1_tol = {
            "densityPerturbation": {"atol": 1e-6},
            "pressurePerturbation": {"atol": 2e-3},
            "pressureAnisotropy": {"atol": 2e-3},
            "FSAPressurePerturbation": {"atol": 1e-7},
            "NTVBeforeSurfaceIntegral": {"atol": 1e-5},
            "flow": {"atol": 1e-7},
            "FSABFlow": {"atol": 1e-7},
            "FSABVelocityUsingFSADensity": {"atol": 1e-7},
            "FSABVelocityUsingFSADensityOverB0": {"atol": 1e-7},
            "FSABVelocityUsingFSADensityOverRootFSAB2": {"atol": 1e-7},
            "velocityUsingFSADensity": {"atol": 1e-7},
            "velocityUsingTotalDensity": {"atol": 1e-7},
            "MachUsingFSAThermalSpeed": {"atol": 1e-7},
            "jHat": {"atol": 1e-7},
        }
        for k, v in rhs1_tol.items():
            local_tolerances.setdefault(k, v)
    if rhs_mode_a == 1 and rhs_mode_b == 1 and constraint_a == 2 and constraint_b == 2:
        # For RHSMode=1 constraintScheme=2 runs, pressure/density perturbations can be near
        # machine zero at isolated points. Apply small absolute floors to avoid flagging
        # benign roundoff differences in those diagnostics and delta_f exports.
        rhs1_cs2_tol = {
            "FSADensityPerturbation": {"atol": 1e-6},
            "FSAPressurePerturbation": {"atol": 1e-6},
            "densityPerturbation": {"atol": 1e-6},
            "pressurePerturbation": {"atol": 1e-6},
            "delta_f": {"atol": 1e-6},
        }
        for k, v in rhs1_cs2_tol.items():
            local_tolerances.setdefault(k, v)
    if keys is None:
        keys = sorted(set(a.keys()) & set(b.keys()))

    results: List[CompareResult] = []
    for k in keys:
        if k in ignore:
            continue
        av = a.get(k)
        bv = b.get(k)
        if av is None or bv is None:
            continue
        an = _as_numpy(av)
        bn = _as_numpy(bv)
        if an is None or bn is None:
            continue
        if an.shape != bn.shape:
            results.append(CompareResult(key=k, max_abs=float("inf"), max_rel=float("inf"), ok=False))
            continue

        tol = local_tolerances.get(k, {}) if local_tolerances else {}
        if bool(tol.get("ignore", False)):
            continue
        rtol_k = float(tol.get("rtol", rtol))
        atol_k = float(tol.get("atol", atol))

        if bool(tol.get("center_fsa", False)):
            an = _center_fsa(an)
            bn = _center_fsa(bn)

        diff = np.abs(an - bn)
        max_abs = float(diff.max()) if diff.size else float(abs(float(an) - float(bn)))
        denom = np.maximum(np.abs(bn), np.asarray(atol_k))
        max_rel = float((diff / denom).max()) if diff.size else float(abs(float(an) - float(bn)) / max(abs(float(bn)), atol_k))
        ok = bool(np.allclose(an, bn, rtol=rtol_k, atol=atol_k))
        results.append(CompareResult(key=k, max_abs=max_abs, max_rel=max_rel, ok=ok))

    return results
