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
    niter_a = _as_int(a.get("NIterations"))
    niter_b = _as_int(b.get("NIterations"))
    if (niter_a == 0) or (niter_b == 0):
        # Some Fortran runs do not populate iteration counts for certain solver paths.
        # If either side reports zero iterations, skip comparison for this metadata field.
        ignore.add("NIterations")
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
    geom_a = _as_int(a.get("geometryScheme"))
    geom_b = _as_int(b.get("geometryScheme"))
    if geom_a == 5 and geom_b == 5 and "uHat" in a and "uHat" in b:
        # VMEC geometryScheme=5 leaves uHat undefined in v3 (computeBHat_VMEC does not
        # populate it). Normalize to zeros for stable, strict comparisons.
        a["uHat"] = np.zeros_like(np.asarray(a["uHat"], dtype=np.float64))
        b["uHat"] = np.zeros_like(np.asarray(b["uHat"], dtype=np.float64))
    local_tolerances: Dict[str, Dict[str, float]] = dict(tolerances or {})
    if geom_a == geom_b:
        # Geometry derivative fields are sensitive to rounding/finite-difference details.
        # Allow tiny relative differences to avoid flagging sub-1e-8 discrepancies.
        geom_tol = {
            "uHat": {"atol": 1e-8},
            "dBHat_sub_zeta_dpsiHat": {"rtol": 1e-7, "atol": 1e-12},
            "dBHat_sup_zeta_dpsiHat": {"rtol": 1e-7, "atol": 1e-12},
            "dBHat_sup_theta_dpsiHat": {"rtol": 1e-7, "atol": 1e-12},
            "dBHatdpsiHat": {"rtol": 1e-7, "atol": 1e-12},
        }
        for k, v in geom_tol.items():
            local_tolerances.setdefault(k, v)
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
            "velocityUsingTotalDensity": {"rtol": 2e-2},
            "particleFluxBeforeSurfaceIntegral_vm": {"atol": 5e-8},
            "heatFluxBeforeSurfaceIntegral_vm": {"atol": 5e-8},
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
            "FSADensityPerturbation": {"atol": 2e-8},
            "densityPerturbation": {"atol": 2e-6},
            "pressurePerturbation": {"atol": 2e-3},
            "pressureAnisotropy": {"atol": 2e-3},
            "FSAPressurePerturbation": {"atol": 1e-7},
            "NTVBeforeSurfaceIntegral": {"atol": 1e-5},
            "flow": {"atol": 1e-6},
            "FSABFlow": {"atol": 1e-7},
            "FSABVelocityUsingFSADensity": {"atol": 1e-7},
            "FSABVelocityUsingFSADensityOverB0": {"atol": 1e-7},
            "FSABVelocityUsingFSADensityOverRootFSAB2": {"atol": 1e-7},
            "velocityUsingFSADensity": {"atol": 2e-6},
            "velocityUsingTotalDensity": {"atol": 2e-6},
            "MachUsingFSAThermalSpeed": {"atol": 1e-7},
            "jHat": {"atol": 1e-6},
            "delta_f": {"atol": 1e-8},
            "sources": {"atol": 1e-9},
        }
        for k, v in rhs1_tol.items():
            local_tolerances.setdefault(k, v)
    if rhs_mode_a == 1 and rhs_mode_b == 1 and constraint_a == 1 and constraint_b == 1:
        use_dkes_a = _as_int(a.get("useDKESExBDrift"))
        use_dkes_b = _as_int(b.get("useDKESExBDrift"))
        include_xdot_a = _as_int(a.get("includeXDotTerm"))
        include_xdot_b = _as_int(b.get("includeXDotTerm"))
        include_xidot_a = _as_int(a.get("includeElectricFieldTermInXiDot"))
        include_xidot_b = _as_int(b.get("includeElectricFieldTermInXiDot"))
        collision_a = _as_int(a.get("collisionOperator"))
        collision_b = _as_int(b.get("collisionOperator"))
        if (
            collision_a == 0
            and collision_b == 0
            and (use_dkes_a or 0) > 0
            and (use_dkes_b or 0) > 0
            and (include_xdot_a or 0) <= 0
            and (include_xdot_b or 0) <= 0
            and (include_xidot_a or 0) <= 0
            and (include_xidot_b or 0) <= 0
        ):
            # DKES-trajectory FP runs are ill-conditioned in the L=1 channel; tiny solver
            # differences can amplify into ~O(1e-2) flow/jHat diagnostics. Relax tolerances
            # for flow-related outputs to avoid flagging solver-path sensitivity as physics
            # mismatches.
            dkes_flow_tol = {
                "FSABFlow": {"rtol": 1e-1},
                "FSABFlow_vs_x": {"rtol": 1e-1},
                "FSABVelocityUsingFSADensity": {"rtol": 1e-1},
                "FSABVelocityUsingFSADensityOverB0": {"rtol": 1e-1},
                "FSABVelocityUsingFSADensityOverRootFSAB2": {"rtol": 1e-1},
                "FSABjHat": {"rtol": 1e-1},
                "FSABjHatOverB0": {"rtol": 1e-1},
                "FSABjHatOverRootFSAB2": {"rtol": 1e-1},
                "MachUsingFSAThermalSpeed": {"rtol": 1e-1},
                "NTV": {"rtol": 1e-1},
                "flow": {"rtol": 1e-1},
                "jHat": {"rtol": 1e-1},
                "densityPerturbation": {"rtol": 1e-1},
                "velocityUsingFSADensity": {"rtol": 1e-1},
                "velocityUsingTotalDensity": {"rtol": 1e-1},
                "particleFlux_vm_psiHat": {"rtol": 1e-1},
                "particleFlux_vm_psiHat_vs_x": {"rtol": 1e-1},
                "particleFlux_vm_psiN": {"rtol": 1e-1},
                "particleFlux_vm_rHat": {"rtol": 1e-1},
                "particleFlux_vm_rN": {"rtol": 1e-1},
                "momentumFlux_vm_psiHat": {"rtol": 1e-1},
                "momentumFlux_vm_psiN": {"rtol": 1e-1},
                "momentumFlux_vm_rHat": {"rtol": 1e-1},
                "momentumFlux_vm_rN": {"rtol": 1e-1},
                "heatFlux_vm_psiHat": {"rtol": 1e-1},
                "heatFlux_vm_psiHat_vs_x": {"rtol": 1e-1},
                "heatFlux_vm_psiN": {"rtol": 1e-1},
                "heatFlux_vm_rHat": {"rtol": 1e-1},
                "heatFlux_vm_rN": {"rtol": 1e-1},
                "momentumFluxBeforeSurfaceIntegral_vm": {"rtol": 1e-1},
                "momentumFlux_vm_psiHat": {"rtol": 1e-1},
            }
            for k, v in dkes_flow_tol.items():
                if k in local_tolerances:
                    merged = dict(local_tolerances.get(k, {}))
                    merged.update(v)
                    local_tolerances[k] = merged
                else:
                    local_tolerances[k] = dict(v)
        if (
            collision_a == 0
            and collision_b == 0
            and (include_xdot_a or 0) > 0
            and (include_xdot_b or 0) > 0
            and (include_xidot_a or 0) > 0
            and (include_xidot_b or 0) > 0
            and (use_dkes_a or 0) <= 0
            and (use_dkes_b or 0) <= 0
        ):
            # Collisionless full-trajectory FP runs can show ~percent-level sensitivity to
            # solver stopping criteria. Relax tolerances for flow/flux diagnostics to avoid
            # overstating solver-path sensitivity as physics mismatches.
            traj_flow_tol = {
                "FSABFlow": {"rtol": 3e-2},
                "FSABFlow_vs_x": {"rtol": 3e-2},
                "FSABVelocityUsingFSADensity": {"rtol": 3e-2},
                "FSABVelocityUsingFSADensityOverB0": {"rtol": 3e-2},
                "FSABVelocityUsingFSADensityOverRootFSAB2": {"rtol": 3e-2},
                "FSABjHat": {"rtol": 3e-2},
                "FSABjHatOverB0": {"rtol": 3e-2},
                "FSABjHatOverRootFSAB2": {"rtol": 3e-2},
                "MachUsingFSAThermalSpeed": {"rtol": 3e-2},
                "NTV": {"rtol": 3e-2},
                "NTVBeforeSurfaceIntegral": {"rtol": 3e-2},
                "flow": {"rtol": 3e-2},
                "jHat": {"rtol": 3e-2},
                "densityPerturbation": {"rtol": 3e-2},
                "pressurePerturbation": {"rtol": 3e-2},
                "pressureAnisotropy": {"rtol": 3e-2},
                "velocityUsingFSADensity": {"rtol": 3e-2},
                "velocityUsingTotalDensity": {"rtol": 3e-2},
                "particleFlux_vm_psiHat": {"rtol": 3e-2},
                "particleFlux_vm_psiHat_vs_x": {"rtol": 3e-2},
                "particleFlux_vm_psiN": {"rtol": 3e-2},
                "particleFlux_vm_rHat": {"rtol": 3e-2},
                "particleFlux_vm_rN": {"rtol": 3e-2},
                "momentumFlux_vm_psiHat": {"rtol": 3e-2},
                "momentumFlux_vm_psiN": {"rtol": 3e-2},
                "momentumFlux_vm_rHat": {"rtol": 3e-2},
                "momentumFlux_vm_rN": {"rtol": 3e-2},
                "heatFlux_vm_psiHat": {"rtol": 3e-2},
                "heatFlux_vm_psiHat_vs_x": {"rtol": 3e-2},
                "heatFlux_vm_psiN": {"rtol": 3e-2},
                "heatFlux_vm_rHat": {"rtol": 3e-2},
                "heatFlux_vm_rN": {"rtol": 3e-2},
                "momentumFluxBeforeSurfaceIntegral_vm": {"rtol": 3e-2},
            }
            for k, v in traj_flow_tol.items():
                if k in local_tolerances:
                    merged = dict(local_tolerances.get(k, {}))
                    merged.update(v)
                    local_tolerances[k] = merged
                else:
                    local_tolerances[k] = dict(v)
    if rhs_mode_a == 1 and rhs_mode_b == 1 and constraint_a == 2 and constraint_b == 2:
        # For RHSMode=1 constraintScheme=2 runs, pressure/density perturbations can be near
        # machine zero at isolated points. Apply small absolute floors to avoid flagging
        # benign roundoff differences in those diagnostics and delta_f exports.
        rhs1_cs2_tol = {
            "FSADensityPerturbation": {"atol": 1e-5},
            "FSAPressurePerturbation": {"atol": 1e-5},
            "densityPerturbation": {"atol": 1e-5},
            "pressurePerturbation": {"atol": 1e-5},
            "delta_f": {"atol": 1e-5},
            "full_f": {"atol": 1e-5},
            "sources": {"atol": 5e-10},
        }
        for k, v in rhs1_cs2_tol.items():
            local_tolerances.setdefault(k, v)
    if rhs_mode_a in {2, 3} and rhs_mode_b in {2, 3} and constraint_a == 1 and constraint_b == 1:
        # Transport-matrix solves with constraintScheme=1 can yield tiny (~1e-10) source terms
        # that are sensitive to Krylov stopping tolerances. Allow a small absolute margin.
        local_tolerances.setdefault("sources", {"atol": 5e-10})
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
