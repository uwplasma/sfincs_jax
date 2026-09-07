"""Native Case normalization and execution without a namelist round-trip."""

from __future__ import annotations

import hashlib
import math
import os
import platform
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from dkx.config import Case, CaseValidationError
from dkx.constants import RadialCoordinates
from dkx.result import Result


_ATOMIC_MASS_KG = 1.66053906892e-27
_PROTON_MASS_KG = 1.67262192369e-27


@dataclass(frozen=True)
class _GeometryState:
    """Geometry data and normalization loaded once for a native profile."""

    source: Any
    n_periods: int
    psi_a_hat: float
    a_hat: float


def _electric_field_kv_m_to_er_hat(value_kv_m: float) -> float:
    """Normalize ``Er`` using ``PhiHat=e*Phi/TBar`` and ``rHat=r/RBar``."""

    from dkx.units import ELEMENTARY_CHARGE, R_BAR, T_BAR  # noqa: PLC0415

    return float(value_kv_m) * 1000.0 * ELEMENTARY_CHARGE * R_BAR / T_BAR


def _unsupported(path: str, value: Any, expected: str, correction: str) -> None:
    raise CaseValidationError(path, value, expected, correction)


def _analytic_scheme(case: Case) -> int:
    token = str(case.geometry.file).lower().replace("-", "_")
    aliases = {
        "tokamak": 1,
        "lhd_standard": 2,
        "lhd_inward": 3,
        "w7x_standard": 4,
    }
    try:
        return aliases[token]
    except KeyError:
        _unsupported(
            "geometry.file",
            str(case.geometry.file),
            "one of tokamak, lhd_standard, lhd_inward, or w7x_standard for analytic geometry",
            "Use a named built-in analytic equilibrium; VMEC/Boozer case execution is the next route.",
        )
        raise AssertionError("unreachable")


def _validate_native_slice(case: Case) -> None:
    if case.run.workflow not in {"profile", "ambipolar_profile"}:
        _unsupported(
            "run.workflow",
            case.run.workflow,
            "profile or ambipolar_profile",
            "Use a supported native profile workflow.",
        )
    if case.geometry.format == "analytic":
        _analytic_scheme(case)
    if case.physics.magnetic_drifts != "dkes":
        _unsupported(
            "physics.magnetic_drifts",
            case.physics.magnetic_drifts,
            "dkes",
            "Set magnetic_drifts = 'dkes' for the supported local analytic route.",
        )
    if case.physics.phi1 != "off":
        _unsupported(
            "physics.phi1",
            case.physics.phi1,
            "off",
            "Set phi1 = 'off' for this route.",
        )
    expected_field_mode = (
        "ambipolar" if case.run.workflow == "ambipolar_profile" else "prescribed"
    )
    if case.electric_field.mode != expected_field_mode:
        _unsupported(
            "electric_field.mode",
            case.electric_field.mode,
            expected_field_mode,
            f"Use mode = '{expected_field_mode}' with workflow = '{case.run.workflow}'.",
        )
    if expected_field_mode == "prescribed" and case.electric_field.value_kV_m is None:
        _unsupported(
            "electric_field.value_kV_m",
            None,
            "a finite electric field in kV/m",
            "Add value_kV_m = 0.0 for a zero-field calculation.",
        )
    if case.scan is not None:
        _unsupported(
            "scan",
            "configured",
            "no scan for dkx.run(case)",
            "Remove [scan] for a single run, or expand it with `dkx scan CASE`.",
        )
    if case.convergence.enabled and case.run.workflow != "ambipolar_profile":
        _unsupported(
            "convergence.enabled",
            True,
            "false for a prescribed-field profile",
            "Use convergence refinement with workflow = 'ambipolar_profile'. Phase-space refinement of theta, zeta, pitch and speed is a separate study: run `dkx converge CASE`.",
        )
    if (
        case.convergence.retain_legendre_tail
        and case.run.workflow != "ambipolar_profile"
    ):
        _unsupported(
            "convergence.retain_legendre_tail",
            True,
            "false for a prescribed-field profile",
            "Selected-tail retention is currently available with workflow = 'ambipolar_profile'.",
        )
    if case.convergence.enabled:
        supported_observables = {
            "electric_field",
            "particle_flux",
            "heat_flux",
            "parallel_current",
            "bootstrap_current",
            "radial_current",
        }
        unsupported_observables = sorted(
            set(case.convergence.observables) - supported_observables
        )
        if unsupported_observables:
            _unsupported(
                "convergence.observables",
                unsupported_observables,
                f"names drawn from {sorted(supported_observables)}",
                "Choose native ambipolar observables that are available at every retained electric-field solve.",
            )
    if case.parallel.strategy == "batch" or case.parallel.shard:
        _unsupported(
            "parallel",
            {"strategy": case.parallel.strategy, "shard": case.parallel.shard},
            "strategy = 'auto' or 'serial' with no shard axes",
            "Remove explicit sharding for this sequential surface route.",
        )
    if case.solver.reuse != "auto":
        _unsupported(
            "solver.reuse",
            case.solver.reuse,
            "auto",
            "Use reuse = 'auto'; explicit reuse policy arrives with surface-state reuse.",
        )
    if len(case.geometry.surfaces) < 2:
        _unsupported(
            "geometry.surfaces",
            case.geometry.surfaces,
            "at least two surfaces",
            "Give at least two profile locations so density and temperature gradients are defined.",
        )
    if any(value <= 0.0 for value in case.geometry.surfaces) or any(
        right <= left
        for left, right in zip(case.geometry.surfaces, case.geometry.surfaces[1:])
    ):
        _unsupported(
            "geometry.surfaces",
            case.geometry.surfaces,
            "strictly increasing normalized toroidal flux values above zero",
            "Sort the surfaces and omit the magnetic axis, where the radial Jacobian is singular.",
        )


def _profile_matrix(case: Case, attribute: str) -> np.ndarray:
    values = np.asarray(
        [getattr(species, attribute) for species in case.species], dtype=np.float64
    )
    return values.T  # (surface, species)


def _profile_gradients(values: np.ndarray, r_hat: np.ndarray) -> np.ndarray:
    edge_order = 2 if r_hat.size >= 3 else 1
    return np.asarray(
        np.gradient(values, r_hat, axis=0, edge_order=edge_order), dtype=np.float64
    )


def _prepare_geometry(case: Case) -> _GeometryState:
    if case.geometry.format == "analytic":
        scheme = _analytic_scheme(case)
        psi_a_hat, a_hat = {
            1: (0.15596, 0.5585),
            2: (0.5585**2 / 2.0, 0.5585),
            3: (0.5400**2 / 2.0, 0.5400),
            4: (-0.384935, 0.5109),
        }[scheme]
        return _GeometryState(
            source=scheme,
            n_periods={1: 10, 2: 10, 3: 10, 4: 5}[scheme],
            psi_a_hat=psi_a_hat,
            a_hat=a_hat,
        )

    if case.geometry.format == "boozer":
        from dkx.magnetic_geometry import read_native_boozer  # noqa: PLC0415

        data = read_native_boozer(case.geometry_path)
        return _GeometryState(
            source=data,
            n_periods=int(data.header.n_periods),
            psi_a_hat=float(data.header.psi_a_hat),
            a_hat=float(data.header.a_hat),
        )

    from dkx.magnetic_geometry import (  # noqa: PLC0415
        psi_a_hat_from_wout,
        read_vmec_wout,
    )

    wout = read_vmec_wout(case.geometry_path)
    return _GeometryState(
        source=wout,
        n_periods=int(wout.nfp),
        psi_a_hat=psi_a_hat_from_wout(wout),
        a_hat=float(wout.aminor_p),
    )


def _make_grids(case: Case, *, n_periods: int):
    from dkx.phase_space import make_grids  # noqa: PLC0415

    return make_grids(
        n_theta=case.resolution.theta,
        n_zeta=case.resolution.zeta,
        n_xi=case.resolution.pitch,
        n_x=case.resolution.speed,
        n_l=min(4, case.resolution.pitch),
        n_periods=n_periods,
        theta_derivative_scheme=2,
        zeta_derivative_scheme=2,
        magnetic_drift_derivative_scheme=3,
        x_grid_scheme=5,
        x_grid_k=0.0,
        x_max=5.0,
        x_dot_derivative_scheme=0,
        n_xi_for_x_option=case.resolution.pitch_speed_ramp,
        n_xi_for_x_override=case.resolution.pitch_modes_by_speed,
        monoenergetic=False,
    )


def _geometry_context(
    case: Case, grids, surface_index: int, geometry_state: _GeometryState
):
    from dkx.magnetic_geometry import FluxSurfaceGeometry  # noqa: PLC0415

    psi_n = float(case.geometry.surfaces[surface_index])
    r_n = math.sqrt(psi_n)
    if case.geometry.format == "analytic":
        geometry = FluxSurfaceGeometry.from_scheme(
            int(geometry_state.source), theta=grids.theta, zeta=grids.zeta
        )
    elif case.geometry.format == "boozer":
        geometry = FluxSurfaceGeometry.from_boozer_data(
            geometry_state.source,
            theta=grids.theta,
            zeta=grids.zeta,
            r_n_wish=r_n,
            vmec_radial_option=0,
        )
    else:
        geometry = FluxSurfaceGeometry.from_vmec(
            geometry_state.source,
            theta=grids.theta,
            zeta=grids.zeta,
            psi_n_wish=psi_n,
            vmec_radial_option=0,
        )
    return geometry, RadialCoordinates(
        psi_a_hat=geometry_state.psi_a_hat,
        a_hat=geometry_state.a_hat,
        r_n=r_n,
    )


def _make_operator(
    case: Case,
    *,
    surface_index: int,
    n_hat: np.ndarray,
    t_hat: np.ndarray,
    dn_dr_hat: np.ndarray,
    dt_dr_hat: np.ndarray,
    grids,
    geometry_state: _GeometryState,
    electric_field_kv_m: float | None = None,
    force_exb_structure: bool = False,
):
    import jax.numpy as jnp  # noqa: PLC0415

    from dkx.collisions import (  # noqa: PLC0415
        make_fokker_planck_v3_operator,
        make_pitch_angle_scattering_v3_operator,
    )
    from dkx.constants import DEFAULT_DELTA, DEFAULT_NU_N  # noqa: PLC0415
    from dkx.drift_kinetic import KineticOperator  # noqa: PLC0415
    from dkx.phase_space import (  # noqa: PLC0415
        legendre_coupling_lower,
        legendre_coupling_upper,
    )

    # nu_n is proportional to the Coulomb logarithm (units.reference_nu_n, where
    # ln-Lambda enters through nuBar), so the case's value scales the pinned
    # default rather than replacing it. At the default 17.0 this is exactly
    # DEFAULT_NU_N, so every existing result is unchanged bit for bit.
    nu_n = DEFAULT_NU_N * (float(case.physics.coulomb_logarithm) / 17.0)

    geometry, radial = _geometry_context(case, grids, surface_index, geometry_state)
    z_s = jnp.asarray([species.charge for species in case.species], dtype=jnp.float64)
    m_hat = jnp.asarray(
        [
            species.mass_amu * _ATOMIC_MASS_KG / _PROTON_MASS_KG
            for species in case.species
        ],
        dtype=jnp.float64,
    )
    n_surface = jnp.asarray(n_hat[surface_index], dtype=jnp.float64)
    t_surface = jnp.asarray(t_hat[surface_index], dtype=jnp.float64)
    dn_dpsi = jnp.asarray(
        radial.d_dr_hat_to_d_dpsi_hat * dn_dr_hat[surface_index], dtype=jnp.float64
    )
    dt_dpsi = jnp.asarray(
        radial.d_dr_hat_to_d_dpsi_hat * dt_dr_hat[surface_index], dtype=jnp.float64
    )
    pas = None
    fp = None
    if case.physics.collisions == "pitch_angle_scattering":
        pas = make_pitch_angle_scattering_v3_operator(
            x=grids.x,
            z_s=z_s,
            m_hats=m_hat,
            n_hats=n_surface,
            t_hats=t_surface,
            nu_n=nu_n,
            n_xi_for_x=grids.n_xi_for_x,
            n_xi=grids.n_xi,
        )
        constraint_scheme = 2
    else:
        fp = make_fokker_planck_v3_operator(
            x=np.asarray(grids.x),
            x_weights=np.asarray(grids.x_weights),
            ddx=np.asarray(grids.ddx),
            d2dx2=np.asarray(grids.d2dx2),
            x_grid_k=0.0,
            z_s=np.asarray(z_s),
            m_hats=np.asarray(m_hat),
            n_hats=np.asarray(n_surface),
            t_hats=np.asarray(t_surface),
            nu_n=nu_n,
            krook=0.0,
            n_xi=grids.n_xi,
            nl=grids.n_l,
            n_xi_for_x=np.asarray(grids.n_xi_for_x),
            strict_parity=len(case.species) > 1,
        )
        constraint_scheme = 1

    field_kv_m = (
        case.electric_field.value_kV_m
        if electric_field_kv_m is None
        else electric_field_kv_m
    )
    if field_kv_m is None:
        raise ValueError("an electric-field value is required to build the operator")
    er_hat = _electric_field_kv_m_to_er_hat(field_kv_m)
    dphi = radial.d_dr_hat_to_d_dpsi_hat * (-er_hat)
    fsab_hat2 = geometry.fsab_hat2(
        theta_weights=grids.theta_weights, zeta_weights=grids.zeta_weights
    )
    op = KineticOperator(
        n_species=len(case.species),
        n_x=grids.n_x,
        n_xi=grids.n_xi,
        n_theta=grids.n_theta,
        n_zeta=grids.n_zeta,
        rhs_mode=1,
        constraint_scheme=constraint_scheme,
        point_at_x0=False,
        use_dkes_exb=True,
        with_exb=force_exb_structure or er_hat != 0.0,
        with_er_xidot=False,
        with_er_xdot=False,
        x=grids.x,
        x_weights=grids.x_weights,
        ddx=grids.ddx,
        ddtheta=grids.ddtheta,
        ddzeta=grids.ddzeta,
        theta_weights=grids.theta_weights,
        zeta_weights=grids.zeta_weights,
        n_xi_for_x=grids.n_xi_for_x,
        xi_coupling_lower=jnp.asarray(legendre_coupling_lower(grids.n_xi)),
        xi_coupling_upper=jnp.asarray(legendre_coupling_upper(grids.n_xi)),
        b_hat=geometry.b_hat,
        db_hat_dtheta=geometry.db_hat_dtheta,
        db_hat_dzeta=geometry.db_hat_dzeta,
        d_hat=geometry.d_hat,
        b_hat_sup_theta=geometry.b_hat_sup_theta,
        b_hat_sup_zeta=geometry.b_hat_sup_zeta,
        b_hat_sub_theta=geometry.b_hat_sub_theta,
        b_hat_sub_zeta=geometry.b_hat_sub_zeta,
        fsab_hat2=jnp.asarray(fsab_hat2, dtype=jnp.float64),
        z_s=z_s,
        m_hat=m_hat,
        t_hat=t_surface,
        n_hat=n_surface,
        dn_hat_dpsi_hat=dn_dpsi,
        dt_hat_dpsi_hat=dt_dpsi,
        alpha=jnp.asarray(1.0, dtype=jnp.float64),
        delta=jnp.asarray(DEFAULT_DELTA, dtype=jnp.float64),
        dphi_hat_dpsi_hat=jnp.asarray(dphi, dtype=jnp.float64),
        dphi_hat_dpsi_hat_kinetic=jnp.asarray(dphi, dtype=jnp.float64),
        e_parallel_hat=jnp.asarray(0.0, dtype=jnp.float64),
        e_parallel_hat_spec=jnp.zeros((len(case.species),), dtype=jnp.float64),
        pas=pas,
        fp=fp,
    )
    return op, grids, geometry, radial


def _route_name(method: str) -> str:
    return {
        "auto": "auto",
        "structured_direct": "block_tridiagonal",
        "recycled_krylov": "gmres",
        "sparse_direct_referee": "direct",
    }[method]


def _sha256(path_or_token: Path) -> str:
    if path_or_token.exists() and path_or_token.is_file():
        digest = hashlib.sha256()
        with path_or_token.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    return hashlib.sha256(str(path_or_token).encode("utf-8")).hexdigest()


def _peak_host_memory_bytes() -> int:
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak if sys.platform == "darwin" else peak * 1024


def _total_host_memory_bytes() -> int:
    """Best-effort physical host memory for the case's fractional budget."""

    try:
        return int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, ValueError):
        return max(_peak_host_memory_bytes(), 1024**3)


def _ambipolar_result_arrays(
    surface_results, n_species: int, speed_nodes: np.ndarray | None = None
):
    """Pad ragged scan/root evidence into named NetCDF-friendly dimensions."""

    n_surface = len(surface_results)
    n_evaluation = max(len(result.evaluations) for result in surface_results)
    n_solver_attempt = max(
        1,
        max(
            len(evaluation.solver_attempts)
            for result in surface_results
            for evaluation in result.evaluations
        ),
    )
    n_root = max(1, max(len(result.roots) for result in surface_results))
    scan_field = np.full((n_surface, n_evaluation), np.nan)
    scan_current = np.full((n_surface, n_evaluation), np.nan)
    scan_residual = np.full((n_surface, n_evaluation), np.nan)
    scan_stage = np.full((n_surface, n_evaluation), "", dtype=object)
    scan_reason = np.full((n_surface, n_evaluation), "", dtype=object)
    scan_level = np.full((n_surface, n_evaluation), -1, dtype=np.int64)
    scan_attempt_count = np.zeros((n_surface, n_evaluation), dtype=np.int64)
    scan_attempt_requested = np.full(
        (n_surface, n_evaluation, n_solver_attempt), "", dtype=object
    )
    scan_attempt_executed = np.full(
        (n_surface, n_evaluation, n_solver_attempt), "", dtype=object
    )
    scan_attempt_residual = np.full((n_surface, n_evaluation, n_solver_attempt), np.nan)
    scan_attempt_accepted = np.zeros(
        (n_surface, n_evaluation, n_solver_attempt), dtype=np.int8
    )
    scan_attempt_reason = np.full(
        (n_surface, n_evaluation, n_solver_attempt), "", dtype=object
    )
    scan_particle = np.full((n_surface, n_evaluation, n_species), np.nan)
    scan_heat = np.full((n_surface, n_evaluation, n_species), np.nan)
    speed_diagnostics = [
        evaluation
        for result in surface_results
        for evaluation in result.evaluations
        if evaluation.particle_flux_m2_s_vs_speed is not None
        and evaluation.heat_flux_w_m2_vs_speed is not None
    ]
    n_speed = (
        int(np.asarray(speed_diagnostics[0].particle_flux_m2_s_vs_speed).shape[0])
        if speed_diagnostics
        else 0
    )
    scan_particle_vs_speed = np.full(
        (n_surface, n_evaluation, n_speed, n_species), np.nan
    )
    scan_heat_vs_speed = np.full((n_surface, n_evaluation, n_speed, n_species), np.nan)
    tail_diagnostics = [
        evaluation
        for result in surface_results
        for evaluation in result.evaluations
        if evaluation.legendre_tail_relative_l2 is not None
    ]
    tail_bound_diagnostics = [
        evaluation
        for result in surface_results
        for evaluation in result.evaluations
        if evaluation.legendre_tail_relative_l2_upper_bound is not None
    ]
    scan_legendre_tail = np.full((n_surface, n_evaluation, n_speed, n_species), np.nan)
    scan_legendre_tail_upper_bound = np.full(
        (n_surface, n_evaluation, n_speed, n_species), np.nan
    )
    scan_parallel = np.full((n_surface, n_evaluation), np.nan)
    root_field = np.full((n_surface, n_root), np.nan)
    root_current = np.full((n_surface, n_root), np.nan)
    root_slope = np.full((n_surface, n_root), np.nan)
    root_type = np.full((n_surface, n_root), "", dtype=object)
    root_bracket = np.full((n_surface, n_root, 2), np.nan)
    root_movement = np.full((n_surface, n_root), np.nan)
    root_observable_movement = np.full((n_surface, n_root), np.nan)
    root_bracket_width = np.full((n_surface, n_root), np.nan)
    root_branch_id = np.full((n_surface, n_root), "", dtype=object)
    root_count = np.zeros((n_surface,), dtype=np.int64)
    selected_root = np.full((n_surface,), -1, dtype=np.int64)
    selected_branch = np.full((n_surface,), "", dtype=object)
    selection_reason = np.full((n_surface,), "", dtype=object)
    status = np.empty((n_surface,), dtype=object)
    chunk_size = np.empty((n_surface,), dtype=np.int64)
    batch_chunks = np.empty((n_surface,), dtype=np.int64)
    refinement_status = np.empty((n_surface,), dtype=object)
    evaluation_budget = np.empty((n_surface,), dtype=np.int64)
    search_strategy = np.empty((n_surface,), dtype=object)
    search_scope = np.empty((n_surface,), dtype=object)
    n_refinement = max(len(result.refinement) for result in surface_results)
    refinement_level = np.full((n_surface, n_refinement), -1, dtype=np.int64)
    refinement_search_count = np.full((n_surface, n_refinement), -1, dtype=np.int64)
    refinement_total_count = np.full((n_surface, n_refinement), -1, dtype=np.int64)
    refinement_root_count = np.full((n_surface, n_refinement), -1, dtype=np.int64)
    refinement_root_movement = np.full((n_surface, n_refinement), np.nan)
    refinement_observable_movement = np.full((n_surface, n_refinement), np.nan)
    refinement_bracket_width = np.full((n_surface, n_refinement), np.nan)
    refinement_converged = np.zeros((n_surface, n_refinement), dtype=np.int8)
    n_event = max(1, max(len(result.branch_events) for result in surface_results))
    event_kind = np.full((n_surface, n_event), "", dtype=object)
    event_branch_id = np.full((n_surface, n_event, 2), "", dtype=object)
    event_root_index = np.full((n_surface, n_event, 2), -1, dtype=np.int64)
    event_field = np.full((n_surface, n_event), np.nan)
    event_detail = np.full((n_surface, n_event), "", dtype=object)
    event_nonsmooth = np.zeros((n_surface, n_event), dtype=np.int8)
    event_count = np.zeros((n_surface,), dtype=np.int64)
    for surface_index, result in enumerate(surface_results):
        status[surface_index] = result.status
        root_count[surface_index] = len(result.roots)
        selected_root[surface_index] = (
            -1 if result.selected_root is None else result.selected_root
        )
        selected_branch[surface_index] = result.selected_branch_id
        selection_reason[surface_index] = result.selection_reason
        chunk_size[surface_index] = result.batch_chunk_size
        batch_chunks[surface_index] = result.batch_chunks
        refinement_status[surface_index] = result.refinement_status
        evaluation_budget[surface_index] = result.evaluation_budget
        search_strategy[surface_index] = result.search_strategy
        search_scope[surface_index] = result.search_scope
        for evaluation_index, evaluation in enumerate(result.evaluations):
            scan_field[surface_index, evaluation_index] = evaluation.electric_field_kv_m
            scan_current[surface_index, evaluation_index] = (
                evaluation.radial_current_a_m2
            )
            scan_residual[surface_index, evaluation_index] = evaluation.residual_norm
            scan_stage[surface_index, evaluation_index] = evaluation.stage
            scan_reason[surface_index, evaluation_index] = evaluation.reason
            scan_level[surface_index, evaluation_index] = evaluation.refinement_level
            scan_attempt_count[surface_index, evaluation_index] = len(
                evaluation.solver_attempts
            )
            for attempt_index, attempt in enumerate(evaluation.solver_attempts):
                scan_attempt_requested[
                    surface_index, evaluation_index, attempt_index
                ] = attempt.requested_method
                scan_attempt_executed[
                    surface_index, evaluation_index, attempt_index
                ] = attempt.executed_method
                scan_attempt_residual[
                    surface_index, evaluation_index, attempt_index
                ] = attempt.residual_norm
                scan_attempt_accepted[
                    surface_index, evaluation_index, attempt_index
                ] = attempt.accepted
                scan_attempt_reason[surface_index, evaluation_index, attempt_index] = (
                    attempt.reason
                )
            scan_particle[surface_index, evaluation_index] = (
                evaluation.particle_flux_m2_s
            )
            scan_heat[surface_index, evaluation_index] = evaluation.heat_flux_w_m2
            if evaluation.particle_flux_m2_s_vs_speed is not None:
                scan_particle_vs_speed[surface_index, evaluation_index] = (
                    evaluation.particle_flux_m2_s_vs_speed
                )
            if evaluation.heat_flux_w_m2_vs_speed is not None:
                scan_heat_vs_speed[surface_index, evaluation_index] = (
                    evaluation.heat_flux_w_m2_vs_speed
                )
            if evaluation.legendre_tail_relative_l2 is not None:
                scan_legendre_tail[surface_index, evaluation_index] = (
                    evaluation.legendre_tail_relative_l2
                )
            if evaluation.legendre_tail_relative_l2_upper_bound is not None:
                scan_legendre_tail_upper_bound[surface_index, evaluation_index] = (
                    evaluation.legendre_tail_relative_l2_upper_bound
                )
            scan_parallel[surface_index, evaluation_index] = (
                evaluation.parallel_current_a_t_m2
            )
        for root_index, root in enumerate(result.roots):
            root_field[surface_index, root_index] = root.electric_field_kv_m
            root_current[surface_index, root_index] = root.radial_current_a_m2
            root_slope[surface_index, root_index] = root.slope_a_m2_per_kv_m
            root_type[surface_index, root_index] = root.root_type
            root_bracket[surface_index, root_index] = root.bracket_kv_m
            root_movement[surface_index, root_index] = root.movement_kv_m
            root_observable_movement[surface_index, root_index] = (
                root.observable_relative_movement
            )
            root_bracket_width[surface_index, root_index] = (
                root.bracket_kv_m[1] - root.bracket_kv_m[0]
            )
            root_branch_id[surface_index, root_index] = root.branch_id
        for refinement_index, evidence in enumerate(result.refinement):
            refinement_level[surface_index, refinement_index] = evidence.level
            refinement_search_count[surface_index, refinement_index] = (
                evidence.search_evaluations
            )
            refinement_total_count[surface_index, refinement_index] = (
                evidence.total_evaluations
            )
            refinement_root_count[surface_index, refinement_index] = evidence.root_count
            refinement_root_movement[surface_index, refinement_index] = (
                evidence.root_movement_kv_m
            )
            refinement_observable_movement[surface_index, refinement_index] = (
                evidence.observable_relative_movement
            )
            refinement_bracket_width[surface_index, refinement_index] = (
                evidence.max_bracket_width_kv_m
            )
            refinement_converged[surface_index, refinement_index] = evidence.converged
        event_count[surface_index] = len(result.branch_events)
        for event_index, event in enumerate(result.branch_events):
            event_kind[surface_index, event_index] = event.kind
            event_field[surface_index, event_index] = event.electric_field_kv_m
            event_detail[surface_index, event_index] = event.detail
            event_nonsmooth[surface_index, event_index] = event.nonsmooth
            for participant, branch_id in enumerate(event.branch_ids):
                event_branch_id[surface_index, event_index, participant] = branch_id
            for participant, root_index in enumerate(event.root_indices):
                event_root_index[surface_index, event_index, participant] = root_index
    arrays = {
        "evaluation": np.arange(n_evaluation, dtype=np.int64),
        "solver_attempt": np.arange(n_solver_attempt, dtype=np.int64),
        "root": np.arange(n_root, dtype=np.int64),
        "refinement": np.arange(n_refinement, dtype=np.int64),
        "branch_event": np.arange(n_event, dtype=np.int64),
        "branch_event_participant": np.asarray(["primary", "secondary"], dtype=object),
        "bracket_endpoint": np.asarray(["minimum", "maximum"], dtype=object),
        "evaluation_electric_field_kV_m": scan_field,
        "radial_current_A_m2": scan_current,
        "evaluation_primal_residual": scan_residual,
        "evaluation_stage": scan_stage,
        "evaluation_reason": scan_reason,
        "evaluation_refinement_level": scan_level,
        "evaluation_solver_attempt_count": scan_attempt_count,
        "evaluation_solver_attempt_requested_method": scan_attempt_requested,
        "evaluation_solver_attempt_executed_method": scan_attempt_executed,
        "evaluation_solver_attempt_residual": scan_attempt_residual,
        "evaluation_solver_attempt_accepted": scan_attempt_accepted,
        "evaluation_solver_attempt_reason": scan_attempt_reason,
        "evaluation_particle_flux_m2_s": scan_particle,
        "evaluation_heat_flux_W_m2": scan_heat,
        "evaluation_parallel_current_A_T_m2": scan_parallel,
        "ambipolar_root_kV_m": root_field,
        "ambipolar_root_current_A_m2": root_current,
        "ambipolar_root_slope_A_m2_per_kV_m": root_slope,
        "ambipolar_root_type": root_type,
        "ambipolar_root_bracket_kV_m": root_bracket,
        "ambipolar_root_movement_kV_m": root_movement,
        "ambipolar_root_observable_relative_movement": root_observable_movement,
        "ambipolar_root_final_bracket_width_kV_m": root_bracket_width,
        "ambipolar_root_branch_id": root_branch_id,
        "ambipolar_root_count": root_count,
        "selected_ambipolar_root": selected_root,
        "selected_ambipolar_branch": selected_branch,
        "ambipolar_selection_reason": selection_reason,
        "ambipolar_status": status,
        "electric_field_batch_chunk_size": chunk_size,
        "electric_field_batch_chunks": batch_chunks,
        "ambipolar_refinement_status": refinement_status,
        "ambipolar_evaluation_budget": evaluation_budget,
        "ambipolar_search_strategy": search_strategy,
        "ambipolar_search_scope": search_scope,
        "ambipolar_refinement_level": refinement_level,
        "ambipolar_refinement_search_evaluations": refinement_search_count,
        "ambipolar_refinement_total_evaluations": refinement_total_count,
        "ambipolar_refinement_root_count": refinement_root_count,
        "ambipolar_refinement_root_movement_kV_m": refinement_root_movement,
        "ambipolar_refinement_observable_relative_movement": refinement_observable_movement,
        "ambipolar_refinement_max_bracket_width_kV_m": refinement_bracket_width,
        "ambipolar_refinement_converged": refinement_converged,
        "ambipolar_branch_event_kind": event_kind,
        "ambipolar_branch_event_branch_id": event_branch_id,
        "ambipolar_branch_event_root_index": event_root_index,
        "ambipolar_branch_event_electric_field_kV_m": event_field,
        "ambipolar_branch_event_detail": event_detail,
        "ambipolar_branch_event_nonsmooth": event_nonsmooth,
        "ambipolar_branch_event_count": event_count,
        "ambipolar_nonsmooth_event": np.any(event_nonsmooth, axis=1).astype(np.int8),
    }
    dimensions = {
        "evaluation": ("evaluation",),
        "solver_attempt": ("solver_attempt",),
        "root": ("root",),
        "refinement": ("refinement",),
        "branch_event": ("branch_event",),
        "branch_event_participant": ("branch_event_participant",),
        "bracket_endpoint": ("bracket_endpoint",),
        "evaluation_electric_field_kV_m": ("surface", "evaluation"),
        "radial_current_A_m2": ("surface", "evaluation"),
        "evaluation_primal_residual": ("surface", "evaluation"),
        "evaluation_stage": ("surface", "evaluation"),
        "evaluation_reason": ("surface", "evaluation"),
        "evaluation_refinement_level": ("surface", "evaluation"),
        "evaluation_solver_attempt_count": ("surface", "evaluation"),
        "evaluation_solver_attempt_requested_method": (
            "surface",
            "evaluation",
            "solver_attempt",
        ),
        "evaluation_solver_attempt_executed_method": (
            "surface",
            "evaluation",
            "solver_attempt",
        ),
        "evaluation_solver_attempt_residual": (
            "surface",
            "evaluation",
            "solver_attempt",
        ),
        "evaluation_solver_attempt_accepted": (
            "surface",
            "evaluation",
            "solver_attempt",
        ),
        "evaluation_solver_attempt_reason": (
            "surface",
            "evaluation",
            "solver_attempt",
        ),
        "evaluation_particle_flux_m2_s": ("surface", "evaluation", "species"),
        "evaluation_heat_flux_W_m2": ("surface", "evaluation", "species"),
        "evaluation_parallel_current_A_T_m2": ("surface", "evaluation"),
        "ambipolar_root_kV_m": ("surface", "root"),
        "ambipolar_root_current_A_m2": ("surface", "root"),
        "ambipolar_root_slope_A_m2_per_kV_m": ("surface", "root"),
        "ambipolar_root_type": ("surface", "root"),
        "ambipolar_root_bracket_kV_m": (
            "surface",
            "root",
            "bracket_endpoint",
        ),
        "ambipolar_root_movement_kV_m": ("surface", "root"),
        "ambipolar_root_observable_relative_movement": ("surface", "root"),
        "ambipolar_root_final_bracket_width_kV_m": ("surface", "root"),
        "ambipolar_root_branch_id": ("surface", "root"),
        "ambipolar_root_count": ("surface",),
        "selected_ambipolar_root": ("surface",),
        "selected_ambipolar_branch": ("surface",),
        "ambipolar_selection_reason": ("surface",),
        "ambipolar_status": ("surface",),
        "electric_field_batch_chunk_size": ("surface",),
        "electric_field_batch_chunks": ("surface",),
        "ambipolar_refinement_status": ("surface",),
        "ambipolar_evaluation_budget": ("surface",),
        "ambipolar_search_strategy": ("surface",),
        "ambipolar_search_scope": ("surface",),
        "ambipolar_refinement_level": ("surface", "refinement"),
        "ambipolar_refinement_search_evaluations": ("surface", "refinement"),
        "ambipolar_refinement_total_evaluations": ("surface", "refinement"),
        "ambipolar_refinement_root_count": ("surface", "refinement"),
        "ambipolar_refinement_root_movement_kV_m": ("surface", "refinement"),
        "ambipolar_refinement_observable_relative_movement": (
            "surface",
            "refinement",
        ),
        "ambipolar_refinement_max_bracket_width_kV_m": (
            "surface",
            "refinement",
        ),
        "ambipolar_refinement_converged": ("surface", "refinement"),
        "ambipolar_branch_event_kind": ("surface", "branch_event"),
        "ambipolar_branch_event_branch_id": (
            "surface",
            "branch_event",
            "branch_event_participant",
        ),
        "ambipolar_branch_event_root_index": (
            "surface",
            "branch_event",
            "branch_event_participant",
        ),
        "ambipolar_branch_event_electric_field_kV_m": (
            "surface",
            "branch_event",
        ),
        "ambipolar_branch_event_detail": ("surface", "branch_event"),
        "ambipolar_branch_event_nonsmooth": ("surface", "branch_event"),
        "ambipolar_branch_event_count": ("surface",),
        "ambipolar_nonsmooth_event": ("surface",),
    }
    if n_speed:
        nodes = (
            np.asarray(speed_nodes, dtype=np.float64)
            if speed_nodes is not None
            else np.arange(n_speed, dtype=np.float64)
        )
        if nodes.shape != (n_speed,):
            raise ValueError(
                "native ambipolar speed coordinate does not match retained "
                f"diagnostics: expected {(n_speed,)}, got {nodes.shape}"
            )
        arrays.update(
            {
                "speed": np.arange(n_speed, dtype=np.int64),
                "speed_v_th": nodes,
                "evaluation_particle_flux_m2_s_vs_speed": scan_particle_vs_speed,
                "evaluation_heat_flux_W_m2_vs_speed": scan_heat_vs_speed,
            }
        )
        dimensions.update(
            {
                "speed": ("speed",),
                "speed_v_th": ("speed",),
                "evaluation_particle_flux_m2_s_vs_speed": (
                    "surface",
                    "evaluation",
                    "speed",
                    "species",
                ),
                "evaluation_heat_flux_W_m2_vs_speed": (
                    "surface",
                    "evaluation",
                    "speed",
                    "species",
                ),
            }
        )
        if tail_diagnostics:
            arrays["evaluation_legendre_tail_relative_l2"] = scan_legendre_tail
            dimensions["evaluation_legendre_tail_relative_l2"] = (
                "surface",
                "evaluation",
                "speed",
                "species",
            )
        if tail_bound_diagnostics:
            arrays["evaluation_legendre_tail_relative_l2_upper_bound"] = (
                scan_legendre_tail_upper_bound
            )
            dimensions["evaluation_legendre_tail_relative_l2_upper_bound"] = (
                "surface",
                "evaluation",
                "speed",
                "species",
            )
    return arrays, dimensions


def _prepare_profile(case: Case):
    """Shared native geometry, grids and physical-profile differentiation."""
    r_n = np.sqrt(np.asarray(case.geometry.surfaces, dtype=np.float64))
    geometry_state = _prepare_geometry(case)
    grids = _make_grids(case, n_periods=geometry_state.n_periods)
    r_hat = geometry_state.a_hat * r_n
    density_m3 = _profile_matrix(case, "density_m3")
    temperature_keV = _profile_matrix(case, "temperature_keV")
    n_hat = density_m3 / 1.0e20
    t_hat = temperature_keV
    dn_dr_hat = _profile_gradients(n_hat, r_hat)
    dt_dr_hat = _profile_gradients(t_hat, r_hat)

    return geometry_state, grids, density_m3, temperature_keV, dn_dr_hat, dt_dr_hat


def run_case(case: Case, *, out: str | Path | None = None, emit=None) -> Result:
    """Execute the supported native profile route and return a Result."""

    _validate_native_slice(case)
    from dkx import __version__  # noqa: PLC0415
    from dkx.run import profile_moments_from_operator  # noqa: PLC0415
    from dkx.solve import solve  # noqa: PLC0415
    from dkx.units import HEAT_FLUX, PARALLEL_CURRENT, PARTICLE_FLUX  # noqa: PLC0415
    import jax  # noqa: PLC0415
    import jaxlib  # noqa: PLC0415

    total_start = time.perf_counter()
    surfaces = np.asarray(case.geometry.surfaces, dtype=np.float64)
    r_n = np.sqrt(surfaces)
    geometry_state, grids, density_m3, temperature_keV, dn_dr_hat, dt_dr_hat = _prepare_profile(case)
    n_hat, t_hat = density_m3 / 1.0e20, temperature_keV

    shape = (surfaces.size, len(case.species))
    particle_flux = np.empty(shape, dtype=np.float64)
    heat_flux = np.empty(shape, dtype=np.float64)
    current = np.empty((surfaces.size,), dtype=np.float64)
    residuals = np.empty((surfaces.size,), dtype=np.float64)
    iterations = np.empty((surfaces.size,), dtype=np.int64)
    solve_seconds = np.empty((surfaces.size,), dtype=np.float64)
    retained_operator = None
    selected_routes: list[str] = []
    ambipolar_surfaces = []
    previous_root_kv_m: float | None = None

    progress = emit if emit is not None else (print if case.run.progress else None)
    solved = None
    for index, surface in enumerate(surfaces):
        if progress is not None:
            progress(f"surface {index + 1}/{len(surfaces)}: psi_N={surface:.6g}")
        ambipolar = case.run.workflow == "ambipolar_profile"
        op, _grids, _geometry, radial = _make_operator(
            case,
            surface_index=index,
            n_hat=n_hat,
            t_hat=t_hat,
            dn_dr_hat=dn_dr_hat,
            dt_dr_hat=dt_dr_hat,
            grids=grids,
            geometry_state=geometry_state,
            electric_field_kv_m=1.0 if ambipolar else None,
            force_exb_structure=ambipolar,
        )
        if ambipolar:
            from dkx.er import ErProblem  # noqa: PLC0415
            from dkx.workflows.ambipolar_native import (  # noqa: PLC0415
                solve_native_ambipolar_surface,
            )

            bounds = case.electric_field.search_kV_m
            assert bounds is not None
            problem = ErProblem(
                operator=op,
                er_units="kV/m",
                dphi_per_er=float(np.asarray(op.dphi_hat_dpsi_hat_kinetic).reshape(())),
                z_s=np.asarray(op.z_s, dtype=np.float64).reshape((-1,)),
                er_initial=0.0,
                er_min=float(bounds[0]),
                er_max=float(bounds[1]),
                solve_method=_route_name(case.solver.method),
                tol=case.solver.relative_tolerance,
            )
            surface_result = solve_native_ambipolar_surface(
                problem,
                electric_field_bounds_kv_m=bounds,
                search_points=case.electric_field.search_points,
                root_tolerance_kv_m=case.electric_field.root_tolerance_kV_m,
                max_root_iterations=case.electric_field.max_root_iterations,
                find_all_roots=case.electric_field.find_all_roots,
                previous_root_kv_m=(
                    previous_root_kv_m
                    if case.electric_field.continue_branches
                    else None
                ),
                # SFINCS diagnostics multiply a psiHat-directed flux by
                # d(rHat)/d(psiHat) to report the physical radial flux.
                radial_factor=radial.d_dr_hat_to_d_dpsi_hat,
                solve_method=_route_name(case.solver.method),
                solve_tolerance=case.solver.relative_tolerance,
                memory_budget_gb=(
                    case.solver.memory_fraction * _total_host_memory_bytes() / (1024**3)
                ),
                convergence_enabled=case.convergence.enabled,
                convergence_observables=case.convergence.observables,
                convergence_relative_tolerance=case.convergence.relative_tolerance,
                max_refinements=case.convergence.max_refinements,
                retain_legendre_tail=case.convergence.retain_legendre_tail,
                seed_brackets_kv_m=(
                    case.electric_field.seed_brackets_kV_m[index]
                    if case.electric_field.search_strategy == "seeded_brackets"
                    and case.electric_field.seed_brackets_kV_m is not None
                    else None
                ),
            )
            ambipolar_surfaces.append(surface_result)
            selected = surface_result.selected
            particle_flux[index] = selected.particle_flux_m2_s
            heat_flux[index] = selected.heat_flux_w_m2
            current[index] = selected.parallel_current_a_t_m2
            residuals[index] = selected.residual_norm
            iterations[index] = len(surface_result.evaluations)
            solve_seconds[index] = surface_result.solve_seconds
            selected_routes.append(problem.solve_method)
            if surface_result.selected_root is not None:
                previous_root_kv_m = surface_result.roots[
                    surface_result.selected_root
                ].electric_field_kv_m
            retained_operator = op
            continue
        solve_start = time.perf_counter()
        solved = solve(
            op,
            op.rhs(),
            method=_route_name(case.solver.method),
            tol=case.solver.relative_tolerance,
            device=None if case.run.device == "auto" else case.run.device,
            tier1_memory_budget_gb=case.solver.memory_fraction
            * _total_host_memory_bytes()
            / (1024**3),
        )
        solve_seconds[index] = time.perf_counter() - solve_start
        if not solved.converged:
            raise RuntimeError(
                f"native profile solve did not converge at geometry.surfaces[{index}]={surface}; "
                f"route={solved.method}, residuals={np.asarray(solved.residual_norms)!r}"
            )
        state = np.asarray(solved.x, dtype=np.float64).reshape((-1,))
        moments = profile_moments_from_operator(op, state)
        # diagnostics.F90 uses d(rHat)/d(psiHat), not its inverse, when
        # converting particleFlux/heatFlux from psiHat to rHat.
        radial_factor = radial.d_dr_hat_to_d_dpsi_hat
        particle_flux[index] = (
            np.asarray(moments["particleFlux_vm_psiHat"], dtype=np.float64)
            * radial_factor
            * PARTICLE_FLUX
        )
        heat_flux[index] = (
            np.asarray(moments["heatFlux_vm_psiHat"], dtype=np.float64)
            * radial_factor
            * HEAT_FLUX
        )
        current[index] = float(np.asarray(moments["FSABjHat"])) * PARALLEL_CURRENT
        norms = np.atleast_1d(np.asarray(solved.residual_norms, dtype=np.float64))
        residuals[index] = float(np.max(norms))
        iterations[index] = 0 if solved.iterations is None else int(solved.iterations)
        selected_routes.append(str(solved.method))
        retained_operator = op

    if ambipolar_surfaces:
        from dkx.workflows.ambipolar_native import (  # noqa: PLC0415
            continue_ambipolar_branches,
        )

        bounds = case.electric_field.search_kV_m
        assert bounds is not None
        ambipolar_surfaces = list(
            continue_ambipolar_branches(
                ambipolar_surfaces,
                surfaces=surfaces,
                electric_field_bounds_kv_m=bounds,
                continue_selection=case.electric_field.continue_branches,
            )
        )
        for index, surface_result in enumerate(ambipolar_surfaces):
            selected = surface_result.selected
            particle_flux[index] = selected.particle_flux_m2_s
            heat_flux[index] = selected.heat_flux_w_m2
            current[index] = selected.parallel_current_a_t_m2
            residuals[index] = selected.residual_norm

    total_seconds = time.perf_counter() - total_start
    output_path = (
        Path(out).expanduser().resolve()
        if out is not None
        else (case.base_directory / case.output.file).resolve()
    )
    if solved is not None:
        device = solved.x.device
    else:
        device = jax.devices()[0]
    selected_electric_field = (
        np.asarray(
            [result.selected.electric_field_kv_m for result in ambipolar_surfaces],
            dtype=np.float64,
        )
        if ambipolar_surfaces
        else np.full((surfaces.size,), case.electric_field.value_kV_m)
    )
    arrays = {
        "surface": surfaces,
        "r_N": r_n,
        "species": np.asarray([species.name for species in case.species], dtype=object),
        "charge_e": np.asarray([species.charge for species in case.species]),
        "mass_amu": np.asarray([species.mass_amu for species in case.species]),
        "density_m3": density_m3,
        "temperature_keV": temperature_keV,
        "electric_field_kV_m": selected_electric_field,
        "particle_flux_m2_s": particle_flux,
        "heat_flux_W_m2": heat_flux,
        "parallel_current_A_T_m2": current,
        "primal_residual": residuals,
        "solver_iterations": iterations,
        "solve_time_s": solve_seconds,
    }
    dimensions = {
        "surface": ("surface",),
        "r_N": ("surface",),
        "species": ("species",),
        "charge_e": ("species",),
        "mass_amu": ("species",),
        "density_m3": ("surface", "species"),
        "temperature_keV": ("surface", "species"),
        "electric_field_kV_m": ("surface",),
        "particle_flux_m2_s": ("surface", "species"),
        "heat_flux_W_m2": ("surface", "species"),
        "parallel_current_A_T_m2": ("surface",),
        "primal_residual": ("surface",),
        "solver_iterations": ("surface",),
        "solve_time_s": ("surface",),
    }
    if ambipolar_surfaces:
        ambipolar_arrays, ambipolar_dimensions = _ambipolar_result_arrays(
            ambipolar_surfaces,
            len(case.species),
            speed_nodes=np.asarray(retained_operator.x, dtype=np.float64),
        )
        arrays.update(ambipolar_arrays)
        dimensions.update(ambipolar_dimensions)
    reported_residual = (
        float(np.nanmax(arrays["evaluation_primal_residual"]))
        if ambipolar_surfaces
        else float(np.max(residuals))
    )
    route_set = sorted(set(selected_routes))
    ambipolar_solver_attempts = [
        attempt
        for surface_result in ambipolar_surfaces
        for evaluation in surface_result.evaluations
        for attempt in evaluation.solver_attempts
    ]
    executed_route_counts = {
        route: sum(
            attempt.executed_method == route for attempt in ambipolar_solver_attempts
        )
        for route in sorted(
            {attempt.executed_method for attempt in ambipolar_solver_attempts}
        )
    }
    recovery_count = sum(
        attempt.reason == "automatic_true_residual_recovery"
        for attempt in ambipolar_solver_attempts
    )
    legendre_tail_retained = any(
        evaluation.legendre_tail_relative_l2 is not None
        for surface_result in ambipolar_surfaces
        for evaluation in surface_result.evaluations
    )
    legendre_tail_upper_bound_retained = any(
        evaluation.legendre_tail_relative_l2_upper_bound is not None
        for surface_result in ambipolar_surfaces
        for evaluation in surface_result.evaluations
    )
    metadata = {
        "canonical_case": case.to_dict(),
        "converged": True,
        "solver_route": route_set[0] if len(route_set) == 1 else route_set,
        "route_reason": "selected from operator structure and requested native solver method",
        "residual_norm": reported_residual,
        "iterations": int(np.sum(iterations)),
        "ambipolar_all_surfaces_bracketed": (
            all(result.status == "bracketed_root" for result in ambipolar_surfaces)
            if ambipolar_surfaces
            else None
        ),
        "ambipolar_search": (
            {
                "strategy": case.electric_field.search_strategy,
                "scope": sorted(
                    {result.search_scope for result in ambipolar_surfaces}
                ),
                "unsampled_crossings_excluded": False,
            }
            if ambipolar_surfaces
            else None
        ),
        "ambipolar_selection": (
            "continue the selected branch identity; nearest zero on the first surface and nearest prior field after branch loss"
            if ambipolar_surfaces and case.electric_field.continue_branches
            else "root nearest zero on each surface"
            if ambipolar_surfaces
            else None
        ),
        "ambipolar_branch_continuation": (
            {
                "method": "linear radial prediction with global minimum-cost matching",
                "continuation_gate_fraction_of_search_span": 0.25,
                "event_evidence": "discrete adjacent-surface observations, not continuously resolved bifurcations",
                "event_count": sum(
                    len(result.branch_events) for result in ambipolar_surfaces
                ),
                "nonsmooth_event_count": sum(
                    event.nonsmooth
                    for result in ambipolar_surfaces
                    for event in result.branch_events
                ),
            }
            if ambipolar_surfaces
            else None
        ),
        "ambipolar_refinement": (
            {
                "enabled": case.convergence.enabled,
                "observables": list(case.convergence.observables),
                "relative_tolerance": case.convergence.relative_tolerance,
                "max_refinements": case.convergence.max_refinements,
                "all_surfaces_resolved": all(
                    result.refinement_status == "resolved"
                    for result in ambipolar_surfaces
                ),
            }
            if ambipolar_surfaces
            else None
        ),
        "ambipolar_solver_attempts": (
            {
                "attempt_count": len(ambipolar_solver_attempts),
                "executed_route_counts": executed_route_counts,
                "automatic_true_residual_recovery_count": recovery_count,
                "policy": "retry only failed automatic-route evaluations with one bounded GMRES solve; explicit methods remain fail-closed",
            }
            if ambipolar_surfaces
            else None
        ),
        "legendre_tail_diagnostic": (
            "retained_full_state_relative_l2"
            if legendre_tail_retained
            else "retained_selected_tail_relative_l2_upper_bound"
            if legendre_tail_upper_bound_retained
            else "unavailable_on_zero_padded_truncated_state"
            if ambipolar_surfaces
            else "not_requested"
        ),
        "normalization": {
            "density_m3": 1.0e20,
            "temperature_keV": 1.0,
            "mass_kg": _PROTON_MASS_KG,
            "a_hat": geometry_state.a_hat,
            "psi_a_hat": geometry_state.psi_a_hat,
        },
        "phase_space": {
            "pitch_speed_ramp": (
                case.resolution.pitch_speed_ramp
                if case.resolution.pitch_modes_by_speed is None
                else None
            ),
            "pitch_allocation_source": (
                "pitch_speed_ramp"
                if case.resolution.pitch_modes_by_speed is None
                else "explicit"
            ),
            "active_pitch_modes_by_speed": np.asarray(
                grids.n_xi_for_x, dtype=np.int64
            ).tolist(),
            "active_pitch_mode_sum": int(np.sum(grids.n_xi_for_x)),
        },
        "geometry_sha256": _sha256(
            case.geometry_path
            if case.geometry.format in {"vmec", "boozer"}
            else case.geometry.file
        ),
        "dkx_version": __version__,
        "python_version": platform.python_version(),
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "platform": platform.platform(),
        "precision": case.run.precision,
        "device": f"{device.platform}:{device.device_kind}",
        "timings_s": {"solve": float(np.sum(solve_seconds)), "total": total_seconds},
        "peak_host_memory_bytes": _peak_host_memory_bytes(),
    }
    warnings = []
    if any(
        result.search_scope == "explicit_seeded_intervals_only"
        for result in ambipolar_surfaces
    ):
        warnings.append(
            "Ambipolar promotion searched only explicit seeded intervals; unsampled crossings outside those intervals are not excluded."
        )
    if any(
        result.refinement_status == "no_bracket_observed"
        for result in ambipolar_surfaces
    ):
        warnings.append(
            "Finite adaptive sampling observed no sign-changing bracket on one or more surfaces; hidden even-number crossings are not excluded."
        )
    if any(
        result.refinement_status == "refinement_exhausted"
        for result in ambipolar_surfaces
    ):
        warnings.append(
            "Adaptive ambipolar refinement exhausted its configured levels before root and observable evidence resolved."
        )
    if any(
        event.nonsmooth
        for result in ambipolar_surfaces
        for event in result.branch_events
    ):
        warnings.append(
            "Discrete ambipolar branch creation, loss, merger, crossing, or classification-transition evidence was observed; branch-local derivatives are nonsmooth or undefined across the flagged interval."
        )
    result = Result(
        case_id=case.case_id,
        case_name=case.name,
        workflow=case.run.workflow,
        arrays=arrays,
        dimensions=dimensions,
        metadata=metadata,
        warnings=tuple(warnings),
        output_path=output_path,
        _runtime={"operator": retained_operator},
    )
    if out is not None:
        result.save(output_path)
    return result
