"""Native physical-unit ambipolar scans and bracket refinement."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np


_MAX_RETAINED_EVALUATIONS = 100_000


@dataclass(frozen=True)
class AmbipolarEvidencePreflight:
    """Conservative work and retained-evidence bounds for a native case."""

    hierarchy_points: int
    evaluations_per_surface: int
    profile_evaluations: int
    retained_bytes_per_surface: int
    retained_profile_bytes: int
    search_strategy: str = "uniform"
    search_points_by_surface: tuple[int, ...] = ()


@dataclass(frozen=True)
class SolverAttempt:
    """One retained linear-solver attempt for a physical-field evaluation."""

    requested_method: str
    executed_method: str
    residual_norm: float
    accepted: bool
    reason: str


@dataclass(frozen=True)
class RootEvaluation:
    """One solved electric-field point retained for root evidence."""

    electric_field_kv_m: float
    radial_current_a_m2: float
    particle_flux_m2_s: np.ndarray
    heat_flux_w_m2: np.ndarray
    parallel_current_a_t_m2: float
    residual_norm: float
    stage: str
    particle_flux_m2_s_vs_speed: np.ndarray | None = None
    heat_flux_w_m2_vs_speed: np.ndarray | None = None
    legendre_tail_relative_l2: np.ndarray | None = None
    legendre_tail_relative_l2_upper_bound: np.ndarray | None = None
    reason: str = "initial_uniform_grid"
    refinement_level: int = 0
    solver_attempts: tuple[SolverAttempt, ...] = ()


@dataclass(frozen=True)
class NativeAmbipolarRoot:
    """One sign-bracketed and physically re-evaluated ambipolar root."""

    electric_field_kv_m: float
    radial_current_a_m2: float
    slope_a_m2_per_kv_m: float
    root_type: str
    bracket_kv_m: tuple[float, float]
    evaluation: RootEvaluation
    movement_kv_m: float = np.nan
    observable_relative_movement: float = np.nan
    branch_id: str = ""


@dataclass(frozen=True)
class BranchEvent:
    """One radially localized branch-continuation event."""

    kind: str
    branch_ids: tuple[str, ...]
    root_indices: tuple[int, ...]
    electric_field_kv_m: float
    detail: str
    nonsmooth: bool = True


@dataclass(frozen=True)
class RefinementEvidence:
    """One deterministic adaptive-search rung retained in the result."""

    level: int
    search_evaluations: int
    total_evaluations: int
    root_count: int
    root_movement_kv_m: float
    observable_relative_movement: float
    max_bracket_width_kv_m: float
    converged: bool


@dataclass(frozen=True)
class NativeAmbipolarSurface:
    """All scan evidence and roots on one flux surface."""

    evaluations: tuple[RootEvaluation, ...]
    roots: tuple[NativeAmbipolarRoot, ...]
    selected_root: int | None
    selected: RootEvaluation
    status: str
    solve_seconds: float
    batch_chunk_size: int
    batch_chunks: int
    refinement: tuple[RefinementEvidence, ...] = ()
    refinement_status: str = "not_requested"
    evaluation_budget: int = 0
    branch_events: tuple[BranchEvent, ...] = ()
    selected_branch_id: str = ""
    selection_reason: str = "not_assigned"
    search_strategy: str = "uniform"
    search_scope: str = "global_uniform_domain"


def _classify_root(electric_field_kv_m: float, slope: float) -> str:
    if slope < 0.0:
        return "unstable"
    return "electron" if electric_field_kv_m > 0.0 else "ion"


def _brackets(fields: np.ndarray, currents: np.ndarray) -> list[tuple[int, int]]:
    brackets: list[tuple[int, int]] = []
    for index in range(fields.size - 1):
        left = float(currents[index])
        right = float(currents[index + 1])
        if left == 0.0:
            brackets.append((index, index))
        elif left * right < 0.0:
            brackets.append((index, index + 1))
    if currents[-1] == 0.0:
        brackets.append((fields.size - 1, fields.size - 1))
    return brackets


def _evaluation_budget(
    *,
    search_points: int,
    max_root_iterations: int,
    find_all_roots: bool,
    convergence_enabled: bool,
    max_refinements: int,
) -> tuple[int, int]:
    """Return final hierarchy size and a conservative retained-solve bound."""

    refinements = int(max_refinements) if convergence_enabled else 0
    # With the schema's minimum three search points, level 16 already contains
    # 131073 hierarchy points. Reject it before constructing an arbitrarily
    # large Python integer from an untrusted configuration value.
    if refinements > 15:
        raise ValueError(
            "native ambipolar refinement preflight exceeds 100000 retained "
            "evaluations; reduce convergence.max_refinements"
        )
    levels = refinements + 1
    hierarchy_points = (
        (int(search_points) - 1) * (2**refinements) + 1
        if convergence_enabled
        else int(search_points)
    )
    brackets_per_level = hierarchy_points if find_all_roots else 1
    budget = hierarchy_points + (levels * brackets_per_level * int(max_root_iterations))
    return hierarchy_points, budget


def _retained_evidence_bytes(
    evaluation_budget: int, species_count: int, speed_count: int
) -> int:
    return int(evaluation_budget) * (
        512 + 16 * int(species_count) + 24 * int(species_count) * int(speed_count)
    )


def preflight_ambipolar_case(case: Any) -> AmbipolarEvidencePreflight:
    """Bound retained solves and evidence bytes without loading JAX or geometry.

    The solve bound is intentionally conservative: it assumes every hierarchy
    point could bracket a root on every refinement rung. It is a capacity bound,
    not a runtime prediction and not a claim that finite sampling finds every
    possible even-numbered crossing.
    """

    if case.run.workflow != "ambipolar_profile":
        raise ValueError("ambipolar preflight requires workflow='ambipolar_profile'")
    strategy = case.electric_field.search_strategy
    if strategy == "seeded_brackets":
        seeds = case.electric_field.seed_brackets_kV_m
        if seeds is None:
            raise ValueError("seeded bracket preflight requires seed_brackets_kV_m")
        search_points_by_surface = tuple(
            len({float(value) for bracket in surface for value in bracket})
            for surface in seeds
        )
        evaluations_by_surface = tuple(
            points
            + len(surface) * int(case.electric_field.max_root_iterations)
            for points, surface in zip(search_points_by_surface, seeds)
        )
        hierarchy_points = max(search_points_by_surface)
    else:
        hierarchy_points, evaluations = _evaluation_budget(
            search_points=case.electric_field.search_points,
            max_root_iterations=case.electric_field.max_root_iterations,
            find_all_roots=case.electric_field.find_all_roots,
            convergence_enabled=case.convergence.enabled,
            max_refinements=case.convergence.max_refinements,
        )
        search_points_by_surface = (hierarchy_points,) * len(case.geometry.surfaces)
        evaluations_by_surface = (evaluations,) * len(case.geometry.surfaces)
    evaluations = max(evaluations_by_surface)
    if evaluations > _MAX_RETAINED_EVALUATIONS:
        raise ValueError(
            "native ambipolar refinement preflight exceeds 100000 retained "
            f"evaluations ({evaluations}); reduce convergence.max_refinements, "
            "electric_field.search_points, or max_root_iterations"
        )
    species_count = max(1, len(case.species))
    speed_count = max(1, int(case.resolution.speed))
    retained_by_surface = tuple(
        _retained_evidence_bytes(value, species_count, speed_count)
        for value in evaluations_by_surface
    )
    return AmbipolarEvidencePreflight(
        hierarchy_points=hierarchy_points,
        evaluations_per_surface=evaluations,
        profile_evaluations=sum(evaluations_by_surface),
        retained_bytes_per_surface=max(retained_by_surface),
        retained_profile_bytes=sum(retained_by_surface),
        search_strategy=strategy,
        search_points_by_surface=search_points_by_surface,
    )


def _observable(evaluation: RootEvaluation, name: str) -> np.ndarray:
    aliases = {
        "electric_field": evaluation.electric_field_kv_m,
        "particle_flux": evaluation.particle_flux_m2_s,
        "heat_flux": evaluation.heat_flux_w_m2,
        "parallel_current": evaluation.parallel_current_a_t_m2,
        "bootstrap_current": evaluation.parallel_current_a_t_m2,
        "radial_current": evaluation.radial_current_a_m2,
    }
    return np.asarray(aliases[name], dtype=np.float64)


def _relative_movement(current: np.ndarray, previous: np.ndarray) -> float:
    scale = np.maximum(np.maximum(np.abs(current), np.abs(previous)), 1.0e-300)
    return float(np.max(np.abs(current - previous) / scale))


def _compare_roots(
    roots: list[NativeAmbipolarRoot],
    previous: tuple[NativeAmbipolarRoot, ...] | None,
    observables: tuple[str, ...],
) -> tuple[list[NativeAmbipolarRoot], float, float]:
    if previous is None or len(roots) != len(previous) or not roots:
        return roots, np.nan, np.nan
    compared: list[NativeAmbipolarRoot] = []
    root_movements: list[float] = []
    observable_movements: list[float] = []
    for root, old_root in zip(roots, previous):
        root_movement = abs(root.electric_field_kv_m - old_root.electric_field_kv_m)
        requested = [
            _relative_movement(
                _observable(root.evaluation, name),
                _observable(old_root.evaluation, name),
            )
            for name in observables
            if name != "electric_field"
        ]
        observable_movement = max(requested, default=0.0)
        compared.append(
            replace(
                root,
                movement_kv_m=float(root_movement),
                observable_relative_movement=float(observable_movement),
            )
        )
        root_movements.append(float(root_movement))
        observable_movements.append(float(observable_movement))
    return compared, max(root_movements), max(observable_movements)


def _branch_prefix(root_type: str) -> str:
    return root_type if root_type in {"ion", "electron", "unstable"} else "branch"


def _predict_branch(history: list[tuple[float, float]], surface: float) -> float:
    if len(history) < 2:
        return history[-1][1]
    previous_surface, previous_field = history[-2]
    last_surface, last_field = history[-1]
    radial_step = last_surface - previous_surface
    if radial_step == 0.0:
        return last_field
    return last_field + (last_field - previous_field) * (
        (surface - last_surface) / radial_step
    )


def continue_ambipolar_branches(
    surface_results: list[NativeAmbipolarSurface] | tuple[NativeAmbipolarSurface, ...],
    *,
    surfaces: np.ndarray,
    electric_field_bounds_kv_m: tuple[float, float],
    continue_selection: bool,
) -> tuple[NativeAmbipolarSurface, ...]:
    """Assign radial branch identity and retain discrete event evidence.

    Matching uses linear prediction from each branch's two most recent points
    and a global minimum-cost assignment. A match is admitted only within one
    quarter of the declared electric-field search span. Events are discrete
    profile evidence; they do not claim a continuously resolved bifurcation.
    """

    from scipy.optimize import linear_sum_assignment  # noqa: PLC0415

    surface_values = np.asarray(surfaces, dtype=np.float64).reshape((-1,))
    if len(surface_results) != surface_values.size:
        raise ValueError("branch continuation needs one result per radial surface")
    if not np.all(np.isfinite(surface_values)) or np.any(
        np.diff(surface_values) <= 0.0
    ):
        raise ValueError(
            "branch continuation needs finite, strictly increasing surfaces"
        )
    span = float(electric_field_bounds_kv_m[1] - electric_field_bounds_kv_m[0])
    if not np.isfinite(span) or span <= 0.0:
        raise ValueError(
            "branch continuation needs increasing finite electric-field bounds"
        )
    gate = 0.25 * span
    counters: dict[str, int] = {"ion": 0, "electron": 0, "unstable": 0, "branch": 0}
    histories: dict[str, list[tuple[float, float]]] = {}
    tracked: list[NativeAmbipolarSurface] = []
    previous_roots: tuple[NativeAmbipolarRoot, ...] = ()

    def new_branch(root: NativeAmbipolarRoot) -> str:
        prefix = _branch_prefix(root.root_type)
        identifier = f"{prefix}-{counters[prefix]:03d}"
        counters[prefix] += 1
        return identifier

    for surface_index, (surface, raw_result) in enumerate(
        zip(surface_values, surface_results)
    ):
        roots = list(raw_result.roots)
        events: list[BranchEvent] = []
        matched_previous: dict[int, int] = {}
        matched_current: dict[int, int] = {}
        predictions: dict[int, float] = {}
        if previous_roots and roots:
            cost = np.empty((len(previous_roots), len(roots)), dtype=np.float64)
            for previous_index, previous_root in enumerate(previous_roots):
                predicted = _predict_branch(
                    histories[previous_root.branch_id], float(surface)
                )
                predictions[previous_index] = predicted
                for root_index, root in enumerate(roots):
                    type_penalty = (
                        0.0
                        if root.root_type == previous_root.root_type
                        else 0.05 * gate
                    )
                    cost[previous_index, root_index] = (
                        abs(root.electric_field_kv_m - predicted)
                        + type_penalty
                        + 1.0e-12 * (previous_index * len(roots) + root_index)
                    )
            previous_indices, root_indices = linear_sum_assignment(cost)
            for previous_index, root_index in zip(previous_indices, root_indices):
                predicted = predictions[previous_index]
                if abs(roots[root_index].electric_field_kv_m - predicted) <= gate:
                    matched_previous[int(previous_index)] = int(root_index)
                    matched_current[int(root_index)] = int(previous_index)

        annotated: list[NativeAmbipolarRoot] = []
        for root_index, root in enumerate(roots):
            if root_index in matched_current:
                previous_index = matched_current[root_index]
                branch_id = previous_roots[previous_index].branch_id
            else:
                branch_id = new_branch(root)
                kind = "boundary_origin" if surface_index == 0 else "creation"
                events.append(
                    BranchEvent(
                        kind=kind,
                        branch_ids=(branch_id,),
                        root_indices=(root_index,),
                        electric_field_kv_m=root.electric_field_kv_m,
                        detail=(
                            "branch first observed at the profile boundary"
                            if kind == "boundary_origin"
                            else "branch first observed between adjacent sampled surfaces"
                        ),
                        nonsmooth=kind != "boundary_origin",
                    )
                )
            annotated.append(replace(root, branch_id=branch_id))

        if previous_roots:
            for previous_index, previous_root in enumerate(previous_roots):
                if previous_index in matched_previous:
                    root_index = matched_previous[previous_index]
                    root = annotated[root_index]
                    if root.root_type != previous_root.root_type:
                        events.append(
                            BranchEvent(
                                kind="classification_transition",
                                branch_ids=(root.branch_id,),
                                root_indices=(root_index,),
                                electric_field_kv_m=root.electric_field_kv_m,
                                detail=(
                                    f"{previous_root.root_type} -> {root.root_type} "
                                    "between adjacent sampled surfaces"
                                ),
                            )
                        )
                    continue
                predicted = _predict_branch(
                    histories[previous_root.branch_id], float(surface)
                )
                events.append(
                    BranchEvent(
                        kind="loss",
                        branch_ids=(previous_root.branch_id,),
                        root_indices=(-1,),
                        electric_field_kv_m=predicted,
                        detail="branch no longer matched within the continuation gate",
                    )
                )
                if annotated:
                    nearest_index = min(
                        range(len(annotated)),
                        key=lambda index: abs(
                            annotated[index].electric_field_kv_m - predicted
                        ),
                    )
                    nearest = annotated[nearest_index]
                    if abs(nearest.electric_field_kv_m - predicted) <= gate:
                        events.append(
                            BranchEvent(
                                kind="merger",
                                branch_ids=(previous_root.branch_id, nearest.branch_id),
                                root_indices=(-1, nearest_index),
                                electric_field_kv_m=nearest.electric_field_kv_m,
                                detail=(
                                    "discrete merger candidate: a lost branch approaches "
                                    "a surviving branch within the continuation gate"
                                ),
                            )
                        )

            matched_items = sorted(matched_previous.items())
            for first in range(len(matched_items)):
                previous_a, current_a = matched_items[first]
                for second in range(first + 1, len(matched_items)):
                    previous_b, current_b = matched_items[second]
                    old_order = (
                        previous_roots[previous_a].electric_field_kv_m
                        - previous_roots[previous_b].electric_field_kv_m
                    )
                    new_order = (
                        annotated[current_a].electric_field_kv_m
                        - annotated[current_b].electric_field_kv_m
                    )
                    if old_order * new_order < 0.0:
                        events.append(
                            BranchEvent(
                                kind="crossing",
                                branch_ids=(
                                    annotated[current_a].branch_id,
                                    annotated[current_b].branch_id,
                                ),
                                root_indices=(current_a, current_b),
                                electric_field_kv_m=0.5
                                * (
                                    annotated[current_a].electric_field_kv_m
                                    + annotated[current_b].electric_field_kv_m
                                ),
                                detail=(
                                    "branch ordering reversed between adjacent sampled surfaces"
                                ),
                            )
                        )

        for root in annotated:
            histories.setdefault(root.branch_id, []).append(
                (float(surface), root.electric_field_kv_m)
            )
        tracked.append(
            replace(
                raw_result,
                roots=tuple(annotated),
                branch_events=tuple(events),
            )
        )
        previous_roots = tuple(annotated)

    selected_branch_id = ""
    previous_selected_field: float | None = None
    selected_results: list[NativeAmbipolarSurface] = []
    for surface_index, result in enumerate(tracked):
        if not result.roots:
            selected_results.append(
                replace(
                    result,
                    selected_root=None,
                    selected_branch_id="",
                    selection_reason=(
                        "seeded_bracket_failed_closest_endpoint"
                        if result.search_strategy == "seeded_brackets"
                        else "no_bracket_closest_sample"
                    ),
                )
            )
            selected_branch_id = ""
            continue
        if not continue_selection or previous_selected_field is None:
            selected_root = min(
                range(len(result.roots)),
                key=lambda index: abs(result.roots[index].electric_field_kv_m),
            )
            selection_reason = (
                "nearest_zero_continuation_disabled"
                if not continue_selection and surface_index > 0
                else "nearest_zero_initial"
                if surface_index == 0
                else "nearest_zero_first_observed"
            )
        else:
            continued = next(
                (
                    index
                    for index, root in enumerate(result.roots)
                    if root.branch_id == selected_branch_id
                ),
                None,
            )
            if continued is not None:
                selected_root = continued
                selection_reason = "continued_selected_branch"
            else:
                target = (
                    0.0 if previous_selected_field is None else previous_selected_field
                )
                selected_root = min(
                    range(len(result.roots)),
                    key=lambda index: abs(
                        result.roots[index].electric_field_kv_m - target
                    ),
                )
                selection_reason = "selected_branch_lost_nearest_root"
        selected = result.roots[selected_root]
        selected_branch_id = selected.branch_id
        previous_selected_field = selected.electric_field_kv_m
        selected_results.append(
            replace(
                result,
                selected_root=selected_root,
                selected=selected.evaluation,
                selected_branch_id=selected_branch_id,
                selection_reason=selection_reason,
            )
        )
    return tuple(selected_results)


def solve_native_ambipolar_surface(
    problem: Any,
    *,
    electric_field_bounds_kv_m: tuple[float, float],
    search_points: int,
    root_tolerance_kv_m: float,
    max_root_iterations: int,
    find_all_roots: bool,
    previous_root_kv_m: float | None,
    radial_factor: float,
    solve_method: str,
    solve_tolerance: float,
    memory_budget_gb: float,
    convergence_enabled: bool = False,
    convergence_observables: tuple[str, ...] = (),
    convergence_relative_tolerance: float = 0.02,
    max_refinements: int = 0,
    retain_legendre_tail: bool = False,
    seed_brackets_kv_m: tuple[tuple[float, float], ...] | None = None,
) -> NativeAmbipolarSurface:
    """Scan, refine, classify, and select native ambipolar roots.

    The coarse search is one bounded :func:`dkx.batch.batched_er_scan`. Each
    sign-changing interval is then refined with bracketed bisection. Every
    candidate is a real kinetic solve; no interpolated point is reported as a
    root. Nearby surfaces select the root nearest ``previous_root_kv_m`` while
    preserving every root in the returned evidence. When
    ``seed_brackets_kv_m`` is supplied, only those explicit intervals are
    searched and the returned scope does not exclude unsampled crossings.
    Complete states supply tail diagnostics at every evaluation;
    ``retain_legendre_tail`` remains accepted for configuration compatibility
    but no longer requests an extra selected-field replay.
    """

    import time

    from dkx.batch import batched_er_scan
    from dkx.units import ELEMENTARY_CHARGE, HEAT_FLUX, PARALLEL_CURRENT, PARTICLE_FLUX

    started = time.perf_counter()
    seeded = seed_brackets_kv_m is not None
    if seeded:
        seed_brackets = tuple(
            (float(bracket[0]), float(bracket[1])) for bracket in seed_brackets_kv_m
        )
        lower, upper = map(float, electric_field_bounds_kv_m)
        if not seed_brackets:
            raise ValueError("seeded ambipolar search requires at least one bracket")
        if convergence_enabled:
            raise ValueError(
                "seeded ambipolar promotion cannot run the global refinement hierarchy"
            )
        if not find_all_roots:
            raise ValueError("seeded ambipolar promotion requires find_all_roots=True")
        previous_right = -np.inf
        for left, right in seed_brackets:
            if (
                not np.isfinite(left)
                or not np.isfinite(right)
                or left >= right
                or left < lower
                or right > upper
                or left < previous_right
            ):
                raise ValueError(
                    "seeded ambipolar brackets must be finite, increasing, ordered, "
                    "non-overlapping, and inside electric_field_bounds_kv_m"
                )
            previous_right = right
        fields = np.asarray(
            sorted({value for bracket in seed_brackets for value in bracket}),
            dtype=np.float64,
        )
        evaluation_budget = len(fields) + len(seed_brackets) * int(max_root_iterations)
    else:
        seed_brackets = ()
        _, evaluation_budget = _evaluation_budget(
            search_points=search_points,
            max_root_iterations=max_root_iterations,
            find_all_roots=find_all_roots,
            convergence_enabled=convergence_enabled,
            max_refinements=max_refinements,
        )
    if evaluation_budget > _MAX_RETAINED_EVALUATIONS:
        raise ValueError(
            "native ambipolar refinement preflight exceeds 100000 retained "
            f"evaluations ({evaluation_budget}); reduce convergence.max_refinements, "
            "electric_field.search_points, or max_root_iterations"
        )
    species_count = max(1, len(np.atleast_1d(getattr(problem, "z_s", [1.0]))))
    speed_count = max(
        1,
        int(getattr(getattr(problem, "operator", None), "n_x", 1)),
    )
    retained_bytes = _retained_evidence_bytes(
        evaluation_budget, species_count, speed_count
    )
    if retained_bytes > float(memory_budget_gb) * (1024**3):
        raise MemoryError(
            "native ambipolar refinement evidence exceeds the memory preflight: "
            f"estimated={retained_bytes} B, budget={float(memory_budget_gb) * (1024**3):.0f} B"
        )
    if not seeded:
        fields = np.linspace(
            float(electric_field_bounds_kv_m[0]),
            float(electric_field_bounds_kv_m[1]),
            int(search_points),
            dtype=np.float64,
        )
    evaluations: dict[float, RootEvaluation] = {}
    chunks: list[int] = []
    chunk_sizes: list[int] = []

    tail_containers = None
    if hasattr(problem, "operator"):
        from dkx.drift_kinetic import KineticOperator
        from dkx.writer import operator_containers

        if isinstance(problem.operator, KineticOperator):
            tail_containers = operator_containers(problem.operator)[:3]

    def physical_outputs(batch):
        particle = (
            np.asarray(batch.moments["particleFlux_vm_psiHat"], dtype=np.float64)
            * float(radial_factor)
            * PARTICLE_FLUX
        )
        heat = (
            np.asarray(batch.moments["heatFlux_vm_psiHat"], dtype=np.float64)
            * float(radial_factor)
            * HEAT_FLUX
        )
        particle_vs_speed = (
            np.asarray(batch.moments["particleFlux_vm_psiHat_vs_x"], dtype=np.float64)
            * float(radial_factor)
            * PARTICLE_FLUX
        )
        heat_vs_speed = (
            np.asarray(batch.moments["heatFlux_vm_psiHat_vs_x"], dtype=np.float64)
            * float(radial_factor)
            * HEAT_FLUX
        )
        parallel = (
            np.asarray(batch.moments["FSABjHat"], dtype=np.float64) * PARALLEL_CURRENT
        )
        radial_current = (
            np.asarray(batch.radial_current, dtype=np.float64)
            * float(radial_factor)
            * PARTICLE_FLUX
            * ELEMENTARY_CHARGE
        )
        residuals = (
            np.asarray(batch.residual_norms, dtype=np.float64).reshape((-1,)).copy()
        )
        legendre_tail = None
        legendre_tail_upper_bound = getattr(
            batch, "legendre_tail_relative_l2_upper_bound", None
        )
        if legendre_tail_upper_bound is not None:
            legendre_tail_upper_bound = np.asarray(
                legendre_tail_upper_bound, dtype=np.float64
            )
        # Every native root evaluation requests a complete state, even when
        # the generated structured kernel retains its historical route name.
        if tail_containers is not None:
            from dkx.moments import legendre_tail_relative_l2_batch

            # np.array, not np.asarray: asarray over a JAX buffer returns a
            # read-only view, and the automatic true-residual recovery below
            # assigns into this array element-wise. Its sibling arrays escape
            # the same trap only because they are scaled by a unit factor on
            # the way out, which copies. Left as asarray this raises
            # "assignment destination is read-only" the first time a deck both
            # retains the Legendre tail and needs a retry.
            legendre_tail = np.array(
                legendre_tail_relative_l2_batch(
                    *tail_containers,
                    batch.states,
                ),
                dtype=np.float64,
            )
        return (
            particle,
            heat,
            particle_vs_speed,
            heat_vs_speed,
            parallel,
            radial_current,
            residuals,
            legendre_tail,
            legendre_tail_upper_bound,
        )

    def evaluate(
        values: np.ndarray, stage: str, reason: str, refinement_level: int
    ) -> list[RootEvaluation]:
        missing = [float(value) for value in values if float(value) not in evaluations]
        if missing:
            batch = batched_er_scan(
                problem,
                np.asarray(missing, dtype=np.float64),
                solve_method=solve_method,
                tol=solve_tolerance,
                memory_budget_gb=memory_budget_gb,
                retain_full_state=True,
            )
            (
                particle,
                heat,
                particle_vs_speed,
                heat_vs_speed,
                parallel,
                radial_current,
                residuals,
                legendre_tail,
                legendre_tail_upper_bound,
            ) = physical_outputs(batch)
            if not np.all(np.isfinite(residuals)):
                raise RuntimeError(
                    "native ambipolar scan produced a non-finite residual"
                )
            requested_method = (
                str(getattr(batch, "method", solve_method)).strip().lower()
            )
            executed_method = (
                str(getattr(batch, "executed_method", requested_method)).strip().lower()
            )
            primary_n_chunks = int(batch.n_chunks)
            primary_chunk_size = int(batch.chunk_size)
            # The accepted physical arrays are now host-owned. Release the
            # potentially large batched states before a targeted recovery so
            # the fallback does not overlap the primary solve's residency.
            del batch
            targets: np.ndarray | None = None
            if hasattr(problem, "operator"):
                from dkx.er import operator_at_er

                rhs_norms = np.asarray(
                    [
                        np.linalg.norm(
                            np.asarray(
                                operator_at_er(
                                    problem.operator,
                                    value,
                                    dphi_per_er=problem.dphi_per_er,
                                ).rhs(),
                                dtype=np.float64,
                            )
                        )
                        for value in missing
                    ]
                )
                targets = float(solve_tolerance) * rhs_norms
                failed = np.flatnonzero(residuals > targets)
            else:
                failed = np.asarray([], dtype=np.int64)
            attempts: list[list[SolverAttempt]] = [
                [
                    SolverAttempt(
                        requested_method=requested_method,
                        executed_method=executed_method,
                        residual_norm=float(residuals[index]),
                        accepted=(
                            targets is None or residuals[index] <= targets[index]
                        ),
                        reason="primary_batch",
                    )
                ]
                for index in range(len(missing))
            ]
            if failed.size and str(solve_method).strip().lower() == "auto":
                for failed_index in failed:
                    index = int(failed_index)
                    retry = batched_er_scan(
                        problem,
                        np.asarray([missing[index]], dtype=np.float64),
                        solve_method="gmres",
                        tol=solve_tolerance,
                        max_batch=1,
                        memory_budget_gb=memory_budget_gb,
                        retain_full_state=True,
                    )
                    (
                        retry_particle,
                        retry_heat,
                        retry_particle_vs_speed,
                        retry_heat_vs_speed,
                        retry_parallel,
                        retry_current,
                        retry_residuals,
                        retry_legendre_tail,
                        retry_legendre_tail_upper_bound,
                    ) = physical_outputs(retry)
                    retry_residual = float(retry_residuals[0])
                    if not np.isfinite(retry_residual):
                        raise RuntimeError(
                            "native ambipolar Krylov recovery produced a non-finite "
                            f"residual at electric_field={missing[index]:.8g} kV/m"
                        )
                    assert targets is not None
                    retry_accepted = retry_residual <= targets[index]
                    retry_requested_method = (
                        str(getattr(retry, "method", "gmres")).strip().lower()
                    )
                    retry_executed_method = (
                        str(getattr(retry, "executed_method", "gmres")).strip().lower()
                    )
                    retry_n_chunks = int(retry.n_chunks)
                    retry_chunk_size = int(retry.chunk_size)
                    del retry
                    attempts[index].append(
                        SolverAttempt(
                            requested_method=retry_requested_method,
                            executed_method=retry_executed_method,
                            residual_norm=retry_residual,
                            accepted=bool(retry_accepted),
                            reason="automatic_true_residual_recovery",
                        )
                    )
                    particle[index] = retry_particle[0]
                    heat[index] = retry_heat[0]
                    particle_vs_speed[index] = retry_particle_vs_speed[0]
                    heat_vs_speed[index] = retry_heat_vs_speed[0]
                    parallel[index] = retry_parallel[0]
                    radial_current[index] = retry_current[0]
                    residuals[index] = retry_residual
                    if legendre_tail is not None and retry_legendre_tail is not None:
                        legendre_tail[index] = retry_legendre_tail[0]
                    elif retry_legendre_tail is not None:
                        legendre_tail = np.full(
                            (len(missing), *retry_legendre_tail.shape[1:]), np.nan
                        )
                        legendre_tail[index] = retry_legendre_tail[0]
                    if (
                        legendre_tail_upper_bound is not None
                        and retry_legendre_tail_upper_bound is not None
                    ):
                        legendre_tail_upper_bound[index] = (
                            retry_legendre_tail_upper_bound[0]
                        )
                    elif retry_legendre_tail_upper_bound is not None:
                        legendre_tail_upper_bound = np.full(
                            (len(missing), *retry_legendre_tail_upper_bound.shape[1:]),
                            np.nan,
                        )
                        legendre_tail_upper_bound[index] = (
                            retry_legendre_tail_upper_bound[0]
                        )
                    elif legendre_tail_upper_bound is not None:
                        # The accepted recovery used a full-state route, so its
                        # exact metric supersedes the primary truncated bound.
                        legendre_tail_upper_bound[index] = np.nan
                    chunks.append(retry_n_chunks)
                    chunk_sizes.append(retry_chunk_size)
                failed = np.flatnonzero(residuals > targets)
            if failed.size:
                index = int(failed[0])
                attempt_summary = ", ".join(
                    f"{attempt.executed_method}:{attempt.residual_norm:.6g}"
                    for attempt in attempts[index]
                )
                assert targets is not None
                raise RuntimeError(
                    "native ambipolar solve did not converge at "
                    f"electric_field={missing[index]:.8g} kV/m: "
                    f"residual={residuals[index]:.6g}, "
                    f"target={targets[index]:.6g}, attempts=[{attempt_summary}]"
                )
            for index, value in enumerate(missing):
                evaluations[value] = RootEvaluation(
                    electric_field_kv_m=value,
                    radial_current_a_m2=float(radial_current[index]),
                    particle_flux_m2_s=np.asarray(particle[index]),
                    heat_flux_w_m2=np.asarray(heat[index]),
                    parallel_current_a_t_m2=float(parallel[index]),
                    residual_norm=float(residuals[index]),
                    stage=stage,
                    particle_flux_m2_s_vs_speed=np.asarray(particle_vs_speed[index]),
                    heat_flux_w_m2_vs_speed=np.asarray(heat_vs_speed[index]),
                    legendre_tail_relative_l2=(
                        None
                        if legendre_tail is None
                        or not np.all(np.isfinite(legendre_tail[index]))
                        else np.asarray(legendre_tail[index])
                    ),
                    legendre_tail_relative_l2_upper_bound=(
                        None
                        if legendre_tail_upper_bound is None
                        or not np.all(np.isfinite(legendre_tail_upper_bound[index]))
                        else np.asarray(legendre_tail_upper_bound[index])
                    ),
                    reason=reason,
                    refinement_level=int(refinement_level),
                    solver_attempts=tuple(attempts[index]),
                )
            chunks.append(primary_n_chunks)
            chunk_sizes.append(primary_chunk_size)
        return [evaluations[float(value)] for value in values]

    evaluate(
        fields,
        "seeded_bracket_scan" if seeded else "coarse_scan",
        "seeded_bracket_endpoint" if seeded else "initial_uniform_grid",
        0,
    )
    roots: list[NativeAmbipolarRoot] = []
    previous_roots: tuple[NativeAmbipolarRoot, ...] | None = None
    refinement: list[RefinementEvidence] = []
    refinement_status = "not_requested"
    observables = tuple(convergence_observables) or ("electric_field",)

    for level in range((int(max_refinements) if convergence_enabled else 0) + 1):
        if level:
            midpoints = 0.5 * (fields[:-1] + fields[1:])
            evaluate(
                midpoints,
                "adaptive_refinement",
                "interval_midpoint",
                level,
            )
            fields = np.sort(np.concatenate((fields, midpoints)))
        search = [evaluations[float(value)] for value in fields]
        currents = np.asarray(
            [evaluation.radial_current_a_m2 for evaluation in search],
            dtype=np.float64,
        )
        if seeded:
            field_indices = {float(value): index for index, value in enumerate(fields)}
            bracket_indices = []
            for left, right in seed_brackets:
                left_index = field_indices[left]
                right_index = field_indices[right]
                left_current = evaluations[left].radial_current_a_m2
                right_current = evaluations[right].radial_current_a_m2
                if left_current == 0.0:
                    bracket_indices.append((left_index, left_index))
                elif right_current == 0.0:
                    bracket_indices.append((right_index, right_index))
                elif left_current * right_current < 0.0:
                    bracket_indices.append((left_index, right_index))
        else:
            bracket_indices = _brackets(fields, currents)
        if not find_all_roots and bracket_indices:
            bracket_indices = bracket_indices[:1]

        level_roots: list[NativeAmbipolarRoot] = []
        for left_index, right_index in bracket_indices:
            left = search[left_index]
            right = search[right_index]
            if left_index != right_index:
                for _ in range(int(max_root_iterations)):
                    width = right.electric_field_kv_m - left.electric_field_kv_m
                    if abs(width) <= float(root_tolerance_kv_m):
                        break
                    trial_field = 0.5 * (
                        left.electric_field_kv_m + right.electric_field_kv_m
                    )
                    trial = evaluate(
                        np.asarray([trial_field]),
                        "root_refinement",
                        "bracket_bisection",
                        level,
                    )[0]
                    if trial.radial_current_a_m2 == 0.0:
                        left = right = trial
                        break
                    if left.radial_current_a_m2 * trial.radial_current_a_m2 < 0.0:
                        right = trial
                    else:
                        left = trial
            root_evaluation = min(
                (left, right), key=lambda item: abs(item.radial_current_a_m2)
            )
            delta_field = right.electric_field_kv_m - left.electric_field_kv_m
            slope = (
                0.0
                if delta_field == 0.0
                else (right.radial_current_a_m2 - left.radial_current_a_m2)
                / delta_field
            )
            level_roots.append(
                NativeAmbipolarRoot(
                    electric_field_kv_m=root_evaluation.electric_field_kv_m,
                    radial_current_a_m2=root_evaluation.radial_current_a_m2,
                    slope_a_m2_per_kv_m=float(slope),
                    root_type=_classify_root(
                        root_evaluation.electric_field_kv_m, slope
                    ),
                    bracket_kv_m=(
                        min(left.electric_field_kv_m, right.electric_field_kv_m),
                        max(left.electric_field_kv_m, right.electric_field_kv_m),
                    ),
                    evaluation=root_evaluation,
                )
            )

        if seeded:
            unique_roots: dict[float, NativeAmbipolarRoot] = {}
            for root in level_roots:
                key = root.evaluation.electric_field_kv_m
                prior = unique_roots.get(key)
                width = root.bracket_kv_m[1] - root.bracket_kv_m[0]
                if prior is None or width < (
                    prior.bracket_kv_m[1] - prior.bracket_kv_m[0]
                ):
                    unique_roots[key] = root
            level_roots = list(unique_roots.values())
        roots, root_movement, observable_movement = _compare_roots(
            level_roots, previous_roots, observables
        )
        max_bracket_width = max(
            (root.bracket_kv_m[1] - root.bracket_kv_m[0] for root in roots),
            default=np.nan,
        )
        resolved = bool(
            convergence_enabled
            and previous_roots is not None
            and roots
            and len(roots) == len(previous_roots)
            and root_movement <= float(root_tolerance_kv_m)
            and observable_movement <= float(convergence_relative_tolerance)
            and max_bracket_width <= float(root_tolerance_kv_m)
        )
        refinement.append(
            RefinementEvidence(
                level=level,
                search_evaluations=len(fields),
                total_evaluations=len(evaluations),
                root_count=len(roots),
                root_movement_kv_m=float(root_movement),
                observable_relative_movement=float(observable_movement),
                max_bracket_width_kv_m=float(max_bracket_width),
                converged=resolved,
            )
        )
        previous_roots = tuple(roots)
    if convergence_enabled:
        refinement_status = (
            "no_bracket_observed"
            if not roots
            else "resolved"
            if refinement[-1].converged
            else "refinement_exhausted"
        )

    selected_root: int | None = None
    if roots:
        target = 0.0 if previous_root_kv_m is None else float(previous_root_kv_m)
        selected_root = min(
            range(len(roots)),
            key=lambda index: abs(roots[index].electric_field_kv_m - target),
        )
        selected = roots[selected_root].evaluation
        status = (
            "seeded_bracket_partial_failure"
            if seeded and len(roots) < len(seed_brackets)
            else "bracketed_root"
        )
    else:
        selected = min(
            (
                evaluation
                for evaluation in evaluations.values()
                if evaluation.stage != "root_refinement"
            ),
            key=lambda item: abs(item.radial_current_a_m2),
        )
        status = "seeded_bracket_failed" if seeded else "no_bracketed_root"

    ordered = tuple(evaluations[key] for key in sorted(evaluations))
    return NativeAmbipolarSurface(
        evaluations=ordered,
        roots=tuple(roots),
        selected_root=selected_root,
        selected=selected,
        status=status,
        solve_seconds=time.perf_counter() - started,
        batch_chunk_size=min(chunk_sizes),
        batch_chunks=sum(chunks),
        refinement=tuple(refinement),
        refinement_status=refinement_status,
        evaluation_budget=evaluation_budget,
        search_strategy="seeded_brackets" if seeded else "uniform",
        search_scope=(
            "explicit_seeded_intervals_only" if seeded else "global_uniform_domain"
        ),
    )


__all__ = [
    "AmbipolarEvidencePreflight",
    "BranchEvent",
    "NativeAmbipolarRoot",
    "NativeAmbipolarSurface",
    "RefinementEvidence",
    "RootEvaluation",
    "_evaluation_budget",
    "continue_ambipolar_branches",
    "preflight_ambipolar_case",
    "solve_native_ambipolar_surface",
]
