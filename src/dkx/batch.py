"""Batched multi-surface / multi-``E_r`` kinetic solves (``jax.vmap`` productized).

Single ``dkx`` solves are CPU/GPU parity on the direct routes; the GPU win
comes from **batching** many solves that share a discretization but differ in
their physics leaves.  Because :class:`dkx.drift_kinetic.KineticOperator`
is a registered, jit-safe pytree, ``jax.vmap`` maps cleanly over its varying
leaves, and the implicit-differentiation solve composes with the batch axis.

This module turns that into a first-class API around the two canonical batch
axes:

* an **``E_r`` scan** — a vector of radial-electric-field values on one geometry
  (:func:`batched_er_scan`); only the two ExB/Er drive scalars vary; and
* a **surface scan** — a batch of geometries sharing grids/layout
  (:func:`batched_surface_scan`); the geometry, species, collision and drive
  leaves vary while the discretization leaves stay shared.

Both are thin builders over the primitive :func:`batched_solve`, which
``jax.vmap``s a solve-plus-moments over an explicit mapping of the varying
operator leaves.  Peak memory is bounded automatically: the per-solve footprint
comes from the route-aware structured direct memory model in :mod:`dkx.solve`
(:func:`dkx.solve.auto_solve_peak_memory_bytes`), the memory budget from the
device/host, and the batch is processed in ``jax.lax.map`` chunks of the
computed size. Reverse mode can retain residuals across chunks; the budget is
an estimate rather than a hard allocation limit.

Independent batches can be split across distinct local devices using JAX's
``shard_map`` primitive. Each device runs the same memory-budgeted local map;
JIT and reverse-mode differentiation preserve this execution path. Uneven
batches repeat the final valid case for padding and discard it from outputs.
No host gather is performed; uneven output trimming can require redistribution.

Design constraints honoured here:

* reuse :func:`dkx.solve.solve` and the operator read-only (no new solver
  code, no operator surgery beyond ``dataclasses.replace`` of leaves);
* differentiable end to end (``jax.grad`` of a scalar over the batch flows
  through the implicit solve) and jit-safe (``jax.jit(batched_solve)`` compiles);
* automatic memory budgeting with only a simple optional ``max_batch`` /
  ``memory_budget_gb`` override — no advanced user knobs, no env-var switches.
"""

from __future__ import annotations

import dataclasses
import math
import os
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

# The JAX backend is imported below; dkx/runtime.py explains why this is here.
from .runtime import configure as _configure_runtime

_configure_runtime()

import jax
import jax.numpy as jnp

from .drift_kinetic import KineticOperator
from .solve import (
    _auto_route_structural,
    _TIER1_KEEP_LOWEST_DEFAULT,
    auto_solve_peak_memory_bytes,
    solve,
    tier1_truncated_tail_blocks,
)

# ---------------------------------------------------------------------------
# Leaf classification and memory-budget defaults
# ---------------------------------------------------------------------------

# Grid / derivative-matrix leaves that define the discretization: they set the
# per-solve memory footprint and are read host-side by the ``solve`` auto-router
# (``auto_solve_peak_memory_bytes`` / ``_auto_route``).  They MUST stay concrete
# (shared, un-vmapped) so the router and the footprint estimate see real values,
# so batching any of them is rejected and a surface scan keeps them from the
# template operator (and requires every surface to agree on them).
_DISCRETIZATION_FIELDS = frozenset(
    {
        "x",
        "x_weights",
        "ddx",
        "ddtheta",
        "ddzeta",
        "theta_weights",
        "zeta_weights",
        "n_xi_for_x",
        "xi_coupling_lower",
        "xi_coupling_upper",
        "ddx_xdot_plus",
        "ddx_xdot_minus",
        "ddtheta_magdrift_plus",
        "ddtheta_magdrift_minus",
        "ddzeta_magdrift_plus",
        "ddzeta_magdrift_minus",
    }
)

# Fraction of the resolved device/host memory used as the default budget: the
# structured direct footprint is an estimate and the runtime keeps other
# buffers live, so a margin below the hard limit keeps the auto-chunked batch
# safely resident.
_DEFAULT_BUDGET_FRACTION = 0.8

# Budget floor when neither the device nor the host expose a memory size.
_FALLBACK_BUDGET_GB = 8.0

# Fixed per-solve allowance for everything the analytic working-set model does
# not see: the JAX/XLA runtime, compiled executables, geometry/coefficient
# leaves held by the operator, and allocator slack.  Measured process baselines
# before the solve phase sit at ~0.4-0.5 GB on the profiling deck ladder; the
# allowance also keeps tiny reduced operators from implying an enormous chunk
# that ignores the runtime's fixed overhead.
_RUNTIME_OVERHEAD_BYTES = 512.0 * 2.0**20  # 512 MB

_BYTES_PER_GB = 2.0**30


# ---------------------------------------------------------------------------
# Result container (registered pytree so ``jax.jit(batched_solve)`` may return it)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BatchedSolveResult:
    """Outcome of a batched solve.

    Attributes:
        states: solved state vectors, shape ``(batch, total_size)``.
        moments: the per-element RHSMode-1 moment table
            (:func:`dkx.run.profile_moments_from_operator`), every entry
            with a leading batch axis, e.g. ``particleFlux_vm_psiHat`` shape
            ``(batch, n_species)`` and ``FSABjHat`` shape ``(batch,)``.
        chunk_size: the memory-budgeted ``jax.lax.map`` chunk size actually
            used.  With a multi-device split this is the per-device chunk
            (the memory budget applies per device).
        n_chunks: number of chunks the batch was processed in
            (``ceil(batch / chunk_size)``).  With a multi-device split this is
            the number of sequential chunks on the busiest device
            (``ceil(largest_shard / chunk_size)``).
        method: the ``solve_method`` requested for every element.
        executed_method: the structural route executed for every element. For
            ``method="auto"`` this resolves the operator-dependent auto policy
            to ``block_tridiagonal``, ``block_tridiagonal_truncated``, or
            ``gmres`` rather than presenting the request as the route.
        radial_current: ``J_r = sum_a Z_a Gamma_a`` per element, shape
            ``(batch,)`` — populated by :func:`batched_er_scan`, ``None``
            otherwise.
        residual_norms: independently recomputed ``||A x - b||`` for the
            returned state. A truncated, zero-padded moment-only state generally
            fails this full-equation check even when its low-order moments are
            accurate. Request ``retain_full_state=True`` to recover every block.
        relative_residual_norms: ``||A x - b|| / ||b||`` per element. With a
            zero drive, entrywise zero defect maps to zero and any nonzero
            defect to inf, even if its squared norm underflows.
        algebraic_converged: per-element finite state/RHS/residual and original
            residual-tolerance check. Zero tolerance requires entrywise zero
            defect. This is not an observable/grid certificate.
        legendre_tail_relative_l2_upper_bound: selected-tail upper bound with
            shape ``(batch, speed, species)`` when explicitly retained on the
            truncated structured route; ``None`` otherwise.
    """

    states: jnp.ndarray
    moments: Mapping[str, jnp.ndarray]
    chunk_size: int
    n_chunks: int
    method: str
    executed_method: str
    radial_current: jnp.ndarray | None = None
    residual_norms: jnp.ndarray | None = None
    legendre_tail_relative_l2_upper_bound: jnp.ndarray | None = None
    relative_residual_norms: jnp.ndarray | None = None
    algebraic_converged: jnp.ndarray | None = None


# The state, moments, radial current, residual norms, and optional tail bound
# are traced arrays. The chunking metadata and method string are static aux so
# the whole result is a valid ``jax.jit`` return value.
jax.tree_util.register_dataclass(
    BatchedSolveResult,
    data_fields=[
        "states",
        "moments",
        "radial_current",
        "residual_norms",
        "relative_residual_norms",
        "algebraic_converged",
        "legendre_tail_relative_l2_upper_bound",
    ],
    meta_fields=["chunk_size", "n_chunks", "method", "executed_method"],
)


# ---------------------------------------------------------------------------
# Memory model: per-solve footprint, budget resolution, and chunk size
# ---------------------------------------------------------------------------


def solve_footprint_bytes(
    op: KineticOperator,
    *,
    memory_budget_gb: float | None = None,
    retain_legendre_tail: bool = False,
    retain_full_state: bool = False,
) -> float:
    """Estimated peak bytes of one ``method="auto"`` solve of ``op``.

    Follows the auto-router's own decision via
    :func:`dkx.solve.auto_solve_peak_memory_bytes`: a solve that routes to the
    truncated block-Thomas kernel (ramped ``Nxi_for_x`` or a full-band peak
    above the structured direct budget) is charged its truncated working set —
    ``O(keep * m^2)`` per subsystem plus the compact coefficient buffers — not
    the full-band factorization peak, which that route never allocates and
    which overstates a production-shaped solve by ~46x (silently serializing
    batched scans through ``chunk=1``).  Full-band and recycled Krylov routes
    keep the conservative full-band peak.  ``memory_budget_gb`` is forwarded to the
    same auto-router used by the solve so budget-forced truncation and chunk
    sizing cannot disagree.  The solved state vector and a fixed runtime
    allowance (:data:`_RUNTIME_OVERHEAD_BYTES`) are added on top.
    ``retain_full_state`` charges recovery of all Legendre blocks.
    """
    route_bytes = float(
        auto_solve_peak_memory_bytes(op, budget_gb=memory_budget_gb,
                                    keep_lowest=op.n_xi if retain_full_state else _TIER1_KEEP_LOWEST_DEFAULT)
    )
    state_bytes = float(op.total_size) * 8.0
    tail_bytes = 0.0
    if retain_legendre_tail and _auto_route_structural(
        op, budget_gb=memory_budget_gb
    ) == "block_tridiagonal_truncated":
        m = op.n_theta * op.n_zeta
        # Conservative allowance for one serial subsystem's Schur/transfer-map
        # sweep, plus selected blocks for every (species, speed) chain.
        tail_bytes = float(4 * 5 * m * m + op.n_species * op.n_x * 2 * m) * 8.0
    return route_bytes + state_bytes + tail_bytes + _RUNTIME_OVERHEAD_BYTES


def _device_memory_bytes() -> float | None:
    """Device memory limit (bytes) if the backend exposes it, else ``None``."""
    try:
        stats = jax.devices()[0].memory_stats()
    except Exception:  # noqa: BLE001 - backends without memory_stats
        return None
    if not stats:
        return None
    limit = stats.get("bytes_limit")
    return float(limit) if limit else None


def _host_memory_bytes() -> float | None:
    """Total physical host RAM (bytes) via ``os.sysconf``, else ``None``."""
    try:
        return float(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (ValueError, OSError, AttributeError):
        return None


def resolve_memory_budget_bytes(memory_budget_gb: float | None = None) -> float:
    """Resolve the batch memory budget in bytes.

    An explicit ``memory_budget_gb`` is taken verbatim; otherwise a fraction
    (:data:`_DEFAULT_BUDGET_FRACTION`) of the device memory limit is used, then
    the host RAM, then a fixed fallback (:data:`_FALLBACK_BUDGET_GB`).
    """
    if memory_budget_gb is not None:
        return float(memory_budget_gb) * _BYTES_PER_GB
    total = _device_memory_bytes()
    if total is None:
        total = _host_memory_bytes()
    if total is None:
        total = _FALLBACK_BUDGET_GB * _BYTES_PER_GB
    return _DEFAULT_BUDGET_FRACTION * total


def auto_chunk_size(
    op: KineticOperator,
    batch: int,
    *,
    max_batch: int | None = None,
    memory_budget_gb: float | None = None,
    retain_legendre_tail: bool = False,
    retain_full_state: bool = False,
) -> int:
    """Largest chunk whose peak stays within the memory budget (>= 1, <= batch).

    ``chunk = floor(budget / per_solve_footprint)`` from
    :func:`resolve_memory_budget_bytes` and :func:`solve_footprint_bytes`, then
    clamped by the optional ``max_batch`` override and the batch length.
    """
    if batch <= 0:
        raise ValueError("batch must be a positive integer.")
    budget = resolve_memory_budget_bytes(memory_budget_gb)
    per_solve = solve_footprint_bytes(
        op,
        memory_budget_gb=memory_budget_gb,
        retain_legendre_tail=retain_legendre_tail,
        retain_full_state=retain_full_state,
    )
    chunk = int(budget // per_solve)
    chunk = max(1, chunk)
    if max_batch is not None:
        if int(max_batch) < 1:
            raise ValueError("max_batch must be a positive integer.")
        chunk = min(chunk, int(max_batch))
    return min(chunk, int(batch))


# ---------------------------------------------------------------------------
# Batch-leaf validation / helpers
# ---------------------------------------------------------------------------


def _batch_size_of(batch_leaves: Mapping[str, Any]) -> int:
    """The shared leading-axis length of every leaf in ``batch_leaves``."""
    leaves = jax.tree_util.tree_leaves(batch_leaves)
    if not leaves:
        raise ValueError("batch_leaves is empty; nothing to batch over.")
    sizes = {int(jnp.asarray(leaf).shape[0]) for leaf in leaves if jnp.ndim(leaf) >= 1}
    if not sizes:
        raise ValueError("batch_leaves has no array leaves with a batch axis.")
    if len(sizes) != 1:
        raise ValueError(
            f"batch_leaves must share one leading batch axis; got sizes {sorted(sizes)}."
        )
    return sizes.pop()


def _validate_batch_leaves(
    op: KineticOperator, batch_leaves: Mapping[str, Any]
) -> None:
    """Reject unknown fields and any discretization leaf (must stay shared)."""
    if not isinstance(batch_leaves, Mapping):
        raise TypeError(
            "batch_leaves must be a mapping of operator field name -> array."
        )
    valid = set(KineticOperator._CHILD_FIELDS)
    for name in batch_leaves:
        if name not in valid:
            raise ValueError(
                f"batch_leaf {name!r} is not a batchable KineticOperator leaf; "
                f"valid leaves are {sorted(valid - _DISCRETIZATION_FIELDS)}."
            )
        if name in _DISCRETIZATION_FIELDS:
            raise ValueError(
                f"batch_leaf {name!r} is a discretization leaf and must stay shared "
                "across the batch (it sets the per-solve footprint and the solver "
                "route); batch geometries/species/drives, not grids."
            )


# ---------------------------------------------------------------------------
# Multi-device split: resolve the device set and shard the batch axis
# ---------------------------------------------------------------------------


def _resolve_devices(
    devices: Sequence[jax.Device] | str | None, batch: int
) -> list[jax.Device] | None:
    """Resolve the ``devices`` argument to a multi-device list, or ``None``.

    ``None`` keeps the single-device path.  ``"auto"`` selects every local
    device of the default backend when more than one is visible.  An explicit
    non-empty sequence must contain distinct devices. Fewer than two
    devices — or a batch smaller than the device count, which would leave
    devices idle — returns ``None`` so the caller degrades to the
    single-device path unchanged.
    """
    if devices is None:
        return None
    if isinstance(devices, str):
        if devices != "auto":
            raise ValueError(
                f"devices={devices!r} is not recognised; pass None, 'auto', or an "
                "explicit sequence of jax.Device objects."
            )
        devs = list(jax.local_devices())
    else:
        devs = list(devices)
        if not devs:
            raise ValueError(
                "devices must be None, 'auto', or a non-empty sequence of "
                "jax.Device objects."
            )
    if len(set(devs)) != len(devs):
        raise ValueError("devices must contain distinct physical devices")
    if len(devs) < 2 or batch < len(devs):
        return None
    return devs


# ---------------------------------------------------------------------------
# The primitive: vmap a solve-plus-moments over varying operator leaves
# ---------------------------------------------------------------------------


def batched_solve(
    op: KineticOperator,
    batch_leaves: Mapping[str, Any],
    *,
    solve_method: str = "auto",
    tol: float = 1e-10,
    differentiable: bool = False,
    max_batch: int | None = None,
    memory_budget_gb: float | None = None,
    ntv_kernel_tz: Any | None = None,
    devices: Sequence[jax.Device] | str | None = None,
    retain_legendre_tail: bool = False,
    retain_full_state: bool = False,
) -> BatchedSolveResult:
    """Solve a batch of kinetic problems sharing ``op``'s discretization.

    Every batch element is ``op`` with the leaves in ``batch_leaves`` replaced
    by their per-element slice (``dataclasses.replace``); the element is solved
    with :func:`dkx.solve.solve` and reduced to the RHSMode-1 moment
    table (:func:`dkx.run.profile_moments_from_operator`).  The map is a
    ``jax.vmap`` executed in memory-budgeted ``jax.lax.map`` chunks, so it is
    differentiable (the implicit solve composes with the batch axis), jit-safe,
    and chunked using an estimated memory budget. Reverse-mode residual storage
    across chunks must be measured separately.

    Only ``op``'s varying leaves are mapped; every other leaf — crucially the
    discretization grids — is closed over and stays concrete, which keeps the
    host-side solver auto-route and the footprint estimate well defined.

    With two or more resolved devices, JAX's ``shard_map`` distributes
    contiguous batch slices on a JAX mesh, including under ``jit`` and ``grad``.
    The memory budget applies per device. Uneven batches are padded with the
    final valid case and trimmed after solving; padding contributes no loss.
    Outputs remain JAX arrays with no explicit host gather. Trimming uneven
    outputs can redistribute them, so inspect their sharding when composing
    downstream distributed work.

    Args:
        op: the base operator (defines grids, layout, and the shared leaves).
        batch_leaves: mapping ``field name -> batched value`` for the leaves
            that vary; each value carries a leading batch axis of the same
            length.  Field names must be batchable
            :class:`~dkx.drift_kinetic.KineticOperator` leaves (no
            discretization leaf; see :data:`_DISCRETIZATION_FIELDS`).
        solve_method: forwarded to :func:`dkx.solve.solve` (``"auto"``
            routes host-side on the shared ``op`` — well defined because the
            discretization is concrete).
        tol: relative residual tolerance forwarded to the solve.
        differentiable: use the implicit-differentiation solve so ``jax.grad``
            of a scalar over the batch flows through each element.
        max_batch: optional hard cap on the chunk size.
        memory_budget_gb: optional memory budget override (GB); defaults to a
            fraction of the device/host memory (:func:`resolve_memory_budget_bytes`).
            The same value controls both the per-element solver route and the
            batch chunk size. With a multi-device split the budget bounds each
            device's chunk.
        ntv_kernel_tz: optional NTV geometric kernel forwarded to the moment
            table (see :func:`dkx.run.profile_moments_from_operator`).
        devices: ``None`` (single-device, the default), ``"auto"`` (every local
            device of the default backend when more than one is visible), or an
            explicit sequence of ``jax.Device`` objects to split the batch
            across.  Fewer than two resolved devices, or a batch smaller than
            the device count, degrades to the single-device path unchanged.
        retain_full_state: recover all Legendre blocks for full-state residual
            certification/restart; includes the extra cost in chunk sizing.
        retain_legendre_tail: on the truncated structured route, perform the
            opt-in selected-tail sweep and retain its relative-L2 upper bound.

    Returns:
        A :class:`BatchedSolveResult` with batched ``states`` and ``moments``.
    """
    from .run import profile_moments_from_operator  # noqa: PLC0415 - heavy run stack

    if not math.isfinite(tol) or tol < 0:
        raise ValueError("tol must be finite and nonnegative")
    _validate_batch_leaves(op, batch_leaves)
    batch = _batch_size_of(batch_leaves)
    leaves_map = dict(batch_leaves)
    executed_method = (
        _auto_route_structural(op, budget_gb=memory_budget_gb,
                               keep_lowest=op.n_xi if retain_full_state else _TIER1_KEEP_LOWEST_DEFAULT)
        if str(solve_method).strip().lower() == "auto"
        else {"iterative": "gmres"}.get(
            str(solve_method).strip().lower(), str(solve_method).strip().lower()
        )
    )
    retain_selected_tail = bool(retain_legendre_tail) and (
        executed_method == "block_tridiagonal_truncated"
    )

    devs = _resolve_devices(devices, batch)

    def solve_one(leaves: Mapping[str, Any]):
        op_i = dataclasses.replace(op, **leaves)
        rhs = op_i.rhs()
        result = solve(
            op_i,
            rhs,
            method=solve_method,
            tol=tol,
            differentiable=differentiable,
            tier1_memory_budget_gb=memory_budget_gb,
            tier1_keep_lowest=op.n_xi if retain_full_state else _TIER1_KEEP_LOWEST_DEFAULT,
            emit=None,
        )
        state = jnp.reshape(result.x, (-1,))
        moments = profile_moments_from_operator(
            op_i, state, ntv_kernel_tz=ntv_kernel_tz
        )
        # Solver diagnostics may cover only the retained low-order rows.
        # Certify the returned state against the original equation instead.
        defect = op_i.apply(state) - rhs.reshape((-1,))
        residual_norm = jnp.linalg.norm(defect)
        rhs_norm = jnp.linalg.norm(rhs)
        zero_defect = jnp.all(defect == 0)
        relative = jnp.where(rhs_norm > 0, residual_norm / jnp.where(rhs_norm > 0, rhs_norm, 1),
                             jnp.where(zero_defect, 0.0, jnp.inf))
        relative = jnp.where(jnp.isfinite(rhs_norm) & jnp.isfinite(residual_norm), relative, jnp.inf)
        # Exact equations must not be admitted by an underflowed squared norm.
        within_tolerance = jnp.where(
            (tol == 0) | jnp.all(rhs == 0), zero_defect, residual_norm <= tol * rhs_norm
        )
        accepted = (jnp.isfinite(rhs_norm) & jnp.isfinite(residual_norm)
                    & jnp.all(jnp.isfinite(state)) & (residual_norm >= 0)
                    & within_tolerance)
        tail_bound = jnp.zeros((op.n_x, op.n_species), dtype=jnp.float64)
        if retain_selected_tail:
            from .moments import (  # noqa: PLC0415
                legendre_tail_relative_l2_upper_bound_batch,
            )
            from .writer import operator_containers  # noqa: PLC0415

            selected = tier1_truncated_tail_blocks(
                op_i, rhs, result.x, keep_highest=2
            )[..., 0]
            tail_bound = legendre_tail_relative_l2_upper_bound_batch(
                *operator_containers(op_i)[:3],
                state[None, :],
                selected[None, ...],
            )[0]
        return state, dict(moments), residual_norm, tail_bound, relative, accepted

    if devs is None:
        chunk = auto_chunk_size(
            op,
            batch,
            max_batch=max_batch,
            memory_budget_gb=memory_budget_gb,
            retain_legendre_tail=retain_selected_tail,
            retain_full_state=retain_full_state,
        )
        states, moments, residual_norms, tail_bounds, relative, accepted = jax.lax.map(
            solve_one, leaves_map, batch_size=chunk
        )
        n_chunks = -(-batch // chunk)
    else:
        from jax.sharding import Mesh, PartitionSpec as P  # noqa: PLC0415

        # SOLVAX owns the local numerical solves. Its shard_batch wrapper does
        # not yet expose this checker option required by custom_linear_solve.
        if hasattr(jax, "shard_map"):
            shard_map = jax.shard_map
            check = {"check_vma": False}
        else:  # JAX 0.4 compatibility
            from jax.experimental.shard_map import shard_map  # noqa: PLC0415
            check = {"check_rep": False}

        shard_max = -(-batch // len(devs))
        chunk = auto_chunk_size(
            op,
            shard_max,
            max_batch=max_batch,
            memory_budget_gb=memory_budget_gb,
            retain_legendre_tail=retain_selected_tail,
            retain_full_state=retain_full_state,
        )
        # Equal local shapes are required by shard_map. Repeat a valid final
        # case instead of inventing zero-density/otherwise invalid physics.
        padding = shard_max * len(devs) - batch
        padded = jax.tree_util.tree_map(
            lambda a: jnp.concatenate(
                [jnp.asarray(a), jnp.repeat(jnp.asarray(a)[-1:], padding, axis=0)],
                axis=0,
            ),
            leaves_map,
        )
        mapped = shard_map(
            lambda local: jax.lax.map(solve_one, local, batch_size=chunk),
            mesh=Mesh(devs, ("device",)),
            in_specs=P("device"),
            out_specs=P("device"),
            **check,
        )
        states, moments, residual_norms, tail_bounds, relative, accepted = jax.tree_util.tree_map(
            lambda a: a[:batch], mapped(padded)
        )
        n_chunks = -(-shard_max // chunk)
    return BatchedSolveResult(
        states=states,
        moments=dict(moments),
        chunk_size=int(chunk),
        n_chunks=int(n_chunks),
        method=str(solve_method),
        executed_method=executed_method,
        residual_norms=residual_norms,
        relative_residual_norms=relative,
        algebraic_converged=accepted,
        legendre_tail_relative_l2_upper_bound=(
            tail_bounds if retain_selected_tail else None
        ),
    )


# ---------------------------------------------------------------------------
# Canonical axis 1: an E_r scan on one geometry
# ---------------------------------------------------------------------------


def batched_er_scan(
    problem: Any,
    er_values: Any,
    *,
    solve_method: str | None = None,
    tol: float | None = None,
    differentiable: bool = False,
    max_batch: int | None = None,
    memory_budget_gb: float | None = None,
    ntv_kernel_tz: Any | None = None,
    devices: Sequence[jax.Device] | str | None = None,
    retain_legendre_tail: bool = False,
    retain_full_state: bool = False,
) -> BatchedSolveResult:
    """Batched radial current / moments over a vector of ``E_r`` on one geometry.

    Each ``E_r`` sets both ExB/Er drive scalars to ``dphi_per_er * E_r`` (the
    ``ambipolarSolver.F90`` ``updateEr`` relation, exactly
    :func:`dkx.er.operator_at_er`); every other leaf — geometry, species,
    collisions, grids — is shared, so the scan is one vmapped solve over the
    single varying drive.  The result's ``radial_current`` is
    ``J_r = sum_a Z_a Gamma_a`` per ``E_r``.

    Args:
        problem: a prepared :class:`dkx.er.ErProblem`
            (:func:`dkx.er.prepare`) — carries the base operator with the
            ExB/Er term flags active, the ``dphi_per_er`` conversion, and ``z_s``.
        er_values: the ``E_r`` scan values, shape ``(batch,)``.
        solve_method, tol: forwarded to the solve (default to the problem's).
        differentiable: differentiable implicit solves (for ``jax.grad``).
        max_batch, memory_budget_gb: memory-budgeting overrides
            (:func:`auto_chunk_size`).
        ntv_kernel_tz: optional NTV kernel forwarded to each profile-moment
            evaluation. Production output workflows use this to retain the
            same NTV diagnostics as scalar profile runs.
        devices: multi-device split of the scan (see :func:`batched_solve`).
        retain_full_state: recover all Legendre blocks for full-state residual
            certification/restart; includes the extra cost in chunk sizing.
        retain_legendre_tail: opt-in selected-tail upper bound on the truncated
            route.

    Returns:
        A :class:`BatchedSolveResult` with ``radial_current`` populated.
    """
    er = jnp.asarray(er_values, dtype=jnp.float64).reshape((-1,))
    dphi = jnp.asarray(problem.dphi_per_er, dtype=jnp.float64) * er
    batch_leaves = {"dphi_hat_dpsi_hat": dphi, "dphi_hat_dpsi_hat_kinetic": dphi}

    result = batched_solve(
        problem.operator,
        batch_leaves,
        solve_method=solve_method or problem.solve_method,
        tol=tol if tol is not None else problem.tol,
        differentiable=differentiable,
        max_batch=max_batch,
        memory_budget_gb=memory_budget_gb,
        ntv_kernel_tz=ntv_kernel_tz,
        devices=devices,
        retain_legendre_tail=retain_legendre_tail,
        retain_full_state=retain_full_state,
    )
    z_s = jnp.asarray(problem.z_s, dtype=jnp.float64)
    gamma = jnp.asarray(result.moments["particleFlux_vm_psiHat"])  # (batch, n_species)
    j_r = gamma @ z_s  # (batch,)
    return dataclasses.replace(result, radial_current=j_r)


# ---------------------------------------------------------------------------
# Canonical axis 2: a batch of flux surfaces sharing discretization
# ---------------------------------------------------------------------------


def _shared_structure(op: KineticOperator) -> Any:
    """The pytree structure (aux + None layout) that batched surfaces must share."""
    return jax.tree_util.tree_structure(op)


def _stack_varying_leaves(operators: Sequence[KineticOperator]) -> dict[str, Any]:
    """Stack every non-discretization, non-``None`` child leaf across surfaces.

    Discretization leaves stay shared (kept from the template and validated
    equal by :func:`_check_shared_discretization`); ``None`` leaves are shared
    structurally.  Every other leaf — geometry, species, drive scalars, and the
    collision sub-pytrees — is stacked with a leading batch axis, tree-aware so
    the ``pas``/``fp`` operator pytrees stack leaf by leaf.
    """
    template = operators[0]
    batch_leaves: dict[str, Any] = {}
    for name in KineticOperator._CHILD_FIELDS:
        if name in _DISCRETIZATION_FIELDS or getattr(template, name) is None:
            continue
        values = [getattr(o, name) for o in operators]
        batch_leaves[name] = jax.tree_util.tree_map(
            lambda *xs: jnp.stack(xs, axis=0), *values
        )
    return batch_leaves


def _check_shared_discretization(operators: Sequence[KineticOperator]) -> None:
    """Require every surface to share structure and discretization leaves."""
    import numpy as np  # noqa: PLC0415 - host-only equality check

    template = operators[0]
    ref_struct = _shared_structure(template)
    for idx, op in enumerate(operators[1:], start=1):
        if _shared_structure(op) != ref_struct:
            raise ValueError(
                f"surface {idx} has a different operator structure (aux flags / "
                "None layout) than surface 0; a batch must share discretization "
                "and configuration."
            )
        for name in _DISCRETIZATION_FIELDS:
            ref = getattr(template, name)
            if ref is None:
                continue
            if not np.array_equal(np.asarray(ref), np.asarray(getattr(op, name))):
                raise ValueError(
                    f"surface {idx} differs from surface 0 in discretization leaf "
                    f"{name!r}; batched surfaces must share grids/derivative "
                    "matrices (only geometry/species/drive leaves may vary)."
                )


def batched_surface_scan(
    operators: Sequence[KineticOperator],
    *,
    solve_method: str = "auto",
    tol: float = 1e-10,
    differentiable: bool = False,
    max_batch: int | None = None,
    memory_budget_gb: float | None = None,
    ntv_kernel_tz: Any | None = None,
    devices: Sequence[jax.Device] | str | None = None,
    retain_legendre_tail: bool = False,
    retain_full_state: bool = False,
) -> BatchedSolveResult:
    """Batched solve / moments over a sequence of flux-surface operators.

    The surfaces share discretization (grids, derivative matrices, layout, and
    option flags) but differ in geometry, species, collision and drive leaves.
    Those varying leaves are stacked (:func:`_stack_varying_leaves`) and mapped
    with :func:`batched_solve`; the shared discretization is validated equal
    across surfaces (:func:`_check_shared_discretization`).

    Args:
        operators: the per-surface
            :class:`~dkx.drift_kinetic.KineticOperator` objects (e.g. one
            per radial location, each built from that surface's geometry).
        solve_method, tol, differentiable: forwarded to :func:`batched_solve`.
        max_batch, memory_budget_gb: memory-budgeting overrides.
        ntv_kernel_tz: optional NTV kernel forwarded to the moment table.
        devices: multi-device split of the batch (see :func:`batched_solve`).
        retain_full_state: recover all Legendre blocks for full-state residual
            certification/restart; includes the extra cost in chunk sizing.
        retain_legendre_tail: opt-in selected-tail upper bound on the truncated
            route.

    Returns:
        A :class:`BatchedSolveResult` with batched ``states`` and ``moments``.
    """
    ops = list(operators)
    if not ops:
        raise ValueError("batched_surface_scan requires at least one operator.")
    _check_shared_discretization(ops)
    batch_leaves = _stack_varying_leaves(ops)
    return batched_solve(
        ops[0],
        batch_leaves,
        solve_method=solve_method,
        tol=tol,
        differentiable=differentiable,
        max_batch=max_batch,
        memory_budget_gb=memory_budget_gb,
        ntv_kernel_tz=ntv_kernel_tz,
        devices=devices,
        retain_legendre_tail=retain_legendre_tail,
        retain_full_state=retain_full_state,
    )


__all__ = [
    "BatchedSolveResult",
    "auto_chunk_size",
    "batched_er_scan",
    "batched_solve",
    "batched_surface_scan",
    "resolve_memory_budget_bytes",
    "solve_footprint_bytes",
]
