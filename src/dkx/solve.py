"""The plan-§2.3 three-route auto-policy linear solver over a :class:`KineticOperator`.

This module is the Phase-3.3 solve track: given the consolidated v3
drift-kinetic operator (:mod:`dkx.drift_kinetic`) and one or more right-hand
sides, pick and run the cheapest adequate linear solver.

The three routes are named as a case file's ``[solver] method`` names them:
``structured_direct``, ``recycled_krylov``, ``sparse_direct_referee``.  Code
identifiers in this module still spell them with the retired tier numbering —
``tier1`` names belong to the structured direct route, ``tier2`` to recycled
Krylov, ``tier3`` to sparse direct — while the prose uses the route names.

Structured direct (``solvax.direct`` block Thomas over Legendre modes)
    Available when :meth:`KineticOperator.to_block_tridiagonal` succeeds (the
    DKES-trajectory / pitch-angle-scattering family: streaming+mirror couple
    L±1, ExB and PAS are diagonal in L, no Er xDot/xiDot L±2 terms, no
    Fokker-Planck (species,x) coupling).  For that family the (species, x)
    axes are mutually uncoupled in the f-block and — for ``constraintScheme=2``
    — the bordered source/constraint machinery is diagonal over (species, x)
    too, so the full system splits into ``n_species * n_x`` independent
    block-tridiagonal systems of ``n_xi`` dense (Ntheta*Nzeta) blocks with a
    rank-one border each.  The border is absorbed exactly with the rank-one
    trick ``A~ = A + gamma B C`` (algebraically exact for any ``gamma != 0``,
    inherited from the retired probing-based RHSMode=3 solver POC) and
    the batch is solved by ``vmap``-ed ``solvax.block_thomas_factor`` /
    ``block_thomas_solve``.  Multi-RHS shares one elimination.

Recycled Krylov — preconditioned, with subspace recycling (``solvax.krylov.gcrot``)
    Matrix-free FGMRES+recycling on :meth:`KineticOperator.apply`,
    right-preconditioned by an exact structured direct solve of the
    SFINCS-simplified coarse operator (the Fortran ``preconditionerOptions``
    idiom): ``preconditioner_species=1`` (self-collisions only) and
    ``preconditioner_x=1`` (x-diagonal collisions) reduce Fokker-Planck to a
    PAS-like L-diagonal coefficient; the Er L±2 terms are dropped, which is
    Fortran's ``preconditioner_xi=1`` applied unconditionally.  The bordered
    constraint rows are eliminated exactly through
    ``solvax.operators.schur_projected_precond``.  The recycle pair (C, U) is
    returned for warm-starting continuation (Er scans, Newton steps); where its
    dense bands exceed RAM it eliminates a generated operator instead.  The
    preconditioner itself, its pins and the routing between its three storage
    policies live in :mod:`dkx.coarse_precond`.

Sparse direct — host fallback and cross-check (``solvax.native.splu_solve``)
    Materializes the operator (vmapped unit vectors; guarded by
    ``max_dense_size``) into CSR and hands it to SuperLU on the host.
    Non-differentiable, non-jittable; prints a loud one-line notice.  Used on
    explicit request (``method="direct"``) or when the recycled Krylov route
    breaches its iteration cap under ``method="auto"``.

Differentiability: the structured direct and recycled Krylov routes are wrapped
with ``solvax.implicit.linear_solve`` (implicit function theorem via
``jax.lax.custom_linear_solve``) when ``differentiable=True``; the adjoint
costs one transposed solve which reuses the same structured direct factors
(``block_thomas_solve(transpose=True)``) or a transposed-preconditioner
GCROT solve.  Sparse direct is a loud, non-differentiable escape hatch.

That transposed solve is the one place where a bad answer is invisible: the
forward solution, its residual, and every other field of :class:`SolveResult`
stay correct while the gradient is silently wrong.  The differentiable
recycled Krylov path therefore recomputes each solve's *true* residual from
the operator (``||A^T y - g||``, never the Krylov method's internal estimate),
records it in :class:`AdjointDiagnostics` on :attr:`SolveResult.adjoint`, and
— unless ``check_adjoint=False`` — raises at execution time when it misses
both the requested tolerance and the double-precision backward-error floor.

Fortran correspondence: ``solver.F90`` (KSP setup / preconditioner matrix
``whichMatrix=0``), ``preconditioner.F90`` (the ``preconditioner_*`` knobs),
and the PETSc ``Pmat`` idiom of production SFINCS.
"""

from __future__ import annotations

import functools
import os
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable

# The JAX backend is imported below; dkx/runtime.py explains why this is here.
from .runtime import configure as _configure_runtime

_configure_runtime()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

# solvax is a core dependency (installed automatically with dkx), but
# keep this module importable without it and raise a clear error on first use
# so broken/partial environments fail with an actionable message.  The guard
# itself (``_require_solvax``) is owned by :mod:`dkx.coarse_precond`, which this
# module imports anyway; there is one copy of that message, not two.
try:  # noqa: E402
    from solvax.direct import (
        BlockTridiagFactors,
        block_thomas_factor,
        block_thomas_selected_tail_fn,
        block_thomas_solve,
        block_thomas_truncated_fn,
    )
    from solvax.implicit import linear_solve as solvax_linear_solve
    from solvax.krylov import gcrot
    from solvax.native import SpluFactorization
    from solvax.native_eigen import sparse_operator_matrix
    from solvax.refine import iterative_refinement
except ImportError:
    BlockTridiagFactors = None  # type: ignore[assignment, misc]
    block_thomas_factor = None  # type: ignore[assignment]
    block_thomas_selected_tail_fn = None  # type: ignore[assignment]
    block_thomas_solve = None  # type: ignore[assignment]
    block_thomas_truncated_fn = None  # type: ignore[assignment]
    solvax_linear_solve = None  # type: ignore[assignment]
    gcrot = None  # type: ignore[assignment]
    SpluFactorization = None  # type: ignore[assignment, misc]
    sparse_operator_matrix = None  # type: ignore[assignment]

from dkx import require_float64
from dkx.coarse_precond import (  # noqa: E402
    _require_solvax,
    _transposed_apply,
    _truncated_block_fn,
    _truncated_blocks,
    _truncated_coefficients,
    _truncated_params,
    build_coarse_preconditioner,
)
from dkx.drift_kinetic import KineticOperator  # noqa: E402

__all__ = [
    "SolveResult",
    "Tier1Solver",
    "auto_solve_peak_memory_bytes",
    "build_coarse_preconditioner",
    "build_tier1_solver",
    "build_tier2_preconditioner",
    "materialize_csr",
    "materialize_dense",
    "solve",
    "tier1_available",
    "tier1_full_band_bytes",
    "tier1_peak_memory_bytes",
    "tier1_truncated_peak_memory_bytes",
    "tier1_truncated_subsystem_width",
    "tier1_truncated_tail_blocks",
]

# Default memory budget above which ``solve(method="auto")`` prefers the
# memory-lean truncated structured direct kernel over the full-band
# factorization.  Chosen to match the validated HSX head-to-head benchmark
# (tools/benchmarks/tier1_hsx_head_to_head.py).  Overridable per call via the
# ``tier1_memory_budget_gb`` argument or the environment variable below.
_TIER1_BUDGET_GB_DEFAULT = 8.0
_TIER1_BUDGET_ENV = "DKX_TIER1_MEMORY_BUDGET_GB"

# Defect-correction sweeps after the structured direct factor solve.  One
# reproduces the hand-rolled pass this replaced and already reaches O(1e-16)
# relative residual in float64; the knob exists because the float32-factor variant needs more.
_TIER1_REFINEMENT_SWEEPS = 1

# RHSMode 1/2/3 drives (radial gradient on L=0,2; inductive E_parallel on L=1)
# and every RHSMode 1/2/3 output moment (fluxes, flows, sources, FSA
# constraints) live on the lowest three Legendre modes, so keeping three
# solution blocks is exact for the standard transport quantities.
_TIER1_KEEP_LOWEST_DEFAULT = 3

# Size-aware device routing (``solve(device=...)``): on an accelerator-default
# host, ``device="auto"`` runs systems at or below these sizes on the host CPU
# instead.  Both thresholds default to 0 — auto-routing OFF — because the
# same-host measurements do not support a nonzero default (36-core Pop!_OS box
# with an RTX A4000, 2026-07-17, docs/performance.rst "Same-host CPU/GPU
# crossover"): the GPU won every structured direct warm solve measured down to
# 6.5k DOFs (2.7x-39x) and every preconditioned recycled Krylov warm solve down
# to 2.8k DOFs (1.5x-2.7x).  The one CPU-wins case — the small unpreconditioned
# recycled Krylov loop of the Phi1 Newton solve (4.5k DOFs: warm 0.048 s CPU
# vs 0.159 s GPU) — did NOT recover its win under solve-level routing
# (0.12-0.13 s: the per-Newton-iteration residuals stay on the GPU and each
# routed solve pays device transfers plus a one-time CPU compile), so routing
# small Phi1 workloads is best done whole-process (``JAX_PLATFORMS=cpu``), not
# per solve.  The knob remains for hosts where the balance differs: set e.g.
# ``DKX_SOLVE_CPU_MAX_SIZE_TIER2=6000`` to route small recycled Krylov solves.
_SOLVE_DEVICE_ENV = "DKX_SOLVE_DEVICE"
_SOLVE_CPU_MAX_TIER1_ENV = "DKX_SOLVE_CPU_MAX_SIZE_TIER1"
_SOLVE_CPU_MAX_TIER2_ENV = "DKX_SOLVE_CPU_MAX_SIZE_TIER2"
_SOLVE_CPU_MAX_TIER1_DEFAULT = 0
_SOLVE_CPU_MAX_TIER2_DEFAULT = 0

# Recycled Krylov preconditioner routes of ``solve(preconditioner=...)``.  All
# three invert the *same* SFINCS-simplified operator and differ only in how.
# ``"coarse"`` is the historical default (exact block-Thomas over L with dense
# ``Ntheta*Nzeta`` blocks); ``"multigrid"`` swaps its inner f-block inverse for
# the semicoarsened V-cycle of :mod:`dkx.multigrid`; ``"sparse"`` keeps the
# inverse exact but eliminates in a fill-reducing order on the host
# (:mod:`dkx.sparse_precond`), which is what the Fortran reference does.
_TIER2_PRECONDITIONERS = ("coarse", "multigrid", "sparse", "none")

# =============================================================================
# Result container
# =============================================================================


@dataclass(frozen=True)
class SolveResidualRecord:
    """One differentiable-solve residual measured on the host at execution time.

    Attributes:
        label: ``"forward"`` or ``"adjoint (transposed)"``.
        rhs_index: column of the right-hand side this solve belongs to.
        residual_norm: the *true* residual norm — ``||A x - b||`` for the
            forward solve, ``||A^T y - g||`` for the adjoint one, recomputed
            from the operator itself rather than read off the Krylov method's
            internal (recursively updated) estimate.
        rhs_norm: ``||b||`` (adjoint: ``||g||``, the cotangent).
        solution_norm: ``||x||`` (adjoint: ``||y||``).
        operator_norm: probe estimate of ``||A||`` (adjoint: ``||A^T||``),
            used for the attainable backward-error floor.
        target: the residual the caller asked for, ``max(atol, tol*||b||)``.
        floor: the float64 backward-error floor
            ``slack * eps * (||A|| ||x|| + ||b||)`` — no solver can push the
            residual of *this* system below it in double precision.
        limit: the accepted residual, ``factor * max(target, floor)``.
        within_tolerance: ``residual_norm <= limit`` and every norm finite.
    """

    label: str
    rhs_index: int
    residual_norm: float
    rhs_norm: float
    solution_norm: float
    operator_norm: float
    target: float
    floor: float
    limit: float
    within_tolerance: bool

    @property
    def relative_residual(self) -> float:
        """``residual_norm / ||b||`` (``inf`` for a zero right-hand side)."""
        if self.rhs_norm > 0.0:
            return self.residual_norm / self.rhs_norm
        return float("inf") if self.residual_norm > 0.0 else 0.0


@dataclass
class AdjointDiagnostics:
    """True residuals of the implicit-differentiation solves of one :func:`solve`.

    The implicit-function-theorem VJP costs one *transposed* solve, and a
    transposed solve that stalls returns a wrong gradient while the forward
    solution stays perfectly good — the failure mode this container exists to
    make visible.  Records are appended by a ``jax.debug.callback`` when the
    corresponding solve actually executes (the adjoint ones during the
    backward pass), so a freshly returned :class:`SolveResult` carries an
    empty one; read it after ``jax.grad``/``jax.vjp`` has run.

    Under ``jax.jit`` the callback closure is captured at *trace* time, so a
    cached jitted function keeps appending to the object built on its first
    trace.  Call :meth:`reset` between measurements.

    Attributes:
        tol, atol: the tolerances the solve was asked for.
        factor: the accepted slack over ``max(target, floor)``
            (``adjoint_residual_factor`` of :func:`solve`).
        checked: whether a failure raises (``check_adjoint`` of :func:`solve`).
        records: the :class:`SolveResidualRecord` list, in execution order.
    """

    tol: float
    atol: float
    factor: float
    checked: bool
    records: list[SolveResidualRecord] = field(default_factory=list)

    def reset(self) -> None:
        """Drop every record (e.g. between two runs of a jitted function)."""
        self.records.clear()

    def _select(self, label_prefix: str) -> list[SolveResidualRecord]:
        return [r for r in self.records if r.label.startswith(label_prefix)]

    @property
    def adjoint_records(self) -> list[SolveResidualRecord]:
        """Only the transposed (adjoint) solves."""
        return self._select("adjoint")

    @property
    def forward_records(self) -> list[SolveResidualRecord]:
        """Only the forward solves run under the implicit-diff wrapper."""
        return self._select("forward")

    @property
    def worst_relative_residual(self) -> float:
        """Largest ``||A^T y - g|| / ||g||`` over the adjoint solves.

        ``0.0`` when the backward pass has not run yet (no records).
        """
        rel = [r.relative_residual for r in self.adjoint_records]
        return max(rel) if rel else 0.0

    @property
    def converged(self) -> bool:
        """Whether every recorded solve landed inside its accepted residual."""
        return all(r.within_tolerance for r in self.records)


@dataclass(frozen=True)
class SolveResult:
    """Outcome of :func:`solve`.

    Attributes:
        x: solution state vector(s), same shape as the ``rhs`` passed in
            (``(n,)`` or ``(n, n_rhs)``).
        method: the implementation that ran --- ``"block_tridiagonal"``,
            ``"gcrot"``, or ``"direct"``.  For "what did it do", read
            :attr:`route` instead, which is ``"direct"`` or ``"iterative"``.
        iterations: total Krylov inner iterations across all right-hand sides
            (recycled Krylov), else ``None``.
        residual_norms: residual norms per right-hand side, shape ``(n_rhs,)``
            (jnp array; traced under ``jax.grad``). Complete states report the
            original ``||b - A x||``; moment-only truncated states report only
            the rows determined by the retained head. Certify full equations
            using complete recovery or an independent original residual.
        converged: every residual below ``max(atol, tol * ||b||)``.  ``True``
            by construction for the direct routes when residuals are finite.
        recycle: GCROT recycle pair ``(C, U)`` from the last right-hand side
            (recycled Krylov), for warm-starting the next solve of a
            continuation.
        timings: wall-clock seconds per phase (``build``, ``solve``).  Each
            phase ends with a ``jax.block_until_ready`` so the numbers are real
            device-compute time, not JAX async-dispatch latency (which would
            under-report by ~10x).  Under ``jit``/``grad`` the blocks are no-ops
            and the values are trace-time only.
        adjoint: :class:`AdjointDiagnostics` of the implicit-differentiation
            solves (differentiable recycled Krylov only, else ``None``).
            Empty when the result is returned: the records are appended at
            *execution* time, i.e. the forward ones when the value is computed
            and the adjoint ones when the backward pass runs, so inspect it
            after the enclosing ``jax.grad`` / ``jax.vjp`` call has
            completed.
        precond: the preconditioner pair this solve built, or ``None`` when the
            route did not need one. Pass it back as ``solve(..., precond=...)``
            to skip the rebuild on a nearby operator -- an ``Er`` scan, a Newton
            step, the next surface. Returning it rather than having the caller
            build one ahead of time is what makes reuse correct: the route is
            not known until the solve runs, so a caller who guesses builds a
            preconditioner the structured-direct route then never uses.
    """

    x: jnp.ndarray
    method: str
    iterations: int | None
    residual_norms: jnp.ndarray
    converged: bool
    recycle: tuple[jnp.ndarray, jnp.ndarray] | None
    timings: dict[str, float]
    adjoint: AdjointDiagnostics | None = None
    precond: tuple[Callable, Callable] | None = None

    @property
    def route(self) -> str:
        """``"direct"`` or ``"iterative"`` --- what the solve did, not how.

        ``method`` names the implementation that ran, which is what the solver
        trace and the benchmarks need.  A reader of a run wants the shorter
        answer, and "block_tridiagonal_truncated" is not it.
        """
        return "iterative" if self.method in {"gcrot", "gmres"} else "direct"


def _as_columns(rhs: jnp.ndarray) -> tuple[jnp.ndarray, bool]:
    rhs = jnp.asarray(rhs, dtype=jnp.float64)
    if rhs.ndim == 1:
        return rhs[:, None], True
    if rhs.ndim == 2:
        return rhs, False
    raise ValueError(f"rhs must be (n,) or (n, n_rhs); got shape {rhs.shape}")


def _is_traced(*arrays: Any) -> bool:
    return any(isinstance(a, jax.core.Tracer) for a in arrays)


def _residual_norms(
    matvec: Callable[[jnp.ndarray], jnp.ndarray], x2d: jnp.ndarray, rhs2d: jnp.ndarray
) -> jnp.ndarray:
    res = jax.vmap(matvec, in_axes=1, out_axes=1)(x2d) - rhs2d
    return jnp.linalg.norm(res, axis=0)


def _converged_flag(
    res_norms: jnp.ndarray, rhs2d: jnp.ndarray, tol: float, atol: float
) -> bool:
    if _is_traced(res_norms):
        return True  # direct routes under trace: exact up to factor accuracy
    rhs_norms = np.linalg.norm(np.asarray(rhs2d), axis=0)
    # Judge every column against the LARGEST column's scale, not its own.  The
    # columns are solved by one shared factorization (or one Krylov space), so
    # they share an accuracy scale: a direct solve's residual tracks the matrix
    # and the overall solve, not the norm of the individual right-hand side.
    #
    # Per-column relative targets punish a small column for being small.  On the
    # monoenergetic decks the two columns differ by 3000x in norm
    # (||b|| = [4.2e-04, 1.3e+00]), so with atol=0 the small one was given a
    # target of 4.2e-14 -- below what double precision delivers for this problem.
    # It came in at 7.7e-14, missed by 1.8x, and vetoed an otherwise exact direct
    # solve; the auto policy then paid a full Krylov re-solve (41 s where the
    # direct answer was already in hand).  A genuinely wrong solve misses by
    # orders of magnitude, not by 1.8x, so this does not weaken the check.
    scale = float(np.max(rhs_norms)) if rhs_norms.size else 0.0
    targets = np.maximum(atol, tol * np.maximum(rhs_norms, scale))
    res = np.asarray(res_norms)
    return bool(np.all(np.isfinite(res)) and np.all(res <= np.maximum(targets, 1e-30)))


def _pinned_matvecs(
    op: KineticOperator,
) -> tuple[Callable[[jnp.ndarray], jnp.ndarray], Callable[[jnp.ndarray], jnp.ndarray]]:
    """Forward/transposed matvecs with the truncated ``Nxi_for_x`` DOFs pinned.

    Fortran v3 packs the ``(x, l >= Nxi_for_x(x))`` DOFs out of the matrix
    (``indices.F90`` packed indexing), so its matrix is nonsingular.  The
    rectangular jax layout keeps those DOFs as exact zero *rows* of
    :meth:`KineticOperator.apply` (with leaked nonzero *columns* from the
    x-dense Fokker-Planck blocks), i.e. the embedded operator is structurally
    singular and its transpose is inconsistent for generic adjoint cotangents
    — the root cause of the FP+constraintScheme=1 silently-wrong gradients.

    Pinning substitutes ``A_pinned = A M + (I - M)`` with ``M`` the
    active-DOF projector from :meth:`KineticOperator.active_dof_mask`:
    identical to ``A`` on the active subspace, identity on the truncated
    DOFs.  For the physical right-hand sides (zero on truncated DOFs, as
    :meth:`KineticOperator.rhs` guarantees) the solution is unchanged, and
    both ``A_pinned`` and ``A_pinned^T`` are nonsingular, so forward solves,
    transposed solves, and implicit-function-theorem adjoints are all
    well-posed.  This is exactly the packed Fortran system, extended with
    trivial identity equations on the DOFs Fortran does not carry.
    """
    apply_t_raw = _transposed_apply(op)
    mask = op.active_dof_mask()
    if mask is None:
        return op.apply, apply_t_raw

    def matvec(v: jnp.ndarray) -> jnp.ndarray:
        return op.apply(mask * v) + (1.0 - mask) * v

    def matvec_t(w: jnp.ndarray) -> jnp.ndarray:
        return mask * apply_t_raw(w) + (1.0 - mask) * w

    return matvec, matvec_t


# Accepted slack over the residual the caller asked for, before a
# differentiable solve is declared failed.  The default of :func:`solve`'s
# ``adjoint_residual_factor``.
DEFAULT_ADJOINT_RESIDUAL_FACTOR = 10.0

# Multiple of the float64 unit roundoff in the attainable backward-error floor
# ``_ADJOINT_FLOOR_SLACK * eps * (||A|| ||x|| + ||b||)``.  A Krylov solve is
# backward stable, so its residual settles at that scale; on a near-singular
# system with a large-norm solution the floor is *above* ``tol * ||b||`` and
# the requested tolerance is simply unreachable in double precision.  Judging
# such a solve failed would abort perfectly good gradients.
_ADJOINT_FLOOR_SLACK = 32.0

# Hard ceiling on that floor, relative to ``||b||``.  Without it a *diverged*
# solve would excuse itself: its solution norm blows up, so a floor
# proportional to ``||A|| ||x||`` grows past the residual it is meant to
# judge.  ``x = 0`` always attains ``||r|| = ||b||``, so no argument about
# attainable accuracy can justify a residual anywhere near ``||b||``; 1e-6
# is far below any converged solve yet far above the near-singular
# stagnation the floor exists to tolerate.
_ADJOINT_FLOOR_MAX_RELATIVE = 1e-6

# Rademacher probes used to estimate ``||A||`` for that floor.  Each costs one
# operator apply against the hundreds the Krylov solve itself runs.
_ADJOINT_NORM_PROBES = 2

_EPS64 = float(np.finfo(np.float64).eps)


def _operator_norm_estimate(
    matvec: Callable[[jnp.ndarray], jnp.ndarray], n: int
) -> jnp.ndarray:
    """Probe estimate of ``||A||_2`` from a few Rademacher matvecs.

    ``max_i ||A u_i|| / ||u_i||`` over fixed pseudo-random sign vectors: a
    lower bound on the spectral norm, tight to within a small factor for the
    operators here, and used only to set the order of magnitude of the
    backward-error floor in :func:`_residual_guard`.
    """
    key = jax.random.PRNGKey(0)
    est = jnp.asarray(0.0, dtype=jnp.float64)
    for i in range(_ADJOINT_NORM_PROBES):
        u = jax.random.rademacher(jax.random.fold_in(key, i), (n,), dtype=jnp.float64)
        est = jnp.maximum(est, jnp.linalg.norm(matvec(u)) / jnp.linalg.norm(u))
    return est


def _residual_guard(
    label: str,
    rhs_index: int,
    *,
    tol: float,
    atol: float,
    factor: float,
    raise_on_failure: bool,
    diagnostics: AdjointDiagnostics | None,
) -> Callable[..., None]:
    """Host callback that records — and by default aborts on — a bad solve.

    Used on both the forward and the adjoint (transposed) GCROT solves of the
    ``differentiable=True`` recycled Krylov path.  A stalled *adjoint* solve is
    the dangerous one: the forward solution stays good, nothing in the returned
    :class:`SolveResult` changes, and the implicit-function-theorem VJP hands
    back a wrong gradient (the singular Fokker-Planck +
    ``constraintScheme=1`` failure mode).  Runs at execution time via
    ``jax.debug.callback``, so it fires under ``jit``/``grad`` tracing and,
    for the adjoint, during the backward pass.

    The residual handed in is recomputed from the operator itself
    (``||A^T y - g||``), never the Krylov method's internal recursive
    estimate.  It is accepted when it meets the requested
    ``max(atol, tol*||b||)`` *or* sits at the double-precision backward-error
    floor ``_ADJOINT_FLOOR_SLACK * eps * (||A|| ||x|| + ||b||)``, both times
    with ``factor`` slack.  The floor matters: when the operator is
    near-singular the adjoint solution norm explodes, and no backward-stable
    method can then drive ``||A^T y - g||`` down to ``tol * ||g||``.  It is
    capped at ``_ADJOINT_FLOOR_MAX_RELATIVE * ||b||`` so a diverged solve
    cannot excuse itself with its own inflated solution norm.
    """

    def guard(
        res_norm: jnp.ndarray,
        rhs_norm: jnp.ndarray,
        sol_norm: jnp.ndarray,
        op_norm: jnp.ndarray,
    ) -> None:
        res = float(np.asarray(res_norm))
        b_norm = float(np.asarray(rhs_norm))
        x_norm = float(np.asarray(sol_norm))
        a_norm = float(np.asarray(op_norm))
        target = max(atol, tol * b_norm)
        floor = min(
            _ADJOINT_FLOOR_SLACK * _EPS64 * (a_norm * x_norm + b_norm),
            _ADJOINT_FLOOR_MAX_RELATIVE * b_norm,
        )
        limit = factor * max(target, floor)
        ok = bool(
            np.isfinite(res)
            and np.isfinite(x_norm)
            and np.isfinite(a_norm)
            and res <= limit
        )
        record = SolveResidualRecord(
            label=label,
            rhs_index=rhs_index,
            residual_norm=res,
            rhs_norm=b_norm,
            solution_norm=x_norm,
            operator_norm=a_norm,
            target=target,
            floor=floor,
            limit=limit,
            within_tolerance=ok,
        )
        if diagnostics is not None:
            diagnostics.records.append(record)
        if ok or not raise_on_failure:
            return
        rel = record.relative_residual
        formula = (
            "||A^T y - g|| / ||g||"
            if label.startswith("adjoint")
            else ("||A x - b|| / ||b||")
        )
        raise RuntimeError(
            f"[dkx.solve] differentiable {label} GCROT solve failed to "
            f"converge on right-hand side {rhs_index}: the true relative "
            f"residual {formula} is {rel:.3e} (residual {res:.3e}, "
            f"rhs norm {b_norm:.3e}, solution norm {x_norm:.3e}), above the accepted "
            f"{limit:.3e} = {factor:g} x max(requested {target:.3e}, "
            f"float64 backward-error floor {floor:.3e}). Returning would "
            "silently corrupt the implicit-differentiation gradient, so the "
            "solve aborts instead.\n"
            "Likely cause: a singular or near-singular operator whose null "
            "space the constraint scheme does not span — full Fokker-Planck "
            "with constraintScheme=1, or an Er xDot/xiDot deck under the "
            "per-speed constraintScheme=2 border. The physical drive stays "
            "in the range of A, so the forward solve converges and only the "
            "transposed solve against a generic cotangent stalls; that is "
            "why nothing else in the SolveResult looks wrong.\n"
            "Remedies: raise the resolution or pick a constraint scheme that "
            "regularizes this operator (SFINCS pairs constraintScheme=1 with "
            "the Fokker-Planck/Sugama operators and 2 with pitch-angle "
            "scattering); referee the case on a direct route "
            "(method='block_tridiagonal' where available, or method='direct' "
            "for a non-differentiable check of the forward solve); tighten "
            "tol and raise max_restarts; or, if you have verified the "
            "gradient against finite differences yourself, pass "
            "check_adjoint=False (or raise adjoint_residual_factor) and read "
            "SolveResult.adjoint for the residuals behind this decision."
        )

    return guard


def _guarded_solve(
    label: str,
    rhs_index: int,
    matvec: Callable[[jnp.ndarray], jnp.ndarray],
    rhs_col: jnp.ndarray,
    x: jnp.ndarray,
    *,
    tol: float,
    atol: float,
    factor: float,
    raise_on_failure: bool,
    diagnostics: AdjointDiagnostics | None,
) -> jnp.ndarray:
    """Measure ``x``'s true residual against ``matvec`` and hand it to the guard.

    Returns ``x`` unchanged: the check observes, it never alters the solution,
    so forward answers are bit-identical with and without it.
    """
    residual = matvec(x) - rhs_col
    jax.debug.callback(
        _residual_guard(
            label,
            rhs_index,
            tol=tol,
            atol=atol,
            factor=factor,
            raise_on_failure=raise_on_failure,
            diagnostics=diagnostics,
        ),
        jnp.linalg.norm(residual),
        jnp.linalg.norm(rhs_col),
        jnp.linalg.norm(x),
        _operator_norm_estimate(matvec, rhs_col.shape[0]),
    )
    return x


# =============================================================================
# Structured direct (block Thomas over Legendre modes)
# =============================================================================


def tier1_available(op: KineticOperator) -> tuple[bool, str]:
    """Check whether the structured direct family applies to ``op``.

    The decision is driven by the operator's own block extraction: if
    :meth:`KineticOperator.legendre_blocks` refuses (Er L±2 terms,
    Fokker-Planck collisions), the structured direct route is off.  On top of
    that the bordered constraint machinery must be diagonal over (species, x)
    (``constraintScheme`` 0 or 2 without ``point_at_x0``).  Non-uniform
    ``Nxi_for_x`` (the production speed-dependent Legendre ramp) is accepted:
    every (species, x) subsystem is closed, so the truncated structured direct
    kernel solves it with its own ``n_blocks = Nxi_for_x[ix]`` — exactly the
    packed Fortran system.  Only the full-band factorization
    (:func:`build_tier1_solver`) additionally requires uniform ``Nxi_for_x``;
    ramped decks always route through the truncated kernel.
    """
    try:
        op._check_block_extraction_supported()
    except NotImplementedError as exc:
        return False, str(exc)
    if op.constraint_scheme not in (0, 2):
        return False, (
            f"constraintScheme={op.constraint_scheme} borders couple speed nodes; "
            "only 0 and 2 keep the (species, x) block split exact"
        )
    if op.constraint_scheme == 2 and op.point_at_x0:
        return False, "point_at_x0 x-grids give the x=0 constraint row a different form"
    return True, ""


def _uniform_nxi_for_x(op: KineticOperator) -> bool:
    """Whether every speed node retains the full Legendre resolution."""
    return int(np.min(np.asarray(op.n_xi_for_x))) >= op.n_xi


# =============================================================================
# Structured direct memory model and the full-vs-truncated route decision
# =============================================================================


def tier1_full_band_bytes(op: KineticOperator) -> float:
    """Bytes of the structured direct Legendre bands (``lower``/``diag``/``upper``).

    :func:`build_tier1_solver` materializes the three block-tridiagonal bands
    of :meth:`KineticOperator.to_block_tridiagonal`, each of shape
    ``(n_xi, n_species, n_x, m, m)`` with block dimension ``m = n_theta *
    n_zeta`` (the dense theta*zeta angular block per Legendre mode, per
    (species, x) subsystem), in float64::

        bytes = 3 * sum_x(Nxi_for_x) * n_species * (n_theta * n_zeta)**2 * 8

    The leading ``3`` counts ``lower``, ``diag`` and ``upper``; a subsystem at
    speed node ``ix`` carries only its own ``Nxi_for_x[ix]`` Legendre blocks
    (``sum_x(Nxi_for_x) = n_xi * n_x`` for uniform ``Nxi_for_x``).  This is
    the ~39 GB figure for the 744k-unknown uniform HSX case (n_theta=25,
    n_zeta=51, n_xi=100, n_x=5, n_species=2).
    """
    m = float(op.n_theta * op.n_zeta)
    n_blocks_total = float(np.sum(np.asarray(op.n_xi_for_x)))
    return 3.0 * n_blocks_total * float(op.n_species) * m * m * 8.0


def tier1_peak_memory_bytes(op: KineticOperator) -> float:
    """Peak-memory estimate of the full structured direct factorization.

    Adds the block-Thomas LU factors and elimination temporaries on top of the
    three input bands (:func:`tier1_full_band_bytes`).  The
    ``BlockTridiagFactors`` store the per-block LU factors plus the two
    off-diagonal bands (~2x the band storage), and the vmapped sweep holds a
    few block temporaries live, so the peak is estimated at ``2.5x`` the band
    storage — the multiplier used by the validated HSX benchmark.
    """
    return 2.5 * tier1_full_band_bytes(op)


def tier1_truncated_peak_memory_bytes(
    op: KineticOperator,
    keep_lowest: int = _TIER1_KEEP_LOWEST_DEFAULT,
    subsystem_batch: int | str = "auto",
) -> float:
    """Working-set estimate of :func:`_solve_tier1_truncated` (structured direct).

    The truncated route never materializes the full Legendre bands, so the
    ~``tier1_peak_memory_bytes`` full-band peak wildly overestimates it (46x on
    the 1.27M-DOF production deck).  Its live buffers, with block dimension
    ``m = n_theta * n_zeta``, subsystem batch ``B = n_species * n_x``, and
    concurrent elimination width ``w = subsystem_batch`` (float64, 8 bytes
    each):

    * the compact coefficient set (:func:`_truncated_coefficients`): the two
      angular derivative matrices, the ExB matrix, the kron assembly
      temporaries, and the per-species streaming matrices —
      ``(5 + n_species) * m^2`` entries;
    * the per-subsystem broadcast of the streaming matrix
      (``jnp.repeat`` to the ``B`` axis) — ``B * m^2`` entries;
    * ``w`` concurrent ``solvax.direct.block_thomas_truncated_fn`` sweeps
      (the batched ``jax.lax.map(..., batch_size=w)`` elimination in
      :func:`_solve_tier1_truncated`): per subsystem the LU carry, the
      assembled ``(L, D, U)`` block triple, elimination temporaries, and the
      stacked ``keep`` head factors — ``w * (2 * keep + 8) * m^2`` entries,
      doubled for the ``jax.lax.map`` pipeline (one batch in flight while the
      next is staged; ``w = 1`` is the fully serial sweep);
    * the state buffers (zero-padded full-shape solution, its RHS reshape,
      and the assembly/concat copies) — ``4 * total_size`` entries.

    ``subsystem_batch="auto"`` models the width the solve itself resolves
    (:func:`_resolve_subsystem_batch`: width 1 on the CPU backend, the
    memory-budgeted :func:`tier1_truncated_subsystem_width` on accelerators);
    an integer models that fixed width (clamped to ``[1, B]``).  The sum is
    doubled as a safety margin for allocator slack and XLA fusion
    temporaries.  Validated against measured process peaks on the profiling
    deck ladder (production 1.27M / mid 337k / small 41k DOFs): the estimate
    lands within about 1.1-1.5x of measurement, on the high side.
    """
    m = float(op.n_theta * op.n_zeta)
    mm_bytes = m * m * 8.0
    n_s = float(op.n_species)
    batch = n_s * float(op.n_x)
    keep = float(min(int(keep_lowest), int(op.n_xi)))
    if isinstance(subsystem_batch, str):
        width = float(_resolve_subsystem_batch(op, subsystem_batch, int(keep)))
    else:
        width = float(max(1, min(int(subsystem_batch), int(batch))))
    coeff_bytes = (5.0 + n_s) * mm_bytes
    stream_broadcast_bytes = batch * mm_bytes
    sweep_bytes = 2.0 * width * (2.0 * keep + 8.0) * mm_bytes
    state_bytes = 4.0 * float(op.total_size) * 8.0
    return 2.0 * (coeff_bytes + stream_broadcast_bytes + sweep_bytes + state_bytes)


def tier1_truncated_subsystem_width(
    op: KineticOperator,
    keep_lowest: int = _TIER1_KEEP_LOWEST_DEFAULT,
    memory_budget_gb: float | None = None,
) -> int:
    """Largest subsystem batch width whose modeled footprint fits the budget.

    The memory-aware chooser behind ``subsystem_batch="auto"`` on accelerator
    backends: the widest ``w in [1, B]`` (``B = n_species * n_x``) such that
    :func:`tier1_truncated_peak_memory_bytes` with ``subsystem_batch=w`` stays
    within :func:`dkx.batch.resolve_memory_budget_bytes` — an explicit
    ``memory_budget_gb``, else a fraction of the device/host memory.  Width 1
    reproduces the fully serial per-subsystem elimination, so a tight budget
    degrades gracefully to the minimum-memory behavior.
    """
    from .batch import resolve_memory_budget_bytes  # local import: batch imports solve

    budget = resolve_memory_budget_bytes(memory_budget_gb)
    b = max(1, int(op.n_species) * int(op.n_x))
    for width in range(b, 1, -1):
        if (
            tier1_truncated_peak_memory_bytes(op, keep_lowest, subsystem_batch=width)
            <= budget
        ):
            return width
    return 1


def _resolve_subsystem_batch(
    op: KineticOperator, subsystem_batch: int | str, keep: int
) -> int:
    """Map the ``solve(subsystem_batch=...)`` knob to a concrete width.

    ``"auto"`` is backend-aware:

    * CPU backend — width 1, the fully serial sweep.  Measured on the
      10-core M4 profiling host (336,610-DOF hsx_pas_dkes_mid warm solves,
      8 threads): every width > 1 is neutral-to-slower than width 1 (ramped
      deck 10.3 s at width 1 vs 11.4 s grouped width 2; uniform-Nxi variant
      16.6 s at width 1 vs 20.5 s at width 10), because XLA:CPU executes the
      batch axis of the LAPACK factor/solve custom calls serially per
      element with extra cache pressure — the batched sweep adds memory,
      not CPU parallelism.
    * accelerator backends — the widest width whose modeled footprint fits
      the memory budget (:func:`tier1_truncated_subsystem_width`); batched
      scans raise device occupancy there, and the budget clamp bounds the
      working set.
    """
    if isinstance(subsystem_batch, str):
        if subsystem_batch.strip().lower() != "auto":
            raise ValueError(
                f"unknown subsystem_batch {subsystem_batch!r}; expected 'auto' "
                "or a positive integer width"
            )
        if jax.default_backend() == "cpu":
            return 1
        return tier1_truncated_subsystem_width(op, keep_lowest=keep)
    width = int(subsystem_batch)
    if width < 1:
        raise ValueError(f"subsystem_batch must be >= 1, got {width}")
    return min(width, max(1, int(op.n_species) * int(op.n_x)))


def _tier1_budget_bytes(budget_gb: float | None) -> tuple[float, float]:
    """Resolve the truncation budget (bytes, GB) from arg / env / default."""
    if budget_gb is None:
        env = os.environ.get(_TIER1_BUDGET_ENV)
        budget_gb = float(env) if env not in (None, "") else _TIER1_BUDGET_GB_DEFAULT
    return float(budget_gb) * 2.0**30, float(budget_gb)


def _truncation_supported(op: KineticOperator, keep: int) -> tuple[bool, str]:
    """Structural check that the truncated structured direct kernel applies.

    Assumes :func:`tier1_available` already passed (PAS/DKES family,
    constraintScheme in {0, 2}, no point_at_x0).  Additionally every closed
    (species, x) subsystem must retain at least ``keep`` Legendre blocks,
    unless full recovery (``keep == n_xi``) is requested. Full recovery retains
    each chain's own ``Nxi_for_x[ix]`` blocks and pads only inactive DOFs.
    """
    if op.constraint_scheme not in (0, 2):
        return (
            False,
            f"constraintScheme={op.constraint_scheme} border couples Legendre modes",
        )
    if op.point_at_x0:
        return False, "point_at_x0 x-grids are not handled by the truncated kernel"
    if keep > op.n_xi:
        return False, f"keep_lowest={keep} exceeds Nxi={op.n_xi}"
    if keep != op.n_xi and int(np.min(np.asarray(op.n_xi_for_x))) < keep:
        return (
            False,
            f"min Nxi_for_x={int(np.min(np.asarray(op.n_xi_for_x)))} < keep_lowest={keep}",
        )
    return True, ""


def _rhs_confined_to_lowest_blocks(
    op: KineticOperator, rhs2d: jnp.ndarray, keep: int
) -> bool | None:
    """Whether the RHS has Legendre support only on modes ``l < keep``.

    Returns ``None`` when ``rhs2d`` is a tracer (support cannot be read under
    jit/grad); callers then fall back to the structural ``rhs_mode`` guarantee.
    The truncated kernel computes exactly the lowest ``keep`` Legendre blocks
    and zero-pads the rest, so it is exact iff both the drive and the requested
    output moments live on ``l < keep`` — true for the RHSMode 1/2/3 transport
    drives and their fluxes/flows/sources, which touch only ``l <= 2``.
    """
    if _is_traced(rhs2d):
        return None
    n_s, n_x, n_xi, n_t, n_z = op.f_shape
    if keep >= n_xi:
        return True
    f = np.asarray(rhs2d)[: op.f_size].reshape(n_s, n_x, n_xi, n_t * n_z, -1)
    return bool(np.max(np.abs(f[:, :, keep:])) == 0.0)


@dataclass(frozen=True)
class Tier1Solver:
    """Factored per-(species, x) bordered block-tridiagonal solver.

    Holds the batched block-Thomas factors of the rank-one-regularized
    Legendre bands ``A~ = A + gamma B C`` for every (species, x) subsystem,
    plus the presolved border columns ``z = A~^{-1} B`` (forward) and
    ``z_t = A~^{-T} C^T`` (transpose), so both the forward and the adjoint
    bordered solve reuse the same elimination.
    """

    op: KineticOperator
    factors: BlockTridiagFactors  # leading batch axis B = S*X
    z_fwd: jnp.ndarray  # (B, L, TZ)
    z_t: jnp.ndarray  # (B, L, TZ)
    gamma: jnp.ndarray  # (B,)
    b0: jnp.ndarray  # (TZ,) source column shape on the l=0 rows
    c0: jnp.ndarray  # (TZ,) constraint row (flux-surface-average weights)

    def solve(self, rhs: jnp.ndarray, transpose: bool = False) -> jnp.ndarray:
        """Solve ``K x = rhs`` (or ``K^T x = rhs``) for flat state vector(s).

        Args:
            rhs: ``(total_size,)`` or ``(total_size, n_rhs)``.
            transpose: solve the transposed bordered system, reusing the same
                factors via ``block_thomas_solve(transpose=True)``.

        Returns:
            Solution(s) with the same shape as ``rhs``.
        """
        op = self.op
        rhs2d, squeeze = _as_columns(rhs)
        n_rhs = rhs2d.shape[1]
        n_s, n_x, n_xi, n_t, n_z = op.f_shape
        batch = n_s * n_x
        n_tz = n_t * n_z

        # f part -> (B, L, TZ, n_rhs)
        b_f = rhs2d[: op.f_size].reshape(n_s, n_x, n_xi, n_tz, n_rhs)
        b_f = b_f.reshape(batch, n_xi, n_tz, n_rhs)

        solve_batched = jax.vmap(
            lambda f, r: block_thomas_solve(f, r, transpose=transpose)
        )
        y = solve_batched(self.factors, b_f)  # (B, L, TZ, n_rhs)

        if op.constraint_scheme == 0:
            x = y.reshape(op.f_size, n_rhs)
            return x[:, 0] if squeeze else x

        # constraintScheme=2: one bordered unknown per (species, x).
        # Forward:  [[A, b0 e0], [c0^T e0^T, 0]];  transpose swaps b0 <-> c0.
        r_c = rhs2d[op.f_size :].reshape(batch, n_rhs)
        z = self.z_t if transpose else self.z_fwd
        w_row = (
            self.b0 if transpose else self.c0
        )  # constraint row of the (transposed) system
        c_y = jnp.einsum("j,bjr->br", w_row, y[:, 0])  # w·y[l=0], (B, n_rhs)
        c_z = jnp.einsum("j,bj->b", w_row, z[:, 0])  # (B,)
        s = self.gamma[:, None] * r_c + (c_y - r_c) / c_z[:, None]
        shift = s - self.gamma[:, None] * r_c  # (B, n_rhs)
        f = y - shift[:, None, None, :] * z[:, :, :, None]

        x = jnp.concatenate(
            [f.reshape(op.f_size, n_rhs), s.reshape(op.extra_size, n_rhs)], axis=0
        )
        return x[:, 0] if squeeze else x


def build_tier1_solver(op: KineticOperator) -> Tier1Solver:
    """Assemble and factor the batched bordered block-tridiagonal solver.

    This is the full-band structured direct route.

    Uses the analytic (probing-free) :meth:`KineticOperator.to_block_tridiagonal`
    blocks — the replacement for the retired probing-based RHSMode=3 solver POC
    — and absorbs the ``constraintScheme=2`` border with the exact rank-one
    trick ``A~ = A + gamma B C`` documented in the module docstring.

    Raises:
        NotImplementedError: when :func:`tier1_available` says no.
    """
    _require_solvax()
    ok, reason = tier1_available(op)
    if not ok:
        raise NotImplementedError(
            f"structured direct route unavailable: {reason}"
        )
    if not _uniform_nxi_for_x(op):
        raise NotImplementedError(
            "the full-band structured direct factorization requires uniform "
            "Nxi_for_x (the ramped bands carry singular zero rows on the "
            "truncated DOFs); ramped decks "
            "route through the truncated kernel (method='block_tridiagonal_truncated')"
        )

    n_s, n_x, n_xi, n_t, n_z = op.f_shape
    n_tz = n_t * n_z
    batch = n_s * n_x

    blocks = op.to_block_tridiagonal()  # (L, S, X, TZ, TZ)
    lower, diag, upper = (
        jnp.transpose(a, (1, 2, 0, 3, 4)).reshape(batch, n_xi, n_tz, n_tz)
        for a in blocks
    )

    b0 = jnp.ones((n_tz,), dtype=jnp.float64)  # source shape on the l=0 rows
    c0 = op._fs_average_factor().reshape(-1)  # flux-surface-average constraint row

    if op.constraint_scheme == 2:
        # Conditioning-friendly rank-one scale per (species, x): mean |diag entry|
        # of the bands over the max magnitude of the rank-one update.
        scale = jnp.mean(jnp.abs(jnp.diagonal(diag, axis1=2, axis2=3)), axis=(1, 2))
        scale = jnp.where(scale > 0.0, scale, jnp.mean(jnp.abs(diag), axis=(1, 2, 3)))
        outer_max = jnp.max(jnp.abs(b0)) * jnp.max(jnp.abs(c0))
        gamma = scale / outer_max
        diag = diag.at[:, 0].add(gamma[:, None, None] * jnp.outer(b0, c0)[None, :, :])
    else:
        gamma = jnp.ones((batch,), dtype=jnp.float64)

    factors = jax.vmap(block_thomas_factor)(lower, diag, upper)

    e0 = jnp.zeros((batch, n_xi, n_tz), dtype=jnp.float64)
    z_fwd = jax.vmap(block_thomas_solve)(factors, e0.at[:, 0, :].set(b0[None, :]))
    z_t = jax.vmap(lambda f, r: block_thomas_solve(f, r, transpose=True))(
        factors, e0.at[:, 0, :].set(c0[None, :])
    )
    return Tier1Solver(
        op=op, factors=factors, z_fwd=z_fwd, z_t=z_t, gamma=gamma, b0=b0, c0=c0
    )


# =============================================================================
# Sparse direct — host fallback and independent cross-check
# =============================================================================


def materialize_csr(
    op: KineticOperator, *, column_chunk: int = 1024, pin_masked_dofs: bool = False
):
    """Materialize the full bordered operator as a scipy CSR matrix.

    ``solvax.native_eigen.sparse_operator_matrix`` owns the sampling: it
    applies the matrix-free operator to identity columns in vmapped chunks and
    sparsifies each chunk as it lands, so no dense ``O(total_size**2)`` array
    is ever formed.  The kinetic operator is a stencil (about 9 of 1121
    entries per angular row; see :mod:`dkx.sparse_precond`), so CSR is what the
    matrix actually is.

    The cost that is *not* removed is the sampling itself: ``total_size``
    operator applications, one per column.  That, not the intermediate, is what
    ``max_dense_size`` guards.

    Args:
        op: the kinetic operator.
        column_chunk: identity columns per vmapped batch.
        pin_masked_dofs: materialize the pinned operator (identity rows and
            columns on the DOFs truncated by ``Nxi_for_x``; see
            :func:`_pinned_matvecs`) instead of the raw rectangular embedding,
            which has exact zero rows on those DOFs.
    """
    _require_solvax()
    apply = _pinned_matvecs(op)[0] if pin_masked_dofs else op.apply
    prototype = jnp.zeros((op.total_size,), dtype=jnp.float64)
    return sparse_operator_matrix(apply, prototype, batch_size=column_chunk)


def materialize_dense(
    op: KineticOperator, *, column_chunk: int = 1024, pin_masked_dofs: bool = False
) -> np.ndarray:
    """:func:`materialize_csr` as a dense numpy matrix, for referee tests.

    Densifying costs ``O(total_size**2)`` memory, which is why the sparse
    direct route uses the CSR form directly and only tests come through here.
    """
    return materialize_csr(
        op, column_chunk=column_chunk, pin_masked_dofs=pin_masked_dofs
    ).toarray()


def _escalate_after_tier2_stall(
    op: KineticOperator,
    rhs2d: jnp.ndarray,
    *,
    stalled: SolveResult,
    tol: float,
    atol: float,
    x0,
    recycle,
    preconditioner: str,
    drop_l_coupling_in_precond: bool,
    restart: int,
    recycle_dim: int,
    max_restarts: int,
    check_adjoint: bool,
    adjoint_residual_factor: float,
    max_dense_size: int,
) -> SolveResult:
    """Work through the remedies a user would otherwise have to know about.

    A stalled Krylov solve is a *preconditioner* problem, not a reason to
    change solver route, and this is where DKX used to get it backwards: it
    announced a fall back to the sparse direct host solve, which obtains its
    matrix by applying the operator to ``n`` identity columns.  At the sizes
    where recycled Krylov actually stalls that is hopeless -- a 66004-DOF deck
    needs 66004 operator applications before the factorization even starts --
    so the guard in :func:`_solve_tier3` refused, and a convergence problem
    surfaced as a hard crash telling the user to raise ``max_dense_size``.
    Sampling into CSR (:func:`materialize_csr`) removed the dense ``O(n^2)``
    intermediate that used to sit on top of that, but not the ``n``
    applications, so the guard stands and this ladder still ends elsewhere.

    SFINCS Fortran v3 does not have this failure mode, and the reason is worth
    stating because it dictates the order below.  It assembles the simplified
    preconditioner matrix *analytically and sparsely*, factorizes it with a
    sparse direct LU (MUMPS or SuperLU_dist), and preconditions GMRES with
    that -- so its "direct solve" is a sparse factorization that handles 66004
    routinely, and GMRES needs few iterations on top of it.  It also retries
    automatically, doubling the MUMPS working-memory factor on a failed
    factorization (``solver.F90``: ``mumps_icntl_14 = mumps_icntl_14 * 2``).

    DKX already owns the equivalent of that preconditioner -- ``"sparse"``
    eliminates in a fill-reducing order and keeps the inverse exact -- and the
    stall above happened without ever trying it.  So the ladder escalates the
    preconditioner first, in increasing cost, and only considers the sparse
    direct route at a size where it can actually run.

    Returns the first converged result, or the best (lowest final residual) if
    none converges, so a caller that can tolerate a loose solve still gets the
    best available answer along with an honest ``converged=False``.
    """
    attempts: list[tuple[str, SolveResult]] = [
        (f"{preconditioner} preconditioner", stalled)
    ]

    def _residual(result: SolveResult) -> float:
        norms = np.asarray(result.residual_norms)
        return float(norms[-1]) if norms.size else float("inf")

    print(
        f"[dkx.solve] recycled Krylov solve stalled with the {preconditioner} "
        f"preconditioner (iterations={stalled.iterations}, residual="
        f"{_residual(stalled):.3e}); escalating rather than giving up."
    )

    # Rung 1: the strong preconditioner.  This is the SFINCS analogue and the
    # single most likely fix, because "coarse" eliminates L first, which fills
    # the angular stencils in, while "sparse" eliminates in a fill-reducing
    # order and stays sparse.
    for kind in ("sparse", "multigrid"):
        if kind == preconditioner:
            continue
        try:
            print(
                f"[dkx.solve]   retrying the iterative solve with the {kind} preconditioner ..."
            )
            candidate = _solve_tier2(
                op,
                rhs2d,
                tol=tol,
                atol=atol,
                x0=x0,
                recycle=recycle,
                preconditioner=kind,
                drop_l_coupling_in_precond=drop_l_coupling_in_precond,
                restart=restart,
                recycle_dim=recycle_dim,
                max_restarts=max_restarts,
                differentiable=False,
                check_adjoint=check_adjoint,
                adjoint_residual_factor=adjoint_residual_factor,
            )
        except (
            Exception
        ) as exc:  # a preconditioner that cannot build is a rung, not a failure
            print(f"[dkx.solve]   {kind} preconditioner unavailable: {exc}")
            continue
        attempts.append((f"{kind} preconditioner", candidate))
        if candidate.converged:
            print(f"[dkx.solve]   converged with the {kind} preconditioner.")
            return candidate

    # Rung 2: more iterations.  Cheap to ask for, and the remedy when the
    # preconditioner is adequate but the cap was simply too low for this Er.
    widened = max(max_restarts * 4, max_restarts + 1)
    best_kind = max(attempts, key=lambda item: -_residual(item[1]))[0].split()[0]
    if best_kind not in _TIER2_PRECONDITIONERS:
        best_kind = preconditioner
    try:
        print(
            f"[dkx.solve]   retrying the iterative solve with the {best_kind} preconditioner "
            f"and {widened} restarts (was {max_restarts}) ..."
        )
        candidate = _solve_tier2(
            op,
            rhs2d,
            tol=tol,
            atol=atol,
            x0=x0,
            recycle=recycle,
            preconditioner=best_kind,
            drop_l_coupling_in_precond=drop_l_coupling_in_precond,
            restart=restart,
            recycle_dim=recycle_dim,
            max_restarts=widened,
            differentiable=False,
            check_adjoint=check_adjoint,
            adjoint_residual_factor=adjoint_residual_factor,
        )
        attempts.append((f"{best_kind} preconditioner, {widened} restarts", candidate))
        if candidate.converged:
            print("[dkx.solve]   converged with the widened iteration budget.")
            return candidate
    except Exception as exc:
        print(f"[dkx.solve]   widened retry failed: {exc}")

    # Rung 3: sparse direct, but only where it can actually run.  Announcing a
    # fallback that the size guard will refuse is what produced the original
    # crash, so the size is checked here rather than discovered downstream.
    if op.total_size <= max_dense_size:
        print("[dkx.solve]   falling back to the sparse direct host solve ...")
        return _solve_tier3(
            op, rhs2d, tol=tol, atol=atol, max_dense_size=max_dense_size
        )

    best_label, best = min(attempts, key=lambda item: _residual(item[1]))
    raise RuntimeError(
        f"the linear solve did not converge at total_size={op.total_size}.\n"
        f"Tried: {'; '.join(label for label, _ in attempts)}.\n"
        f"Best final residual {_residual(best):.3e} ({best_label}), tolerance {tol:.1e}.\n"
        "The sparse direct fallback was not attempted. Sampling the operator is "
        f"not the obstacle -- it is vmapped in column chunks, and at n={op.total_size} "
        "was measured at about 160 s -- but the factorization is: SuperLU ran for "
        "75 minutes on a matrix of this size and shape (n=66004, 33.5 nonzeros per "
        "row) without finishing, so raising max_dense_size trades a stall for a "
        "longer one.\n"
        "What usually helps, in order:\n"
        "  1. RAISE Nxi. A stall at large |Er| and low collisionality is usually "
        "under-resolution in pitch angle, not a solver defect: the distribution "
        "develops fine structure at the trapped-passing boundary that a coarse Nxi "
        "cannot represent, and the discrete operator is then nearly singular. "
        "Measured on one such deck at Er=15: Nxi=20 failed after 6182 s, while "
        "Nxi=30 converged in 406 s and Nxi=80 in 359 s -- more resolution ran "
        "*faster*, because the better-conditioned operator needs far fewer "
        "iterations.\n"
        "  2. Check that the answer is converged, not merely obtained. On that same "
        "deck FSABjHat was +5.9e-3 at Nxi=30, +3.0e-3 at Nxi=40, -3.4e-3 at Nxi=60 "
        "and -4.1e-3 at Nxi=80: it changes sign. A solve that converges at low Nxi "
        "is not evidence that the physics is resolved, so scan Nxi before trusting "
        "any of it.\n"
        "  3. Lower |Er|, or raise solverTolerance, if the point is not needed at "
        "full accuracy. Reducing Nxi/Ntheta/Nzeta makes this failure mode worse, "
        "not better.\n"
        "Pass solver=SolverOptions(...) to control the solver directly."
    )


def _solve_tier3(
    op: KineticOperator,
    rhs2d: jnp.ndarray,
    *,
    tol: float,
    atol: float,
    max_dense_size: int,
) -> SolveResult:
    _require_solvax()
    if _is_traced(rhs2d):
        raise RuntimeError(
            "the sparse direct (host SuperLU) solve is non-differentiable and "
            "cannot run "
            "under jit/vmap/grad; use method='block_tridiagonal' or 'gmres' with "
            "differentiable=True."
        )
    n = op.total_size
    if n > max_dense_size:
        raise RuntimeError(
            f"sparse direct materialization refused: total_size={n} > "
            f"max_dense_size={max_dense_size}, i.e. {n} operator applications to "
            "sample the matrix column by column; raise max_dense_size explicitly "
            "if you really want this."
        )
    print(
        f"[dkx.solve] sparse direct route (host SuperLU, n={n}): "
        "non-differentiable fallback path."
    )
    t0 = time.perf_counter()
    lu = SpluFactorization(materialize_csr(op, pin_masked_dofs=True))
    t1 = time.perf_counter()
    x2d = jnp.asarray(lu.solve(np.asarray(rhs2d)))
    if x2d.ndim == 1:
        x2d = x2d[:, None]
    t2 = time.perf_counter()
    res = _residual_norms(_pinned_matvecs(op)[0], x2d, rhs2d)
    return SolveResult(
        x=x2d,
        method="direct",
        iterations=None,
        residual_norms=res,
        converged=_converged_flag(res, rhs2d, tol, atol),
        recycle=None,
        timings={"build": t1 - t0, "solve": t2 - t1},
    )


# =============================================================================
# Route drivers
# =============================================================================


def _implicit_solve(
    matvec: Callable[[jnp.ndarray], jnp.ndarray],
    matvec_t: Callable[[jnp.ndarray], jnp.ndarray],
    rhs_col: jnp.ndarray,
    fwd_solve: Callable[[jnp.ndarray], jnp.ndarray],
    t_solve: Callable[[jnp.ndarray], jnp.ndarray],
    *,
    has_aux: bool = False,
):
    """One differentiable column solve via ``solvax.implicit.linear_solve``.

    The single ``solver`` callable required by the API dispatches between the
    forward and transposed factorized solves by identity of the matvec it is
    handed (``linear_solve`` passes ``transpose_matvec`` through verbatim).

    With ``has_aux`` both callables return ``(x, aux)`` and the *forward*
    solve's ``aux`` comes back beside the solution.  That is how the recycled
    Krylov route reads its iteration count, residual and recycle pair off the
    one solve the wrapper runs, instead of running a second solve outside it
    just to see them.
    """

    def solver(mv: Callable, b: jnp.ndarray):
        return t_solve(b) if mv is matvec_t else fwd_solve(b)

    return solvax_linear_solve(
        matvec, rhs_col, solver, transpose_matvec=matvec_t, has_aux=has_aux
    )


def _solve_tier1(
    op: KineticOperator,
    rhs2d: jnp.ndarray,
    *,
    tol: float,
    atol: float,
    differentiable: bool,
) -> SolveResult:
    t0 = time.perf_counter()
    t1_solver = build_tier1_solver(op)
    # Force the async block-Thomas factorization to complete so the "build"
    # timing reflects real compute, not JAX dispatch latency.  We block on the
    # array fields (the Tier1Solver dataclass itself is not a pytree, so
    # block_until_ready would treat it as an opaque leaf); a no-op under
    # jit/grad tracing.
    jax.block_until_ready(
        (t1_solver.factors, t1_solver.z_fwd, t1_solver.z_t, t1_solver.gamma)
    )
    t1 = time.perf_counter()

    def _solve_refined(b: jnp.ndarray, *, transpose: bool = False) -> jnp.ndarray:
        """Factor solve plus iterative refinement (defect correction).

        The block-Thomas elimination is backward-stable but its rounding can
        leave the true relative residual a small multiple of eps above the
        strict production gate; refinement ``x += solve(b - A x)`` (one extra
        apply and substitution on the existing factors per sweep) takes it to
        O(1e-16).

        :func:`solvax.refine.iterative_refinement` owns the recurrence.  It is
        the same fixed point as the hand-rolled single pass this replaces --
        one sweep reproduces it exactly -- and it is where the float32-factor
        variant lives (``solvax.refine.as_low_precision``), which is the
        documented route to a GPU that runs FP64 at 1/32 rate.  Reusable
        numerical kernels belong upstream in ``solvax`` rather than here.
        """
        apply = _transposed_apply(op) if transpose else op.apply
        apply2d = apply if b.ndim == 1 else jax.vmap(apply, in_axes=1, out_axes=1)
        x, _residual_norms = iterative_refinement(
            apply2d,
            b,
            lambda r: t1_solver.solve(r, transpose=transpose),
            iterations=_TIER1_REFINEMENT_SWEEPS,
        )
        return x

    if differentiable:
        apply_t = _transposed_apply(op)
        cols = [
            _implicit_solve(
                op.apply,
                apply_t,
                rhs2d[:, j],
                lambda b: _solve_refined(b),
                lambda b: _solve_refined(b, transpose=True),
            )
            for j in range(rhs2d.shape[1])
        ]
        x2d = jnp.stack(cols, axis=1)
    else:
        x2d = _solve_refined(rhs2d)
    x2d = jax.block_until_ready(x2d)  # real solve compute, not just dispatch
    t2 = time.perf_counter()
    res = _residual_norms(op.apply, x2d, rhs2d)
    return SolveResult(
        x=x2d,
        method="block_tridiagonal",
        iterations=None,
        residual_norms=res,
        converged=_converged_flag(res, rhs2d, tol, atol),
        recycle=None,
        timings={"build": t1 - t0, "solve": t2 - t1},
    )


# =============================================================================
# Structured direct (truncated) — block Thomas over the lowest K Legendre modes
# =============================================================================


def _solve_tier1_truncated(
    op: KineticOperator,
    rhs2d: jnp.ndarray,
    *,
    keep: int,
    tol: float,
    atol: float,
    subsystem_batch: int | str = "auto",
    adjoint_window: int | None = None,
) -> SolveResult:
    """Memory-lean structured direct solve: the lowest ``keep`` Legendre blocks.

    Assembles each ``(m, m)`` Legendre block on the fly inside
    ``solvax.direct.block_thomas_truncated_fn`` (peak memory ``O(keep * m^2)``
    per subsystem, independent of ``n_xi``), so the ~39 GB full-band storage is
    never allocated.  The ``constraintScheme=2`` border is absorbed with the
    same exact rank-one trick as :func:`build_tier1_solver`.  The lowest
    ``keep`` blocks (and the source unknowns) are exact; blocks ``l >= keep``
    are zero-padded — valid because the drive and all requested output moments
    live on ``l < keep`` (see :func:`_rhs_confined_to_lowest_blocks`). With full
    recovery, ``keep`` grows with pitch resolution: per-chain memory then grows
    too, although global band arrays are still never materialized.

    Non-uniform ``Nxi_for_x`` (the production speed-dependent Legendre ramp)
    is exact too: each (species, x) subsystem is closed, so it is eliminated
    with its own ``n_blocks = Nxi_for_x[ix]`` — precisely the packed Fortran
    discretization (``indices.F90``), whose truncated DOFs the zero-padded
    tail already covers. Full recovery retains ``min(keep, n_blocks)`` per
    chain and pads only inactive DOFs; partial recovery still requires
    ``keep <= min Nxi_for_x``.

    ``subsystem_batch`` sets how many of the ``B = n_species * n_x``
    independent subsystems are eliminated concurrently (the
    ``jax.lax.map(..., batch_size=w)`` vmapped-chunk axis): width 1 is the
    fully serial minimum-memory sweep, width ``B`` eliminates every subsystem
    at once, and ``"auto"`` resolves backend-aware — width 1 on CPU, the
    widest memory-budget-fitting width on accelerators
    (:func:`_resolve_subsystem_batch`).  On the ramped path subsystems are
    grouped by equal ``n_blocks`` (all species at the speed nodes sharing one
    ``Nxi_for_x`` value) and batched within each group, so every subsystem
    keeps exactly its own static block count at any width — identical
    per-subsystem arithmetic to the serial sweep.

    Differentiability: the whole solve is a pure-JAX composition of
    ``block_thomas_truncated_fn`` sweeps, so ``jax.grad`` differentiates
    straight through it.  It is *not* wrapped in the full-operator
    implicit-function-theorem adjoint used by the full structured direct and
    recycled Krylov paths: this solve inverts the *reduced* Schur-complemented
    operator on the lowest ``keep`` blocks, not the full band, so a
    full-operator ``A^T`` adjoint would be inconsistent and silently corrupt
    gradients.  Two consistent
    reverse-mode paths exist instead.  The default tapes the generated sweeps
    (exact, but the tape grows with ``n_xi``: ``O(n_xi * m^2)`` per
    subsystem).  With ``adjoint_window=w`` the
    solve uses solvax's structure-preserving custom VJP for generated blocks:
    the right-hand-side gradient is an exactly *generated* truncated solve of
    the transposed operator, and the coefficient gradients are pulled back
    through the block assembly's own derivative on the leading ``keep + w``
    Legendre blocks — reverse mode then runs at ``O((keep + w) * m^2)`` per
    subsystem, independent of ``n_xi``, matching the forward sweep.  The
    window trades nothing on the right-hand-side gradient and has
    ``O(rho^{2w})`` coefficient-gradient error for the block-dominant
    collisional operators this kernel targets; ``w >= n_xi`` reproduces the
    taped gradient exactly (solvax pins full-window bitwise equality).
    """
    _require_solvax()
    n_s, n_x, n_xi, n_t, n_z = op.f_shape
    n_tz = n_t * n_z
    batch = n_s * n_x
    n_rhs = rhs2d.shape[1]
    cs = op.constraint_scheme

    t0 = time.perf_counter()
    coef = _truncated_coefficients(op)

    # Per-subsystem inputs, flattened to batch index b = s * n_x + x (matching
    # the (S, X) reshape used by KineticOperator.apply / build_tier1_solver).
    stream_b = jnp.repeat(coef["stream"], n_x, axis=0)  # (B, TZ, TZ)
    mirror_b = jnp.repeat(coef["mirror"], n_x, axis=0)  # (B, TZ)
    pas_b = coef["pas"].reshape(batch, n_xi)  # (B, L)
    x_b = jnp.tile(op.x, n_s)  # (B,)
    gamma_b = coef["gamma"].reshape(batch)  # (B,)
    b0, c0 = coef["b0"], coef["c0"]

    rhs_f = rhs2d[: op.f_size].reshape(n_s, n_x, n_xi, n_tz, n_rhs)
    rhs_low_b = rhs_f[:, :, :keep].reshape(batch, keep, n_tz, n_rhs)  # (B, keep, TZ, R)
    if cs == 2:
        r_c_b = rhs2d[op.f_size :].reshape(batch, n_rhs)  # (B, R)
    else:
        r_c_b = jnp.zeros((batch, n_rhs), dtype=jnp.float64)

    def solve_one(inputs, n_blocks: int):
        stream, mirror, pas_row, x_val, gamma, rhs_low, r_c = inputs
        local_keep = min(keep, n_blocks)
        rhs_low = rhs_low[:local_keep]

        def truncated_solve(rhs_cols: jnp.ndarray) -> jnp.ndarray:
            if adjoint_window is not None:
                # Structure-preserving bounded adjoint: the
                # differentiable coefficients travel as an explicit params
                # pytree, and reverse mode runs at O((keep + w) * m^2) per
                # subsystem instead of taping the full n_blocks sweep.
                params = _truncated_params(coef, stream, mirror, pas_row, x_val, gamma)
                block_fn_pk = functools.partial(
                    _truncated_blocks, n_xi=n_xi, shift_border=(cs == 2)
                )
                return block_thomas_truncated_fn(
                    block_fn_pk,
                    n_blocks,
                    rhs_cols,
                    local_keep,
                    params=params,
                    adjoint_window=adjoint_window,
                )
            block_fn = _truncated_block_fn(
                coef,
                n_xi,
                stream,
                mirror,
                pas_row,
                x_val,
                gamma,
                shift_border=(cs == 2),
            )
            return block_thomas_truncated_fn(block_fn, n_blocks, rhs_cols, local_keep)

        if cs == 2:
            z_col = jnp.zeros((local_keep, n_tz, 1), dtype=jnp.float64).at[0, :, 0].set(b0)
            rhs_stack = jnp.concatenate([rhs_low, z_col], axis=2)  # (keep, TZ, R+1)
            sol = truncated_solve(rhs_stack)
            y = sol[:, :, :n_rhs]  # (keep, TZ, R)
            z = sol[:, :, n_rhs]  # (keep, TZ)
            c_y0 = c0 @ y[0]  # (R,)
            c_z0 = c0 @ z[0]  # scalar
            shift = (c_y0 - r_c) / c_z0  # (R,)
            s = gamma * r_c + shift  # (R,)
            f_low = y - shift[None, None, :] * z[:, :, None]  # (keep, TZ, R)
        else:
            f_low = truncated_solve(rhs_low)
            s = jnp.zeros((n_rhs,), dtype=jnp.float64)
        return jnp.pad(f_low, ((0, keep - local_keep), (0, 0), (0, 0))), s

    # Concurrency across the B independent subsystems: lax.map with
    # batch_size=w eliminates w subsystems per vmapped chunk (the peak holds w
    # concurrent O(keep * m^2) sweeps); width 1 (batch_size=None) is the fully
    # serial scan-based sweep with a single subsystem's working set.
    width = _resolve_subsystem_batch(op, subsystem_batch, keep)

    def _map_subsystems(fn, inputs, n_sub: int):
        w = min(width, n_sub)
        return jax.lax.map(fn, inputs, batch_size=None if w == 1 else w)

    if _uniform_nxi_for_x(op):
        f_low_b, s_b = _map_subsystems(
            lambda t: solve_one(t, n_xi),
            (stream_b, mirror_b, pas_b, x_b, gamma_b, rhs_low_b, r_c_b),
            batch,
        )
    else:
        # Ramped Nxi_for_x: each (species, x) subsystem is closed, so it is
        # eliminated with its own static n_blocks = Nxi_for_x[ix] (the packed
        # Fortran discretization).  Subsystems are grouped by equal n_blocks —
        # all species at the speed nodes sharing one Nxi_for_x value — and
        # batched within each group, preserving the exact per-subsystem block
        # count at any width.
        rhs_low_sx = rhs_low_b.reshape(n_s, n_x, keep, n_tz, n_rhs)
        r_c_sx = r_c_b.reshape(n_s, n_x, n_rhs)
        groups: dict[int, list[int]] = {}
        for ix, nb in enumerate(int(v) for v in np.asarray(op.n_xi_for_x)):
            groups.setdefault(nb, []).append(ix)
        f_low_sx = jnp.zeros((n_s, n_x, keep, n_tz, n_rhs), dtype=jnp.float64)
        s_sx = jnp.zeros((n_s, n_x, n_rhs), dtype=jnp.float64)
        for nb, ixs in groups.items():
            g = len(ixs)
            idx = np.asarray(ixs)
            if g == 1:
                # Single-speed-node group (all ramp values distinct — the
                # production shape): slice views, no gather/repeat copies.
                ix = ixs[0]
                inputs = (
                    coef["stream"], coef["mirror"], coef["pas"][:, ix],
                    jnp.broadcast_to(op.x[ix], (n_s,)), coef["gamma"][:, ix],
                    rhs_low_sx[:, ix], r_c_sx[:, ix],
                )  # fmt: skip
            else:
                inputs = (
                    jnp.repeat(coef["stream"], g, axis=0),
                    jnp.repeat(coef["mirror"], g, axis=0),
                    coef["pas"][:, idx].reshape(n_s * g, n_xi),
                    jnp.tile(op.x[idx], n_s),
                    coef["gamma"][:, idx].reshape(n_s * g),
                    rhs_low_sx[:, idx].reshape(n_s * g, keep, n_tz, n_rhs),
                    r_c_sx[:, idx].reshape(n_s * g, n_rhs),
                )  # fmt: skip
            f_g, s_g = _map_subsystems(
                lambda t, nb=nb: solve_one(t, nb), inputs, n_s * g
            )
            f_low_sx = f_low_sx.at[:, idx].set(f_g.reshape(n_s, g, keep, n_tz, n_rhs))
            s_sx = s_sx.at[:, idx].set(s_g.reshape(n_s, g, n_rhs))
        f_low_b = f_low_sx.reshape(batch, keep, n_tz, n_rhs)
        s_b = s_sx.reshape(batch, n_rhs)
    # Force the async truncated block-Thomas sweep to complete so the timing is
    # real compute, not JAX dispatch latency (a no-op under jit/grad tracing).
    f_low_b, s_b = jax.block_until_ready((f_low_b, s_b))
    t1 = time.perf_counter()

    f_full = jnp.zeros((n_s, n_x, n_xi, n_tz, n_rhs), dtype=jnp.float64)
    f_full = f_full.at[:, :, :keep].set(f_low_b.reshape(n_s, n_x, keep, n_tz, n_rhs))
    parts = [f_full.reshape(op.f_size, n_rhs)]
    if op.extra_size:
        parts.append(s_b.reshape(op.extra_size, n_rhs))
    x2d = jnp.concatenate(parts, axis=0)

    if keep == n_xi:
        res = _residual_norms(op.apply, x2d, rhs2d)
    else:
        res = _truncated_partial_residual(
            op, coef, stream_b, mirror_b, pas_b, x_b, f_low_b, s_b, rhs_low_b, r_c_b, keep
        )
    x2d, res = jax.block_until_ready(
        (x2d, res)
    )  # real residual/assembly time, not dispatch
    t2 = time.perf_counter()
    return SolveResult(
        x=x2d,
        method="block_tridiagonal_truncated",
        iterations=None,
        residual_norms=res,
        converged=_converged_flag(res, rhs2d, tol, atol),
        recycle=None,
        timings={"build": t1 - t0, "solve": t2 - t1},
    )


def tier1_truncated_tail_blocks(
    op: KineticOperator,
    rhs: jnp.ndarray,
    solution: jnp.ndarray,
    *,
    keep_lowest: int = _TIER1_KEEP_LOWEST_DEFAULT,
    keep_highest: int = 2,
) -> jnp.ndarray:
    """Recover exact final Legendre blocks after a truncated structured solve.

    The production truncated solve retains exact low transport modes and the
    ``constraintScheme=2`` source unknowns.  This opt-in diagnostic performs a
    second, forward Schur sweep through the same generated blocks and retains
    only the final ``keep_highest`` modes.  It never materializes full bands or
    a full modal state; dense workspace is independent of ``Nxi``.

    Returns ``(species, speed, keep_highest, theta*zeta, rhs)``.  The tail is
    ordered by increasing Legendre index, and each speed subsystem uses its own
    active ``Nxi_for_x`` count.
    """
    _require_solvax()
    if int(keep_highest) < 1:
        raise ValueError("keep_highest must be at least 1")
    ok, reason = _truncation_supported(op, int(keep_lowest))
    if not ok:
        raise NotImplementedError(
            f"structured direct selected-tail reconstruction unavailable: {reason}"
        )
    if int(np.min(np.asarray(op.n_xi_for_x))) < int(keep_highest):
        raise ValueError("keep_highest exceeds the minimum active Nxi_for_x")

    rhs2d, _ = _as_columns(rhs)
    solution2d, _ = _as_columns(solution)
    if rhs2d.shape != solution2d.shape or rhs2d.shape[0] != op.total_size:
        raise ValueError(
            "rhs and solution must have matching state-vector shapes, got "
            f"{rhs2d.shape} and {solution2d.shape}"
        )

    n_s, n_x, n_xi, n_t, n_z = op.f_shape
    n_tz = n_t * n_z
    n_rhs = rhs2d.shape[1]
    batch = n_s * n_x
    coef = _truncated_coefficients(op)
    rhs_f = rhs2d[: op.f_size].reshape(n_s, n_x, n_xi, n_tz, n_rhs)
    rhs_low_b = rhs_f[:, :, :keep_lowest].reshape(
        batch, keep_lowest, n_tz, n_rhs
    )
    solution_f = solution2d[: op.f_size].reshape(
        n_s, n_x, n_xi, n_tz, n_rhs
    )
    solution_low_b = solution_f[:, :, :keep_lowest].reshape(
        batch, keep_lowest, n_tz, n_rhs
    )
    if op.constraint_scheme == 2:
        r_c_b = rhs2d[op.f_size :].reshape(batch, n_rhs)
        source_b = solution2d[op.f_size :].reshape(batch, n_rhs)
    else:
        r_c_b = jnp.zeros((batch, n_rhs), dtype=jnp.float64)
        source_b = jnp.zeros((batch, n_rhs), dtype=jnp.float64)

    def solve_one(inputs, n_blocks: int):
        stream, mirror, pas_row, x_val, gamma, rhs_low, solution_low, r_c, source = inputs
        block_fn = _truncated_block_fn(
            coef,
            n_xi,
            stream,
            mirror,
            pas_row,
            x_val,
            gamma,
            shift_border=(op.constraint_scheme == 2),
        )
        if op.constraint_scheme == 2:
            shift = source - gamma * r_c
            rhs_low = rhs_low.at[0].add(-coef["b0"][:, None] * shift[None, :])
        return block_thomas_selected_tail_fn(
            block_fn,
            n_blocks,
            rhs_low,
            int(keep_highest),
            solution_low=solution_low,
        )

    groups: dict[int, list[int]] = {}
    for ix, count in enumerate(int(v) for v in np.asarray(op.n_xi_for_x)):
        groups.setdefault(count, []).append(ix)
    tails = jnp.zeros(
        (n_s, n_x, keep_highest, n_tz, n_rhs), dtype=jnp.float64
    )
    rhs_low_sx = rhs_low_b.reshape(n_s, n_x, keep_lowest, n_tz, n_rhs)
    solution_low_sx = solution_low_b.reshape(
        n_s, n_x, keep_lowest, n_tz, n_rhs
    )
    r_c_sx = r_c_b.reshape(n_s, n_x, n_rhs)
    source_sx = source_b.reshape(n_s, n_x, n_rhs)
    for n_blocks, ixs in groups.items():
        count = len(ixs)
        idx = np.asarray(ixs)
        inputs = (
            jnp.repeat(coef["stream"], count, axis=0),
            jnp.repeat(coef["mirror"], count, axis=0),
            coef["pas"][:, idx].reshape(n_s * count, n_xi),
            jnp.tile(op.x[idx], n_s),
            coef["gamma"][:, idx].reshape(n_s * count),
            rhs_low_sx[:, idx].reshape(n_s * count, keep_lowest, n_tz, n_rhs),
            solution_low_sx[:, idx].reshape(
                n_s * count, keep_lowest, n_tz, n_rhs
            ),
            r_c_sx[:, idx].reshape(n_s * count, n_rhs),
            source_sx[:, idx].reshape(n_s * count, n_rhs),
        )
        recovered = jax.lax.map(
            lambda values, n_blocks=n_blocks: solve_one(values, n_blocks), inputs
        )
        tails = tails.at[:, idx].set(
            recovered.reshape(n_s, count, keep_highest, n_tz, n_rhs)
        )
    return tails


def _truncated_partial_residual(
    op: KineticOperator,
    coef: dict[str, jnp.ndarray],
    stream_b: jnp.ndarray,
    mirror_b: jnp.ndarray,
    pas_b: jnp.ndarray,
    x_b: jnp.ndarray,
    f_low_b: jnp.ndarray,
    s_b: jnp.ndarray,
    rhs_low_b: jnp.ndarray,
    r_c_b: jnp.ndarray,
    keep: int,
) -> jnp.ndarray:
    """Residual over the rows fully determined by the computed lowest-K blocks.

    Legendre row ``l`` couples to columns ``l-1, l, l+1``, so rows
    ``l = 0 .. keep-2`` (plus the ``constraintScheme=2`` FSA rows) are entirely
    fixed by the ``keep`` computed blocks and must vanish to machine precision;
    row ``keep-1`` couples to the (deliberately unsolved) block ``keep`` and is
    excluded.  This mirrors ``TruncatedTier1.partial_residual`` and is the
    honest convergence signal for the truncated solve.  Returns per-column
    norms of shape ``(n_rhs,)``.
    """
    cs = op.constraint_scheme
    n_rhs = rhs_low_b.shape[-1]
    b0, c0 = coef["b0"], coef["c0"]

    def per_subsystem(inputs):
        stream, mirror, pas_row, x_val, f_low, s, rhs_low, r_c = inputs
        raw = _truncated_block_fn(
            coef, op.n_xi, stream, mirror, pas_row, x_val, 0.0, shift_border=False
        )
        acc = jnp.zeros((n_rhs,), dtype=jnp.float64)
        for ell in range(keep - 1):
            lo, di, up = raw(jnp.asarray(ell, dtype=jnp.int32))
            r = jnp.einsum("ij,jr->ir", di, f_low[ell]) - rhs_low[ell]
            if ell > 0:
                r = r + jnp.einsum("ij,jr->ir", lo, f_low[ell - 1])
            r = r + jnp.einsum("ij,jr->ir", up, f_low[ell + 1])
            if ell == 0 and cs == 2:
                r = r + b0[:, None] * s[None, :]
            acc = acc + jnp.sum(r * r, axis=0)
        if cs == 2:
            rc = (c0 @ f_low[0]) - r_c  # (R,)
            acc = acc + rc * rc
        return acc

    sq = jax.lax.map(
        per_subsystem, (stream_b, mirror_b, pas_b, x_b, f_low_b, s_b, rhs_low_b, r_c_b)
    )  # (B, R)
    return jnp.sqrt(jnp.sum(sq, axis=0))


def _resolve_preconditioner(
    preconditioner: str | None, use_preconditioner: bool
) -> str:
    """Normalize the recycled Krylov preconditioner request.

    ``preconditioner=None`` keeps the historical ``use_preconditioner`` boolean
    (``True`` -> the coarse block-Thomas preconditioner, ``False`` -> none), so
    every existing caller is byte-for-byte unaffected.
    """
    if preconditioner is None:
        return "coarse" if use_preconditioner else "none"
    name = str(preconditioner).strip().lower()
    if name not in _TIER2_PRECONDITIONERS:
        raise ValueError(
            f"unknown preconditioner {preconditioner!r}; expected one of "
            f"{sorted(_TIER2_PRECONDITIONERS)}"
        )
    return name


def build_tier2_preconditioner(
    op: KineticOperator, kind: str, *, drop_l_coupling: bool = False
) -> tuple[Callable[[jnp.ndarray], jnp.ndarray], Callable[[jnp.ndarray], jnp.ndarray]]:
    """``(precond, precond_t)`` for the requested Krylov preconditioner.

    ``"coarse"`` is :func:`build_coarse_preconditioner` (the exact block-Thomas
    factorization of the SFINCS-simplified operator); ``"multigrid"`` is
    :func:`dkx.multigrid.build_multigrid_preconditioner`, which approximates
    the inverse of that *same* operator with a semicoarsened multigrid V-cycle
    and so is affordable where the cubic-in-``Ntheta*Nzeta`` factorization is
    not; ``"sparse"`` is :func:`dkx.sparse_precond.build_sparse_preconditioner`,
    which keeps the inverse exact but eliminates in a fill-reducing order on the
    host instead of eliminating ``L`` first, so the angular stencils stay sparse.
    All three eliminate the bordered constraint / ``Phi1`` rows identically, and
    none changes the solution: they change how fast the Krylov route gets
    there.
    """
    if kind == "coarse":
        return build_coarse_preconditioner(op, drop_l_coupling=drop_l_coupling)
    if kind == "sparse":
        from dkx.sparse_precond import build_sparse_preconditioner  # noqa: PLC0415

        return build_sparse_preconditioner(op, drop_l_coupling=drop_l_coupling)
    from dkx.multigrid import build_multigrid_preconditioner  # noqa: PLC0415

    return build_multigrid_preconditioner(op, drop_l_coupling=drop_l_coupling)


def _solve_tier2(
    op: KineticOperator,
    rhs2d: jnp.ndarray,
    *,
    tol: float,
    atol: float,
    x0: jnp.ndarray | None,
    recycle: tuple[jnp.ndarray, jnp.ndarray] | None,
    preconditioner: str,
    drop_l_coupling_in_precond: bool,
    restart: int,
    recycle_dim: int,
    max_restarts: int,
    differentiable: bool,
    check_adjoint: bool,
    adjoint_residual_factor: float = DEFAULT_ADJOINT_RESIDUAL_FACTOR,
    prebuilt_precond: tuple[Callable, Callable] | None = None,
) -> SolveResult:
    if int(recycle_dim) < 1:
        # solvax's GCROT scatters the recycled subspace into a (n, k) array, so
        # k = 0 dies inside jax indexing with "index is out of bounds for axis 1
        # with size 0" -- a traceback naming neither the parameter nor DKX.
        # Plain restarted FGMRES is not reachable through this route; refusing
        # by name is better than a crash that looks like a library bug.
        raise ValueError(
            f"recycle_dim={recycle_dim}: the recycled Krylov route needs at least "
            "one recycled direction. Use recycle_dim >= 1, or method='direct' for "
            "a non-recycling solve."
        )
    traced = _is_traced(rhs2d, *jax.tree_util.tree_leaves(op))
    t0 = time.perf_counter()
    precond = precond_t = None
    # The escalation path deliberately does NOT pass a prebuilt pair: it is
    # searching over preconditioner kinds, so pinning one would defeat the
    # search it exists to perform.
    if prebuilt_precond is not None:
        # A caller that solves the same operator family repeatedly -- an Er
        # scan, a Newton loop, a batched sweep -- can build once and hand the
        # pair in. The build is a large fraction of a single call (39% on the
        # Sugama collisionOperator=3 deck), so paying it per point is the
        # single largest avoidable cost in a scan.
        precond, precond_t = prebuilt_precond
    elif preconditioner != "none":
        precond, precond_t = build_tier2_preconditioner(
            op, preconditioner, drop_l_coupling=drop_l_coupling_in_precond
        )
        # The preconditioner closure captures the async coarse block-Thomas
        # factorization; force it to complete (a zero probe) so the "build"
        # timing is real compute, not JAX dispatch latency.  Skipped under
        # jit/grad tracing, where block_until_ready is a no-op on tracers and
        # the probe would only add dead nodes to the trace.
        if not traced:
            jax.block_until_ready(
                precond(jnp.zeros((op.total_size,), dtype=jnp.float64))
            )
    t1 = time.perf_counter()

    x0_2d = None
    if x0 is not None:
        x0_2d, _ = _as_columns(x0)
        if x0_2d.shape != rhs2d.shape:
            raise ValueError(
                f"x0 shape {x0_2d.shape} must match rhs shape {rhs2d.shape}"
            )

    # Pinned matvecs: identical to op.apply on the physical subspace, identity
    # on the Nxi_for_x-truncated DOFs, so the system (and in particular its
    # transpose, used by the differentiable adjoint) is nonsingular.
    matvec, matvec_t = _pinned_matvecs(op)
    diagnostics = (
        AdjointDiagnostics(
            tol=float(tol),
            atol=float(atol),
            factor=float(adjoint_residual_factor),
            checked=bool(check_adjoint),
        )
        if differentiable
        else None
    )
    cols: list[jnp.ndarray] = []
    total_iters: int | None = 0
    converged = True
    res_norms: list[jnp.ndarray] = []
    for j in range(rhs2d.shape[1]):
        b = rhs2d[:, j]
        if differentiable:
            # One forward solve, not two.  ``custom_linear_solve`` *defines*
            # the primal as ``solve(matvec, b)``, so the wrapper runs a GCROT
            # of its own no matter what; this used to sit on top of a plain
            # GCROT whose solution was then thrown away, and every
            # differentiable Krylov solve paid for both.  ``has_aux`` carries
            # the recycle pair, iteration count, convergence flag and residual
            # out of the wrapper's own solve instead, so there is nothing left
            # to recompute outside it.
            #
            # The wrapped solve takes no ``x0``: ``custom_linear_solve``
            # applies the same solver to *tangent* right-hand sides, for which
            # the primal's warm start is a bad initial guess.  The recycle pair
            # is right-hand-side independent -- a deflation subspace, with
            # ``A U`` recomputed and re-orthonormalized against the current
            # operator on entry -- so it is still threaded across columns.
            #
            # Both the forward and the adjoint (transposed) solves have their
            # *true* residual recomputed from the operator and recorded in
            # ``diagnostics``; with check_adjoint on, a miss aborts loudly
            # instead of silently corrupting the gradient
            # (jax.debug.callback fires at execution time, i.e. during the
            # backward pass for the adjoint).
            def _measured(
                label: str,
                mv: Callable[[jnp.ndarray], jnp.ndarray],
                pc: Callable[[jnp.ndarray], jnp.ndarray] | None,
                rhs_col: jnp.ndarray,
                warm: tuple[jnp.ndarray, jnp.ndarray] | None,
                index: int = j,
            ):
                s = gcrot(
                    mv,
                    rhs_col,
                    precond=pc,
                    m=restart,
                    k=recycle_dim,
                    rtol=tol,
                    atol=atol,
                    max_restarts=max_restarts,
                    recycle=warm,
                )
                measured = _guarded_solve(
                    label,
                    index,
                    mv,
                    rhs_col,
                    s.x,
                    tol=tol,
                    atol=atol,
                    factor=adjoint_residual_factor,
                    raise_on_failure=check_adjoint,
                    diagnostics=diagnostics,
                )
                return measured, (
                    s.recycle,
                    s.iterations,
                    s.converged,
                    s.residual_norm,
                )

            def fwd_solve(rhs_col: jnp.ndarray, warm=recycle):
                return _measured("forward", matvec, precond, rhs_col, warm)

            def t_solve(rhs_col: jnp.ndarray):
                # The adjoint keeps its own cold start: the forward pair is
                # built inside the wrapper's trace and cannot reach here.
                return _measured(
                    "adjoint (transposed)", matvec_t, precond_t, rhs_col, None
                )

            x_col, aux = _implicit_solve(
                matvec, matvec_t, b, fwd_solve, t_solve, has_aux=True
            )
            recycle, iterations, col_converged, residual_norm = aux
        else:
            sol = gcrot(
                matvec,
                b,
                x0=None if x0_2d is None else x0_2d[:, j],
                precond=precond,
                m=restart,
                k=recycle_dim,
                rtol=tol,
                atol=atol,
                max_restarts=max_restarts,
                recycle=recycle,
            )
            x_col = sol.x
            recycle, iterations = sol.recycle, sol.iterations
            col_converged, residual_norm = sol.converged, sol.residual_norm
        if traced:
            total_iters = None  # iteration counts are tracers under jit/grad
        else:
            total_iters += int(iterations)
            converged = converged and bool(col_converged)
        res_norms.append(residual_norm)
        cols.append(x_col)
    x_stacked = jax.block_until_ready(
        jnp.stack(cols, axis=1)
    )  # real solve time, not dispatch
    t2 = time.perf_counter()
    return SolveResult(
        x=x_stacked,
        method="gcrot",
        iterations=total_iters,
        residual_norms=jnp.stack(res_norms),
        converged=converged,
        recycle=recycle,
        timings={"build": t1 - t0, "solve": t2 - t1},
        adjoint=diagnostics,
        # Only the recycled-Krylov route builds one, so a caller that threads
        # this forward gets a preconditioner exactly when there is one to reuse
        # and None when the route was direct.
        precond=None if precond is None else (precond, precond_t),
    )


# =============================================================================
# The auto-policy entry point
# =============================================================================


def _auto_route_structural(
    op: KineticOperator,
    budget_gb: float | None = None,
    keep_lowest: int = _TIER1_KEEP_LOWEST_DEFAULT,
) -> str:
    """The ``method="auto"`` structured route decided from operator structure.

    The RHS-free twin of :func:`_auto_route`, for callers that must predict the
    route without a right-hand side (the memory model behind
    :func:`auto_solve_peak_memory_bytes` and the batched-scan chunk sizing in
    :mod:`dkx.batch`).  The one RHS-dependent check —
    :func:`_rhs_confined_to_lowest_blocks` — is replaced by the structural
    RHSMode 1/2/3 guarantee (drives and output moments on ``l <= 2``), exactly
    the fallback :func:`_auto_route` itself uses when the RHS is traced.  Any
    change to the routing conditions must be applied to both functions.

    Returns ``"block_tridiagonal"``, ``"block_tridiagonal_truncated"``, or
    ``"gmres"``.
    """
    ok, _reason = tier1_available(op)
    if not ok:
        return "gmres"
    budget_bytes, _ = _tier1_budget_bytes(budget_gb)
    if _uniform_nxi_for_x(op) and tier1_peak_memory_bytes(op) <= budget_bytes:
        return "block_tridiagonal"
    keep = min(keep_lowest, op.n_xi)
    sup_ok, _sup_reason = _truncation_supported(op, keep)
    if sup_ok and int(op.rhs_mode) in (1, 2, 3):
        return "block_tridiagonal_truncated"
    return "gmres"


def auto_solve_peak_memory_bytes(
    op: KineticOperator,
    budget_gb: float | None = None,
    keep_lowest: int = _TIER1_KEEP_LOWEST_DEFAULT,
) -> float:
    """Peak-memory estimate of the solve ``method="auto"`` would run on ``op``.

    Follows the auto-router's own decision (:func:`_auto_route_structural`) so
    the estimate models the kernel that actually executes: a solve that routes
    to the truncated block-Thomas kernel is charged its truncated working set
    (:func:`tier1_truncated_peak_memory_bytes`) — the full Legendre bands are
    never allocated on that route, and charging their factorization peak
    overstates a ramped or budget-forced production solve by ~46x.  The
    full-band route keeps the factorization peak
    (:func:`tier1_peak_memory_bytes`); the recycled Krylov GCROT fallback keeps
    it too, as a deliberately conservative stand-in (its matvec working set is
    smaller, but it has no validated model of its own).
    """
    route = _auto_route_structural(op, budget_gb, keep_lowest)
    if route == "block_tridiagonal_truncated":
        return tier1_truncated_peak_memory_bytes(op, keep_lowest=keep_lowest)
    return tier1_peak_memory_bytes(op)


def _auto_route(
    op: KineticOperator,
    rhs2d: jnp.ndarray,
    budget_gb: float | None,
    keep_lowest: int,
    subsystem_batch: int | str = "auto",
    emit: Callable[[str], None] | None = print,
) -> str:
    """Pick the route for ``method="auto"`` and print a Fortran-style one-liner.

    Structural changes to the routing conditions here must be mirrored in
    :func:`_auto_route_structural` (the RHS-free twin used by the memory
    model).
    """
    ok, _reason = tier1_available(op)
    if not ok:
        return "gmres"

    peak = tier1_peak_memory_bytes(op)
    bands = tier1_full_band_bytes(op)
    budget_bytes, budget_gb_val = _tier1_budget_bytes(budget_gb)
    peak_gb = peak / 2.0**30
    uniform = _uniform_nxi_for_x(op)
    if uniform and peak <= budget_bytes:
        if emit is not None:
            emit(
                f"[dkx.solve] structured direct route: full factorization; "
                f"peak estimate {peak_gb:.2f} GB <= budget {budget_gb_val:.1f} GB "
                f"(bands {bands / 2.0**30:.2f} GB x2.5)."
            )
        return "block_tridiagonal"

    keep = min(keep_lowest, op.n_xi)
    sup_ok, sup_reason = _truncation_supported(op, keep)
    rhs_ok = _rhs_confined_to_lowest_blocks(op, rhs2d, keep)
    # Under trace the RHS support is unreadable; trust the structural RHSMode
    # 1/2/3 guarantee (drives + moments on l <= 2).
    rhs_valid = rhs_ok if rhs_ok is not None else (int(op.rhs_mode) in (1, 2, 3))
    if sup_ok and rhs_valid:
        because = (
            "non-uniform Nxi_for_x (per-subsystem n_blocks = Nxi_for_x[ix]; "
            "the full bands do not support the ramp)"
            if not uniform
            else f"peak estimate {peak_gb:.2f} GB > budget {budget_gb_val:.1f} GB "
            f"(bands {bands / 2.0**30:.2f} GB x2.5)"
        )
        width = _resolve_subsystem_batch(op, subsystem_batch, keep)
        n_sub = max(1, int(op.n_species) * int(op.n_x))
        trunc_gb = (
            tier1_truncated_peak_memory_bytes(op, keep, subsystem_batch=width) / 2.0**30
        )
        if emit is not None:
            emit(
                f"[dkx.solve] memory-bounded structured direct route: "
                f"truncated block-Thomas (keep_lowest={keep}); {because}, "
                f"solving the lowest {keep} Legendre blocks "
                f"with subsystem batch width {width} of {n_sub} "
                f"(working-set estimate {trunc_gb:.2f} GB)."
            )
        return "block_tridiagonal_truncated"

    why = sup_reason if not sup_ok else "RHS/output needs Legendre modes l >= keep"
    blocker = (
        "non-uniform Nxi_for_x rules out the full bands"
        if not uniform
        else f"full-band estimate {peak_gb:.2f} GB > budget {budget_gb_val:.1f} GB"
    )
    if emit is not None:
        emit(
            f"[dkx.solve] structured direct route unavailable: {blocker} and "
            f"truncation is invalid ({why}); using the recycled Krylov solve."
        )
    return "gmres"


def _env_size(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(float(raw))
    except ValueError:
        return default


def _cpu_device_or_none() -> "jax.Device | None":
    """The first host-CPU device, or ``None`` when the CPU backend is absent
    (e.g. ``JAX_PLATFORMS=cuda`` initializes only the CUDA platform)."""
    try:
        return jax.local_devices(backend="cpu")[0]
    except RuntimeError:
        return None


def _single_device_of(arr: jnp.ndarray) -> "jax.Device | None":
    """The unique device holding ``arr``, or ``None`` (sharded/unknown)."""
    try:
        devices = arr.devices()
    except Exception:
        return None
    if len(devices) != 1:
        return None
    return next(iter(devices))


def _resolve_solve_device(
    device: "str | jax.Device | None",
    chosen: str,
    op: KineticOperator,
    traced: bool,
) -> "jax.Device | None":
    """Map the ``solve(device=...)`` knob to a target device (or ``None``).

    ``None`` means "stay put" (no array movement).  Under jit/grad tracing the
    knob is inert: arrays cannot be moved mid-trace, and the enclosing ``jit``
    already pinned the computation's devices.
    """
    if traced or chosen == "direct":  # sparse direct is a host solve already
        return None
    if device is None:
        device = os.environ.get(_SOLVE_DEVICE_ENV, "").strip().lower() or "auto"
    if isinstance(device, jax.Device):
        return device
    device = str(device).strip().lower()
    if device == "default":
        return None
    backend = jax.default_backend()
    if device == "cpu":
        if backend == "cpu":
            return None
        cpu = _cpu_device_or_none()
        if cpu is None:
            raise ValueError(
                "solve(device='cpu') requested but no CPU backend is available "
                "(JAX_PLATFORMS excludes 'cpu'?)."
            )
        return cpu
    if device in ("gpu", "cuda", "tpu", "accelerator"):
        if backend == "cpu":
            raise ValueError(
                f"solve(device={device!r}) requested but the default JAX backend is CPU "
                "(no accelerator available)."
            )
        return None
    if device != "auto":
        raise ValueError(
            f"unknown solve device {device!r}; expected 'auto', 'default', 'cpu', "
            "'gpu', or a jax.Device"
        )
    if backend == "cpu":
        return None
    if chosen in ("block_tridiagonal", "block_tridiagonal_truncated"):
        max_size = _env_size(_SOLVE_CPU_MAX_TIER1_ENV, _SOLVE_CPU_MAX_TIER1_DEFAULT)
    else:
        max_size = _env_size(_SOLVE_CPU_MAX_TIER2_ENV, _SOLVE_CPU_MAX_TIER2_DEFAULT)
    if int(op.total_size) <= max_size:
        cpu = _cpu_device_or_none()
        if cpu is None:  # e.g. JAX_PLATFORMS=cuda: no CPU backend to route to
            return None
        print(
            f"[dkx.solve] device route: total_size={int(op.total_size)} <= "
            f"{max_size} — running this "
            f"{'structured direct' if chosen.startswith('block') else 'recycled Krylov'} "
            f"solve on the host CPU (small solves are dispatch-bound on {backend}; "
            f"override with device='default' or {_SOLVE_DEVICE_ENV})."
        )
        return cpu
    return None


def solve(
    op: KineticOperator,
    rhs: jnp.ndarray,
    *,
    method: str = "auto",
    tol: float = 1e-10,
    atol: float = 0.0,
    x0: jnp.ndarray | None = None,
    recycle: tuple[jnp.ndarray, jnp.ndarray] | None = None,
    differentiable: bool = False,
    check_adjoint: bool = True,
    adjoint_residual_factor: float = DEFAULT_ADJOINT_RESIDUAL_FACTOR,
    use_preconditioner: bool = True,
    preconditioner: str | None = None,
    drop_l_coupling_in_precond: bool = False,
    restart: int = 30,
    recycle_dim: int = 8,
    max_restarts: int = 200,
    max_dense_size: int = 8192,
    tier1_memory_budget_gb: float | None = None,
    tier1_keep_lowest: int = _TIER1_KEEP_LOWEST_DEFAULT,
    subsystem_batch: int | str = "auto",
    tier1_adjoint_window: int | None = None,
    device: str | jax.Device | None = None,
    emit: Callable[[str], None] | None = print,
    precond: tuple[Callable, Callable] | None = None,
) -> SolveResult:
    """Solve ``K x = rhs`` with the plan-§2.3 three-route auto-policy.

    Policy (``method="auto"``):

    1. **structured direct** (``"block_tridiagonal"``) when
       :func:`tier1_available` — PAS/DKES family, exact direct solve,
       multi-RHS in one elimination;
    2. **recycled Krylov** (``"gmres"``) otherwise — GCROT-recycled FGMRES on
       the matrix-free operator, right-preconditioned by an exact structured
       direct solve of the Fortran-style simplified coarse operator;
    3. **sparse direct** (``"direct"``) on explicit request, or automatically
       when recycled Krylov breaches its iteration cap — host SuperLU on the
       materialized matrix, non-differentiable, loud.

    Args:
        op: the kinetic operator (:class:`dkx.drift_kinetic.KineticOperator`).
        rhs: right-hand side(s), ``(total_size,)`` or ``(total_size, n_rhs)``
            — e.g. columns of :meth:`KineticOperator.rhs` for RHSMode 2/3.
        method: ``"auto"`` | ``"block_tridiagonal"`` | ``"gmres"`` |
            ``"direct"``.  Explicit route requests raise if unsupported.
        tol: relative residual tolerance (on ``||rhs||``, per column).
        atol: absolute residual floor.
        x0: warm-start solution (recycled Krylov), same shape as ``rhs``.
        recycle: GCROT recycle pair from a previous :class:`SolveResult`
            (recycled Krylov continuation warm start).
        differentiable: wrap the solution in
            ``solvax.implicit.linear_solve`` so ``jax.grad`` flows through
            (structured direct and recycled Krylov; sparse direct refuses).
            The forward solve is the same one either way; a *gradient* costs
            one extra (transposed) solve.
        check_adjoint: (differentiable recycled Krylov only, default on) abort
            loudly — a ``RuntimeError`` raised from a ``jax.debug.callback`` at
            execution time — when the forward or the adjoint (transposed)
            GCROT solve misses its residual.  A stalled *adjoint* solve is
            invisible from the outside: the forward solution and every field
            of the returned :class:`SolveResult` are fine and only the
            implicit-function-theorem gradient is wrong, which is how the
            singular Fokker-Planck + ``constraintScheme=1`` system corrupts
            an optimization.  Set ``False`` for the explicitly unchecked
            path; the residuals are still recorded in
            :attr:`SolveResult.adjoint`, so a caller that opts out can
            inspect them and decide.  The check recomputes the true residual
            from the operator (``||A^T y - g||``, one extra apply plus two
            norm probes per solve) rather than trusting the Krylov method's
            internal estimate, and it never touches the solution — forward
            answers are identical either way.
        adjoint_residual_factor: how far over its residual a differentiable
            solve may land before ``check_adjoint`` calls it a failure
            (default 10).  The accepted residual is ``factor *
            max(atol, tol*||b||, floor)``, where ``floor = 32 * eps *
            (||A|| ||x|| + ||b||)`` is the double-precision backward-error
            floor of that solve.  The floor is what keeps healthy cases
            quiet: on a near-singular operator the adjoint solution norm
            explodes (the cotangent has a component along an almost-null
            direction), and no backward-stable method can then drive
            ``||A^T y - g||`` down to ``tol * ||g||`` — the gradient is fine,
            the requested residual is simply unreachable.  The floor is
            capped at ``1e-6 * ||b||`` so a solve that diverges outright
            cannot excuse itself with its own inflated solution norm; a
            genuinely inconsistent adjoint misses by many orders of
            magnitude and is caught regardless.

    Operators with a truncated Legendre resolution (non-uniform ``Nxi_for_x``)
    are structurally singular in the rectangular state layout: the truncated
    DOFs are exact zero rows of :meth:`KineticOperator.apply` (Fortran v3
    never carries them — packed indexing in ``indices.F90``).  The recycled
    Krylov and sparse direct routes therefore solve the *pinned* system
    ``(A M + I - M) x = rhs`` with ``M`` the active-DOF projector
    (:meth:`KineticOperator.active_dof_mask`): it is nonsingular, agrees with
    ``A`` on the physical subspace, and forces
    ``x = rhs = 0`` on the truncated DOFs, so solutions, residuals, and
    implicit-function-theorem gradients all match the packed Fortran system.
    The truncated structured direct kernel is consistent with the same
    pinning: it eliminates each closed (species, x) subsystem with its own
    ``n_blocks = Nxi_for_x[ix]`` (exactly the packed Fortran system) and
    zero-pads everything above, so ramped PAS/DKES decks route through it;
    only the full structured direct factorization requires uniform
    ``Nxi_for_x``.
        use_preconditioner: recycled-Krylov preconditioner on/off (legacy
            boolean; a non-``None`` ``preconditioner`` overrides it).
        preconditioner: which recycled-Krylov preconditioner to build —

            ``"coarse"``
                the classical one (:func:`build_coarse_preconditioner`): an
                exact batched block-Thomas factorization of the
                SFINCS-simplified operator, with dense ``Ntheta*Nzeta``
                blocks.  ``O(Nxi Nspecies Nx (Ntheta Nzeta)**3)`` time and
                ``O(Nxi Nspecies Nx (Ntheta Nzeta)**2)`` memory *per solve*.
            ``"multigrid"``
                the same simplified operator, inverted approximately by the
                semicoarsened geometric multigrid V-cycle of
                :mod:`dkx.multigrid`: linear in the grid size and therefore
                affordable at angular resolutions where the cubic
                factorization is not.  It buys that affordability at the cost
                of preconditioner quality — on the measured NCSX
                full-Fokker-Planck ladder it does *not* reach the Krylov
                tolerance, and ``docs/performance.rst`` records both the table
                and the diagnosis — so it stays opt-in.
            ``"sparse"``
                the same simplified operator, inverted *exactly* but in a
                fill-reducing elimination order
                (:mod:`dkx.sparse_precond`): the angular blocks carry the 3- or
                5-point ``createGrids.F90`` stencils, and eliminating ``L``
                first is what fills them in, so ``dkx`` assembles the operator
                in CSR and factors it with host SuperLU — which is what the
                Fortran reference does with MUMPS.  Same map as ``"coarse"`` to
                factorization round-off; it cannot run with traced operator
                leaves, since the assembly reads values on the host.
            ``"none"``
                unpreconditioned GCROT.

            ``None`` (the default) keeps the historical behaviour selected by
            ``use_preconditioner``, so nothing changes silently.  A
            preconditioner cannot change the answer — only the iteration
            count and the wall time.
        drop_l_coupling_in_precond: sever the L±1 coupling in the coarse
            operator.  Not Fortran's ``preconditioner_xi``, which drops L±2,
            and expensive; see :func:`dkx.coarse_precond.build_coarse_preconditioner`.
        restart: FGMRES cycle size ``m``.
        recycle_dim: GCROT recycle directions ``k``.
        max_restarts: recycled-Krylov outer-cycle cap (what makes ``auto``
            fall through to the sparse direct route).
        max_dense_size: sparse direct materialization guard.
        tier1_memory_budget_gb: budget (GB) above which ``method="auto"``
            prefers the memory-lean truncated structured direct kernel over
            the full-band factorization.  ``None`` reads the ``DKX_TIER1_MEMORY_BUDGET_GB``
            environment variable, else the 8 GB default.  The full-band peak is
            estimated by :func:`tier1_peak_memory_bytes`.
        tier1_keep_lowest: number of Legendre blocks the truncated structured
            direct kernel computes exactly (default 3 — the RHSMode 1/2/3 drives and
            output moments live on ``l <= 2``).
        subsystem_batch: how many of the ``B = n_species * n_x`` independent
            (species, x) subsystems the truncated structured direct kernel
            eliminates concurrently.  ``"auto"`` (default) is backend-aware: width 1 on
            the CPU backend (XLA:CPU runs batched LAPACK factor/solve calls
            serially per batch element, so wider sweeps measure
            neutral-to-slower — see :func:`_resolve_subsystem_batch`), and on
            accelerators the widest width whose modeled footprint
            (:func:`tier1_truncated_peak_memory_bytes`) fits the memory
            budget (:func:`tier1_truncated_subsystem_width`).  An integer
            fixes the width (clamped to ``[1, B]``; 1 is the fully serial
            minimum-memory sweep).  Any width computes identical
            per-subsystem arithmetic — the knob trades memory for batched
            parallel work.  Ignored by the non-truncated routes.
        tier1_adjoint_window: opt-in bounded reverse mode for the truncated
            structured direct kernel.  ``None`` (default)
            keeps the taped gradient — bit-identical behavior to previous
            releases.  An integer ``w`` selects solvax's structure-preserving
            custom VJP: ``jax.grad`` through the truncated solve then runs at
            ``O((keep + w) m^2)`` memory per (species, x) subsystem instead of
            taping the full ``Nxi`` sweep, with an exact right-hand-side
            gradient and ``O(rho^{2w})`` coefficient-gradient error;
            ``w >= Nxi`` reproduces the taped gradient exactly.

            **Reverse mode only.**  The bounded path is a ``jax.custom_vjp``,
            and JAX cannot push forward-mode autodiff through one, so
            ``jax.jacfwd`` and ``jax.jvp`` over a solve that sets this option
            raise ``TypeError: can't apply forward-mode autodiff (jvp) to a
            custom_vjp function`` rather than falling back.  That matters here
            because :func:`dkx.sensitivity.jvp_flux` is forward mode: a
            sensitivity study that mixes the two must leave
            ``tier1_adjoint_window=None`` on the forward-mode calls.  Choosing
            the window is therefore a per-call-site decision, not a global
            switch.  See :func:`_solve_tier1_truncated`.
        device: where to run the solve.  ``"cpu"``/``"gpu"`` force a backend
            and a ``jax.Device`` pins the solve to that device: inputs are
            moved with ``jax.device_put`` and the solution is returned on
            the device that held ``rhs``.  ``"auto"`` (the default, also
            read from the ``DKX_SOLVE_DEVICE`` environment variable)
            additionally routes solves at or below the
            ``DKX_SOLVE_CPU_MAX_SIZE_TIER1`` / ``_TIER2`` thresholds
            to the host CPU on accelerator-default hosts — but both
            thresholds default to 0 (no routing), because the same-host
            measurements in docs/performance.rst found the GPU faster at
            every practical size; the knobs exist for hosts where that
            balance differs.  ``"default"`` disables all movement.  Under
            ``jit``/``grad`` tracing the knob is inert (arrays cannot move
            mid-trace), so jitted callers are unaffected.
        emit: route-status sink. The default prints one physical route
            description; ``None`` suppresses routine route presentation while
            retaining exceptions and convergence failures.

    Auto-policy structured routing (``method="auto"``, when
    :func:`tier1_available` is true):

    ================================  =============================================
    condition                         route
    ================================  =============================================
    uniform, peak estimate <= budget  full ``"block_tridiagonal"`` (any output
                                      mode, multi-RHS factor reuse)
    ramped Nxi_for_x or estimate >    ``"block_tridiagonal_truncated"`` when the
    budget                            truncation is valid (lowest ``keep`` blocks
                                      only, ~O(keep m^2) memory; ramps solved with
                                      per-subsystem ``n_blocks = Nxi_for_x[ix]``)
    …and truncation invalid           ``"gcrot"`` recycled Krylov, with a
                                      printed notice (high-l output the
                                      truncation cannot supply)
    ================================  =============================================

    "Truncation valid" means the operator admits the truncated kernel
    (:func:`_truncation_supported`) and the RHS support is confined to
    ``l < keep`` (:func:`_rhs_confined_to_lowest_blocks`; under jit/grad the
    structural ``rhs_mode in {1,2,3}`` guarantee is used).

    Returns:
        A :class:`SolveResult`; ``x`` matches the shape of ``rhs``.
    """
    _require_solvax()
    method = str(method).strip().lower()
    # "direct" and "iterative" are the names to reach for: they say what the
    # route does rather than how it is built.  The implementation names stay
    # accepted because the benchmarks and the solver trace speak them.
    method = {"iterative": "gmres"}.get(method, method)
    if method not in {
        "auto",
        "block_tridiagonal",
        "block_tridiagonal_truncated",
        "gmres",
        "direct",
    }:
        raise ValueError(
            f"unknown method {method!r}; use 'auto' (default), 'direct', or 'iterative'"
        )
    if method == "block_tridiagonal_truncated":
        ok, reason = tier1_available(op)
        if not ok:
            raise NotImplementedError(
                f"truncated structured direct route unavailable: {reason}"
            )
        keep = min(tier1_keep_lowest, op.n_xi)
        sup_ok, sup_reason = _truncation_supported(op, keep)
        if not sup_ok:
            raise NotImplementedError(
                f"truncated structured direct route unavailable: {sup_reason}"
            )
    require_float64()
    rhs2d, squeeze = _as_columns(rhs)
    if rhs2d.shape[0] != op.total_size:
        raise ValueError(
            f"rhs has {rhs2d.shape[0]} rows; operator expects {op.total_size}"
        )

    chosen = method
    if method == "auto":
        chosen = _auto_route(
            op,
            rhs2d,
            tier1_memory_budget_gb,
            tier1_keep_lowest,
            subsystem_batch,
            emit,
        )

    target_device = _resolve_solve_device(
        device, chosen, op, _is_traced(rhs2d, *jax.tree_util.tree_leaves(op))
    )
    home_device: jax.Device | None = None
    if target_device is not None:
        home = _single_device_of(rhs2d)
        if home is not None and home != target_device:
            home_device = home
            op = jax.device_put(op, target_device)
            rhs2d = jax.device_put(rhs2d, target_device)
            if x0 is not None:
                x0 = jax.device_put(x0, target_device)
            if recycle is not None:
                recycle = jax.device_put(recycle, target_device)

    if chosen in ("block_tridiagonal", "block_tridiagonal_truncated"):
        if chosen == "block_tridiagonal":
            result = _solve_tier1(
                op, rhs2d, tol=tol, atol=atol, differentiable=differentiable
            )
        else:
            keep = min(tier1_keep_lowest, op.n_xi)
            result = _solve_tier1_truncated(
                op,
                rhs2d,
                keep=keep,
                tol=tol,
                atol=atol,
                subsystem_batch=subsystem_batch,
                adjoint_window=tier1_adjoint_window,
            )
        if method == "auto" and not result.converged and not differentiable:
            # Structured elimination has no pivoting across blocks: on
            # near-singular systems (e.g. a nu_n=0 collisionless deck, whose
            # bordered constraint leaves the operator with condition numbers
            # ~1e18) its residual can miss the tolerance even though the
            # system is consistent.  Mirror the recycled-Krylov -> sparse-direct
            # pattern and fall through to the preconditioned Krylov route.
            print(
                "[dkx.solve] the structured direct solve missed the "
                f"tolerance (residuals={np.asarray(result.residual_norms)}); "
                "falling back to the recycled Krylov route."
            )
            chosen = "gmres"
    if chosen in ("block_tridiagonal", "block_tridiagonal_truncated"):
        pass  # structured direct result stands.
    elif chosen == "gmres":
        result = _solve_tier2(
            op,
            rhs2d,
            tol=tol,
            atol=atol,
            x0=x0,
            recycle=recycle,
            preconditioner=_resolve_preconditioner(preconditioner, use_preconditioner),
            prebuilt_precond=precond,
            drop_l_coupling_in_precond=drop_l_coupling_in_precond,
            restart=restart,
            recycle_dim=recycle_dim,
            max_restarts=max_restarts,
            differentiable=differentiable,
            check_adjoint=check_adjoint,
            adjoint_residual_factor=adjoint_residual_factor,
        )
        if method == "auto" and not result.converged and not differentiable:
            requested = _resolve_preconditioner(preconditioner, use_preconditioner)
            result = _escalate_after_tier2_stall(
                op,
                rhs2d,
                stalled=result,
                tol=tol,
                atol=atol,
                x0=x0,
                recycle=recycle,
                preconditioner=requested,
                drop_l_coupling_in_precond=drop_l_coupling_in_precond,
                restart=restart,
                recycle_dim=recycle_dim,
                max_restarts=max_restarts,
                check_adjoint=check_adjoint,
                adjoint_residual_factor=adjoint_residual_factor,
                max_dense_size=max_dense_size,
            )
    else:  # direct
        if differentiable:
            raise RuntimeError(
                "the sparse direct route (method='direct') is non-differentiable."
            )
        result = _solve_tier3(
            op, rhs2d, tol=tol, atol=atol, max_dense_size=max_dense_size
        )

    if home_device is not None:
        # Return the solution (and warm-start state) on the device that held
        # the inputs, so downstream pipelines are unaffected by the routing.
        result = replace(
            result,
            x=jax.device_put(result.x, home_device),
            residual_norms=jax.device_put(result.residual_norms, home_device),
            recycle=(
                None
                if result.recycle is None
                else jax.device_put(result.recycle, home_device)
            ),
        )

    if squeeze:
        result = replace(result, x=result.x[:, 0])
    return result
