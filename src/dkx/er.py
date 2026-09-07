"""Ambipolar radial electric field on the canonical stack.

Finds the radial electric field ``E_r`` that zeroes the radial current

    J_r(E_r) = sum_a Z_a Gamma_a(E_r) = 0,

by driving the canonical ``inputs -> drift_kinetic -> solve -> moments`` stack
at a sequence of ``E_r`` values.  It replaces the legacy in-process Brent owner
``problems/ambipolar.py`` and adds a differentiable ambipolar root.

Fortran counterpart (``ambipolarSolver.F90``): ``updateEr`` sets
``dPhiHatdpsiHat = ddrHat2ddpsiHat * (-Er)`` and defines the radial current as
``sum_s Zs * particleFlux_vm(1:Nspecies)``; ``ambipolarSolverBrent``
(``ambipolarSolveOption==2``) evaluates ``Er_min`` and ``Er_max``, expands the
bracket until the radial current changes sign, then refines the root with the
Numerical-Recipes ``zbrent`` update using ``Er_search_tolerance_f``.

Public entry points:

- :func:`radial_current` — one canonical solve at a given ``E_r`` returning
  ``(J_r, per-species Gamma, ErSolveState)``.  ``x0``/``recycle`` thread warm
  starts and GCROT recycling across ``E_r`` evaluations (recycled Krylov).  When
  called on a base :class:`~dkx.drift_kinetic.KineticOperator` (or an
  :class:`ErProblem`) it is a differentiable function of ``E_r`` and of the
  operator's parameters.
- :func:`find_ambipolar_er` — the Fortran-parity Brent root solve with bracket
  expansion, per-species fluxes, an iteration history, and
  ion / electron / unstable classification from the sign of ``dJr/dEr``.
- :func:`ambipolar_er` — the *differentiable* ambipolar ``E_r``: the root
  condition is wrapped with :func:`solvax.implicit.root_solve` so ``jax.grad``
  flows through ``E_r`` via the implicit function theorem
  ``dEr/dp = -(dJr/dEr)^{-1} dJr/dp``, with both Jacobians taken from autodiff
  of :func:`radial_current` (not finite differences).

Units follow SFINCS: ``E_r`` is the deck's normalized ``Er`` entry and the
per-species fluxes are ``particleFlux_vm_psiHat`` (the ``sum_s Z_s Gamma_s``
root is coordinate-independent because the ``psiHat`` <-> ``rHat`` Jacobian is a
positive species-independent factor).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np

from dkx.drift_kinetic import (
    KineticOperator,
    kinetic_operator_from_namelist,
)
from dkx.inputs import (
    RawNamelist,
    SfincsInput,
    load_sfincs_input,
    sfincs_input_from_raw,
)
from dkx.solve import SolveResult, solve

__all__ = [
    "AmbipolarIteration",
    "AmbipolarResult",
    "AmbipolarRoot",
    "ErProblem",
    "ErSolveState",
    "ambipolar_er",
    "find_ambipolar_er",
    "operator_at_er",
    "prepare",
    "radial_current",
]


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AmbipolarIteration:
    """One radial-current evaluation during an ambipolar solve."""

    index: int
    er: float
    radial_current: float
    stage: str


@dataclass(frozen=True)
class AmbipolarRoot:
    """A classified ambipolar root.

    Attributes:
        er: the root ``E_r``.
        radial_current: ``J_r`` at the root (residual; near zero).
        slope: ``dJr/dEr`` at the root (central finite difference, used only to
            classify the root — the differentiable gradient uses autodiff).
        root_type: ``"ion"`` (stable, ``E_r < 0``), ``"electron"`` (stable,
            ``E_r > 0``), or ``"unstable"`` (``dJr/dEr > 0`` on the standard
            stellarator S-curve, the middle branch).
    """

    er: float
    radial_current: float
    slope: float
    root_type: str


@dataclass(frozen=True)
class AmbipolarResult:
    """Result of an ambipolar ``E_r`` solve.

    Attributes:
        converged: whether the primary Brent solve converged.
        method: ``"brent"``.
        status: ``"converged"`` | ``"unbracketed"`` | ``"max_evaluations"``.
        er: the selected (primary) root ``E_r`` (``None`` if unbracketed).
        radial_current: ``J_r`` at the selected root.
        root_type: classification of the selected root.
        per_species_flux: ``particleFlux_vm_psiHat`` at the selected root, shape
            ``(n_species,)``.
        iterations: the ordered radial-current evaluations (Fortran-parity
            history).
        roots: every root found in the bracket, classified (length 1 for a
            single-root case; the differentiable :func:`ambipolar_er` wrapper
            differentiates one *selected* root).
        message: human-readable status detail.
    """

    converged: bool
    method: str
    status: str
    er: float | None
    radial_current: float | None
    root_type: str
    per_species_flux: np.ndarray | None
    iterations: tuple[AmbipolarIteration, ...] = ()
    roots: tuple[AmbipolarRoot, ...] = ()
    message: str = ""

    @property
    def er_values(self) -> tuple[float, ...]:
        return tuple(it.er for it in self.iterations)

    @property
    def radial_currents(self) -> tuple[float, ...]:
        return tuple(it.radial_current for it in self.iterations)


@dataclass(frozen=True)
class ErSolveState:
    """Warm-start payload threaded across ``E_r`` evaluations.

    Attributes:
        x: the solved state, shape ``(total_size, 1)`` — a recycled Krylov warm
            start (``x0``) for the next nearby ``E_r``.
        recycle: the GCROT recycle pair from the previous Krylov solve, or
            ``None`` (structured direct solves do not recycle).
        result: the underlying :class:`~dkx.solve.SolveResult`
            (``method``, ``iterations``, ``residual_norms``, ``timings``).
        precond: the preconditioner the previous point's solve built, or
            ``None`` when its route did not need one. Threaded forward so a
            bracket search builds one preconditioner rather than one per point.
            Reuse is safe because a preconditioner never changes the converged
            answer, only the iteration count.
    """

    x: Any
    recycle: Any
    result: SolveResult
    precond: Any = None


@dataclass(frozen=True)
class ErProblem:
    """A prepared, shape-stable ambipolar problem.

    Built by :func:`prepare` or :func:`dkx.prepare_er_scan`; every field evaluation reuses ``operator``
    (a base :class:`~dkx.drift_kinetic.KineticOperator` with the ExB / Er
    term flags already switched on) and overrides only ``dPhiHatdpsiHat``, so
    the per-evaluation cost is one solve and the transform is differentiable.

    Attributes:
        operator: the base operator (built at a nonzero reference ``E_r`` so
            the electric-field terms enabled by the selected trajectory model
            have a stable structure).
        dphi_per_er: the conversion factor ``c`` with
            ``dPhiHatdpsiHat = c * E_r`` (``= -ddrHat2ddpsiHat``, the
            ``ambipolarSolver.F90`` ``updateEr`` relation).
        z_s: species charges, shape ``(n_species,)``.
        er_units: input-field units (normalized for decks, kV/m for native Cases).
        er_initial, er_min, er_max: the initial guess and default bracket read
            from the deck (``Er`` / ``ErMin`` / ``ErMax``).
        solve_method, tol: forwarded to :func:`dkx.solve.solve`.
    """

    operator: KineticOperator
    dphi_per_er: float
    z_s: np.ndarray
    er_initial: float
    er_min: float
    er_max: float
    solve_method: str = "auto"
    tol: float = 1e-10
    er_units: str = "normalized"


# ---------------------------------------------------------------------------
# Namelist helpers
# ---------------------------------------------------------------------------


def _phys_value(raw: RawNamelist, key: str, default: float) -> float:
    """Case-insensitive scalar lookup in ``&physicsParameters`` of a raw deck."""
    group = raw.groups.get("physicsparameters", {})
    want = key.upper()
    for name, value in group.items():
        if str(name).upper() == want:
            return float(np.asarray(value).reshape(()))
    return float(default)


def _raw_with_er(raw: RawNamelist, er: float) -> RawNamelist:
    """Return a raw-deck copy with ``&physicsParameters ER`` overridden.

    Mirrors :func:`dkx.run._raw_with_validated_overrides`: the operator
    builder reads the raw namelist, and ``kinetic_operator_from_namelist`` maps
    ``Er`` to ``dPhiHatdpsiHat`` via the ``inputRadialCoordinate=4`` (Er) path.
    """
    groups = {name: dict(values) for name, values in raw.groups.items()}
    groups.setdefault("physicsparameters", {})["ER"] = float(er)
    return replace(raw, groups=groups)


def _as_input(inp: SfincsInput | RawNamelist | str | Path) -> SfincsInput:
    if isinstance(inp, SfincsInput):
        return inp
    if isinstance(inp, RawNamelist):
        return sfincs_input_from_raw(inp)
    return load_sfincs_input(Path(inp))


# ---------------------------------------------------------------------------
# Operator override at a given E_r (differentiable)
# ---------------------------------------------------------------------------


def operator_at_er(op_base: KineticOperator, er, *, dphi_per_er) -> KineticOperator:
    """Return ``op_base`` with the electric-field drive set for ``E_r``.

    Overrides both the RHS-drive value ``dphi_hat_dpsi_hat`` and the kinetic
    ExB/Er value ``dphi_hat_dpsi_hat_kinetic`` to ``dphi_per_er * E_r`` (the
    ``ambipolarSolver.F90`` ``updateEr`` relation) while keeping every other
    leaf — geometry, collisions, speed grid — and the static term flags fixed.
    Because the ExB, Er-xiDot and Er-xDot terms are *linear* in
    ``dphi_hat_dpsi_hat_kinetic``, this reproduces a fresh
    ``kinetic_operator_from_namelist`` build at ``E_r`` exactly (the base must
    carry the term flags, i.e. be built at a nonzero reference ``E_r``), and it
    is a differentiable function of ``E_r``.
    """
    import jax.numpy as jnp  # noqa: PLC0415

    dphi = jnp.asarray(dphi_per_er, dtype=jnp.float64) * jnp.asarray(er, dtype=jnp.float64)
    return replace(op_base, dphi_hat_dpsi_hat=dphi, dphi_hat_dpsi_hat_kinetic=dphi)


# ---------------------------------------------------------------------------
# Prepare a problem from a deck
# ---------------------------------------------------------------------------


def prepare(
    inp: SfincsInput | RawNamelist | str | Path,
    *,
    solve_method: str = "auto",
    tol: float = 1e-10,
    er_bracket: tuple[float, float] | None = None,
    er_initial: float | None = None,
) -> ErProblem:
    """Build a shape-stable :class:`ErProblem` from a SFINCS deck.

    The base operator is built once at a nonzero reference ``E_r`` so the ExB /
    Er term flags are active; the ``E_r`` -> ``dPhiHatdpsiHat`` factor is read
    back from that operator (``dphi_hat_dpsi_hat_kinetic`` at the reference).
    Requires a deck whose electric field is the ``inputRadialCoordinate=4``
    (Er) knob — the only mode ``ambipolarSolver.F90`` drives.
    """
    typed = _as_input(inp)
    if typed.general.rhs_mode != 1:
        raise NotImplementedError(
            "ambipolar E_r solves require RHSMode=1 (single-RHS profile drive); "
            f"got RHSMode={typed.general.rhs_mode}."
        )
    raw = typed.raw
    if raw is None:
        raise ValueError("prepare requires an input parsed from a namelist file.")

    # Build the base operator at a nonzero reference E_r so with_exb / with_er_*
    # are switched on; dphi_hat_dpsi_hat_kinetic then equals the E_r factor.
    er_ref = 1.0
    op_base = kinetic_operator_from_namelist(_raw_with_er(raw, er_ref))
    dphi_per_er = float(np.asarray(op_base.dphi_hat_dpsi_hat_kinetic).reshape(())) / er_ref
    z_s = np.asarray(op_base.z_s, dtype=np.float64).reshape((-1,))

    er_deck = _phys_value(raw, "Er", 0.0)
    if er_bracket is None:
        er_min = _phys_value(raw, "ErMin", er_deck - 5.0)
        er_max = _phys_value(raw, "ErMax", er_deck + 5.0)
    else:
        er_min, er_max = float(er_bracket[0]), float(er_bracket[1])
    er_init = float(er_deck if er_initial is None else er_initial)

    return ErProblem(
        operator=op_base,
        dphi_per_er=dphi_per_er,
        z_s=z_s,
        er_initial=er_init,
        er_min=float(er_min),
        er_max=float(er_max),
        solve_method=solve_method,
        tol=tol,
    )


def _resolve_problem(
    inp_or_operator: Any,
    *,
    dphi_per_er: float | None,
    z_s: Any | None,
    solve_method: str,
    tol: float,
) -> ErProblem:
    """Coerce the ``radial_current`` first argument to an :class:`ErProblem`."""
    if isinstance(inp_or_operator, ErProblem):
        return inp_or_operator
    if isinstance(inp_or_operator, KineticOperator):
        if dphi_per_er is None:
            raise ValueError(
                "radial_current(operator, ...) needs dphi_per_er (dPhiHatdpsiHat "
                "per unit E_r); use er.prepare(deck) to obtain it, or pass an ErProblem."
            )
        charges = (
            np.asarray(z_s, dtype=np.float64).reshape((-1,))
            if z_s is not None
            else np.asarray(inp_or_operator.z_s, dtype=np.float64).reshape((-1,))
        )
        return ErProblem(
            operator=inp_or_operator,
            dphi_per_er=float(dphi_per_er),
            z_s=charges,
            er_initial=0.0,
            er_min=0.0,
            er_max=0.0,
            solve_method=solve_method,
            tol=tol,
        )
    return prepare(inp_or_operator, solve_method=solve_method, tol=tol)


# ---------------------------------------------------------------------------
# Radial current at one E_r
# ---------------------------------------------------------------------------


def radial_current(
    inp_or_operator: Any,
    er,
    *,
    x0: Any | None = None,
    recycle: Any | None = None,
    precond: Any | None = None,
    dphi_per_er: float | None = None,
    z_s: Any | None = None,
    solve_method: str | None = None,
    tol: float | None = None,
    differentiable: bool = False,
):
    """Radial current ``J_r`` and per-species fluxes at one ``E_r``.

    Builds the canonical operator at ``E_r`` (overriding ``dPhiHatdpsiHat`` from
    ``E_r`` through the ``ambipolarSolver.F90`` ``updateEr`` conversion — the
    same value :func:`dkx.run.run_profile` sets), solves the single-RHS
    system with :func:`dkx.solve.solve`, and forms

        J_r = sum_a Z_a Gamma_a,   Gamma_a = particleFlux_vm_psiHat[a].

    Args:
        inp_or_operator: an :class:`ErProblem`, a base
            :class:`~dkx.drift_kinetic.KineticOperator` (needs
            ``dphi_per_er``), or a deck (``SfincsInput`` / path).
        er: the radial electric field (scalar; may be a traced JAX value).
        x0: Krylov warm-start state from a previous :class:`ErSolveState`.
        recycle: GCROT recycle pair from a previous Krylov solve.
        dphi_per_er: ``dPhiHatdpsiHat`` per unit ``E_r`` (required only when the
            first argument is a bare operator).
        z_s: species charges override (defaults to the operator's).
        solve_method, tol: forwarded to :func:`dkx.solve.solve`.
        differentiable: wrap the solve in ``solvax.implicit.linear_solve`` so
            ``jax.grad`` flows through ``J_r`` (used by :func:`ambipolar_er`).

    Returns:
        ``(J_r, per_species_flux, ErSolveState)`` — ``J_r`` a scalar JAX array,
        ``per_species_flux`` shape ``(n_species,)``, and the warm-start state to
        thread into the next call.
    """
    import jax.numpy as jnp  # noqa: PLC0415

    from dkx.run import profile_moments_from_operator  # noqa: PLC0415

    problem = _resolve_problem(
        inp_or_operator,
        dphi_per_er=dphi_per_er,
        z_s=z_s,
        solve_method=solve_method or "auto",
        tol=tol if tol is not None else 1e-10,
    )
    method = solve_method or problem.solve_method
    rtol = tol if tol is not None else problem.tol

    op = operator_at_er(problem.operator, er, dphi_per_er=problem.dphi_per_er)
    rhs = op.rhs()
    if differentiable:
        # Fully traceable path for autodiff / implicit differentiation: assemble
        # the operator densely (all jnp) and solve exactly with jnp.linalg.solve.
        # The routed ``solve`` builds its factorization with host numpy, which
        # cannot run under ``solvax.root_solve`` closure conversion; the dense
        # solve is exact and matches the structured direct route to machine
        # precision on the tiny ambipolar decks that carry a differentiable
        # objective.
        x_full = _dense_solve(op, rhs)
        state = None
    else:
        # Reuse whatever the previous point's route actually built. Nothing is
        # constructed speculatively here: a structured-direct point returns
        # precond=None and the next point simply builds nothing, which is why an
        # earlier attempt that guessed ahead of the route was a pessimization.
        result = solve(
            op, rhs, method=method, tol=rtol, x0=x0, recycle=recycle, precond=precond
        )
        x_full = jnp.reshape(result.x, (-1,))
        state = ErSolveState(
            x=result.x,
            recycle=result.recycle,
            result=result,
            precond=result.precond,
        )
    table = profile_moments_from_operator(op, x_full)
    gamma = table["particleFlux_vm_psiHat"]  # (n_species,)
    charges = jnp.asarray(problem.z_s, dtype=jnp.float64)
    j_r = jnp.tensordot(charges, gamma, axes=1)
    return j_r, gamma, state


def _dense_solve(op: KineticOperator, rhs):
    """Exact dense solve of ``op x = rhs`` (traceable, differentiable).

    Assembles the matrix column by column from the matrix-free ``op.apply`` and
    solves with :func:`jax.numpy.linalg.solve`.  Requires the un-truncated
    embedding (``Nxi_for_x_option=0``, no structurally singular DOFs) — the
    caller (:func:`ambipolar_er`) checks this on the concrete operator.
    """
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    n = int(op.total_size)
    eye = jnp.eye(n, dtype=jnp.float64)
    a = jnp.transpose(jax.vmap(op.apply)(eye))  # column i = op.apply(e_i)
    return jnp.linalg.solve(a, jnp.reshape(rhs, (-1,)))


# ---------------------------------------------------------------------------
# Fortran-parity Brent root solve (option 2)
# ---------------------------------------------------------------------------


def _same_sign(a: float, b: float) -> bool:
    return (a > 0.0 and b > 0.0) or (a < 0.0 and b < 0.0)


def _classify(er: float, slope: float) -> str:
    """ion / electron / unstable from ``E_r`` sign and the ``dJr/dEr`` sign.

    The radial field relaxes as ``dEr/dt ~ -J_r``, so a root is *stable* iff
    ``dJr/dEr > 0``.  On the standard stellarator S-curve the outer stable ion
    (``E_r < 0``) and electron (``E_r > 0``) roots have ``dJr/dEr > 0`` and the
    middle root has ``dJr/dEr < 0`` (unstable); a single root is always stable.
    """
    if slope < 0.0:
        return "unstable"
    return "electron" if er > 0.0 else "ion"


def _brent(
    eval_jr: Callable[[float, str], float],
    *,
    er_min: float,
    er_max: float,
    er_initial: float,
    max_iter: int,
    current_tol: float,
    max_expansions: int,
    emit: Callable[[str], None] | None,
) -> tuple[float | None, bool, str, str]:
    """Bracket-expanding Numerical-Recipes zbrent (``ambipolarSolverBrent``).

    Returns ``(root_er, converged, status, message)``.
    """
    eps = 1.0e-15
    a = float(er_min)
    fa = eval_jr(a, "bracket_min")
    c = float(er_max)
    fc = eval_jr(c, "bracket_max")

    # Expand the bracket until the radial current changes sign.
    expansions = 0
    while fa * fc > 0.0:
        if expansions >= max_expansions:
            return None, False, "unbracketed", (
                "Radial current did not change sign after "
                f"{max_expansions} bracket expansions in [{a:g}, {c:g}]."
            )
        if emit is not None:
            emit("Warning: root not bracketed in Brent solve! Expanding search bounds...")
        if abs(fa) < abs(fc):
            a = a - (c - a)
            fa = eval_jr(a, "expand_min")
        else:
            c = c + (c - a)
            fc = eval_jr(c, "expand_max")
        expansions += 1

    b = float(er_initial)
    fb = eval_jr(b, "initial")

    # Orient the initial guess into the bracket (ambipolarSolver.F90 lines 119-125).
    if _same_sign(fa, fb):
        fa, a = fb, b
    elif _same_sign(fc, fb):
        fc, c = fb, b

    d = b - a
    e = d
    for _ in range(4, int(max_iter) + 1):
        if _same_sign(fb, fc):
            c, fc = a, fa
            e = b - a
            d = e
        if abs(fc) < abs(fb):
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb
        tol1 = 2.0 * eps * abs(b) + 0.5 * float(current_tol)
        xm = 0.5 * (c - b)
        if abs(xm) <= tol1 or abs(fb) < float(current_tol):
            return b, True, "converged", "Brent algorithm successful."
        if abs(e) >= tol1 and abs(fa) > abs(fb):
            s = fb / fa
            if a == c:
                p = 2.0 * xm * s
                q = 1.0 - s
            else:
                q = fa / fc
                r = fb / fc
                p = s * (2.0 * xm * q * (q - r) - (b - a) * (r - 1.0))
                q = (q - 1.0) * (r - 1.0) * (s - 1.0)
            if p > 0.0:
                q = -q
            p = abs(p)
            if 2.0 * p < min(3.0 * xm * q - abs(tol1 * q), abs(e * q)):
                e = d
                d = p / q
            else:
                d = xm
                e = d
        else:
            d = xm
            e = d
        a, fa = b, fb
        if abs(d) > tol1:
            b = b + d
        else:
            b = b + math.copysign(tol1, xm)
        fb = eval_jr(b, "brent")

    return b, False, "max_evaluations", (
        "The E_r search did not converge within max_iter evaluations."
    )


def find_ambipolar_er(
    inp: SfincsInput | RawNamelist | str | Path | ErProblem,
    *,
    er_bracket: tuple[float, float] | None = None,
    er_initial: float | None = None,
    max_iter: int = 20,
    current_tol: float = 1e-10,
    solve_method: str = "auto",
    tol: float = 1e-10,
    warm_start: bool = True,
    all_roots: bool = True,
    n_scan: int = 9,
    slope_step: float | None = None,
    emit: Callable[[str], None] | None = print,
) -> AmbipolarResult:
    """Solve ``J_r(E_r) = 0`` with the Fortran-parity Brent method.

    Evaluates ``E_r_min`` and ``E_r_max``, expands the bracket until the radial
    current changes sign, then refines the root with the ``ambipolarSolver.F90``
    ``zbrent`` update (``Er_search_tolerance_f = current_tol``,
    ``NEr_ambipolarSolve = max_iter``).  Warm starts and GCROT recycling are
    threaded across evaluations when ``warm_start`` is set (a benefit only on
    recycled Krylov solves; structured direct solves ignore them).

    With ``all_roots`` the bracket is additionally coarse-scanned so every root
    is returned classified (ion / electron / unstable), while the *selected*
    root remains the Brent result.

    Returns an :class:`AmbipolarResult`.
    """
    problem = (
        inp
        if isinstance(inp, ErProblem)
        else prepare(
            inp,
            solve_method=solve_method,
            tol=tol,
            er_bracket=er_bracket,
            er_initial=er_initial,
        )
    )
    er_min, er_max = problem.er_min, problem.er_max
    if er_bracket is not None:
        er_min, er_max = float(er_bracket[0]), float(er_bracket[1])
    er_init = problem.er_initial if er_initial is None else float(er_initial)

    iterations: list[AmbipolarIteration] = []
    state_box: dict[str, Any] = {"state": None, "gamma": None}
    flux_cache: dict[float, np.ndarray] = {}

    def eval_jr(er: float, stage: str) -> float:
        er = float(er)
        prev = state_box["state"] if warm_start else None
        j_r, gamma, st = radial_current(
            problem,
            er,
            x0=(prev.x if prev is not None else None),
            recycle=(prev.recycle if prev is not None else None),
            precond=(prev.precond if prev is not None else None),
            solve_method=solve_method,
            tol=tol,
        )
        state_box["state"] = st
        gamma_np = np.asarray(gamma, dtype=np.float64).reshape((-1,))
        state_box["gamma"] = gamma_np
        flux_cache[er] = gamma_np
        value = float(j_r)
        iterations.append(AmbipolarIteration(len(iterations) + 1, er, value, stage))
        if emit is not None:
            emit(f"Solving with Er = {er:.15g}   radialCurrent = {value:.8e}")
        return value

    t0 = time.perf_counter()
    root_er, converged, status, message = _brent(
        eval_jr,
        er_min=er_min,
        er_max=er_max,
        er_initial=er_init,
        max_iter=max_iter,
        current_tol=current_tol,
        max_expansions=50,
        emit=emit,
    )
    elapsed = time.perf_counter() - t0

    if root_er is None:
        if emit is not None:
            emit(message)
        return AmbipolarResult(
            converged=False,
            method="brent",
            status=status,
            er=None,
            radial_current=None,
            root_type="unknown",
            per_species_flux=None,
            iterations=tuple(iterations),
            message=message,
        )

    # One clean evaluation at the root for the reported fluxes.
    jr_root = eval_jr(root_er, "root")
    gamma_root = state_box["gamma"]

    # Classify the selected root by the sign of dJr/dEr (central difference).
    span = max(abs(er_max - er_min), 1.0)
    h = float(slope_step) if slope_step is not None else 1e-3 * span
    slope = (eval_jr(root_er + h, "slope_plus") - eval_jr(root_er - h, "slope_minus")) / (2.0 * h)
    root_type = _classify(root_er, slope)

    roots: list[AmbipolarRoot] = [AmbipolarRoot(root_er, jr_root, slope, root_type)]
    if all_roots and n_scan >= 3:
        roots = _enumerate_roots(
            eval_jr,
            er_min=er_min,
            er_max=er_max,
            n_scan=n_scan,
            current_tol=current_tol,
            slope_step=h,
            primary=roots[0],
        )
        # Keep the Brent root classification/slope in the returned selected root.
        root_type = next(
            (r.root_type for r in roots if abs(r.er - root_er) <= 2.0 * h), root_type
        )

    if emit is not None:
        emit("Brent algorithm successful." if converged else message)
        emit("Here are the Ers we used: " + " ".join(f"{it.er:.8g}" for it in iterations))
        emit(
            "Here are the radial currents: "
            + " ".join(f"{it.radial_current:.4e}" for it in iterations)
        )
        emit(f"Time for ambipolar solve: {elapsed:.6g} seconds.")
        emit(f"Ambipolar Er = {root_er:.15g}  ({root_type} root)")

    return AmbipolarResult(
        converged=converged,
        method="brent",
        status=status,
        er=float(root_er),
        radial_current=float(jr_root),
        root_type=root_type,
        per_species_flux=gamma_root,
        iterations=tuple(iterations),
        roots=tuple(roots),
        message=message,
    )


def _refine_secant(
    eval_jr: Callable[[float, str], float],
    lo: float,
    flo: float,
    hi: float,
    fhi: float,
    *,
    current_tol: float,
    max_steps: int = 40,
) -> float:
    """Bracketed secant/bisection refinement of a single sign-changing bracket."""
    for _ in range(max_steps):
        if fhi == flo:
            mid = 0.5 * (lo + hi)
        else:
            mid = hi - fhi * (hi - lo) / (fhi - flo)
            if not (min(lo, hi) < mid < max(lo, hi)):
                mid = 0.5 * (lo + hi)
        fmid = eval_jr(mid, "scan_refine")
        if abs(fmid) < current_tol or abs(hi - lo) < 1e-13 * max(1.0, abs(hi)):
            return mid
        if _same_sign(flo, fmid):
            lo, flo = mid, fmid
        else:
            hi, fhi = mid, fmid
    return 0.5 * (lo + hi)


def _enumerate_roots(
    eval_jr: Callable[[float, str], float],
    *,
    er_min: float,
    er_max: float,
    n_scan: int,
    current_tol: float,
    slope_step: float,
    primary: AmbipolarRoot,
) -> list[AmbipolarRoot]:
    """Coarse-scan the bracket and classify every root (ion/electron/unstable)."""
    grid = np.linspace(float(er_min), float(er_max), int(n_scan))
    fvals = np.asarray([eval_jr(float(e), "scan") for e in grid], dtype=np.float64)
    roots: list[AmbipolarRoot] = []
    for i in range(len(grid) - 1):
        flo, fhi = float(fvals[i]), float(fvals[i + 1])
        if flo == 0.0:
            er = float(grid[i])
        elif _same_sign(flo, fhi):
            continue
        else:
            er = _refine_secant(
                eval_jr, float(grid[i]), flo, float(grid[i + 1]), fhi, current_tol=current_tol
            )
        jr = eval_jr(er, "scan_root")
        slope = (
            eval_jr(er + slope_step, "scan_slope_plus")
            - eval_jr(er - slope_step, "scan_slope_minus")
        ) / (2.0 * slope_step)
        roots.append(AmbipolarRoot(er, jr, slope, _classify(er, slope)))
    if not roots:
        roots.append(primary)
    return roots


# ---------------------------------------------------------------------------
# Differentiable ambipolar E_r (implicit function theorem)
# ---------------------------------------------------------------------------


def ambipolar_er(
    inp_or_operator: Any,
    *,
    er0: float = 0.0,
    dphi_per_er: float | None = None,
    z_s: Any | None = None,
    solve_method: str = "auto",
    tol: float = 1e-10,
    root_tol: float = 1e-11,
    max_root_iter: int = 60,
):
    """Differentiable ambipolar ``E_r`` (a scalar JAX array).

    The residual ``f(E_r) = J_r(E_r)`` is a differentiable function of ``E_r``
    and of the operator's parameters (:func:`radial_current` with
    ``differentiable=True``).  The forward root is found with a black-box
    bracketed secant, wrapped by :func:`solvax.implicit.root_solve`
    (``jax.lax.custom_root``): ``jax.grad`` of the returned ``E_r`` w.r.t. any
    parameter ``p`` that the operator closes over follows the implicit function
    theorem

        dEr/dp = -(dJr/dEr)^{-1} dJr/dp,

    with ``dJr/dEr`` and ``dJr/dp`` from autodiff of :func:`radial_current` — no
    finite differences.  When the bracket contains several roots this
    differentiates the one selected by ``er0`` (seed it near the desired root,
    e.g. with :func:`find_ambipolar_er`).

    Args:
        inp_or_operator: an :class:`ErProblem`, a base
            :class:`~dkx.drift_kinetic.KineticOperator` (needs
            ``dphi_per_er``), or a deck.
        er0: initial guess selecting the root and seeding the secant.
        dphi_per_er, z_s: overrides for the bare-operator path.
        solve_method, tol: forwarded to the differentiable solve.
        root_tol, max_root_iter: forward secant tolerance and iteration cap.

    Returns:
        The ambipolar ``E_r`` as a scalar JAX array, differentiable via
        ``jax.grad`` / ``jax.jacobian``.
    """
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    from solvax.implicit import root_solve  # noqa: PLC0415

    problem = _resolve_problem(
        inp_or_operator, dphi_per_er=dphi_per_er, z_s=z_s, solve_method=solve_method, tol=tol
    )
    # The differentiable residual uses an exact dense solve, which needs the
    # un-truncated embedding.  Check on the concrete operator (before tracing).
    if problem.operator.active_dof_mask() is not None:
        raise NotImplementedError(
            "ambipolar_er's differentiable dense solve requires Nxi_for_x_option=0 "
            "(no Legendre truncation); use find_ambipolar_er for the truncated case."
        )

    def residual(er):
        j_r, _gamma, _state = radial_current(
            problem.operator,
            er,
            dphi_per_er=problem.dphi_per_er,
            z_s=problem.z_s,
            solve_method=solve_method,
            tol=tol,
            differentiable=True,
        )
        return j_r

    def solver(f, x_init):
        # Black-box forward root: a bracketed secant (value only, so no nested
        # gradient); custom_root supplies the implicit-function-theorem tangent.
        x_init = jnp.asarray(x_init, dtype=jnp.float64)
        step = jnp.where(jnp.abs(x_init) > 0.0, 1e-3 * jnp.abs(x_init), 1e-3)
        x_prev = x_init
        f_prev = f(x_prev)
        x_cur = x_init + step

        def body(state):
            xp, fp, xc, i = state
            fc = f(xc)
            denom = fc - fp
            denom = jnp.where(jnp.abs(denom) < 1e-300, 1e-300, denom)
            xn = xc - fc * (xc - xp) / denom
            return (xc, fc, xn, i + 1)

        def cond(state):
            xp, _fp, xc, i = state
            return (i < max_root_iter) & (jnp.abs(xc - xp) > root_tol)

        _xp, _fp, x_root, _i = jax.lax.while_loop(cond, body, (x_prev, f_prev, x_cur, 0))
        return x_root

    return root_solve(residual, jnp.asarray(er0, dtype=jnp.float64), solver)
