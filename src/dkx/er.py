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
from dataclasses import dataclass, field, replace
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
            ``E_r > 0``), ``"unstable"`` (``dJr/dEr < 0``), ``"marginal"``
            (zero slope), or ``"unknown"`` (nonfinite field/slope).
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
        status: ``"converged"`` | ``"unbracketed"`` | ``"max_evaluations"`` |
            ``"current_tolerance"``.
        er: the selected (primary) root ``E_r`` (``None`` if unbracketed).
        radial_current: ``J_r`` at the selected root.
        root_type: classification of the selected root.
        per_species_flux: ``particleFlux_vm_psiHat`` at the selected root, shape
            ``(n_species,)``.
        iterations: the ordered radial-current evaluations (Fortran-parity
            history).
        roots: current-accepted roots found by the finite scan (length 1 for a
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
            It remains an approximate inverse for subsequent operators;
            original-residual and observable checks still govern acceptance.
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
    _profile_builder: Callable | None = field(default=None, repr=False, compare=False)

    def with_profiles(self, *, density_m3, temperature_keV) -> ErProblem:
        """Return updated native profiles, collisions and radial drives.

        Enable with ``prepare_er_scan(..., differentiable_profiles=True)``.
        Both inputs cover every Case surface/species; geometry, normalization
        and Coulomb logarithm stay fixed. Usable inside JAX transformations;
        the returned problem itself is a Python container, not a JAX array.
        Solver policy and field bounds are preserved; reuse factors separately
        only after checking their validity for the changed operator.
        """
        if self._profile_builder is None:
            raise ValueError("prepare_er_scan requires differentiable_profiles=True for profile updates")
        return replace(self, operator=self._profile_builder(density_m3, temperature_keV))


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
        thread into the next call. Both paths retain the complete Legendre
        state, including when a memory-bounded structured route is selected.
        Differentiable evaluations require the original kinetic residual to
        satisfy ``||Ax-b|| <= tol*||b||`` before forming the current; failures
        raise at execution time, including under JIT. This primal check does
        not certify the adjoint equation or grid convergence.
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
    if not np.isfinite(rtol) or rtol <= 0:
        raise ValueError("tol must be finite and positive")

    op = operator_at_er(problem.operator, er, dphi_per_er=problem.dphi_per_er)
    rhs = op.rhs()
    if differentiable:
        result = solve(
            op, rhs, method=method, tol=rtol, differentiable=True, emit=None,
            tier1_keep_lowest=op.n_xi,
        )
        x_full = jnp.reshape(result.x, (-1,))
        _check_kinetic_solution(op, er, x_full, rtol)
        state = None
    else:
        # Reuse whatever the previous point's route actually built. Nothing is
        # constructed speculatively here: a structured-direct point returns
        # precond=None and the next point simply builds nothing, which is why an
        # earlier attempt that guessed ahead of the route was a pessimization.
        result = solve(
            op, rhs, method=method, tol=rtol, x0=x0, recycle=recycle, precond=precond,
            # A reusable host state must include the Legendre tail: a moment-only
            # zero-padded head cannot satisfy the original kinetic equation.
            tier1_keep_lowest=op.n_xi,
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
    middle root has ``dJr/dEr < 0`` (unstable). Root count alone does not
    determine stability; a zero slope is marginal.
    """
    if not math.isfinite(er) or not math.isfinite(slope):
        return "unknown"
    if slope == 0.0:
        return "marginal"
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
    field_tol: float = 1e-10,
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
    while _same_sign(fa, fc):
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

    b = min(max(float(er_initial), a), c)
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
        tol1 = 2.0 * eps * abs(b) + 0.5 * float(field_tol)
        xm = 0.5 * (c - b)
        if abs(fb) <= float(current_tol):
            return b, True, "converged", "Brent algorithm successful."
        if abs(xm) <= tol1:
            return b, False, "current_tolerance", "Field bracket closed without satisfying the current tolerance."
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


def _check_host_kinetic_state(problem, er, state, tol):
    """Certify the original single-RHS equation before root acceptance/reuse."""
    if state is None:
        raise RuntimeError("Ambipolar kinetic solve returned no state")
    op = operator_at_er(problem.operator, er, dphi_per_er=problem.dphi_per_er)
    _check_kinetic_solution(op, er, state.x, tol)


def _check_kinetic_solution(op, er, x, tol):
    """Original-equation admission; valid scalar JIT checks stay on device."""
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    x = jnp.asarray(x).reshape((-1,))
    rhs = jnp.asarray(op.rhs()).reshape((-1,))
    defect = op.apply(x) - rhs
    residual = jnp.linalg.norm(defect)
    rhs_norm = jnp.linalg.norm(rhs)
    # A squared norm can underflow: homogeneous admission requires actual zeros.
    within_tolerance = jnp.where(
        jnp.all(rhs == 0), jnp.all(defect == 0), residual <= tol * rhs_norm
    )
    accepted = (jnp.all(jnp.isfinite(x)) & jnp.isfinite(rhs_norm)
                & jnp.isfinite(residual) & within_tolerance)

    def check(ok, field, residual, rhs_norm):
        if not bool(ok):
            raise RuntimeError(
                f"Ambipolar kinetic residual failed at Er={float(field):g}: "
                f"||Ax-b||={float(residual):.8e}, ||b||={float(rhs_norm):.8e}, tol={tol:.8e}; "
                "a homogeneous RHS requires entrywise zero defect"
            )

    if isinstance(accepted, jax.core.Tracer):
        jax.lax.cond(
            accepted, lambda: None,
            lambda: jax.debug.callback(check, accepted, er, residual, rhs_norm),
        )
    else:
        check(accepted, er, residual, rhs_norm)


def find_ambipolar_er(
    inp: SfincsInput | RawNamelist | str | Path | ErProblem,
    *,
    er_bracket: tuple[float, float] | None = None,
    er_initial: float | None = None,
    max_iter: int = 20,
    current_tol: float = 1e-10,
    field_tol: float = 1e-10,
    solve_method: str | None = None,
    tol: float | None = None,
    warm_start: bool = True,
    all_roots: bool = True,
    n_scan: int = 9,
    slope_step: float | None = None,
    emit: Callable[[str], None] | None = print,
) -> AmbipolarResult:
    """Solve ``J_r(E_r) = 0`` with the Fortran-parity Brent method.

    Evaluates ``E_r_min`` and ``E_r_max``, expands the bracket until the radial
    current changes sign, then refines the root with the ``ambipolarSolver.F90``
    ``zbrent`` update. Unlike its shared stopping tolerance, ``current_tol``
    bounds normalized radial current and ``field_tol`` bounds bracket width
    in the prepared problem's field units. A narrow bracket alone does not
    establish convergence. ``NEr_ambipolarSolve`` corresponds to ``max_iter``.
    Warm starts and GCROT recycling are
    threaded across evaluations when ``warm_start`` is set (a benefit only on
    recycled Krylov solves; structured direct solves ignore them).

    With ``all_roots`` the bracket is additionally coarse-scanned for sampled
    zeros and sign-changing intervals. Only current-accepted candidates are
    returned, while the selected root remains the Brent result. A finite scan
    cannot guarantee finding tangencies or every root. Zero slope is marginal;
    a small nonzero slope still requires an uncertainty study.

    Omitted ``solve_method`` and ``tol`` preserve a prepared problem's solver
    policy. Unprepared inputs default to ``"auto"`` and ``1e-10``. Explicit
    overrides apply to every kinetic solve, including the cold final check.
    Each state must satisfy the original equation ``||Ax-b|| <= tol*||b||``
    with finite state/RHS/residual before acceptance or reuse. A failed check
    raises ``RuntimeError``; a zero RHS requires a zero residual. This adds one
    operator application per field evaluation and does not certify grid accuracy.

    Returns an :class:`AmbipolarResult`.
    """
    for name, value in (("current_tol", current_tol), ("field_tol", field_tol)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if slope_step is not None and (not math.isfinite(slope_step) or slope_step <= 0):
        raise ValueError("slope_step must be finite and positive")
    problem = (
        inp
        if isinstance(inp, ErProblem)
        else prepare(
            inp,
            solve_method="auto" if solve_method is None else solve_method,
            tol=1e-10 if tol is None else tol,
            er_bracket=er_bracket,
            er_initial=er_initial,
        )
    )
    er_min, er_max = problem.er_min, problem.er_max
    if er_bracket is not None:
        er_min, er_max = float(er_bracket[0]), float(er_bracket[1])
    er_init = problem.er_initial if er_initial is None else float(er_initial)
    if not all(math.isfinite(e) for e in (er_min, er_max, er_init)) or er_min >= er_max:
        raise ValueError("The field bracket must be finite and increasing, with a finite initial field")

    iterations: list[AmbipolarIteration] = []
    state_box: dict[str, Any] = {"state": None, "gamma": None}
    kinetic_tol = problem.tol if tol is None else tol
    if not math.isfinite(kinetic_tol) or kinetic_tol <= 0:
        raise ValueError("tol must be finite and positive")

    def eval_jr(er: float, stage: str) -> float:
        er = float(er)
        if not math.isfinite(er):
            raise RuntimeError(f"Nonfinite ambipolar field at stage {stage}: Er={er}")
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
        _check_host_kinetic_state(problem, er, st, kinetic_tol)
        state_box["state"] = st
        gamma_np = np.asarray(gamma, dtype=np.float64).reshape((-1,))
        state_box["gamma"] = gamma_np
        value = float(j_r)
        if not math.isfinite(value) or not np.all(np.isfinite(gamma_np)):
            raise RuntimeError(f"Nonfinite ambipolar field/current/flux at stage {stage}: Er={er}, Jr={value}")
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
        field_tol=field_tol,
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

    # Accept against a cold final solve, independent of the continuation state.
    state_box["state"] = None
    jr_root = eval_jr(root_er, "root")
    gamma_root = state_box["gamma"]
    if not converged or abs(jr_root) > current_tol:
        return AmbipolarResult(
            converged=False, method="brent",
            status=status if not converged else "current_tolerance",
            er=float(root_er), radial_current=float(jr_root), root_type="unknown",
            per_species_flux=gamma_root, iterations=tuple(iterations),
            message=message if not converged else "Final current evaluation failed its tolerance.",
        )

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
            field_tol=field_tol,
            slope_step=h,
            primary=roots[0],
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
    field_tol: float = 1e-10,
) -> float | None:
    """Bracketed secant/bisection refinement of a single sign-changing bracket."""
    for _ in range(max_steps):
        if fhi == flo:
            mid = 0.5 * (lo + hi)
        else:
            mid = hi - fhi * (hi - lo) / (fhi - flo)
            if not (min(lo, hi) < mid < max(lo, hi)):
                mid = 0.5 * (lo + hi)
        fmid = eval_jr(mid, "scan_refine")
        if abs(fmid) <= current_tol:
            return mid
        if abs(hi - lo) <= field_tol:
            return None
        if _same_sign(flo, fmid):
            lo, flo = mid, fmid
        else:
            hi, fhi = mid, fmid
    return None


def _enumerate_roots(
    eval_jr: Callable[[float, str], float],
    *,
    er_min: float,
    er_max: float,
    n_scan: int,
    current_tol: float,
    slope_step: float,
    primary: AmbipolarRoot,
    field_tol: float = 1e-10,
) -> list[AmbipolarRoot]:
    """Classify accepted sampled zeros and sign-changing scan intervals."""
    grid = np.linspace(float(er_min), float(er_max), int(n_scan))
    fvals = np.asarray([eval_jr(float(e), "scan") for e in grid], dtype=np.float64)
    candidates = [float(e) for e, f in zip(grid, fvals, strict=True) if f == 0.0]
    for i in range(len(grid) - 1):
        flo, fhi = float(fvals[i]), float(fvals[i + 1])
        if flo == 0.0 or fhi == 0.0 or _same_sign(flo, fhi):
            continue
        er = _refine_secant(
            eval_jr, float(grid[i]), flo, float(grid[i + 1]), fhi,
            current_tol=current_tol, field_tol=field_tol,
        )
        if er is not None:
            candidates.append(er)
    roots: list[AmbipolarRoot] = []
    for er in sorted(candidates):
        jr = eval_jr(er, "scan_root")
        if not math.isfinite(jr) or abs(jr) > current_tol:
            continue
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
    solve_method: str | None = None,
    tol: float | None = None,
    root_tol: float = 1e-11,
    max_root_iter: int = 60,
    current_tol: float = 1e-12,
    min_abs_slope: float = 0.0,
):
    """Differentiable ambipolar ``E_r`` (a scalar JAX array).

    The residual ``f(E_r) = J_r(E_r)`` is a differentiable function of ``E_r``
    and of the operator's parameters (:func:`radial_current` with
    ``differentiable=True``).  The forward root is found with a black-box
    secant, wrapped by :func:`solvax.implicit.root_solve`
    (``jax.lax.custom_root``): ``jax.grad`` of the returned ``E_r`` w.r.t. any
    parameter ``p`` that the operator closes over follows the implicit function
    theorem

        dEr/dp = -(dJr/dEr)^{-1} dJr/dp,

    with ``dJr/dEr`` and ``dJr/dp`` from autodiff of :func:`radial_current` — no
    finite differences. When several roots exist this
    differentiates the one selected by ``er0`` (seed it near the desired root,
    e.g. with :func:`find_ambipolar_er`).

    Args:
        inp_or_operator: an :class:`ErProblem`, a base
            :class:`~dkx.drift_kinetic.KineticOperator` (needs
            ``dphi_per_er``), or a deck.
        er0: initial guess selecting the root and seeding the secant.
        dphi_per_er, z_s: overrides for the bare-operator path.
        solve_method, tol: overrides for the routed differentiable solve;
            omitted values preserve a prepared problem's policy (deck defaults
            are auto and 1e-10).
        root_tol: secant step tolerance and maximum final local Newton
            correction ``abs(Jr / (dJr/dEr))``, in the problem's field units.
        max_root_iter: positive integer forward iteration cap.
        current_tol: maximum absolute final normalized radial current.
        min_abs_slope: strictly exceeded by ``abs(dJr/dEr)`` at the root,
            in normalized current per field unit. Zero rejects exactly flat
            roots; choose a positive threshold for application-specific
            marginal-root rejection.

    Returns:
        The ambipolar ``E_r`` as a scalar JAX array, differentiable via
        ``jax.grad`` / ``jax.jacobian``.

    Raises:
        ValueError: invalid static acceptance controls.
        RuntimeError: nonfinite or unacceptable final root, current, slope,
            or local correction (wrapped by JAX under compilation). These
            local checks do not establish uniqueness or grid convergence.
    """
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    from solvax.implicit import root_solve  # noqa: PLC0415

    for name, value, positive in (
        ("root_tol", root_tol, True), ("current_tol", current_tol, False),
        ("min_abs_slope", min_abs_slope, False),
    ):
        if not np.isfinite(value) or value < 0 or (positive and value == 0):
            raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    if isinstance(max_root_iter, bool) or not isinstance(max_root_iter, (int, np.integer)) or max_root_iter < 1:
        raise ValueError("max_root_iter must be a positive integer")

    problem = _resolve_problem(
        inp_or_operator, dphi_per_er=dphi_per_er, z_s=z_s,
        solve_method=solve_method or "auto", tol=tol if tol is not None else 1e-10,
    )

    def residual(er):
        j_r, _gamma, _state = radial_current(
            problem,
            er,
            solve_method=solve_method,
            tol=tol,
            differentiable=True,
        )
        return j_r

    def solver(f, x_init):
        # Value-only secant iterations; one field JVP checks final acceptance.
        # custom_root supplies the implicit-function-theorem parameter tangent.
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
        current, slope = jax.jvp(f, (x_root,), (jnp.ones_like(x_root),))

        def check(root, current, slope):
            root, current, slope = float(root), float(current), float(slope)
            if not (
                np.isfinite(root) and np.isfinite(current) and np.isfinite(slope)
                and abs(current) <= current_tol and abs(slope) > min_abs_slope
                and abs(current / slope) <= root_tol
            ):
                raise RuntimeError(
                    "Ambipolar root acceptance failed: "
                    f"Er={root}, Jr={current}, dJr/dEr={slope}; "
                    f"require |Jr| <= {current_tol}, |dJr/dEr| > {min_abs_slope}, "
                    f"and |Jr/(dJr/dEr)| <= {root_tol}."
                )

        jax.debug.callback(check, x_root, current, slope)
        return x_root

    return root_solve(residual, jnp.asarray(er0, dtype=jnp.float64), solver)
