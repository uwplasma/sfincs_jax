"""Implicit and adjoint sensitivity helpers.

The functions here are intentionally small and model-agnostic. They implement
the derivative identity used by SFINCS Fortran v3 adjoint diagnostics and by
JAX implicit differentiation:

``A(p) x(p) = b(p)``, ``J(p) = c(p)^T x(p) + J_0(p)``.

The tangent equation is ``A x_p = b_p - A_p x``. The scalar adjoint equation is
``A^T lambda = c``. Both routes must agree before a derivative is used in an
optimization or ambipolar Newton step.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

# The JAX backend is imported below; dkx/runtime.py explains why this is here.
from .runtime import configure as _configure_runtime

_configure_runtime()

import jax
import jax.numpy as jnp

from .input_compat import (
    bool_config_values as _bool_values,
    config_bool as _config_bool,
    config_float as _config_float,
    config_int as _config_int,
    lookup_config_value as _lookup_config_value,
)


Array = Any
LinearSolver = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
VectorSolver = Callable[[jnp.ndarray], jnp.ndarray]
LinearOperatorApply = Callable[[jnp.ndarray], jnp.ndarray]
ObservableFn = Callable[[float], float]
LinearObservableBuilder = Callable[[float], "LinearObservableSystem"]
MatrixFreeLinearObservableBuilder = Callable[[float], "MatrixFreeLinearObservableSystem"]
StateObservableFn = Callable[[jnp.ndarray], Any]
FluxFn = Callable[[Any], Any]


def _immutable_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType(dict(value or {}))


_BASE_ADJOINT_DEBUG_FIELDS = (
    "dHeatFluxdLambda_finitediff",
    "dParticleFluxdLambda_finitediff",
    "dParallelFlowdLambda_finitediff",
    "dTotalHeatFluxdLambda_finitediff",
    "dRadialCurrentdLambda_finitediff",
    "dBootstrapdLambda_finitediff",
    "particleFluxPercentError",
    "heatFluxPercentError",
    "parallelFlowPercentError",
    "bootstrapPercentError",
    "totalHeatFluxPercentError",
    "radialCurrentPercentError",
)


_ADJOINT_SPECIES_FIELD_RANKS = {
    "dHeatFluxdLambda": 4,
    "dParticleFluxdLambda": 4,
    "dParallelFlowdLambda": 4,
    "dHeatFluxdLambda_finitediff": 4,
    "dParticleFluxdLambda_finitediff": 4,
    "dParallelFlowdLambda_finitediff": 4,
    "heatFluxPercentError": 4,
    "parallelFlowPercentError": 4,
    "particleFluxPercentError": 4,
}

_ADJOINT_SURFACE_FIELD_RANKS = {
    "dTotalHeatFluxdLambda": 3,
    "dRadialCurrentdLambda": 3,
    "dBootstrapdLambda": 3,
    "dPhidPsidLambda": 3,
    "dTotalHeatFluxdLambda_finitediff": 3,
    "dRadialCurrentdLambda_finitediff": 3,
    "dBootstrapdLambda_finitediff": 3,
    "dPhidPsidLambda_finitediff": 3,
    "bootstrapPercentError": 3,
    "dPhidPsiPercentError": 3,
    "radialCurrentPercentError": 3,
    "totalHeatFluxPercentError": 3,
}


def _output_shape(value: Any) -> tuple[int, ...] | None:
    if isinstance(value, Mapping) and "shape" in value:
        return tuple(int(item) for item in value["shape"])
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return tuple(int(item) for item in shape)


def fortran_v3_adjoint_sensitivity_output_fields(config: Any) -> tuple[str, ...]:
    """Return RHSMode-4/5 sensitivity fields written by SFINCS Fortran v3.

    The field list follows ``writeHDF5Output.F90``.  One source-code quirk is
    preserved intentionally: ``dParallelFlowdLambda`` is gated by
    ``adjointParticleFluxOption`` or ``debugAdjoint``, not by
    ``adjointParallelFlowOption``.
    """

    general = ("general", "generalparameters")
    adjoint = ("adjointOptions", "adjointoptions")
    rhs_mode = _config_int(config, general, "RHSMode", 1)
    if rhs_mode not in (4, 5):
        return ()

    debug = _config_bool(config, adjoint, "debugAdjoint", False)
    heat = any(_bool_values(_lookup_config_value(config, adjoint, "adjointHeatFluxOption", ())))
    particle = any(_bool_values(_lookup_config_value(config, adjoint, "adjointParticleFluxOption", ())))
    total_heat = _config_bool(config, adjoint, "adjointTotalHeatFluxOption", False)
    radial_current = _config_bool(config, adjoint, "adjointRadialCurrentOption", False)
    bootstrap = _config_bool(config, adjoint, "adjointBootstrapOption", False)

    fields: list[str] = []
    if heat or debug:
        fields.append("dHeatFluxdLambda")
    if particle or debug:
        fields.append("dParticleFluxdLambda")
    if particle or debug:
        fields.append("dParallelFlowdLambda")
    if total_heat or debug:
        fields.append("dTotalHeatFluxdLambda")
    if radial_current or debug:
        fields.append("dRadialCurrentdLambda")
    if bootstrap or debug:
        fields.append("dBootstrapdLambda")
    if rhs_mode == 5:
        fields.append("dPhidPsidLambda")
    if debug:
        fields.extend(_BASE_ADJOINT_DEBUG_FIELDS)
        if rhs_mode == 5:
            fields.extend(("dPhidPsiPercentError", "dPhidPsidLambda_finitediff"))
    return tuple(dict.fromkeys(fields))


def fortran_v3_adjoint_sensitivity_output_ranks(config: Any) -> Mapping[str, int]:
    """Return expected tensor ranks for Fortran-v3 RHSMode-4/5 output fields."""

    ranks: dict[str, int] = {}
    for field_name in fortran_v3_adjoint_sensitivity_output_fields(config):
        rank = _ADJOINT_SPECIES_FIELD_RANKS.get(field_name)
        if rank is None:
            rank = _ADJOINT_SURFACE_FIELD_RANKS.get(field_name)
        if rank is not None:
            ranks[field_name] = rank
    return MappingProxyType(ranks)


def validate_fortran_v3_adjoint_sensitivity_output_surface(
    config: Any,
    outputs: Mapping[str, Any],
) -> tuple[str, ...]:
    """Validate a RHSMode-4/5 sensitivity output surface against v3 fields.

    ``outputs`` can be an HDF5-like mapping of arrays or a lightweight summary
    mapping whose values contain a ``shape`` entry.  This helper intentionally
    checks only names and tensor ranks; numerical physics identities are pinned
    by separate tests because they depend on the selected adjoint options.
    """

    errors = list(validate_fortran_v3_adjoint_sensitivity_constraints(config))
    ranks = fortran_v3_adjoint_sensitivity_output_ranks(config)
    for field_name in fortran_v3_adjoint_sensitivity_output_fields(config):
        if field_name not in outputs:
            errors.append(f"Missing Fortran-v3 RHSMode 4/5 sensitivity output field: {field_name}.")
            continue
        expected_rank = ranks.get(field_name)
        if expected_rank is None:
            continue
        shape = _output_shape(outputs[field_name])
        if shape is None:
            errors.append(f"Cannot determine shape for sensitivity output field: {field_name}.")
            continue
        if len(shape) != expected_rank:
            errors.append(
                f"Sensitivity output field {field_name} has rank {len(shape)}; expected {expected_rank}."
            )
    return tuple(errors)


def validate_fortran_v3_adjoint_sensitivity_constraints(config: Any) -> tuple[str, ...]:
    """Return Fortran-v3 compatibility errors for RHSMode 4/5 adjoint runs."""

    general = ("general", "generalparameters")
    physics = ("physicsParameters", "physicsparameters")
    resolution = ("resolutionParameters", "resolutionparameters")
    geometry = ("geometryParameters", "geometryparameters")
    adjoint = ("adjointOptions", "adjointoptions")

    rhs_mode = _config_int(config, general, "RHSMode", 1)
    geometry_scheme = _config_int(config, geometry, "geometryScheme", 1)
    include_phi1 = _config_bool(config, physics, "includePhi1", False)
    e_parallel = _config_float(config, physics, "EParallelHat", 0.0)
    magnetic_drift_scheme = _config_int(config, physics, "magneticDriftScheme", 0)
    constraint_scheme = _config_int(config, resolution, "constraintScheme", -1)
    collision_operator = _config_int(config, physics, "collisionOperator", 0)
    discrete_adjoint = _config_bool(config, adjoint, "discreteAdjointOption", True)
    radial_current = _config_bool(config, adjoint, "adjointRadialCurrentOption", False)
    include_xdot = _config_bool(config, physics, "includeXDotTerm", True)
    use_dkes_exb = _config_bool(config, physics, "useDKESExBDrift", False)
    include_er_xidot = _config_bool(config, physics, "includeElectricFieldTermInXiDot", True)

    errors: list[str] = []
    if rhs_mode not in (4, 5):
        errors.append("RHSMode 4 or 5 is required for Fortran-v3 adjoint sensitivities.")
    if rhs_mode == 5 and radial_current:
        errors.append("RHSMode=5 cannot be used with adjointRadialCurrentOption in SFINCS Fortran v3.")
    if 4 < geometry_scheme < 8:
        errors.append("RHSMode>3 must use a Boozer-coordinate geometry scheme in SFINCS Fortran v3.")
    if include_phi1:
        errors.append("RHSMode>3 cannot be used with includePhi1 in SFINCS Fortran v3.")
    if e_parallel != 0.0:
        errors.append("RHSMode>3 cannot be used with EParallelHat != 0 in SFINCS Fortran v3.")
    if magnetic_drift_scheme > 0:
        errors.append("RHSMode>3 cannot use tangential magnetic drifts in SFINCS Fortran v3.")
    if constraint_scheme not in (-1, 1):
        errors.append("RHSMode>3 requires constraintScheme=-1 or 1 in SFINCS Fortran v3.")
    if collision_operator != 0:
        errors.append("RHSMode>3 requires collisionOperator=0 in SFINCS Fortran v3.")
    full_trajectories = include_xdot and (not use_dkes_exb) and include_er_xidot
    dkes_trajectories = (not include_xdot) and use_dkes_exb and (not include_er_xidot)
    if (not discrete_adjoint) and not (full_trajectories or dkes_trajectories):
        errors.append(
            "discreteAdjointOption=false must use either full trajectories or DKES trajectories in SFINCS Fortran v3."
        )
    return tuple(errors)


@dataclass(frozen=True, slots=True, kw_only=True)
class LinearObservableSystem:
    """Linearized observable system at one scalar parameter value.

    A concrete SFINCS problem owner should build this object from the same
    operator/RHS graph used for the solve. Keeping the fields explicit makes the
    finite-difference gate compare against the real assembled terms, not a
    separately coded analytic shortcut.
    """

    parameter: float
    matrix: Array
    rhs: Array
    matrix_derivative: Array
    rhs_derivative: Array
    observable_vector: Array
    observable_vector_derivative: Array | None = None
    observable_offset: float = 0.0
    observable_offset_derivative: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameter", float(self.parameter))
        object.__setattr__(self, "observable_offset", float(self.observable_offset))
        object.__setattr__(
            self,
            "observable_offset_derivative",
            float(self.observable_offset_derivative),
        )
        object.__setattr__(self, "metadata", _immutable_mapping(self.metadata))


@dataclass(frozen=True, slots=True, kw_only=True)
class MatrixFreeLinearObservableSystem:
    """Matrix-free observable derivative system for production-size solves.

    Concrete problem owners provide the selected production solver, transpose
    solver, operator action, transpose action, and parameter-derivative action.
    This is the promotion path from dense small-deck certificates to RHSMode-1
    and RHSMode-4/5 derivative gates that should not assemble a dense matrix.
    """

    parameter: float
    size: int
    rhs: Array
    rhs_derivative: Array
    apply: LinearOperatorApply
    transpose_apply: LinearOperatorApply
    derivative_apply: LinearOperatorApply
    solve: VectorSolver
    transpose_solve: VectorSolver
    observable_vector: Array
    observable_vector_derivative: Array | None = None
    observable_offset: float = 0.0
    observable_offset_derivative: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameter", float(self.parameter))
        object.__setattr__(self, "size", int(self.size))
        object.__setattr__(self, "observable_offset", float(self.observable_offset))
        object.__setattr__(
            self,
            "observable_offset_derivative",
            float(self.observable_offset_derivative),
        )
        object.__setattr__(self, "metadata", _immutable_mapping(self.metadata))
        if self.size <= 0:
            raise ValueError("size must be positive.")


@dataclass(frozen=True, slots=True)
class LinearObservableDerivativeResult:
    """Derivative certificate for a scalar observable of a linear solve."""

    parameter: float
    observable: float
    derivative: float
    tangent_derivative: float
    adjoint_derivative: float
    primal_residual_norm: float
    tangent_residual_norm: float
    adjoint_residual_norm: float
    tangent_adjoint_abs_error: float
    finite_difference_derivative: float | None = None
    finite_difference_abs_error: float | None = None
    finite_difference_step: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "parameter",
            "observable",
            "derivative",
            "tangent_derivative",
            "adjoint_derivative",
            "primal_residual_norm",
            "tangent_residual_norm",
            "adjoint_residual_norm",
            "tangent_adjoint_abs_error",
        ):
            object.__setattr__(self, name, float(getattr(self, name)))
        if self.finite_difference_derivative is not None:
            object.__setattr__(
                self,
                "finite_difference_derivative",
                float(self.finite_difference_derivative),
            )
        if self.finite_difference_abs_error is not None:
            object.__setattr__(
                self,
                "finite_difference_abs_error",
                float(self.finite_difference_abs_error),
            )
        if self.finite_difference_step is not None:
            object.__setattr__(self, "finite_difference_step", float(self.finite_difference_step))
        object.__setattr__(self, "metadata", _immutable_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class JvpVjpDotProductResult:
    """Adjoint consistency certificate for one JVP/VJP pair."""

    primal_value: Any
    tangent_value: Any
    cotangent_value: Any
    vjp_value: Any
    lhs: float
    rhs: float
    abs_error: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "lhs", float(self.lhs))
        object.__setattr__(self, "rhs", float(self.rhs))
        object.__setattr__(self, "abs_error", float(self.abs_error))


def _as_vector(name: str, value: Array) -> jnp.ndarray:
    array = jnp.asarray(value, dtype=jnp.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D vector.")
    return array


def _as_matrix(name: str, value: Array) -> jnp.ndarray:
    array = jnp.asarray(value, dtype=jnp.float64)
    if array.ndim != 2 or int(array.shape[0]) != int(array.shape[1]):
        raise ValueError(f"{name} must be a square 2D matrix.")
    return array


def _default_solve(matrix: jnp.ndarray, rhs: jnp.ndarray) -> jnp.ndarray:
    return jnp.linalg.solve(matrix, rhs)


def jvp_flux(
    flux: FluxFn,
    parameters: Any,
    tangent: Any,
) -> tuple[Any, Any]:
    """Return ``flux(parameters)`` and its Jacobian-vector product."""

    return jax.jvp(flux, (parameters,), (tangent,))


def vjp_flux(
    flux: FluxFn,
    parameters: Any,
    cotangent: Any,
) -> tuple[Any, Any]:
    """Return ``flux(parameters)`` and the vector-Jacobian product."""

    value, pullback = jax.vjp(flux, parameters)
    return value, pullback(cotangent)[0]


def adjoint_dot_product_check(
    flux: FluxFn,
    parameters: Any,
    tangent: Any,
    cotangent: Any,
) -> JvpVjpDotProductResult:
    """Check ``<JVP(dp), y> = <dp, VJP(y)>`` for one flux diagnostic."""

    primal, tangent_value = jvp_flux(flux, parameters, tangent)
    primal_vjp, vjp_value = vjp_flux(flux, parameters, cotangent)
    del primal_vjp
    lhs = jnp.vdot(jnp.ravel(jnp.asarray(tangent_value)), jnp.ravel(jnp.asarray(cotangent)))
    rhs = jnp.vdot(jnp.ravel(jnp.asarray(tangent)), jnp.ravel(jnp.asarray(vjp_value)))
    return JvpVjpDotProductResult(
        primal_value=primal,
        tangent_value=tangent_value,
        cotangent_value=cotangent,
        vjp_value=vjp_value,
        lhs=float(lhs),
        rhs=float(rhs),
        abs_error=abs(float(lhs) - float(rhs)),
    )


def evaluate_linear_observable(
    system: LinearObservableSystem,
    *,
    solve: LinearSolver | None = None,
) -> float:
    """Solve one linear-observable system and return ``c^T x + J0``."""

    a = _as_matrix("matrix", system.matrix)
    b = _as_vector("rhs", system.rhs)
    c = _as_vector("observable_vector", system.observable_vector)
    if int(b.shape[0]) != int(a.shape[0]):
        raise ValueError("rhs length must match matrix size.")
    if int(c.shape[0]) != int(a.shape[0]):
        raise ValueError("observable_vector length must match matrix size.")
    solve_fn = solve or _default_solve
    x = solve_fn(a, b)
    return float(jnp.vdot(c, x) + float(system.observable_offset))


def evaluate_matrix_free_linear_observable(system: MatrixFreeLinearObservableSystem) -> float:
    """Solve one matrix-free linear-observable system and return ``c^T x + J0``."""

    b = _as_vector("rhs", system.rhs)
    c = _as_vector("observable_vector", system.observable_vector)
    if int(b.shape[0]) != int(system.size):
        raise ValueError("rhs length must match system.size.")
    if int(c.shape[0]) != int(system.size):
        raise ValueError("observable_vector length must match system.size.")
    x = _as_vector("solution", system.solve(b))
    if int(x.shape[0]) != int(system.size):
        raise ValueError("solve(rhs) length must match system.size.")
    return float(jnp.vdot(c, x) + float(system.observable_offset))


def probe_linear_observable_vector(
    observable: StateObservableFn,
    *,
    size: int,
    chunk_size: int = 128,
    dtype: Any = jnp.float64,
) -> tuple[jnp.ndarray, float]:
    """Return ``(c, J0)`` for a scalar diagnostic ``J(x) = c^T x + J0``.

    Existing SFINCS diagnostics are often easier to trust than a manually
    duplicated formula. This utility probes those diagnostics with basis
    vectors in bounded chunks, which is suitable for small validation decks and
    for deriving checked observable weights during development. Production
    owners should replace it with analytic weights once the formula is pinned.
    """

    n = int(size)
    chunk = int(chunk_size)
    if n <= 0:
        raise ValueError("size must be positive.")
    if chunk <= 0:
        raise ValueError("chunk_size must be positive.")
    zero = jnp.zeros((n,), dtype=dtype)
    offset = float(jnp.asarray(observable(zero), dtype=dtype).reshape(()))
    parts: list[jnp.ndarray] = []
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        width = stop - start
        basis = jnp.zeros((width, n), dtype=dtype)
        basis = basis.at[jnp.arange(width), jnp.arange(start, stop)].set(1.0)
        values = jax.vmap(lambda vec: jnp.asarray(observable(vec), dtype=dtype).reshape(()))(basis)
        parts.append(values - offset)
    return jnp.concatenate(parts, axis=0), offset


def _centered_finite_difference(
    observable: ObservableFn,
    *,
    parameter: float,
    step: float,
) -> float:
    if step <= 0.0:
        raise ValueError("finite_difference_step must be positive.")
    plus = float(observable(float(parameter) + step))
    minus = float(observable(float(parameter) - step))
    return (plus - minus) / (2.0 * step)


def implicit_linear_observable_derivative_from_builder(
    build_system: LinearObservableBuilder,
    *,
    parameter: float,
    solve: LinearSolver | None = None,
    transpose_solve: LinearSolver | None = None,
    finite_difference_step: float | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> LinearObservableDerivativeResult:
    """Build a parameter-local system and return its derivative certificate.

    This is the integration point for concrete RHSMode 1/RHSMode 4/5 owners: they
    provide one fixed-shape builder for ``A(p)``, ``b(p)``, and the observable.
    The optional finite-difference gate is generated by reusing that builder at
    ``p +/- h``.
    """

    system = build_system(float(parameter))
    merged_metadata = {**dict(system.metadata), **dict(metadata or {})}
    fd_observable = None
    if finite_difference_step is not None:
        def fd_observable(value: float) -> float:
            return evaluate_linear_observable(
                build_system(float(value)),
                solve=solve,
            )

    return implicit_linear_observable_derivative(
        matrix=system.matrix,
        rhs=system.rhs,
        matrix_derivative=system.matrix_derivative,
        rhs_derivative=system.rhs_derivative,
        observable_vector=system.observable_vector,
        observable_vector_derivative=system.observable_vector_derivative,
        observable_offset=system.observable_offset,
        observable_offset_derivative=system.observable_offset_derivative,
        parameter=float(parameter),
        solve=solve,
        transpose_solve=transpose_solve,
        finite_difference_observable=fd_observable,
        finite_difference_step=finite_difference_step,
        metadata=merged_metadata,
    )


def linear_observable_algebraic_error(
    *, rhs: Array, state: Array, observable_vector: Array,
    apply: LinearOperatorApply, transpose_apply: LinearOperatorApply,
    transpose_solve: VectorSolver, primal_rtol: float, adjoint_rtol: float,
) -> dict[str, Any]:
    """Audit a linear moment's algebraic error without assembling a matrix.

    With r=b-Ax and A^T lambda=c, c^T(x_exact-x)=lambda^T r. A numerical
    adjoint leaves a remainder (c-A^T lambda)^T(x_exact-x), so this is an
    estimate, not a rigorous bound or a discretization/geometry error estimate.
    Callers supply original constrained operator actions and a qualified
    transpose solver; tighten that solve to check estimate stability. Reporting
    is synchronous and outside AD. The returned scalars fit Result.metadata.
    """
    import math  # noqa: PLC0415

    primal_rtol, adjoint_rtol = float(primal_rtol), float(adjoint_rtol)
    if any(not math.isfinite(t) or t < 0 for t in (primal_rtol, adjoint_rtol)):
        raise ValueError("residual tolerances must be finite and nonnegative")
    b = _as_vector("rhs", rhs)

    def checked(name, value):
        v = _as_vector(name, value)
        if not v.size or v.shape != b.shape or not bool(jnp.all(jnp.isfinite(v))):
            raise ValueError(f"{name} must be finite, nonempty, and match rhs shape")
        return v

    def relative_norm(defect, reference):
        # A shared scale avoids both norm underflow and inf/inf on finite data.
        scale = jnp.maximum(jnp.max(jnp.abs(defect)), jnp.max(jnp.abs(reference)))
        scale = jnp.where(scale == 0, 1.0, scale)
        numerator = float(jnp.linalg.norm(defect / scale))
        denominator = float(jnp.linalg.norm(reference / scale))
        return numerator / denominator if denominator else (0.0 if numerator == 0 else math.inf)

    b = checked("rhs", b)
    x = checked("state", state)
    c = checked("observable_vector", observable_vector)
    r = checked("primal defect", b - checked("apply(state)", apply(x)))
    primal_residual = relative_norm(r, b)
    if primal_residual > primal_rtol:
        raise ValueError(f"original primal relative residual {primal_residual} exceeds {primal_rtol}")
    lam = checked("adjoint", transpose_solve(c))
    dual_defect = checked("adjoint defect", c - checked("transpose_apply(adjoint)", transpose_apply(lam)))
    adjoint_residual = relative_norm(dual_defect, c)
    if adjoint_residual > adjoint_rtol:
        raise ValueError(f"original adjoint relative residual {adjoint_residual} exceeds {adjoint_rtol}")
    value, correction = float(jnp.vdot(c, x)), float(jnp.vdot(lam, r))
    if not all(math.isfinite(v) for v in (value, correction, value + correction)):
        raise ValueError("nonfinite observable or algebraic correction")
    return {
        "kind": "linear_observable_algebraic_estimate",
        "observable": value, "signed_correction": correction,
        "absolute_estimate": abs(correction), "corrected_observable": value + correction,
        "primal_relative_residual": primal_residual,
        "adjoint_relative_residual": adjoint_residual,
        "primal_rtol": primal_rtol, "adjoint_rtol": adjoint_rtol,
        "is_bound": False,
    }


def implicit_matrix_free_linear_observable_derivative(
    system: MatrixFreeLinearObservableSystem,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> LinearObservableDerivativeResult:
    """Return tangent/adjoint derivative certificate without dense assembly."""

    n = int(system.size)
    b = _as_vector("rhs", system.rhs)
    bp = _as_vector("rhs_derivative", system.rhs_derivative)
    c = _as_vector("observable_vector", system.observable_vector)
    if int(b.shape[0]) != n:
        raise ValueError("rhs length must match system.size.")
    if int(bp.shape[0]) != n:
        raise ValueError("rhs_derivative length must match system.size.")
    if int(c.shape[0]) != n:
        raise ValueError("observable_vector length must match system.size.")
    cp = (
        jnp.zeros_like(c)
        if system.observable_vector_derivative is None
        else _as_vector("observable_vector_derivative", system.observable_vector_derivative)
    )
    if int(cp.shape[0]) != n:
        raise ValueError("observable_vector_derivative length must match system.size.")

    x = _as_vector("solution", system.solve(b))
    if int(x.shape[0]) != n:
        raise ValueError("solve(rhs) length must match system.size.")
    ax = _as_vector("apply(solution)", system.apply(x))
    if int(ax.shape[0]) != n:
        raise ValueError("apply(solution) length must match system.size.")
    primal_residual = ax - b

    apx = _as_vector("derivative_apply(solution)", system.derivative_apply(x))
    if int(apx.shape[0]) != n:
        raise ValueError("derivative_apply(solution) length must match system.size.")
    tangent_rhs = bp - apx
    x_tangent = _as_vector("tangent_solution", system.solve(tangent_rhs))
    if int(x_tangent.shape[0]) != n:
        raise ValueError("solve(tangent_rhs) length must match system.size.")
    tangent_residual = _as_vector("apply(tangent_solution)", system.apply(x_tangent)) - tangent_rhs

    lam = _as_vector("adjoint_solution", system.transpose_solve(c))
    if int(lam.shape[0]) != n:
        raise ValueError("transpose_solve(observable_vector) length must match system.size.")
    adjoint_residual = _as_vector("transpose_apply(adjoint_solution)", system.transpose_apply(lam)) - c

    observable = jnp.vdot(c, x) + float(system.observable_offset)
    tangent_derivative = (
        jnp.vdot(cp, x)
        + jnp.vdot(c, x_tangent)
        + float(system.observable_offset_derivative)
    )
    adjoint_derivative = (
        jnp.vdot(cp, x)
        + jnp.vdot(lam, tangent_rhs)
        + float(system.observable_offset_derivative)
    )
    derivative = 0.5 * (tangent_derivative + adjoint_derivative)
    combined_metadata = {
        **dict(system.metadata),
        **dict(metadata or {}),
        "system_kind": "matrix_free_linear_observable",
    }
    return LinearObservableDerivativeResult(
        parameter=float(system.parameter),
        observable=float(observable),
        derivative=float(derivative),
        tangent_derivative=float(tangent_derivative),
        adjoint_derivative=float(adjoint_derivative),
        primal_residual_norm=float(jnp.linalg.norm(primal_residual)),
        tangent_residual_norm=float(jnp.linalg.norm(tangent_residual)),
        adjoint_residual_norm=float(jnp.linalg.norm(adjoint_residual)),
        tangent_adjoint_abs_error=abs(float(tangent_derivative) - float(adjoint_derivative)),
        metadata=combined_metadata,
    )


def implicit_matrix_free_linear_observable_derivative_from_builder(
    build_system: MatrixFreeLinearObservableBuilder,
    *,
    parameter: float,
    finite_difference_step: float | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> LinearObservableDerivativeResult:
    """Build a matrix-free system and optionally compare to centered finite differences."""

    fd_derivative = None
    fd_abs_error = None
    fd_step = None
    if finite_difference_step is not None:
        fd_step = float(finite_difference_step)
        fd_derivative = _centered_finite_difference(
            lambda value: evaluate_matrix_free_linear_observable(build_system(float(value))),
            parameter=float(parameter),
            step=fd_step,
        )
    result = implicit_matrix_free_linear_observable_derivative(
        build_system(float(parameter)),
        metadata=metadata,
    )
    if fd_derivative is not None:
        fd_abs_error = abs(float(result.derivative) - float(fd_derivative))
    return LinearObservableDerivativeResult(
        parameter=result.parameter,
        observable=result.observable,
        derivative=result.derivative,
        tangent_derivative=result.tangent_derivative,
        adjoint_derivative=result.adjoint_derivative,
        primal_residual_norm=result.primal_residual_norm,
        tangent_residual_norm=result.tangent_residual_norm,
        adjoint_residual_norm=result.adjoint_residual_norm,
        tangent_adjoint_abs_error=result.tangent_adjoint_abs_error,
        finite_difference_derivative=fd_derivative,
        finite_difference_abs_error=fd_abs_error,
        finite_difference_step=fd_step,
        metadata=result.metadata,
    )


def implicit_linear_observable_derivative(
    *,
    matrix: Array,
    rhs: Array,
    matrix_derivative: Array,
    rhs_derivative: Array,
    observable_vector: Array,
    observable_vector_derivative: Array | None = None,
    observable_offset: float = 0.0,
    observable_offset_derivative: float = 0.0,
    parameter: float = 0.0,
    solve: LinearSolver | None = None,
    transpose_solve: LinearSolver | None = None,
    finite_difference_observable: ObservableFn | None = None,
    finite_difference_step: float | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> LinearObservableDerivativeResult:
    """Differentiate a scalar observable of ``A(p) x(p) = b(p)``.

    The helper returns both tangent and adjoint derivatives and their residuals.
    Larger matrix-free SFINCS operators can supply custom ``solve`` and
    ``transpose_solve`` callbacks; small validation gates use the dense default.
    """

    a = _as_matrix("matrix", matrix)
    ap = _as_matrix("matrix_derivative", matrix_derivative)
    b = _as_vector("rhs", rhs)
    bp = _as_vector("rhs_derivative", rhs_derivative)
    c = _as_vector("observable_vector", observable_vector)
    if a.shape != ap.shape:
        raise ValueError("matrix and matrix_derivative must have the same shape.")
    if int(b.shape[0]) != int(a.shape[0]):
        raise ValueError("rhs length must match matrix size.")
    if int(bp.shape[0]) != int(a.shape[0]):
        raise ValueError("rhs_derivative length must match matrix size.")
    if int(c.shape[0]) != int(a.shape[0]):
        raise ValueError("observable_vector length must match matrix size.")
    if observable_vector_derivative is None:
        cp = jnp.zeros_like(c)
    else:
        cp = _as_vector("observable_vector_derivative", observable_vector_derivative)
        if int(cp.shape[0]) != int(a.shape[0]):
            raise ValueError("observable_vector_derivative length must match matrix size.")

    solve_fn = solve or _default_solve
    transpose_solve_fn = transpose_solve or solve_fn

    x = solve_fn(a, b)
    primal_residual = a @ x - b
    tangent_rhs = bp - ap @ x
    x_tangent = solve_fn(a, tangent_rhs)
    tangent_residual = a @ x_tangent - tangent_rhs
    lam = transpose_solve_fn(a.T, c)
    adjoint_residual = a.T @ lam - c

    direct_derivative = jnp.vdot(cp, x) + float(observable_offset_derivative)
    observable = jnp.vdot(c, x) + float(observable_offset)
    tangent_derivative = direct_derivative + jnp.vdot(c, x_tangent)
    adjoint_derivative = direct_derivative + jnp.vdot(lam, tangent_rhs)
    derivative = 0.5 * (tangent_derivative + adjoint_derivative)

    fd_derivative: float | None = None
    fd_abs_error: float | None = None
    if finite_difference_observable is not None:
        step = 1.0e-6 if finite_difference_step is None else float(finite_difference_step)
        fd_derivative = _centered_finite_difference(
            finite_difference_observable,
            parameter=float(parameter),
            step=step,
        )
        fd_abs_error = abs(float(derivative) - float(fd_derivative))
    else:
        step = None if finite_difference_step is None else float(finite_difference_step)

    return LinearObservableDerivativeResult(
        parameter=float(parameter),
        observable=float(observable),
        derivative=float(derivative),
        tangent_derivative=float(tangent_derivative),
        adjoint_derivative=float(adjoint_derivative),
        primal_residual_norm=float(jnp.linalg.norm(primal_residual)),
        tangent_residual_norm=float(jnp.linalg.norm(tangent_residual)),
        adjoint_residual_norm=float(jnp.linalg.norm(adjoint_residual)),
        tangent_adjoint_abs_error=abs(float(tangent_derivative) - float(adjoint_derivative)),
        finite_difference_derivative=fd_derivative,
        finite_difference_abs_error=fd_abs_error,
        finite_difference_step=step,
        metadata=metadata,
    )


__all__ = (
    "FluxFn",
    "JvpVjpDotProductResult",
    "LinearObservableBuilder",
    "LinearObservableDerivativeResult",
    "LinearOperatorApply",
    "LinearObservableSystem",
    "MatrixFreeLinearObservableBuilder",
    "MatrixFreeLinearObservableSystem",
    "StateObservableFn",
    "VectorSolver",
    "adjoint_dot_product_check",
    "evaluate_linear_observable",
    "evaluate_matrix_free_linear_observable",
    "fortran_v3_adjoint_sensitivity_output_fields",
    "fortran_v3_adjoint_sensitivity_output_ranks",
    "implicit_linear_observable_derivative",
    "implicit_linear_observable_derivative_from_builder",
    "implicit_matrix_free_linear_observable_derivative",
    "implicit_matrix_free_linear_observable_derivative_from_builder",
    "jvp_flux",
    "linear_observable_algebraic_error",
    "probe_linear_observable_vector",
    "validate_fortran_v3_adjoint_sensitivity_constraints",
    "validate_fortran_v3_adjoint_sensitivity_output_surface",
    "vjp_flux",
)
