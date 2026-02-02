from __future__ import annotations

from dataclasses import dataclass

from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
from jax import tree_util as jtu

from .namelist import Namelist
from .solver import GMRESSolveResult, gmres_solve
from .transport_matrix import transport_matrix_size_from_rhs_mode, v3_transport_matrix_from_state_vectors
from .v3 import geometry_from_namelist, grids_from_namelist
from .v3_system import (
    V3FullSystemOperator,
    apply_v3_full_system_operator,
    full_system_operator_from_namelist,
    residual_v3_full_system,
    rhs_v3_full_system,
    with_transport_rhs_settings,
)


@jtu.register_pytree_node_class
@dataclass(frozen=True)
class V3LinearSolveResult:
    """Result of a single matrix-free GMRES solve for the (currently supported) v3 full-system operator."""

    op: V3FullSystemOperator
    rhs: jnp.ndarray
    gmres: GMRESSolveResult

    def tree_flatten(self):
        children = (self.op, self.rhs, self.gmres)
        aux = None
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        del aux
        op, rhs, gmres_result = children
        return cls(op=op, rhs=rhs, gmres=gmres_result)

    @property
    def x(self) -> jnp.ndarray:
        return self.gmres.x

    @property
    def residual_norm(self) -> jnp.ndarray:
        return self.gmres.residual_norm


def solve_v3_full_system_linear_gmres(
    *,
    nml: Namelist,
    which_rhs: int | None = None,
    x0: jnp.ndarray | None = None,
    tol: float = 1e-10,
    atol: float = 0.0,
    restart: int = 80,
    maxiter: int | None = 400,
    solve_method: str = "batched",
    identity_shift: float = 0.0,
    phi1_hat_base: jnp.ndarray | None = None,
) -> V3LinearSolveResult:
    """Solve the current v3 full-system linear problem `A x = rhs` matrix-free using GMRES.

    Notes
    -----
    This helper currently targets the linear runs exercised in the parity fixtures
    (e.g. includePhi1InKineticEquation=false). For nonlinear runs, use `residual_v3_full_system`
    and an outer Newton-Krylov iteration (not yet shipped as a stable API).
    """
    op = full_system_operator_from_namelist(nml=nml, identity_shift=identity_shift, phi1_hat_base=phi1_hat_base)
    if int(op.rhs_mode) in {2, 3}:
        # v3 sets (dnHatdpsiHats, dTHatdpsiHats, EParallelHat) internally based on whichRHS.
        # If the input file omits gradients (common for monoenergetic runs), callers must select whichRHS.
        if which_rhs is None:
            which_rhs = 1
        op = with_transport_rhs_settings(op, which_rhs=int(which_rhs))
    rhs = rhs_v3_full_system(op)

    def mv(x):
        return apply_v3_full_system_operator(op, x)

    result = gmres_solve(
        matvec=mv,
        b=rhs,
        x0=x0,
        tol=tol,
        atol=atol,
        restart=restart,
        maxiter=maxiter,
        solve_method=solve_method,
    )
    return V3LinearSolveResult(op=op, rhs=rhs, gmres=result)


solve_v3_full_system_linear_gmres_jit = jax.jit(
    solve_v3_full_system_linear_gmres,
    static_argnames=("tol", "atol", "restart", "maxiter", "solve_method", "identity_shift"),
)


@jtu.register_pytree_node_class
@dataclass(frozen=True)
class V3NewtonKrylovResult:
    """Result of a simple Newton–Krylov solve for `residual_v3_full_system` (experimental)."""

    op: V3FullSystemOperator
    x: jnp.ndarray
    residual_norm: jnp.ndarray
    n_newton: int
    last_linear_residual_norm: jnp.ndarray

    def tree_flatten(self):
        children = (self.op, self.x, self.residual_norm, self.last_linear_residual_norm)
        aux = int(self.n_newton)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        op, x, residual_norm, last_linear_residual_norm = children
        return cls(op=op, x=x, residual_norm=residual_norm, n_newton=int(aux), last_linear_residual_norm=last_linear_residual_norm)


def solve_v3_full_system_newton_krylov(
    *,
    nml: Namelist,
    x0: jnp.ndarray | None = None,
    tol: float = 1e-10,
    max_newton: int = 12,
    gmres_tol: float = 1e-10,
    gmres_restart: int = 80,
    gmres_maxiter: int | None = 400,
    solve_method: str = "batched",
    identity_shift: float = 0.0,
) -> V3NewtonKrylovResult:
    """Solve `residual_v3_full_system(op, x) = 0` using a basic Newton–Krylov iteration.

    This is intended for small parity fixtures and developer experimentation. It is **not**
    yet a stable API for production runs.
    """
    op = full_system_operator_from_namelist(nml=nml, identity_shift=identity_shift)
    if x0 is None:
        x = jnp.zeros((op.total_size,), dtype=jnp.float64)
    else:
        x = jnp.asarray(x0, dtype=jnp.float64)
        if x.shape != (op.total_size,):
            raise ValueError(f"x0 must have shape {(op.total_size,)}, got {x.shape}")

    last_linear_resid = jnp.asarray(jnp.inf, dtype=jnp.float64)

    for k in range(int(max_newton)):
        r = residual_v3_full_system(op, x)
        rnorm = jnp.linalg.norm(r)
        if float(rnorm) < float(tol):
            return V3NewtonKrylovResult(
                op=op,
                x=x,
                residual_norm=rnorm,
                n_newton=k,
                last_linear_residual_norm=last_linear_resid,
            )

        def jvp(v):
            # J(x) v via forward-mode AD.
            return jax.jvp(lambda xx: residual_v3_full_system(op, xx), (x,), (v,))[1]

        # Solve J s = -r
        lin = gmres_solve(
            matvec=jvp,
            b=-r,
            tol=float(gmres_tol),
            restart=int(gmres_restart),
            maxiter=gmres_maxiter,
            solve_method=str(solve_method),
        )
        s = lin.x
        last_linear_resid = lin.residual_norm

        # Backtracking line search on ||r|| (very simple Armijo-style criterion).
        step = 1.0
        rnorm0 = float(rnorm)
        for _ in range(12):
            x_try = x + step * s
            r_try = residual_v3_full_system(op, x_try)
            rnorm_try = float(jnp.linalg.norm(r_try))
            if rnorm_try <= 0.9 * rnorm0:
                x = x_try
                break
            step *= 0.5
        else:
            # If we fail to reduce the residual, still take a small step to avoid stalling.
            x = x + (1.0 / 64.0) * s

    r = residual_v3_full_system(op, x)
    return V3NewtonKrylovResult(
        op=op,
        x=x,
        residual_norm=jnp.linalg.norm(r),
        n_newton=int(max_newton),
        last_linear_residual_norm=last_linear_resid,
    )


@dataclass(frozen=True)
class V3TransportMatrixSolveResult:
    """Result of assembling a RHSMode=2/3 transport matrix by looping `whichRHS` solves."""

    op0: V3FullSystemOperator
    transport_matrix: jnp.ndarray  # (N,N) in mathematical (row, col) order
    state_vectors_by_rhs: dict[int, jnp.ndarray]
    residual_norms_by_rhs: dict[int, jnp.ndarray]


def solve_v3_transport_matrix_linear_gmres(
    *,
    nml: Namelist,
    x0: jnp.ndarray | None = None,
    tol: float = 1e-10,
    atol: float = 0.0,
    restart: int = 80,
    maxiter: int | None = 400,
    solve_method: str = "batched",
    identity_shift: float = 0.0,
    phi1_hat_base: jnp.ndarray | None = None,
) -> V3TransportMatrixSolveResult:
    """Compute a RHSMode=2/3 transport matrix by running all `whichRHS` solves matrix-free in JAX.

    Notes
    -----
    This mirrors the v3 `solver.F90` RHSMode=2/3 path:
    - Loop `whichRHS`
    - Overwrite (dnHatdpsiHats, dTHatdpsiHats, EParallelHat)
    - Build the RHS via `evaluateResidual(f=0)`
    - Solve `A x = rhs`
    - Use `diagnostics.F90` formulas to fill `transportMatrix`
    """
    op0 = full_system_operator_from_namelist(nml=nml, identity_shift=identity_shift, phi1_hat_base=phi1_hat_base)
    rhs_mode = int(op0.rhs_mode)
    n = transport_matrix_size_from_rhs_mode(rhs_mode)

    # Geometry scalars needed for the transport-matrix formulas.
    grids = grids_from_namelist(nml)
    geom = geometry_from_namelist(nml=nml, grids=grids)

    def mv(x):
        return apply_v3_full_system_operator(op0, x)

    state_vectors: dict[int, jnp.ndarray] = {}
    residual_norms: dict[int, jnp.ndarray] = {}
    x_guess = x0

    for which_rhs in range(1, n + 1):
        op_rhs = with_transport_rhs_settings(op0, which_rhs=which_rhs)
        rhs = rhs_v3_full_system(op_rhs)
        res = gmres_solve(
            matvec=mv,
            b=rhs,
            x0=x_guess,
            tol=tol,
            atol=atol,
            restart=restart,
            maxiter=maxiter,
            solve_method=solve_method,
        )
        x_guess = res.x
        state_vectors[which_rhs] = res.x
        residual_norms[which_rhs] = res.residual_norm

    tm = v3_transport_matrix_from_state_vectors(op0=op0, geom=geom, state_vectors_by_rhs=state_vectors)
    return V3TransportMatrixSolveResult(
        op0=op0,
        transport_matrix=tm,
        state_vectors_by_rhs=state_vectors,
        residual_norms_by_rhs=residual_norms,
    )
