from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Callable

import jax
import jax.numpy as jnp
from jax import tree_util as jtu

from .solver import GMRESSolveResult, bicgstab_solve, bicgstab_solve_jit, gmres_solve, gmres_solve_jit


MatVec = Callable[[jnp.ndarray], jnp.ndarray]


@jtu.register_pytree_node_class
@dataclass(frozen=True)
class ImplicitGMRESSolveResult:
    """Result wrapper for implicit-diff GMRES solves.

    Notes
    -----
    The returned `x` is differentiable w.r.t. any JAX parameters used inside `matvec` and/or `b`,
    using `jax.lax.custom_linear_solve` (implicit differentiation). The `gmres` metadata is
    returned as a convenience for diagnostics, but is not intended as a stable API.
    """

    x: jnp.ndarray
    gmres: GMRESSolveResult

    def tree_flatten(self):
        children = (self.x, self.gmres)
        aux = None
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        del aux
        x, gmres = children
        return cls(x=x, gmres=gmres)


@jtu.register_pytree_node_class
@dataclass(frozen=True)
class ImplicitLinearSolveResult:
    """Result wrapper for implicit-diff linear solves (GMRES/BiCGStab/dense)."""

    x: jnp.ndarray
    residual_norm: jnp.ndarray

    def tree_flatten(self):
        children = (self.x, self.residual_norm)
        aux = None
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        del aux
        x, residual_norm = children
        return cls(x=x, residual_norm=residual_norm)


def gmres_custom_linear_solve(
    *,
    matvec: MatVec,
    b: jnp.ndarray,
    tol: float = 1e-10,
    atol: float = 0.0,
    restart: int = 80,
    maxiter: int | None = 400,
    solve_method: str = "batched",
) -> ImplicitGMRESSolveResult:
    """Solve `A x = b` with GMRES and define gradients via implicit differentiation.

    This is the recommended way to obtain gradients through a **linear** SFINCS solve without
    backpropagating through GMRES iterations.

    Implementation uses `jax.lax.custom_linear_solve`, which requires:
      - a forward solve for `A x = b`
      - a transpose solve for `A^T y = g` during reverse-mode differentiation

    Parameters
    ----------
    matvec:
      Function computing `A @ x` for the current operator.
    b:
      RHS vector.
    """
    result = linear_custom_solve(
        matvec=matvec,
        b=b,
        tol=tol,
        atol=atol,
        restart=restart,
        maxiter=maxiter,
        solve_method=solve_method,
        solver="gmres",
    )
    gmres = gmres_solve(
        matvec=matvec,
        b=jnp.asarray(b, dtype=jnp.float64),
        tol=tol,
        atol=atol,
        restart=restart,
        maxiter=maxiter,
        solve_method=solve_method,
    )
    return ImplicitGMRESSolveResult(x=result.x, gmres=gmres)


def _linear_custom_solve_core(
    *,
    matvec: MatVec,
    b: jnp.ndarray,
    tol: float,
    atol: float,
    restart: int,
    maxiter: int | None,
    solve_method: str,
    solver: str,
    preconditioner: MatVec | None,
    preconditioner_transpose: MatVec | None,
    x0: jnp.ndarray | None,
    precondition_side: str,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    b = jnp.asarray(b, dtype=jnp.float64)

    solver_kind = str(solver).lower()
    if solver_kind in {"auto", "default"}:
        solver_kind = "bicgstab"

    solver_jit_env = os.environ.get("SFINCS_JAX_SOLVER_JIT", "").strip().lower()
    use_solver_jit = solver_jit_env not in {"0", "false", "no", "off"}

    def _solve_direct(mv: MatVec, rhs: jnp.ndarray) -> GMRESSolveResult:
        if solver_kind in {"bicgstab", "bicgstab_jax"}:
            solver_fn = bicgstab_solve_jit if use_solver_jit else bicgstab_solve
            return solver_fn(
                matvec=mv,
                b=rhs,
                preconditioner=preconditioner,
                x0=x0,
                tol=tol,
                atol=atol,
                maxiter=maxiter,
                precondition_side=precondition_side,
            )
        solver_fn = gmres_solve_jit if use_solver_jit else gmres_solve
        return solver_fn(
            matvec=mv,
            b=rhs,
            preconditioner=preconditioner,
            x0=x0,
            tol=tol,
            atol=atol,
            restart=restart,
            maxiter=maxiter,
            solve_method=solve_method,
            precondition_side=precondition_side,
        )

    def solve(mv: MatVec, rhs: jnp.ndarray) -> jnp.ndarray:
        return _solve_direct(mv, rhs).x

    def transpose_solve(mv_T: MatVec, rhs: jnp.ndarray) -> jnp.ndarray:
        precond_T = preconditioner_transpose if preconditioner_transpose is not None else preconditioner
        if solver_kind in {"bicgstab", "bicgstab_jax"}:
            solver_fn = bicgstab_solve_jit if use_solver_jit else bicgstab_solve
            return solver_fn(
                matvec=mv_T,
                b=rhs,
                preconditioner=precond_T,
                x0=None,
                tol=tol,
                atol=atol,
                maxiter=maxiter,
                precondition_side=precondition_side,
            ).x
        solver_fn = gmres_solve_jit if use_solver_jit else gmres_solve
        return solver_fn(
            matvec=mv_T,
            b=rhs,
            preconditioner=precond_T,
            x0=None,
            tol=tol,
            atol=atol,
            restart=restart,
            maxiter=maxiter,
            solve_method=solve_method,
            precondition_side=precondition_side,
        ).x

    x = jax.lax.custom_linear_solve(matvec, b, solve=solve, transpose_solve=transpose_solve, symmetric=False)
    r = b - matvec(x)
    return x, r


def linear_custom_solve(
    *,
    matvec: MatVec,
    b: jnp.ndarray,
    tol: float = 1e-10,
    atol: float = 0.0,
    restart: int = 80,
    maxiter: int | None = 400,
    solve_method: str = "batched",
    solver: str = "auto",
    preconditioner: MatVec | None = None,
    preconditioner_transpose: MatVec | None = None,
    x0: jnp.ndarray | None = None,
    precondition_side: str = "left",
) -> ImplicitLinearSolveResult:
    """Implicit-diff linear solve wrapper using `jax.lax.custom_linear_solve`."""
    x, r = _linear_custom_solve_core(
        matvec=matvec,
        b=b,
        tol=tol,
        atol=atol,
        restart=restart,
        maxiter=maxiter,
        solve_method=solve_method,
        solver=solver,
        preconditioner=preconditioner,
        preconditioner_transpose=preconditioner_transpose,
        x0=x0,
        precondition_side=precondition_side,
    )
    return ImplicitLinearSolveResult(x=x, residual_norm=jnp.linalg.norm(r))


def linear_custom_solve_with_residual(
    *,
    matvec: MatVec,
    b: jnp.ndarray,
    tol: float = 1e-10,
    atol: float = 0.0,
    restart: int = 80,
    maxiter: int | None = 400,
    solve_method: str = "batched",
    solver: str = "auto",
    preconditioner: MatVec | None = None,
    preconditioner_transpose: MatVec | None = None,
    x0: jnp.ndarray | None = None,
    precondition_side: str = "left",
) -> tuple[ImplicitLinearSolveResult, jnp.ndarray]:
    """Implicit-diff linear solve that also returns the residual vector."""
    x, r = _linear_custom_solve_core(
        matvec=matvec,
        b=b,
        tol=tol,
        atol=atol,
        restart=restart,
        maxiter=maxiter,
        solve_method=solve_method,
        solver=solver,
        preconditioner=preconditioner,
        preconditioner_transpose=preconditioner_transpose,
        x0=x0,
        precondition_side=precondition_side,
    )
    return ImplicitLinearSolveResult(x=x, residual_norm=jnp.linalg.norm(r)), r
