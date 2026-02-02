from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp

from .solver import GMRESSolveResult, gmres_solve


MatVec = Callable[[jnp.ndarray], jnp.ndarray]


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
    b = jnp.asarray(b, dtype=jnp.float64)

    def solve(mv: MatVec, rhs: jnp.ndarray) -> jnp.ndarray:
        return gmres_solve(
            matvec=mv,
            b=rhs,
            tol=tol,
            atol=atol,
            restart=restart,
            maxiter=maxiter,
            solve_method=solve_method,
        ).x

    def transpose_solve(mv_T: MatVec, rhs: jnp.ndarray) -> jnp.ndarray:
        return gmres_solve(
            matvec=mv_T,
            b=rhs,
            tol=tol,
            atol=atol,
            restart=restart,
            maxiter=maxiter,
            solve_method=solve_method,
        ).x

    # Return x with a custom transpose rule (implicit differentiation).
    x = jax.lax.custom_linear_solve(matvec, b, solve=solve, transpose_solve=transpose_solve, symmetric=False)

    # Re-run GMRES once to attach residual_norm for user diagnostics (not part of the AD rule).
    gmres = gmres_solve(
        matvec=matvec,
        b=b,
        tol=tol,
        atol=atol,
        restart=restart,
        maxiter=maxiter,
        solve_method=solve_method,
    )
    return ImplicitGMRESSolveResult(x=x, gmres=gmres)

