from __future__ import annotations

from dataclasses import dataclass

from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
from jax import tree_util as jtu
from jax import vmap
from jax.scipy.sparse.linalg import gmres


@jtu.register_pytree_node_class
@dataclass(frozen=True)
class GMRESSolveResult:
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


def gmres_solve(
    *,
    matvec,
    b: jnp.ndarray,
    x0: jnp.ndarray | None = None,
    tol: float = 1e-10,
    atol: float = 0.0,
    restart: int = 50,
    maxiter: int | None = None,
    solve_method: str = "batched",
) -> GMRESSolveResult:
    """Solve `A x = b` using JAX's GMRES.

    Notes
    -----
    - `matvec` must be callable like `matvec(x)` and return the same shape as `x`.
    - JAX's `gmres` currently returns `info=None` (SciPy-style iteration info is planned).
    """
    b = jnp.asarray(b)
    if x0 is not None:
        x0 = jnp.asarray(x0)

    method = str(solve_method).lower()
    if method == "dense":
        n = int(b.size)
        if b.ndim != 1:
            raise ValueError(f"dense solve requires a 1D vector b, got shape {b.shape}")
        # Guardrail: dense assembly is quadratic memory/time.
        if n > 5000:
            raise ValueError(f"dense solve is disabled for n={n} (too large). Use GMRES.")

        eye = jnp.eye(n, dtype=b.dtype)
        # Assemble columns A[:,j] = matvec(e_j).
        a = vmap(matvec, in_axes=1, out_axes=1)(eye)
        x = jnp.linalg.solve(a, b)
        r = b - a @ x
        return GMRESSolveResult(x=x, residual_norm=jnp.linalg.norm(r))

    x, _info = gmres(
        matvec,
        b,
        x0=x0,
        tol=float(tol),
        atol=float(atol),
        restart=int(restart),
        maxiter=maxiter,
        solve_method=solve_method,
    )
    r = b - matvec(x)
    return GMRESSolveResult(x=x, residual_norm=jnp.linalg.norm(r))


gmres_solve_jit = jax.jit(gmres_solve, static_argnames=("matvec", "restart", "maxiter", "solve_method"))
