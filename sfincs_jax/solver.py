from __future__ import annotations

from dataclasses import dataclass
import os

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


def assemble_dense_matrix_from_matvec(*, matvec, n: int, dtype: jnp.dtype) -> jnp.ndarray:
    """Assemble a dense matrix from a matrix-free `matvec`."""
    eye = jnp.eye(int(n), dtype=dtype)
    return vmap(matvec, in_axes=1, out_axes=1)(eye)


def dense_solve_from_matrix(*, a: jnp.ndarray, b: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Solve `A X = B` with v3-compatible singular handling.

    Parameters
    ----------
    a:
        Dense square matrix, shape `(N,N)`.
    b:
        Right-hand side, shape `(N,)` or `(N,K)`.
    """
    a = jnp.asarray(a)
    b = jnp.asarray(b)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"dense_solve_from_matrix expects a square matrix, got shape {a.shape}")
    if b.ndim not in (1, 2):
        raise ValueError(f"dense_solve_from_matrix expects b.ndim in {{1,2}}, got {b.ndim}")
    if b.shape[0] != a.shape[0]:
        raise ValueError(f"dense_solve_from_matrix shape mismatch: a={a.shape}, b={b.shape}")

    n = int(a.shape[0])
    b2 = b[:, None] if b.ndim == 1 else b
    eye = jnp.eye(n, dtype=a.dtype)

    x_direct = jnp.linalg.solve(a, b2)
    rank = jnp.linalg.matrix_rank(a)
    needs_reg = rank < jnp.asarray(n, dtype=rank.dtype)

    reg_val = 2.2e-10
    env_reg = os.environ.get("SFINCS_JAX_DENSE_REG", "").strip()
    if env_reg:
        reg_val = float(env_reg)
    reg = jnp.asarray(reg_val, dtype=a.dtype)

    x_reg = jnp.linalg.solve(a + reg * eye, b2)
    x_lstsq = jnp.linalg.lstsq(a, b2, rcond=None)[0]

    singular_mode = os.environ.get("SFINCS_JAX_DENSE_SINGULAR_MODE", "").strip().lower()
    if singular_mode == "lstsq":
        x2 = jnp.where(needs_reg, x_lstsq, x_direct)
    else:
        x2 = jnp.where(needs_reg, x_reg, x_direct)

    x2 = jnp.where(jnp.all(jnp.isfinite(x2)), x2, x_lstsq)
    r2 = b2 - a @ x2
    rn = jnp.linalg.norm(r2, axis=0)

    if b.ndim == 1:
        return x2[:, 0], rn[0]
    return x2, rn


def gmres_solve(
    *,
    matvec,
    b: jnp.ndarray,
    preconditioner=None,
    x0: jnp.ndarray | None = None,
    tol: float = 1e-10,
    atol: float = 0.0,
    restart: int = 50,
    maxiter: int | None = None,
    solve_method: str = "batched",
    precondition_side: str = "left",
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

        a = assemble_dense_matrix_from_matvec(matvec=matvec, n=n, dtype=b.dtype)
        x, residual_norm = dense_solve_from_matrix(a=a, b=b)
        return GMRESSolveResult(x=x, residual_norm=residual_norm)

    side = str(precondition_side).strip().lower()
    if side not in {"left", "right", "none"}:
        side = "left"

    if side == "right" and preconditioner is not None:
        # PETSc's GMRES defaults to right preconditioning: solve A P^{-1} y = b, x = P^{-1} y.
        # Here, `preconditioner` is expected to apply P^{-1}.
        def matvec_right(y):
            return matvec(preconditioner(y))

        y, _info = gmres(
            matvec_right,
            b,
            x0=None,
            tol=float(tol),
            atol=float(atol),
            restart=int(restart),
            maxiter=maxiter,
            M=None,
            solve_method=solve_method,
        )
        x = preconditioner(y)
    else:
        # Left preconditioning (SciPy-style): solve P^{-1} A x = P^{-1} b.
        M = preconditioner if side == "left" else None
        x, _info = gmres(
            matvec,
            b,
            x0=x0,
            tol=float(tol),
            atol=float(atol),
            restart=int(restart),
            maxiter=maxiter,
            M=M,
            solve_method=solve_method,
        )

    r = b - matvec(x)
    return GMRESSolveResult(x=x, residual_norm=jnp.linalg.norm(r))


gmres_solve_jit = jax.jit(
    gmres_solve,
    static_argnames=("matvec", "preconditioner", "restart", "maxiter", "solve_method", "precondition_side"),
)
