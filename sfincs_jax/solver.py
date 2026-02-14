from __future__ import annotations

from dataclasses import dataclass
import os
import numpy as np

from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
from jax import tree_util as jtu
from jax import vmap
from jax.scipy.sparse.linalg import bicgstab, gmres
from scipy.sparse.linalg import LinearOperator as _LinearOperator
from scipy.sparse.linalg import gmres as _scipy_gmres


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


def _maybe_limit_restart(n: int, restart: int, dtype: jnp.dtype) -> int:
    if n <= 0 or restart <= 1:
        return restart
    auto_env = os.environ.get("SFINCS_JAX_GMRES_AUTO_RESTART", "").strip().lower()
    if auto_env in {"0", "false", "no", "off"}:
        return restart
    max_mb_env = os.environ.get("SFINCS_JAX_GMRES_MAX_MB", "").strip()
    if max_mb_env:
        try:
            max_mb = float(max_mb_env)
        except ValueError:
            max_mb = 2048.0
    else:
        max_mb = 2048.0
    if max_mb <= 0:
        return restart
    bytes_per_elem = int(np.dtype(dtype).itemsize)
    if bytes_per_elem <= 0:
        return restart
    max_bytes = max_mb * 1e6
    # Estimate Krylov basis storage ~ (restart+1) * n * bytes_per_elem.
    max_restart = int(max_bytes // (bytes_per_elem * n)) - 1
    if max_restart < 1:
        max_restart = 1
    return min(int(restart), int(max_restart))


def gmres_solve_with_history_scipy(
    *,
    matvec,
    b: jnp.ndarray,
    preconditioner=None,
    x0: jnp.ndarray | None = None,
    tol: float = 1e-10,
    atol: float = 0.0,
    restart: int = 50,
    maxiter: int | None = None,
    precondition_side: str = "left",
) -> tuple[np.ndarray, float, list[float]]:
    """Run SciPy GMRES to collect residual history for Fortran-style logging."""
    b_np = np.asarray(b, dtype=np.float64).reshape((-1,))
    n = int(b_np.size)
    x0_np = np.asarray(x0, dtype=np.float64).reshape((-1,)) if x0 is not None else None
    restart_use = _maybe_limit_restart(n, int(restart), np.dtype(np.float64))

    def _mv(x_np: np.ndarray) -> np.ndarray:
        return np.asarray(matvec(jnp.asarray(x_np, dtype=jnp.float64)), dtype=np.float64)

    def _prec(x_np: np.ndarray) -> np.ndarray:
        if preconditioner is None:
            return x_np
        return np.asarray(preconditioner(jnp.asarray(x_np, dtype=jnp.float64)), dtype=np.float64)

    side = str(precondition_side).strip().lower()
    if side not in {"left", "right", "none"}:
        side = "left"

    if side == "right" and preconditioner is not None:
        def _mv_right(y_np: np.ndarray) -> np.ndarray:
            return _mv(_prec(y_np))

        A = _LinearOperator((n, n), matvec=_mv_right, dtype=np.float64)
        M = None
    else:
        A = _LinearOperator((n, n), matvec=_mv, dtype=np.float64)
        M = _LinearOperator((n, n), matvec=_prec, dtype=np.float64) if preconditioner is not None else None

    history: list[float] = []

    def _cb(arg):
        # SciPy passes residual norm when callback_type='pr_norm'.
        if np.isscalar(arg):
            history.append(float(arg))
        else:
            history.append(float(np.linalg.norm(arg)))

    x_np, info = _scipy_gmres(
        A,
        b_np,
        x0=x0_np,
        rtol=float(tol),
        atol=float(atol),
        restart=int(restart_use),
        maxiter=int(maxiter) if maxiter is not None else None,
        M=M,
        callback=_cb,
        callback_type="pr_norm",
    )

    if side == "right" and preconditioner is not None:
        x_np = _prec(x_np)

    res = b_np - _mv(x_np)
    rn = float(np.linalg.norm(res))
    return x_np, rn, history


def bicgstab_solve(
    *,
    matvec,
    b: jnp.ndarray,
    preconditioner=None,
    x0: jnp.ndarray | None = None,
    tol: float = 1e-10,
    atol: float = 0.0,
    maxiter: int | None = None,
    precondition_side: str = "left",
) -> GMRESSolveResult:
    """Solve `A x = b` using JAX's BiCGStab (short-recurrence Krylov, O(n) memory)."""
    b = jnp.asarray(b)
    if x0 is not None:
        x0 = jnp.asarray(x0)

    side = str(precondition_side).strip().lower()
    if side not in {"left", "right", "none"}:
        side = "left"

    if side == "right" and preconditioner is not None:
        def matvec_right(y):
            return matvec(preconditioner(y))

        y, _info = bicgstab(
            matvec_right,
            b,
            x0=None,
            tol=float(tol),
            atol=float(atol),
            maxiter=maxiter,
            M=None,
        )
        x = preconditioner(y)
    else:
        M = preconditioner if side == "left" else None
        x, _info = bicgstab(
            matvec,
            b,
            x0=x0,
            tol=float(tol),
            atol=float(atol),
            maxiter=maxiter,
            M=M,
        )

    r = b - matvec(x)
    return GMRESSolveResult(x=x, residual_norm=jnp.linalg.norm(r))


bicgstab_solve_jit = jax.jit(
    bicgstab_solve,
    static_argnames=("matvec", "preconditioner", "tol", "atol", "maxiter", "precondition_side"),
)


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
    direct_finite = jnp.all(jnp.isfinite(x_direct))

    reg_val = 2.2e-10
    env_reg = os.environ.get("SFINCS_JAX_DENSE_REG", "").strip()
    if env_reg:
        reg_val = float(env_reg)
    reg = jnp.asarray(reg_val, dtype=a.dtype)

    singular_mode = os.environ.get("SFINCS_JAX_DENSE_SINGULAR_MODE", "").strip().lower()
    force_reg = os.environ.get("SFINCS_JAX_DENSE_FORCE_REG", "").strip().lower() in {"1", "true", "yes", "on"}
    force_lstsq = singular_mode == "lstsq"

    def _solve_lstsq(_):
        return jnp.linalg.lstsq(a, b2, rcond=None)[0]

    if force_reg:
        x2 = jnp.linalg.solve(a + reg * eye, b2)
    elif force_lstsq:
        x2 = _solve_lstsq(None)
    else:
        x2 = jax.lax.cond(direct_finite, lambda _: x_direct, _solve_lstsq, operand=None)

    x2 = jax.lax.cond(jnp.all(jnp.isfinite(x2)), lambda _: x2, _solve_lstsq, operand=None)
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
    if method in {"auto", "default"}:
        method = "bicgstab"
    if method in {"bicgstab", "bicgstab_jax"}:
        res = bicgstab_solve(
            matvec=matvec,
            b=b,
            preconditioner=preconditioner,
            x0=x0,
            tol=tol,
            atol=atol,
            maxiter=maxiter,
            precondition_side=precondition_side,
        )
        target = max(float(atol), float(tol) * float(jnp.linalg.norm(b)))
        if not bool(jnp.isfinite(res.residual_norm)) or float(res.residual_norm) > target:
            # Fallback to GMRES when BiCGStab stagnates or returns non-finite residuals.
            return gmres_solve(
                matvec=matvec,
                b=b,
                preconditioner=preconditioner,
                x0=x0,
                tol=tol,
                atol=atol,
                restart=restart,
                maxiter=maxiter,
                solve_method="incremental",
                precondition_side=precondition_side,
            )
        return res
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

    restart_use = _maybe_limit_restart(int(b.size), int(restart), b.dtype)

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
            restart=int(restart_use),
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
            restart=int(restart_use),
            maxiter=maxiter,
            M=M,
            solve_method=solve_method,
        )

    r = b - matvec(x)
    return GMRESSolveResult(x=x, residual_norm=jnp.linalg.norm(r))


gmres_solve_jit = jax.jit(
    gmres_solve,
    static_argnames=("matvec", "preconditioner", "tol", "atol", "restart", "maxiter", "solve_method", "precondition_side"),
)
