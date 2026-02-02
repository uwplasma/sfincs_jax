from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from sfincs_jax.implicit_solve import gmres_custom_linear_solve


def test_custom_linear_solve_grad_matches_finite_difference() -> None:
    # A small, well-conditioned linear system A(p) x = b with A(p) = A0 + p*I.
    a0 = jnp.asarray(
        [
            [4.0, 1.0, 0.0, 0.0],
            [1.0, 3.0, 1.0, 0.0],
            [0.0, 1.0, 2.5, 1.0],
            [0.0, 0.0, 1.0, 2.0],
        ],
        dtype=jnp.float64,
    )
    b = jnp.asarray([1.0, 2.0, -1.0, 0.5], dtype=jnp.float64)

    def objective(p: jnp.ndarray) -> jnp.ndarray:
        p = jnp.asarray(p, dtype=jnp.float64)
        a = a0 + p * jnp.eye(4, dtype=jnp.float64)

        def mv(x: jnp.ndarray) -> jnp.ndarray:
            return a @ x

        x = gmres_custom_linear_solve(matvec=mv, b=b, tol=1e-12, restart=20, maxiter=50).x
        return 0.5 * jnp.vdot(x, x)

    p0 = jnp.asarray(0.2, dtype=jnp.float64)
    g = float(jax.grad(objective)(p0))

    eps = 1e-6
    fd = (float(objective(p0 + eps)) - float(objective(p0 - eps))) / (2.0 * eps)

    assert np.isfinite(g)
    assert abs(g - fd) < 5e-6

