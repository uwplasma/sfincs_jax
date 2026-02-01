from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from sfincs_jax.solver import gmres_solve


def test_gmres_solve_matches_numpy_for_spd_matrix() -> None:
    rng = np.random.default_rng(0)
    n = 24
    m = rng.normal(size=(n, n)).astype(np.float64)
    a = m.T @ m + 0.5 * np.eye(n)  # SPD, well-conditioned enough.
    b = rng.normal(size=(n,)).astype(np.float64)
    x_ref = np.linalg.solve(a, b)

    a_j = jnp.asarray(a)
    b_j = jnp.asarray(b)

    def mv(x):
        return a_j @ x

    result = gmres_solve(matvec=mv, b=b_j, tol=1e-12, restart=30, maxiter=200)
    x = np.asarray(result.x)

    np.testing.assert_allclose(x, x_ref, rtol=1e-8, atol=1e-8)
    assert float(result.residual_norm) < 1e-8

