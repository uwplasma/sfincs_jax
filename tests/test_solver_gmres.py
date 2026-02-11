from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from sfincs_jax.solver import assemble_dense_matrix_from_matvec, dense_solve_from_matrix, gmres_solve


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


def test_dense_solve_from_matrix_supports_multiple_rhs() -> None:
    rng = np.random.default_rng(3)
    n = 18
    k = 3
    m = rng.normal(size=(n, n)).astype(np.float64)
    a = m.T @ m + 0.3 * np.eye(n)
    b = rng.normal(size=(n, k)).astype(np.float64)
    x_ref = np.linalg.solve(a, b)

    x, rn = dense_solve_from_matrix(a=jnp.asarray(a), b=jnp.asarray(b))
    np.testing.assert_allclose(np.asarray(x), x_ref, rtol=1e-10, atol=1e-10)
    assert np.asarray(rn).shape == (k,)
    assert float(np.max(np.asarray(rn))) < 1e-9


def test_assemble_dense_matrix_from_matvec_recovers_operator() -> None:
    rng = np.random.default_rng(7)
    n = 13
    a = rng.normal(size=(n, n)).astype(np.float64)
    a_j = jnp.asarray(a)

    def mv(x):
        return a_j @ x

    assembled = assemble_dense_matrix_from_matvec(matvec=mv, n=n, dtype=jnp.float64)
    np.testing.assert_allclose(np.asarray(assembled), a, rtol=0.0, atol=1e-12)
