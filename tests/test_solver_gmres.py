from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest

from sfincs_jax.solver import (
    _distributed_solver_kind,
    assemble_dense_matrix_from_matvec,
    bicgstab_solve_with_residual,
    dense_krylov_solve_from_matrix,
    dense_solve_from_matrix,
    explicit_left_preconditioned_gmres_scipy,
    gmres_solve,
    gmres_solve_with_residual,
)


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


def test_dense_solve_from_matrix_regularizes_singular_system() -> None:
    a = np.array(
        [
            [2.0, -1.0, 0.0],
            [4.0, -2.0, 0.0],
            [0.0, 0.0, 3.0],
        ],
        dtype=np.float64,
    )
    b = np.array([1.0, 2.0, -3.0], dtype=np.float64)

    x, rn = dense_solve_from_matrix(a=jnp.asarray(a), b=jnp.asarray(b))
    x_np = np.asarray(x)

    assert np.all(np.isfinite(x_np))
    np.testing.assert_allclose(a @ x_np, b, rtol=1e-8, atol=1e-8)
    assert float(rn) < 1e-8


def test_dense_krylov_solve_from_matrix_matches_numpy() -> None:
    rng = np.random.default_rng(5)
    n = 20
    m = rng.normal(size=(n, n)).astype(np.float64)
    a = m.T @ m + 0.25 * np.eye(n)
    b = rng.normal(size=(n,)).astype(np.float64)
    x_ref = np.linalg.solve(a, b)

    x, rn = dense_krylov_solve_from_matrix(
        a=jnp.asarray(a),
        b=jnp.asarray(b),
        tol=1e-12,
        restart=n,
        maxiter=8,
        solve_method="incremental",
    )

    np.testing.assert_allclose(np.asarray(x), x_ref, rtol=1e-8, atol=1e-8)
    assert float(rn) < 1e-8


def test_dense_krylov_row_scaled_handles_diagonal_imbalance() -> None:
    diag = np.array([1.0e-8, 2.0, 5.0e4, 7.0], dtype=np.float64)
    a = np.diag(diag)
    a[0, 1] = -3.0e-7
    a[1, 0] = 2.0e-7
    b = np.array([2.0e-8, -4.0, 1.5e5, 3.5], dtype=np.float64)
    x_ref = np.linalg.solve(a, b)

    x, rn = dense_krylov_solve_from_matrix(
        a=jnp.asarray(a),
        b=jnp.asarray(b),
        tol=1e-12,
        restart=a.shape[0],
        maxiter=4,
        solve_method="incremental",
        row_scaled=True,
    )

    np.testing.assert_allclose(np.asarray(x), x_ref, rtol=1e-7, atol=1e-9)
    assert float(rn) < 1e-7


def test_assemble_dense_matrix_from_matvec_recovers_operator() -> None:
    rng = np.random.default_rng(7)
    n = 13
    a = rng.normal(size=(n, n)).astype(np.float64)
    a_j = jnp.asarray(a)

    def mv(x):
        return a_j @ x

    assembled = assemble_dense_matrix_from_matvec(matvec=mv, n=n, dtype=jnp.float64)
    np.testing.assert_allclose(np.asarray(assembled), a, rtol=0.0, atol=1e-12)


def test_gmres_solve_with_residual_matches_matvec() -> None:
    rng = np.random.default_rng(11)
    n = 16
    m = rng.normal(size=(n, n)).astype(np.float64)
    a = m.T @ m + 0.4 * np.eye(n)
    b = rng.normal(size=(n,)).astype(np.float64)

    a_j = jnp.asarray(a)
    b_j = jnp.asarray(b)

    def mv(x):
        return a_j @ x

    result, residual = gmres_solve_with_residual(matvec=mv, b=b_j, tol=1e-12, restart=30, maxiter=200)
    r_expected = b_j - mv(result.x)

    np.testing.assert_allclose(np.asarray(residual), np.asarray(r_expected), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(
        float(result.residual_norm),
        float(jnp.linalg.norm(residual)),
        rtol=1e-12,
        atol=1e-12,
    )


def test_bicgstab_solve_with_residual_matches_matvec() -> None:
    rng = np.random.default_rng(23)
    n = 14
    m = rng.normal(size=(n, n)).astype(np.float64)
    a = m.T @ m + 0.6 * np.eye(n)
    b = rng.normal(size=(n,)).astype(np.float64)

    a_j = jnp.asarray(a)
    b_j = jnp.asarray(b)

    def mv(x):
        return a_j @ x

    result, residual = bicgstab_solve_with_residual(matvec=mv, b=b_j, tol=1e-12, maxiter=400)
    r_expected = b_j - mv(result.x)

    np.testing.assert_allclose(np.asarray(residual), np.asarray(r_expected), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(
        float(result.residual_norm),
        float(jnp.linalg.norm(residual)),
        rtol=1e-12,
        atol=1e-12,
    )


def test_explicit_left_preconditioned_gmres_scipy_matches_numpy() -> None:
    a = np.array(
        [
            [4.0, 1.0, 0.0],
            [1.0, 3.0, -1.0],
            [0.0, -1.0, 2.0],
        ],
        dtype=np.float64,
    )
    b = np.array([1.0, -2.0, 3.0], dtype=np.float64)
    x_ref = np.linalg.solve(a, b)
    a_j = jnp.asarray(a)

    def mv(x):
        return a_j @ x

    inv_diag = 1.0 / np.diag(a)

    def precond(x):
        return jnp.asarray(inv_diag, dtype=jnp.float64) * x

    x, rn_true, rn_pc, history = explicit_left_preconditioned_gmres_scipy(
        matvec=mv,
        b=jnp.asarray(b),
        preconditioner=precond,
        tol=1e-12,
        restart=6,
        maxiter=20,
    )

    np.testing.assert_allclose(x, x_ref, rtol=1e-10, atol=1e-10)
    assert rn_true < 1e-10
    assert rn_pc < 1e-10
    assert history


def test_explicit_left_preconditioned_gmres_scipy_zero_preconditioned_rhs_is_finite() -> None:
    a = np.eye(3, dtype=np.float64)
    b = np.array([1.0, -2.0, 3.0], dtype=np.float64)
    a_j = jnp.asarray(a)

    def mv(x):
        return a_j @ x

    def zero_precond(x):
        return jnp.zeros_like(x)

    x, rn_true, rn_pc, history = explicit_left_preconditioned_gmres_scipy(
        matvec=mv,
        b=jnp.asarray(b),
        preconditioner=zero_precond,
        tol=1e-12,
        restart=4,
        maxiter=4,
    )

    np.testing.assert_allclose(x, np.zeros_like(b), rtol=0.0, atol=0.0)
    assert rn_true == pytest.approx(np.linalg.norm(b))
    assert rn_pc == 0.0
    assert history == [0.0]


def test_distributed_solver_kind_auto_defaults_to_bicgstab(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_DISTRIBUTED_KRYLOV", raising=False)
    kind, method = _distributed_solver_kind("auto")
    assert kind == "bicgstab"
    assert method == "batched"


def test_distributed_solver_kind_auto_can_force_gmres(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_DISTRIBUTED_KRYLOV", "gmres")
    kind, method = _distributed_solver_kind("auto")
    assert kind == "gmres"
    assert method == "incremental"
