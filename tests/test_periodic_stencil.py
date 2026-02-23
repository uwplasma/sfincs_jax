from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from sfincs_jax.grids import uniform_diff_matrices
from sfincs_jax.periodic_stencil import (
    apply_periodic_stencil_roll,
    apply_sparse_row_stencil_gather,
    extract_sparse_circulant_stencil,
    extract_sparse_row_stencil,
)


def test_extract_sparse_circulant_stencil_scheme10_matches_dense() -> None:
    _, _, ddx, _ = uniform_diff_matrices(n=17, x_min=0.0, x_max=2.0 * np.pi, scheme=10)
    shifts, coeffs = extract_sparse_circulant_stencil(np.asarray(ddx))
    assert shifts
    assert coeffs
    x = np.linspace(0.0, 1.0, 17, dtype=np.float64)
    y_dense = np.asarray(ddx) @ x
    y_stencil = np.zeros_like(x)
    for shift, coeff in zip(shifts, coeffs):
        y_stencil += coeff * np.roll(x, shift)
    np.testing.assert_allclose(y_stencil, y_dense, rtol=0.0, atol=1e-13)


def test_extract_sparse_circulant_stencil_nonperiodic_returns_empty() -> None:
    _, _, ddx, _ = uniform_diff_matrices(n=17, x_min=0.0, x_max=1.0, scheme=12)
    shifts, coeffs = extract_sparse_circulant_stencil(np.asarray(ddx))
    assert shifts == ()
    assert coeffs == ()


def test_apply_periodic_stencil_roll_matches_dense_theta_einsum() -> None:
    _, _, ddx, _ = uniform_diff_matrices(n=19, x_min=0.0, x_max=2.0 * np.pi, scheme=10)
    shifts, coeffs = extract_sparse_circulant_stencil(np.asarray(ddx))
    rng = np.random.default_rng(3)
    f = jnp.asarray(rng.normal(size=(2, 3, 4, 19, 5)), dtype=jnp.float64)
    y_dense = jnp.einsum("ij,sxljz->sxliz", ddx, f)
    y_stencil = apply_periodic_stencil_roll(f, shifts=shifts, coeffs=coeffs, axis=3)
    np.testing.assert_allclose(np.asarray(y_stencil), np.asarray(y_dense), rtol=0.0, atol=1e-12)


def test_apply_sparse_row_stencil_matches_dense_theta_einsum() -> None:
    _, _, ddx, _ = uniform_diff_matrices(n=19, x_min=0.0, x_max=1.0, scheme=2)
    cols, vals = extract_sparse_row_stencil(np.asarray(ddx), max_row_nnz=5)
    assert cols.shape == vals.shape
    rng = np.random.default_rng(7)
    f = jnp.asarray(rng.normal(size=(2, 3, 4, 19, 5)), dtype=jnp.float64)
    y_dense = jnp.einsum("ij,sxljz->sxliz", ddx, f)
    y_sparse = apply_sparse_row_stencil_gather(
        f,
        cols=jnp.asarray(cols, dtype=jnp.int32),
        vals=jnp.asarray(vals, dtype=jnp.float64),
        axis=3,
    )
    np.testing.assert_allclose(np.asarray(y_sparse), np.asarray(y_dense), rtol=0.0, atol=1e-12)
