from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from sfincs_jax.v3_driver import _small_regularized_lstsq


def test_small_regularized_lstsq_matches_numpy_tall_system() -> None:
    a = jnp.asarray(
        [
            [2.0, -1.0, 0.5],
            [0.0, 3.0, 1.0],
            [1.0, 1.0, -2.0],
            [4.0, 0.5, 1.5],
            [-1.0, 2.0, 0.0],
        ],
        dtype=jnp.float64,
    )
    b = jnp.asarray([1.0, -2.0, 0.5, 3.0, -1.5], dtype=jnp.float64)

    coeff = np.asarray(_small_regularized_lstsq(a, b))
    coeff_ref, *_ = np.linalg.lstsq(np.asarray(a), np.asarray(b), rcond=None)

    assert np.allclose(coeff, coeff_ref, rtol=1e-9, atol=1e-9)


def test_small_regularized_lstsq_handles_near_rank_deficiency() -> None:
    a = jnp.asarray(
        [
            [1.0, 1.0],
            [2.0, 2.0 + 1e-10],
            [3.0, 3.0 - 1e-10],
            [4.0, 4.0 + 2e-10],
        ],
        dtype=jnp.float64,
    )
    b = jnp.asarray([1.0, 2.0, 3.0, 4.0], dtype=jnp.float64)

    coeff = np.asarray(_small_regularized_lstsq(a, b))
    residual = np.linalg.norm(np.asarray(a) @ coeff - np.asarray(b))

    assert np.all(np.isfinite(coeff))
    assert residual < 1e-8
