from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from sfincs_jax.sparse import csr_matvec


def test_csr_matvec_matches_dense() -> None:
    rng = np.random.default_rng(0)
    n = 6
    # Deterministic sparsity pattern.
    dense = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        dense[i, i] = 2.0 + 0.1 * i
        if i + 1 < n:
            dense[i, i + 1] = -0.3
        if i - 2 >= 0:
            dense[i, i - 2] = 0.2

    # Build CSR.
    data = []
    indices = []
    indptr = [0]
    for i in range(n):
        cols = np.nonzero(dense[i])[0]
        data.extend(dense[i, cols].tolist())
        indices.extend(cols.tolist())
        indptr.append(len(data))

    data = np.asarray(data, dtype=np.float64)
    indices = np.asarray(indices, dtype=np.int32)
    indptr = np.asarray(indptr, dtype=np.int32)

    x = rng.normal(size=(n,)).astype(np.float64)
    y_ref = dense @ x
    y = np.asarray(
        csr_matvec(
            data=jnp.asarray(data),
            indices=jnp.asarray(indices),
            indptr=jnp.asarray(indptr),
            x=jnp.asarray(x),
            n_rows=n,
        )
    )
    np.testing.assert_allclose(y, y_ref, rtol=0, atol=1e-13)

