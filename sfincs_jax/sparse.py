from __future__ import annotations

from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp


def csr_matvec(
    *,
    data: jnp.ndarray,  # (nnz,)
    indices: jnp.ndarray,  # (nnz,)
    indptr: jnp.ndarray,  # (n_rows+1,)
    x: jnp.ndarray,  # (n_cols,)
    n_rows: int | None = None,
) -> jnp.ndarray:
    """Sparse CSR matrix-vector product.

    Parameters
    ----------
    data, indices, indptr:
      CSR arrays, as in SciPy / PETSc AIJ.
    x:
      Dense vector.
    n_rows:
      Optional override for the number of rows (otherwise inferred from `indptr`).
    """
    data = jnp.asarray(data)
    indices = jnp.asarray(indices)
    indptr = jnp.asarray(indptr)
    x = jnp.asarray(x)

    if indptr.ndim != 1:
        raise ValueError("indptr must be 1D")
    if indices.ndim != 1 or data.ndim != 1:
        raise ValueError("data and indices must be 1D")

    if n_rows is None:
        n_rows = int(indptr.shape[0] - 1)
    if int(indptr.shape[0]) != n_rows + 1:
        raise ValueError("indptr has incompatible length")

    counts = indptr[1:] - indptr[:-1]
    # When JIT compiling, `jnp.repeat` requires the total output length to be static.
    # For CSR, this length is exactly nnz = len(data).
    nnz = int(data.shape[0])
    row_ids = jnp.repeat(
        jnp.arange(n_rows, dtype=indices.dtype),
        counts,
        total_repeat_length=nnz,
    )
    y_vals = data * x[indices]
    return jax.ops.segment_sum(y_vals, row_ids, n_rows)
