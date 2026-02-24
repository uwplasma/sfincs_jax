from __future__ import annotations

import os

import jax.numpy as jnp
import numpy as np


def _sharding_active_hint() -> bool:
    shard_axis = os.environ.get("SFINCS_JAX_MATVEC_SHARD_AXIS", "").strip().lower()
    if shard_axis in {"theta", "zeta", "x", "flat", "vector", "p"}:
        return True
    if shard_axis in {"off", "none", "0", "false", "no"}:
        return False
    gmres_dist = os.environ.get("SFINCS_JAX_GMRES_DISTRIBUTED", "").strip().lower()
    if gmres_dist in {"1", "true", "yes", "on", "auto", "theta", "zeta"}:
        return True
    return False


def periodic_stencil_runtime_enabled() -> bool:
    env = os.environ.get("SFINCS_JAX_PERIODIC_STENCIL", "").strip().lower()
    if env in {"0", "false", "no", "off"}:
        return False
    on_sharded = os.environ.get("SFINCS_JAX_PERIODIC_STENCIL_ON_SHARDED", "").strip().lower()
    try:
        import jax  # noqa: PLC0415

        n_local = int(jax.local_device_count())
    except Exception:
        n_local = 1
    if n_local > 1:
        if on_sharded in {"0", "false", "no", "off"}:
            return False
        if on_sharded in {"1", "true", "yes", "on"}:
            return True
        # Auto mode: enable on sharded runs so derivatives use local stencil/halo
        # kernels instead of dense contractions.
        if not _sharding_active_hint():
            return False
    return True


def _periodic_stencil_atol() -> float:
    env = os.environ.get("SFINCS_JAX_PERIODIC_STENCIL_ATOL", "").strip()
    if not env:
        return 1e-14
    try:
        return float(env)
    except ValueError:
        return 1e-14


def _periodic_stencil_max_nnz() -> int:
    env = os.environ.get("SFINCS_JAX_PERIODIC_STENCIL_MAX_NNZ", "").strip()
    if not env:
        return 9
    try:
        return max(1, int(env))
    except ValueError:
        return 9


def _sparse_row_stencil_max_nnz() -> int:
    env = os.environ.get("SFINCS_JAX_DERIV_SPARSE_MAX_ROW_NNZ", "").strip()
    if not env:
        return 9
    try:
        return max(1, int(env))
    except ValueError:
        return 9


def extract_sparse_circulant_stencil(
    matrix: np.ndarray,
    *,
    atol: float | None = None,
    max_nnz: int | None = None,
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """Extract a compact circular-stencil representation for a periodic derivative matrix.

    Returns `(shifts, coeffs)` such that:
    `matrix @ x == sum(coeffs[k] * roll(x, shifts[k]))` for all periodic vectors `x`.
    If extraction is not possible (non-circulant or too dense), returns empty tuples.
    """
    if not periodic_stencil_runtime_enabled():
        return (), ()
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        return (), ()
    n = int(arr.shape[0])
    if n <= 1:
        return (), ()
    atol_use = _periodic_stencil_atol() if atol is None else float(atol)
    max_nnz_use = _periodic_stencil_max_nnz() if max_nnz is None else int(max_nnz)
    row0 = np.asarray(arr[0], dtype=np.float64)
    nz = np.flatnonzero(np.abs(row0) > atol_use)
    if nz.size == 0 or nz.size > max_nnz_use:
        return (), ()
    # Circulant check: each row is a right-roll of the first row.
    for i in range(n):
        if not np.allclose(arr[i], np.roll(row0, i), atol=atol_use, rtol=0.0):
            return (), ()
    shifts = tuple(int(-int(j)) for j in nz.tolist())
    coeffs = tuple(float(row0[j]) for j in nz.tolist())
    return shifts, coeffs


def extract_sparse_row_stencil(
    matrix: np.ndarray,
    *,
    atol: float | None = None,
    max_row_nnz: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract padded sparse rows `(cols, vals)` for dense->sparse derivative application."""
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        return np.zeros((0, 0), dtype=np.int32), np.zeros((0, 0), dtype=np.float64)
    n = int(arr.shape[0])
    if n <= 1:
        return np.zeros((0, 0), dtype=np.int32), np.zeros((0, 0), dtype=np.float64)
    atol_use = _periodic_stencil_atol() if atol is None else float(atol)
    max_row_nnz_use = _sparse_row_stencil_max_nnz() if max_row_nnz is None else int(max_row_nnz)
    row_nnz = np.sum(np.abs(arr) > atol_use, axis=1)
    if row_nnz.size == 0:
        return np.zeros((0, 0), dtype=np.int32), np.zeros((0, 0), dtype=np.float64)
    kmax = int(np.max(row_nnz))
    if kmax <= 0 or kmax > max_row_nnz_use:
        return np.zeros((0, 0), dtype=np.int32), np.zeros((0, 0), dtype=np.float64)
    cols = np.zeros((n, kmax), dtype=np.int32)
    vals = np.zeros((n, kmax), dtype=np.float64)
    for i in range(n):
        idx = np.flatnonzero(np.abs(arr[i]) > atol_use)
        if idx.size == 0:
            continue
        cols[i, : idx.size] = idx.astype(np.int32, copy=False)
        vals[i, : idx.size] = arr[i, idx]
    return cols, vals


def apply_periodic_stencil_roll(
    f: jnp.ndarray,
    *,
    shifts: tuple[int, ...],
    coeffs: tuple[float, ...],
    axis: int,
) -> jnp.ndarray:
    """Apply a periodic derivative stencil using `jnp.roll` on the given axis."""
    out = jnp.zeros_like(f)
    for shift, coeff in zip(shifts, coeffs):
        if coeff != 0.0:
            out = out + jnp.asarray(coeff, dtype=f.dtype) * jnp.roll(f, shift=int(shift), axis=axis)
    return out


def apply_sparse_row_stencil_gather(
    f: jnp.ndarray,
    *,
    cols: jnp.ndarray,
    vals: jnp.ndarray,
    axis: int,
) -> jnp.ndarray:
    """Apply row-sparse stencil via gather on a specific axis."""
    f_take = jnp.take(f, cols, axis=axis)
    reshape = [1] * f_take.ndim
    reshape[axis] = int(vals.shape[0])
    reshape[axis + 1] = int(vals.shape[1])
    weighted = f_take * jnp.asarray(vals, dtype=f.dtype).reshape(tuple(reshape))
    return jnp.sum(weighted, axis=axis + 1)
