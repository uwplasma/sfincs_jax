from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib

from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)

from collections.abc import Callable, Sequence
import os
import concurrent.futures
from pathlib import Path
import numpy as np

import jax
import jax.numpy as jnp
from jax import tree_util as jtu

from .namelist import Namelist, read_sfincs_input
from .solver import (
    GMRESSolveResult,
    assemble_dense_matrix_from_matvec,
    bicgstab_solve_with_residual,
    bicgstab_solve_with_residual_jit,
    bicgstab_solve_with_history_scipy,
    dense_solve_from_matrix,
    dense_solve_from_matrix_row_scaled,
    gmres_solve,
    gmres_solve_jit,
    gmres_solve_with_residual,
    gmres_solve_with_residual_jit,
    gmres_solve_with_history_scipy,
)
from .implicit_solve import linear_custom_solve, linear_custom_solve_with_residual
from .transport_matrix import (
    _flux_functions_from_op,
    transport_matrix_size_from_rhs_mode,
    v3_rhsmode1_output_fields_vm_only_jit,
    v3_transport_diagnostics_vm_only,
    v3_transport_diagnostics_vm_only_precompute,
    v3_transport_diagnostics_vm_only_batch_jit,
    v3_transport_diagnostics_vm_only_batch_remat_jit,
    v3_transport_diagnostics_vm_only_batch_op0_jit,
    v3_transport_diagnostics_vm_only_batch_op0_remat_jit,
    v3_transport_diagnostics_vm_only_batch_op0_precomputed_jit,
    v3_transport_diagnostics_vm_only_batch_op0_precomputed_remat_jit,
    v3_transport_matrix_from_flux_arrays,
)
from .v3_system import _fs_average_factor, _ix_min, _source_basis_constraint_scheme_1
from .verbose import Timer
from .v3 import geometry_from_namelist, grids_from_namelist
from .v3_system import (
    V3FullSystemOperator,
    _operator_signature_cached,
    apply_v3_full_system_jacobian,
    apply_v3_full_system_jacobian_jit,
    apply_v3_full_system_operator_cached,
    full_system_operator_from_namelist,
    residual_v3_full_system,
    rhs_v3_full_system,
    rhs_v3_full_system_jit,
    with_transport_rhs_settings,
)
from .profiling import maybe_profiler


def _use_solver_jit(size_hint: int | None = None) -> bool:
    env = os.environ.get("SFINCS_JAX_SOLVER_JIT", "").strip().lower()
    if env in {"1", "true", "yes", "on"}:
        return True
    if env in {"0", "false", "no", "off"}:
        return False
    if size_hint is None:
        size_hint = _PRECOND_SIZE_HINT or 0
    thresh_env = os.environ.get("SFINCS_JAX_SOLVER_JIT_MAX_SIZE", "").strip()
    try:
        thresh = int(thresh_env) if thresh_env else 20000
    except ValueError:
        thresh = 20000
    return int(size_hint) <= thresh


_PRECOND_SIZE_HINT: int | None = None


def _set_precond_size_hint(n: int | None) -> None:
    global _PRECOND_SIZE_HINT
    if n is None:
        _PRECOND_SIZE_HINT = None
    else:
        _PRECOND_SIZE_HINT = int(n)


def _precond_dtype(size_hint: int | None = None) -> jnp.dtype:
    env = os.environ.get("SFINCS_JAX_PRECOND_DTYPE", "").strip().lower()
    if env in {"", "auto", "mixed"}:
        if size_hint is None:
            size_hint = _PRECOND_SIZE_HINT or 0
            thresh_env = os.environ.get("SFINCS_JAX_PRECOND_FP32_MIN_SIZE", "").strip()
            thresh_default = 20000
        else:
            thresh_env = os.environ.get("SFINCS_JAX_PRECOND_FP32_MIN_BLOCK", "").strip()
            thresh_default = 500000
        try:
            thresh = int(thresh_env) if thresh_env else thresh_default
        except ValueError:
            thresh = thresh_default
        return jnp.float32 if size_hint >= thresh else jnp.float64
    if env in {"float32", "fp32", "f32", "32"}:
        return jnp.float32
    if env in {"float64", "fp64", "f64", "64"}:
        return jnp.float64
    return jnp.float64


def _gmres_solve_dispatch(**kwargs):
    solver_fn = gmres_solve_jit if _use_solver_jit() else gmres_solve
    return solver_fn(**kwargs)


@dataclass(frozen=True)
class _RHSMode1PrecondCache:
    idx_map_jnp: jnp.ndarray
    flat_idx_jnp: jnp.ndarray
    block_inv_jnp: jnp.ndarray
    extra_idx_jnp: jnp.ndarray
    extra_inv_jnp: jnp.ndarray | None


_RHSMODE1_PRECOND_CACHE: dict[tuple[object, ...], _RHSMode1PrecondCache] = {}


@dataclass(frozen=True)
class _RHSMode1PrecondListCache:
    block_inv_list: tuple[jnp.ndarray, ...]
    block_slices: tuple[tuple[int, int], ...]
    extra_idx_jnp: jnp.ndarray
    extra_inv_jnp: jnp.ndarray | None


_RHSMODE1_PRECOND_LIST_CACHE: dict[tuple[object, ...], _RHSMode1PrecondListCache] = {}


@dataclass(frozen=True)
class _RHSMode1PrecondGlobalCache:
    idx_map_jnp: jnp.ndarray
    flat_idx_jnp: jnp.ndarray
    block_inv_jnp: jnp.ndarray
    extra_idx_jnp: jnp.ndarray
    extra_inv_jnp: jnp.ndarray | None


_RHSMODE1_PRECOND_GLOBAL_CACHE: dict[tuple[object, ...], _RHSMode1PrecondGlobalCache] = {}


@dataclass(frozen=True)
class _RHSMode1PrecondDiagXCache:
    block_inv_list: tuple[tuple[jnp.ndarray, ...], ...]
    idx_map_list: tuple[tuple[jnp.ndarray, ...], ...]
    extra_idx_jnp: jnp.ndarray
    extra_inv_jnp: jnp.ndarray | None


_RHSMODE1_PRECOND_DIAGX_CACHE: dict[tuple[object, ...], _RHSMode1PrecondDiagXCache] = {}


@dataclass(frozen=True)
class _RHSMode1PrecondIdxCache:
    block_inv_list: tuple[jnp.ndarray, ...]
    block_idx_list: tuple[jnp.ndarray, ...]
    extra_idx_jnp: jnp.ndarray
    extra_inv_jnp: jnp.ndarray | None


_RHSMODE1_PRECOND_IDX_CACHE: dict[tuple[object, ...], _RHSMode1PrecondIdxCache] = {}
_RHSMODE1_PAS_PRECOND_PROBE_CACHE: dict[tuple[object, ...], bool] = {}


@dataclass(frozen=True)
class _SparseILUCache:
    a_csr_full: object
    a_csr_drop: object
    ilu: object | None
    a_dense: np.ndarray | None
    l_dense: np.ndarray | None
    u_dense: np.ndarray | None
    l_unit_diag: bool


_RHSMODE1_SPARSE_ILU_CACHE: dict[tuple[object, ...], _SparseILUCache] = {}


def _rhsmode1_sparse_cache_key(
    op: V3FullSystemOperator,
    *,
    kind: str,
    active_size: int,
    use_active_dof_mode: bool,
    use_pas_projection: bool,
    drop_tol: float,
    drop_rel: float,
    ilu_drop_tol: float,
    fill_factor: float,
) -> tuple[object, ...]:
    return (
        *_rhsmode1_precond_cache_key(op, kind),
        int(active_size),
        int(bool(use_active_dof_mode)),
        int(bool(use_pas_projection)),
        float(drop_tol),
        float(drop_rel),
        float(ilu_drop_tol),
        float(fill_factor),
    )


def _build_sparse_ilu_from_matvec(
    *,
    matvec: Callable[[jnp.ndarray], jnp.ndarray],
    n: int,
    dtype: jnp.dtype,
    cache_key: tuple[object, ...],
    drop_tol: float,
    drop_rel: float,
    ilu_drop_tol: float,
    fill_factor: float,
    build_dense_factors: bool,
    build_ilu: bool,
    store_dense: bool,
    emit: Callable[[int, str], None] | None = None,
) -> tuple[object, object, object | None, np.ndarray | None, np.ndarray | None, np.ndarray | None, bool]:
    cached = _RHSMODE1_SPARSE_ILU_CACHE.get(cache_key)
    if cached is not None:
        return (
            cached.a_csr_full,
            cached.a_csr_drop,
            cached.ilu,
            cached.a_dense,
            cached.l_dense,
            cached.u_dense,
            cached.l_unit_diag,
        )

    if emit is not None:
        emit(1, f"sparse_ilu: assembling dense operator (n={n})")
    a_dense = assemble_dense_matrix_from_matvec(matvec=matvec, n=int(n), dtype=dtype)
    a_np_full = np.array(a_dense, dtype=np.float64, copy=True)
    max_abs = float(np.max(np.abs(a_np_full))) if a_np_full.size else 0.0
    thresh = max(float(drop_tol), float(drop_rel) * max_abs)
    a_np_drop = a_np_full
    if thresh > 0.0:
        if emit is not None:
            emit(1, f"sparse_ilu: dropping entries |a| < {thresh:.3e}")
        a_np_drop = a_np_full.copy()
        a_np_drop[np.abs(a_np_drop) < thresh] = 0.0

    import scipy.sparse as sp  # noqa: PLC0415
    from scipy.sparse.linalg import spilu  # noqa: PLC0415

    a_csr_full = sp.csr_matrix(a_np_full)
    a_csr_full.eliminate_zeros()
    a_csr_drop = sp.csr_matrix(a_np_drop)
    a_csr_drop.eliminate_zeros()
    if emit is not None:
        nnz = int(a_csr_drop.nnz)
        emit(1, f"sparse_ilu: nnz={nnz} ({nnz / max(1, n * n):.3e} density)")

    ilu = None
    if build_ilu:
        ilu = spilu(
            a_csr_drop.tocsc(),
            drop_tol=float(ilu_drop_tol),
            fill_factor=float(fill_factor),
            permc_spec="COLAMD",
        )
    a_dense = a_np_full if store_dense else None
    l_dense = None
    u_dense = None
    l_unit_diag = True
    if build_dense_factors and ilu is not None:
        l_dense = np.asarray(ilu.L.todense(), dtype=np.float64)
        u_dense = np.asarray(ilu.U.todense(), dtype=np.float64)
        diag_l = np.diag(l_dense)
        l_unit_diag = bool(np.allclose(diag_l, 1.0))
    _RHSMODE1_SPARSE_ILU_CACHE[cache_key] = _SparseILUCache(
        a_csr_full=a_csr_full,
        a_csr_drop=a_csr_drop,
        ilu=ilu,
        a_dense=a_dense,
        l_dense=l_dense,
        u_dense=u_dense,
        l_unit_diag=l_unit_diag,
    )
    return a_csr_full, a_csr_drop, ilu, a_dense, l_dense, u_dense, l_unit_diag


def _build_sparse_jax_preconditioner_from_matvec(
    *,
    matvec: Callable[[jnp.ndarray], jnp.ndarray],
    n: int,
    dtype: jnp.dtype,
    cache_key: tuple[object, ...],
    drop_tol: float,
    drop_rel: float,
    reg: float,
    omega: float,
    sweeps: int,
    emit: Callable[[int, str], None] | None = None,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    cached = _RHSMODE1_SPARSE_JAX_CACHE.get(cache_key)
    if cached is not None:
        a_sp = cached.a_sp
        d_inv = cached.d_inv
        omega = cached.omega
        sweeps = cached.sweeps
    else:
        if emit is not None:
            emit(1, f"sparse_jax: assembling dense operator (n={n})")
        a_dense = assemble_dense_matrix_from_matvec(matvec=matvec, n=int(n), dtype=dtype)
        a_dense = jnp.asarray(a_dense, dtype=dtype)
        max_abs = jnp.max(jnp.abs(a_dense)) if int(n) > 0 else jnp.asarray(0.0, dtype=dtype)
        thresh = jnp.maximum(jnp.asarray(drop_tol, dtype=dtype), jnp.asarray(drop_rel, dtype=dtype) * max_abs)
        if drop_tol > 0.0 or drop_rel > 0.0:
            a_drop = jnp.where(jnp.abs(a_dense) >= thresh, a_dense, jnp.zeros_like(a_dense))
        else:
            a_drop = a_dense
        diag_idx = jnp.arange(int(n), dtype=jnp.int32)
        diag = a_dense[diag_idx, diag_idx]
        diag_safe = diag + jnp.asarray(reg, dtype=dtype)
        a_drop = a_drop.at[diag_idx, diag_idx].set(diag_safe)
        d_inv = jnp.where(diag_safe != 0, 1.0 / diag_safe, jnp.asarray(0.0, dtype=dtype))
        try:
            from jax.experimental import sparse as jsparse  # noqa: PLC0415

            a_sp = jsparse.BCOO.fromdense(a_drop)
        except Exception as exc:  # noqa: BLE001
            if emit is not None:
                emit(1, f"sparse_jax: failed to build BCOO ({type(exc).__name__}: {exc})")
            a_sp = None
        if a_sp is None:
            raise RuntimeError("sparse_jax: failed to build sparse operator")
        _RHSMODE1_SPARSE_JAX_CACHE[cache_key] = _SparseJaxPrecondCache(
            a_sp=a_sp,
            d_inv=jnp.asarray(d_inv, dtype=dtype),
            omega=float(omega),
            sweeps=int(sweeps),
        )

    def _apply(v: jnp.ndarray) -> jnp.ndarray:
        v = jnp.asarray(v, dtype=d_inv.dtype)
        x0 = jnp.zeros_like(v)

        def _body(i, x):
            r = v - a_sp @ x
            return x + omega * d_inv * r

        x = jax.lax.fori_loop(0, int(sweeps), _body, x0)
        return jnp.asarray(x, dtype=jnp.float64)

    return _apply


@dataclass(frozen=True)
class _TransportPrecondCache:
    inv_diag_f: jnp.ndarray


@dataclass(frozen=True)
class _TransportXBlockPrecondCache:
    inv_xblock: jnp.ndarray


@dataclass(frozen=True)
class _LowRankXBlockPrecondCache:
    d_inv: jnp.ndarray
    d_inv_u: jnp.ndarray
    v: jnp.ndarray
    m_inv: jnp.ndarray


@dataclass(frozen=True)
class _TransportXmgPrecondCache:
    inv_diag_f: jnp.ndarray
    coarse_inv: jnp.ndarray
    coarse_idx: jnp.ndarray


@dataclass(frozen=True)
class _SparseJaxPrecondCache:
    a_sp: object
    d_inv: jnp.ndarray
    omega: float
    sweeps: int


_TRANSPORT_PRECOND_CACHE: dict[tuple[object, ...], _TransportPrecondCache] = {}
_RHSMODE1_DIAG_PRECOND_CACHE: dict[tuple[object, ...], _TransportPrecondCache] = {}
_RHSMODE1_XBLOCK_PRECOND_CACHE: dict[tuple[object, ...], _TransportXBlockPrecondCache] = {}
_RHSMODE1_SCHUR_CACHE: dict[tuple[object, ...], jnp.ndarray] = {}
_TRANSPORT_SXBLOCK_LR_PRECOND_CACHE: dict[tuple[object, ...], _LowRankXBlockPrecondCache] = {}
_RHSMODE1_SXBLOCK_LR_PRECOND_CACHE: dict[tuple[object, ...], _LowRankXBlockPrecondCache] = {}
_TRANSPORT_XMG_PRECOND_CACHE: dict[tuple[object, ...], _TransportXmgPrecondCache] = {}
_RHSMODE1_SXBLOCK_PRECOND_CACHE: dict[tuple[object, ...], _TransportXBlockPrecondCache] = {}
_TRANSPORT_XBLOCK_PRECOND_CACHE: dict[tuple[object, ...], _TransportXBlockPrecondCache] = {}
_TRANSPORT_SXBLOCK_PRECOND_CACHE: dict[tuple[object, ...], _TransportXBlockPrecondCache] = {}
_RHSMODE23_PRECOND_CACHE: dict[tuple[object, ...], _RHSMode1PrecondCache] = {}
_RHSMODE1_SPARSE_JAX_CACHE: dict[tuple[object, ...], _SparseJaxPrecondCache] = {}


def _precond_chunk_cols(total_size: int, n_cols: int) -> int:
    env_cols = os.environ.get("SFINCS_JAX_PRECOND_CHUNK", "").strip()
    if env_cols:
        try:
            cols = int(env_cols)
            if cols > 0:
                return min(cols, n_cols)
        except ValueError:
            pass
    env_max_mb = os.environ.get("SFINCS_JAX_PRECOND_MAX_MB", "").strip()
    try:
        max_mb = float(env_max_mb) if env_max_mb else 256.0
    except ValueError:
        max_mb = 256.0
    if max_mb <= 0:
        return n_cols
    bytes_per_row = int(total_size) * 8
    if bytes_per_row <= 0:
        return n_cols
    max_cols = max(1, int((max_mb * 1e6) // bytes_per_row))
    return min(n_cols, max_cols)


def _matvec_submatrix(
    op_pc: V3FullSystemOperator,
    *,
    col_idx: np.ndarray,
    row_idx: np.ndarray,
    total_size: int,
    chunk_cols: int,
) -> np.ndarray:
    col_idx = np.asarray(col_idx, dtype=np.int32)
    row_idx_jnp = jnp.asarray(row_idx, dtype=jnp.int32)
    blocks: list[np.ndarray] = []
    for start in range(0, int(col_idx.shape[0]), int(chunk_cols)):
        idx = col_idx[start : start + int(chunk_cols)]
        basis = jax.nn.one_hot(jnp.asarray(idx, dtype=jnp.int32), total_size, dtype=jnp.float64)
        y = jax.vmap(lambda v: apply_v3_full_system_operator_cached(op_pc, v))(basis)
        y_sub = y[:, row_idx_jnp]
        blocks.append(np.asarray(y_sub, dtype=np.float64))
    if len(blocks) == 1:
        return blocks[0]
    return np.concatenate(blocks, axis=0)


def _hash_array(arr: jnp.ndarray | np.ndarray) -> str:
    arr_np = np.asarray(arr, dtype=np.float64)
    return hashlib.blake2b(arr_np.tobytes(), digest_size=8).hexdigest()


def _rhsmode1_dense_fallback_max(op: V3FullSystemOperator) -> int:
    dense_env = os.environ.get("SFINCS_JAX_RHSMODE1_DENSE_FALLBACK_MAX", "").strip()
    try:
        base_max = int(dense_env) if dense_env else 400
    except ValueError:
        base_max = 0
    if op.fblock.fp is None:
        if int(op.constraint_scheme) != 2:
            return base_max
        dense_pas_env = os.environ.get("SFINCS_JAX_RHSMODE1_DENSE_PAS_MAX", "").strip()
        try:
            dense_pas_max = int(dense_pas_env) if dense_pas_env else 5000
        except ValueError:
            dense_pas_max = base_max
        if dense_pas_max <= 0:
            return base_max
        return max(base_max, dense_pas_max)
    dense_fp_env = os.environ.get("SFINCS_JAX_RHSMODE1_DENSE_FP_MAX", "").strip()
    try:
        dense_fp_max = int(dense_fp_env) if dense_fp_env else 5000
    except ValueError:
        dense_fp_max = base_max
    if dense_fp_max <= 0:
        return base_max
    return max(base_max, dense_fp_max)


def _rhsmode1_precond_cache_key(op: V3FullSystemOperator, kind: str) -> tuple[object, ...]:
    nxi_for_x = np.asarray(op.fblock.collisionless.n_xi_for_x, dtype=np.int32)
    precond_dtype = str(_precond_dtype())
    # RHS-only gradients do not affect the operator; omit them so preconditioners
    # can be reused across whichRHS/scan points.
    return (
        kind,
        precond_dtype,
        int(op.rhs_mode),
        int(op.n_species),
        int(op.n_x),
        int(op.n_xi),
        int(op.n_theta),
        int(op.n_zeta),
        int(op.constraint_scheme),
        int(op.quasineutrality_option),
        bool(op.include_phi1),
        bool(op.include_phi1_in_kinetic),
        bool(op.with_adiabatic),
        float(op.alpha),
        float(op.delta),
        float(op.dphi_hat_dpsi_hat),
        _hash_array(op.adiabatic_z),
        _hash_array(op.adiabatic_nhat),
        _hash_array(op.adiabatic_that),
        _hash_array(op.z_s),
        _hash_array(op.m_hat),
        _hash_array(op.t_hat),
        _hash_array(op.n_hat),
        _hash_array(op.theta_weights),
        _hash_array(op.zeta_weights),
        _hash_array(op.b_hat),
        _hash_array(op.d_hat),
        _hash_array(op.b_hat_sub_theta),
        _hash_array(op.b_hat_sub_zeta),
        _hash_array(op.x),
        _hash_array(op.x_weights),
        tuple(nxi_for_x.tolist()),
    )


def _transport_precond_cache_key(op: V3FullSystemOperator, kind: str) -> tuple[object, ...]:
    nxi_for_x = np.asarray(op.fblock.collisionless.n_xi_for_x, dtype=np.int32)
    pas = op.fblock.pas
    fp = op.fblock.fp
    precond_dtype = str(_precond_dtype())
    return (
        kind,
        precond_dtype,
        int(op.n_species),
        int(op.n_x),
        int(op.n_xi),
        int(op.n_theta),
        int(op.n_zeta),
        float(op.fblock.identity_shift),
        bool(pas is not None),
        float(pas.nu_n) if pas is not None else None,
        float(pas.krook) if pas is not None else None,
        _hash_array(pas.nu_d_hat) if pas is not None else None,
        bool(fp is not None),
        _hash_array(fp.mat) if fp is not None else None,
        tuple(nxi_for_x.tolist()),
    )


def _build_rhsmode23_collision_preconditioner(
    *,
    op: V3FullSystemOperator,
    reduce_full: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    expand_reduced: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Cheap diagonal preconditioner for RHSMode=2/3 transport solves.

    Uses analytic diagonal contributions from the collision operator (PAS or FP) plus
    the identity shift, and is diagonal in (theta, zeta).
    """
    cache_key = _transport_precond_cache_key(op, "collision_diag")
    precond_dtype = _precond_dtype()
    cached = _TRANSPORT_PRECOND_CACHE.get(cache_key)
    if cached is None:
        f_shape = op.fblock.f_shape
        n_species, n_x, n_l, _, _ = f_shape
        diag = jnp.zeros(f_shape, dtype=jnp.float64)

        # Identity shift contribution.
        if float(op.fblock.identity_shift) != 0.0:
            diag = diag + jnp.asarray(op.fblock.identity_shift, dtype=jnp.float64)

        # Pitch-angle scattering diagonal term.
        if op.fblock.pas is not None:
            pas = op.fblock.pas
            l = jnp.arange(n_l, dtype=jnp.float64)
            factor_l = 0.5 * (l * (l + 1.0) + 2.0 * pas.krook)
            pas_diag = pas.nu_n * pas.nu_d_hat[:, :, None] * factor_l[None, None, :]
            diag = diag + pas_diag[:, :, :, None, None]

        # Fokker-Planck diagonal term (self-species, diagonal in x).
        if op.fblock.fp is not None:
            mat = op.fblock.fp.mat  # (S,S,L,X,X)
            diag_x = jnp.diagonal(mat, axis1=3, axis2=4)  # (S,S,L,X)
            diag_self = jnp.diagonal(diag_x, axis1=0, axis2=1)  # (L,X,S)
            diag_self = jnp.transpose(diag_self, (2, 1, 0))  # (S,X,L)
            diag = diag + diag_self[:, :, :, None, None]

        # Mask out inactive L-modes.
        nxi_for_x = op.fblock.collisionless.n_xi_for_x.astype(jnp.int32)
        mask = jnp.arange(n_l, dtype=jnp.int32)[None, :] < nxi_for_x[:, None]  # (X,L)
        mask = mask[None, :, :, None, None]  # (1,X,L,1,1)
        diag = jnp.where(mask, diag, jnp.asarray(1.0, dtype=jnp.float64))

        reg_env = os.environ.get("SFINCS_JAX_TRANSPORT_PRECOND_REG", "").strip()
        try:
            reg = float(reg_env) if reg_env else 1e-10
        except ValueError:
            reg = 1e-10
        inv_diag_f = 1.0 / (diag + float(reg))
        cached = _TransportPrecondCache(inv_diag_f=jnp.asarray(inv_diag_f, dtype=precond_dtype))
        _TRANSPORT_PRECOND_CACHE[cache_key] = cached

    inv_diag_f = cached.inv_diag_f

    def _apply_full(r_full: jnp.ndarray) -> jnp.ndarray:
        r_full = jnp.asarray(r_full, dtype=precond_dtype)
        f = r_full[: op.f_size].reshape(op.fblock.f_shape)
        z_f = f * inv_diag_f
        tail = r_full[op.f_size :]
        z_full = jnp.concatenate([z_f.reshape((-1,)), tail], axis=0)
        return jnp.asarray(z_full, dtype=jnp.float64)

    if reduce_full is None or expand_reduced is None:
        return _apply_full

    def _apply_reduced(r_reduced: jnp.ndarray) -> jnp.ndarray:
        z_full = _apply_full(expand_reduced(r_reduced))
        return reduce_full(z_full)

    return _apply_reduced


def _build_rhsmode23_sxblock_preconditioner(
    *,
    op: V3FullSystemOperator,
    reduce_full: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    expand_reduced: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Lightweight block-Jacobi preconditioner for RHSMode=2/3 using species/x blocks.

    Builds per-L blocks across species and x from the collision operator (PAS/FP) plus
    identity shift. This avoids matvec-based assembly while capturing cross-species/x
    coupling in the FP operator.
    """
    if op.fblock.fp is None:
        return _build_rhsmode23_collision_preconditioner(op=op, reduce_full=reduce_full, expand_reduced=expand_reduced)

    low_rank_env = os.environ.get("SFINCS_JAX_TRANSPORT_FP_LOW_RANK_K", "").strip()
    if not low_rank_env:
        low_rank_env = os.environ.get("SFINCS_JAX_FP_LOW_RANK_K", "").strip()
    low_rank_env = low_rank_env.strip().lower()
    low_rank_auto = low_rank_env in {"", "auto"}
    if low_rank_env and low_rank_env != "auto":
        try:
            low_rank_k = int(low_rank_env)
        except ValueError:
            low_rank_k = 0
    else:
        low_rank_k = 0

    f_shape = op.fblock.f_shape
    n_species, n_x, n_l, _, _ = f_shape
    n_block = n_species * n_x
    if low_rank_auto and low_rank_k <= 0 and n_block >= 24:
        low_rank_k = min(8, n_block)

    precond_dtype = _precond_dtype()
    if low_rank_k > 0:
        rank_k = min(int(low_rank_k), int(n_block))
        cache_key = _transport_precond_cache_key(op, f"collision_sxblock_lr_{rank_k}")
        cached_lr = _TRANSPORT_SXBLOCK_LR_PRECOND_CACHE.get(cache_key)
        if cached_lr is None:
            mat = np.asarray(op.fblock.fp.mat, dtype=np.float64)  # (S,S,L,X,X)
            reg_env = os.environ.get("SFINCS_JAX_TRANSPORT_PRECOND_REG", "").strip()
            try:
                reg = float(reg_env) if reg_env else 1e-10
            except ValueError:
                reg = 1e-10
            identity_shift = float(op.fblock.identity_shift)
            pas_diag = None
            if op.fblock.pas is not None:
                pas = op.fblock.pas
                l_arr = np.arange(n_l, dtype=np.float64)
                factor_l = 0.5 * (l_arr * (l_arr + 1.0) + 2.0 * float(pas.krook))
                pas_diag = float(pas.nu_n) * np.asarray(pas.nu_d_hat, dtype=np.float64)[:, :, None] * factor_l[None, None, :]

            nxi_for_x = np.asarray(op.fblock.collisionless.n_xi_for_x, dtype=np.int32)
            d_inv = np.zeros((n_l, n_block), dtype=np.float64)
            d_inv_u = np.zeros((n_l, n_block, rank_k), dtype=np.float64)
            v_lr = np.zeros((n_l, rank_k, n_block), dtype=np.float64)
            m_inv = np.zeros((n_l, rank_k, rank_k), dtype=np.float64)

            for l in range(n_l):
                a_fp = np.array(mat[:, :, l, :, :], dtype=np.float64, copy=True)  # (S,S,X,X)
                a_fp = a_fp.transpose(0, 2, 1, 3).reshape((n_block, n_block))
                diag = np.full((n_block,), identity_shift + reg, dtype=np.float64)
                if pas_diag is not None:
                    diag += pas_diag[:, :, l].reshape((n_block,))
                inactive_x = np.where(nxi_for_x <= l)[0]
                if inactive_x.size:
                    for ix in inactive_x:
                        for s in range(n_species):
                            idx = s * n_x + int(ix)
                            a_fp[idx, :] = 0.0
                            a_fp[:, idx] = 0.0
                            diag[idx] = 1.0
                d_inv_l = 1.0 / diag
                d_inv[l, :] = d_inv_l
                if rank_k > 0:
                    try:
                        u, svals, vt = np.linalg.svd(a_fp, full_matrices=False)
                    except np.linalg.LinAlgError:
                        u, svals, vt = np.linalg.svd(a_fp + 1e-12 * np.eye(n_block), full_matrices=False)
                    k_use = min(rank_k, int(svals.shape[0]))
                    if k_use > 0:
                        u = u[:, :k_use]
                        svals = svals[:k_use]
                        vt = vt[:k_use, :]
                        s_sqrt = np.sqrt(np.maximum(svals, 0.0))
                        u_lr = u * s_sqrt[None, :]
                        v_lr_l = s_sqrt[:, None] * vt
                        d_inv_u_l = d_inv_l[:, None] * u_lr
                        m = np.eye(k_use, dtype=np.float64) + v_lr_l @ d_inv_u_l
                        try:
                            m_inv_l = np.linalg.inv(m)
                        except np.linalg.LinAlgError:
                            m_inv_l = np.linalg.pinv(m, rcond=1e-12)
                        if not np.all(np.isfinite(m_inv_l)):
                            m_inv_l = np.linalg.pinv(m, rcond=1e-12)
                        d_inv_u[l, :, :k_use] = d_inv_u_l
                        v_lr[l, :k_use, :] = v_lr_l
                        m_inv[l, :k_use, :k_use] = m_inv_l

            cached_lr = _LowRankXBlockPrecondCache(
                d_inv=jnp.asarray(d_inv, dtype=precond_dtype),
                d_inv_u=jnp.asarray(d_inv_u, dtype=precond_dtype),
                v=jnp.asarray(v_lr, dtype=precond_dtype),
                m_inv=jnp.asarray(m_inv, dtype=precond_dtype),
            )
            _TRANSPORT_SXBLOCK_LR_PRECOND_CACHE[cache_key] = cached_lr

        d_inv = cached_lr.d_inv
        d_inv_u = cached_lr.d_inv_u
        v_lr = cached_lr.v
        m_inv = cached_lr.m_inv

        def _apply_full(r_full: jnp.ndarray) -> jnp.ndarray:
            r_full = jnp.asarray(r_full, dtype=precond_dtype)
            f = r_full[: op.f_size].reshape(op.fblock.f_shape)  # (S,X,L,T,Z)
            f_l = jnp.transpose(f, (2, 0, 1, 3, 4))  # (L,S,X,T,Z)
            f_l = f_l.reshape((int(op.n_xi), int(op.n_species) * int(op.n_x), int(op.n_theta), int(op.n_zeta)))
            d_r = d_inv[:, :, None, None] * f_l
            if rank_k > 0:
                tmp = jnp.einsum("lkn,lntz->lktz", v_lr, d_r)
                tmp2 = jnp.einsum("lkm,lmtz->lktz", m_inv, tmp)
                corr = jnp.einsum("lnk,lktz->lntz", d_inv_u, tmp2)
                z_l = d_r - corr
            else:
                z_l = d_r
            z_l = z_l.reshape((int(op.n_xi), int(op.n_species), int(op.n_x), int(op.n_theta), int(op.n_zeta)))
            z_f = jnp.transpose(z_l, (1, 2, 0, 3, 4))  # (S,X,L,T,Z)
            tail = r_full[op.f_size :]
            z_full = jnp.concatenate([z_f.reshape((-1,)), tail], axis=0)
            return jnp.asarray(z_full, dtype=jnp.float64)

        if reduce_full is None or expand_reduced is None:
            return _apply_full

        def _apply_reduced(r_reduced: jnp.ndarray) -> jnp.ndarray:
            z_full = _apply_full(expand_reduced(r_reduced))
            return reduce_full(z_full)

        return _apply_reduced

    cache_key = _transport_precond_cache_key(op, "collision_sxblock")
    cached = _TRANSPORT_SXBLOCK_PRECOND_CACHE.get(cache_key)
    if cached is None:
        f_shape = op.fblock.f_shape
        n_species, n_x, n_l, _, _ = f_shape
        n_block = n_species * n_x
        inv_block = np.zeros((n_l, n_block, n_block), dtype=np.float64)
        mat = np.asarray(op.fblock.fp.mat, dtype=np.float64)  # (S,S,L,X,X)

        reg_env = os.environ.get("SFINCS_JAX_TRANSPORT_PRECOND_REG", "").strip()
        try:
            reg = float(reg_env) if reg_env else 1e-10
        except ValueError:
            reg = 1e-10

        identity_shift = float(op.fblock.identity_shift)
        pas_diag = None
        if op.fblock.pas is not None:
            pas = op.fblock.pas
            l_arr = np.arange(n_l, dtype=np.float64)
            factor_l = 0.5 * (l_arr * (l_arr + 1.0) + 2.0 * float(pas.krook))
            pas_diag = float(pas.nu_n) * np.asarray(pas.nu_d_hat, dtype=np.float64)[:, :, None] * factor_l[None, None, :]

        nxi_for_x = np.asarray(op.fblock.collisionless.n_xi_for_x, dtype=np.int32)
        for l in range(n_l):
            a = np.array(mat[:, :, l, :, :], dtype=np.float64, copy=True)  # (S,S,X,X)
            a = a.transpose(0, 2, 1, 3).reshape((n_block, n_block))
            if identity_shift != 0.0:
                a[np.arange(n_block), np.arange(n_block)] += identity_shift
            if pas_diag is not None:
                diag_add = pas_diag[:, :, l].reshape((n_block,))
                a[np.arange(n_block), np.arange(n_block)] += diag_add
            if reg != 0.0:
                a[np.arange(n_block), np.arange(n_block)] += reg

            inactive_x = np.where(nxi_for_x <= l)[0]
            if inactive_x.size:
                for ix in inactive_x:
                    for s in range(n_species):
                        idx = s * n_x + int(ix)
                        a[idx, :] = 0.0
                        a[:, idx] = 0.0
                        a[idx, idx] = 1.0

            try:
                inv = np.linalg.inv(a)
            except np.linalg.LinAlgError:
                inv = np.linalg.pinv(a, rcond=1e-12)
            if not np.all(np.isfinite(inv)):
                inv = np.linalg.pinv(a, rcond=1e-12)
            inv_block[l, :, :] = inv

        cached = _TransportXBlockPrecondCache(inv_xblock=jnp.asarray(inv_block, dtype=precond_dtype))
        _TRANSPORT_SXBLOCK_PRECOND_CACHE[cache_key] = cached

    inv_block = cached.inv_xblock  # (L, S*X, S*X)

    def _apply_full(r_full: jnp.ndarray) -> jnp.ndarray:
        r_full = jnp.asarray(r_full, dtype=precond_dtype)
        f = r_full[: op.f_size].reshape(op.fblock.f_shape)  # (S,X,L,T,Z)
        f_l = jnp.transpose(f, (2, 0, 1, 3, 4))  # (L,S,X,T,Z)
        f_l = f_l.reshape((int(op.n_xi), int(op.n_species) * int(op.n_x), int(op.n_theta), int(op.n_zeta)))
        z_l = jnp.einsum("lmn,lntz->lmtz", inv_block, f_l)
        z_l = z_l.reshape((int(op.n_xi), int(op.n_species), int(op.n_x), int(op.n_theta), int(op.n_zeta)))
        z_f = jnp.transpose(z_l, (1, 2, 0, 3, 4))  # (S,X,L,T,Z)
        tail = r_full[op.f_size :]
        z_full = jnp.concatenate([z_f.reshape((-1,)), tail], axis=0)
        return jnp.asarray(z_full, dtype=jnp.float64)

    if reduce_full is None or expand_reduced is None:
        return _apply_full

    def _apply_reduced(r_reduced: jnp.ndarray) -> jnp.ndarray:
        z_full = _apply_full(expand_reduced(r_reduced))
        return reduce_full(z_full)

    return _apply_reduced


def _build_rhsmode23_xmg_preconditioner(
    *,
    op: V3FullSystemOperator,
    reduce_full: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    expand_reduced: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Two-level additive x-grid preconditioner for RHSMode=2/3 collision operators.

    Applies a fine-grid diagonal inverse plus a coarse-grid correction on the speed grid.
    The coarse solve is block-diagonal in species and L, ignoring cross-species coupling.
    """
    stride_env = os.environ.get("SFINCS_JAX_XMG_STRIDE", "").strip()
    try:
        stride = int(stride_env) if stride_env else 2
    except ValueError:
        stride = 2
    stride = max(1, stride)
    cache_key = _transport_precond_cache_key(op, f"xmg_{stride}")
    precond_dtype = _precond_dtype()
    cached = _TRANSPORT_XMG_PRECOND_CACHE.get(cache_key)
    if cached is None:
        f_shape = op.fblock.f_shape
        n_species, n_x, n_l, _, _ = f_shape
        coarse_idx = np.arange(0, n_x, stride, dtype=np.int32)
        n_coarse = int(coarse_idx.shape[0])
        coarse_map = {int(ix): int(i) for i, ix in enumerate(coarse_idx)}

        diag = np.zeros(f_shape, dtype=np.float64)
        if float(op.fblock.identity_shift) != 0.0:
            diag = diag + float(op.fblock.identity_shift)
        if op.fblock.pas is not None:
            pas = op.fblock.pas
            l_arr = np.arange(n_l, dtype=np.float64)
            factor_l = 0.5 * (l_arr * (l_arr + 1.0) + 2.0 * float(pas.krook))
            pas_diag = float(pas.nu_n) * np.asarray(pas.nu_d_hat, dtype=np.float64)[:, :, None] * factor_l[None, None, :]
            diag = diag + pas_diag[:, :, :, None, None]
        if op.fblock.fp is not None:
            mat = np.asarray(op.fblock.fp.mat, dtype=np.float64)  # (S,S,L,X,X)
            diag_x = np.diagonal(mat, axis1=3, axis2=4)  # (S,S,L,X)
            diag_self = np.diagonal(diag_x, axis1=0, axis2=1)  # (L,X,S)
            diag_self = np.transpose(diag_self, (2, 1, 0))  # (S,X,L)
            diag = diag + diag_self[:, :, :, None, None]

        nxi_for_x = np.asarray(op.fblock.collisionless.n_xi_for_x, dtype=np.int32)
        mask = np.arange(n_l, dtype=np.int32)[None, :] < nxi_for_x[:, None]  # (X,L)
        mask = mask[None, :, :, None, None]
        diag = np.where(mask, diag, 1.0)

        reg_env = os.environ.get("SFINCS_JAX_TRANSPORT_PRECOND_REG", "").strip()
        try:
            reg = float(reg_env) if reg_env else 1e-10
        except ValueError:
            reg = 1e-10
        inv_diag_f = 1.0 / (diag + float(reg))

        coarse_inv = np.zeros((n_species, n_l, n_coarse, n_coarse), dtype=np.float64)
        mat_fp = None
        if op.fblock.fp is not None:
            mat_fp = np.asarray(op.fblock.fp.mat, dtype=np.float64)  # (S,S,L,X,X)
        identity_shift = float(op.fblock.identity_shift)
        pas_diag = None
        if op.fblock.pas is not None:
            pas = op.fblock.pas
            l_arr = np.arange(n_l, dtype=np.float64)
            factor_l = 0.5 * (l_arr * (l_arr + 1.0) + 2.0 * float(pas.krook))
            pas_diag = float(pas.nu_n) * np.asarray(pas.nu_d_hat, dtype=np.float64)[:, :, None] * factor_l[None, None, :]

        for s in range(n_species):
            for l in range(n_l):
                if mat_fp is None:
                    a = np.zeros((n_x, n_x), dtype=np.float64)
                else:
                    a = np.array(mat_fp[s, s, l, :, :], dtype=np.float64, copy=True)
                a = a[np.ix_(coarse_idx, coarse_idx)]
                diag_vec = np.full((n_coarse,), identity_shift + reg, dtype=np.float64)
                if pas_diag is not None:
                    diag_vec += pas_diag[s, coarse_idx, l]
                a[np.arange(n_coarse), np.arange(n_coarse)] += diag_vec

                inactive_x = np.where(nxi_for_x <= l)[0]
                if inactive_x.size:
                    for ix in inactive_x:
                        j = coarse_map.get(int(ix))
                        if j is not None:
                            a[j, :] = 0.0
                            a[:, j] = 0.0
                            a[j, j] = 1.0

                try:
                    inv = np.linalg.inv(a)
                except np.linalg.LinAlgError:
                    inv = np.linalg.pinv(a, rcond=1e-12)
                if not np.all(np.isfinite(inv)):
                    inv = np.linalg.pinv(a, rcond=1e-12)
                coarse_inv[s, l, :, :] = inv

        cached = _TransportXmgPrecondCache(
            inv_diag_f=jnp.asarray(inv_diag_f, dtype=precond_dtype),
            coarse_inv=jnp.asarray(coarse_inv, dtype=precond_dtype),
            coarse_idx=jnp.asarray(coarse_idx, dtype=jnp.int32),
        )
        _TRANSPORT_XMG_PRECOND_CACHE[cache_key] = cached

    inv_diag_f = cached.inv_diag_f
    coarse_inv = cached.coarse_inv
    coarse_idx = cached.coarse_idx

    def _apply_full(r_full: jnp.ndarray) -> jnp.ndarray:
        r_full = jnp.asarray(r_full, dtype=precond_dtype)
        f = r_full[: op.f_size].reshape(op.fblock.f_shape)  # (S,X,L,T,Z)
        z_f = f * inv_diag_f
        f_sl = jnp.transpose(f, (0, 2, 1, 3, 4))  # (S,L,X,T,Z)
        f_coarse = f_sl[:, :, coarse_idx, :, :]
        z_coarse = jnp.einsum("slij,sljtz->slitz", coarse_inv, f_coarse)
        corr_sl = jnp.zeros_like(f_sl)
        corr_sl = corr_sl.at[:, :, coarse_idx, :, :].set(z_coarse)
        corr = jnp.transpose(corr_sl, (0, 2, 1, 3, 4))
        z_f = z_f + corr
        tail = r_full[op.f_size :]
        z_full = jnp.concatenate([z_f.reshape((-1,)), tail], axis=0)
        return jnp.asarray(z_full, dtype=jnp.float64)

    if reduce_full is None or expand_reduced is None:
        return _apply_full

    def _apply_reduced(r_reduced: jnp.ndarray) -> jnp.ndarray:
        z_full = _apply_full(expand_reduced(r_reduced))
        return reduce_full(z_full)

    return _apply_reduced


def _build_rhsmode1_collision_preconditioner(
    *,
    op: V3FullSystemOperator,
    reduce_full: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    expand_reduced: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Cheap collision-based preconditioner for RHSMode=1 solves (BiCGStab-friendly)."""
    use_xblock = False
    use_sxblock = False
    precond_dtype = _precond_dtype()
    kind_env = os.environ.get("SFINCS_JAX_RHSMODE1_COLLISION_PRECOND_KIND", "").strip().lower()
    if kind_env in {"xblock", "block_x", "x"}:
        use_xblock = True
    elif kind_env in {"sxblock", "species_block", "block"}:
        use_sxblock = True
    elif kind_env in {"", "auto"} and op.fblock.fp is not None:
        f_shape = op.fblock.f_shape
        n_species, n_x, _, _, _ = f_shape
        n_block = int(n_species) * int(n_x)
        sxblock_max_env = os.environ.get("SFINCS_JAX_RHSMODE1_COLLISION_SXBLOCK_MAX", "").strip()
        try:
            sxblock_max = int(sxblock_max_env) if sxblock_max_env else 64
        except ValueError:
            sxblock_max = 64
        xblock_max_env = os.environ.get("SFINCS_JAX_RHSMODE1_COLLISION_XBLOCK_MAX", "").strip()
        try:
            xblock_max = int(xblock_max_env) if xblock_max_env else 256
        except ValueError:
            xblock_max = 256
        if sxblock_max >= 0 and n_block <= sxblock_max:
            use_sxblock = True
        elif xblock_max >= 0 and int(n_x) <= xblock_max:
            use_xblock = True

    if use_sxblock and op.fblock.fp is not None:
        low_rank_env = os.environ.get("SFINCS_JAX_RHSMODE1_FP_LOW_RANK_K", "").strip()
        if not low_rank_env:
            low_rank_env = os.environ.get("SFINCS_JAX_FP_LOW_RANK_K", "").strip()
        low_rank_env = low_rank_env.strip().lower()
        low_rank_auto = low_rank_env in {"", "auto"}
        if low_rank_env and low_rank_env != "auto":
            try:
                low_rank_k = int(low_rank_env)
            except ValueError:
                low_rank_k = 0
        else:
            low_rank_k = 0
        f_shape = op.fblock.f_shape
        n_species, n_x, n_l, _, _ = f_shape
        n_block = n_species * n_x
        if low_rank_auto and low_rank_k <= 0 and n_block >= 24:
            low_rank_k = min(8, n_block)

        if low_rank_k > 0:
            rank_k = min(int(low_rank_k), int(n_block))
            cache_key = _transport_precond_cache_key(op, f"rhs1_collision_sxblock_lr_{rank_k}")
            cached_lr = _RHSMODE1_SXBLOCK_LR_PRECOND_CACHE.get(cache_key)
            if cached_lr is None:
                mat = np.asarray(op.fblock.fp.mat, dtype=np.float64)  # (S,S,L,X,X)
                reg_env = os.environ.get("SFINCS_JAX_RHSMODE1_BICGSTAB_PRECOND_REG", "").strip()
                try:
                    reg = float(reg_env) if reg_env else 1e-10
                except ValueError:
                    reg = 1e-10
                identity_shift = float(op.fblock.identity_shift)
                pas_diag = None
                if op.fblock.pas is not None:
                    pas = op.fblock.pas
                    l_arr = np.arange(n_l, dtype=np.float64)
                    factor_l = 0.5 * (l_arr * (l_arr + 1.0) + 2.0 * float(pas.krook))
                    pas_diag = float(pas.nu_n) * np.asarray(pas.nu_d_hat, dtype=np.float64)[:, :, None] * factor_l[None, None, :]

                nxi_for_x = np.asarray(op.fblock.collisionless.n_xi_for_x, dtype=np.int32)
                d_inv = np.zeros((n_l, n_block), dtype=np.float64)
                d_inv_u = np.zeros((n_l, n_block, rank_k), dtype=np.float64)
                v_lr = np.zeros((n_l, rank_k, n_block), dtype=np.float64)
                m_inv = np.zeros((n_l, rank_k, rank_k), dtype=np.float64)

                for l in range(n_l):
                    a_fp = np.array(mat[:, :, l, :, :], dtype=np.float64, copy=True)  # (S,S,X,X)
                    a_fp = a_fp.transpose(0, 2, 1, 3).reshape((n_block, n_block))
                    diag = np.full((n_block,), identity_shift + reg, dtype=np.float64)
                    if pas_diag is not None:
                        diag += pas_diag[:, :, l].reshape((n_block,))
                    inactive_x = np.where(nxi_for_x <= l)[0]
                    if inactive_x.size:
                        for ix in inactive_x:
                            for s in range(n_species):
                                idx = s * n_x + int(ix)
                                a_fp[idx, :] = 0.0
                                a_fp[:, idx] = 0.0
                                diag[idx] = 1.0
                    d_inv_l = 1.0 / diag
                    d_inv[l, :] = d_inv_l
                    if rank_k > 0:
                        try:
                            u, svals, vt = np.linalg.svd(a_fp, full_matrices=False)
                        except np.linalg.LinAlgError:
                            u, svals, vt = np.linalg.svd(a_fp + 1e-12 * np.eye(n_block), full_matrices=False)
                        k_use = min(rank_k, int(svals.shape[0]))
                        if k_use > 0:
                            u = u[:, :k_use]
                            svals = svals[:k_use]
                            vt = vt[:k_use, :]
                            s_sqrt = np.sqrt(np.maximum(svals, 0.0))
                            u_lr = u * s_sqrt[None, :]
                            v_lr_l = s_sqrt[:, None] * vt
                            d_inv_u_l = d_inv_l[:, None] * u_lr
                            m = np.eye(k_use, dtype=np.float64) + v_lr_l @ d_inv_u_l
                            try:
                                m_inv_l = np.linalg.inv(m)
                            except np.linalg.LinAlgError:
                                m_inv_l = np.linalg.pinv(m, rcond=1e-12)
                            if not np.all(np.isfinite(m_inv_l)):
                                m_inv_l = np.linalg.pinv(m, rcond=1e-12)
                            d_inv_u[l, :, :k_use] = d_inv_u_l
                            v_lr[l, :k_use, :] = v_lr_l
                            m_inv[l, :k_use, :k_use] = m_inv_l

                cached_lr = _LowRankXBlockPrecondCache(
                    d_inv=jnp.asarray(d_inv, dtype=precond_dtype),
                    d_inv_u=jnp.asarray(d_inv_u, dtype=precond_dtype),
                    v=jnp.asarray(v_lr, dtype=precond_dtype),
                    m_inv=jnp.asarray(m_inv, dtype=precond_dtype),
                )
                _RHSMODE1_SXBLOCK_LR_PRECOND_CACHE[cache_key] = cached_lr

            d_inv = cached_lr.d_inv
            d_inv_u = cached_lr.d_inv_u
            v_lr = cached_lr.v
            m_inv = cached_lr.m_inv

            def _apply_full(r_full: jnp.ndarray) -> jnp.ndarray:
                r_full = jnp.asarray(r_full, dtype=precond_dtype)
                f = r_full[: op.f_size].reshape(op.fblock.f_shape)  # (S,X,L,T,Z)
                f_l = jnp.transpose(f, (2, 0, 1, 3, 4))  # (L,S,X,T,Z)
                f_l = f_l.reshape((int(op.n_xi), int(op.n_species) * int(op.n_x), int(op.n_theta), int(op.n_zeta)))
                d_r = d_inv[:, :, None, None] * f_l
                if rank_k > 0:
                    tmp = jnp.einsum("lkn,lntz->lktz", v_lr, d_r)
                    tmp2 = jnp.einsum("lkm,lmtz->lktz", m_inv, tmp)
                    corr = jnp.einsum("lnk,lktz->lntz", d_inv_u, tmp2)
                    z_l = d_r - corr
                else:
                    z_l = d_r
                z_l = z_l.reshape((int(op.n_xi), int(op.n_species), int(op.n_x), int(op.n_theta), int(op.n_zeta)))
                z_f = jnp.transpose(z_l, (1, 2, 0, 3, 4))  # (S,X,L,T,Z)
                tail = r_full[op.f_size :]
                z_full = jnp.concatenate([z_f.reshape((-1,)), tail], axis=0)
                return jnp.asarray(z_full, dtype=jnp.float64)
        else:
            cache_key = _transport_precond_cache_key(op, "rhs1_collision_sxblock")
            cached = _RHSMODE1_SXBLOCK_PRECOND_CACHE.get(cache_key)
            if cached is None:
                f_shape = op.fblock.f_shape
                n_species, n_x, n_l, _, _ = f_shape
                n_block = n_species * n_x
                inv_block = np.zeros((n_l, n_block, n_block), dtype=np.float64)
                mat = np.asarray(op.fblock.fp.mat, dtype=np.float64)  # (S,S,L,X,X)
                reg_env = os.environ.get("SFINCS_JAX_RHSMODE1_BICGSTAB_PRECOND_REG", "").strip()
                try:
                    reg = float(reg_env) if reg_env else 1e-10
                except ValueError:
                    reg = 1e-10
                identity_shift = float(op.fblock.identity_shift)
                pas_diag = None
                if op.fblock.pas is not None:
                    pas = op.fblock.pas
                    l_arr = np.arange(n_l, dtype=np.float64)
                    factor_l = 0.5 * (l_arr * (l_arr + 1.0) + 2.0 * float(pas.krook))
                    pas_diag = float(pas.nu_n) * np.asarray(pas.nu_d_hat, dtype=np.float64)[:, :, None] * factor_l[None, None, :]

                for l in range(n_l):
                    a = np.array(mat[:, :, l, :, :], dtype=np.float64, copy=True)  # (S,S,X,X)
                    a = a.transpose(0, 2, 1, 3).reshape((n_block, n_block))
                    if identity_shift != 0.0:
                        a[np.arange(n_block), np.arange(n_block)] += identity_shift
                    if pas_diag is not None:
                        diag_add = pas_diag[:, :, l].reshape((n_block,))
                        a[np.arange(n_block), np.arange(n_block)] += diag_add
                    if reg != 0.0:
                        a[np.arange(n_block), np.arange(n_block)] += reg
                    try:
                        inv = np.linalg.inv(a)
                    except np.linalg.LinAlgError:
                        inv = np.linalg.pinv(a, rcond=1e-12)
                    if not np.all(np.isfinite(inv)):
                        inv = np.linalg.pinv(a, rcond=1e-12)
                    inv_block[l, :, :] = inv

                cached = _TransportXBlockPrecondCache(inv_xblock=jnp.asarray(inv_block, dtype=precond_dtype))
                _RHSMODE1_SXBLOCK_PRECOND_CACHE[cache_key] = cached

            inv_block = cached.inv_xblock  # (L, S*X, S*X)

            def _apply_full(r_full: jnp.ndarray) -> jnp.ndarray:
                r_full = jnp.asarray(r_full, dtype=precond_dtype)
                f = r_full[: op.f_size].reshape(op.fblock.f_shape)  # (S,X,L,T,Z)
                f_l = jnp.transpose(f, (2, 0, 1, 3, 4))  # (L,S,X,T,Z)
                f_l = f_l.reshape((int(op.n_xi), int(op.n_species) * int(op.n_x), int(op.n_theta), int(op.n_zeta)))
                z_l = jnp.einsum("lmn,lntz->lmtz", inv_block, f_l)
                z_l = z_l.reshape((int(op.n_xi), int(op.n_species), int(op.n_x), int(op.n_theta), int(op.n_zeta)))
                z_f = jnp.transpose(z_l, (1, 2, 0, 3, 4))  # (S,X,L,T,Z)
                tail = r_full[op.f_size :]
                z_full = jnp.concatenate([z_f.reshape((-1,)), tail], axis=0)
                return jnp.asarray(z_full, dtype=jnp.float64)

    elif use_xblock and op.fblock.fp is not None:
        cache_key = _transport_precond_cache_key(op, "rhs1_collision_xblock")
        cached = _RHSMODE1_XBLOCK_PRECOND_CACHE.get(cache_key)
        if cached is None:
            f_shape = op.fblock.f_shape
            n_species, n_x, n_l, _, _ = f_shape
            inv_xblock = np.zeros((n_species, n_l, n_x, n_x), dtype=np.float64)
            mat = np.asarray(op.fblock.fp.mat, dtype=np.float64)  # (S,S,L,X,X)
            reg_env = os.environ.get("SFINCS_JAX_RHSMODE1_BICGSTAB_PRECOND_REG", "").strip()
            try:
                reg = float(reg_env) if reg_env else 1e-10
            except ValueError:
                reg = 1e-10
            identity_shift = float(op.fblock.identity_shift)
            pas_diag = None
            if op.fblock.pas is not None:
                pas = op.fblock.pas
                l_arr = np.arange(n_l, dtype=np.float64)
                factor_l = 0.5 * (l_arr * (l_arr + 1.0) + 2.0 * float(pas.krook))
                pas_diag = float(pas.nu_n) * np.asarray(pas.nu_d_hat, dtype=np.float64)[:, :, None] * factor_l[None, None, :]

            for s in range(n_species):
                for l in range(n_l):
                    a = np.array(mat[s, s, l, :, :], dtype=np.float64, copy=True)
                    if identity_shift != 0.0:
                        a[np.arange(n_x), np.arange(n_x)] += identity_shift
                    if pas_diag is not None:
                        a[np.arange(n_x), np.arange(n_x)] += pas_diag[s, :, l]
                    if reg != 0.0:
                        a[np.arange(n_x), np.arange(n_x)] += reg
                    try:
                        inv = np.linalg.inv(a)
                    except np.linalg.LinAlgError:
                        inv = np.linalg.pinv(a, rcond=1e-12)
                    if not np.all(np.isfinite(inv)):
                        inv = np.linalg.pinv(a, rcond=1e-12)
                    inv_xblock[s, l, :, :] = inv

            cached = _TransportXBlockPrecondCache(inv_xblock=jnp.asarray(inv_xblock, dtype=precond_dtype))
            _RHSMODE1_XBLOCK_PRECOND_CACHE[cache_key] = cached

        inv_xblock = cached.inv_xblock

        def _apply_full(r_full: jnp.ndarray) -> jnp.ndarray:
            r_full = jnp.asarray(r_full, dtype=precond_dtype)
            f = r_full[: op.f_size].reshape(op.fblock.f_shape)  # (S,X,L,T,Z)
            f_sl = jnp.transpose(f, (0, 2, 1, 3, 4))  # (S,L,X,T,Z)
            z_sl = jnp.einsum("slij,sljtz->slitz", inv_xblock, f_sl)
            z_f = jnp.transpose(z_sl, (0, 2, 1, 3, 4))  # (S,X,L,T,Z)
            tail = r_full[op.f_size :]
            z_full = jnp.concatenate([z_f.reshape((-1,)), tail], axis=0)
            return jnp.asarray(z_full, dtype=jnp.float64)
    else:
        cache_key = _transport_precond_cache_key(op, "rhs1_collision_diag")
        cached = _RHSMODE1_DIAG_PRECOND_CACHE.get(cache_key)
        if cached is None:
            f_shape = op.fblock.f_shape
            n_species, n_x, n_l, _, _ = f_shape
            diag = jnp.zeros(f_shape, dtype=jnp.float64)

            if float(op.fblock.identity_shift) != 0.0:
                diag = diag + jnp.asarray(op.fblock.identity_shift, dtype=jnp.float64)

            if op.fblock.pas is not None:
                pas = op.fblock.pas
                l = jnp.arange(n_l, dtype=jnp.float64)
                factor_l = 0.5 * (l * (l + 1.0) + 2.0 * pas.krook)
                pas_diag = pas.nu_n * pas.nu_d_hat[:, :, None] * factor_l[None, None, :]
                diag = diag + pas_diag[:, :, :, None, None]

            if op.fblock.fp is not None:
                mat = op.fblock.fp.mat  # (S,S,L,X,X)
                diag_x = jnp.diagonal(mat, axis1=3, axis2=4)  # (S,S,L,X)
                diag_self = jnp.diagonal(diag_x, axis1=0, axis2=1)  # (L,X,S)
                diag_self = jnp.transpose(diag_self, (2, 1, 0))  # (S,X,L)
                diag = diag + diag_self[:, :, :, None, None]

            nxi_for_x = op.fblock.collisionless.n_xi_for_x.astype(jnp.int32)
            mask = jnp.arange(n_l, dtype=jnp.int32)[None, :] < nxi_for_x[:, None]  # (X,L)
            mask = mask[None, :, :, None, None]
            diag = jnp.where(mask, diag, jnp.asarray(1.0, dtype=jnp.float64))

            reg_env = os.environ.get("SFINCS_JAX_RHSMODE1_BICGSTAB_PRECOND_REG", "").strip()
            try:
                reg = float(reg_env) if reg_env else 1e-10
            except ValueError:
                reg = 1e-10
            inv_diag_f = 1.0 / (diag + float(reg))
            cached = _TransportPrecondCache(inv_diag_f=jnp.asarray(inv_diag_f, dtype=precond_dtype))
            _RHSMODE1_DIAG_PRECOND_CACHE[cache_key] = cached

        inv_diag_f = cached.inv_diag_f

        def _apply_full(r_full: jnp.ndarray) -> jnp.ndarray:
            r_full = jnp.asarray(r_full, dtype=precond_dtype)
            f = r_full[: op.f_size].reshape(op.fblock.f_shape)
            z_f = f * inv_diag_f
            tail = r_full[op.f_size :]
            z_full = jnp.concatenate([z_f.reshape((-1,)), tail], axis=0)
            return jnp.asarray(z_full, dtype=jnp.float64)

    if reduce_full is None or expand_reduced is None:
        return _apply_full

    def _apply_reduced(r_reduced: jnp.ndarray) -> jnp.ndarray:
        z_full = _apply_full(expand_reduced(r_reduced))
        return reduce_full(z_full)

    return _apply_reduced


@jtu.register_pytree_node_class
@dataclass(frozen=True)
class V3LinearSolveResult:
    """Result of a single matrix-free GMRES solve for the (currently supported) v3 full-system operator."""

    op: V3FullSystemOperator
    rhs: jnp.ndarray
    gmres: GMRESSolveResult

    def tree_flatten(self):
        children = (self.op, self.rhs, self.gmres)
        aux = None
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        del aux
        op, rhs, gmres_result = children
        return cls(op=op, rhs=rhs, gmres=gmres_result)

    @property
    def x(self) -> jnp.ndarray:
        return self.gmres.x

    @property
    def residual_norm(self) -> jnp.ndarray:
        return self.gmres.residual_norm


def _gmres_result_is_finite(res: GMRESSolveResult) -> bool:
    """Return True when GMRES returned finite state and residual."""
    return bool(jnp.all(jnp.isfinite(res.x)) and jnp.isfinite(res.residual_norm))


def _diag_only(m: jnp.ndarray) -> jnp.ndarray:
    """Return a diagonal-only copy of a square matrix."""
    return jnp.diag(jnp.diag(m))


def _build_rhsmode1_preconditioner_operator_point(op: V3FullSystemOperator) -> V3FullSystemOperator:
    """Return a simplified RHSMode=1 operator for point-block preconditioning.

    This is the original cheap RHSMode=1 preconditioner: it retains local x/L
    couplings and collisions while dropping theta/zeta derivative couplings
    (streaming, ExB, and magnetic-drift derivatives) by diagonalizing the derivative
    matrices.
    """
    if int(op.rhs_mode) != 1:
        return op

    fblock = op.fblock
    coll = replace(
        fblock.collisionless,
        ddtheta=_diag_only(fblock.collisionless.ddtheta),
        ddzeta=_diag_only(fblock.collisionless.ddzeta),
    )
    exb_theta = None if fblock.exb_theta is None else replace(
        fblock.exb_theta, ddtheta=_diag_only(fblock.exb_theta.ddtheta)
    )
    exb_zeta = None if fblock.exb_zeta is None else replace(
        fblock.exb_zeta, ddzeta=_diag_only(fblock.exb_zeta.ddzeta)
    )
    mag_theta = None
    if fblock.magdrift_theta is not None:
        mag_theta = replace(
            fblock.magdrift_theta,
            ddtheta_plus=_diag_only(fblock.magdrift_theta.ddtheta_plus),
            ddtheta_minus=_diag_only(fblock.magdrift_theta.ddtheta_minus),
        )
    mag_zeta = None
    if fblock.magdrift_zeta is not None:
        mag_zeta = replace(
            fblock.magdrift_zeta,
            ddzeta_plus=_diag_only(fblock.magdrift_zeta.ddzeta_plus),
            ddzeta_minus=_diag_only(fblock.magdrift_zeta.ddzeta_minus),
        )
    fblock_pc = replace(
        fblock,
        collisionless=coll,
        exb_theta=exb_theta,
        exb_zeta=exb_zeta,
        magdrift_theta=mag_theta,
        magdrift_zeta=mag_zeta,
    )
    return replace(op, fblock=fblock_pc)


def _build_transport_preconditioner_operator_point(op: V3FullSystemOperator) -> V3FullSystemOperator:
    """Return a simplified transport operator for point-block preconditioning.

    This mirrors `_build_rhsmode1_preconditioner_operator_point` but does not
    require RHSMode=1, since RHSMode=2/3 transport solves reuse the same operator
    structure with different right-hand sides.
    """
    fblock = op.fblock
    coll = replace(
        fblock.collisionless,
        ddtheta=_diag_only(fblock.collisionless.ddtheta),
        ddzeta=_diag_only(fblock.collisionless.ddzeta),
    )
    exb_theta = None if fblock.exb_theta is None else replace(
        fblock.exb_theta, ddtheta=_diag_only(fblock.exb_theta.ddtheta)
    )
    exb_zeta = None if fblock.exb_zeta is None else replace(
        fblock.exb_zeta, ddzeta=_diag_only(fblock.exb_zeta.ddzeta)
    )
    mag_theta = None
    if fblock.magdrift_theta is not None:
        mag_theta = replace(
            fblock.magdrift_theta,
            ddtheta_plus=_diag_only(fblock.magdrift_theta.ddtheta_plus),
            ddtheta_minus=_diag_only(fblock.magdrift_theta.ddtheta_minus),
        )
    mag_zeta = None
    if fblock.magdrift_zeta is not None:
        mag_zeta = replace(
            fblock.magdrift_zeta,
            ddzeta_plus=_diag_only(fblock.magdrift_zeta.ddzeta_plus),
            ddzeta_minus=_diag_only(fblock.magdrift_zeta.ddzeta_minus),
        )
    fblock_pc = replace(
        fblock,
        collisionless=coll,
        exb_theta=exb_theta,
        exb_zeta=exb_zeta,
        magdrift_theta=mag_theta,
        magdrift_zeta=mag_zeta,
    )
    return replace(op, fblock=fblock_pc)


def _build_rhsmode1_preconditioner_operator_theta_line(op: V3FullSystemOperator) -> V3FullSystemOperator:
    """Return a simplified RHSMode=1 operator for theta-line preconditioning.

    Keep full theta derivative couplings but drop zeta derivative couplings. This enables
    a significantly stronger preconditioner than point-block Jacobi, while remaining
    much cheaper than a full (theta,zeta)-coupled preconditioner.
    """
    if int(op.rhs_mode) != 1:
        return op

    fblock = op.fblock
    coll = replace(
        fblock.collisionless,
        ddzeta=_diag_only(fblock.collisionless.ddzeta),
    )
    exb_theta = fblock.exb_theta
    exb_zeta = None if fblock.exb_zeta is None else replace(
        fblock.exb_zeta, ddzeta=_diag_only(fblock.exb_zeta.ddzeta)
    )
    mag_theta = fblock.magdrift_theta
    mag_zeta = None
    if fblock.magdrift_zeta is not None:
        mag_zeta = replace(
            fblock.magdrift_zeta,
            ddzeta_plus=_diag_only(fblock.magdrift_zeta.ddzeta_plus),
            ddzeta_minus=_diag_only(fblock.magdrift_zeta.ddzeta_minus),
        )
    fblock_pc = replace(
        fblock,
        collisionless=coll,
        exb_theta=exb_theta,
        exb_zeta=exb_zeta,
        magdrift_theta=mag_theta,
        magdrift_zeta=mag_zeta,
    )
    return replace(op, fblock=fblock_pc)


def _build_rhsmode1_preconditioner_operator_zeta_line(op: V3FullSystemOperator) -> V3FullSystemOperator:
    """Return a simplified RHSMode=1 operator for zeta-line preconditioning.

    Keep full zeta derivative couplings but drop theta derivative couplings. This is the
    zeta-analog of `_build_rhsmode1_preconditioner_operator_theta_line`.
    """
    if int(op.rhs_mode) != 1:
        return op

    fblock = op.fblock
    coll = replace(
        fblock.collisionless,
        ddtheta=_diag_only(fblock.collisionless.ddtheta),
    )
    exb_theta = None if fblock.exb_theta is None else replace(
        fblock.exb_theta, ddtheta=_diag_only(fblock.exb_theta.ddtheta)
    )
    exb_zeta = fblock.exb_zeta
    mag_theta = None
    if fblock.magdrift_theta is not None:
        mag_theta = replace(
            fblock.magdrift_theta,
            ddtheta_plus=_diag_only(fblock.magdrift_theta.ddtheta_plus),
            ddtheta_minus=_diag_only(fblock.magdrift_theta.ddtheta_minus),
        )
    mag_zeta = fblock.magdrift_zeta
    fblock_pc = replace(
        fblock,
        collisionless=coll,
        exb_theta=exb_theta,
        exb_zeta=exb_zeta,
        magdrift_theta=mag_theta,
        magdrift_zeta=mag_zeta,
    )
    return replace(op, fblock=fblock_pc)


def _build_rhsmode1_block_preconditioner_xdiag(
    *,
    op: V3FullSystemOperator,
    reduce_full: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    expand_reduced: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    preconditioner_xi: int = 1,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Block preconditioner with x-diagonal blocks (per species, per x).

    This matches v3's `preconditioner_x=1` behavior by dropping x-couplings while
    retaining the requested xi coupling.
    """
    cache_key = _rhsmode1_precond_cache_key(op, f"point_xdiag_xi{int(preconditioner_xi)}")
    cached = _RHSMODE1_PRECOND_DIAGX_CACHE.get(cache_key)
    if cached is None:
        op_pc = _build_rhsmode1_preconditioner_operator_point(op)
        n_s = int(op.n_species)
        n_x = int(op.n_x)
        n_l = int(op.n_xi)
        n_t = int(op.n_theta)
        n_z = int(op.n_zeta)
        total = int(op.total_size)

        nxi_for_x = np.asarray(op.fblock.collisionless.n_xi_for_x, dtype=np.int32)
        max_l = int(np.max(nxi_for_x)) if nxi_for_x.size else 0
        precond_dtype = _precond_dtype(max_l * max_l)

        reg_env = os.environ.get("SFINCS_JAX_RHSMODE1_PRECOND_REG", "").strip()
        reg_val = float(reg_env) if reg_env else 1e-10
        reg = np.float64(reg_val)

        block_inv_list: list[list[jnp.ndarray]] = []
        idx_map_list: list[list[jnp.ndarray]] = []
        for s in range(n_s):
            inv_row: list[jnp.ndarray] = []
            idx_row: list[jnp.ndarray] = []
            for ix in range(n_x):
                max_lx = int(nxi_for_x[ix])
                if max_lx <= 0:
                    inv_row.append(jnp.zeros((0, 0), dtype=precond_dtype))
                    idx_row.append(jnp.zeros((n_t, n_z, 0), dtype=jnp.int32))
                    continue
                rep_idx = np.zeros((max_lx,), dtype=np.int32)
                for il in range(max_lx):
                    rep_idx[il] = int(((((s * n_x + ix) * n_l + il) * n_t + 0) * n_z + 0))
                chunk_cols = _precond_chunk_cols(total, int(rep_idx.shape[0]))
                y_sub = _matvec_submatrix(
                    op_pc,
                    col_idx=rep_idx,
                    row_idx=rep_idx,
                    total_size=total,
                    chunk_cols=chunk_cols,
                )
                a = np.asarray(y_sub.T, dtype=np.float64)
                if preconditioner_xi != 0:
                    a = np.diag(np.diag(a))
                a = a + reg * np.eye(max_lx, dtype=np.float64)
                try:
                    inv = np.linalg.inv(a)
                except np.linalg.LinAlgError:
                    inv = np.linalg.pinv(a, rcond=1e-12)
                if not np.all(np.isfinite(inv)):
                    inv = np.linalg.pinv(a, rcond=1e-12)
                inv_row.append(jnp.asarray(inv, dtype=precond_dtype))
                idx_map = np.zeros((n_t, n_z, max_lx), dtype=np.int32)
                for it in range(n_t):
                    for iz in range(n_z):
                        for il in range(max_lx):
                            idx_map[it, iz, il] = int(
                                ((((s * n_x + ix) * n_l + il) * n_t + it) * n_z + iz)
                            )
                idx_row.append(jnp.asarray(idx_map, dtype=jnp.int32))
            block_inv_list.append(inv_row)
            idx_map_list.append(idx_row)

        extra_start = int(op.f_size + op.phi1_size)
        extra_size = int(op.extra_size)
        extra_idx_np = np.arange(extra_start, extra_start + extra_size, dtype=np.int32)
        extra_idx_jnp = jnp.asarray(extra_idx_np, dtype=jnp.int32)
        extra_inv_jnp: jnp.ndarray | None = None
        if extra_size > 0:
            chunk_cols = _precond_chunk_cols(total, int(extra_idx_np.shape[0]))
            y_sub = _matvec_submatrix(
                op_pc,
                col_idx=extra_idx_np,
                row_idx=extra_idx_np,
                total_size=total,
                chunk_cols=chunk_cols,
            )
            ee = np.asarray(y_sub.T, dtype=np.float64)
            ee = ee + reg * np.eye(extra_size, dtype=np.float64)
            try:
                ee_inv = np.linalg.inv(ee)
            except np.linalg.LinAlgError:
                ee_inv = np.linalg.pinv(ee, rcond=1e-12)
            if not np.all(np.isfinite(ee_inv)):
                ee_inv = np.linalg.pinv(ee, rcond=1e-12)
            extra_inv_jnp = jnp.asarray(ee_inv, dtype=precond_dtype)

        cached = _RHSMode1PrecondDiagXCache(
            block_inv_list=tuple(tuple(row) for row in block_inv_list),
            idx_map_list=tuple(tuple(row) for row in idx_map_list),
            extra_idx_jnp=extra_idx_jnp,
            extra_inv_jnp=extra_inv_jnp,
        )
        _RHSMODE1_PRECOND_DIAGX_CACHE[cache_key] = cached

    block_inv_list = cached.block_inv_list
    idx_map_list = cached.idx_map_list
    extra_idx_jnp = cached.extra_idx_jnp
    extra_inv_jnp = cached.extra_inv_jnp
    n_s = int(op.n_species)
    n_x = int(op.n_x)

    def _apply_full(r_full: jnp.ndarray) -> jnp.ndarray:
        r_full = jnp.asarray(r_full, dtype=block_inv_list[0][0].dtype if n_s > 0 and n_x > 0 else _precond_dtype())
        z_full = jnp.zeros_like(r_full)
        for s in range(n_s):
            for ix in range(n_x):
                inv = block_inv_list[s][ix]
                idx_map = idx_map_list[s][ix]
                if idx_map.size == 0:
                    continue
                r_loc = r_full[idx_map].reshape((int(op.n_theta), int(op.n_zeta), int(inv.shape[0])))
                z_loc = jnp.einsum("ij,tzj->tzi", inv, r_loc)
                z_full = z_full.at[idx_map].set(z_loc, unique_indices=True)
        if extra_inv_jnp is not None:
            r_extra = r_full[extra_idx_jnp]
            z_extra = extra_inv_jnp @ r_extra
            z_full = z_full.at[extra_idx_jnp].set(z_extra, unique_indices=True)
        return jnp.asarray(z_full, dtype=jnp.float64)

    if reduce_full is None or expand_reduced is None:
        return _apply_full

    def _apply_reduced(r_reduced: jnp.ndarray) -> jnp.ndarray:
        z_full = _apply_full(expand_reduced(r_reduced))
        return reduce_full(z_full)

    return _apply_reduced


def _build_rhsmode1_block_preconditioner(
    *,
    op: V3FullSystemOperator,
    reduce_full: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    expand_reduced: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    preconditioner_species: int = 1,
    preconditioner_x: int = 1,
    preconditioner_xi: int = 1,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Build a PETSc-like block preconditioner for RHSMode=1 solves.

    Structure:
    - x/L local block solve per species at each (theta,zeta), using a representative
      per-species block matrix from a simplified operator.
    - explicit extra/source-row solve via a dense small block.
    """
    if int(preconditioner_species) == 0:
        return _build_rhsmode1_species_xblock_preconditioner(op=op, reduce_full=reduce_full, expand_reduced=expand_reduced)
    if int(preconditioner_x) == 1:
        return _build_rhsmode1_block_preconditioner_xdiag(
            op=op, reduce_full=reduce_full, expand_reduced=expand_reduced, preconditioner_xi=preconditioner_xi
        )
    cache_key = _rhsmode1_precond_cache_key(op, "point")
    precond_dtype = _precond_dtype()
    cached = _RHSMODE1_PRECOND_CACHE.get(cache_key)
    if cached is None:
        op_pc = _build_rhsmode1_preconditioner_operator_point(op)
        n_s = int(op.n_species)
        n_x = int(op.n_x)
        n_l = int(op.n_xi)
        n_t = int(op.n_theta)
        n_z = int(op.n_zeta)
        total = int(op.total_size)

        nxi_for_x = np.asarray(op.fblock.collisionless.n_xi_for_x, dtype=np.int32)
        local_per_species = int(np.sum(nxi_for_x))

        # Representative local x/L blocks at (theta,zeta)=(0,0), one per species.
        rep_indices_by_species: list[np.ndarray] = []
        for s in range(n_s):
            idx: list[int] = []
            for ix in range(n_x):
                max_l = int(nxi_for_x[ix])
                for il in range(max_l):
                    f_idx = ((((s * n_x + ix) * n_l + il) * n_t + 0) * n_z + 0)
                    idx.append(int(f_idx))
            rep_indices_by_species.append(np.asarray(idx, dtype=np.int32))

        reg_env = os.environ.get("SFINCS_JAX_RHSMODE1_PRECOND_REG", "").strip()
        reg_val = float(reg_env) if reg_env else 1e-10
        reg = np.float64(reg_val)

        block_inv = np.zeros((n_s, local_per_species, local_per_species), dtype=np.float64)
        for s in range(n_s):
            rep_idx = rep_indices_by_species[s]
            chunk_cols = _precond_chunk_cols(total, int(np.asarray(rep_idx).shape[0]))
            y_sub = _matvec_submatrix(
                op_pc,
                col_idx=rep_idx,
                row_idx=rep_idx,
                total_size=total,
                chunk_cols=chunk_cols,
            )
            a = np.asarray(y_sub.T, dtype=np.float64)
            a = a + reg * np.eye(local_per_species, dtype=np.float64)
            try:
                inv = np.linalg.inv(a)
            except np.linalg.LinAlgError:
                inv = np.linalg.pinv(a, rcond=1e-12)
            if not np.all(np.isfinite(inv)):
                inv = np.linalg.pinv(a, rcond=1e-12)
            block_inv[s, :, :] = inv

        # Build per-(s,theta,zeta) gather map for active x/L rows.
        idx_map = np.zeros((n_s, n_t, n_z, local_per_species), dtype=np.int32)
        for s in range(n_s):
            for it in range(n_t):
                for iz in range(n_z):
                    k = 0
                    for ix in range(n_x):
                        max_l = int(nxi_for_x[ix])
                        for il in range(max_l):
                            idx_map[s, it, iz, k] = int(
                                ((((s * n_x + ix) * n_l + il) * n_t + it) * n_z + iz)
                            )
                            k += 1

        idx_map_jnp = jnp.asarray(idx_map, dtype=jnp.int32)
        flat_idx_jnp = idx_map_jnp.reshape((-1,))
        block_inv_jnp = jnp.asarray(block_inv, dtype=precond_dtype)

        extra_start = int(op.f_size + op.phi1_size)
        extra_size = int(op.extra_size)
        extra_idx_np = np.arange(extra_start, extra_start + extra_size, dtype=np.int32)
        extra_idx_jnp = jnp.asarray(extra_idx_np, dtype=jnp.int32)
        extra_inv_jnp: jnp.ndarray | None = None
        if extra_size > 0:
            chunk_cols = _precond_chunk_cols(total, int(extra_idx_np.shape[0]))
            y_sub = _matvec_submatrix(
                op_pc,
                col_idx=extra_idx_np,
                row_idx=extra_idx_np,
                total_size=total,
                chunk_cols=chunk_cols,
            )
            ee = np.asarray(y_sub.T, dtype=np.float64)
            ee = ee + reg * np.eye(extra_size, dtype=np.float64)
            try:
                ee_inv = np.linalg.inv(ee)
            except np.linalg.LinAlgError:
                ee_inv = np.linalg.pinv(ee, rcond=1e-12)
            if not np.all(np.isfinite(ee_inv)):
                ee_inv = np.linalg.pinv(ee, rcond=1e-12)
            extra_inv_jnp = jnp.asarray(ee_inv, dtype=precond_dtype)

        cached = _RHSMode1PrecondCache(
            idx_map_jnp=idx_map_jnp,
            flat_idx_jnp=flat_idx_jnp,
            block_inv_jnp=block_inv_jnp,
            extra_idx_jnp=extra_idx_jnp,
            extra_inv_jnp=extra_inv_jnp,
        )
        _RHSMODE1_PRECOND_CACHE[cache_key] = cached

    n_s = int(op.n_species)
    n_t = int(op.n_theta)
    n_z = int(op.n_zeta)
    local_per_species = int(cached.block_inv_jnp.shape[-1])
    flat_idx_jnp = cached.flat_idx_jnp
    block_inv_jnp = cached.block_inv_jnp
    extra_idx_jnp = cached.extra_idx_jnp
    extra_inv_jnp = cached.extra_inv_jnp

    def _apply_full(r_full: jnp.ndarray) -> jnp.ndarray:
        r_full = jnp.asarray(r_full, dtype=precond_dtype)
        r_loc = r_full[flat_idx_jnp].reshape((n_s, n_t, n_z, local_per_species))
        z_loc = jnp.einsum("sab,stzb->stza", block_inv_jnp, r_loc)
        z_full = jnp.zeros_like(r_full)
        z_full = z_full.at[flat_idx_jnp].set(z_loc.reshape((-1,)), unique_indices=True)
        if extra_inv_jnp is not None:
            r_extra = r_full[extra_idx_jnp]
            z_extra = extra_inv_jnp @ r_extra
            z_full = z_full.at[extra_idx_jnp].set(z_extra, unique_indices=True)
        return jnp.asarray(z_full, dtype=jnp.float64)

    if reduce_full is None or expand_reduced is None:
        return _apply_full

    def _apply_reduced(r_reduced: jnp.ndarray) -> jnp.ndarray:
        z_full = _apply_full(expand_reduced(r_reduced))
        return reduce_full(z_full)

    return _apply_reduced


def _build_rhsmode23_block_preconditioner(
    *,
    op: V3FullSystemOperator,
    reduce_full: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    expand_reduced: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Build a block-Jacobi preconditioner for RHSMode=2/3 transport solves.

    Uses the same local x/L block structure as RHSMode=1 preconditioning, but
    applies it to the transport operator (RHSMode=2/3) using a simplified
    operator with diagonalized theta/zeta derivatives.
    """
    cache_key = _transport_precond_cache_key(op, "block")
    precond_dtype = _precond_dtype()
    cached = _RHSMODE23_PRECOND_CACHE.get(cache_key)
    if cached is None:
        op_pc = _build_transport_preconditioner_operator_point(op)
        n_s = int(op.n_species)
        n_x = int(op.n_x)
        n_l = int(op.n_xi)
        n_t = int(op.n_theta)
        n_z = int(op.n_zeta)
        total = int(op.total_size)

        nxi_for_x = np.asarray(op.fblock.collisionless.n_xi_for_x, dtype=np.int32)
        local_per_species = int(np.sum(nxi_for_x))

        rep_indices_by_species: list[np.ndarray] = []
        for s in range(n_s):
            idx: list[int] = []
            for ix in range(n_x):
                max_l = int(nxi_for_x[ix])
                for il in range(max_l):
                    f_idx = ((((s * n_x + ix) * n_l + il) * n_t + 0) * n_z + 0)
                    idx.append(int(f_idx))
            rep_indices_by_species.append(np.asarray(idx, dtype=np.int32))

        reg_env = os.environ.get("SFINCS_JAX_TRANSPORT_PRECOND_BLOCK_REG", "").strip()
        if not reg_env:
            reg_env = os.environ.get("SFINCS_JAX_TRANSPORT_PRECOND_REG", "").strip()
        try:
            reg = float(reg_env) if reg_env else 1e-10
        except ValueError:
            reg = 1e-10

        block_inv = np.zeros((n_s, local_per_species, local_per_species), dtype=np.float64)
        for s in range(n_s):
            rep_idx = rep_indices_by_species[s]
            chunk_cols = _precond_chunk_cols(total, int(np.asarray(rep_idx).shape[0]))
            y_sub = _matvec_submatrix(
                op_pc,
                col_idx=rep_idx,
                row_idx=rep_idx,
                total_size=total,
                chunk_cols=chunk_cols,
            )
            a = np.asarray(y_sub.T, dtype=np.float64)
            a = a + reg * np.eye(local_per_species, dtype=np.float64)
            try:
                inv = np.linalg.inv(a)
            except np.linalg.LinAlgError:
                inv = np.linalg.pinv(a, rcond=1e-12)
            if not np.all(np.isfinite(inv)):
                inv = np.linalg.pinv(a, rcond=1e-12)
            block_inv[s, :, :] = inv

        idx_map = np.zeros((n_s, n_t, n_z, local_per_species), dtype=np.int32)
        for s in range(n_s):
            for it in range(n_t):
                for iz in range(n_z):
                    k = 0
                    for ix in range(n_x):
                        max_l = int(nxi_for_x[ix])
                        for il in range(max_l):
                            idx_map[s, it, iz, k] = int(
                                ((((s * n_x + ix) * n_l + il) * n_t + it) * n_z + iz)
                            )
                            k += 1

        idx_map_jnp = jnp.asarray(idx_map, dtype=jnp.int32)
        flat_idx_jnp = idx_map_jnp.reshape((-1,))
        block_inv_jnp = jnp.asarray(block_inv, dtype=precond_dtype)

        extra_start = int(op.f_size + op.phi1_size)
        extra_size = int(op.extra_size)
        extra_idx_np = np.arange(extra_start, extra_start + extra_size, dtype=np.int32)
        extra_idx_jnp = jnp.asarray(extra_idx_np, dtype=jnp.int32)
        extra_inv_jnp: jnp.ndarray | None = None
        if extra_size > 0:
            chunk_cols = _precond_chunk_cols(total, int(extra_idx_np.shape[0]))
            y_sub = _matvec_submatrix(
                op_pc,
                col_idx=extra_idx_np,
                row_idx=extra_idx_np,
                total_size=total,
                chunk_cols=chunk_cols,
            )
            a = np.asarray(y_sub.T, dtype=np.float64)
            a = a + reg * np.eye(extra_size, dtype=np.float64)
            try:
                extra_inv = np.linalg.inv(a)
            except np.linalg.LinAlgError:
                extra_inv = np.linalg.pinv(a, rcond=1e-12)
            if not np.all(np.isfinite(extra_inv)):
                extra_inv = np.linalg.pinv(a, rcond=1e-12)
            extra_inv_jnp = jnp.asarray(extra_inv, dtype=precond_dtype)

        cached = _RHSMode1PrecondCache(
            idx_map_jnp=idx_map_jnp,
            flat_idx_jnp=flat_idx_jnp,
            block_inv_jnp=block_inv_jnp,
            extra_idx_jnp=extra_idx_jnp,
            extra_inv_jnp=extra_inv_jnp,
        )
        _RHSMODE23_PRECOND_CACHE[cache_key] = cached

    idx_map_jnp = cached.idx_map_jnp
    flat_idx_jnp = cached.flat_idx_jnp
    block_inv_jnp = cached.block_inv_jnp
    extra_idx_jnp = cached.extra_idx_jnp
    extra_inv_jnp = cached.extra_inv_jnp
    n_s = int(op.n_species)
    n_t = int(op.n_theta)
    n_z = int(op.n_zeta)
    local_per_species = int(idx_map_jnp.shape[-1])

    def _apply_full(r_full: jnp.ndarray) -> jnp.ndarray:
        r_full = jnp.asarray(r_full, dtype=precond_dtype)
        r_loc = r_full[flat_idx_jnp].reshape((n_s, n_t, n_z, local_per_species))
        z_loc = jnp.einsum("sab,stzb->stza", block_inv_jnp, r_loc)
        z_full = jnp.zeros_like(r_full)
        z_full = z_full.at[flat_idx_jnp].set(z_loc.reshape((-1,)), unique_indices=True)
        if extra_inv_jnp is not None:
            r_extra = r_full[extra_idx_jnp]
            z_extra = extra_inv_jnp @ r_extra
            z_full = z_full.at[extra_idx_jnp].set(z_extra, unique_indices=True)
        return jnp.asarray(z_full, dtype=jnp.float64)

    if reduce_full is None or expand_reduced is None:
        return _apply_full

    def _apply_reduced(r_reduced: jnp.ndarray) -> jnp.ndarray:
        z_full = _apply_full(expand_reduced(r_reduced))
        return reduce_full(z_full)

    return _apply_reduced


def _build_rhsmode1_schur_preconditioner(
    *,
    op: V3FullSystemOperator,
    reduce_full: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    expand_reduced: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Approximate Schur-complement preconditioner for constraintScheme=2 RHSMode=1 solves."""
    precond_dtype = _precond_dtype()
    species_block_max_env = os.environ.get("SFINCS_JAX_RHSMODE1_SPECIES_BLOCK_MAX", "").strip()
    try:
        species_block_max = int(species_block_max_env) if species_block_max_env else 1600
    except ValueError:
        species_block_max = 1600
    base_kind_env = os.environ.get("SFINCS_JAX_RHSMODE1_SCHUR_BASE", "").strip().lower()
    if base_kind_env in {"theta", "theta_line", "line_theta"}:
        base_kind = "theta_line"
    elif base_kind_env in {"zeta", "zeta_line", "line_zeta"}:
        base_kind = "zeta_line"
    elif base_kind_env in {"adi", "adi_line", "theta_zeta", "zeta_theta"}:
        base_kind = "adi"
    elif base_kind_env in {"species", "species_block", "speciesblock"}:
        base_kind = "species_block"
    elif base_kind_env in {"sxblock_tz", "sxblock_theta_zeta", "species_xblock_tz", "sx_tz"}:
        base_kind = "sxblock_tz"
    elif base_kind_env in {"xblock_tz", "xblock", "x_tz", "xtz", "xblock_theta_zeta"}:
        base_kind = "xblock_tz"
    elif base_kind_env in {"point", "block", "jacobi"}:
        base_kind = "point"
    else:
        nxi_for_x = np.asarray(op.fblock.collisionless.n_xi_for_x, dtype=np.int32)
        max_l = int(np.max(nxi_for_x)) if nxi_for_x.size else 0
        local_per_species = int(np.sum(nxi_for_x))
        dke_size = int(local_per_species * int(op.n_theta) * int(op.n_zeta))
        if int(op.n_theta) > 1 or int(op.n_zeta) > 1:
            tz_max_env = os.environ.get("SFINCS_JAX_RHSMODE1_TZ_PRECOND_MAX", "").strip()
            try:
                tz_max = int(tz_max_env) if tz_max_env else 128
            except ValueError:
                tz_max = 128
            xblock_tz_max_env = os.environ.get("SFINCS_JAX_RHSMODE1_XBLOCK_TZ_MAX", "").strip()
            try:
                xblock_tz_max = int(xblock_tz_max_env) if xblock_tz_max_env else 1200
            except ValueError:
                xblock_tz_max = 1200
            if (
                op.fblock.pas is not None
                and int(op.n_theta) > 1
                and int(op.n_zeta) > 1
                and species_block_max > 0
                and dke_size <= species_block_max
            ):
                base_kind = "species_block"
            elif (
                op.fblock.pas is not None
                and int(op.n_theta) > 1
                and int(op.n_zeta) > 1
                and xblock_tz_max > 0
                and int(max_l) * int(op.n_theta) * int(op.n_zeta) <= xblock_tz_max
            ):
                base_kind = "xblock_tz"
            elif (
                op.fblock.pas is not None
                and int(op.n_theta) > 1
                and int(op.n_zeta) > 1
                and int(op.n_theta) * int(op.n_zeta) <= tz_max
            ):
                base_kind = "theta_zeta"
            else:
                base_kind = "theta_line" if int(op.n_theta) >= int(op.n_zeta) else "zeta_line"
        else:
            base_kind = "point"

    if base_kind == "theta_line":
        base_precond = _build_rhsmode1_theta_line_preconditioner(op=op)
    elif base_kind == "species_block":
        base_precond = _build_rhsmode1_species_block_preconditioner(op=op)
    elif base_kind == "sxblock_tz":
        base_precond = _build_rhsmode1_sxblock_tz_preconditioner(op=op)
    elif base_kind == "xblock_tz":
        base_precond = _build_rhsmode1_xblock_tz_preconditioner(op=op)
    elif base_kind == "theta_zeta":
        base_precond = _build_rhsmode1_theta_zeta_preconditioner(op=op)
    elif base_kind == "zeta_line":
        base_precond = _build_rhsmode1_zeta_line_preconditioner(op=op)
    elif base_kind == "adi":
        pre_theta = _build_rhsmode1_theta_line_preconditioner(op=op)
        pre_zeta = _build_rhsmode1_zeta_line_preconditioner(op=op)
        sweeps_env = os.environ.get("SFINCS_JAX_RHSMODE1_ADI_SWEEPS", "").strip()
        try:
            sweeps = int(sweeps_env) if sweeps_env else 2
        except ValueError:
            sweeps = 2
        sweeps = max(1, sweeps)

        def base_precond(v: jnp.ndarray) -> jnp.ndarray:
            out = v
            for _ in range(sweeps):
                out = pre_zeta(pre_theta(out))
            return out
    else:
        base_precond = _build_rhsmode1_block_preconditioner(op=op)
    f_size = int(op.f_size)
    extra_size = int(op.extra_size)
    n_species = int(op.n_species)
    n_x = int(op.n_x)
    n_constraints = int(extra_size)

    def _schur_inv_diag() -> jnp.ndarray:
        cache_key = _rhsmode1_precond_cache_key(op, "schur_diag")
        cached = _RHSMODE1_SCHUR_CACHE.get(cache_key)
        if cached is not None:
            return cached
        # Ensure base block preconditioner cache exists.
        _build_rhsmode1_block_preconditioner(
            op=op,
            preconditioner_species=1,
            preconditioner_x=0,
            preconditioner_xi=1,
        )
        block_key = _rhsmode1_precond_cache_key(op, "point")
        block_cached = _RHSMODE1_PRECOND_CACHE.get(block_key)
        if block_cached is None:
            raise RuntimeError("Schur preconditioner requires block preconditioner cache.")
        block_inv = np.asarray(block_cached.block_inv_jnp, dtype=np.float64)
        nxi_for_x = np.asarray(op.fblock.collisionless.n_xi_for_x, dtype=np.int32)
        offsets = np.concatenate([[0], np.cumsum(nxi_for_x)])
        idx = offsets[:-1]
        diag = np.zeros((n_species, n_x), dtype=np.float64)
        for s in range(n_species):
            diag[s, :] = block_inv[s, idx, idx]
        theta_w = np.asarray(op.theta_weights, dtype=np.float64)
        zeta_w = np.asarray(op.zeta_weights, dtype=np.float64)
        d_hat = np.asarray(op.d_hat, dtype=np.float64)
        fs_sum = float(np.sum((theta_w[:, None] * zeta_w[None, :]) / d_hat))
        diag = diag * fs_sum
        eps_env = os.environ.get("SFINCS_JAX_RHSMODE1_SCHUR_EPS", "").strip()
        try:
            eps = float(eps_env) if eps_env else 1e-14
        except ValueError:
            eps = 1e-14
        inv_diag = np.zeros_like(diag)
        mask = np.abs(diag) > eps
        inv_diag[mask] = 1.0 / diag[mask]
        ix0 = _ix_min(bool(op.point_at_x0))
        if ix0 > 0:
            inv_diag[:, :ix0] = 0.0
        inv_diag_jnp = jnp.asarray(inv_diag, dtype=precond_dtype)
        _RHSMODE1_SCHUR_CACHE[cache_key] = inv_diag_jnp
        return inv_diag_jnp

    def _schur_inv_full() -> jnp.ndarray:
        cache_key = _rhsmode1_precond_cache_key(op, f"schur_full_{n_constraints}")
        cached = _RHSMODE1_SCHUR_CACHE.get(cache_key)
        if cached is not None:
            return cached
        # Build a dense approximate Schur complement: S ~= C M^{-1} B.
        # M^{-1} is the block preconditioner for the f-block.
        if n_constraints <= 0:
            inv = np.zeros((0, 0), dtype=np.float64)
            inv_jnp = jnp.asarray(inv, dtype=jnp.float64)
            _RHSMODE1_SCHUR_CACHE[cache_key] = inv_jnp
            return inv_jnp
        zeros_e = jnp.zeros((extra_size,), dtype=jnp.float64)
        s_mat = np.zeros((n_constraints, n_constraints), dtype=np.float64)
        # Build columns of S by applying M^{-1} to constraint injections.
        for j in range(n_constraints):
            basis = np.zeros((n_species, n_x), dtype=np.float64)
            basis.reshape(-1)[j] = 1.0
            f_src = _constraint_scheme2_inject_source(op, basis).reshape((-1,))
            y_full = base_precond(jnp.concatenate([jnp.asarray(f_src, dtype=jnp.float64), zeros_e], axis=0))
            y_f = np.asarray(y_full[:f_size], dtype=np.float64).reshape(op.fblock.f_shape)
            c_y = np.asarray(_constraint_scheme2_source_from_f(op, y_f), dtype=np.float64).reshape((-1,))
            s_mat[:, j] = c_y
        reg_env = os.environ.get("SFINCS_JAX_RHSMODE1_SCHUR_REG", "").strip()
        try:
            reg = float(reg_env) if reg_env else 1e-12
        except ValueError:
            reg = 1e-12
        s_mat = s_mat + reg * np.eye(n_constraints, dtype=np.float64)
        try:
            s_inv = np.linalg.inv(s_mat)
        except np.linalg.LinAlgError:
            s_inv = np.linalg.pinv(s_mat, rcond=1e-12)
        if not np.all(np.isfinite(s_inv)):
            s_inv = np.linalg.pinv(s_mat, rcond=1e-12)
        s_inv_jnp = jnp.asarray(s_inv, dtype=precond_dtype)
        _RHSMODE1_SCHUR_CACHE[cache_key] = s_inv_jnp
        return s_inv_jnp

    schur_mode_env = os.environ.get("SFINCS_JAX_RHSMODE1_SCHUR_MODE", "").strip().lower()
    if schur_mode_env in {"full", "dense"}:
        schur_mode = "full"
    elif schur_mode_env in {"diag", "diagonal"}:
        schur_mode = "diag"
    elif schur_mode_env in {"auto", ""}:
        schur_mode = "auto"
    else:
        schur_mode = "diag"
    schur_full_max_env = os.environ.get("SFINCS_JAX_RHSMODE1_SCHUR_FULL_MAX", "").strip()
    try:
        schur_full_max = int(schur_full_max_env) if schur_full_max_env else 256
    except ValueError:
        schur_full_max = 256
    use_full_schur = bool(schur_mode == "full" or (schur_mode == "auto" and n_constraints <= schur_full_max))

    inv_diag_cached = _schur_inv_diag()
    inv_schur_cached = _schur_inv_full() if use_full_schur else None

    def _apply_full(r_full: jnp.ndarray) -> jnp.ndarray:
        r_full = jnp.asarray(r_full, dtype=jnp.float64)
        if int(op.rhs_mode) != 1 or int(op.constraint_scheme) != 2 or int(op.phi1_size) != 0 or extra_size == 0:
            return base_precond(r_full)
        r_f = r_full[:f_size]
        r_e = r_full[f_size:]
        zeros_e = jnp.zeros((extra_size,), dtype=r_full.dtype)
        y_full = base_precond(jnp.concatenate([r_f, zeros_e], axis=0))
        y_f = y_full[:f_size]
        f = y_f.reshape(op.fblock.f_shape)
        c_y = _constraint_scheme2_source_from_f(op, f).reshape((-1,))
        if use_full_schur and inv_schur_cached is not None:
            x_e = inv_schur_cached @ (c_y - r_e)
        else:
            inv_diag = inv_diag_cached.reshape((-1,))
            x_e = (c_y - r_e) * inv_diag
        f_corr = _constraint_scheme2_inject_source(op, x_e.reshape((n_species, n_x)))
        r_corr = r_f - f_corr
        y_corr = base_precond(jnp.concatenate([r_corr, zeros_e], axis=0))
        x_f = y_corr[:f_size]
        return jnp.concatenate([x_f, x_e], axis=0)

    if reduce_full is None or expand_reduced is None:
        return _apply_full

    def _apply_reduced(r_reduced: jnp.ndarray) -> jnp.ndarray:
        z_full = _apply_full(expand_reduced(r_reduced))
        return reduce_full(z_full)

    return _apply_reduced


def _build_rhsmode1_theta_line_preconditioner(
    *,
    op: V3FullSystemOperator,
    reduce_full: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    expand_reduced: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Build a stronger RHSMode=1 preconditioner using theta-line blocks.

    For each species and each zeta plane, solve a representative block that couples
    all theta points for all local (x,L) unknowns at that zeta. This approximates the
    dominant streaming/mirror couplings along theta while ignoring zeta derivatives.
    """
    cache_key = _rhsmode1_precond_cache_key(op, "theta_line")
    precond_dtype = _precond_dtype()
    cached = _RHSMODE1_PRECOND_CACHE.get(cache_key)
    if cached is None:
        op_pc = _build_rhsmode1_preconditioner_operator_theta_line(op)
        n_species = int(op.n_species)
        n_x = int(op.n_x)
        n_l = int(op.n_xi)
        n_theta = int(op.n_theta)
        n_zeta = int(op.n_zeta)
        total_size = int(op.total_size)

        nxi_for_x = np.asarray(op.fblock.collisionless.n_xi_for_x, dtype=np.int32)
        local_per_species = int(np.sum(nxi_for_x))
        line_size = int(n_theta * local_per_species)

        reg_env = os.environ.get("SFINCS_JAX_RHSMODE1_PRECOND_REG", "").strip()
        reg_val = float(reg_env) if reg_env else 1e-10
        reg = np.float64(reg_val)

        # Build per-(species,zeta) gather map for all theta points and local (x,L) indices.
        idx_map = np.zeros((n_species, n_zeta, line_size), dtype=np.int32)
        for s in range(n_species):
            for iz in range(n_zeta):
                k = 0
                for it in range(n_theta):
                    for ix in range(n_x):
                        max_l = int(nxi_for_x[ix])
                        for il in range(max_l):
                            idx_map[s, iz, k] = int(
                                ((((s * n_x + ix) * n_l + il) * n_theta + it) * n_zeta + iz)
                            )
                            k += 1

        idx_map_jnp = jnp.asarray(idx_map, dtype=jnp.int32)
        flat_idx_jnp = idx_map_jnp.reshape((-1,))

        # Invert a theta-line block for each zeta plane.
        block_inv = np.zeros((n_species, n_zeta, line_size, line_size), dtype=np.float64)
        for s in range(n_species):
            for iz in range(n_zeta):
                rep_idx = np.asarray(idx_map[s, iz, :], dtype=np.int32)
                chunk_cols = _precond_chunk_cols(total_size, int(rep_idx.shape[0]))
                y_sub = _matvec_submatrix(
                    op_pc,
                    col_idx=rep_idx,
                    row_idx=rep_idx,
                    total_size=total_size,
                    chunk_cols=chunk_cols,
                )
                a = np.asarray(y_sub.T, dtype=np.float64)  # (line_size, line_size)
                a = a + reg * np.eye(line_size, dtype=np.float64)
                try:
                    inv = np.linalg.inv(a)
                except np.linalg.LinAlgError:
                    inv = np.linalg.pinv(a, rcond=1e-12)
                if not np.all(np.isfinite(inv)):
                    inv = np.linalg.pinv(a, rcond=1e-12)
                block_inv[s, iz, :, :] = inv
        block_inv_jnp = jnp.asarray(block_inv, dtype=precond_dtype)

        extra_start = int(op.f_size + op.phi1_size)
        extra_size = int(op.extra_size)
        extra_idx_np = np.arange(extra_start, extra_start + extra_size, dtype=np.int32)
        extra_idx_jnp = jnp.asarray(extra_idx_np, dtype=jnp.int32)
        extra_inv_jnp: jnp.ndarray | None = None
        if extra_size > 0:
            chunk_cols = _precond_chunk_cols(total_size, int(extra_idx_np.shape[0]))
            y_sub = _matvec_submatrix(
                op_pc,
                col_idx=extra_idx_np,
                row_idx=extra_idx_np,
                total_size=total_size,
                chunk_cols=chunk_cols,
            )
            ee = np.asarray(y_sub.T, dtype=np.float64)
            ee = ee + reg * np.eye(extra_size, dtype=np.float64)
            try:
                ee_inv = np.linalg.inv(ee)
            except np.linalg.LinAlgError:
                ee_inv = np.linalg.pinv(ee, rcond=1e-12)
            if not np.all(np.isfinite(ee_inv)):
                ee_inv = np.linalg.pinv(ee, rcond=1e-12)
            extra_inv_jnp = jnp.asarray(ee_inv, dtype=precond_dtype)

        cached = _RHSMode1PrecondCache(
            idx_map_jnp=idx_map_jnp,
            flat_idx_jnp=flat_idx_jnp,
            block_inv_jnp=block_inv_jnp,
            extra_idx_jnp=extra_idx_jnp,
            extra_inv_jnp=extra_inv_jnp,
        )
        _RHSMODE1_PRECOND_CACHE[cache_key] = cached

    n_species = int(op.n_species)
    n_zeta = int(op.n_zeta)
    line_size = int(cached.block_inv_jnp.shape[-1])
    flat_idx_jnp = cached.flat_idx_jnp
    block_inv_jnp = cached.block_inv_jnp
    extra_idx_jnp = cached.extra_idx_jnp
    extra_inv_jnp = cached.extra_inv_jnp

    def _apply_full(r_full: jnp.ndarray) -> jnp.ndarray:
        r_full = jnp.asarray(r_full, dtype=precond_dtype)
        r_loc = r_full[flat_idx_jnp].reshape((n_species, n_zeta, line_size))
        z_loc = jnp.einsum("szab,szb->sza", block_inv_jnp, r_loc)
        z_full = jnp.zeros_like(r_full)
        z_full = z_full.at[flat_idx_jnp].set(z_loc.reshape((-1,)), unique_indices=True)
        if extra_inv_jnp is not None:
            r_extra = r_full[extra_idx_jnp]
            z_extra = extra_inv_jnp @ r_extra
            z_full = z_full.at[extra_idx_jnp].set(z_extra, unique_indices=True)
        return jnp.asarray(z_full, dtype=jnp.float64)

    if reduce_full is None or expand_reduced is None:
        return _apply_full

    def _apply_reduced(r_reduced: jnp.ndarray) -> jnp.ndarray:
        z_full = _apply_full(expand_reduced(r_reduced))
        return reduce_full(z_full)

    return _apply_reduced


def _build_rhsmode1_theta_zeta_preconditioner(
    *,
    op: V3FullSystemOperator,
    reduce_full: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    expand_reduced: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Build a RHSMode=1 preconditioner using full (theta,zeta) blocks.

    For each species and each active (x,L) pair, solve the full angular block over
    all theta/zeta points. This captures 2D angular coupling while treating x/L
    coupling as diagonal, making it stronger than the theta- or zeta-line options.
    """
    cache_key = _rhsmode1_precond_cache_key(op, "theta_zeta")
    precond_dtype = _precond_dtype()
    cached = _RHSMODE1_PRECOND_CACHE.get(cache_key)
    if cached is None:
        op_pc = op
        n_species = int(op.n_species)
        n_x = int(op.n_x)
        n_l = int(op.n_xi)
        n_theta = int(op.n_theta)
        n_zeta = int(op.n_zeta)
        total_size = int(op.total_size)

        nxi_for_x = np.asarray(op.fblock.collisionless.n_xi_for_x, dtype=np.int32)
        local_per_species = int(np.sum(nxi_for_x))
        block_count = int(n_species * local_per_species)
        block_size = int(n_theta * n_zeta)

        reg_env = os.environ.get("SFINCS_JAX_RHSMODE1_PRECOND_REG", "").strip()
        reg_val = float(reg_env) if reg_env else 1e-10
        reg = np.float64(reg_val)

        idx_map = np.zeros((block_count, block_size), dtype=np.int32)
        block_inv = np.zeros((block_count, block_size, block_size), dtype=np.float64)

        block_idx = 0
        for s in range(n_species):
            for ix in range(n_x):
                max_l = int(nxi_for_x[ix])
                for il in range(max_l):
                    k = 0
                    for it in range(n_theta):
                        for iz in range(n_zeta):
                            idx_map[block_idx, k] = int(
                                ((((s * n_x + ix) * n_l + il) * n_theta + it) * n_zeta + iz)
                            )
                            k += 1
                    rep_idx = idx_map[block_idx, :]
                    chunk_cols = _precond_chunk_cols(total_size, int(rep_idx.shape[0]))
                    y_sub = _matvec_submatrix(
                        op_pc,
                        col_idx=rep_idx,
                        row_idx=rep_idx,
                        total_size=total_size,
                        chunk_cols=chunk_cols,
                    )
                    a = np.asarray(y_sub.T, dtype=np.float64)
                    a = a + reg * np.eye(block_size, dtype=np.float64)
                    try:
                        inv = np.linalg.inv(a)
                    except np.linalg.LinAlgError:
                        inv = np.linalg.pinv(a, rcond=1e-12)
                    if not np.all(np.isfinite(inv)):
                        inv = np.linalg.pinv(a, rcond=1e-12)
                    block_inv[block_idx, :, :] = inv
                    block_idx += 1

        idx_map_jnp = jnp.asarray(idx_map, dtype=jnp.int32)
        flat_idx_jnp = idx_map_jnp.reshape((-1,))
        block_inv_jnp = jnp.asarray(block_inv, dtype=precond_dtype)

        extra_start = int(op.f_size + op.phi1_size)
        extra_size = int(op.extra_size)
        extra_idx_np = np.arange(extra_start, extra_start + extra_size, dtype=np.int32)
        extra_idx_jnp = jnp.asarray(extra_idx_np, dtype=jnp.int32)
        extra_inv_jnp: jnp.ndarray | None = None
        if extra_size > 0:
            chunk_cols = _precond_chunk_cols(total_size, int(extra_idx_np.shape[0]))
            y_sub = _matvec_submatrix(
                op_pc,
                col_idx=extra_idx_np,
                row_idx=extra_idx_np,
                total_size=total_size,
                chunk_cols=chunk_cols,
            )
            ee = np.asarray(y_sub.T, dtype=np.float64)
            ee = ee + reg * np.eye(extra_size, dtype=np.float64)
            try:
                ee_inv = np.linalg.inv(ee)
            except np.linalg.LinAlgError:
                ee_inv = np.linalg.pinv(ee, rcond=1e-12)
            if not np.all(np.isfinite(ee_inv)):
                ee_inv = np.linalg.pinv(ee, rcond=1e-12)
            extra_inv_jnp = jnp.asarray(ee_inv, dtype=precond_dtype)

        cached = _RHSMode1PrecondCache(
            idx_map_jnp=idx_map_jnp,
            flat_idx_jnp=flat_idx_jnp,
            block_inv_jnp=block_inv_jnp,
            extra_idx_jnp=extra_idx_jnp,
            extra_inv_jnp=extra_inv_jnp,
        )
        _RHSMODE1_PRECOND_CACHE[cache_key] = cached

    block_inv_jnp = cached.block_inv_jnp
    flat_idx_jnp = cached.flat_idx_jnp
    extra_idx_jnp = cached.extra_idx_jnp
    extra_inv_jnp = cached.extra_inv_jnp
    block_size = int(block_inv_jnp.shape[-1])
    block_count = int(block_inv_jnp.shape[0])

    def _apply_full(r_full: jnp.ndarray) -> jnp.ndarray:
        r_full = jnp.asarray(r_full, dtype=precond_dtype)
        r_loc = r_full[flat_idx_jnp].reshape((block_count, block_size))
        z_loc = jnp.einsum("bij,bj->bi", block_inv_jnp, r_loc)
        z_full = jnp.zeros_like(r_full)
        z_full = z_full.at[flat_idx_jnp].set(z_loc.reshape((-1,)), unique_indices=True)
        if extra_inv_jnp is not None:
            r_extra = r_full[extra_idx_jnp]
            z_extra = extra_inv_jnp @ r_extra
            z_full = z_full.at[extra_idx_jnp].set(z_extra, unique_indices=True)
        return jnp.asarray(z_full, dtype=jnp.float64)

    if reduce_full is None or expand_reduced is None:
        return _apply_full

    def _apply_reduced(r_reduced: jnp.ndarray) -> jnp.ndarray:
        z_full = _apply_full(expand_reduced(r_reduced))
        return reduce_full(z_full)

    return _apply_reduced


def _build_rhsmode1_species_block_preconditioner(
    *,
    op: V3FullSystemOperator,
    reduce_full: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    expand_reduced: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Build a RHSMode=1 preconditioner using full species blocks.

    For each species, solve a block over all (x,L,theta,zeta) points. This captures
    both angular and x/L coupling within each species while ignoring inter-species
    coupling, providing a much stronger preconditioner for PAS systems.
    """
    cache_key = _rhsmode1_precond_cache_key(op, "species_block")
    # Use block-size hint for mixed-precision selection on large species blocks.
    nxi_for_x = np.asarray(op.fblock.collisionless.n_xi_for_x, dtype=np.int32)
    local_per_species = int(np.sum(nxi_for_x))
    block_size_hint = int(local_per_species * int(op.n_theta) * int(op.n_zeta))
    precond_dtype = _precond_dtype(block_size_hint * block_size_hint)
    cached = _RHSMODE1_PRECOND_CACHE.get(cache_key)
    if cached is None:
        op_pc = op
        n_species = int(op.n_species)
        n_x = int(op.n_x)
        n_l = int(op.n_xi)
        n_theta = int(op.n_theta)
        n_zeta = int(op.n_zeta)
        total_size = int(op.total_size)

        nxi_for_x = np.asarray(op.fblock.collisionless.n_xi_for_x, dtype=np.int32)
        local_per_species = int(np.sum(nxi_for_x))
        block_size = int(local_per_species * n_theta * n_zeta)

        reg_env = os.environ.get("SFINCS_JAX_RHSMODE1_PRECOND_REG", "").strip()
        reg_val = float(reg_env) if reg_env else 1e-10
        reg = np.float64(reg_val)

        idx_map = np.zeros((n_species, block_size), dtype=np.int32)
        block_inv = np.zeros((n_species, block_size, block_size), dtype=np.float64)

        for s in range(n_species):
            k = 0
            for ix in range(n_x):
                max_l = int(nxi_for_x[ix])
                for il in range(max_l):
                    for it in range(n_theta):
                        for iz in range(n_zeta):
                            idx_map[s, k] = int(
                                ((((s * n_x + ix) * n_l + il) * n_theta + it) * n_zeta + iz)
                            )
                            k += 1
            rep_idx = idx_map[s, :]
            chunk_cols = _precond_chunk_cols(total_size, int(rep_idx.shape[0]))
            y_sub = _matvec_submatrix(
                op_pc,
                col_idx=rep_idx,
                row_idx=rep_idx,
                total_size=total_size,
                chunk_cols=chunk_cols,
            )
            a = np.asarray(y_sub.T, dtype=np.float64)
            a = a + reg * np.eye(block_size, dtype=np.float64)
            try:
                inv = np.linalg.inv(a)
            except np.linalg.LinAlgError:
                inv = np.linalg.pinv(a, rcond=1e-12)
            if not np.all(np.isfinite(inv)):
                inv = np.linalg.pinv(a, rcond=1e-12)
            block_inv[s, :, :] = inv

        idx_map_jnp = jnp.asarray(idx_map, dtype=jnp.int32)
        flat_idx_jnp = idx_map_jnp.reshape((-1,))
        block_inv_jnp = jnp.asarray(block_inv, dtype=precond_dtype)

        extra_start = int(op.f_size + op.phi1_size)
        extra_size = int(op.extra_size)
        extra_idx_np = np.arange(extra_start, extra_start + extra_size, dtype=np.int32)
        extra_idx_jnp = jnp.asarray(extra_idx_np, dtype=jnp.int32)
        extra_inv_jnp: jnp.ndarray | None = None
        if extra_size > 0:
            chunk_cols = _precond_chunk_cols(total_size, int(extra_idx_np.shape[0]))
            y_sub = _matvec_submatrix(
                op_pc,
                col_idx=extra_idx_np,
                row_idx=extra_idx_np,
                total_size=total_size,
                chunk_cols=chunk_cols,
            )
            ee = np.asarray(y_sub.T, dtype=np.float64)
            ee = ee + reg * np.eye(extra_size, dtype=np.float64)
            try:
                ee_inv = np.linalg.inv(ee)
            except np.linalg.LinAlgError:
                ee_inv = np.linalg.pinv(ee, rcond=1e-12)
            if not np.all(np.isfinite(ee_inv)):
                ee_inv = np.linalg.pinv(ee, rcond=1e-12)
            extra_inv_jnp = jnp.asarray(ee_inv, dtype=precond_dtype)

        cached = _RHSMode1PrecondCache(
            idx_map_jnp=idx_map_jnp,
            flat_idx_jnp=flat_idx_jnp,
            block_inv_jnp=block_inv_jnp,
            extra_idx_jnp=extra_idx_jnp,
            extra_inv_jnp=extra_inv_jnp,
        )
        _RHSMODE1_PRECOND_CACHE[cache_key] = cached

    block_inv_jnp = cached.block_inv_jnp
    flat_idx_jnp = cached.flat_idx_jnp
    extra_idx_jnp = cached.extra_idx_jnp
    extra_inv_jnp = cached.extra_inv_jnp
    n_species = int(block_inv_jnp.shape[0])
    block_size = int(block_inv_jnp.shape[-1])

    def _apply_full(r_full: jnp.ndarray) -> jnp.ndarray:
        r_full = jnp.asarray(r_full, dtype=precond_dtype)
        r_loc = r_full[flat_idx_jnp].reshape((n_species, block_size))
        z_loc = jnp.einsum("sij,sj->si", block_inv_jnp, r_loc)
        z_full = jnp.zeros_like(r_full)
        z_full = z_full.at[flat_idx_jnp].set(z_loc.reshape((-1,)), unique_indices=True)
        if extra_inv_jnp is not None:
            r_extra = r_full[extra_idx_jnp]
            z_extra = extra_inv_jnp @ r_extra
            z_full = z_full.at[extra_idx_jnp].set(z_extra, unique_indices=True)
        return jnp.asarray(z_full, dtype=jnp.float64)

    if reduce_full is None or expand_reduced is None:
        return _apply_full

    def _apply_reduced(r_reduced: jnp.ndarray) -> jnp.ndarray:
        z_full = _apply_full(expand_reduced(r_reduced))
        return reduce_full(z_full)

    return _apply_reduced


def _build_rhsmode1_species_xblock_preconditioner(
    *,
    op: V3FullSystemOperator,
    reduce_full: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    expand_reduced: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Build a RHSMode=1 preconditioner with a single species×x/L block per (theta,zeta).

    This includes inter-species coupling in the local (x,L) block, which is useful for
    FP collision cases where the field-particle terms couple species.
    """
    cache_key = _rhsmode1_precond_cache_key(op, "sxblock")
    nxi_for_x = np.asarray(op.fblock.collisionless.n_xi_for_x, dtype=np.int32)
    local_per_species = int(np.sum(nxi_for_x))
    block_size = int(int(op.n_species) * local_per_species)
    precond_dtype = _precond_dtype(block_size * block_size)
    cached = _RHSMODE1_PRECOND_GLOBAL_CACHE.get(cache_key)
    if cached is None:
        op_pc = op
        n_species = int(op.n_species)
        n_x = int(op.n_x)
        n_l = int(op.n_xi)
        n_theta = int(op.n_theta)
        n_zeta = int(op.n_zeta)
        total_size = int(op.total_size)

        reg_env = os.environ.get("SFINCS_JAX_RHSMODE1_PRECOND_REG", "").strip()
        reg_val = float(reg_env) if reg_env else 1e-10
        reg = np.float64(reg_val)

        rep_idx: list[int] = []
        for s in range(n_species):
            for ix in range(n_x):
                max_l = int(nxi_for_x[ix])
                for il in range(max_l):
                    f_idx = ((((s * n_x + ix) * n_l + il) * n_theta + 0) * n_zeta + 0)
                    rep_idx.append(int(f_idx))
        rep_idx_np = np.asarray(rep_idx, dtype=np.int32)
        chunk_cols = _precond_chunk_cols(total_size, int(rep_idx_np.shape[0]))
        y_sub = _matvec_submatrix(
            op_pc,
            col_idx=rep_idx_np,
            row_idx=rep_idx_np,
            total_size=total_size,
            chunk_cols=chunk_cols,
        )
        a = np.asarray(y_sub.T, dtype=np.float64)
        a = a + reg * np.eye(block_size, dtype=np.float64)
        try:
            inv = np.linalg.inv(a)
        except np.linalg.LinAlgError:
            inv = np.linalg.pinv(a, rcond=1e-12)
        if not np.all(np.isfinite(inv)):
            inv = np.linalg.pinv(a, rcond=1e-12)

        idx_map = np.zeros((n_theta, n_zeta, block_size), dtype=np.int32)
        for it in range(n_theta):
            for iz in range(n_zeta):
                k = 0
                for s in range(n_species):
                    for ix in range(n_x):
                        max_l = int(nxi_for_x[ix])
                        for il in range(max_l):
                            idx_map[it, iz, k] = int(
                                ((((s * n_x + ix) * n_l + il) * n_theta + it) * n_zeta + iz)
                            )
                            k += 1

        idx_map_jnp = jnp.asarray(idx_map, dtype=jnp.int32)
        flat_idx_jnp = idx_map_jnp.reshape((-1,))
        block_inv_jnp = jnp.asarray(inv, dtype=precond_dtype)

        extra_start = int(op.f_size + op.phi1_size)
        extra_size = int(op.extra_size)
        extra_idx_np = np.arange(extra_start, extra_start + extra_size, dtype=np.int32)
        extra_idx_jnp = jnp.asarray(extra_idx_np, dtype=jnp.int32)
        extra_inv_jnp: jnp.ndarray | None = None
        if extra_size > 0:
            chunk_cols = _precond_chunk_cols(total_size, int(extra_idx_np.shape[0]))
            y_sub = _matvec_submatrix(
                op_pc,
                col_idx=extra_idx_np,
                row_idx=extra_idx_np,
                total_size=total_size,
                chunk_cols=chunk_cols,
            )
            ee = np.asarray(y_sub.T, dtype=np.float64)
            ee = ee + reg * np.eye(extra_size, dtype=np.float64)
            try:
                ee_inv = np.linalg.inv(ee)
            except np.linalg.LinAlgError:
                ee_inv = np.linalg.pinv(ee, rcond=1e-12)
            if not np.all(np.isfinite(ee_inv)):
                ee_inv = np.linalg.pinv(ee, rcond=1e-12)
            extra_inv_jnp = jnp.asarray(ee_inv, dtype=precond_dtype)

        cached = _RHSMode1PrecondGlobalCache(
            idx_map_jnp=idx_map_jnp,
            flat_idx_jnp=flat_idx_jnp,
            block_inv_jnp=block_inv_jnp,
            extra_idx_jnp=extra_idx_jnp,
            extra_inv_jnp=extra_inv_jnp,
        )
        _RHSMODE1_PRECOND_GLOBAL_CACHE[cache_key] = cached

    block_inv_jnp = cached.block_inv_jnp
    flat_idx_jnp = cached.flat_idx_jnp
    extra_idx_jnp = cached.extra_idx_jnp
    extra_inv_jnp = cached.extra_inv_jnp

    def _apply_full(r_full: jnp.ndarray) -> jnp.ndarray:
        r_full = jnp.asarray(r_full, dtype=precond_dtype)
        r_loc = r_full[flat_idx_jnp].reshape((int(op.n_theta), int(op.n_zeta), block_size))
        z_loc = jnp.einsum("ij,tzj->tzi", block_inv_jnp, r_loc)
        z_full = jnp.zeros_like(r_full)
        z_full = z_full.at[flat_idx_jnp].set(z_loc.reshape((-1,)), unique_indices=True)
        if extra_inv_jnp is not None:
            r_extra = r_full[extra_idx_jnp]
            z_extra = extra_inv_jnp @ r_extra
            z_full = z_full.at[extra_idx_jnp].set(z_extra, unique_indices=True)
        return jnp.asarray(z_full, dtype=jnp.float64)

    if reduce_full is None or expand_reduced is None:
        return _apply_full

    def _apply_reduced(r_reduced: jnp.ndarray) -> jnp.ndarray:
        z_full = _apply_full(expand_reduced(r_reduced))
        return reduce_full(z_full)

    return _apply_reduced


def _build_rhsmode1_xblock_tz_preconditioner(
    *,
    op: V3FullSystemOperator,
    reduce_full: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    expand_reduced: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Build a RHSMode=1 preconditioner using full (theta,zeta) blocks per x.

    For each species and each x, solve the block over all (L,theta,zeta) points.
    This captures angular coupling while retaining x/L coupling without assembling
    full species blocks.
    """
    cache_key = _rhsmode1_precond_cache_key(op, "xblock_tz")
    cached = _RHSMODE1_PRECOND_LIST_CACHE.get(cache_key)
    if cached is None:
        op_pc = op
        n_species = int(op.n_species)
        n_x = int(op.n_x)
        n_l = int(op.n_xi)
        n_theta = int(op.n_theta)
        n_zeta = int(op.n_zeta)
        total_size = int(op.total_size)

        nxi_for_x = np.asarray(op.fblock.collisionless.n_xi_for_x, dtype=np.int32)
        max_l = int(np.max(nxi_for_x)) if nxi_for_x.size else 0
        max_block_size = int(max_l * n_theta * n_zeta)
        precond_dtype = _precond_dtype(max_block_size * max_block_size)

        reg_env = os.environ.get("SFINCS_JAX_RHSMODE1_PRECOND_REG", "").strip()
        reg_val = float(reg_env) if reg_env else 1e-10
        reg = np.float64(reg_val)

        block_inv_list: list[jnp.ndarray] = []
        block_slices: list[tuple[int, int]] = []

        for s in range(n_species):
            for ix in range(n_x):
                max_l = int(nxi_for_x[ix])
                block_size = int(max_l * n_theta * n_zeta)
                start = int((((s * n_x + ix) * n_l) * n_theta) * n_zeta)
                if block_size <= 0:
                    continue
                rep_idx = np.arange(start, start + block_size, dtype=np.int32)
                chunk_cols = _precond_chunk_cols(total_size, int(rep_idx.shape[0]))
                y_sub = _matvec_submatrix(
                    op_pc,
                    col_idx=rep_idx,
                    row_idx=rep_idx,
                    total_size=total_size,
                    chunk_cols=chunk_cols,
                )
                a = np.asarray(y_sub.T, dtype=np.float64)
                a = a + reg * np.eye(block_size, dtype=np.float64)
                try:
                    inv = np.linalg.inv(a)
                except np.linalg.LinAlgError:
                    inv = np.linalg.pinv(a, rcond=1e-12)
                if not np.all(np.isfinite(inv)):
                    inv = np.linalg.pinv(a, rcond=1e-12)
                block_inv_list.append(jnp.asarray(inv, dtype=precond_dtype))
                block_slices.append((start, block_size))

        extra_start = int(op.f_size + op.phi1_size)
        extra_size = int(op.extra_size)
        extra_idx_np = np.arange(extra_start, extra_start + extra_size, dtype=np.int32)
        extra_idx_jnp = jnp.asarray(extra_idx_np, dtype=jnp.int32)
        extra_inv_jnp: jnp.ndarray | None = None
        if extra_size > 0:
            chunk_cols = _precond_chunk_cols(total_size, int(extra_idx_np.shape[0]))
            y_sub = _matvec_submatrix(
                op_pc,
                col_idx=extra_idx_np,
                row_idx=extra_idx_np,
                total_size=total_size,
                chunk_cols=chunk_cols,
            )
            ee = np.asarray(y_sub.T, dtype=np.float64)
            ee = ee + reg * np.eye(extra_size, dtype=np.float64)
            try:
                ee_inv = np.linalg.inv(ee)
            except np.linalg.LinAlgError:
                ee_inv = np.linalg.pinv(ee, rcond=1e-12)
            if not np.all(np.isfinite(ee_inv)):
                ee_inv = np.linalg.pinv(ee, rcond=1e-12)
            extra_inv_jnp = jnp.asarray(ee_inv, dtype=precond_dtype)

        cached = _RHSMode1PrecondListCache(
            block_inv_list=tuple(block_inv_list),
            block_slices=tuple(block_slices),
            extra_idx_jnp=extra_idx_jnp,
            extra_inv_jnp=extra_inv_jnp,
        )
        _RHSMODE1_PRECOND_LIST_CACHE[cache_key] = cached

    block_inv_list = cached.block_inv_list
    block_slices = cached.block_slices
    extra_idx_jnp = cached.extra_idx_jnp
    extra_inv_jnp = cached.extra_inv_jnp
    precond_dtype = block_inv_list[0].dtype if block_inv_list else _precond_dtype()

    def _apply_full(r_full: jnp.ndarray) -> jnp.ndarray:
        r_full = jnp.asarray(r_full, dtype=precond_dtype)
        z_full = jnp.zeros_like(r_full)
        for inv, (start, block_size) in zip(block_inv_list, block_slices, strict=True):
            r_loc = r_full[start : start + block_size]
            z_loc = inv @ r_loc
            z_full = z_full.at[start : start + block_size].set(z_loc, unique_indices=True)
        if extra_inv_jnp is not None:
            r_extra = r_full[extra_idx_jnp]
            z_extra = extra_inv_jnp @ r_extra
            z_full = z_full.at[extra_idx_jnp].set(z_extra, unique_indices=True)
        return jnp.asarray(z_full, dtype=jnp.float64)

    if reduce_full is None or expand_reduced is None:
        return _apply_full

    def _apply_reduced(r_reduced: jnp.ndarray) -> jnp.ndarray:
        z_full = _apply_full(expand_reduced(r_reduced))
        return reduce_full(z_full)

    return _apply_reduced


def _build_rhsmode1_sxblock_tz_preconditioner(
    *,
    op: V3FullSystemOperator,
    reduce_full: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    expand_reduced: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Build a RHSMode=1 preconditioner with species/x blocks over (theta,zeta) per L."""
    cache_key = _rhsmode1_precond_cache_key(op, "sxblock_tz")
    cached = _RHSMODE1_PRECOND_IDX_CACHE.get(cache_key)
    if cached is None:
        n_s = int(op.n_species)
        n_x = int(op.n_x)
        n_l = int(op.n_xi)
        n_t = int(op.n_theta)
        n_z = int(op.n_zeta)
        total = int(op.total_size)

        nxi_for_x = np.asarray(op.fblock.collisionless.n_xi_for_x, dtype=np.int32)
        max_block_size = 0
        for l in range(n_l):
            active_x = np.where(nxi_for_x > l)[0]
            if active_x.size:
                max_block_size = max(max_block_size, int(active_x.size) * n_s * n_t * n_z)
        precond_dtype = _precond_dtype(max_block_size * max_block_size)

        reg_env = os.environ.get("SFINCS_JAX_RHSMODE1_PRECOND_REG", "").strip()
        reg_val = float(reg_env) if reg_env else 1e-10
        reg = np.float64(reg_val)

        block_inv_list: list[jnp.ndarray] = []
        block_idx_list: list[jnp.ndarray] = []
        for l in range(n_l):
            active_x = np.where(nxi_for_x > l)[0]
            if active_x.size == 0:
                continue
            idx: list[int] = []
            for s in range(n_s):
                for ix in active_x:
                    base = int((((s * n_x + int(ix)) * n_l + l) * n_t) * n_z)
                    for it in range(n_t):
                        for iz in range(n_z):
                            idx.append(base + it * n_z + iz)
            rep_idx = np.asarray(idx, dtype=np.int32)
            chunk_cols = _precond_chunk_cols(total, int(rep_idx.shape[0]))
            y_sub = _matvec_submatrix(
                op,
                col_idx=rep_idx,
                row_idx=rep_idx,
                total_size=total,
                chunk_cols=chunk_cols,
            )
            a = np.asarray(y_sub.T, dtype=np.float64)
            a = a + reg * np.eye(int(rep_idx.shape[0]), dtype=np.float64)
            try:
                inv = np.linalg.inv(a)
            except np.linalg.LinAlgError:
                inv = np.linalg.pinv(a, rcond=1e-12)
            if not np.all(np.isfinite(inv)):
                inv = np.linalg.pinv(a, rcond=1e-12)
            block_inv_list.append(jnp.asarray(inv, dtype=precond_dtype))
            block_idx_list.append(jnp.asarray(rep_idx, dtype=jnp.int32))

        extra_start = int(op.f_size + op.phi1_size)
        extra_size = int(op.extra_size)
        extra_idx_np = np.arange(extra_start, extra_start + extra_size, dtype=np.int32)
        extra_idx_jnp = jnp.asarray(extra_idx_np, dtype=jnp.int32)
        extra_inv_jnp: jnp.ndarray | None = None
        if extra_size > 0:
            chunk_cols = _precond_chunk_cols(total, int(extra_idx_np.shape[0]))
            y_sub = _matvec_submatrix(
                op,
                col_idx=extra_idx_np,
                row_idx=extra_idx_np,
                total_size=total,
                chunk_cols=chunk_cols,
            )
            ee = np.asarray(y_sub.T, dtype=np.float64)
            ee = ee + reg * np.eye(extra_size, dtype=np.float64)
            try:
                ee_inv = np.linalg.inv(ee)
            except np.linalg.LinAlgError:
                ee_inv = np.linalg.pinv(ee, rcond=1e-12)
            if not np.all(np.isfinite(ee_inv)):
                ee_inv = np.linalg.pinv(ee, rcond=1e-12)
            extra_inv_jnp = jnp.asarray(ee_inv, dtype=precond_dtype)

        cached = _RHSMode1PrecondIdxCache(
            block_inv_list=tuple(block_inv_list),
            block_idx_list=tuple(block_idx_list),
            extra_idx_jnp=extra_idx_jnp,
            extra_inv_jnp=extra_inv_jnp,
        )
        _RHSMODE1_PRECOND_IDX_CACHE[cache_key] = cached

    block_inv_list = cached.block_inv_list
    block_idx_list = cached.block_idx_list
    extra_idx_jnp = cached.extra_idx_jnp
    extra_inv_jnp = cached.extra_inv_jnp
    precond_dtype = block_inv_list[0].dtype if block_inv_list else _precond_dtype()

    def _apply_full(r_full: jnp.ndarray) -> jnp.ndarray:
        r_full = jnp.asarray(r_full, dtype=precond_dtype)
        z_full = jnp.zeros_like(r_full)
        for inv, idx in zip(block_inv_list, block_idx_list, strict=True):
            r_loc = r_full[idx]
            z_loc = inv @ r_loc
            z_full = z_full.at[idx].set(z_loc, unique_indices=True)
        if extra_inv_jnp is not None:
            r_extra = r_full[extra_idx_jnp]
            z_extra = extra_inv_jnp @ r_extra
            z_full = z_full.at[extra_idx_jnp].set(z_extra, unique_indices=True)
        return jnp.asarray(z_full, dtype=jnp.float64)

    if reduce_full is None or expand_reduced is None:
        return _apply_full

    def _apply_reduced(r_reduced: jnp.ndarray) -> jnp.ndarray:
        z_full = _apply_full(expand_reduced(r_reduced))
        return reduce_full(z_full)

    return _apply_reduced


def _build_rhsmode1_zeta_line_preconditioner(
    *,
    op: V3FullSystemOperator,
    reduce_full: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    expand_reduced: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Build a stronger RHSMode=1 preconditioner using zeta-line blocks.

    For each species and each theta plane, solve a representative block that couples
    all zeta points for all local (x,L) unknowns at that theta. This approximates the
    dominant derivative couplings along zeta while ignoring theta derivatives.
    """
    cache_key = _rhsmode1_precond_cache_key(op, "zeta_line")
    precond_dtype = _precond_dtype()
    cached = _RHSMODE1_PRECOND_CACHE.get(cache_key)
    if cached is None:
        op_pc = _build_rhsmode1_preconditioner_operator_zeta_line(op)
        n_species = int(op.n_species)
        n_x = int(op.n_x)
        n_l = int(op.n_xi)
        n_theta = int(op.n_theta)
        n_zeta = int(op.n_zeta)
        total_size = int(op.total_size)

        nxi_for_x = np.asarray(op.fblock.collisionless.n_xi_for_x, dtype=np.int32)
        local_per_species = int(np.sum(nxi_for_x))
        line_size = int(n_zeta * local_per_species)

        reg_env = os.environ.get("SFINCS_JAX_RHSMODE1_PRECOND_REG", "").strip()
        reg_val = float(reg_env) if reg_env else 1e-10
        reg = np.float64(reg_val)

        # Build per-(species,theta) gather map for all zeta points and local (x,L) indices.
        idx_map = np.zeros((n_species, n_theta, line_size), dtype=np.int32)
        for s in range(n_species):
            for it in range(n_theta):
                k = 0
                for iz in range(n_zeta):
                    for ix in range(n_x):
                        max_l = int(nxi_for_x[ix])
                        for il in range(max_l):
                            idx_map[s, it, k] = int(
                                ((((s * n_x + ix) * n_l + il) * n_theta + it) * n_zeta + iz)
                            )
                            k += 1

        idx_map_jnp = jnp.asarray(idx_map, dtype=jnp.int32)
        flat_idx_jnp = idx_map_jnp.reshape((-1,))

        # Invert a zeta-line block for each theta plane.
        block_inv = np.zeros((n_species, n_theta, line_size, line_size), dtype=np.float64)
        for s in range(n_species):
            for it in range(n_theta):
                rep_idx = np.asarray(idx_map[s, it, :], dtype=np.int32)
                chunk_cols = _precond_chunk_cols(total_size, int(rep_idx.shape[0]))
                y_sub = _matvec_submatrix(
                    op_pc,
                    col_idx=rep_idx,
                    row_idx=rep_idx,
                    total_size=total_size,
                    chunk_cols=chunk_cols,
                )
                a = np.asarray(y_sub.T, dtype=np.float64)  # (line_size, line_size)
                a = a + reg * np.eye(line_size, dtype=np.float64)
                try:
                    inv = np.linalg.inv(a)
                except np.linalg.LinAlgError:
                    inv = np.linalg.pinv(a, rcond=1e-12)
                if not np.all(np.isfinite(inv)):
                    inv = np.linalg.pinv(a, rcond=1e-12)
                block_inv[s, it, :, :] = inv
        block_inv_jnp = jnp.asarray(block_inv, dtype=precond_dtype)

        extra_start = int(op.f_size + op.phi1_size)
        extra_size = int(op.extra_size)
        extra_idx_np = np.arange(extra_start, extra_start + extra_size, dtype=np.int32)
        extra_idx_jnp = jnp.asarray(extra_idx_np, dtype=jnp.int32)
        extra_inv_jnp: jnp.ndarray | None = None
        if extra_size > 0:
            chunk_cols = _precond_chunk_cols(total_size, int(extra_idx_np.shape[0]))
            y_sub = _matvec_submatrix(
                op_pc,
                col_idx=extra_idx_np,
                row_idx=extra_idx_np,
                total_size=total_size,
                chunk_cols=chunk_cols,
            )
            ee = np.asarray(y_sub.T, dtype=np.float64)
            ee = ee + reg * np.eye(extra_size, dtype=np.float64)
            try:
                ee_inv = np.linalg.inv(ee)
            except np.linalg.LinAlgError:
                ee_inv = np.linalg.pinv(ee, rcond=1e-12)
            if not np.all(np.isfinite(ee_inv)):
                ee_inv = np.linalg.pinv(ee, rcond=1e-12)
            extra_inv_jnp = jnp.asarray(ee_inv, dtype=precond_dtype)

        cached = _RHSMode1PrecondCache(
            idx_map_jnp=idx_map_jnp,
            flat_idx_jnp=flat_idx_jnp,
            block_inv_jnp=block_inv_jnp,
            extra_idx_jnp=extra_idx_jnp,
            extra_inv_jnp=extra_inv_jnp,
        )
        _RHSMODE1_PRECOND_CACHE[cache_key] = cached

    n_species = int(op.n_species)
    n_theta = int(op.n_theta)
    line_size = int(cached.block_inv_jnp.shape[-1])
    flat_idx_jnp = cached.flat_idx_jnp
    block_inv_jnp = cached.block_inv_jnp
    extra_idx_jnp = cached.extra_idx_jnp
    extra_inv_jnp = cached.extra_inv_jnp

    def _apply_full(r_full: jnp.ndarray) -> jnp.ndarray:
        r_full = jnp.asarray(r_full, dtype=precond_dtype)
        r_loc = r_full[flat_idx_jnp].reshape((n_species, n_theta, line_size))
        z_loc = jnp.einsum("stab,stb->sta", block_inv_jnp, r_loc)
        z_full = jnp.zeros_like(r_full)
        z_full = z_full.at[flat_idx_jnp].set(z_loc.reshape((-1,)), unique_indices=True)
        if extra_inv_jnp is not None:
            r_extra = r_full[extra_idx_jnp]
            z_extra = extra_inv_jnp @ r_extra
            z_full = z_full.at[extra_idx_jnp].set(z_extra, unique_indices=True)
        return jnp.asarray(z_full, dtype=jnp.float64)

    if reduce_full is None or expand_reduced is None:
        return _apply_full

    def _apply_reduced(r_reduced: jnp.ndarray) -> jnp.ndarray:
        z_full = _apply_full(expand_reduced(r_reduced))
        return reduce_full(z_full)

    return _apply_reduced


def solve_v3_full_system_linear_gmres(
    *,
    nml: Namelist,
    which_rhs: int | None = None,
    x0: jnp.ndarray | None = None,
    tol: float = 1e-10,
    atol: float = 0.0,
    restart: int = 80,
    maxiter: int | None = 400,
    solve_method: str = "auto",
    identity_shift: float = 0.0,
    phi1_hat_base: jnp.ndarray | None = None,
    emit: Callable[[int, str], None] | None = None,
    recycle_basis: Sequence[jnp.ndarray] | None = None,
) -> V3LinearSolveResult:
    """Solve the current v3 full-system linear problem `A x = rhs` matrix-free using GMRES.

    Notes
    -----
    This helper currently targets the linear runs exercised in the parity fixtures
    (e.g. includePhi1InKineticEquation=false). For nonlinear runs, use `residual_v3_full_system`
    and an outer Newton-Krylov iteration (not yet shipped as a stable API).
    """
    t = Timer()
    profiler = maybe_profiler(emit=emit)

    def _mark(label: str) -> None:
        if profiler is not None:
            profiler.mark(label)
    restart_env = os.environ.get("SFINCS_JAX_GMRES_RESTART", "").strip()
    if restart_env:
        try:
            restart = int(restart_env)
        except ValueError:
            pass
    maxiter_env = os.environ.get("SFINCS_JAX_GMRES_MAXITER", "").strip()
    if maxiter_env:
        try:
            maxiter = int(maxiter_env)
        except ValueError:
            pass
    if emit is not None:
        emit(1, "solve_v3_full_system_linear_gmres: building operator")
    op = full_system_operator_from_namelist(nml=nml, identity_shift=identity_shift, phi1_hat_base=phi1_hat_base)
    _mark("operator_built")
    _set_precond_size_hint(int(op.total_size))
    if int(op.rhs_mode) in {2, 3}:
        # v3 sets (dnHatdpsiHats, dTHatdpsiHats, EParallelHat) internally based on whichRHS.
        # If the input file omits gradients (common for monoenergetic runs), callers must select whichRHS.
        if which_rhs is None:
            which_rhs = 1
        op = with_transport_rhs_settings(op, which_rhs=int(which_rhs))
        if emit is not None:
            emit(1, f"solve_v3_full_system_linear_gmres: applied transport RHS settings whichRHS={int(which_rhs)}")
    if emit is not None:
        emit(1, f"solve_v3_full_system_linear_gmres: total_size={int(op.total_size)}")
        emit(1, "solve_v3_full_system_linear_gmres: assembling RHS")
    rhs = rhs_v3_full_system(op)
    _mark("rhs_assembled")
    rhs_norm = jnp.linalg.norm(rhs)
    if emit is not None:
        emit(2, f"solve_v3_full_system_linear_gmres: rhs_norm={float(rhs_norm):.6e}")

    recycle_k_env = os.environ.get("SFINCS_JAX_RHSMODE1_RECYCLE_K", "").strip()
    try:
        recycle_k = int(recycle_k_env) if recycle_k_env else 4
    except ValueError:
        recycle_k = 4
    recycle_k = max(0, recycle_k)
    recycle_basis_use: list[jnp.ndarray] = []
    if recycle_k > 0 and recycle_basis:
        for vec in recycle_basis:
            v = jnp.asarray(vec)
            if v.shape == (op.total_size,):
                recycle_basis_use.append(v)
        if len(recycle_basis_use) > recycle_k:
            recycle_basis_use = recycle_basis_use[-recycle_k:]

    def mv(x):
        # Use the JIT-compiled operator application to reduce Python overhead in repeated matvecs
        # (e.g. during GMRES iterations and Er scans).
        return apply_v3_full_system_operator_cached(op, x)

    def _recycled_initial_guess(
        rhs_vec: jnp.ndarray,
        basis: list[jnp.ndarray],
        basis_au: list[jnp.ndarray],
    ) -> jnp.ndarray | None:
        if not basis or not basis_au:
            return None
        u = jnp.stack(basis, axis=1)  # (N, k)
        au = jnp.stack(basis_au, axis=1)  # (N, k)
        coeff, *_ = jnp.linalg.lstsq(au, rhs_vec, rcond=None)
        x0_guess = u @ coeff
        if not jnp.all(jnp.isfinite(x0_guess)):
            return None
        return x0_guess

    active_env = os.environ.get("SFINCS_JAX_ACTIVE_DOF", "").strip().lower()
    nxi_for_x = np.asarray(op.fblock.collisionless.n_xi_for_x, dtype=np.int32)
    has_reduced_modes = bool(np.any(nxi_for_x < int(op.n_xi)))
    phys_params = nml.group("physicsParameters")
    def _nml_get(group: dict, key: str, default=None):
        if key in group:
            return group[key]
        key_upper = key.upper()
        if key_upper in group:
            return group[key_upper]
        key_lower = key.lower()
        if key_lower in group:
            return group[key_lower]
        return default

    def _nml_bool(val: object | None) -> bool:
        if val is None:
            return False
        if isinstance(val, bool):
            return bool(val)
        if isinstance(val, (int, np.integer)):
            return bool(int(val))
        if isinstance(val, (float, np.floating)):
            return bool(float(val))
        if isinstance(val, str):
            return val.strip().lower() in {"t", "true", "1", "yes", ".true."}
        return False
    use_dkes = _nml_bool(
        _nml_get(
            phys_params,
            "useDKESExBDrift",
            _nml_get(phys_params, "useDKESExBdrift", _nml_get(phys_params, "use_dkes_exb_drift", None)),
        )
    )
    use_active_dof_mode = False
    if active_env in {"1", "true", "yes", "on"}:
        use_active_dof_mode = True
    elif active_env in {"0", "false", "no", "off"}:
        use_active_dof_mode = False
    else:
        # Auto mode:
        # - Always use active-DOF reduction for RHSMode=2/3 with truncated pitch grid.
        # - Also use it for RHSMode=1 when includePhi1 is off and pitch truncation is
        #   present. This reduces solve size and JIT cost in upstream-like reduced runs.
        # - Keep includePhi1 RHSMode=1 on the full system to preserve sensitive parity
        #   branches in tiny includePhi1 fixtures.
        use_active_dof_mode = has_reduced_modes and (
            int(op.rhs_mode) in {2, 3} or (int(op.rhs_mode) == 1 and (not bool(op.include_phi1)))
        )
        if use_active_dof_mode and int(op.rhs_mode) == 1 and op.fblock.pas is not None and use_dkes:
            # DKES trajectories are sensitive to active-DOF reduction for PAS runs.
            use_active_dof_mode = False

    precond_opts = nml.group("preconditionerOptions")
    pas_project_env = os.environ.get("SFINCS_JAX_PAS_PROJECT_CONSTRAINTS", "").strip().lower()
    if pas_project_env in {"1", "true", "yes", "on"}:
        pas_project_mode = "on"
    elif pas_project_env in {"0", "false", "no", "off"}:
        pas_project_mode = "off"
    elif pas_project_env in {"", "auto"}:
        pas_project_mode = "auto"
    else:
        pas_project_mode = "off"
    def _precond_opt_int(key: str, default: int) -> int:
        val = precond_opts.get(key, None)
        if val is None:
            return default
        try:
            return int(val)
        except (TypeError, ValueError):
            return default

    preconditioner_species = _precond_opt_int("PRECONDITIONER_SPECIES", 1)
    preconditioner_x = _precond_opt_int("PRECONDITIONER_X", 1)
    preconditioner_xi = _precond_opt_int("PRECONDITIONER_XI", 1)
    full_precond_requested = (
        preconditioner_species == 0 and preconditioner_x == 0 and preconditioner_xi == 0
    )
    pas_project_enabled = bool(
        pas_project_mode == "on"
        or (pas_project_mode == "auto" and int(op.n_zeta) == 1 and not full_precond_requested)
    )
    if (
        pas_project_mode == "auto"
        and (not pas_project_enabled)
        and op.fblock.pas is not None
        and use_dkes
        and (not full_precond_requested)
    ):
        pas_project_enabled = True
    use_pas_projection = bool(
        pas_project_enabled
        and int(op.rhs_mode) == 1
        and (not bool(op.include_phi1))
        and int(op.constraint_scheme) == 2
        and op.fblock.pas is not None
        and int(op.phi1_size) == 0
    )
    if use_pas_projection:
        # Force a reduced system when projecting out constraintScheme=2 sources.
        use_active_dof_mode = True

    active_idx_jnp: jnp.ndarray | None = None
    full_to_active_jnp: jnp.ndarray | None = None
    active_size = int(op.total_size)
    if use_active_dof_mode:
        if use_pas_projection:
            active_idx_np = _transport_active_dof_indices(op)
            active_idx_np = active_idx_np[active_idx_np < int(op.f_size)]
            active_idx_jnp = jnp.asarray(active_idx_np, dtype=jnp.int32)
            full_to_active_np = np.zeros((int(op.f_size),), dtype=np.int32)
            full_to_active_np[np.asarray(active_idx_np, dtype=np.int32)] = np.arange(
                1, int(active_idx_np.shape[0]) + 1, dtype=np.int32
            )
            full_to_active_jnp = jnp.asarray(full_to_active_np, dtype=jnp.int32)
            active_size = int(active_idx_np.shape[0])
            if emit is not None:
                emit(
                    1,
                    "solve_v3_full_system_linear_gmres: PAS constraint projection enabled "
                    f"(size={active_size}/{int(op.total_size)})",
                )
        else:
            active_idx_np = _transport_active_dof_indices(op)
            active_idx_jnp = jnp.asarray(active_idx_np, dtype=jnp.int32)
            full_to_active_np = np.zeros((int(op.total_size),), dtype=np.int32)
            full_to_active_np[np.asarray(active_idx_np, dtype=np.int32)] = np.arange(
                1, int(active_idx_np.shape[0]) + 1, dtype=np.int32
            )
            full_to_active_jnp = jnp.asarray(full_to_active_np, dtype=jnp.int32)
            active_size = int(active_idx_np.shape[0])
            if emit is not None:
                emit(1, f"solve_v3_full_system_linear_gmres: active-DOF mode enabled (size={active_size}/{int(op.total_size)})")

    full_precond_env = os.environ.get("SFINCS_JAX_RHSMODE1_FULL_PRECOND", "").strip().lower()
    if full_precond_env in {"0", "false", "no", "off"}:
        full_precond_mode = "off"
    elif full_precond_env in {"dense", "dense_ksp"}:
        full_precond_mode = full_precond_env
    else:
        full_precond_mode = "auto"

    full_precond_max_env = os.environ.get("SFINCS_JAX_RHSMODE1_FULL_PRECOND_DENSE_MAX", "").strip()
    try:
        full_precond_dense_max = int(full_precond_max_env) if full_precond_max_env else 2500
    except ValueError:
        full_precond_dense_max = 2500

    full_precond_size = active_size if use_active_dof_mode else int(op.total_size)
    auto_dense_full_precond = bool(
        full_precond_mode == "auto"
        and full_precond_requested
        and int(op.rhs_mode) == 1
        and (not bool(op.include_phi1))
        and int(op.constraint_scheme) != 0
        and full_precond_dense_max > 0
        and int(full_precond_size) <= int(full_precond_dense_max)
        and str(solve_method).strip().lower() in {"auto", "default"}
    )
    if (
        full_precond_requested
        and (full_precond_mode in {"dense", "dense_ksp"} or auto_dense_full_precond)
        and int(op.rhs_mode) == 1
        and (not bool(op.include_phi1))
        and full_precond_dense_max > 0
        and int(full_precond_size) <= int(full_precond_dense_max)
        and str(solve_method).strip().lower() in {"auto", "default"}
    ):
        solve_method = "dense" if (full_precond_mode != "dense_ksp" or use_active_dof_mode) else "dense_ksp"
        if emit is not None:
            emit(
                0,
                "solve_v3_full_system_linear_gmres: full preconditioner requested; "
                f"using solve_method={solve_method} (size={int(full_precond_size)})",
            )

    if emit is not None:
        emit(1, f"solve_v3_full_system_linear_gmres: GMRES tol={tol} atol={atol} restart={restart} maxiter={maxiter} solve_method={solve_method}")
        emit(1, "solve_v3_full_system_linear_gmres: evaluateJacobian called (matrix-free)")
    rhs1_precond_env = os.environ.get("SFINCS_JAX_RHSMODE1_PRECONDITIONER", "").strip().lower()
    rhs1_bicgstab_env = os.environ.get("SFINCS_JAX_RHSMODE1_BICGSTAB_PRECOND", "").strip().lower()
    try:
        pre_theta = int(precond_opts.get("PRECONDITIONER_THETA", 0) or 0)
    except (TypeError, ValueError):
        pre_theta = 0
    try:
        pre_zeta = int(precond_opts.get("PRECONDITIONER_ZETA", 0) or 0)
    except (TypeError, ValueError):
        pre_zeta = 0
    rhs1_precond_kind: str | None
    if rhs1_precond_env:
        if rhs1_precond_env in {"0", "false", "no", "off"}:
            rhs1_precond_kind = None
        elif rhs1_precond_env in {"theta", "theta_line", "line_theta"}:
            rhs1_precond_kind = "theta_line"
        elif rhs1_precond_env in {"species", "species_block", "speciesblock"}:
            rhs1_precond_kind = "species_block"
        elif rhs1_precond_env in {"sxblock", "species_xblock", "species_x"}:
            rhs1_precond_kind = "sxblock"
        elif rhs1_precond_env in {"sxblock_tz", "sxblock_theta_zeta", "species_xblock_tz", "sx_tz"}:
            rhs1_precond_kind = "sxblock_tz"
        elif rhs1_precond_env in {"xblock_tz", "xblock", "x_tz", "xtz", "xblock_theta_zeta"}:
            rhs1_precond_kind = "xblock_tz"
        elif rhs1_precond_env in {"theta_zeta", "theta_zeta_line", "tz", "tz_line"}:
            rhs1_precond_kind = "theta_zeta"
        elif rhs1_precond_env in {"zeta", "zeta_line", "line_zeta"}:
            rhs1_precond_kind = "zeta_line"
        elif rhs1_precond_env in {"adi", "adi_line", "line_adi", "theta_zeta", "zeta_theta"}:
            rhs1_precond_kind = "adi"
        elif rhs1_precond_env in {"1", "true", "yes", "on", "point", "point_block"}:
            rhs1_precond_kind = "point"
        elif rhs1_precond_env in {"schur", "schur_complement", "constraint_schur"}:
            rhs1_precond_kind = "schur"
        elif rhs1_precond_env in {"collision", "diag", "collision_diag"}:
            rhs1_precond_kind = "collision"
        else:
            rhs1_precond_kind = None
    else:
        # Default to v3-like preconditioner options: when preconditioner_theta/zeta are 0,
        # use point-block Jacobi. Enable line preconditioning only when explicitly requested.
        if int(op.rhs_mode) == 1 and (not bool(op.include_phi1)):
            if pre_theta == 0 and pre_zeta == 0:
                tz_max_env = os.environ.get("SFINCS_JAX_RHSMODE1_TZ_PRECOND_MAX", "").strip()
                try:
                    tz_max = int(tz_max_env) if tz_max_env else 128
                except ValueError:
                    tz_max = 128
                xblock_tz_max_env = os.environ.get("SFINCS_JAX_RHSMODE1_XBLOCK_TZ_MAX", "").strip()
                try:
                    xblock_tz_max = int(xblock_tz_max_env) if xblock_tz_max_env else 1200
                except ValueError:
                    xblock_tz_max = 1200
                species_block_max_env = os.environ.get("SFINCS_JAX_RHSMODE1_SPECIES_BLOCK_MAX", "").strip()
                try:
                    species_block_max = int(species_block_max_env) if species_block_max_env else 1600
                except ValueError:
                    species_block_max = 1600
                nxi_for_x = np.asarray(op.fblock.collisionless.n_xi_for_x, dtype=np.int32)
                max_l = int(np.max(nxi_for_x)) if nxi_for_x.size else 0
                local_per_species = int(np.sum(nxi_for_x))
                dke_size = int(local_per_species * int(op.n_theta) * int(op.n_zeta))
                sxblock_max_env = os.environ.get("SFINCS_JAX_RHSMODE1_SXBLOCK_MAX", "").strip()
                try:
                    sxblock_max = int(sxblock_max_env) if sxblock_max_env else 64
                except ValueError:
                    sxblock_max = 64
                sxblock_tz_max_env = os.environ.get("SFINCS_JAX_RHSMODE1_SXBLOCK_TZ_MAX", "").strip()
                try:
                    sxblock_tz_max = int(sxblock_tz_max_env) if sxblock_tz_max_env else 0
                except ValueError:
                    sxblock_tz_max = 0
                if sxblock_tz_max == 0 and op.fblock.fp is not None and (
                    int(op.n_theta) > 1 or int(op.n_zeta) > 1
                ):
                    # Allow a modest FP sxblock_tz preconditioner in multi-angle FP cases
                    # to avoid RHSMode=1 stagnation without large dense fallbacks.
                    sxblock_tz_max = 2000
                sxblock_size = int(int(op.n_species) * local_per_species)
                sxblock_tz_size = int(int(op.n_species) * int(op.n_x) * int(op.n_theta) * int(op.n_zeta))
                schur_auto = False
                if (
                    int(op.constraint_scheme) == 2
                    and int(op.extra_size) > 0
                    and op.fblock.pas is not None
                    and (int(op.n_theta) > 1 or int(op.n_zeta) > 1)
                ):
                    schur_auto_min_env = os.environ.get("SFINCS_JAX_RHSMODE1_SCHUR_AUTO_MIN", "").strip()
                    try:
                        schur_auto_min = int(schur_auto_min_env) if schur_auto_min_env else 2500
                    except ValueError:
                        schur_auto_min = 2500
                    schur_auto = int(op.total_size) >= schur_auto_min
                phys_params = nml.group("physicsParameters")
                er_val = phys_params.get("ER", phys_params.get("Er", phys_params.get("er", None)))
                er_abs = 0.0
                if er_val is not None:
                    try:
                        er_abs = float(er_val)
                    except (TypeError, ValueError):
                        er_abs = 0.0
                er_abs = abs(er_abs)
                epar_val = phys_params.get("EPARALLELHAT", phys_params.get("EParallelHat", None))
                try:
                    epar_abs = abs(float(epar_val)) if epar_val is not None else 0.0
                except (TypeError, ValueError):
                    epar_abs = 0.0
                if epar_abs > 0.0 and sxblock_tz_max == 0:
                    sxblock_tz_max = 2000
                schur_er_env = os.environ.get("SFINCS_JAX_RHSMODE1_SCHUR_ER_ABS_MIN", "").strip()
                try:
                    schur_er_min = float(schur_er_env) if schur_er_env else 0.0
                except ValueError:
                    schur_er_min = 0.0
                schur_tokamak_env = os.environ.get("SFINCS_JAX_RHSMODE1_SCHUR_TOKAMAK", "").strip().lower()
                schur_tokamak = schur_tokamak_env in {"1", "true", "yes", "on"}
                tokamak_like = int(op.n_zeta) == 1
                if full_precond_requested and int(op.constraint_scheme) == 2 and int(op.extra_size) > 0:
                    if tokamak_like and (not schur_tokamak) and er_abs <= schur_er_min:
                        rhs1_precond_kind = "theta_line" if int(op.n_theta) >= int(op.n_zeta) else "zeta_line"
                    else:
                        rhs1_precond_kind = "schur"
                elif full_precond_requested and (int(op.n_theta) > 1 or int(op.n_zeta) > 1):
                    rhs1_precond_kind = "theta_line" if int(op.n_theta) >= int(op.n_zeta) else "zeta_line"
                elif schur_auto:
                    rhs1_precond_kind = "schur"
                elif (
                    op.fblock.fp is not None
                    and (int(op.n_theta) > 1 or int(op.n_zeta) > 1)
                    and sxblock_tz_max > 0
                    and sxblock_tz_size <= sxblock_tz_max
                ):
                    rhs1_precond_kind = "sxblock_tz"
                elif (
                    op.fblock.fp is not None
                    and (int(op.n_theta) > 1 or int(op.n_zeta) > 1)
                    and int(op.n_theta) * int(op.n_zeta) <= tz_max
                ):
                    rhs1_precond_kind = "theta_zeta"
                elif op.fblock.fp is not None and sxblock_max > 0 and sxblock_size <= sxblock_max:
                    rhs1_precond_kind = "sxblock"
                elif (
                    op.fblock.pas is not None
                    and int(op.n_theta) > 1
                    and int(op.n_zeta) > 1
                    and species_block_max > 0
                    and dke_size <= species_block_max
                ):
                    rhs1_precond_kind = "species_block"
                elif (
                    op.fblock.pas is not None
                    and int(op.n_theta) > 1
                    and int(op.n_zeta) > 1
                    and xblock_tz_max > 0
                    and int(max_l) * int(op.n_theta) * int(op.n_zeta) <= xblock_tz_max
                ):
                    rhs1_precond_kind = "xblock_tz"
                elif (
                    op.fblock.pas is not None
                    and int(op.n_theta) > 1
                    and int(op.n_zeta) > 1
                    and int(op.n_theta) * int(op.n_zeta) <= tz_max
                ):
                    rhs1_precond_kind = "theta_zeta"
                else:
                    collision_precond_min_env = os.environ.get("SFINCS_JAX_RHSMODE1_COLLISION_PRECOND_MIN", "").strip()
                    try:
                        collision_precond_min = int(collision_precond_min_env) if collision_precond_min_env else 600
                    except ValueError:
                        collision_precond_min = 600
                    use_collision_precond = (
                        (op.fblock.fp is not None or op.fblock.pas is not None)
                        and int(op.total_size) >= collision_precond_min
                    )
                    rhs1_precond_kind = "collision" if use_collision_precond else "point"
            elif pre_theta > 0 and pre_zeta > 0:
                rhs1_precond_kind = "adi"
            elif pre_theta > 0:
                rhs1_precond_kind = "theta_line"
            elif pre_zeta > 0:
                rhs1_precond_kind = "zeta_line"
            else:
                rhs1_precond_kind = "point"
        else:
            rhs1_precond_kind = None
    if rhs1_precond_env == "" and rhs1_precond_kind == "point" and use_pas_projection:
        # PAS tokamak-like cases benefit from a stronger line preconditioner by default.
        rhs1_precond_kind = "theta_line" if int(op.n_theta) >= int(op.n_zeta) else "zeta_line"
    if str(solve_method).strip().lower() in {"dense", "dense_ksp", "dense_row_scaled"}:
        rhs1_precond_kind = None
    rhs1_precond_enabled = (
        rhs1_precond_kind is not None
        and int(op.rhs_mode) == 1
        and (not bool(op.include_phi1))
    )
    if rhs1_bicgstab_env in {"0", "false", "no", "off"}:
        rhs1_bicgstab_kind = None
    elif rhs1_bicgstab_env in {"rhs1", "same", "preconditioner"}:
        rhs1_bicgstab_kind = "rhs1"
    elif rhs1_bicgstab_env in {"", "1", "true", "yes", "on", "collision", "diag"}:
        rhs1_bicgstab_kind = "collision"
    else:
        rhs1_bicgstab_kind = None
    if (
        rhs1_bicgstab_kind == "collision"
        and op.fblock.fp is not None
        and rhs1_precond_kind not in {None, "collision"}
    ):
        rhs1_bicgstab_kind = "rhs1"
    solve_method_kind = str(solve_method).strip().lower()
    if solve_method_kind == "dense_ksp":
        # `dense_ksp` uses its own PETSc-like block preconditioner on the assembled dense system.
        rhs1_precond_enabled = False
    gmres_precond_side_env = os.environ.get("SFINCS_JAX_GMRES_PRECONDITION_SIDE", "").strip().lower()
    if gmres_precond_side_env not in {"", "left", "right", "none"}:
        gmres_precond_side_env = ""
    # Upstream SFINCS v3 reports KSP residual norms for the *preconditioned* residual, matching
    # a left-preconditioned solve. Default to left to align solver-branch parity.
    gmres_precond_side = gmres_precond_side_env or "left"

    bicgstab_fallback_env = os.environ.get("SFINCS_JAX_BICGSTAB_FALLBACK", "").strip().lower()
    if bicgstab_fallback_env in {"0", "false", "no", "off"}:
        bicgstab_fallback_strict = False
    elif bicgstab_fallback_env in {"1", "true", "yes", "on", "strict"}:
        bicgstab_fallback_strict = True
    else:
        # Default to strict fallback to preserve parity when BiCGStab stagnates.
        bicgstab_fallback_strict = True
    implicit_env = os.environ.get("SFINCS_JAX_IMPLICIT_SOLVE", "").strip().lower()
    use_implicit = implicit_env not in {"0", "false", "no", "off"}

    sparse_precond_env = os.environ.get("SFINCS_JAX_RHSMODE1_SPARSE_PRECOND", "").strip().lower()
    if sparse_precond_env in {"jax", "jax_native", "native"}:
        sparse_precond_mode = "on"
        sparse_precond_kind = "jax"
    elif sparse_precond_env in {"scipy", "ilu", "spilu"}:
        sparse_precond_mode = "on"
        sparse_precond_kind = "scipy"
    elif sparse_precond_env in {"1", "true", "yes", "on"}:
        sparse_precond_mode = "on"
        sparse_precond_kind = "auto"
    elif sparse_precond_env in {"0", "false", "no", "off"}:
        sparse_precond_mode = "off"
        sparse_precond_kind = "auto"
    else:
        sparse_precond_mode = "auto"
        sparse_precond_kind = "auto"
    sparse_allow_nondiff_env = os.environ.get("SFINCS_JAX_RHSMODE1_SPARSE_ALLOW_NONDIFF", "").strip().lower()
    sparse_allow_nondiff = sparse_allow_nondiff_env in {"1", "true", "yes", "on"}
    sparse_matvec_env = os.environ.get("SFINCS_JAX_RHSMODE1_SPARSE_MATVEC", "").strip().lower()
    if sparse_matvec_env in {"1", "true", "yes", "on"}:
        sparse_use_matvec = True
    elif sparse_matvec_env in {"0", "false", "no", "off"}:
        sparse_use_matvec = False
    else:
        sparse_use_matvec = False
    sparse_operator_env = os.environ.get("SFINCS_JAX_RHSMODE1_SPARSE_OPERATOR", "").strip().lower()
    if sparse_operator_env in {"1", "true", "yes", "on"}:
        sparse_operator_mode = "on"
    elif sparse_operator_env in {"0", "false", "no", "off"}:
        sparse_operator_mode = "off"
    else:
        sparse_operator_mode = "auto"
    sparse_max_env = os.environ.get("SFINCS_JAX_RHSMODE1_SPARSE_MAX", "").strip()
    try:
        sparse_max_size = int(sparse_max_env) if sparse_max_env else 4000
    except ValueError:
        sparse_max_size = 4000
    sparse_drop_tol_env = os.environ.get("SFINCS_JAX_RHSMODE1_SPARSE_DROP_TOL", "").strip()
    sparse_drop_rel_env = os.environ.get("SFINCS_JAX_RHSMODE1_SPARSE_DROP_REL", "").strip()
    sparse_ilu_drop_env = os.environ.get("SFINCS_JAX_RHSMODE1_SPARSE_ILU_DROP_TOL", "").strip()
    sparse_ilu_fill_env = os.environ.get("SFINCS_JAX_RHSMODE1_SPARSE_ILU_FILL_FACTOR", "").strip()
    sparse_ilu_dense_env = os.environ.get("SFINCS_JAX_RHSMODE1_SPARSE_ILU_DENSE_MAX", "").strip()
    sparse_dense_cache_env = os.environ.get("SFINCS_JAX_RHSMODE1_SPARSE_DENSE_CACHE_MAX", "").strip()
    try:
        sparse_drop_tol = float(sparse_drop_tol_env) if sparse_drop_tol_env else 0.0
    except ValueError:
        sparse_drop_tol = 0.0
    try:
        sparse_drop_rel = float(sparse_drop_rel_env) if sparse_drop_rel_env else 1.0e-6
    except ValueError:
        sparse_drop_rel = 1.0e-6
    try:
        sparse_ilu_drop_tol = float(sparse_ilu_drop_env) if sparse_ilu_drop_env else 1.0e-4
    except ValueError:
        sparse_ilu_drop_tol = 1.0e-4
    try:
        sparse_ilu_fill = float(sparse_ilu_fill_env) if sparse_ilu_fill_env else 10.0
    except ValueError:
        sparse_ilu_fill = 10.0
    try:
        sparse_ilu_dense_max = int(sparse_ilu_dense_env) if sparse_ilu_dense_env else 2500
    except ValueError:
        sparse_ilu_dense_max = 2500
    try:
        sparse_dense_cache_max = int(sparse_dense_cache_env) if sparse_dense_cache_env else 3000
    except ValueError:
        sparse_dense_cache_max = 3000
    sparse_jax_max_env = os.environ.get("SFINCS_JAX_RHSMODE1_SPARSE_JAX_MAX_MB", "").strip()
    try:
        sparse_jax_max_mb = float(sparse_jax_max_env) if sparse_jax_max_env else 128.0
    except ValueError:
        sparse_jax_max_mb = 128.0
    sparse_jax_sweeps_env = os.environ.get("SFINCS_JAX_RHSMODE1_SPARSE_JAX_SWEEPS", "").strip()
    try:
        sparse_jax_sweeps = int(sparse_jax_sweeps_env) if sparse_jax_sweeps_env else 2
    except ValueError:
        sparse_jax_sweeps = 2
    sparse_jax_sweeps = max(1, sparse_jax_sweeps)
    sparse_jax_omega_env = os.environ.get("SFINCS_JAX_RHSMODE1_SPARSE_JAX_OMEGA", "").strip()
    try:
        sparse_jax_omega = float(sparse_jax_omega_env) if sparse_jax_omega_env else 0.8
    except ValueError:
        sparse_jax_omega = 0.8
    sparse_jax_reg_env = os.environ.get("SFINCS_JAX_RHSMODE1_SPARSE_JAX_REG", "").strip()
    try:
        sparse_jax_reg = float(sparse_jax_reg_env) if sparse_jax_reg_env else 1e-10
    except ValueError:
        sparse_jax_reg = 1e-10

    def _solver_kind(method: str) -> tuple[str, str]:
        method_l = str(method).strip().lower()
        if method_l in {"auto", "default"}:
            # Transport matrices involve constrained/near-singular systems; GMRES is
            # generally more reliable for parity, while BiCGStab remains available.
            if int(op.rhs_mode) in {2, 3}:
                return "gmres", "incremental"
            small_gmres_env = os.environ.get("SFINCS_JAX_RHSMODE1_GMRES_SMALL_MAX", "").strip()
            try:
                small_gmres_max = int(small_gmres_env) if small_gmres_env else 600
            except ValueError:
                small_gmres_max = 600
            if small_gmres_max > 0 and int(op.total_size) <= small_gmres_max:
                return "gmres", "incremental"
            # Default RHSMode=1 to GMRES to match PETSc/KSP parity.
            return "gmres", "incremental"
        if method_l in {"bicgstab", "bicgstab_jax"}:
            return "bicgstab", "batched"
        return "gmres", method_l

    stage2_env = os.environ.get("SFINCS_JAX_LINEAR_STAGE2", "").strip().lower()
    if stage2_env in {"0", "false", "no", "off"}:
        stage2_enabled = False
    elif stage2_env in {"1", "true", "yes", "on"}:
        stage2_enabled = True
    else:
        solver_kind_default = _solver_kind(solve_method)[0]
        stage2_enabled = (
            int(op.rhs_mode) == 1
            and (not bool(op.include_phi1))
            and solver_kind_default == "gmres"
        )
    # Stage-2 is a "stronger" fallback solve for difficult cases. The default time cap
    # must be large enough to still trigger after any one-time preconditioner setup,
    # while remaining bounded for interactive use and CI.
    stage2_time_cap_s = float(os.environ.get("SFINCS_JAX_LINEAR_STAGE2_MAX_ELAPSED_S", "30.0"))

    def _solve_linear(
        *,
        matvec_fn,
        b_vec: jnp.ndarray,
        precond_fn,
        x0_vec: jnp.ndarray | None,
        tol_val: float,
        atol_val: float,
        restart_val: int,
        maxiter_val: int | None,
        solve_method_val: str,
        precond_side: str,
    ):
        if use_implicit:
            solver_kind, gmres_method = _solver_kind(solve_method_val)
            return linear_custom_solve(
                matvec=matvec_fn,
                b=b_vec,
                preconditioner=precond_fn,
                x0=x0_vec,
                tol=tol_val,
                atol=atol_val,
                restart=restart_val,
                maxiter=maxiter_val,
                solve_method=gmres_method,
                solver=solver_kind,
                precondition_side=precond_side,
                size_hint=int(op.total_size),
            )
        return _gmres_solve_dispatch(
            matvec=matvec_fn,
            b=b_vec,
            preconditioner=precond_fn,
            x0=x0_vec,
            tol=tol_val,
            atol=atol_val,
            restart=restart_val,
            maxiter=maxiter_val,
            solve_method=solve_method_val,
            precondition_side=precond_side,
        )

    def _solve_linear_with_residual(
        *,
        matvec_fn,
        b_vec: jnp.ndarray,
        precond_fn,
        x0_vec: jnp.ndarray | None,
        tol_val: float,
        atol_val: float,
        restart_val: int,
        maxiter_val: int | None,
        solve_method_val: str,
        precond_side: str,
    ) -> tuple[GMRESSolveResult, jnp.ndarray]:
        solver_kind, gmres_method = _solver_kind(solve_method_val)
        if use_implicit:
            return linear_custom_solve_with_residual(
                matvec=matvec_fn,
                b=b_vec,
                preconditioner=precond_fn,
                x0=x0_vec,
                tol=tol_val,
                atol=atol_val,
                restart=restart_val,
                maxiter=maxiter_val,
                solve_method=gmres_method,
                solver=solver_kind,
                precondition_side=precond_side,
                size_hint=int(op.total_size),
            )
        if solver_kind == "bicgstab":
            solver_fn = bicgstab_solve_with_residual_jit if _use_solver_jit() else bicgstab_solve_with_residual
            return solver_fn(
                matvec=matvec_fn,
                b=b_vec,
                preconditioner=precond_fn,
                x0=x0_vec,
                tol=tol_val,
                atol=atol_val,
                maxiter=maxiter_val,
                precondition_side=precond_side,
            )
        solver_fn = gmres_solve_with_residual_jit if _use_solver_jit() else gmres_solve_with_residual
        return solver_fn(
            matvec=matvec_fn,
            b=b_vec,
            preconditioner=precond_fn,
            x0=x0_vec,
            tol=tol_val,
            atol=atol_val,
            restart=restart_val,
            maxiter=maxiter_val,
            solve_method=gmres_method,
            precondition_side=precond_side,
        )

    fortran_stdout_env = os.environ.get("SFINCS_JAX_FORTRAN_STDOUT", "").strip().lower()
    if fortran_stdout_env in {"0", "false", "no", "off"}:
        fortran_stdout = False
    elif fortran_stdout_env in {"1", "true", "yes", "on"}:
        fortran_stdout = True
    else:
        fortran_stdout = emit is not None
    ksp_history_max_env = os.environ.get("SFINCS_JAX_KSP_HISTORY_MAX_SIZE", "").strip().lower()
    if ksp_history_max_env in {"none", "inf", "infinite", "unlimited"}:
        ksp_history_max_size = None
    else:
        try:
            ksp_history_max_size = int(ksp_history_max_env) if ksp_history_max_env else 800
        except ValueError:
            ksp_history_max_size = 800
    ksp_history_max_iter_env = os.environ.get("SFINCS_JAX_KSP_HISTORY_MAX_ITER", "").strip()
    try:
        ksp_history_max_iter = int(ksp_history_max_iter_env) if ksp_history_max_iter_env else 2000
    except ValueError:
        ksp_history_max_iter = 2000
    iter_stats_env = os.environ.get("SFINCS_JAX_SOLVER_ITER_STATS", "").strip().lower()
    iter_stats_enabled = iter_stats_env in {"1", "true", "yes", "on"}
    iter_stats_max_env = os.environ.get("SFINCS_JAX_SOLVER_ITER_STATS_MAX_SIZE", "").strip()
    try:
        iter_stats_max_size = int(iter_stats_max_env) if iter_stats_max_env else None
    except ValueError:
        iter_stats_max_size = None

    ksp_matvec = None
    ksp_b = None
    ksp_precond = None
    ksp_x0 = None
    ksp_restart = restart
    ksp_maxiter = maxiter
    ksp_precond_side = gmres_precond_side
    ksp_solver_kind = "gmres"
    residual_vec: jnp.ndarray | None = None

    def _emit_ksp_history(
        *,
        matvec_fn,
        b_vec: jnp.ndarray,
        precond_fn,
        x0_vec: jnp.ndarray | None,
        tol_val: float,
        atol_val: float,
        restart_val: int,
        maxiter_val: int | None,
        precond_side: str,
        solver_kind: str,
        solve_method_val: str,
    ) -> list[float] | None:
        if emit is None or not fortran_stdout:
            return None
        if str(solver_kind).strip().lower() != "gmres":
            return None
        size = int(b_vec.size)
        if ksp_history_max_size is not None and size > int(ksp_history_max_size):
            emit(1, f"fortran-stdout: KSP history skipped (size={size} > max={int(ksp_history_max_size)})")
            return None
        if maxiter_val is not None and ksp_history_max_iter is not None:
            est_iters = int(maxiter_val)
            if str(solver_kind).strip().lower() == "gmres":
                est_iters *= max(1, int(restart_val))
            if est_iters > int(ksp_history_max_iter):
                emit(
                    1,
                    "fortran-stdout: KSP history skipped "
                    f"(estimated_iters={est_iters} > max={int(ksp_history_max_iter)})",
                )
                return None
        try:
            _x_hist, _rn, history = gmres_solve_with_history_scipy(
                matvec=matvec_fn,
                b=b_vec,
                preconditioner=precond_fn,
                x0=x0_vec,
                tol=tol_val,
                atol=atol_val,
                restart=restart_val,
                maxiter=maxiter_val,
                precondition_side=precond_side,
            )
        except Exception as exc:  # noqa: BLE001
            emit(1, f"fortran-stdout: KSP history unavailable ({type(exc).__name__}: {exc})")
            return None
        for k, rn in enumerate(history):
            emit(0, f"{k:4d} KSP Residual norm {rn: .12e} ")
        if history:
            emit(0, " Linear iteration (KSP) converged.  KSPConvergedReason =            2")
            emit(0, "   KSP_CONVERGED_RTOL: Norm decreased by rtol.")
        return history

    def _emit_ksp_iter_stats(
        *,
        matvec_fn,
        b_vec: jnp.ndarray,
        precond_fn,
        x0_vec: jnp.ndarray | None,
        tol_val: float,
        atol_val: float,
        restart_val: int,
        maxiter_val: int | None,
        precond_side: str,
        solver_kind: str,
        history: list[float] | None,
        solve_method_val: str,
    ) -> None:
        if emit is None or not iter_stats_enabled:
            return
        size = int(b_vec.size)
        if iter_stats_max_size is not None and size > int(iter_stats_max_size):
            emit(1, f"ksp_iterations skipped (size={size} > max={int(iter_stats_max_size)})")
            return
        solver_kind_l = str(solver_kind).strip().lower()
        iter_stats_max_iter_env = os.environ.get("SFINCS_JAX_SOLVER_ITER_STATS_MAX_ITER", "").strip()
        try:
            iter_stats_max_iter = int(iter_stats_max_iter_env) if iter_stats_max_iter_env else 2000
        except ValueError:
            iter_stats_max_iter = 2000
        if maxiter_val is not None and iter_stats_max_iter is not None:
            est_iters = int(maxiter_val)
            if solver_kind_l == "gmres":
                est_iters *= max(1, int(restart_val))
            if est_iters > int(iter_stats_max_iter):
                emit(
                    1,
                    "ksp_iterations skipped "
                    f"(estimated_iters={est_iters} > max={int(iter_stats_max_iter)})",
                )
                return
        try:
            if solver_kind_l == "gmres":
                if history is None:
                    _x_hist, _rn, history = gmres_solve_with_history_scipy(
                        matvec=matvec_fn,
                        b=b_vec,
                        preconditioner=precond_fn,
                        x0=x0_vec,
                        tol=tol_val,
                        atol=atol_val,
                        restart=restart_val,
                        maxiter=maxiter_val,
                        precondition_side=precond_side,
                    )
                iters = len(history or [])
            elif solver_kind_l == "bicgstab":
                _x_hist, _rn, history = bicgstab_solve_with_history_scipy(
                    matvec=matvec_fn,
                    b=b_vec,
                    preconditioner=precond_fn,
                    x0=x0_vec,
                    tol=tol_val,
                    atol=atol_val,
                    maxiter=maxiter_val,
                    precondition_side=precond_side,
                )
                iters = len(history or [])
            else:
                return
        except Exception as exc:  # noqa: BLE001
            emit(1, f"ksp_iterations unavailable ({type(exc).__name__}: {exc})")
            return
        emit(0, f"ksp_iterations={iters} solver={solver_kind_l}")
    if use_active_dof_mode:
        assert active_idx_jnp is not None
        assert full_to_active_jnp is not None

        def reduce_full(v_full: jnp.ndarray) -> jnp.ndarray:
            return v_full[active_idx_jnp]

        if use_pas_projection:
            fs_factor = _fs_average_factor(op.theta_weights, op.zeta_weights, op.d_hat)
            fs_sum = jnp.sum(fs_factor)
            fs_sum_safe = jnp.where(fs_sum != 0, fs_sum, jnp.asarray(1.0, dtype=jnp.float64))
            ix0 = _ix_min(bool(op.point_at_x0))
            mask_x = (jnp.arange(int(op.n_x)) >= ix0).astype(jnp.float64)

            def _project_pas_f(f_flat: jnp.ndarray) -> jnp.ndarray:
                f = f_flat.reshape(op.fblock.f_shape)
                avg = jnp.einsum("tz,sxtz->sx", fs_factor, f[:, :, 0, :, :])
                avg = avg * mask_x[None, :]
                avg = avg / fs_sum_safe
                f = f.at[:, :, 0, :, :].add(-avg[:, :, None, None])
                return f.reshape((-1,))

            def _expand_active_f(v_reduced: jnp.ndarray) -> jnp.ndarray:
                z0 = jnp.zeros((1,), dtype=v_reduced.dtype)
                padded = jnp.concatenate([z0, v_reduced], axis=0)
                return padded[full_to_active_jnp]

            def _project_reduced(v_reduced: jnp.ndarray) -> jnp.ndarray:
                f_full = _expand_active_f(v_reduced)
                f_proj = _project_pas_f(f_full)
                return reduce_full(f_proj)

            def _wrap_pas_precond(precond_fn: Callable[[jnp.ndarray], jnp.ndarray]) -> Callable[[jnp.ndarray], jnp.ndarray]:
                def _apply(v_reduced: jnp.ndarray) -> jnp.ndarray:
                    z_reduced = precond_fn(v_reduced)
                    return _project_reduced(z_reduced)
                return _apply

            def expand_reduced(v_reduced: jnp.ndarray) -> jnp.ndarray:
                f_full = _expand_active_f(v_reduced)
                if int(op.extra_size) > 0:
                    zeros_e = jnp.zeros((int(op.extra_size),), dtype=v_reduced.dtype)
                    return jnp.concatenate([f_full, zeros_e], axis=0)
                return f_full

            zeros_extra = jnp.zeros((int(op.extra_size),), dtype=jnp.float64)

            def mv_reduced(x_reduced: jnp.ndarray) -> jnp.ndarray:
                f_full = _expand_active_f(x_reduced)
                f_proj = _project_pas_f(f_full)
                x_full = jnp.concatenate([f_proj, zeros_extra], axis=0) if int(op.extra_size) > 0 else f_proj
                y_full = mv(x_full)
                y_f = y_full[: op.f_size]
                y_proj = _project_pas_f(y_f)
                return reduce_full(y_proj)

            rhs_f = rhs[: op.f_size]
            rhs_proj = _project_pas_f(rhs_f)
            rhs_reduced = reduce_full(rhs_proj)
            x0_reduced = None
            if x0 is not None:
                x0_arr = jnp.asarray(x0)
                if x0_arr.shape == (active_size,):
                    x0_reduced = _project_reduced(x0_arr)
                elif x0_arr.shape == (op.total_size,):
                    f0_proj = _project_pas_f(x0_arr[: op.f_size])
                    x0_reduced = reduce_full(f0_proj)
                elif x0_arr.shape == (op.f_size,):
                    f0_proj = _project_pas_f(x0_arr)
                    x0_reduced = reduce_full(f0_proj)
            if recycle_basis_use:
                basis_reduced: list[jnp.ndarray] = []
                for vec in recycle_basis_use:
                    if vec.shape != (op.total_size,):
                        continue
                    f_proj = _project_pas_f(vec[: op.f_size])
                    basis_reduced.append(reduce_full(f_proj))
                if basis_reduced:
                    basis_au = [mv_reduced(b) for b in basis_reduced]
                    x0_recycled = _recycled_initial_guess(rhs_reduced, basis_reduced, basis_au)
                    if x0_recycled is not None:
                        if x0_reduced is None:
                            x0_reduced = x0_recycled
                        else:
                            r0 = jnp.linalg.norm(mv_reduced(x0_reduced) - rhs_reduced)
                            r1 = jnp.linalg.norm(mv_reduced(x0_recycled) - rhs_reduced)
                            if jnp.isfinite(r1) and (not jnp.isfinite(r0) or float(r1) < float(r0)):
                                x0_reduced = x0_recycled
        else:
            def expand_reduced(v_reduced: jnp.ndarray) -> jnp.ndarray:
                z0 = jnp.zeros((1,), dtype=v_reduced.dtype)
                padded = jnp.concatenate([z0, v_reduced], axis=0)
                return padded[full_to_active_jnp]

            def mv_reduced(x_reduced: jnp.ndarray) -> jnp.ndarray:
                return reduce_full(mv(expand_reduced(x_reduced)))

            rhs_reduced = reduce_full(rhs)
            x0_reduced = None
            if x0 is not None:
                x0_arr = jnp.asarray(x0)
                if x0_arr.shape == (active_size,):
                    x0_reduced = x0_arr
                elif x0_arr.shape == (op.total_size,):
                    x0_reduced = reduce_full(x0_arr)
                elif use_pas_projection and x0_arr.shape == (op.f_size,):
                    x0_reduced = x0_arr[active_idx_jnp]
            if recycle_basis_use:
                basis_reduced = []
                for vec in recycle_basis_use:
                    if vec.shape != (op.total_size,):
                        continue
                    basis_reduced.append(reduce_full(vec))
                if basis_reduced:
                    basis_au = [mv_reduced(b) for b in basis_reduced]
                    x0_recycled = _recycled_initial_guess(rhs_reduced, basis_reduced, basis_au)
                    if x0_recycled is not None:
                        if x0_reduced is None:
                            x0_reduced = x0_recycled
                        else:
                            r0 = jnp.linalg.norm(mv_reduced(x0_reduced) - rhs_reduced)
                            r1 = jnp.linalg.norm(mv_reduced(x0_recycled) - rhs_reduced)
                            if jnp.isfinite(r1) and (not jnp.isfinite(r0) or float(r1) < float(r0)):
                                x0_reduced = x0_recycled
        target_reduced = max(float(atol), float(tol) * float(jnp.linalg.norm(rhs_reduced)))
        dense_shortcut_ratio_env = os.environ.get("SFINCS_JAX_RHSMODE1_DENSE_SHORTCUT_RATIO", "").strip()
        try:
            dense_shortcut_ratio = float(dense_shortcut_ratio_env) if dense_shortcut_ratio_env else 1.0e6
        except ValueError:
            dense_shortcut_ratio = 1.0e6
        dense_fallback_max = _rhsmode1_dense_fallback_max(op)
        early_dense_shortcut = False
        probe_shortcut = False
        probe_x0: jnp.ndarray | None = None
        preconditioner_reduced = None
        bicgstab_preconditioner_reduced = None
        pas_precond_force_collision = False
        if rhs1_bicgstab_kind is not None:
            if emit is not None:
                emit(1, f"solve_v3_full_system_linear_gmres: RHSMode=1 BiCGStab preconditioner={rhs1_bicgstab_kind}")
            if rhs1_bicgstab_kind == "collision":
                bicgstab_preconditioner_reduced = _build_rhsmode1_collision_preconditioner(
                    op=op, reduce_full=reduce_full, expand_reduced=expand_reduced
                )
            if use_pas_projection:
                bicgstab_preconditioner_reduced = _wrap_pas_precond(bicgstab_preconditioner_reduced)

        # PAS probe shortcut: avoid expensive block/line preconditioner builds when a
        # cheap collision-based preconditioner already provides a strong residual drop.
        pas_probe_env = os.environ.get("SFINCS_JAX_PAS_PRECOND_PROBE", "").strip().lower()
        pas_probe_enabled = pas_probe_env not in {"0", "false", "no", "off"}
        pas_probe_rel_env = os.environ.get("SFINCS_JAX_PAS_PRECOND_PROBE_REL_MAX", "").strip()
        try:
            pas_probe_rel_max = float(pas_probe_rel_env) if pas_probe_rel_env else 0.9
        except ValueError:
            pas_probe_rel_max = 0.9
        pas_build_max_env = os.environ.get("SFINCS_JAX_PAS_PRECOND_BUILD_MAX", "").strip()
        try:
            pas_build_max = int(pas_build_max_env) if pas_build_max_env else 20000
        except ValueError:
            pas_build_max = 20000
        heavy_precond_kinds = {
            "point",
            "theta_line",
            "zeta_line",
            "theta_zeta",
            "adi",
            "xblock_tz",
            "sxblock_tz",
            "species_block",
            "schur",
        }
        if (
            pas_probe_enabled
            and rhs1_precond_kind in heavy_precond_kinds
            and rhs1_precond_enabled
            and solve_method_kind not in {"dense", "dense_ksp"}
            and op.fblock.pas is not None
        ):
            probe_key = _rhsmode1_precond_cache_key(op, "pas_probe_decision")
            use_collision_precond = _RHSMODE1_PAS_PRECOND_PROBE_CACHE.get(probe_key)
            if use_collision_precond is None and int(op.total_size) >= int(pas_build_max):
                use_collision_precond = True
                _RHSMODE1_PAS_PRECOND_PROBE_CACHE[probe_key] = True
                if emit is not None:
                    emit(
                        1,
                        "solve_v3_full_system_linear_gmres: PAS precond skip "
                        f"(size={int(op.total_size)} >= {int(pas_build_max)}) -> collision",
                    )
            if use_collision_precond is None:
                try:
                    probe_precond = _build_rhsmode1_collision_preconditioner(
                        op=op, reduce_full=reduce_full, expand_reduced=expand_reduced
                    )
                    if use_pas_projection:
                        probe_precond = _wrap_pas_precond(probe_precond)
                    probe_x = probe_precond(rhs_reduced)
                    probe_r = rhs_reduced - mv_reduced(probe_x)
                    rhs_norm = float(jnp.linalg.norm(rhs_reduced))
                    probe_rel = float(jnp.linalg.norm(probe_r)) / rhs_norm if rhs_norm > 0 else 0.0
                    use_collision_precond = probe_rel <= pas_probe_rel_max
                    _RHSMODE1_PAS_PRECOND_PROBE_CACHE[probe_key] = bool(use_collision_precond)
                    if emit is not None:
                        emit(
                            1,
                            "solve_v3_full_system_linear_gmres: PAS precond probe "
                            f"(rel={probe_rel:.3e}, max={pas_probe_rel_max:.3e}) -> "
                            f"{'collision' if use_collision_precond else 'full'}",
                        )
                except Exception as exc:  # noqa: BLE001
                    use_collision_precond = None
                    if emit is not None:
                        emit(1, f"solve_v3_full_system_linear_gmres: PAS precond probe failed ({type(exc).__name__}: {exc})")
            if use_collision_precond:
                preconditioner_reduced = _build_rhsmode1_collision_preconditioner(
                    op=op, reduce_full=reduce_full, expand_reduced=expand_reduced
                )
                if use_pas_projection:
                    preconditioner_reduced = _wrap_pas_precond(preconditioner_reduced)
                rhs1_precond_kind = "collision"
                pas_precond_force_collision = True
                if rhs1_bicgstab_kind == "rhs1":
                    bicgstab_preconditioner_reduced = preconditioner_reduced

        def _build_rhs1_preconditioner_reduced():
            _mark("rhs1_precond_build_start")
            if emit is not None:
                emit(
                    1,
                    "solve_v3_full_system_linear_gmres: building RHSMode=1 preconditioner="
                    f"{rhs1_precond_kind} (active-DOF)",
                )
            if rhs1_precond_kind == "theta_line":
                precond = _build_rhsmode1_theta_line_preconditioner(
                    op=op, reduce_full=reduce_full, expand_reduced=expand_reduced
                )
            elif rhs1_precond_kind == "species_block":
                precond = _build_rhsmode1_species_block_preconditioner(
                    op=op, reduce_full=reduce_full, expand_reduced=expand_reduced
                )
            elif rhs1_precond_kind == "sxblock":
                precond = _build_rhsmode1_species_xblock_preconditioner(
                    op=op, reduce_full=reduce_full, expand_reduced=expand_reduced
                )
            elif rhs1_precond_kind == "sxblock_tz":
                precond = _build_rhsmode1_sxblock_tz_preconditioner(
                    op=op, reduce_full=reduce_full, expand_reduced=expand_reduced
                )
            elif rhs1_precond_kind == "xblock_tz":
                precond = _build_rhsmode1_xblock_tz_preconditioner(
                    op=op, reduce_full=reduce_full, expand_reduced=expand_reduced
                )
            elif rhs1_precond_kind == "theta_zeta":
                precond = _build_rhsmode1_theta_zeta_preconditioner(
                    op=op, reduce_full=reduce_full, expand_reduced=expand_reduced
                )
            elif rhs1_precond_kind == "zeta_line":
                precond = _build_rhsmode1_zeta_line_preconditioner(
                    op=op, reduce_full=reduce_full, expand_reduced=expand_reduced
                )
            elif rhs1_precond_kind == "schur":
                precond = _build_rhsmode1_schur_preconditioner(
                    op=op, reduce_full=reduce_full, expand_reduced=expand_reduced
                )
            elif rhs1_precond_kind == "collision":
                precond = _build_rhsmode1_collision_preconditioner(
                    op=op, reduce_full=reduce_full, expand_reduced=expand_reduced
                )
            elif rhs1_precond_kind == "adi":
                pre_theta = _build_rhsmode1_theta_line_preconditioner(
                    op=op, reduce_full=reduce_full, expand_reduced=expand_reduced
                )
                pre_zeta = _build_rhsmode1_zeta_line_preconditioner(
                    op=op, reduce_full=reduce_full, expand_reduced=expand_reduced
                )

                sweeps_env = os.environ.get("SFINCS_JAX_RHSMODE1_ADI_SWEEPS", "").strip()
                try:
                    sweeps = int(sweeps_env) if sweeps_env else 2
                except ValueError:
                    sweeps = 2
                sweeps = max(1, sweeps)

                def preconditioner_reduced(v: jnp.ndarray) -> jnp.ndarray:
                    out = v
                    for _ in range(sweeps):
                        out = pre_zeta(pre_theta(out))
                    return out
                precond = preconditioner_reduced
            else:
                precond = _build_rhsmode1_block_preconditioner(
                    op=op,
                    reduce_full=reduce_full,
                    expand_reduced=expand_reduced,
                    preconditioner_species=preconditioner_species,
                    preconditioner_x=preconditioner_x,
                    preconditioner_xi=preconditioner_xi,
                )
            precond = _wrap_pas_precond(precond) if use_pas_projection else precond
            _mark("rhs1_precond_build_done")
            return precond

        if rhs1_precond_enabled:
            solver_kind = _solver_kind(solve_method)[0]
            build_rhs1 = (
                (solver_kind != "bicgstab" and solve_method_kind != "dense")
                or (rhs1_bicgstab_kind == "rhs1" and solve_method_kind != "dense")
            )
            if build_rhs1 and preconditioner_reduced is None:
                preconditioner_reduced = _build_rhs1_preconditioner_reduced()
                if rhs1_bicgstab_kind == "rhs1":
                    bicgstab_preconditioner_reduced = preconditioner_reduced
        if preconditioner_reduced is None and bicgstab_preconditioner_reduced is not None:
            preconditioner_reduced = bicgstab_preconditioner_reduced
        sparse_operator_use = False
        if sparse_operator_mode == "on":
            sparse_operator_use = True
        elif sparse_operator_mode == "auto":
            sparse_operator_use = sparse_use_matvec and (op.fblock.fp is not None)
        if sparse_operator_use:
            sparse_operator_use = int(op.rhs_mode) == 1 and (not bool(op.include_phi1))
        if sparse_operator_use:
            if use_implicit and not sparse_allow_nondiff:
                sparse_operator_use = False
                if emit is not None:
                    emit(1, "sparse_operator: disabled for implicit solves (set SFINCS_JAX_RHSMODE1_SPARSE_ALLOW_NONDIFF=1 to override)")
            elif int(active_size) > sparse_max_size:
                sparse_operator_use = False
                if emit is not None:
                    emit(1, f"sparse_operator: disabled (size={int(active_size)} > max={int(sparse_max_size)})")
        if sparse_operator_use:
            try:
                cache_key = _rhsmode1_sparse_cache_key(
                    op,
                    kind="sparse_operator",
                    active_size=int(active_size),
                    use_active_dof_mode=True,
                    use_pas_projection=use_pas_projection,
                    drop_tol=sparse_drop_tol,
                    drop_rel=sparse_drop_rel,
                    ilu_drop_tol=sparse_ilu_drop_tol,
                    fill_factor=sparse_ilu_fill,
                )
                a_csr_full, _a_csr_drop, _ilu, _a_dense, _l_dense, _u_dense, _l_unit = _build_sparse_ilu_from_matvec(
                    matvec=mv_reduced,
                    n=int(active_size),
                    dtype=rhs_reduced.dtype,
                    cache_key=cache_key,
                    drop_tol=sparse_drop_tol,
                    drop_rel=sparse_drop_rel,
                    ilu_drop_tol=sparse_ilu_drop_tol,
                    fill_factor=sparse_ilu_fill,
                    build_dense_factors=False,
                    build_ilu=False,
                    store_dense=False,
                    emit=emit,
                )

                def _mv_sparse(v: jnp.ndarray) -> jnp.ndarray:
                    x_np = np.asarray(v, dtype=np.float64).reshape((-1,))
                    y_np = a_csr_full @ x_np
                    return jnp.asarray(y_np, dtype=jnp.float64)

                mv_reduced = _mv_sparse
                if emit is not None:
                    emit(0, "solve_v3_full_system_linear_gmres: using sparse operator matvec")
            except Exception as exc:  # noqa: BLE001
                if emit is not None:
                    emit(1, f"sparse_operator: failed ({type(exc).__name__}: {exc})")
        probe_env = os.environ.get("SFINCS_JAX_RHSMODE1_DENSE_PROBE", "").strip().lower()
        probe_enabled = probe_env not in {"0", "false", "no", "off"}
        if (
            probe_enabled
            and (not probe_shortcut)
            and preconditioner_reduced is not None
            and solve_method_kind not in {"dense", "dense_ksp"}
        ):
            try:
                probe_x0 = preconditioner_reduced(rhs_reduced)
                probe_r = rhs_reduced - mv_reduced(probe_x0)
                probe_norm = float(jnp.linalg.norm(probe_r))
                probe_ratio = probe_norm / max(float(target_reduced), 1e-300)
                if dense_shortcut_ratio > 0 and probe_ratio >= dense_shortcut_ratio:
                    early_dense_shortcut = True
                    allow_probe_shortcut = dense_fallback_max > 0 and int(active_size) <= dense_fallback_max
                    if allow_probe_shortcut:
                        probe_shortcut = True
                        res_reduced = GMRESSolveResult(x=probe_x0, residual_norm=jnp.asarray(probe_norm))
                        ksp_matvec = mv_reduced
                        ksp_b = rhs_reduced
                        ksp_precond = preconditioner_reduced
                        ksp_x0 = probe_x0
                        ksp_precond_side = gmres_precond_side
                        ksp_solver_kind = _solver_kind(solve_method)[0]
                        if emit is not None:
                            emit(
                                0,
                                "solve_v3_full_system_linear_gmres: dense fallback shortcut (probe) "
                                f"(ratio={probe_ratio:.3e} >= {dense_shortcut_ratio:.1e})",
                            )
                    else:
                        if emit is not None:
                            emit(
                                1,
                                "solve_v3_full_system_linear_gmres: probe shortcut skipped "
                                f"(size={int(active_size)} > dense_max={dense_fallback_max})",
                            )
                        if x0_reduced is None:
                            x0_reduced = probe_x0
                elif x0_reduced is None:
                    x0_reduced = probe_x0
            except Exception as exc:  # noqa: BLE001
                if emit is not None:
                    emit(1, f"solve_v3_full_system_linear_gmres: probe failed ({type(exc).__name__}: {exc})")
        if probe_shortcut:
            pass
        elif solve_method_kind == "dense_ksp":
            if int(op.phi1_size) != 0:
                raise NotImplementedError("dense_ksp is only supported for includePhi1=false RHSMode=1 solves.")
            if emit is not None:
                emit(1, "solve_v3_full_system_linear_gmres: assembling dense reduced matrix for dense_ksp")
            a_dense = assemble_dense_matrix_from_matvec(matvec=mv_reduced, n=active_size, dtype=rhs_reduced.dtype)

            if emit is not None:
                emit(1, "solve_v3_full_system_linear_gmres: building PETSc-like species-block preconditioner (dense_ksp)")

            import jax.scipy.linalg as jla  # noqa: PLC0415

            n_species = int(op.n_species)
            n_theta = int(op.n_theta)
            n_zeta = int(op.n_zeta)
            local_per_species = int(np.sum(nxi_for_x))
            dke_size = int(local_per_species * n_theta * n_zeta)
            extra_size = int(op.extra_size)
            extra_per_species = int(extra_size // max(1, n_species)) if extra_size else 0
            if extra_size and (extra_per_species * n_species != extra_size):
                extra_per_species = 0

            f_size = int(n_species * dke_size)
            expected_active = int(f_size + int(op.phi1_size) + extra_size)
            if int(active_size) != expected_active:
                raise RuntimeError(f"dense_ksp expects active_size={expected_active}, got {active_size}")

            lu_factors: list[tuple[jnp.ndarray, jnp.ndarray]] = []
            idx_blocks: list[jnp.ndarray] = []
            for s in range(n_species):
                f_idx = np.arange(s * dke_size, (s + 1) * dke_size, dtype=np.int32)
                extra_idx = np.arange(f_size + s * extra_per_species, f_size + (s + 1) * extra_per_species, dtype=np.int32)
                block_idx_np = np.concatenate([f_idx, extra_idx], axis=0) if extra_per_species else f_idx
                block_idx = jnp.asarray(block_idx_np, dtype=jnp.int32)
                a_block = a_dense[jnp.ix_(block_idx, block_idx)]
                lu, piv = jla.lu_factor(a_block)
                lu_factors.append((lu, piv))
                idx_blocks.append(block_idx)

            def preconditioner_dense(v: jnp.ndarray) -> jnp.ndarray:
                out = jnp.zeros_like(v)
                for block_idx, (lu, piv) in zip(idx_blocks, lu_factors, strict=True):
                    rhs_block = v[block_idx]
                    sol_block = jla.lu_solve((lu, piv), rhs_block)
                    out = out.at[block_idx].set(sol_block, unique_indices=True)
                return out

            def mv_dense(x: jnp.ndarray) -> jnp.ndarray:
                return a_dense @ x

            # PETSc v3 uses *left* preconditioning and checks convergence in the
            # preconditioned residual norm ||M^{-1} r||. To match this behavior with
            # JAX's GMRES (which uses a SciPy-style convergence check), solve the
            # explicitly left-preconditioned system:
            #   (M^{-1} A) x = (M^{-1} b).
            rhs_pc = preconditioner_dense(rhs_reduced)

            def mv_pc(x: jnp.ndarray) -> jnp.ndarray:
                return preconditioner_dense(mv_dense(x))

            res_reduced = _solve_linear(
                matvec_fn=mv_pc,
                b_vec=rhs_pc,
                precond_fn=None,
                x0_vec=x0_reduced,
                tol_val=tol,
                atol_val=atol,
                restart_val=restart,
                maxiter_val=maxiter,
                solve_method_val="incremental",
                precond_side="none",
            )
            ksp_matvec = mv_pc
            ksp_b = rhs_pc
            ksp_precond = None
            ksp_x0 = x0_reduced
            ksp_precond_side = "none"
            ksp_solver_kind = _solver_kind("incremental")[0]
        else:
            res_reduced = _solve_linear(
                matvec_fn=mv_reduced,
                b_vec=rhs_reduced,
                precond_fn=preconditioner_reduced,
                x0_vec=x0_reduced,
                tol_val=tol,
                atol_val=atol,
                restart_val=restart,
                maxiter_val=maxiter,
                solve_method_val=solve_method,
                precond_side=gmres_precond_side,
            )
            ksp_matvec = mv_reduced
            ksp_b = rhs_reduced
            ksp_precond = preconditioner_reduced
            ksp_x0 = x0_reduced
            ksp_precond_side = gmres_precond_side
            ksp_solver_kind = _solver_kind(solve_method)[0]
        if (not probe_shortcut) and preconditioner_reduced is not None and (not _gmres_result_is_finite(res_reduced)):
            if emit is not None:
                emit(0, "solve_v3_full_system_linear_gmres: preconditioned reduced GMRES returned non-finite result; retrying without preconditioner")
            res_reduced = _solve_linear(
                matvec_fn=mv_reduced,
                b_vec=rhs_reduced,
                precond_fn=None,
                x0_vec=x0_reduced,
                tol_val=tol,
                atol_val=atol,
                restart_val=restart,
                maxiter_val=maxiter,
                solve_method_val=solve_method,
                precond_side=gmres_precond_side,
            )
            ksp_matvec = mv_reduced
            ksp_b = rhs_reduced
            ksp_precond = None
            ksp_x0 = x0_reduced
            ksp_precond_side = gmres_precond_side
            ksp_solver_kind = _solver_kind(solve_method)[0]
        residual_norm_check = float(res_reduced.residual_norm)
        residual_norm_true = residual_norm_check
        try:
            r_vec = rhs_reduced - mv_reduced(res_reduced.x)
            residual_norm_true = float(jnp.linalg.norm(r_vec))
            if not np.isfinite(residual_norm_true):
                residual_norm_true = residual_norm_check
        except Exception:
            residual_norm_true = residual_norm_check
        if np.isfinite(residual_norm_true):
            res_reduced = GMRESSolveResult(
                x=res_reduced.x, residual_norm=jnp.asarray(residual_norm_true, dtype=jnp.float64)
            )
        res_ratio = float(residual_norm_true) / max(float(target_reduced), 1e-300)
        stage2_ratio_env = os.environ.get("SFINCS_JAX_LINEAR_STAGE2_RATIO", "").strip()
        try:
            stage2_ratio = float(stage2_ratio_env) if stage2_ratio_env else 1.0e2
        except ValueError:
            stage2_ratio = 1.0e2
        stage2_trigger = bool(res_ratio > stage2_ratio) if stage2_ratio > 0 else True
        if (not early_dense_shortcut) and dense_shortcut_ratio > 0 and res_ratio >= dense_shortcut_ratio:
            early_dense_shortcut = True
            if emit is not None:
                emit(
                    0,
                    "solve_v3_full_system_linear_gmres: dense fallback shortcut (early) "
                    f"(ratio={res_ratio:.3e} >= {dense_shortcut_ratio:.1e})",
                )
        solver_kind = _solver_kind(solve_method)[0]
        if solver_kind == "bicgstab" and (
            (not _gmres_result_is_finite(res_reduced))
            or (bicgstab_fallback_strict and float(res_reduced.residual_norm) > target_reduced)
        ):
            if emit is not None:
                emit(
                    0,
                    "solve_v3_full_system_linear_gmres: BiCGStab fallback to GMRES "
                    f"(residual={float(res_reduced.residual_norm):.3e} > target={target_reduced:.3e})",
                )
            if preconditioner_reduced is None and rhs1_precond_enabled:
                preconditioner_reduced = _build_rhs1_preconditioner_reduced()
            res_reduced = _solve_linear(
                matvec_fn=mv_reduced,
                b_vec=rhs_reduced,
                precond_fn=preconditioner_reduced,
                x0_vec=x0_reduced,
                tol_val=tol,
                atol_val=atol,
                restart_val=restart,
                maxiter_val=maxiter,
                solve_method_val="incremental",
                precond_side=gmres_precond_side,
            )
            ksp_matvec = mv_reduced
            ksp_b = rhs_reduced
            ksp_precond = preconditioner_reduced
            ksp_x0 = x0_reduced
            ksp_precond_side = gmres_precond_side
            ksp_solver_kind = "gmres"
        if (
            float(res_reduced.residual_norm) > target_reduced
            and stage2_enabled
            and stage2_trigger
            and not early_dense_shortcut
            and t.elapsed_s() < stage2_time_cap_s
        ):
            if preconditioner_reduced is None and rhs1_precond_enabled:
                preconditioner_reduced = _build_rhs1_preconditioner_reduced()
            stage2_maxiter = int(os.environ.get("SFINCS_JAX_LINEAR_STAGE2_MAXITER", str(max(600, int(maxiter or 400) * 2))))
            stage2_restart = int(os.environ.get("SFINCS_JAX_LINEAR_STAGE2_RESTART", str(max(120, int(restart)))))
            stage2_method = os.environ.get("SFINCS_JAX_LINEAR_STAGE2_METHOD", "incremental").strip().lower()
            if stage2_method not in {"batched", "incremental", "dense"}:
                stage2_method = "incremental"
            if emit is not None:
                emit(
                    0,
                    "solve_v3_full_system_linear_gmres: stage2 reduced GMRES "
                    f"(residual={float(res_reduced.residual_norm):.3e} > target={target_reduced:.3e}) "
                    f"restart={stage2_restart} maxiter={stage2_maxiter} method={stage2_method}",
                )
            res2 = _solve_linear(
                matvec_fn=mv_reduced,
                b_vec=rhs_reduced,
                precond_fn=preconditioner_reduced,
                x0_vec=res_reduced.x,
                tol_val=tol,
                atol_val=atol,
                restart_val=stage2_restart,
                maxiter_val=stage2_maxiter,
                solve_method_val=stage2_method,
                precond_side=gmres_precond_side,
            )
            if float(res2.residual_norm) < float(res_reduced.residual_norm):
                res_reduced = res2
                ksp_matvec = mv_reduced
                ksp_b = rhs_reduced
                ksp_precond = preconditioner_reduced
                ksp_x0 = res_reduced.x
                ksp_restart = stage2_restart
                ksp_maxiter = stage2_maxiter
                ksp_precond_side = gmres_precond_side
                ksp_solver_kind = _solver_kind(stage2_method)[0]
        res_ratio = float(res_reduced.residual_norm) / max(float(target_reduced), 1e-300)
        strong_ratio_env = os.environ.get("SFINCS_JAX_RHSMODE1_STRONG_PRECOND_RATIO", "").strip()
        try:
            strong_ratio = float(strong_ratio_env) if strong_ratio_env else 1.0
        except ValueError:
            strong_ratio = 1.0
        strong_precond_trigger = bool(res_ratio > strong_ratio) if strong_ratio > 0 else True
        if (
            float(res_reduced.residual_norm) > target_reduced
            and int(op.rhs_mode) == 1
            and (not bool(op.include_phi1))
            and rhs1_precond_kind == "point"
            and (op.fblock.fp is not None or op.fblock.pas is not None)
            and strong_precond_trigger
        ):
            if bicgstab_preconditioner_reduced is None:
                bicgstab_preconditioner_reduced = _build_rhsmode1_collision_preconditioner(
                    op=op, reduce_full=reduce_full, expand_reduced=expand_reduced
                )
            if bicgstab_preconditioner_reduced is not None:
                if emit is not None:
                    emit(
                        0,
                        "solve_v3_full_system_linear_gmres: retry with collision preconditioner "
                        f"(residual={float(res_reduced.residual_norm):.3e} > target={target_reduced:.3e})",
                    )
                res_collision = _solve_linear(
                    matvec_fn=mv_reduced,
                    b_vec=rhs_reduced,
                    precond_fn=bicgstab_preconditioner_reduced,
                    x0_vec=res_reduced.x,
                    tol_val=tol,
                    atol_val=atol,
                    restart_val=restart,
                    maxiter_val=maxiter,
                    solve_method_val="incremental",
                    precond_side=gmres_precond_side,
                )
                if float(res_collision.residual_norm) < float(res_reduced.residual_norm):
                    res_reduced = res_collision
                    ksp_matvec = mv_reduced
                    ksp_b = rhs_reduced
                    ksp_precond = bicgstab_preconditioner_reduced
                    ksp_x0 = res_reduced.x
                    ksp_restart = restart
                    ksp_maxiter = maxiter
                    ksp_precond_side = gmres_precond_side
                    ksp_solver_kind = _solver_kind("incremental")[0]
        strong_precond_min_env = os.environ.get("SFINCS_JAX_RHSMODE1_STRONG_PRECOND_MIN", "").strip()
        try:
            strong_precond_min = int(strong_precond_min_env) if strong_precond_min_env else 800
        except ValueError:
            strong_precond_min = 800
        strong_precond_env = os.environ.get("SFINCS_JAX_RHSMODE1_STRONG_PRECOND", "").strip().lower()
        strong_precond_disabled = strong_precond_env in {"0", "false", "no", "off"}
        strong_precond_auto = strong_precond_env == "auto"
        if pas_precond_force_collision and strong_precond_env in {"", "auto"}:
            strong_precond_disabled = True
            strong_precond_auto = False
            if emit is not None:
                emit(1, "solve_v3_full_system_linear_gmres: PAS collision probe disabled strong preconditioner auto")
        if (
            strong_precond_env == ""
            and int(op.constraint_scheme) == 2
            and int(op.extra_size) > 0
            and (not use_pas_projection)
        ):
            strong_precond_auto = True
        if (
            strong_precond_env == ""
            and op.fblock.fp is not None
            and int(active_size) >= strong_precond_min
            and (int(op.n_theta) > 1 or int(op.n_zeta) > 1)
        ):
            strong_precond_auto = True
        if (
            strong_precond_env == ""
            and op.fblock.pas is not None
            and int(active_size) >= strong_precond_min
            and (int(op.n_theta) > 1 or int(op.n_zeta) > 1)
        ):
            strong_precond_auto = True
        strong_precond_kind: str | None = None
        if strong_precond_disabled:
            strong_precond_kind = None
        elif strong_precond_env in {"theta", "theta_line", "line_theta"}:
            strong_precond_kind = "theta_line"
        elif strong_precond_env in {"species", "species_block", "speciesblock"}:
            strong_precond_kind = "species_block"
        elif strong_precond_env in {"sxblock", "species_xblock", "species_x"}:
            strong_precond_kind = "sxblock"
        elif strong_precond_env in {"sxblock_tz", "sxblock_theta_zeta", "species_xblock_tz", "sx_tz"}:
            strong_precond_kind = "sxblock_tz"
        elif strong_precond_env in {"zeta", "zeta_line", "line_zeta"}:
            strong_precond_kind = "zeta_line"
        elif strong_precond_env in {"xblock_tz", "xblock", "x_tz", "xtz", "xblock_theta_zeta"}:
            strong_precond_kind = "xblock_tz"
        elif strong_precond_env in {"theta_zeta", "theta_zeta_line", "tz", "tz_line"}:
            strong_precond_kind = "theta_zeta"
        elif strong_precond_env in {"adi", "adi_line", "line_adi", "theta_zeta", "zeta_theta"}:
            strong_precond_kind = "adi"
        elif strong_precond_env in {"schur", "schur_complement", "constraint_schur"}:
            strong_precond_kind = "schur"
        elif strong_precond_env == "auto":
            strong_precond_kind = None
        else:
            strong_precond_kind = None

        if strong_precond_kind is None and (not strong_precond_disabled) and strong_precond_auto:
            if int(op.constraint_scheme) == 2 and int(op.extra_size) > 0:
                strong_precond_kind = "schur"
            elif (
                rhs1_precond_env == ""
                and int(op.rhs_mode) == 1
                and (not bool(op.include_phi1))
                and op.fblock.pas is not None
                and int(active_size) >= strong_precond_min
                and (int(op.n_theta) > 1 or int(op.n_zeta) > 1)
            ):
                tz_max_env = os.environ.get("SFINCS_JAX_RHSMODE1_TZ_PRECOND_MAX", "").strip()
                try:
                    tz_max = int(tz_max_env) if tz_max_env else 128
                except ValueError:
                    tz_max = 128
                xblock_tz_max_env = os.environ.get("SFINCS_JAX_RHSMODE1_XBLOCK_TZ_MAX", "").strip()
                try:
                    xblock_tz_max = int(xblock_tz_max_env) if xblock_tz_max_env else 1200
                except ValueError:
                    xblock_tz_max = 1200
                max_l = int(np.max(nxi_for_x)) if nxi_for_x.size else 0
                if (
                    int(op.n_theta) > 1
                    and int(op.n_zeta) > 1
                    and xblock_tz_max > 0
                    and int(max_l) * int(op.n_theta) * int(op.n_zeta) <= xblock_tz_max
                ):
                    strong_precond_kind = "xblock_tz"
                elif int(op.n_theta) > 1 and int(op.n_zeta) > 1 and int(op.n_theta) * int(op.n_zeta) <= tz_max:
                    strong_precond_kind = "theta_zeta"
                else:
                    strong_precond_kind = "theta_line" if int(op.n_theta) >= int(op.n_zeta) else "zeta_line"
            elif (
                rhs1_precond_env == ""
                and int(op.rhs_mode) == 1
                and (not bool(op.include_phi1))
                and op.fblock.fp is not None
                and int(active_size) >= strong_precond_min
                and (int(op.n_theta) > 1 or int(op.n_zeta) > 1)
            ):
                tz_max_env = os.environ.get("SFINCS_JAX_RHSMODE1_TZ_PRECOND_MAX", "").strip()
                try:
                    tz_max = int(tz_max_env) if tz_max_env else 128
                except ValueError:
                    tz_max = 128
                xblock_tz_max_env = os.environ.get("SFINCS_JAX_RHSMODE1_XBLOCK_TZ_MAX", "").strip()
                try:
                    xblock_tz_max = int(xblock_tz_max_env) if xblock_tz_max_env else 1200
                except ValueError:
                    xblock_tz_max = 1200
                max_l = int(np.max(nxi_for_x)) if nxi_for_x.size else 0
                if (
                    int(op.n_theta) > 1
                    and int(op.n_zeta) > 1
                    and xblock_tz_max > 0
                    and int(max_l) * int(op.n_theta) * int(op.n_zeta) <= xblock_tz_max
                ):
                    strong_precond_kind = "xblock_tz"
                elif int(op.n_theta) > 1 and int(op.n_zeta) > 1 and int(op.n_theta) * int(op.n_zeta) <= tz_max:
                    strong_precond_kind = "theta_zeta"
                else:
                    strong_precond_kind = "theta_line" if int(op.n_theta) >= int(op.n_zeta) else "zeta_line"

        if (
            strong_precond_kind is not None
            and float(res_reduced.residual_norm) > target_reduced
            and strong_precond_trigger
            and not early_dense_shortcut
        ):
            _mark("rhs1_strong_precond_build_start")
            if emit is not None:
                emit(
                    0,
                    "solve_v3_full_system_linear_gmres: strong preconditioner fallback "
                    f"kind={strong_precond_kind} (residual={float(res_reduced.residual_norm):.3e} > target={target_reduced:.3e})",
                )

            if strong_precond_kind == "theta_line":
                strong_preconditioner_reduced = _build_rhsmode1_theta_line_preconditioner(
                    op=op, reduce_full=reduce_full, expand_reduced=expand_reduced
                )
            elif strong_precond_kind == "species_block":
                strong_preconditioner_reduced = _build_rhsmode1_species_block_preconditioner(
                    op=op, reduce_full=reduce_full, expand_reduced=expand_reduced
                )
            elif strong_precond_kind == "sxblock":
                strong_preconditioner_reduced = _build_rhsmode1_species_xblock_preconditioner(
                    op=op, reduce_full=reduce_full, expand_reduced=expand_reduced
                )
            elif strong_precond_kind == "sxblock_tz":
                strong_preconditioner_reduced = _build_rhsmode1_sxblock_tz_preconditioner(
                    op=op, reduce_full=reduce_full, expand_reduced=expand_reduced
                )
            elif strong_precond_kind == "theta_zeta":
                strong_preconditioner_reduced = _build_rhsmode1_theta_zeta_preconditioner(
                    op=op, reduce_full=reduce_full, expand_reduced=expand_reduced
                )
            elif strong_precond_kind == "xblock_tz":
                strong_preconditioner_reduced = _build_rhsmode1_xblock_tz_preconditioner(
                    op=op, reduce_full=reduce_full, expand_reduced=expand_reduced
                )
            elif strong_precond_kind == "zeta_line":
                strong_preconditioner_reduced = _build_rhsmode1_zeta_line_preconditioner(
                    op=op, reduce_full=reduce_full, expand_reduced=expand_reduced
                )
            elif strong_precond_kind == "schur":
                strong_preconditioner_reduced = _build_rhsmode1_schur_preconditioner(
                    op=op, reduce_full=reduce_full, expand_reduced=expand_reduced
                )
            else:
                pre_theta = _build_rhsmode1_theta_line_preconditioner(
                    op=op, reduce_full=reduce_full, expand_reduced=expand_reduced
                )
                pre_zeta = _build_rhsmode1_zeta_line_preconditioner(
                    op=op, reduce_full=reduce_full, expand_reduced=expand_reduced
                )
                sweeps_env = os.environ.get("SFINCS_JAX_RHSMODE1_ADI_SWEEPS", "").strip()
                try:
                    sweeps = int(sweeps_env) if sweeps_env else 2
                except ValueError:
                    sweeps = 2
                sweeps = max(1, sweeps)

                def strong_preconditioner_reduced(v: jnp.ndarray) -> jnp.ndarray:
                    out = v
                    for _ in range(sweeps):
                        out = pre_zeta(pre_theta(out))
                    return out
            _mark("rhs1_strong_precond_build_done")
            if use_pas_projection:
                strong_preconditioner_reduced = _wrap_pas_precond(strong_preconditioner_reduced)

            strong_restart_env = os.environ.get("SFINCS_JAX_RHSMODE1_STRONG_PRECOND_RESTART", "").strip()
            strong_maxiter_env = os.environ.get("SFINCS_JAX_RHSMODE1_STRONG_PRECOND_MAXITER", "").strip()
            try:
                strong_restart = int(strong_restart_env) if strong_restart_env else max(120, int(restart))
            except ValueError:
                strong_restart = max(120, int(restart))
            try:
                strong_maxiter = int(strong_maxiter_env) if strong_maxiter_env else max(800, int(maxiter or 400) * 2)
            except ValueError:
                strong_maxiter = max(800, int(maxiter or 400) * 2)
            res_strong = _solve_linear(
                matvec_fn=mv_reduced,
                b_vec=rhs_reduced,
                precond_fn=strong_preconditioner_reduced,
                x0_vec=res_reduced.x,
                tol_val=tol,
                atol_val=atol,
                restart_val=strong_restart,
                maxiter_val=strong_maxiter,
                solve_method_val="incremental",
                precond_side=gmres_precond_side,
            )
            if float(res_strong.residual_norm) < float(res_reduced.residual_norm):
                res_reduced = res_strong
                ksp_matvec = mv_reduced
                ksp_b = rhs_reduced
                ksp_precond = strong_preconditioner_reduced
                ksp_x0 = res_reduced.x
                ksp_restart = strong_restart
                ksp_maxiter = strong_maxiter
                ksp_precond_side = gmres_precond_side
                ksp_solver_kind = _solver_kind("incremental")[0]

        dense_shortcut = early_dense_shortcut
        if not dense_shortcut and dense_shortcut_ratio > 0:
            quick_ratio = float(res_reduced.residual_norm) / max(float(target_reduced), 1e-300)
            if quick_ratio >= dense_shortcut_ratio:
                dense_fallback_max = _rhsmode1_dense_fallback_max(op)
                dense_fallback_max_huge = 0
                dense_fallback_ratio = 1.0e2
                if dense_fallback_max > 0:
                    dense_fallback_max_huge_env = os.environ.get("SFINCS_JAX_RHSMODE1_DENSE_FALLBACK_MAX_HUGE", "").strip()
                    try:
                        dense_fallback_max_huge = int(dense_fallback_max_huge_env) if dense_fallback_max_huge_env else dense_fallback_max
                    except ValueError:
                        dense_fallback_max_huge = dense_fallback_max
                    dense_fallback_ratio_env = os.environ.get("SFINCS_JAX_RHSMODE1_DENSE_FALLBACK_RATIO", "").strip()
                    try:
                        dense_fallback_ratio = float(dense_fallback_ratio_env) if dense_fallback_ratio_env else 1.0e2
                    except ValueError:
                        dense_fallback_ratio = 1.0e2
                r_vec = rhs_reduced - mv_reduced(res_reduced.x)
                residual_norm_true = float(jnp.linalg.norm(r_vec))
                if not np.isfinite(residual_norm_true):
                    residual_norm_true = float("inf")
                res_ratio = float(residual_norm_true) / max(float(target_reduced), 1e-300)
                dense_fallback_limit = dense_fallback_max_huge if res_ratio > dense_fallback_ratio else dense_fallback_max
                force_dense_cs0 = int(op.constraint_scheme) == 0
                if force_dense_cs0:
                    dense_fallback_limit = max(dense_fallback_limit, dense_fallback_max)
                dense_fallback_trigger = bool(res_ratio > dense_fallback_ratio) if dense_fallback_ratio > 0 else True
                if (
                    dense_fallback_limit > 0
                    and int(active_size) <= dense_fallback_limit
                    and dense_fallback_trigger
                    and (float(residual_norm_true) > target_reduced or force_dense_cs0)
                    and res_ratio >= dense_shortcut_ratio
                ):
                    dense_shortcut = True
                    if emit is not None:
                        emit(
                            0,
                            "solve_v3_full_system_linear_gmres: dense fallback shortcut "
                            f"(ratio={res_ratio:.3e} >= {dense_shortcut_ratio:.1e})",
                        )

        sparse_kind_use = sparse_precond_kind
        sparse_enabled = False
        if sparse_precond_mode == "on":
            sparse_enabled = True
        elif sparse_precond_mode == "auto":
            sparse_enabled = op.fblock.fp is not None
        if sparse_enabled:
            sparse_enabled = int(op.rhs_mode) == 1 and (not bool(op.include_phi1))
        if sparse_enabled:
            if dense_shortcut:
                sparse_enabled = False
            sparse_kind_use = sparse_precond_kind
            if sparse_kind_use == "auto":
                if use_implicit or (not sparse_allow_nondiff):
                    sparse_kind_use = "jax"
                else:
                    sparse_kind_use = "scipy"
            if sparse_kind_use == "scipy" and use_implicit and (not sparse_allow_nondiff):
                sparse_enabled = False
                if emit is not None:
                    emit(
                        1,
                        "sparse_ilu: disabled for implicit solves "
                        "(set SFINCS_JAX_RHSMODE1_SPARSE_PRECOND=jax to enable a differentiable sparse preconditioner)",
                    )
            elif int(active_size) > sparse_max_size:
                sparse_enabled = False
                if emit is not None:
                    emit(1, f"sparse_ilu: disabled (size={int(active_size)} > max={int(sparse_max_size)})")
            elif sparse_kind_use == "jax":
                precond_dtype = _precond_dtype(int(active_size))
                bytes_per = 4.0 if precond_dtype == jnp.float32 else 8.0
                est_mb = (int(active_size) ** 2) * bytes_per / 1.0e6
                if sparse_jax_max_mb > 0.0 and est_mb > sparse_jax_max_mb:
                    sparse_enabled = False
                    if emit is not None:
                        emit(
                            1,
                            "sparse_jax: disabled "
                            f"(est_mem={est_mb:.1f} MB > max_mb={sparse_jax_max_mb:.1f})",
                        )

        dense_matrix_cache: np.ndarray | None = None
        if sparse_enabled and float(res_reduced.residual_norm) > target_reduced:
            if sparse_kind_use == "jax":
                try:
                    _mark("rhs1_sparse_precond_build_start")
                    cache_key = _rhsmode1_sparse_cache_key(
                        op,
                        kind="sparse_jax",
                        active_size=int(active_size),
                        use_active_dof_mode=True,
                        use_pas_projection=use_pas_projection,
                        drop_tol=sparse_drop_tol,
                        drop_rel=sparse_drop_rel,
                        ilu_drop_tol=sparse_ilu_drop_tol,
                        fill_factor=sparse_ilu_fill,
                    )
                    precond_dtype = _precond_dtype(int(active_size))
                    precond_sparse = _build_sparse_jax_preconditioner_from_matvec(
                        matvec=mv_reduced,
                        n=int(active_size),
                        dtype=precond_dtype,
                        cache_key=cache_key,
                        drop_tol=sparse_drop_tol,
                        drop_rel=sparse_drop_rel,
                        reg=sparse_jax_reg,
                        omega=sparse_jax_omega,
                        sweeps=sparse_jax_sweeps,
                        emit=emit,
                    )
                    _mark("rhs1_sparse_precond_build_done")
                    if emit is not None:
                        emit(
                            0,
                            "solve_v3_full_system_linear_gmres: sparse JAX Jacobi fallback "
                            f"(sweeps={int(sparse_jax_sweeps)} omega={float(sparse_jax_omega):.2f})",
                        )
                    res_sparse = _solve_linear(
                        matvec_fn=mv_reduced,
                        b_vec=rhs_reduced,
                        precond_fn=precond_sparse,
                        x0_vec=res_reduced.x,
                        tol_val=tol,
                        atol_val=atol,
                        restart_val=restart,
                        maxiter_val=maxiter,
                        solve_method_val="incremental",
                        precond_side=gmres_precond_side,
                    )
                    if res_sparse is not None and float(res_sparse.residual_norm) < float(res_reduced.residual_norm):
                        res_reduced = res_sparse
                        ksp_matvec = mv_reduced
                        ksp_b = rhs_reduced
                        ksp_precond = precond_sparse
                        ksp_x0 = res_reduced.x
                        ksp_restart = restart
                        ksp_maxiter = maxiter
                        ksp_precond_side = gmres_precond_side
                        ksp_solver_kind = _solver_kind("incremental")[0]
                except Exception as exc:  # noqa: BLE001
                    if emit is not None:
                        emit(1, f"sparse_jax: failed ({type(exc).__name__}: {exc})")
            else:
                try:
                    _mark("rhs1_sparse_precond_build_start")
                    cache_key = _rhsmode1_sparse_cache_key(
                        op,
                        kind="sparse_ilu",
                        active_size=int(active_size),
                        use_active_dof_mode=True,
                        use_pas_projection=use_pas_projection,
                        drop_tol=sparse_drop_tol,
                        drop_rel=sparse_drop_rel,
                        ilu_drop_tol=sparse_ilu_drop_tol,
                        fill_factor=sparse_ilu_fill,
                    )
                    build_dense_factors = bool(use_implicit) and int(active_size) <= int(sparse_ilu_dense_max)
                    store_dense = int(active_size) <= int(sparse_dense_cache_max)
                    a_csr_full, _a_csr_drop, ilu, a_dense_cache, l_dense, u_dense, l_unit_diag = _build_sparse_ilu_from_matvec(
                        matvec=mv_reduced,
                        n=int(active_size),
                        dtype=rhs_reduced.dtype,
                        cache_key=cache_key,
                        drop_tol=sparse_drop_tol,
                        drop_rel=sparse_drop_rel,
                        ilu_drop_tol=sparse_ilu_drop_tol,
                        fill_factor=sparse_ilu_fill,
                        build_dense_factors=build_dense_factors,
                        build_ilu=True,
                        store_dense=store_dense,
                        emit=emit,
                    )
                    dense_matrix_cache = a_dense_cache
                    _mark("rhs1_sparse_precond_build_done")

                    if use_implicit:
                        if l_dense is None or u_dense is None:
                            if emit is not None:
                                emit(1, "sparse_ilu: dense factors unavailable for implicit solve; skipping")
                            res_sparse = None
                        else:
                            import jax.scipy.linalg as jla  # noqa: PLC0415

                            l_jnp = jnp.asarray(l_dense, dtype=jnp.float64)
                            u_jnp = jnp.asarray(u_dense, dtype=jnp.float64)

                            def _precond_sparse(v: jnp.ndarray) -> jnp.ndarray:
                                y = jla.solve_triangular(l_jnp, v, lower=True, unit_diagonal=l_unit_diag)
                                return jla.solve_triangular(u_jnp, y, lower=False)

                            if emit is not None:
                                emit(0, "solve_v3_full_system_linear_gmres: sparse ILU (JAX) fallback")
                            res_sparse = _solve_linear(
                                matvec_fn=mv_reduced,
                                b_vec=rhs_reduced,
                                precond_fn=_precond_sparse,
                                x0_vec=res_reduced.x,
                                tol_val=tol,
                                atol_val=atol,
                                restart_val=restart,
                                maxiter_val=maxiter,
                                solve_method_val="incremental",
                                precond_side=gmres_precond_side,
                            )
                    else:
                        if ilu is None:
                            raise RuntimeError("sparse_ilu: ILU factors unavailable")

                        def _precond_sparse(v: jnp.ndarray) -> jnp.ndarray:
                            x_np = np.asarray(v, dtype=np.float64).reshape((-1,))
                            y_np = ilu.solve(x_np)
                            return jnp.asarray(y_np, dtype=jnp.float64)

                        if sparse_use_matvec:
                            def _mv_sparse(v: jnp.ndarray) -> jnp.ndarray:
                                x_np = np.asarray(v, dtype=np.float64).reshape((-1,))
                                y_np = a_csr_full @ x_np
                                return jnp.asarray(y_np, dtype=jnp.float64)
                        else:
                            _mv_sparse = mv_reduced

                        if emit is not None:
                            emit(0, "solve_v3_full_system_linear_gmres: sparse ILU GMRES fallback")
                        x_np, rn_sparse, _history = gmres_solve_with_history_scipy(
                            matvec=_mv_sparse,
                            b=rhs_reduced,
                            preconditioner=_precond_sparse,
                            x0=res_reduced.x,
                            tol=tol,
                            atol=atol,
                            restart=restart,
                            maxiter=maxiter,
                            precondition_side=gmres_precond_side,
                        )
                        res_sparse = GMRESSolveResult(
                            x=jnp.asarray(x_np, dtype=jnp.float64),
                            residual_norm=jnp.asarray(rn_sparse, dtype=jnp.float64),
                        )
                    if res_sparse is not None and float(res_sparse.residual_norm) < float(res_reduced.residual_norm):
                        res_reduced = res_sparse
                        ksp_matvec = mv_reduced if use_implicit else _mv_sparse
                        ksp_b = rhs_reduced
                        ksp_precond = _precond_sparse
                        ksp_x0 = res_reduced.x
                        ksp_restart = restart
                        ksp_maxiter = maxiter
                        ksp_precond_side = gmres_precond_side
                        ksp_solver_kind = _solver_kind("incremental")[0]
                except Exception as exc:  # noqa: BLE001
                    if emit is not None:
                        emit(1, f"sparse_ilu: failed ({type(exc).__name__}: {exc})")
        residual_norm_check = float(res_reduced.residual_norm)
        residual_norm_true = residual_norm_check
        if ksp_precond is not None and ksp_precond_side == "left":
            try:
                r_vec = ksp_b - ksp_matvec(res_reduced.x)
                residual_norm_true = float(jnp.linalg.norm(r_vec))
                if not np.isfinite(residual_norm_true):
                    residual_norm_true = float("inf")
                r_pc = ksp_precond(r_vec)
                r_pc_norm = float(jnp.linalg.norm(r_pc))
                if np.isfinite(r_pc_norm):
                    residual_norm_check = r_pc_norm
            except Exception:
                pass
        dense_fallback_max = _rhsmode1_dense_fallback_max(op)
        dense_fallback_max_huge = 0
        dense_fallback_ratio = 1.0e2
        if dense_fallback_max > 0:
            dense_fallback_max_huge_env = os.environ.get("SFINCS_JAX_RHSMODE1_DENSE_FALLBACK_MAX_HUGE", "").strip()
            try:
                dense_fallback_max_huge = int(dense_fallback_max_huge_env) if dense_fallback_max_huge_env else dense_fallback_max
            except ValueError:
                dense_fallback_max_huge = dense_fallback_max
            dense_fallback_ratio_env = os.environ.get("SFINCS_JAX_RHSMODE1_DENSE_FALLBACK_RATIO", "").strip()
            try:
                dense_fallback_ratio = float(dense_fallback_ratio_env) if dense_fallback_ratio_env else 1.0e2
            except ValueError:
                dense_fallback_ratio = 1.0e2
        res_ratio = float(residual_norm_true) / max(float(target_reduced), 1e-300)
        dense_fallback_trigger = bool(res_ratio > dense_fallback_ratio) if dense_fallback_ratio > 0 else True
        dense_fallback_limit = dense_fallback_max_huge if res_ratio > dense_fallback_ratio else dense_fallback_max
        pas_force_dense = (
            op.fblock.fp is None
            and int(op.constraint_scheme) == 2
            and dense_fallback_limit > 0
            and int(active_size) <= dense_fallback_limit
            and float(res_reduced.residual_norm) > target_reduced
        )
        if pas_force_dense:
            dense_fallback_trigger = True
        fp_force_dense = (
            op.fblock.fp is not None
            and dense_fallback_max > 0
            and int(active_size) <= dense_fallback_max
            and float(residual_norm_true) > target_reduced
        )
        if fp_force_dense:
            dense_fallback_trigger = True
            dense_fallback_limit = max(dense_fallback_limit, dense_fallback_max)
        force_dense_cs0 = int(op.constraint_scheme) == 0
        if force_dense_cs0:
            # constraintScheme=0 systems are singular; keep the dense fallback
            # available even when the residual ratio is huge.
            dense_fallback_limit = max(dense_fallback_limit, dense_fallback_max)
            dense_fallback_trigger = True
        if (
            dense_fallback_limit > 0
            and int(op.rhs_mode) == 1
            and (not bool(op.include_phi1))
            and int(active_size) <= dense_fallback_limit
            and dense_fallback_trigger
            and (float(residual_norm_true) > target_reduced or force_dense_cs0)
        ):
            _mark("rhs1_dense_fallback_start")
            use_row_scaled = bool(
                int(op.constraint_scheme) == 0
                or (int(op.constraint_scheme) == 1 and op.fblock.fp is not None)
            )
            if emit is not None:
                emit(
                    0,
                    "solve_v3_full_system_linear_gmres: dense fallback "
                    f"(size={active_size} residual={float(res_reduced.residual_norm):.3e} > target={target_reduced:.3e})",
                )
            try:
                if dense_matrix_cache is not None:
                    a_dense_jnp = jnp.asarray(dense_matrix_cache, dtype=rhs_reduced.dtype)
                    if use_row_scaled:
                        x_dense, _rn = dense_solve_from_matrix_row_scaled(a=a_dense_jnp, b=rhs_reduced)
                    else:
                        x_dense, _rn = dense_solve_from_matrix(a=a_dense_jnp, b=rhs_reduced)
                    r_dense = rhs_reduced - mv_reduced(x_dense)
                    res_dense = GMRESSolveResult(x=x_dense, residual_norm=jnp.linalg.norm(r_dense))
                else:
                    dense_method = "dense"
                    if use_row_scaled:
                        dense_method = "dense_row_scaled"
                    res_dense = _solve_linear(
                        matvec_fn=mv_reduced,
                        b_vec=rhs_reduced,
                        precond_fn=None,
                        x0_vec=None,
                        tol_val=tol,
                        atol_val=atol,
                        restart_val=restart,
                        maxiter_val=maxiter,
                        solve_method_val=dense_method,
                        precond_side="none",
                    )
                if float(res_dense.residual_norm) < float(res_reduced.residual_norm):
                    res_reduced = res_dense
            except Exception as exc:  # noqa: BLE001
                if emit is not None:
                    emit(1, f"solve_v3_full_system_linear_gmres: dense fallback failed ({type(exc).__name__}: {exc})")
            _mark("rhs1_dense_fallback_done")
        if use_pas_projection:
            f_full = _expand_active_f(res_reduced.x)
            f_full = _project_pas_f(f_full)
            if int(op.extra_size) > 0:
                zeros_extra = jnp.zeros((int(op.extra_size),), dtype=jnp.float64)
                y_full = mv(jnp.concatenate([f_full, zeros_extra], axis=0))
                r_f = rhs[: op.f_size] - y_full[: op.f_size]
                extra = _constraint_scheme2_source_from_f(op, r_f.reshape(op.fblock.f_shape)) / fs_sum_safe
                if ix0 > 0:
                    extra = extra.at[:, :ix0].set(0.0)
                zero_env = os.environ.get("SFINCS_JAX_PAS_SOURCE_ZERO_TOL", "").strip()
                try:
                    zero_tol = float(zero_env) if zero_env else 2e-9
                except ValueError:
                    zero_tol = 2e-9
                if zero_tol > 0.0:
                    max_abs = jnp.max(jnp.abs(extra))
                    extra = jnp.where(max_abs <= zero_tol, jnp.zeros_like(extra), extra)
                x_full = jnp.concatenate([f_full, extra.reshape((-1,))], axis=0)
            else:
                x_full = f_full
        else:
            x_full = expand_reduced(res_reduced.x)
        # Residuals in active-DOF mode are computed on the reduced system to avoid an
        # extra full matvec; this matches the reduced KSP system used upstream.
        residual_norm_full = res_reduced.residual_norm
        result = GMRESSolveResult(x=x_full, residual_norm=residual_norm_full)
    else:
        if solve_method_kind == "dense_ksp":
            if int(op.phi1_size) != 0:
                raise NotImplementedError("dense_ksp is only supported for includePhi1=false RHSMode=1 solves.")
            if emit is not None:
                emit(1, "solve_v3_full_system_linear_gmres: assembling dense full matrix for dense_ksp")
            a_dense = assemble_dense_matrix_from_matvec(matvec=mv, n=int(op.total_size), dtype=rhs.dtype)

            if emit is not None:
                emit(1, "solve_v3_full_system_linear_gmres: building PETSc-like species-block preconditioner (dense_ksp)")

            import jax.scipy.linalg as jla  # noqa: PLC0415

            n_species = int(op.n_species)
            n_theta = int(op.n_theta)
            n_zeta = int(op.n_zeta)
            local_per_species = int(np.sum(nxi_for_x))
            dke_size = int(local_per_species * n_theta * n_zeta)
            extra_size = int(op.extra_size)
            extra_per_species = int(extra_size // max(1, n_species)) if extra_size else 0
            if extra_size and (extra_per_species * n_species != extra_size):
                extra_per_species = 0

            f_size = int(n_species * dke_size)
            expected_size = int(f_size + int(op.phi1_size) + extra_size)
            if int(op.total_size) != expected_size:
                raise RuntimeError(f"dense_ksp expects total_size={expected_size}, got {int(op.total_size)}")

            lu_factors: list[tuple[jnp.ndarray, jnp.ndarray]] = []
            idx_blocks: list[jnp.ndarray] = []
            for s in range(n_species):
                f_idx = np.arange(s * dke_size, (s + 1) * dke_size, dtype=np.int32)
                extra_idx = np.arange(
                    f_size + s * extra_per_species,
                    f_size + (s + 1) * extra_per_species,
                    dtype=np.int32,
                )
                block_idx_np = np.concatenate([f_idx, extra_idx], axis=0) if extra_per_species else f_idx
                block_idx = jnp.asarray(block_idx_np, dtype=jnp.int32)
                a_block = a_dense[jnp.ix_(block_idx, block_idx)]
                lu, piv = jla.lu_factor(a_block)
                lu_factors.append((lu, piv))
                idx_blocks.append(block_idx)

            def preconditioner_dense(v: jnp.ndarray) -> jnp.ndarray:
                out = jnp.zeros_like(v)
                for block_idx, (lu, piv) in zip(idx_blocks, lu_factors, strict=True):
                    rhs_block = v[block_idx]
                    sol_block = jla.lu_solve((lu, piv), rhs_block)
                    out = out.at[block_idx].set(sol_block, unique_indices=True)
                return out

            def mv_dense(x: jnp.ndarray) -> jnp.ndarray:
                return a_dense @ x

            rhs_pc = preconditioner_dense(rhs)

            def mv_pc(x: jnp.ndarray) -> jnp.ndarray:
                return preconditioner_dense(mv_dense(x))

            res_pc = _solve_linear(
                matvec_fn=mv_pc,
                b_vec=rhs_pc,
                precond_fn=None,
                x0_vec=x0,
                tol_val=tol,
                atol_val=atol,
                restart_val=restart,
                maxiter_val=maxiter,
                solve_method_val="incremental",
                precond_side="none",
            )
            ksp_matvec = mv_pc
            ksp_b = rhs_pc
            ksp_precond = None
            ksp_x0 = x0
            ksp_precond_side = "none"
            ksp_solver_kind = _solver_kind("incremental")[0]
            residual_norm_full = jnp.linalg.norm(mv(res_pc.x) - rhs)
            result = GMRESSolveResult(x=res_pc.x, residual_norm=residual_norm_full)
        else:
            preconditioner_full = None
            bicgstab_preconditioner_full = None

            if rhs1_bicgstab_kind is not None:
                if emit is not None:
                    emit(1, f"solve_v3_full_system_linear_gmres: RHSMode=1 BiCGStab preconditioner={rhs1_bicgstab_kind}")
                if rhs1_bicgstab_kind == "collision":
                    bicgstab_preconditioner_full = _build_rhsmode1_collision_preconditioner(op=op)

            def _build_rhs1_preconditioner_full():
                _mark("rhs1_precond_build_start")
                if emit is not None:
                    emit(1, f"solve_v3_full_system_linear_gmres: building RHSMode=1 preconditioner={rhs1_precond_kind}")
                if rhs1_precond_kind == "theta_line":
                    precond = _build_rhsmode1_theta_line_preconditioner(op=op)
                elif rhs1_precond_kind == "species_block":
                    precond = _build_rhsmode1_species_block_preconditioner(op=op)
                elif rhs1_precond_kind == "sxblock":
                    precond = _build_rhsmode1_species_xblock_preconditioner(op=op)
                elif rhs1_precond_kind == "sxblock_tz":
                    precond = _build_rhsmode1_sxblock_tz_preconditioner(op=op)
                elif rhs1_precond_kind == "xblock_tz":
                    precond = _build_rhsmode1_xblock_tz_preconditioner(op=op)
                elif rhs1_precond_kind == "theta_zeta":
                    precond = _build_rhsmode1_theta_zeta_preconditioner(op=op)
                elif rhs1_precond_kind == "zeta_line":
                    precond = _build_rhsmode1_zeta_line_preconditioner(op=op)
                elif rhs1_precond_kind == "schur":
                    precond = _build_rhsmode1_schur_preconditioner(op=op)
                elif rhs1_precond_kind == "collision":
                    precond = _build_rhsmode1_collision_preconditioner(op=op)
                elif rhs1_precond_kind == "adi":
                    pre_theta = _build_rhsmode1_theta_line_preconditioner(op=op)
                    pre_zeta = _build_rhsmode1_zeta_line_preconditioner(op=op)

                    sweeps_env = os.environ.get("SFINCS_JAX_RHSMODE1_ADI_SWEEPS", "").strip()
                    try:
                        sweeps = int(sweeps_env) if sweeps_env else 2
                    except ValueError:
                        sweeps = 2
                    sweeps = max(1, sweeps)

                    def preconditioner_full(v: jnp.ndarray) -> jnp.ndarray:
                        out = v
                        for _ in range(sweeps):
                            out = pre_zeta(pre_theta(out))
                        return out
                    precond = preconditioner_full
                else:
                    precond = _build_rhsmode1_block_preconditioner(
                        op=op,
                        preconditioner_species=preconditioner_species,
                        preconditioner_x=preconditioner_x,
                        preconditioner_xi=preconditioner_xi,
                    )
                _mark("rhs1_precond_build_done")
                return precond

            if rhs1_precond_enabled:
                solver_kind = _solver_kind(solve_method)[0]
                build_rhs1 = (
                    (solver_kind != "bicgstab" and solve_method_kind != "dense")
                    or (rhs1_bicgstab_kind == "rhs1" and solve_method_kind != "dense")
                )
                if build_rhs1:
                    preconditioner_full = _build_rhs1_preconditioner_full()
                    if rhs1_bicgstab_kind == "rhs1":
                        bicgstab_preconditioner_full = preconditioner_full
            if preconditioner_full is None and bicgstab_preconditioner_full is not None:
                preconditioner_full = bicgstab_preconditioner_full
            if recycle_basis_use:
                basis_full: list[jnp.ndarray] = []
                for vec in recycle_basis_use:
                    if vec.shape == (op.total_size,):
                        basis_full.append(vec)
                if basis_full:
                    basis_au = [mv(v) for v in basis_full]
                    x0_recycled = _recycled_initial_guess(rhs, basis_full, basis_au)
                    if x0_recycled is not None:
                        if x0 is None:
                            x0 = x0_recycled
                        else:
                            r0 = jnp.linalg.norm(mv(jnp.asarray(x0)) - rhs)
                            r1 = jnp.linalg.norm(mv(x0_recycled) - rhs)
                            if jnp.isfinite(r1) and (not jnp.isfinite(r0) or float(r1) < float(r0)):
                                x0 = x0_recycled
            result, residual_vec = _solve_linear_with_residual(
                matvec_fn=mv,
                b_vec=rhs,
                precond_fn=preconditioner_full,
                x0_vec=x0,
                tol_val=tol,
                atol_val=atol,
                restart_val=restart,
                maxiter_val=maxiter,
                solve_method_val=solve_method,
                precond_side=gmres_precond_side,
            )
            ksp_matvec = mv
            ksp_b = rhs
            ksp_precond = preconditioner_full
            ksp_x0 = x0
            ksp_precond_side = gmres_precond_side
            ksp_solver_kind = _solver_kind(solve_method)[0]
            if preconditioner_full is not None and (not _gmres_result_is_finite(result)):
                if emit is not None:
                    emit(0, "solve_v3_full_system_linear_gmres: preconditioned GMRES returned non-finite result; retrying without preconditioner")
                result, residual_vec = _solve_linear_with_residual(
                    matvec_fn=mv,
                    b_vec=rhs,
                    precond_fn=None,
                    x0_vec=x0,
                    tol_val=tol,
                    atol_val=atol,
                    restart_val=restart,
                    maxiter_val=maxiter,
                    solve_method_val=solve_method,
                    precond_side=gmres_precond_side,
                )
                ksp_matvec = mv
                ksp_b = rhs
                ksp_precond = None
                ksp_x0 = x0
                ksp_precond_side = gmres_precond_side
                ksp_solver_kind = _solver_kind(solve_method)[0]
            # If GMRES does not reach the requested tolerance (common without preconditioning),
            # retry with a larger iteration budget and the more robust incremental mode.
            target = max(float(atol), float(tol) * float(rhs_norm))
            res_ratio = float(result.residual_norm) / max(float(target), 1e-300)
            stage2_ratio_env = os.environ.get("SFINCS_JAX_LINEAR_STAGE2_RATIO", "").strip()
            try:
                stage2_ratio = float(stage2_ratio_env) if stage2_ratio_env else 1.0e2
            except ValueError:
                stage2_ratio = 1.0e2
            stage2_trigger = bool(res_ratio > stage2_ratio) if stage2_ratio > 0 else True
            solver_kind = _solver_kind(solve_method)[0]
            if solver_kind == "bicgstab" and (
                (not _gmres_result_is_finite(result))
                or (bicgstab_fallback_strict and float(result.residual_norm) > target)
            ):
                if emit is not None:
                    emit(
                        0,
                        "solve_v3_full_system_linear_gmres: BiCGStab fallback to GMRES "
                        f"(residual={float(result.residual_norm):.3e} > target={target:.3e})",
                    )
                if preconditioner_full is None and rhs1_precond_enabled:
                    preconditioner_full = _build_rhs1_preconditioner_full()
                result, residual_vec = _solve_linear_with_residual(
                    matvec_fn=mv,
                    b_vec=rhs,
                    precond_fn=preconditioner_full,
                    x0_vec=x0,
                    tol_val=tol,
                    atol_val=atol,
                    restart_val=restart,
                    maxiter_val=maxiter,
                    solve_method_val="incremental",
                    precond_side=gmres_precond_side,
                )
                ksp_matvec = mv
                ksp_b = rhs
                ksp_precond = preconditioner_full
                ksp_x0 = x0
                ksp_precond_side = gmres_precond_side
                ksp_solver_kind = "gmres"
        if (
            float(result.residual_norm) > target
            and stage2_enabled
            and stage2_trigger
            and t.elapsed_s() < stage2_time_cap_s
        ):
            if preconditioner_full is None and rhs1_precond_enabled:
                preconditioner_full = _build_rhs1_preconditioner_full()
            stage2_maxiter = int(os.environ.get("SFINCS_JAX_LINEAR_STAGE2_MAXITER", str(max(600, int(maxiter or 400) * 2))))
            stage2_restart = int(os.environ.get("SFINCS_JAX_LINEAR_STAGE2_RESTART", str(max(120, int(restart)))))
            stage2_method = os.environ.get("SFINCS_JAX_LINEAR_STAGE2_METHOD", "incremental").strip().lower()
            if stage2_method not in {"batched", "incremental", "dense"}:
                stage2_method = "incremental"
            if emit is not None:
                emit(
                    0,
                    "solve_v3_full_system_linear_gmres: stage2 GMRES "
                    f"(residual={float(result.residual_norm):.3e} > target={target:.3e}) "
                    f"restart={stage2_restart} maxiter={stage2_maxiter} method={stage2_method}",
                )
            res2, residual_vec2 = _solve_linear_with_residual(
                matvec_fn=mv,
                b_vec=rhs,
                precond_fn=preconditioner_full,
                x0_vec=result.x,
                tol_val=tol,
                atol_val=atol,
                restart_val=stage2_restart,
                maxiter_val=stage2_maxiter,
                solve_method_val=stage2_method,
                precond_side=gmres_precond_side,
            )
            if float(res2.residual_norm) < float(result.residual_norm):
                result = res2
                residual_vec = residual_vec2
                ksp_matvec = mv
                ksp_b = rhs
                ksp_precond = preconditioner_full
                ksp_x0 = result.x
                ksp_restart = stage2_restart
                ksp_maxiter = stage2_maxiter
                ksp_precond_side = gmres_precond_side
                ksp_solver_kind = _solver_kind(stage2_method)[0]
        res_ratio = float(result.residual_norm) / max(float(target), 1e-300)
        strong_ratio_env = os.environ.get("SFINCS_JAX_RHSMODE1_STRONG_PRECOND_RATIO", "").strip()
        try:
            strong_ratio = float(strong_ratio_env) if strong_ratio_env else 1.0
        except ValueError:
            strong_ratio = 1.0
        strong_precond_trigger = bool(res_ratio > strong_ratio) if strong_ratio > 0 else True
        if (
            float(result.residual_norm) > target
            and int(op.rhs_mode) == 1
            and (not bool(op.include_phi1))
            and rhs1_precond_kind == "point"
            and (op.fblock.fp is not None or op.fblock.pas is not None)
            and strong_precond_trigger
        ):
            if bicgstab_preconditioner_full is None:
                bicgstab_preconditioner_full = _build_rhsmode1_collision_preconditioner(op=op)
            if bicgstab_preconditioner_full is not None:
                if emit is not None:
                    emit(
                        0,
                        "solve_v3_full_system_linear_gmres: retry with collision preconditioner "
                        f"(residual={float(result.residual_norm):.3e} > target={target:.3e})",
                    )
                res_collision, residual_vec_collision = _solve_linear_with_residual(
                    matvec_fn=mv,
                    b_vec=rhs,
                    precond_fn=bicgstab_preconditioner_full,
                    x0_vec=result.x,
                    tol_val=tol,
                    atol_val=atol,
                    restart_val=restart,
                    maxiter_val=maxiter,
                    solve_method_val="incremental",
                    precond_side=gmres_precond_side,
                )
                if float(res_collision.residual_norm) < float(result.residual_norm):
                    result = res_collision
                    residual_vec = residual_vec_collision
                    ksp_matvec = mv
                    ksp_b = rhs
                    ksp_precond = bicgstab_preconditioner_full
                    ksp_x0 = result.x
                    ksp_restart = restart
                    ksp_maxiter = maxiter
                    ksp_precond_side = gmres_precond_side
                    ksp_solver_kind = _solver_kind("incremental")[0]
            strong_precond_min_env = os.environ.get("SFINCS_JAX_RHSMODE1_STRONG_PRECOND_MIN", "").strip()
            try:
                strong_precond_min = int(strong_precond_min_env) if strong_precond_min_env else 800
            except ValueError:
                strong_precond_min = 800
            strong_precond_env = os.environ.get("SFINCS_JAX_RHSMODE1_STRONG_PRECOND", "").strip().lower()
            strong_precond_disabled = strong_precond_env in {"0", "false", "no", "off"}
            strong_precond_auto = strong_precond_env == "auto"
            if (
                strong_precond_env == ""
                and int(op.constraint_scheme) == 2
                and int(op.extra_size) > 0
            ):
                strong_precond_auto = True
            if (
                strong_precond_env == ""
                and op.fblock.fp is not None
                and int(op.total_size) >= strong_precond_min
                and (int(op.n_theta) > 1 or int(op.n_zeta) > 1)
            ):
                strong_precond_auto = True
            strong_precond_kind: str | None = None
            if strong_precond_disabled:
                strong_precond_kind = None
            elif strong_precond_env in {"theta", "theta_line", "line_theta"}:
                strong_precond_kind = "theta_line"
            elif strong_precond_env in {"species", "species_block", "speciesblock"}:
                strong_precond_kind = "species_block"
            elif strong_precond_env in {"sxblock_tz", "sxblock_theta_zeta", "species_xblock_tz", "sx_tz"}:
                strong_precond_kind = "sxblock_tz"
            elif strong_precond_env in {"zeta", "zeta_line", "line_zeta"}:
                strong_precond_kind = "zeta_line"
            elif strong_precond_env in {"xblock_tz", "xblock", "x_tz", "xtz", "xblock_theta_zeta"}:
                strong_precond_kind = "xblock_tz"
            elif strong_precond_env in {"adi", "adi_line", "line_adi", "theta_zeta", "zeta_theta"}:
                strong_precond_kind = "adi"
            elif strong_precond_env in {"schur", "schur_complement", "constraint_schur"}:
                strong_precond_kind = "schur"
            elif strong_precond_env == "auto":
                strong_precond_kind = None
            else:
                strong_precond_kind = None

            if strong_precond_kind is None and (not strong_precond_disabled) and strong_precond_auto:
                if int(op.constraint_scheme) == 2 and int(op.extra_size) > 0:
                    strong_precond_kind = "schur"
                elif (
                    rhs1_precond_env == ""
                    and rhs1_precond_kind == "point"
                    and int(op.rhs_mode) == 1
                    and (not bool(op.include_phi1))
                    and op.fblock.pas is not None
                    and int(op.total_size) >= strong_precond_min
                    and (int(op.n_theta) > 1 or int(op.n_zeta) > 1)
                ):
                    xblock_tz_max_env = os.environ.get("SFINCS_JAX_RHSMODE1_XBLOCK_TZ_MAX", "").strip()
                    try:
                        xblock_tz_max = int(xblock_tz_max_env) if xblock_tz_max_env else 1200
                    except ValueError:
                        xblock_tz_max = 1200
                    max_l = int(np.max(nxi_for_x)) if nxi_for_x.size else 0
                    if (
                        int(op.n_theta) > 1
                        and int(op.n_zeta) > 1
                        and xblock_tz_max > 0
                        and int(max_l) * int(op.n_theta) * int(op.n_zeta) <= xblock_tz_max
                    ):
                        strong_precond_kind = "xblock_tz"
                    else:
                        strong_precond_kind = "theta_line" if int(op.n_theta) >= int(op.n_zeta) else "zeta_line"
                elif (
                    rhs1_precond_env == ""
                    and int(op.rhs_mode) == 1
                    and (not bool(op.include_phi1))
                    and op.fblock.fp is not None
                    and int(op.total_size) >= strong_precond_min
                    and (int(op.n_theta) > 1 or int(op.n_zeta) > 1)
                ):
                    tz_max_env = os.environ.get("SFINCS_JAX_RHSMODE1_TZ_PRECOND_MAX", "").strip()
                    try:
                        tz_max = int(tz_max_env) if tz_max_env else 128
                    except ValueError:
                        tz_max = 128
                    xblock_tz_max_env = os.environ.get("SFINCS_JAX_RHSMODE1_XBLOCK_TZ_MAX", "").strip()
                    try:
                        xblock_tz_max = int(xblock_tz_max_env) if xblock_tz_max_env else 1200
                    except ValueError:
                        xblock_tz_max = 1200
                    max_l = int(np.max(nxi_for_x)) if nxi_for_x.size else 0
                    if (
                        int(op.n_theta) > 1
                        and int(op.n_zeta) > 1
                        and xblock_tz_max > 0
                        and int(max_l) * int(op.n_theta) * int(op.n_zeta) <= xblock_tz_max
                    ):
                        strong_precond_kind = "xblock_tz"
                    elif int(op.n_theta) > 1 and int(op.n_zeta) > 1 and int(op.n_theta) * int(op.n_zeta) <= tz_max:
                        strong_precond_kind = "theta_zeta"
                    else:
                        strong_precond_kind = "theta_line" if int(op.n_theta) >= int(op.n_zeta) else "zeta_line"

            if strong_precond_kind is not None and float(result.residual_norm) > target:
                _mark("rhs1_strong_precond_build_start")
                if emit is not None:
                    emit(
                        0,
                        "solve_v3_full_system_linear_gmres: strong preconditioner fallback "
                        f"kind={strong_precond_kind} (residual={float(result.residual_norm):.3e} > target={target:.3e})",
                    )

                if strong_precond_kind == "theta_line":
                    strong_preconditioner_full = _build_rhsmode1_theta_line_preconditioner(op=op)
                elif strong_precond_kind == "species_block":
                    strong_preconditioner_full = _build_rhsmode1_species_block_preconditioner(op=op)
                elif strong_precond_kind == "sxblock":
                    strong_preconditioner_full = _build_rhsmode1_species_xblock_preconditioner(op=op)
                elif strong_precond_kind == "sxblock_tz":
                    strong_preconditioner_full = _build_rhsmode1_sxblock_tz_preconditioner(op=op)
                elif strong_precond_kind == "xblock_tz":
                    strong_preconditioner_full = _build_rhsmode1_xblock_tz_preconditioner(op=op)
                elif strong_precond_kind == "theta_zeta":
                    strong_preconditioner_full = _build_rhsmode1_theta_zeta_preconditioner(op=op)
                elif strong_precond_kind == "zeta_line":
                    strong_preconditioner_full = _build_rhsmode1_zeta_line_preconditioner(op=op)
                elif strong_precond_kind == "schur":
                    strong_preconditioner_full = _build_rhsmode1_schur_preconditioner(op=op)
                else:
                    pre_theta = _build_rhsmode1_theta_line_preconditioner(op=op)
                    pre_zeta = _build_rhsmode1_zeta_line_preconditioner(op=op)
                    sweeps_env = os.environ.get("SFINCS_JAX_RHSMODE1_ADI_SWEEPS", "").strip()
                    try:
                        sweeps = int(sweeps_env) if sweeps_env else 2
                    except ValueError:
                        sweeps = 2
                    sweeps = max(1, sweeps)

                    def strong_preconditioner_full(v: jnp.ndarray) -> jnp.ndarray:
                        out = v
                        for _ in range(sweeps):
                            out = pre_zeta(pre_theta(out))
                        return out
                _mark("rhs1_strong_precond_build_done")

                strong_restart_env = os.environ.get("SFINCS_JAX_RHSMODE1_STRONG_PRECOND_RESTART", "").strip()
                strong_maxiter_env = os.environ.get("SFINCS_JAX_RHSMODE1_STRONG_PRECOND_MAXITER", "").strip()
                try:
                    strong_restart = int(strong_restart_env) if strong_restart_env else max(120, int(restart))
                except ValueError:
                    strong_restart = max(120, int(restart))
                try:
                    strong_maxiter = int(strong_maxiter_env) if strong_maxiter_env else max(800, int(maxiter or 400) * 2)
                except ValueError:
                    strong_maxiter = max(800, int(maxiter or 400) * 2)
                res_strong, residual_vec_strong = _solve_linear_with_residual(
                    matvec_fn=mv,
                    b_vec=rhs,
                    precond_fn=strong_preconditioner_full,
                    x0_vec=result.x,
                    tol_val=tol,
                    atol_val=atol,
                    restart_val=strong_restart,
                    maxiter_val=strong_maxiter,
                    solve_method_val="incremental",
                    precond_side=gmres_precond_side,
                )
                if float(res_strong.residual_norm) < float(result.residual_norm):
                    result = res_strong
                    residual_vec = residual_vec_strong
                    ksp_matvec = mv
                    ksp_b = rhs
                    ksp_precond = strong_preconditioner_full
                    ksp_x0 = result.x
                    ksp_restart = strong_restart
                    ksp_maxiter = strong_maxiter
                    ksp_precond_side = gmres_precond_side
                    ksp_solver_kind = _solver_kind("incremental")[0]
        residual_norm_check = float(result.residual_norm)
        residual_norm_true = residual_norm_check
        if ksp_precond is not None and ksp_precond_side == "left":
            try:
                if residual_vec is None:
                    residual_vec = rhs - mv(result.x)
                residual_norm_true = float(jnp.linalg.norm(residual_vec))
                if not np.isfinite(residual_norm_true):
                    residual_norm_true = float("inf")
                r_pc = ksp_precond(residual_vec)
                r_pc_norm = float(jnp.linalg.norm(r_pc))
                if np.isfinite(r_pc_norm):
                    residual_norm_check = r_pc_norm
            except Exception:
                pass
        dense_fallback_max = _rhsmode1_dense_fallback_max(op)
        dense_fallback_ratio_env = os.environ.get("SFINCS_JAX_RHSMODE1_DENSE_FALLBACK_RATIO", "").strip()
        try:
            dense_fallback_ratio = float(dense_fallback_ratio_env) if dense_fallback_ratio_env else 1.0e2
        except ValueError:
            dense_fallback_ratio = 1.0e2
        res_ratio = float(residual_norm_true) / max(float(target), 1e-300)
        dense_fallback_trigger = bool(res_ratio > dense_fallback_ratio) if dense_fallback_ratio > 0 else True
        fp_force_dense = (
            op.fblock.fp is not None
            and dense_fallback_max > 0
            and int(active_size) <= dense_fallback_max
            and float(residual_norm_true) > target
        )
        if fp_force_dense:
            dense_fallback_trigger = True
        force_dense_cs0 = int(op.constraint_scheme) == 0
        if force_dense_cs0:
            dense_fallback_trigger = True
        if (
            dense_fallback_max > 0
            and int(op.rhs_mode) == 1
            and (not bool(op.include_phi1))
            and int(op.total_size) <= dense_fallback_max
            and dense_fallback_trigger
            and float(residual_norm_true) > target
        ):
            _mark("rhs1_dense_fallback_start")
            if emit is not None:
                emit(
                    0,
                    "solve_v3_full_system_linear_gmres: dense fallback "
                    f"(size={int(op.total_size)} residual={float(residual_norm_check):.3e} > target={target:.3e})",
                )
            try:
                dense_method = "dense"
                if int(op.constraint_scheme) == 0:
                    dense_method = "dense_row_scaled"
                res_dense, residual_vec_dense = _solve_linear_with_residual(
                    matvec_fn=mv,
                    b_vec=rhs,
                    precond_fn=None,
                    x0_vec=None,
                    tol_val=tol,
                    atol_val=atol,
                    restart_val=restart,
                    maxiter_val=maxiter,
                    solve_method_val=dense_method,
                    precond_side="none",
                )
                if float(res_dense.residual_norm) < float(result.residual_norm):
                    result = res_dense
                    residual_vec = residual_vec_dense
            except Exception as exc:  # noqa: BLE001
                if emit is not None:
                    emit(1, f"solve_v3_full_system_linear_gmres: dense fallback failed ({type(exc).__name__}: {exc})")
            _mark("rhs1_dense_fallback_done")
    if int(op.rhs_mode) == 1:
        project_env = os.environ.get("SFINCS_JAX_RHSMODE1_PROJECT_NULLSPACE", "").strip().lower()
        if project_env in {"0", "false", "no", "off"}:
            project_rhs1 = False
        elif project_env in {"1", "true", "yes", "on"}:
            project_rhs1 = True
        else:
            # Default parity-first behavior: enforce constraintScheme=1 nullspace projection
            # for linear RHSMode=1 solves without Phi1.
            project_rhs1 = bool(int(op.constraint_scheme) == 1 and (not bool(op.include_phi1)))
        if project_rhs1:
            x_projected, residual_projected = _project_constraint_scheme1_nullspace_solution_with_residual(
                op=op,
                x_vec=result.x,
                rhs_vec=rhs,
                matvec_op=op,
                enabled_env_var="SFINCS_JAX_RHSMODE1_PROJECT_NULLSPACE",
                residual_vec=residual_vec if residual_vec is not None and residual_vec.shape == rhs.shape else None,
            )
            if not bool(jnp.allclose(x_projected, result.x)):
                residual_norm_projected = jnp.linalg.norm(residual_projected)
                result = GMRESSolveResult(x=x_projected, residual_norm=residual_norm_projected)
        if int(op.constraint_scheme) == 2 and int(op.extra_size) > 0:
            zero_env = os.environ.get("SFINCS_JAX_PAS_SOURCE_ZERO_TOL", "").strip()
            try:
                zero_tol = float(zero_env) if zero_env else 2e-9
            except ValueError:
                zero_tol = 2e-9
            if zero_tol > 0.0:
                extra = result.x[-int(op.extra_size) :]
                max_abs = jnp.max(jnp.abs(extra))
                extra = jnp.where(max_abs <= zero_tol, jnp.zeros_like(extra), extra)
                x_new = jnp.concatenate([result.x[: -int(op.extra_size)], extra], axis=0)
                result = GMRESSolveResult(x=x_new, residual_norm=result.residual_norm)
    if ksp_matvec is not None and ksp_b is not None:
        ksp_history = _emit_ksp_history(
            matvec_fn=ksp_matvec,
            b_vec=ksp_b,
            precond_fn=ksp_precond,
            x0_vec=ksp_x0,
            tol_val=tol,
            atol_val=atol,
            restart_val=int(ksp_restart),
            maxiter_val=ksp_maxiter,
            precond_side=ksp_precond_side,
            solver_kind=ksp_solver_kind,
            solve_method_val=str(solve_method),
        )
        _emit_ksp_iter_stats(
            matvec_fn=ksp_matvec,
            b_vec=ksp_b,
            precond_fn=ksp_precond,
            x0_vec=ksp_x0,
            tol_val=float(tol),
            atol_val=float(atol),
            restart_val=int(ksp_restart),
            maxiter_val=ksp_maxiter,
            precond_side=ksp_precond_side,
            solver_kind=ksp_solver_kind,
            history=ksp_history,
            solve_method_val=str(solve_method),
        )
    if emit is not None:
        emit(0, f"solve_v3_full_system_linear_gmres: residual_norm={float(result.residual_norm):.6e}")
        emit(1, f"solve_v3_full_system_linear_gmres: elapsed_s={t.elapsed_s():.3f}")
    return V3LinearSolveResult(op=op, rhs=rhs, gmres=result)


solve_v3_full_system_linear_gmres_jit = jax.jit(
    solve_v3_full_system_linear_gmres,
    static_argnames=("tol", "atol", "restart", "maxiter", "solve_method", "identity_shift"),
)


@jtu.register_pytree_node_class
@dataclass(frozen=True)
class V3NewtonKrylovResult:
    """Result of a simple Newton–Krylov solve for `residual_v3_full_system` (experimental)."""

    op: V3FullSystemOperator
    x: jnp.ndarray
    residual_norm: jnp.ndarray
    n_newton: int
    last_linear_residual_norm: jnp.ndarray

    def tree_flatten(self):
        children = (self.op, self.x, self.residual_norm, self.last_linear_residual_norm)
        aux = int(self.n_newton)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        op, x, residual_norm, last_linear_residual_norm = children
        return cls(op=op, x=x, residual_norm=residual_norm, n_newton=int(aux), last_linear_residual_norm=last_linear_residual_norm)


def solve_v3_full_system_newton_krylov(
    *,
    nml: Namelist,
    x0: jnp.ndarray | None = None,
    tol: float = 1e-10,
    max_newton: int = 12,
    gmres_tol: float = 1e-10,
    gmres_restart: int = 80,
    gmres_maxiter: int | None = 400,
    solve_method: str = "batched",
    identity_shift: float = 0.0,
) -> V3NewtonKrylovResult:
    """Solve `residual_v3_full_system(op, x) = 0` using a basic Newton–Krylov iteration.

    This is intended for small parity fixtures and developer experimentation. It is **not**
    yet a stable API for production runs.
    """
    op = full_system_operator_from_namelist(nml=nml, identity_shift=identity_shift)
    _set_precond_size_hint(int(op.total_size))
    _set_precond_size_hint(int(op.total_size))
    if x0 is None:
        x = jnp.zeros((op.total_size,), dtype=jnp.float64)
    else:
        x = jnp.asarray(x0, dtype=jnp.float64)
        if x.shape != (op.total_size,):
            raise ValueError(f"x0 must have shape {(op.total_size,)}, got {x.shape}")

    last_linear_resid = jnp.asarray(jnp.inf, dtype=jnp.float64)

    for k in range(int(max_newton)):
        # Compute residual and a *single* linearization for this Newton step that can be reused
        # by GMRES. This avoids applying JAX's autodiff transform inside every matvec call,
        # which is a major performance bottleneck for includePhi1 solves.
        r, jvp = jax.linearize(lambda xx: residual_v3_full_system(op, xx), x)
        rnorm = jnp.linalg.norm(r)
        if float(rnorm) < float(tol):
            return V3NewtonKrylovResult(
                op=op,
                x=x,
                residual_norm=rnorm,
                n_newton=k,
                last_linear_residual_norm=last_linear_resid,
            )

        # Solve J s = -r
        lin = _gmres_solve_dispatch(
            matvec=jvp,
            b=-r,
            tol=float(gmres_tol),
            restart=int(gmres_restart),
            maxiter=gmres_maxiter,
            solve_method=str(solve_method),
        )
        s = lin.x
        last_linear_resid = lin.residual_norm

        # Backtracking line search on ||r|| (very simple Armijo-style criterion).
        step = 1.0
        step_scale_env = os.environ.get("SFINCS_JAX_PHI1_STEP_SCALE", "").strip()
        try:
            step_scale = float(step_scale_env) if step_scale_env else 1.0
        except ValueError:
            step_scale = 1.0
        rnorm0 = float(rnorm)
        for _ in range(12):
            x_try = x + (step * step_scale) * s
            r_try = residual_v3_full_system(op, x_try)
            rnorm_try = float(jnp.linalg.norm(r_try))
            if rnorm_try <= 0.9 * rnorm0:
                x = x_try
                break
            step *= 0.5
        else:
            # If we fail to reduce the residual, still take a small step to avoid stalling.
            x = x + (1.0 / 64.0) * s

    r = residual_v3_full_system(op, x)
    return V3NewtonKrylovResult(
        op=op,
        x=x,
        residual_norm=jnp.linalg.norm(r),
        n_newton=int(max_newton),
        last_linear_residual_norm=last_linear_resid,
    )


def solve_v3_full_system_newton_krylov_history(
    *,
    nml: Namelist,
    x0: jnp.ndarray | None = None,
    tol: float = 1e-10,
    max_newton: int = 12,
    gmres_tol: float = 1e-10,
    gmres_restart: int = 80,
    gmres_maxiter: int | None = 400,
    solve_method: str = "batched",
    identity_shift: float = 0.0,
    nonlinear_rtol: float = 0.0,
    use_frozen_linearization: bool = False,
    emit: Callable[[int, str], None] | None = None,
) -> tuple[V3NewtonKrylovResult, list[jnp.ndarray]]:
    """Newton–Krylov solve that also returns the per-iteration accepted states.

    The returned history matches v3's convention of saving diagnostics for iteration numbers
    starting at 1, i.e. it includes the sequence of *accepted* Newton iterates and excludes
    the initial guess `x0`.

    Optionally, this routine can use a v3-parity-oriented solve path with a frozen
    (`whichMatrix=1`-like) linearization and relative residual stopping
    (`||F|| <= nonlinear_rtol * ||F_0||`) in addition to absolute `tol`.
    """
    op = full_system_operator_from_namelist(nml=nml, identity_shift=identity_shift)
    if emit is not None:
        emit(1, f"solve_v3_full_system_newton_krylov_history: total_size={int(op.total_size)}")
    fortran_stdout_env = os.environ.get("SFINCS_JAX_FORTRAN_STDOUT", "").strip().lower()
    if fortran_stdout_env in {"0", "false", "no", "off"}:
        fortran_stdout = False
    elif fortran_stdout_env in {"1", "true", "yes", "on"}:
        fortran_stdout = True
    else:
        fortran_stdout = emit is not None
    env_gmres_tol = os.environ.get("SFINCS_JAX_PHI1_GMRES_TOL", "").strip()
    if env_gmres_tol:
        gmres_tol = float(env_gmres_tol)

    if x0 is None:
        x = jnp.zeros((op.total_size,), dtype=jnp.float64)
    else:
        x = jnp.asarray(x0, dtype=jnp.float64)
        if x.shape != (op.total_size,):
            raise ValueError(f"x0 must have shape {(op.total_size,)}, got {x.shape}")

    active_env = os.environ.get("SFINCS_JAX_PHI1_ACTIVE_DOF", "").strip().lower()
    nxi_for_x = np.asarray(op.fblock.collisionless.n_xi_for_x, dtype=np.int32)
    has_reduced_modes = bool(np.any(nxi_for_x < int(op.n_xi)))
    if active_env in {"1", "true", "yes", "on"}:
        use_active_dof_mode = True
    elif active_env in {"0", "false", "no", "off"}:
        use_active_dof_mode = False
    else:
        # Auto mode: for includePhi1 nonlinear RHSMode=1 solves with a truncated pitch grid,
        # solve only active DOFs to avoid singular inactive rows and reduce Krylov cost.
        use_active_dof_mode = bool(
            int(op.rhs_mode) == 1
            and bool(op.include_phi1)
            and has_reduced_modes
        )

    active_idx_jnp: jnp.ndarray | None = None
    full_to_active_jnp: jnp.ndarray | None = None
    active_size = int(op.total_size)
    if use_active_dof_mode:
        active_idx_np = _transport_active_dof_indices(op)
        active_idx_jnp = jnp.asarray(active_idx_np, dtype=jnp.int32)
        full_to_active_np = np.zeros((int(op.total_size),), dtype=np.int32)
        full_to_active_np[np.asarray(active_idx_np, dtype=np.int32)] = np.arange(
            1, int(active_idx_np.shape[0]) + 1, dtype=np.int32
        )
        full_to_active_jnp = jnp.asarray(full_to_active_np, dtype=jnp.int32)
        active_size = int(active_idx_np.shape[0])
        if emit is not None:
            emit(
                1,
                "solve_v3_full_system_newton_krylov_history: active-DOF mode enabled "
                f"(size={active_size}/{int(op.total_size)})",
            )
    gmres_restart_use = int(gmres_restart)
    if active_size <= 1000:
        gmres_restart_use = min(gmres_restart_use, 200)
    gmres_restart_use = max(1, gmres_restart_use)

    def _reduce_full(v_full: jnp.ndarray) -> jnp.ndarray:
        assert active_idx_jnp is not None
        return v_full[active_idx_jnp]

    def _expand_reduced(v_reduced: jnp.ndarray) -> jnp.ndarray:
        assert full_to_active_jnp is not None
        z0 = jnp.zeros((1,), dtype=v_reduced.dtype)
        padded = jnp.concatenate([z0, v_reduced], axis=0)
        return padded[full_to_active_jnp]

    preconditioner = None
    # Only enable block preconditioning in the PETSc-like parity mode that freezes the
    # Jacobian/linearization. For autodiff-linearized Newton steps, JAX's GMRES can
    # behave differently with an approximate preconditioner, which impacts iteration
    # histories and `sfincsOutput.h5` shape parity for linear Phi1 fixtures.
    pc_env = os.environ.get("SFINCS_JAX_PHI1_USE_PRECONDITIONER", "").strip().lower()
    use_preconditioner = pc_env not in {"0", "false", "no", "off"}
    dense_cutoff_env = os.environ.get("SFINCS_JAX_PHI1_NK_DENSE_CUTOFF", "").strip()
    try:
        dense_cutoff = int(dense_cutoff_env) if dense_cutoff_env else 5000
    except ValueError:
        dense_cutoff = 5000
    linear_size = active_size if use_active_dof_mode else int(op.total_size)
    solve_method_in = str(solve_method).strip().lower()
    use_dense_linear = solve_method_in in {"dense", "dense_row_scaled"} or (
        use_frozen_linearization and int(linear_size) <= int(dense_cutoff)
    )
    if use_dense_linear:
        use_preconditioner = False
    precond_opts = nml.group("preconditionerOptions")

    def _phi1_precond_opt_int(key: str, default: int) -> int:
        val = precond_opts.get(key, None)
        if val is None:
            return default
        try:
            return int(val)
        except (TypeError, ValueError):
            return default

    preconditioner_species = _phi1_precond_opt_int("PRECONDITIONER_SPECIES", 1)
    preconditioner_x = _phi1_precond_opt_int("PRECONDITIONER_X", 1)
    preconditioner_xi = _phi1_precond_opt_int("PRECONDITIONER_XI", 1)
    if use_preconditioner and use_frozen_linearization and int(op.rhs_mode) == 1:
        precond_kind_env = os.environ.get("SFINCS_JAX_PHI1_PRECOND_KIND", "").strip().lower()
        if not precond_kind_env:
            precond_kind = "collision" if bool(op.include_phi1) else "block"
        elif precond_kind_env in {"collision", "diag"}:
            precond_kind = "collision"
        elif precond_kind_env in {"block", "block_jacobi", "point"}:
            precond_kind = "block"
        else:
            precond_kind = "block"
        if emit is not None:
            emit(1, f"solve_v3_full_system_newton_krylov_history: preconditioner={precond_kind}")
        if precond_kind == "collision":
            if use_active_dof_mode:
                preconditioner = _build_rhsmode1_collision_preconditioner(
                    op=op, reduce_full=_reduce_full, expand_reduced=_expand_reduced
                )
            else:
                preconditioner = _build_rhsmode1_collision_preconditioner(op=op)
        else:
            if use_active_dof_mode:
                preconditioner = _build_rhsmode1_block_preconditioner(
                    op=op,
                    reduce_full=_reduce_full,
                    expand_reduced=_expand_reduced,
                    preconditioner_species=preconditioner_species,
                    preconditioner_x=preconditioner_x,
                    preconditioner_xi=preconditioner_xi,
                )
            else:
                preconditioner = _build_rhsmode1_block_preconditioner(
                    op=op,
                    preconditioner_species=preconditioner_species,
                    preconditioner_x=preconditioner_x,
                    preconditioner_xi=preconditioner_xi,
                )

    last_linear_resid = jnp.asarray(jnp.inf, dtype=jnp.float64)
    accepted: list[jnp.ndarray] = []
    rnorm_initial: float | None = None
    cached_jvp = None
    cached_jvp_iter = -1
    frozen_jac_cache_env = os.environ.get("SFINCS_JAX_PHI1_FROZEN_JAC_CACHE", "").strip().lower()
    if frozen_jac_cache_env in {"0", "false", "no", "off"}:
        use_frozen_jac_cache = False
    elif frozen_jac_cache_env in {"1", "true", "yes", "on"}:
        use_frozen_jac_cache = True
    else:
        use_frozen_jac_cache = True
    frozen_jac_every_env = os.environ.get("SFINCS_JAX_PHI1_FROZEN_JAC_CACHE_EVERY", "").strip()
    try:
        frozen_jac_every = int(frozen_jac_every_env) if frozen_jac_every_env else 1
    except ValueError:
        frozen_jac_every = 1
    frozen_jac_every = max(1, frozen_jac_every)
    ksp_history_max_env = os.environ.get("SFINCS_JAX_KSP_HISTORY_MAX_SIZE", "").strip().lower()
    if ksp_history_max_env in {"none", "inf", "infinite", "unlimited"}:
        ksp_history_max_size = None
    else:
        try:
            ksp_history_max_size = int(ksp_history_max_env) if ksp_history_max_env else 800
        except ValueError:
            ksp_history_max_size = 800
    ksp_history_max_iter_env = os.environ.get("SFINCS_JAX_KSP_HISTORY_MAX_ITER", "").strip()
    try:
        ksp_history_max_iter = int(ksp_history_max_iter_env) if ksp_history_max_iter_env else 2000
    except ValueError:
        ksp_history_max_iter = 2000

    def _emit_ksp_history_nk(
        *,
        matvec_fn,
        b_vec: jnp.ndarray,
        precond_fn,
        x0_vec: jnp.ndarray | None,
        tol_val: float,
        atol_val: float,
        restart_val: int,
        maxiter_val: int | None,
        precond_side: str,
    ) -> None:
        if emit is None or not fortran_stdout:
            return
        size = int(b_vec.size)
        if ksp_history_max_size is not None and size > int(ksp_history_max_size):
            emit(1, f"fortran-stdout: KSP history skipped (size={size} > max={int(ksp_history_max_size)})")
            return
        if maxiter_val is not None and ksp_history_max_iter is not None:
            est_iters = int(maxiter_val) * max(1, int(restart_val))
            if est_iters > int(ksp_history_max_iter):
                emit(
                    1,
                    "fortran-stdout: KSP history skipped "
                    f"(estimated_iters={est_iters} > max={int(ksp_history_max_iter)})",
                )
                return
        try:
            _x_hist, _rn, history = gmres_solve_with_history_scipy(
                matvec=matvec_fn,
                b=b_vec,
                preconditioner=precond_fn,
                x0=x0_vec,
                tol=tol_val,
                atol=atol_val,
                restart=restart_val,
                maxiter=maxiter_val,
                precondition_side=precond_side,
            )
        except Exception as exc:  # noqa: BLE001
            emit(1, f"fortran-stdout: KSP history unavailable ({type(exc).__name__}: {exc})")
            return
        for k_hist, rn in enumerate(history):
            emit(0, f"{k_hist:4d} KSP Residual norm {rn: .12e} ")
        if history:
            emit(0, " Linear iteration (KSP) converged.  KSPConvergedReason =            2")
            emit(0, "   KSP_CONVERGED_RTOL: Norm decreased by rtol.")

    for k in range(int(max_newton)):
        if emit is not None:
            emit(1, f"newton_iter={k}: evaluateResidual called")
        op_use = op
        if bool(op.include_phi1):
            phi1_flat = x[op.f_size : op.f_size + op.n_theta * op.n_zeta]
            phi1 = phi1_flat.reshape((op.n_theta, op.n_zeta))
            op_use = replace(op, phi1_hat_base=phi1)

        r = apply_v3_full_system_operator_cached(op_use, x, include_jacobian_terms=False) - rhs_v3_full_system_jit(op_use)
        rnorm = jnp.linalg.norm(r)
        rnorm_f = float(rnorm)
        if rnorm_initial is None:
            rnorm_initial = max(rnorm_f, 1e-300)
        if emit is not None:
            emit(0, f"newton_iter={k}: residual_norm={rnorm_f:.6e}")
        if emit is not None and fortran_stdout:
            emit(0, f"{k:4d} SNES Function norm {rnorm_f: .12e} ")
        if not np.isfinite(rnorm_f):
            # Keep the latest finite iterate. This mirrors PETSc's behavior of
            # stopping when the nonlinear residual becomes invalid instead of
            # continuing with NaN/Inf states.
            x_return = accepted[-1] if accepted else x
            r_return = residual_v3_full_system(op, x_return)
            return (
                V3NewtonKrylovResult(
                    op=op,
                    x=x_return,
                    residual_norm=jnp.linalg.norm(r_return),
                    n_newton=k,
                    last_linear_residual_norm=last_linear_resid,
                ),
                accepted,
            )

        converged_abs = rnorm_f < float(tol)
        converged_rel = rnorm_f <= float(nonlinear_rtol) * float(rnorm_initial)
        if converged_abs or converged_rel:
            if not accepted:
                accepted.append(x)
            return (
                V3NewtonKrylovResult(
                    op=op,
                    x=x,
                    residual_norm=rnorm,
                    n_newton=k,
                    last_linear_residual_norm=last_linear_resid,
                ),
                accepted,
            )

        frozen_jac_mode = None
        if use_frozen_linearization:
            jac_mode = os.environ.get("SFINCS_JAX_PHI1_FROZEN_JAC_MODE", "").strip().lower()
            if jac_mode not in {"frozen", "frozen_rhs", "frozen_op"}:
                jac_mode = "frozen" if bool(op.include_phi1) else "frozen_rhs"
            frozen_jac_mode = jac_mode

            if jac_mode == "frozen_rhs":
                # Keep the kinetic/collision operator frozen at the current iterate (op_use),
                # but let the RHS keep its explicit Phi1 dependence. This is closer to v3's
                # nonlinear Jacobian path for includePhi1 while retaining robust parity behavior.
                def residual_for_jac(xx: jnp.ndarray) -> jnp.ndarray:
                    if bool(op.include_phi1):
                        phi1_flat_x = xx[op.f_size : op.f_size + op.n_theta * op.n_zeta]
                        phi1_x = phi1_flat_x.reshape((op.n_theta, op.n_zeta))
                        op_rhs_x = replace(op, phi1_hat_base=phi1_x)
                    else:
                        op_rhs_x = op
                    return (
                        apply_v3_full_system_operator_cached(op_use, xx, include_jacobian_terms=True)
                        - rhs_v3_full_system_jit(op_rhs_x)
                    )

                reuse_cached = (
                    use_frozen_jac_cache
                    and cached_jvp is not None
                    and (k - cached_jvp_iter) < frozen_jac_every
                )
                if reuse_cached:
                    matvec = cached_jvp
                    if emit is not None:
                        emit(1, f"newton_iter={k}: evaluateJacobian reused (frozen_rhs cache)")
                else:
                    _r_lin, jvp = jax.linearize(residual_for_jac, x)
                    matvec = jvp
                    if use_frozen_jac_cache:
                        cached_jvp = jvp
                        cached_jvp_iter = k
                    if emit is not None:
                        emit(1, f"newton_iter={k}: evaluateJacobian called (frozen operator + dynamic RHS)")
            elif jac_mode == "frozen_op":
                # Keep RHS frozen at the current iterate, but let the operator
                # carry the Phi1 dependence. This emulates partial Jacobian updates
                # in upstream SNES paths.
                def residual_for_jac(xx: jnp.ndarray) -> jnp.ndarray:
                    if bool(op.include_phi1):
                        phi1_flat_x = xx[op.f_size : op.f_size + op.n_theta * op.n_zeta]
                        phi1_x = phi1_flat_x.reshape((op.n_theta, op.n_zeta))
                        op_mat_x = replace(op, phi1_hat_base=phi1_x)
                    else:
                        op_mat_x = op
                    return (
                        apply_v3_full_system_operator_cached(op_mat_x, xx, include_jacobian_terms=True)
                        - rhs_v3_full_system_jit(op_use)
                    )

                _r_lin, jvp = jax.linearize(residual_for_jac, x)
                matvec = jvp
                if emit is not None:
                    emit(1, f"newton_iter={k}: evaluateJacobian called (dynamic operator + frozen RHS)")
            else:
                matvec = lambda dx: apply_v3_full_system_jacobian_jit(op_use, x, dx)
                if emit is not None:
                    emit(1, f"newton_iter={k}: evaluateJacobian called (fully frozen linearization)")
        else:
            # Optional exact mode for debugging/experimentation.
            _r_lin, jvp = jax.linearize(lambda xx: residual_v3_full_system(op, xx), x)
            matvec = jvp
            if emit is not None:
                emit(1, f"newton_iter={k}: evaluateJacobian called (autodiff linearization)")

        solve_method_linear = str(solve_method)
        if use_frozen_linearization:
            if int(linear_size) <= int(dense_cutoff):
                solve_method_linear = "dense"

        if use_active_dof_mode:
            rhs_reduced = _reduce_full(-r)

            def matvec_reduced(dx_reduced: jnp.ndarray) -> jnp.ndarray:
                return _reduce_full(matvec(_expand_reduced(dx_reduced)))

            lin = _gmres_solve_dispatch(
                matvec=matvec_reduced,
                b=rhs_reduced,
                preconditioner=preconditioner,
                tol=float(gmres_tol),
                restart=int(gmres_restart_use),
                maxiter=gmres_maxiter,
                solve_method=solve_method_linear,
            )
            _emit_ksp_history_nk(
                matvec_fn=matvec_reduced,
                b_vec=rhs_reduced,
                precond_fn=preconditioner,
                x0_vec=None,
                tol_val=float(gmres_tol),
                atol_val=0.0,
                restart_val=int(gmres_restart_use),
                maxiter_val=gmres_maxiter,
                precond_side="left",
            )
            if preconditioner is not None and (not _gmres_result_is_finite(lin)):
                if emit is not None:
                    emit(
                        0,
                        "newton_iter="
                        f"{k}: preconditioned GMRES returned non-finite result; retrying without preconditioner",
                    )
                lin = _gmres_solve_dispatch(
                    matvec=matvec_reduced,
                    b=rhs_reduced,
                    preconditioner=None,
                    tol=float(gmres_tol),
                    restart=int(gmres_restart_use),
                    maxiter=gmres_maxiter,
                    solve_method=solve_method_linear,
                )
                _emit_ksp_history_nk(
                    matvec_fn=matvec_reduced,
                    b_vec=rhs_reduced,
                    precond_fn=None,
                    x0_vec=None,
                    tol_val=float(gmres_tol),
                    atol_val=0.0,
                    restart_val=int(gmres_restart_use),
                    maxiter_val=gmres_maxiter,
                    precond_side="left",
                )
            s = _expand_reduced(lin.x)
            linear_resid_norm = jnp.linalg.norm(matvec(s) + r)
        else:
            lin = _gmres_solve_dispatch(
                matvec=matvec,
                b=-r,
                preconditioner=preconditioner,
                tol=float(gmres_tol),
                restart=int(gmres_restart_use),
                maxiter=gmres_maxiter,
                solve_method=solve_method_linear,
            )
            _emit_ksp_history_nk(
                matvec_fn=matvec,
                b_vec=-r,
                precond_fn=preconditioner,
                x0_vec=None,
                tol_val=float(gmres_tol),
                atol_val=0.0,
                restart_val=int(gmres_restart_use),
                maxiter_val=gmres_maxiter,
                precond_side="left",
            )
            if preconditioner is not None and (not _gmres_result_is_finite(lin)):
                if emit is not None:
                    emit(
                        0,
                        "newton_iter="
                        f"{k}: preconditioned GMRES returned non-finite result; retrying without preconditioner",
                    )
                lin = _gmres_solve_dispatch(
                    matvec=matvec,
                    b=-r,
                    preconditioner=None,
                    tol=float(gmres_tol),
                    restart=int(gmres_restart_use),
                    maxiter=gmres_maxiter,
                    solve_method=solve_method_linear,
                )
                _emit_ksp_history_nk(
                    matvec_fn=matvec,
                    b_vec=-r,
                    precond_fn=None,
                    x0_vec=None,
                    tol_val=float(gmres_tol),
                    atol_val=0.0,
                    restart_val=int(gmres_restart_use),
                    maxiter_val=gmres_maxiter,
                    precond_side="left",
                )
            s = lin.x
            linear_resid_norm = lin.residual_norm

        if emit is not None:
            emit(1, f"newton_iter={k}: gmres_residual={float(linear_resid_norm):.6e}")
        if not _gmres_result_is_finite(lin):
            x_return = accepted[-1] if accepted else x
            r_return = residual_v3_full_system(op, x_return)
            return (
                V3NewtonKrylovResult(
                    op=op,
                    x=x_return,
                    residual_norm=jnp.linalg.norm(r_return),
                    n_newton=k,
                    last_linear_residual_norm=last_linear_resid,
                ),
                accepted,
            )
        last_linear_resid = linear_resid_norm

        step = 1.0
        step_scale_env = os.environ.get("SFINCS_JAX_PHI1_STEP_SCALE", "").strip()
        try:
            step_scale = float(step_scale_env) if step_scale_env else 1.0
        except ValueError:
            step_scale = 1.0
        rnorm0 = float(rnorm)
        ls_factor_env = os.environ.get("SFINCS_JAX_PHI1_LINESEARCH_FACTOR", "").strip()
        ls_c1_env = os.environ.get("SFINCS_JAX_PHI1_LINESEARCH_C1", "").strip()
        try:
            ls_factor = float(ls_factor_env) if ls_factor_env else None
        except ValueError:
            ls_factor = None
        try:
            ls_c1 = float(ls_c1_env) if ls_c1_env else 1.0e-4
        except ValueError:
            ls_c1 = 1.0e-4
        ls_mode_env = os.environ.get("SFINCS_JAX_PHI1_LINESEARCH_MODE", "").strip().lower()
        if ls_mode_env:
            ls_mode = ls_mode_env
        else:
            ls_mode = "petsc" if (use_frozen_linearization and bool(op.include_phi1)) else "best"
        best_x = None
        best_rnorm = float("inf")
        if ls_mode == "best":
            step_candidates = [1.0, 1.5, 2.0, 0.5, 0.25, 0.125, 0.0625, 0.03125]
        max_ls_env = os.environ.get("SFINCS_JAX_PHI1_LINESEARCH_MAXITER", "").strip()
        try:
            max_ls = int(max_ls_env) if max_ls_env else (40 if ls_mode == "petsc" else 12)
        except ValueError:
            max_ls = 40 if ls_mode == "petsc" else 12
        if ls_mode in {"basic", "full"}:
            x = x + (step * step_scale) * s
            accepted.append(x)
            continue
        for _ in range(max_ls):
            if ls_mode == "best":
                try_step = step_candidates.pop(0) if step_candidates else step
            else:
                try_step = step
            x_try = x + (try_step * step_scale) * s
            # Always evaluate the true nonlinear residual for line-search acceptance,
            # even when using a frozen linearization for the Jacobian. This matches PETSc's
            # SNES line-search behavior and improves includePhi1 iteration parity.
            r_try = residual_v3_full_system(op, x_try)
            rnorm_try = float(jnp.linalg.norm(r_try))
            if not np.isfinite(rnorm_try):
                if ls_mode != "best":
                    step *= 0.5
                continue
            if rnorm_try < best_rnorm:
                best_rnorm = rnorm_try
                best_x = x_try
            if ls_mode != "best":
                if ls_factor is not None:
                    accept = rnorm_try <= ls_factor * rnorm0
                else:
                    accept = rnorm_try <= (1.0 - ls_c1 * step) * rnorm0
                if accept:
                    x = x_try
                    accepted.append(x)
                    break
            if ls_mode != "best":
                step *= 0.5
        else:
            if ls_mode == "best" and best_x is not None and best_rnorm < rnorm0:
                x = best_x
            elif best_x is not None and np.isfinite(best_rnorm):
                # Accept the best finite trial even if not strictly improving.
                x = best_x
            elif accepted:
                # Reuse the last accepted finite state to avoid propagating non-finite updates.
                x = accepted[-1]
            else:
                x = x + (1.0 / 64.0) * s
            accepted.append(x)

    r = residual_v3_full_system(op, x)
    return (
        V3NewtonKrylovResult(
            op=op,
            x=x,
            residual_norm=jnp.linalg.norm(r),
            n_newton=int(max_newton),
            last_linear_residual_norm=last_linear_resid,
        ),
        accepted,
    )


@dataclass(frozen=True)
class V3TransportMatrixSolveResult:
    """Result of assembling a RHSMode=2/3 transport matrix by looping `whichRHS` solves."""

    op0: V3FullSystemOperator
    transport_matrix: jnp.ndarray  # (N,N) in mathematical (row, col) order
    state_vectors_by_rhs: dict[int, jnp.ndarray]
    residual_norms_by_rhs: dict[int, jnp.ndarray]
    # Diagnostics in the same normalized units as v3 `diagnostics.F90`.
    # Arrays are returned in mathematical axis order: (species, whichRHS).
    fsab_flow: jnp.ndarray  # (S, N)
    particle_flux_vm_psi_hat: jnp.ndarray  # (S, N)
    heat_flux_vm_psi_hat: jnp.ndarray  # (S, N)
    elapsed_time_s: jnp.ndarray  # (N,)
    transport_output_fields: dict[str, np.ndarray] | None = None


def _transport_parallel_worker(payload: dict[str, object]) -> dict[str, object]:
    """Worker entry point for parallel whichRHS transport solves."""
    input_path = Path(str(payload["input_path"]))
    which_rhs_values = [int(v) for v in payload["which_rhs_values"]]  # type: ignore[assignment]
    tol = float(payload.get("tol", 1e-10))
    atol = float(payload.get("atol", 0.0))
    restart = int(payload.get("restart", 80))
    maxiter = payload.get("maxiter")
    solve_method = str(payload.get("solve_method", "auto"))
    identity_shift = float(payload.get("identity_shift", 0.0))
    phi1_hat_base = payload.get("phi1_hat_base")
    if phi1_hat_base is not None:
        phi1_hat_base = jnp.asarray(phi1_hat_base, dtype=jnp.float64)

    # Prevent recursive parallelism inside workers.
    os.environ["SFINCS_JAX_TRANSPORT_PARALLEL"] = "off"
    os.environ["SFINCS_JAX_TRANSPORT_PARALLEL_CHILD"] = "1"

    nml = read_sfincs_input(input_path)
    result = solve_v3_transport_matrix_linear_gmres(
        nml=nml,
        tol=tol,
        atol=atol,
        restart=restart,
        maxiter=maxiter,
        solve_method=solve_method,
        identity_shift=identity_shift,
        phi1_hat_base=phi1_hat_base,
        input_namelist=input_path,
        which_rhs_values=which_rhs_values,
        force_stream_diagnostics=True,
        force_store_state=False,
        parallel_workers=1,
    )
    transport_fields = result.transport_output_fields or {}
    return {
        "which_rhs_values": which_rhs_values,
        "particle_flux_vm_psi_hat": np.asarray(result.particle_flux_vm_psi_hat),
        "heat_flux_vm_psi_hat": np.asarray(result.heat_flux_vm_psi_hat),
        "fsab_flow": np.asarray(result.fsab_flow),
        "transport_output_fields": {k: np.asarray(v) for k, v in transport_fields.items()},
        "residual_norms_by_rhs": {int(k): float(np.asarray(v)) for k, v in result.residual_norms_by_rhs.items()},
        "elapsed_time_s": np.asarray(result.elapsed_time_s, dtype=np.float64),
    }


def _transport_active_dof_indices(op: V3FullSystemOperator) -> np.ndarray:
    """Return full-vector indices for active transport solve DOFs.

    For v3 RHSMode=2/3 transport solves, Fortran only includes active Legendre
    modes for each x (as set by `Nxi_for_x`) in the linear system unknown vector.
    This helper builds that reduced active set so matrix-free solves can mirror
    Fortran's non-singular system size.
    """
    s = int(op.n_species)
    x = int(op.n_x)
    l = int(op.n_xi)
    t = int(op.n_theta)
    z = int(op.n_zeta)

    nxi_for_x = np.asarray(op.fblock.collisionless.n_xi_for_x, dtype=np.int32)  # (X,)
    l_idx = np.arange(l, dtype=np.int32)[None, :]  # (1,L)
    xl_mask = l_idx < nxi_for_x[:, None]  # (X,L)
    f_mask = np.broadcast_to(xl_mask[None, :, :, None, None], (s, x, l, t, z))
    f_active = np.flatnonzero(f_mask.reshape((-1,)))  # within f block

    tail_active = np.arange(int(op.f_size), int(op.total_size), dtype=np.int32)
    return np.concatenate([f_active.astype(np.int32), tail_active], axis=0)


def _constraint_scheme2_source_from_f(op: V3FullSystemOperator, f: jnp.ndarray) -> jnp.ndarray:
    """Return constraintScheme=2 source terms from L=0 flux-surface averages."""
    factor = _fs_average_factor(op.theta_weights, op.zeta_weights, op.d_hat)  # (T,Z)
    y_avg = jnp.einsum("tz,sxtz->sx", factor, f[:, :, 0, :, :])  # (S,X)
    return y_avg


def _constraint_scheme2_inject_source(op: V3FullSystemOperator, src: jnp.ndarray) -> jnp.ndarray:
    """Inject constraintScheme=2 source terms into the L=0 rows of the f block."""
    f = jnp.zeros(op.fblock.f_shape, dtype=jnp.float64)
    ix0 = _ix_min(bool(op.point_at_x0))
    f = f.at[:, ix0:, 0, :, :].set(src[:, ix0:, None, None])
    return f.reshape((-1,))


def _project_constraint_scheme1_nullspace_solution_with_residual(
    *,
    op: V3FullSystemOperator,
    x_vec: jnp.ndarray,
    rhs_vec: jnp.ndarray,
    matvec_op: V3FullSystemOperator,
    enabled_env_var: str,
    residual_vec: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Project solution to constraintScheme=1 nullspace complement and return residual."""
    if int(op.constraint_scheme) != 1:
        if residual_vec is not None and residual_vec.shape == rhs_vec.shape:
            return x_vec, residual_vec
        return x_vec, apply_v3_full_system_operator_cached(matvec_op, x_vec) - rhs_vec
    if int(op.phi1_size) != 0:
        if residual_vec is not None and residual_vec.shape == rhs_vec.shape:
            return x_vec, residual_vec
        return x_vec, apply_v3_full_system_operator_cached(matvec_op, x_vec) - rhs_vec
    if int(op.extra_size) == 0:
        if residual_vec is not None and residual_vec.shape == rhs_vec.shape:
            return x_vec, residual_vec
        return x_vec, apply_v3_full_system_operator_cached(matvec_op, x_vec) - rhs_vec

    project_env = os.environ.get(enabled_env_var, "").strip().lower()
    if project_env in {"0", "false", "no", "off"}:
        if residual_vec is not None and residual_vec.shape == rhs_vec.shape:
            return x_vec, residual_vec
        return x_vec, apply_v3_full_system_operator_cached(matvec_op, x_vec) - rhs_vec

    xpart1, xpart2 = _source_basis_constraint_scheme_1(op.x)
    ix0 = 1 if bool(op.point_at_x0) else 0
    f_shape = op.fblock.f_shape
    n_s, _, _, _, _ = f_shape

    def _basis(species_index: int, src_index: int, xpart: jnp.ndarray) -> jnp.ndarray:
        f = jnp.zeros(f_shape, dtype=jnp.float64)
        f = f.at[species_index, ix0:, 0, :, :].set(xpart[ix0:][:, None, None])
        extra = jnp.zeros((n_s, 2), dtype=jnp.float64)
        extra = extra.at[species_index, src_index].set(-1.0)
        return jnp.concatenate([f.reshape((-1,)), extra.reshape((-1,))], axis=0)

    basis: list[jnp.ndarray] = []
    for s in range(n_s):
        basis.append(_basis(s, 0, xpart1))
        basis.append(_basis(s, 1, xpart2))

    if residual_vec is not None and residual_vec.shape == rhs_vec.shape:
        r = residual_vec
    else:
        r = apply_v3_full_system_operator_cached(matvec_op, x_vec) - rhs_vec
    r_extra = r[-op.extra_size :]
    proj_atol_env = os.environ.get(f"{enabled_env_var}_ATOL", "").strip()
    if proj_atol_env:
        try:
            proj_atol = float(proj_atol_env)
        except ValueError:
            proj_atol = 0.0
    else:
        # For transport-matrix solves, skip projection when the constraint residuals are
        # already at roundoff. Keep RHSMode=1 default behavior unchanged.
        proj_atol = 0.0 if "RHSMODE1" in enabled_env_var else 1e-9
    if proj_atol > 0.0:
        max_res = float(np.max(np.abs(np.asarray(r_extra, dtype=np.float64))))
        if max_res <= proj_atol:
            return x_vec, r
    cols_full = [apply_v3_full_system_operator_cached(matvec_op, v) for v in basis]
    cols_extra = [col[-op.extra_size :] for col in cols_full]
    m = jnp.stack(cols_extra, axis=1)
    c_res, *_ = jnp.linalg.lstsq(m, -r_extra, rcond=None)
    x_corr = sum(v * c_res[i] for i, v in enumerate(basis))
    r_corr = sum(col * c_res[i] for i, col in enumerate(cols_full))
    # For constraintScheme=1, enforce the source rows directly and keep the corrected
    # solution. Projecting out the basis reintroduces the constraint residuals.
    return x_vec + x_corr, r + r_corr


def _project_constraint_scheme1_nullspace_solution(
    *,
    op: V3FullSystemOperator,
    x_vec: jnp.ndarray,
    rhs_vec: jnp.ndarray,
    matvec_op: V3FullSystemOperator,
    enabled_env_var: str,
) -> jnp.ndarray:
    """Project solution to constraintScheme=1 nullspace complement and enforce source rows."""
    x_proj, _r = _project_constraint_scheme1_nullspace_solution_with_residual(
        op=op,
        x_vec=x_vec,
        rhs_vec=rhs_vec,
        matvec_op=matvec_op,
        enabled_env_var=enabled_env_var,
    )
    return x_proj


def solve_v3_transport_matrix_linear_gmres(
    *,
    nml: Namelist,
    x0: jnp.ndarray | None = None,
    x0_by_rhs: dict[int, jnp.ndarray] | None = None,
    tol: float = 1e-10,
    atol: float = 0.0,
    restart: int = 80,
    maxiter: int | None = 400,
    solve_method: str = "auto",
    identity_shift: float = 0.0,
    phi1_hat_base: jnp.ndarray | None = None,
    emit: Callable[[int, str], None] | None = None,
    input_namelist: Path | None = None,
    which_rhs_values: Sequence[int] | None = None,
    force_stream_diagnostics: bool | None = None,
    force_store_state: bool | None = None,
    parallel_workers: int | None = None,
) -> V3TransportMatrixSolveResult:
    """Compute a RHSMode=2/3 transport matrix by running all `whichRHS` solves matrix-free in JAX.

    Notes
    -----
    This mirrors the v3 `solver.F90` RHSMode=2/3 path:
    - Loop `whichRHS`
    - Overwrite (dnHatdpsiHats, dTHatdpsiHats, EParallelHat)
    - Build the RHS via `evaluateResidual(f=0)`
    - Solve `A x = rhs`
    - Use `diagnostics.F90` formulas to fill `transportMatrix`
    """
    t_all = Timer()
    if emit is not None:
        emit(0, "solve_v3_transport_matrix_linear_gmres: starting whichRHS loop")
    op0 = full_system_operator_from_namelist(nml=nml, identity_shift=identity_shift, phi1_hat_base=phi1_hat_base)
    _set_precond_size_hint(int(op0.total_size))
    state_in_env = os.environ.get("SFINCS_JAX_STATE_IN", "").strip()
    state_out_env = os.environ.get("SFINCS_JAX_STATE_OUT", "").strip()
    state_x_by_rhs: dict[int, jnp.ndarray] | None = None
    if state_in_env:
        try:
            from .solver_state import load_krylov_state  # noqa: PLC0415

            state = load_krylov_state(path=state_in_env, op=op0)
        except Exception:
            state = None
        if state:
            state_x_by_rhs = state.get("x_by_rhs")
            if x0_by_rhs is None:
                x0_by_rhs = state_x_by_rhs
            if x0 is None:
                x0 = state.get("x_full")
    rhs_mode = int(op0.rhs_mode)
    n = transport_matrix_size_from_rhs_mode(rhs_mode)
    if which_rhs_values is None:
        which_rhs_values = list(range(1, n + 1))
    else:
        which_rhs_values = [int(v) for v in which_rhs_values]
        which_rhs_values = [v for v in which_rhs_values if 1 <= v <= n]
        which_rhs_values = sorted(set(which_rhs_values))
    subset_mode = len(which_rhs_values) < n
    parallel_child = os.environ.get("SFINCS_JAX_TRANSPORT_PARALLEL_CHILD", "").strip().lower() in {"1", "true", "yes", "on"}
    parallel_env = os.environ.get("SFINCS_JAX_TRANSPORT_PARALLEL", "").strip().lower()
    if parallel_workers is None:
        workers_env = os.environ.get("SFINCS_JAX_TRANSPORT_PARALLEL_WORKERS", "").strip()
        try:
            workers_val = int(workers_env) if workers_env else 0
        except ValueError:
            workers_val = 0
        if parallel_env in {"", "0", "false", "no", "off"}:
            parallel_workers = 1
        elif parallel_env in {"process", "auto", "1", "true", "yes", "on"}:
            if workers_val > 0:
                parallel_workers = workers_val
            else:
                cpu_count = os.cpu_count() or 1
                parallel_workers = min(cpu_count, n) if n > 1 else 1
        else:
            parallel_workers = 1
    else:
        parallel_workers = max(1, int(parallel_workers))

    if (
        (not parallel_child)
        and parallel_workers > 1
        and len(which_rhs_values) > 1
        and (input_namelist is not None)
    ):
        if emit is not None:
            emit(
                0,
                "solve_v3_transport_matrix_linear_gmres: parallel whichRHS "
                f"(workers={int(parallel_workers)} rhs_count={len(which_rhs_values)}/{n})",
            )

        def _partition(values: list[int], workers: int) -> list[list[int]]:
            chunks: list[list[int]] = [[] for _ in range(workers)]
            for i, val in enumerate(values):
                chunks[i % workers].append(val)
            return [c for c in chunks if c]

        chunks = _partition(list(which_rhs_values), int(parallel_workers))
        payloads: list[dict[str, object]] = []
        phi1_payload = np.asarray(phi1_hat_base) if phi1_hat_base is not None else None
        for chunk in chunks:
            payloads.append(
                {
                    "input_path": str(input_namelist),
                    "which_rhs_values": chunk,
                    "tol": tol,
                    "atol": atol,
                    "restart": restart,
                    "maxiter": maxiter,
                    "solve_method": solve_method,
                    "identity_shift": identity_shift,
                    "phi1_hat_base": phi1_payload,
                }
            )

        results: list[dict[str, object]] = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=int(parallel_workers)) as pool:
            futures = [pool.submit(_transport_parallel_worker, payload) for payload in payloads]
            for fut in concurrent.futures.as_completed(futures):
                results.append(fut.result())

        s = int(op0.n_species)
        diag_pf = np.zeros((s, n), dtype=np.float64)
        diag_hf = np.zeros((s, n), dtype=np.float64)
        diag_flow = np.zeros((s, n), dtype=np.float64)
        transport_output_fields: dict[str, np.ndarray] = {}
        residual_norms: dict[int, jnp.ndarray] = {}
        elapsed_s = np.zeros((n,), dtype=np.float64)
        for res in results:
            rhs_vals = [int(v) for v in res.get("which_rhs_values", [])]
            idxs = [v - 1 for v in rhs_vals]
            pf = np.asarray(res["particle_flux_vm_psi_hat"], dtype=np.float64)
            hf = np.asarray(res["heat_flux_vm_psi_hat"], dtype=np.float64)
            fl = np.asarray(res["fsab_flow"], dtype=np.float64)
            diag_pf[:, idxs] = pf[:, idxs]
            diag_hf[:, idxs] = hf[:, idxs]
            diag_flow[:, idxs] = fl[:, idxs]
            elapsed_chunk = np.asarray(res.get("elapsed_time_s", np.zeros((n,))), dtype=np.float64)
            elapsed_s[idxs] = elapsed_chunk[idxs]
            residual_norms.update({int(k): jnp.asarray(v, dtype=jnp.float64) for k, v in res.get("residual_norms_by_rhs", {}).items()})

            fields = res.get("transport_output_fields", {})
            if isinstance(fields, dict):
                for key, arr in fields.items():
                    arr_np = np.asarray(arr)
                    if key not in transport_output_fields:
                        transport_output_fields[key] = np.zeros_like(arr_np)
                    if arr_np.ndim > 0 and arr_np.shape[-1] == n:
                        transport_output_fields[key][..., idxs] = arr_np[..., idxs]
                    else:
                        transport_output_fields[key] = arr_np

        tm = v3_transport_matrix_from_flux_arrays(
            op=op0,
            geom=geometry_from_namelist(nml=nml, grids=grids_from_namelist(nml)),
            particle_flux_vm_psi_hat=jnp.asarray(diag_pf, dtype=jnp.float64),
            heat_flux_vm_psi_hat=jnp.asarray(diag_hf, dtype=jnp.float64),
            fsab_flow=jnp.asarray(diag_flow, dtype=jnp.float64),
        )
        return V3TransportMatrixSolveResult(
            op0=op0,
            transport_matrix=tm,
            state_vectors_by_rhs={},
            residual_norms_by_rhs=residual_norms,
            fsab_flow=jnp.asarray(diag_flow, dtype=jnp.float64),
            particle_flux_vm_psi_hat=jnp.asarray(diag_pf, dtype=jnp.float64),
            heat_flux_vm_psi_hat=jnp.asarray(diag_hf, dtype=jnp.float64),
            elapsed_time_s=jnp.asarray(elapsed_s, dtype=jnp.float64),
            transport_output_fields=transport_output_fields,
        )
    if emit is not None:
        emit(1, f"solve_v3_transport_matrix_linear_gmres: rhs_mode={rhs_mode} whichRHS_count={n} total_size={int(op0.total_size)}")

    low_memory_env = os.environ.get("SFINCS_JAX_TRANSPORT_LOW_MEMORY", "").strip().lower()
    if low_memory_env in {"1", "true", "yes", "on"}:
        low_memory_outputs = True
    elif low_memory_env in {"0", "false", "no", "off"}:
        low_memory_outputs = False
    else:
        low_memory_outputs = int(op0.total_size) * int(n) >= 200_000
    stream_env = os.environ.get("SFINCS_JAX_TRANSPORT_STREAM_DIAGNOSTICS", "").strip().lower()
    if stream_env in {"1", "true", "yes", "on"}:
        stream_diagnostics = True
    elif stream_env in {"0", "false", "no", "off"}:
        stream_diagnostics = False
    else:
        stream_diagnostics = low_memory_outputs
    store_state_env = os.environ.get("SFINCS_JAX_TRANSPORT_STORE_STATE", "").strip().lower()
    if store_state_env in {"1", "true", "yes", "on"}:
        store_state_vectors = True
    elif store_state_env in {"0", "false", "no", "off"}:
        store_state_vectors = False
    else:
        store_state_vectors = not stream_diagnostics
    if state_out_env:
        store_state_vectors = True
    if (not stream_diagnostics) and (not store_state_vectors):
        store_state_vectors = True
        if emit is not None:
            emit(1, "solve_v3_transport_matrix_linear_gmres: forcing state storage (streaming disabled)")
    if force_stream_diagnostics is not None:
        stream_diagnostics = bool(force_stream_diagnostics)
    if force_store_state is not None:
        store_state_vectors = bool(force_store_state)
    if subset_mode and not stream_diagnostics:
        stream_diagnostics = True
        if emit is not None:
            emit(1, "solve_v3_transport_matrix_linear_gmres: streaming diagnostics forced for subset whichRHS")

    solve_method_use = solve_method
    force_krylov_env = os.environ.get("SFINCS_JAX_TRANSPORT_FORCE_KRYLOV", "").strip().lower()
    force_krylov = force_krylov_env in {"1", "true", "yes", "on"}
    force_dense_env = os.environ.get("SFINCS_JAX_TRANSPORT_FORCE_DENSE", "").strip().lower()
    force_dense = force_dense_env in {"1", "true", "yes", "on"}
    if low_memory_outputs:
        force_krylov = True
        force_dense = False
    dense_fallback_env = os.environ.get("SFINCS_JAX_TRANSPORT_DENSE_FALLBACK", "").strip().lower()
    dense_fallback_max_env = os.environ.get("SFINCS_JAX_TRANSPORT_DENSE_FALLBACK_MAX", "").strip()
    try:
        dense_fallback_max = int(dense_fallback_max_env) if dense_fallback_max_env else 0
    except ValueError:
        dense_fallback_max = 0
    dense_retry_env = os.environ.get("SFINCS_JAX_TRANSPORT_DENSE_RETRY_MAX", "").strip()
    try:
        if dense_retry_env:
            dense_retry_max = int(dense_retry_env)
        else:
            dense_retry_max = 6000 if int(rhs_mode) in {2, 3} else 0
    except ValueError:
        dense_retry_max = 3000 if int(rhs_mode) in {2, 3} else 0
    if low_memory_outputs:
        dense_retry_max = 0
    dense_mem_env = os.environ.get("SFINCS_JAX_TRANSPORT_DENSE_MAX_MB", "").strip()
    try:
        dense_mem_max_mb = float(dense_mem_env) if dense_mem_env else 128.0
    except ValueError:
        dense_mem_max_mb = 128.0
    dense_mem_est_mb64 = (int(op0.total_size) ** 2) * 8.0 / 1.0e6
    dense_mem_est_mb32 = (int(op0.total_size) ** 2) * 4.0 / 1.0e6
    dense_mem_block64 = bool(dense_mem_max_mb > 0.0 and dense_mem_est_mb64 > dense_mem_max_mb)
    dense_mem_block32 = bool(dense_mem_max_mb > 0.0 and dense_mem_est_mb32 > dense_mem_max_mb)
    dense_mem_block = dense_mem_block32
    dense_use_mixed = dense_mem_block64 and not dense_mem_block32
    dense_fallback_enabled_env = dense_fallback_env in {"1", "true", "yes", "on"}
    dense_fallback_disabled_env = dense_fallback_env in {"0", "false", "no", "off"}
    if dense_fallback_enabled_env:
        dense_fallback = True
        if not dense_fallback_max_env:
            dense_fallback_max = 1600
    elif dense_fallback_disabled_env:
        dense_fallback = False
    else:
        # Default to a dense fallback for RHSMode=3 monoenergetic solves when the system is modest,
        # since iterative Krylov often stalls for the E_parallel RHS.
        dense_fallback = int(rhs_mode) == 3
        if dense_fallback and not dense_fallback_max_env:
            dense_fallback_max = 6000
    if dense_mem_block:
        dense_fallback = False
        dense_retry_max = 0
        force_dense = False
        if emit is not None:
            emit(
                1,
                "solve_v3_transport_matrix_linear_gmres: dense fallback disabled "
                f"(est_mem32={dense_mem_est_mb32:.1f} MB > {dense_mem_max_mb:.1f} MB)",
            )
        if str(solve_method_use).lower() in {"auto", "default", "batched"}:
            solve_method_use = "incremental"
    elif dense_use_mixed and emit is not None:
        emit(
            1,
            "solve_v3_transport_matrix_linear_gmres: dense fallback using float32 "
            f"(est_mem64={dense_mem_est_mb64:.1f} MB > {dense_mem_max_mb:.1f} MB)",
        )
    if low_memory_outputs:
        dense_fallback = False
    if int(rhs_mode) in {2, 3}:
        if force_dense:
            solve_method_use = "dense"
            if emit is not None:
                emit(0, f"solve_v3_transport_matrix_linear_gmres: forced dense solve for RHSMode={rhs_mode} (n={int(op0.total_size)})")
        elif (
            int(rhs_mode) == 2
            and (not force_krylov)
            and str(solve_method_use).lower() in {"auto", "default", "batched", "incremental"}
            and int(op0.total_size) <= 1500
            and (not dense_mem_block)
        ):
            solve_method_use = "dense"
            if emit is not None:
                emit(
                    0,
                    "solve_v3_transport_matrix_linear_gmres: auto dense solve for RHSMode=2 "
                    f"(n={int(op0.total_size)})",
                )
        elif (
            dense_fallback
            and (not force_krylov)
            and int(op0.total_size) <= dense_fallback_max
            and str(solve_method_use).lower() in {"auto", "default", "batched", "incremental"}
            and (not dense_mem_block)
        ):
            # On some JAX versions/platforms, `jax.scipy.sparse.linalg.gmres` can return NaNs for
            # small ill-conditioned problems (observed in CI for RHSMode=3 scheme12 fixtures).
            # Enable a dense fallback only when explicitly requested.
            solve_method_use = "dense"
            if emit is not None:
                emit(0, f"solve_v3_transport_matrix_linear_gmres: dense fallback enabled for RHSMode={rhs_mode} (n={int(op0.total_size)})")

    implicit_env = os.environ.get("SFINCS_JAX_IMPLICIT_SOLVE", "").strip().lower()
    use_implicit = implicit_env not in {"0", "false", "no", "off"}

    gmres_restart_env = os.environ.get("SFINCS_JAX_TRANSPORT_GMRES_RESTART", "").strip()
    try:
        gmres_restart = int(gmres_restart_env) if gmres_restart_env else min(int(restart), 40)
    except ValueError:
        gmres_restart = min(int(restart), 40)
    if dense_mem_block and gmres_restart < 80:
        gmres_restart = 80

    if dense_mem_block:
        if maxiter is None:
            maxiter = 800
        else:
            maxiter = max(int(maxiter), 800)

    use_solver_jit = _use_solver_jit(int(op0.total_size))

    def _dense_dtype(dtype_in: jnp.dtype) -> jnp.dtype:
        return jnp.float32 if dense_use_mixed else dtype_in

    def _solver_kind(method: str) -> tuple[str, str]:
        method_l = str(method).strip().lower()
        if method_l in {"auto", "default"}:
            if int(rhs_mode) in {2, 3}:
                # Favor short-recurrence Krylov for RHSMode=2/3; fall back to GMRES if needed.
                return "bicgstab", "batched"
            return "bicgstab", "batched"
        if method_l in {"bicgstab", "bicgstab_jax"}:
            return "bicgstab", "batched"
        return "gmres", method_l

    def _restart_for_method(method: str) -> int:
        return gmres_restart if _solver_kind(method)[0] == "gmres" else int(restart)

    def _solve_linear(
        *,
        matvec_fn,
        b_vec: jnp.ndarray,
        x0_vec: jnp.ndarray | None,
        tol_val: float,
        atol_val: float,
        restart_val: int,
        maxiter_val: int | None,
        solve_method_val: str,
        preconditioner_val=None,
        precondition_side_val: str = "left",
    ):
        if use_implicit:
            solver_kind, gmres_method = _solver_kind(solve_method_val)
            return linear_custom_solve(
                matvec=matvec_fn,
                b=b_vec,
                preconditioner=preconditioner_val,
                x0=x0_vec,
                tol=tol_val,
                atol=atol_val,
                restart=restart_val,
                maxiter=maxiter_val,
                solve_method=gmres_method,
                solver=solver_kind,
                precondition_side=precondition_side_val,
                size_hint=int(op0.total_size),
            )
        solver_fn = gmres_solve_jit if use_solver_jit else gmres_solve
        return solver_fn(
            matvec=matvec_fn,
            b=b_vec,
            preconditioner=preconditioner_val,
            x0=x0_vec,
            tol=tol_val,
            atol=atol_val,
            restart=restart_val,
            maxiter=maxiter_val,
            solve_method=solve_method_val,
            precondition_side=precondition_side_val,
        )

    def _solve_linear_with_residual(
        *,
        matvec_fn,
        b_vec: jnp.ndarray,
        x0_vec: jnp.ndarray | None,
        tol_val: float,
        atol_val: float,
        restart_val: int,
        maxiter_val: int | None,
        solve_method_val: str,
        preconditioner_val=None,
        precondition_side_val: str = "left",
    ) -> tuple[GMRESSolveResult, jnp.ndarray]:
        solver_kind, gmres_method = _solver_kind(solve_method_val)
        if use_implicit:
            return linear_custom_solve_with_residual(
                matvec=matvec_fn,
                b=b_vec,
                preconditioner=preconditioner_val,
                x0=x0_vec,
                tol=tol_val,
                atol=atol_val,
                restart=restart_val,
                maxiter=maxiter_val,
                solve_method=gmres_method,
                solver=solver_kind,
                precondition_side=precondition_side_val,
                size_hint=int(op0.total_size),
            )
        if solver_kind == "bicgstab":
            solver_fn = bicgstab_solve_with_residual_jit if use_solver_jit else bicgstab_solve_with_residual
            return solver_fn(
                matvec=matvec_fn,
                b=b_vec,
                preconditioner=preconditioner_val,
                x0=x0_vec,
                tol=tol_val,
                atol=atol_val,
                maxiter=maxiter_val,
                precondition_side=precondition_side_val,
            )
        solver_fn = gmres_solve_with_residual_jit if use_solver_jit else gmres_solve_with_residual
        return solver_fn(
            matvec=matvec_fn,
            b=b_vec,
            preconditioner=preconditioner_val,
            x0=x0_vec,
            tol=tol_val,
            atol=atol_val,
            restart=restart_val,
            maxiter=maxiter_val,
            solve_method=gmres_method,
            precondition_side=precondition_side_val,
        )

    active_dof_env = os.environ.get("SFINCS_JAX_TRANSPORT_ACTIVE_DOF", "").strip().lower()
    active_dof_reason: str | None = None
    if active_dof_env in {"0", "false", "no", "off"}:
        use_active_dof_mode = False
    elif active_dof_env in {"1", "true", "yes", "on"}:
        use_active_dof_mode = True
        active_dof_reason = "env"
    else:
        if int(rhs_mode) in {2, 3}:
            nxi_for_x = np.asarray(op0.fblock.collisionless.n_xi_for_x, dtype=np.int32)
            use_active_dof_mode = bool(np.any(nxi_for_x < int(op0.n_xi)))
            if use_active_dof_mode:
                active_dof_reason = "auto"
        else:
            use_active_dof_mode = False
    # For reduced active-DOF parity mode, prefer Krylov iterations over dense direct
    # solves to stay closer to upstream PETSc/KSP behavior for singular transport systems.
    if use_active_dof_mode and str(solve_method_use).lower() == "dense":
        solve_method_use = str(solve_method)
    elif int(rhs_mode) in {2, 3} and emit is not None and active_dof_env not in {"0", "false", "no", "off"}:
        emit(
            1,
            "solve_v3_transport_matrix_linear_gmres: active-DOF mode disabled "
            "(set SFINCS_JAX_TRANSPORT_ACTIVE_DOF=1 to enable; "
            "SFINCS_JAX_TRANSPORT_ACTIVE_DOF=0 to force full-size solve)",
        )
    active_idx_np: np.ndarray | None = None
    active_idx_jnp: jnp.ndarray | None = None
    full_to_active_jnp: jnp.ndarray | None = None
    active_size = int(op0.total_size)
    if use_active_dof_mode:
        active_idx_np = _transport_active_dof_indices(op0)
        active_idx_jnp = jnp.asarray(active_idx_np, dtype=jnp.int32)
        full_to_active_np = np.zeros((int(op0.total_size),), dtype=np.int32)
        full_to_active_np[np.asarray(active_idx_np, dtype=np.int32)] = np.arange(1, int(active_idx_np.shape[0]) + 1, dtype=np.int32)
        full_to_active_jnp = jnp.asarray(full_to_active_np, dtype=jnp.int32)
        active_size = int(active_idx_np.shape[0])
        if emit is not None:
            reason = f" ({active_dof_reason})" if active_dof_reason else ""
            emit(
                1,
                "solve_v3_transport_matrix_linear_gmres: active-DOF mode enabled "
                f"(size={active_size}/{int(op0.total_size)}){reason}",
            )

    dense_mem_est_active_mb64 = (int(active_size) ** 2) * 8.0 / 1.0e6
    dense_mem_est_active_mb32 = (int(active_size) ** 2) * 4.0 / 1.0e6
    dense_mem_block_active32 = bool(dense_mem_max_mb > 0.0 and dense_mem_est_active_mb32 > dense_mem_max_mb)
    dense_mem_block_active64 = bool(dense_mem_max_mb > 0.0 and dense_mem_est_active_mb64 > dense_mem_max_mb)
    if dense_mem_block_active32 and not dense_mem_block:
        dense_mem_block = True
        dense_use_mixed = False
        dense_fallback = False
        dense_retry_max = 0
        force_dense = False
        if emit is not None:
            emit(
                1,
                "solve_v3_transport_matrix_linear_gmres: dense fallback disabled "
                f"(active_est_mem32={dense_mem_est_active_mb32:.1f} MB > {dense_mem_max_mb:.1f} MB)",
            )
        if str(solve_method_use).lower() == "dense":
            solve_method_use = "incremental"
    elif dense_mem_block_active64 and not dense_mem_block and not dense_use_mixed:
        dense_use_mixed = True
        if emit is not None:
            emit(
                1,
                "solve_v3_transport_matrix_linear_gmres: dense fallback using float32 "
                f"(active_est_mem64={dense_mem_est_active_mb64:.1f} MB > {dense_mem_max_mb:.1f} MB)",
            )
    if dense_mem_block:
        dense_precond_enabled = False

    if (
        int(rhs_mode) == 2
        and (not force_krylov)
        and (not force_dense)
        and str(solve_method_use).lower() in {"auto", "default", "batched", "incremental"}
    ):
        auto_dense_size = int(active_size) if use_active_dof_mode else int(op0.total_size)
        if auto_dense_size <= 1500 and (not dense_mem_block):
            solve_method_use = "dense"
            if emit is not None:
                emit(
                    0,
                    "solve_v3_transport_matrix_linear_gmres: auto dense solve for RHSMode=2 "
                    f"(n={auto_dense_size})",
                )

    dense_precond_env = os.environ.get("SFINCS_JAX_TRANSPORT_DENSE_PRECOND_MAX", "").strip()
    try:
        if dense_precond_env:
            dense_precond_max = int(dense_precond_env)
        else:
            dense_precond_max = 1600 if int(rhs_mode) == 2 else 600
    except ValueError:
        dense_precond_max = 1600 if int(rhs_mode) == 2 else 600
    dense_precond_mem_env = os.environ.get("SFINCS_JAX_TRANSPORT_DENSE_PRECOND_MAX_MB", "").strip()
    try:
        dense_precond_mem_max_mb = float(dense_precond_mem_env) if dense_precond_mem_env else min(32.0, dense_mem_max_mb or 32.0)
    except ValueError:
        dense_precond_mem_max_mb = min(32.0, dense_mem_max_mb or 32.0)
    dense_precond_size = int(active_size) if use_active_dof_mode else int(op0.total_size)
    dense_precond_bytes = 4.0 if dense_use_mixed else 8.0
    dense_precond_est_mb = (dense_precond_size**2) * dense_precond_bytes / 1.0e6
    dense_precond_mem_block = bool(dense_precond_mem_max_mb > 0.0 and dense_precond_est_mb > dense_precond_mem_max_mb)
    if dense_precond_mem_block and emit is not None:
        emit(
            1,
            "solve_v3_transport_matrix_linear_gmres: dense preconditioner disabled "
            f"(est_mem={dense_precond_est_mb:.1f} MB > {dense_precond_mem_max_mb:.1f} MB)",
        )
    dense_precond_enabled = (
        dense_precond_max > 0
        and int(rhs_mode) in {2, 3}
        and int(dense_precond_size) <= dense_precond_max
        and str(solve_method_use).lower() != "dense"
        and (not low_memory_outputs)
        and (not dense_mem_block)
        and (not dense_precond_mem_block)
    )
    dense_precond_cache_full: dict[tuple[object, int], Callable[[jnp.ndarray], jnp.ndarray]] = {}
    dense_precond_cache_reduced: dict[tuple[object, int], Callable[[jnp.ndarray], jnp.ndarray]] = {}
    dense_solver_cache_full: dict[tuple[object, int], Callable[[jnp.ndarray], jnp.ndarray]] = {}
    dense_solver_cache_reduced: dict[tuple[object, int], Callable[[jnp.ndarray], jnp.ndarray]] = {}

    reduce_full = None
    expand_reduced = None
    if use_active_dof_mode:
        assert active_idx_jnp is not None
        assert full_to_active_jnp is not None

        def reduce_full(v_full: jnp.ndarray) -> jnp.ndarray:
            return v_full[active_idx_jnp]

        def expand_reduced(v_reduced: jnp.ndarray) -> jnp.ndarray:
            z0 = jnp.zeros((1,), dtype=v_reduced.dtype)
            padded = jnp.concatenate([z0, v_reduced], axis=0)
            return padded[full_to_active_jnp]

    transport_precond_env = os.environ.get("SFINCS_JAX_TRANSPORT_PRECOND", "").strip().lower()
    if transport_precond_env in {"0", "none", "off", "false", "no"}:
        transport_precond_kind = None
    elif transport_precond_env in {
        "collision",
        "block",
        "block_jacobi",
        "sxblock",
        "block_sx",
        "species_x",
        "xmg",
        "multigrid",
        "sparse_jax",
    }:
        transport_precond_kind = transport_precond_env
    else:
        transport_precond_kind = "auto"

    preconditioner_full = None
    preconditioner_reduced = None
    strong_precond_kind: str | None = None
    default_solver_kind = _solver_kind(solve_method_use)[0]
    precond_kind_used: str | None = None
    transport_sparse_drop_tol_env = os.environ.get("SFINCS_JAX_TRANSPORT_SPARSE_DROP_TOL", "").strip()
    transport_sparse_drop_rel_env = os.environ.get("SFINCS_JAX_TRANSPORT_SPARSE_DROP_REL", "").strip()
    try:
        transport_sparse_drop_tol = float(transport_sparse_drop_tol_env) if transport_sparse_drop_tol_env else 0.0
    except ValueError:
        transport_sparse_drop_tol = 0.0
    try:
        transport_sparse_drop_rel = float(transport_sparse_drop_rel_env) if transport_sparse_drop_rel_env else 1.0e-6
    except ValueError:
        transport_sparse_drop_rel = 1.0e-6
    transport_sparse_reg_env = os.environ.get("SFINCS_JAX_TRANSPORT_SPARSE_JAX_REG", "").strip()
    try:
        transport_sparse_reg = float(transport_sparse_reg_env) if transport_sparse_reg_env else 1e-10
    except ValueError:
        transport_sparse_reg = 1e-10
    transport_sparse_omega_env = os.environ.get("SFINCS_JAX_TRANSPORT_SPARSE_JAX_OMEGA", "").strip()
    try:
        transport_sparse_omega = float(transport_sparse_omega_env) if transport_sparse_omega_env else 0.8
    except ValueError:
        transport_sparse_omega = 0.8
    transport_sparse_sweeps_env = os.environ.get("SFINCS_JAX_TRANSPORT_SPARSE_JAX_SWEEPS", "").strip()
    try:
        transport_sparse_sweeps = int(transport_sparse_sweeps_env) if transport_sparse_sweeps_env else 2
    except ValueError:
        transport_sparse_sweeps = 2
    transport_sparse_sweeps = max(1, transport_sparse_sweeps)
    transport_sparse_max_env = os.environ.get("SFINCS_JAX_TRANSPORT_SPARSE_JAX_MAX_MB", "").strip()
    try:
        transport_sparse_max_mb = float(transport_sparse_max_env) if transport_sparse_max_env else 128.0
    except ValueError:
        transport_sparse_max_mb = 128.0
    if transport_precond_kind is not None and int(rhs_mode) in {2, 3}:
        precond_kind = transport_precond_kind
        if precond_kind == "auto":
            block_max_env = os.environ.get("SFINCS_JAX_TRANSPORT_PRECOND_BLOCK_MAX", "").strip()
            try:
                block_max = int(block_max_env) if block_max_env else 5000
            except ValueError:
                block_max = 5000
            sxblock_max_env = os.environ.get("SFINCS_JAX_TRANSPORT_SXBLOCK_MAX", "").strip()
            try:
                sxblock_max = int(sxblock_max_env) if sxblock_max_env else 64
            except ValueError:
                sxblock_max = 64
            n_block = int(op0.n_species) * int(op0.n_x)
            if op0.fblock.fp is not None:
                if n_block <= sxblock_max:
                    precond_kind = "sxblock"
                elif int(op0.total_size) <= block_max and default_solver_kind != "bicgstab":
                    precond_kind = "sxblock"
                elif default_solver_kind == "bicgstab":
                    precond_kind = "collision"
                else:
                    precond_kind = "collision"
                if n_block <= sxblock_max:
                    strong_precond_kind = "sxblock"
                elif int(op0.total_size) <= block_max:
                    strong_precond_kind = "block"
                else:
                    strong_precond_kind = "xmg"
            else:
                if int(op0.total_size) <= block_max:
                    precond_kind = "block"
                    strong_precond_kind = "block"
                else:
                    precond_kind = "collision"
                    strong_precond_kind = "xmg"
            if dense_mem_block and strong_precond_kind is not None:
                precond_kind = strong_precond_kind
        precond_kind_used = precond_kind
        if precond_kind in {"xmg", "multigrid"}:
            preconditioner_full = _build_rhsmode23_xmg_preconditioner(op=op0)
            if use_active_dof_mode and reduce_full is not None and expand_reduced is not None:
                preconditioner_reduced = _build_rhsmode23_xmg_preconditioner(
                    op=op0, reduce_full=reduce_full, expand_reduced=expand_reduced
                )
        elif precond_kind == "sparse_jax":
            precond_dtype = _precond_dtype(int(active_size) if use_active_dof_mode else int(op0.total_size))
            bytes_per = 4.0 if precond_dtype == jnp.float32 else 8.0
            size_est = int(active_size) if use_active_dof_mode else int(op0.total_size)
            est_mb = (size_est**2) * bytes_per / 1.0e6
            if transport_sparse_max_mb > 0.0 and est_mb > transport_sparse_max_mb:
                preconditioner_full = _build_rhsmode23_collision_preconditioner(op=op0)
                if use_active_dof_mode and reduce_full is not None and expand_reduced is not None:
                    preconditioner_reduced = _build_rhsmode23_collision_preconditioner(
                        op=op0, reduce_full=reduce_full, expand_reduced=expand_reduced
                    )
                if emit is not None:
                    emit(
                        1,
                        "solve_v3_transport_matrix_linear_gmres: sparse_jax preconditioner disabled "
                        f"(est_mem={est_mb:.1f} MB > max_mb={transport_sparse_max_mb:.1f})",
                    )
            else:
                cache_key_full = _transport_precond_cache_key(op0, f"sparse_jax_{size_est}")
                def _mv_sparse_full(x: jnp.ndarray, op=op0) -> jnp.ndarray:
                    return apply_v3_full_system_operator_cached(op, x)
                preconditioner_full = _build_sparse_jax_preconditioner_from_matvec(
                    matvec=_mv_sparse_full,
                    n=int(op0.total_size),
                    dtype=precond_dtype,
                    cache_key=cache_key_full,
                    drop_tol=transport_sparse_drop_tol,
                    drop_rel=transport_sparse_drop_rel,
                    reg=transport_sparse_reg,
                    omega=transport_sparse_omega,
                    sweeps=transport_sparse_sweeps,
                    emit=emit,
                )
                if use_active_dof_mode and reduce_full is not None and expand_reduced is not None:
                    cache_key_reduced = _transport_precond_cache_key(op0, f"sparse_jax_active_{int(active_size)}")
                    def _mv_sparse_reduced(x_reduced: jnp.ndarray, op=op0) -> jnp.ndarray:
                        y_full = apply_v3_full_system_operator_cached(op, expand_reduced(x_reduced))
                        return reduce_full(y_full)
                    preconditioner_reduced = _build_sparse_jax_preconditioner_from_matvec(
                        matvec=_mv_sparse_reduced,
                        n=int(active_size),
                        dtype=precond_dtype,
                        cache_key=cache_key_reduced,
                        drop_tol=transport_sparse_drop_tol,
                        drop_rel=transport_sparse_drop_rel,
                        reg=transport_sparse_reg,
                        omega=transport_sparse_omega,
                        sweeps=transport_sparse_sweeps,
                        emit=emit,
                    )
        elif precond_kind in {"sxblock", "block_sx", "species_x"}:
            preconditioner_full = _build_rhsmode23_sxblock_preconditioner(op=op0)
            if use_active_dof_mode and reduce_full is not None and expand_reduced is not None:
                preconditioner_reduced = _build_rhsmode23_sxblock_preconditioner(
                    op=op0, reduce_full=reduce_full, expand_reduced=expand_reduced
                )
        elif precond_kind in {"block", "block_jacobi"}:
            preconditioner_full = _build_rhsmode23_block_preconditioner(op=op0)
            if use_active_dof_mode and reduce_full is not None and expand_reduced is not None:
                preconditioner_reduced = _build_rhsmode23_block_preconditioner(
                    op=op0, reduce_full=reduce_full, expand_reduced=expand_reduced
                )
        else:
            preconditioner_full = _build_rhsmode23_collision_preconditioner(op=op0)
            if use_active_dof_mode and reduce_full is not None and expand_reduced is not None:
                preconditioner_reduced = _build_rhsmode23_collision_preconditioner(
                    op=op0, reduce_full=reduce_full, expand_reduced=expand_reduced
                )

    strong_preconditioner_full = None
    strong_preconditioner_reduced = None

    def _get_strong_preconditioner(use_reduced: bool) -> Callable[[jnp.ndarray], jnp.ndarray] | None:
        nonlocal strong_preconditioner_full, strong_preconditioner_reduced
        if strong_precond_kind is None:
            return None
        if precond_kind_used is not None and strong_precond_kind == precond_kind_used:
            return preconditioner_reduced if use_reduced else preconditioner_full
        if use_reduced:
            if strong_preconditioner_reduced is None:
                if strong_precond_kind in {"xmg", "multigrid"}:
                    strong_preconditioner_reduced = _build_rhsmode23_xmg_preconditioner(
                        op=op0, reduce_full=reduce_full, expand_reduced=expand_reduced
                    )
                elif strong_precond_kind in {"sxblock", "block_sx", "species_x"}:
                    strong_preconditioner_reduced = _build_rhsmode23_sxblock_preconditioner(
                        op=op0, reduce_full=reduce_full, expand_reduced=expand_reduced
                    )
                elif strong_precond_kind in {"block", "block_jacobi"}:
                    strong_preconditioner_reduced = _build_rhsmode23_block_preconditioner(
                        op=op0, reduce_full=reduce_full, expand_reduced=expand_reduced
                    )
                else:
                    strong_preconditioner_reduced = _build_rhsmode23_collision_preconditioner(
                        op=op0, reduce_full=reduce_full, expand_reduced=expand_reduced
                    )
            return strong_preconditioner_reduced
        if strong_preconditioner_full is None:
            if strong_precond_kind in {"xmg", "multigrid"}:
                strong_preconditioner_full = _build_rhsmode23_xmg_preconditioner(op=op0)
            elif strong_precond_kind in {"sxblock", "block_sx", "species_x"}:
                strong_preconditioner_full = _build_rhsmode23_sxblock_preconditioner(op=op0)
            elif strong_precond_kind in {"block", "block_jacobi"}:
                strong_preconditioner_full = _build_rhsmode23_block_preconditioner(op=op0)
            else:
                strong_preconditioner_full = _build_rhsmode23_collision_preconditioner(op=op0)
        return strong_preconditioner_full

    def _dense_preconditioner_for_matvec(
        *,
        matvec_fn,
        n: int,
        dtype: jnp.dtype,
        cache: dict[tuple[object, int], Callable[[jnp.ndarray], jnp.ndarray]],
        key: tuple[object, int],
    ) -> Callable[[jnp.ndarray], jnp.ndarray]:
        if key in cache:
            return cache[key]
        import jax.scipy.linalg as jla  # noqa: PLC0415

        a_dense = assemble_dense_matrix_from_matvec(matvec=matvec_fn, n=n, dtype=dtype)
        a_dense = jnp.asarray(a_dense, dtype=dtype)
        lu, piv = jla.lu_factor(a_dense)

        def precond(v: jnp.ndarray) -> jnp.ndarray:
            return jla.lu_solve((lu, piv), v)

        cache[key] = precond
        return precond

    def _dense_solver_for_matvec(
        *,
        matvec_fn,
        n: int,
        dtype: jnp.dtype,
        cache: dict[tuple[object, int], Callable[[jnp.ndarray], jnp.ndarray]],
        key: tuple[object, int],
    ) -> Callable[[jnp.ndarray], jnp.ndarray]:
        if key in cache:
            return cache[key]
        import jax.scipy.linalg as jla  # noqa: PLC0415

        a_dense = assemble_dense_matrix_from_matvec(matvec=matvec_fn, n=n, dtype=dtype)
        a_dense = jnp.asarray(a_dense, dtype=dtype)
        lu, piv = jla.lu_factor(a_dense)

        def solve(v: jnp.ndarray) -> jnp.ndarray:
            return jla.lu_solve((lu, piv), v)

        cache[key] = solve
        return solve

    # Geometry scalars needed for the transport-matrix formulas.
    grids = grids_from_namelist(nml)
    geom = geometry_from_namelist(nml=nml, grids=grids)

    state_vectors: dict[int, jnp.ndarray] = {}
    residual_norms: dict[int, jnp.ndarray] = {}
    elapsed_s = np.zeros((n,), dtype=np.float64)
    op_rhs_by_index = [with_transport_rhs_settings(op0, which_rhs=which_rhs) for which_rhs in which_rhs_values]
    rhs_by_index = [rhs_v3_full_system_jit(op_rhs) for op_rhs in op_rhs_by_index]

    use_op_rhs_in_matvec = bool(op0.include_phi1_in_kinetic)
    env_transport_matvec = os.environ.get("SFINCS_JAX_TRANSPORT_MATVEC_MODE", "").strip().lower()
    if env_transport_matvec == "rhs":
        use_op_rhs_in_matvec = True
    elif env_transport_matvec == "base":
        use_op_rhs_in_matvec = False
    op_matvec_by_index = [op_rhs if use_op_rhs_in_matvec else op0 for op_rhs in op_rhs_by_index]

    env_diag_op = os.environ.get("SFINCS_JAX_TRANSPORT_DIAG_OP", "").strip().lower()
    use_diag_op0 = env_diag_op != "rhs"
    diag_op_by_index = op_rhs_by_index if not use_diag_op0 else None

    transport_output_fields: dict[str, np.ndarray] | None = None
    diag_pf_arr: np.ndarray | None = None
    diag_hf_arr: np.ndarray | None = None
    diag_flow_arr: np.ndarray | None = None
    if stream_diagnostics:
        s = int(op0.n_species)
        t = int(op0.n_theta)
        z = int(op0.n_zeta)
        x = int(op0.n_x)
        diag_pf_arr = np.zeros((s, n), dtype=np.float64)
        diag_hf_arr = np.zeros((s, n), dtype=np.float64)
        diag_flow_arr = np.zeros((s, n), dtype=np.float64)

        def _alloc_ztsn() -> np.ndarray:
            return np.zeros((z, t, s, n), dtype=np.float64)

        def _alloc_zt_n() -> np.ndarray:
            return np.zeros((z, t, n), dtype=np.float64)

        def _alloc_sn() -> np.ndarray:
            return np.zeros((s, n), dtype=np.float64)

        def _alloc_xsn() -> np.ndarray:
            return np.zeros((x, s, n), dtype=np.float64)

        def _alloc_2sn() -> np.ndarray:
            return np.zeros((2, s, n), dtype=np.float64)

        dens = _alloc_ztsn()
        pres = _alloc_ztsn()
        pres_aniso = _alloc_ztsn()
        flow = _alloc_ztsn()
        total_dens = _alloc_ztsn()
        total_pres = _alloc_ztsn()
        vel_fsadens = _alloc_ztsn()
        vel_total = _alloc_ztsn()
        mach = _alloc_ztsn()
        j_hat = _alloc_zt_n()
        fsa_dens = _alloc_sn()
        fsa_pres = _alloc_sn()

        mf_before_vm = _alloc_ztsn()
        mf_before_vm0 = _alloc_ztsn()
        mf_before_vE = _alloc_ztsn()
        mf_before_vE0 = _alloc_ztsn()
        mf_vm_psi_hat = _alloc_sn()
        mf_vm0_psi_hat = _alloc_sn()

        ntv_before = _alloc_ztsn()
        ntv = _alloc_sn()

        pf_before_vm = _alloc_ztsn()
        hf_before_vm = _alloc_ztsn()
        pf_before_vm0 = _alloc_ztsn()
        hf_before_vm0 = _alloc_ztsn()
        pf_before_ve = _alloc_ztsn()
        hf_before_ve = _alloc_ztsn()
        pf_before_ve0 = _alloc_ztsn()
        hf_before_ve0 = _alloc_ztsn()

        pf_vm_psi_hat = _alloc_sn()
        hf_vm_psi_hat = _alloc_sn()
        pf_vm0_psi_hat = _alloc_sn()
        hf_vm0_psi_hat = _alloc_sn()
        pf_vs_x = _alloc_xsn()
        hf_vs_x = _alloc_xsn()
        fsab_flow = _alloc_sn()
        fsab_flow_vs_x = _alloc_xsn()

        sources = None
        if int(op0.constraint_scheme) == 2:
            sources = _alloc_xsn()
        elif int(op0.constraint_scheme) in {1, 3, 4}:
            sources = _alloc_2sn()

        geom_params = nml.group("geometryParameters")
        geometry_scheme = int(geom_params.get("GEOMETRYSCHEME", geom_params.get("geometryScheme", -1)))
        compute_ntv = geometry_scheme != 5
        if compute_ntv:
            from .diagnostics import u_hat_np  # noqa: PLC0415

            uhat_np = u_hat_np(grids=grids, geom=geom)
            uhat = jnp.asarray(uhat_np, dtype=jnp.float64)
            bh = jnp.asarray(op0.b_hat, dtype=jnp.float64)
            dbt = jnp.asarray(op0.db_hat_dtheta, dtype=jnp.float64)
            dbz = jnp.asarray(op0.db_hat_dzeta, dtype=jnp.float64)
            inv_fsa_b2 = 1.0 / jnp.asarray(op0.fsab_hat2, dtype=jnp.float64)
            ghat = jnp.asarray(float(geom.g_hat), dtype=jnp.float64)
            ihat = jnp.asarray(float(geom.i_hat), dtype=jnp.float64)
            iota = jnp.asarray(float(geom.iota), dtype=jnp.float64)
            ntv_kernel = (2.0 / 5.0) / bh * (
                (uhat - ghat * inv_fsa_b2) * (iota * dbt + dbz)
                + iota * (1.0 / (bh * bh)) * (ghat * dbt - ihat * dbz)
            )
        else:
            ntv_kernel = jnp.zeros_like(jnp.asarray(op0.b_hat, dtype=jnp.float64))

        w2d = jnp.asarray(op0.theta_weights, dtype=jnp.float64)[:, None] * jnp.asarray(op0.zeta_weights, dtype=jnp.float64)[None, :]
        vprime_hat = jnp.sum(w2d / jnp.asarray(op0.d_hat, dtype=jnp.float64))
        x_grid = jnp.asarray(op0.x, dtype=jnp.float64)
        xw = jnp.asarray(op0.x_weights, dtype=jnp.float64)
        w_ntv = xw * (x_grid**4)
        z_s = jnp.asarray(op0.z_s, dtype=jnp.float64)
        t_hat = jnp.asarray(op0.t_hat, dtype=jnp.float64)
        m_hat = jnp.asarray(op0.m_hat, dtype=jnp.float64)
        sqrt_t = jnp.sqrt(t_hat)
        sqrt_m = jnp.sqrt(m_hat)
        b0, _g, _i = _flux_functions_from_op(op0)
        fsab2 = jnp.asarray(op0.fsab_hat2, dtype=jnp.float64)

        w2d_np = np.asarray(w2d, dtype=np.float64)
        b0_val = float(np.asarray(b0, dtype=np.float64))
        fsab2_val = float(np.asarray(fsab2, dtype=np.float64))
        n_hat_np = np.asarray(op0.n_hat, dtype=np.float64)

        def _collect_transport_outputs(which_rhs: int, x_full: jnp.ndarray) -> None:
            """Populate streaming diagnostics for a single whichRHS solve."""
            j = int(which_rhs) - 1
            op_rhs = with_transport_rhs_settings(op0, which_rhs=int(which_rhs))
            d = v3_rhsmode1_output_fields_vm_only_jit(op_rhs, x_full=x_full)
            diag = v3_transport_diagnostics_vm_only(op_rhs, x_full=x_full)

            dens[:, :, :, j] = np.asarray(jnp.transpose(d["densityPerturbation"], (2, 1, 0)), dtype=np.float64)
            pres[:, :, :, j] = np.asarray(jnp.transpose(d["pressurePerturbation"], (2, 1, 0)), dtype=np.float64)
            pres_aniso[:, :, :, j] = np.asarray(jnp.transpose(d["pressureAnisotropy"], (2, 1, 0)), dtype=np.float64)
            flow[:, :, :, j] = np.asarray(jnp.transpose(d["flow"], (2, 1, 0)), dtype=np.float64)
            total_dens[:, :, :, j] = np.asarray(jnp.transpose(d["totalDensity"], (2, 1, 0)), dtype=np.float64)
            total_pres[:, :, :, j] = np.asarray(jnp.transpose(d["totalPressure"], (2, 1, 0)), dtype=np.float64)
            vel_fsadens[:, :, :, j] = np.asarray(jnp.transpose(d["velocityUsingFSADensity"], (2, 1, 0)), dtype=np.float64)
            vel_total[:, :, :, j] = np.asarray(jnp.transpose(d["velocityUsingTotalDensity"], (2, 1, 0)), dtype=np.float64)
            mach[:, :, :, j] = np.asarray(jnp.transpose(d["MachUsingFSAThermalSpeed"], (2, 1, 0)), dtype=np.float64)
            j_hat[:, :, j] = np.asarray(jnp.transpose(d["jHat"], (1, 0)), dtype=np.float64)
            fsa_dens[:, j] = np.asarray(d["FSADensityPerturbation"], dtype=np.float64)
            fsa_pres[:, j] = np.asarray(d["FSAPressurePerturbation"], dtype=np.float64)

            mf_before_vm[:, :, :, j] = np.asarray(
                jnp.transpose(d["momentumFluxBeforeSurfaceIntegral_vm"], (2, 1, 0)), dtype=np.float64
            )
            mf_before_vm0[:, :, :, j] = np.asarray(
                jnp.transpose(d["momentumFluxBeforeSurfaceIntegral_vm0"], (2, 1, 0)), dtype=np.float64
            )
            mf_before_vE[:, :, :, j] = np.asarray(
                jnp.transpose(d["momentumFluxBeforeSurfaceIntegral_vE"], (2, 1, 0)), dtype=np.float64
            )
            mf_before_vE0[:, :, :, j] = np.asarray(
                jnp.transpose(d["momentumFluxBeforeSurfaceIntegral_vE0"], (2, 1, 0)), dtype=np.float64
            )
            mf_vm_psi_hat[:, j] = np.asarray(d["momentumFlux_vm_psiHat"], dtype=np.float64)
            mf_vm0_psi_hat[:, j] = np.asarray(d["momentumFlux_vm0_psiHat"], dtype=np.float64)

            pf_before_vm[:, :, :, j] = np.asarray(
                jnp.transpose(diag.particle_flux_before_surface_integral_vm, (2, 1, 0)), dtype=np.float64
            )
            hf_before_vm[:, :, :, j] = np.asarray(
                jnp.transpose(diag.heat_flux_before_surface_integral_vm, (2, 1, 0)), dtype=np.float64
            )
            pf_before_vm0[:, :, :, j] = np.asarray(
                jnp.transpose(diag.particle_flux_before_surface_integral_vm0, (2, 1, 0)), dtype=np.float64
            )
            hf_before_vm0[:, :, :, j] = np.asarray(
                jnp.transpose(diag.heat_flux_before_surface_integral_vm0, (2, 1, 0)), dtype=np.float64
            )
            pf_vs_x[:, :, j] = np.asarray(diag.particle_flux_vm_psi_hat_vs_x, dtype=np.float64)
            hf_vs_x[:, :, j] = np.asarray(diag.heat_flux_vm_psi_hat_vs_x, dtype=np.float64)
            fsab_flow_vs_x[:, :, j] = np.asarray(diag.fsab_flow_vs_x, dtype=np.float64)

            pf_vm_psi_hat[:, j] = np.asarray(diag.particle_flux_vm_psi_hat, dtype=np.float64)
            hf_vm_psi_hat[:, j] = np.asarray(diag.heat_flux_vm_psi_hat, dtype=np.float64)
            fsab_flow[:, j] = np.asarray(diag.fsab_flow, dtype=np.float64)
            diag_pf_arr[:, j] = pf_vm_psi_hat[:, j]
            diag_hf_arr[:, j] = hf_vm_psi_hat[:, j]
            diag_flow_arr[:, j] = fsab_flow[:, j]

            pf_vm0_psi_hat[:, j] = np.einsum(
                "tz,stz->s",
                w2d_np,
                np.asarray(diag.particle_flux_before_surface_integral_vm0, dtype=np.float64),
            )
            hf_vm0_psi_hat[:, j] = np.einsum(
                "tz,stz->s",
                w2d_np,
                np.asarray(diag.heat_flux_before_surface_integral_vm0, dtype=np.float64),
            )

            if compute_ntv and int(op0.n_xi) > 2:
                f_delta = np.asarray(x_full[: op0.f_size], dtype=np.float64).reshape(op0.fblock.f_shape)
                sum_ntv = np.einsum("x,sxtz->stz", np.asarray(w_ntv, dtype=np.float64), f_delta[:, :, 2, :, :])
                ntv_before_stz = (
                    (4.0 * np.pi * (np.asarray(t_hat) ** 2) * np.asarray(sqrt_t) / (np.asarray(m_hat) * np.asarray(sqrt_m) * float(np.asarray(vprime_hat))))
                )[:, None, None] * np.asarray(ntv_kernel, dtype=np.float64)[None, :, :] * sum_ntv
                ntv_s = np.einsum("tz,stz->s", w2d_np, ntv_before_stz)
            else:
                ntv_before_stz = np.zeros((int(op0.n_species), int(op0.n_theta), int(op0.n_zeta)), dtype=np.float64)
                ntv_s = np.zeros((int(op0.n_species),), dtype=np.float64)
            ntv[:, j] = ntv_s
            ntv_before[:, :, :, j] = np.asarray(np.transpose(ntv_before_stz, (2, 1, 0)), dtype=np.float64)

            if sources is not None:
                extra = np.asarray(x_full[op0.f_size + op0.phi1_size :], dtype=np.float64)
                if int(op0.constraint_scheme) == 2:
                    src = extra.reshape((int(op0.n_species), int(op0.n_x))).T
                else:
                    src = extra.reshape((int(op0.n_species), 2)).T
                sources[:, :, j] = src


    matvec_full_cache: dict[tuple[object, ...], Callable[[jnp.ndarray], jnp.ndarray]] = {}
    matvec_reduced_cache: dict[tuple[object, ...], Callable[[jnp.ndarray], jnp.ndarray]] = {}

    def _get_full_matvec(op_matvec: V3FullSystemOperator) -> Callable[[jnp.ndarray], jnp.ndarray]:
        sig = _operator_signature_cached(op_matvec)
        fn = matvec_full_cache.get(sig)
        if fn is None:
            def mv(x: jnp.ndarray, op=op_matvec) -> jnp.ndarray:
                return apply_v3_full_system_operator_cached(op, x)

            matvec_full_cache[sig] = mv
            fn = mv
        return fn

    def _get_reduced_matvec(op_matvec: V3FullSystemOperator) -> Callable[[jnp.ndarray], jnp.ndarray]:
        if not use_active_dof_mode or reduce_full is None or expand_reduced is None:
            return _get_full_matvec(op_matvec)
        sig = _operator_signature_cached(op_matvec)
        key = (sig, int(active_size))
        fn = matvec_reduced_cache.get(key)
        if fn is None:
            def mv(x_reduced: jnp.ndarray, op=op_matvec) -> jnp.ndarray:
                y_full = apply_v3_full_system_operator_cached(op, expand_reduced(x_reduced))
                return reduce_full(y_full)

            matvec_reduced_cache[key] = mv
            fn = mv
        return fn

    recycle_k_env = os.environ.get("SFINCS_JAX_TRANSPORT_RECYCLE_K", "").strip()
    try:
        recycle_k = int(recycle_k_env) if recycle_k_env else 4
    except ValueError:
        recycle_k = 4
    recycle_k = max(0, recycle_k)
    if recycle_k > 0:
        sig_ref = _operator_signature_cached(op_matvec_by_index[0])
        for op_probe in op_matvec_by_index[1:]:
            if _operator_signature_cached(op_probe) != sig_ref:
                recycle_k = 0
                if emit is not None:
                    emit(1, "solve_v3_transport_matrix_linear_gmres: recycle disabled (matvec operator varies across whichRHS)")
                break

    recycle_basis_full: list[jnp.ndarray] = []
    recycle_basis_full_au: list[jnp.ndarray] = []
    recycle_basis_reduced: list[jnp.ndarray] = []
    recycle_basis_reduced_au: list[jnp.ndarray] = []
    state_recycle_env = os.environ.get("SFINCS_JAX_TRANSPORT_RECYCLE_STATE", "").strip().lower()
    state_recycle_enabled = state_recycle_env not in {"0", "false", "no", "off"}
    if recycle_k > 0 and state_recycle_enabled and state_x_by_rhs:
        mv_ref_full = _get_full_matvec(op_matvec_by_index[0])
        mv_ref_reduced = _get_reduced_matvec(op_matvec_by_index[0])
        for which_rhs in sorted(state_x_by_rhs.keys()):
            x_full = jnp.asarray(state_x_by_rhs[int(which_rhs)])
            if x_full.shape == (op0.total_size,):
                recycle_basis_full.append(x_full)
                recycle_basis_full_au.append(mv_ref_full(x_full))
                if use_active_dof_mode and reduce_full is not None:
                    x_red = reduce_full(x_full)
                    recycle_basis_reduced.append(x_red)
                    recycle_basis_reduced_au.append(mv_ref_reduced(x_red))
            elif use_active_dof_mode and x_full.shape == (active_size,) and reduce_full is not None:
                recycle_basis_reduced.append(x_full)
                recycle_basis_reduced_au.append(mv_ref_reduced(x_full))
        if len(recycle_basis_full) > recycle_k:
            recycle_basis_full = recycle_basis_full[-recycle_k:]
            recycle_basis_full_au = recycle_basis_full_au[-recycle_k:]
        if len(recycle_basis_reduced) > recycle_k:
            recycle_basis_reduced = recycle_basis_reduced[-recycle_k:]
            recycle_basis_reduced_au = recycle_basis_reduced_au[-recycle_k:]

    def _residual_value(res: GMRESSolveResult) -> float:
        val = float(res.residual_norm)
        return val if np.isfinite(val) else float("inf")

    def _needs_retry(res: GMRESSolveResult, target: float) -> bool:
        return (not _gmres_result_is_finite(res)) or (_residual_value(res) > target)

    def _recycled_initial_guess(
        rhs_vec: jnp.ndarray,
        basis: list[jnp.ndarray],
        basis_au: list[jnp.ndarray],
    ) -> jnp.ndarray | None:
        if not basis or not basis_au:
            return None
        u = jnp.stack(basis, axis=1)  # (N, k)
        au = jnp.stack(basis_au, axis=1)  # (N, k)
        coeff, *_ = jnp.linalg.lstsq(au, rhs_vec, rcond=None)
        x0 = u @ coeff
        if not jnp.all(jnp.isfinite(x0)):
            return None
        return x0

    loose_env = os.environ.get("SFINCS_JAX_TRANSPORT_EPAR_LOOSE", "").strip().lower()
    krylov_env = os.environ.get("SFINCS_JAX_TRANSPORT_EPAR_KRYLOV", "").strip().lower()

    def _rhs3_krylov_flags(which_rhs: int) -> tuple[bool, bool]:
        use_loose = (
            loose_env in {"1", "true", "yes", "on"}
            and int(rhs_mode) == 2
            and int(which_rhs) == 3
            and int(op0.constraint_scheme) == 1
        )
        force_k = (
            krylov_env in {"1", "true", "yes", "on"}
            and int(rhs_mode) == 2
            and int(which_rhs) == 3
            and int(op0.constraint_scheme) == 1
        )
        return use_loose, force_k

    project_env = os.environ.get("SFINCS_JAX_TRANSPORT_PROJECT_NULLSPACE", "").strip().lower()
    project_nullspace_enabled = (
        int(op0.constraint_scheme) == 1
        and int(op0.phi1_size) == 0
        and int(op0.extra_size) > 0
        and project_env not in {"0", "false", "no", "off"}
    )

    def _projection_needed(which_rhs: int) -> bool:
        if not project_nullspace_enabled:
            return False
        return (
            (int(rhs_mode) == 2 and int(which_rhs) == 3)
            or (int(rhs_mode) == 3 and int(which_rhs) == 2)
        )

    iter_stats_env = os.environ.get("SFINCS_JAX_SOLVER_ITER_STATS", "").strip().lower()
    iter_stats_enabled = iter_stats_env in {"1", "true", "yes", "on"}
    iter_stats_max_env = os.environ.get("SFINCS_JAX_SOLVER_ITER_STATS_MAX_SIZE", "").strip()
    try:
        iter_stats_max_size = int(iter_stats_max_env) if iter_stats_max_env else None
    except ValueError:
        iter_stats_max_size = None

    def _emit_ksp_iter_stats_transport(
        *,
        which_rhs: int,
        matvec_fn,
        b_vec: jnp.ndarray,
        precond_fn,
        x0_vec: jnp.ndarray | None,
        tol_val: float,
        atol_val: float,
        restart_val: int,
        maxiter_val: int | None,
        precond_side: str,
        solver_kind: str,
    ) -> None:
        if emit is None or not iter_stats_enabled:
            return
        size = int(b_vec.size)
        if iter_stats_max_size is not None and size > int(iter_stats_max_size):
            emit(1, f"whichRHS={which_rhs} ksp_iterations skipped (size={size} > max={int(iter_stats_max_size)})")
            return
        solver_kind_l = str(solver_kind).strip().lower()
        try:
            if solver_kind_l == "gmres":
                _x_hist, _rn, history = gmres_solve_with_history_scipy(
                    matvec=matvec_fn,
                    b=b_vec,
                    preconditioner=precond_fn,
                    x0=x0_vec,
                    tol=tol_val,
                    atol=atol_val,
                    restart=restart_val,
                    maxiter=maxiter_val,
                    precondition_side=precond_side,
                )
                iters = len(history)
            elif solver_kind_l == "bicgstab":
                _x_hist, _rn, history = bicgstab_solve_with_history_scipy(
                    matvec=matvec_fn,
                    b=b_vec,
                    preconditioner=precond_fn,
                    x0=x0_vec,
                    tol=tol_val,
                    atol=atol_val,
                    maxiter=maxiter_val,
                    precondition_side=precond_side,
                )
                iters = len(history)
            else:
                return
        except Exception as exc:  # noqa: BLE001
            emit(1, f"whichRHS={which_rhs} ksp_iterations unavailable ({type(exc).__name__}: {exc})")
            return
        emit(0, f"whichRHS={which_rhs} ksp_iterations={iters} solver={solver_kind_l}")

    def _maybe_project_constraint_nullspace(
        x_vec: jnp.ndarray,
        *,
        which_rhs: int,
        op_matvec: V3FullSystemOperator,
        rhs_vec: jnp.ndarray,
    ) -> jnp.ndarray:
        apply_projection = (
            (int(rhs_mode) == 2 and int(which_rhs) == 3)
            or (int(rhs_mode) == 3 and int(which_rhs) == 2)
        )
        if not apply_projection:
            return x_vec
        return _project_constraint_scheme1_nullspace_solution(
            op=op0,
            x_vec=x_vec,
            rhs_vec=rhs_vec,
            matvec_op=op_matvec,
            enabled_env_var="SFINCS_JAX_TRANSPORT_PROJECT_NULLSPACE",
        )

    dense_batch_done = False
    if str(solve_method_use).lower() == "dense":
        requested_epar_krylov = any((_rhs3_krylov_flags(which_rhs)[0] or _rhs3_krylov_flags(which_rhs)[1]) for which_rhs in which_rhs_values)
        if not requested_epar_krylov:
            op_probe_ref = op_matvec_by_index[0]
            sig_ref = _operator_signature_cached(op_probe_ref)
            same_operator = True
            for op_probe in op_matvec_by_index[1:]:
                if _operator_signature_cached(op_probe) != sig_ref:
                    same_operator = False
                    break

            if same_operator:
                if emit is not None:
                    emit(1, "solve_v3_transport_matrix_linear_gmres: evaluateJacobian called (matrix-free)")
                    emit(1, "solve_v3_transport_matrix_linear_gmres: dense batched solve across all whichRHS")
                t_dense = Timer()

                if use_active_dof_mode:
                    assert reduce_full is not None
                    assert expand_reduced is not None

                    def _mv_dense(x: jnp.ndarray) -> jnp.ndarray:
                        y_full = apply_v3_full_system_operator_cached(op_probe_ref, expand_reduced(x))
                        return reduce_full(y_full)

                    dense_dtype = _dense_dtype(jnp.float64)
                    rhs_mat = jnp.stack([reduce_full(rhs) for rhs in rhs_by_index], axis=1)
                    a_dense = assemble_dense_matrix_from_matvec(
                        matvec=_mv_dense, n=int(active_size), dtype=dense_dtype
                    )
                    rhs_mat = jnp.asarray(rhs_mat, dtype=dense_dtype)
                    x_mat, _ = dense_solve_from_matrix(a=a_dense, b=rhs_mat)
                    if dense_use_mixed:
                        r_mat = rhs_mat - a_dense @ x_mat
                        dx_mat, _ = dense_solve_from_matrix(a=a_dense, b=r_mat)
                        x_mat = x_mat + dx_mat
                    x_mat = jnp.asarray(x_mat, dtype=jnp.float64)
                    res_mat = a_dense @ x_mat - rhs_mat
                    res_norms = jnp.linalg.norm(res_mat, axis=0)
                    for idx, which_rhs in enumerate(which_rhs_values):
                        x_col = expand_reduced(x_mat[:, idx])
                        rhs_vec = rhs_by_index[idx]
                        x_col = _maybe_project_constraint_nullspace(
                            x_col, which_rhs=int(which_rhs), op_matvec=op_probe_ref, rhs_vec=rhs_vec
                        )
                        if store_state_vectors:
                            state_vectors[which_rhs] = x_col
                        if stream_diagnostics:
                            _collect_transport_outputs(int(which_rhs), x_col)
                        residual_norms[which_rhs] = res_norms[idx]
                        elapsed_s[int(which_rhs) - 1] = float(t_dense.elapsed_s() / float(n))
                        if emit is not None:
                            emit(
                                0,
                                f"whichRHS={which_rhs}: residual_norm={float(residual_norms[which_rhs]):.6e} "
                                f"elapsed_s={float(elapsed_s[-1]):.3f}",
                            )
                    dense_batch_done = True
                else:
                    def _mv_dense(x: jnp.ndarray) -> jnp.ndarray:
                        return apply_v3_full_system_operator_cached(op_probe_ref, x)

                    a_dense = assemble_dense_matrix_from_matvec(
                        matvec=_mv_dense, n=int(op0.total_size), dtype=_dense_dtype(jnp.float64)
                    )
                    rhs_mat = jnp.stack(rhs_by_index, axis=1)
                    rhs_mat = jnp.asarray(rhs_mat, dtype=a_dense.dtype)
                    x_mat, _ = dense_solve_from_matrix(a=a_dense, b=rhs_mat)
                    if dense_use_mixed:
                        r_mat = rhs_mat - a_dense @ x_mat
                        dx_mat, _ = dense_solve_from_matrix(a=a_dense, b=r_mat)
                        x_mat = x_mat + dx_mat
                    x_mat = jnp.asarray(x_mat, dtype=jnp.float64)
                    x_cols: list[jnp.ndarray] = []
                    for idx, which_rhs in enumerate(which_rhs_values):
                        x_col = x_mat[:, idx]
                        rhs_vec = rhs_by_index[idx]
                        x_col = _maybe_project_constraint_nullspace(
                            x_col, which_rhs=int(which_rhs), op_matvec=op_probe_ref, rhs_vec=rhs_vec
                        )
                        x_cols.append(x_col)

                    x_mat_proj = jnp.stack(x_cols, axis=1)
                    res_mat = a_dense @ x_mat_proj - rhs_mat
                    res_norms = jnp.linalg.norm(res_mat, axis=0)

                    for idx, which_rhs in enumerate(which_rhs_values):
                        x_col = x_mat_proj[:, idx]
                        if store_state_vectors:
                            state_vectors[which_rhs] = x_col
                        if stream_diagnostics:
                            _collect_transport_outputs(int(which_rhs), x_col)
                        residual_norms[which_rhs] = res_norms[idx]
                        elapsed_s[int(which_rhs) - 1] = float(t_dense.elapsed_s() / float(n))
                        if emit is not None:
                            emit(
                                0,
                                f"whichRHS={which_rhs}: residual_norm={float(residual_norms[which_rhs]):.6e} "
                                f"elapsed_s={float(elapsed_s[-1]):.3f}",
                            )
                    dense_batch_done = True

    if not dense_batch_done:
        for idx, which_rhs in enumerate(which_rhs_values):
            t_rhs = Timer()
            op_rhs = op_rhs_by_index[idx]
            rhs = rhs_by_index[idx]
            op_matvec = op_matvec_by_index[idx]
            if emit is not None:
                emit(0, f"whichRHS={which_rhs}/{n}: assembling+solving (rhs_norm={float(jnp.linalg.norm(rhs)):.6e})")
                emit(1, f"whichRHS={which_rhs}/{n}: evaluateJacobian called (matrix-free)")

            use_loose_epar_krylov, force_epar_krylov = _rhs3_krylov_flags(which_rhs)
            solve_method_rhs = solve_method_use
            tol_rhs = tol
            if force_epar_krylov or use_loose_epar_krylov:
                solve_method_rhs = "incremental"
                if use_loose_epar_krylov:
                    epar_tol_env = os.environ.get("SFINCS_JAX_TRANSPORT_EPAR_TOL", "").strip()
                    try:
                        epar_tol = float(epar_tol_env) if epar_tol_env else 1e-8
                    except ValueError:
                        epar_tol = 1e-8
                    tol_rhs = max(float(tol), float(epar_tol))

            if use_active_dof_mode:
                assert active_idx_jnp is not None
                assert full_to_active_jnp is not None
                assert reduce_full is not None
                assert expand_reduced is not None
                mv_reduced = _get_reduced_matvec(op_matvec)

                rhs_reduced = reduce_full(rhs)
                preconditioner_use = preconditioner_reduced
                if dense_precond_enabled:
                    sig = _operator_signature_cached(op_matvec)
                    preconditioner_use = _dense_preconditioner_for_matvec(
                        matvec_fn=mv_reduced,
                        n=active_size,
                        dtype=_dense_dtype(rhs_reduced.dtype),
                        cache=dense_precond_cache_reduced,
                        key=(sig, int(active_size)),
                    )
                x0_reduced = None
                x0_local = x0_by_rhs.get(int(which_rhs)) if x0_by_rhs else x0
                if x0_local is not None:
                    x0_arr = jnp.asarray(x0_local)
                    if x0_arr.shape == (active_size,):
                        x0_reduced = x0_arr
                    elif x0_arr.shape == (op0.total_size,):
                        x0_reduced = reduce_full(x0_arr)
                if recycle_k > 0:
                    x0_recycled = _recycled_initial_guess(
                        rhs_reduced,
                        recycle_basis_reduced[-recycle_k:],
                        recycle_basis_reduced_au[-recycle_k:],
                    )
                    if x0_reduced is None and x0_recycled is not None:
                        x0_reduced = x0_recycled

                solver_kind_used = _solver_kind(solve_method_rhs)[0]
                solve_method_used = solve_method_rhs
                restart_used = _restart_for_method(solve_method_rhs)
                preconditioner_used = preconditioner_use
                x0_used = x0_reduced
                dense_used = False

                res_reduced = _solve_linear(
                    matvec_fn=mv_reduced,
                    b_vec=rhs_reduced,
                    x0_vec=x0_reduced,
                    tol_val=tol_rhs,
                    atol_val=atol,
                    restart_val=_restart_for_method(solve_method_rhs),
                    maxiter_val=maxiter,
                    solve_method_val=solve_method_rhs,
                    preconditioner_val=preconditioner_use,
                    precondition_side_val="left",
                )
                target_rhs = max(float(atol), float(tol_rhs) * float(jnp.linalg.norm(rhs_reduced)))
                solver_kind = _solver_kind(solve_method_rhs)[0]
                if solver_kind == "bicgstab" and (not _gmres_result_is_finite(res_reduced) or float(res_reduced.residual_norm) > target_rhs):
                    if emit is not None:
                        emit(
                            0,
                            "solve_v3_transport_matrix_linear_gmres: BiCGStab fallback to GMRES "
                            f"(residual={float(res_reduced.residual_norm):.3e} > target={target_rhs:.3e})",
                        )
                    res_reduced = _solve_linear(
                        matvec_fn=mv_reduced,
                        b_vec=rhs_reduced,
                        x0_vec=x0_reduced,
                        tol_val=tol_rhs,
                        atol_val=atol,
                        restart_val=gmres_restart,
                        maxiter_val=maxiter,
                        solve_method_val="incremental",
                        preconditioner_val=preconditioner_use,
                        precondition_side_val="left",
                    )
                    solver_kind_used = "gmres"
                    solve_method_used = "incremental"
                    restart_used = gmres_restart
                if _needs_retry(res_reduced, target_rhs) and preconditioner_use is not None:
                    if emit is not None:
                        emit(
                            0,
                            "solve_v3_transport_matrix_linear_gmres: retry without preconditioner "
                            f"(residual={float(res_reduced.residual_norm):.3e} > target={target_rhs:.3e})",
                        )
                    res_retry = _solve_linear(
                        matvec_fn=mv_reduced,
                        b_vec=rhs_reduced,
                        x0_vec=x0_reduced,
                        tol_val=tol_rhs,
                        atol_val=atol,
                        restart_val=_restart_for_method(solve_method_rhs),
                        maxiter_val=maxiter,
                        solve_method_val=solve_method_rhs,
                        preconditioner_val=None,
                        precondition_side_val="left",
                    )
                    if _residual_value(res_retry) < _residual_value(res_reduced):
                        res_reduced = res_retry
                        preconditioner_use = None
                        preconditioner_used = None
                if _needs_retry(res_reduced, target_rhs):
                    strong_precond = _get_strong_preconditioner(True)
                    if strong_precond is not None and strong_precond is not preconditioner_use:
                        if emit is not None:
                            emit(
                                0,
                                "solve_v3_transport_matrix_linear_gmres: retry with strong preconditioner "
                                f"(residual={float(res_reduced.residual_norm):.3e} > target={target_rhs:.3e})",
                            )
                        res_strong = _solve_linear(
                            matvec_fn=mv_reduced,
                            b_vec=rhs_reduced,
                            x0_vec=res_reduced.x,
                            tol_val=tol_rhs,
                            atol_val=atol,
                            restart_val=gmres_restart,
                            maxiter_val=maxiter,
                            solve_method_val="incremental",
                            preconditioner_val=strong_precond,
                            precondition_side_val="left",
                        )
                        if _residual_value(res_strong) < _residual_value(res_reduced):
                            res_reduced = res_strong
                            preconditioner_use = strong_precond
                            preconditioner_used = strong_precond
                            solver_kind_used = "gmres"
                            solve_method_used = "incremental"
                            restart_used = gmres_restart
                if _needs_retry(res_reduced, target_rhs) and dense_retry_max > 0 and int(active_size) <= int(dense_retry_max):
                    if emit is not None:
                        emit(
                            0,
                            "solve_v3_transport_matrix_linear_gmres: dense fallback "
                            f"(size={int(active_size)} residual={float(res_reduced.residual_norm):.3e} > target={target_rhs:.3e})",
                        )
                    try:
                        sig = _operator_signature_cached(op_matvec)
                        dense_solver = _dense_solver_for_matvec(
                            matvec_fn=mv_reduced,
                            n=int(active_size),
                            dtype=_dense_dtype(rhs_reduced.dtype),
                            cache=dense_solver_cache_reduced,
                            key=(sig, int(active_size), str(_dense_dtype(rhs_reduced.dtype))),
                        )
                        rhs_dense = jnp.asarray(rhs_reduced, dtype=_dense_dtype(rhs_reduced.dtype))
                        x_dense = dense_solver(rhs_dense)
                        if dense_use_mixed:
                            r_dense0 = rhs_reduced - mv_reduced(jnp.asarray(x_dense, dtype=rhs_reduced.dtype))
                            dx = dense_solver(jnp.asarray(r_dense0, dtype=_dense_dtype(rhs_reduced.dtype)))
                            x_dense = jnp.asarray(x_dense, dtype=rhs_reduced.dtype) + jnp.asarray(dx, dtype=rhs_reduced.dtype)
                        r_dense = rhs_reduced - mv_reduced(x_dense)
                        res_dense = GMRESSolveResult(x=x_dense, residual_norm=jnp.linalg.norm(r_dense))
                        if _residual_value(res_dense) < _residual_value(res_reduced):
                            res_reduced = res_dense
                            dense_used = True
                            solver_kind_used = "dense"
                            solve_method_used = "dense"
                    except Exception as exc:  # noqa: BLE001
                        if emit is not None:
                            emit(
                                1,
                                "solve_v3_transport_matrix_linear_gmres: dense fallback failed "
                                f"({type(exc).__name__}: {exc})",
                            )
                x_full = expand_reduced(res_reduced.x)
                x_full = _maybe_project_constraint_nullspace(
                    x_full, which_rhs=int(which_rhs), op_matvec=op_matvec, rhs_vec=rhs
                )
                ax_full = apply_v3_full_system_operator_cached(op_matvec, x_full)
                res_norm_full = jnp.linalg.norm(ax_full - rhs)
                if (not dense_used) and dense_retry_max > 0 and int(active_size) <= int(dense_retry_max):
                    target_full = max(float(atol), float(tol_rhs) * float(jnp.linalg.norm(rhs)))
                    if float(res_norm_full) > target_full:
                        if emit is not None:
                            emit(
                                0,
                                "solve_v3_transport_matrix_linear_gmres: dense fallback (true residual) "
                                f"(size={int(active_size)} residual={float(res_norm_full):.3e} > target={target_full:.3e})",
                            )
                        try:
                            sig = _operator_signature_cached(op_matvec)
                            dense_solver = _dense_solver_for_matvec(
                                matvec_fn=mv_reduced,
                                n=int(active_size),
                                dtype=_dense_dtype(rhs_reduced.dtype),
                                cache=dense_solver_cache_reduced,
                                key=(sig, int(active_size), str(_dense_dtype(rhs_reduced.dtype))),
                            )
                            rhs_dense = jnp.asarray(rhs_reduced, dtype=_dense_dtype(rhs_reduced.dtype))
                            x_dense = dense_solver(rhs_dense)
                            if dense_use_mixed:
                                r_dense0 = rhs_reduced - mv_reduced(jnp.asarray(x_dense, dtype=rhs_reduced.dtype))
                                dx = dense_solver(jnp.asarray(r_dense0, dtype=_dense_dtype(rhs_reduced.dtype)))
                                x_dense = jnp.asarray(x_dense, dtype=rhs_reduced.dtype) + jnp.asarray(dx, dtype=rhs_reduced.dtype)
                            x_full_dense = expand_reduced(x_dense)
                            x_full_dense = _maybe_project_constraint_nullspace(
                                x_full_dense, which_rhs=int(which_rhs), op_matvec=op_matvec, rhs_vec=rhs
                            )
                            ax_dense = apply_v3_full_system_operator_cached(op_matvec, x_full_dense)
                            res_dense_norm = jnp.linalg.norm(ax_dense - rhs)
                            if float(res_dense_norm) < float(res_norm_full):
                                x_full = x_full_dense
                                ax_full = ax_dense
                                res_norm_full = res_dense_norm
                                dense_used = True
                                solver_kind_used = "dense"
                                solve_method_used = "dense"
                        except Exception as exc:  # noqa: BLE001
                            if emit is not None:
                                emit(
                                    1,
                                    "solve_v3_transport_matrix_linear_gmres: dense fallback failed "
                                    f"({type(exc).__name__}: {exc})",
                                )
                if store_state_vectors:
                    state_vectors[which_rhs] = x_full
                residual_norms[which_rhs] = res_norm_full
                if stream_diagnostics:
                    _collect_transport_outputs(int(which_rhs), x_full)
                if recycle_k > 0:
                    recycle_basis_reduced.append(res_reduced.x)
                    recycle_basis_reduced_au.append(reduce_full(ax_full))
                    if len(recycle_basis_reduced) > recycle_k:
                        recycle_basis_reduced = recycle_basis_reduced[-recycle_k:]
                        recycle_basis_reduced_au = recycle_basis_reduced_au[-recycle_k:]
                    recycle_basis_full.append(x_full)
                    recycle_basis_full_au.append(ax_full)
                    if len(recycle_basis_full) > recycle_k:
                        recycle_basis_full = recycle_basis_full[-recycle_k:]
                        recycle_basis_full_au = recycle_basis_full_au[-recycle_k:]
                if not dense_used:
                    _emit_ksp_iter_stats_transport(
                        which_rhs=int(which_rhs),
                        matvec_fn=mv_reduced,
                        b_vec=rhs_reduced,
                        precond_fn=preconditioner_used,
                        x0_vec=x0_used,
                        tol_val=float(tol_rhs),
                        atol_val=float(atol),
                        restart_val=int(restart_used),
                        maxiter_val=maxiter,
                        precond_side="left",
                        solver_kind=solver_kind_used,
                    )
            else:
                mv = _get_full_matvec(op_matvec)

                preconditioner_use = preconditioner_full
                if dense_precond_enabled:
                    sig = _operator_signature_cached(op_matvec)
                    preconditioner_use = _dense_preconditioner_for_matvec(
                        matvec_fn=mv,
                        n=int(op0.total_size),
                        dtype=_dense_dtype(rhs.dtype),
                        cache=dense_precond_cache_full,
                        key=(sig, int(op0.total_size)),
                    )
                x0_full = x0_by_rhs.get(int(which_rhs)) if x0_by_rhs else x0
                if recycle_k > 0:
                    x0_recycled = _recycled_initial_guess(
                        rhs,
                        recycle_basis_full[-recycle_k:],
                        recycle_basis_full_au[-recycle_k:],
                    )
                    if x0_full is None and x0_recycled is not None:
                        x0_full = x0_recycled

                solver_kind_used = _solver_kind(solve_method_rhs)[0]
                solve_method_used = solve_method_rhs
                restart_used = _restart_for_method(solve_method_rhs)
                preconditioner_used = preconditioner_use
                x0_used = x0_full
                dense_used = False
                res, residual_vec = _solve_linear_with_residual(
                    matvec_fn=mv,
                    b_vec=rhs,
                    x0_vec=x0_full,
                    tol_val=tol_rhs,
                    atol_val=atol,
                    restart_val=_restart_for_method(solve_method_rhs),
                    maxiter_val=maxiter,
                    solve_method_val=solve_method_rhs,
                    preconditioner_val=preconditioner_use,
                    precondition_side_val="left",
                )
                target_rhs = max(float(atol), float(tol_rhs) * float(jnp.linalg.norm(rhs)))
                solver_kind = _solver_kind(solve_method_rhs)[0]
                if solver_kind == "bicgstab" and (not _gmres_result_is_finite(res) or float(res.residual_norm) > target_rhs):
                    if emit is not None:
                        emit(
                            0,
                            "solve_v3_transport_matrix_linear_gmres: BiCGStab fallback to GMRES "
                            f"(residual={float(res.residual_norm):.3e} > target={target_rhs:.3e})",
                        )
                    res, residual_vec = _solve_linear_with_residual(
                        matvec_fn=mv,
                        b_vec=rhs,
                        x0_vec=x0_full,
                        tol_val=tol_rhs,
                        atol_val=atol,
                        restart_val=gmres_restart,
                        maxiter_val=maxiter,
                        solve_method_val="incremental",
                        preconditioner_val=preconditioner_use,
                        precondition_side_val="left",
                    )
                    solver_kind_used = "gmres"
                    solve_method_used = "incremental"
                    restart_used = gmres_restart
                if _needs_retry(res, target_rhs) and preconditioner_use is not None:
                    if emit is not None:
                        emit(
                            0,
                            "solve_v3_transport_matrix_linear_gmres: retry without preconditioner "
                            f"(residual={float(res.residual_norm):.3e} > target={target_rhs:.3e})",
                        )
                    res_retry, residual_retry = _solve_linear_with_residual(
                        matvec_fn=mv,
                        b_vec=rhs,
                        x0_vec=x0_full,
                        tol_val=tol_rhs,
                        atol_val=atol,
                        restart_val=_restart_for_method(solve_method_rhs),
                        maxiter_val=maxiter,
                        solve_method_val=solve_method_rhs,
                        preconditioner_val=None,
                        precondition_side_val="left",
                    )
                    if _residual_value(res_retry) < _residual_value(res):
                        res = res_retry
                        residual_vec = residual_retry
                        preconditioner_use = None
                        preconditioner_used = None
                if _needs_retry(res, target_rhs):
                    strong_precond = _get_strong_preconditioner(False)
                    if strong_precond is not None and strong_precond is not preconditioner_use:
                        if emit is not None:
                            emit(
                                0,
                                "solve_v3_transport_matrix_linear_gmres: retry with strong preconditioner "
                                f"(residual={float(res.residual_norm):.3e} > target={target_rhs:.3e})",
                            )
                        res_strong, residual_vec_strong = _solve_linear_with_residual(
                            matvec_fn=mv,
                            b_vec=rhs,
                            x0_vec=res.x,
                            tol_val=tol_rhs,
                            atol_val=atol,
                            restart_val=gmres_restart,
                            maxiter_val=maxiter,
                            solve_method_val="incremental",
                            preconditioner_val=strong_precond,
                            precondition_side_val="left",
                        )
                        if _residual_value(res_strong) < _residual_value(res):
                            res = res_strong
                            residual_vec = residual_vec_strong
                            preconditioner_use = strong_precond
                            preconditioner_used = strong_precond
                            solver_kind_used = "gmres"
                            solve_method_used = "incremental"
                            restart_used = gmres_restart
                if _needs_retry(res, target_rhs) and dense_retry_max > 0 and int(op0.total_size) <= int(dense_retry_max):
                    if emit is not None:
                        emit(
                            0,
                            "solve_v3_transport_matrix_linear_gmres: dense fallback "
                            f"(size={int(op0.total_size)} residual={float(res.residual_norm):.3e} > target={target_rhs:.3e})",
                        )
                    try:
                        sig = _operator_signature_cached(op_matvec)
                        dense_solver = _dense_solver_for_matvec(
                            matvec_fn=mv,
                            n=int(op0.total_size),
                            dtype=_dense_dtype(rhs.dtype),
                            cache=dense_solver_cache_full,
                            key=(sig, int(op0.total_size), str(_dense_dtype(rhs.dtype))),
                        )
                        rhs_dense = jnp.asarray(rhs, dtype=_dense_dtype(rhs.dtype))
                        x_dense = dense_solver(rhs_dense)
                        if dense_use_mixed:
                            r_dense0 = rhs - mv(jnp.asarray(x_dense, dtype=rhs.dtype))
                            dx = dense_solver(jnp.asarray(r_dense0, dtype=_dense_dtype(rhs.dtype)))
                            x_dense = jnp.asarray(x_dense, dtype=rhs.dtype) + jnp.asarray(dx, dtype=rhs.dtype)
                        residual_dense = rhs - mv(x_dense)
                        res_dense = GMRESSolveResult(x=x_dense, residual_norm=jnp.linalg.norm(residual_dense))
                        if _residual_value(res_dense) < _residual_value(res):
                            res = res_dense
                            residual_vec = residual_dense
                            dense_used = True
                            solver_kind_used = "dense"
                            solve_method_used = "dense"
                    except Exception as exc:  # noqa: BLE001
                        if emit is not None:
                            emit(
                                1,
                                "solve_v3_transport_matrix_linear_gmres: dense fallback failed "
                                f"({type(exc).__name__}: {exc})",
                            )
                x_full = res.x
                projection_needed = _projection_needed(which_rhs)
                if projection_needed:
                    x_full = _maybe_project_constraint_nullspace(
                        x_full, which_rhs=int(which_rhs), op_matvec=op_matvec, rhs_vec=rhs
                    )
                if store_state_vectors:
                    state_vectors[which_rhs] = x_full
                if (not projection_needed) and residual_vec is not None and residual_vec.shape == rhs.shape:
                    ax_full = rhs - residual_vec
                    residual_norms[which_rhs] = res.residual_norm
                else:
                    ax_full = apply_v3_full_system_operator_cached(op_matvec, x_full)
                    residual_vec = ax_full - rhs
                    residual_norms[which_rhs] = jnp.linalg.norm(residual_vec)
                if stream_diagnostics:
                    _collect_transport_outputs(int(which_rhs), x_full)
                if recycle_k > 0:
                    recycle_basis_full.append(x_full)
                    recycle_basis_full_au.append(ax_full)
                    if len(recycle_basis_full) > recycle_k:
                        recycle_basis_full = recycle_basis_full[-recycle_k:]
                        recycle_basis_full_au = recycle_basis_full_au[-recycle_k:]
                if not dense_used:
                    _emit_ksp_iter_stats_transport(
                        which_rhs=int(which_rhs),
                        matvec_fn=mv,
                        b_vec=rhs,
                        precond_fn=preconditioner_used,
                        x0_vec=x0_used,
                        tol_val=float(tol_rhs),
                        atol_val=float(atol),
                        restart_val=int(restart_used),
                        maxiter_val=maxiter,
                        precond_side="left",
                        solver_kind=solver_kind_used,
                    )
            if emit is not None:
                emit(
                    0,
                    f"whichRHS={which_rhs}: residual_norm={float(residual_norms[which_rhs]):.6e} "
                    f"elapsed_s={t_rhs.elapsed_s():.3f}",
                )
            elapsed_s[int(which_rhs) - 1] = float(t_rhs.elapsed_s())

    if emit is not None:
        emit(0, "solve_v3_transport_matrix_linear_gmres: computing whichRHS diagnostics (batched)")
    if stream_diagnostics:
        assert diag_pf_arr is not None
        assert diag_hf_arr is not None
        assert diag_flow_arr is not None
        diag_pf_jnp = jnp.asarray(diag_pf_arr, dtype=jnp.float64)
        diag_hf_jnp = jnp.asarray(diag_hf_arr, dtype=jnp.float64)
        diag_flow_jnp = jnp.asarray(diag_flow_arr, dtype=jnp.float64)

        z_s_np = np.asarray(op0.z_s, dtype=np.float64)
        n_hat_np = np.asarray(op0.n_hat, dtype=np.float64)
        fsab_jhat = np.einsum("s,sn->n", z_s_np, fsab_flow)
        b0_val = float(b0)
        fsab2_val = float(fsab2)
        transport_output_fields = {
            "densityPerturbation": dens,
            "pressurePerturbation": pres,
            "pressureAnisotropy": pres_aniso,
            "flow": flow,
            "totalDensity": total_dens,
            "totalPressure": total_pres,
            "velocityUsingFSADensity": vel_fsadens,
            "velocityUsingTotalDensity": vel_total,
            "MachUsingFSAThermalSpeed": mach,
            "jHat": j_hat,
            "FSADensityPerturbation": fsa_dens,
            "FSAPressurePerturbation": fsa_pres,
            "momentumFluxBeforeSurfaceIntegral_vm": mf_before_vm,
            "momentumFluxBeforeSurfaceIntegral_vm0": mf_before_vm0,
            "momentumFluxBeforeSurfaceIntegral_vE": mf_before_vE,
            "momentumFluxBeforeSurfaceIntegral_vE0": mf_before_vE0,
            "momentumFlux_vm_psiHat": mf_vm_psi_hat,
            "momentumFlux_vm0_psiHat": mf_vm0_psi_hat,
            "NTVBeforeSurfaceIntegral": ntv_before,
            "NTV": ntv,
            "FSABFlow": fsab_flow,
            "FSABFlow_vs_x": fsab_flow_vs_x,
            "FSABVelocityUsingFSADensity": fsab_flow / n_hat_np[:, None],
            "FSABVelocityUsingFSADensityOverB0": (fsab_flow / n_hat_np[:, None]) / b0_val,
            "FSABVelocityUsingFSADensityOverRootFSAB2": (fsab_flow / n_hat_np[:, None]) / np.sqrt(fsab2_val),
            "FSABjHat": fsab_jhat,
            "FSABjHatOverB0": fsab_jhat / b0_val,
            "FSABjHatOverRootFSAB2": fsab_jhat / np.sqrt(fsab2_val),
            "particleFlux_vm_psiHat": pf_vm_psi_hat,
            "heatFlux_vm_psiHat": hf_vm_psi_hat,
            "particleFlux_vm0_psiHat": pf_vm0_psi_hat,
            "heatFlux_vm0_psiHat": hf_vm0_psi_hat,
            "particleFluxBeforeSurfaceIntegral_vm": pf_before_vm,
            "heatFluxBeforeSurfaceIntegral_vm": hf_before_vm,
            "particleFluxBeforeSurfaceIntegral_vm0": pf_before_vm0,
            "heatFluxBeforeSurfaceIntegral_vm0": hf_before_vm0,
            "particleFluxBeforeSurfaceIntegral_vE": pf_before_ve,
            "heatFluxBeforeSurfaceIntegral_vE": hf_before_ve,
            "particleFluxBeforeSurfaceIntegral_vE0": pf_before_ve0,
            "heatFluxBeforeSurfaceIntegral_vE0": hf_before_ve0,
            "particleFlux_vm_psiHat_vs_x": pf_vs_x,
            "heatFlux_vm_psiHat_vs_x": hf_vs_x,
        }
        if sources is not None:
            transport_output_fields["sources"] = sources
    else:
        remat_env = os.environ.get("SFINCS_JAX_REMAT_TRANSPORT_DIAGNOSTICS", "").strip().lower()
        if remat_env in {"1", "true", "yes", "on"}:
            use_remat_diag = True
        elif remat_env in {"0", "false", "no", "off"}:
            use_remat_diag = False
        else:
            remat_min_env = os.environ.get("SFINCS_JAX_REMAT_TRANSPORT_DIAGNOSTICS_MIN", "").strip()
            try:
                remat_min = int(remat_min_env) if remat_min_env else 20000
            except ValueError:
                remat_min = 20000
            use_remat_diag = int(op0.total_size) * int(n) >= remat_min
        diag_chunk_env = os.environ.get("SFINCS_JAX_TRANSPORT_DIAG_CHUNK", "").strip()
        try:
            diag_chunk = int(diag_chunk_env) if diag_chunk_env else None
        except ValueError:
            diag_chunk = None
        if diag_chunk is None or int(diag_chunk) <= 0:
            diag_chunk = 0
        if diag_chunk == 0 and int(op0.total_size) * int(n) >= 200_000:
            diag_chunk = 4

        if use_diag_op0:
            precompute_env = os.environ.get("SFINCS_JAX_TRANSPORT_DIAG_PRECOMPUTE", "").strip().lower()
            use_precompute = precompute_env not in {"0", "false", "no", "off"}
            if use_precompute:
                precomputed = v3_transport_diagnostics_vm_only_precompute(op0)
                diag_fn = (
                    v3_transport_diagnostics_vm_only_batch_op0_precomputed_remat_jit
                    if use_remat_diag
                    else v3_transport_diagnostics_vm_only_batch_op0_precomputed_jit
                )
            else:
                diag_fn = (
                    v3_transport_diagnostics_vm_only_batch_op0_remat_jit
                    if use_remat_diag
                    else v3_transport_diagnostics_vm_only_batch_op0_jit
                )
        else:
            diag_op_stack = jtu.tree_map(lambda *xs: jnp.stack(xs, axis=0), *diag_op_by_index)
            diag_fn = (
                v3_transport_diagnostics_vm_only_batch_remat_jit
                if use_remat_diag
                else v3_transport_diagnostics_vm_only_batch_jit
            )

        if diag_chunk <= 0 or int(diag_chunk) >= int(n):
            x_stack = jnp.stack([state_vectors[which_rhs] for which_rhs in which_rhs_values], axis=0)  # (N,total)
            if use_diag_op0:
                if use_precompute:
                    diag_stack = diag_fn(op0=op0, precomputed=precomputed, x_full_stack=x_stack)
                else:
                    diag_stack = diag_fn(op0=op0, x_full_stack=x_stack)
            else:
                diag_stack = diag_fn(op_stack=diag_op_stack, x_full_stack=x_stack)
            diag_pf_jnp = jnp.transpose(diag_stack.particle_flux_vm_psi_hat, (1, 0))  # (S,N)
            diag_hf_jnp = jnp.transpose(diag_stack.heat_flux_vm_psi_hat, (1, 0))  # (S,N)
            diag_flow_jnp = jnp.transpose(diag_stack.fsab_flow, (1, 0))  # (S,N)
        else:
            s = int(op0.n_species)
            diag_pf_arr = np.zeros((s, n), dtype=np.float64)
            diag_hf_arr = np.zeros((s, n), dtype=np.float64)
            diag_flow_arr = np.zeros((s, n), dtype=np.float64)
            for start in range(0, n, int(diag_chunk)):
                end = min(n, start + int(diag_chunk))
                rhs_chunk = which_rhs_values[start:end]
                x_stack_chunk = jnp.stack([state_vectors[which_rhs] for which_rhs in rhs_chunk], axis=0)
                if use_diag_op0:
                    if use_precompute:
                        diag_stack = diag_fn(op0=op0, precomputed=precomputed, x_full_stack=x_stack_chunk)
                    else:
                        diag_stack = diag_fn(op0=op0, x_full_stack=x_stack_chunk)
                else:
                    op_chunk = jtu.tree_map(lambda arr: arr[start:end], diag_op_stack)
                    diag_stack = diag_fn(op_stack=op_chunk, x_full_stack=x_stack_chunk)
                diag_pf_arr[:, start:end] = np.asarray(jnp.transpose(diag_stack.particle_flux_vm_psi_hat, (1, 0)))
                diag_hf_arr[:, start:end] = np.asarray(jnp.transpose(diag_stack.heat_flux_vm_psi_hat, (1, 0)))
                diag_flow_arr[:, start:end] = np.asarray(jnp.transpose(diag_stack.fsab_flow, (1, 0)))
            diag_pf_jnp = jnp.asarray(diag_pf_arr, dtype=jnp.float64)
            diag_hf_jnp = jnp.asarray(diag_hf_arr, dtype=jnp.float64)
            diag_flow_jnp = jnp.asarray(diag_flow_arr, dtype=jnp.float64)

    tm = v3_transport_matrix_from_flux_arrays(
        op=op0,
        geom=geom,
        particle_flux_vm_psi_hat=diag_pf_jnp,
        heat_flux_vm_psi_hat=diag_hf_jnp,
        fsab_flow=diag_flow_jnp,
    )
    if state_out_env:
        try:
            from .solver_state import save_krylov_state  # noqa: PLC0415

            save_krylov_state(path=state_out_env, op=op0, x_by_rhs=state_vectors)
        except Exception:
            if emit is not None:
                emit(1, f"solve_v3_transport_matrix_linear_gmres: failed to write state {state_out_env}")
    if emit is not None:
        emit(0, "solve_v3_transport_matrix_linear_gmres: done")
        emit(1, f"solve_v3_transport_matrix_linear_gmres: elapsed_s={t_all.elapsed_s():.3f}")
    return V3TransportMatrixSolveResult(
        op0=op0,
        transport_matrix=tm,
        state_vectors_by_rhs=state_vectors,
        residual_norms_by_rhs=residual_norms,
        fsab_flow=diag_flow_jnp,
        particle_flux_vm_psi_hat=diag_pf_jnp,
        heat_flux_vm_psi_hat=diag_hf_jnp,
        elapsed_time_s=jnp.asarray(elapsed_s, dtype=jnp.float64),
        transport_output_fields=transport_output_fields,
    )
