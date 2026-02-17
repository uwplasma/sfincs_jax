from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib

from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)

from collections.abc import Callable
import os
import numpy as np

import jax
import jax.numpy as jnp
from jax import tree_util as jtu

from .namelist import Namelist
from .solver import (
    GMRESSolveResult,
    assemble_dense_matrix_from_matvec,
    bicgstab_solve_with_residual,
    bicgstab_solve_with_residual_jit,
    bicgstab_solve_with_history_scipy,
    dense_solve_from_matrix,
    gmres_solve,
    gmres_solve_jit,
    gmres_solve_with_residual,
    gmres_solve_with_residual_jit,
    gmres_solve_with_history_scipy,
)
from .implicit_solve import linear_custom_solve, linear_custom_solve_with_residual
from .transport_matrix import (
    transport_matrix_size_from_rhs_mode,
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


def _use_solver_jit() -> bool:
    env = os.environ.get("SFINCS_JAX_SOLVER_JIT", "").strip().lower()
    return env not in {"0", "false", "no", "off"}


_PRECOND_SIZE_HINT: int | None = None


def _set_precond_size_hint(n: int | None) -> None:
    global _PRECOND_SIZE_HINT
    if n is None:
        _PRECOND_SIZE_HINT = None
    else:
        _PRECOND_SIZE_HINT = int(n)


def _precond_dtype() -> jnp.dtype:
    env = os.environ.get("SFINCS_JAX_PRECOND_DTYPE", "").strip().lower()
    if env in {"", "auto", "mixed"}:
        size_hint = _PRECOND_SIZE_HINT or 0
        thresh_env = os.environ.get("SFINCS_JAX_PRECOND_FP32_MIN_SIZE", "").strip()
        try:
            thresh = int(thresh_env) if thresh_env else 20000
        except ValueError:
            thresh = 20000
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


def _rhsmode1_precond_cache_key(op: V3FullSystemOperator, kind: str) -> tuple[object, ...]:
    nxi_for_x = np.asarray(op.fblock.collisionless.n_xi_for_x, dtype=np.int32)
    precond_dtype = str(_precond_dtype())
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
        float(op.e_parallel_hat),
        _hash_array(op.adiabatic_z),
        _hash_array(op.adiabatic_nhat),
        _hash_array(op.adiabatic_that),
        _hash_array(op.z_s),
        _hash_array(op.m_hat),
        _hash_array(op.t_hat),
        _hash_array(op.n_hat),
        _hash_array(op.dn_hat_dpsi_hat),
        _hash_array(op.dt_hat_dpsi_hat),
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
    try:
        low_rank_k = int(low_rank_env) if low_rank_env else 0
    except ValueError:
        low_rank_k = 0
    low_rank_k = max(0, low_rank_k)

    precond_dtype = _precond_dtype()
    if low_rank_k > 0:
        f_shape = op.fblock.f_shape
        n_species, n_x, n_l, _, _ = f_shape
        n_block = n_species * n_x
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

    if use_sxblock and op.fblock.fp is not None:
        low_rank_env = os.environ.get("SFINCS_JAX_RHSMODE1_FP_LOW_RANK_K", "").strip()
        if not low_rank_env:
            low_rank_env = os.environ.get("SFINCS_JAX_FP_LOW_RANK_K", "").strip()
        try:
            low_rank_k = int(low_rank_env) if low_rank_env else 0
        except ValueError:
            low_rank_k = 0
        low_rank_k = max(0, low_rank_k)

        if low_rank_k > 0:
            f_shape = op.fblock.f_shape
            n_species, n_x, n_l, _, _ = f_shape
            n_block = n_species * n_x
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


def _build_rhsmode1_block_preconditioner(
    *,
    op: V3FullSystemOperator,
    reduce_full: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    expand_reduced: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Build a PETSc-like block preconditioner for RHSMode=1 solves.

    Structure:
    - x/L local block solve per species at each (theta,zeta), using a representative
      per-species block matrix from a simplified operator.
    - explicit extra/source-row solve via a dense small block.
    """
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
        _build_rhsmode1_block_preconditioner(op=op)
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
        schur_full_max = int(schur_full_max_env) if schur_full_max_env else 128
    except ValueError:
        schur_full_max = 128
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
) -> V3LinearSolveResult:
    """Solve the current v3 full-system linear problem `A x = rhs` matrix-free using GMRES.

    Notes
    -----
    This helper currently targets the linear runs exercised in the parity fixtures
    (e.g. includePhi1InKineticEquation=false). For nonlinear runs, use `residual_v3_full_system`
    and an outer Newton-Krylov iteration (not yet shipped as a stable API).
    """
    t = Timer()
    if emit is not None:
        emit(1, "solve_v3_full_system_linear_gmres: building operator")
    op = full_system_operator_from_namelist(nml=nml, identity_shift=identity_shift, phi1_hat_base=phi1_hat_base)
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
    rhs_norm = jnp.linalg.norm(rhs)
    if emit is not None:
        emit(2, f"solve_v3_full_system_linear_gmres: rhs_norm={float(rhs_norm):.6e}")

    def mv(x):
        # Use the JIT-compiled operator application to reduce Python overhead in repeated matvecs
        # (e.g. during GMRES iterations and Er scans).
        return apply_v3_full_system_operator_cached(op, x)

    active_env = os.environ.get("SFINCS_JAX_ACTIVE_DOF", "").strip().lower()
    nxi_for_x = np.asarray(op.fblock.collisionless.n_xi_for_x, dtype=np.int32)
    has_reduced_modes = bool(np.any(nxi_for_x < int(op.n_xi)))
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

    pas_project_env = os.environ.get("SFINCS_JAX_PAS_PROJECT_CONSTRAINTS", "").strip().lower()
    if pas_project_env in {"1", "true", "yes", "on"}:
        pas_project_mode = "on"
    elif pas_project_env in {"0", "false", "no", "off"}:
        pas_project_mode = "off"
    elif pas_project_env in {"", "auto"}:
        pas_project_mode = "auto"
    else:
        pas_project_mode = "off"
    pas_project_enabled = bool(pas_project_mode == "on" or (pas_project_mode == "auto" and int(op.n_zeta) == 1))
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

    if emit is not None:
        emit(1, f"solve_v3_full_system_linear_gmres: GMRES tol={tol} atol={atol} restart={restart} maxiter={maxiter} solve_method={solve_method}")
        emit(1, "solve_v3_full_system_linear_gmres: evaluateJacobian called (matrix-free)")
    rhs1_precond_env = os.environ.get("SFINCS_JAX_RHSMODE1_PRECONDITIONER", "").strip().lower()
    rhs1_bicgstab_env = os.environ.get("SFINCS_JAX_RHSMODE1_BICGSTAB_PRECOND", "").strip().lower()
    precond_opts = nml.group("preconditionerOptions")
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
    rhs1_precond_enabled = (
        rhs1_precond_kind is not None
        and int(op.rhs_mode) == 1
        and (not bool(op.include_phi1))
    )
    if rhs1_bicgstab_env in {"0", "false", "no", "off"}:
        rhs1_bicgstab_kind = None
    elif rhs1_bicgstab_env in {"", "1", "true", "yes", "on", "collision", "diag"}:
        rhs1_bicgstab_kind = "collision"
    else:
        rhs1_bicgstab_kind = None
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
    iter_stats_env = os.environ.get("SFINCS_JAX_SOLVER_ITER_STATS", "").strip().lower()
    iter_stats_enabled = iter_stats_env in {"1", "true", "yes", "on"}
    iter_stats_max_env = os.environ.get("SFINCS_JAX_SOLVER_ITER_STATS_MAX_SIZE", "").strip()
    try:
        iter_stats_max_size = int(iter_stats_max_env) if iter_stats_max_env else None
    except ValueError:
        iter_stats_max_size = None
    ksp_history_max_env = os.environ.get("SFINCS_JAX_KSP_HISTORY_MAX_SIZE", "").strip().lower()
    if ksp_history_max_env in {"none", "inf", "infinite", "unlimited"}:
        ksp_history_max_size = None
    else:
        try:
            ksp_history_max_size = int(ksp_history_max_env) if ksp_history_max_env else 800
        except ValueError:
            ksp_history_max_size = 800

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
    ) -> list[float] | None:
        if emit is None or not fortran_stdout:
            return None
        if str(solver_kind).strip().lower() != "gmres":
            return None
        size = int(b_vec.size)
        if ksp_history_max_size is not None and size > int(ksp_history_max_size):
            emit(1, f"fortran-stdout: KSP history skipped (size={size} > max={int(ksp_history_max_size)})")
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
    ) -> None:
        if emit is None or not iter_stats_enabled:
            return
        size = int(b_vec.size)
        if iter_stats_max_size is not None and size > int(iter_stats_max_size):
            emit(1, f"ksp_iterations skipped (size={size} > max={int(iter_stats_max_size)})")
            return
        solver_kind_l = str(solver_kind).strip().lower()
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
            def expand_reduced(v_reduced: jnp.ndarray) -> jnp.ndarray:
                z0 = jnp.zeros((1,), dtype=v_reduced.dtype)
                padded = jnp.concatenate([z0, v_reduced], axis=0)
                f_full = padded[full_to_active_jnp]
                f = f_full.reshape(op.fblock.f_shape)
                src = _constraint_scheme2_source_from_f(op, f)
                return jnp.concatenate([f_full, src.reshape((-1,))], axis=0)
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
        preconditioner_reduced = None
        bicgstab_preconditioner_reduced = None
        if rhs1_bicgstab_kind is not None:
            if emit is not None:
                emit(1, "solve_v3_full_system_linear_gmres: RHSMode=1 BiCGStab preconditioner=collision")
            bicgstab_preconditioner_reduced = _build_rhsmode1_collision_preconditioner(
                op=op, reduce_full=reduce_full, expand_reduced=expand_reduced
            )

        def _build_rhs1_preconditioner_reduced():
            if emit is not None:
                emit(
                    1,
                    "solve_v3_full_system_linear_gmres: building RHSMode=1 preconditioner="
                    f"{rhs1_precond_kind} (active-DOF)",
                )
            if rhs1_precond_kind == "theta_line":
                return _build_rhsmode1_theta_line_preconditioner(
                    op=op, reduce_full=reduce_full, expand_reduced=expand_reduced
                )
            if rhs1_precond_kind == "zeta_line":
                return _build_rhsmode1_zeta_line_preconditioner(
                    op=op, reduce_full=reduce_full, expand_reduced=expand_reduced
                )
            if rhs1_precond_kind == "schur":
                return _build_rhsmode1_schur_preconditioner(
                    op=op, reduce_full=reduce_full, expand_reduced=expand_reduced
                )
            if rhs1_precond_kind == "collision":
                return _build_rhsmode1_collision_preconditioner(
                    op=op, reduce_full=reduce_full, expand_reduced=expand_reduced
                )
            if rhs1_precond_kind == "adi":
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

                return preconditioner_reduced
            return _build_rhsmode1_block_preconditioner(op=op, reduce_full=reduce_full, expand_reduced=expand_reduced)

        if rhs1_precond_enabled:
            solver_kind = _solver_kind(solve_method)[0]
            if solver_kind != "bicgstab" and solve_method_kind != "dense":
                preconditioner_reduced = _build_rhs1_preconditioner_reduced()
        if preconditioner_reduced is None and bicgstab_preconditioner_reduced is not None:
            preconditioner_reduced = bicgstab_preconditioner_reduced
        if solve_method_kind == "dense_ksp":
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
        if preconditioner_reduced is not None and (not _gmres_result_is_finite(res_reduced)):
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
        target_reduced = max(float(atol), float(tol) * float(jnp.linalg.norm(rhs_reduced)))
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
        if float(res_reduced.residual_norm) > target_reduced and stage2_enabled and t.elapsed_s() < stage2_time_cap_s:
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
        if (
            float(res_reduced.residual_norm) > target_reduced
            and int(op.rhs_mode) == 1
            and (not bool(op.include_phi1))
            and rhs1_precond_kind == "point"
            and (op.fblock.fp is not None or op.fblock.pas is not None)
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
        strong_precond_kind: str | None = None
        if strong_precond_disabled:
            strong_precond_kind = None
        elif strong_precond_env in {"theta", "theta_line", "line_theta"}:
            strong_precond_kind = "theta_line"
        elif strong_precond_env in {"zeta", "zeta_line", "line_zeta"}:
            strong_precond_kind = "zeta_line"
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
                and int(active_size) >= strong_precond_min
                and (int(op.n_theta) > 1 or int(op.n_zeta) > 1)
            ):
                strong_precond_kind = "theta_line" if int(op.n_theta) >= int(op.n_zeta) else "zeta_line"

        if strong_precond_kind is not None and float(res_reduced.residual_norm) > target_reduced:
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
        dense_fallback_env = os.environ.get("SFINCS_JAX_RHSMODE1_DENSE_FALLBACK_MAX", "").strip()
        try:
            dense_fallback_max = int(dense_fallback_env) if dense_fallback_env else 3000
        except ValueError:
            dense_fallback_max = 0
        dense_fallback_max_huge = 0
        dense_fallback_ratio = 1.0e8
        if dense_fallback_max > 0:
            dense_fallback_max_huge_env = os.environ.get("SFINCS_JAX_RHSMODE1_DENSE_FALLBACK_MAX_HUGE", "").strip()
            try:
                dense_fallback_max_huge = int(dense_fallback_max_huge_env) if dense_fallback_max_huge_env else dense_fallback_max
            except ValueError:
                dense_fallback_max_huge = dense_fallback_max
            dense_fallback_ratio_env = os.environ.get("SFINCS_JAX_RHSMODE1_DENSE_FALLBACK_RATIO", "").strip()
            try:
                dense_fallback_ratio = float(dense_fallback_ratio_env) if dense_fallback_ratio_env else 1.0e8
            except ValueError:
                dense_fallback_ratio = 1.0e8
        res_ratio = float(res_reduced.residual_norm) / max(float(target_reduced), 1e-300)
        dense_fallback_limit = dense_fallback_max_huge if res_ratio > dense_fallback_ratio else dense_fallback_max
        force_dense_cs0 = int(op.constraint_scheme) == 0
        if force_dense_cs0:
            # constraintScheme=0 systems are singular; keep the dense fallback
            # available even when the residual ratio is huge.
            dense_fallback_limit = max(dense_fallback_limit, dense_fallback_max)
        if (
            dense_fallback_limit > 0
            and int(op.rhs_mode) == 1
            and (not bool(op.include_phi1))
            and int(active_size) <= dense_fallback_limit
            and (float(res_reduced.residual_norm) > target_reduced or force_dense_cs0)
        ):
            if emit is not None:
                emit(
                    0,
                    "solve_v3_full_system_linear_gmres: dense fallback "
                    f"(size={active_size} residual={float(res_reduced.residual_norm):.3e} > target={target_reduced:.3e})",
                )
            try:
                dense_method = "dense"
                if int(op.constraint_scheme) == 0:
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
                    emit(1, "solve_v3_full_system_linear_gmres: RHSMode=1 BiCGStab preconditioner=collision")
                bicgstab_preconditioner_full = _build_rhsmode1_collision_preconditioner(op=op)

            def _build_rhs1_preconditioner_full():
                if emit is not None:
                    emit(1, f"solve_v3_full_system_linear_gmres: building RHSMode=1 preconditioner={rhs1_precond_kind}")
                if rhs1_precond_kind == "theta_line":
                    return _build_rhsmode1_theta_line_preconditioner(op=op)
                if rhs1_precond_kind == "zeta_line":
                    return _build_rhsmode1_zeta_line_preconditioner(op=op)
                if rhs1_precond_kind == "schur":
                    return _build_rhsmode1_schur_preconditioner(op=op)
                if rhs1_precond_kind == "collision":
                    return _build_rhsmode1_collision_preconditioner(op=op)
                if rhs1_precond_kind == "adi":
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

                    return preconditioner_full
                return _build_rhsmode1_block_preconditioner(op=op)

            if rhs1_precond_enabled:
                solver_kind = _solver_kind(solve_method)[0]
                if solver_kind != "bicgstab" and solve_method_kind != "dense":
                    preconditioner_full = _build_rhs1_preconditioner_full()
            if preconditioner_full is None and bicgstab_preconditioner_full is not None:
                preconditioner_full = bicgstab_preconditioner_full
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
        if float(result.residual_norm) > target and stage2_enabled and t.elapsed_s() < stage2_time_cap_s:
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
        if (
            float(result.residual_norm) > target
            and int(op.rhs_mode) == 1
            and (not bool(op.include_phi1))
            and rhs1_precond_kind == "point"
            and (op.fblock.fp is not None or op.fblock.pas is not None)
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
            strong_precond_kind: str | None = None
            if strong_precond_disabled:
                strong_precond_kind = None
            elif strong_precond_env in {"theta", "theta_line", "line_theta"}:
                strong_precond_kind = "theta_line"
            elif strong_precond_env in {"zeta", "zeta_line", "line_zeta"}:
                strong_precond_kind = "zeta_line"
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
                    strong_precond_kind = "theta_line" if int(op.n_theta) >= int(op.n_zeta) else "zeta_line"

            if strong_precond_kind is not None and float(result.residual_norm) > target:
                if emit is not None:
                    emit(
                        0,
                        "solve_v3_full_system_linear_gmres: strong preconditioner fallback "
                        f"kind={strong_precond_kind} (residual={float(result.residual_norm):.3e} > target={target:.3e})",
                    )

                if strong_precond_kind == "theta_line":
                    strong_preconditioner_full = _build_rhsmode1_theta_line_preconditioner(op=op)
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
            dense_fallback_env = os.environ.get("SFINCS_JAX_RHSMODE1_DENSE_FALLBACK_MAX", "").strip()
            try:
                dense_fallback_max = int(dense_fallback_env) if dense_fallback_env else 3000
            except ValueError:
                dense_fallback_max = 3000
            if (
                dense_fallback_max > 0
                and int(op.rhs_mode) == 1
                and (not bool(op.include_phi1))
                and int(op.total_size) <= dense_fallback_max
                and float(result.residual_norm) > target
            ):
                if emit is not None:
                    emit(
                        0,
                        "solve_v3_full_system_linear_gmres: dense fallback "
                        f"(size={int(op.total_size)} residual={float(result.residual_norm):.3e} > target={target:.3e})",
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
                    op=op, reduce_full=_reduce_full, expand_reduced=_expand_reduced
                )
            else:
                preconditioner = _build_rhsmode1_block_preconditioner(op=op)

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
            dense_cutoff_env = os.environ.get("SFINCS_JAX_PHI1_NK_DENSE_CUTOFF", "").strip()
            try:
                dense_cutoff = int(dense_cutoff_env) if dense_cutoff_env else 5000
            except ValueError:
                dense_cutoff = 5000
            linear_size = active_size if use_active_dof_mode else int(op.total_size)
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
                restart=int(gmres_restart),
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
                restart_val=int(gmres_restart),
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
                    restart=int(gmres_restart),
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
                    restart_val=int(gmres_restart),
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
                restart=int(gmres_restart),
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
                restart_val=int(gmres_restart),
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
                    restart=int(gmres_restart),
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
                    restart_val=int(gmres_restart),
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
    if emit is not None:
        emit(1, f"solve_v3_transport_matrix_linear_gmres: rhs_mode={rhs_mode} whichRHS_count={n} total_size={int(op0.total_size)}")

    solve_method_use = solve_method
    force_krylov_env = os.environ.get("SFINCS_JAX_TRANSPORT_FORCE_KRYLOV", "").strip().lower()
    force_krylov = force_krylov_env in {"1", "true", "yes", "on"}
    force_dense_env = os.environ.get("SFINCS_JAX_TRANSPORT_FORCE_DENSE", "").strip().lower()
    force_dense = force_dense_env in {"1", "true", "yes", "on"}
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
            dense_retry_max = 3000 if int(rhs_mode) in {2, 3} else 0
    except ValueError:
        dense_retry_max = 3000 if int(rhs_mode) in {2, 3} else 0
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
            dense_fallback_max = 3000
    if int(rhs_mode) in {2, 3}:
        if force_dense:
            solve_method_use = "dense"
            if emit is not None:
                emit(0, f"solve_v3_transport_matrix_linear_gmres: forced dense solve for RHSMode={rhs_mode} (n={int(op0.total_size)})")
        elif (
            dense_fallback
            and (not force_krylov)
            and int(op0.total_size) <= dense_fallback_max
            and str(solve_method_use).lower() in {"auto", "default", "batched", "incremental"}
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

    solver_jit_env = os.environ.get("SFINCS_JAX_SOLVER_JIT", "").strip().lower()
    use_solver_jit = solver_jit_env not in {"0", "false", "no", "off"}

    dense_precond_env = os.environ.get("SFINCS_JAX_TRANSPORT_DENSE_PRECOND_MAX", "").strip()
    try:
        if dense_precond_env:
            dense_precond_max = int(dense_precond_env)
        else:
            dense_precond_max = 1600 if int(rhs_mode) == 2 else 600
    except ValueError:
        dense_precond_max = 1600 if int(rhs_mode) == 2 else 600
    dense_precond_enabled = (
        dense_precond_max > 0
        and int(rhs_mode) in {2, 3}
        and int(op0.total_size) <= dense_precond_max
        and str(solve_method_use).lower() != "dense"
    )
    dense_precond_cache_full: dict[tuple[object, int], Callable[[jnp.ndarray], jnp.ndarray]] = {}
    dense_precond_cache_reduced: dict[tuple[object, int], Callable[[jnp.ndarray], jnp.ndarray]] = {}

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
    }:
        transport_precond_kind = transport_precond_env
    else:
        transport_precond_kind = "auto"

    preconditioner_full = None
    preconditioner_reduced = None
    default_solver_kind = _solver_kind(solve_method_use)[0]
    if transport_precond_kind is not None and int(rhs_mode) in {2, 3}:
        precond_kind = transport_precond_kind
        if precond_kind == "auto":
            if default_solver_kind == "bicgstab":
                precond_kind = "collision"
            else:
                block_max_env = os.environ.get("SFINCS_JAX_TRANSPORT_PRECOND_BLOCK_MAX", "").strip()
                try:
                    block_max = int(block_max_env) if block_max_env else 5000
                except ValueError:
                    block_max = 5000
                if op0.fblock.fp is not None and int(op0.total_size) <= block_max:
                    precond_kind = "sxblock"
                else:
                    precond_kind = "collision"
        if precond_kind in {"xmg", "multigrid"}:
            preconditioner_full = _build_rhsmode23_xmg_preconditioner(op=op0)
            if use_active_dof_mode and reduce_full is not None and expand_reduced is not None:
                preconditioner_reduced = _build_rhsmode23_xmg_preconditioner(
                    op=op0, reduce_full=reduce_full, expand_reduced=expand_reduced
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
        lu, piv = jla.lu_factor(a_dense)

        def precond(v: jnp.ndarray) -> jnp.ndarray:
            return jla.lu_solve((lu, piv), v)

        cache[key] = precond
        return precond

    # Geometry scalars needed for the transport-matrix formulas.
    grids = grids_from_namelist(nml)
    geom = geometry_from_namelist(nml=nml, grids=grids)

    state_vectors: dict[int, jnp.ndarray] = {}
    residual_norms: dict[int, jnp.ndarray] = {}
    elapsed_s: list[jnp.ndarray] = []
    which_rhs_values = list(range(1, n + 1))
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
    if str(solve_method_use).lower() == "dense" and not use_active_dof_mode:
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

                def _mv_dense(x: jnp.ndarray) -> jnp.ndarray:
                    return apply_v3_full_system_operator_cached(op_probe_ref, x)

                a_dense = assemble_dense_matrix_from_matvec(
                    matvec=_mv_dense, n=int(op0.total_size), dtype=jnp.float64
                )
                rhs_mat = jnp.stack(rhs_by_index, axis=1)
                x_mat, _ = dense_solve_from_matrix(a=a_dense, b=rhs_mat)
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
                    state_vectors[which_rhs] = x_col
                    residual_norms[which_rhs] = res_norms[idx]
                    elapsed_s.append(jnp.asarray(t_dense.elapsed_s() / float(n), dtype=jnp.float64))
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
                        dtype=rhs_reduced.dtype,
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
                if _needs_retry(res_reduced, target_rhs) and dense_retry_max > 0 and int(active_size) <= int(dense_retry_max):
                    if emit is not None:
                        emit(
                            0,
                            "solve_v3_transport_matrix_linear_gmres: dense fallback "
                            f"(size={int(active_size)} residual={float(res_reduced.residual_norm):.3e} > target={target_rhs:.3e})",
                        )
                    try:
                        res_dense = _solve_linear(
                            matvec_fn=mv_reduced,
                            b_vec=rhs_reduced,
                            x0_vec=None,
                            tol_val=tol_rhs,
                            atol_val=atol,
                            restart_val=_restart_for_method("dense"),
                            maxiter_val=maxiter,
                            solve_method_val="dense",
                            preconditioner_val=None,
                            precondition_side_val="none",
                        )
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
                state_vectors[which_rhs] = x_full
                residual_norms[which_rhs] = res_norm_full
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
                        dtype=rhs.dtype,
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
                if _needs_retry(res, target_rhs) and dense_retry_max > 0 and int(op0.total_size) <= int(dense_retry_max):
                    if emit is not None:
                        emit(
                            0,
                            "solve_v3_transport_matrix_linear_gmres: dense fallback "
                            f"(size={int(op0.total_size)} residual={float(res.residual_norm):.3e} > target={target_rhs:.3e})",
                        )
                    try:
                        res_dense, residual_dense = _solve_linear_with_residual(
                            matvec_fn=mv,
                            b_vec=rhs,
                            x0_vec=None,
                            tol_val=tol_rhs,
                            atol_val=atol,
                            restart_val=_restart_for_method("dense"),
                            maxiter_val=maxiter,
                            solve_method_val="dense",
                            preconditioner_val=None,
                            precondition_side_val="none",
                        )
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
                state_vectors[which_rhs] = x_full
                if (not projection_needed) and residual_vec is not None and residual_vec.shape == rhs.shape:
                    ax_full = rhs - residual_vec
                    residual_norms[which_rhs] = res.residual_norm
                else:
                    ax_full = apply_v3_full_system_operator_cached(op_matvec, x_full)
                    residual_vec = ax_full - rhs
                    residual_norms[which_rhs] = jnp.linalg.norm(residual_vec)
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
            elapsed_s.append(jnp.asarray(t_rhs.elapsed_s(), dtype=jnp.float64))

    if emit is not None:
        emit(0, "solve_v3_transport_matrix_linear_gmres: computing whichRHS diagnostics (batched)")
    x_stack = jnp.stack([state_vectors[which_rhs] for which_rhs in which_rhs_values], axis=0)  # (N,total)
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
        use_remat_diag = int(x_stack.size) >= remat_min
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
            diag_stack = diag_fn(op0=op0, precomputed=precomputed, x_full_stack=x_stack)
        else:
            diag_fn = (
                v3_transport_diagnostics_vm_only_batch_op0_remat_jit
                if use_remat_diag
                else v3_transport_diagnostics_vm_only_batch_op0_jit
            )
            diag_stack = diag_fn(op0=op0, x_full_stack=x_stack)
    else:
        diag_op_stack = jtu.tree_map(lambda *xs: jnp.stack(xs, axis=0), *diag_op_by_index)
        diag_fn = (
            v3_transport_diagnostics_vm_only_batch_remat_jit
            if use_remat_diag
            else v3_transport_diagnostics_vm_only_batch_jit
        )
        diag_stack = diag_fn(op_stack=diag_op_stack, x_full_stack=x_stack)

    diag_pf_arr = jnp.transpose(diag_stack.particle_flux_vm_psi_hat, (1, 0))  # (S,N)
    diag_hf_arr = jnp.transpose(diag_stack.heat_flux_vm_psi_hat, (1, 0))  # (S,N)
    diag_flow_arr = jnp.transpose(diag_stack.fsab_flow, (1, 0))  # (S,N)

    tm = v3_transport_matrix_from_flux_arrays(
        op=op0,
        geom=geom,
        particle_flux_vm_psi_hat=diag_pf_arr,
        heat_flux_vm_psi_hat=diag_hf_arr,
        fsab_flow=diag_flow_arr,
    )
    state_out_env = os.environ.get("SFINCS_JAX_STATE_OUT", "").strip()
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
        fsab_flow=diag_flow_arr,
        particle_flux_vm_psi_hat=diag_pf_arr,
        heat_flux_vm_psi_hat=diag_hf_arr,
        elapsed_time_s=jnp.stack(elapsed_s, axis=0),
    )
