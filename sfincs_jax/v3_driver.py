from __future__ import annotations

from dataclasses import dataclass, replace

from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)

from collections.abc import Callable
import os
import numpy as np

import jax
import jax.numpy as jnp
from jax import tree_util as jtu

from .namelist import Namelist
from .solver import GMRESSolveResult, assemble_dense_matrix_from_matvec, dense_solve_from_matrix, gmres_solve
from .transport_matrix import (
    V3TransportDiagnostics,
    transport_matrix_size_from_rhs_mode,
    v3_transport_diagnostics_vm_only_batch_jit,
    v3_transport_matrix_column,
)
from .v3_system import _source_basis_constraint_scheme_1
from .verbose import Timer
from .v3 import geometry_from_namelist, grids_from_namelist
from .v3_system import (
    V3FullSystemOperator,
    apply_v3_full_system_jacobian,
    apply_v3_full_system_jacobian_jit,
    apply_v3_full_system_operator,
    apply_v3_full_system_operator_jit,
    full_system_operator_from_namelist,
    residual_v3_full_system,
    rhs_v3_full_system,
    rhs_v3_full_system_jit,
    with_transport_rhs_settings,
)


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


def _build_rhsmode1_preconditioner_operator(op: V3FullSystemOperator) -> V3FullSystemOperator:
    """Return a simplified RHSMode=1 operator for block preconditioning.

    The preconditioner retains local x/L couplings and collisions while dropping
    theta/zeta derivative couplings (streaming, ExB, and magnetic-drift derivatives).
    """
    if int(op.rhs_mode) != 1:
        return op

    fblock = op.fblock
    def _diag_only(m: jnp.ndarray) -> jnp.ndarray:
        return jnp.diag(jnp.diag(m))

    coll = replace(
        fblock.collisionless,
        ddtheta=_diag_only(fblock.collisionless.ddtheta),
        ddzeta=_diag_only(fblock.collisionless.ddzeta),
    )
    exb_theta = replace(fblock.exb_theta, ddtheta=_diag_only(fblock.exb_theta.ddtheta))
    exb_zeta = replace(fblock.exb_zeta, ddzeta=_diag_only(fblock.exb_zeta.ddzeta))
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
    op_pc = _build_rhsmode1_preconditioner_operator(op)
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
        m = int(rep_idx.shape[0])
        b = np.zeros((m, m), dtype=np.float64)
        for j, col in enumerate(rep_idx.tolist()):
            e = jnp.zeros((total,), dtype=jnp.float64).at[col].set(1.0)
            y = np.asarray(apply_v3_full_system_operator(op_pc, e), dtype=np.float64)
            b[:, j] = y[rep_idx]
        b = b + reg * np.eye(m, dtype=np.float64)
        try:
            inv = np.linalg.inv(b)
        except np.linalg.LinAlgError:
            inv = np.linalg.pinv(b, rcond=1e-12)
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
                        idx_map[s, it, iz, k] = int(((((s * n_x + ix) * n_l + il) * n_t + it) * n_z + iz))
                        k += 1

    idx_map_jnp = jnp.asarray(idx_map, dtype=jnp.int32)
    flat_idx_jnp = idx_map_jnp.reshape((-1,))
    block_inv_jnp = jnp.asarray(block_inv, dtype=jnp.float64)

    extra_start = int(op.f_size + op.phi1_size)
    extra_size = int(op.extra_size)
    extra_idx_np = np.arange(extra_start, extra_start + extra_size, dtype=np.int32)
    extra_idx_jnp = jnp.asarray(extra_idx_np, dtype=jnp.int32)
    extra_inv_jnp: jnp.ndarray | None = None
    if extra_size > 0:
        ee = np.zeros((extra_size, extra_size), dtype=np.float64)
        for j, col in enumerate(extra_idx_np.tolist()):
            e = jnp.zeros((total,), dtype=jnp.float64).at[col].set(1.0)
            y = np.asarray(apply_v3_full_system_operator(op_pc, e), dtype=np.float64)
            ee[:, j] = y[extra_idx_np]
        ee = ee + reg * np.eye(extra_size, dtype=np.float64)
        try:
            ee_inv = np.linalg.inv(ee)
        except np.linalg.LinAlgError:
            ee_inv = np.linalg.pinv(ee, rcond=1e-12)
        extra_inv_jnp = jnp.asarray(ee_inv, dtype=jnp.float64)

    def _apply_full(r_full: jnp.ndarray) -> jnp.ndarray:
        r_full = jnp.asarray(r_full, dtype=jnp.float64)
        r_loc = r_full[flat_idx_jnp].reshape((n_s, n_t, n_z, local_per_species))
        z_loc = jnp.einsum("sab,stzb->stza", block_inv_jnp, r_loc)
        z_full = jnp.zeros_like(r_full)
        z_full = z_full.at[flat_idx_jnp].set(z_loc.reshape((-1,)))
        if extra_inv_jnp is not None:
            r_extra = r_full[extra_idx_jnp]
            z_extra = extra_inv_jnp @ r_extra
            z_full = z_full.at[extra_idx_jnp].set(z_extra)
        return z_full

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
    solve_method: str = "batched",
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
        return apply_v3_full_system_operator_jit(op, x)

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

    active_idx_jnp: jnp.ndarray | None = None
    full_to_active_jnp: jnp.ndarray | None = None
    active_size = int(op.total_size)
    if use_active_dof_mode:
        active_idx_np = _transport_active_dof_indices(op)
        active_idx_jnp = jnp.asarray(active_idx_np, dtype=jnp.int32)
        full_to_active_np = np.zeros((int(op.total_size),), dtype=np.int32)
        full_to_active_np[np.asarray(active_idx_np, dtype=np.int32)] = np.arange(1, int(active_idx_np.shape[0]) + 1, dtype=np.int32)
        full_to_active_jnp = jnp.asarray(full_to_active_np, dtype=jnp.int32)
        active_size = int(active_idx_np.shape[0])
        if emit is not None:
            emit(1, f"solve_v3_full_system_linear_gmres: active-DOF mode enabled (size={active_size}/{int(op.total_size)})")

    if emit is not None:
        emit(1, f"solve_v3_full_system_linear_gmres: GMRES tol={tol} atol={atol} restart={restart} maxiter={maxiter} solve_method={solve_method}")
        emit(1, "solve_v3_full_system_linear_gmres: evaluateJacobian called (matrix-free)")
    rhs1_precond_env = os.environ.get("SFINCS_JAX_RHSMODE1_PRECONDITIONER", "").strip().lower()
    rhs1_precond_enabled = (
        int(op.rhs_mode) == 1
        and (not bool(op.include_phi1))
        and int(op.extra_size) > 0
        and rhs1_precond_env in {"1", "true", "yes", "on"}
    )
    stage2_env = os.environ.get("SFINCS_JAX_LINEAR_STAGE2", "").strip().lower()
    if stage2_env in {"0", "false", "no", "off"}:
        stage2_enabled = False
    elif stage2_env in {"1", "true", "yes", "on"}:
        stage2_enabled = True
    else:
        stage2_enabled = int(op.rhs_mode) == 1 and (not bool(op.include_phi1))
    stage2_time_cap_s = float(os.environ.get("SFINCS_JAX_LINEAR_STAGE2_MAX_ELAPSED_S", "10.0"))
    if use_active_dof_mode:
        assert active_idx_jnp is not None
        assert full_to_active_jnp is not None

        def reduce_full(v_full: jnp.ndarray) -> jnp.ndarray:
            return v_full[active_idx_jnp]

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
        preconditioner_reduced = None
        if rhs1_precond_enabled:
            if emit is not None:
                emit(1, "solve_v3_full_system_linear_gmres: building RHSMode=1 block preconditioner (active-DOF)")
            preconditioner_reduced = _build_rhsmode1_block_preconditioner(
                op=op, reduce_full=reduce_full, expand_reduced=expand_reduced
            )
        res_reduced = gmres_solve(
            matvec=mv_reduced,
            b=rhs_reduced,
            preconditioner=preconditioner_reduced,
            x0=x0_reduced,
            tol=tol,
            atol=atol,
            restart=restart,
            maxiter=maxiter,
            solve_method=solve_method,
        )
        if preconditioner_reduced is not None and (not _gmres_result_is_finite(res_reduced)):
            if emit is not None:
                emit(0, "solve_v3_full_system_linear_gmres: preconditioned reduced GMRES returned non-finite result; retrying without preconditioner")
            res_reduced = gmres_solve(
                matvec=mv_reduced,
                b=rhs_reduced,
                preconditioner=None,
                x0=x0_reduced,
                tol=tol,
                atol=atol,
                restart=restart,
                maxiter=maxiter,
                solve_method=solve_method,
            )
        target_reduced = max(float(atol), float(tol) * float(jnp.linalg.norm(rhs_reduced)))
        if float(res_reduced.residual_norm) > target_reduced and stage2_enabled and t.elapsed_s() < stage2_time_cap_s:
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
            res2 = gmres_solve(
                matvec=mv_reduced,
                b=rhs_reduced,
                preconditioner=preconditioner_reduced,
                x0=res_reduced.x,
                tol=tol,
                atol=atol,
                restart=stage2_restart,
                maxiter=stage2_maxiter,
                solve_method=stage2_method,
            )
            if float(res2.residual_norm) < float(res_reduced.residual_norm):
                res_reduced = res2
        x_full = expand_reduced(res_reduced.x)
        residual_norm_full = jnp.linalg.norm(mv(x_full) - rhs)
        result = GMRESSolveResult(x=x_full, residual_norm=residual_norm_full)
    else:
        preconditioner_full = None
        if rhs1_precond_enabled:
            if emit is not None:
                emit(1, "solve_v3_full_system_linear_gmres: building RHSMode=1 block preconditioner")
            preconditioner_full = _build_rhsmode1_block_preconditioner(op=op)
        result = gmres_solve(
            matvec=mv,
            b=rhs,
            preconditioner=preconditioner_full,
            x0=x0,
            tol=tol,
            atol=atol,
            restart=restart,
            maxiter=maxiter,
            solve_method=solve_method,
        )
        if preconditioner_full is not None and (not _gmres_result_is_finite(result)):
            if emit is not None:
                emit(0, "solve_v3_full_system_linear_gmres: preconditioned GMRES returned non-finite result; retrying without preconditioner")
            result = gmres_solve(
                matvec=mv,
                b=rhs,
                preconditioner=None,
                x0=x0,
                tol=tol,
                atol=atol,
                restart=restart,
                maxiter=maxiter,
                solve_method=solve_method,
            )
        # If GMRES does not reach the requested tolerance (common without preconditioning),
        # retry with a larger iteration budget and the more robust incremental mode.
        target = max(float(atol), float(tol) * float(rhs_norm))
        if float(result.residual_norm) > target and stage2_enabled and t.elapsed_s() < stage2_time_cap_s:
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
            res2 = gmres_solve(
                matvec=mv,
                b=rhs,
                preconditioner=preconditioner_full,
                x0=result.x,
                tol=tol,
                atol=atol,
                restart=stage2_restart,
                maxiter=stage2_maxiter,
                solve_method=stage2_method,
            )
            if float(res2.residual_norm) < float(result.residual_norm):
                result = res2
    if int(op.rhs_mode) == 1:
        project_rhs1 = os.environ.get("SFINCS_JAX_RHSMODE1_PROJECT_NULLSPACE", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if project_rhs1:
            x_projected = _project_constraint_scheme1_nullspace_solution(
                op=op,
                x_vec=result.x,
                rhs_vec=rhs,
                matvec_op=op,
                enabled_env_var="SFINCS_JAX_RHSMODE1_PROJECT_NULLSPACE",
            )
            if not bool(jnp.allclose(x_projected, result.x)):
                residual_norm_projected = jnp.linalg.norm(mv(x_projected) - rhs)
                result = GMRESSolveResult(x=x_projected, residual_norm=residual_norm_projected)
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
        lin = gmres_solve(
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
    env_gmres_tol = os.environ.get("SFINCS_JAX_PHI1_GMRES_TOL", "").strip()
    if env_gmres_tol:
        gmres_tol = float(env_gmres_tol)

    if x0 is None:
        x = jnp.zeros((op.total_size,), dtype=jnp.float64)
    else:
        x = jnp.asarray(x0, dtype=jnp.float64)
        if x.shape != (op.total_size,):
            raise ValueError(f"x0 must have shape {(op.total_size,)}, got {x.shape}")

    preconditioner = None
    # Only enable block preconditioning in the PETSc-like parity mode that freezes the
    # Jacobian/linearization. For autodiff-linearized Newton steps, JAX's GMRES can
    # behave differently with an approximate preconditioner, which impacts iteration
    # histories and `sfincsOutput.h5` shape parity for linear Phi1 fixtures.
    pc_env = os.environ.get("SFINCS_JAX_PHI1_USE_PRECONDITIONER", "").strip().lower()
    use_preconditioner = pc_env not in {"0", "false", "no", "off"}
    if use_preconditioner and use_frozen_linearization and int(op.rhs_mode) == 1:
        if emit is not None:
            emit(1, "solve_v3_full_system_newton_krylov_history: building RHSMode=1 block preconditioner")
        preconditioner = _build_rhsmode1_block_preconditioner(op=op)

    last_linear_resid = jnp.asarray(jnp.inf, dtype=jnp.float64)
    accepted: list[jnp.ndarray] = []
    rnorm_initial: float | None = None

    for k in range(int(max_newton)):
        if emit is not None:
            emit(1, f"newton_iter={k}: evaluateResidual called")
        op_use = op
        if bool(op.include_phi1):
            phi1_flat = x[op.f_size : op.f_size + op.n_theta * op.n_zeta]
            phi1 = phi1_flat.reshape((op.n_theta, op.n_zeta))
            op_use = replace(op, phi1_hat_base=phi1)

        r = apply_v3_full_system_operator_jit(op_use, x, include_jacobian_terms=False) - rhs_v3_full_system_jit(op_use)
        rnorm = jnp.linalg.norm(r)
        rnorm_f = float(rnorm)
        if rnorm_initial is None:
            rnorm_initial = max(rnorm_f, 1e-300)
        if emit is not None:
            emit(0, f"newton_iter={k}: residual_norm={rnorm_f:.6e}")
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
                        apply_v3_full_system_operator_jit(op_use, xx, include_jacobian_terms=True)
                        - rhs_v3_full_system_jit(op_rhs_x)
                    )

                _r_lin, jvp = jax.linearize(residual_for_jac, x)
                matvec = jvp
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
                        apply_v3_full_system_operator_jit(op_mat_x, xx, include_jacobian_terms=True)
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
            if int(op.total_size) <= int(dense_cutoff):
                solve_method_linear = "dense"

        lin = gmres_solve(
            matvec=matvec,
            b=-r,
            preconditioner=preconditioner,
            tol=float(gmres_tol),
            restart=int(gmres_restart),
            maxiter=gmres_maxiter,
            solve_method=solve_method_linear,
        )
        if preconditioner is not None and (not _gmres_result_is_finite(lin)):
            if emit is not None:
                emit(
                    0,
                    "newton_iter="
                    f"{k}: preconditioned GMRES returned non-finite result; retrying without preconditioner",
                )
            lin = gmres_solve(
                matvec=matvec,
                b=-r,
                preconditioner=None,
                tol=float(gmres_tol),
                restart=int(gmres_restart),
                maxiter=gmres_maxiter,
                solve_method=solve_method_linear,
            )
        if emit is not None:
            emit(1, f"newton_iter={k}: gmres_residual={float(lin.residual_norm):.6e}")
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
        s = lin.x
        last_linear_resid = lin.residual_norm

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


def _project_constraint_scheme1_nullspace_solution(
    *,
    op: V3FullSystemOperator,
    x_vec: jnp.ndarray,
    rhs_vec: jnp.ndarray,
    matvec_op: V3FullSystemOperator,
    enabled_env_var: str,
) -> jnp.ndarray:
    """Project solution to constraintScheme=1 nullspace complement and enforce source rows."""
    if int(op.constraint_scheme) != 1:
        return x_vec
    if int(op.phi1_size) != 0:
        return x_vec
    if int(op.extra_size) == 0:
        return x_vec

    project_env = os.environ.get(enabled_env_var, "").strip().lower()
    if project_env in {"0", "false", "no", "off"}:
        return x_vec

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

    r = apply_v3_full_system_operator_jit(matvec_op, x_vec) - rhs_vec
    r_extra = r[-op.extra_size :]
    cols = [apply_v3_full_system_operator_jit(matvec_op, v)[-op.extra_size :] for v in basis]
    m = jnp.stack(cols, axis=1)
    c_res, *_ = jnp.linalg.lstsq(m, -r_extra, rcond=None)
    x_corr = sum(v * c_res[i] for i, v in enumerate(basis))
    x_vec = x_vec + x_corr

    b_mat = jnp.stack(basis, axis=1)
    gram = b_mat.T @ b_mat
    proj_rhs = b_mat.T @ x_vec
    c_proj = jnp.linalg.solve(gram, proj_rhs)
    return x_vec - b_mat @ c_proj


def solve_v3_transport_matrix_linear_gmres(
    *,
    nml: Namelist,
    x0: jnp.ndarray | None = None,
    tol: float = 1e-10,
    atol: float = 0.0,
    restart: int = 80,
    maxiter: int | None = 400,
    solve_method: str = "batched",
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
    rhs_mode = int(op0.rhs_mode)
    n = transport_matrix_size_from_rhs_mode(rhs_mode)
    if emit is not None:
        emit(1, f"solve_v3_transport_matrix_linear_gmres: rhs_mode={rhs_mode} whichRHS_count={n} total_size={int(op0.total_size)}")

    # JAX's built-in GMRES stagnates at ~1e-6 residual on some small RHSMode=2 parity fixtures,
    # which prevents end-to-end transport-matrix parity. For small systems, fall back to a dense
    # JAX solve (still differentiable) by assembling the matrix from matvecs.
    solve_method_use = solve_method
    force_krylov_env = os.environ.get("SFINCS_JAX_TRANSPORT_FORCE_KRYLOV", "").strip().lower()
    force_krylov = force_krylov_env in {"1", "true", "yes", "on"}
    if (not force_krylov) and int(op0.total_size) <= 5000 and str(solve_method).lower() in {"batched", "incremental"}:
        # On some JAX versions/platforms, `jax.scipy.sparse.linalg.gmres` can return NaNs for
        # small ill-conditioned problems (observed in CI for RHSMode=3 scheme12 fixtures).
        # Dense assembly is cheap at these sizes and improves robustness.
        if int(rhs_mode) in {2, 3}:
            solve_method_use = "dense"
            if emit is not None:
                emit(0, f"solve_v3_transport_matrix_linear_gmres: using dense solve for RHSMode={rhs_mode} (n={int(op0.total_size)})")

    active_dof_env = os.environ.get("SFINCS_JAX_TRANSPORT_ACTIVE_DOF", "").strip().lower()
    use_active_dof_mode = int(rhs_mode) in {2, 3} and active_dof_env in {"1", "true", "yes", "on"}
    # For reduced active-DOF parity mode, prefer Krylov iterations over dense direct
    # solves to stay closer to upstream PETSc/KSP behavior for singular transport systems.
    if use_active_dof_mode and str(solve_method_use).lower() == "dense":
        solve_method_use = str(solve_method)
    elif int(rhs_mode) in {2, 3} and emit is not None:
        emit(
            1,
            "solve_v3_transport_matrix_linear_gmres: active-DOF mode disabled "
            "(set SFINCS_JAX_TRANSPORT_ACTIVE_DOF=1 to enable experimental reduced solve)",
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
            emit(
                1,
                "solve_v3_transport_matrix_linear_gmres: active-DOF mode enabled "
                f"(size={active_size}/{int(op0.total_size)})",
            )

    # Geometry scalars needed for the transport-matrix formulas.
    grids = grids_from_namelist(nml)
    geom = geometry_from_namelist(nml=nml, grids=grids)

    state_vectors: dict[int, jnp.ndarray] = {}
    residual_norms: dict[int, jnp.ndarray] = {}
    diag_fsab_flow: list[jnp.ndarray] = []
    diag_pf: list[jnp.ndarray] = []
    diag_hf: list[jnp.ndarray] = []
    diag_by_rhs: list[V3TransportDiagnostics] = []
    elapsed_s: list[jnp.ndarray] = []
    which_rhs_values = list(range(1, n + 1))
    op_rhs_by_index = [with_transport_rhs_settings(op0, which_rhs=which_rhs) for which_rhs in which_rhs_values]
    rhs_by_index = [rhs_v3_full_system(op_rhs) for op_rhs in op_rhs_by_index]

    use_op_rhs_in_matvec = op0.fblock.pas is not None
    env_transport_matvec = os.environ.get("SFINCS_JAX_TRANSPORT_MATVEC_MODE", "").strip().lower()
    if env_transport_matvec == "rhs":
        use_op_rhs_in_matvec = True
    elif env_transport_matvec == "base":
        use_op_rhs_in_matvec = False
    op_matvec_by_index = [op_rhs if use_op_rhs_in_matvec else op0 for op_rhs in op_rhs_by_index]

    env_diag_op = os.environ.get("SFINCS_JAX_TRANSPORT_DIAG_OP", "").strip().lower()
    if env_diag_op == "rhs":
        diag_op_by_index = op_rhs_by_index
    else:
        diag_op_by_index = [op0 for _ in which_rhs_values]

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
            probe0 = jnp.zeros((int(op0.total_size),), dtype=jnp.float64).at[0].set(1.0)
            probe1 = jnp.zeros((int(op0.total_size),), dtype=jnp.float64).at[-1].set(1.0)
            same_operator = True
            y_ref0 = apply_v3_full_system_operator_jit(op_probe_ref, probe0)
            y_ref1 = apply_v3_full_system_operator_jit(op_probe_ref, probe1)
            for op_probe in op_matvec_by_index[1:]:
                d0 = float(jnp.linalg.norm(apply_v3_full_system_operator_jit(op_probe, probe0) - y_ref0))
                d1 = float(jnp.linalg.norm(apply_v3_full_system_operator_jit(op_probe, probe1) - y_ref1))
                if max(d0, d1) > 1e-13:
                    same_operator = False
                    break

            if same_operator:
                if emit is not None:
                    emit(1, "solve_v3_transport_matrix_linear_gmres: dense batched solve across all whichRHS")
                t_dense = Timer()

                def _mv_dense(x: jnp.ndarray) -> jnp.ndarray:
                    return apply_v3_full_system_operator_jit(op_probe_ref, x)

                a_dense = assemble_dense_matrix_from_matvec(
                    matvec=_mv_dense, n=int(op0.total_size), dtype=jnp.float64
                )
                rhs_mat = jnp.stack(rhs_by_index, axis=1)
                x_mat, _ = dense_solve_from_matrix(a=a_dense, b=rhs_mat)

                for idx, which_rhs in enumerate(which_rhs_values):
                    x_col = x_mat[:, idx]
                    rhs_vec = rhs_by_index[idx]
                    x_col = _maybe_project_constraint_nullspace(
                        x_col, which_rhs=int(which_rhs), op_matvec=op_probe_ref, rhs_vec=rhs_vec
                    )
                    state_vectors[which_rhs] = x_col
                    residual_norms[which_rhs] = jnp.linalg.norm(
                        apply_v3_full_system_operator_jit(op_probe_ref, x_col) - rhs_vec
                    )
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

                def reduce_full(v_full: jnp.ndarray) -> jnp.ndarray:
                    return v_full[active_idx_jnp]

                def expand_reduced(v_reduced: jnp.ndarray) -> jnp.ndarray:
                    z0 = jnp.zeros((1,), dtype=v_reduced.dtype)
                    padded = jnp.concatenate([z0, v_reduced], axis=0)
                    return padded[full_to_active_jnp]

                def mv_reduced(x_reduced: jnp.ndarray) -> jnp.ndarray:
                    y_full = apply_v3_full_system_operator_jit(op_matvec, expand_reduced(x_reduced))
                    return reduce_full(y_full)

                rhs_reduced = reduce_full(rhs)
                x0_reduced = None
                if x0 is not None:
                    x0_arr = jnp.asarray(x0)
                    if x0_arr.shape == (active_size,):
                        x0_reduced = x0_arr
                    elif x0_arr.shape == (op0.total_size,):
                        x0_reduced = reduce_full(x0_arr)

                res_reduced = gmres_solve(
                    matvec=mv_reduced,
                    b=rhs_reduced,
                    x0=x0_reduced,
                    tol=tol_rhs,
                    atol=atol,
                    restart=restart,
                    maxiter=maxiter,
                    solve_method=solve_method_rhs,
                )
                x_full = expand_reduced(res_reduced.x)
                x_full = _maybe_project_constraint_nullspace(
                    x_full, which_rhs=int(which_rhs), op_matvec=op_matvec, rhs_vec=rhs
                )
                res_norm_full = jnp.linalg.norm(apply_v3_full_system_operator_jit(op_matvec, x_full) - rhs)
                state_vectors[which_rhs] = x_full
                residual_norms[which_rhs] = res_norm_full
            else:
                def mv(x: jnp.ndarray) -> jnp.ndarray:
                    return apply_v3_full_system_operator_jit(op_matvec, x)

                res = gmres_solve(
                    matvec=mv,
                    b=rhs,
                    x0=x0,
                    tol=tol_rhs,
                    atol=atol,
                    restart=restart,
                    maxiter=maxiter,
                    solve_method=solve_method_rhs,
                )
                x_full = res.x
                x_full = _maybe_project_constraint_nullspace(
                    x_full, which_rhs=int(which_rhs), op_matvec=op_matvec, rhs_vec=rhs
                )
                state_vectors[which_rhs] = x_full
                residual_norms[which_rhs] = jnp.linalg.norm(apply_v3_full_system_operator_jit(op_matvec, x_full) - rhs)
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
    diag_op_stack = jtu.tree_map(lambda *xs: jnp.stack(xs, axis=0), *diag_op_by_index)
    diag_stack = v3_transport_diagnostics_vm_only_batch_jit(op_stack=diag_op_stack, x_full_stack=x_stack)

    diag_pf_arr = jnp.transpose(diag_stack.particle_flux_vm_psi_hat, (1, 0))  # (S,N)
    diag_hf_arr = jnp.transpose(diag_stack.heat_flux_vm_psi_hat, (1, 0))  # (S,N)
    diag_flow_arr = jnp.transpose(diag_stack.fsab_flow, (1, 0))  # (S,N)

    diag_pf = [diag_pf_arr[:, i] for i in range(n)]
    diag_hf = [diag_hf_arr[:, i] for i in range(n)]
    diag_fsab_flow = [diag_flow_arr[:, i] for i in range(n)]
    diag_by_rhs = [jtu.tree_map(lambda a, i=i: a[i], diag_stack) for i in range(n)]

    tm_cols = [
        v3_transport_matrix_column(op=op_rhs_by_index[i], geom=geom, which_rhs=int(which_rhs_values[i]), diag=diag_by_rhs[i])
        for i in range(n)
    ]
    tm = jnp.stack(tm_cols, axis=1)
    if emit is not None:
        emit(0, "solve_v3_transport_matrix_linear_gmres: done")
        emit(1, f"solve_v3_transport_matrix_linear_gmres: elapsed_s={t_all.elapsed_s():.3f}")
    return V3TransportMatrixSolveResult(
        op0=op0,
        transport_matrix=tm,
        state_vectors_by_rhs=state_vectors,
        residual_norms_by_rhs=residual_norms,
        fsab_flow=jnp.stack(diag_fsab_flow, axis=1),
        particle_flux_vm_psi_hat=jnp.stack(diag_pf, axis=1),
        heat_flux_vm_psi_hat=jnp.stack(diag_hf, axis=1),
        elapsed_time_s=jnp.stack(elapsed_s, axis=0),
    )
