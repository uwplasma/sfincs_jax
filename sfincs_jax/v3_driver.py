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
from .solver import GMRESSolveResult, gmres_solve
from .transport_matrix import transport_matrix_size_from_rhs_mode, v3_transport_matrix_from_state_vectors
from .v3_system import _source_basis_constraint_scheme_1
from .verbose import Timer
from .v3 import geometry_from_namelist, grids_from_namelist
from .v3_system import (
    V3FullSystemOperator,
    apply_v3_full_system_operator,
    apply_v3_full_system_operator_jit,
    full_system_operator_from_namelist,
    residual_v3_full_system,
    rhs_v3_full_system,
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
    if emit is not None:
        emit(2, f"solve_v3_full_system_linear_gmres: rhs_norm={float(jnp.linalg.norm(rhs)):.6e}")

    def mv(x):
        # Use the JIT-compiled operator application to reduce Python overhead in repeated matvecs
        # (e.g. during GMRES iterations and Er scans).
        return apply_v3_full_system_operator_jit(op, x)

    if emit is not None:
        emit(1, f"solve_v3_full_system_linear_gmres: GMRES tol={tol} atol={atol} restart={restart} maxiter={maxiter} solve_method={solve_method}")
        emit(1, "solve_v3_full_system_linear_gmres: evaluateJacobian called (matrix-free)")
    result = gmres_solve(
        matvec=mv,
        b=rhs,
        x0=x0,
        tol=tol,
        atol=atol,
        restart=restart,
        maxiter=maxiter,
        solve_method=solve_method,
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

        r = apply_v3_full_system_operator(op_use, x) - rhs_v3_full_system(op_use)
        rnorm = jnp.linalg.norm(r)
        rnorm_f = float(rnorm)
        if rnorm_initial is None:
            rnorm_initial = max(rnorm_f, 1e-300)
        if emit is not None:
            emit(0, f"newton_iter={k}: residual_norm={rnorm_f:.6e}")

        converged_abs = rnorm_f < float(tol)
        converged_rel = rnorm_f <= float(nonlinear_rtol) * float(rnorm_initial)
        if converged_abs or converged_rel:
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
            if jac_mode not in {"frozen", "frozen_rhs"}:
                jac_mode = "frozen_rhs"
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
                    return apply_v3_full_system_operator(op_use, xx) - rhs_v3_full_system(op_rhs_x)

                _r_lin, jvp = jax.linearize(residual_for_jac, x)
                matvec = jvp
                if emit is not None:
                    emit(1, f"newton_iter={k}: evaluateJacobian called (frozen operator + dynamic RHS)")
            else:
                matvec = lambda dx: apply_v3_full_system_operator(op_use, dx)
                if emit is not None:
                    emit(1, f"newton_iter={k}: evaluateJacobian called (fully frozen linearization)")
        else:
            # Optional exact mode for debugging/experimentation.
            _r_lin, jvp = jax.linearize(lambda xx: residual_v3_full_system(op, xx), x)
            matvec = jvp
            if emit is not None:
                emit(1, f"newton_iter={k}: evaluateJacobian called (autodiff linearization)")

        lin = gmres_solve(
            matvec=matvec,
            b=-r,
            tol=float(gmres_tol),
            restart=int(gmres_restart),
            maxiter=gmres_maxiter,
            solve_method=str(solve_method),
        )
        if emit is not None:
            emit(1, f"newton_iter={k}: gmres_residual={float(lin.residual_norm):.6e}")
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
        try:
            ls_factor = float(ls_factor_env) if ls_factor_env else 0.9
        except ValueError:
            ls_factor = 0.9
        for _ in range(12):
            x_try = x + (step * step_scale) * s
            if use_frozen_linearization and frozen_jac_mode == "frozen_rhs":
                if bool(op.include_phi1):
                    phi1_flat_x = x_try[op.f_size : op.f_size + op.n_theta * op.n_zeta]
                    phi1_x = phi1_flat_x.reshape((op.n_theta, op.n_zeta))
                    op_rhs_x = replace(op, phi1_hat_base=phi1_x)
                else:
                    op_rhs_x = op
                r_try = apply_v3_full_system_operator(op_use, x_try) - rhs_v3_full_system(op_rhs_x)
            elif use_frozen_linearization and frozen_jac_mode == "frozen":
                r_try = apply_v3_full_system_operator(op_use, x_try) - rhs_v3_full_system(op_use)
            else:
                r_try = residual_v3_full_system(op, x_try)
            rnorm_try = float(jnp.linalg.norm(r_try))
            if rnorm_try <= ls_factor * rnorm0:
                x = x_try
                accepted.append(x)
                break
            step *= 0.5
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
    elapsed_s: list[jnp.ndarray] = []

    for which_rhs in range(1, n + 1):
        t_rhs = Timer()
        op_rhs = with_transport_rhs_settings(op0, which_rhs=which_rhs)
        rhs = rhs_v3_full_system(op_rhs)
        if emit is not None:
            emit(0, f"whichRHS={which_rhs}/{n}: assembling+solving (rhs_norm={float(jnp.linalg.norm(rhs)):.6e})")
            emit(1, f"whichRHS={which_rhs}/{n}: evaluateJacobian called (matrix-free)")

        # In upstream v3, the transport RHS settings are applied globally before each solve.
        # For PAS parity fixtures, matching this behavior requires using `op_rhs` in the matvec.
        # For full FP, default to `op0` for robustness, with an env override for parity tuning.
        use_op_rhs_in_matvec = op0.fblock.pas is not None
        env_transport_matvec = os.environ.get("SFINCS_JAX_TRANSPORT_MATVEC_MODE", "").strip().lower()
        if env_transport_matvec == "rhs":
            use_op_rhs_in_matvec = True
        elif env_transport_matvec == "base":
            use_op_rhs_in_matvec = False
        op_matvec = op_rhs if use_op_rhs_in_matvec else op0

        # For RHSMode=2 whichRHS=3 (E_parallel drive), PETSc tends to stop at a looser
        # tolerance, leaving small constraint residuals that appear in the moment-family
        # diagnostics. Mimic that behavior by using a Krylov solve with a looser tol.
        loose_env = os.environ.get("SFINCS_JAX_TRANSPORT_EPAR_LOOSE", "").strip().lower()
        krylov_env = os.environ.get("SFINCS_JAX_TRANSPORT_EPAR_KRYLOV", "").strip().lower()
        use_loose_epar_krylov = (
            loose_env in {"1", "true", "yes", "on"}
            and int(rhs_mode) == 2
            and which_rhs == 3
            and int(op0.constraint_scheme) == 1
        )
        force_epar_krylov = (
            krylov_env in {"1", "true", "yes", "on"}
            and int(rhs_mode) == 2
            and which_rhs == 3
            and int(op0.constraint_scheme) == 1
        )
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

        def _maybe_project_constraint_nullspace(x_vec: jnp.ndarray) -> jnp.ndarray:
            if int(rhs_mode) != 2 or which_rhs != 3:
                return x_vec
            if int(op0.constraint_scheme) != 1:
                return x_vec
            if int(op0.phi1_size) != 0:
                return x_vec
            project_env = os.environ.get("SFINCS_JAX_TRANSPORT_PROJECT_NULLSPACE", "").strip().lower()
            if project_env in {"0", "false", "no", "off"}:
                return x_vec

            # Build basis vectors from constraintScheme=1 source basis functions.
            xpart1, xpart2 = _source_basis_constraint_scheme_1(op0.x)
            ix0 = 1 if bool(op0.point_at_x0) else 0
            f_shape = op0.fblock.f_shape
            n_s, n_x, n_l, n_t, n_z = f_shape

            def _basis(src_index: int, xpart: jnp.ndarray) -> jnp.ndarray:
                f = jnp.zeros(f_shape, dtype=jnp.float64)
                f = f.at[:, ix0:, 0, :, :].set(xpart[ix0:][None, :, None, None])
                extra = jnp.zeros((n_s, 2), dtype=jnp.float64)
                extra = extra.at[:, src_index].set(-1.0)
                flat = jnp.concatenate([f.reshape((-1,)), extra.reshape((-1,))], axis=0)
                return flat

            v1 = _basis(0, xpart1)
            v2 = _basis(1, xpart2)

            r = apply_v3_full_system_operator(op_matvec, x_vec) - rhs
            m1 = apply_v3_full_system_operator(op_matvec, v1)
            m2 = apply_v3_full_system_operator(op_matvec, v2)
            m = jnp.stack([m1, m2], axis=1)  # (N,2)
            c, *_ = jnp.linalg.lstsq(m, -r, rcond=None)
            return x_vec + v1 * c[0] + v2 * c[1]

        if use_active_dof_mode:
            assert active_idx_jnp is not None
            assert full_to_active_jnp is not None

            def reduce_full(v_full: jnp.ndarray) -> jnp.ndarray:
                return v_full[active_idx_jnp]

            def expand_reduced(v_reduced: jnp.ndarray) -> jnp.ndarray:
                # Gather-based expansion (instead of scatter) keeps the reduced
                # matvec compatible with JAX's transpose rules used by GMRES.
                z0 = jnp.zeros((1,), dtype=v_reduced.dtype)
                padded = jnp.concatenate([z0, v_reduced], axis=0)
                return padded[full_to_active_jnp]

            def mv_reduced(x_reduced: jnp.ndarray) -> jnp.ndarray:
                y_full = apply_v3_full_system_operator(op_matvec, expand_reduced(x_reduced))
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
                # Match v3 solver.F90 RHSMode=2/3 loop: solutionVec is reset to 0
                # before each whichRHS solve (no warm-start from previous RHS).
                x0=x0_reduced,
                tol=tol_rhs,
                atol=atol,
                restart=restart,
                maxiter=maxiter,
                solve_method=solve_method_rhs,
            )
            x_full = expand_reduced(res_reduced.x)
            x_full = _maybe_project_constraint_nullspace(x_full)
            res_norm_full = jnp.linalg.norm(apply_v3_full_system_operator(op_matvec, x_full) - rhs)
            state_vectors[which_rhs] = x_full
            residual_norms[which_rhs] = res_norm_full
        else:
            def mv(x):
                return apply_v3_full_system_operator(op_matvec, x)

            res = gmres_solve(
                matvec=mv,
                b=rhs,
                # Match v3 solver.F90 RHSMode=2/3 loop: solutionVec is reset to 0
                # before each whichRHS solve (no warm-start from previous RHS).
                x0=x0,
                tol=tol_rhs,
                atol=atol,
                restart=restart,
                maxiter=maxiter,
                solve_method=solve_method_rhs,
            )
            x_full = res.x
            x_full = _maybe_project_constraint_nullspace(x_full)
            state_vectors[which_rhs] = x_full
            residual_norms[which_rhs] = res.residual_norm
        if emit is not None:
            emit(
                0,
                f"whichRHS={which_rhs}: residual_norm={float(residual_norms[which_rhs]):.6e} "
                f"elapsed_s={t_rhs.elapsed_s():.3f}",
            )
        # Collect diagnostics for parity with v3 `sfincsOutput.h5` and postprocessing scripts.
        from .transport_matrix import v3_transport_diagnostics_vm_only

        if emit is not None:
            emit(1, f"whichRHS={which_rhs}/{n}: Computing diagnostics")
        # v3 diagnostics are evaluated in the same global state used for each whichRHS solve.
        # Keep `op0` as default for current parity baselines, with an opt-in override.
        diag_op = op0
        env_diag_op = os.environ.get("SFINCS_JAX_TRANSPORT_DIAG_OP", "").strip().lower()
        if env_diag_op == "rhs":
            diag_op = op_rhs
        elif env_diag_op == "base":
            diag_op = op0
        diag = v3_transport_diagnostics_vm_only(diag_op, x_full=x_full)
        diag_fsab_flow.append(diag.fsab_flow)
        diag_pf.append(diag.particle_flux_vm_psi_hat)
        diag_hf.append(diag.heat_flux_vm_psi_hat)
        elapsed_s.append(jnp.asarray(t_rhs.elapsed_s(), dtype=jnp.float64))

    tm = v3_transport_matrix_from_state_vectors(op0=op0, geom=geom, state_vectors_by_rhs=state_vectors)
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
