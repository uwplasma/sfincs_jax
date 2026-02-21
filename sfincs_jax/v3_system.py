from __future__ import annotations

from dataclasses import dataclass, replace
import os
import contextlib
import weakref
from functools import lru_cache
from pathlib import Path

from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
import numpy as np
from jax import tree_util as jtu
from jax.sharding import Mesh, PartitionSpec
try:  # JAX>=0.4
    from jax import pjit as _pjit
except Exception:  # pragma: no cover
    from jax.experimental import pjit as _pjit  # type: ignore[no-redef]

from .boozer_bc import read_boozer_bc_header, selected_r_n_from_bc
from .diagnostics import b0_over_bbar as b0_over_bbar_jax
from .diagnostics import fsab_hat2 as fsab_hat2_jax
from .diagnostics import g_hat_i_hat as g_hat_i_hat_jax
from .namelist import Namelist
from .paths import resolve_existing_path
from .v3 import V3Grids, geometry_from_namelist, grids_from_namelist
from .collisionless import CollisionlessV3Operator
from .collisionless_er import ErXiDotV3Operator, ErXDotV3Operator
from .collisionless_exb import ExBThetaV3Operator, ExBZetaV3Operator
from .magnetic_drifts import (
    MagneticDriftThetaV3Operator,
    MagneticDriftZetaV3Operator,
    MagneticDriftXiDotV3Operator,
)
from .v3_fblock import V3FBlockOperator, apply_v3_fblock_operator, fblock_operator_from_namelist
from .vmec_wout import psi_a_hat_from_wout, read_vmec_wout, vmec_interpolation

_THRESHOLD_FOR_INCLUSION = 1e-12  # Matches v3 `sparsify.F90`.
_V3_DEFAULT_DELTA = 4.5694e-3  # v3 `globalVariables.F90`

_SHARDING_CONSTRAINTS_ENABLED = False


def _shard_pad_enabled() -> bool:
    env = os.environ.get("SFINCS_JAX_SHARD_PAD", "").strip().lower()
    if env in {"0", "false", "no", "off"}:
        return False
    if env in {"1", "true", "yes", "on"}:
        return True
    # Default on: allow padding to make odd grids divisible by device count.
    return True


@contextlib.contextmanager
def sharding_constraints(enabled: bool = True):
    global _SHARDING_CONSTRAINTS_ENABLED
    prev = _SHARDING_CONSTRAINTS_ENABLED
    _SHARDING_CONSTRAINTS_ENABLED = bool(enabled)
    try:
        yield
    finally:
        _SHARDING_CONSTRAINTS_ENABLED = prev


def _get_int(group: dict, key: str, default: int) -> int:
    v = group.get(key.upper(), default)
    if isinstance(v, list):
        v = v[0] if v else default
    return int(v)


def _get_bool(group: dict, key: str, default: bool = False) -> bool:
    return bool(group.get(key.upper(), default))


@jtu.register_pytree_node_class
@dataclass(frozen=True)
class V3FullSystemOperator:
    """Matrix-free operator for a subset of the full v3 linear system.

    This operator extends the F-block (distribution function) operator with the constraint rows/cols
    used to remove nullspaces and enforce moments:

    - If ``includePhi1 = .true.`` (and ``readExternalPhi1 = .false.``), the operator includes the
      quasineutrality (QN) block and the flux-surface-average constraint on ``Phi1`` (lambda),
      matching v3 `indices.F90` ordering:
      ``[F-block, Phi1(theta,zeta), lambda, constraint unknowns]``.
    - ``constraintScheme = 2`` (common default when ``collisionOperator != 0``):
      adds an L=0 source unknown at each x, and enforces flux-surface average of ``f1`` is 0 at each x.
    - ``constraintScheme = 1`` (common default when ``collisionOperator = 0``):
      adds particle+energy source unknowns per species, and enforces density and pressure moments are 0.

    Notes
    -----
    - Phi1 coupling in the **kinetic equation** is partially implemented for
      ``includePhi1InKineticEquation = .true.`` by matching the v3 whichMatrix=3 linearization.
      This requires a base ``Phi1Hat`` field (the linearization point) which is supplied via
      ``phi1_hat_base`` when constructing the operator.
    - Phi1 coupling inside the **collision operator** and the full nonlinear residual/RHS assembly
      are not yet implemented end-to-end.
    """

    fblock: V3FBlockOperator
    constraint_scheme: int
    point_at_x0: bool

    include_phi1: bool
    quasineutrality_option: int
    with_adiabatic: bool
    alpha: jnp.ndarray  # scalar
    delta: jnp.ndarray  # scalar
    adiabatic_z: jnp.ndarray  # scalar
    adiabatic_nhat: jnp.ndarray  # scalar
    adiabatic_that: jnp.ndarray  # scalar

    include_phi1_in_kinetic: bool
    dphi_hat_dpsi_hat: jnp.ndarray  # scalar
    phi1_hat_base: jnp.ndarray  # (T,Z)

    rhs_mode: int
    e_parallel_hat: jnp.ndarray  # scalar
    e_parallel_hat_spec: jnp.ndarray  # (S,)
    fsab_hat2: jnp.ndarray  # scalar

    z_s: jnp.ndarray  # (S,)
    m_hat: jnp.ndarray  # (S,)
    t_hat: jnp.ndarray  # (S,)
    n_hat: jnp.ndarray  # (S,)
    dn_hat_dpsi_hat: jnp.ndarray  # (S,)
    dt_hat_dpsi_hat: jnp.ndarray  # (S,)

    theta_weights: jnp.ndarray  # (T,)
    zeta_weights: jnp.ndarray  # (Z,)
    d_hat: jnp.ndarray  # (T,Z)
    b_hat: jnp.ndarray  # (T,Z)
    db_hat_dtheta: jnp.ndarray  # (T,Z)
    db_hat_dzeta: jnp.ndarray  # (T,Z)
    b_hat_sup_theta: jnp.ndarray  # (T,Z)
    b_hat_sup_zeta: jnp.ndarray  # (T,Z)
    b_hat_sub_theta: jnp.ndarray  # (T,Z)
    b_hat_sub_zeta: jnp.ndarray  # (T,Z)

    x: jnp.ndarray  # (X,)
    x_weights: jnp.ndarray  # (X,)
    ddx: jnp.ndarray  # (X,X)

    def tree_flatten(self):
        # Keep Python ints/bools in `aux` so the operator can be used as a JAX PyTree in JITted code.
        # Shape-affecting options are static in practice (e.g. includePhi1 changes the vector layout).
        aux = (
            int(self.constraint_scheme),
            bool(self.point_at_x0),
            bool(self.include_phi1),
            int(self.quasineutrality_option),
            bool(self.with_adiabatic),
            bool(self.include_phi1_in_kinetic),
            int(self.rhs_mode),
        )
        children = (
            self.fblock,
            self.alpha,
            self.delta,
            self.adiabatic_z,
            self.adiabatic_nhat,
            self.adiabatic_that,
            self.dphi_hat_dpsi_hat,
            self.phi1_hat_base,
            self.e_parallel_hat,
            self.e_parallel_hat_spec,
            self.fsab_hat2,
            self.z_s,
            self.m_hat,
            self.t_hat,
            self.n_hat,
            self.dn_hat_dpsi_hat,
            self.dt_hat_dpsi_hat,
            self.theta_weights,
            self.zeta_weights,
            self.d_hat,
            self.b_hat,
            self.db_hat_dtheta,
            self.db_hat_dzeta,
            self.b_hat_sup_theta,
            self.b_hat_sup_zeta,
            self.b_hat_sub_theta,
            self.b_hat_sub_zeta,
            self.x,
            self.x_weights,
            self.ddx,
        )
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        (
            constraint_scheme,
            point_at_x0,
            include_phi1,
            quasineutrality_option,
            with_adiabatic,
            include_phi1_in_kinetic,
            rhs_mode,
        ) = aux
        (
            fblock,
            alpha,
            delta,
            adiabatic_z,
            adiabatic_nhat,
            adiabatic_that,
            dphi_hat_dpsi_hat,
            phi1_hat_base,
            e_parallel_hat,
            e_parallel_hat_spec,
            fsab_hat2,
            z_s,
            m_hat,
            t_hat,
            n_hat,
            dn_hat_dpsi_hat,
            dt_hat_dpsi_hat,
            theta_weights,
            zeta_weights,
            d_hat,
            b_hat,
            db_hat_dtheta,
            db_hat_dzeta,
            b_hat_sup_theta,
            b_hat_sup_zeta,
            b_hat_sub_theta,
            b_hat_sub_zeta,
            x,
            x_weights,
            ddx,
        ) = children
        return cls(
            fblock=fblock,
            constraint_scheme=int(constraint_scheme),
            point_at_x0=bool(point_at_x0),
            include_phi1=bool(include_phi1),
            quasineutrality_option=int(quasineutrality_option),
            with_adiabatic=bool(with_adiabatic),
            alpha=alpha,
            delta=delta,
            adiabatic_z=adiabatic_z,
            adiabatic_nhat=adiabatic_nhat,
            adiabatic_that=adiabatic_that,
            include_phi1_in_kinetic=bool(include_phi1_in_kinetic),
            dphi_hat_dpsi_hat=dphi_hat_dpsi_hat,
            phi1_hat_base=phi1_hat_base,
            rhs_mode=int(rhs_mode),
            e_parallel_hat=e_parallel_hat,
            e_parallel_hat_spec=e_parallel_hat_spec,
            fsab_hat2=fsab_hat2,
            z_s=z_s,
            m_hat=m_hat,
            t_hat=t_hat,
            n_hat=n_hat,
            dn_hat_dpsi_hat=dn_hat_dpsi_hat,
            dt_hat_dpsi_hat=dt_hat_dpsi_hat,
            theta_weights=theta_weights,
            zeta_weights=zeta_weights,
            d_hat=d_hat,
            b_hat=b_hat,
            db_hat_dtheta=db_hat_dtheta,
            db_hat_dzeta=db_hat_dzeta,
            b_hat_sup_theta=b_hat_sup_theta,
            b_hat_sup_zeta=b_hat_sup_zeta,
            b_hat_sub_theta=b_hat_sub_theta,
            b_hat_sub_zeta=b_hat_sub_zeta,
            x=x,
            x_weights=x_weights,
            ddx=ddx,
        )

    @property
    def n_species(self) -> int:
        return int(self.fblock.n_species)

    @property
    def n_x(self) -> int:
        return int(self.fblock.n_x)

    @property
    def n_xi(self) -> int:
        return int(self.fblock.n_xi)

    @property
    def n_theta(self) -> int:
        return int(self.fblock.n_theta)

    @property
    def n_zeta(self) -> int:
        return int(self.fblock.n_zeta)

    @property
    def f_size(self) -> int:
        return int(self.fblock.flat_size)

    @property
    def phi1_size(self) -> int:
        if bool(self.include_phi1):
            return int(self.n_theta * self.n_zeta + 1)
        return 0

    @property
    def extra_size(self) -> int:
        if int(self.constraint_scheme) == 2:
            return int(self.n_species * self.n_x)
        if int(self.constraint_scheme) in {1, 3, 4}:
            return int(2 * self.n_species)
        if int(self.constraint_scheme) == 0:
            return 0
        raise NotImplementedError(f"constraintScheme={int(self.constraint_scheme)} is not supported.")

    @property
    def total_size(self) -> int:
        return int(self.f_size + self.phi1_size + self.extra_size)


def _ix_min(point_at_x0: bool) -> int:
    # Matches populateMatrix.F90:
    #   if (pointAtX0) ixMin = 2 else ixMin = 1   (1-based)
    return 1 if point_at_x0 else 0


def _fs_average_factor(theta_weights: jnp.ndarray, zeta_weights: jnp.ndarray, d_hat: jnp.ndarray) -> jnp.ndarray:
    return (theta_weights[:, None] * zeta_weights[None, :]) / d_hat


def _source_basis_constraint_scheme_1(x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return (xPartOfSource1, xPartOfSource2) for constraintScheme=1 and whichMatrix != 4,5."""
    x2 = x * x
    sqrt_pi = jnp.sqrt(jnp.pi)
    coef = jnp.exp(-x2) / (jnp.pi * sqrt_pi)
    s1 = (-x2 + 2.5) * coef
    s2 = ((2.0 / 3.0) * x2 - 1.0) * coef
    return s1, s2


def _nonlinear_temp_vector(
    op: V3FullSystemOperator, f: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute tempVector2-like L-coupling (out_nl) and masks for nonlinear Phi1 terms."""
    n_xi = int(op.n_xi)
    inv_x = 1.0 / op.x  # (X,)
    ddx_to_use = jnp.where(jnp.abs(op.ddx) > _THRESHOLD_FOR_INCLUSION, op.ddx, 0.0)
    ddx_f = jnp.einsum("ij,sltzj->sltzi", ddx_to_use, jnp.transpose(f, (0, 2, 3, 4, 1)))  # (S,L,T,Z,X)
    ddx_f = jnp.transpose(ddx_f, (0, 4, 1, 2, 3))  # (S,X,L,T,Z)

    out_nl = jnp.zeros_like(f, dtype=jnp.float64)
    l = jnp.arange(n_xi, dtype=jnp.float64)

    if n_xi > 1:
        lp1 = l[:-1]
        coef = (lp1 + 1.0) / (2.0 * lp1 + 3.0)
        diag_xl = (((lp1 + 1.0) * (lp1 + 2.0) / (2.0 * lp1 + 3.0))[:, None] * inv_x[None, :]).T
        src = f[:, :, 1:, :, :]
        ddx_src = ddx_f[:, :, 1:, :, :]
        term = coef[None, None, :, None, None] * ddx_src + diag_xl[None, :, :, None, None] * src
        out_nl = out_nl.at[:, :, :-1, :, :].add(term)

        lm1 = l[1:]
        coef = lm1 / (2.0 * lm1 - 1.0)
        diag_xl = ((-(lm1 - 1.0) * lm1 / (2.0 * lm1 - 1.0))[:, None] * inv_x[None, :]).T
        src = f[:, :, :-1, :, :]
        ddx_src = ddx_f[:, :, :-1, :, :]
        term = coef[None, None, :, None, None] * ddx_src + diag_xl[None, :, :, None, None] * src
        out_nl = out_nl.at[:, :, 1:, :, :].add(term)

    ix0 = _ix_min(bool(op.point_at_x0))
    mask_x = (jnp.arange(op.n_x) >= ix0).astype(jnp.float64)  # (X,)
    mask_l = (
        jnp.arange(n_xi, dtype=jnp.int32)[None, :] < op.fblock.collisionless.n_xi_for_x.astype(jnp.int32)[:, None]
    ).astype(jnp.float64)  # (X,L)
    return out_nl, mask_l, mask_x


def _nonlinear_temp_vector_phi1(
    op: V3FullSystemOperator, f: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute tempVector2 for d(kinetic)/dPhi1 terms (matches populateMatrix.F90 logic)."""
    n_xi = int(op.n_xi)
    inv_x = 1.0 / op.x  # (X,)
    ddx_to_use = jnp.where(jnp.abs(op.ddx) > _THRESHOLD_FOR_INCLUSION, op.ddx, 0.0)
    ddx_f = jnp.einsum("ij,sltzj->sltzi", ddx_to_use, jnp.transpose(f, (0, 2, 3, 4, 1)))  # (S,L,T,Z,X)
    ddx_f = jnp.transpose(ddx_f, (0, 4, 1, 2, 3))  # (S,X,L,T,Z)

    out_phi1 = jnp.zeros_like(f, dtype=jnp.float64)
    l = jnp.arange(n_xi, dtype=jnp.float64)

    if n_xi > 1:
        # For L < Nxi-1, populateMatrix.F90 overwrites tempVector2 with the L+1 contribution.
        lp1 = l[:-1]
        coef = (lp1 + 1.0) / (2.0 * lp1 + 3.0)
        diag_xl = (((lp1 + 1.0) * (lp1 + 2.0) / (2.0 * lp1 + 3.0))[:, None] * inv_x[None, :]).T
        src = f[:, :, 1:, :, :]
        ddx_src = ddx_f[:, :, 1:, :, :]
        term = coef[None, None, :, None, None] * ddx_src + diag_xl[None, :, :, None, None] * src
        out_phi1 = out_phi1.at[:, :, :-1, :, :].add(term)

        # For L = Nxi-1, populateMatrix.F90 uses only the L-1 contribution.
        l_last = int(n_xi - 1)
        coef_last = l_last / (2.0 * l_last - 1.0)
        diag_last = (-(l_last - 1.0) * l_last / (2.0 * l_last - 1.0)) * inv_x
        src_last = f[:, :, l_last - 1, :, :]
        ddx_last = ddx_f[:, :, l_last - 1, :, :]
        term_last = coef_last * ddx_last + diag_last[None, :, None, None] * src_last
        out_phi1 = out_phi1.at[:, :, l_last, :, :].add(term_last)

    ix0 = _ix_min(bool(op.point_at_x0))
    mask_x = (jnp.arange(op.n_x) >= ix0).astype(jnp.float64)  # (X,)
    mask_l = (
        jnp.arange(n_xi, dtype=jnp.int32)[None, :] < op.fblock.collisionless.n_xi_for_x.astype(jnp.int32)[:, None]
    ).astype(jnp.float64)  # (X,L)
    return out_phi1, mask_l, mask_x


def apply_v3_full_system_operator(
    op: V3FullSystemOperator,
    x_full: jnp.ndarray,
    *,
    include_jacobian_terms: bool = True,
    allow_sharding: bool = False,
) -> jnp.ndarray:
    """Apply the matrix-free full-system operator.

    By default this routine applies the linearized (`whichMatrix=3`-style) operator
    for includePhi1-in-kinetic runs, i.e. coefficients are frozen at ``phi1_hat_base``.
    Use ``include_jacobian_terms=False`` for nonlinear residual evaluations.
    """
    x_full = jnp.asarray(x_full)
    if x_full.shape != (op.total_size,):
        raise ValueError(f"x_full must have shape {(op.total_size,)}, got {x_full.shape}")

    f_flat = x_full[: op.f_size]
    rest = x_full[op.f_size :]
    f = f_flat.reshape(op.fblock.f_shape)
    shard_axis = _matvec_shard_axis(op) if (allow_sharding and _SHARDING_CONSTRAINTS_ENABLED) else None
    use_sharding = shard_axis in {"theta", "zeta", "x"} and (jax.device_count() > 1)
    if use_sharding:
        n_devices = jax.local_device_count()
        if n_devices <= 1:
            use_sharding = False
        elif shard_axis == "theta" and int(op.n_theta) % n_devices != 0:
            use_sharding = False
        elif shard_axis == "zeta" and int(op.n_zeta) % n_devices != 0:
            use_sharding = False
        elif shard_axis == "x" and int(op.n_x) % n_devices != 0:
            use_sharding = False
    if use_sharding and shard_axis == "theta":
        f = jax.lax.with_sharding_constraint(f, PartitionSpec(None, None, None, "theta", None))
    elif use_sharding and shard_axis == "zeta":
        f = jax.lax.with_sharding_constraint(f, PartitionSpec(None, None, None, None, "zeta"))
    elif use_sharding and shard_axis == "x":
        f = jax.lax.with_sharding_constraint(f, PartitionSpec(None, "x", None, None, None))

    phi1 = None
    lam = None
    extra = rest
    if op.include_phi1:
        phi1_flat = rest[: op.n_theta * op.n_zeta]
        lam = rest[op.n_theta * op.n_zeta]
        extra = rest[op.phi1_size :]
        phi1 = phi1_flat.reshape((op.n_theta, op.n_zeta))
        if use_sharding and shard_axis == "theta":
            phi1 = jax.lax.with_sharding_constraint(phi1, PartitionSpec("theta", None))
        elif use_sharding and shard_axis == "zeta":
            phi1 = jax.lax.with_sharding_constraint(phi1, PartitionSpec(None, "zeta"))

    phi1_for_collisions = None
    if op.fblock.fp_phi1 is not None:
        if include_jacobian_terms:
            phi1_for_collisions = op.phi1_hat_base
        else:
            phi1_for_collisions = phi1 if phi1 is not None else op.phi1_hat_base

    y_f = apply_v3_fblock_operator(op.fblock, f, phi1_hat_base=phi1_for_collisions)
    factor = _fs_average_factor(op.theta_weights, op.zeta_weights, op.d_hat)  # (T,Z)
    ix0 = _ix_min(op.point_at_x0)

    y_phi1 = jnp.zeros((0,), dtype=jnp.float64)
    if op.include_phi1:

        # Quasineutrality equation block (in v3, this is appended after the DKE rows).
        # For the linear subset we currently support, this block includes:
        #   - charge density from f1 (L=0)
        #   - a diagonal phi1 term for quasineutralityOption=2 with adiabatic response
        #   - the lambda Lagrange multiplier
        x2w = (op.x * op.x) * op.x_weights  # (X,)
        species_factor = 4.0 * jnp.pi * op.z_s * op.t_hat / op.m_hat * jnp.sqrt(op.t_hat / op.m_hat)  # (S,)

        if int(op.quasineutrality_option) == 2:
            # EUTERPE equations: only the first kinetic species appears in QN.
            qn_from_f = species_factor[0] * jnp.einsum("x,xtz->tz", x2w, f[0, :, 0, :, :])
        else:
            qn_from_f = jnp.einsum("s,x,sxtz->tz", species_factor, x2w, f[:, :, 0, :, :])

        if int(op.quasineutrality_option) == 1:
            # Nonlinear QN RHS terms live in evaluateResidual.F90 (not in the matrix).
            # Therefore: residual uses qn_from_f + lam, while Jacobian uses qn_from_f + diag*phi1 + lam.
            if include_jacobian_terms:
                exp_phi = jnp.exp(
                    -(op.z_s[:, None, None] * op.alpha / op.t_hat[:, None, None]) * op.phi1_hat_base[None, :, :]
                )
                diag = -jnp.sum(
                    (op.z_s * op.z_s * op.alpha * op.n_hat / op.t_hat)[:, None, None] * exp_phi, axis=0
                )
                if op.with_adiabatic:
                    diag = diag - (
                        (op.adiabatic_z * op.adiabatic_z * op.alpha * op.adiabatic_nhat / op.adiabatic_that)
                        * jnp.exp(-(op.adiabatic_z * op.alpha / op.adiabatic_that) * op.phi1_hat_base)
                    )
                diag_scale_env = os.environ.get("SFINCS_JAX_PHI1_QN_DIAG_SCALE", "").strip()
                if diag_scale_env:
                    try:
                        diag = diag * float(diag_scale_env)
                    except ValueError:
                        pass
                qn = qn_from_f + diag * phi1 + lam
            else:
                qn = qn_from_f + lam
        else:
            phi1_diag = 0.0
            if int(op.quasineutrality_option) == 2 and op.with_adiabatic and op.n_species > 0:
                phi1_diag = -op.alpha * (
                    (op.z_s[0] * op.z_s[0]) * op.n_hat[0] / op.t_hat[0]
                    + (op.adiabatic_z * op.adiabatic_z) * op.adiabatic_nhat / op.adiabatic_that
                )
            qn = qn_from_f + phi1_diag * phi1 + lam

        # <Phi1> = 0 constraint row ("lambda row"):
        y_lam = jnp.sum(factor * phi1)

        y_phi1 = jnp.concatenate([qn.reshape((-1,)), jnp.asarray([y_lam])], axis=0)
    else:
        extra = rest
        phi1 = None

    if op.include_phi1 and op.include_phi1_in_kinetic:
        # Parity-first subset of the Phi1-in-kinetic-equation couplings (v3):
        # add terms proportional to dPhi1/dtheta and dPhi1/dzeta into the L=0 DKE rows.
        #
        # We implement the matrix action corresponding to the blocks:
        #   "Add the inhomogeneous drive term multiplied by exp(-Ze Phi1 / T)"
        # and related linear Phi1-gradient terms in populateMatrix.F90.
        #
        # Notes:
        # - For residual evaluations we use the current Phi1; for Jacobian matvecs we use
        #   the frozen base-state Phi1 to match v3's linearization behavior.
        assert phi1 is not None
        ddtheta = op.fblock.collisionless.ddtheta
        ddzeta = op.fblock.collisionless.ddzeta
        dphi1_dtheta = ddtheta @ phi1  # (T,Z)
        dphi1_dzeta = phi1 @ ddzeta.T  # (T,Z)
        dphi1_base_dtheta = ddtheta @ op.phi1_hat_base
        dphi1_base_dzeta = op.phi1_hat_base @ ddzeta.T

        # Nonlinear term in the v3 residual that is linear in f but proportional to grad(Phi1Hat).
        # For Jacobian matvecs, use the base Phi1 gradients; for residual evaluation, use the current Phi1.
        if include_jacobian_terms:
            e_term = op.b_hat_sup_theta * dphi1_base_dtheta + op.b_hat_sup_zeta * dphi1_base_dzeta  # (T,Z)
        else:
            e_term = op.b_hat_sup_theta * dphi1_dtheta + op.b_hat_sup_zeta * dphi1_dzeta  # (T,Z)
        nonlinear_factor = (
            -(op.alpha * op.z_s)[:, None, None]
            / (2.0 * op.b_hat[None, :, :] * jnp.sqrt(op.t_hat)[:, None, None] * jnp.sqrt(op.m_hat)[:, None, None])
            * e_term[None, :, :]
        )  # (S,T,Z)

        out_nl, mask, mask_x = _nonlinear_temp_vector(op, f)
        y_f = y_f + out_nl * nonlinear_factor[:, None, None, :, :] * mask[None, :, :, None, None] * mask_x[None, :, None, None, None]

        # Jacobian d(kinetic)/dPhi1 term associated with the nonlinear operator is handled
        # in apply_v3_full_system_jacobian so it can use the base-state f.

        x2 = op.x * op.x  # (X,)
        expx2 = jnp.exp(-x2)  # (X,)

        sqrt_pi = jnp.sqrt(jnp.pi)
        norm = jnp.pi * sqrt_pi

        phi1_use = op.phi1_hat_base if include_jacobian_terms else phi1
        exp_phi = jnp.exp(
            -(op.z_s[:, None, None] * op.alpha / op.t_hat[:, None, None]) * phi1_use[None, :, :]
        )  # (S,T,Z)

        # Species-dependent Maxwellian normalization (v3):
        #   nHat * (mHat*sqrt(mHat)) / (THat*sqrt(THat)*pi*sqrt(pi))
        sp_pref1 = op.n_hat * (op.m_hat * jnp.sqrt(op.m_hat)) / (op.t_hat * jnp.sqrt(op.t_hat) * norm)  # (S,)

        bracket = (op.dn_hat_dpsi_hat / op.n_hat)[:, None] + (x2[None, :] - 1.5) * (
            op.dt_hat_dpsi_hat / op.t_hat
        )[:, None]  # (S,X)
        fm = sp_pref1[:, None] * expx2[None, :] * bracket  # (S,X)

        geom_theta = -op.alpha * op.delta * op.d_hat * op.b_hat_sub_zeta / (2.0 * (op.b_hat * op.b_hat))  # (T,Z)
        geom_zeta = op.alpha * op.delta * op.d_hat * op.b_hat_sub_theta / (2.0 * (op.b_hat * op.b_hat))  # (T,Z)

        coeff1_theta = fm[:, :, None, None] * geom_theta[None, None, :, :] * exp_phi[:, None, :, :]  # (S,X,T,Z)
        coeff1_zeta = fm[:, :, None, None] * geom_zeta[None, None, :, :] * exp_phi[:, None, :, :]  # (S,X,T,Z)

        # factor2 term from populateMatrix.F90 (adds an extra piece proportional to dPhiHatdpsiHat + Phi1Hat*dTHatdpsiHat/THat)
        sp_pref2 = op.z_s * op.n_hat * (op.m_hat * jnp.sqrt(op.m_hat)) / (op.t_hat * op.t_hat * jnp.sqrt(op.t_hat))  # (S,)
        phi_term = op.dphi_hat_dpsi_hat + phi1_use[None, :, :] * (op.dt_hat_dpsi_hat / op.t_hat)[:, None, None]  # (S,T,Z)

        geom2_theta = -(op.alpha * op.alpha) * op.delta * op.d_hat * op.b_hat_sub_zeta / (
            2.0 * norm * (op.b_hat * op.b_hat)
        )  # (T,Z)
        geom2_zeta = (op.alpha * op.alpha) * op.delta * op.d_hat * op.b_hat_sub_theta / (
            2.0 * norm * (op.b_hat * op.b_hat)
        )  # (T,Z)

        coeff2_theta = (
            sp_pref2[:, None, None, None]
            * expx2[None, :, None, None]
            * exp_phi[:, None, :, :]
            * phi_term[:, None, :, :]
            * geom2_theta[None, None, :, :]
        )
        coeff2_zeta = (
            sp_pref2[:, None, None, None]
            * expx2[None, :, None, None]
            * exp_phi[:, None, :, :]
            * phi_term[:, None, :, :]
            * geom2_zeta[None, None, :, :]
        )

        y_f = y_f.at[:, :, 0, :, :].add((coeff1_theta + coeff2_theta) * dphi1_dtheta[None, None, :, :])
        y_f = y_f.at[:, :, 0, :, :].add((coeff1_zeta + coeff2_zeta) * dphi1_dzeta[None, None, :, :])

    if int(op.constraint_scheme) == 0:
        y_extra = jnp.zeros((0,), dtype=jnp.float64)

    elif int(op.constraint_scheme) == 2:
        # Unknowns: per-species per-x L=0 source (constant on the flux surface).
        src = extra.reshape((op.n_species, op.n_x))  # (S,X)

        # DKE rows: add the source into L=0 for ix>=ixMin.
        y_f = y_f.at[:, ix0:, 0, :, :].add(src[:, ix0:, None, None])

        # Constraint rows: y = <f> at each x (L=0), with special handling for pointAtX0.
        # y[s,ix] = Σ_{θ,ζ} factor(θ,ζ) * f[s,ix,L=0,θ,ζ]
        y_avg = jnp.einsum("tz,sxtz->sx", factor, f[:, :, 0, :, :])
        if op.point_at_x0:
            y_avg = y_avg.at[:, 0].set(src[:, 0])
        y_extra = y_avg.reshape((-1,))

    elif int(op.constraint_scheme) in {1, 3, 4}:
        if int(op.constraint_scheme) != 1:
            raise NotImplementedError("Only constraintScheme=1 is implemented in sfincs_jax so far.")
        # Unknowns: per-species (particle source, energy source).
        src = extra.reshape((op.n_species, 2))  # (S,2)
        src_p = src[:, 0]
        src_e = src[:, 1]

        # DKE rows: add source basis functions at L=0.
        xpart1, xpart2 = _source_basis_constraint_scheme_1(op.x)
        y_f = y_f.at[:, ix0:, 0, :, :].add(
            xpart1[ix0:][None, :, None, None] * src_p[:, None, None, None]
            + xpart2[ix0:][None, :, None, None] * src_e[:, None, None, None]
        )

        # Constraint rows: density and pressure moments are zero (L=0 only).
        x2 = op.x * op.x
        x4 = x2 * x2
        w2 = x2 * op.x_weights
        w4 = x4 * op.x_weights

        # y_dens[s] = Σ_{x,θ,ζ} w2[x] * factor[θ,ζ] * f[s,x,L=0,θ,ζ]
        # y_pres[s] = Σ_{x,θ,ζ} w4[x] * factor[θ,ζ] * f[s,x,L=0,θ,ζ]
        y_dens = jnp.einsum("x,tz,sxtz->s", w2, factor, f[:, :, 0, :, :])
        y_pres = jnp.einsum("x,tz,sxtz->s", w4, factor, f[:, :, 0, :, :])
        y_extra = jnp.stack([y_dens, y_pres], axis=1).reshape((-1,))

    else:
        raise NotImplementedError(f"constraintScheme={int(op.constraint_scheme)} is not supported.")

    return jnp.concatenate([y_f.reshape((-1,)), y_phi1, y_extra], axis=0)


def apply_v3_full_system_jacobian(
    op: V3FullSystemOperator, x_state: jnp.ndarray, dx: jnp.ndarray
) -> jnp.ndarray:
    """Apply the Jacobian at x_state to dx, using matrix-free parity terms."""
    dx = jnp.asarray(dx)
    x_state = jnp.asarray(x_state)
    if dx.shape != (op.total_size,) or x_state.shape != (op.total_size,):
        raise ValueError("x_state and dx must have shape (total_size,)")

    # Start from the linearized operator with Jacobian terms (this includes factorJ1..5).
    y = apply_v3_full_system_operator(op, dx, include_jacobian_terms=True, allow_sharding=False)

    if op.include_phi1 and op.include_phi1_in_kinetic:
        # Replace the d(kinetic)/dPhi1 nonlinear-term contribution so it uses the base f_state.
        f_state = x_state[: op.f_size].reshape(op.fblock.f_shape)

        phi1_flat = dx[op.f_size : op.f_size + op.n_theta * op.n_zeta]
        phi1 = phi1_flat.reshape((op.n_theta, op.n_zeta))
        ddtheta = op.fblock.collisionless.ddtheta
        ddzeta = op.fblock.collisionless.ddzeta
        dphi1_dtheta = ddtheta @ phi1
        dphi1_dzeta = phi1 @ ddzeta.T
        phi1_grad = op.b_hat_sup_theta * dphi1_dtheta + op.b_hat_sup_zeta * dphi1_dzeta  # (T,Z)

        factor_j = (
            -(op.alpha * op.z_s)[:, None, None]
            / (2.0 * op.b_hat[None, :, :] * jnp.sqrt(op.t_hat)[:, None, None] * jnp.sqrt(op.m_hat)[:, None, None])
        )  # (S,T,Z)

        out_state, mask_l_state, mask_x_state = _nonlinear_temp_vector_phi1(op, f_state)
        term_state = out_state * factor_j[:, None, None, :, :] * phi1_grad[None, None, None, :, :] * mask_l_state[None, :, :, None, None] * mask_x_state[None, :, None, None, None]

        y_f = y[: op.f_size].reshape(op.fblock.f_shape)
        y_f = y_f + term_state

        # Add Jacobian-only diagonal d(kinetic)/dPhi1 terms from populateMatrix.F90 factorJ1..J5.
        # These are present in whichMatrix=1/0 but not in whichMatrix=3.
        phi1_base = op.phi1_hat_base
        phi1_perturb = phi1  # (T,Z)

        x = op.x
        x2 = x * x
        expx2 = jnp.exp(-x2)
        x2_expx2 = x2 * expx2
        sqrt_pi = jnp.sqrt(jnp.pi)
        norm = jnp.pi * sqrt_pi
        two_pi = jnp.asarray(2.0 * jnp.pi, dtype=jnp.float64)

        z = op.z_s
        n_hat = op.n_hat
        m_hat = op.m_hat
        t_hat = op.t_hat
        dn = op.dn_hat_dpsi_hat
        dt = op.dt_hat_dpsi_hat

        dphi_hat_dpsi_hat_to_use = jnp.where(
            (op.rhs_mode == 1) | (op.rhs_mode > 3),
            op.dphi_hat_dpsi_hat,
            jnp.asarray(0.0, dtype=jnp.float64),
        )

        x_part = x2_expx2[None, :] * (
            dn[:, None] / n_hat[:, None]
            + (op.alpha * z / t_hat)[:, None] * dphi_hat_dpsi_hat_to_use
            + (x2[None, :] - 1.5) * (dt / t_hat)[:, None]
        )  # (S,X)
        x_part2 = x2_expx2[None, :] * (dt / (t_hat * t_hat))[:, None]  # (S,X)

        exp_phi = jnp.exp(
            -(z[:, None, None] * op.alpha / t_hat[:, None, None]) * phi1_base[None, :, :]
        )  # (S,T,Z)
        geom_b = (
            (-op.b_hat_sub_zeta * op.db_hat_dtheta + op.b_hat_sub_theta * op.db_hat_dzeta)
            * op.d_hat
            / (op.b_hat * op.b_hat * op.b_hat)
        )  # (T,Z)

        pref_common = op.delta * n_hat * m_hat * jnp.sqrt(m_hat) / (two_pi * sqrt_pi * jnp.sqrt(t_hat))  # (S,)
        factor_j3 = (
            (pref_common / z)[:, None, None, None]
            * geom_b[None, None, :, :]
            * (
                x_part[:, :, None, None]
                + x_part2[:, :, None, None] * (z * op.alpha)[:, None, None, None] * phi1_base[None, None, :, :]
            )
            * exp_phi[:, None, :, :]
        )  # (S,X,T,Z)
        factor_j5 = (
            pref_common[:, None, None, None]
            * geom_b[None, None, :, :]
            * x_part2[:, :, None, None]
            * op.alpha
            * exp_phi[:, None, :, :]
        )  # (S,X,T,Z)

        ddtheta = op.fblock.collisionless.ddtheta
        ddzeta = op.fblock.collisionless.ddzeta
        dphi1_base_dtheta = ddtheta @ phi1_base
        dphi1_base_dzeta = phi1_base @ ddzeta.T
        geom_phi = (
            -op.b_hat_sub_zeta * dphi1_base_dtheta + op.b_hat_sub_theta * dphi1_base_dzeta
        )  # (T,Z)

        sp_pref1 = n_hat * (m_hat * jnp.sqrt(m_hat)) / (t_hat * jnp.sqrt(t_hat) * norm)  # (S,)
        bracket_no_phi = (dn / n_hat)[:, None] + (x2[None, :] - 1.5) * (dt / t_hat)[:, None]  # (S,X)
        factor_j1 = (
            (op.alpha * op.delta * op.d_hat / (2.0 * op.b_hat * op.b_hat))[None, None, :, :]
            * geom_phi[None, None, :, :]
            * sp_pref1[:, None, None, None]
            * expx2[None, :, None, None]
            * bracket_no_phi[:, :, None, None]
            * exp_phi[:, None, :, :]
        )  # (S,X,T,Z)

        sp_pref2 = z * n_hat * (m_hat * jnp.sqrt(m_hat)) / (t_hat * t_hat * jnp.sqrt(t_hat))  # (S,)
        phi_term = dphi_hat_dpsi_hat_to_use + phi1_base[None, :, :] * (dt / t_hat)[:, None, None]  # (S,T,Z)
        factor_j2_pref = (op.alpha * op.alpha * op.delta * op.d_hat) / (2.0 * norm * (op.b_hat * op.b_hat))  # (T,Z)
        factor_j2 = (
            factor_j2_pref[None, None, :, :]
            * geom_phi[None, None, :, :]
            * sp_pref2[:, None, None, None]
            * expx2[None, :, None, None]
            * exp_phi[:, None, :, :]
            * phi_term[:, None, :, :]
        )  # (S,X,T,Z)
        factor_j4 = (
            factor_j2_pref[None, None, :, :]
            * geom_phi[None, None, :, :]
            * sp_pref2[:, None, None, None]
            * expx2[None, :, None, None]
            * exp_phi[:, None, :, :]
            * (dt / t_hat)[:, None, None, None]
        )  # (S,X,T,Z)

        j35 = (-(z * op.alpha / t_hat)[:, None, None, None] * factor_j3 + factor_j5)  # (S,X,T,Z)
        coeff_l0_diag = (
            -(z * op.alpha / t_hat)[:, None, None, None] * (factor_j1 + factor_j2)
            + factor_j4
            + (4.0 / 3.0) * j35
        )
        coeff_l2_diag = (2.0 / 3.0) * j35

        ix0 = _ix_min(bool(op.point_at_x0))
        mask_x = (jnp.arange(op.n_x) >= ix0).astype(jnp.float64)  # (X,)
        nxi_for_x = op.fblock.collisionless.n_xi_for_x.astype(jnp.int32)
        mask_l0 = (nxi_for_x > 0).astype(jnp.float64) * mask_x
        mask_l2 = (nxi_for_x > 2).astype(jnp.float64) * mask_x

        y_f = y_f.at[:, :, 0, :, :].add(
            coeff_l0_diag * phi1_perturb[None, None, :, :] * mask_l0[None, :, None, None]
        )
        if op.n_xi > 2:
            y_f = y_f.at[:, :, 2, :, :].add(
                coeff_l2_diag * phi1_perturb[None, None, :, :] * mask_l2[None, :, None, None]
            )

        y = jnp.concatenate([y_f.reshape((-1,)), y[op.f_size :]], axis=0)

    return y


apply_v3_full_system_operator_jit = jax.jit(
    apply_v3_full_system_operator,
    static_argnames=("include_jacobian_terms",),
)

apply_v3_full_system_jacobian_jit = jax.jit(apply_v3_full_system_jacobian)


def _operator_signature(op: V3FullSystemOperator) -> tuple[object, ...]:
    return (
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
        bool(op.point_at_x0),
    )


_OPERATOR_SIGNATURE_CACHE: dict[int, tuple[weakref.ReferenceType[V3FullSystemOperator], tuple[object, ...]]] = {}
_PADDED_OPERATOR_CACHE: dict[
    tuple[int, str, int], tuple[weakref.ReferenceType[V3FullSystemOperator], V3FullSystemOperator]
] = {}


def _operator_signature_cached(op: V3FullSystemOperator) -> tuple[object, ...]:
    key = id(op)
    cached = _OPERATOR_SIGNATURE_CACHE.get(key)
    if cached is not None:
        ref, sig = cached
        if ref() is op:
            return sig
    sig = _operator_signature(op)
    _OPERATOR_SIGNATURE_CACHE[key] = (weakref.ref(op), sig)
    return sig


def _pad_1d(arr: jnp.ndarray, pad: int, *, fill: float = 0.0) -> jnp.ndarray:
    if pad <= 0:
        return arr
    return jnp.pad(arr, ((0, pad),), constant_values=fill)


def _pad_square(arr: jnp.ndarray, pad: int) -> jnp.ndarray:
    if pad <= 0:
        return arr
    return jnp.pad(arr, ((0, pad), (0, pad)), constant_values=0.0)


def _pad_2d_theta(arr: jnp.ndarray, pad: int, *, fill: float = 0.0) -> jnp.ndarray:
    if pad <= 0:
        return arr
    return jnp.pad(arr, ((0, pad), (0, 0)), constant_values=fill)


def _pad_2d_zeta(arr: jnp.ndarray, pad: int, *, fill: float = 0.0) -> jnp.ndarray:
    if pad <= 0:
        return arr
    return jnp.pad(arr, ((0, 0), (0, pad)), constant_values=fill)


def _pad_2d(arr: jnp.ndarray, *, axis: str, pad: int, fill: float = 0.0) -> jnp.ndarray:
    if axis == "theta":
        return _pad_2d_theta(arr, pad, fill=fill)
    if axis == "zeta":
        return _pad_2d_zeta(arr, pad, fill=fill)
    return arr


def _pad_collisionless(op: CollisionlessV3Operator, *, axis: str, pad: int) -> CollisionlessV3Operator:
    if pad <= 0:
        return op
    if axis == "theta":
        return replace(
            op,
            ddtheta=_pad_square(op.ddtheta, pad),
            b_hat=_pad_2d_theta(op.b_hat, pad, fill=1.0),
            b_hat_sup_theta=_pad_2d_theta(op.b_hat_sup_theta, pad, fill=0.0),
            b_hat_sup_zeta=_pad_2d_theta(op.b_hat_sup_zeta, pad, fill=0.0),
            db_hat_dtheta=_pad_2d_theta(op.db_hat_dtheta, pad, fill=0.0),
            db_hat_dzeta=_pad_2d_theta(op.db_hat_dzeta, pad, fill=0.0),
        )
    if axis == "zeta":
        return replace(
            op,
            ddzeta=_pad_square(op.ddzeta, pad),
            b_hat=_pad_2d_zeta(op.b_hat, pad, fill=1.0),
            b_hat_sup_theta=_pad_2d_zeta(op.b_hat_sup_theta, pad, fill=0.0),
            b_hat_sup_zeta=_pad_2d_zeta(op.b_hat_sup_zeta, pad, fill=0.0),
            db_hat_dtheta=_pad_2d_zeta(op.db_hat_dtheta, pad, fill=0.0),
            db_hat_dzeta=_pad_2d_zeta(op.db_hat_dzeta, pad, fill=0.0),
        )
    return op


def _pad_exb_theta(op: ExBThetaV3Operator, *, axis: str, pad: int) -> ExBThetaV3Operator:
    if pad <= 0:
        return op
    if axis == "theta":
        return replace(
            op,
            ddtheta=_pad_square(op.ddtheta, pad),
            d_hat=_pad_2d_theta(op.d_hat, pad, fill=0.0),
            b_hat=_pad_2d_theta(op.b_hat, pad, fill=1.0),
            b_hat_sub_zeta=_pad_2d_theta(op.b_hat_sub_zeta, pad, fill=0.0),
        )
    if axis == "zeta":
        return replace(
            op,
            d_hat=_pad_2d_zeta(op.d_hat, pad, fill=0.0),
            b_hat=_pad_2d_zeta(op.b_hat, pad, fill=1.0),
            b_hat_sub_zeta=_pad_2d_zeta(op.b_hat_sub_zeta, pad, fill=0.0),
        )
    return op


def _pad_exb_zeta(op: ExBZetaV3Operator, *, axis: str, pad: int) -> ExBZetaV3Operator:
    if pad <= 0:
        return op
    if axis == "zeta":
        return replace(
            op,
            ddzeta=_pad_square(op.ddzeta, pad),
            d_hat=_pad_2d_zeta(op.d_hat, pad, fill=0.0),
            b_hat=_pad_2d_zeta(op.b_hat, pad, fill=1.0),
            b_hat_sub_theta=_pad_2d_zeta(op.b_hat_sub_theta, pad, fill=0.0),
        )
    if axis == "theta":
        return replace(
            op,
            d_hat=_pad_2d_theta(op.d_hat, pad, fill=0.0),
            b_hat=_pad_2d_theta(op.b_hat, pad, fill=1.0),
            b_hat_sub_theta=_pad_2d_theta(op.b_hat_sub_theta, pad, fill=0.0),
        )
    return op


def _pad_magdrift_theta(
    op: MagneticDriftThetaV3Operator, *, axis: str, pad: int
) -> MagneticDriftThetaV3Operator:
    if pad <= 0:
        return op
    if axis == "theta":
        return replace(
            op,
            ddtheta_plus=_pad_square(op.ddtheta_plus, pad),
            ddtheta_minus=_pad_square(op.ddtheta_minus, pad),
            d_hat=_pad_2d_theta(op.d_hat, pad, fill=0.0),
            b_hat=_pad_2d_theta(op.b_hat, pad, fill=1.0),
            b_hat_sub_zeta=_pad_2d_theta(op.b_hat_sub_zeta, pad, fill=0.0),
            b_hat_sub_psi=_pad_2d_theta(op.b_hat_sub_psi, pad, fill=0.0),
            db_hat_dzeta=_pad_2d_theta(op.db_hat_dzeta, pad, fill=0.0),
            db_hat_dpsi_hat=_pad_2d_theta(op.db_hat_dpsi_hat, pad, fill=0.0),
            db_hat_sub_psi_dzeta=_pad_2d_theta(op.db_hat_sub_psi_dzeta, pad, fill=0.0),
            db_hat_sub_zeta_dpsi_hat=_pad_2d_theta(op.db_hat_sub_zeta_dpsi_hat, pad, fill=0.0),
        )
    if axis == "zeta":
        return replace(
            op,
            d_hat=_pad_2d_zeta(op.d_hat, pad, fill=0.0),
            b_hat=_pad_2d_zeta(op.b_hat, pad, fill=1.0),
            b_hat_sub_zeta=_pad_2d_zeta(op.b_hat_sub_zeta, pad, fill=0.0),
            b_hat_sub_psi=_pad_2d_zeta(op.b_hat_sub_psi, pad, fill=0.0),
            db_hat_dzeta=_pad_2d_zeta(op.db_hat_dzeta, pad, fill=0.0),
            db_hat_dpsi_hat=_pad_2d_zeta(op.db_hat_dpsi_hat, pad, fill=0.0),
            db_hat_sub_psi_dzeta=_pad_2d_zeta(op.db_hat_sub_psi_dzeta, pad, fill=0.0),
            db_hat_sub_zeta_dpsi_hat=_pad_2d_zeta(op.db_hat_sub_zeta_dpsi_hat, pad, fill=0.0),
        )
    return op


def _pad_magdrift_zeta(
    op: MagneticDriftZetaV3Operator, *, axis: str, pad: int
) -> MagneticDriftZetaV3Operator:
    if pad <= 0:
        return op
    if axis == "zeta":
        return replace(
            op,
            ddzeta_plus=_pad_square(op.ddzeta_plus, pad),
            ddzeta_minus=_pad_square(op.ddzeta_minus, pad),
            d_hat=_pad_2d_zeta(op.d_hat, pad, fill=0.0),
            b_hat=_pad_2d_zeta(op.b_hat, pad, fill=1.0),
            b_hat_sub_theta=_pad_2d_zeta(op.b_hat_sub_theta, pad, fill=0.0),
            b_hat_sub_psi=_pad_2d_zeta(op.b_hat_sub_psi, pad, fill=0.0),
            db_hat_dtheta=_pad_2d_zeta(op.db_hat_dtheta, pad, fill=0.0),
            db_hat_dpsi_hat=_pad_2d_zeta(op.db_hat_dpsi_hat, pad, fill=0.0),
            db_hat_sub_theta_dpsi_hat=_pad_2d_zeta(op.db_hat_sub_theta_dpsi_hat, pad, fill=0.0),
            db_hat_sub_psi_dtheta=_pad_2d_zeta(op.db_hat_sub_psi_dtheta, pad, fill=0.0),
        )
    if axis == "theta":
        return replace(
            op,
            d_hat=_pad_2d_theta(op.d_hat, pad, fill=0.0),
            b_hat=_pad_2d_theta(op.b_hat, pad, fill=1.0),
            b_hat_sub_theta=_pad_2d_theta(op.b_hat_sub_theta, pad, fill=0.0),
            b_hat_sub_psi=_pad_2d_theta(op.b_hat_sub_psi, pad, fill=0.0),
            db_hat_dtheta=_pad_2d_theta(op.db_hat_dtheta, pad, fill=0.0),
            db_hat_dpsi_hat=_pad_2d_theta(op.db_hat_dpsi_hat, pad, fill=0.0),
            db_hat_sub_theta_dpsi_hat=_pad_2d_theta(op.db_hat_sub_theta_dpsi_hat, pad, fill=0.0),
            db_hat_sub_psi_dtheta=_pad_2d_theta(op.db_hat_sub_psi_dtheta, pad, fill=0.0),
        )
    return op


def _pad_magdrift_xidot(
    op: MagneticDriftXiDotV3Operator, *, axis: str, pad: int
) -> MagneticDriftXiDotV3Operator:
    if pad <= 0:
        return op
    if axis == "theta":
        return replace(
            op,
            d_hat=_pad_2d_theta(op.d_hat, pad, fill=0.0),
            b_hat=_pad_2d_theta(op.b_hat, pad, fill=1.0),
            db_hat_dtheta=_pad_2d_theta(op.db_hat_dtheta, pad, fill=0.0),
            db_hat_dzeta=_pad_2d_theta(op.db_hat_dzeta, pad, fill=0.0),
            db_hat_sub_psi_dzeta=_pad_2d_theta(op.db_hat_sub_psi_dzeta, pad, fill=0.0),
            db_hat_sub_zeta_dpsi_hat=_pad_2d_theta(op.db_hat_sub_zeta_dpsi_hat, pad, fill=0.0),
            db_hat_sub_theta_dpsi_hat=_pad_2d_theta(op.db_hat_sub_theta_dpsi_hat, pad, fill=0.0),
            db_hat_sub_psi_dtheta=_pad_2d_theta(op.db_hat_sub_psi_dtheta, pad, fill=0.0),
        )
    if axis == "zeta":
        return replace(
            op,
            d_hat=_pad_2d_zeta(op.d_hat, pad, fill=0.0),
            b_hat=_pad_2d_zeta(op.b_hat, pad, fill=1.0),
            db_hat_dtheta=_pad_2d_zeta(op.db_hat_dtheta, pad, fill=0.0),
            db_hat_dzeta=_pad_2d_zeta(op.db_hat_dzeta, pad, fill=0.0),
            db_hat_sub_psi_dzeta=_pad_2d_zeta(op.db_hat_sub_psi_dzeta, pad, fill=0.0),
            db_hat_sub_zeta_dpsi_hat=_pad_2d_zeta(op.db_hat_sub_zeta_dpsi_hat, pad, fill=0.0),
            db_hat_sub_theta_dpsi_hat=_pad_2d_zeta(op.db_hat_sub_theta_dpsi_hat, pad, fill=0.0),
            db_hat_sub_psi_dtheta=_pad_2d_zeta(op.db_hat_sub_psi_dtheta, pad, fill=0.0),
        )
    return op


def _pad_er_xidot(op: ErXiDotV3Operator, *, axis: str, pad: int) -> ErXiDotV3Operator:
    if pad <= 0:
        return op
    if axis == "theta":
        return replace(
            op,
            d_hat=_pad_2d_theta(op.d_hat, pad, fill=0.0),
            b_hat=_pad_2d_theta(op.b_hat, pad, fill=1.0),
            b_hat_sub_theta=_pad_2d_theta(op.b_hat_sub_theta, pad, fill=0.0),
            b_hat_sub_zeta=_pad_2d_theta(op.b_hat_sub_zeta, pad, fill=0.0),
            db_hat_dtheta=_pad_2d_theta(op.db_hat_dtheta, pad, fill=0.0),
            db_hat_dzeta=_pad_2d_theta(op.db_hat_dzeta, pad, fill=0.0),
        )
    if axis == "zeta":
        return replace(
            op,
            d_hat=_pad_2d_zeta(op.d_hat, pad, fill=0.0),
            b_hat=_pad_2d_zeta(op.b_hat, pad, fill=1.0),
            b_hat_sub_theta=_pad_2d_zeta(op.b_hat_sub_theta, pad, fill=0.0),
            b_hat_sub_zeta=_pad_2d_zeta(op.b_hat_sub_zeta, pad, fill=0.0),
            db_hat_dtheta=_pad_2d_zeta(op.db_hat_dtheta, pad, fill=0.0),
            db_hat_dzeta=_pad_2d_zeta(op.db_hat_dzeta, pad, fill=0.0),
        )
    return op


def _pad_er_xdot(op: ErXDotV3Operator, *, axis: str, pad: int) -> ErXDotV3Operator:
    if pad <= 0:
        return op
    if axis == "theta":
        return replace(
            op,
            d_hat=_pad_2d_theta(op.d_hat, pad, fill=0.0),
            b_hat=_pad_2d_theta(op.b_hat, pad, fill=1.0),
            b_hat_sub_theta=_pad_2d_theta(op.b_hat_sub_theta, pad, fill=0.0),
            b_hat_sub_zeta=_pad_2d_theta(op.b_hat_sub_zeta, pad, fill=0.0),
            db_hat_dtheta=_pad_2d_theta(op.db_hat_dtheta, pad, fill=0.0),
            db_hat_dzeta=_pad_2d_theta(op.db_hat_dzeta, pad, fill=0.0),
        )
    if axis == "zeta":
        return replace(
            op,
            d_hat=_pad_2d_zeta(op.d_hat, pad, fill=0.0),
            b_hat=_pad_2d_zeta(op.b_hat, pad, fill=1.0),
            b_hat_sub_theta=_pad_2d_zeta(op.b_hat_sub_theta, pad, fill=0.0),
            b_hat_sub_zeta=_pad_2d_zeta(op.b_hat_sub_zeta, pad, fill=0.0),
            db_hat_dtheta=_pad_2d_zeta(op.db_hat_dtheta, pad, fill=0.0),
            db_hat_dzeta=_pad_2d_zeta(op.db_hat_dzeta, pad, fill=0.0),
        )
    return op


def _pad_fblock_operator(
    op: V3FBlockOperator, *, axis: str, pad: int
) -> V3FBlockOperator:
    if pad <= 0:
        return op
    collisionless = _pad_collisionless(op.collisionless, axis=axis, pad=pad)
    exb_theta = _pad_exb_theta(op.exb_theta, axis=axis, pad=pad) if op.exb_theta is not None else None
    exb_zeta = _pad_exb_zeta(op.exb_zeta, axis=axis, pad=pad) if op.exb_zeta is not None else None
    magdrift_theta = (
        _pad_magdrift_theta(op.magdrift_theta, axis=axis, pad=pad) if op.magdrift_theta is not None else None
    )
    magdrift_zeta = (
        _pad_magdrift_zeta(op.magdrift_zeta, axis=axis, pad=pad) if op.magdrift_zeta is not None else None
    )
    magdrift_xidot = (
        _pad_magdrift_xidot(op.magdrift_xidot, axis=axis, pad=pad) if op.magdrift_xidot is not None else None
    )
    er_xidot = _pad_er_xidot(op.er_xidot, axis=axis, pad=pad) if op.er_xidot is not None else None
    er_xdot = _pad_er_xdot(op.er_xdot, axis=axis, pad=pad) if op.er_xdot is not None else None
    n_theta = int(op.n_theta + pad) if axis == "theta" else int(op.n_theta)
    n_zeta = int(op.n_zeta + pad) if axis == "zeta" else int(op.n_zeta)
    return replace(
        op,
        collisionless=collisionless,
        exb_theta=exb_theta,
        exb_zeta=exb_zeta,
        magdrift_theta=magdrift_theta,
        magdrift_zeta=magdrift_zeta,
        magdrift_xidot=magdrift_xidot,
        er_xidot=er_xidot,
        er_xdot=er_xdot,
        n_theta=n_theta,
        n_zeta=n_zeta,
    )


def _pad_full_system_operator(op: V3FullSystemOperator, *, axis: str, pad: int) -> V3FullSystemOperator:
    key = (id(op), axis, pad)
    cached = _PADDED_OPERATOR_CACHE.get(key)
    if cached is not None:
        ref, value = cached
        if ref() is op:
            return value
    if pad <= 0:
        _PADDED_OPERATOR_CACHE[key] = (weakref.ref(op), op)
        return op
    fblock = _pad_fblock_operator(op.fblock, axis=axis, pad=pad)
    theta_weights = _pad_1d(op.theta_weights, pad, fill=0.0) if axis == "theta" else op.theta_weights
    zeta_weights = _pad_1d(op.zeta_weights, pad, fill=0.0) if axis == "zeta" else op.zeta_weights
    # d_hat appears in denominators for flux-surface averaging; use 1.0 to avoid 0/0.
    d_hat = _pad_2d(op.d_hat, axis=axis, pad=pad, fill=1.0)
    b_hat = _pad_2d(op.b_hat, axis=axis, pad=pad, fill=1.0)
    db_hat_dtheta = _pad_2d(op.db_hat_dtheta, axis=axis, pad=pad, fill=0.0)
    db_hat_dzeta = _pad_2d(op.db_hat_dzeta, axis=axis, pad=pad, fill=0.0)
    b_hat_sup_theta = _pad_2d(op.b_hat_sup_theta, axis=axis, pad=pad, fill=0.0)
    b_hat_sup_zeta = _pad_2d(op.b_hat_sup_zeta, axis=axis, pad=pad, fill=0.0)
    b_hat_sub_theta = _pad_2d(op.b_hat_sub_theta, axis=axis, pad=pad, fill=0.0)
    b_hat_sub_zeta = _pad_2d(op.b_hat_sub_zeta, axis=axis, pad=pad, fill=0.0)
    phi1_hat_base = _pad_2d(op.phi1_hat_base, axis=axis, pad=pad, fill=0.0)
    padded = replace(
        op,
        fblock=fblock,
        theta_weights=theta_weights,
        zeta_weights=zeta_weights,
        d_hat=d_hat,
        b_hat=b_hat,
        db_hat_dtheta=db_hat_dtheta,
        db_hat_dzeta=db_hat_dzeta,
        b_hat_sup_theta=b_hat_sup_theta,
        b_hat_sup_zeta=b_hat_sup_zeta,
        b_hat_sub_theta=b_hat_sub_theta,
        b_hat_sub_zeta=b_hat_sub_zeta,
        phi1_hat_base=phi1_hat_base,
    )
    _PADDED_OPERATOR_CACHE[key] = (weakref.ref(op), padded)
    return padded


def _pad_full_vector(
    x_full: jnp.ndarray,
    *,
    op: V3FullSystemOperator,
    op_pad: V3FullSystemOperator,
    axis: str,
    pad: int,
) -> jnp.ndarray:
    if pad <= 0:
        return x_full
    f = x_full[: op.f_size].reshape(op.fblock.f_shape)
    if axis == "theta":
        f_pad = jnp.pad(f, ((0, 0), (0, 0), (0, 0), (0, pad), (0, 0)))
    else:
        f_pad = jnp.pad(f, ((0, 0), (0, 0), (0, 0), (0, 0), (0, pad)))
    f_pad_flat = f_pad.reshape((-1,))
    rest = x_full[op.f_size :]
    if op.include_phi1:
        phi1_size = int(op.n_theta * op.n_zeta)
        phi1 = rest[:phi1_size].reshape((op.n_theta, op.n_zeta))
        if axis == "theta":
            phi1_pad = jnp.pad(phi1, ((0, pad), (0, 0)))
        else:
            phi1_pad = jnp.pad(phi1, ((0, 0), (0, pad)))
        phi1_pad_flat = phi1_pad.reshape((-1,))
        lam = rest[phi1_size : phi1_size + 1]
        extra = rest[op.phi1_size :]
        rest_pad = jnp.concatenate([phi1_pad_flat, lam, extra])
    else:
        rest_pad = rest
    return jnp.concatenate([f_pad_flat, rest_pad])


def _unpad_full_vector(
    x_pad: jnp.ndarray,
    *,
    op: V3FullSystemOperator,
    op_pad: V3FullSystemOperator,
    axis: str,
    pad: int,
) -> jnp.ndarray:
    if pad <= 0:
        return x_pad
    f_pad = x_pad[: op_pad.f_size].reshape(op_pad.fblock.f_shape)
    if axis == "theta":
        f = f_pad[:, :, :, : op.n_theta, :]
    else:
        f = f_pad[:, :, :, :, : op.n_zeta]
    f_flat = f.reshape((-1,))
    rest_pad = x_pad[op_pad.f_size :]
    if op.include_phi1:
        phi1_size_pad = int(op_pad.n_theta * op_pad.n_zeta)
        phi1_pad = rest_pad[:phi1_size_pad].reshape((op_pad.n_theta, op_pad.n_zeta))
        if axis == "theta":
            phi1 = phi1_pad[: op.n_theta, :]
        else:
            phi1 = phi1_pad[:, : op.n_zeta]
        phi1_flat = phi1.reshape((-1,))
        lam = rest_pad[phi1_size_pad : phi1_size_pad + 1]
        extra = rest_pad[op_pad.phi1_size :]
        rest = jnp.concatenate([phi1_flat, lam, extra])
    else:
        rest = rest_pad
    return jnp.concatenate([f_flat, rest])


def _matvec_shard_axis(op: V3FullSystemOperator | None = None) -> str | None:
    env = os.environ.get("SFINCS_JAX_MATVEC_SHARD_AXIS", "").strip().lower()
    if env in {"off", "none", "0", "false", "no"}:
        return None
    if env in {"flat", "vector", "p"}:
        return "flat"
    if env in {"theta", "zeta", "x"}:
        return env
    auto_env = os.environ.get("SFINCS_JAX_AUTO_SHARD", "").strip().lower()
    if env not in {"", "auto"} and auto_env in {"0", "false", "no", "off"}:
        return None
    if op is None:
        return None
    try:
        min_tz_env = os.environ.get("SFINCS_JAX_MATVEC_SHARD_MIN_TZ", "").strip()
        min_tz = int(min_tz_env) if min_tz_env else 128
    except ValueError:
        min_tz = 128
    if int(op.n_theta) * int(op.n_zeta) < max(1, min_tz):
        return None
    ntheta = int(op.n_theta)
    nzeta = int(op.n_zeta)
    prefer_x_env = os.environ.get("SFINCS_JAX_MATVEC_SHARD_PREFER_X", "").strip().lower()
    prefer_x = prefer_x_env in {"1", "true", "yes", "on"}
    try:
        min_x_env = os.environ.get("SFINCS_JAX_MATVEC_SHARD_MIN_X", "").strip()
        min_x = int(min_x_env) if min_x_env else 16
    except ValueError:
        min_x = 16
    if prefer_x and int(op.n_x) >= max(1, min_x):
        return "x"
    if ntheta > 1 and nzeta > 1:
        return "theta" if ntheta >= nzeta else "zeta"
    if ntheta > 1:
        return "theta"
    if nzeta > 1:
        return "zeta"
    if int(op.n_x) >= max(1, min_x):
        return "x"
    return None


@lru_cache(maxsize=4)
def _get_matvec_mesh(axis_name: str) -> Mesh | None:
    devices = jax.local_devices()
    if len(devices) <= 1:
        return None
    mesh_devices = np.array(devices)
    return Mesh(mesh_devices, (axis_name,))


@lru_cache(maxsize=32)
def _get_apply_full_system_operator_jit(_signature: tuple[object, ...]):
    def _apply(
        op: V3FullSystemOperator,
        x_full: jnp.ndarray,
        include_jacobian_terms: bool = True,
        pad: int = 0,
    ) -> jnp.ndarray:
        x_use = x_full[: int(op.total_size)] if pad else x_full
        y = apply_v3_full_system_operator(
            op,
            x_use,
            include_jacobian_terms=include_jacobian_terms,
            allow_sharding=False,
        )
        if pad:
            y = jnp.pad(y, (0, int(pad)))
        return y

    return jax.jit(_apply, static_argnums=(2, 3))


@lru_cache(maxsize=32)
def _get_apply_full_system_operator_pjit(_signature: tuple[object, ...], shard_axis: str):
    def _apply(
        op: V3FullSystemOperator,
        x_full: jnp.ndarray,
        include_jacobian_terms: bool = True,
    ) -> jnp.ndarray:
        return apply_v3_full_system_operator(
            op,
            x_full,
            include_jacobian_terms=include_jacobian_terms,
            allow_sharding=True,
        )

    if _pjit is None:
        return jax.jit(_apply, static_argnums=(2,))
    return _pjit.pjit(
        _apply,
        in_shardings=(None, None),
        out_shardings=None,
        static_argnums=(2,),
    )


@lru_cache(maxsize=32)
def _get_apply_full_system_operator_pjit_flat(_signature: tuple[object, ...]):
    def _apply(
        op: V3FullSystemOperator,
        x_full: jnp.ndarray,
        include_jacobian_terms: bool = True,
        pad: int = 0,
    ) -> jnp.ndarray:
        x_use = x_full[: int(op.total_size)] if pad else x_full
        y = apply_v3_full_system_operator(
            op,
            x_use,
            include_jacobian_terms=include_jacobian_terms,
            allow_sharding=False,
        )
        if pad:
            y = jnp.pad(y, (0, int(pad)))
        return y

    if _pjit is None:
        return jax.jit(_apply, static_argnums=(2, 3))
    return _pjit.pjit(
        _apply,
        in_shardings=(None, PartitionSpec("p")),
        out_shardings=PartitionSpec("p"),
        static_argnums=(2, 3),
    )


def apply_v3_full_system_operator_cached(
    op: V3FullSystemOperator, x_full: jnp.ndarray, *, include_jacobian_terms: bool = True
) -> jnp.ndarray:
    shard_axis = _matvec_shard_axis(op)
    if shard_axis is not None:
        axis_name = "p" if shard_axis == "flat" else shard_axis
        mesh = _get_matvec_mesh(axis_name)
        if mesh is not None:
            n_devices = int(np.prod(mesh.devices.shape))
            if shard_axis in {"theta", "zeta", "x"}:
                if shard_axis == "theta":
                    n_dim = int(op.n_theta)
                elif shard_axis == "zeta":
                    n_dim = int(op.n_zeta)
                else:
                    n_dim = int(op.n_x)
                pad = (-n_dim) % max(1, n_devices)
                if pad != 0 and shard_axis in {"theta", "zeta"} and _shard_pad_enabled():
                    op_pad = _pad_full_system_operator(op, axis=shard_axis, pad=pad)
                    x_pad = _pad_full_vector(x_full, op=op, op_pad=op_pad, axis=shard_axis, pad=pad)
                    fn = _get_apply_full_system_operator_pjit(_operator_signature_cached(op_pad), shard_axis)
                    with mesh:
                        y_pad = fn(op_pad, x_pad, include_jacobian_terms)
                    y = _unpad_full_vector(y_pad, op=op, op_pad=op_pad, axis=shard_axis, pad=pad)
                    return y
                if pad != 0:
                    fn = _get_apply_full_system_operator_jit(_operator_signature_cached(op))
                    return fn(op, x_full, include_jacobian_terms, 0)
                fn = _get_apply_full_system_operator_pjit(_operator_signature_cached(op), shard_axis)
                with mesh:
                    y = fn(op, x_full, include_jacobian_terms)
                return y
            if shard_axis == "flat":
                fn = _get_apply_full_system_operator_pjit_flat(_operator_signature_cached(op))
                n = int(x_full.shape[0])
                pad = (-n) % n_devices if n_devices > 0 else 0
                x_use = jnp.pad(x_full, (0, pad)) if pad else x_full
                with mesh:
                    x_use = jax.lax.with_sharding_constraint(x_use, PartitionSpec("p"))
                    y = fn(op, x_use, include_jacobian_terms, pad)
                return y[:n] if pad else y
    fn = _get_apply_full_system_operator_jit(_operator_signature_cached(op))
    return fn(op, x_full, include_jacobian_terms, 0)


def rhs_v3_full_system(op: V3FullSystemOperator) -> jnp.ndarray:
    """Assemble the v3 RHS vector used in `evaluateResidual.F90` (subset).

    This implements the parts of `evaluateResidual.F90` that are independent of the unknown
    distribution function `f1`, but may depend on the background Phi1 field:

    - `dot(psi) * d f_M / d psi` drive (adds to L=0 and L=2 rows)
    - inductive E_parallel term (adds to L=1 rows)

    For `includePhi1InKineticEquation = .true.`, the drive is multiplied by
    `exp(-Z*alpha*Phi1Hat/THat)` and includes the additional `Phi1Hat*dTHat/dpsiHat` term,
    matching `evaluateResidual.F90` lines ~89-165.

    Notes
    -----
    - `readExternalPhi1` and the specialized `EParallelHatSpec_bcdatFile` branch are not supported.
    - QuasineutralityOption=1 RHS terms (nonlinear QN) are not yet implemented here.
    """
    f_rhs = jnp.zeros(op.fblock.f_shape, dtype=jnp.float64)

    ix_min = _ix_min(bool(op.point_at_x0))
    x = op.x
    x2 = x * x
    expx2 = jnp.exp(-x2)

    dphi_hat_dpsi_hat_to_use = jnp.where(
        (op.rhs_mode == 1) | (op.rhs_mode > 3),
        op.dphi_hat_dpsi_hat,
        jnp.asarray(0.0, dtype=jnp.float64),
    )

    geom2 = (
        (op.b_hat_sub_zeta * op.db_hat_dtheta - op.b_hat_sub_theta * op.db_hat_dzeta)
        * op.d_hat
        / (op.b_hat * op.b_hat * op.b_hat)
    )  # (T,Z)

    mask_x = (jnp.arange(op.n_x) >= ix_min).astype(jnp.float64)  # (X,)

    sqrt_pi = jnp.sqrt(jnp.pi)
    two_pi = jnp.asarray(2.0 * jnp.pi, dtype=jnp.float64)
    x2_expx2 = x2 * expx2  # (X,)

    # Vectorize across species to reduce Python-loop overhead and improve XLA fusion for multi-species runs.
    z = op.z_s  # (S,)
    m_hat = op.m_hat  # (S,)
    t_hat = op.t_hat  # (S,)
    n_hat = op.n_hat  # (S,)
    dn = op.dn_hat_dpsi_hat  # (S,)
    dt = op.dt_hat_dpsi_hat  # (S,)

    sqrt_t = jnp.sqrt(t_hat)  # (S,)
    sqrt_m = jnp.sqrt(m_hat)  # (S,)

    # (S,X)
    x_part = x2_expx2[None, :] * (
        dn[:, None] / n_hat[:, None]
        + (op.alpha * z / t_hat)[:, None] * dphi_hat_dpsi_hat_to_use
        + (x2[None, :] - 1.5) * (dt / t_hat)[:, None]
    )

    if bool(op.include_phi1) and bool(op.include_phi1_in_kinetic):
        # (S,X)
        x_part2 = x2_expx2[None, :] * (dt / (t_hat * t_hat))[:, None]
        phi1 = op.phi1_hat_base  # (T,Z)
        exp_phi1 = jnp.exp(-(z[:, None, None] * op.alpha / t_hat[:, None, None]) * phi1[None, :, :])  # (S,T,Z)
        x_part_total = x_part[:, :, None, None] + (x_part2[:, :, None, None] * (z * op.alpha)[:, None, None, None] * phi1[None, None, :, :])
        x_part_total = x_part_total * exp_phi1[:, None, :, :]  # (S,X,T,Z)
    else:
        x_part_total = x_part[:, :, None, None]  # (S,X,1,1)

    pref = op.delta * n_hat * m_hat * sqrt_m / (two_pi * sqrt_pi * z * sqrt_t)  # (S,)

    factor = pref[:, None, None, None] * geom2[None, None, :, :] * x_part_total  # (S,X,T,Z)
    factor = factor * mask_x[None, :, None, None]

    if op.n_xi > 0:
        mask_l0 = (op.fblock.collisionless.n_xi_for_x > 0).astype(jnp.float64) * mask_x  # (X,)
        f_rhs = f_rhs.at[:, :, 0, :, :].add((4.0 / 3.0) * factor * mask_l0[None, :, None, None])
    if op.n_xi > 2:
        mask_l2 = (op.fblock.collisionless.n_xi_for_x > 2).astype(jnp.float64) * mask_x  # (X,)
        f_rhs = f_rhs.at[:, :, 2, :, :].add((2.0 / 3.0) * factor * mask_l2[None, :, None, None])

    if op.n_xi > 1:
        epar = op.e_parallel_hat + op.e_parallel_hat_spec  # (S,)
        factor_e = (
            op.alpha
            * z[:, None]
            * x[None, :]
            * expx2[None, :]
            * epar[:, None]
            * n_hat[:, None]
            * m_hat[:, None]
            / (jnp.pi * sqrt_pi * (t_hat * t_hat)[:, None] * op.fsab_hat2)
        )  # (S,X)
        factor_e = factor_e * mask_x[None, :]
        f_rhs = f_rhs.at[:, :, 1, :, :].add(factor_e[:, :, None, None] * op.b_hat[None, None, :, :])

    rhs_f_flat = f_rhs.reshape((-1,))
    rhs_phi1 = jnp.zeros((op.phi1_size,), dtype=jnp.float64)
    if bool(op.include_phi1) and int(op.quasineutrality_option) == 1:
        exp_phi = jnp.exp(
            -(op.z_s[:, None, None] * op.alpha / op.t_hat[:, None, None]) * op.phi1_hat_base[None, :, :]
        )
        qn_nonlin = -jnp.sum((op.z_s * op.n_hat)[:, None, None] * exp_phi, axis=0)
        if op.with_adiabatic:
            qn_nonlin = qn_nonlin - op.adiabatic_z * op.adiabatic_nhat * jnp.exp(
                -(op.adiabatic_z * op.alpha / op.adiabatic_that) * op.phi1_hat_base
            )
        qn_flat = qn_nonlin.reshape((-1,))
        # Append a zero for the lambda constraint row.
        rhs_phi1 = jnp.concatenate([qn_flat, jnp.asarray([0.0], dtype=jnp.float64)], axis=0)
    rhs_extra = jnp.zeros((op.extra_size,), dtype=jnp.float64)
    return jnp.concatenate([rhs_f_flat, rhs_phi1, rhs_extra], axis=0)


rhs_v3_full_system_jit = jax.jit(rhs_v3_full_system)


def precompile_v3_full_system(op: V3FullSystemOperator, *, include_jacobian: bool = True) -> None:
    """Ahead-of-time compile core v3 kernels for a given operator shape."""
    x_full = jnp.zeros((int(op.total_size),), dtype=jnp.float64)
    fn = _get_apply_full_system_operator_jit(_operator_signature_cached(op))
    fn.lower(op, x_full, include_jacobian_terms=True).compile()
    fn.lower(op, x_full, include_jacobian_terms=False).compile()
    if include_jacobian:
        apply_v3_full_system_jacobian_jit.lower(op, x_full, x_full).compile()
    rhs_v3_full_system_jit.lower(op).compile()


def with_transport_rhs_settings(op: V3FullSystemOperator, *, which_rhs: int) -> V3FullSystemOperator:
    """Return an operator with v3's internal RHSMode-dependent RHS settings applied.

    In v3, when `RHSMode` is used to compute a transport matrix (e.g. monoenergetic coefficients),
    the solver loops over `whichRHS` and *overwrites* (dnHatdpsiHats, dTHatdpsiHats, EParallelHat)
    before building the RHS via `evaluateResidual(f=0)`.

    This helper replicates that behavior for the currently supported modes:
    - RHSMode=3 (monoenergetic): which_rhs=1..2
    - RHSMode=2 (energy-integrated): which_rhs=1..3
    """
    w = int(which_rhs)
    if int(op.rhs_mode) == 3:
        if w == 1:
            dn = jnp.ones_like(op.dn_hat_dpsi_hat)
            dt = jnp.zeros_like(op.dt_hat_dpsi_hat)
            epar = jnp.asarray(0.0, dtype=jnp.float64)
        elif w == 2:
            dn = jnp.zeros_like(op.dn_hat_dpsi_hat)
            dt = jnp.zeros_like(op.dt_hat_dpsi_hat)
            epar = jnp.asarray(1.0, dtype=jnp.float64)
        else:
            raise ValueError("RHSMode=3 expects which_rhs in {1,2}.")
        return replace(op, dn_hat_dpsi_hat=dn, dt_hat_dpsi_hat=dt, e_parallel_hat=epar)

    if int(op.rhs_mode) == 2:
        if w == 1:
            dn = jnp.ones_like(op.dn_hat_dpsi_hat)
            dt = jnp.zeros_like(op.dt_hat_dpsi_hat)
            epar = jnp.asarray(0.0, dtype=jnp.float64)
        elif w == 2:
            # v3 sets (1/n)*dn/dpsi + (3/2)*dT/dpsi = 0 while dT/dpsi is nonzero:
            # dnHatdpsiHats = (3/2)*nHats(1)*THats(1), dTHatdpsiHats = 1.
            dn_val = (1.5) * op.n_hat[0] * op.t_hat[0]
            dn = jnp.broadcast_to(dn_val, op.dn_hat_dpsi_hat.shape)
            dt = jnp.ones_like(op.dt_hat_dpsi_hat)
            epar = jnp.asarray(0.0, dtype=jnp.float64)
        elif w == 3:
            dn = jnp.zeros_like(op.dn_hat_dpsi_hat)
            dt = jnp.zeros_like(op.dt_hat_dpsi_hat)
            epar = jnp.asarray(1.0, dtype=jnp.float64)
        else:
            raise ValueError("RHSMode=2 expects which_rhs in {1,2,3}.")
        return replace(op, dn_hat_dpsi_hat=dn, dt_hat_dpsi_hat=dt, e_parallel_hat=epar)

    return op


def residual_v3_full_system(op: V3FullSystemOperator, x_full: jnp.ndarray) -> jnp.ndarray:
    """Compute the full v3 residual `A(x) x - rhs(x)` for the currently implemented subset."""
    x_full = jnp.asarray(x_full, dtype=jnp.float64)
    op_use = op
    if bool(op.include_phi1):
        phi1_flat = x_full[op.f_size : op.f_size + op.n_theta * op.n_zeta]
        phi1 = phi1_flat.reshape((op.n_theta, op.n_zeta))
        op_use = replace(op, phi1_hat_base=phi1)
    return (
        apply_v3_full_system_operator_cached(op_use, x_full, include_jacobian_terms=False)
        - rhs_v3_full_system_jit(op_use)
    )


def full_system_operator_from_namelist(
    *, nml: Namelist, identity_shift: float = 0.0, phi1_hat_base: jnp.ndarray | None = None
) -> V3FullSystemOperator:
    """Build the full-system operator (subset) from an input namelist."""
    general = nml.group("general")
    phys = nml.group("physicsParameters")
    other = nml.group("otherNumericalParameters")
    species = nml.group("speciesParameters")
    geom_params = nml.group("geometryParameters")

    include_phi1 = _get_bool(phys, "includePhi1", False)
    read_external_phi1 = _get_bool(phys, "readExternalPhi1", False)
    if include_phi1 and read_external_phi1:
        raise NotImplementedError("readExternalPhi1 is not yet supported in sfincs_jax.")
    include_phi1 = include_phi1 and (not read_external_phi1)

    collision_operator = _get_int(phys, "collisionOperator", 0)
    delta = float(phys.get("DELTA", _V3_DEFAULT_DELTA))
    rhs_mode = _get_int(general, "RHSMode", 1)
    # In v3, `constraintScheme` is a physics input (readInput.F90) and is finalized in createGrids.F90.
    constraint_scheme = _get_int(phys, "constraintScheme", -1)
    if constraint_scheme < 0:
        constraint_scheme = 1 if collision_operator == 0 else 2

    grids: V3Grids = grids_from_namelist(nml)
    geom = geometry_from_namelist(nml=nml, grids=grids)
    fblock: V3FBlockOperator = fblock_operator_from_namelist(nml=nml, identity_shift=identity_shift)

    x_grid_scheme = _get_int(other, "xGridScheme", 5)
    point_at_x0 = x_grid_scheme in {2, 6}

    def _as_1d_float_array(v) -> jnp.ndarray:
        if isinstance(v, list):
            vv = v
        else:
            vv = [v]
        return jnp.asarray(vv, dtype=jnp.float64)

    zs = _as_1d_float_array(species.get("ZS", 1.0))
    mhat = _as_1d_float_array(species.get("MHATS", 1.0))
    that = _as_1d_float_array(species.get("THATS", 1.0))
    nhat = _as_1d_float_array(species.get("NHATS", 1.0))

    quasineutrality_option = _get_int(phys, "quasineutralityOption", 1)
    # In v3, adiabatic-species settings live in the speciesParameters namelist.
    with_adiabatic = _get_bool(species, "withAdiabatic", False)
    adiabatic_z = float(species.get("ADIABATICZ", 1.0))
    adiabatic_nhat = float(species.get("ADIABATICNHAT", 0.0))
    adiabatic_that = float(species.get("ADIABATICTHAT", 1.0))
    alpha = float(phys.get("ALPHA", 1.0))
    include_phi1_in_kinetic = bool(phys.get("INCLUDEPHI1INKINETICEQUATION", False))

    # Radial normalization factors (radialCoordinates.F90).
    input_radial = _get_int(geom_params, "inputRadialCoordinate", 3)
    input_radial_grad = _get_int(geom_params, "inputRadialCoordinateForGradients", 4)
    if input_radial != 3 or input_radial_grad not in {0, 4}:
        raise NotImplementedError(
            "sfincs_jax currently supports inputRadialCoordinate=3 (rN) with "
            "inputRadialCoordinateForGradients in {0 (psiHat), 4 (rHat)}."
        )

    geometry_scheme = _get_int(geom_params, "geometryScheme", -1)
    if geometry_scheme == 1:
        # v3 defaults are in `globalVariables.F90`; allow the namelist to override them.
        psi_a_hat = float(geom_params.get("PSIAHAT", 0.15596))
        a_hat = float(geom_params.get("AHAT", 0.5585))
        r_n = float(geom_params.get("RN_WISH", 0.5))
    elif geometry_scheme == 2:
        # v3 ignores *_wish and uses rN=0.5 for this simplified LHD model.
        a_hat = 0.5585
        psi_a_hat = (a_hat * a_hat) / 2.0
        r_n = 0.5
    elif geometry_scheme == 4:
        psi_a_hat = -0.384935
        a_hat = 0.5109
        r_n = 0.5  # v3 forces rN=0.5 for geometryScheme=4.
    elif geometry_scheme in {11, 12}:
        eq = geom_params.get("EQUILIBRIUMFILE", None)
        if eq is None:
            raise ValueError("geometryScheme=11/12 requires equilibriumFile in geometryParameters.")
        base_dir = nml.source_path.parent if nml.source_path is not None else None
        repo_root = Path(__file__).resolve().parents[1]
        extra = (repo_root / "tests" / "ref", repo_root / "sfincs_jax" / "data" / "equilibria")
        p = resolve_existing_path(str(eq), base_dir=base_dir, extra_search_dirs=extra).path
        header = read_boozer_bc_header(path=str(p), geometry_scheme=int(geometry_scheme))
        psi_a_hat = float(header.psi_a_hat)
        a_hat = float(header.a_hat)
        r_n_wish = float(geom_params.get("RN_WISH", 0.5))
        vmecradial_option = _get_int(geom_params, "VMECRadialOption", _get_int(geom_params, "VMECRADIALOPTION", 1))
        r_n = selected_r_n_from_bc(
            path=str(p),
            geometry_scheme=int(geometry_scheme),
            r_n_wish=r_n_wish,
            vmecradial_option=int(vmecradial_option),
        )
    elif geometry_scheme == 5:
        eq = geom_params.get("EQUILIBRIUMFILE", None)
        if eq is None:
            raise ValueError("geometryScheme=5 requires equilibriumFile in geometryParameters.")
        base_dir = nml.source_path.parent if nml.source_path is not None else None
        repo_root = Path(__file__).resolve().parents[1]
        extra = (repo_root / "tests" / "ref", repo_root / "sfincs_jax" / "data" / "equilibria")
        # Allow `.txt -> .nc` fallback for VMEC wout files.
        try:
            p = resolve_existing_path(str(eq), base_dir=base_dir, extra_search_dirs=extra).path
        except FileNotFoundError:
            p2 = Path(str(eq).strip().strip('"').strip("'")).with_suffix(".nc")
            p = resolve_existing_path(str(p2), base_dir=base_dir, extra_search_dirs=extra).path

        w = read_vmec_wout(p)
        psi_a_hat = float(psi_a_hat_from_wout(w))
        a_hat = float(w.aminor_p)

        r_n_wish = float(geom_params.get("RN_WISH", 0.5))
        psi_n_wish = float(r_n_wish) * float(r_n_wish)
        vmecradial_option = _get_int(geom_params, "VMECRadialOption", _get_int(geom_params, "VMECRADIALOPTION", 1))
        interp = vmec_interpolation(w=w, psi_n_wish=psi_n_wish, vmec_radial_option=vmecradial_option)
        r_n = float(interp.psi_n) ** 0.5
    else:
        raise NotImplementedError(f"Radial conversions are not implemented for geometryScheme={geometry_scheme}.")

    # With rHat = aHat * rN and psiHat = psiAHat * (rN^2):
    # dpsiHat/drHat = 2*psiAHat*rN/aHat -> drHat/dpsiHat = aHat/(2*psiAHat*rN).
    ddrhat2ddpsihat = float(a_hat) / (2.0 * float(psi_a_hat) * float(r_n))

    def _grad_in_psihat(key_drhat: str, key_psihat: str) -> jnp.ndarray:
        if key_drhat.upper() in species:
            return ddrhat2ddpsihat * _as_1d_float_array(species.get(key_drhat.upper(), 0.0))
        return _as_1d_float_array(species.get(key_psihat.upper(), 0.0))

    dn_hat_dpsi_hat = _grad_in_psihat("dNHatdrHats", "dNHatdpsiHats")
    dt_hat_dpsi_hat = _grad_in_psihat("dTHatdrHats", "dTHatdpsiHats")

    # dPhiHat/dpsiHat:
    # - if inputRadialCoordinateForGradients=4, v3 uses Er with dPhiHat/drHat = -Er.
    # - if inputRadialCoordinateForGradients=0, v3 expects dPhiHat/dpsiHat directly.
    if int(input_radial_grad) == 4:
        er = float(phys.get("ER", 0.0))
        dphi_hat_dpsi_hat = jnp.asarray(ddrhat2ddpsihat * (-er), dtype=jnp.float64)
    else:
        dphi_hat_dpsi_hat = jnp.asarray(float(phys.get("DPHIHATDPSIHAT", 0.0)), dtype=jnp.float64)

    if int(rhs_mode) == 3:
        e_star = float(phys.get("ESTAR", phys.get("EStar", 0.0)))
        g_hat_eff = float(geom.g_hat)
        b0_eff = float(geom.b0_over_bbar)
        if abs(g_hat_eff) < 1e-30 or abs(b0_eff) < 1e-30:
            g_tmp, _i_tmp = g_hat_i_hat_jax(grids=grids, geom=geom)
            b0_tmp = b0_over_bbar_jax(grids=grids, geom=geom)
            g_hat_eff = float(g_tmp)
            b0_eff = float(b0_tmp)
        dphi_hat_dpsi_hat = jnp.asarray(
            (2.0 / (float(alpha) * float(delta)))
            * float(e_star)
            * float(geom.iota)
            * float(b0_eff)
            / float(g_hat_eff),
            dtype=jnp.float64,
        )

    e_parallel_hat = float(phys.get("EPARALLELHAT", 0.0))
    e_parallel_hat_spec_raw = phys.get("EPARALLELHATSPEC", None)
    if e_parallel_hat_spec_raw is None:
        e_parallel_hat_spec = jnp.zeros_like(zs)
    else:
        e_parallel_hat_spec = _as_1d_float_array(e_parallel_hat_spec_raw)
        if e_parallel_hat_spec.shape == (1,) and zs.shape != (1,):
            e_parallel_hat_spec = jnp.broadcast_to(e_parallel_hat_spec, zs.shape)
        if e_parallel_hat_spec.shape != zs.shape:
            raise ValueError(f"EParallelHatSpec must have shape {zs.shape}, got {e_parallel_hat_spec.shape}")

    fsab_hat2 = jnp.asarray(fsab_hat2_jax(grids=grids, geom=geom), dtype=jnp.float64)

    if phi1_hat_base is None:
        phi1_hat_base = jnp.zeros((int(grids.theta.shape[0]), int(grids.zeta.shape[0])), dtype=jnp.float64)
    else:
        phi1_hat_base = jnp.asarray(phi1_hat_base, dtype=jnp.float64)
        if phi1_hat_base.shape != (int(grids.theta.shape[0]), int(grids.zeta.shape[0])):
            raise ValueError(
                f"phi1_hat_base must have shape {(int(grids.theta.shape[0]), int(grids.zeta.shape[0]))}, got {phi1_hat_base.shape}"
            )

    return V3FullSystemOperator(
        fblock=fblock,
        constraint_scheme=int(constraint_scheme),
        point_at_x0=bool(point_at_x0),
        include_phi1=bool(include_phi1),
        quasineutrality_option=int(quasineutrality_option),
        with_adiabatic=bool(with_adiabatic),
        alpha=jnp.asarray(alpha, dtype=jnp.float64),
        delta=jnp.asarray(delta, dtype=jnp.float64),
        adiabatic_z=jnp.asarray(adiabatic_z, dtype=jnp.float64),
        adiabatic_nhat=jnp.asarray(adiabatic_nhat, dtype=jnp.float64),
        adiabatic_that=jnp.asarray(adiabatic_that, dtype=jnp.float64),
        include_phi1_in_kinetic=bool(include_phi1_in_kinetic),
        dphi_hat_dpsi_hat=dphi_hat_dpsi_hat,
        phi1_hat_base=phi1_hat_base,
        rhs_mode=int(rhs_mode),
        e_parallel_hat=jnp.asarray(e_parallel_hat, dtype=jnp.float64),
        e_parallel_hat_spec=e_parallel_hat_spec,
        fsab_hat2=fsab_hat2,
        z_s=zs,
        m_hat=mhat,
        t_hat=that,
        n_hat=nhat,
        dn_hat_dpsi_hat=dn_hat_dpsi_hat,
        dt_hat_dpsi_hat=dt_hat_dpsi_hat,
        theta_weights=grids.theta_weights,
        zeta_weights=grids.zeta_weights,
        d_hat=geom.d_hat,
        b_hat=geom.b_hat,
        db_hat_dtheta=geom.db_hat_dtheta,
        db_hat_dzeta=geom.db_hat_dzeta,
        b_hat_sup_theta=geom.b_hat_sup_theta,
        b_hat_sup_zeta=geom.b_hat_sup_zeta,
        b_hat_sub_theta=geom.b_hat_sub_theta,
        b_hat_sub_zeta=geom.b_hat_sub_zeta,
        x=grids.x,
        x_weights=grids.x_weights,
        ddx=grids.ddx,
    )
