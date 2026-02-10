from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
from jax import tree_util as jtu

from .boozer_bc import read_boozer_bc_header
from .diagnostics import b0_over_bbar as b0_over_bbar_jax
from .diagnostics import fsab_hat2 as fsab_hat2_jax
from .diagnostics import g_hat_i_hat as g_hat_i_hat_jax
from .namelist import Namelist
from .paths import resolve_existing_path
from .v3 import V3Grids, geometry_from_namelist, grids_from_namelist
from .v3_fblock import V3FBlockOperator, apply_v3_fblock_operator, fblock_operator_from_namelist
from .vmec_wout import psi_a_hat_from_wout, read_vmec_wout, vmec_interpolation

_THRESHOLD_FOR_INCLUSION = 1e-12  # Matches v3 `sparsify.F90`.
_V3_DEFAULT_DELTA = 4.5694e-3  # v3 `globalVariables.F90`


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


def apply_v3_full_system_operator(op: V3FullSystemOperator, x_full: jnp.ndarray) -> jnp.ndarray:
    """Apply the matrix-free full-system operator."""
    x_full = jnp.asarray(x_full)
    if x_full.shape != (op.total_size,):
        raise ValueError(f"x_full must have shape {(op.total_size,)}, got {x_full.shape}")

    f_flat = x_full[: op.f_size]
    rest = x_full[op.f_size :]
    f = f_flat.reshape(op.fblock.f_shape)

    y_f = apply_v3_fblock_operator(op.fblock, f, phi1_hat_base=op.phi1_hat_base if op.fblock.fp_phi1 is not None else None)
    factor = _fs_average_factor(op.theta_weights, op.zeta_weights, op.d_hat)  # (T,Z)
    ix0 = _ix_min(op.point_at_x0)

    y_phi1 = jnp.zeros((0,), dtype=jnp.float64)
    if op.include_phi1:
        phi1_flat = rest[: op.n_theta * op.n_zeta]
        lam = rest[op.n_theta * op.n_zeta]
        extra = rest[op.phi1_size :]

        phi1 = phi1_flat.reshape((op.n_theta, op.n_zeta))

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

        phi1_diag = 0.0
        if int(op.quasineutrality_option) == 2 and op.with_adiabatic and op.n_species > 0:
            phi1_diag = -op.alpha * (
                (op.z_s[0] * op.z_s[0]) * op.n_hat[0] / op.t_hat[0]
                + (op.adiabatic_z * op.adiabatic_z) * op.adiabatic_nhat / op.adiabatic_that
            )
        elif int(op.quasineutrality_option) == 1:
            # Parity-first stabilization for the includePhi1 + quasineutralityOption=1 branch:
            # include the leading-order Boltzmann-response diagonal from all kinetic species and
            # the adiabatic species when present. Without this term, tiny reduced fixtures can
            # pick an unphysical large-Phi1 nullspace branch.
            phi1_diag = -op.alpha * jnp.sum((op.z_s * op.z_s) * op.n_hat / op.t_hat)
            if op.with_adiabatic:
                phi1_diag = phi1_diag - op.alpha * (
                    (op.adiabatic_z * op.adiabatic_z) * op.adiabatic_nhat / op.adiabatic_that
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
        # - For now we treat exp(-Ze alpha Phi1 / T) as 1.0 (linearization about Phi1 ~ 0),
        #   matching the parity fixtures which use tiny Phi1 amplitudes.
        assert phi1 is not None
        ddtheta = op.fblock.collisionless.ddtheta
        ddzeta = op.fblock.collisionless.ddzeta
        dphi1_dtheta = ddtheta @ phi1  # (T,Z)
        dphi1_dzeta = phi1 @ ddzeta.T  # (T,Z)

        # Nonlinear term in the v3 residual that is linear in f but proportional to grad(Phi1Hat).
        # This contributes to the F-block (d(kinetic eqn)/df) in whichMatrix=3.
        dphi1_hat_base_dtheta = ddtheta @ op.phi1_hat_base
        dphi1_hat_base_dzeta = op.phi1_hat_base @ ddzeta.T
        e_term = op.b_hat_sup_theta * dphi1_hat_base_dtheta + op.b_hat_sup_zeta * dphi1_hat_base_dzeta  # (T,Z)
        nonlinear_factor = (
            -(op.alpha * op.z_s)[:, None, None]
            / (2.0 * op.b_hat[None, :, :] * jnp.sqrt(op.t_hat)[:, None, None] * jnp.sqrt(op.m_hat)[:, None, None])
            * e_term[None, :, :]
        )  # (S,T,Z)

        n_xi = int(op.n_xi)
        inv_x = 1.0 / op.x  # (X,)

        # ddx applied along x for all L at once:
        ddx_to_use = jnp.where(jnp.abs(op.ddx) > _THRESHOLD_FOR_INCLUSION, op.ddx, 0.0)
        ddx_f = jnp.einsum("ij,sltzj->sltzi", ddx_to_use, jnp.transpose(f, (0, 2, 3, 4, 1)))  # (S,L,T,Z,X)
        ddx_f = jnp.transpose(ddx_f, (0, 4, 1, 2, 3))  # (S,X,L,T,Z)

        out_nl = jnp.zeros_like(f, dtype=jnp.float64)
        l = jnp.arange(n_xi, dtype=jnp.float64)

        # Super-diagonal (output L receives from source L+1):
        if n_xi > 1:
            lp1 = l[:-1]  # output L indices (0..Nxi-2)
            coef = (lp1 + 1.0) / (2.0 * lp1 + 3.0)  # (Nxi-1,)
            diag_xl = (((lp1 + 1.0) * (lp1 + 2.0) / (2.0 * lp1 + 3.0))[:, None] * inv_x[None, :]).T  # (X,Nxi-1)
            src = f[:, :, 1:, :, :]  # (S,X,Nxi-1,T,Z)
            ddx_src = ddx_f[:, :, 1:, :, :]
            term = coef[None, None, :, None, None] * ddx_src + diag_xl[None, :, :, None, None] * src
            out_nl = out_nl.at[:, :, :-1, :, :].add(term)

        # Sub-diagonal (output L receives from source L-1):
        if n_xi > 1:
            lm1 = l[1:]  # output L indices (1..Nxi-1)
            coef = lm1 / (2.0 * lm1 - 1.0)  # (Nxi-1,)
            diag_xl = ((-(lm1 - 1.0) * lm1 / (2.0 * lm1 - 1.0))[:, None] * inv_x[None, :]).T  # (X,Nxi-1)
            src = f[:, :, :-1, :, :]  # (S,X,Nxi-1,T,Z)
            ddx_src = ddx_f[:, :, :-1, :, :]
            term = coef[None, None, :, None, None] * ddx_src + diag_xl[None, :, :, None, None] * src
            out_nl = out_nl.at[:, :, 1:, :, :].add(term)

        mask = (
            jnp.arange(n_xi, dtype=jnp.int32)[None, :] < op.fblock.collisionless.n_xi_for_x.astype(jnp.int32)[:, None]
        ).astype(jnp.float64)  # (X,L)
        y_f = y_f + out_nl * nonlinear_factor[:, None, None, :, :] * mask[None, :, :, None, None]

        x2 = op.x * op.x  # (X,)
        expx2 = jnp.exp(-x2)  # (X,)

        sqrt_pi = jnp.sqrt(jnp.pi)
        norm = jnp.pi * sqrt_pi

        # Evaluate exp(-Z*alpha*Phi1Hat/THat) using the base-state Phi1Hat used to assemble the v3 matrix.
        exp_phi = jnp.exp(
            -(op.z_s[:, None, None] * op.alpha / op.t_hat[:, None, None]) * op.phi1_hat_base[None, :, :]
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
        phi_term = op.dphi_hat_dpsi_hat + op.phi1_hat_base[None, :, :] * (op.dt_hat_dpsi_hat / op.t_hat)[:, None, None]  # (S,T,Z)

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


apply_v3_full_system_operator_jit = jax.jit(apply_v3_full_system_operator, static_argnums=())


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
    rhs_extra = jnp.zeros((op.extra_size,), dtype=jnp.float64)
    return jnp.concatenate([rhs_f_flat, rhs_phi1, rhs_extra], axis=0)


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
    return apply_v3_full_system_operator(op_use, x_full) - rhs_v3_full_system(op_use)


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
        r_n = float(geom_params.get("RN_WISH", 0.5))
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
        vmecradial_option = int(geom_params.get("VMECRADIALOPTION", 0))
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
