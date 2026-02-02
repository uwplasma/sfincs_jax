from __future__ import annotations

from dataclasses import dataclass

from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
from jax import tree_util as jtu

from .namelist import Namelist
from .v3 import V3Grids, geometry_from_namelist, grids_from_namelist
from .v3_fblock import V3FBlockOperator, apply_v3_fblock_operator, fblock_operator_from_namelist

_THRESHOLD_FOR_INCLUSION = 1e-12  # Matches v3 `sparsify.F90`.


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
    b_hat_sup_theta: jnp.ndarray  # (T,Z)
    b_hat_sup_zeta: jnp.ndarray  # (T,Z)
    b_hat_sub_theta: jnp.ndarray  # (T,Z)
    b_hat_sub_zeta: jnp.ndarray  # (T,Z)

    x: jnp.ndarray  # (X,)
    x_weights: jnp.ndarray  # (X,)
    ddx: jnp.ndarray  # (X,X)

    def tree_flatten(self):
        children = (
            self.fblock,
            self.constraint_scheme,
            self.point_at_x0,
            self.include_phi1,
            self.quasineutrality_option,
            self.with_adiabatic,
            self.alpha,
            self.delta,
            self.adiabatic_z,
            self.adiabatic_nhat,
            self.adiabatic_that,
            self.include_phi1_in_kinetic,
            self.dphi_hat_dpsi_hat,
            self.phi1_hat_base,
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
            self.b_hat_sup_theta,
            self.b_hat_sup_zeta,
            self.b_hat_sub_theta,
            self.b_hat_sub_zeta,
            self.x,
            self.x_weights,
            self.ddx,
        )
        aux = None
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        del aux
        (
            fblock,
            constraint_scheme,
            point_at_x0,
            include_phi1,
            quasineutrality_option,
            with_adiabatic,
            alpha,
            delta,
            adiabatic_z,
            adiabatic_nhat,
            adiabatic_that,
            include_phi1_in_kinetic,
            dphi_hat_dpsi_hat,
            phi1_hat_base,
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

    y_f = apply_v3_fblock_operator(op.fblock, f)
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


def full_system_operator_from_namelist(
    *, nml: Namelist, identity_shift: float = 0.0, phi1_hat_base: jnp.ndarray | None = None
) -> V3FullSystemOperator:
    """Build the full-system operator (subset) from an input namelist."""
    phys = nml.group("physicsParameters")
    other = nml.group("otherNumericalParameters")
    species = nml.group("speciesParameters")

    include_phi1 = _get_bool(phys, "includePhi1", False)
    read_external_phi1 = _get_bool(phys, "readExternalPhi1", False)
    if include_phi1 and read_external_phi1:
        raise NotImplementedError("readExternalPhi1 is not yet supported in sfincs_jax.")
    include_phi1 = include_phi1 and (not read_external_phi1)

    collision_operator = _get_int(phys, "collisionOperator", 0)
    delta = float(phys.get("DELTA", 0.0))
    constraint_scheme = _get_int(other, "constraintScheme", -1)
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

    # v3 default for geometryScheme=4 forces rN=0.5, so psiN=0.25. Use the same conversion
    # factor as radialCoordinates.F90 for inputRadialCoordinateForGradients=4.
    psi_a_hat = float(nml.group("geometryParameters").get("PSIAHAT", -0.384935))
    a_hat = float(nml.group("geometryParameters").get("AHAT", 0.5109))
    psi_n = 0.25
    ddrhat2ddpsihat = a_hat / (2.0 * psi_a_hat * jnp.sqrt(psi_n))

    def _grad_in_psihat(key_drhat: str, key_psihat: str) -> jnp.ndarray:
        if key_drhat.upper() in species:
            return ddrhat2ddpsihat * _as_1d_float_array(species.get(key_drhat.upper(), 0.0))
        return _as_1d_float_array(species.get(key_psihat.upper(), 0.0))

    dn_hat_dpsi_hat = _grad_in_psihat("dNHatdrHats", "dNHatdpsiHats")
    dt_hat_dpsi_hat = _grad_in_psihat("dTHatdrHats", "dTHatdpsiHats")

    # dPhiHat/dpsiHat from Er (radialCoordinates.F90 inputRadialCoordinateForGradients=4):
    er = float(phys.get("ER", 0.0))
    dphi_hat_dpsi_hat = jnp.asarray(ddrhat2ddpsihat * (-er), dtype=jnp.float64)

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
        b_hat_sup_theta=geom.b_hat_sup_theta,
        b_hat_sup_zeta=geom.b_hat_sup_zeta,
        b_hat_sub_theta=geom.b_hat_sub_theta,
        b_hat_sub_zeta=geom.b_hat_sub_zeta,
        x=grids.x,
        x_weights=grids.x_weights,
        ddx=grids.ddx,
    )
