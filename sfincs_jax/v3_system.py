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
    - Phi1 coupling terms in the kinetic equation and collision operator are not yet implemented.
      For now, `sfincs_jax` supports the *linear* QN/lambda blocks used when Phi1 is present but does
      not enter the kinetic equation operator.
    """

    fblock: V3FBlockOperator
    constraint_scheme: int
    point_at_x0: bool

    include_phi1: bool
    quasineutrality_option: int
    with_adiabatic: bool
    alpha: jnp.ndarray  # scalar
    adiabatic_z: jnp.ndarray  # scalar
    adiabatic_nhat: jnp.ndarray  # scalar
    adiabatic_that: jnp.ndarray  # scalar

    z_s: jnp.ndarray  # (S,)
    m_hat: jnp.ndarray  # (S,)
    t_hat: jnp.ndarray  # (S,)
    n_hat: jnp.ndarray  # (S,)

    theta_weights: jnp.ndarray  # (T,)
    zeta_weights: jnp.ndarray  # (Z,)
    d_hat: jnp.ndarray  # (T,Z)

    x: jnp.ndarray  # (X,)
    x_weights: jnp.ndarray  # (X,)

    def tree_flatten(self):
        children = (
            self.fblock,
            self.constraint_scheme,
            self.point_at_x0,
            self.include_phi1,
            self.quasineutrality_option,
            self.with_adiabatic,
            self.alpha,
            self.adiabatic_z,
            self.adiabatic_nhat,
            self.adiabatic_that,
            self.z_s,
            self.m_hat,
            self.t_hat,
            self.n_hat,
            self.theta_weights,
            self.zeta_weights,
            self.d_hat,
            self.x,
            self.x_weights,
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
            adiabatic_z,
            adiabatic_nhat,
            adiabatic_that,
            z_s,
            m_hat,
            t_hat,
            n_hat,
            theta_weights,
            zeta_weights,
            d_hat,
            x,
            x_weights,
        ) = children
        return cls(
            fblock=fblock,
            constraint_scheme=int(constraint_scheme),
            point_at_x0=bool(point_at_x0),
            include_phi1=bool(include_phi1),
            quasineutrality_option=int(quasineutrality_option),
            with_adiabatic=bool(with_adiabatic),
            alpha=alpha,
            adiabatic_z=adiabatic_z,
            adiabatic_nhat=adiabatic_nhat,
            adiabatic_that=adiabatic_that,
            z_s=z_s,
            m_hat=m_hat,
            t_hat=t_hat,
            n_hat=n_hat,
            theta_weights=theta_weights,
            zeta_weights=zeta_weights,
            d_hat=d_hat,
            x=x,
            x_weights=x_weights,
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


def full_system_operator_from_namelist(*, nml: Namelist, identity_shift: float = 0.0) -> V3FullSystemOperator:
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

    return V3FullSystemOperator(
        fblock=fblock,
        constraint_scheme=int(constraint_scheme),
        point_at_x0=bool(point_at_x0),
        include_phi1=bool(include_phi1),
        quasineutrality_option=int(quasineutrality_option),
        with_adiabatic=bool(with_adiabatic),
        alpha=jnp.asarray(alpha, dtype=jnp.float64),
        adiabatic_z=jnp.asarray(adiabatic_z, dtype=jnp.float64),
        adiabatic_nhat=jnp.asarray(adiabatic_nhat, dtype=jnp.float64),
        adiabatic_that=jnp.asarray(adiabatic_that, dtype=jnp.float64),
        z_s=zs,
        m_hat=mhat,
        t_hat=that,
        n_hat=nhat,
        theta_weights=grids.theta_weights,
        zeta_weights=grids.zeta_weights,
        d_hat=geom.d_hat,
        x=grids.x,
        x_weights=grids.x_weights,
    )
