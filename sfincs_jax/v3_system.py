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
    """Matrix-free operator for the full v3 linear system in the common includePhi1 = false modes.

    This operator extends the F-block (distribution function) operator with the constraint rows/cols
    used to remove nullspaces and enforce moments:

    - ``constraintScheme = 2`` (common default when ``collisionOperator != 0``):
      adds an L=0 source unknown at each x, and enforces flux-surface average of ``f1`` is 0 at each x.
    - ``constraintScheme = 1`` (common default when ``collisionOperator = 0``):
      adds particle+energy source unknowns per species, and enforces density and pressure moments are 0.

    Notes
    -----
    - This currently supports only ``includePhi1 = .false.`` (no quasineutrality / lambda blocks).
    - The ordering of unknowns matches v3 `indices.F90` for this subset:
      ``[F-block, constraint unknowns]``.
    """

    fblock: V3FBlockOperator
    constraint_scheme: int
    point_at_x0: jnp.ndarray  # scalar bool

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
            theta_weights,
            zeta_weights,
            d_hat,
            x,
            x_weights,
        ) = children
        return cls(
            fblock=fblock,
            constraint_scheme=int(constraint_scheme),
            point_at_x0=point_at_x0,
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
        return int(self.f_size + self.extra_size)


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
    extra = x_full[op.f_size :]
    f = f_flat.reshape(op.fblock.f_shape)

    y_f = apply_v3_fblock_operator(op.fblock, f)
    factor = _fs_average_factor(op.theta_weights, op.zeta_weights, op.d_hat)  # (T,Z)
    ix0 = _ix_min(bool(op.point_at_x0))

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
        if bool(op.point_at_x0):
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

    return jnp.concatenate([y_f.reshape((-1,)), y_extra], axis=0)


apply_v3_full_system_operator_jit = jax.jit(apply_v3_full_system_operator, static_argnums=())


def full_system_operator_from_namelist(*, nml: Namelist, identity_shift: float = 0.0) -> V3FullSystemOperator:
    """Build the full-system operator (subset) from an input namelist."""
    phys = nml.group("physicsParameters")
    other = nml.group("otherNumericalParameters")

    include_phi1 = _get_bool(phys, "includePhi1", False)
    if include_phi1:
        raise NotImplementedError("full-system assembly currently supports includePhi1 = .false. only.")

    collision_operator = _get_int(phys, "collisionOperator", 0)
    constraint_scheme = _get_int(other, "constraintScheme", -1)
    if constraint_scheme < 0:
        constraint_scheme = 1 if collision_operator == 0 else 2

    grids: V3Grids = grids_from_namelist(nml)
    geom = geometry_from_namelist(nml=nml, grids=grids)
    fblock: V3FBlockOperator = fblock_operator_from_namelist(nml=nml, identity_shift=identity_shift)

    x_grid_scheme = _get_int(other, "xGridScheme", 5)
    point_at_x0 = x_grid_scheme in {2, 6}

    return V3FullSystemOperator(
        fblock=fblock,
        constraint_scheme=int(constraint_scheme),
        point_at_x0=jnp.asarray(point_at_x0),
        theta_weights=grids.theta_weights,
        zeta_weights=grids.zeta_weights,
        d_hat=geom.d_hat,
        x=grids.x,
        x_weights=grids.x_weights,
    )
