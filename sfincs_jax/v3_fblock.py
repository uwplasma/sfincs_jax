from __future__ import annotations

from dataclasses import dataclass
import math

from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
import numpy as np
from jax import tree_util as jtu

from .collisionless import CollisionlessV3Operator, apply_collisionless_v3
from .collisionless_er import ErXDotV3Operator, ErXiDotV3Operator, apply_er_xdot_v3, apply_er_xidot_v3
from .collisions import PitchAngleScatteringV3Operator, apply_pitch_angle_scattering_v3, make_pitch_angle_scattering_v3_operator
from .geometry import BoozerGeometry
from .namelist import Namelist
from .solver import GMRESSolveResult, gmres_solve
from .v3 import V3Grids, geometry_from_namelist, grids_from_namelist


def _as_1d_float(group: dict, key: str) -> np.ndarray:
    """Read a namelist value as a 1D float64 numpy array."""
    v = group[key.upper()]
    return np.atleast_1d(np.asarray(v, dtype=np.float64))


def _get_float(group: dict, key: str, default: float) -> float:
    v = group.get(key.upper(), default)
    if isinstance(v, list):
        v = v[0] if v else default
    return float(v)


def _get_int(group: dict, key: str, default: int) -> int:
    v = group.get(key.upper(), default)
    if isinstance(v, list):
        v = v[0] if v else default
    return int(v)


def _dphi_hat_dpsi_hat_from_er_scheme4(er: float) -> float:
    """Compute dPhiHat/dpsiHat from Er for geometryScheme=4 (v3 defaults).

    Matches geometry.F90 (scheme=4) + radialCoordinates.F90 with
    inputRadialCoordinateForGradients=4 (default).
    """
    psi_a_hat = -0.384935
    a_hat = 0.5109
    psi_n = 0.25  # rN=0.5 is forced for geometryScheme=4
    ddrhat2ddpsihat = a_hat / (2.0 * psi_a_hat * math.sqrt(psi_n))
    return float(ddrhat2ddpsihat * (-er))


def collisionless_operator_from_namelist(
    *,
    nml: Namelist,
    grids: V3Grids,
    geom: BoozerGeometry,
) -> CollisionlessV3Operator:
    species = nml.group("speciesParameters")
    t_hats = _as_1d_float(species, "THats")
    m_hats = _as_1d_float(species, "mHats")
    return CollisionlessV3Operator(
        x=grids.x,
        ddtheta=grids.ddtheta,
        ddzeta=grids.ddzeta,
        b_hat=geom.b_hat,
        b_hat_sup_theta=geom.b_hat_sup_theta,
        b_hat_sup_zeta=geom.b_hat_sup_zeta,
        db_hat_dtheta=geom.db_hat_dtheta,
        db_hat_dzeta=geom.db_hat_dzeta,
        t_hats=jnp.asarray(t_hats),
        m_hats=jnp.asarray(m_hats),
        n_xi_for_x=grids.n_xi_for_x,
    )


def pas_collision_operator_from_namelist(*, nml: Namelist, grids: V3Grids) -> PitchAngleScatteringV3Operator:
    species = nml.group("speciesParameters")
    phys = nml.group("physicsParameters")

    collision_operator = _get_int(phys, "collisionOperator", 0)
    if collision_operator != 1:
        raise NotImplementedError(
            "sfincs_jax currently only builds a collision operator for collisionOperator=1 "
            "(pitch-angle scattering)."
        )

    z_s = _as_1d_float(species, "Zs")
    m_hats = _as_1d_float(species, "mHats")
    n_hats = _as_1d_float(species, "nHats")
    t_hats = _as_1d_float(species, "THats")
    nu_n = _get_float(phys, "nu_n", 0.0)
    krook = _get_float(phys, "Krook", 0.0)

    return make_pitch_angle_scattering_v3_operator(
        x=grids.x,
        z_s=jnp.asarray(z_s),
        m_hats=jnp.asarray(m_hats),
        n_hats=jnp.asarray(n_hats),
        t_hats=jnp.asarray(t_hats),
        nu_n=nu_n,
        krook=krook,
        n_xi_for_x=grids.n_xi_for_x,
    )


@jtu.register_pytree_node_class
@dataclass(frozen=True)
class V3FBlockOperator:
    """Matrix-free operator for the v3 distribution-function block (BLOCK_F).

    This is intentionally incomplete. Today it includes:
    - collisionless streaming + mirror (±1 couplings in L)
    - pitch-angle scattering collisions (diagonal in L)
    - collisionless Er terms (xiDot and xDot), when enabled in the namelist

    As more v3 terms are ported, they will be composed here.
    """

    collisionless: CollisionlessV3Operator
    pas: PitchAngleScatteringV3Operator
    er_xidot: ErXiDotV3Operator | None
    er_xdot: ErXDotV3Operator | None
    identity_shift: jnp.ndarray  # scalar, helps make toy solves well-conditioned

    n_species: int
    n_x: int
    n_xi: int
    n_theta: int
    n_zeta: int

    @property
    def f_shape(self) -> tuple[int, int, int, int, int]:
        return (self.n_species, self.n_x, self.n_xi, self.n_theta, self.n_zeta)

    @property
    def flat_size(self) -> int:
        s, x, l, t, z = self.f_shape
        return int(s * x * l * t * z)

    def tree_flatten(self):
        children = (self.collisionless, self.pas, self.er_xidot, self.er_xdot, self.identity_shift)
        aux = (self.n_species, self.n_x, self.n_xi, self.n_theta, self.n_zeta)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        collisionless, pas, er_xidot, er_xdot, identity_shift = children
        n_species, n_x, n_xi, n_theta, n_zeta = aux
        return cls(
            collisionless=collisionless,
            pas=pas,
            er_xidot=er_xidot,
            er_xdot=er_xdot,
            identity_shift=identity_shift,
            n_species=n_species,
            n_x=n_x,
            n_xi=n_xi,
            n_theta=n_theta,
            n_zeta=n_zeta,
        )


def fblock_operator_from_namelist(*, nml: Namelist, identity_shift: float = 0.0) -> V3FBlockOperator:
    grids = grids_from_namelist(nml)
    geom = geometry_from_namelist(nml=nml, grids=grids)
    colless = collisionless_operator_from_namelist(nml=nml, grids=grids, geom=geom)
    pas = pas_collision_operator_from_namelist(nml=nml, grids=grids)

    phys = nml.group("physicsParameters")
    include_xdot = bool(phys.get("INCLUDEXDOTTERM", False))
    include_er_xidot = bool(phys.get("INCLUDEELECTRICFIELDTERMINXIDOT", False))
    er = float(phys.get("ER", 0.0))
    alpha = float(phys.get("ALPHA", 1.0))
    delta = float(phys.get("DELTA", 0.0))

    # For now we assume geometryScheme=4 and v3 defaults for the radial conversion.
    dphi = _dphi_hat_dpsi_hat_from_er_scheme4(er)

    er_xidot = None
    if include_er_xidot:
        er_xidot = ErXiDotV3Operator(
            alpha=jnp.asarray(alpha, dtype=jnp.float64),
            delta=jnp.asarray(delta, dtype=jnp.float64),
            dphi_hat_dpsi_hat=jnp.asarray(dphi, dtype=jnp.float64),
            d_hat=geom.d_hat,
            b_hat=geom.b_hat,
            b_hat_sub_theta=geom.b_hat_sub_theta,
            b_hat_sub_zeta=geom.b_hat_sub_zeta,
            db_hat_dtheta=geom.db_hat_dtheta,
            db_hat_dzeta=geom.db_hat_dzeta,
            force0_radial_current=jnp.asarray(True),
            n_xi_for_x=grids.n_xi_for_x,
        )

    er_xdot = None
    if include_xdot:
        er_xdot = ErXDotV3Operator(
            alpha=jnp.asarray(alpha, dtype=jnp.float64),
            delta=jnp.asarray(delta, dtype=jnp.float64),
            dphi_hat_dpsi_hat=jnp.asarray(dphi, dtype=jnp.float64),
            x=grids.x,
            ddx_plus=grids.ddx,
            ddx_minus=grids.ddx,
            d_hat=geom.d_hat,
            b_hat=geom.b_hat,
            b_hat_sub_theta=geom.b_hat_sub_theta,
            b_hat_sub_zeta=geom.b_hat_sub_zeta,
            db_hat_dtheta=geom.db_hat_dtheta,
            db_hat_dzeta=geom.db_hat_dzeta,
            force0_radial_current=jnp.asarray(True),
            n_xi_for_x=grids.n_xi_for_x,
        )

    return V3FBlockOperator(
        collisionless=colless,
        pas=pas,
        er_xidot=er_xidot,
        er_xdot=er_xdot,
        identity_shift=jnp.asarray(identity_shift, dtype=jnp.float64),
        n_species=int(colless.n_species),
        n_x=int(colless.n_x),
        n_xi=int(grids.n_xi),
        n_theta=int(colless.n_theta),
        n_zeta=int(colless.n_zeta),
    )


def apply_v3_fblock_operator(op: V3FBlockOperator, f: jnp.ndarray) -> jnp.ndarray:
    out = op.identity_shift * f
    out = out + apply_collisionless_v3(op.collisionless, f)
    if op.er_xidot is not None:
        out = out + apply_er_xidot_v3(op.er_xidot, f)
    if op.er_xdot is not None:
        out = out + apply_er_xdot_v3(op.er_xdot, f)
    out = out + apply_pitch_angle_scattering_v3(op.pas, f)
    return out


def matvec_v3_fblock_flat(op: V3FBlockOperator, x_flat: jnp.ndarray) -> jnp.ndarray:
    x_flat = jnp.asarray(x_flat)
    f = x_flat.reshape(op.f_shape)
    y = apply_v3_fblock_operator(op, f)
    return y.reshape((-1,))


def solve_v3_fblock_gmres(
    *,
    op: V3FBlockOperator,
    b_flat: jnp.ndarray,
    x0_flat: jnp.ndarray | None = None,
    tol: float = 1e-10,
    atol: float = 0.0,
    restart: int = 50,
    maxiter: int | None = None,
    solve_method: str = "batched",
) -> GMRESSolveResult:
    b_flat = jnp.asarray(b_flat)
    if b_flat.shape != (op.flat_size,):
        raise ValueError(f"b_flat must have shape {(op.flat_size,)}, got {b_flat.shape}")

    def mv(x):
        return matvec_v3_fblock_flat(op, x)

    return gmres_solve(
        matvec=mv,
        b=b_flat,
        x0=x0_flat,
        tol=tol,
        atol=atol,
        restart=restart,
        maxiter=maxiter,
        solve_method=solve_method,
    )


apply_v3_fblock_operator_jit = jax.jit(apply_v3_fblock_operator, static_argnums=())
matvec_v3_fblock_flat_jit = jax.jit(matvec_v3_fblock_flat, static_argnums=())
