from __future__ import annotations

from dataclasses import dataclass

from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
from jax import tree_util as jtu


@jtu.register_pytree_node_class
@dataclass(frozen=True)
class CollisionlessV3Operator:
    """Collisionless part of the v3 kinetic operator (streaming + mirror).

    This operator is intentionally partial. It is meant as an incremental, parity-tested
    building block for the full v3 Jacobian/residual.
    """

    x: jnp.ndarray  # (Nx,)
    ddtheta: jnp.ndarray  # (Ntheta, Ntheta)
    ddzeta: jnp.ndarray  # (Nzeta, Nzeta)

    b_hat: jnp.ndarray  # (Ntheta, Nzeta)
    b_hat_sup_theta: jnp.ndarray  # (Ntheta, Nzeta)
    b_hat_sup_zeta: jnp.ndarray  # (Ntheta, Nzeta)
    db_hat_dtheta: jnp.ndarray  # (Ntheta, Nzeta)
    db_hat_dzeta: jnp.ndarray  # (Ntheta, Nzeta)

    t_hats: jnp.ndarray  # (Nspecies,)
    m_hats: jnp.ndarray  # (Nspecies,)
    n_xi_for_x: jnp.ndarray  # (Nx,) int32

    @property
    def n_species(self) -> int:
        return int(self.t_hats.shape[0])

    @property
    def n_x(self) -> int:
        return int(self.x.shape[0])

    @property
    def n_theta(self) -> int:
        return int(self.ddtheta.shape[0])

    @property
    def n_zeta(self) -> int:
        return int(self.ddzeta.shape[0])

    def tree_flatten(self):
        children = (
            self.x,
            self.ddtheta,
            self.ddzeta,
            self.b_hat,
            self.b_hat_sup_theta,
            self.b_hat_sup_zeta,
            self.db_hat_dtheta,
            self.db_hat_dzeta,
            self.t_hats,
            self.m_hats,
            self.n_xi_for_x,
        )
        aux = None
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        del aux
        (
            x,
            ddtheta,
            ddzeta,
            b_hat,
            b_hat_sup_theta,
            b_hat_sup_zeta,
            db_hat_dtheta,
            db_hat_dzeta,
            t_hats,
            m_hats,
            n_xi_for_x,
        ) = children
        return cls(
            x=x,
            ddtheta=ddtheta,
            ddzeta=ddzeta,
            b_hat=b_hat,
            b_hat_sup_theta=b_hat_sup_theta,
            b_hat_sup_zeta=b_hat_sup_zeta,
            db_hat_dtheta=db_hat_dtheta,
            db_hat_dzeta=db_hat_dzeta,
            t_hats=t_hats,
            m_hats=m_hats,
            n_xi_for_x=n_xi_for_x,
        )


def _mask_xi(n_xi_for_x: jnp.ndarray, n_xi_max: int) -> jnp.ndarray:
    # (Nx, Nxi)
    l = jnp.arange(n_xi_max, dtype=jnp.int32)[None, :]
    return l < n_xi_for_x[:, None]


def apply_collisionless_v3(op: CollisionlessV3Operator, f: jnp.ndarray) -> jnp.ndarray:
    """Apply streaming+mirror terms to `f`.

    Parameters
    ----------
    f:
      Array of shape (Nspecies, Nx, Nxi, Ntheta, Nzeta).

    Returns
    -------
    out:
      Array of same shape.
    """
    if f.ndim != 5:
        raise ValueError("f must have shape (Nspecies, Nx, Nxi, Ntheta, Nzeta)")

    n_species, n_x, n_xi, n_theta, n_zeta = f.shape
    if n_species != op.t_hats.shape[0]:
        raise ValueError("f species axis does not match t_hats")
    if n_x != op.x.shape[0]:
        raise ValueError("f x axis does not match x")
    if n_theta != op.ddtheta.shape[0]:
        raise ValueError("f theta axis does not match ddtheta")
    if n_zeta != op.ddzeta.shape[0]:
        raise ValueError("f zeta axis does not match ddzeta")

    l = jnp.arange(n_xi, dtype=jnp.float64)  # row L
    x = op.x.astype(jnp.float64)

    # Broadcast helpers
    sqrt_t_over_m = jnp.sqrt(op.t_hats / op.m_hats).astype(jnp.float64)  # (S,)

    # -------------------------------------------------------------------------
    # Streaming terms (off-diagonal in L): couple L <-> L±1
    # -------------------------------------------------------------------------
    v_theta = (op.b_hat_sup_theta / op.b_hat).astype(jnp.float64)  # (T,Z)
    v_zeta = (op.b_hat_sup_zeta / op.b_hat).astype(jnp.float64)  # (T,Z)
    v_theta_s = sqrt_t_over_m[:, None, None] * v_theta[None, :, :]  # (S,T,Z)
    v_zeta_s = sqrt_t_over_m[:, None, None] * v_zeta[None, :, :]  # (S,T,Z)

    # d/dtheta applied to f at each (s,x,l,zeta): (S,X,L,T,Z)
    dtheta_f = jnp.einsum("ij,sxljz->sxliz", op.ddtheta.astype(jnp.float64), f.astype(jnp.float64))
    dtheta_f = dtheta_f * v_theta_s[:, None, None, :, :]  # row-scaling

    # d/dzeta applied to f at each (s,x,l,theta): (S,X,L,T,Z)
    dzeta_f = jnp.einsum("ij,sxltj->sxlti", op.ddzeta.astype(jnp.float64), f.astype(jnp.float64))
    dzeta_f = dzeta_f * v_zeta_s[:, None, None, :, :]

    # L-coupling coefficients for row L:
    coef_plus = (l + 1.0) / (2.0 * l + 3.0)  # (L,)
    coef_minus = jnp.where(l > 0, l / (2.0 * l - 1.0), 0.0)  # (L,)

    coef_plus_x = x[:, None] * coef_plus[None, :]  # (X,L)
    coef_minus_x = x[:, None] * coef_minus[None, :]  # (X,L)

    def couple_l(g: jnp.ndarray) -> jnp.ndarray:
        # g shape (S,X,L,T,Z) is evaluated at column ell.
        # out row L receives from ell=L+1 and ell=L-1.
        term_plus = coef_plus_x[None, :, :-1, None, None] * g[:, :, 1:, :, :]
        term_plus = jnp.pad(term_plus, ((0, 0), (0, 0), (0, 1), (0, 0), (0, 0)))

        term_minus = coef_minus_x[None, :, 1:, None, None] * g[:, :, :-1, :, :]
        term_minus = jnp.pad(term_minus, ((0, 0), (0, 0), (1, 0), (0, 0), (0, 0)))
        return term_plus + term_minus

    out_streaming = couple_l(dtheta_f) + couple_l(dzeta_f)

    # -------------------------------------------------------------------------
    # Mirror term (off-diagonal in L): couple L <-> L±1
    # -------------------------------------------------------------------------
    mirror_geom = (
        op.b_hat_sup_theta * op.db_hat_dtheta + op.b_hat_sup_zeta * op.db_hat_dzeta
    ).astype(jnp.float64)
    mirror_factor = -sqrt_t_over_m[:, None, None] * mirror_geom[None, :, :] / (
        2.0 * (op.b_hat.astype(jnp.float64) ** 2)
    )  # (S,T,Z)

    coef_mirror_plus = (l + 1.0) * (l + 2.0) / (2.0 * l + 3.0)  # (L,)
    coef_mirror_minus = jnp.where(l > 1, -l * (l - 1.0) / (2.0 * l - 1.0), 0.0)  # (L,)
    coef_mirror_plus_x = x[:, None] * coef_mirror_plus[None, :]
    coef_mirror_minus_x = x[:, None] * coef_mirror_minus[None, :]

    # mirror uses f at column ell=L±1 without derivatives.
    term_plus = coef_mirror_plus_x[None, :, :-1, None, None] * f[:, :, 1:, :, :]
    term_plus = jnp.pad(term_plus, ((0, 0), (0, 0), (0, 1), (0, 0), (0, 0)))

    term_minus = coef_mirror_minus_x[None, :, 1:, None, None] * f[:, :, :-1, :, :]
    term_minus = jnp.pad(term_minus, ((0, 0), (0, 0), (1, 0), (0, 0), (0, 0)))

    out_mirror = (term_plus + term_minus) * mirror_factor[:, None, None, :, :]

    out = out_streaming + out_mirror

    # Mask invalid xi modes per x.
    mask = _mask_xi(op.n_xi_for_x.astype(jnp.int32), n_xi).astype(out.dtype)  # (X,L)
    out = out * mask[None, :, :, None, None]
    return out


apply_collisionless_v3_jit = jax.jit(apply_collisionless_v3, static_argnums=())
