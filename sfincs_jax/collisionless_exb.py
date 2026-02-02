from __future__ import annotations

from dataclasses import dataclass

from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
from jax import tree_util as jtu


def _mask_xi(n_xi_for_x: jnp.ndarray, n_xi_max: int) -> jnp.ndarray:
    # (Nx, Nxi)
    l = jnp.arange(n_xi_max, dtype=jnp.int32)[None, :]
    return l < n_xi_for_x[:, None]


@jtu.register_pytree_node_class
@dataclass(frozen=True)
class ExBThetaV3Operator:
    """ExB drift term proportional to ∂/∂θ (v3 Boozer formulation).

    This term is present in the v3 Jacobian whenever `whichMatrix != 2` and is
    proportional to `dPhiHat/dpsiHat`, which is in turn derived from `Er` by
    radial-coordinate conventions.
    """

    alpha: jnp.ndarray  # scalar
    delta: jnp.ndarray  # scalar
    dphi_hat_dpsi_hat: jnp.ndarray  # scalar

    ddtheta: jnp.ndarray  # (Ntheta, Ntheta)
    d_hat: jnp.ndarray  # (Ntheta, Nzeta)
    b_hat: jnp.ndarray  # (Ntheta, Nzeta)
    b_hat_sub_zeta: jnp.ndarray  # (Ntheta, Nzeta)

    use_dkes_exb_drift: bool
    fsab_hat2: jnp.ndarray  # scalar, only used when use_dkes_exb_drift=True

    n_xi_for_x: jnp.ndarray  # (Nx,) int32

    def tree_flatten(self):
        children = (
            self.alpha,
            self.delta,
            self.dphi_hat_dpsi_hat,
            self.ddtheta,
            self.d_hat,
            self.b_hat,
            self.b_hat_sub_zeta,
            self.fsab_hat2,
            self.n_xi_for_x,
        )
        aux = bool(self.use_dkes_exb_drift)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        (
            alpha,
            delta,
            dphi_hat_dpsi_hat,
            ddtheta,
            d_hat,
            b_hat,
            b_hat_sub_zeta,
            fsab_hat2,
            n_xi_for_x,
        ) = children
        return cls(
            alpha=alpha,
            delta=delta,
            dphi_hat_dpsi_hat=dphi_hat_dpsi_hat,
            ddtheta=ddtheta,
            d_hat=d_hat,
            b_hat=b_hat,
            b_hat_sub_zeta=b_hat_sub_zeta,
            use_dkes_exb_drift=bool(aux),
            fsab_hat2=fsab_hat2,
            n_xi_for_x=n_xi_for_x,
        )


@jtu.register_pytree_node_class
@dataclass(frozen=True)
class ExBZetaV3Operator:
    """ExB drift term proportional to ∂/∂ζ (v3 Boozer formulation)."""

    alpha: jnp.ndarray  # scalar
    delta: jnp.ndarray  # scalar
    dphi_hat_dpsi_hat: jnp.ndarray  # scalar

    ddzeta: jnp.ndarray  # (Nzeta, Nzeta)
    d_hat: jnp.ndarray  # (Ntheta, Nzeta)
    b_hat: jnp.ndarray  # (Ntheta, Nzeta)
    b_hat_sub_theta: jnp.ndarray  # (Ntheta, Nzeta)

    use_dkes_exb_drift: bool
    fsab_hat2: jnp.ndarray  # scalar, only used when use_dkes_exb_drift=True

    n_xi_for_x: jnp.ndarray  # (Nx,) int32

    def tree_flatten(self):
        children = (
            self.alpha,
            self.delta,
            self.dphi_hat_dpsi_hat,
            self.ddzeta,
            self.d_hat,
            self.b_hat,
            self.b_hat_sub_theta,
            self.fsab_hat2,
            self.n_xi_for_x,
        )
        aux = bool(self.use_dkes_exb_drift)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        (
            alpha,
            delta,
            dphi_hat_dpsi_hat,
            ddzeta,
            d_hat,
            b_hat,
            b_hat_sub_theta,
            fsab_hat2,
            n_xi_for_x,
        ) = children
        return cls(
            alpha=alpha,
            delta=delta,
            dphi_hat_dpsi_hat=dphi_hat_dpsi_hat,
            ddzeta=ddzeta,
            d_hat=d_hat,
            b_hat=b_hat,
            b_hat_sub_theta=b_hat_sub_theta,
            use_dkes_exb_drift=bool(aux),
            fsab_hat2=fsab_hat2,
            n_xi_for_x=n_xi_for_x,
        )


def apply_exb_theta_v3(op: ExBThetaV3Operator, f: jnp.ndarray) -> jnp.ndarray:
    """Apply the ExB `d/dtheta` term to `f`.

    Parameters
    ----------
    f:
      Array of shape (Nspecies, Nx, Nxi, Ntheta, Nzeta).
    """
    if f.ndim != 5:
        raise ValueError("f must have shape (Nspecies, Nx, Nxi, Ntheta, Nzeta)")

    _n_species, n_x, n_xi, n_theta, n_zeta = f.shape
    if n_theta != op.ddtheta.shape[0]:
        raise ValueError("f theta axis does not match ddtheta")
    if n_zeta != op.b_hat.shape[1]:
        raise ValueError("f zeta axis does not match geometry arrays")
    if n_x != op.n_xi_for_x.shape[0]:
        raise ValueError("f x axis does not match n_xi_for_x")

    if op.use_dkes_exb_drift:
        denom = op.fsab_hat2.astype(jnp.float64)
        coef = (op.d_hat * op.b_hat_sub_zeta / denom).astype(jnp.float64)  # (T,Z)
    else:
        denom = (op.b_hat.astype(jnp.float64) ** 2)
        coef = (op.d_hat * op.b_hat_sub_zeta / denom).astype(jnp.float64)  # (T,Z)

    factor = (op.alpha * op.delta * 0.5 * op.dphi_hat_dpsi_hat).astype(jnp.float64)

    dtheta_f = jnp.einsum("ij,sxljz->sxliz", op.ddtheta.astype(jnp.float64), f.astype(jnp.float64))
    out = factor * dtheta_f * coef[None, None, None, :, :]

    mask = _mask_xi(op.n_xi_for_x.astype(jnp.int32), n_xi).astype(out.dtype)  # (X,L)
    return out * mask[None, :, :, None, None]


def apply_exb_zeta_v3(op: ExBZetaV3Operator, f: jnp.ndarray) -> jnp.ndarray:
    """Apply the ExB `d/dzeta` term to `f`."""
    if f.ndim != 5:
        raise ValueError("f must have shape (Nspecies, Nx, Nxi, Ntheta, Nzeta)")

    _n_species, n_x, n_xi, n_theta, n_zeta = f.shape
    if n_zeta != op.ddzeta.shape[0]:
        raise ValueError("f zeta axis does not match ddzeta")
    if n_theta != op.b_hat.shape[0]:
        raise ValueError("f theta axis does not match geometry arrays")
    if n_x != op.n_xi_for_x.shape[0]:
        raise ValueError("f x axis does not match n_xi_for_x")

    if op.use_dkes_exb_drift:
        denom = op.fsab_hat2.astype(jnp.float64)
        coef = (op.d_hat * op.b_hat_sub_theta / denom).astype(jnp.float64)  # (T,Z)
    else:
        denom = (op.b_hat.astype(jnp.float64) ** 2)
        coef = (op.d_hat * op.b_hat_sub_theta / denom).astype(jnp.float64)  # (T,Z)

    factor = (-op.alpha * op.delta * 0.5 * op.dphi_hat_dpsi_hat).astype(jnp.float64)

    dzeta_f = jnp.einsum("ij,sxltj->sxlti", op.ddzeta.astype(jnp.float64), f.astype(jnp.float64))
    out = factor * dzeta_f * coef[None, None, None, :, :]

    mask = _mask_xi(op.n_xi_for_x.astype(jnp.int32), n_xi).astype(out.dtype)  # (X,L)
    return out * mask[None, :, :, None, None]


apply_exb_theta_v3_jit = jax.jit(apply_exb_theta_v3, static_argnums=())
apply_exb_zeta_v3_jit = jax.jit(apply_exb_zeta_v3, static_argnums=())
