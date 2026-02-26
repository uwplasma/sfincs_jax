from __future__ import annotations

from dataclasses import dataclass

from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
from jax import tree_util as jtu

from .periodic_stencil import (
    apply_periodic_stencil_halo,
    apply_periodic_stencil_roll,
    apply_sparse_row_stencil_gather,
    periodic_stencil_runtime_enabled,
)


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
    ddtheta_stencil_shifts: tuple[int, ...] = ()
    ddtheta_stencil_coeffs: tuple[float, ...] = ()
    ddtheta_sparse_cols: jnp.ndarray | None = None
    ddtheta_sparse_vals: jnp.ndarray | None = None

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
            self.ddtheta_sparse_cols,
            self.ddtheta_sparse_vals,
        )
        aux = (
            bool(self.use_dkes_exb_drift),
            self.ddtheta_stencil_shifts,
            self.ddtheta_stencil_coeffs,
        )
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
            ddtheta_sparse_cols,
            ddtheta_sparse_vals,
        ) = children
        if isinstance(aux, tuple) and len(aux) == 3:
            use_dkes_exb_drift, ddtheta_stencil_shifts, ddtheta_stencil_coeffs = aux
        else:
            # Backward compatibility with older pytree metadata.
            use_dkes_exb_drift = bool(aux)
            ddtheta_stencil_shifts = ()
            ddtheta_stencil_coeffs = ()
        return cls(
            alpha=alpha,
            delta=delta,
            dphi_hat_dpsi_hat=dphi_hat_dpsi_hat,
            ddtheta=ddtheta,
            d_hat=d_hat,
            b_hat=b_hat,
            b_hat_sub_zeta=b_hat_sub_zeta,
            use_dkes_exb_drift=bool(use_dkes_exb_drift),
            fsab_hat2=fsab_hat2,
            n_xi_for_x=n_xi_for_x,
            ddtheta_stencil_shifts=tuple(int(v) for v in ddtheta_stencil_shifts),
            ddtheta_stencil_coeffs=tuple(float(v) for v in ddtheta_stencil_coeffs),
            ddtheta_sparse_cols=ddtheta_sparse_cols,
            ddtheta_sparse_vals=ddtheta_sparse_vals,
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
    ddzeta_stencil_shifts: tuple[int, ...] = ()
    ddzeta_stencil_coeffs: tuple[float, ...] = ()
    ddzeta_sparse_cols: jnp.ndarray | None = None
    ddzeta_sparse_vals: jnp.ndarray | None = None

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
            self.ddzeta_sparse_cols,
            self.ddzeta_sparse_vals,
        )
        aux = (
            bool(self.use_dkes_exb_drift),
            self.ddzeta_stencil_shifts,
            self.ddzeta_stencil_coeffs,
        )
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
            ddzeta_sparse_cols,
            ddzeta_sparse_vals,
        ) = children
        if isinstance(aux, tuple) and len(aux) == 3:
            use_dkes_exb_drift, ddzeta_stencil_shifts, ddzeta_stencil_coeffs = aux
        else:
            use_dkes_exb_drift = bool(aux)
            ddzeta_stencil_shifts = ()
            ddzeta_stencil_coeffs = ()
        return cls(
            alpha=alpha,
            delta=delta,
            dphi_hat_dpsi_hat=dphi_hat_dpsi_hat,
            ddzeta=ddzeta,
            d_hat=d_hat,
            b_hat=b_hat,
            b_hat_sub_theta=b_hat_sub_theta,
            use_dkes_exb_drift=bool(use_dkes_exb_drift),
            fsab_hat2=fsab_hat2,
            n_xi_for_x=n_xi_for_x,
            ddzeta_stencil_shifts=tuple(int(v) for v in ddzeta_stencil_shifts),
            ddzeta_stencil_coeffs=tuple(float(v) for v in ddzeta_stencil_coeffs),
            ddzeta_sparse_cols=ddzeta_sparse_cols,
            ddzeta_sparse_vals=ddzeta_sparse_vals,
        )


def apply_exb_theta_v3(
    op: ExBThetaV3Operator,
    f: jnp.ndarray,
    *,
    shard_axis: str | None = None,
) -> jnp.ndarray:
    """Apply the ExB `d/dtheta` term to `f`.

    Parameters
    ----------
    f:
      Array of shape (Nspecies, Nx, Nxi, Ntheta, Nzeta).
    """
    if f.ndim != 5:
        raise ValueError("f must have shape (Nspecies, Nx, Nxi, Ntheta, Nzeta)")
    f = jnp.asarray(f, dtype=jnp.float64)

    _n_species, n_x, n_xi, n_theta, n_zeta = f.shape
    if n_theta != op.ddtheta.shape[0]:
        raise ValueError("f theta axis does not match ddtheta")
    if n_zeta != op.b_hat.shape[1]:
        raise ValueError("f zeta axis does not match geometry arrays")
    if n_x != op.n_xi_for_x.shape[0]:
        raise ValueError("f x axis does not match n_xi_for_x")

    if op.use_dkes_exb_drift:
        denom = op.fsab_hat2
        coef = (op.d_hat * op.b_hat_sub_zeta / denom)  # (T,Z)
    else:
        denom = (op.b_hat ** 2)
        coef = (op.d_hat * op.b_hat_sub_zeta / denom)  # (T,Z)

    factor = (op.alpha * op.delta * 0.5 * op.dphi_hat_dpsi_hat)

    if periodic_stencil_runtime_enabled() and op.ddtheta_stencil_shifts:
        if shard_axis == "theta" and jax.device_count() > 1:
            dtheta_f = apply_periodic_stencil_halo(
                f,
                shifts=op.ddtheta_stencil_shifts,
                coeffs=op.ddtheta_stencil_coeffs,
                axis=3,
                axis_name="theta",
            )
        else:
            dtheta_f = apply_periodic_stencil_roll(
                f,
                shifts=op.ddtheta_stencil_shifts,
                coeffs=op.ddtheta_stencil_coeffs,
                axis=3,
            )
    elif (
        op.ddtheta_sparse_cols is not None
        and op.ddtheta_sparse_vals is not None
        and int(op.ddtheta_sparse_cols.size) > 0
    ):
        dtheta_f = apply_sparse_row_stencil_gather(
            f,
            cols=op.ddtheta_sparse_cols,
            vals=op.ddtheta_sparse_vals,
            axis=3,
        )
    else:
        dtheta_f = jnp.einsum("ij,sxljz->sxliz", op.ddtheta, f)
    out = factor * dtheta_f * coef[None, None, None, :, :]

    mask = _mask_xi(op.n_xi_for_x.astype(jnp.int32), n_xi).astype(out.dtype)  # (X,L)
    return out * mask[None, :, :, None, None]


def apply_exb_zeta_v3(
    op: ExBZetaV3Operator,
    f: jnp.ndarray,
    *,
    shard_axis: str | None = None,
) -> jnp.ndarray:
    """Apply the ExB `d/dzeta` term to `f`."""
    if f.ndim != 5:
        raise ValueError("f must have shape (Nspecies, Nx, Nxi, Ntheta, Nzeta)")
    f = jnp.asarray(f, dtype=jnp.float64)

    _n_species, n_x, n_xi, n_theta, n_zeta = f.shape
    if n_zeta != op.ddzeta.shape[0]:
        raise ValueError("f zeta axis does not match ddzeta")
    if n_theta != op.b_hat.shape[0]:
        raise ValueError("f theta axis does not match geometry arrays")
    if n_x != op.n_xi_for_x.shape[0]:
        raise ValueError("f x axis does not match n_xi_for_x")

    if op.use_dkes_exb_drift:
        denom = op.fsab_hat2
        coef = (op.d_hat * op.b_hat_sub_theta / denom)  # (T,Z)
    else:
        denom = (op.b_hat ** 2)
        coef = (op.d_hat * op.b_hat_sub_theta / denom)  # (T,Z)

    factor = (-op.alpha * op.delta * 0.5 * op.dphi_hat_dpsi_hat)

    if periodic_stencil_runtime_enabled() and op.ddzeta_stencil_shifts:
        if shard_axis == "zeta" and jax.device_count() > 1:
            dzeta_f = apply_periodic_stencil_halo(
                f,
                shifts=op.ddzeta_stencil_shifts,
                coeffs=op.ddzeta_stencil_coeffs,
                axis=4,
                axis_name="zeta",
            )
        else:
            dzeta_f = apply_periodic_stencil_roll(
                f,
                shifts=op.ddzeta_stencil_shifts,
                coeffs=op.ddzeta_stencil_coeffs,
                axis=4,
            )
    elif (
        op.ddzeta_sparse_cols is not None
        and op.ddzeta_sparse_vals is not None
        and int(op.ddzeta_sparse_cols.size) > 0
    ):
        dzeta_f = apply_sparse_row_stencil_gather(
            f,
            cols=op.ddzeta_sparse_cols,
            vals=op.ddzeta_sparse_vals,
            axis=4,
        )
    else:
        dzeta_f = jnp.einsum("ij,sxltj->sxlti", op.ddzeta, f)
    out = factor * dzeta_f * coef[None, None, None, :, :]

    mask = _mask_xi(op.n_xi_for_x.astype(jnp.int32), n_xi).astype(out.dtype)  # (X,L)
    return out * mask[None, :, :, None, None]


apply_exb_theta_v3_jit = jax.jit(apply_exb_theta_v3, static_argnames=("shard_axis",))
apply_exb_zeta_v3_jit = jax.jit(apply_exb_zeta_v3, static_argnames=("shard_axis",))
