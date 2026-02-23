from __future__ import annotations

from dataclasses import dataclass

from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
from jax import tree_util as jtu

from .periodic_stencil import (
    apply_periodic_stencil_roll,
    apply_sparse_row_stencil_gather,
    periodic_stencil_runtime_enabled,
)


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
    ddtheta_stencil_shifts: tuple[int, ...] = ()
    ddtheta_stencil_coeffs: tuple[float, ...] = ()
    ddzeta_stencil_shifts: tuple[int, ...] = ()
    ddzeta_stencil_coeffs: tuple[float, ...] = ()
    ddtheta_sparse_cols: jnp.ndarray | None = None
    ddtheta_sparse_vals: jnp.ndarray | None = None
    ddzeta_sparse_cols: jnp.ndarray | None = None
    ddzeta_sparse_vals: jnp.ndarray | None = None

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
            self.ddtheta_sparse_cols,
            self.ddtheta_sparse_vals,
            self.ddzeta_sparse_cols,
            self.ddzeta_sparse_vals,
        )
        aux = (
            self.ddtheta_stencil_shifts,
            self.ddtheta_stencil_coeffs,
            self.ddzeta_stencil_shifts,
            self.ddzeta_stencil_coeffs,
        )
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
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
            ddtheta_sparse_cols,
            ddtheta_sparse_vals,
            ddzeta_sparse_cols,
            ddzeta_sparse_vals,
        ) = children
        if aux is None:
            ddtheta_stencil_shifts = ()
            ddtheta_stencil_coeffs = ()
            ddzeta_stencil_shifts = ()
            ddzeta_stencil_coeffs = ()
        else:
            (
                ddtheta_stencil_shifts,
                ddtheta_stencil_coeffs,
                ddzeta_stencil_shifts,
                ddzeta_stencil_coeffs,
            ) = aux
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
            ddtheta_stencil_shifts=tuple(int(v) for v in ddtheta_stencil_shifts),
            ddtheta_stencil_coeffs=tuple(float(v) for v in ddtheta_stencil_coeffs),
            ddzeta_stencil_shifts=tuple(int(v) for v in ddzeta_stencil_shifts),
            ddzeta_stencil_coeffs=tuple(float(v) for v in ddzeta_stencil_coeffs),
            ddtheta_sparse_cols=ddtheta_sparse_cols,
            ddtheta_sparse_vals=ddtheta_sparse_vals,
            ddzeta_sparse_cols=ddzeta_sparse_cols,
            ddzeta_sparse_vals=ddzeta_sparse_vals,
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

    f = jnp.asarray(f, dtype=jnp.float64)
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
    x = op.x

    # Broadcast helpers
    sqrt_t_over_m = jnp.sqrt(op.t_hats / op.m_hats)  # (S,)

    # -------------------------------------------------------------------------
    # Streaming terms (off-diagonal in L): couple L <-> L±1
    # -------------------------------------------------------------------------
    v_theta = (op.b_hat_sup_theta / op.b_hat)  # (T,Z)
    v_zeta = (op.b_hat_sup_zeta / op.b_hat)  # (T,Z)
    v_theta_s = sqrt_t_over_m[:, None, None] * v_theta[None, :, :]  # (S,T,Z)
    v_zeta_s = sqrt_t_over_m[:, None, None] * v_zeta[None, :, :]  # (S,T,Z)

    # d/dtheta applied to f at each (s,x,l,zeta): (S,X,L,T,Z)
    if periodic_stencil_runtime_enabled() and op.ddtheta_stencil_shifts:
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
    dtheta_f = dtheta_f * v_theta_s[:, None, None, :, :]  # row-scaling

    # d/dzeta applied to f at each (s,x,l,theta): (S,X,L,T,Z)
    if periodic_stencil_runtime_enabled() and op.ddzeta_stencil_shifts:
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
    mirror_geom = op.b_hat_sup_theta * op.db_hat_dtheta + op.b_hat_sup_zeta * op.db_hat_dzeta
    mirror_factor = -sqrt_t_over_m[:, None, None] * mirror_geom[None, :, :] / (
        2.0 * (op.b_hat ** 2)
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
