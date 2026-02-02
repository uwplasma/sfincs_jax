from __future__ import annotations

from dataclasses import dataclass

from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
from jax import tree_util as jtu


def _mask_xi(n_xi_for_x: jnp.ndarray, n_xi_max: int) -> jnp.ndarray:
    l = jnp.arange(n_xi_max, dtype=jnp.int32)[None, :]
    return l < n_xi_for_x[:, None]


@jtu.register_pytree_node_class
@dataclass(frozen=True)
class ErXiDotV3Operator:
    """Non-standard d/dxi term associated with E_r (v3 collisionless term).

    Matches the block in `populateMatrix.F90` labeled:
      \"Add the non-standard d/dxi term associated with E_r\".

    This term couples Legendre modes L <-> L±2 (and has a diagonal-in-L piece).
    """

    alpha: jnp.ndarray  # scalar
    delta: jnp.ndarray  # scalar
    dphi_hat_dpsi_hat: jnp.ndarray  # scalar

    d_hat: jnp.ndarray  # (T,Z)
    b_hat: jnp.ndarray  # (T,Z)

    b_hat_sub_theta: jnp.ndarray  # (T,Z)
    b_hat_sub_zeta: jnp.ndarray  # (T,Z)

    db_hat_dtheta: jnp.ndarray  # (T,Z)
    db_hat_dzeta: jnp.ndarray  # (T,Z)

    # For now we assume force0RadialCurrentInEquilibrium=.true. (v3 default),
    # so the extra derivative term is omitted.
    force0_radial_current: jnp.ndarray  # scalar bool (stored as bool array)

    n_xi_for_x: jnp.ndarray  # (X,) int32

    def tree_flatten(self):
        children = (
            self.alpha,
            self.delta,
            self.dphi_hat_dpsi_hat,
            self.d_hat,
            self.b_hat,
            self.b_hat_sub_theta,
            self.b_hat_sub_zeta,
            self.db_hat_dtheta,
            self.db_hat_dzeta,
            self.force0_radial_current,
            self.n_xi_for_x,
        )
        aux = None
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        del aux
        (
            alpha,
            delta,
            dphi_hat_dpsi_hat,
            d_hat,
            b_hat,
            b_hat_sub_theta,
            b_hat_sub_zeta,
            db_hat_dtheta,
            db_hat_dzeta,
            force0_radial_current,
            n_xi_for_x,
        ) = children
        return cls(
            alpha=alpha,
            delta=delta,
            dphi_hat_dpsi_hat=dphi_hat_dpsi_hat,
            d_hat=d_hat,
            b_hat=b_hat,
            b_hat_sub_theta=b_hat_sub_theta,
            b_hat_sub_zeta=b_hat_sub_zeta,
            db_hat_dtheta=db_hat_dtheta,
            db_hat_dzeta=db_hat_dzeta,
            force0_radial_current=force0_radial_current,
            n_xi_for_x=n_xi_for_x,
        )


def apply_er_xidot_v3(op: ErXiDotV3Operator, f: jnp.ndarray) -> jnp.ndarray:
    """Apply the v3 Er xiDot term to `f`.

    Parameters
    ----------
    f:
      Array of shape (Nspecies, Nx, Nxi, Ntheta, Nzeta).
    """
    if f.ndim != 5:
        raise ValueError("f must have shape (Nspecies, Nx, Nxi, Ntheta, Nzeta)")
    _, n_x, n_xi, _, _ = f.shape
    if n_x != int(op.n_xi_for_x.shape[0]):
        raise ValueError("n_x axis does not match n_xi_for_x")

    # For geometryScheme=4 and default force0RadialCurrentInEquilibrium=.true.,
    # temp = (B_sub_zeta dB/dtheta - B_sub_theta dB/dzeta).
    temp = op.b_hat_sub_zeta * op.db_hat_dtheta - op.b_hat_sub_theta * op.db_hat_dzeta
    factor = (
        op.alpha
        * op.delta
        * op.dphi_hat_dpsi_hat
        / (4.0 * (op.b_hat**3))
        * op.d_hat
        * temp
    )  # (T,Z)

    l = jnp.arange(n_xi, dtype=jnp.float64)  # row L
    denom0 = (2.0 * l - 1.0) * (2.0 * l + 3.0)
    diag_coef = (l + 1.0) * l / denom0  # (L,)

    # ±2 couplings:
    sup2_coef = (l + 3.0) * (l + 2.0) * (l + 1.0) / ((2.0 * l + 5.0) * (2.0 * l + 3.0))
    sub2_coef = -l * (l - 1.0) * (l - 2.0) / ((2.0 * l - 3.0) * (2.0 * l - 1.0))

    out = jnp.zeros_like(f, dtype=jnp.float64)

    # Diagonal in L:
    out = out + (factor[None, None, None, :, :] * diag_coef[None, None, :, None, None]) * f

    # Super-super (L receives from L+2):
    term_sup2 = sup2_coef[None, None, :-2, None, None] * f[:, :, 2:, :, :]
    term_sup2 = jnp.pad(term_sup2, ((0, 0), (0, 0), (0, 2), (0, 0), (0, 0)))
    out = out + factor[None, None, None, :, :] * term_sup2

    # Sub-sub (L receives from L-2):
    term_sub2 = sub2_coef[None, None, 2:, None, None] * f[:, :, :-2, :, :]
    term_sub2 = jnp.pad(term_sub2, ((0, 0), (0, 0), (2, 0), (0, 0), (0, 0)))
    out = out + factor[None, None, None, :, :] * term_sub2

    # Mask invalid xi modes per x.
    mask = _mask_xi(op.n_xi_for_x.astype(jnp.int32), n_xi).astype(out.dtype)  # (X,L)
    return out * mask[None, :, :, None, None]


apply_er_xidot_v3_jit = jax.jit(apply_er_xidot_v3, static_argnums=())


def apply_er_xidot_v3_offdiag2(op: ErXiDotV3Operator, f: jnp.ndarray) -> jnp.ndarray:
    """Apply only the :math:`\\Delta L = \\pm 2` couplings of the v3 Er xiDot term."""
    if f.ndim != 5:
        raise ValueError("f must have shape (Nspecies, Nx, Nxi, Ntheta, Nzeta)")
    _, n_x, n_xi, _, _ = f.shape
    if n_x != int(op.n_xi_for_x.shape[0]):
        raise ValueError("n_x axis does not match n_xi_for_x")

    temp = op.b_hat_sub_zeta * op.db_hat_dtheta - op.b_hat_sub_theta * op.db_hat_dzeta
    factor = (
        op.alpha
        * op.delta
        * op.dphi_hat_dpsi_hat
        / (4.0 * (op.b_hat**3))
        * op.d_hat
        * temp
    )  # (T,Z)

    l = jnp.arange(n_xi, dtype=jnp.float64)
    sup2_coef = (l + 3.0) * (l + 2.0) * (l + 1.0) / ((2.0 * l + 5.0) * (2.0 * l + 3.0))
    sub2_coef = -l * (l - 1.0) * (l - 2.0) / ((2.0 * l - 3.0) * (2.0 * l - 1.0))

    out = jnp.zeros_like(f, dtype=jnp.float64)

    term_sup2 = sup2_coef[None, None, :-2, None, None] * f[:, :, 2:, :, :]
    term_sup2 = jnp.pad(term_sup2, ((0, 0), (0, 0), (0, 2), (0, 0), (0, 0)))
    out = out + factor[None, None, None, :, :] * term_sup2

    term_sub2 = sub2_coef[None, None, 2:, None, None] * f[:, :, :-2, :, :]
    term_sub2 = jnp.pad(term_sub2, ((0, 0), (0, 0), (2, 0), (0, 0), (0, 0)))
    out = out + factor[None, None, None, :, :] * term_sub2

    mask = _mask_xi(op.n_xi_for_x.astype(jnp.int32), n_xi).astype(out.dtype)
    return out * mask[None, :, :, None, None]


apply_er_xidot_v3_offdiag2_jit = jax.jit(apply_er_xidot_v3_offdiag2, static_argnums=())


@jtu.register_pytree_node_class
@dataclass(frozen=True)
class ErXDotV3Operator:
    """Collisionless d/dx term associated with E_r (v3 xDot term).

    Matches the block in `populateMatrix.F90` labeled:
      \"Add the collisionless d/dx term associated with E_r\".

    This term couples in x (dense ddx matvec) and couples L <-> L±2.
    """

    alpha: jnp.ndarray  # scalar
    delta: jnp.ndarray  # scalar
    dphi_hat_dpsi_hat: jnp.ndarray  # scalar

    x: jnp.ndarray  # (X,)
    ddx_plus: jnp.ndarray  # (X,X)
    ddx_minus: jnp.ndarray  # (X,X)

    d_hat: jnp.ndarray  # (T,Z)
    b_hat: jnp.ndarray  # (T,Z)

    b_hat_sub_theta: jnp.ndarray  # (T,Z)
    b_hat_sub_zeta: jnp.ndarray  # (T,Z)

    db_hat_dtheta: jnp.ndarray  # (T,Z)
    db_hat_dzeta: jnp.ndarray  # (T,Z)

    # default v3: force0RadialCurrentInEquilibrium=.true., so xDotFactor2=0
    force0_radial_current: jnp.ndarray  # scalar bool

    n_xi_for_x: jnp.ndarray  # (X,) int32

    def tree_flatten(self):
        children = (
            self.alpha,
            self.delta,
            self.dphi_hat_dpsi_hat,
            self.x,
            self.ddx_plus,
            self.ddx_minus,
            self.d_hat,
            self.b_hat,
            self.b_hat_sub_theta,
            self.b_hat_sub_zeta,
            self.db_hat_dtheta,
            self.db_hat_dzeta,
            self.force0_radial_current,
            self.n_xi_for_x,
        )
        aux = None
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        del aux
        (
            alpha,
            delta,
            dphi_hat_dpsi_hat,
            x,
            ddx_plus,
            ddx_minus,
            d_hat,
            b_hat,
            b_hat_sub_theta,
            b_hat_sub_zeta,
            db_hat_dtheta,
            db_hat_dzeta,
            force0_radial_current,
            n_xi_for_x,
        ) = children
        return cls(
            alpha=alpha,
            delta=delta,
            dphi_hat_dpsi_hat=dphi_hat_dpsi_hat,
            x=x,
            ddx_plus=ddx_plus,
            ddx_minus=ddx_minus,
            d_hat=d_hat,
            b_hat=b_hat,
            b_hat_sub_theta=b_hat_sub_theta,
            b_hat_sub_zeta=b_hat_sub_zeta,
            db_hat_dtheta=db_hat_dtheta,
            db_hat_dzeta=db_hat_dzeta,
            force0_radial_current=force0_radial_current,
            n_xi_for_x=n_xi_for_x,
        )


def apply_er_xdot_v3(op: ErXDotV3Operator, f: jnp.ndarray) -> jnp.ndarray:
    """Apply the v3 Er xDot term to `f`.

    Parameters
    ----------
    f:
      Array of shape (Nspecies, Nx, Nxi, Ntheta, Nzeta).
    """
    if f.ndim != 5:
        raise ValueError("f must have shape (Nspecies, Nx, Nxi, Ntheta, Nzeta)")
    _, n_x, n_xi, _, _ = f.shape
    if n_x != int(op.x.shape[0]):
        raise ValueError("x axis does not match")

    factor0 = -(op.alpha * op.delta * op.dphi_hat_dpsi_hat) / 4.0  # adjointFactor=1
    xdot_factor = (
        factor0
        * op.d_hat
        / (op.b_hat**3)
        * (op.b_hat_sub_theta * op.db_hat_dzeta - op.b_hat_sub_zeta * op.db_hat_dtheta)
    )  # (T,Z)

    # xDotFactor2 is omitted for force0RadialCurrentInEquilibrium=.true. (v3 default).
    xdot_factor2 = jnp.zeros_like(xdot_factor)

    # Upwinding in x (choose plus/minus based on sign of xdot_factor).
    x_part_plus = op.x[:, None] * op.ddx_plus  # (X,X)
    x_part_minus = op.x[:, None] * op.ddx_minus

    def xmatmul(x_part: jnp.ndarray, g: jnp.ndarray) -> jnp.ndarray:
        # x_part (X,X), g (...,X) -> (...,X) with last axis being X.
        return jnp.einsum("ij,...j->...i", x_part, g)

    # Apply along x for each (s,L,theta,zeta).
    f_sxltz = f.astype(jnp.float64)

    l = jnp.arange(n_xi, dtype=jnp.float64)
    denom = (2.0 * l + 3.0) * (2.0 * l - 1.0)
    diag_coef = 2.0 * (3.0 * l * l + 3.0 * l - 2.0) / denom
    diag_coef2 = (2.0 * l * l + 2.0 * l - 1.0) / denom
    diag_stuff = diag_coef * xdot_factor[:, :, None] + diag_coef2 * xdot_factor2[:, :, None]  # (T,Z,L)
    diag_stuff_ltz = jnp.transpose(diag_stuff, (2, 0, 1))  # (L,T,Z)

    # Prepare sign-based selection per (theta,zeta).
    use_plus = (xdot_factor > 0.0)  # (T,Z)

    # Diagonal in L term:
    g_diag = f_sxltz  # (S,X,L,T,Z)
    # Move X last to use einsum helper:
    g_diag_xlast = jnp.transpose(g_diag, (0, 2, 3, 4, 1))  # (S,L,T,Z,X)
    y_plus = xmatmul(x_part_plus, g_diag_xlast)
    y_minus = xmatmul(x_part_minus, g_diag_xlast)
    y = jnp.where(use_plus[None, None, :, :, None], y_plus, y_minus)  # (S,L,T,Z,X)
    y = jnp.transpose(y, (0, 4, 1, 2, 3))  # (S,X,L,T,Z)
    out = y * diag_stuff_ltz[None, None, :, :, :]

    # Off-by-2 terms (L±2):
    off_stuff = (xdot_factor + xdot_factor2)  # (T,Z)

    # Super-super: rows L get columns ell=L+2
    if n_xi >= 3:
        l0 = l[:-2]
        sup_stuff = (l0 + 1.0) * (l0 + 2.0) / ((2.0 * l0 + 5.0) * (2.0 * l0 + 3.0))  # (L-2,)
        g_sup = f_sxltz[:, :, 2:, :, :]  # (S,X,L-2,T,Z) columns
        g_sup_xlast = jnp.transpose(g_sup, (0, 2, 3, 4, 1))  # (S,L-2,T,Z,X)
        y_plus = xmatmul(x_part_plus, g_sup_xlast)
        y_minus = xmatmul(x_part_minus, g_sup_xlast)
        y_sup = jnp.where(use_plus[None, None, :, :, None], y_plus, y_minus)  # (S,L-2,T,Z,X)
        y_sup = jnp.transpose(y_sup, (0, 4, 1, 2, 3))  # (S,X,L-2,T,Z)
        # Pad back to L dimension:
        y_sup = jnp.pad(y_sup, ((0, 0), (0, 0), (0, 2), (0, 0), (0, 0)))
        sup_coef = jnp.pad(sup_stuff, (0, 2))  # (L,)
        coef = sup_coef[:, None, None] * off_stuff[None, :, :]  # (L,T,Z)
        out = out + y_sup * coef[None, None, :, :, :]

    # Sub-sub: rows L get columns ell=L-2
    if n_xi >= 3:
        l2 = l[2:]
        sub_stuff = l2 * (l2 - 1.0) / ((2.0 * l2 - 3.0) * (2.0 * l2 - 1.0))  # (L-2,)
        g_sub = f_sxltz[:, :, :-2, :, :]  # columns ell=L-2
        g_sub_xlast = jnp.transpose(g_sub, (0, 2, 3, 4, 1))  # (S,L-2,T,Z,X)
        y_plus = xmatmul(x_part_plus, g_sub_xlast)
        y_minus = xmatmul(x_part_minus, g_sub_xlast)
        y_sub = jnp.where(use_plus[None, None, :, :, None], y_plus, y_minus)
        y_sub = jnp.transpose(y_sub, (0, 4, 1, 2, 3))  # (S,X,L-2,T,Z)
        y_sub = jnp.pad(y_sub, ((0, 0), (0, 0), (2, 0), (0, 0), (0, 0)))
        sub_coef = jnp.pad(sub_stuff, (2, 0))  # (L,)
        coef = sub_coef[:, None, None] * off_stuff[None, :, :]  # (L,T,Z)
        out = out + y_sub * coef[None, None, :, :, :]

    # Mask invalid xi modes per x.
    mask = _mask_xi(op.n_xi_for_x.astype(jnp.int32), n_xi).astype(out.dtype)  # (X,L)
    return out * mask[None, :, :, None, None]


apply_er_xdot_v3_jit = jax.jit(apply_er_xdot_v3, static_argnums=())


def apply_er_xdot_v3_offdiag2(op: ErXDotV3Operator, f: jnp.ndarray) -> jnp.ndarray:
    """Apply only the :math:`\\Delta L = \\pm 2` couplings of the v3 Er xDot term."""
    if f.ndim != 5:
        raise ValueError("f must have shape (Nspecies, Nx, Nxi, Ntheta, Nzeta)")
    _, n_x, n_xi, _, _ = f.shape
    if n_x != int(op.x.shape[0]):
        raise ValueError("x axis does not match")

    factor0 = -(op.alpha * op.delta * op.dphi_hat_dpsi_hat) / 4.0  # adjointFactor=1
    xdot_factor = (
        factor0
        * op.d_hat
        / (op.b_hat**3)
        * (op.b_hat_sub_theta * op.db_hat_dzeta - op.b_hat_sub_zeta * op.db_hat_dtheta)
    )  # (T,Z)
    xdot_factor2 = jnp.zeros_like(xdot_factor)
    off_stuff = (xdot_factor + xdot_factor2)  # (T,Z)

    x_part_plus = op.x[:, None] * op.ddx_plus
    x_part_minus = op.x[:, None] * op.ddx_minus

    def xmatmul(x_part: jnp.ndarray, g: jnp.ndarray) -> jnp.ndarray:
        return jnp.einsum("ij,...j->...i", x_part, g)

    use_plus = (xdot_factor > 0.0)  # (T,Z)

    out = jnp.zeros_like(f, dtype=jnp.float64)
    f_sxltz = f.astype(jnp.float64)

    l = jnp.arange(n_xi, dtype=jnp.float64)

    # Super-super: rows L get columns ell=L+2
    if n_xi >= 3:
        l0 = l[:-2]
        sup_stuff = (l0 + 1.0) * (l0 + 2.0) / ((2.0 * l0 + 5.0) * (2.0 * l0 + 3.0))  # (L-2,)
        g_sup = f_sxltz[:, :, 2:, :, :]  # (S,X,L-2,T,Z)
        g_sup_xlast = jnp.transpose(g_sup, (0, 2, 3, 4, 1))  # (S,L-2,T,Z,X)
        y_plus = xmatmul(x_part_plus, g_sup_xlast)
        y_minus = xmatmul(x_part_minus, g_sup_xlast)
        y_sup = jnp.where(use_plus[None, None, :, :, None], y_plus, y_minus)
        y_sup = jnp.transpose(y_sup, (0, 4, 1, 2, 3))  # (S,X,L-2,T,Z)
        y_sup = jnp.pad(y_sup, ((0, 0), (0, 0), (0, 2), (0, 0), (0, 0)))
        sup_coef = jnp.pad(sup_stuff, (0, 2))
        coef = sup_coef[:, None, None] * off_stuff[None, :, :]
        out = out + y_sup * coef[None, None, :, :, :]

    # Sub-sub: rows L get columns ell=L-2
    if n_xi >= 3:
        l2 = l[2:]
        sub_stuff = l2 * (l2 - 1.0) / ((2.0 * l2 - 3.0) * (2.0 * l2 - 1.0))
        g_sub = f_sxltz[:, :, :-2, :, :]
        g_sub_xlast = jnp.transpose(g_sub, (0, 2, 3, 4, 1))
        y_plus = xmatmul(x_part_plus, g_sub_xlast)
        y_minus = xmatmul(x_part_minus, g_sub_xlast)
        y_sub = jnp.where(use_plus[None, None, :, :, None], y_plus, y_minus)
        y_sub = jnp.transpose(y_sub, (0, 4, 1, 2, 3))
        y_sub = jnp.pad(y_sub, ((0, 0), (0, 0), (2, 0), (0, 0), (0, 0)))
        sub_coef = jnp.pad(sub_stuff, (2, 0))
        coef = sub_coef[:, None, None] * off_stuff[None, :, :]
        out = out + y_sub * coef[None, None, :, :, :]

    mask = _mask_xi(op.n_xi_for_x.astype(jnp.int32), n_xi).astype(out.dtype)
    return out * mask[None, :, :, None, None]


apply_er_xdot_v3_offdiag2_jit = jax.jit(apply_er_xdot_v3_offdiag2, static_argnums=())
