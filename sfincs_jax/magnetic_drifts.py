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


def _offdiag2_coupling_plus(n_xi: int) -> jnp.ndarray:
    l = jnp.arange(n_xi, dtype=jnp.float64)
    return (l + 2.0) * (l + 1.0) / ((2.0 * l + 5.0) * (2.0 * l + 3.0))


def _offdiag2_coupling_minus(n_xi: int) -> jnp.ndarray:
    l = jnp.arange(n_xi, dtype=jnp.float64)
    return jnp.where(l > 1, (l - 1.0) * l / ((2.0 * l - 3.0) * (2.0 * l - 1.0)), 0.0)


def _xidot_offdiag2_coupling_plus(n_xi: int) -> jnp.ndarray:
    l = jnp.arange(n_xi, dtype=jnp.float64)
    return (l + 3.0) * (l + 2.0) * (l + 1.0) / ((2.0 * l + 5.0) * (2.0 * l + 3.0))


def _xidot_offdiag2_coupling_minus(n_xi: int) -> jnp.ndarray:
    l = jnp.arange(n_xi, dtype=jnp.float64)
    return jnp.where(l > 1, -l * (l - 1.0) * (l - 2.0) / ((2.0 * l - 3.0) * (2.0 * l - 1.0)), 0.0)


def _diag_l_coupling(n_xi: int) -> jnp.ndarray:
    l = jnp.arange(n_xi, dtype=jnp.float64)
    return jnp.where(l > 0, (l + 1.0) * l / ((2.0 * l - 1.0) * (2.0 * l + 3.0)), 0.0)


def _magdrift_diag_coeffs(n_xi: int) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return the diagonal-in-L coefficients multiplying (gf1, gf2, gf3).

    Matches v3 `populateMatrix.F90` for both the d/dtheta and d/dzeta magnetic-drift terms:

    stuffToAdd = factor * (c1(L)*gf1 + c2(L)*gf2 + c3(L)*gf3)
    """
    l = jnp.arange(n_xi, dtype=jnp.float64)
    denom = (2.0 * l + 3.0) * (2.0 * l - 1.0)
    c1 = 2.0 * (3.0 * l * l + 3.0 * l - 2.0) / denom
    c2 = (2.0 * l * l + 2.0 * l - 1.0) / denom
    c3 = (-2.0) * l * (l + 1.0) / denom
    return c1, c2, c3


@jtu.register_pytree_node_class
@dataclass(frozen=True)
class MagneticDriftThetaV3Operator:
    """Magnetic-drift `d/dtheta` term (v3, scheme=1/2/7/8/9 form).

    This operator currently implements only the `magneticDriftScheme=1` geometric factors
    and assumes `force0RadialCurrentInEquilibrium = .true.` (so no extra shear/current term).
    """

    # Scalars:
    delta: jnp.ndarray  # scalar
    t_hat: jnp.ndarray  # scalar
    z: jnp.ndarray  # scalar (charge number for this species)

    # Grids:
    x: jnp.ndarray  # (Nx,)
    ddtheta_plus: jnp.ndarray  # (Ntheta, Ntheta)
    ddtheta_minus: jnp.ndarray  # (Ntheta, Ntheta)

    # Geometry:
    d_hat: jnp.ndarray  # (Ntheta, Nzeta)
    b_hat: jnp.ndarray  # (Ntheta, Nzeta)
    b_hat_sub_zeta: jnp.ndarray  # (Ntheta, Nzeta) (constant in Boozer)
    b_hat_sub_psi: jnp.ndarray  # (Ntheta, Nzeta)
    db_hat_dzeta: jnp.ndarray  # (Ntheta, Nzeta)
    db_hat_dpsi_hat: jnp.ndarray  # (Ntheta, Nzeta)
    db_hat_sub_psi_dzeta: jnp.ndarray  # (Ntheta, Nzeta)
    db_hat_sub_zeta_dpsi_hat: jnp.ndarray  # (Ntheta, Nzeta)

    n_xi_for_x: jnp.ndarray  # (Nx,) int32

    def tree_flatten(self):
        children = (
            self.delta,
            self.t_hat,
            self.z,
            self.x,
            self.ddtheta_plus,
            self.ddtheta_minus,
            self.d_hat,
            self.b_hat,
            self.b_hat_sub_zeta,
            self.b_hat_sub_psi,
            self.db_hat_dzeta,
            self.db_hat_dpsi_hat,
            self.db_hat_sub_psi_dzeta,
            self.db_hat_sub_zeta_dpsi_hat,
            self.n_xi_for_x,
        )
        aux = None
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        del aux
        (
            delta,
            t_hat,
            z,
            x,
            ddtheta_plus,
            ddtheta_minus,
            d_hat,
            b_hat,
            b_hat_sub_zeta,
            b_hat_sub_psi,
            db_hat_dzeta,
            db_hat_dpsi_hat,
            db_hat_sub_psi_dzeta,
            db_hat_sub_zeta_dpsi_hat,
            n_xi_for_x,
        ) = children
        return cls(
            delta=delta,
            t_hat=t_hat,
            z=z,
            x=x,
            ddtheta_plus=ddtheta_plus,
            ddtheta_minus=ddtheta_minus,
            d_hat=d_hat,
            b_hat=b_hat,
            b_hat_sub_zeta=b_hat_sub_zeta,
            b_hat_sub_psi=b_hat_sub_psi,
            db_hat_dzeta=db_hat_dzeta,
            db_hat_dpsi_hat=db_hat_dpsi_hat,
            db_hat_sub_psi_dzeta=db_hat_sub_psi_dzeta,
            db_hat_sub_zeta_dpsi_hat=db_hat_sub_zeta_dpsi_hat,
            n_xi_for_x=n_xi_for_x,
        )


@jtu.register_pytree_node_class
@dataclass(frozen=True)
class MagneticDriftZetaV3Operator:
    """Magnetic-drift `d/dzeta` term (v3, scheme=1/2/7/9 form)."""

    delta: jnp.ndarray  # scalar
    t_hat: jnp.ndarray  # scalar
    z: jnp.ndarray  # scalar

    x: jnp.ndarray  # (Nx,)
    ddzeta_plus: jnp.ndarray  # (Nzeta, Nzeta)
    ddzeta_minus: jnp.ndarray  # (Nzeta, Nzeta)

    d_hat: jnp.ndarray  # (Ntheta, Nzeta)
    b_hat: jnp.ndarray  # (Ntheta, Nzeta)
    b_hat_sub_theta: jnp.ndarray  # (Ntheta, Nzeta) (constant in Boozer)
    b_hat_sub_psi: jnp.ndarray  # (Ntheta, Nzeta)
    db_hat_dtheta: jnp.ndarray  # (Ntheta, Nzeta)
    db_hat_dpsi_hat: jnp.ndarray  # (Ntheta, Nzeta)
    db_hat_sub_theta_dpsi_hat: jnp.ndarray  # (Ntheta, Nzeta)
    db_hat_sub_psi_dtheta: jnp.ndarray  # (Ntheta, Nzeta)

    n_xi_for_x: jnp.ndarray  # (Nx,) int32

    def tree_flatten(self):
        children = (
            self.delta,
            self.t_hat,
            self.z,
            self.x,
            self.ddzeta_plus,
            self.ddzeta_minus,
            self.d_hat,
            self.b_hat,
            self.b_hat_sub_theta,
            self.b_hat_sub_psi,
            self.db_hat_dtheta,
            self.db_hat_dpsi_hat,
            self.db_hat_sub_theta_dpsi_hat,
            self.db_hat_sub_psi_dtheta,
            self.n_xi_for_x,
        )
        aux = None
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        del aux
        (
            delta,
            t_hat,
            z,
            x,
            ddzeta_plus,
            ddzeta_minus,
            d_hat,
            b_hat,
            b_hat_sub_theta,
            b_hat_sub_psi,
            db_hat_dtheta,
            db_hat_dpsi_hat,
            db_hat_sub_theta_dpsi_hat,
            db_hat_sub_psi_dtheta,
            n_xi_for_x,
        ) = children
        return cls(
            delta=delta,
            t_hat=t_hat,
            z=z,
            x=x,
            ddzeta_plus=ddzeta_plus,
            ddzeta_minus=ddzeta_minus,
            d_hat=d_hat,
            b_hat=b_hat,
            b_hat_sub_theta=b_hat_sub_theta,
            b_hat_sub_psi=b_hat_sub_psi,
            db_hat_dtheta=db_hat_dtheta,
            db_hat_dpsi_hat=db_hat_dpsi_hat,
            db_hat_sub_theta_dpsi_hat=db_hat_sub_theta_dpsi_hat,
            db_hat_sub_psi_dtheta=db_hat_sub_psi_dtheta,
            n_xi_for_x=n_xi_for_x,
        )


@jtu.register_pytree_node_class
@dataclass(frozen=True)
class MagneticDriftXiDotV3Operator:
    """Non-standard magnetic-drift `d/dxi` term (v3, scheme=1/2/8/9 form)."""

    delta: jnp.ndarray  # scalar
    t_hat: jnp.ndarray  # scalar
    z: jnp.ndarray  # scalar

    x: jnp.ndarray  # (Nx,)

    d_hat: jnp.ndarray  # (Ntheta, Nzeta)
    b_hat: jnp.ndarray  # (Ntheta, Nzeta)
    db_hat_dtheta: jnp.ndarray  # (Ntheta, Nzeta)
    db_hat_dzeta: jnp.ndarray  # (Ntheta, Nzeta)

    db_hat_sub_psi_dzeta: jnp.ndarray  # (Ntheta, Nzeta)
    db_hat_sub_zeta_dpsi_hat: jnp.ndarray  # (Ntheta, Nzeta)
    db_hat_sub_theta_dpsi_hat: jnp.ndarray  # (Ntheta, Nzeta)
    db_hat_sub_psi_dtheta: jnp.ndarray  # (Ntheta, Nzeta)

    n_xi_for_x: jnp.ndarray  # (Nx,) int32

    def tree_flatten(self):
        children = (
            self.delta,
            self.t_hat,
            self.z,
            self.x,
            self.d_hat,
            self.b_hat,
            self.db_hat_dtheta,
            self.db_hat_dzeta,
            self.db_hat_sub_psi_dzeta,
            self.db_hat_sub_zeta_dpsi_hat,
            self.db_hat_sub_theta_dpsi_hat,
            self.db_hat_sub_psi_dtheta,
            self.n_xi_for_x,
        )
        aux = None
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        del aux
        (
            delta,
            t_hat,
            z,
            x,
            d_hat,
            b_hat,
            db_hat_dtheta,
            db_hat_dzeta,
            db_hat_sub_psi_dzeta,
            db_hat_sub_zeta_dpsi_hat,
            db_hat_sub_theta_dpsi_hat,
            db_hat_sub_psi_dtheta,
            n_xi_for_x,
        ) = children
        return cls(
            delta=delta,
            t_hat=t_hat,
            z=z,
            x=x,
            d_hat=d_hat,
            b_hat=b_hat,
            db_hat_dtheta=db_hat_dtheta,
            db_hat_dzeta=db_hat_dzeta,
            db_hat_sub_psi_dzeta=db_hat_sub_psi_dzeta,
            db_hat_sub_zeta_dpsi_hat=db_hat_sub_zeta_dpsi_hat,
            db_hat_sub_theta_dpsi_hat=db_hat_sub_theta_dpsi_hat,
            db_hat_sub_psi_dtheta=db_hat_sub_psi_dtheta,
            n_xi_for_x=n_xi_for_x,
        )


def apply_magnetic_drift_theta_v3_offdiag2(op: MagneticDriftThetaV3Operator, f: jnp.ndarray) -> jnp.ndarray:
    """Apply only the :math:`\\Delta L = \\pm 2` part of the magnetic-drift d/dtheta term."""
    if f.ndim != 5:
        raise ValueError("f must have shape (Nspecies, Nx, Nxi, Ntheta, Nzeta)")
    _n_species, n_x, n_xi, n_theta, n_zeta = f.shape
    if n_theta != op.ddtheta_plus.shape[0]:
        raise ValueError("f theta axis does not match ddtheta_plus/minus")
    if n_x != op.x.shape[0]:
        raise ValueError("f x axis does not match x")

    # Geometric factors for magneticDriftScheme=1.
    gf1 = (op.b_hat_sub_zeta * op.db_hat_dpsi_hat - op.b_hat_sub_psi * op.db_hat_dzeta).astype(jnp.float64)
    gf2 = (2.0 * op.b_hat * (op.db_hat_sub_psi_dzeta - op.db_hat_sub_zeta_dpsi_hat)).astype(jnp.float64)
    gf12 = gf1 + gf2

    base = (op.delta * op.t_hat * op.d_hat / (2.0 * op.z * (op.b_hat**3))).astype(jnp.float64)  # (T,Z)
    x2 = (op.x.astype(jnp.float64) ** 2)  # (X,)

    # d/dtheta applied to f:
    dtheta_plus = jnp.einsum("ij,sxljz->sxliz", op.ddtheta_plus.astype(jnp.float64), f.astype(jnp.float64))
    dtheta_minus = jnp.einsum("ij,sxljz->sxliz", op.ddtheta_minus.astype(jnp.float64), f.astype(jnp.float64))

    # Upwind selection matches v3 `magneticDriftDerivativeScheme != 0`:
    # ddthetaToUse depends on sign(geometricFactor1 * DHat(1,1) / Z).
    dhat11 = op.d_hat[0, 0].astype(jnp.float64)
    use_plus = (gf1 * dhat11 / op.z.astype(jnp.float64)) > 0  # (T,Z) bool
    dtheta_f = jnp.where(use_plus[None, None, None, :, :], dtheta_plus, dtheta_minus)

    c_plus = _offdiag2_coupling_plus(n_xi)  # (L,)
    c_minus = _offdiag2_coupling_minus(n_xi)  # (L,)

    term_plus = c_plus[None, None, :-2, None, None] * dtheta_f[:, :, 2:, :, :]
    term_plus = jnp.pad(term_plus, ((0, 0), (0, 0), (0, 2), (0, 0), (0, 0)))

    term_minus = c_minus[None, None, 2:, None, None] * dtheta_f[:, :, :-2, :, :]
    term_minus = jnp.pad(term_minus, ((0, 0), (0, 0), (2, 0), (0, 0), (0, 0)))

    out = (
        x2[None, :, None, None, None]
        * base[None, None, None, :, :]
        * gf12[None, None, None, :, :]
        * (term_plus + term_minus)
    )

    mask = _mask_xi(op.n_xi_for_x.astype(jnp.int32), n_xi).astype(out.dtype)
    return out * mask[None, :, :, None, None]


def apply_magnetic_drift_theta_v3(op: MagneticDriftThetaV3Operator, f: jnp.ndarray) -> jnp.ndarray:
    """Apply the full magnetic-drift d/dtheta term for the currently supported physics.

    Notes
    -----
    - Implements the v3 `magneticDriftScheme=1` geometric factors.
    - Assumes `magneticDriftDerivativeScheme != 0` and uses ddtheta_plus/minus with the
      v3 upwind selector based on `sign(geometricFactor1 * DHat(1,1) / Z)`.
    - Assumes `magneticDriftScheme != 2` so `geometricFactor3 = 0`.
    """
    if f.ndim != 5:
        raise ValueError("f must have shape (Nspecies, Nx, Nxi, Ntheta, Nzeta)")
    _n_species, n_x, n_xi, n_theta, _n_zeta = f.shape
    if n_theta != op.ddtheta_plus.shape[0]:
        raise ValueError("f theta axis does not match ddtheta_plus/minus")
    if n_x != op.x.shape[0]:
        raise ValueError("f x axis does not match x")

    gf1 = (op.b_hat_sub_zeta * op.db_hat_dpsi_hat - op.b_hat_sub_psi * op.db_hat_dzeta).astype(jnp.float64)
    gf2 = (2.0 * op.b_hat * (op.db_hat_sub_psi_dzeta - op.db_hat_sub_zeta_dpsi_hat)).astype(jnp.float64)
    gf3 = jnp.zeros_like(gf1)
    gf12 = gf1 + gf2

    base = (op.delta * op.t_hat * op.d_hat / (2.0 * op.z * (op.b_hat**3))).astype(jnp.float64)  # (T,Z)
    x2 = (op.x.astype(jnp.float64) ** 2)  # (X,)

    dtheta_plus = jnp.einsum("ij,sxljz->sxliz", op.ddtheta_plus.astype(jnp.float64), f.astype(jnp.float64))
    dtheta_minus = jnp.einsum("ij,sxljz->sxliz", op.ddtheta_minus.astype(jnp.float64), f.astype(jnp.float64))

    dhat11 = op.d_hat[0, 0].astype(jnp.float64)
    use_plus = (gf1 * dhat11 / op.z.astype(jnp.float64)) > 0
    dtheta_f = jnp.where(use_plus[None, None, None, :, :], dtheta_plus, dtheta_minus)

    c1, c2, c3 = _magdrift_diag_coeffs(n_xi)  # (L,), (L,), (L,)
    diag_l = (
        c1[:, None, None] * gf1[None, :, :] + c2[:, None, None] * gf2[None, :, :] + c3[:, None, None] * gf3[None, :, :]
    ).astype(jnp.float64)  # (L,T,Z)

    diag_part = diag_l[None, None, :, :, :] * dtheta_f

    c_plus = _offdiag2_coupling_plus(n_xi)
    c_minus = _offdiag2_coupling_minus(n_xi)

    term_plus = c_plus[None, None, :-2, None, None] * dtheta_f[:, :, 2:, :, :]
    term_plus = jnp.pad(term_plus, ((0, 0), (0, 0), (0, 2), (0, 0), (0, 0)))

    term_minus = c_minus[None, None, 2:, None, None] * dtheta_f[:, :, :-2, :, :]
    term_minus = jnp.pad(term_minus, ((0, 0), (0, 0), (2, 0), (0, 0), (0, 0)))

    offdiag2_part = gf12[None, None, None, :, :] * (term_plus + term_minus)

    out = x2[None, :, None, None, None] * base[None, None, None, :, :] * (diag_part + offdiag2_part)
    mask = _mask_xi(op.n_xi_for_x.astype(jnp.int32), n_xi).astype(out.dtype)
    return out * mask[None, :, :, None, None]


def apply_magnetic_drift_zeta_v3_offdiag2(op: MagneticDriftZetaV3Operator, f: jnp.ndarray) -> jnp.ndarray:
    """Apply only the :math:`\\Delta L = \\pm 2` part of the magnetic-drift d/dzeta term."""
    if f.ndim != 5:
        raise ValueError("f must have shape (Nspecies, Nx, Nxi, Ntheta, Nzeta)")
    _n_species, n_x, n_xi, n_theta, n_zeta = f.shape
    if n_zeta != op.ddzeta_plus.shape[0]:
        raise ValueError("f zeta axis does not match ddzeta_plus/minus")
    if n_x != op.x.shape[0]:
        raise ValueError("f x axis does not match x")

    # Geometric factors for magneticDriftScheme=1 (same as case (1,2,7,9) in Fortran).
    gf1 = (op.b_hat_sub_psi * op.db_hat_dtheta - op.b_hat_sub_theta * op.db_hat_dpsi_hat).astype(jnp.float64)
    gf2 = (2.0 * op.b_hat * (op.db_hat_sub_theta_dpsi_hat - op.db_hat_sub_psi_dtheta)).astype(jnp.float64)
    gf12 = gf1 + gf2

    base = (op.delta * op.t_hat * op.d_hat / (2.0 * op.z * (op.b_hat**3))).astype(jnp.float64)  # (T,Z)
    x2 = (op.x.astype(jnp.float64) ** 2)  # (X,)

    dzeta_plus = jnp.einsum("ij,sxltj->sxlti", op.ddzeta_plus.astype(jnp.float64), f.astype(jnp.float64))
    dzeta_minus = jnp.einsum("ij,sxltj->sxlti", op.ddzeta_minus.astype(jnp.float64), f.astype(jnp.float64))

    dhat11 = op.d_hat[0, 0].astype(jnp.float64)
    use_plus = (gf1 * dhat11 / op.z.astype(jnp.float64)) > 0  # (T,Z) bool
    dzeta_f = jnp.where(use_plus[None, None, None, :, :], dzeta_plus, dzeta_minus)

    c_plus = _offdiag2_coupling_plus(n_xi)
    c_minus = _offdiag2_coupling_minus(n_xi)

    term_plus = c_plus[None, None, :-2, None, None] * dzeta_f[:, :, 2:, :, :]
    term_plus = jnp.pad(term_plus, ((0, 0), (0, 0), (0, 2), (0, 0), (0, 0)))

    term_minus = c_minus[None, None, 2:, None, None] * dzeta_f[:, :, :-2, :, :]
    term_minus = jnp.pad(term_minus, ((0, 0), (0, 0), (2, 0), (0, 0), (0, 0)))

    out = (
        x2[None, :, None, None, None]
        * base[None, None, None, :, :]
        * gf12[None, None, None, :, :]
        * (term_plus + term_minus)
    )

    mask = _mask_xi(op.n_xi_for_x.astype(jnp.int32), n_xi).astype(out.dtype)
    return out * mask[None, :, :, None, None]


def apply_magnetic_drift_zeta_v3(op: MagneticDriftZetaV3Operator, f: jnp.ndarray) -> jnp.ndarray:
    """Apply the full magnetic-drift d/dzeta term for the currently supported physics.

    Notes
    -----
    This uses the same (gf1, gf2) definitions as `apply_magnetic_drift_zeta_v3_offdiag2`,
    which are parity-tested against a v3 PETSc matrix slice. As for the theta term, we
    assume `geometricFactor3 = 0`.
    """
    if f.ndim != 5:
        raise ValueError("f must have shape (Nspecies, Nx, Nxi, Ntheta, Nzeta)")
    _n_species, n_x, n_xi, _n_theta, n_zeta = f.shape
    if n_zeta != op.ddzeta_plus.shape[0]:
        raise ValueError("f zeta axis does not match ddzeta_plus/minus")
    if n_x != op.x.shape[0]:
        raise ValueError("f x axis does not match x")

    gf1 = (op.b_hat_sub_psi * op.db_hat_dtheta - op.b_hat_sub_theta * op.db_hat_dpsi_hat).astype(jnp.float64)
    gf2 = (2.0 * op.b_hat * (op.db_hat_sub_theta_dpsi_hat - op.db_hat_sub_psi_dtheta)).astype(jnp.float64)
    gf3 = jnp.zeros_like(gf1)
    gf12 = gf1 + gf2

    base = (op.delta * op.t_hat * op.d_hat / (2.0 * op.z * (op.b_hat**3))).astype(jnp.float64)  # (T,Z)
    x2 = (op.x.astype(jnp.float64) ** 2)  # (X,)

    dzeta_plus = jnp.einsum("ij,sxltj->sxlti", op.ddzeta_plus.astype(jnp.float64), f.astype(jnp.float64))
    dzeta_minus = jnp.einsum("ij,sxltj->sxlti", op.ddzeta_minus.astype(jnp.float64), f.astype(jnp.float64))

    dhat11 = op.d_hat[0, 0].astype(jnp.float64)
    use_plus = (gf1 * dhat11 / op.z.astype(jnp.float64)) > 0
    dzeta_f = jnp.where(use_plus[None, None, None, :, :], dzeta_plus, dzeta_minus)

    c1, c2, c3 = _magdrift_diag_coeffs(n_xi)
    diag_l = (
        c1[:, None, None] * gf1[None, :, :] + c2[:, None, None] * gf2[None, :, :] + c3[:, None, None] * gf3[None, :, :]
    ).astype(jnp.float64)  # (L,T,Z)
    diag_part = diag_l[None, None, :, :, :] * dzeta_f

    c_plus = _offdiag2_coupling_plus(n_xi)
    c_minus = _offdiag2_coupling_minus(n_xi)

    term_plus = c_plus[None, None, :-2, None, None] * dzeta_f[:, :, 2:, :, :]
    term_plus = jnp.pad(term_plus, ((0, 0), (0, 0), (0, 2), (0, 0), (0, 0)))

    term_minus = c_minus[None, None, 2:, None, None] * dzeta_f[:, :, :-2, :, :]
    term_minus = jnp.pad(term_minus, ((0, 0), (0, 0), (2, 0), (0, 0), (0, 0)))

    offdiag2_part = gf12[None, None, None, :, :] * (term_plus + term_minus)

    out = x2[None, :, None, None, None] * base[None, None, None, :, :] * (diag_part + offdiag2_part)
    mask = _mask_xi(op.n_xi_for_x.astype(jnp.int32), n_xi).astype(out.dtype)
    return out * mask[None, :, :, None, None]


def apply_magnetic_drift_xidot_v3_offdiag2(op: MagneticDriftXiDotV3Operator, f: jnp.ndarray) -> jnp.ndarray:
    """Apply only the :math:`\\Delta L = \\pm 2` part of the non-standard magnetic-drift d/dxi term."""
    if f.ndim != 5:
        raise ValueError("f must have shape (Nspecies, Nx, Nxi, Ntheta, Nzeta)")
    _n_species, n_x, n_xi, _n_theta, _n_zeta = f.shape
    if n_x != op.x.shape[0]:
        raise ValueError("f x axis does not match x")

    temp = (
        (op.db_hat_sub_psi_dzeta - op.db_hat_sub_zeta_dpsi_hat) * op.db_hat_dtheta
        + (op.db_hat_sub_theta_dpsi_hat - op.db_hat_sub_psi_dtheta) * op.db_hat_dzeta
    ).astype(jnp.float64)

    factor = (-(op.delta * op.t_hat) * op.d_hat / (2.0 * op.z * (op.b_hat**3)) * temp).astype(jnp.float64)  # (T,Z)
    x2 = (op.x.astype(jnp.float64) ** 2)  # (X,)

    c_plus = _xidot_offdiag2_coupling_plus(n_xi)  # (L,)
    c_minus = _xidot_offdiag2_coupling_minus(n_xi)  # (L,)

    term_plus = c_plus[None, None, :-2, None, None] * f[:, :, 2:, :, :].astype(jnp.float64)
    term_plus = jnp.pad(term_plus, ((0, 0), (0, 0), (0, 2), (0, 0), (0, 0)))

    term_minus = c_minus[None, None, 2:, None, None] * f[:, :, :-2, :, :].astype(jnp.float64)
    term_minus = jnp.pad(term_minus, ((0, 0), (0, 0), (2, 0), (0, 0), (0, 0)))

    out = x2[None, :, None, None, None] * factor[None, None, None, :, :] * (term_plus + term_minus)
    mask = _mask_xi(op.n_xi_for_x.astype(jnp.int32), n_xi).astype(out.dtype)
    return out * mask[None, :, :, None, None]


def apply_magnetic_drift_xidot_v3(op: MagneticDriftXiDotV3Operator, f: jnp.ndarray) -> jnp.ndarray:
    """Apply the full non-standard magnetic-drift d/dxi term (diag-in-L + :math:`\\Delta L = \\pm 2`)."""
    if f.ndim != 5:
        raise ValueError("f must have shape (Nspecies, Nx, Nxi, Ntheta, Nzeta)")
    _n_species, n_x, n_xi, _n_theta, _n_zeta = f.shape
    if n_x != op.x.shape[0]:
        raise ValueError("f x axis does not match x")

    temp = (
        (op.db_hat_sub_psi_dzeta - op.db_hat_sub_zeta_dpsi_hat) * op.db_hat_dtheta
        + (op.db_hat_sub_theta_dpsi_hat - op.db_hat_sub_psi_dtheta) * op.db_hat_dzeta
    ).astype(jnp.float64)

    factor = (-(op.delta * op.t_hat) * op.d_hat / (2.0 * op.z * (op.b_hat**3)) * temp).astype(jnp.float64)  # (T,Z)
    x2 = (op.x.astype(jnp.float64) ** 2)  # (X,)

    diag_c = _diag_l_coupling(n_xi)  # (L,)
    diag_part = diag_c[None, None, :, None, None] * f.astype(jnp.float64)

    c_plus = _xidot_offdiag2_coupling_plus(n_xi)
    c_minus = _xidot_offdiag2_coupling_minus(n_xi)

    term_plus = c_plus[None, None, :-2, None, None] * f[:, :, 2:, :, :].astype(jnp.float64)
    term_plus = jnp.pad(term_plus, ((0, 0), (0, 0), (0, 2), (0, 0), (0, 0)))

    term_minus = c_minus[None, None, 2:, None, None] * f[:, :, :-2, :, :].astype(jnp.float64)
    term_minus = jnp.pad(term_minus, ((0, 0), (0, 0), (2, 0), (0, 0), (0, 0)))

    out = x2[None, :, None, None, None] * factor[None, None, None, :, :] * (diag_part + term_plus + term_minus)
    mask = _mask_xi(op.n_xi_for_x.astype(jnp.int32), n_xi).astype(out.dtype)
    return out * mask[None, :, :, None, None]


apply_magnetic_drift_theta_v3_offdiag2_jit = jax.jit(apply_magnetic_drift_theta_v3_offdiag2, static_argnums=())
apply_magnetic_drift_zeta_v3_offdiag2_jit = jax.jit(apply_magnetic_drift_zeta_v3_offdiag2, static_argnums=())
apply_magnetic_drift_xidot_v3_offdiag2_jit = jax.jit(apply_magnetic_drift_xidot_v3_offdiag2, static_argnums=())

apply_magnetic_drift_theta_v3_jit = jax.jit(apply_magnetic_drift_theta_v3, static_argnums=())
apply_magnetic_drift_zeta_v3_jit = jax.jit(apply_magnetic_drift_zeta_v3, static_argnums=())
apply_magnetic_drift_xidot_v3_jit = jax.jit(apply_magnetic_drift_xidot_v3, static_argnums=())
