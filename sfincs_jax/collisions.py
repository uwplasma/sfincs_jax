from __future__ import annotations

from dataclasses import dataclass

from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
from jax import tree_util as jtu
from jax.scipy.special import erf


def _psi_chandra(x: jnp.ndarray) -> jnp.ndarray:
    """Chandrasekhar function Ψ(x).

    Matches the definition used in SFINCS v3 Fortran (`populateMatrix.F90`):

      Ψ = (erf(x) - (2/sqrt(pi)) x exp(-x^2)) / (2 x^2)
    """
    x = x.astype(jnp.float64)
    sqrt_pi = jnp.sqrt(jnp.pi)
    num = erf(x) - (2.0 / sqrt_pi) * x * jnp.exp(-(x * x))
    den = 2.0 * x * x
    # Avoid NaNs at x=0 (not typically used with v3 default x grids, but keep robust).
    eps = jnp.asarray(1e-14, dtype=jnp.float64)
    small = jnp.abs(x) < eps
    # Series: Ψ(x) ~ (2/sqrt(pi)) * x/3 + O(x^3)
    series = (2.0 / sqrt_pi) * x / 3.0
    return jnp.where(small, series, num / den)


def nu_d_hat_pitch_angle_scattering_v3(
    *,
    x: jnp.ndarray,  # (Nx,)
    z_s: jnp.ndarray,  # (S,)
    m_hats: jnp.ndarray,  # (S,)
    n_hats: jnp.ndarray,  # (S,)
    t_hats: jnp.ndarray,  # (S,)
) -> jnp.ndarray:
    """Compute the v3 pitch-angle-scattering deflection frequency `nuDHat`.

    This function matches the "WITHOUT PHI1" branch in `populateMatrix.F90` for
    `collisionOperator = 1`.
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    z_s = jnp.asarray(z_s, dtype=jnp.float64)
    m_hats = jnp.asarray(m_hats, dtype=jnp.float64)
    n_hats = jnp.asarray(n_hats, dtype=jnp.float64)
    t_hats = jnp.asarray(t_hats, dtype=jnp.float64)

    z2 = z_s * z_s  # (S,)
    # T32m = THat * sqrt(THat * mHat) in the Fortran code:
    t32m = t_hats * jnp.sqrt(t_hats * m_hats)  # (S,)

    # speciesFactor(A,B) = sqrt(THat_A*mHat_B / (THat_B*mHat_A))
    species_factor = jnp.sqrt(
        (t_hats[:, None] * m_hats[None, :]) / (t_hats[None, :] * m_hats[:, None])
    )  # (S,S)

    xb = x[None, None, :] * species_factor[:, :, None]  # (S,S,X)
    psi = _psi_chandra(xb)
    term = (erf(xb) - psi)  # (S,S,X)

    # Divide by x^3 (note: Fortran uses the base x-grid, not xb):
    x3 = x * x * x  # (X,)
    # Avoid div-by-0 if a point at x=0 is used:
    x3 = jnp.where(x3 == 0, jnp.asarray(jnp.inf, dtype=jnp.float64), x3)
    term = term / x3[None, None, :]  # (S,S,X)

    prefac = (3.0 * jnp.sqrt(jnp.pi) / 4.0) / t32m  # (S,)
    sum_b = jnp.sum((z2[None, :, None] * n_hats[None, :, None]) * term, axis=1)  # (S,X)
    return prefac[:, None] * z2[:, None] * sum_b


@jtu.register_pytree_node_class
@dataclass(frozen=True)
class PitchAngleScatteringV3Operator:
    """Pure pitch-angle scattering collision operator in the v3 Legendre basis.

    Notes
    -----
    - This is `collisionOperator = 1` without Phi1.
    - The operator is diagonal in (theta, zeta) and in Legendre index L.
    """

    nu_n: jnp.ndarray  # scalar
    krook: jnp.ndarray  # scalar
    nu_d_hat: jnp.ndarray  # (S, X)
    n_xi_for_x: jnp.ndarray  # (X,) int32

    def tree_flatten(self):
        children = (self.nu_n, self.krook, self.nu_d_hat, self.n_xi_for_x)
        aux = None
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        del aux
        nu_n, krook, nu_d_hat, n_xi_for_x = children
        return cls(nu_n=nu_n, krook=krook, nu_d_hat=nu_d_hat, n_xi_for_x=n_xi_for_x)


def make_pitch_angle_scattering_v3_operator(
    *,
    x: jnp.ndarray,
    z_s: jnp.ndarray,
    m_hats: jnp.ndarray,
    n_hats: jnp.ndarray,
    t_hats: jnp.ndarray,
    nu_n: float,
    krook: float = 0.0,
    n_xi_for_x: jnp.ndarray,
) -> PitchAngleScatteringV3Operator:
    nu_d_hat = nu_d_hat_pitch_angle_scattering_v3(
        x=x, z_s=z_s, m_hats=m_hats, n_hats=n_hats, t_hats=t_hats
    )
    return PitchAngleScatteringV3Operator(
        nu_n=jnp.asarray(nu_n, dtype=jnp.float64),
        krook=jnp.asarray(krook, dtype=jnp.float64),
        nu_d_hat=nu_d_hat,
        n_xi_for_x=jnp.asarray(n_xi_for_x, dtype=jnp.int32),
    )


def _mask_xi(n_xi_for_x: jnp.ndarray, n_xi_max: int) -> jnp.ndarray:
    l = jnp.arange(n_xi_max, dtype=jnp.int32)[None, :]
    return l < n_xi_for_x[:, None]


def apply_pitch_angle_scattering_v3(op: PitchAngleScatteringV3Operator, f: jnp.ndarray) -> jnp.ndarray:
    """Apply v3 pitch-angle scattering collisions to `f`.

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
    n_species, n_x, n_xi, _, _ = f.shape
    if op.nu_d_hat.shape != (n_species, n_x):
        raise ValueError(
            f"op.nu_d_hat has shape {op.nu_d_hat.shape}, expected {(n_species, n_x)}"
        )

    l = jnp.arange(n_xi, dtype=jnp.float64)  # row L
    factor_l = 0.5 * (l * (l + 1.0) + 2.0 * op.krook)  # (L,)
    coef = op.nu_n * op.nu_d_hat[:, :, None] * factor_l[None, None, :]  # (S,X,L)

    out = coef[:, :, :, None, None] * f

    mask = _mask_xi(op.n_xi_for_x.astype(jnp.int32), n_xi).astype(out.dtype)  # (X,L)
    return out * mask[None, :, :, None, None]


apply_pitch_angle_scattering_v3_jit = jax.jit(apply_pitch_angle_scattering_v3, static_argnums=())

