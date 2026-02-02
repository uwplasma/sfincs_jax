from __future__ import annotations

from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)

import jax.numpy as jnp

import numpy as np

from .geometry import BoozerGeometry
from .v3 import V3Grids


def vprime_hat(*, grids: V3Grids, geom: BoozerGeometry) -> jnp.ndarray:
    """Compute `VPrimeHat` as in v3 `geometry.F90:computeBIntegrals`.

    Returns
    -------
    vprime_hat:
      Scalar JAX array.
    """
    tw = jnp.asarray(grids.theta_weights, dtype=jnp.float64)  # (T,)
    zw = jnp.asarray(grids.zeta_weights, dtype=jnp.float64)  # (Z,)
    w = tw[:, None] * zw[None, :]  # (T,Z)
    return jnp.sum(w / geom.d_hat)


def fsab_hat2(*, grids: V3Grids, geom: BoozerGeometry) -> jnp.ndarray:
    """Compute `FSABHat2` as in v3 `geometry.F90:computeBIntegrals`.

    Returns
    -------
    fsab_hat2:
      Scalar JAX array.
    """
    tw = jnp.asarray(grids.theta_weights, dtype=jnp.float64)  # (T,)
    zw = jnp.asarray(grids.zeta_weights, dtype=jnp.float64)  # (Z,)
    w = tw[:, None] * zw[None, :]  # (T,Z)
    vph = vprime_hat(grids=grids, geom=geom)
    return jnp.sum(w * (geom.b_hat**2) / geom.d_hat) / vph


def u_hat(*, grids: V3Grids, geom: BoozerGeometry) -> jnp.ndarray:
    """Compute `uHat` as in v3 `geometry.F90` (cos-only branch).

    Notes
    -----
    In v3, `uHat` is computed from discrete harmonics of

      hHat = 1 / BHat^2

    via a projection onto modes `cos(m*theta - n*NPeriods*zeta)`. This implementation
    follows the **cosine-only** branch, which is the branch used for the `geometryScheme=4`
    fixture suite (stellarator-symmetric BHarmonics only).
    """
    theta = jnp.asarray(grids.theta, dtype=jnp.float64)
    zeta = jnp.asarray(grids.zeta, dtype=jnp.float64)
    b_hat = jnp.asarray(geom.b_hat, dtype=jnp.float64)
    ntheta = int(theta.shape[0])
    nzeta = int(zeta.shape[0])

    h_hat = 1.0 / (b_hat * b_hat)
    u = jnp.zeros_like(b_hat)

    m_max = int(ntheta // 2)
    n_max = int(nzeta // 2)

    theta_half = ntheta / 2.0
    zeta_half = nzeta / 2.0
    zeta_is_even = float(int(zeta_half)) == zeta_half

    for m in range(m_max + 1):
        if m == 0:
            startn = 1  # omit (0,0) mode
        elif float(m) == theta_half:
            startn = 0
        elif zeta_is_even:
            startn = -n_max + 1
        else:
            startn = -n_max
        stopn = n_max

        for n in range(startn, stopn + 1):
            angle = float(m) * theta[:, None] - float(n * geom.n_periods) * zeta[None, :]
            cos_angle = jnp.cos(angle)

            at_nyquist = (
                (m == 0 and float(n) == zeta_half)
                or (float(m) == theta_half and n == 0)
                or (float(m) == theta_half and float(n) == zeta_half)
            )
            weight = (1.0 if at_nyquist else 2.0) / float(ntheta * nzeta)

            h_amp = weight * jnp.sum(cos_angle * h_hat)

            denom = float(n * geom.n_periods) - float(geom.iota) * float(m)
            numer = float(geom.iota) * (float(geom.g_hat) * float(m) + float(geom.i_hat) * float(n * geom.n_periods))
            u_amp = jnp.where(jnp.abs(denom) < 1e-30, 0.0, (numer / denom) * h_amp)

            u = u + u_amp * cos_angle

    return u


def u_hat_np(*, grids: V3Grids, geom: BoozerGeometry) -> np.ndarray:
    """NumPy reference for `uHat` matching v3's transcendental evaluation closely.

    This function is intended for file-level parity against a frozen Fortran output.
    It is not differentiable.
    """
    theta = np.asarray(grids.theta, dtype=np.float64)
    zeta = np.asarray(grids.zeta, dtype=np.float64)
    b_hat = np.asarray(geom.b_hat, dtype=np.float64)

    ntheta = int(theta.shape[0])
    nzeta = int(zeta.shape[0])
    h_hat = 1.0 / (b_hat * b_hat)

    u = np.zeros_like(b_hat)

    m_max = int(ntheta // 2)
    n_max = int(nzeta // 2)

    theta_half = ntheta / 2.0
    zeta_half = nzeta / 2.0
    zeta_is_even = float(int(zeta_half)) == zeta_half

    for m in range(m_max + 1):
        if m == 0:
            startn = 1
        elif float(m) == theta_half:
            startn = 0
        elif zeta_is_even:
            startn = -n_max + 1
        else:
            startn = -n_max
        stopn = n_max

        for n in range(startn, stopn + 1):
            angle = float(m) * theta[:, None] - float(n * geom.n_periods) * zeta[None, :]
            cos_angle = np.cos(angle)

            at_nyquist = (
                (m == 0 and float(n) == zeta_half)
                or (float(m) == theta_half and n == 0)
                or (float(m) == theta_half and float(n) == zeta_half)
            )
            weight = (1.0 if at_nyquist else 2.0) / float(ntheta * nzeta)

            h_amp = weight * float(np.sum(cos_angle * h_hat))

            denom = float(n * geom.n_periods) - float(geom.iota) * float(m)
            numer = float(geom.iota) * (float(geom.g_hat) * float(m) + float(geom.i_hat) * float(n * geom.n_periods))
            u_amp = 0.0 if abs(denom) < 1e-30 else (numer / denom) * h_amp

            u = u + u_amp * cos_angle

    return u
