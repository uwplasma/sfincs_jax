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


def b0_over_bbar(*, grids: V3Grids, geom: BoozerGeometry) -> jnp.ndarray:
    """Compute `B0OverBBar` as in v3 `geometry.F90:computeBIntegrals` for non-Boozer coordinates.

    v3 uses:
      B0OverBBar = <B^3> / <B^2>

    where <> denotes the Jacobian-weighted flux-surface average.
    """
    tw = jnp.asarray(grids.theta_weights, dtype=jnp.float64)  # (T,)
    zw = jnp.asarray(grids.zeta_weights, dtype=jnp.float64)  # (Z,)
    w = tw[:, None] * zw[None, :]  # (T,Z)
    vph = vprime_hat(grids=grids, geom=geom)
    fsab2 = fsab_hat2(grids=grids, geom=geom)
    num = jnp.sum(w * (geom.b_hat**3) / geom.d_hat)
    return num / (vph * fsab2)


def g_hat_i_hat(*, grids: V3Grids, geom: BoozerGeometry) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute (GHat, IHat) from covariant components as in v3 `geometry.F90:computeBIntegrals`.

    For non-Boozer coordinates, v3 reports:
      GHat = <B_sub_zeta>, IHat = <B_sub_theta>

    where <> is the (theta,zeta) average over 4π² using trapezoid weights.
    """
    tw = jnp.asarray(grids.theta_weights, dtype=jnp.float64)  # (T,)
    zw = jnp.asarray(grids.zeta_weights, dtype=jnp.float64)  # (Z,)
    w = tw[:, None] * zw[None, :]  # (T,Z)
    denom = jnp.asarray(4.0 * jnp.pi * jnp.pi, dtype=jnp.float64)
    g_hat = jnp.sum(w * geom.b_hat_sub_zeta) / denom
    i_hat = jnp.sum(w * geom.b_hat_sub_theta) / denom
    return g_hat, i_hat


def _u_hat_loop(*, grids: V3Grids, geom: BoozerGeometry) -> jnp.ndarray:
    """Reference implementation of `uHat` using explicit harmonic loops (slow but transparent).

    This follows the cosine-only branch in v3 `geometry.F90` and is useful for validating
    faster implementations.
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


def u_hat(*, grids: V3Grids, geom: BoozerGeometry) -> jnp.ndarray:
    """Compute `uHat` using an FFT-based mode-by-mode transform (fast, JIT-friendly).

    Notes
    -----
    v3 computes `uHat` by projecting :math:`\\hat h = 1/\\hat B^2` onto harmonics

      cos(m*theta - n*NPeriods*zeta)

    and applying a per-mode scaling. On a uniform periodic grid, this projection is exactly
    a discrete Fourier transform, so we can compute `uHat` efficiently via `fft2/ifft2`.
    """
    b_hat = jnp.asarray(geom.b_hat, dtype=jnp.float64)
    ntheta, nzeta = int(b_hat.shape[0]), int(b_hat.shape[1])

    # Work in zeta' = NPeriods*zeta, which is 2π-periodic on the v3 grids.
    # The grid samples already correspond to zeta' = 2π*j/Nzeta, so the standard FFT basis applies.
    h_hat = 1.0 / (b_hat * b_hat)
    f = jnp.fft.fft2(h_hat)  # complex128 when x64 is enabled

    m = jnp.fft.fftfreq(ntheta, d=1.0 / ntheta)  # integer-valued float frequencies
    k = jnp.fft.fftfreq(nzeta, d=1.0 / nzeta)
    mm, kk = jnp.meshgrid(m, k, indexing="ij")

    # Map FFT basis exp(i(m*theta + k*zeta')) to the v3 harmonic exp(i(m*theta - n*NPeriods*zeta)):
    #   zeta' = NPeriods*zeta so exp(i(m*theta - n*NPeriods*zeta)) = exp(i(m*theta - n*zeta')).
    # In the FFT basis k multiplies +zeta', so k = -n  =>  n = -k, and the uHat scale uses n*NPeriods.
    n_periods = jnp.asarray(float(geom.n_periods), dtype=jnp.float64)
    n_eff = (-kk) * n_periods
    denom = n_eff - jnp.asarray(geom.iota, dtype=jnp.float64) * mm
    numer = jnp.asarray(geom.iota, dtype=jnp.float64) * (
        jnp.asarray(geom.g_hat, dtype=jnp.float64) * mm + jnp.asarray(geom.i_hat, dtype=jnp.float64) * n_eff
    )
    scale = jnp.where(jnp.abs(denom) < 1e-30, 0.0, numer / denom).astype(f.dtype)

    # Omit the (0,0) mode.
    scale = scale.at[0, 0].set(jnp.asarray(0.0, dtype=f.dtype))

    u = jnp.fft.ifft2(scale * f).real
    return jnp.asarray(u, dtype=jnp.float64)


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

            h_amp = 0.0
            for itheta in range(ntheta):
                for izeta in range(nzeta):
                    h_amp += float(cos_angle[itheta, izeta] * h_hat[itheta, izeta])
            h_amp = weight * h_amp

            denom = float(n * geom.n_periods) - float(geom.iota) * float(m)
            numer = float(geom.iota) * (float(geom.g_hat) * float(m) + float(geom.i_hat) * float(n * geom.n_periods))
            u_amp = 0.0 if abs(denom) < 1e-30 else (numer / denom) * h_amp

            u = u + u_amp * cos_angle

    return u
