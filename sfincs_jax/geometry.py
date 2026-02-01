from __future__ import annotations

from dataclasses import dataclass

from jax import config as _jax_config
_jax_config.update("jax_enable_x64", True)
import jax.numpy as jnp


@dataclass(frozen=True)
class BoozerGeometry:
    # Scalars:
    n_periods: int
    b0_over_bbar: float
    iota: float
    g_hat: float
    i_hat: float

    # (Ntheta, Nzeta) arrays:
    b_hat: jnp.ndarray
    db_hat_dtheta: jnp.ndarray
    db_hat_dzeta: jnp.ndarray

    d_hat: jnp.ndarray
    b_hat_sup_theta: jnp.ndarray
    b_hat_sup_zeta: jnp.ndarray
    b_hat_sub_theta: jnp.ndarray
    b_hat_sub_zeta: jnp.ndarray

    # Boozer-only, but included for parity with Fortran output:
    b_hat_sub_psi: jnp.ndarray
    db_hat_dpsi_hat: jnp.ndarray
    db_hat_sub_psi_dtheta: jnp.ndarray
    db_hat_sub_psi_dzeta: jnp.ndarray


def _scheme4_default_harmonics() -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return the (l, n, amp0) harmonic table for `geometryScheme=4`.

    The amplitudes `amp0` are normalized by B0 in the Fortran code, and later multiplied
    by `B0OverBBar`.
    """
    l = jnp.asarray([0, 1, 1], dtype=jnp.int32)
    n = jnp.asarray([1, 1, 0], dtype=jnp.int32)
    amp0 = jnp.asarray([0.04645, -0.04351, -0.01902], dtype=jnp.float64)
    return l, n, amp0


def boozer_geometry_scheme4(
    *,
    theta: jnp.ndarray,
    zeta: jnp.ndarray,
    iota: float = 0.8700,
    b0_over_bbar: float = 3.089,
    g_hat: float = -17.885,
    i_hat: float = 0.0,
    harmonics_amp0: jnp.ndarray | None = None,
) -> BoozerGeometry:
    """Simplified W7-X Boozer model used by SFINCS v3 (`geometryScheme = 4`).

    Parameters
    ----------
    harmonics_amp0:
      Optional array of shape `(3,)` overriding the default harmonic amplitudes (normalized
      by B0). This hook exists mostly for differentiable-geometry demos and optimization
      examples.
    """
    n_periods = 5

    l, n, default_amp0 = _scheme4_default_harmonics()
    if harmonics_amp0 is None:
        amp0 = default_amp0
    else:
        amp0 = jnp.asarray(harmonics_amp0, dtype=jnp.float64)
        if amp0.shape != default_amp0.shape:
            raise ValueError(f"harmonics_amp0 must have shape {default_amp0.shape}, got {amp0.shape}")

    theta2 = theta[:, None]
    zeta2 = zeta[None, :]

    # Harmonics (l, n, amplitude) from Beidler et al, NF 51, 076001 (2011), Table 1.
    amp = (amp0 * b0_over_bbar)[:, None, None]  # (H,1,1)
    angle = l[:, None, None] * theta2[None, :, :] - n_periods * n[:, None, None] * zeta2[None, :, :]
    cos_angle = jnp.cos(angle)
    sin_angle = jnp.sin(angle)

    b_hat = b0_over_bbar + jnp.sum(amp * cos_angle, axis=0)
    db_hat_dtheta = -jnp.sum(amp * (l[:, None, None]) * sin_angle, axis=0)
    db_hat_dzeta = jnp.sum(amp * (n_periods * n[:, None, None]) * sin_angle, axis=0)

    denom = g_hat + iota * i_hat
    d_hat = b_hat * b_hat / denom
    b_hat_sup_theta = iota * d_hat
    b_hat_sup_zeta = d_hat
    b_hat_sub_theta = jnp.full_like(b_hat, i_hat)
    b_hat_sub_zeta = jnp.full_like(b_hat, g_hat)

    zeros = jnp.zeros_like(b_hat)
    return BoozerGeometry(
        n_periods=n_periods,
        b0_over_bbar=b0_over_bbar,
        iota=iota,
        g_hat=g_hat,
        i_hat=i_hat,
        b_hat=b_hat,
        db_hat_dtheta=db_hat_dtheta,
        db_hat_dzeta=db_hat_dzeta,
        d_hat=d_hat,
        b_hat_sup_theta=b_hat_sup_theta,
        b_hat_sup_zeta=b_hat_sup_zeta,
        b_hat_sub_theta=b_hat_sub_theta,
        b_hat_sub_zeta=b_hat_sub_zeta,
        b_hat_sub_psi=zeros,
        db_hat_dpsi_hat=zeros,
        db_hat_sub_psi_dtheta=zeros,
        db_hat_sub_psi_dzeta=zeros,
    )
