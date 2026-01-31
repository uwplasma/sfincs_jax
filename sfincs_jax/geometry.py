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


def boozer_geometry_scheme4(*, theta: jnp.ndarray, zeta: jnp.ndarray) -> BoozerGeometry:
    """Simplified W7-X Boozer model used by SFINCS v3 (`geometryScheme = 4`)."""
    n_periods = 5

    iota = 0.8700
    b0_over_bbar = 3.089
    g_hat = -17.885
    i_hat = 0.0

    # Harmonics (l, n, amplitude) from Beidler et al, NF 51, 076001 (2011), Table 1.
    # Note: the Fortran code stores amplitudes normalized by B0 then multiplies by B0OverBBar.
    harmonics = [
        (0, 1, 0.04645),
        (1, 1, -0.04351),
        (1, 0, -0.01902),
    ]

    theta2 = theta[:, None]
    zeta2 = zeta[None, :]

    b_hat = jnp.full((theta.size, zeta.size), b0_over_bbar)
    db_hat_dtheta = jnp.zeros_like(b_hat)
    db_hat_dzeta = jnp.zeros_like(b_hat)

    for l, n, amp0 in harmonics:
        amp = amp0 * b0_over_bbar
        angle = l * theta2 - n_periods * n * zeta2
        b_hat = b_hat + amp * jnp.cos(angle)
        db_hat_dtheta = db_hat_dtheta - amp * l * jnp.sin(angle)
        db_hat_dzeta = db_hat_dzeta + amp * n_periods * n * jnp.sin(angle)

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
