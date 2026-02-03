from __future__ import annotations

from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)

import jax.numpy as jnp


def classical_flux_v3(
    *,
    use_phi1: bool,
    # Geometry / grids:
    theta_weights: jnp.ndarray,  # (T,)
    zeta_weights: jnp.ndarray,  # (Z,)
    d_hat: jnp.ndarray,  # (T,Z)
    gpsipsi: jnp.ndarray,  # (T,Z) written as gpsiHatpsiHat
    b_hat: jnp.ndarray,  # (T,Z)
    vprime_hat: jnp.ndarray,  # scalar
    # Plasma parameters:
    alpha: jnp.ndarray,  # scalar
    phi1_hat: jnp.ndarray,  # (T,Z) (ignored if use_phi1=False)
    delta: jnp.ndarray,  # scalar
    nu_n: jnp.ndarray,  # scalar
    z_s: jnp.ndarray,  # (S,)
    m_hat: jnp.ndarray,  # (S,)
    t_hat: jnp.ndarray,  # (S,)
    n_hat: jnp.ndarray,  # (S,)
    dn_hat_dpsi_hat: jnp.ndarray,  # (S,)
    dt_hat_dpsi_hat: jnp.ndarray,  # (S,)
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute the v3 classical particle/heat fluxes (psiHat projection).

    This matches `sfincs/fortran/version3/classicalTransport.F90:calculateClassicalFlux`.

    Returns
    -------
    classical_pf_psi_hat:
      (S,) array, v3-normalized classical particle flux projected onto ∇psiHat.
    classical_hf_psi_hat:
      (S,) array, v3-normalized total classical heat flux projected onto ∇psiHat.
    """
    theta_weights = jnp.asarray(theta_weights, dtype=jnp.float64)
    zeta_weights = jnp.asarray(zeta_weights, dtype=jnp.float64)
    d_hat = jnp.asarray(d_hat, dtype=jnp.float64)
    gpsipsi = jnp.asarray(gpsipsi, dtype=jnp.float64)
    b_hat = jnp.asarray(b_hat, dtype=jnp.float64)
    vprime_hat = jnp.asarray(vprime_hat, dtype=jnp.float64)

    alpha = jnp.asarray(alpha, dtype=jnp.float64)
    phi1_hat = jnp.asarray(phi1_hat, dtype=jnp.float64)
    delta = jnp.asarray(delta, dtype=jnp.float64)
    nu_n = jnp.asarray(nu_n, dtype=jnp.float64)
    z_s = jnp.asarray(z_s, dtype=jnp.float64)
    m_hat = jnp.asarray(m_hat, dtype=jnp.float64)
    t_hat = jnp.asarray(t_hat, dtype=jnp.float64)
    n_hat = jnp.asarray(n_hat, dtype=jnp.float64)
    dn_hat_dpsi_hat = jnp.asarray(dn_hat_dpsi_hat, dtype=jnp.float64)
    dt_hat_dpsi_hat = jnp.asarray(dt_hat_dpsi_hat, dtype=jnp.float64)

    w = (theta_weights[:, None] * zeta_weights[None, :]) / d_hat  # (T,Z)
    integrand = w * (gpsipsi / (b_hat * b_hat))  # (T,Z)

    s = int(z_s.shape[0])
    # Species-pair coefficients:
    xab2 = (m_hat[:, None] * t_hat[None, :]) / (m_hat[None, :] * t_hat[:, None])  # (S,S)
    m_ratio = m_hat[:, None] / m_hat[None, :]  # (S,S)
    one_plus_x = 1.0 + xab2
    denom = one_plus_x ** (2.5)

    mab00 = -((1.0 + m_ratio) * one_plus_x) / denom
    mab01 = -(1.5) * (1.0 + m_ratio) / denom
    mab11 = -(13.0 + 16.0 * xab2 + 30.0 * (xab2**2)) / 4.0 / denom
    nab11 = (27.0 * m_ratio) / 4.0 / denom

    if use_phi1:
        coef_ab = alpha * ((z_s / t_hat)[:, None] + (z_s / t_hat)[None, :])  # (S,S)
        exp_ab = jnp.exp(-coef_ab[:, :, None, None] * phi1_hat[None, None, :, :])  # (S,S,T,Z)
        geom1 = jnp.einsum("tz,abtz->ab", integrand, exp_ab) / vprime_hat
        geom2 = jnp.einsum("tz,abtz->ab", integrand * phi1_hat, exp_ab) / vprime_hat
    else:
        geom1_scalar = jnp.sum(integrand) / vprime_hat
        geom1 = jnp.broadcast_to(geom1_scalar, (s, s))
        geom2 = jnp.zeros((s, s), dtype=jnp.float64)

    geom1 = geom1 * (n_hat[:, None] * n_hat[None, :])
    geom2 = geom2 * (n_hat[:, None] * n_hat[None, :])

    u_dn = (t_hat * dn_hat_dpsi_hat) / (n_hat * z_s)  # (S,)
    u_dt = dt_hat_dpsi_hat / t_hat  # (S,)
    u_dt_over_z = dt_hat_dpsi_hat / z_s  # (S,)

    term_dn = u_dn[:, None] - u_dn[None, :]
    term_dt = u_dt[:, None] - u_dt[None, :]

    pf_ab = (
        geom1 * mab00 * term_dn
        + geom2 * alpha * mab00 * term_dt
        + geom1
        * (
            (mab00 - mab01) * u_dt_over_z[:, None]
            - (mab00 - xab2 * mab01) * u_dt_over_z[None, :]
        )
    )
    hf_ab = (
        geom1 * mab01 * term_dn
        + geom2 * alpha * mab01 * term_dt
        + geom1
        * (
            (mab01 - mab11) * u_dt_over_z[:, None]
            - (mab01 + nab11) * u_dt_over_z[None, :]
        )
    )

    z2_b = (z_s[None, :] ** 2)  # (1,S)
    pf_a = jnp.sum(z2_b * pf_ab, axis=1)  # (S,)
    hf_a = jnp.sum(z2_b * hf_ab, axis=1)  # (S,)

    # Final prefactors and "total" heat flux definition matching v3 comments:
    pf_a = z_s * (delta**2) * nu_n * jnp.sqrt(m_hat) * pf_a / (2.0 * (t_hat**1.5))
    hf_a = -z_s * (delta**2) * nu_n * jnp.sqrt(m_hat) * hf_a / (4.0 * jnp.sqrt(t_hat))
    hf_a = hf_a + 1.25 * t_hat * pf_a
    return pf_a, hf_a

