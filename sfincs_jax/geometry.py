from __future__ import annotations

from dataclasses import dataclass

from jax import config as _jax_config
_jax_config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from .boozer_bc import read_boozer_bc_bracketing_surfaces


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
    db_hat_sub_theta_dpsi_hat: jnp.ndarray
    db_hat_sub_zeta_dpsi_hat: jnp.ndarray

    # Additional derivative datasets written by v3 `sfincsOutput.h5`:
    db_hat_sub_theta_dzeta: jnp.ndarray
    db_hat_sub_zeta_dtheta: jnp.ndarray
    db_hat_sup_theta_dpsi_hat: jnp.ndarray
    db_hat_sup_theta_dzeta: jnp.ndarray
    db_hat_sup_zeta_dpsi_hat: jnp.ndarray
    db_hat_sup_zeta_dtheta: jnp.ndarray


def boozer_geometry_scheme1(
    *,
    theta: jnp.ndarray,
    zeta: jnp.ndarray,
    epsilon_t: float,
    epsilon_h: float,
    epsilon_antisymm: float,
    iota: float,
    g_hat: float,
    i_hat: float,
    b0_over_bbar: float,
    helicity_l: int,
    helicity_n: int,
    helicity_antisymm_l: int,
    helicity_antisymm_n: int,
) -> BoozerGeometry:
    """Three-helicity analytic Boozer model (`geometryScheme = 1` in v3).

    v3 defines:

      BHat = B0OverBBar
             + (epsilon_t * B0OverBBar) * cos(theta)
             + (epsilon_h * B0OverBBar) * cos(helicity_l*theta - helicity_n*zeta)
             + (epsilon_antisymm * B0OverBBar) * sin(helicity_antisymm_l*theta - helicity_antisymm_n*zeta)

    with `NPeriods = max(1, helicity_n)` and the Boozer toroidal angle sampled on
    `[0, 2π/NPeriods)`.
    """
    n_periods = max(1, int(helicity_n))
    theta2 = jnp.asarray(theta, dtype=jnp.float64)[:, None]
    zeta2 = jnp.asarray(zeta, dtype=jnp.float64)[None, :]

    # v3 stores harmonics as cos(m*theta - NPeriods*n*zeta) / sin(...). With NPeriods=helicity_n,
    # the main helical ripple term uses n=1 to represent helicity_n*zeta.
    n2 = 0 if int(helicity_n) == 0 else 1
    if int(helicity_n) == 0:
        n3 = int(helicity_antisymm_n)
    else:
        n3 = int(helicity_antisymm_n) // int(helicity_n) if int(helicity_n) != 0 else 0

    m = jnp.asarray([1, int(helicity_l), int(helicity_antisymm_l)], dtype=jnp.float64)[:, None, None]
    n = jnp.asarray([0, int(n2), int(n3)], dtype=jnp.float64)[:, None, None]
    parity = jnp.asarray([True, True, False], dtype=bool)[:, None, None]
    amp = (
        jnp.asarray([float(epsilon_t), float(epsilon_h), float(epsilon_antisymm)], dtype=jnp.float64) * float(b0_over_bbar)
    )[:, None, None]

    angle = m * theta2[None, :, :] - float(n_periods) * n * zeta2[None, :, :]
    cos_a = jnp.cos(angle)
    sin_a = jnp.sin(angle)

    basis = jnp.where(parity, cos_a, sin_a)
    b_hat = float(b0_over_bbar) + jnp.sum(amp * basis, axis=0)

    # Derivatives:
    dtheta_basis = jnp.where(parity, -m * sin_a, m * cos_a)
    db_hat_dtheta = jnp.sum(amp * dtheta_basis, axis=0)

    dzeta_factor = float(n_periods) * n
    dzeta_basis = jnp.where(parity, dzeta_factor * sin_a, -dzeta_factor * cos_a)
    db_hat_dzeta = jnp.sum(amp * dzeta_basis, axis=0)

    denom = float(g_hat) + float(iota) * float(i_hat)
    d_hat = (b_hat * b_hat) / denom
    b_hat_sup_theta = float(iota) * d_hat
    b_hat_sup_zeta = d_hat
    b_hat_sub_theta = jnp.full_like(b_hat, float(i_hat))
    b_hat_sub_zeta = jnp.full_like(b_hat, float(g_hat))

    zeros = jnp.zeros_like(b_hat)
    return BoozerGeometry(
        n_periods=n_periods,
        b0_over_bbar=float(b0_over_bbar),
        iota=float(iota),
        g_hat=float(g_hat),
        i_hat=float(i_hat),
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
        db_hat_sub_theta_dpsi_hat=zeros,
        db_hat_sub_zeta_dpsi_hat=zeros,
        db_hat_sub_theta_dzeta=zeros,
        db_hat_sub_zeta_dtheta=zeros,
        db_hat_sup_theta_dpsi_hat=zeros,
        db_hat_sup_theta_dzeta=zeros,
        db_hat_sup_zeta_dpsi_hat=zeros,
        db_hat_sup_zeta_dtheta=zeros,
    )


def boozer_geometry_scheme2(*, theta: jnp.ndarray, zeta: jnp.ndarray) -> BoozerGeometry:
    """Simplified LHD model (`geometryScheme = 2` in v3).

    Values are from Beidler et al, Nuclear Fusion 51, 076001 (2011), Table 1, matching v3.
    """
    # v3 hard-codes these values in `geometry.F90` for scheme 2:
    n_periods = 10
    iota = 0.4542
    b0_over_bbar = 1.0
    r0 = 3.7481
    g_hat = b0_over_bbar * r0
    i_hat = 0.0

    theta2 = jnp.asarray(theta, dtype=jnp.float64)[:, None]
    zeta2 = jnp.asarray(zeta, dtype=jnp.float64)[None, :]

    # Harmonics: (m, n, amplitude) with cosine parity.
    m = jnp.asarray([1, 2, 1], dtype=jnp.float64)[:, None, None]
    n = jnp.asarray([0, 1, 1], dtype=jnp.float64)[:, None, None]
    amp = jnp.asarray([-0.07053, 0.05067, -0.01476], dtype=jnp.float64)[:, None, None] * b0_over_bbar

    angle = m * theta2[None, :, :] - float(n_periods) * n * zeta2[None, :, :]
    cos_a = jnp.cos(angle)
    sin_a = jnp.sin(angle)

    b_hat = b0_over_bbar + jnp.sum(amp * cos_a, axis=0)
    db_hat_dtheta = -jnp.sum(amp * m * sin_a, axis=0)
    db_hat_dzeta = jnp.sum(amp * (float(n_periods) * n) * sin_a, axis=0)

    denom = float(g_hat) + float(iota) * float(i_hat)
    d_hat = (b_hat * b_hat) / denom
    b_hat_sup_theta = float(iota) * d_hat
    b_hat_sup_zeta = d_hat
    b_hat_sub_theta = jnp.full_like(b_hat, float(i_hat))
    b_hat_sub_zeta = jnp.full_like(b_hat, float(g_hat))

    zeros = jnp.zeros_like(b_hat)
    return BoozerGeometry(
        n_periods=n_periods,
        b0_over_bbar=float(b0_over_bbar),
        iota=float(iota),
        g_hat=float(g_hat),
        i_hat=float(i_hat),
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
        db_hat_sub_theta_dpsi_hat=zeros,
        db_hat_sub_zeta_dpsi_hat=zeros,
        db_hat_sub_theta_dzeta=zeros,
        db_hat_sub_zeta_dtheta=zeros,
        db_hat_sup_theta_dpsi_hat=zeros,
        db_hat_sup_theta_dzeta=zeros,
        db_hat_sup_zeta_dpsi_hat=zeros,
        db_hat_sup_zeta_dtheta=zeros,
    )


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
        db_hat_sub_theta_dpsi_hat=zeros,
        db_hat_sub_zeta_dpsi_hat=zeros,
        db_hat_sub_theta_dzeta=zeros,
        db_hat_sub_zeta_dtheta=zeros,
        db_hat_sup_theta_dpsi_hat=zeros,
        db_hat_sup_theta_dzeta=zeros,
        db_hat_sup_zeta_dpsi_hat=zeros,
        db_hat_sup_zeta_dtheta=zeros,
    )


def _eval_b_series_and_derivatives(
    *,
    theta: np.ndarray,
    zeta: np.ndarray,
    n_periods: int,
    b0_over_bbar: float,
    m: np.ndarray,
    n: np.ndarray,
    parity: np.ndarray,
    b_amp: np.ndarray,
    chunk: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate BHat(theta,zeta) and its theta/zeta derivatives from Fourier tables."""
    theta1 = theta[None, :, None]  # (1,T,1)
    zeta1 = zeta[None, None, :]  # (1,1,Z)
    out = np.full((theta.shape[0], zeta.shape[0]), float(b0_over_bbar), dtype=np.float64)
    dbdtheta = np.zeros_like(out)
    dbdzeta = np.zeros_like(out)

    # v3 truncates the harmonic table to modes representable on the (theta,zeta) grid.
    ntheta = int(theta.shape[0])
    nzeta = int(zeta.shape[0])
    m_max_grid = int(ntheta / 2.0)
    n_max_grid = int(nzeta / 2.0)
    if nzeta == 1:
        include = np.ones((int(m.shape[0]),), dtype=bool)
    else:
        include = (np.abs(n) <= n_max_grid) & (m <= m_max_grid)

    # Additional Nyquist exclusions for sine components (see v3 `computeBHat`):
    is_sin = ~parity.astype(bool)
    if nzeta != 1 and np.any(is_sin):
        at_m_nyq = (m == 0) | (m.astype(np.float64) == (ntheta / 2.0))
        at_n_nyq = (n == 0) | (np.abs(n.astype(np.float64)) == (nzeta / 2.0))
        include = include & ~(is_sin & at_m_nyq & at_n_nyq)

    m = m[include]
    n = n[include]
    parity = parity[include]
    b_amp = b_amp[include]

    h = int(m.shape[0])
    for i0 in range(0, h, chunk):
        i1 = min(h, i0 + chunk)
        mc = m[i0:i1].astype(np.float64)[:, None, None]
        nc = n[i0:i1].astype(np.float64)[:, None, None]
        bc = b_amp[i0:i1].astype(np.float64)[:, None, None]
        pc = parity[i0:i1].astype(bool)[:, None, None]

        angle = mc * theta1 - float(n_periods) * nc * zeta1
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        basis = np.where(pc, cos_a, sin_a)
        out = out + np.sum(bc * basis, axis=0)

        # d/dtheta:
        dtheta_basis = np.where(pc, -mc * sin_a, mc * cos_a)
        dbdtheta = dbdtheta + np.sum(bc * dtheta_basis, axis=0)

        # d/dzeta:
        dzeta_factor = float(n_periods) * nc
        dzeta_basis = np.where(pc, dzeta_factor * sin_a, -dzeta_factor * cos_a)
        dbdzeta = dbdzeta + np.sum(bc * dzeta_basis, axis=0)

    return out, dbdtheta, dbdzeta


def _compute_u_and_bsubpsi(
    *,
    theta: np.ndarray,
    zeta: np.ndarray,
    n_periods: int,
    b_hat: np.ndarray,
    iota: float,
    g_hat: float,
    i_hat: float,
    p_prime_hat: float,
    non_stel_sym: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute BHat_sub_psi and its derivatives using v3's harmonic projection algorithm."""
    ntheta = int(theta.shape[0])
    nzeta = int(zeta.shape[0])

    h_hat = 1.0 / (b_hat * b_hat)

    b_sub_psi = np.zeros_like(b_hat)
    db_sub_psi_dtheta = np.zeros_like(b_hat)
    db_sub_psi_dzeta = np.zeros_like(b_hat)

    m_max = int(ntheta / 2.0)
    n_max = int(nzeta / 2.0)
    theta_half = ntheta / 2.0
    zeta_half = nzeta / 2.0
    zeta_is_even = float(int(zeta_half)) == zeta_half

    denom_eps = 0.0  # v3 does not guard the denominator except by skipping (0,0).

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
            # Cosine component:
            h_amp = 0.0
            for itheta in range(ntheta):
                ang = float(m) * float(theta[itheta]) - float(n * n_periods) * zeta
                c = np.cos(ang)
                nyquist = (
                    (m == 0 and float(n) == zeta_half)
                    or (float(m) == theta_half and n == 0)
                    or (float(m) == theta_half and float(n) == zeta_half)
                )
                w = (1.0 if nyquist else 2.0) / float(ntheta * nzeta)
                h_amp += w * float(np.dot(c, h_hat[itheta, :]))

            denom = float(n * n_periods) - float(iota) * float(m)
            numer = float(iota) * (float(g_hat) * float(m) + float(i_hat) * float(n * n_periods))
            if abs(denom) <= denom_eps:
                u_amp = 0.0
            else:
                u_amp = (numer / denom) * h_amp

            d_bsubpsi_dtheta_amp = -float(p_prime_hat) / float(iota) * (u_amp - float(iota) * float(i_hat) * h_amp)
            d_bsubpsi_dzeta_amp = float(p_prime_hat) * (u_amp + float(g_hat) * h_amp)

            for itheta in range(ntheta):
                ang = float(m) * float(theta[itheta]) - float(n * n_periods) * zeta
                c = np.cos(ang)
                s = np.sin(ang)
                db_sub_psi_dtheta[itheta, :] += d_bsubpsi_dtheta_amp * c
                db_sub_psi_dzeta[itheta, :] += d_bsubpsi_dzeta_amp * c

                if n == 0:
                    b_sub_psi[itheta, :] += (d_bsubpsi_dtheta_amp / float(m)) * s
                else:
                    b_sub_psi[itheta, :] += -(d_bsubpsi_dzeta_amp / float(n) / float(n_periods)) * s

            if not non_stel_sym:
                continue

            # Sine component:
            h_amp = 0.0
            nyquist = (
                (m == 0 and float(n) == zeta_half)
                or (float(m) == theta_half and n == 0)
                or (float(m) == theta_half and float(n) == zeta_half)
            )
            if not nyquist:
                for itheta in range(ntheta):
                    ang = float(m) * float(theta[itheta]) - float(n * n_periods) * zeta
                    s = np.sin(ang)
                    w = 2.0 / float(ntheta * nzeta)
                    h_amp += w * float(np.dot(s, h_hat[itheta, :]))

            denom = float(n * n_periods) - float(iota) * float(m)
            numer = float(iota) * (float(g_hat) * float(m) + float(i_hat) * float(n * n_periods))
            if abs(denom) <= denom_eps:
                u_amp = 0.0
            else:
                u_amp = (numer / denom) * h_amp

            d_bsubpsi_dtheta_amp = -float(p_prime_hat) / float(iota) * (u_amp - float(iota) * float(i_hat) * h_amp)
            d_bsubpsi_dzeta_amp = float(p_prime_hat) * (u_amp + float(g_hat) * h_amp)

            for itheta in range(ntheta):
                ang = float(m) * float(theta[itheta]) - float(n * n_periods) * zeta
                c = np.cos(ang)
                s = np.sin(ang)
                db_sub_psi_dtheta[itheta, :] += d_bsubpsi_dtheta_amp * s
                db_sub_psi_dzeta[itheta, :] += d_bsubpsi_dzeta_amp * s

                if n == 0:
                    b_sub_psi[itheta, :] += -(d_bsubpsi_dtheta_amp / float(m)) * c
                else:
                    b_sub_psi[itheta, :] += (d_bsubpsi_dzeta_amp / float(n) / float(n_periods)) * c

    return b_sub_psi, db_sub_psi_dtheta, db_sub_psi_dzeta


def boozer_geometry_from_bc_file(
    *,
    path: str,
    theta: jnp.ndarray,
    zeta: jnp.ndarray,
    r_n_wish: float,
    vmecradial_option: int = 0,
    geometry_scheme: int = 11,
) -> BoozerGeometry:
    """Compute Boozer geometry from a v3 `.bc` file (geometryScheme 11/12).

    Notes
    -----
    - This routine follows the conventions and sign switches used in SFINCS v3 `geometry.F90`.
    - The `.bc` file read is not differentiable. Once loaded, the resulting geometry arrays
      are standard JAX arrays and can be used in JIT-compiled kernels.
    """
    if geometry_scheme not in {11, 12}:
        raise ValueError(f"geometry_scheme must be 11 or 12, got {geometry_scheme}")

    header, surf_old, surf_new = read_boozer_bc_bracketing_surfaces(
        path=path, geometry_scheme=geometry_scheme, r_n_wish=float(r_n_wish)
    )

    r_old = float(surf_old.r_n)
    r_new = float(surf_new.r_n)
    if r_new == r_old:
        radial_weight = 1.0
    else:
        if int(vmecradial_option) == 1:
            if abs(r_old - float(r_n_wish)) < abs(r_new - float(r_n_wish)):
                radial_weight = 1.0
            else:
                radial_weight = 0.0
        else:
            radial_weight = (r_new * r_new - float(r_n_wish) * float(r_n_wish)) / (r_new * r_new - r_old * r_old)

    # Interpolate scalar surface quantities:
    iota_old, iota_new = float(surf_old.iota), float(surf_new.iota)
    g_old, g_new = float(surf_old.g_hat), float(surf_new.g_hat)
    i_old, i_new = float(surf_old.i_hat), float(surf_new.i_hat)
    b0_old, b0_new = float(surf_old.b0_over_bbar), float(surf_new.b0_over_bbar)
    p_old, p_new = float(surf_old.p_prime_hat), float(surf_new.p_prime_hat)

    iota = iota_old * radial_weight + iota_new * (1.0 - radial_weight)
    g_hat = g_old * radial_weight + g_new * (1.0 - radial_weight)
    i_hat = i_old * radial_weight + i_new * (1.0 - radial_weight)
    b0_over_bbar = b0_old * radial_weight + b0_new * (1.0 - radial_weight)
    p_prime_hat = p_old * radial_weight + p_new * (1.0 - radial_weight)

    delta_psi_hat = float(header.psi_a_hat) * (r_new * r_new - r_old * r_old)
    if delta_psi_hat == 0.0:
        raise ValueError("delta_psi_hat is zero; cannot compute radial derivatives from nearby radii.")

    # Sign flips:
    if geometry_scheme == 11 and (g_hat * float(header.psi_a_hat) > 0.0):
        g_hat *= -1.0
        g_old *= -1.0
        g_new *= -1.0
        i_hat *= -1.0
        i_old *= -1.0
        i_new *= -1.0

    # Toroidal direction sign switch:
    g_hat *= -1.0
    g_old *= -1.0
    g_new *= -1.0
    iota *= -1.0
    iota_old *= -1.0
    iota_new *= -1.0

    n_old = -np.asarray(surf_old.n, dtype=np.int32)
    n_new = -np.asarray(surf_new.n, dtype=np.int32)

    # Evaluate BHat on the theta/zeta grids for the bracketing surfaces.
    theta_np = np.asarray(theta, dtype=np.float64)
    zeta_np = np.asarray(zeta, dtype=np.float64)

    b_l, dbdtheta_l, dbdzeta_l = _eval_b_series_and_derivatives(
        theta=theta_np,
        zeta=zeta_np,
        n_periods=int(header.n_periods),
        b0_over_bbar=b0_old,
        m=np.asarray(surf_old.m, dtype=np.int32),
        n=n_old,
        parity=np.asarray(surf_old.parity, dtype=bool),
        b_amp=np.asarray(surf_old.b_amp, dtype=np.float64),
    )
    b_h, dbdtheta_h, dbdzeta_h = _eval_b_series_and_derivatives(
        theta=theta_np,
        zeta=zeta_np,
        n_periods=int(header.n_periods),
        b0_over_bbar=b0_new,
        m=np.asarray(surf_new.m, dtype=np.int32),
        n=n_new,
        parity=np.asarray(surf_new.parity, dtype=bool),
        b_amp=np.asarray(surf_new.b_amp, dtype=np.float64),
    )

    b_hat = b_l * radial_weight + b_h * (1.0 - radial_weight)
    db_hat_dtheta = dbdtheta_l * radial_weight + dbdtheta_h * (1.0 - radial_weight)
    db_hat_dzeta = dbdzeta_l * radial_weight + dbdzeta_h * (1.0 - radial_weight)
    db_hat_dpsi_hat = (b_h - b_l) / float(delta_psi_hat)

    # Covariant components of B in Boozer coords:
    b_hat_sub_theta = np.full_like(b_hat, float(i_hat), dtype=np.float64)
    b_hat_sub_zeta = np.full_like(b_hat, float(g_hat), dtype=np.float64)
    db_hat_sub_theta_dpsi_hat = np.full_like(b_hat, (i_new - i_old) / float(delta_psi_hat), dtype=np.float64)
    db_hat_sub_zeta_dpsi_hat = np.full_like(b_hat, (g_new - g_old) / float(delta_psi_hat), dtype=np.float64)

    # Contravariant components via D = B^2 / (G + iota I)
    denom = float(g_hat) + float(iota) * float(i_hat)
    d_hat = (b_hat * b_hat) / denom
    b_hat_sup_theta = float(iota) * d_hat
    b_hat_sup_zeta = d_hat

    # Additional v3 output derivatives (computed when nearby radii are available, as they are here).
    denom = float(g_hat) + float(iota) * float(i_hat)
    diotadpsi_hat = (float(iota_new) - float(iota_old)) / float(delta_psi_hat)

    d_bsup_zeta_dpsi_hat = (
        2.0 * b_hat * db_hat_dpsi_hat / denom
        - (
            db_hat_sub_zeta_dpsi_hat
            + float(iota) * db_hat_sub_theta_dpsi_hat
            + float(diotadpsi_hat) * float(i_hat)
        )
        / (denom * denom)
    )
    d_bsup_zeta_dtheta = 2.0 * b_hat * db_hat_dtheta / denom
    d_bsup_theta_dpsi_hat = float(iota) * d_bsup_zeta_dpsi_hat + float(diotadpsi_hat) * d_hat
    d_bsup_theta_dzeta = float(iota) * 2.0 * b_hat * db_hat_dzeta / denom

    # Compute BHat_sub_psi and its derivatives as in v3 (used in magnetic drifts):
    non_stel_sym = bool(np.any(~np.asarray(surf_old.parity)) or np.any(~np.asarray(surf_new.parity)))
    b_hat_sub_psi, db_hat_sub_psi_dtheta, db_hat_sub_psi_dzeta = _compute_u_and_bsubpsi(
        theta=theta_np,
        zeta=zeta_np,
        n_periods=int(header.n_periods),
        b_hat=b_hat,
        iota=float(iota),
        g_hat=float(g_hat),
        i_hat=float(i_hat),
        p_prime_hat=float(p_prime_hat),
        non_stel_sym=non_stel_sym,
    )

    zeros = np.zeros_like(b_hat)
    return BoozerGeometry(
        n_periods=int(header.n_periods),
        b0_over_bbar=float(b0_over_bbar),
        iota=float(iota),
        g_hat=float(g_hat),
        i_hat=float(i_hat),
        b_hat=jnp.asarray(b_hat),
        db_hat_dtheta=jnp.asarray(db_hat_dtheta),
        db_hat_dzeta=jnp.asarray(db_hat_dzeta),
        d_hat=jnp.asarray(d_hat),
        b_hat_sup_theta=jnp.asarray(b_hat_sup_theta),
        b_hat_sup_zeta=jnp.asarray(b_hat_sup_zeta),
        b_hat_sub_theta=jnp.asarray(b_hat_sub_theta),
        b_hat_sub_zeta=jnp.asarray(b_hat_sub_zeta),
        b_hat_sub_psi=jnp.asarray(b_hat_sub_psi),
        db_hat_dpsi_hat=jnp.asarray(db_hat_dpsi_hat),
        db_hat_sub_psi_dtheta=jnp.asarray(db_hat_sub_psi_dtheta),
        db_hat_sub_psi_dzeta=jnp.asarray(db_hat_sub_psi_dzeta),
        db_hat_sub_theta_dpsi_hat=jnp.asarray(db_hat_sub_theta_dpsi_hat),
        db_hat_sub_zeta_dpsi_hat=jnp.asarray(db_hat_sub_zeta_dpsi_hat),
        db_hat_sub_theta_dzeta=jnp.asarray(zeros),
        db_hat_sub_zeta_dtheta=jnp.asarray(zeros),
        db_hat_sup_theta_dpsi_hat=jnp.asarray(d_bsup_theta_dpsi_hat),
        db_hat_sup_theta_dzeta=jnp.asarray(d_bsup_theta_dzeta),
        db_hat_sup_zeta_dpsi_hat=jnp.asarray(d_bsup_zeta_dpsi_hat),
        db_hat_sup_zeta_dtheta=jnp.asarray(d_bsup_zeta_dtheta),
    )
