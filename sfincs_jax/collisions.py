from __future__ import annotations

import math
from dataclasses import dataclass

from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
import numpy as np
from jax import tree_util as jtu
from jax.scipy.special import erf
from scipy import special as sp_special
from scipy.integrate import quad

from .xgrid import XGrid, make_x_grid


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


def _psi_chandra_np(x: np.ndarray) -> np.ndarray:
    """NumPy version of :func:`_psi_chandra` used in collision-operator precomputations."""
    x = np.asarray(x, dtype=np.float64)
    sqrt_pi = float(np.sqrt(np.pi))
    num = sp_special.erf(x) - (2.0 / sqrt_pi) * x * np.exp(-(x * x))
    den = 2.0 * x * x
    out = np.empty_like(x)
    eps = 1e-14
    small = np.abs(x) < eps
    out[small] = (2.0 / sqrt_pi) * x[small] / 3.0
    out[~small] = num[~small] / den[~small]
    return out


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


def polynomial_interpolation_matrix_np(
    *,
    xk: np.ndarray,  # (N,)
    x: np.ndarray,  # (M,)
    alpxk: np.ndarray,  # (N,)
    alpx: np.ndarray,  # (M,)
) -> np.ndarray:
    """Port of v3 `polynomialInterpolationMatrix` (barycentric spectral interpolation).

    This routine is used by the v3 Fokker-Planck collision operator to interpolate between
    species-specific speed variables.
    """
    xk = np.asarray(xk, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    alpxk = np.asarray(alpxk, dtype=np.float64)
    alpx = np.asarray(alpx, dtype=np.float64)
    n = int(xk.size)
    m = int(x.size)
    if alpxk.shape != (n,):
        raise ValueError(f"alpxk must have shape {(n,)}, got {alpxk.shape}")
    if alpx.shape != (m,):
        raise ValueError(f"alpx must have shape {(m,)}, got {alpx.shape}")

    # Barycentric weights: w(j) = 1 / Π_{i≠j} (xk(i) - xk(j)).
    d = xk[:, None] - xk[None, :]
    np.fill_diagonal(d, 1.0)
    w = 1.0 / np.prod(d, axis=0)

    # D(i,j) = 1/(x(i) - xk(j)) with a tiny regularization at exact matches.
    dx = x[:, None] - xk[None, :]
    dx = np.where(dx == 0.0, 1e-15, dx)
    inv_dx = 1.0 / dx

    denom = np.sum(inv_dx * w[None, :], axis=1)  # (M,)
    mat = inv_dx * (alpx[:, None] / denom[:, None])  # (M,N)
    mat = mat * (w[None, :] / alpxk[None, :])
    return mat


def _poly_coeffs_monomial(xg: XGrid) -> list[np.ndarray]:
    """Return monomial coefficients for the orthogonal polynomials used by v3 `xGrid`.

    The returned list is 0-based: `coeffs[j-1]` corresponds to the Fortran polynomial index `j`.
    Coefficients are in ascending powers, i.e. `p(x) = sum_m coeff[m] * x**m`.
    """
    n = int(xg.x.size)
    a = np.asarray(xg.poly_a, dtype=np.float64)
    b = np.asarray(xg.poly_b, dtype=np.float64)

    coeffs: list[np.ndarray] = []
    coeffs.append(np.array([1.0], dtype=np.float64))  # j=1
    if n == 1:
        return coeffs

    coeffs.append(np.array([-float(a[1]), 1.0], dtype=np.float64))  # j=2
    for j in range(2, n):
        aj = float(a[j])
        bj = float(b[j])
        p_j = coeffs[j - 1]
        p_jm1 = coeffs[j - 2]

        # (x - aj) * p_j
        out = np.zeros((p_j.size + 1,), dtype=np.float64)
        out[0] = -aj * p_j[0]
        for m in range(1, p_j.size):
            out[m] = p_j[m - 1] - aj * p_j[m]
        out[-1] = p_j[-1]

        # - bj * p_{j-1}
        out[: p_jm1.size] -= bj * p_jm1
        coeffs.append(out)
    return coeffs


def _monomial_int_lower(xb: float, n: int) -> float:
    """∫_0^xb t^n e^{-t^2} dt (n >= 0)."""
    if n < 0:
        raise ValueError("lower monomial integral is only used for n >= 0 in v3")
    a = 0.5 * (n + 1.0)
    return float(0.5 * sp_special.gamma(a) * sp_special.gammainc(a, xb * xb))


def _monomial_int_upper(xb: float, n: int) -> float:
    """∫_xb^∞ t^n e^{-t^2} dt (supports n < 0 via quadrature)."""
    if n >= 0:
        a = 0.5 * (n + 1.0)
        return float(0.5 * sp_special.gamma(a) * sp_special.gammaincc(a, xb * xb))

    def f(t: float) -> float:
        return (t**n) * math.exp(-(t * t))

    val, _ = quad(f, xb, np.inf, epsabs=1e-13, epsrel=1e-13, limit=5000)
    return float(val)


def rosenbluth_potential_terms_v3_np(
    *,
    x: np.ndarray,  # (X,)
    x_weights: np.ndarray,  # (X,) dx weights (as in v3 createGrids.F90)
    x_grid_k: float,
    xg: XGrid,
    z_s: np.ndarray,  # (S,)
    m_hats: np.ndarray,  # (S,)
    n_hats: np.ndarray,  # (S,)
    t_hats: np.ndarray,  # (S,)
    nl: int,
) -> np.ndarray:
    """Compute v3 `RosenbluthPotentialTerms` for xGridScheme=5/6 (new scheme).

    Returns
    -------
    terms:
      Array of shape (S, S, NL, X, X) with index ordering:
      (species_row, species_col, L, x_row, x_col).
    """
    x = np.asarray(x, dtype=np.float64)
    x_weights = np.asarray(x_weights, dtype=np.float64)
    z_s = np.asarray(z_s, dtype=np.float64)
    m_hats = np.asarray(m_hats, dtype=np.float64)
    n_hats = np.asarray(n_hats, dtype=np.float64)
    t_hats = np.asarray(t_hats, dtype=np.float64)

    n_x = int(x.size)
    n_species = int(z_s.size)
    expx2 = np.exp(-(x * x))

    # collocation2modal(j,i) in the Fortran code:
    poly_coeffs = _poly_coeffs_monomial(xg)
    poly_c = np.asarray(xg.poly_c, dtype=np.float64)
    pvals = np.zeros((n_x, n_x), dtype=np.float64)  # (j,i)
    for j in range(1, n_x + 1):
        coeff = poly_coeffs[j - 1]
        # Evaluate with ascending coefficients:
        p = np.zeros_like(x)
        for m, cm in enumerate(coeff):
            p += cm * (x**m)
        pvals[j - 1, :] = p
    collocation2modal = (x_weights[None, :] * (x[None, :] ** float(x_grid_k)) * pvals) / (
        poly_c[1 : n_x + 1, None]
    )

    terms = np.zeros((n_species, n_species, int(nl), n_x, n_x), dtype=np.float64)
    pi = float(np.pi)

    for l in range(int(nl)):
        alpha = -float(2 * l - 1) / float(2 * l + 3)
        denom_h = float(2 * l + 1)
        denom_g = float(4 * l * l - 1)
        for ia in range(n_species):
            for ib in range(n_species):
                species_factor = float(
                    math.sqrt((t_hats[ia] * m_hats[ib]) / (t_hats[ib] * m_hats[ia]))
                )
                species_factor2 = float(3.0 / (2.0 * pi)) * float(n_hats[ia]) * float(z_s[ia] ** 2) * float(
                    z_s[ib] ** 2
                )
                species_factor2 *= float(t_hats[ib] * m_hats[ia]) / float(t_hats[ia] * m_hats[ib])
                species_factor2 /= float(t_hats[ia] * math.sqrt(t_hats[ia] * m_hats[ia]))

                temp_h = np.zeros((n_x, n_x), dtype=np.float64)
                temp_dh = np.zeros((n_x, n_x), dtype=np.float64)
                temp_d2g = np.zeros((n_x, n_x), dtype=np.float64)

                for ix in range(n_x):
                    xb = float(x[ix] * species_factor)
                    xb_safe = xb if xb > 0 else 1e-14
                    xb_pow_l = xb_safe**l
                    xb_pow_lm1 = xb_safe ** (l - 1) if l >= 1 else xb_safe ** (-1)
                    xb_pow_lm2 = xb_safe ** (l - 2) if l >= 2 else xb_safe ** (-2)

                    for j in range(1, n_x + 1):
                        coeff = poly_coeffs[j - 1]

                        def poly_int_lower(base_power: int) -> float:
                            acc = 0.0
                            for m, cm in enumerate(coeff):
                                acc += float(cm) * _monomial_int_lower(xb_safe, base_power + m)
                            return float(acc)

                        def poly_int_upper(base_power: int) -> float:
                            acc = 0.0
                            for m, cm in enumerate(coeff):
                                acc += float(cm) * _monomial_int_upper(xb_safe, base_power + m)
                            return float(acc)

                        i_2pl = poly_int_lower(l + 2)
                        i_4pl = poly_int_lower(l + 4)
                        i_1ml = poly_int_upper(1 - l)
                        i_3ml = poly_int_upper(3 - l)

                        temp_h[ix, j - 1] = (4.0 * pi / denom_h) * (
                            i_2pl / (xb_safe ** (l + 1)) + xb_pow_l * i_1ml
                        )
                        temp_dh[ix, j - 1] = (4.0 * pi / denom_h) * (
                            -(l + 1) * i_2pl / (xb_safe ** (l + 2)) + l * xb_pow_lm1 * i_1ml
                        )
                        temp_d2g[ix, j - 1] = (-4.0 * pi / denom_g) * (
                            l * (l - 1) * xb_pow_lm2 * i_3ml
                            + alpha * (l + 1) * (l + 2) * xb_pow_l * i_1ml
                            + alpha * (l + 1) * (l + 2) * i_4pl / (xb_safe ** (l + 3))
                            + l * (l - 1) * i_2pl / (xb_safe ** (l + 1))
                        )

                temp_combined = np.zeros((n_x, n_x), dtype=np.float64)
                mass_ratio = float(m_hats[ia] / m_hats[ib])
                for i in range(n_x):
                    xb = float(x[i] * species_factor)
                    temp_combined[i, :] = species_factor2 * expx2[i] * (
                        -temp_h[i, :]
                        - (1.0 - mass_ratio) * xb * temp_dh[i, :]
                        + float(x[i] * x[i]) * temp_d2g[i, :]
                    )

                terms[ia, ib, l, :, :] = temp_combined @ collocation2modal

    return terms


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


@jtu.register_pytree_node_class
@dataclass(frozen=True)
class FokkerPlanckV3Operator:
    """Full linearized v3 Fokker-Planck collision operator (no Phi1).

    Notes
    -----
    - Matches `collisionOperator = 0` in v3 `populateMatrix.F90` for the "original code"
      branch without Phi1 variations.
    - The operator is diagonal in (theta, zeta) and in Legendre index L, but dense in x
      and can couple multiple species.
    """

    mat: jnp.ndarray  # (S,S,L,X,X) already multiplied by (-nu_n)
    n_xi_for_x: jnp.ndarray  # (X,) int32

    def tree_flatten(self):
        children = (self.mat, self.n_xi_for_x)
        aux = None
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        del aux
        mat, n_xi_for_x = children
        return cls(mat=mat, n_xi_for_x=n_xi_for_x)


def make_fokker_planck_v3_operator(
    *,
    x: np.ndarray,  # (X,)
    x_weights: np.ndarray,  # (X,) dx weights
    ddx: np.ndarray,  # (X,X)
    d2dx2: np.ndarray,  # (X,X)
    x_grid_k: float,
    z_s: np.ndarray,  # (S,)
    m_hats: np.ndarray,  # (S,)
    n_hats: np.ndarray,  # (S,)
    t_hats: np.ndarray,  # (S,)
    nu_n: float,
    krook: float,
    n_xi: int,
    nl: int,
    n_xi_for_x: np.ndarray,
) -> FokkerPlanckV3Operator:
    """Construct the collisionOperator=0 (no-Phi1) v3 collision operator."""
    x = np.asarray(x, dtype=np.float64)
    x_weights = np.asarray(x_weights, dtype=np.float64)
    ddx = np.asarray(ddx, dtype=np.float64)
    d2dx2 = np.asarray(d2dx2, dtype=np.float64)
    z_s = np.asarray(z_s, dtype=np.float64)
    m_hats = np.asarray(m_hats, dtype=np.float64)
    n_hats = np.asarray(n_hats, dtype=np.float64)
    t_hats = np.asarray(t_hats, dtype=np.float64)
    n_xi_for_x = np.asarray(n_xi_for_x, dtype=np.int32)

    n_species = int(z_s.size)
    n_x = int(x.size)
    sqrt_pi = float(np.sqrt(np.pi))
    expx2 = np.exp(-(x * x))
    x2 = x * x
    x3 = x2 * x

    # Precompute the Rosenbluth response matrices (new scheme used for xGridScheme=5/6).
    xg = make_x_grid(n=n_x, k=float(x_grid_k), include_point_at_x0=False)
    rosen = rosenbluth_potential_terms_v3_np(
        x=x,
        x_weights=x_weights,
        x_grid_k=float(x_grid_k),
        xg=xg,
        z_s=z_s,
        m_hats=m_hats,
        n_hats=n_hats,
        t_hats=t_hats,
        nl=int(nl),
    )  # (S,S,NL,X,X)

    # Build nuDHat and CECD (both omit the overall factor nu_n, matching v3).
    nu_d_hat = np.zeros((n_species, n_x), dtype=np.float64)
    cecd = np.zeros((n_species, n_species, n_x, n_x), dtype=np.float64)

    for ia in range(n_species):
        t32m = float(t_hats[ia]) * math.sqrt(float(t_hats[ia]) * float(m_hats[ia]))
        for ib in range(n_species):
            species_factor = float(
                math.sqrt((t_hats[ia] * m_hats[ib]) / (t_hats[ib] * m_hats[ia]))
            )
            xb = x * species_factor
            expxb2 = np.exp(-(xb * xb))
            erfs = sp_special.erf(xb)
            psi = (erfs - (2.0 / sqrt_pi) * xb * expxb2) / (2.0 * xb * xb)

            # nuDHat: uses base x-grid x^3 in the denominator (matching Fortran).
            nu_d_hat[ia, :] += (3.0 * sqrt_pi / 4.0) / t32m * float(z_s[ia] ** 2) * float(
                z_s[ib] ** 2
            ) * float(n_hats[ib]) * (erfs - psi) / x3

            # Interpolate species-B f(x_b) onto the species-A x grid.
            if ia == ib:
                f_to_f = np.eye(n_x, dtype=np.float64)
            else:
                alpxk = expx2 * (x**float(x_grid_k))
                alpx = expxb2 * (xb**float(x_grid_k))
                f_to_f = polynomial_interpolation_matrix_np(xk=x, x=xb, alpxk=alpxk, alpx=alpx)

            # CD: field term independent of Rosenbluth potentials (dense in species).
            species_factor_cd = (
                3.0
                * float(n_hats[ia])
                * float(m_hats[ia] / m_hats[ib])
                * float(z_s[ia] ** 2)
                * float(z_s[ib] ** 2)
                / t32m
            )
            cecd[ia, ib, :, :] += (species_factor_cd * expx2)[:, None] * f_to_f

            # CE: energy scattering (diagonal in species indices, but depends on species B).
            species_factor_ce = (
                3.0
                * sqrt_pi
                / 4.0
                * float(n_hats[ib])
                * float(z_s[ia] ** 2)
                * float(z_s[ib] ** 2)
                / t32m
            )
            coef_d2 = (psi / x)[:, None] * d2dx2
            coef_dx = (
                (
                    -2.0
                    * float(t_hats[ia] * m_hats[ib] / (t_hats[ib] * m_hats[ia]))
                    * psi
                    * (1.0 - float(m_hats[ia] / m_hats[ib]))
                    + (erfs - psi) / x2
                )[:, None]
                * ddx
            )
            cecd[ia, ia, :, :] += species_factor_ce * (coef_d2 + coef_dx)

            diag_extra = (
                species_factor_ce
                * 4.0
                / sqrt_pi
                * float(t_hats[ia] / t_hats[ib])
                * math.sqrt(float(t_hats[ia] * m_hats[ib] / (t_hats[ib] * m_hats[ia])))
                * expxb2
            )
            cecd[ia, ia, range(n_x), range(n_x)] += diag_extra

    # Assemble per-L matrices and include the overall (-nu_n) factor to match the PETSc Jacobian entries.
    mat = np.zeros((n_species, n_species, int(n_xi), n_x, n_x), dtype=np.float64)
    for l in range(int(n_xi)):
        m11 = cecd.copy()
        diag = -0.5 * nu_d_hat * (float(l * (l + 1)) + 2.0 * float(krook))
        for s in range(n_species):
            m11[s, s, range(n_x), range(n_x)] += diag[s, :]
        if l < int(nl):
            m11 = m11 + rosen[:, :, l, :, :]
        mat[:, :, l, :, :] = -float(nu_n) * m11

    return FokkerPlanckV3Operator(
        mat=jnp.asarray(mat),
        n_xi_for_x=jnp.asarray(n_xi_for_x, dtype=jnp.int32),
    )


def apply_fokker_planck_v3(op: FokkerPlanckV3Operator, f: jnp.ndarray) -> jnp.ndarray:
    """Apply the v3 `collisionOperator=0` collision operator to `f` (no Phi1)."""
    if f.ndim != 5:
        raise ValueError("f must have shape (Nspecies, Nx, Nxi, Ntheta, Nzeta)")
    n_species, n_x, n_xi, _, _ = f.shape
    if op.mat.shape != (n_species, n_species, n_xi, n_x, n_x):
        raise ValueError(f"op.mat has shape {op.mat.shape}, expected {(n_species, n_species, n_xi, n_x, n_x)}")

    # Compute: y[a,i,l,t,z] = Σ_{b,j} mat[a,b,l,i,j] * f[b,j,l,t,z].
    f2 = jnp.transpose(f, (0, 2, 1, 3, 4))  # (S,L,X,T,Z)
    y2 = jnp.einsum("abLij,bLjtz->aLitz", op.mat, f2)  # (S,L,X,T,Z)
    y = jnp.transpose(y2, (0, 2, 1, 3, 4))  # (S,X,L,T,Z)

    mask = _mask_xi(op.n_xi_for_x.astype(jnp.int32), n_xi).astype(y.dtype)  # (X,L)
    return y * mask[None, :, :, None, None]


apply_fokker_planck_v3_jit = jax.jit(apply_fokker_planck_v3, static_argnums=())
