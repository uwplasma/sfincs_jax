"""Collision physics kernels used by profile-response and transport solves."""

from __future__ import annotations

import math
from collections.abc import Callable
import os
from dataclasses import dataclass, replace

# The JAX backend is imported below; dkx/runtime.py explains why this is here.
from .runtime import configure as _configure_runtime

_configure_runtime()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax import tree_util as jtu  # noqa: E402
from jax.scipy.special import erf  # noqa: E402
from scipy import special as sp_special  # noqa: E402
from scipy.integrate import quad  # noqa: E402

from dkx.xgrid import XGrid, make_x_grid  # noqa: E402

_V3_PI = 3.14159265358979
_V3_SQRTPI = 1.77245385090552

def _erf_np(x: np.ndarray) -> np.ndarray:
    """Use libm-based erf for closer parity with Fortran's intrinsic."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    vec = np.vectorize(math.erf, otypes=[np.float64])
    return vec(x)

def _psi_chandra(
    x: jnp.ndarray, *, series_threshold: float = 1e-5, sqrt_pi: float = _V3_SQRTPI,
) -> jnp.ndarray:
    """Chandrasekhar function Ψ(x).

    Matches the definition used in SFINCS v3 Fortran (`populateMatrix.F90`):

      Ψ = (erf(x) - (2/sqrt(pi)) x exp(-x^2)) / (2 x^2)
    """
    x = x.astype(jnp.float64)
    sqrt_pi = jnp.asarray(sqrt_pi, dtype=jnp.float64)
    num = erf(x) - (2.0 / sqrt_pi) * x * jnp.exp(-(x * x))
    den = 2.0 * x * x
    # Avoid NaNs at x=0 (not typically used with v3 default x grids, but keep robust).
    eps = jnp.asarray(series_threshold, dtype=jnp.float64)
    small = jnp.abs(x) < eps
    # Series after cancellation:
    # Ψ(x) = x/sqrt(pi) sum_n 2(-x²)^n / (n! (2n+3)).
    # Through x^9 the relative truncation error is <2e-16 for |x|<0.05.
    x2 = x * x
    series = x/sqrt_pi * (2/3 + x2*(-2/5 + x2*(1/7 + x2*(-1/27 + x2/132))))
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

    prefac = (3.0 * jnp.asarray(_V3_SQRTPI, dtype=jnp.float64) / 4.0) / t32m  # (S,)
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

    # Mirror v3 Fortran (polynomialInterpolationMatrix.F90) with explicit loops
    # to reduce rounding-order differences in strict parity tests.
    d = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            d[i, j] = xk[i] - xk[j]
    for i in range(n):
        d[i, i] = 1.0

    w = np.zeros((n,), dtype=np.float64)
    for j in range(n):
        prod = 1.0
        for i in range(n):
            prod *= d[i, j]
        w[j] = 1.0 / prod

    mat = np.zeros((m, n), dtype=np.float64)
    for i in range(m):
        for j in range(n):
            dx = x[i] - xk[j]
            if dx == 0.0:
                dx = 1e-15
            mat[i, j] = 1.0 / dx

    for i in range(m):
        denom = 0.0
        for j in range(n):
            denom += mat[i, j] * w[j]
        factor = alpx[i] / denom
        for j in range(n):
            mat[i, j] *= factor

    for j in range(n):
        factor = w[j] / alpxk[j]
        for i in range(m):
            mat[i, j] *= factor
    return mat

def _poly_coeffs_monomial(xg: XGrid) -> list[np.ndarray]:
    """Return monomial coefficients for the orthogonal polynomials used by v3 `xGrid`.

    The returned list is 0-based: `coeffs[j-1]` corresponds to the Fortran polynomial index `j`.
    Coefficients are in ascending powers, i.e. `p(x) = sum_m coeff[m] * x**m`.
    """
    n = int(xg.x.size)
    # Forming high-order monomial coefficients is intrinsically more poorly
    # conditioned than evaluating the polynomials with their 3-term
    # recurrence.  Keep the coefficient construction in extended precision:
    # these coefficients are only used by the analytic moment integrals below,
    # where several individually large monomial moments can cancel.
    dtype = np.longdouble
    a = np.asarray(xg.poly_a, dtype=dtype)
    b = np.asarray(xg.poly_b, dtype=dtype)

    coeffs: list[np.ndarray] = []
    coeffs.append(np.array([1.0], dtype=dtype))  # j=1
    if n == 1:
        return coeffs

    coeffs.append(np.array([-a[1], 1.0], dtype=dtype))  # j=2
    for j in range(2, n):
        aj = a[j]
        bj = b[j]
        p_j = coeffs[j - 1]
        p_jm1 = coeffs[j - 2]

        # (x - aj) * p_j
        out = np.zeros((p_j.size + 1,), dtype=dtype)
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
    """∫_xb^∞ t^n e^{-t^2} dt for any integer ``n`` used by v3.

    The result is ``Gamma((n+1)/2, xb**2) / 2``.  SciPy's regularized
    incomplete-gamma functions do not accept non-positive shape parameters,
    so continue the upper incomplete gamma downward from ``a=0`` (integer
    branch) or ``a=1/2`` (half-integer branch).  This removes the last
    QUADPACK calls from the analytic Rosenbluth path, including the sharply
    peaked negative-power integrals at electron/ion mass ratios.
    """
    if n >= 0:
        a = 0.5 * (n + 1.0)
        return float(0.5 * sp_special.gamma(a) * sp_special.gammaincc(a, xb * xb))

    x2 = float(xb * xb)
    target = 0.5 * (n + 1.0)
    if target < 0.0 and x2 >= 1.0:
        # Downward recurrence subtracts two nearly equal exponentially small
        # numbers when x is large.  Lentz's continued fraction evaluates the
        # *unregularized* Gamma(a,x) directly and remains well scaled there.
        tiny = 1e-300
        b_cf = x2 + 1.0 - target
        c_cf = 1.0 / tiny
        d_cf = 1.0 / b_cf
        h_cf = d_cf
        for iteration in range(1, 10001):
            an = -float(iteration) * (float(iteration) - target)
            b_cf += 2.0
            d_cf = an * d_cf + b_cf
            if abs(d_cf) < tiny:
                d_cf = tiny
            c_cf = b_cf + an / c_cf
            if abs(c_cf) < tiny:
                c_cf = tiny
            d_cf = 1.0 / d_cf
            update = d_cf * c_cf
            h_cf *= update
            if abs(update - 1.0) < 2e-15:
                break
        upper_gamma = math.exp(-x2 + target * math.log(x2)) * h_cf
        return float(0.5 * upper_gamma)

    if n % 2:
        # Odd n -> integer target.  Gamma(0, x) = E1(x).
        current = 0.0
        upper_gamma = float(sp_special.exp1(x2))
    else:
        # Even n -> half-integer target.  Start from Gamma(1/2, x).
        current = 0.5
        upper_gamma = float(math.sqrt(math.pi) * sp_special.gammaincc(0.5, x2))

    exp_minus_x2 = math.exp(-x2)
    while current > target:
        next_a = current - 1.0
        # Gamma(a+1,x) = a Gamma(a,x) + x^a exp(-x).
        upper_gamma = (upper_gamma - (x2**next_a) * exp_minus_x2) / next_a
        current = next_a
    return float(0.5 * upper_gamma)

def _evaluate_polynomial_v3(x: float, *, j: int, a: np.ndarray, b: np.ndarray) -> float:
    """Evaluate v3's orthogonal polynomial p_j(x) using the 3-term recurrence.

    Mirrors `xGrid.F90:evaluatePolynomial()`. Index `j` is 1-based.
    """
    if j == 1:
        return 1.0
    pj_minus1 = 0.0
    pj = 1.0
    y = 0.0
    for ii in range(1, j):
        y = (x - float(a[ii])) * pj - float(b[ii]) * pj_minus1
        pj_minus1, pj = pj, y
    return float(y)

def _rosenbluth_potential_terms_v3_np_quadpack(
    *,
    x: np.ndarray,
    x_weights: np.ndarray,
    x_grid_k: float,
    xg: XGrid,
    z_s: np.ndarray,
    m_hats: np.ndarray,
    n_hats: np.ndarray,
    t_hats: np.ndarray,
    nl: int,
) -> np.ndarray:
    """Quadpack-based Rosenbluth response matrices matching v3 `xGrid.F90`.

    This implementation intentionally mirrors the upstream Fortran algorithm:
    - Polynomials are evaluated via the 3-term recurrence (not monomial expansion).
    - All required integrals are evaluated using QUADPACK (`scipy.integrate.quad`) with
      epsabs=epsrel=1e-13 and an upper split at `partition=max(10, 2*xb)` for semi-infinite
      integrals, matching `xGrid.F90`.

    Notes
    -----
    This function is not JAX-differentiable (SciPy). It is used to precompute the
    linearized v3 Fokker-Planck collision operator coefficients.
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
    a = np.asarray(xg.poly_a, dtype=np.float64)
    b = np.asarray(xg.poly_b, dtype=np.float64)
    poly_c = np.asarray(xg.poly_c, dtype=np.float64)

    # collocation2modal(j,i) in the Fortran code:
    pvals = np.zeros((n_x, n_x), dtype=np.float64)  # (j,i)
    for j in range(1, n_x + 1):
        for i in range(n_x):
            pvals[j - 1, i] = _evaluate_polynomial_v3(float(x[i]), j=j, a=a, b=b)
    collocation2modal = (x_weights[None, :] * (x[None, :] ** float(x_grid_k)) * pvals) / (
        poly_c[1 : n_x + 1, None]
    )

    pi = float(_V3_PI)
    epsabs = 1e-13
    epsrel = 1e-13
    limit = 5000

    terms = np.zeros((n_species, n_species, int(nl), n_x, n_x), dtype=np.float64)
    for ell in range(int(nl)):
        alpha = -float(2 * ell - 1) / float(2 * ell + 3)
        denom_h = float(2 * ell + 1)
        denom_g = float(4 * ell * ell - 1)

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

                    # For semi-infinite integrals, v3 splits at `partition=max(10,2*xb)`.
                    partition = float(max(10.0, 2.0 * xb_safe))

                    for j in range(1, n_x + 1):
                        def poly(t: float) -> float:
                            return _evaluate_polynomial_v3(t, j=j, a=a, b=b)

                        def integrand(t: float, power: int) -> float:
                            # Note: do NOT include t**xGrid_k here; v3 excludes it in these integrals.
                            return (t**power) * poly(t) * math.exp(-(t * t))

                        def quad_finite(power: int, a0: float, b0: float) -> float:
                            val, _ = quad(
                                lambda tt: integrand(tt, power),
                                a0,
                                b0,
                                epsabs=epsabs,
                                epsrel=epsrel,
                                limit=limit,
                            )
                            return float(val)

                        def quad_semiinf(power: int, a0: float) -> float:
                            val, _ = quad(
                                lambda tt: integrand(tt, power),
                                a0,
                                np.inf,
                                epsabs=epsabs,
                                epsrel=epsrel,
                                limit=limit,
                            )
                            return float(val)

                        i_2pl = quad_finite(ell + 2, 0.0, xb_safe)
                        i_4pl = quad_finite(ell + 4, 0.0, xb_safe)

                        i_1ml = quad_finite(1 - ell, xb_safe, partition) + quad_semiinf(1 - ell, partition)
                        i_3ml = quad_finite(3 - ell, xb_safe, partition) + quad_semiinf(3 - ell, partition)

                        xb_pow_l = xb_safe**ell
                        xb_pow_lm1 = xb_safe ** (ell - 1) if ell >= 1 else xb_safe ** (-1)
                        xb_pow_lm2 = xb_safe ** (ell - 2) if ell >= 2 else xb_safe ** (-2)

                        temp_h[ix, j - 1] = (4.0 * pi / denom_h) * (
                            i_2pl / (xb_safe ** (ell + 1)) + xb_pow_l * i_1ml
                        )
                        temp_dh[ix, j - 1] = (4.0 * pi / denom_h) * (
                            -(ell + 1) * i_2pl / (xb_safe ** (ell + 2)) + ell * xb_pow_lm1 * i_1ml
                        )
                        temp_d2g[ix, j - 1] = (-4.0 * pi / denom_g) * (
                            ell * (ell - 1) * xb_pow_lm2 * i_3ml
                            + alpha * (ell + 1) * (ell + 2) * xb_pow_l * i_1ml
                            + alpha * (ell + 1) * (ell + 2) * i_4pl / (xb_safe ** (ell + 3))
                            + ell * (ell - 1) * i_2pl / (xb_safe ** (ell + 1))
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

                terms[ia, ib, ell, :, :] = temp_combined @ collocation2modal

    return terms

ROSENBLUTH_METHODS: tuple[str, ...] = ("quadpack", "analytic", "hybrid")
"""Accepted ``rosenbluth_method`` values, in order of increasing Fortran divergence."""

def resolve_rosenbluth_method(method: str | None) -> str:
    """Normalize and validate a Rosenbluth-quadrature selector.

    ``method`` is the explicit route — the ``rosenbluth_method=`` argument of
    the collision-operator builders, which the ``RosenbluthMethod`` key of
    ``&otherNumericalParameters`` feeds.  Only when it is ``None`` does the
    ``DKX_ROSENBLUTH_METHOD`` environment variable act as an override, and
    with neither set the default is ``"quadpack"`` — the upstream Fortran v3
    algorithm, for parity.

    Raises:
        ValueError: for any selector outside :data:`ROSENBLUTH_METHODS`, so a
            mistyped namelist key or environment override fails loudly instead
            of silently falling back to the default.
    """
    raw = method
    if raw is None:
        raw = os.environ.get("DKX_ROSENBLUTH_METHOD", "") or None
    if raw is None:
        return "quadpack"
    resolved = str(raw).strip().lower()
    if not resolved:
        return "quadpack"
    if resolved not in ROSENBLUTH_METHODS:
        raise ValueError(
            f"Unknown RosenbluthPotentialTerms method={raw!r}. "
            f"Use one of {', '.join(repr(m) for m in ROSENBLUTH_METHODS)} "
            "(namelist RosenbluthMethod, the rosenbluth_method= builder "
            "argument, or the DKX_ROSENBLUTH_METHOD override)."
        )
    return resolved

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
    method: str | None = None,
) -> np.ndarray:
    """Compute v3 `RosenbluthPotentialTerms` for xGridScheme=5/6 (new scheme).

    Args:
        method: one of :data:`ROSENBLUTH_METHODS`, resolved by
            :func:`resolve_rosenbluth_method` (``None`` consults
            ``DKX_ROSENBLUTH_METHOD`` and then falls back to ``"quadpack"``).

    Returns
    -------
    terms:
      Array of shape (S, S, NL, X, X) with index ordering:
      (species_row, species_col, L, x_row, x_col).
    """
    method = resolve_rosenbluth_method(method)

    if method == "quadpack":
        return _rosenbluth_potential_terms_v3_np_quadpack(
            x=x,
            x_weights=x_weights,
            x_grid_k=x_grid_k,
            xg=xg,
            z_s=z_s,
            m_hats=m_hats,
            n_hats=n_hats,
            t_hats=t_hats,
            nl=nl,
        )
    if method == "hybrid":
        # L=0 and L=1 own the particle/energy and parallel-momentum
        # conservation moments.  Keep the warning-free low-L QUADPACK values
        # (and L=2,3 for a small parity margin), then use closed-form moments
        # for the high-L blocks in which near-zero cancellation and negative
        # powers make the fixed 1e-13 QUADPACK tolerance ill-conditioned.
        terms = rosenbluth_potential_terms_v3_np(
            x=x,
            x_weights=x_weights,
            x_grid_k=x_grid_k,
            xg=xg,
            z_s=z_s,
            m_hats=m_hats,
            n_hats=n_hats,
            t_hats=t_hats,
            nl=nl,
            method="analytic",
        )
        nl_quadpack = min(int(nl), 4)
        if nl_quadpack:
            terms[:, :, :nl_quadpack, :, :] = _rosenbluth_potential_terms_v3_np_quadpack(
                x=x,
                x_weights=x_weights,
                x_grid_k=x_grid_k,
                xg=xg,
                z_s=z_s,
                m_hats=m_hats,
                n_hats=n_hats,
                t_hats=t_hats,
                nl=nl_quadpack,
            )
        return terms
    if method != "analytic":
        # Unreachable through resolve_rosenbluth_method; a structural guard so
        # a new entry in ROSENBLUTH_METHODS without a branch here fails loudly
        # rather than falling through to the analytic path.
        raise ValueError(f"RosenbluthPotentialTerms method={method!r} has no branch.")

    x = np.asarray(x, dtype=np.float64)
    x_weights = np.asarray(x_weights, dtype=np.float64)
    z_s = np.asarray(z_s, dtype=np.float64)
    m_hats = np.asarray(m_hats, dtype=np.float64)
    n_hats = np.asarray(n_hats, dtype=np.float64)
    t_hats = np.asarray(t_hats, dtype=np.float64)

    n_x = int(x.size)
    n_species = int(z_s.size)
    expx2 = np.exp(-(x * x))

    # collocation2modal(j,i) in the Fortran code.  Use the stable 3-term
    # recurrence for point evaluation; monomial coefficients are reserved for
    # the analytic integrals below.
    poly_coeffs = _poly_coeffs_monomial(xg)
    poly_c = np.asarray(xg.poly_c, dtype=np.float64)
    poly_a = np.asarray(xg.poly_a, dtype=np.float64)
    poly_b = np.asarray(xg.poly_b, dtype=np.float64)
    pvals = np.zeros((n_x, n_x), dtype=np.float64)  # (j,i)
    for j in range(1, n_x + 1):
        for i in range(n_x):
            pvals[j - 1, i] = _evaluate_polynomial_v3(float(x[i]), j=j, a=poly_a, b=poly_b)
    collocation2modal = (x_weights[None, :] * (x[None, :] ** float(x_grid_k)) * pvals) / (
        poly_c[1 : n_x + 1, None]
    )

    terms = np.zeros((n_species, n_species, int(nl), n_x, n_x), dtype=np.float64)
    pi = float(_V3_PI)

    for ell in range(int(nl)):
        alpha = -float(2 * ell - 1) / float(2 * ell + 3)
        denom_h = float(2 * ell + 1)
        denom_g = float(4 * ell * ell - 1)
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
                    xb_pow_l = xb_safe**ell
                    xb_pow_lm1 = xb_safe ** (ell - 1) if ell >= 1 else xb_safe ** (-1)
                    xb_pow_lm2 = xb_safe ** (ell - 2) if ell >= 2 else xb_safe ** (-2)

                    for j in range(1, n_x + 1):
                        coeff = poly_coeffs[j - 1]

                        def poly_int_lower(base_power: int) -> float:
                            moments = np.asarray(
                                [_monomial_int_lower(xb_safe, base_power + m) for m in range(coeff.size)],
                                dtype=np.longdouble,
                            )
                            return float(np.sum(coeff * moments, dtype=np.longdouble))

                        def poly_int_upper(base_power: int) -> float:
                            moments = np.asarray(
                                [_monomial_int_upper(xb_safe, base_power + m) for m in range(coeff.size)],
                                dtype=np.longdouble,
                            )
                            return float(np.sum(coeff * moments, dtype=np.longdouble))

                        i_2pl = poly_int_lower(ell + 2)
                        i_4pl = poly_int_lower(ell + 4)
                        i_1ml = poly_int_upper(1 - ell)
                        i_3ml = poly_int_upper(3 - ell)

                        temp_h[ix, j - 1] = (4.0 * pi / denom_h) * (
                            i_2pl / (xb_safe ** (ell + 1)) + xb_pow_l * i_1ml
                        )
                        temp_dh[ix, j - 1] = (4.0 * pi / denom_h) * (
                            -(ell + 1) * i_2pl / (xb_safe ** (ell + 2)) + ell * xb_pow_lm1 * i_1ml
                        )
                        temp_d2g[ix, j - 1] = (-4.0 * pi / denom_g) * (
                            ell * (ell - 1) * xb_pow_lm2 * i_3ml
                            + alpha * (ell + 1) * (ell + 2) * xb_pow_l * i_1ml
                            + alpha * (ell + 1) * (ell + 2) * i_4pl / (xb_safe ** (ell + 3))
                            + ell * (ell - 1) * i_2pl / (xb_safe ** (ell + 1))
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

                terms[ia, ib, ell, :, :] = temp_combined @ collocation2modal

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
    coef: jnp.ndarray  # (S, X, L)
    mask_xi: jnp.ndarray  # (X, L)

    def tree_flatten(self):
        children = (self.nu_n, self.krook, self.nu_d_hat, self.n_xi_for_x, self.coef, self.mask_xi)
        aux = None
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        del aux
        nu_n, krook, nu_d_hat, n_xi_for_x, coef, mask_xi = children
        return cls(
            nu_n=nu_n,
            krook=krook,
            nu_d_hat=nu_d_hat,
            n_xi_for_x=n_xi_for_x,
            coef=coef,
            mask_xi=mask_xi,
        )

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
    n_xi: int,
) -> PitchAngleScatteringV3Operator:
    nu_d_hat = nu_d_hat_pitch_angle_scattering_v3(
        x=x, z_s=z_s, m_hats=m_hats, n_hats=n_hats, t_hats=t_hats
    )
    n_xi_int = int(n_xi)
    ell = jnp.arange(n_xi_int, dtype=jnp.float64)
    factor_l = 0.5 * (ell * (ell + 1.0) + 2.0 * krook)
    coef = jnp.asarray(nu_n, dtype=jnp.float64) * nu_d_hat[:, :, None] * factor_l[None, None, :]
    mask = _mask_xi(jnp.asarray(n_xi_for_x, dtype=jnp.int32), n_xi_int)
    return PitchAngleScatteringV3Operator(
        nu_n=jnp.asarray(nu_n, dtype=jnp.float64),
        krook=jnp.asarray(krook, dtype=jnp.float64),
        nu_d_hat=nu_d_hat,
        n_xi_for_x=jnp.asarray(n_xi_for_x, dtype=jnp.int32),
        coef=coef,
        mask_xi=mask,
    )

def _mask_xi(n_xi_for_x: jnp.ndarray, n_xi_max: int) -> jnp.ndarray:
    ell = jnp.arange(n_xi_max, dtype=jnp.int32)[None, :]
    return ell < n_xi_for_x[:, None]

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

    if op.coef.shape[-1] != n_xi:
        ell = jnp.arange(n_xi, dtype=jnp.float64)  # row L
        factor_l = 0.5 * (ell * (ell + 1.0) + 2.0 * op.krook)  # (L,)
        coef = op.nu_n * op.nu_d_hat[:, :, None] * factor_l[None, None, :]  # (S,X,L)
        mask = _mask_xi(op.n_xi_for_x.astype(jnp.int32), n_xi).astype(coef.dtype)  # (X,L)
    else:
        coef = op.coef
        mask = op.mask_xi.astype(coef.dtype)

    out = coef[:, :, :, None, None] * f
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
    mask_xi: jnp.ndarray  # (X, L)

    def tree_flatten(self):
        children = (self.mat, self.n_xi_for_x, self.mask_xi)
        aux = None
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        del aux
        mat, n_xi_for_x, mask_xi = children
        return cls(mat=mat, n_xi_for_x=n_xi_for_x, mask_xi=mask_xi)

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
    strict_parity: bool = False,
    rosenbluth_method: str | None = None,
) -> FokkerPlanckV3Operator:
    """Construct the collisionOperator=0 (no-Phi1) v3 collision operator.

    Args:
        rosenbluth_method: how the Rosenbluth potential response matrices are
            evaluated — one of :data:`ROSENBLUTH_METHODS`, resolved by
            :func:`resolve_rosenbluth_method`.  ``None`` keeps the Fortran-parity
            ``"quadpack"`` default unless ``DKX_ROSENBLUTH_METHOD`` overrides it.
    """
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
    sqrt_pi = float(_V3_SQRTPI)
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
        method=rosenbluth_method,
    )  # (S,S,NL,X,X)

    # Build nuDHat and CECD (both omit the overall factor nu_n, matching v3).
    nu_d_hat = np.zeros((n_species, n_x), dtype=np.float64)
    cecd = np.zeros((n_species, n_species, n_x, n_x), dtype=np.float64)

    strict_fp = bool(strict_parity)

    for ia in range(n_species):
        t32m = float(t_hats[ia]) * math.sqrt(float(t_hats[ia]) * float(m_hats[ia]))
        for ib in range(n_species):
            species_factor = float(
                math.sqrt((t_hats[ia] * m_hats[ib]) / (t_hats[ib] * m_hats[ia]))
            )
            xb = x * species_factor
            if strict_fp:
                expxb2 = np.empty((n_x,), dtype=np.float64)
                erfs = np.empty((n_x,), dtype=np.float64)
                psi = np.empty((n_x,), dtype=np.float64)
                for ix in range(n_x):
                    xb_val = float(xb[ix])
                    exp_val = math.exp(-(xb_val * xb_val))
                    erf_val = math.erf(xb_val)
                    expxb2[ix] = exp_val
                    erfs[ix] = erf_val
                    if abs(xb_val) < 1e-14:
                        psi[ix] = (2.0 / sqrt_pi) * xb_val / 3.0
                    else:
                        psi[ix] = (erf_val - (2.0 / sqrt_pi) * xb_val * exp_val) / (2.0 * xb_val * xb_val)
            else:
                expxb2 = np.exp(-(xb * xb))
                erfs = _erf_np(xb)
                psi = (erfs - (2.0 / sqrt_pi) * xb * expxb2) / (2.0 * xb * xb)

            # nuDHat: uses base x-grid x^3 in the denominator (matching Fortran).
            nu_factor = (3.0 * sqrt_pi / 4.0) / t32m * float(z_s[ia] ** 2) * float(
                z_s[ib] ** 2
            ) * float(n_hats[ib])
            if strict_fp:
                for ix in range(n_x):
                    nu_d_hat[ia, ix] += nu_factor * (erfs[ix] - psi[ix]) / x3[ix]
            else:
                nu_d_hat[ia, :] += nu_factor * (erfs - psi) / x3

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
            if strict_fp:
                for ix in range(n_x):
                    for jx in range(n_x):
                        cecd[ia, ib, ix, jx] += species_factor_cd * expx2[ix] * f_to_f[ix, jx]
            else:
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
            if strict_fp:
                for ix in range(n_x):
                    for jx in range(n_x):
                        cecd[ia, ia, ix, jx] += species_factor_ce * (coef_d2[ix, jx] + coef_dx[ix, jx])
            else:
                cecd[ia, ia, :, :] += species_factor_ce * (coef_d2 + coef_dx)

            diag_extra = (
                species_factor_ce
                * 4.0
                / sqrt_pi
                * float(t_hats[ia] / t_hats[ib])
                * math.sqrt(float(t_hats[ia] * m_hats[ib] / (t_hats[ib] * m_hats[ia])))
                * expxb2
            )
            if strict_fp:
                for ix in range(n_x):
                    cecd[ia, ia, ix, ix] += diag_extra[ix]
            else:
                cecd[ia, ia, range(n_x), range(n_x)] += diag_extra

    # Assemble per-L matrices and include the overall (-nu_n) factor to match the PETSc Jacobian entries.
    mat = np.zeros((n_species, n_species, int(n_xi), n_x, n_x), dtype=np.float64)
    for ell in range(int(n_xi)):
        m11 = cecd.copy()
        diag = -0.5 * nu_d_hat * (float(ell * (ell + 1)) + 2.0 * float(krook))
        for s in range(n_species):
            m11[s, s, range(n_x), range(n_x)] += diag[s, :]
        if ell < int(nl):
            m11 = m11 + rosen[:, :, ell, :, :]
        mat[:, :, ell, :, :] = -float(nu_n) * m11

    return FokkerPlanckV3Operator(
        mat=jnp.asarray(mat),
        n_xi_for_x=jnp.asarray(n_xi_for_x, dtype=jnp.int32),
        mask_xi=_mask_xi(jnp.asarray(n_xi_for_x, dtype=jnp.int32), int(n_xi)),
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

    if op.mask_xi.shape[-1] != n_xi:
        mask = _mask_xi(op.n_xi_for_x.astype(jnp.int32), n_xi).astype(y.dtype)  # (X,L)
    else:
        mask = op.mask_xi.astype(y.dtype)
    return y * mask[None, :, :, None, None]

apply_fokker_planck_v3_jit = jax.jit(apply_fokker_planck_v3, static_argnums=())

@jtu.register_pytree_node_class
@dataclass(frozen=True)
class FokkerPlanckV3Phi1Operator:
    """v3 `collisionOperator=0` collision operator including Phi1 in the collision operator.

    This corresponds to the `includePhi1InCollisionOperator = .true.` branch in v3
    `populateMatrix.F90`, in which the collisional coefficients become poloidally varying
    through the factor `exp(-Z*alpha*Phi1Hat/THat)`.

    Notes
    -----
    - The operator remains diagonal in (theta, zeta) and in Legendre index L, but it is no longer
      uniform over the flux surface.
    - This implementation targets the residual/Jacobian matrices used in parity fixtures
      (notably `whichMatrix=3` in v3), i.e. it treats Phi1 as a frozen background field
      when applying the matrix-free operator.
    """

    nu_n: jnp.ndarray  # scalar
    krook: jnp.ndarray  # scalar
    alpha: jnp.ndarray  # scalar
    z_s: jnp.ndarray  # (S,)
    n_hats: jnp.ndarray  # (S,)
    t_hats: jnp.ndarray  # (S,)
    nl: int

    # Base tensors, independent of densities and Phi1:
    # - nuDHat = sum_b n_pol[b] * k_nu[a,b,x]
    k_nu: jnp.ndarray  # (S,S,X)
    # - CD term: scales with n_pol[a]
    k_cd: jnp.ndarray  # (S,S,X,X)
    # - CE term: contributes to diagonal (a,a) and scales with n_pol[b]
    k_ce: jnp.ndarray  # (S,S,X,X)
    # - Rosenbluth term: scales with n_pol[a]
    k_rosen: jnp.ndarray  # (S,S,NL,X,X)

    n_xi_for_x: jnp.ndarray  # (X,) int32

    def rescale_temperature(self, scale: jnp.ndarray) -> FokkerPlanckV3Phi1Operator:
        """Scale every species temperature by the same positive scalar.

        Species speed ratios and interpolation/Rosenbluth responses are then
        unchanged; all four unit-density kernels scale as ``scale**(-3/2)``.
        This exact fixed-grid update supports JIT and AD, including the changed
        Phi1 Boltzmann response. Masses, charges, densities, ``nu_n`` (including
        any Coulomb logarithm), and normalization stay fixed. Callers must also
        update the kinetic operator temperatures, drives and desired profiles.
        Independent species temperature changes still require a full rebuild.

        Nonpositive or nonfinite scales produce NaNs, including under JIT.
        """
        scale = jnp.asarray(scale, dtype=jnp.float64)
        if scale.shape != ():
            raise ValueError("temperature scale must be scalar")
        scale = jnp.where(jnp.isfinite(scale) & (scale > 0), scale, jnp.nan)
        factor = scale ** -1.5
        return replace(
            self, t_hats=self.t_hats * scale,
            **{name: getattr(self, name) * factor
               for name in ("k_nu", "k_cd", "k_ce", "k_rosen")},
        )

    def at_uniform_density(self, n_hats: jnp.ndarray, *, n_xi: int) -> FokkerPlanckV3Operator:
        """Refresh uniform FP coefficients from the stored unit-density kernels.

        Jittable and differentiable in density at fixed temperatures, masses,
        charges and speed grid. Changing those kernel dependencies requires a
        rebuild; this method does not apply a Phi1 Boltzmann response. ``n_xi``
        is the static rectangular pitch resolution, including truncated rows.
        """
        n = jnp.asarray(n_hats, dtype=jnp.float64)
        if n.shape != self.n_hats.shape:
            raise ValueError(f"n_hats must have shape {self.n_hats.shape}, got {n.shape}")
        if n_xi < 1:
            raise ValueError("n_xi must be positive")
        ns, _, nx = self.k_nu.shape
        ell = jnp.arange(n_xi, dtype=jnp.float64)
        nu_d = jnp.einsum("b,abx->ax", n, self.k_nu)
        ce = jnp.einsum("b,abij->aij", n, self.k_ce)
        diagonal = -0.5 * nu_d[:, None, :] * (ell * (ell + 1) + 2 * self.krook)[None, :, None]
        same_species = ce[:, None, :, :] + diagonal[:, :, :, None] * jnp.eye(nx)
        mat = n[:, None, None, None, None] * self.k_cd[:, :, None, :, :]
        mat = mat + jnp.eye(ns)[:, :, None, None, None] * same_species[:, None, :, :, :]
        nl = min(self.nl, n_xi)
        mat = mat.at[:, :, :nl].add(n[:, None, None, None, None] * self.k_rosen[:, :, :nl])
        return FokkerPlanckV3Operator(
            mat=-self.nu_n * mat, n_xi_for_x=self.n_xi_for_x,
            mask_xi=_mask_xi(self.n_xi_for_x, n_xi),
        )

    def tree_flatten(self):
        children = (
            self.nu_n,
            self.krook,
            self.alpha,
            self.z_s,
            self.n_hats,
            self.t_hats,
            self.k_nu,
            self.k_cd,
            self.k_ce,
            self.k_rosen,
            self.n_xi_for_x,
        )
        aux = int(self.nl)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        (
            nu_n,
            krook,
            alpha,
            z_s,
            n_hats,
            t_hats,
            k_nu,
            k_cd,
            k_ce,
            k_rosen,
            n_xi_for_x,
        ) = children
        return cls(
            nu_n=nu_n,
            krook=krook,
            alpha=alpha,
            z_s=z_s,
            n_hats=n_hats,
            t_hats=t_hats,
            nl=int(aux),
            k_nu=k_nu,
            k_cd=k_cd,
            k_ce=k_ce,
            k_rosen=k_rosen,
            n_xi_for_x=n_xi_for_x,
        )

def prepare_fokker_planck_v3_profiles(
    *, x: np.ndarray, x_weights: np.ndarray, ddx: np.ndarray, d2dx2: np.ndarray,
    x_grid_k: float, z_s: np.ndarray, m_hats: np.ndarray, nu_n: float,
    krook: float, nl: int, n_xi_for_x: np.ndarray, alpha: float = 1.0,
    quadrature_order: int = 128,
) -> Callable[[jnp.ndarray, jnp.ndarray], FokkerPlanckV3Phi1Operator]:
    """Prepare an opt-in JAX full-FP builder ``build(n_hats, t_hats)``.

    Species densities and temperatures may vary independently under JIT/AD.
    Layout, masses, charges, normalization, nu_n/Coulomb logarithm and grids
    stay fixed. The returned kernels support uniform or frozen-Phi1 application.
    This does not update kinetic profiles/drives or certify reusable factors.

    Rosenbluth integrals use composite Gauss-Legendre quadrature with a static
    order per panel, retaining the v3 polynomial recurrence. Callers must check
    order and physical-grid convergence for their parameter domain. This is an
    explicit alternative to the host QUADPACK parity builder, not its default
    replacement. Positive speed nodes (v3 schemes 5/6 without x=0) are required.
    Invalid dynamic profiles produce NaNs; incompatible shapes raise ValueError.
    """
    if isinstance(quadrature_order, bool) or not isinstance(quadrature_order, int) or quadrature_order < 2:
        raise ValueError("quadrature_order must be an integer >= 2")
    if isinstance(nl, bool) or not isinstance(nl, int) or nl < 1:
        raise ValueError("nl must be a positive integer")
    x = np.asarray(x, dtype=np.float64)
    z = np.asarray(z_s, dtype=np.float64)
    m = np.asarray(m_hats, dtype=np.float64)
    nx = x.size
    if x.ndim != 1 or nx < 1 or not np.all(np.isfinite(x) & (x > 0)):
        raise ValueError("x must be a nonempty positive finite vector")
    if z.ndim != 1 or z.size < 1 or m.shape != z.shape or not np.all(np.isfinite(z)) or not np.all(np.isfinite(m) & (m > 0)):
        raise ValueError("species charges/masses must be finite vectors with positive masses and matching shapes")
    if np.shape(x_weights) != (nx,) or np.shape(ddx) != (nx, nx) or np.shape(d2dx2) != (nx, nx) or np.shape(n_xi_for_x) != (nx,):
        raise ValueError("speed weights, derivatives and pitch layout must match x")
    xg = make_x_grid(n=nx, k=float(x_grid_k), include_point_at_x0=False)
    a, b = np.asarray(xg.poly_a), np.asarray(xg.poly_b)
    pvals = np.array([[_evaluate_polynomial_v3(float(t), j=j, a=a, b=b)
                       for t in x] for j in range(1, nx + 1)])
    modal = jnp.asarray(np.asarray(x_weights)[None, :] * x[None, :]**x_grid_k * pvals
                        / np.asarray(xg.poly_c)[1:nx + 1, None])
    # Polynomial interpolation through a nonsingular evaluation basis avoids
    # the barycentric 1/(xb-x_j) removable singularity and its incorrect AD at
    # coincident speed nodes. Normalize modal columns before the static solve.
    norms = np.sqrt(np.asarray(xg.poly_c)[1:nx + 1])
    basis = (np.exp(-x*x) * x**x_grid_k)[:, None] * pvals.T / norms
    inverse_basis = jnp.asarray(np.linalg.solve(basis, np.eye(nx)))
    nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
    u, w = jnp.asarray((nodes + 1)/2), jnp.asarray(weights/2)
    x, z, m = jnp.asarray(x), jnp.asarray(z), jnp.asarray(m)
    ddx, d2dx2 = jnp.asarray(ddx), jnp.asarray(d2dx2)
    layout = jnp.asarray(n_xi_for_x, dtype=jnp.int32)

    def weighted_polynomials(t):
        # Carry exp(-t²) inside the recurrence to avoid huge polynomial values
        # multiplying an underflowed Maxwellian in the semi-infinite tail.
        vals = [jnp.exp(-t*t)]
        if nx > 1:
            vals.append((t - a[1]) * vals[0])
        for j in range(2, nx):
            vals.append((t - a[j]) * vals[-1] - b[j] * vals[-2])
        return jnp.stack(vals, axis=-1)

    def build(n_hats, t_hats):
        n, temp = jnp.asarray(n_hats, dtype=jnp.float64), jnp.asarray(t_hats, dtype=jnp.float64)
        if n.shape != z.shape or temp.shape != z.shape:
            raise ValueError(f"n_hats and t_hats must have shape {z.shape}")
        n = jnp.where(jnp.isfinite(n) & (n >= 0), n, jnp.nan)
        temp = jnp.where(jnp.isfinite(temp) & (temp > 0), temp, jnp.nan)
        factor = jnp.sqrt(temp[:, None] * m[None, :] / (temp[None, :] * m[:, None]))
        xb = x[None, None, :] * factor[:, :, None]
        partition, split = jnp.maximum(10., 2*xb), jnp.minimum(10., xb)
        # Lower: two linear panels. Upper: logarithmic near-field panel
        # resolves small-x electron/ion negative powers, then a rational map
        # to infinity. Panel locations are differentiated along with xb.
        lower = jnp.concatenate([split[..., None]*u, split[..., None] + (xb-split)[..., None]*u], axis=-1)
        lw = jnp.concatenate([split[..., None]*w, (xb-split)[..., None]*w], axis=-1)
        logspan = jnp.log(partition/xb)
        mid = xb[..., None] * jnp.exp(logspan[..., None]*u)
        upper = jnp.concatenate([mid, partition[..., None] + u/(1-u)], axis=-1)
        uw = jnp.concatenate([mid*logspan[..., None]*w, jnp.broadcast_to(w/(1-u)**2, mid.shape)], axis=-1)
        lp, up = weighted_polynomials(lower)*lw[..., None], weighted_polynomials(upper)*uw[..., None]
        lo, hi = lower/xb[..., None], xb[..., None]/upper
        t32m = temp * jnp.sqrt(temp*m)
        prefactor = z[:, None]**2 * z[None, :]**2 / t32m[:, None]
        rosen = []
        for ell in range(nl):
            # These are the four v3 incomplete integrals after absorbing all
            # xb powers: l2=I_(l+2)/xb^(l+1), u1=xb^l I_(1-l),
            # l4=I_(l+4)/xb^(l+3), u3=xb^(l-2) I_(3-l).
            l2 = jnp.einsum("abxq,abxqj->abxj", lo**(ell+2)*xb[..., None], lp)
            l4 = jnp.einsum("abxq,abxqj->abxj", lo**(ell+4)*xb[..., None], lp)
            u1 = jnp.einsum("abxq,abxqj->abxj", hi**ell*upper, up)
            u3 = (jnp.einsum("abxq,abxqj->abxj", hi**(ell-2)*upper, up)
                  if ell >= 2 else jnp.zeros_like(u1))
            ratio = -(2*ell-1)/(2*ell+3)
            h = 4*_V3_PI/(2*ell+1)*(l2+u1)
            xdh = 4*_V3_PI/(2*ell+1)*(-(ell+1)*l2+ell*u1)
            d2g = -4*_V3_PI/(4*ell*ell-1)*(ell*(ell-1)*(u3+l2) + ratio*(ell+1)*(ell+2)*(u1+l4))
            combined = -h - (1-m[:, None, None, None]/m[None, :, None, None])*xdh + x[None, None, :, None]**2*d2g
            sf = 3/(2*_V3_PI) * prefactor / factor**2
            rosen.append(jnp.einsum("abij,jk->abik", sf[:, :, None, None]*jnp.exp(-x*x)[None, None, :, None]*combined, modal))
        interpolation = (weighted_polynomials(xb)*xb[..., None]**x_grid_k/norms) @ inverse_basis
        # Self-collisions have identically coincident speed grids.
        interpolation = jnp.where(jnp.eye(z.size, dtype=bool)[:, :, None, None], jnp.eye(nx), interpolation)
        expb, erfs = jnp.exp(-xb*xb), erf(xb)
        # The host parity route subtracts nearly equal erf/Maxwellian terms
        # using v3's rounded sqrt(pi). At small electron/ion speed this loses
        # accuracy even before device-dependent rounding. This opt-in route
        # uses mathematical sqrt(pi) and a stable series, checked independently.
        sqrt_pi = math.sqrt(math.pi)
        psi = _psi_chandra(xb, series_threshold=.05, sqrt_pi=sqrt_pi)
        ce_factor = (3*sqrt_pi/4)*prefactor[:, :, None]
        k_nu = ce_factor*(erfs-psi)/x**3
        k_cd = (3*prefactor*m[:, None]/m[None, :])[:, :, None, None] * jnp.exp(-x*x)[None, None, :, None]*interpolation
        dx_coeff = -2*factor[:, :, None]**2*psi*(1-m[:, None, None]/m[None, :, None]) + (erfs-psi)/x**2
        k_ce = ce_factor[..., None]*((psi/x)[..., None]*d2dx2 + dx_coeff[..., None]*ddx)
        diag = ce_factor*(4/sqrt_pi)*(temp[:, None]/temp[None, :]*factor)[:, :, None]*expb
        k_ce = k_ce + diag[..., None]*jnp.eye(nx)
        return FokkerPlanckV3Phi1Operator(
            nu_n=jnp.asarray(nu_n), krook=jnp.asarray(krook), alpha=jnp.asarray(alpha),
            z_s=z, n_hats=n, t_hats=temp, nl=nl, k_nu=k_nu, k_cd=k_cd,
            k_ce=k_ce, k_rosen=jnp.stack(rosen, axis=2), n_xi_for_x=layout,
        )

    return build

def make_fokker_planck_v3_phi1_operator(
    *,
    x: np.ndarray,
    x_weights: np.ndarray,
    ddx: np.ndarray,
    d2dx2: np.ndarray,
    x_grid_k: float,
    z_s: np.ndarray,
    m_hats: np.ndarray,
    n_hats: np.ndarray,
    t_hats: np.ndarray,
    nu_n: float,
    krook: float,
    n_xi: int,
    nl: int,
    alpha: float,
    n_xi_for_x: np.ndarray,
    rosenbluth_method: str | None = None,
) -> FokkerPlanckV3Phi1Operator:
    """Construct the poloidally varying v3 FP collision operator (`includePhi1InCollisionOperator=true`).

    The returned operator factors out the (theta,zeta) dependence into a runtime scaling by
    `n_pol = nHat * exp(-Z*alpha*Phi1Hat/THat)`.

    Args:
        rosenbluth_method: as in :func:`make_fokker_planck_v3_operator`.
    """
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
    sqrt_pi = float(_V3_SQRTPI)
    expx2 = np.exp(-(x * x))
    x2 = x * x
    x3 = x2 * x

    xg = make_x_grid(n=n_x, k=float(x_grid_k), include_point_at_x0=False)

    # Base Rosenbluth term, with nHat factored out (set nHat=1 in the helper).
    rosen_base = rosenbluth_potential_terms_v3_np(
        x=x,
        x_weights=x_weights,
        x_grid_k=float(x_grid_k),
        xg=xg,
        z_s=z_s,
        m_hats=m_hats,
        n_hats=np.ones_like(n_hats),
        t_hats=t_hats,
        nl=int(nl),
        method=rosenbluth_method,
    )  # (S,S,NL,X,X)

    k_nu = np.zeros((n_species, n_species, n_x), dtype=np.float64)
    k_cd = np.zeros((n_species, n_species, n_x, n_x), dtype=np.float64)
    k_ce = np.zeros((n_species, n_species, n_x, n_x), dtype=np.float64)

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

            # nuDHat contribution per unit nHat_b.
            k_nu[ia, ib, :] = (3.0 * sqrt_pi / 4.0) / t32m * float(z_s[ia] ** 2) * float(
                z_s[ib] ** 2
            ) * (erfs - psi) / x3

            if ia == ib:
                f_to_f = np.eye(n_x, dtype=np.float64)
            else:
                alpxk = expx2 * (x**float(x_grid_k))
                alpx = expxb2 * (xb**float(x_grid_k))
                f_to_f = polynomial_interpolation_matrix_np(xk=x, x=xb, alpxk=alpxk, alpx=alpx)

            # CD term per unit nHat_a.
            species_factor_cd = 3.0 * float(m_hats[ia] / m_hats[ib]) * float(z_s[ia] ** 2) * float(
                z_s[ib] ** 2
            ) / t32m
            k_cd[ia, ib, :, :] += (species_factor_cd * expx2)[:, None] * f_to_f

            # CE term per unit nHat_b (adds into the diagonal block for species a).
            species_factor_ce = 3.0 * sqrt_pi / 4.0 * float(z_s[ia] ** 2) * float(z_s[ib] ** 2) / t32m
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
            k_ce[ia, ib, :, :] += species_factor_ce * (coef_d2 + coef_dx)

            diag_extra = (
                species_factor_ce
                * 4.0
                / sqrt_pi
                * float(t_hats[ia] / t_hats[ib])
                * math.sqrt(float(t_hats[ia] * m_hats[ib] / (t_hats[ib] * m_hats[ia])))
                * expxb2
            )
            k_ce[ia, ib, range(n_x), range(n_x)] += diag_extra

    return FokkerPlanckV3Phi1Operator(
        nu_n=jnp.asarray(float(nu_n), dtype=jnp.float64),
        krook=jnp.asarray(float(krook), dtype=jnp.float64),
        alpha=jnp.asarray(float(alpha), dtype=jnp.float64),
        z_s=jnp.asarray(z_s, dtype=jnp.float64),
        n_hats=jnp.asarray(n_hats, dtype=jnp.float64),
        t_hats=jnp.asarray(t_hats, dtype=jnp.float64),
        nl=int(nl),
        k_nu=jnp.asarray(k_nu, dtype=jnp.float64),
        k_cd=jnp.asarray(k_cd, dtype=jnp.float64),
        k_ce=jnp.asarray(k_ce, dtype=jnp.float64),
        k_rosen=jnp.asarray(rosen_base, dtype=jnp.float64),
        n_xi_for_x=jnp.asarray(n_xi_for_x, dtype=jnp.int32),
    )

def apply_fokker_planck_v3_phi1(op: FokkerPlanckV3Phi1Operator, f: jnp.ndarray, *, phi1_hat: jnp.ndarray) -> jnp.ndarray:
    """Apply the v3 `collisionOperator=0` collision operator including Phi1 in collisions."""
    if f.ndim != 5:
        raise ValueError("f must have shape (Nspecies, Nx, Nxi, Ntheta, Nzeta)")
    n_species, n_x, n_xi, n_theta, n_zeta = f.shape
    if phi1_hat.shape != (n_theta, n_zeta):
        raise ValueError(f"phi1_hat must have shape {(n_theta, n_zeta)}, got {phi1_hat.shape}")
    if op.k_nu.shape != (n_species, n_species, n_x):
        raise ValueError(f"op.k_nu has shape {op.k_nu.shape}, expected {(n_species, n_species, n_x)}")
    if op.k_cd.shape != (n_species, n_species, n_x, n_x):
        raise ValueError(f"op.k_cd has shape {op.k_cd.shape}, expected {(n_species, n_species, n_x, n_x)}")
    if op.k_ce.shape != (n_species, n_species, n_x, n_x):
        raise ValueError(f"op.k_ce has shape {op.k_ce.shape}, expected {(n_species, n_species, n_x, n_x)}")

    # Effective poloidally varying densities: n_pol[s,t,z] = nHat[s] * exp(-Z*alpha*Phi1Hat/THat).
    n_pol = op.n_hats[:, None, None] * jnp.exp(
        -(op.z_s[:, None, None] * op.alpha / op.t_hats[:, None, None]) * phi1_hat[None, :, :]
    )  # (S,T,Z)

    # nuDHat_pol[a,x,t,z] = sum_b n_pol[b,t,z] * k_nu[a,b,x]
    nu_d_hat = jnp.einsum("bTZ,abx->axTZ", n_pol, op.k_nu)  # (S,X,T,Z)

    # Work in (S,L,X,T,Z) order for matrix multiplies.
    f2 = jnp.transpose(f, (0, 2, 1, 3, 4))  # (S,L,X,T,Z)

    # CD contribution (dense in species indices), scaled by n_pol[a].
    y_cd = jnp.einsum("abij,bLjTZ->aLiTZ", op.k_cd, f2)  # (S,L,X,T,Z)
    y_cd = y_cd * n_pol[:, None, None, :, :]

    # CE contribution (diagonal in species index a, but sums over b).
    ce_mat = jnp.einsum("bTZ,abij->aijTZ", n_pol, op.k_ce)  # (S,X,X,T,Z)
    y_ce = jnp.einsum("aijTZ,aLjTZ->aLiTZ", ce_mat, f2)  # (S,L,X,T,Z)

    # Pitch-angle scattering / Krook diagonal term.
    ell = jnp.arange(n_xi, dtype=jnp.float64)  # (L,)
    factor_l = ell * (ell + 1.0) + 2.0 * op.krook  # (L,)
    y_diag = 0.5 * op.nu_n * nu_d_hat[:, None, :, :, :] * factor_l[None, :, None, None, None] * f2

    y = (-op.nu_n) * (y_cd + y_ce) + y_diag

    # Rosenbluth term (only for L < NL), scaled by n_pol[a].
    nl = int(min(int(op.nl), int(n_xi)))
    if nl > 0:
        y_rosen = jnp.einsum("abLij,bLjTZ->aLiTZ", op.k_rosen[:, :, :nl, :, :], f2[:, :nl, :, :, :])
        y_rosen = y_rosen * n_pol[:, None, None, :, :]
        y = y.at[:, :nl, :, :, :].add((-op.nu_n) * y_rosen)

    # Back to (S,X,L,T,Z).
    y_out = jnp.transpose(y, (0, 2, 1, 3, 4))

    mask = _mask_xi(op.n_xi_for_x.astype(jnp.int32), n_xi).astype(y_out.dtype)  # (X,L)
    return y_out * mask[None, :, :, None, None]

apply_fokker_planck_v3_phi1_jit = jax.jit(apply_fokker_planck_v3_phi1, static_argnums=())

# ---------------------------------------------------------------------------
# Improved Sugama linearized model collision operator (collisionOperator = 3).
#
# This is a research extension BEYOND SFINCS v3 (which ships only the full
# linearized Fokker-Planck operator, collisionOperator = 0, and pitch-angle
# scattering, collisionOperator = 1).  It implements the momentum- and
# energy-conserving *improved* linearized model collision operator of
#
#   H. Sugama, S. Matsuoka, S. Satake, M. Nunami, and T.-H. Watanabe,
#   "Improved linearized model collision operator for the highly collisional
#   regime", Phys. Plasmas 26, 102108 (2019),
#
# using the moment-based numerical construction of the field-particle term of
#
#   B. J. Frei, S. Ernst, and P. Ricci, "Numerical implementation of the
#   improved Sugama collision operator using a moment approach",
#   Phys. Plasmas 29, 093902 (2022) (arXiv:2202.06293).
#
# Structure.  The operator is the test-particle part (pitch-angle deflection +
# energy/speed diffusion -- identical velocity kernels to the v3 Fokker-Planck
# operator's ``nuD`` and ``CE`` blocks) PLUS a field-particle (back-reaction)
# term built from low-order velocity moments whose coefficients are fixed to
# enforce the conservation laws that the test-particle part alone violates.
# The improved-Sugama field term lives only in the L=0 (particle + energy) and
# L=1 (parallel momentum) Legendre components, so it assembles into exactly the
# same ``mat[a,b,L,i,j]`` block structure as the Fokker-Planck operator and
# reuses :func:`apply_fokker_planck_v3` for the matvec.
#
# Conservation is EXACT at the collocation level (not merely to the spectral
# quadrature error), because each field block is constructed as a low-rank
# operator whose moment functional cancels the test-particle moment functional
# algebraically -- the essence of the moment approach.
# ---------------------------------------------------------------------------

def _improved_sugama_pair_kernels(
    *,
    x: np.ndarray,
    ddx: np.ndarray,
    d2dx2: np.ndarray,
    z_s: np.ndarray,
    m_hats: np.ndarray,
    n_hats: np.ndarray,
    t_hats: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-species-pair test-particle velocity kernels.

    Returns ``(nu_d_ab, ce_ab)`` where ``nu_d_ab[a,b,:]`` is the pitch-angle
    deflection frequency of species ``a`` off background ``b`` (same expression
    as the v3 ``nuDHat`` summand, WITHOUT the overall ``nu_n``) and
    ``ce_ab[a,b,:,:]`` is the energy/speed-diffusion (``CE``) collocation block
    of species ``a`` due to ``b`` (same expression as the v3 Fokker-Planck
    ``CE`` summand, WITHOUT ``nu_n``).  Both mirror the coefficients already
    used in :func:`make_fokker_planck_v3_operator`.
    """
    x = np.asarray(x, dtype=np.float64)
    n_x = int(x.size)
    n_species = int(z_s.size)
    sqrt_pi = float(_V3_SQRTPI)
    x2 = x * x
    x3 = x2 * x

    nu_d_ab = np.zeros((n_species, n_species, n_x), dtype=np.float64)
    ce_ab = np.zeros((n_species, n_species, n_x, n_x), dtype=np.float64)
    for ia in range(n_species):
        t32m = float(t_hats[ia]) * math.sqrt(float(t_hats[ia]) * float(m_hats[ia]))
        for ib in range(n_species):
            species_factor = float(
                math.sqrt((t_hats[ia] * m_hats[ib]) / (t_hats[ib] * m_hats[ia]))
            )
            xb = x * species_factor
            expxb2 = np.exp(-(xb * xb))
            erfs = _erf_np(xb)
            psi = (erfs - (2.0 / sqrt_pi) * xb * expxb2) / (2.0 * xb * xb)

            nu_d_ab[ia, ib, :] = (
                (3.0 * sqrt_pi / 4.0)
                / t32m
                * float(z_s[ia] ** 2)
                * float(z_s[ib] ** 2)
                * float(n_hats[ib])
                * (erfs - psi)
                / x3
            )

            species_factor_ce = (
                3.0 * sqrt_pi / 4.0 * float(n_hats[ib]) * float(z_s[ia] ** 2) * float(z_s[ib] ** 2) / t32m
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
            ce = species_factor_ce * (coef_d2 + coef_dx)
            diag_extra = (
                species_factor_ce
                * 4.0
                / sqrt_pi
                * float(t_hats[ia] / t_hats[ib])
                * math.sqrt(float(t_hats[ia] * m_hats[ib] / (t_hats[ib] * m_hats[ia])))
                * expxb2
            )
            ce[range(n_x), range(n_x)] += diag_extra
            ce_ab[ia, ib, :, :] = ce
    return nu_d_ab, ce_ab

@jtu.register_pytree_node_class
@dataclass(frozen=True)
class ImprovedSugamaV3Operator:
    """Improved Sugama momentum/energy-conserving model collision operator.

    Research extension beyond v3 (``collisionOperator = 3``).  See the module
    section above for the physics and primary references (Sugama et al., Phys.
    Plasmas 26, 102108 (2019); Frei et al., arXiv:2202.06293).  The assembled
    ``mat`` has the same ``(S, S, L, X, X)`` block layout as
    :class:`FokkerPlanckV3Operator` (already multiplied by ``-nu_n``), so the
    matvec is shared with :func:`apply_fokker_planck_v3`.
    """

    mat: jnp.ndarray  # (S,S,L,X,X)
    n_xi_for_x: jnp.ndarray  # (X,) int32
    mask_xi: jnp.ndarray  # (X, L)
    # The same operator, kept apart as ``test`` plus a rank-K correction:
    # ``mat[a,b,L,i,j] = delta_ab test[a,L,i,j]
    #                    + sum_c columns[a,i,L,c] extraction[b,j,L,c]``.
    # ``test`` is species-diagonal and dense only in ``x``, so a velocity-space
    # block smoother inverts it directly; the field-particle back-reaction is
    # the part that couples species and carries the conservation laws, and it
    # is low rank (``K = 2 S``, of which L=1 uses ``S`` and L >= 2 uses none).
    # Splitting them is what lets a relaxation see a local operator while the
    # conserving terms are still applied exactly
    # (:func:`solvax.precond.low_rank_corrected`).
    # ``apply`` uses ``mat`` alone, so these stay optional: a caller assembling
    # an operator by hand (or an older pickle) simply has no split to offer.
    test: jnp.ndarray | None = None  # (S,L,X,X)
    columns: jnp.ndarray | None = None  # (S,X,L,K)
    extraction: jnp.ndarray | None = None  # (S,X,L,K)

    def tree_flatten(self):
        children = (
            self.mat,
            self.n_xi_for_x,
            self.mask_xi,
            self.test,
            self.columns,
            self.extraction,
        )
        aux = None
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        del aux
        mat, n_xi_for_x, mask_xi, test, columns, extraction = children
        return cls(
            mat=mat,
            n_xi_for_x=n_xi_for_x,
            mask_xi=mask_xi,
            test=test,
            columns=columns,
            extraction=extraction,
        )

def make_improved_sugama_v3_operator(
    *,
    x: np.ndarray,  # (X,)
    x_weights: np.ndarray,  # (X,) dx weights (createGrids.F90 convention)
    ddx: np.ndarray,  # (X,X)
    d2dx2: np.ndarray,  # (X,X)
    z_s: np.ndarray,  # (S,)
    m_hats: np.ndarray,  # (S,)
    n_hats: np.ndarray,  # (S,)
    t_hats: np.ndarray,  # (S,)
    nu_n: float,
    krook: float,
    n_xi: int,
    n_xi_for_x: np.ndarray,
) -> ImprovedSugamaV3Operator:
    """Construct the improved Sugama model collision operator (``collisionOperator=3``).

    The operator is ``C = C_test + C_field`` where ``C_test`` is the
    test-particle pitch-angle + energy-diffusion operator (same velocity
    kernels as the v3 Fokker-Planck ``nuD``/``CE`` blocks) and ``C_field`` is
    the moment-based back-reaction that restores the conservation laws.

    Conservation-enforcing coefficient logic
    ----------------------------------------
    Write the assembled test-particle block for the ordered pair ``(a,b)`` and
    Legendre index ``L`` as ``T[a,b,L]`` (already carrying the ``-nu_n``
    factor); it acts on ``f_a`` and is diagonal in the species index.  Let the
    discrete moment weights be ``p_i = x_weights_i x_i^3`` (parallel momentum,
    L=1), ``n_i = x_weights_i x_i^2`` (particle number, L=0), and
    ``e_i = x_weights_i x_i^4`` (kinetic energy, L=0), and the per-species
    physical prefactors ``P_a = t_a^2/m_a`` (``~ m_a v_{th,a}^4``, momentum)
    and ``E_a = t_a^{5/2}/m_a^{3/2}`` (``~ m_a v_{th,a}^5``, energy).

    * **L=1 parallel momentum.**  With the Maxwellian-flow invariant
      ``phi_i = x_i exp(-x_i^2)``, the response profile is the test-particle
      action on the invariant ``r_a = (sum_b T[a,b,1]) phi`` and the extraction
      is the exact test momentum functional ``tau_ab = p^T T[a,b,1]``.  The
      field block is ``F[a,b,1] = kappa_ab * outer(r_a, tau_ba)`` with
      ``kappa_ab = -P_b / (P_a * (p . r_a))``.  Summing the momentum functional
      ``sum_a P_a p^T (T + F)`` telescopes to zero for arbitrary ``f`` (the
      ``b<->a`` relabelling uses ``P_a (p.r_a) kappa_ab = -P_b``), so total
      parallel momentum is conserved exactly; for a single species this also
      makes the operator annihilate the shifted Maxwellian ``phi`` (Galilean
      invariance of like-particle collisions).

    * **L=0 particle number + energy.**  With the density shape
      ``h1_i = exp(-x_i^2)`` and the particle-neutral energy shape
      ``h2_i = (x_i^2 - c) exp(-x_i^2)`` (``c`` chosen so ``n . h2 = 0``), the
      per-species particle functional ``nu_ab = n^T T[a,b,0]`` is cancelled by
      ``-h1 (n^T . )/(n . h1)`` (particle conservation, diagonal in species),
      and the residual energy functional after particle restoration
      ``mu_ab = e^T T[a,b,0] - (e.h1)/(n.h1) nu_ab`` is redistributed by
      ``F_energy[a,b,0] = lambda_ab outer(h2, mu_ba)`` with
      ``lambda_ab = -E_b/(E_a (e.h2))``, which telescopes exactly like the
      momentum term so total kinetic energy is conserved (energy is exchanged
      between species; like-particle collisions conserve it per species).

    Keeping ``collisionOperator`` 0/1 byte-identical, this builder shares no
    code path with :func:`make_fokker_planck_v3_operator`.
    """
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
    n_xi_int = int(n_xi)
    nu_n_f = float(nu_n)
    krook_f = float(krook)

    nu_d_ab, ce_ab = _improved_sugama_pair_kernels(
        x=x, ddx=ddx, d2dx2=d2dx2, z_s=z_s, m_hats=m_hats, n_hats=n_hats, t_hats=t_hats
    )

    # Discrete moment weights and per-species physical prefactors.
    w_mom = x_weights * (x**3)  # parallel momentum weight (L=1)
    w_part = x_weights * (x**2)  # particle-number weight (L=0)
    w_energy = x_weights * (x**4)  # kinetic-energy weight (L=0)
    p_mom = t_hats**2 / m_hats  # ~ m_a v_th,a^4
    p_energy = t_hats**2.5 / m_hats**1.5  # ~ m_a v_th,a^5

    # Maxwellian invariants / response shapes.
    expx2 = np.exp(-(x * x))
    phi = x * expx2  # L=1 flow invariant
    h1 = expx2.copy()  # L=0 density shape
    c_energy = float((w_part @ (x * x * h1)) / (w_part @ h1))
    h2 = (x * x - c_energy) * expx2  # L=0 particle-neutral energy shape
    n1 = float(w_part @ h1)
    e2 = float(w_energy @ h2)
    e_h1 = float(w_energy @ h1)

    # Per-pair, per-L assembled test-particle blocks T[a,b,L] (carry -nu_n).
    ell = np.arange(n_xi_int, dtype=np.float64)
    pitch_factor_l = 0.5 * (ell * (ell + 1.0) + 2.0 * krook_f)  # (L,)
    test_ab_l = np.zeros((n_species, n_species, n_xi_int, n_x, n_x), dtype=np.float64)
    for ia in range(n_species):
        for ib in range(n_species):
            base = -nu_n_f * ce_ab[ia, ib]  # energy diffusion (all L)
            for ell_i in range(n_xi_int):
                blk = base.copy()
                # pitch-angle diagonal: v3 diag = -0.5 nu_d (l(l+1)+2 krook); mat = -nu_n * diag.
                blk[range(n_x), range(n_x)] += nu_n_f * pitch_factor_l[ell_i] * nu_d_ab[ia, ib]
                test_ab_l[ia, ib, ell_i] = blk

    # ---- field-particle factors, as a rank-K correction over (species, x) ----
    # Every field term is one outer product per species pair, so the whole
    # back-reaction factors with the rank index running over species:
    # F[(a,i),(b,j),L] = sum_c columns[a,i,L,c] extraction[b,j,L,c].
    # K = 2 S covers L=0 (particle + energy); L=1 uses S of the slots
    # (momentum) and L >= 2 carries no field term at all.
    rank = 2 * n_species
    columns = np.zeros((n_species, n_x, n_xi_int, rank), dtype=np.float64)
    extraction = np.zeros((n_species, n_x, n_xi_int, rank), dtype=np.float64)

    # L=1 parallel momentum: U = r_a, V = kappa_cb tau_bc.
    if n_xi_int > 1:
        test_l1 = test_ab_l[:, :, 1, :, :]  # (S,S,X,X)
        r_mom = np.einsum("abij,j->ai", test_l1, phi)  # response r_a (S,X)
        tau_mom = np.einsum("i,abij->abj", w_mom, test_l1)  # tau_ab (S,S,X)
        r_dot = w_mom @ r_mom.T  # (S,) = p . r_a
        for ic in range(n_species):
            columns[ic, :, 1, ic] = r_mom[ic]
            for ib in range(n_species):
                kappa = -float(p_mom[ib]) / (float(p_mom[ic]) * float(r_dot[ic]))
                extraction[ib, :, 1, ic] = kappa * tau_mom[ib, ic]

    # L=0 particle number (U = h1, species-diagonal) and kinetic energy (U = h2).
    test_l0 = test_ab_l[:, :, 0, :, :]  # (S,S,X,X)
    nu_part = np.einsum("i,abij->abj", w_part, test_l0)  # nu_ab (S,S,X)
    eps_energy = np.einsum("i,abij->abj", w_energy, test_l0)  # eps_ab (S,S,X)
    mu_energy = eps_energy - (e_h1 / n1) * nu_part  # residual energy functional (S,S,X)
    for ic in range(n_species):
        columns[ic, :, 0, ic] = h1
        extraction[ic, :, 0, ic] = -(1.0 / n1) * nu_part[ic].sum(axis=0)
        columns[ic, :, 0, n_species + ic] = h2
        for ib in range(n_species):
            lam = -float(p_energy[ib]) / (float(p_energy[ic]) * e2)
            extraction[ib, :, 0, n_species + ic] = lam * mu_energy[ib, ic]

    # Test-particle part is diagonal in the species index (acts on f_a).
    test = np.einsum("abLij->aLij", test_ab_l)

    mat = np.einsum("ab,aLij->abLij", np.eye(n_species), test)
    mat += np.einsum("aiLc,bjLc->abLij", columns, extraction)

    return ImprovedSugamaV3Operator(
        mat=jnp.asarray(mat),
        n_xi_for_x=jnp.asarray(n_xi_for_x, dtype=jnp.int32),
        mask_xi=_mask_xi(jnp.asarray(n_xi_for_x, dtype=jnp.int32), n_xi_int),
        test=jnp.asarray(test),
        columns=jnp.asarray(columns),
        extraction=jnp.asarray(extraction),
    )

def apply_improved_sugama_v3(op: ImprovedSugamaV3Operator, f: jnp.ndarray) -> jnp.ndarray:
    """Apply the improved Sugama model collision operator to ``f``.

    Shares the block matvec with :func:`apply_fokker_planck_v3` (identical
    ``mat`` layout): ``y[a,i,l,t,z] = sum_{b,j} mat[a,b,l,i,j] f[b,j,l,t,z]``.
    """
    if f.ndim != 5:
        raise ValueError("f must have shape (Nspecies, Nx, Nxi, Ntheta, Nzeta)")
    n_species, n_x, n_xi, _, _ = f.shape
    if op.mat.shape != (n_species, n_species, n_xi, n_x, n_x):
        raise ValueError(
            f"op.mat has shape {op.mat.shape}, expected {(n_species, n_species, n_xi, n_x, n_x)}"
        )

    f2 = jnp.transpose(f, (0, 2, 1, 3, 4))  # (S,L,X,T,Z)
    y2 = jnp.einsum("abLij,bLjtz->aLitz", op.mat, f2)
    y = jnp.transpose(y2, (0, 2, 1, 3, 4))  # (S,X,L,T,Z)

    if op.mask_xi.shape[-1] != n_xi:
        mask = _mask_xi(op.n_xi_for_x.astype(jnp.int32), n_xi).astype(y.dtype)
    else:
        mask = op.mask_xi.astype(y.dtype)
    return y * mask[None, :, :, None, None]

apply_improved_sugama_v3_jit = jax.jit(apply_improved_sugama_v3, static_argnums=())
