from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np
from scipy.integrate import quad


def _weight(x: float, k: float) -> float:
    return math.exp(-(x * x)) * (x**k)


def x_weight_np(x: np.ndarray, k: float) -> np.ndarray:
    """Weight function used by SFINCS v3 for the polynomial x grid."""
    x = np.asarray(x, dtype=np.float64)
    return np.exp(-(x * x)) * (x**k)


def x_weight_d1_over_weight_np(x: np.ndarray, k: float) -> np.ndarray:
    """(d/dx weight) / weight, matching v3 `polynomialDiffMatrices.F90`."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    mask0 = np.abs(x) < 1e-12
    out[mask0] = 0.0
    out[~mask0] = k / x[~mask0] - 2.0 * x[~mask0]
    return out


def x_weight_d2_over_weight_np(x: np.ndarray, k: float) -> np.ndarray:
    """(d^2/dx^2 weight) / weight, matching v3 `polynomialDiffMatrices.F90`."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    mask0 = np.abs(x) < 1e-12
    out[mask0] = -2.0
    out[~mask0] = k * (k - 1.0) / (x[~mask0] * x[~mask0]) - 2.0 * (2.0 * k + 1.0) + 4.0 * (
        x[~mask0] * x[~mask0]
    )
    return out


def make_x_polynomial_diff_matrices(x: np.ndarray, *, k: float) -> tuple[np.ndarray, np.ndarray]:
    """Port of v3 `makeXPolynomialDiffMatrices` (polynomialDiffMatrices.F90).

    Returns
    -------
    ddx, d2dx2:
      First and second derivative matrices on the (possibly nonuniform) x-grid.
    """
    x = np.asarray(x, dtype=np.float64)
    n = int(x.size)
    if n < 1:
        raise ValueError("x must have at least one point")

    xx = np.broadcast_to(x[:, None], (n, n)).copy()
    dx = xx - xx.T
    np.fill_diagonal(dx, 1.0)

    c = np.prod(dx, axis=1)
    c = c * x_weight_np(x, k)
    ccc = c[:, None] / c[None, :]

    z = 1.0 / dx
    np.fill_diagonal(z, 0.0)

    xxx = np.zeros((n - 1, n), dtype=np.float64)
    for i in range(n):
        if i + 1 < n:
            xxx[i:, i] = z[i, i + 1 :]
        if i > 0:
            xxx[:i, i] = z[i, :i]

    y = np.zeros((n, n), dtype=np.float64)
    y[0, :] = x_weight_d1_over_weight_np(x, k)
    for i in range(1, n):
        y[i, :] = y[i - 1, :] + xxx[i - 1, :]

    ddx = z * ccc
    np.fill_diagonal(ddx, y[-1, :])

    old_y = y
    y2 = np.zeros((n, n), dtype=np.float64)
    y2[0, :] = x_weight_d2_over_weight_np(x, k)
    for i in range(1, n):
        y2[i, :] = y2[i - 1, :] + 2.0 * old_y[i - 1, :] * xxx[i - 1, :]

    repmat_diag_ddx = np.broadcast_to(np.diag(ddx)[:, None], (n, n))
    d2dx2 = 2.0 * z * (ccc * repmat_diag_ddx - ddx)
    np.fill_diagonal(d2dx2, y2[-1, :])

    return ddx, d2dx2

def _integrate_split(
    f: Callable[[float], float],
    *,
    finite_bound: float = 10.0,
    epsabs: float = 0.0,
    epsrel: float = 1e-13,
    limit: int = 5000,
) -> float:
    a1, _ = quad(f, 0.0, finite_bound, epsabs=epsabs, epsrel=epsrel, limit=limit)
    a2, _ = quad(f, finite_bound, np.inf, epsabs=epsabs, epsrel=epsrel, limit=limit)
    return float(a1 + a2)


def _evaluate_polynomial(x: float, *, j: int, a: np.ndarray, b: np.ndarray) -> float:
    # Mirrors xGrid.F90:evaluatePolynomial().
    if j == 1:
        return 1.0
    pj_minus1 = 0.0
    pj = 1.0
    y = 0.0
    for ii in range(1, j):
        y = (x - float(a[ii])) * pj - float(b[ii]) * pj_minus1
        pj_minus1, pj = pj, y
    return float(y)


@dataclass(frozen=True)
class XGrid:
    x: np.ndarray
    gaussian_weights: np.ndarray

    def dx_weights(self, k: float) -> np.ndarray:
        """Return weights for plain `dx` integrals (Fortran divides by the weight function)."""
        w = np.exp(-(self.x * self.x)) * (self.x**k)
        return self.gaussian_weights / w


def make_x_grid(
    *,
    n: int,
    k: float = 0.0,
    include_point_at_x0: bool = False,
    x0: float = 0.0,
    finite_bound: float = 10.0,
) -> XGrid:
    """Port of SFINCS v3 `makeXGrid` for the built-in weight exp(-x^2)*x^k on [0, +inf).

    This routine is not performance-critical for the JAX solve (the grid is static). It is
    implemented primarily for parity with the Fortran v3 grids.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if k < 0:
        raise ValueError("k must be >= 0 for the built-in SFINCS weight")

    # Use 1-based indexing to stay close to the Fortran code.
    a = np.zeros((n + 1,), dtype=float)
    b = np.zeros((n + 2,), dtype=float)
    c = np.zeros((n + 1,), dtype=float)
    d = np.zeros((n + 1,), dtype=float)

    oldc = 1.0
    last_poly_x0 = 0.0
    penult_poly_x0 = 0.0

    for j in range(1, n + 1):
        def p(xx: float) -> float:
            return _evaluate_polynomial(xx, j=j, a=a, b=b)

        def integrand_c(xx: float) -> float:
            pj = p(xx)
            return pj * _weight(xx, k) * pj

        def integrand_d(xx: float) -> float:
            pj = p(xx)
            return xx * pj * _weight(xx, k) * pj

        c[j] = _integrate_split(integrand_c, finite_bound=finite_bound)
        d[j] = _integrate_split(integrand_d, finite_bound=finite_bound)

        b[j] = c[j] / oldc
        a[j] = d[j] / c[j]
        oldc = c[j]

        penult_poly_x0 = last_poly_x0
        last_poly_x0 = p(x0)

    if include_point_at_x0:
        a[n] = x0 - b[n] * penult_poly_x0 / last_poly_x0

    # Jacobi matrix eigen-decomposition (Golub-Welsch).
    diag = a[1 : n + 1].copy()
    off = np.sqrt(b[2 : n + 1].copy())

    jmat = np.diag(diag)
    for i in range(n - 1):
        jmat[i, i + 1] = off[i]
        jmat[i + 1, i] = off[i]

    abscissae, eigenvectors = np.linalg.eigh(jmat)
    weights = c[1] * (eigenvectors[0, :] ** 2)

    if include_point_at_x0:
        # Match the Fortran behavior: force the smallest node to be exactly x0 (typically 0).
        abscissae = abscissae.copy()
        abscissae[0] = x0

    return XGrid(x=abscissae, gaussian_weights=weights)
