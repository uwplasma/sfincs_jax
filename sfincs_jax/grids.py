from __future__ import annotations

import math
from typing import Tuple

from jax import config as _jax_config
_jax_config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np


def uniform_diff_matrices(
    *,
    n: int,
    x_min: float,
    x_max: float,
    scheme: int,
    dtype: np.dtype = np.float64,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Port of SFINCS v3 `uniformDiffMatrices`.

    Returns `x`, `weights`, `ddx`, `d2dx2` as JAX arrays.
    """
    if n < 2:
        raise ValueError(f"n must be at least 2, got {n}")
    if x_min > x_max:
        raise ValueError(f"x_max must be >= x_min, got x_min={x_min}, x_max={x_max}")
    if x_min == x_max:
        raise ValueError("x_max cannot equal x_min")

    x_min = float(x_min)
    x_max = float(x_max)

    aperiodic = scheme in {2, 3, 12, 13, 32, 42, 52, 62, 82, 92, 102, 112, 122, 132}
    include_xmin_not_xmax = scheme in {0, 10, 20, 30, 40, 50, 60, 80, 90, 100, 110, 120, 130}
    include_xmax_not_xmin = scheme in {1, 11, 21, 31, 41, 51, 61, 81, 91, 101, 111, 121, 131}

    if aperiodic:
        x = np.linspace(x_min, x_max, n, dtype=dtype)
    elif include_xmin_not_xmax:
        x = x_min + (x_max - x_min) * (np.arange(n, dtype=dtype) / n)
    elif include_xmax_not_xmin:
        x = x_min + (x_max - x_min) * (np.arange(1, n + 1, dtype=dtype) / n)
    else:
        raise ValueError(f"Invalid scheme: {scheme}")

    dx = float(x[1] - x[0])
    dx2 = dx * dx

    weights = np.full((n,), dx, dtype=dtype)
    if aperiodic:
        weights[0] *= 0.5
        weights[-1] *= 0.5

    ddx = np.zeros((n, n), dtype=dtype)
    d2dx2 = np.zeros((n, n), dtype=dtype)

    # Fill the interior of the differentiation matrices:
    if scheme in {0, 1, 2, 3}:
        if n < 3:
            raise ValueError("n must be at least 3 for 3-point stencil schemes")
        for i in range(1, n - 1):
            ddx[i, i + 1] = 1.0 / (2 * dx)
            ddx[i, i - 1] = -1.0 / (2 * dx)
            d2dx2[i, i + 1] = 1.0 / dx2
            d2dx2[i, i] = -2.0 / dx2
            d2dx2[i, i - 1] = 1.0 / dx2

    elif scheme in {10, 11, 12, 13}:
        if n < 5:
            raise ValueError("n must be at least 5 for 5-point stencil schemes")
        for i in range(2, n - 2):
            ddx[i, i + 2] = -1.0 / (12 * dx)
            ddx[i, i + 1] = 2.0 / (3 * dx)
            ddx[i, i - 1] = -2.0 / (3 * dx)
            ddx[i, i - 2] = 1.0 / (12 * dx)

            d2dx2[i, i + 2] = -1.0 / (12 * dx2)
            d2dx2[i, i + 1] = 4.0 / (3 * dx2)
            d2dx2[i, i] = -5.0 / (2 * dx2)
            d2dx2[i, i - 1] = 4.0 / (3 * dx2)
            d2dx2[i, i - 2] = -1.0 / (12 * dx2)

    elif scheme in {30, 31, 32}:
        if n < 3:
            raise ValueError("n must be at least 3 for this scheme")
        for i in range(1, n):
            ddx[i, i] = 1.0 / dx
            ddx[i, i - 1] = -1.0 / dx
        for i in range(2, n):
            d2dx2[i, i] = 1.0 / dx2
            d2dx2[i, i - 1] = -2.0 / dx2
            d2dx2[i, i - 2] = 1.0 / dx2

    elif scheme in {40, 41, 42}:
        if n < 3:
            raise ValueError("n must be at least 3 for this scheme")
        for i in range(0, n - 1):
            ddx[i, i] = -1.0 / dx
            ddx[i, i + 1] = 1.0 / dx
        for i in range(0, n - 2):
            d2dx2[i, i] = 1.0 / dx2
            d2dx2[i, i + 1] = -2.0 / dx2
            d2dx2[i, i + 2] = 1.0 / dx2

    elif scheme in {50, 51, 52}:
        if n < 3:
            raise ValueError("n must be at least 3 for this scheme")
        for i in range(2, n):
            ddx[i, i] = 1.5 / dx
            ddx[i, i - 1] = -2.0 / dx
            ddx[i, i - 2] = 1.0 / (2 * dx)

            d2dx2[i, i] = 1.0 / dx2
            d2dx2[i, i - 1] = -2.0 / dx2
            d2dx2[i, i - 2] = 1.0 / dx2

    elif scheme in {60, 61, 62}:
        if n < 3:
            raise ValueError("n must be at least 3 for this scheme")
        for i in range(0, n - 2):
            ddx[i, i] = -1.5 / dx
            ddx[i, i + 1] = 2.0 / dx
            ddx[i, i + 2] = -1.0 / (2 * dx)

            d2dx2[i, i] = 1.0 / dx2
            d2dx2[i, i + 1] = -2.0 / dx2
            d2dx2[i, i + 2] = 1.0 / dx2

    elif scheme in {20, 21, 22}:
        # Spectral differentiation matrices (periodic).
        pi = math.pi
        h = 2 * pi / n
        n1 = int(math.floor((n - 1.0) / 2))
        n2 = int(math.ceil((n - 1.0) / 2))

        col1 = np.zeros((n,), dtype=dtype)
        if n % 2 == 0:
            topc = np.array([0.5 / math.tan(i * h / 2) for i in range(1, n2 + 1)], dtype=dtype)
            col1[1 : n2 + 1] = topc
            col1[n2 + 1 :] = -topc[n1 - 1 :: -1]
            col1[1::2] *= -1
            col1 *= 2 * pi / (x_max - x_min)
            for i in range(n):
                ddx[i, i:] = -col1[: n - i]
                ddx[i, :i] = col1[i:0:-1]
        else:
            topc = np.array([0.5 / math.sin(i * h / 2) for i in range(1, n2 + 1)], dtype=dtype)
            col1[1 : n2 + 1] = topc
            col1[n2 + 1 :] = topc[n1 - 1 :: -1]
            col1[1::2] *= -1
            col1 *= 2 * pi / (x_max - x_min)
            for i in range(n):
                ddx[i, i:] = -col1[: n - i]
                ddx[i, :i] = col1[i:0:-1]

        # Second derivative matrix:
        col1 = np.zeros((n,), dtype=dtype)
        if n % 2 == 0:
            col1[0] = -pi * pi / (3 * h * h) - 1.0 / 6
            topc = np.array([-(0.5) / (math.sin(i * h / 2) ** 2) for i in range(1, n2 + 1)], dtype=dtype)
            col1[1 : n2 + 1] = topc
            col1[n2 + 1 :] = topc[n1 - 1 :: -1]
            col1[1::2] *= -1
            col1 *= (2 * pi / (x_max - x_min)) ** 2
            for i in range(n):
                d2dx2[i, i:] = col1[: n - i]
                d2dx2[i, :i] = col1[i:0:-1]
        else:
            col1[0] = -pi * pi / (3 * h * h) + 1.0 / 12
            topc = np.array(
                [-(0.5) / (math.sin(i * h / 2) * math.tan(i * h / 2)) for i in range(1, n2 + 1)],
                dtype=dtype,
            )
            col1[1 : n2 + 1] = topc
            col1[n2 + 1 :] = -topc[n1 - 1 :: -1]
            col1[1::2] *= -1
            col1 *= (2 * pi / (x_max - x_min)) ** 2
            for i in range(n):
                d2dx2[i, i:] = col1[: n - i]
                d2dx2[i, :i] = col1[i:0:-1]

    elif scheme in {80, 81}:
        if n < 5:
            raise ValueError("n must be at least 5 for 4 point stencil schemes")
        for i in range(n):
            ddx[i, (i + 1) % n] = 1.0 / (3 * dx)
            ddx[i, i] = 1.0 / (2 * dx)
            ddx[i, (i - 1) % n] = -1.0 / dx
            ddx[i, (i - 2) % n] = 1.0 / (6 * dx)

            d2dx2[i, (i + 1) % n] = 1.0 / dx2
            d2dx2[i, i] = -2.0 / dx2
            d2dx2[i, (i - 1) % n] = 1.0 / dx2

    elif scheme == 82:
        if n < 5:
            raise ValueError("n must be at least 5 for 4 point stencil schemes")
        for i in range(2, n - 1):
            ddx[i, i + 1] = 1.0 / (3 * dx)
            ddx[i, i] = 1.0 / (2 * dx)
            ddx[i, i - 1] = -1.0 / dx
            ddx[i, i - 2] = 1.0 / (6 * dx)
        for i in range(1, n - 1):
            d2dx2[i, i + 1] = 1.0 / dx2
            d2dx2[i, i] = -2.0 / dx2
            d2dx2[i, i - 1] = 1.0 / dx2

    elif scheme in {90, 91}:
        if n < 5:
            raise ValueError("n must be at least 5 for 4 point stencil schemes")
        for i in range(n):
            ddx[i, (i - 1) % n] = -1.0 / (3 * dx)
            ddx[i, i] = -1.0 / (2 * dx)
            ddx[i, (i + 1) % n] = 1.0 / dx
            ddx[i, (i + 2) % n] = -1.0 / (6 * dx)

            d2dx2[i, (i + 1) % n] = 1.0 / dx2
            d2dx2[i, i] = -2.0 / dx2
            d2dx2[i, (i - 1) % n] = 1.0 / dx2

    elif scheme == 92:
        if n < 5:
            raise ValueError("n must be at least 5 for 4 point stencil schemes")
        for i in range(1, n - 2):
            ddx[i, i - 1] = -1.0 / (3 * dx)
            ddx[i, i] = -1.0 / (2 * dx)
            ddx[i, i + 1] = 1.0 / dx
            ddx[i, i + 2] = -1.0 / (6 * dx)
        for i in range(1, n - 1):
            d2dx2[i, i + 1] = 1.0 / dx2
            d2dx2[i, i] = -2.0 / dx2
            d2dx2[i, i - 1] = 1.0 / dx2

    elif scheme in {100, 101}:
        if n < 5:
            raise ValueError("n must be at least 5 for schemes 100, 101")
        for i in range(n):
            ddx[i, (i + 1) % n] = 1.0 / (4 * dx)
            ddx[i, i] = 5.0 / (6 * dx)
            ddx[i, (i - 1) % n] = -3.0 / (2 * dx)
            ddx[i, (i - 2) % n] = 1.0 / (2 * dx)
            ddx[i, (i - 3) % n] = -1.0 / (12 * dx)

    elif scheme == 102:
        if n < 5:
            raise ValueError("n must be at least 5 for scheme 102")
        for i in range(3, n - 1):
            ddx[i, i + 1] = 1.0 / (4 * dx)
            ddx[i, i] = 5.0 / (6 * dx)
            ddx[i, i - 1] = -3.0 / (2 * dx)
            ddx[i, i - 2] = 1.0 / (2 * dx)
            ddx[i, i - 3] = -1.0 / (12 * dx)

    elif scheme in {110, 111}:
        if n < 5:
            raise ValueError("n must be at least 5 for schemes 110, 111")
        for i in range(n):
            ddx[i, (i - 1) % n] = -1.0 / (4 * dx)
            ddx[i, i] = -5.0 / (6 * dx)
            ddx[i, (i + 1) % n] = 3.0 / (2 * dx)
            ddx[i, (i + 2) % n] = -1.0 / (2 * dx)
            ddx[i, (i + 3) % n] = 1.0 / (12 * dx)

    elif scheme == 112:
        if n < 5:
            raise ValueError("n must be at least 5 for scheme 112")
        for i in range(1, n - 3):
            ddx[i, i - 1] = -1.0 / (4 * dx)
            ddx[i, i] = -5.0 / (6 * dx)
            ddx[i, i + 1] = 3.0 / (2 * dx)
            ddx[i, i + 2] = -1.0 / (2 * dx)
            ddx[i, i + 3] = 1.0 / (12 * dx)

    elif scheme in {120, 121}:
        if n < 5:
            raise ValueError("n must be at least 5 for schemes 120, 121")
        for i in range(n):
            ddx[i, (i + 2) % n] = -1.0 / (20 * dx)
            ddx[i, (i + 1) % n] = 1.0 / (2 * dx)
            ddx[i, i] = 1.0 / (3 * dx)
            ddx[i, (i - 1) % n] = -1.0 / dx
            ddx[i, (i - 2) % n] = 1.0 / (4 * dx)
            ddx[i, (i - 3) % n] = -1.0 / (30 * dx)

    elif scheme == 122:
        raise NotImplementedError("scheme 122 is not implemented in SFINCS v3 either")

    elif scheme in {130, 131}:
        if n < 5:
            raise ValueError("n must be at least 5 for schemes 130, 131")
        for i in range(n):
            ddx[i, (i - 2) % n] = 1.0 / (20 * dx)
            ddx[i, (i - 1) % n] = -1.0 / (2 * dx)
            ddx[i, i] = -1.0 / (3 * dx)
            ddx[i, (i + 1) % n] = 1.0 / dx
            ddx[i, (i + 2) % n] = -1.0 / (4 * dx)
            ddx[i, (i + 3) % n] = 1.0 / (30 * dx)

    elif scheme == 132:
        raise NotImplementedError("scheme 132 is not implemented in SFINCS v3 either")

    else:
        raise ValueError(f"Invalid scheme: {scheme}")

    # Handle endpoints:
    if scheme in {0, 1}:
        ddx[0, -1] = -1.0 / (2 * dx)
        ddx[0, 1] = 1.0 / (2 * dx)
        ddx[-1, 0] = 1.0 / (2 * dx)
        ddx[-1, -2] = -1.0 / (2 * dx)

        d2dx2[0, 0] = -2.0 / dx2
        d2dx2[-1, -1] = -2.0 / dx2
        d2dx2[0, -1] = 1.0 / dx2
        d2dx2[0, 1] = 1.0 / dx2
        d2dx2[-1, 0] = 1.0 / dx2
        d2dx2[-1, -2] = 1.0 / dx2

    elif scheme == 2:
        ddx[0, 0] = -1.5 / dx
        ddx[0, 1] = 2.0 / dx
        ddx[0, 2] = -0.5 / dx
        ddx[-1, -1] = 1.5 / dx
        ddx[-1, -2] = -2.0 / dx
        ddx[-1, -3] = 0.5 / dx

        d2dx2[0, 0] = 1.0 / dx2
        d2dx2[0, 1] = -2.0 / dx2
        d2dx2[0, 2] = 1.0 / dx2
        d2dx2[-1, -1] = 1.0 / dx2
        d2dx2[-1, -2] = -2.0 / dx2
        d2dx2[-1, -3] = 1.0 / dx2

    elif scheme == 3:
        ddx[0, 0] = -1.0 / dx
        ddx[0, 1] = 1.0 / dx
        ddx[-1, -1] = 1.0 / dx
        ddx[-1, -2] = -1.0 / dx

        d2dx2[0, 0] = 1.0 / dx2
        d2dx2[0, 1] = -2.0 / dx2
        d2dx2[0, 2] = 1.0 / dx2
        d2dx2[-1, -1] = 1.0 / dx2
        d2dx2[-1, -2] = -2.0 / dx2
        d2dx2[-1, -3] = 1.0 / dx2

    elif scheme in {10, 11}:
        # Wrap-around terms:
        ddx[0, -1] = -(4.0 / 3) / (2 * dx)
        ddx[0, -2] = (1.0 / 6) / (2 * dx)
        ddx[1, -1] = (1.0 / 6) / (2 * dx)

        ddx[-1, 0] = (4.0 / 3) / (2 * dx)
        ddx[-1, 1] = -(1.0 / 6) / (2 * dx)
        ddx[-2, 0] = -(1.0 / 6) / (2 * dx)

        d2dx2[0, -1] = (4.0 / 3) / dx2
        d2dx2[0, -2] = -(1.0 / 12) / dx2
        d2dx2[1, -1] = -(1.0 / 12) / dx2

        d2dx2[-1, 0] = (4.0 / 3) / dx2
        d2dx2[-1, 1] = -(1.0 / 12) / dx2
        d2dx2[-2, 0] = -(1.0 / 12) / dx2

        # i = 1
        ddx[0, 2] = -1.0 / (12 * dx)
        ddx[0, 1] = 2.0 / (3 * dx)
        d2dx2[0, 2] = -1.0 / (12 * dx2)
        d2dx2[0, 1] = 4.0 / (3 * dx2)
        d2dx2[0, 0] = -5.0 / (2 * dx2)

        # i = 2
        ddx[1, 3] = -1.0 / (12 * dx)
        ddx[1, 2] = 2.0 / (3 * dx)
        ddx[1, 0] = -2.0 / (3 * dx)
        d2dx2[1, 3] = -1.0 / (12 * dx2)
        d2dx2[1, 2] = 4.0 / (3 * dx2)
        d2dx2[1, 1] = -5.0 / (2 * dx2)
        d2dx2[1, 0] = 4.0 / (3 * dx2)

        # i = N
        ddx[-1, -2] = -2.0 / (3 * dx)
        ddx[-1, -3] = 1.0 / (12 * dx)
        d2dx2[-1, -1] = -5.0 / (2 * dx2)
        d2dx2[-1, -2] = 4.0 / (3 * dx2)
        d2dx2[-1, -3] = -1.0 / (12 * dx2)

        # i = N-1
        ddx[-2, -1] = 2.0 / (3 * dx)
        ddx[-2, -3] = -2.0 / (3 * dx)
        ddx[-2, -4] = 1.0 / (12 * dx)
        d2dx2[-2, -1] = 4.0 / (3 * dx2)
        d2dx2[-2, -2] = -5.0 / (2 * dx2)
        d2dx2[-2, -3] = 4.0 / (3 * dx2)
        d2dx2[-2, -4] = -1.0 / (12 * dx2)

    elif scheme == 12:
        ddx[0, 0] = -25.0 / (12 * dx)
        ddx[0, 1] = 4.0 / dx
        ddx[0, 2] = -3.0 / dx
        ddx[0, 3] = 4.0 / (3 * dx)
        ddx[0, 4] = -1.0 / (4 * dx)

        ddx[1, 0] = -1.0 / (4 * dx)
        ddx[1, 1] = -5.0 / (6 * dx)
        ddx[1, 2] = 3.0 / (2 * dx)
        ddx[1, 3] = -1.0 / (2 * dx)
        ddx[1, 4] = 1.0 / (12 * dx)

        ddx[-1, -1] = 25.0 / (12 * dx)
        ddx[-1, -2] = -4.0 / dx
        ddx[-1, -3] = 3.0 / dx
        ddx[-1, -4] = -4.0 / (3 * dx)
        ddx[-1, -5] = 1.0 / (4 * dx)

        ddx[-2, -1] = 1.0 / (4 * dx)
        ddx[-2, -2] = 5.0 / (6 * dx)
        ddx[-2, -3] = -3.0 / (2 * dx)
        ddx[-2, -4] = 1.0 / (2 * dx)
        ddx[-2, -5] = -1.0 / (12 * dx)

        d2dx2[0, 0] = 35.0 / (12 * dx2)
        d2dx2[0, 1] = -26.0 / (3 * dx2)
        d2dx2[0, 2] = 19.0 / (2 * dx2)
        d2dx2[0, 3] = -14.0 / (3 * dx2)
        d2dx2[0, 4] = 11.0 / (12 * dx2)

        d2dx2[1, 0] = 11.0 / (12 * dx2)
        d2dx2[1, 1] = -5.0 / (3 * dx2)
        d2dx2[1, 2] = 1.0 / (2 * dx2)
        d2dx2[1, 3] = 1.0 / (3 * dx2)
        d2dx2[1, 4] = -1.0 / (12 * dx2)

        d2dx2[-1, -1] = 35.0 / (12 * dx2)
        d2dx2[-1, -2] = -26.0 / (3 * dx2)
        d2dx2[-1, -3] = 19.0 / (2 * dx2)
        d2dx2[-1, -4] = -14.0 / (3 * dx2)
        d2dx2[-1, -5] = 11.0 / (12 * dx2)

        d2dx2[-2, -1] = 11.0 / (12 * dx2)
        d2dx2[-2, -2] = -5.0 / (3 * dx2)
        d2dx2[-2, -3] = 1.0 / (2 * dx2)
        d2dx2[-2, -4] = 1.0 / (3 * dx2)
        d2dx2[-2, -5] = -1.0 / (12 * dx2)

    elif scheme == 13:
        ddx[0, 0] = -1.5 / dx
        ddx[0, 1] = 2.0 / dx
        ddx[0, 2] = -0.5 / dx

        ddx[-1, -1] = 1.5 / dx
        ddx[-1, -2] = -2.0 / dx
        ddx[-1, -3] = 0.5 / dx

        ddx[1, 0] = -1.0 / (3 * dx)
        ddx[1, 1] = -1.0 / (2 * dx)
        ddx[1, 2] = 1.0 / dx
        ddx[1, 3] = -1.0 / (6 * dx)

        ddx[-2, -1] = 1.0 / (3 * dx)
        ddx[-2, -2] = 1.0 / (2 * dx)
        ddx[-2, -3] = -1.0 / dx
        ddx[-2, -4] = 1.0 / (6 * dx)

        d2dx2[0, 0] = 1.0 / dx2
        d2dx2[0, 1] = -2.0 / dx2
        d2dx2[0, 2] = 1.0 / dx2

        d2dx2[-1, -1] = 1.0 / dx2
        d2dx2[-1, -2] = -2.0 / dx2
        d2dx2[-1, -3] = 1.0 / dx2

        d2dx2[1, 0] = 1.0 / dx2
        d2dx2[1, 1] = -2.0 / dx2
        d2dx2[1, 2] = 1.0 / dx2
        d2dx2[1, 3] = 0.0

        d2dx2[-2, -1] = 1.0 / dx2
        d2dx2[-2, -2] = -2.0 / dx2
        d2dx2[-2, -3] = 1.0 / dx2
        d2dx2[-2, -4] = 0.0

    elif scheme in {20, 21, 32, 42, 80, 81, 90, 91, 100, 101, 110, 111, 120, 121, 130, 131}:
        # Nothing additional to do (either already periodic-handled, or scheme defines
        # explicit zero rows at boundaries for aperiodic upwinding).
        pass

    elif scheme in {30, 31}:
        ddx[0, 0] = 1.0 / dx
        ddx[0, -1] = -1.0 / dx
        d2dx2[0, 0] = 1.0 / dx2
        d2dx2[1, 1] = 1.0 / dx2
        d2dx2[1, -1] = 1.0 / dx2
        d2dx2[0, -2] = 1.0 / dx2
        d2dx2[0, -1] = -2.0 / dx2
        d2dx2[1, 0] = -2.0 / dx2

    elif scheme in {40, 41}:
        ddx[-1, 0] = 1.0 / dx
        ddx[-1, -1] = -1.0 / dx
        d2dx2[-2, -2] = 1.0 / dx2
        d2dx2[-1, -1] = 1.0 / dx2
        d2dx2[-2, 0] = 1.0 / dx2
        d2dx2[-1, 1] = 1.0 / dx2
        d2dx2[-1, 0] = -2.0 / dx2
        d2dx2[-2, -1] = -2.0 / dx2

    elif scheme in {50, 51}:
        ddx[0, 0] = 1.5 / dx
        ddx[1, 1] = 1.5 / dx
        ddx[0, -1] = -2.0 / dx
        ddx[1, 0] = -2.0 / dx
        ddx[1, -1] = 1.0 / (2 * dx)
        ddx[0, -2] = 1.0 / (2 * dx)

        d2dx2[0, 0] = 1.0 / dx2
        d2dx2[1, 1] = 1.0 / dx2
        d2dx2[1, -1] = 1.0 / dx2
        d2dx2[0, -2] = 1.0 / dx2
        d2dx2[0, -1] = -2.0 / dx2
        d2dx2[1, 0] = -2.0 / dx2

    elif scheme == 52:
        ddx[1, 0] = -1.0 / dx
        ddx[1, 1] = 1.0 / dx

    elif scheme in {60, 61}:
        ddx[-2, 0] = -1.0 / (2 * dx)
        ddx[-1, 1] = -1.0 / (2 * dx)
        ddx[-1, 0] = 2.0 / dx
        ddx[-2, -1] = 2.0 / dx
        ddx[-2, -2] = -1.5 / dx
        ddx[-1, -1] = -1.5 / dx

        d2dx2[-2, -2] = 1.0 / dx2
        d2dx2[-1, -1] = 1.0 / dx2
        d2dx2[-2, 0] = 1.0 / dx2
        d2dx2[-1, 1] = 1.0 / dx2
        d2dx2[-1, 0] = -2.0 / dx2
        d2dx2[-2, -1] = -2.0 / dx2

    elif scheme == 62:
        ddx[-2, -2] = -1.0 / dx
        ddx[-2, -1] = 1.0 / dx

    elif scheme == 82:
        ddx[1, 1] = 1.0 / dx
        ddx[1, 0] = -1.0 / dx
        ddx[-1, -1] = 1.5 / dx
        ddx[-1, -2] = -2.0 / dx
        ddx[-1, -3] = 1.0 / (2 * dx)

    elif scheme == 92:
        ddx[-2, -2] = -1.0 / dx
        ddx[-2, -1] = 1.0 / dx
        ddx[0, 0] = -1.5 / dx
        ddx[0, 1] = 2.0 / dx
        ddx[0, 2] = -1.0 / (2 * dx)

    elif scheme == 102:
        ddx[1, 1] = 1.0 / dx
        ddx[1, 0] = -1.0 / dx
        ddx[2, 3] = 1.0 / (3 * dx)
        ddx[2, 2] = 1.0 / (2 * dx)
        ddx[2, 1] = -1.0 / dx
        ddx[2, 0] = 1.0 / (6 * dx)
        ddx[-1, -1] = 5.0 / (6 * dx)
        ddx[-1, -2] = -3.0 / (2 * dx)
        ddx[-1, -3] = 1.0 / (2 * dx)
        ddx[-1, -4] = -1.0 / (12 * dx)

    elif scheme == 112:
        ddx[-2, -2] = -1.0 / dx
        ddx[-2, -1] = 1.0 / dx
        ddx[0, 0] = -1.5 / dx
        ddx[0, 1] = 2.0 / dx
        ddx[0, 2] = -1.0 / (2 * dx)

    # else: schemes 122/132 already raise.

    return (
        jnp.asarray(x),
        jnp.asarray(weights),
        jnp.asarray(ddx),
        jnp.asarray(d2dx2),
    )
