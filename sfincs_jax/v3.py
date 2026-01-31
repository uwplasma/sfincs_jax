from __future__ import annotations

import math
from dataclasses import dataclass

from jax import config as _jax_config
_jax_config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from .geometry import BoozerGeometry, boozer_geometry_scheme4
from .grids import uniform_diff_matrices
from .namelist import Namelist
from .xgrid import XGrid, make_x_grid


@dataclass(frozen=True)
class V3Grids:
    theta: jnp.ndarray
    zeta: jnp.ndarray
    x: jnp.ndarray

    theta_weights: jnp.ndarray
    zeta_weights: jnp.ndarray
    x_weights: jnp.ndarray


def _get_int(group: dict, key: str, default: int) -> int:
    v = group.get(key.upper(), default)
    if isinstance(v, list):
        v = v[0] if v else default
    return int(v)


def _get_float(group: dict, key: str, default: float) -> float:
    v = group.get(key.upper(), default)
    if isinstance(v, list):
        v = v[0] if v else default
    return float(v)


def grids_from_namelist(nml: Namelist) -> V3Grids:
    """Construct v3 grids using the same defaults as the Fortran code (where implemented)."""
    res = nml.group("resolutionParameters")
    other = nml.group("otherNumericalParameters")
    geom = nml.group("geometryParameters")

    ntheta = _get_int(res, "Ntheta", 15)
    nzeta = _get_int(res, "Nzeta", 15)
    nx = _get_int(res, "Nx", 5)

    # SFINCS v3 defaults:
    force_odd = True
    if force_odd:
        if ntheta % 2 == 0:
            ntheta += 1
        if nzeta % 2 == 0:
            nzeta += 1

    theta_derivative_scheme = _get_int(other, "thetaDerivativeScheme", 2)
    zeta_derivative_scheme = _get_int(other, "zetaDerivativeScheme", 2)
    x_grid_scheme = _get_int(other, "xGridScheme", 5)
    x_grid_k = _get_float(other, "xGrid_k", 0.0)

    geometry_scheme = _get_int(geom, "geometryScheme", -1)
    if geometry_scheme == 4:
        n_periods = 5
    else:
        raise NotImplementedError(
            "Only geometryScheme=4 is implemented in sfincs_jax so far."
        )

    # theta grid
    theta_scheme_map = {0: 20, 1: 0, 2: 10}
    theta_scheme = theta_scheme_map.get(theta_derivative_scheme)
    if theta_scheme is None:
        raise ValueError(f"Invalid thetaDerivativeScheme={theta_derivative_scheme}")
    theta, theta_weights, _, _ = uniform_diff_matrices(
        n=ntheta, x_min=0.0, x_max=2 * math.pi, scheme=theta_scheme
    )

    # zeta grid
    zeta_max = 2 * math.pi / n_periods
    zeta_scheme_map = {0: 20, 1: 0, 2: 10}
    zeta_scheme = zeta_scheme_map.get(zeta_derivative_scheme)
    if zeta_scheme is None:
        raise ValueError(f"Invalid zetaDerivativeScheme={zeta_derivative_scheme}")
    if nzeta == 1:
        zeta = jnp.asarray(np.array([0.0], dtype=np.float64))
        zeta_weights = jnp.asarray(np.array([2 * math.pi], dtype=np.float64))
    else:
        zeta, zeta_weights, _, _ = uniform_diff_matrices(
            n=nzeta, x_min=0.0, x_max=zeta_max, scheme=zeta_scheme
        )
        zeta_weights = zeta_weights * n_periods

    # x grid
    if x_grid_scheme in {1, 5}:
        include_x0 = False
    elif x_grid_scheme in {2, 6}:
        include_x0 = True
    else:
        raise NotImplementedError(
            f"Only xGridScheme in {{1,2,5,6}} is implemented (got {x_grid_scheme})."
        )

    xg: XGrid = make_x_grid(n=nx, k=x_grid_k, include_point_at_x0=include_x0)
    x = jnp.asarray(xg.x)
    x_weights = jnp.asarray(xg.dx_weights(x_grid_k))

    return V3Grids(
        theta=theta,
        zeta=zeta,
        x=x,
        theta_weights=theta_weights,
        zeta_weights=zeta_weights,
        x_weights=x_weights,
    )


def geometry_from_namelist(*, nml: Namelist, grids: V3Grids) -> BoozerGeometry:
    geom = nml.group("geometryParameters")
    geometry_scheme = _get_int(geom, "geometryScheme", -1)
    if geometry_scheme != 4:
        raise NotImplementedError("Only geometryScheme=4 is implemented so far.")
    return boozer_geometry_scheme4(theta=grids.theta, zeta=grids.zeta)
