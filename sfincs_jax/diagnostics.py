from __future__ import annotations

from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)

import jax.numpy as jnp

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

