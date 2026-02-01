"""Differentiate a simple geometry-based scalar with respect to harmonic amplitudes.

This is a tiny example of what becomes possible once SFINCS is fully ported to JAX:
sensitivity analysis and gradient-based optimization w.r.t. geometry parameters.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sfincs_jax.geometry import boozer_geometry_scheme4


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n-theta", type=int, default=31)
    p.add_argument("--n-zeta", type=int, default=31)
    args = p.parse_args()

    theta = jnp.linspace(0.0, 2 * jnp.pi, args.n_theta, endpoint=False, dtype=jnp.float64)
    zeta = jnp.linspace(0.0, 2 * jnp.pi / 5, args.n_zeta, endpoint=False, dtype=jnp.float64)

    def objective(harmonics_amp0: jnp.ndarray) -> jnp.ndarray:
        geom = boozer_geometry_scheme4(theta=theta, zeta=zeta, harmonics_amp0=harmonics_amp0)
        # A made-up scalar objective: mean(BHat^2).
        return jnp.mean(geom.b_hat**2)

    amps0 = jnp.asarray([0.04645, -0.04351, -0.01902], dtype=jnp.float64)
    g = jax.grad(objective)(amps0)
    g_jit = jax.jit(jax.grad(objective))(amps0)

    print("amps0 =", amps0)
    print("grad  =", g)
    print("grad(jit) =", g_jit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
