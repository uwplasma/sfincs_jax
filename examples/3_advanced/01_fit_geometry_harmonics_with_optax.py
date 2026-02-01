"""Fit geometryScheme=4 harmonic amplitudes using optax.

This example requires:

```bash
pip install -e \"sfincs_jax[opt]\"
```

It demonstrates how a differentiable geometry model can be optimized with standard JAX
optimizers (optax). The *real* use case for SFINCS is analogous, but with transport
targets and with the kinetic solve inside the objective.
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

try:
    import optax
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "This example requires optax. Install with: pip install -e \"sfincs_jax[opt]\""
    ) from e

from sfincs_jax.geometry import boozer_geometry_scheme4


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=5e-2)
    args = p.parse_args()

    n_theta = 41
    n_zeta = 41
    theta = jnp.linspace(0.0, 2 * jnp.pi, n_theta, endpoint=False, dtype=jnp.float64)
    zeta = jnp.linspace(0.0, 2 * jnp.pi / 5, n_zeta, endpoint=False, dtype=jnp.float64)

    amps_true = jnp.asarray([0.04645, -0.04351, -0.01902], dtype=jnp.float64)
    amps_target = amps_true + jnp.asarray([0.01, -0.01, 0.005], dtype=jnp.float64)
    bhat_target = boozer_geometry_scheme4(theta=theta, zeta=zeta, harmonics_amp0=amps_target).b_hat

    def loss_fn(amps: jnp.ndarray) -> jnp.ndarray:
        bhat = boozer_geometry_scheme4(theta=theta, zeta=zeta, harmonics_amp0=amps).b_hat
        return jnp.mean((bhat - bhat_target) ** 2)

    opt = optax.adam(args.lr)
    amps = jnp.zeros_like(amps_true)
    opt_state = opt.init(amps)

    @jax.jit
    def step(amps: jnp.ndarray, opt_state):
        loss, g = jax.value_and_grad(loss_fn)(amps)
        updates, opt_state = opt.update(g, opt_state, amps)
        amps = optax.apply_updates(amps, updates)
        return amps, opt_state, loss

    for k in range(int(args.steps)):
        amps, opt_state, loss = step(amps, opt_state)
        if k % 20 == 0 or k == int(args.steps) - 1:
            print(f"step {k:4d}  loss={float(loss):.3e}  amps={amps}")

    print("\nTarget amps:", amps_target)
    print("Fit amps:   ", amps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
