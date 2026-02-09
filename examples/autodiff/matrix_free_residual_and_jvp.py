"""Matrix-free residual + Jacobian-vector products (JVP) for the v3 F-block.

This example demonstrates a key architectural milestone for a full SFINCS v3 port:

- Express the discrete problem as a residual function r(x) = A x - b.
- Apply the Jacobian matrix-free using JVPs (or explicit matvecs for linear pieces).
- Use `jax.grad`/`jax.jvp` to get sensitivities without assembling sparse matrices.

Today the residual is linear in x because we only consider the distribution-function block.
The same structure extends naturally to nonlinear residuals (e.g. when including Phi1 or
additional constraints), while keeping the Jacobian application matrix-free.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.residual import V3FBlockLinearSystem
from sfincs_jax.v3_fblock import fblock_operator_from_namelist
from sfincs_jax.v3_fblock import matvec_v3_fblock_flat


def _default_input() -> Path:
    return Path(__file__).parents[2] / "tests" / "ref" / "pas_1species_PAS_noEr_tiny.input.namelist"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=str(_default_input()))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--plot", action="store_true", help="Write a simple PNG (requires matplotlib).")
    args = p.parse_args()

    nml = read_sfincs_input(Path(args.input))
    op = fblock_operator_from_namelist(nml=nml, identity_shift=0.0)

    rng = np.random.default_rng(args.seed)
    x0 = jnp.asarray(rng.normal(size=(op.flat_size,)).astype(np.float64))
    v = jnp.asarray(rng.normal(size=(op.flat_size,)).astype(np.float64))

    # Choose b so that the "true" solution is x0 (just for demonstration).
    b = matvec_v3_fblock_flat(op, x0)

    sys = V3FBlockLinearSystem(op=op, b_flat=b)

    # Residual and JVP:
    r0, jv = sys.jvp(x0, v)

    # A small objective to demonstrate reverse-mode AD through the residual:
    #   phi(x) = 0.5 * ||r(x)||^2
    def phi(x):
        r = sys.residual(x)
        return 0.5 * jnp.vdot(r, r)

    grad_phi = jax.grad(phi)(x0)

    # Finite-difference check of directional derivative:
    eps = 1e-6
    fd = float((phi(x0 + eps * v) - phi(x0 - eps * v)) / (2 * eps))
    ad = float(jnp.vdot(grad_phi, v))

    print(f"n={op.flat_size}")
    print(f"||r(x0)||_2 = {float(jnp.linalg.norm(r0)):.3e}  (should be ~0)")
    print(f"||J v||_2   = {float(jnp.linalg.norm(jv)):.3e}")
    print(f"directional derivative: finite-diff={fd:.6e}  autodiff={ad:.6e}  abs_err={abs(fd-ad):.3e}")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except Exception as e:  # noqa: BLE001
            raise SystemExit(f"--plot requested but matplotlib is unavailable: {e}")

        # Plot phi(x0 + t v) for a small range of t to visualize smoothness.
        ts = np.linspace(-5e-3, 5e-3, 101)
        vals = np.array([float(phi(x0 + float(t) * v)) for t in ts])
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        ax.plot(ts, vals, lw=2)
        ax.set_xlabel("t")
        ax.set_ylabel(r"$\phi(x_0 + t v)$")
        ax.set_title("Matrix-free residual objective (autodiff-ready)")
        ax.grid(True, alpha=0.3)
        out = Path("residual_objective.png")
        fig.tight_layout()
        fig.savefig(out, dpi=200)
        print(f"Wrote {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
