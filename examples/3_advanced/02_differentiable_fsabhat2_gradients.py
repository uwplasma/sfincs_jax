"""Differentiate a v3 geometry diagnostic with respect to Boozer harmonic amplitudes.

This example demonstrates a key benefit of a JAX port: gradients through geometry-derived
quantities are easy and fast.

We compute the flux-surface average <B^2> (in v3-normalized form `FSABHat2`) for the
`geometryScheme=4` simplified W7-X model, and take derivatives with respect to the
three harmonic amplitudes used in that model.

Run:
  python examples/3_advanced/02_differentiable_fsabhat2_gradients.py

Plotting requires:
  pip install -e ".[viz]"
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sfincs_jax.diagnostics import fsab_hat2
from sfincs_jax.geometry import boozer_geometry_scheme4
from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.v3 import grids_from_namelist


def main() -> int:
    input_path = _REPO_ROOT / "tests" / "ref" / "output_scheme4_1species_tiny.input.namelist"
    nml = read_sfincs_input(input_path)
    grids = grids_from_namelist(nml)

    amp0 = jnp.asarray([0.04645, -0.04351, -0.01902], dtype=jnp.float64)

    def f(a):
        geom = boozer_geometry_scheme4(theta=grids.theta, zeta=grids.zeta, harmonics_amp0=a)
        return fsab_hat2(grids=grids, geom=geom)

    val = float(f(amp0))
    g = jax.grad(f)(amp0)
    print(f"FSABHat2 = {val:.12g}")
    print("d(FSABHat2)/d(amp0) =", g)

    try:
        import matplotlib.pyplot as plt
    except Exception:
        return 0

    labels = [r"$(\ell,n)=(0,1)$", r"$(\ell,n)=(1,1)$", r"$(\ell,n)=(1,0)$"]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.bar(range(3), [float(x) for x in g], tick_label=labels)
    ax.set_title(r"Sensitivity of $\\langle \\hat B^2 \\rangle$ to Boozer harmonic amplitudes")
    ax.set_ylabel(r"$\\partial\\,\\mathrm{FSABHat2} / \\partial\\,a_k$")
    ax.grid(True, axis="y", alpha=0.25)

    fig_dir = Path(__file__).with_suffix("").parent / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "fsabhat2_gradients.png", bbox_inches="tight", dpi=200)
    fig.savefig(fig_dir / "fsabhat2_gradients.pdf", bbox_inches="tight")
    print(f"Wrote figures to {fig_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

