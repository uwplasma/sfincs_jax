"""Differentiate through a v3 operator term (Er xiDot) using JAX.

This script demonstrates one of the core motivations for a JAX port: once the operator is written
as a pure JAX compute graph, we can obtain derivatives of physics-relevant scalars with respect to
inputs (here, the Er-driven factor dPhiHat/dpsiHat) essentially "for free".

We:
  1) Build a tiny v3 geometryScheme=4 configuration from an input.namelist fixture.
  2) Apply only the Er xiDot term to a random distribution `f`.
  3) Compute the gradient of a scalar objective with respect to dPhiHat/dpsiHat.
  4) Produce a small, publication-style figure.

Run:
  pip install -e ".[viz]"
  python examples/2_intermediate/11_autodiff_er_xidot_term.py
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

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
except Exception as e:  # pragma: no cover
    raise SystemExit('This example requires matplotlib. Install with: pip install -e ".[viz]"') from e

from sfincs_jax.collisionless_er import ErXiDotV3Operator, apply_er_xidot_v3
from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.v3 import geometry_from_namelist, grids_from_namelist


def _setup_mpl() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "-",
            "lines.linewidth": 2.0,
        }
    )


def _save(fig, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}.png", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--out-dir",
        default=str(Path(__file__).with_suffix("").parent / "figures"),
        help="Directory to write figures",
    )
    args = p.parse_args()

    _setup_mpl()
    out_dir = Path(args.out_dir)

    input_path = _REPO_ROOT / "tests" / "ref" / "er_xidot_1species_tiny.input.namelist"
    nml = read_sfincs_input(input_path)
    grids = grids_from_namelist(nml)
    geom = geometry_from_namelist(nml=nml, grids=grids)

    species = nml.group("speciesParameters")
    zs = species.get("ZS", [1])
    if not isinstance(zs, list):
        zs = [zs]
    n_species = len(zs)

    phys = nml.group("physicsParameters")
    alpha = float(phys.get("ALPHA", 1.0))
    delta = float(phys.get("DELTA", 0.0))

    # Build a fixed operator template and treat only dPhiHat/dpsiHat as a differentiable scalar input.
    def apply_term(dphi_hat_dpsi_hat: jnp.ndarray, f: jnp.ndarray) -> jnp.ndarray:
        op = ErXiDotV3Operator(
            alpha=jnp.asarray(alpha, dtype=jnp.float64),
            delta=jnp.asarray(delta, dtype=jnp.float64),
            dphi_hat_dpsi_hat=jnp.asarray(dphi_hat_dpsi_hat, dtype=jnp.float64),
            d_hat=geom.d_hat,
            b_hat=geom.b_hat,
            b_hat_sub_theta=geom.b_hat_sub_theta,
            b_hat_sub_zeta=geom.b_hat_sub_zeta,
            db_hat_dtheta=geom.db_hat_dtheta,
            db_hat_dzeta=geom.db_hat_dzeta,
            force0_radial_current=jnp.asarray(True),
            n_xi_for_x=grids.n_xi_for_x,
        )
        return apply_er_xidot_v3(op, f)

    key = jax.random.key(args.seed)
    f = jax.random.normal(
        key,
        shape=(n_species, int(grids.x.shape[0]), int(grids.n_xi), int(grids.theta.shape[0]), int(grids.zeta.shape[0])),
        dtype=jnp.float64,
    )

    def loss(dphi_hat_dpsi_hat: jnp.ndarray) -> jnp.ndarray:
        y = apply_term(dphi_hat_dpsi_hat, f)
        return 0.5 * jnp.sum(y * y)

    grad_loss = jax.grad(loss)
    loss_jit = jax.jit(loss)
    grad_jit = jax.jit(grad_loss)

    dphi0 = jnp.asarray(0.02, dtype=jnp.float64)
    g0 = float(grad_jit(dphi0))

    # Small scan for a figure + sanity check against finite differences.
    dphis = jnp.linspace(-0.05, 0.05, 41, dtype=jnp.float64)
    losses = jax.vmap(loss_jit)(dphis)
    grads = jax.vmap(grad_jit)(dphis)

    eps = 1e-6
    g_fd = float((loss_jit(dphi0 + eps) - loss_jit(dphi0 - eps)) / (2.0 * eps))

    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.6))
    ax[0].plot(np.asarray(dphis), np.asarray(losses))
    ax[0].set_title(r"Objective vs $d\hat\Phi/d\hat\psi$")
    ax[0].set_xlabel(r"$d\hat\Phi/d\hat\psi$")
    ax[0].set_ylabel(r"$\tfrac12\| \mathcal{L}_{E_r,\xi}[f]\|_2^2$")

    ax[1].plot(np.asarray(dphis), np.asarray(grads), label="JAX grad")
    ax[1].axvline(float(dphi0), color="k", alpha=0.25, linestyle="--")
    ax[1].set_title(r"Gradient (autodiff)")
    ax[1].set_xlabel(r"$d\hat\Phi/d\hat\psi$")
    ax[1].set_ylabel(r"$d\,\mathrm{loss}/d(d\hat\Phi/d\hat\psi)$")
    ax[1].legend(loc="best")

    _save(fig, out_dir, "autodiff_er_xidot_term")

    print(f"Wrote figures to {out_dir}")
    print(f"grad(loss) at dphi={float(dphi0):.3g}: autodiff={g0:.6e}  finite-diff={g_fd:.6e}  abs_err={abs(g0-g_fd):.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
