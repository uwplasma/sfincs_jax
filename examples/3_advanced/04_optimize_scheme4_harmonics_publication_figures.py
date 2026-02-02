"""Optimize `geometryScheme=4` harmonic amplitudes with optax and write publication-ready figures.

This example is intentionally small and pedagogical. It demonstrates a pattern that becomes
important once the *full* SFINCS solve is differentiable:

  - build a differentiable objective from a plasma/geometry model
  - use autodiff to get gradients
  - run a gradient-based optimization loop (here: optax/Adam)
  - produce figures suitable for a report/paper

Here we use a simple *synthetic* target: a reference `BHat(θ,ζ)` field from the same
`geometryScheme=4` model with slightly perturbed harmonic amplitudes. In real applications,
the target might come from a VMEC equilibrium, a Boozer transform, or a design goal metric
(e.g. reducing a neoclassical proxy).

Requirements:
  pip install -e ".[opt,viz]"
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
    import optax
except Exception as e:  # pragma: no cover
    raise SystemExit('This example requires optax. Install with: pip install -e ".[opt]"') from e

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
except Exception as e:  # pragma: no cover
    raise SystemExit('This example requires matplotlib. Install with: pip install -e ".[viz]"') from e

from sfincs_jax.geometry import boozer_geometry_scheme4


def _setup_mpl() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
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
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=5e-2)
    p.add_argument(
        "--out-dir",
        default=str(Path(__file__).with_suffix("").parent / "figures"),
        help="Directory to write figures",
    )
    args = p.parse_args()

    _setup_mpl()
    out_dir = Path(args.out_dir)

    # Use an odd grid like v3 defaults (avoids Nyquist subtleties).
    n_theta = 61
    n_zeta = 61
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, n_theta, endpoint=False, dtype=jnp.float64)
    zeta = jnp.linspace(0.0, 2.0 * jnp.pi / 5.0, n_zeta, endpoint=False, dtype=jnp.float64)

    amps_nominal = jnp.asarray([0.04645, -0.04351, -0.01902], dtype=jnp.float64)
    amps_target = amps_nominal + jnp.asarray([0.010, -0.008, 0.004], dtype=jnp.float64)

    bhat_target = boozer_geometry_scheme4(theta=theta, zeta=zeta, harmonics_amp0=amps_target).b_hat

    def loss_fn(amps: jnp.ndarray) -> jnp.ndarray:
        bhat = boozer_geometry_scheme4(theta=theta, zeta=zeta, harmonics_amp0=amps).b_hat
        # MSE fit to the target field.
        return jnp.mean((bhat - bhat_target) ** 2)

    opt = optax.adam(float(args.lr))

    amps = jnp.zeros_like(amps_nominal)
    opt_state = opt.init(amps)

    history_loss: list[float] = []
    history_amps: list[np.ndarray] = []

    @jax.jit
    def step(a: jnp.ndarray, s):
        loss, g = jax.value_and_grad(loss_fn)(a)
        updates, s = opt.update(g, s, a)
        a = optax.apply_updates(a, updates)
        return a, s, loss

    for k in range(int(args.steps)):
        amps, opt_state, loss = step(amps, opt_state)
        history_loss.append(float(loss))
        history_amps.append(np.asarray(amps))
        if k % 50 == 0 or k == int(args.steps) - 1:
            print(f"step {k:4d}  loss={float(loss):.3e}  amps={np.asarray(amps)}")

    bhat_fit = boozer_geometry_scheme4(theta=theta, zeta=zeta, harmonics_amp0=amps).b_hat
    bhat_nom = boozer_geometry_scheme4(theta=theta, zeta=zeta, harmonics_amp0=amps_nominal).b_hat

    print("\nNominal amps:", np.asarray(amps_nominal))
    print("Target amps: ", np.asarray(amps_target))
    print("Fit amps:    ", np.asarray(amps))

    th = np.asarray(theta)
    ze = np.asarray(zeta)
    bh_nom = np.asarray(bhat_nom)
    bh_tgt = np.asarray(bhat_target)
    bh_fit = np.asarray(bhat_fit)

    # Figure 1: loss curve
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    ax.plot(history_loss)
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("MSE loss")
    ax.set_title("Optax fit of geometryScheme=4 harmonics")
    _save(fig, out_dir, "scheme4_optax_loss")

    # Figure 2: lineouts at zeta=0 for nominal/target/fit
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    iz0 = 0
    ax.plot(th, bh_nom[:, iz0], label="nominal")
    ax.plot(th, bh_tgt[:, iz0], label="target", linestyle="--")
    ax.plot(th, bh_fit[:, iz0], label="fit")
    ax.set_xlabel(r"$\\theta$")
    ax.set_ylabel(r"$\\hat B$")
    ax.set_title(r"Lineout at $\\zeta=0$")
    ax.legend(loc="best", frameon=True)
    _save(fig, out_dir, "scheme4_optax_lineout_zeta0")

    # Figure 3: heatmaps of target and fit error
    def heat(ax, data, title: str) -> None:
        im = ax.pcolormesh(ze, th, data, shading="auto")
        ax.set_xlabel(r"$\\zeta$")
        ax.set_ylabel(r"$\\theta$")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, shrink=0.9)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.4), constrained_layout=True)
    heat(axes[0], bh_tgt, "Target $\\hat B(\\theta,\\zeta)$")
    heat(axes[1], (bh_fit - bh_tgt), "Fit error (fit - target)")
    fig.savefig(out_dir / "scheme4_optax_heatmaps.png", bbox_inches="tight")
    fig.savefig(out_dir / "scheme4_optax_heatmaps.pdf", bbox_inches="tight")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

