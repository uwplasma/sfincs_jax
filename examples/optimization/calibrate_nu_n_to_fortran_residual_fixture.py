"""Calibrate `nu_n` to match a frozen Fortran v3 residual fixture (autodiff + optax).

This example demonstrates a practical use of differentiability during the parity-port phase:
parameter identification. We treat the normalized collisionality parameter `nu_n` as a
differentiable scalar and fit it so that the JAX residual matches a frozen Fortran v3 residual.

The workflow is:
  1) read an upstream v3 input.namelist
  2) load frozen PETSc vectors from a Fortran v3 run: `stateVector` and `residual`
  3) minimize

       0.5 * || r_jax(nu_n; x_ref) - r_fortran ||^2

     using optax/Adam.

Requirements:
  pip install -e ".[opt]"

Optional (for figures):
  pip install -e ".[viz]"
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
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
except Exception:  # pragma: no cover
    mpl = None
    plt = None

from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.petsc_binary import read_petsc_vec
from sfincs_jax.v3_system import full_system_operator_from_namelist, residual_v3_full_system


def _setup_mpl() -> None:
    if mpl is None:
        return
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


def _default_input() -> Path:
    return Path(__file__).parents[2] / "tests" / "ref" / "pas_1species_PAS_noEr_tiny_scheme5.input.namelist"


def _default_statevector() -> Path:
    return Path(__file__).parents[2] / "tests" / "ref" / "pas_1species_PAS_noEr_tiny_scheme5.stateVector.petscbin"


def _default_residual() -> Path:
    return Path(__file__).parents[2] / "tests" / "ref" / "pas_1species_PAS_noEr_tiny_scheme5.residual.petscbin"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=str(_default_input()))
    p.add_argument("--statevector", default=str(_default_statevector()))
    p.add_argument("--residual", default=str(_default_residual()))
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-1)
    p.add_argument(
        "--out-dir",
        default=str(Path(__file__).with_suffix("").parent / "figures"),
        help="Directory to write figures (only if matplotlib is installed)",
    )
    args = p.parse_args()

    _setup_mpl()
    out_dir = Path(args.out_dir)

    nml = read_sfincs_input(Path(args.input))
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)

    x_ref = jnp.asarray(read_petsc_vec(Path(args.statevector)).values)
    r_ref = jnp.asarray(read_petsc_vec(Path(args.residual)).values)

    if op.fblock.pas is None:
        raise SystemExit("This example expects collisionOperator=1 (PAS), but op.fblock.pas is None.")

    nu_true = jnp.asarray(op.fblock.pas.nu_n, dtype=jnp.float64)

    def objective(nu_n: jnp.ndarray) -> jnp.ndarray:
        pas2 = replace(op.fblock.pas, nu_n=jnp.asarray(nu_n, dtype=jnp.float64))
        op2 = replace(op, fblock=replace(op.fblock, pas=pas2))
        r = residual_v3_full_system(op2, x_ref)
        d = r - r_ref
        return 0.5 * jnp.vdot(d, d)

    # Start away from the true value to show convergence.
    nu0 = jnp.where(nu_true == 0.0, jnp.asarray(1.0, dtype=jnp.float64), 1.7 * nu_true)

    opt = optax.adam(float(args.lr))
    nu = nu0
    opt_state = opt.init(nu)

    history: list[float] = []
    history_nu: list[float] = []

    @jax.jit
    def step(nu_n: jnp.ndarray, state):
        loss, g = jax.value_and_grad(objective)(nu_n)
        updates, state = opt.update(g, state, nu_n)
        nu_n = optax.apply_updates(nu_n, updates)
        return nu_n, state, loss

    for k in range(int(args.steps)):
        nu, opt_state, loss = step(nu, opt_state)
        history.append(float(loss))
        history_nu.append(float(nu))
        if k % 25 == 0 or k == int(args.steps) - 1:
            print(f"step {k:4d}  loss={float(loss):.3e}  nu_n={float(nu):.6g}")

    print("\nnu_n_true:", float(nu_true))
    print("nu_n_fit: ", float(nu))

    if plt is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(6.0, 3.2))
        ax.plot(history)
        ax.set_yscale("log")
        ax.set_xlabel("iteration")
        ax.set_ylabel(r"$\\frac{1}{2}\\|r_{jax}-r_{fortran}\\|^2$")
        ax.set_title("Calibrating $\\nu_n$ to a Fortran residual fixture")
        fig.tight_layout()
        fig.savefig(out_dir / "calibrate_nu_n_loss.png", bbox_inches="tight")
        fig.savefig(out_dir / "calibrate_nu_n_loss.pdf", bbox_inches="tight")

        fig, ax = plt.subplots(figsize=(6.0, 3.2))
        ax.plot(history_nu)
        ax.axhline(float(nu_true), color="k", linestyle="--", label="true")
        ax.set_xlabel("iteration")
        ax.set_ylabel(r"$\\nu_n$")
        ax.set_title("Recovered $\\nu_n$")
        ax.legend(loc="best", frameon=True)
        fig.tight_layout()
        fig.savefig(out_dir / "calibrate_nu_n_trace.png", bbox_inches="tight")
        fig.savefig(out_dir / "calibrate_nu_n_trace.pdf", bbox_inches="tight")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

