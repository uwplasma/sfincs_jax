"""Generate publication-style figures for the collisionless Er terms (v3).

This script produces a small set of polished figures that illustrate:
  - The simplified W7-X Boozer field magnitude BHat(θ, ζ) (geometryScheme=4)
  - The Er-driven coefficients appearing in the v3 xiDot and xDot terms
  - The ExB drift coefficient multiplying the v3 d/dtheta term (for this geometry, IHat=0 so d/dzeta vanishes)

Outputs are written as both PNG and PDF in `examples/2_intermediate/figures/`.

Requirements:
  pip install -e ".[viz]"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
except Exception as e:  # pragma: no cover
    raise SystemExit('This example requires matplotlib. Install with: pip install -e ".[viz]"') from e

from sfincs_jax.geometry import boozer_geometry_scheme4
from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.v3 import grids_from_namelist


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


def _dphi_hat_dpsi_hat_from_er_scheme4(er: float) -> float:
    psi_a_hat = -0.384935
    a_hat = 0.5109
    psi_n = 0.25
    return float(a_hat / (2.0 * psi_a_hat * np.sqrt(psi_n)) * (-er))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=None, help="Path to an input.namelist (default: repo fixture)")
    p.add_argument(
        "--out-dir",
        default=str(Path(__file__).with_suffix("").parent / "figures"),
        help="Directory to write figures",
    )
    args = p.parse_args()

    _setup_mpl()
    out_dir = Path(args.out_dir)

    if args.input is None:
        input_path = Path(__file__).parents[2] / "tests" / "ref" / "er_xidot_1species_tiny.input.namelist"
    else:
        input_path = Path(args.input)

    nml = read_sfincs_input(input_path)
    grids = grids_from_namelist(nml)
    geom = boozer_geometry_scheme4(theta=grids.theta, zeta=grids.zeta)

    phys = nml.group("physicsParameters")
    alpha = float(phys.get("ALPHA", 1.0))
    delta = float(phys.get("DELTA", 0.0))
    er = float(phys.get("ER", 0.0))
    dphi = _dphi_hat_dpsi_hat_from_er_scheme4(er)

    theta = np.asarray(grids.theta)
    zeta = np.asarray(grids.zeta)
    th2, ze2 = np.meshgrid(theta, zeta, indexing="ij")

    bhat = np.asarray(geom.b_hat)

    # Coefficient for the xiDot Er term (force0RadialCurrentInEquilibrium=.true.).
    temp_xi = np.asarray(geom.b_hat_sub_zeta) * np.asarray(geom.db_hat_dtheta) - np.asarray(geom.b_hat_sub_theta) * np.asarray(
        geom.db_hat_dzeta
    )
    f_xi = (alpha * delta * dphi / 4.0) * np.asarray(geom.d_hat) * temp_xi / (bhat**3)

    # Coefficient for the xDot Er term (xDotFactor) used by the default implementation (force0 => xDotFactor2=0).
    temp_x = np.asarray(geom.b_hat_sub_theta) * np.asarray(geom.db_hat_dzeta) - np.asarray(geom.b_hat_sub_zeta) * np.asarray(
        geom.db_hat_dtheta
    )
    f_x = (-(alpha * delta * dphi) / 4.0) * np.asarray(geom.d_hat) * temp_x / (bhat**3)

    # ExB drift coefficient multiplying d/dtheta (non-DKES ExB drift).
    # For geometryScheme=4, BHat_sub_theta = IHat = 0, so the ExB d/dzeta term is identically zero.
    f_exb_theta = (alpha * delta * dphi / 2.0) * np.asarray(geom.d_hat) * np.asarray(geom.b_hat_sub_zeta) / (bhat**2)

    # Figure 1: BHat(θ, ζ)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    im = ax.pcolormesh(th2, ze2, bhat, shading="auto")
    ax.set_title(r"SFINCS v3 geometryScheme=4: $\hat B(\theta,\zeta)$")
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$\zeta$")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(r"$\hat B$")
    _save(fig, out_dir, "er_terms_bhat")

    # Figure 2: xiDot coefficient
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    im = ax.pcolormesh(th2, ze2, f_xi, shading="auto")
    ax.set_title(r"Er xiDot coefficient $F_{\xi}(\theta,\zeta)$")
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$\zeta$")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(r"$F_{\xi}$")
    _save(fig, out_dir, "er_terms_xidot_coeff")

    # Figure 3: xDot coefficient
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    im = ax.pcolormesh(th2, ze2, f_x, shading="auto")
    ax.set_title(r"Er xDot coefficient $F_x(\theta,\zeta)$ (force0)")
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$\zeta$")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(r"$F_x$")
    _save(fig, out_dir, "er_terms_xdot_coeff")

    # Figure 4: ExB drift d/dtheta coefficient
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    im = ax.pcolormesh(th2, ze2, f_exb_theta, shading="auto")
    ax.set_title(r"ExB $d/d\theta$ coefficient $F_{E\\times B,\\theta}(\theta,\zeta)$")
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$\zeta$")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(r"$F_{E\\times B,\\theta}$")
    _save(fig, out_dir, "exb_terms_theta_coeff")

    print(f"Wrote figures to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
