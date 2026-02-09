"""Generate publication-style figures for magnetic drift terms (v3).

This script produces a small set of polished figures that illustrate:
  - The v3 Boozer field magnitude BHat(θ, ζ) for a `geometryScheme=11` W7-X equilibrium
  - The magnetic-drift geometric factors used for upwinding
  - The upwind-selector mask over (θ, ζ)
  - Representative coefficient fields that multiply angular derivatives and |ΔL|=2 couplings

Outputs are written as both PNG and PDF in `examples/publication_figures/figures/`.

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


def _load_geom_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    out: dict[str, np.ndarray] = {}
    for k in data.files:
        arr = np.asarray(data[k])
        if arr.ndim == 2:
            arr = arr.T  # see tests/test_magnetic_drifts_parity.py for details
        out[k] = arr
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=None, help="Path to an input.namelist (default: repo fixture)")
    p.add_argument("--geom-npz", default=None, help="Path to a geometry .npz (default: repo fixture)")
    p.add_argument(
        "--out-dir",
        default=str(Path(__file__).with_suffix("").parent / "figures"),
        help="Directory to write figures",
    )
    p.add_argument("--ix", type=int, default=1, help="x-grid index to visualize for coefficient fields")
    p.add_argument("--l", type=int, default=1, help="Legendre index L to visualize for coefficient fields")
    args = p.parse_args()

    _setup_mpl()
    out_dir = Path(args.out_dir)

    if args.input is None:
        input_path = _REPO_ROOT / "tests" / "ref" / "magdrift_1species_tiny.input.namelist"
    else:
        input_path = Path(args.input)

    if args.geom_npz is None:
        geom_path = _REPO_ROOT / "tests" / "ref" / "magdrift_1species_tiny.geometry.npz"
    else:
        geom_path = Path(args.geom_npz)

    nml = read_sfincs_input(input_path)
    grids = grids_from_namelist(nml)
    geom = _load_geom_npz(geom_path)

    species = nml.group("speciesParameters")
    phys = nml.group("physicsParameters")
    delta = float(phys.get("DELTA", 0.0))
    t_hat = float(np.atleast_1d(np.asarray(species["THATS"], dtype=np.float64))[0])
    z = float(np.atleast_1d(np.asarray(species["ZS"], dtype=np.float64))[0])

    theta = np.asarray(grids.theta)
    zeta = np.asarray(grids.zeta)
    th2, ze2 = np.meshgrid(theta, zeta, indexing="ij")

    bhat = np.asarray(geom["BHat"], dtype=np.float64)
    dhat = np.asarray(geom["DHat"], dtype=np.float64)

    # Magnetic-drift geometric factors for scheme=1, matching sfincs_jax.magnetic_drifts.
    gf1_theta = np.asarray(geom["BHat_sub_zeta"]) * np.asarray(geom["dBHatdpsiHat"]) - np.asarray(geom["BHat_sub_psi"]) * np.asarray(
        geom["dBHatdzeta"]
    )
    gf2_theta = 2.0 * bhat * (np.asarray(geom["dBHat_sub_psi_dzeta"]) - np.asarray(geom["dBHat_sub_zeta_dpsiHat"]))

    # The v3 upwind selector uses sign(gf1 * DHat(1,1) / Z).
    use_plus_theta = (gf1_theta * float(dhat[0, 0]) / z) > 0

    # Representative coefficient that multiplies ddtheta and the L->L+2 coupling for the theta drift term.
    ix = int(args.ix)
    l = int(args.l)
    x = float(np.asarray(grids.x)[ix])
    coupling_lp2 = (l + 2.0) * (l + 1.0) / ((2.0 * l + 5.0) * (2.0 * l + 3.0))
    pref = (delta * t_hat) * dhat * (x * x) / (2.0 * z * (bhat**3))
    coeff_theta_lp2 = pref * (gf1_theta + gf2_theta) * coupling_lp2

    # Figure 1: BHat(θ, ζ)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    im = ax.pcolormesh(th2, ze2, bhat, shading="auto")
    ax.set_title(r"SFINCS v3 geometryScheme=11 fixture: $\hat B(\theta,\zeta)$")
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$\zeta$")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(r"$\hat B$")
    _save(fig, out_dir, "magdrift_bhat")

    # Figure 2: geometricFactor1 (theta drift)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    im = ax.pcolormesh(th2, ze2, gf1_theta, shading="auto")
    ax.set_title(r"Magnetic drift: $g_1^{(\theta)}(\theta,\zeta)$")
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$\zeta$")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(r"$g_1^{(\theta)}$")
    _save(fig, out_dir, "magdrift_gf1_theta")

    # Figure 3: upwind selector mask
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    im = ax.pcolormesh(th2, ze2, use_plus_theta.astype(float), shading="auto", vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_title(r"Upwind selector for $d/d\theta$: $\mathbf{1}[g_1^{(\theta)} \hat D(1,1)/Z > 0]$")
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$\zeta$")
    cb = fig.colorbar(im, ax=ax, ticks=[0.0, 1.0])
    cb.set_label("use_plus")
    _save(fig, out_dir, "magdrift_upwind_mask_theta")

    # Figure 4: representative |ΔL|=2 coefficient field
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    im = ax.pcolormesh(th2, ze2, coeff_theta_lp2, shading="auto")
    ax.set_title(rf"Theta drift |ΔL|=2 coefficient (ix={ix}, L→L+2 with L={l})")
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$\zeta$")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(r"$C_{\theta, L\to L+2}(\theta,\zeta)$")
    _save(fig, out_dir, "magdrift_coeff_theta_offdiag2")

    print(f"Wrote figures to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
