"""Compare a `sfincs_jax` output file against a frozen Fortran v3 output fixture.

This example demonstrates:
  - writing `sfincsOutput.h5` using `sfincs_jax`
  - comparing dataset values against a reference Fortran `sfincsOutput.h5`
  - (optional) plotting a heatmap of BHat differences

Run:
  python examples/parity/07_output_parity_vs_fortran_fixture.py

Plotting requires:
  pip install -e ".[viz]"
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sfincs_jax.compare import compare_sfincs_outputs
from sfincs_jax.io import read_sfincs_h5, write_sfincs_jax_output_h5


def main() -> int:
    input_path = _REPO_ROOT / "tests" / "ref" / "output_scheme4_1species_tiny.input.namelist"
    fortran_path = _REPO_ROOT / "tests" / "ref" / "output_scheme4_1species_tiny.sfincsOutput.h5"
    out_dir = Path(__file__).with_suffix("").parent / "output"
    out_path = out_dir / "sfincsOutput_jax.h5"
    out_dir.mkdir(parents=True, exist_ok=True)

    write_sfincs_jax_output_h5(input_namelist=input_path, output_path=out_path)

    keys = [
        "Nspecies",
        "Ntheta",
        "Nzeta",
        "Nxi",
        "NL",
        "Nx",
        "theta",
        "zeta",
        "x",
        "Nxi_for_x",
        "geometryScheme",
        "Delta",
        "alpha",
        "nu_n",
        "Er",
        "dPhiHatdpsiHat",
        "NPeriods",
        "B0OverBBar",
        "iota",
        "GHat",
        "IHat",
        "VPrimeHat",
        "FSABHat2",
        "DHat",
        "BHat",
        "dBHatdtheta",
        "dBHatdzeta",
        "BHat_sub_theta",
        "BHat_sub_zeta",
        "BHat_sup_theta",
        "BHat_sup_zeta",
    ]

    results = compare_sfincs_outputs(a_path=out_path, b_path=fortran_path, keys=keys, rtol=0, atol=1e-12)
    bad = [r for r in results if not r.ok]
    if bad:
        print("FAIL:")
        for r in bad:
            print(f"  {r.key}: max_abs={r.max_abs:.3e} max_rel={r.max_rel:.3e}")
        return 2
    print(f"OK: {len(results)} keys match (rtol=0, atol=1e-12)")

    # Optional visualization of BHat difference.
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return 0

    a = read_sfincs_h5(out_path)
    b = read_sfincs_h5(fortran_path)
    bhat_a = np.asarray(a["BHat"])
    bhat_b = np.asarray(b["BHat"])
    diff = bhat_a - bhat_b

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    im = ax.pcolormesh(diff, shading="auto", cmap="coolwarm")
    ax.set_title(r"$\hat B$ difference (JAX - Fortran), stored layout")
    ax.set_xlabel("col")
    ax.set_ylabel("row")
    fig.colorbar(im, ax=ax, label="ΔBHat")

    fig_dir = Path(__file__).with_suffix("").parent / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "output_parity_bhat_diff.png", bbox_inches="tight", dpi=200)
    fig.savefig(fig_dir / "output_parity_bhat_diff.pdf", bbox_inches="tight")
    print(f"Wrote figures to {fig_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
