"""Sweep Er, write outputs via CLI, and plot the derived `dPhiHatdpsiHat` conversion.

This example demonstrates using the `sfincs_jax` CLI in a simple parametric study:
  - create a few small `input.namelist` variants with different Er values
  - call `sfincs_jax write-output` for each
  - read outputs and visualize the resulting `dPhiHatdpsiHat`

Run:
  python examples/transport/write_output_cli_sweep_er.py

Plotting requires:
  pip install -e ".[viz]"
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sfincs_jax.io import read_sfincs_h5


def _set_er(text: str, er: float) -> str:
    # Replace a line like "Er = 0.5d+0" or "Er = 0.5" within physicsParameters.
    pat = re.compile(r"^\\s*Er\\s*=\\s*[^\\n]+$", flags=re.MULTILINE)
    repl = f"  Er = {er:.12g}d+0"
    if not pat.search(text):
        raise ValueError("Could not find an 'Er =' line to replace.")
    return pat.sub(repl, text, count=1)


def main() -> int:
    template_path = _REPO_ROOT / "tests" / "ref" / "output_scheme4_1species_tiny.input.namelist"
    template = template_path.read_text()

    out_dir = Path(__file__).with_suffix("").parent / "output" / "er_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    ers = np.linspace(-1.0, 1.0, 9)
    dphis = []

    for er in ers:
        input_path = out_dir / f"input_Er_{er:+.2f}.namelist"
        output_path = out_dir / f"sfincsOutput_Er_{er:+.2f}.h5"
        input_path.write_text(_set_er(template, float(er)))

        cmd = [
            sys.executable,
            "-m",
            "sfincs_jax",
            "write-output",
            "--input",
            str(input_path),
            "--out",
            str(output_path),
        ]
        subprocess.run(cmd, check=True)

        data = read_sfincs_h5(output_path)
        dphis.append(float(np.asarray(data["dPhiHatdpsiHat"])))

    try:
        import matplotlib.pyplot as plt
    except Exception:
        for er, dphi in zip(ers, dphis):
            print(er, dphi)
        return 0

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(ers, dphis, marker="o")
    ax.set_title(r"geometryScheme=4: $d\\hat\\Phi/d\\hat\\psi$ vs $E_r$")
    ax.set_xlabel(r"$E_r$")
    ax.set_ylabel(r"$d\\hat\\Phi/d\\hat\\psi$")
    ax.grid(True, alpha=0.25)

    fig_dir = Path(__file__).with_suffix("").parent / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "er_sweep_dphidpsi.png", bbox_inches="tight", dpi=200)
    fig.savefig(fig_dir / "er_sweep_dphidpsi.pdf", bbox_inches="tight")
    print(f"Wrote outputs to {out_dir}")
    print(f"Wrote figures to {fig_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
