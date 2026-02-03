from __future__ import annotations

"""
Run a tiny RHSMode=2 transport-matrix case with `sfincs_jax`, then run the upstream v3
postprocessing script `sfincsScanPlot_1` to generate publication-style figures.

This script demonstrates two things:

1) `sfincs_jax write-output --compute-transport-matrix` writes the minimal datasets that the
   upstream scan plotting scripts expect:
   - transportMatrix
   - FSABFlow
   - particleFlux_vm_psiHat
   - heatFlux_vm_psiHat

2) The `sfincs_jax postprocess-upstream` CLI can execute the vendored upstream `utils/` scripts
   in a non-interactive way (no `input()` pauses) and with a headless matplotlib backend.

Requirements
------------
- `matplotlib` installed: `pip install -e ".[viz]"`
"""

import subprocess
import sys
from pathlib import Path

from sfincs_jax.io import write_sfincs_jax_output_h5
from sfincs_jax.verbose import make_emit


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    fixture = repo_root / "tests" / "ref" / "transportMatrix_PAS_tiny_rhsMode2_scheme2.input.namelist"
    out_dir = Path("/tmp") / "sfincs_jax_postprocess_demo"
    out_dir.mkdir(parents=True, exist_ok=True)

    emit = make_emit(verbose=2, prefix="[demo] ")
    emit(0, f"fixture={fixture}")
    emit(0, f"out_dir={out_dir}")

    out_h5 = out_dir / "sfincsOutput.h5"
    write_sfincs_jax_output_h5(
        input_namelist=fixture,
        output_path=out_h5,
        compute_transport_matrix=True,
        emit=emit,
    )

    emit(0, "Running upstream plotting script (sfincsScanPlot_1) via `sfincs_jax postprocess-upstream`…")
    cmd = [
        sys.executable,
        "-m",
        "sfincs_jax",
        "-v",
        "postprocess-upstream",
        "--case-dir",
        str(out_dir),
        "--util",
        "sfincsScanPlot_1",
        "--",
        "pdf",
    ]
    subprocess.run(cmd, check=True)

    # The upstream script writes several PDF files in the current directory.
    pdfs = sorted(out_dir.glob("*.pdf"))
    emit(0, f"generated_pdfs={len(pdfs)}")
    for p in pdfs[:20]:
        emit(0, f" - {p}")

    emit(0, "Done.")


if __name__ == "__main__":
    main()
