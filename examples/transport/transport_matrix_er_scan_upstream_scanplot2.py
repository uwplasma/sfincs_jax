from __future__ import annotations

"""
Transport-matrix scan + upstream plotting (publication-style PDF).

This example demonstrates one of the main goals of `sfincs_jax`:

- run a (small) scan in a fully Python/JAX-native way
- write outputs compatible with upstream Fortran v3 postprocessing scripts
- generate a polished PDF figure via the vendored upstream `utils/` script

Requirements
------------
- `pip install -e .[viz]` (matplotlib)
- A repo checkout (so `examples/sfincs_examples/utils` and `examples/sfincs_examples/globalVariables.F90` exist)
"""

from pathlib import Path

import numpy as np

from sfincs_jax.postprocess_upstream import run_upstream_util
from sfincs_jax.scans import run_er_scan
from sfincs_jax.verbose import make_emit


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    out_dir = Path(__file__).resolve().parent / "output" / "transport_matrix_er_scan"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Small RHSMode=2 template used by the parity fixtures.
    template = repo_root / "tests" / "ref" / "transportMatrix_PAS_tiny_rhsMode2_scheme2.input.namelist"
    if not template.exists():
        raise FileNotFoundError(f"Missing template: {template}")

    emit = make_emit(verbose=1)

    # Match upstream `sfincsScan_2`: values are typically generated as linspace(max, min, N).
    values = list(np.linspace(0.1, -0.1, 5))
    scan_dir = out_dir / "scan"

    run_er_scan(
        input_namelist=template,
        out_dir=scan_dir,
        values=values,
        compute_transport_matrix=True,
        emit=emit,
    )

    # Generate a PDF figure using the upstream script. It will write into `scan_dir`.
    run_upstream_util(util="sfincsScanPlot_2", case_dir=scan_dir, args=("pdf",), emit=emit)

    pdf = scan_dir / "sfincsScanPlot_2.pdf"
    print(f"Wrote {pdf}")


if __name__ == "__main__":
    main()

