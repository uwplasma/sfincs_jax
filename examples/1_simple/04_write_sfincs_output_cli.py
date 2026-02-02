"""Write a SFINCS-style `sfincsOutput.h5` using the CLI.

This example demonstrates:
  - invoking the `sfincs_jax` CLI from Python
  - loading the resulting HDF5 output

Run:
  python examples/1_simple/04_write_sfincs_output_cli.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sfincs_jax.io import read_sfincs_h5


def main() -> int:
    input_path = _REPO_ROOT / "tests" / "ref" / "output_scheme4_1species_tiny.input.namelist"
    out_dir = Path(__file__).with_suffix("").parent / "output"
    out_path = out_dir / "sfincsOutput_cli.h5"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "sfincs_jax.cli",
        "write-output",
        "--input",
        str(input_path),
        "--out",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)

    data = read_sfincs_h5(out_path)
    print(f"Wrote: {out_path}")
    print(f"Keys: {len(data)}")
    print(f"FSABHat2 = {np.asarray(data['FSABHat2'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

