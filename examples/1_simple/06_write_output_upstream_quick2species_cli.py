"""Write `sfincsOutput.h5` using an upstream Fortran v3 example input (CLI).

This example uses the vendored upstream input:
  examples/upstream/fortran_v3/quick_2species_FPCollisions_noEr/input.namelist

Run:
  python examples/1_simple/06_write_output_upstream_quick2species_cli.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    input_path = (
        _REPO_ROOT / "examples" / "upstream" / "fortran_v3" / "quick_2species_FPCollisions_noEr" / "input.namelist"
    )
    out_dir = Path(__file__).with_suffix("").parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sfincsOutput_upstream_quick2species_cli.h5"

    cmd = [
        sys.executable,
        "-m",
        "sfincs_jax",
        "write-output",
        "--input",
        str(input_path),
        "--out",
        str(out_path),
    ]
    subprocess.run(cmd, cwd=str(_REPO_ROOT), check=True)
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
