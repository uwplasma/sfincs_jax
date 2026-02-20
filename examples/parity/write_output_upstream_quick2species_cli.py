"""Write `sfincsOutput.h5` using an upstream Fortran v3 example input (CLI).

This example uses the vendored upstream input:
  examples/upstream/fortran_v3/quick_2species_FPCollisions_noEr/input.namelist

Run:
  python examples/parity/write_output_upstream_quick2species_cli.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _select_input_path() -> Path:
    if os.environ.get("SFINCS_JAX_CI") == "1" or os.environ.get("SFINCS_JAX_FAST_EXAMPLES") == "1":
        return _REPO_ROOT / "tests" / "reduced_inputs" / "quick_2species_FPCollisions_noEr.input.namelist"
    return (
        _REPO_ROOT
        / "examples"
        / "upstream"
        / "fortran_v3"
        / "quick_2species_FPCollisions_noEr"
        / "input.namelist"
    )


def main() -> int:
    input_path = _select_input_path()
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
    env = os.environ.copy()
    if os.environ.get("SFINCS_JAX_CI") == "1" or os.environ.get("SFINCS_JAX_FAST_EXAMPLES") == "1":
        env.setdefault("SFINCS_JAX_FORTRAN_STDOUT", "0")
        env.setdefault("SFINCS_JAX_SOLVER_ITER_STATS", "0")
    subprocess.run(cmd, cwd=str(_REPO_ROOT), check=True, env=env)
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
