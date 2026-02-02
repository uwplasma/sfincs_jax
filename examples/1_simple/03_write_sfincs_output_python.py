"""Write a SFINCS-style `sfincsOutput.h5` using the Python API.

This example demonstrates:
  - parsing a Fortran `input.namelist`
  - generating `sfincsOutput.h5` for supported modes (currently geometryScheme=4)
  - loading the resulting HDF5 file and inspecting a few datasets

Run:
  python examples/1_simple/03_write_sfincs_output_python.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sfincs_jax.io import read_sfincs_h5, write_sfincs_jax_output_h5


def main() -> int:
    input_path = _REPO_ROOT / "tests" / "ref" / "output_scheme4_1species_tiny.input.namelist"
    out_dir = Path(__file__).with_suffix("").parent / "output"
    out_path = out_dir / "sfincsOutput_python.h5"

    write_sfincs_jax_output_h5(input_namelist=input_path, output_path=out_path)
    data = read_sfincs_h5(out_path)

    print(f"Wrote: {out_path}")
    print(f"Keys: {len(data)}")
    print("Selected datasets:")
    for k in ["Ntheta", "Nzeta", "Nx", "Delta", "Er", "FSABHat2"]:
        print(f"  {k} = {np.asarray(data[k])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

