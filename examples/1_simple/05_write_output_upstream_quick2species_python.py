"""Write `sfincsOutput.h5` using an upstream Fortran v3 example input (Python API).

This example uses the vendored upstream input:
  examples/upstream/fortran_v3/quick_2species_FPCollisions_noEr/input.namelist

Run:
  python examples/1_simple/05_write_output_upstream_quick2species_python.py
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
    input_path = (
        _REPO_ROOT / "examples" / "upstream" / "fortran_v3" / "quick_2species_FPCollisions_noEr" / "input.namelist"
    )
    out_dir = Path(__file__).with_suffix("").parent / "output"
    out_path = out_dir / "sfincsOutput_upstream_quick2species_python.h5"

    write_sfincs_jax_output_h5(input_namelist=input_path, output_path=out_path)
    data = read_sfincs_h5(out_path)

    print(f"Wrote: {out_path}")
    print(f"Keys: {len(data)}")
    print("Selected datasets:")
    for k in ["Nspecies", "Ntheta", "Nzeta", "Nx", "Nxi", "Delta", "Er", "FSABHat2"]:
        print(f"  {k} = {np.asarray(data[k])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

