"""Report which `sfincsOutput.h5` datasets are missing in `sfincs_jax`.

This script compares:
  - a `sfincsOutput.h5` written by `sfincs_jax write-output`
  - a frozen Fortran v3 `sfincsOutput.h5` fixture

and prints keys that exist in the Fortran file but are not written by `sfincs_jax` yet.

Run:
  python examples/2_intermediate/09_output_key_coverage_report.py
"""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from sfincs_jax.io import read_sfincs_h5, write_sfincs_jax_output_h5


def main() -> int:
    input_path = repo_root / "tests" / "ref" / "output_scheme4_1species_tiny.input.namelist"
    fortran_path = repo_root / "tests" / "ref" / "output_scheme4_1species_tiny.sfincsOutput.h5"

    out_dir = Path(__file__).with_suffix("").parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    jax_path = out_dir / "sfincsOutput_jax.h5"
    write_sfincs_jax_output_h5(input_namelist=input_path, output_path=jax_path)

    a = read_sfincs_h5(jax_path)
    b = read_sfincs_h5(fortran_path)

    missing = sorted(set(b.keys()) - set(a.keys()))
    extra = sorted(set(a.keys()) - set(b.keys()))

    print(f"Fortran keys: {len(b)}")
    print(f"JAX keys:    {len(a)}")
    print(f"Missing in JAX: {len(missing)}")
    for k in missing:
        print(f"  {k}")
    if extra:
        print(f"Extra in JAX: {len(extra)}")
        for k in extra:
            print(f"  {k}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
