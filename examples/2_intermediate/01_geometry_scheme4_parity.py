"""Compare sfincs_jax `geometryScheme=4` fields against a Fortran v3 run.

This script is useful when extending the geometry implementation. It compares:
- `BHat`
- `dBHatdtheta`
- `dBHatdzeta`

You can either point to an existing `sfincsOutput.h5`, or ask the script to run the
Fortran executable (set `SFINCS_FORTRAN_EXE` or pass `--fortran-exe`).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sfincs_jax.fortran import run_sfincs_fortran
from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.v3 import geometry_from_namelist, grids_from_namelist


def main() -> int:
    p = argparse.ArgumentParser()
    default_input = _REPO_ROOT / "tests" / "ref" / "output_scheme4_1species_tiny.input.namelist"
    default_output = _REPO_ROOT / "tests" / "ref" / "output_scheme4_1species_tiny.sfincsOutput.h5"
    p.add_argument("--input", default=str(default_input), help="Path to SFINCS input.namelist (default: repo fixture)")
    p.add_argument(
        "--sfincs-output",
        default=str(default_output),
        help="Path to sfincsOutput.h5 (default: matching repo fixture; optional if --run-fortran)",
    )
    p.add_argument("--run-fortran", action="store_true", help="Run Fortran first to generate sfincsOutput.h5")
    p.add_argument("--fortran-exe", default=None, help="Path to Fortran executable (or set SFINCS_FORTRAN_EXE)")
    args = p.parse_args()

    input_path = Path(args.input)
    if args.run_fortran:
        out_path = run_sfincs_fortran(
            input_namelist=input_path,
            exe=Path(args.fortran_exe) if args.fortran_exe else None,
        )
    else:
        out_path = Path(args.sfincs_output)

    nml = read_sfincs_input(input_path)
    grids = grids_from_namelist(nml)
    geom = geometry_from_namelist(nml=nml, grids=grids)

    with h5py.File(out_path, "r") as f:
        bhat_f = f["BHat"][...]
        dbth_f = f["dBHatdtheta"][...]
        dbze_f = f["dBHatdzeta"][...]

    # Fortran HDF5 output uses (Nzeta, Ntheta); sfincs_jax uses (Ntheta, Nzeta).
    bhat = np.asarray(geom.b_hat).T
    dbth = np.asarray(geom.db_hat_dtheta).T
    dbze = np.asarray(geom.db_hat_dzeta).T

    print("max |BHat - Fortran|:", float(np.max(np.abs(bhat - bhat_f))))
    print("max |dBHatdtheta - Fortran|:", float(np.max(np.abs(dbth - dbth_f))))
    print("max |dBHatdzeta - Fortran|:", float(np.max(np.abs(dbze - dbze_f))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
