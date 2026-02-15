#!/usr/bin/env python

"""Run sfincs_jax and write a SFINCS-style sfincsOutput.h5.

This module is the shared execution backend for the utils scripts. It defaults
to reading ``input.namelist`` in the current directory and writing a local
``sfincsOutput.h5``. Use ``--input`` and ``--out`` to target other paths, or
import ``run_sfincs_jax`` from Python to embed it in custom workflows.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sfincs_jax.io import localize_equilibrium_file_in_place, write_sfincs_jax_output_h5  # noqa: E402
from sfincs_jax.namelist import read_sfincs_input  # noqa: E402


def run_sfincs_jax(
    *,
    input_namelist: Path,
    output_path: Optional[Path] = None,
    compute_transport_matrix: Optional[bool] = None,
    compute_solution: Optional[bool] = None,
    overwrite: bool = True,
    verbose: bool = True,
    ensure_equilibrium: bool = True,
) -> Path:
    input_namelist = Path(input_namelist).resolve()
    if output_path is None:
        output_path = input_namelist.parent / "sfincsOutput.h5"

    nml = read_sfincs_input(input_namelist)
    rhs_mode = int(nml.group("general").get("RHSMODE", 1))

    if compute_transport_matrix is None:
        compute_transport_matrix = rhs_mode in {2, 3}
    if compute_solution is None:
        compute_solution = rhs_mode == 1

    if ensure_equilibrium:
        localize_equilibrium_file_in_place(input_namelist=input_namelist, overwrite=False)

    return write_sfincs_jax_output_h5(
        input_namelist=input_namelist,
        output_path=output_path,
        compute_transport_matrix=bool(compute_transport_matrix),
        compute_solution=bool(compute_solution),
        overwrite=overwrite,
        verbose=verbose,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="sfincs_jax_driver",
        description="Run sfincs_jax to write a SFINCS-style sfincsOutput.h5.",
    )
    parser.add_argument("--input", default="input.namelist", help="Path to input.namelist.")
    parser.add_argument("--out", default="sfincsOutput.h5", help="Output sfincsOutput.h5 path.")
    parser.add_argument(
        "--transport",
        action="store_true",
        help="Force transport-matrix solve (RHSMode=2/3).",
    )
    parser.add_argument(
        "--solution",
        action="store_true",
        help="Force RHSMode=1 solve and diagnostics.",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Do not overwrite existing output file.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose sfincs-style logging.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    input_path = Path(args.input)
    output_path = Path(args.out)

    compute_transport = True if args.transport else None
    compute_solution = True if args.solution else None

    run_sfincs_jax(
        input_namelist=input_path,
        output_path=output_path,
        compute_transport_matrix=compute_transport,
        compute_solution=compute_solution,
        overwrite=not args.no_overwrite,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
