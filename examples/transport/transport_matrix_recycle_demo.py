from __future__ import annotations

"""
Demonstrate recycled-Krylov warm starts for RHSMode=2/3 transport-matrix solves.

This script runs a tiny RHSMode=3 transport-matrix example twice:
  1) with recycling disabled
  2) with recycling enabled (keep the last k solution vectors)

The recycle path uses a small least-squares projection onto the recycled subspace
to generate an initial guess for the next `whichRHS` solve.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.v3_driver import solve_v3_transport_matrix_linear_gmres


def _default_input() -> Path:
    return _REPO_ROOT / "tests" / "ref" / "monoenergetic_PAS_tiny_scheme11.input.namelist"


def _run_case(input_path: Path, *, recycle_k: int) -> None:
    os.environ["SFINCS_JAX_TRANSPORT_RECYCLE_K"] = str(int(recycle_k))
    # Force Krylov so the warm-start path is exercised.
    os.environ["SFINCS_JAX_TRANSPORT_FORCE_KRYLOV"] = "1"

    nml = read_sfincs_input(input_path)
    result = solve_v3_transport_matrix_linear_gmres(nml=nml, tol=1e-10, restart=80, maxiter=400)

    residuals = {k: float(v) for k, v in result.residual_norms_by_rhs.items()}
    elapsed = np.asarray(result.elapsed_time_s, dtype=np.float64)
    print(f"\nrecycle_k={recycle_k}")
    print(f"  residuals_by_rhs: {residuals}")
    print(f"  elapsed_s_by_rhs: {elapsed}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=str(_default_input()))
    p.add_argument("--recycle-k", type=int, default=4)
    args = p.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"input not found: {input_path}")

    _run_case(input_path, recycle_k=0)
    _run_case(input_path, recycle_k=int(args.recycle_k))


if __name__ == "__main__":
    main()
