"""End-to-end linear solve parity for VMEC `geometryScheme=5` (tiny PAS case).

This script exercises the full matrix-free v3 linear solve stack for a VMEC equilibrium:

- Builds the v3 grids and `geometryScheme=5` VMEC geometry from `wout_*.nc`.
- Builds the full-system operator (F-block + constraint rows/cols).
- Assembles the v3 RHS and solves `A x = rhs` using matrix-free GMRES.
- Compares the resulting solution vector to a frozen Fortran v3 `stateVector` fixture.

This is a key milestone toward full upstream v3 example-suite parity: not just geometry/output
parity, but a full solve with VMEC geometry.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.petsc_binary import read_petsc_vec
from sfincs_jax.v3_driver import solve_v3_full_system_linear_gmres


def _default_input() -> Path:
    return Path(__file__).parents[2] / "tests" / "ref" / "pas_1species_PAS_noEr_tiny_scheme5.input.namelist"


def _default_statevector() -> Path:
    return Path(__file__).parents[2] / "tests" / "ref" / "pas_1species_PAS_noEr_tiny_scheme5.stateVector.petscbin"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=str(_default_input()))
    p.add_argument("--statevector", default=str(_default_statevector()))
    p.add_argument("--tol", type=float, default=1e-12)
    p.add_argument("--restart", type=int, default=80)
    p.add_argument("--maxiter", type=int, default=300)
    args = p.parse_args()

    nml = read_sfincs_input(Path(args.input))
    ref = read_petsc_vec(Path(args.statevector)).values

    sol = solve_v3_full_system_linear_gmres(
        nml=nml,
        tol=float(args.tol),
        restart=int(args.restart),
        maxiter=int(args.maxiter),
        solve_method="batched",
    )
    x = np.asarray(sol.x)

    err = x - ref
    rel = float(np.linalg.norm(err) / np.linalg.norm(ref))
    print(f"n={ref.size}")
    print(f"GMRES residual norm: {float(sol.residual_norm):.3e}")
    print(f"||x - x_ref|| / ||x_ref|| = {rel:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

