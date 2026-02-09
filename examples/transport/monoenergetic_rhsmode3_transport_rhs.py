"""
Monoenergetic (RHSMode=3) transport-matrix runs: selecting `whichRHS`.

In upstream SFINCS v3, `RHSMode=3` (monoenergetic coefficients) runs a *loop* over `whichRHS` and
overwrites (dnHatdpsiHats, dTHatdpsiHats, EParallelHat) internally before constructing each RHS.

In `sfincs_jax`, the same behavior is exposed via:
  - `sfincs_jax.v3_system.with_transport_rhs_settings` (low-level operator manipulation), and
  - `sfincs_jax.v3_driver.solve_v3_full_system_linear_gmres(..., which_rhs=...)` (convenience driver),
  - `sfincs_jax solve-v3 --which-rhs ...` (CLI).

This example uses the tiny monoenergetic parity fixture shipped in `tests/ref/` so it runs quickly
and does not require the Fortran executable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.v3_driver import solve_v3_full_system_linear_gmres


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    input_path = repo_root / "tests" / "ref" / "monoenergetic_PAS_tiny_scheme1.input.namelist"

    nml = read_sfincs_input(input_path)

    # whichRHS=1: density-gradient drive, E_parallel = 0
    res1 = solve_v3_full_system_linear_gmres(nml=nml, which_rhs=1, tol=1e-12, restart=100, maxiter=2000)
    x1 = np.asarray(res1.x)
    print(f"which_rhs=1 residual_norm={float(res1.residual_norm):.3e}  ||x||_2={np.linalg.norm(x1):.3e}")

    # whichRHS=2: inductive E_parallel drive, dn/dpsi = 0
    res2 = solve_v3_full_system_linear_gmres(nml=nml, which_rhs=2, tol=1e-12, restart=100, maxiter=2000)
    x2 = np.asarray(res2.x)
    print(f"which_rhs=2 residual_norm={float(res2.residual_norm):.3e}  ||x||_2={np.linalg.norm(x2):.3e}")


if __name__ == "__main__":
    main()

