"""Solve a frozen Fortran v3 PETSc matrix using JAX GMRES (matrix-free matvec).

This script is a stepping stone for the full sfincs_jax solver:

- Today: we read a PETSc AIJ matrix saved by the Fortran code and solve A x = b.
- Later: we replace the matrix with a pure-JAX operator matvec (no assembly).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sfincs_jax.petsc_binary import read_petsc_mat_aij, read_petsc_vec
from sfincs_jax.solver import gmres_solve
from sfincs_jax.sparse import csr_matvec


def _default_prefix() -> Path:
    return Path(__file__).parents[2] / "tests" / "ref" / "pas_1species_PAS_noEr_tiny"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mat", default=None, help="Path to PETSc AIJ matrix (default: fixture)")
    p.add_argument("--x-ref", default=None, help="Path to PETSc Vec to use as x_ref (default: fixture)")
    p.add_argument("--tol", type=float, default=1e-12)
    p.add_argument("--restart", type=int, default=80)
    p.add_argument("--maxiter", type=int, default=200)
    args = p.parse_args()

    prefix = _default_prefix()
    mat_path = Path(args.mat) if args.mat else Path(str(prefix) + ".whichMatrix_3.petscbin")
    xref_path = Path(args.x_ref) if args.x_ref else Path(str(prefix) + ".stateVector.petscbin")

    a = read_petsc_mat_aij(mat_path)
    x_ref = read_petsc_vec(xref_path).values

    data = jnp.asarray(a.data)
    indices = jnp.asarray(a.col_ind)
    indptr = jnp.asarray(a.row_ptr, dtype=jnp.int32)
    n_rows, n_cols = a.shape
    if n_rows != n_cols:
        raise SystemExit(f"Matrix is not square: {a.shape}")

    def A_mv(x: jnp.ndarray) -> jnp.ndarray:
        return csr_matvec(data=data, indices=indices, indptr=indptr, x=x, n_rows=n_rows)

    x_ref_j = jnp.asarray(x_ref)
    b = A_mv(x_ref_j)

    result = gmres_solve(matvec=A_mv, b=b, tol=args.tol, restart=args.restart, maxiter=args.maxiter)
    x = np.asarray(result.x)

    err = float(np.max(np.abs(x - x_ref)))
    print(f"n={n_rows}  residual_norm={float(result.residual_norm):.3e}  max|x-x_ref|={err:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
