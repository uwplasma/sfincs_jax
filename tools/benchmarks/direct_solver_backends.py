#!/usr/bin/env python
"""Inspect sparse factorization costs on an explicitly selected SFINCS matrix.

Run on an idle machine, with binary matrix dumps and ``-log_view`` saved as
``petsc_log.txt``. Select ``--matrix-index 0`` for the simplified preconditioner
of an iterative solve, or ``--matrix-index 1`` for a direct Jacobian solve.
Check ``-ksp_view`` to establish the actual backend and factored matrix.
The residual operator (index 3) need not be the matrix PETSc factorized.

PETSc's numeric-factor event and scipy's complete ``splu`` call have different
timing boundaries, so this diagnostic deliberately prints no speedup ratio.
A controlled comparison must also match ordering, scaling, pivoting, threads,
precision, and symbolic-analysis reuse. Sequential SuperLU is supernodal and
supports fill-reducing column orderings (COLAMD is scipy's default); it is a
different implementation from SuperLU_DIST and multifrontal MUMPS.

Usage:
  python tools/benchmarks/direct_solver_backends.py --matrix-index 0 WORKDIR
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path


def petsc_lu_seconds(log: Path) -> float | None:
    """Numeric factorization time from PETSc; the caller verifies the backend."""
    if not log.is_file():
        return None
    for line in log.read_text(errors="replace").splitlines():
        if line.strip().startswith("MatLUFactorNum"):
            fields = line.split()
            try:
                return float(fields[3])
            except (IndexError, ValueError):
                return None
    return None


def superlu_seconds(matrix_path: Path) -> tuple[float, int, int]:
    """Time scipy's SuperLU on the same matrix; returns (seconds, n, fill)."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import splu

    from dkx.validation.fortran import read_petsc_mat_aij

    aij = read_petsc_mat_aij(matrix_path)
    operator = csr_matrix(
        (aij.data, aij.col_ind, aij.row_ptr), shape=aij.shape
    ).tocsc()
    started = time.perf_counter()
    factors = splu(operator)
    elapsed = time.perf_counter() - started
    fill = int(factors.L.nnz + factors.U.nnz)
    del factors
    return elapsed, int(operator.shape[0]), fill


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("workdirs", nargs="+", type=Path)
    parser.add_argument(
        "--matrix-index", type=int, choices=(0, 1), required=True,
        help="Factored matrix: 0 = iterative preconditioner; 1 = direct Jacobian.",
    )
    parser.add_argument(
        "--timeout-note",
        action="store_true",
        help="only report sizes and MUMPS times, without running SuperLU",
    )
    args = parser.parse_args()

    print(
        f"{'run':<40s} {'n':>8s} {'nnz(LU)':>12s} "
        f"{'PETSc numeric [s]':>18s} {'SuperLU total [s]':>18s}"
    )
    for work in args.workdirs:
        matrix = work / f"sfincsBinary_iteration_000_whichMatrix_{args.matrix_index}"
        if not matrix.is_file():
            print(f"{work.name[:40]:<40s} {'-':>8s} (no matrix dump)")
            continue
        mumps = petsc_lu_seconds(work / "petsc_log.txt")
        if args.timeout_note:
            print(
                f"{work.name[:40]:<40s} {'?':>8s} {'?':>12s} "
                f"{(mumps if mumps is not None else float('nan')):10.4f}"
            )
            continue
        seconds, size, fill = superlu_seconds(matrix)
        print(
            f"{work.name[:40]:<40s} {size:8d} {fill:12d} "
            f"{(mumps if mumps is not None else float('nan')):10.4f} "
            f"{seconds:18.4f}"
        )


if __name__ == "__main__":
    main()
