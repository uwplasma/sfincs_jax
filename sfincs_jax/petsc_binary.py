from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

_VEC_CLASSID = 1211214
_MAT_CLASSID = 1211216


@dataclass(frozen=True)
class PetscVec:
    values: np.ndarray

    @property
    def size(self) -> int:
        return int(self.values.size)


@dataclass(frozen=True)
class PetscCSRMatrix:
    shape: Tuple[int, int]
    row_ptr: np.ndarray
    col_ind: np.ndarray
    data: np.ndarray

    def get(self, i: int, j: int) -> float:
        """Get A[i, j] for a CSR matrix with sorted column indices per row."""
        i = int(i)
        j = int(j)
        start = int(self.row_ptr[i])
        end = int(self.row_ptr[i + 1])
        cols = self.col_ind[start:end]
        k = int(np.searchsorted(cols, j))
        if k < cols.size and int(cols[k]) == j:
            return float(self.data[start + k])
        return 0.0


def read_petsc_vec(path: str | Path) -> PetscVec:
    """Read a PETSc Vec binary file (big-endian header + float64 data)."""
    b = Path(path).read_bytes()
    header = np.frombuffer(b, dtype=">i4", count=2)
    if header.size != 2:
        raise ValueError("File too small to be a PETSc Vec.")
    classid, n = (int(header[0]), int(header[1]))
    if classid != _VEC_CLASSID:
        raise ValueError(f"Unexpected PETSc Vec classid={classid} (expected {_VEC_CLASSID}).")
    values = np.frombuffer(b, dtype=">f8", offset=2 * 4, count=n).astype(np.float64, copy=False)
    return PetscVec(values=values)


def read_petsc_mat_aij(path: str | Path) -> PetscCSRMatrix:
    """Read a PETSc AIJ Mat binary file into CSR.

    This reader supports the common format written by `MatView(..., PETSC_VIEWER_BINARY_)`.
    """
    b = Path(path).read_bytes()
    header = np.frombuffer(b, dtype=">i4", count=4)
    if header.size != 4:
        raise ValueError("File too small to be a PETSc Mat.")
    classid, m, n, nnz = (int(header[0]), int(header[1]), int(header[2]), int(header[3]))
    if classid != _MAT_CLASSID:
        raise ValueError(f"Unexpected PETSc Mat classid={classid} (expected {_MAT_CLASSID}).")

    offset = 4 * 4
    row_nnz = np.frombuffer(b, dtype=">i4", offset=offset, count=m).astype(np.int32, copy=False)
    offset += m * 4
    col_ind = np.frombuffer(b, dtype=">i4", offset=offset, count=nnz).astype(np.int32, copy=False)
    offset += nnz * 4
    data = np.frombuffer(b, dtype=">f8", offset=offset, count=nnz).astype(np.float64, copy=False)

    row_ptr = np.zeros((m + 1,), dtype=np.int64)
    row_ptr[1:] = np.cumsum(row_nnz, dtype=np.int64)

    if int(row_ptr[-1]) != nnz:
        raise ValueError("Invalid PETSc Mat: row pointers do not sum to nnz.")

    return PetscCSRMatrix(
        shape=(m, n),
        row_ptr=row_ptr,
        col_ind=col_ind,
        data=data,
    )

