from __future__ import annotations

from pathlib import Path

import numpy as np
import jax.numpy as jnp

from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.petsc_binary import read_petsc_mat_aij, read_petsc_vec
from sfincs_jax.v3 import grids_from_namelist
from sfincs_jax.v3_fblock import fblock_operator_from_namelist, matvec_v3_fblock_flat
from sfincs_jax.indices import V3Indexing


def test_fblock_collisionless_plus_fp_matvec_matches_fortran_matrix() -> None:
    """Parity test for the combined (collisionless + Fokker-Planck) v3 F-block operator.

    Uses a frozen Fortran v3 PETSc matrix (whichMatrix_3) and restricts A @ x to the
    distribution-function block (rows/cols corresponding to f in the PETSc ordering).
    """
    here = Path(__file__).parent
    input_path = here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist"
    mat_path = here / "ref" / "quick_2species_FPCollisions_noEr.whichMatrix_3.petscbin"
    vec_path = here / "ref" / "quick_2species_FPCollisions_noEr.stateVector.petscbin"

    nml = read_sfincs_input(input_path)
    grids = grids_from_namelist(nml)

    op = fblock_operator_from_namelist(nml=nml, identity_shift=0.0)
    a = read_petsc_mat_aij(mat_path)
    x_full = read_petsc_vec(vec_path).values

    indexing = V3Indexing(
        n_species=op.n_species,
        n_x=op.n_x,
        n_theta=op.n_theta,
        n_zeta=op.n_zeta,
        n_xi_max=op.n_xi,
        n_xi_for_x=np.asarray(grids.n_xi_for_x, dtype=int),
    )
    inv = indexing.build_inverse_f_map()
    n_f = len(inv)
    assert n_f == op.flat_size

    x_f = np.asarray(x_full[:n_f], dtype=np.float64)

    y_jax = np.asarray(matvec_v3_fblock_flat(op, jnp.asarray(x_f)))

    y_ref = np.zeros((n_f,), dtype=np.float64)
    for row in range(n_f):
        start = int(a.row_ptr[row])
        end = int(a.row_ptr[row + 1])
        cols = a.col_ind[start:end]
        vals = a.data[start:end]
        acc = 0.0
        for c, v in zip(cols.tolist(), vals.tolist()):
            if c < 0 or c >= n_f:
                continue
            acc += float(v) * float(x_f[c])
        y_ref[row] = acc

    # This case includes dense x-couplings from the collision operator, so allow a small
    # absolute tolerance beyond 1e-12 to accommodate summation order differences.
    np.testing.assert_allclose(y_jax, y_ref, rtol=0, atol=5e-11)

