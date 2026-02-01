from __future__ import annotations

from pathlib import Path

import numpy as np
import jax.numpy as jnp

from sfincs_jax.collisionless import CollisionlessV3Operator, apply_collisionless_v3
from sfincs_jax.indices import V3Indexing
from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.petsc_binary import read_petsc_mat_aij, read_petsc_vec
from sfincs_jax.v3 import geometry_from_namelist, grids_from_namelist


def test_collisionless_offdiag_xi_matvec_matches_fortran_matrix() -> None:
    """Parity test for the collisionless ±1-in-L couplings (streaming + mirror).

    We compare against the saved Fortran v3 PETSc binary matrix for the quick example by
    restricting the matrix-vector product to columns with `|ΔL| = 1`, which isolates
    streaming+mirror for this configuration.
    """
    here = Path(__file__).parent
    input_path = here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist"
    mat_path = here / "ref" / "quick_2species_FPCollisions_noEr.whichMatrix_3.petscbin"
    vec_path = here / "ref" / "quick_2species_FPCollisions_noEr.stateVector.petscbin"

    nml = read_sfincs_input(input_path)
    grids = grids_from_namelist(nml)
    geom = geometry_from_namelist(nml=nml, grids=grids)

    species = nml.group("speciesParameters")
    t_hats = np.asarray(species["THATS"], dtype=np.float64)
    m_hats = np.asarray(species["MHATS"], dtype=np.float64)

    op = CollisionlessV3Operator(
        x=grids.x,
        ddtheta=grids.ddtheta,
        ddzeta=grids.ddzeta,
        b_hat=geom.b_hat,
        b_hat_sup_theta=geom.b_hat_sup_theta,
        b_hat_sup_zeta=geom.b_hat_sup_zeta,
        db_hat_dtheta=geom.db_hat_dtheta,
        db_hat_dzeta=geom.db_hat_dzeta,
        t_hats=jnp.asarray(t_hats),
        m_hats=jnp.asarray(m_hats),
        n_xi_for_x=grids.n_xi_for_x,
    )

    a = read_petsc_mat_aij(mat_path)
    x_full = read_petsc_vec(vec_path).values

    indexing = V3Indexing(
        n_species=int(t_hats.size),
        n_x=int(grids.x.shape[0]),
        n_theta=int(grids.theta.shape[0]),
        n_zeta=int(grids.zeta.shape[0]),
        n_xi_max=int(grids.n_xi),
        n_xi_for_x=np.asarray(grids.n_xi_for_x, dtype=int),
    )
    inv = indexing.build_inverse_f_map()
    n_f = len(inv)

    # Build padded f tensor from the full PETSc state vector.
    f = np.zeros(
        (
            indexing.n_species,
            indexing.n_x,
            indexing.n_xi_max,
            indexing.n_theta,
            indexing.n_zeta,
        ),
        dtype=np.float64,
    )
    for g, (s, ix, l, it, iz) in enumerate(inv):
        f[s, ix, l, it, iz] = x_full[g]

    # JAX result (collisionless only, offdiag in L only).
    y_jax = np.asarray(apply_collisionless_v3(op, jnp.asarray(f)))

    # Fortran reference: restrict A @ x to columns with |ΔL|=1 within the distribution block.
    y_ref = np.zeros((n_f,), dtype=np.float64)
    for row in range(n_f):
        s_r, ix_r, l_r, it_r, iz_r = inv[row]
        start = int(a.row_ptr[row])
        end = int(a.row_ptr[row + 1])
        cols = a.col_ind[start:end]
        vals = a.data[start:end]
        acc = 0.0
        for c, v in zip(cols.tolist(), vals.tolist()):
            if c < 0 or c >= n_f:
                continue
            s_c, ix_c, l_c, it_c, iz_c = inv[c]
            if s_c != s_r or ix_c != ix_r:
                continue
            if abs(l_c - l_r) != 1:
                continue
            acc += float(v) * float(x_full[c])
        y_ref[row] = acc

    y_jax_vec = np.zeros((n_f,), dtype=np.float64)
    for g, (s, ix, l, it, iz) in enumerate(inv):
        y_jax_vec[g] = y_jax[s, ix, l, it, iz]
    np.testing.assert_allclose(y_jax_vec, y_ref, rtol=0, atol=1e-12)
