from __future__ import annotations

from pathlib import Path

import numpy as np
import jax.numpy as jnp

from sfincs_jax.collisions import make_pitch_angle_scattering_v3_operator, apply_pitch_angle_scattering_v3
from sfincs_jax.indices import V3Indexing
from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.petsc_binary import read_petsc_mat_aij, read_petsc_vec
from sfincs_jax.v3 import grids_from_namelist


def test_pas_collisions_diagonal_matvec_matches_fortran_matrix() -> None:
    """Parity test for v3 pitch-angle scattering (collisionOperator=1, no Phi1).

    We compare sfincs_jax against a saved Fortran v3 PETSc binary matrix by restricting
    the matrix-vector product to the diagonal entries inside the distribution-function block.
    For this configuration (no Er, no magnetic drifts), the diagonal-in-L piece is dominated
    by the PAS collision operator.
    """
    here = Path(__file__).parent
    input_path = here / "ref" / "pas_1species_PAS_noEr_tiny.input.namelist"
    mat_path = here / "ref" / "pas_1species_PAS_noEr_tiny.whichMatrix_3.petscbin"
    vec_path = here / "ref" / "pas_1species_PAS_noEr_tiny.stateVector.petscbin"

    nml = read_sfincs_input(input_path)
    grids = grids_from_namelist(nml)

    species = nml.group("speciesParameters")
    z_s = np.atleast_1d(np.asarray(species["ZS"], dtype=np.float64))
    m_hats = np.atleast_1d(np.asarray(species["MHATS"], dtype=np.float64))
    n_hats = np.atleast_1d(np.asarray(species["NHATS"], dtype=np.float64))
    t_hats = np.atleast_1d(np.asarray(species["THATS"], dtype=np.float64))

    phys = nml.group("physicsParameters")
    nu_n = float(phys.get("NU_N", 0.0))

    op = make_pitch_angle_scattering_v3_operator(
        x=grids.x,
        z_s=jnp.asarray(z_s),
        m_hats=jnp.asarray(m_hats),
        n_hats=jnp.asarray(n_hats),
        t_hats=jnp.asarray(t_hats),
        nu_n=nu_n,
        krook=float(phys.get("KROOK", 0.0)),
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

    y_jax = np.asarray(apply_pitch_angle_scattering_v3(op, jnp.asarray(f)))

    # Fortran reference: diagonal-only contribution inside the distribution block.
    y_ref = np.zeros((n_f,), dtype=np.float64)
    for row in range(n_f):
        start = int(a.row_ptr[row])
        end = int(a.row_ptr[row + 1])
        cols = a.col_ind[start:end]
        vals = a.data[start:end]
        diag = 0.0
        for c, v in zip(cols.tolist(), vals.tolist()):
            if c == row:
                diag = float(v)
                break
        y_ref[row] = diag * float(x_full[row])

    y_jax_vec = np.zeros((n_f,), dtype=np.float64)
    for g, (s, ix, l, it, iz) in enumerate(inv):
        y_jax_vec[g] = y_jax[s, ix, l, it, iz]

    np.testing.assert_allclose(y_jax_vec, y_ref, rtol=0, atol=1e-12)
