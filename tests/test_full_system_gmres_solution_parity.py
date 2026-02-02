from __future__ import annotations

from pathlib import Path

import numpy as np
import jax.numpy as jnp
from scipy.sparse import csr_matrix

from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.petsc_binary import read_petsc_mat_aij, read_petsc_vec
from sfincs_jax.solver import gmres_solve
from sfincs_jax.v3_system import apply_v3_full_system_operator, full_system_operator_from_namelist


def test_full_system_gmres_recovers_fortran_statevector_pas_tiny() -> None:
    """Solve A x = b matrix-free and recover the Fortran v3 stateVector (tiny PAS case)."""
    here = Path(__file__).parent
    input_path = here / "ref" / "pas_1species_PAS_noEr_tiny.input.namelist"
    mat_path = here / "ref" / "pas_1species_PAS_noEr_tiny.whichMatrix_3.petscbin"
    vec_path = here / "ref" / "pas_1species_PAS_noEr_tiny.stateVector.petscbin"

    nml = read_sfincs_input(input_path)
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)

    a = read_petsc_mat_aij(mat_path)
    x_ref = read_petsc_vec(vec_path).values
    assert x_ref.shape == (op.total_size,)
    a_csr = csr_matrix((a.data, a.col_ind, a.row_ptr), shape=a.shape)

    # Build b using the Fortran matrix so the reference solution is exact.
    b = a_csr.dot(x_ref)

    def mv(x):
        return apply_v3_full_system_operator(op, x)

    result = gmres_solve(matvec=mv, b=jnp.asarray(b), tol=1e-12, restart=80, maxiter=200)
    x = np.asarray(result.x)

    np.testing.assert_allclose(x, x_ref, rtol=0, atol=1e-9)
    assert float(result.residual_norm) < 1e-9


def test_full_system_gmres_recovers_fortran_statevector_pas_tiny_with_phi1_linear() -> None:
    """Solve A x = b matrix-free and recover the Fortran v3 stateVector (tiny PAS + Phi1 blocks)."""
    here = Path(__file__).parent
    input_path = here / "ref" / "pas_1species_PAS_noEr_tiny_withPhi1_linear.input.namelist"
    mat_path = here / "ref" / "pas_1species_PAS_noEr_tiny_withPhi1_linear.whichMatrix_3.petscbin"
    vec_path = here / "ref" / "pas_1species_PAS_noEr_tiny_withPhi1_linear.stateVector.petscbin"

    nml = read_sfincs_input(input_path)
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)

    a = read_petsc_mat_aij(mat_path)
    x_ref = read_petsc_vec(vec_path).values
    assert x_ref.shape == (op.total_size,)
    a_csr = csr_matrix((a.data, a.col_ind, a.row_ptr), shape=a.shape)

    b = a_csr.dot(x_ref)

    def mv(x):
        return apply_v3_full_system_operator(op, x)

    result = gmres_solve(matvec=mv, b=jnp.asarray(b), tol=1e-12, restart=100, maxiter=400)
    x = np.asarray(result.x)

    np.testing.assert_allclose(x, x_ref, rtol=0, atol=1e-9)
    assert float(result.residual_norm) < 1e-9


def test_full_system_gmres_recovers_fortran_statevector_pas_tiny_with_phi1_in_kinetic() -> None:
    """Solve A x = b matrix-free and recover the Fortran v3 stateVector (tiny PAS + Phi1 in kinetic equation)."""
    here = Path(__file__).parent
    input_path = here / "ref" / "pas_1species_PAS_noEr_tiny_withPhi1_inKinetic_linear.input.namelist"
    mat_path = here / "ref" / "pas_1species_PAS_noEr_tiny_withPhi1_inKinetic_linear.whichMatrix_3.petscbin"
    vec_path = here / "ref" / "pas_1species_PAS_noEr_tiny_withPhi1_inKinetic_linear.stateVector.petscbin"

    nml = read_sfincs_input(input_path)

    # The v3 whichMatrix=3 assembly for includePhi1InKineticEquation linearizes around a base Phi1.
    # Extract this base Phi1 from the frozen Fortran stateVector fixture and pass it into the
    # matrix-free operator so we match the same linearization point.
    op0 = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)
    x_ref = read_petsc_vec(vec_path).values
    phi1_flat = x_ref[op0.f_size : op0.f_size + op0.n_theta * op0.n_zeta]
    phi1_hat_base = phi1_flat.reshape((op0.n_theta, op0.n_zeta))

    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0, phi1_hat_base=jnp.asarray(phi1_hat_base))

    a = read_petsc_mat_aij(mat_path)
    assert x_ref.shape == (op.total_size,)
    a_csr = csr_matrix((a.data, a.col_ind, a.row_ptr), shape=a.shape)

    b = a_csr.dot(x_ref)

    def mv(x):
        return apply_v3_full_system_operator(op, x)

    result = gmres_solve(matvec=mv, b=jnp.asarray(b), tol=1e-12, restart=120, maxiter=600)
    x = np.asarray(result.x)

    np.testing.assert_allclose(x, x_ref, rtol=0, atol=1e-9)
    assert float(result.residual_norm) < 1e-9
