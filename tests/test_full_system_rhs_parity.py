from __future__ import annotations

from pathlib import Path

import numpy as np
import jax.numpy as jnp
from scipy.sparse import csr_matrix

from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.petsc_binary import read_petsc_mat_aij, read_petsc_vec
from sfincs_jax.v3_system import full_system_operator_from_namelist, rhs_v3_full_system


def _csr_from_petsc(a) -> csr_matrix:
    return csr_matrix((a.data, a.col_ind, a.row_ptr), shape=a.shape)


def _rhs_ref_from_fortran_matrix_and_residual(*, mat_path: Path, vec_path: Path, residual_path: Path) -> np.ndarray:
    a = read_petsc_mat_aij(mat_path)
    x_ref = read_petsc_vec(vec_path).values
    r_ref = read_petsc_vec(residual_path).values
    # Fortran evaluateResidual: residual = A(stateVec)*stateVec - rhs.
    return _csr_from_petsc(a).dot(x_ref) - r_ref


def test_full_system_rhs_pas_tiny_matches_fortran() -> None:
    here = Path(__file__).parent
    input_path = here / "ref" / "pas_1species_PAS_noEr_tiny.input.namelist"
    mat_path = here / "ref" / "pas_1species_PAS_noEr_tiny.whichMatrix_3.petscbin"
    vec_path = here / "ref" / "pas_1species_PAS_noEr_tiny.stateVector.petscbin"
    residual_path = here / "ref" / "pas_1species_PAS_noEr_tiny.residual.petscbin"

    nml = read_sfincs_input(input_path)
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)

    rhs = np.asarray(rhs_v3_full_system(op))
    rhs_ref = _rhs_ref_from_fortran_matrix_and_residual(mat_path=mat_path, vec_path=vec_path, residual_path=residual_path)

    np.testing.assert_allclose(rhs, rhs_ref, rtol=0, atol=3e-12)


def test_full_system_rhs_fp_2species_matches_fortran() -> None:
    here = Path(__file__).parent
    input_path = here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist"
    mat_path = here / "ref" / "quick_2species_FPCollisions_noEr.whichMatrix_3.petscbin"
    vec_path = here / "ref" / "quick_2species_FPCollisions_noEr.stateVector.petscbin"
    residual_path = here / "ref" / "quick_2species_FPCollisions_noEr.residual.petscbin"

    nml = read_sfincs_input(input_path)
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)

    rhs = np.asarray(rhs_v3_full_system(op))
    rhs_ref = _rhs_ref_from_fortran_matrix_and_residual(mat_path=mat_path, vec_path=vec_path, residual_path=residual_path)

    np.testing.assert_allclose(rhs, rhs_ref, rtol=0, atol=3e-12)


def test_full_system_rhs_pas_tiny_with_phi1_linear_matches_fortran() -> None:
    here = Path(__file__).parent
    input_path = here / "ref" / "pas_1species_PAS_noEr_tiny_withPhi1_linear.input.namelist"
    mat_path = here / "ref" / "pas_1species_PAS_noEr_tiny_withPhi1_linear.whichMatrix_3.petscbin"
    vec_path = here / "ref" / "pas_1species_PAS_noEr_tiny_withPhi1_linear.stateVector.petscbin"
    residual_path = here / "ref" / "pas_1species_PAS_noEr_tiny_withPhi1_linear.residual.petscbin"

    nml = read_sfincs_input(input_path)
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)

    rhs = np.asarray(rhs_v3_full_system(op))
    rhs_ref = _rhs_ref_from_fortran_matrix_and_residual(mat_path=mat_path, vec_path=vec_path, residual_path=residual_path)

    np.testing.assert_allclose(rhs, rhs_ref, rtol=0, atol=3e-12)


def test_full_system_rhs_pas_tiny_with_phi1_in_kinetic_matches_fortran() -> None:
    here = Path(__file__).parent
    input_path = here / "ref" / "pas_1species_PAS_noEr_tiny_withPhi1_inKinetic_linear.input.namelist"
    mat_path = here / "ref" / "pas_1species_PAS_noEr_tiny_withPhi1_inKinetic_linear.whichMatrix_3.petscbin"
    vec_path = here / "ref" / "pas_1species_PAS_noEr_tiny_withPhi1_inKinetic_linear.stateVector.petscbin"
    residual_path = here / "ref" / "pas_1species_PAS_noEr_tiny_withPhi1_inKinetic_linear.residual.petscbin"

    nml = read_sfincs_input(input_path)
    op0 = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)
    x_ref = read_petsc_vec(vec_path).values
    phi1_flat = x_ref[op0.f_size : op0.f_size + op0.n_theta * op0.n_zeta]
    phi1_hat_base = jnp.asarray(phi1_flat.reshape((op0.n_theta, op0.n_zeta)))
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0, phi1_hat_base=phi1_hat_base)

    rhs = np.asarray(rhs_v3_full_system(op))
    rhs_ref = _rhs_ref_from_fortran_matrix_and_residual(mat_path=mat_path, vec_path=vec_path, residual_path=residual_path)

    np.testing.assert_allclose(rhs, rhs_ref, rtol=0, atol=3e-12)
