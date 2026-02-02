from __future__ import annotations

from pathlib import Path

import numpy as np
import jax.numpy as jnp
from scipy.sparse import csr_matrix

from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.petsc_binary import read_petsc_mat_aij, read_petsc_vec
from sfincs_jax.v3_system import apply_v3_full_system_operator, full_system_operator_from_namelist


def _csr_from_petsc(a) -> csr_matrix:
    return csr_matrix((a.data, a.col_ind, a.row_ptr), shape=a.shape)


def test_full_system_matvec_pas_tiny_matches_fortran_matrix() -> None:
    here = Path(__file__).parent
    input_path = here / "ref" / "pas_1species_PAS_noEr_tiny.input.namelist"
    mat_path = here / "ref" / "pas_1species_PAS_noEr_tiny.whichMatrix_3.petscbin"
    vec_path = here / "ref" / "pas_1species_PAS_noEr_tiny.stateVector.petscbin"

    nml = read_sfincs_input(input_path)
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)

    a = read_petsc_mat_aij(mat_path)
    x_ref = read_petsc_vec(vec_path).values
    assert x_ref.shape == (op.total_size,)
    assert a.shape == (op.total_size, op.total_size)

    y_jax = np.asarray(apply_v3_full_system_operator(op, jnp.asarray(x_ref)))
    y_ref = _csr_from_petsc(a).dot(x_ref)

    np.testing.assert_allclose(y_jax, y_ref, rtol=0, atol=1e-12)


def test_full_system_matvec_fp_2species_matches_fortran_matrix() -> None:
    here = Path(__file__).parent
    input_path = here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist"
    mat_path = here / "ref" / "quick_2species_FPCollisions_noEr.whichMatrix_3.petscbin"
    vec_path = here / "ref" / "quick_2species_FPCollisions_noEr.stateVector.petscbin"

    nml = read_sfincs_input(input_path)
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)

    a = read_petsc_mat_aij(mat_path)
    x_ref = read_petsc_vec(vec_path).values
    assert x_ref.shape == (op.total_size,)
    assert a.shape == (op.total_size, op.total_size)

    y_jax = np.asarray(apply_v3_full_system_operator(op, jnp.asarray(x_ref)))
    y_ref = _csr_from_petsc(a).dot(x_ref)

    # The v3 matrix is a fixed numeric object; any mismatch here indicates a true parity issue.
    np.testing.assert_allclose(y_jax, y_ref, rtol=0, atol=3e-12)


def test_full_system_matvec_pas_tiny_with_phi1_linear_matches_fortran_matrix() -> None:
    """IncludePhi1 parity check for the QN/lambda blocks in a tiny PAS case.

    This fixture uses includePhi1=.true. but includePhi1InKineticEquation=.false., so Phi1 enters only
    through the quasineutrality and <Phi1>=0 constraint blocks (parity-first step).
    """
    here = Path(__file__).parent
    input_path = here / "ref" / "pas_1species_PAS_noEr_tiny_withPhi1_linear.input.namelist"
    mat_path = here / "ref" / "pas_1species_PAS_noEr_tiny_withPhi1_linear.whichMatrix_3.petscbin"
    vec_path = here / "ref" / "pas_1species_PAS_noEr_tiny_withPhi1_linear.stateVector.petscbin"

    nml = read_sfincs_input(input_path)
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)

    a = read_petsc_mat_aij(mat_path)
    x_ref = read_petsc_vec(vec_path).values
    assert x_ref.shape == (op.total_size,)
    assert a.shape == (op.total_size, op.total_size)

    y_jax = np.asarray(apply_v3_full_system_operator(op, jnp.asarray(x_ref)))
    y_ref = _csr_from_petsc(a).dot(x_ref)

    np.testing.assert_allclose(y_jax, y_ref, rtol=0, atol=3e-12)
