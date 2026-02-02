from __future__ import annotations

from pathlib import Path

import numpy as np
import jax.numpy as jnp
from scipy.sparse import csr_matrix

from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.petsc_binary import read_petsc_mat_aij, read_petsc_vec
from sfincs_jax.v3_system import (
    apply_v3_full_system_operator,
    full_system_operator_from_namelist,
    residual_v3_full_system,
    rhs_v3_full_system,
    with_transport_rhs_settings,
)


def _csr_from_petsc(a) -> csr_matrix:
    return csr_matrix((a.data, a.col_ind, a.row_ptr), shape=a.shape)


def test_monoenergetic_full_system_matvec_matches_fortran_whichmatrix1() -> None:
    """RHSMode=3: matvec parity vs frozen PETSc whichMatrix=1 (solver matrix)."""
    here = Path(__file__).parent
    input_path = here / "ref" / "monoenergetic_PAS_tiny_scheme1.input.namelist"
    mat_path = here / "ref" / "monoenergetic_PAS_tiny_scheme1.whichMatrix_1.petscbin"
    vec_path = here / "ref" / "monoenergetic_PAS_tiny_scheme1.stateVector.petscbin"

    nml = read_sfincs_input(input_path)
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)

    a = read_petsc_mat_aij(mat_path)
    x_ref = read_petsc_vec(vec_path).values
    assert x_ref.shape == (op.total_size,)
    assert a.shape == (op.total_size, op.total_size)

    y_jax = np.asarray(apply_v3_full_system_operator(op, jnp.asarray(x_ref)))
    y_ref = _csr_from_petsc(a).dot(x_ref)

    np.testing.assert_allclose(y_jax, y_ref, rtol=0, atol=3e-12)


def test_monoenergetic_rhsmode3_rhs_matches_fortran_residual_at_zero_state() -> None:
    """RHSMode=3: RHS is built from `evaluateResidual(f=0)`, so residual(0) == -rhs."""
    here = Path(__file__).parent
    input_path = here / "ref" / "monoenergetic_PAS_tiny_scheme1.input.namelist"
    residual_path = here / "ref" / "monoenergetic_PAS_tiny_scheme1.residual.petscbin"

    nml = read_sfincs_input(input_path)
    op0 = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)
    op = with_transport_rhs_settings(op0, which_rhs=1)

    r_ref = read_petsc_vec(residual_path).values
    assert r_ref.shape == (op.total_size,)

    r_jax = np.asarray(residual_v3_full_system(op, jnp.zeros((op.total_size,), dtype=jnp.float64)))
    np.testing.assert_allclose(r_jax, r_ref, rtol=0, atol=3e-12)


def test_monoenergetic_rhsmode3_fortran_statevector_solves_physical_rhs() -> None:
    """RHSMode=3: the frozen v3 stateVector should satisfy A x = RHS assembled by sfincs_jax."""
    here = Path(__file__).parent
    input_path = here / "ref" / "monoenergetic_PAS_tiny_scheme1.input.namelist"
    vec_path = here / "ref" / "monoenergetic_PAS_tiny_scheme1.stateVector.petscbin"

    nml = read_sfincs_input(input_path)
    op0 = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)
    op = with_transport_rhs_settings(op0, which_rhs=1)
    rhs = rhs_v3_full_system(op)

    x_ref = read_petsc_vec(vec_path).values
    assert x_ref.shape == (op.total_size,)

    ax = np.asarray(apply_v3_full_system_operator(op, jnp.asarray(x_ref)))
    np.testing.assert_allclose(ax, np.asarray(rhs), rtol=0, atol=3e-12)
