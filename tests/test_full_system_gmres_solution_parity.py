from __future__ import annotations

from pathlib import Path

import numpy as np
import jax.numpy as jnp
from scipy.sparse import csr_matrix

from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.petsc_binary import read_petsc_mat_aij, read_petsc_vec
from sfincs_jax.solver import gmres_solve
from sfincs_jax.v3_system import apply_v3_full_system_operator, full_system_operator_from_namelist, rhs_v3_full_system


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

    np.testing.assert_allclose(x, x_ref, rtol=1e-5, atol=1e-9)
    assert float(result.residual_norm) < 1e-9


def test_full_system_gmres_solves_physical_rhs_pas_tiny() -> None:
    """Solve A x = RHS assembled by sfincs_jax and recover the Fortran v3 solution (tiny PAS case)."""
    here = Path(__file__).parent
    input_path = here / "ref" / "pas_1species_PAS_noEr_tiny.input.namelist"
    vec_path = here / "ref" / "pas_1species_PAS_noEr_tiny.stateVector.petscbin"

    nml = read_sfincs_input(input_path)
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)
    rhs = rhs_v3_full_system(op)

    x_ref = read_petsc_vec(vec_path).values
    assert x_ref.shape == (op.total_size,)

    def mv(x):
        return apply_v3_full_system_operator(op, x)

    result = gmres_solve(matvec=mv, b=rhs, tol=1e-12, restart=80, maxiter=250)
    x = np.asarray(result.x)

    np.testing.assert_allclose(x, x_ref, rtol=1e-5, atol=1e-9)
    assert float(result.residual_norm) < 1e-9


def test_full_system_gmres_recovers_fortran_statevector_pas_tiny_scheme1() -> None:
    """Solve A x = b matrix-free and recover the Fortran v3 stateVector (tiny PAS, geometryScheme=1)."""
    here = Path(__file__).parent
    input_path = here / "ref" / "pas_1species_PAS_noEr_tiny_scheme1.input.namelist"
    mat_path = here / "ref" / "pas_1species_PAS_noEr_tiny_scheme1.whichMatrix_3.petscbin"
    vec_path = here / "ref" / "pas_1species_PAS_noEr_tiny_scheme1.stateVector.petscbin"

    nml = read_sfincs_input(input_path)
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)

    a = read_petsc_mat_aij(mat_path)
    x_ref = read_petsc_vec(vec_path).values
    assert x_ref.shape == (op.total_size,)
    a_csr = csr_matrix((a.data, a.col_ind, a.row_ptr), shape=a.shape)

    b = a_csr.dot(x_ref)

    def mv(x):
        return apply_v3_full_system_operator(op, x)

    result = gmres_solve(matvec=mv, b=jnp.asarray(b), tol=1e-12, restart=80, maxiter=220)
    x = np.asarray(result.x)

    np.testing.assert_allclose(x, x_ref, rtol=1e-5, atol=1e-9)
    assert float(result.residual_norm) < 1e-9


def test_full_system_gmres_solves_physical_rhs_pas_tiny_scheme1() -> None:
    """Solve A x = RHS assembled by sfincs_jax and recover the Fortran v3 solution (tiny PAS, geometryScheme=1)."""
    here = Path(__file__).parent
    input_path = here / "ref" / "pas_1species_PAS_noEr_tiny_scheme1.input.namelist"
    vec_path = here / "ref" / "pas_1species_PAS_noEr_tiny_scheme1.stateVector.petscbin"

    nml = read_sfincs_input(input_path)
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)
    rhs = rhs_v3_full_system(op)

    x_ref = read_petsc_vec(vec_path).values
    assert x_ref.shape == (op.total_size,)

    def mv(x):
        return apply_v3_full_system_operator(op, x)

    result = gmres_solve(matvec=mv, b=rhs, tol=1e-12, restart=80, maxiter=250)
    x = np.asarray(result.x)

    np.testing.assert_allclose(x, x_ref, rtol=1e-5, atol=1e-9)
    assert float(result.residual_norm) < 1e-9


def test_full_system_gmres_recovers_fortran_statevector_pas_tiny_scheme11() -> None:
    """Solve A x = b matrix-free and recover the Fortran v3 stateVector (tiny PAS, geometryScheme=11)."""
    here = Path(__file__).parent
    input_path = here / "ref" / "pas_1species_PAS_noEr_tiny_scheme11.input.namelist"
    mat_path = here / "ref" / "pas_1species_PAS_noEr_tiny_scheme11.whichMatrix_3.petscbin"
    vec_path = here / "ref" / "pas_1species_PAS_noEr_tiny_scheme11.stateVector.petscbin"

    nml = read_sfincs_input(input_path)
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)

    a = read_petsc_mat_aij(mat_path)
    x_ref = read_petsc_vec(vec_path).values
    assert x_ref.shape == (op.total_size,)
    a_csr = csr_matrix((a.data, a.col_ind, a.row_ptr), shape=a.shape)

    b = a_csr.dot(x_ref)

    def mv(x):
        return apply_v3_full_system_operator(op, x)

    result = gmres_solve(matvec=mv, b=jnp.asarray(b), tol=1e-12, restart=80, maxiter=250)
    x = np.asarray(result.x)

    np.testing.assert_allclose(x, x_ref, rtol=1e-5, atol=1e-9)
    assert float(result.residual_norm) < 1e-9


def test_full_system_gmres_solves_physical_rhs_pas_tiny_scheme11() -> None:
    """Solve A x = RHS assembled by sfincs_jax and recover the Fortran v3 solution (tiny PAS, geometryScheme=11)."""
    here = Path(__file__).parent
    input_path = here / "ref" / "pas_1species_PAS_noEr_tiny_scheme11.input.namelist"
    vec_path = here / "ref" / "pas_1species_PAS_noEr_tiny_scheme11.stateVector.petscbin"

    nml = read_sfincs_input(input_path)
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)
    rhs = rhs_v3_full_system(op)

    x_ref = read_petsc_vec(vec_path).values
    assert x_ref.shape == (op.total_size,)

    def mv(x):
        return apply_v3_full_system_operator(op, x)

    result = gmres_solve(matvec=mv, b=rhs, tol=1e-12, restart=80, maxiter=300)
    x = np.asarray(result.x)

    np.testing.assert_allclose(x, x_ref, rtol=1e-5, atol=1e-9)
    assert float(result.residual_norm) < 1e-9


def test_full_system_gmres_recovers_fortran_statevector_pas_tiny_scheme12() -> None:
    """Solve A x = b matrix-free and recover the Fortran v3 stateVector (tiny PAS, geometryScheme=12)."""
    here = Path(__file__).parent
    input_path = here / "ref" / "pas_1species_PAS_noEr_tiny_scheme12.input.namelist"
    mat_path = here / "ref" / "pas_1species_PAS_noEr_tiny_scheme12.whichMatrix_3.petscbin"
    vec_path = here / "ref" / "pas_1species_PAS_noEr_tiny_scheme12.stateVector.petscbin"

    nml = read_sfincs_input(input_path)
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)

    a = read_petsc_mat_aij(mat_path)
    x_ref = read_petsc_vec(vec_path).values
    assert x_ref.shape == (op.total_size,)
    a_csr = csr_matrix((a.data, a.col_ind, a.row_ptr), shape=a.shape)

    b = a_csr.dot(x_ref)

    def mv(x):
        return apply_v3_full_system_operator(op, x)

    result = gmres_solve(matvec=mv, b=jnp.asarray(b), tol=1e-12, restart=80, maxiter=250)
    x = np.asarray(result.x)

    np.testing.assert_allclose(x, x_ref, rtol=0, atol=1e-9)
    assert float(result.residual_norm) < 1e-9


def test_full_system_gmres_solves_physical_rhs_pas_tiny_scheme12() -> None:
    """Solve A x = RHS assembled by sfincs_jax and recover the Fortran v3 solution (tiny PAS, geometryScheme=12)."""
    here = Path(__file__).parent
    input_path = here / "ref" / "pas_1species_PAS_noEr_tiny_scheme12.input.namelist"
    vec_path = here / "ref" / "pas_1species_PAS_noEr_tiny_scheme12.stateVector.petscbin"

    nml = read_sfincs_input(input_path)
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)
    rhs = rhs_v3_full_system(op)

    x_ref = read_petsc_vec(vec_path).values
    assert x_ref.shape == (op.total_size,)

    def mv(x):
        return apply_v3_full_system_operator(op, x)

    result = gmres_solve(matvec=mv, b=rhs, tol=1e-12, restart=80, maxiter=300)
    x = np.asarray(result.x)

    np.testing.assert_allclose(x, x_ref, rtol=0, atol=1e-9)
    assert float(result.residual_norm) < 1e-9


def test_full_system_gmres_recovers_fortran_statevector_pas_tiny_scheme5() -> None:
    """Solve A x = b matrix-free and recover the Fortran v3 stateVector (tiny PAS, geometryScheme=5)."""
    here = Path(__file__).parent
    input_path = here / "ref" / "pas_1species_PAS_noEr_tiny_scheme5.input.namelist"
    mat_path = here / "ref" / "pas_1species_PAS_noEr_tiny_scheme5.whichMatrix_3.petscbin"
    vec_path = here / "ref" / "pas_1species_PAS_noEr_tiny_scheme5.stateVector.petscbin"

    nml = read_sfincs_input(input_path)
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)

    a = read_petsc_mat_aij(mat_path)
    x_ref = read_petsc_vec(vec_path).values
    assert x_ref.shape == (op.total_size,)
    a_csr = csr_matrix((a.data, a.col_ind, a.row_ptr), shape=a.shape)

    b = a_csr.dot(x_ref)

    def mv(x):
        return apply_v3_full_system_operator(op, x)

    result = gmres_solve(matvec=mv, b=jnp.asarray(b), tol=1e-12, restart=80, maxiter=220)
    x = np.asarray(result.x)

    np.testing.assert_allclose(x, x_ref, rtol=0, atol=1e-9)
    assert float(result.residual_norm) < 1e-9


def test_full_system_gmres_solves_physical_rhs_pas_tiny_scheme5() -> None:
    """Solve A x = RHS assembled by sfincs_jax and recover the Fortran v3 solution (tiny PAS, geometryScheme=5)."""
    here = Path(__file__).parent
    input_path = here / "ref" / "pas_1species_PAS_noEr_tiny_scheme5.input.namelist"
    vec_path = here / "ref" / "pas_1species_PAS_noEr_tiny_scheme5.stateVector.petscbin"

    nml = read_sfincs_input(input_path)
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)
    rhs = rhs_v3_full_system(op)

    x_ref = read_petsc_vec(vec_path).values
    assert x_ref.shape == (op.total_size,)

    def mv(x):
        return apply_v3_full_system_operator(op, x)

    result = gmres_solve(matvec=mv, b=rhs, tol=1e-12, restart=80, maxiter=280)
    x = np.asarray(result.x)

    np.testing.assert_allclose(x, x_ref, rtol=0, atol=1e-9)
    assert float(result.residual_norm) < 1e-9


def test_full_system_gmres_recovers_fortran_statevector_pas_tiny_scheme5_with_phi1_linear() -> None:
    """Solve A x = b matrix-free and recover the Fortran v3 stateVector (tiny PAS + Phi1 blocks, geometryScheme=5)."""
    here = Path(__file__).parent
    input_path = here / "ref" / "pas_1species_PAS_noEr_tiny_scheme5_withPhi1_linear.input.namelist"
    mat_path = here / "ref" / "pas_1species_PAS_noEr_tiny_scheme5_withPhi1_linear.whichMatrix_3.petscbin"
    vec_path = here / "ref" / "pas_1species_PAS_noEr_tiny_scheme5_withPhi1_linear.stateVector.petscbin"

    nml = read_sfincs_input(input_path)
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)

    a = read_petsc_mat_aij(mat_path)
    x_ref = read_petsc_vec(vec_path).values
    assert x_ref.shape == (op.total_size,)
    a_csr = csr_matrix((a.data, a.col_ind, a.row_ptr), shape=a.shape)

    b = a_csr.dot(x_ref)

    def mv(x):
        return apply_v3_full_system_operator(op, x)

    result = gmres_solve(matvec=mv, b=jnp.asarray(b), tol=1e-12, restart=120, maxiter=500)
    x = np.asarray(result.x)

    np.testing.assert_allclose(x, x_ref, rtol=0, atol=1e-9)
    assert float(result.residual_norm) < 1e-9


def test_full_system_gmres_solves_physical_rhs_pas_tiny_scheme5_with_phi1_linear() -> None:
    """Solve A x = RHS assembled by sfincs_jax and recover the Fortran v3 solution (tiny PAS + Phi1 blocks, geometryScheme=5)."""
    here = Path(__file__).parent
    input_path = here / "ref" / "pas_1species_PAS_noEr_tiny_scheme5_withPhi1_linear.input.namelist"
    vec_path = here / "ref" / "pas_1species_PAS_noEr_tiny_scheme5_withPhi1_linear.stateVector.petscbin"

    nml = read_sfincs_input(input_path)
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)
    rhs = rhs_v3_full_system(op)

    x_ref = read_petsc_vec(vec_path).values
    assert x_ref.shape == (op.total_size,)

    def mv(x):
        return apply_v3_full_system_operator(op, x)

    result = gmres_solve(matvec=mv, b=rhs, tol=1e-12, restart=120, maxiter=600)
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


def test_full_system_gmres_solves_physical_rhs_pas_tiny_with_phi1_linear() -> None:
    """Solve A x = RHS assembled by sfincs_jax and recover the Fortran v3 solution (tiny PAS + Phi1 blocks)."""
    here = Path(__file__).parent
    input_path = here / "ref" / "pas_1species_PAS_noEr_tiny_withPhi1_linear.input.namelist"
    vec_path = here / "ref" / "pas_1species_PAS_noEr_tiny_withPhi1_linear.stateVector.petscbin"

    nml = read_sfincs_input(input_path)
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)
    rhs = rhs_v3_full_system(op)

    x_ref = read_petsc_vec(vec_path).values
    assert x_ref.shape == (op.total_size,)

    def mv(x):
        return apply_v3_full_system_operator(op, x)

    result = gmres_solve(matvec=mv, b=rhs, tol=1e-12, restart=120, maxiter=600)
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


def test_full_system_gmres_recovers_fortran_statevector_fp_tiny_with_phi1_in_collision() -> None:
    """Solve A x = b matrix-free and recover the Fortran v3 stateVector (tiny FP + Phi1 in collision operator)."""
    here = Path(__file__).parent
    input_path = here / "ref" / "fp_1species_FPCollisions_noEr_tiny_withPhi1_inCollision.input.namelist"
    mat_path = here / "ref" / "fp_1species_FPCollisions_noEr_tiny_withPhi1_inCollision.whichMatrix_3.petscbin"
    vec_path = here / "ref" / "fp_1species_FPCollisions_noEr_tiny_withPhi1_inCollision.stateVector.petscbin"

    nml = read_sfincs_input(input_path)
    op0 = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)

    a = read_petsc_mat_aij(mat_path)
    x_ref = read_petsc_vec(vec_path).values
    assert x_ref.shape == (op0.total_size,)
    a_csr = csr_matrix((a.data, a.col_ind, a.row_ptr), shape=a.shape)

    phi1_flat = x_ref[op0.f_size : op0.f_size + op0.n_theta * op0.n_zeta]
    phi1_hat_base = jnp.asarray(phi1_flat.reshape((op0.n_theta, op0.n_zeta)))
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0, phi1_hat_base=phi1_hat_base)

    b = a_csr.dot(x_ref)

    def mv(x):
        return apply_v3_full_system_operator(op, x)

    result = gmres_solve(matvec=mv, b=jnp.asarray(b), tol=1e-12, restart=160, maxiter=900)
    x = np.asarray(result.x)

    np.testing.assert_allclose(x, x_ref, rtol=0, atol=2e-9)
    assert float(result.residual_norm) < 2e-9
