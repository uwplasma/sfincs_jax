from __future__ import annotations

from pathlib import Path

import numpy as np
import jax.numpy as jnp
import pytest

from sfincs_jax.io import read_sfincs_h5
from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.petsc_binary import read_petsc_vec
from sfincs_jax.transport_matrix import v3_transport_matrix_from_state_vectors
from sfincs_jax.transport_matrix import v3_transport_output_fields_vm_only
from sfincs_jax.v3 import geometry_from_namelist, grids_from_namelist
from sfincs_jax.v3_system import (
    apply_v3_full_system_operator,
    full_system_operator_from_namelist,
    residual_v3_full_system,
    rhs_v3_full_system,
    with_transport_rhs_settings,
)


def test_transport_matrix_rhsmode2_matches_fortran_output() -> None:
    here = Path(__file__).parent
    base = "transportMatrix_PAS_tiny_rhsMode2_scheme2"
    input_path = here / "ref" / f"{base}.input.namelist"
    out_path = here / "ref" / f"{base}.sfincsOutput.h5"

    nml = read_sfincs_input(input_path)
    grids = grids_from_namelist(nml)
    geom = geometry_from_namelist(nml=nml, grids=grids)
    op0 = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)
    assert int(op0.rhs_mode) == 2

    state_vecs = {
        1: jnp.asarray(read_petsc_vec(here / "ref" / f"{base}.whichRHS1.stateVector.petscbin").values),
        2: jnp.asarray(read_petsc_vec(here / "ref" / f"{base}.whichRHS2.stateVector.petscbin").values),
        3: jnp.asarray(read_petsc_vec(here / "ref" / f"{base}.whichRHS3.stateVector.petscbin").values),
    }
    tm = np.asarray(v3_transport_matrix_from_state_vectors(op0=op0, geom=geom, state_vectors_by_rhs=state_vecs))
    out = read_sfincs_h5(out_path)
    tm_ref = np.asarray(out["transportMatrix"], dtype=np.float64)

    assert tm.shape == (3, 3)
    assert tm_ref.shape == (3, 3)
    # Fortran writes arrays in column-major order; as read by Python, the dataset appears transposed.
    np.testing.assert_allclose(tm.T, tm_ref, rtol=0, atol=5e-10)

    pf_ref = np.asarray(out["particleFlux_vm_psiHat"], dtype=np.float64)
    hf_ref = np.asarray(out["heatFlux_vm_psiHat"], dtype=np.float64)
    fsab_ref = np.asarray(out["FSABFlow"], dtype=np.float64)

    from sfincs_jax.transport_matrix import v3_transport_diagnostics_vm_only

    d = [v3_transport_diagnostics_vm_only(op0, x_full=state_vecs[k]) for k in (1, 2, 3)]
    pf = np.stack([np.asarray(di.particle_flux_vm_psi_hat) for di in d], axis=1)
    hf = np.stack([np.asarray(di.heat_flux_vm_psi_hat) for di in d], axis=1)
    fsab = np.stack([np.asarray(di.fsab_flow) for di in d], axis=1)

    np.testing.assert_allclose(pf, pf_ref, rtol=0, atol=5e-10)
    np.testing.assert_allclose(hf, hf_ref, rtol=0, atol=5e-10)
    np.testing.assert_allclose(fsab, fsab_ref, rtol=0, atol=5e-10)

    # Validate additional output fields used by the upstream scan plotting scripts.
    fields = v3_transport_output_fields_vm_only(op0=op0, state_vectors_by_rhs=state_vecs)

    np.testing.assert_allclose(np.asarray(fields["FSABjHat"]), np.asarray(out["FSABjHat"]), rtol=0, atol=5e-10)
    np.testing.assert_allclose(
        np.asarray(fields["FSABjHatOverRootFSAB2"]),
        np.asarray(out["FSABjHatOverRootFSAB2"]),
        rtol=0,
        atol=5e-10,
    )
    np.testing.assert_allclose(
        np.asarray(fields["FSABVelocityUsingFSADensity"]),
        np.asarray(out["FSABVelocityUsingFSADensity"]),
        rtol=0,
        atol=5e-10,
    )

    # Before-surface-integral and vs_x fields:
    np.testing.assert_allclose(
        np.asarray(fields["particleFluxBeforeSurfaceIntegral_vm"]),
        np.asarray(out["particleFluxBeforeSurfaceIntegral_vm"]),
        rtol=0,
        atol=5e-10,
    )
    np.testing.assert_allclose(
        np.asarray(fields["heatFluxBeforeSurfaceIntegral_vm"]),
        np.asarray(out["heatFluxBeforeSurfaceIntegral_vm"]),
        rtol=0,
        atol=5e-10,
    )
    np.testing.assert_allclose(
        np.asarray(fields["particleFlux_vm_psiHat_vs_x"]),
        np.asarray(out["particleFlux_vm_psiHat_vs_x"]),
        rtol=0,
        atol=5e-10,
    )
    np.testing.assert_allclose(
        np.asarray(fields["heatFlux_vm_psiHat_vs_x"]),
        np.asarray(out["heatFlux_vm_psiHat_vs_x"]),
        rtol=0,
        atol=5e-10,
    )

    # Coordinate variants follow v3 radial-coordinate conversion factors:
    psi_a_hat = float(np.asarray(out["psiAHat"]))
    a_hat = float(np.asarray(out["aHat"]))
    r_n = float(np.asarray(out["rN"]))
    ddpsiN2ddpsiHat = 1.0 / psi_a_hat
    ddrHat2ddpsiHat = a_hat / (2.0 * psi_a_hat * r_n)
    ddrN2ddpsiHat = 1.0 / (2.0 * psi_a_hat * r_n)

    pf = np.asarray(fields["particleFlux_vm_psiHat"], dtype=np.float64)
    hf = np.asarray(fields["heatFlux_vm_psiHat"], dtype=np.float64)
    np.testing.assert_allclose(pf * ddpsiN2ddpsiHat, np.asarray(out["particleFlux_vm_psiN"]), rtol=0, atol=5e-10)
    np.testing.assert_allclose(pf * ddrHat2ddpsiHat, np.asarray(out["particleFlux_vm_rHat"]), rtol=0, atol=5e-10)
    np.testing.assert_allclose(pf * ddrN2ddpsiHat, np.asarray(out["particleFlux_vm_rN"]), rtol=0, atol=5e-10)
    np.testing.assert_allclose(hf * ddpsiN2ddpsiHat, np.asarray(out["heatFlux_vm_psiN"]), rtol=0, atol=5e-10)
    np.testing.assert_allclose(hf * ddrHat2ddpsiHat, np.asarray(out["heatFlux_vm_rHat"]), rtol=0, atol=5e-10)
    np.testing.assert_allclose(hf * ddrN2ddpsiHat, np.asarray(out["heatFlux_vm_rN"]), rtol=0, atol=5e-10)

    # sources are the constraint-scheme-2 extra unknowns (per-x L=0 source), in (X,S,whichRHS) order.
    np.testing.assert_allclose(np.asarray(fields["sources"]), np.asarray(out["sources"]), rtol=0, atol=5e-10)


@pytest.mark.parametrize("which_rhs", (1, 2, 3))
def test_transport_matrix_rhsmode2_rhs_matches_fortran_residual_at_zero_state(which_rhs: int) -> None:
    here = Path(__file__).parent
    base = "transportMatrix_PAS_tiny_rhsMode2_scheme2"
    input_path = here / "ref" / f"{base}.input.namelist"
    residual_path = here / "ref" / f"{base}.whichRHS{int(which_rhs)}.residual.petscbin"

    nml = read_sfincs_input(input_path)
    op0 = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)
    op = with_transport_rhs_settings(op0, which_rhs=int(which_rhs))

    r_ref = read_petsc_vec(residual_path).values
    r_jax = np.asarray(residual_v3_full_system(op, jnp.zeros((op.total_size,), dtype=jnp.float64)))
    np.testing.assert_allclose(r_jax, r_ref, rtol=0, atol=3e-12)


@pytest.mark.parametrize("which_rhs", (1, 2, 3))
def test_transport_matrix_rhsmode2_fortran_statevector_solves_physical_rhs(which_rhs: int) -> None:
    here = Path(__file__).parent
    base = "transportMatrix_PAS_tiny_rhsMode2_scheme2"
    input_path = here / "ref" / f"{base}.input.namelist"
    vec_path = here / "ref" / f"{base}.whichRHS{int(which_rhs)}.stateVector.petscbin"

    nml = read_sfincs_input(input_path)
    op0 = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)
    op = with_transport_rhs_settings(op0, which_rhs=int(which_rhs))
    rhs = rhs_v3_full_system(op)

    x_ref = read_petsc_vec(vec_path).values
    ax = np.asarray(apply_v3_full_system_operator(op, jnp.asarray(x_ref)))

    # These vectors come from PETSc solves with a finite tolerance; allow a modest absolute tolerance.
    np.testing.assert_allclose(ax, np.asarray(rhs), rtol=0, atol=5e-8)
