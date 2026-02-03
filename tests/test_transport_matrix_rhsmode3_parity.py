from __future__ import annotations

from pathlib import Path

import numpy as np
import jax.numpy as jnp
import pytest

from sfincs_jax.io import read_sfincs_h5
from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.petsc_binary import read_petsc_vec
from sfincs_jax.transport_matrix import v3_transport_matrix_from_state_vectors
from sfincs_jax.v3 import geometry_from_namelist, grids_from_namelist
from sfincs_jax.v3_system import full_system_operator_from_namelist


@pytest.mark.parametrize(
    "base",
    (
        "monoenergetic_PAS_tiny_scheme1",
        "monoenergetic_PAS_tiny_scheme11",
        "monoenergetic_PAS_tiny_scheme5_filtered",
    ),
)
def test_transport_matrix_rhsmode3_matches_fortran_output(base: str) -> None:
    here = Path(__file__).parent
    input_path = here / "ref" / f"{base}.input.namelist"
    out_path = here / "ref" / f"{base}.sfincsOutput.h5"

    nml = read_sfincs_input(input_path)
    grids = grids_from_namelist(nml)
    geom = geometry_from_namelist(nml=nml, grids=grids)
    op0 = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)
    assert int(op0.rhs_mode) == 3

    vec1 = here / "ref" / f"{base}.whichRHS1.stateVector.petscbin"
    vec2 = here / "ref" / f"{base}.whichRHS2.stateVector.petscbin"
    state_vecs = {
        1: jnp.asarray(read_petsc_vec(vec1).values),
        2: jnp.asarray(read_petsc_vec(vec2).values),
    }

    tm = np.asarray(v3_transport_matrix_from_state_vectors(op0=op0, geom=geom, state_vectors_by_rhs=state_vecs))
    out = read_sfincs_h5(out_path)
    tm_ref = np.asarray(out["transportMatrix"], dtype=np.float64)

    assert tm.shape == (2, 2)
    assert tm_ref.shape == (2, 2)
    # Fortran writes arrays in column-major order; as read by Python, the dataset appears transposed.
    np.testing.assert_allclose(tm.T, tm_ref, rtol=0, atol=2e-10)

    # Also validate the diagnostic fields used by upstream scan plotting scripts.
    # For these extensible diagnostic arrays, the Fortran output is already in (species, whichRHS) order as read by Python.
    pf_ref = np.asarray(out["particleFlux_vm_psiHat"], dtype=np.float64)
    hf_ref = np.asarray(out["heatFlux_vm_psiHat"], dtype=np.float64)
    fsab_ref = np.asarray(out["FSABFlow"], dtype=np.float64)

    # Compute from the solved state vectors:
    from sfincs_jax.transport_matrix import v3_transport_diagnostics_vm_only

    d1 = v3_transport_diagnostics_vm_only(op0, x_full=state_vecs[1])
    d2 = v3_transport_diagnostics_vm_only(op0, x_full=state_vecs[2])
    pf = np.stack([np.asarray(d1.particle_flux_vm_psi_hat), np.asarray(d2.particle_flux_vm_psi_hat)], axis=1)
    hf = np.stack([np.asarray(d1.heat_flux_vm_psi_hat), np.asarray(d2.heat_flux_vm_psi_hat)], axis=1)
    fsab = np.stack([np.asarray(d1.fsab_flow), np.asarray(d2.fsab_flow)], axis=1)

    np.testing.assert_allclose(pf, pf_ref, rtol=0, atol=5e-10)
    np.testing.assert_allclose(hf, hf_ref, rtol=0, atol=5e-10)
    np.testing.assert_allclose(fsab, fsab_ref, rtol=0, atol=5e-10)
