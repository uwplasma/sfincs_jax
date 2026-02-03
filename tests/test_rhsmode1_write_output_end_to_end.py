from __future__ import annotations

from pathlib import Path

import numpy as np

from sfincs_jax.io import read_sfincs_h5, write_sfincs_jax_output_h5


def test_write_output_rhsmode1_solution_fields_match_fortran_fixture(tmp_path: Path) -> None:
    """End-to-end: from input.namelist, solve RHSMode=1 and write solution-derived fields."""
    here = Path(__file__).parent
    base = "pas_1species_PAS_noEr_tiny_scheme1"
    input_path = here / "ref" / f"{base}.input.namelist"
    ref_path = here / "ref" / f"{base}.sfincsOutput.h5"
    out_path = tmp_path / f"{base}.sfincsOutput_jax.h5"

    write_sfincs_jax_output_h5(
        input_namelist=input_path,
        output_path=out_path,
        compute_solution=True,
    )

    out = read_sfincs_h5(out_path)
    ref = read_sfincs_h5(ref_path)

    # For these tiny fixtures, output parity should be tight for the vm-only branch.
    # (NTV and vE-related quantities are currently written as 0 placeholders.)
    checks: list[tuple[str, float]] = [
        # Moments on the grid:
        ("densityPerturbation", 5e-10),
        ("pressurePerturbation", 5e-10),
        ("pressureAnisotropy", 5e-10),
        ("flow", 5e-10),
        ("totalDensity", 5e-10),
        ("totalPressure", 5e-10),
        ("velocityUsingFSADensity", 5e-10),
        ("velocityUsingTotalDensity", 5e-10),
        ("MachUsingFSAThermalSpeed", 5e-10),
        ("jHat", 5e-10),
        # Flux surface averages:
        ("FSADensityPerturbation", 5e-10),
        ("FSAPressurePerturbation", 5e-10),
        ("FSABFlow", 5e-10),
        ("FSABFlow_vs_x", 5e-10),
        ("FSABVelocityUsingFSADensity", 5e-10),
        ("FSABVelocityUsingFSADensityOverB0", 5e-10),
        ("FSABVelocityUsingFSADensityOverRootFSAB2", 5e-10),
        ("FSABjHat", 5e-10),
        ("FSABjHatOverB0", 5e-10),
        ("FSABjHatOverRootFSAB2", 5e-10),
        # Fluxes:
        ("particleFluxBeforeSurfaceIntegral_vm", 5e-10),
        ("particleFluxBeforeSurfaceIntegral_vm0", 5e-10),
        ("heatFluxBeforeSurfaceIntegral_vm", 5e-10),
        ("heatFluxBeforeSurfaceIntegral_vm0", 5e-10),
        ("momentumFluxBeforeSurfaceIntegral_vm", 5e-10),
        ("momentumFluxBeforeSurfaceIntegral_vm0", 5e-10),
        ("particleFlux_vm_psiHat", 5e-10),
        ("particleFlux_vm0_psiHat", 5e-10),
        ("heatFlux_vm_psiHat", 5e-10),
        ("heatFlux_vm0_psiHat", 5e-10),
        ("momentumFlux_vm_psiHat", 5e-10),
        ("momentumFlux_vm0_psiHat", 5e-10),
        ("particleFlux_vm_psiHat_vs_x", 5e-10),
        ("heatFlux_vm_psiHat_vs_x", 5e-10),
        ("sources", 5e-10),
        # Coordinate variants (v3 conversions):
        ("particleFlux_vm_psiN", 5e-10),
        ("particleFlux_vm_rHat", 5e-10),
        ("particleFlux_vm_rN", 5e-10),
        ("particleFlux_vm0_psiN", 5e-10),
        ("particleFlux_vm0_rHat", 5e-10),
        ("particleFlux_vm0_rN", 5e-10),
        ("heatFlux_vm_psiN", 5e-10),
        ("heatFlux_vm_rHat", 5e-10),
        ("heatFlux_vm_rN", 5e-10),
        ("heatFlux_vm0_psiN", 5e-10),
        ("heatFlux_vm0_rHat", 5e-10),
        ("heatFlux_vm0_rN", 5e-10),
        ("momentumFlux_vm_psiN", 5e-10),
        ("momentumFlux_vm_rHat", 5e-10),
        ("momentumFlux_vm_rN", 5e-10),
        ("momentumFlux_vm0_psiN", 5e-10),
        ("momentumFlux_vm0_rHat", 5e-10),
        ("momentumFlux_vm0_rN", 5e-10),
        # Classical terms are present-but-zero for this fixture:
        ("classicalParticleFlux_psiHat", 0.0),
        ("classicalParticleFlux_psiN", 0.0),
        ("classicalParticleFlux_rHat", 0.0),
        ("classicalParticleFlux_rN", 0.0),
        ("classicalHeatFlux_psiHat", 0.0),
        ("classicalHeatFlux_psiN", 0.0),
        ("classicalHeatFlux_rHat", 0.0),
        ("classicalHeatFlux_rN", 0.0),
        # Present-but-zero placeholders:
        ("particleFluxBeforeSurfaceIntegral_vE", 0.0),
        ("particleFluxBeforeSurfaceIntegral_vE0", 0.0),
        ("heatFluxBeforeSurfaceIntegral_vE", 0.0),
        ("heatFluxBeforeSurfaceIntegral_vE0", 0.0),
        ("momentumFluxBeforeSurfaceIntegral_vE", 0.0),
        ("momentumFluxBeforeSurfaceIntegral_vE0", 0.0),
        ("NTVBeforeSurfaceIntegral", 1e-15),
        ("NTV", 1e-15),
    ]

    for key, atol in checks:
        np.testing.assert_allclose(
            np.asarray(out[key], dtype=np.float64),
            np.asarray(ref[key], dtype=np.float64),
            rtol=0.0,
            atol=float(atol),
        )

    # Timings are expected to differ between Fortran and JAX runs, but we still write them for provenance.
    assert "elapsed time (s)" in out
    assert np.asarray(out["elapsed time (s)"]).shape == np.asarray(ref["elapsed time (s)"]).shape
    assert np.all(np.asarray(out["elapsed time (s)"], dtype=np.float64) >= 0.0)
