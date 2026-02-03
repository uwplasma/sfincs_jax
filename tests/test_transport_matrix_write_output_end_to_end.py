from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sfincs_jax.io import read_sfincs_h5, write_sfincs_jax_output_h5


@pytest.mark.parametrize(
    "base",
    (
        "transportMatrix_PAS_tiny_rhsMode2_scheme2",
        "monoenergetic_PAS_tiny_scheme11",
        "monoenergetic_PAS_tiny_scheme5_filtered",
    ),
)
def test_write_output_compute_transport_matrix_matches_fortran_fixture(base: str, tmp_path: Path) -> None:
    """End-to-end: from input.namelist, solve whichRHS and write transport-matrix fields."""
    here = Path(__file__).parent
    input_path = here / "ref" / f"{base}.input.namelist"
    ref_path = here / "ref" / f"{base}.sfincsOutput.h5"
    out_path = tmp_path / f"{base}.sfincsOutput_jax.h5"

    write_sfincs_jax_output_h5(
        input_namelist=input_path,
        output_path=out_path,
        compute_transport_matrix=True,
    )

    out = read_sfincs_h5(out_path)
    ref = read_sfincs_h5(ref_path)

    # For these tiny fixtures, transport-matrix solve parity is expected to be tight.
    for key, atol in (
        ("transportMatrix", 5e-10),
        ("FSABFlow", 5e-10),
        ("FSABjHat", 5e-10),
        ("FSABjHatOverRootFSAB2", 5e-10),
        ("FSABVelocityUsingFSADensity", 5e-10),
        ("particleFlux_vm_psiHat", 5e-10),
        ("heatFlux_vm_psiHat", 5e-10),
        ("particleFlux_vm0_psiHat", 5e-10),
        ("heatFlux_vm0_psiHat", 5e-10),
        ("particleFluxBeforeSurfaceIntegral_vm", 5e-10),
        ("heatFluxBeforeSurfaceIntegral_vm", 5e-10),
        ("particleFluxBeforeSurfaceIntegral_vm0", 5e-10),
        ("heatFluxBeforeSurfaceIntegral_vm0", 5e-10),
        ("particleFlux_vm_psiHat_vs_x", 5e-10),
        ("heatFlux_vm_psiHat_vs_x", 5e-10),
        ("sources", 5e-10),
        ("particleFlux_vm_psiN", 5e-10),
        ("particleFlux_vm_rHat", 5e-10),
        ("particleFlux_vm_rN", 5e-10),
        ("heatFlux_vm_psiN", 5e-10),
        ("heatFlux_vm_rHat", 5e-10),
        ("heatFlux_vm_rN", 5e-10),
    ):
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
