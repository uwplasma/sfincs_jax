from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sfincs_jax.io import read_sfincs_h5, write_sfincs_jax_output_h5


def _is_numeric_dataset(x) -> bool:
    if isinstance(x, (str, bytes)):
        return False
    if isinstance(x, np.ndarray) and x.dtype.kind in {"S", "U", "O"}:
        return False
    try:
        np.asarray(x, dtype=np.float64)
        return True
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.parametrize(
    "base",
    (
        "transportMatrix_PAS_tiny_rhsMode2_scheme2",
        "transportMatrix_PAS_tiny_rhsMode2_scheme11",
        "transportMatrix_PAS_tiny_rhsMode2_scheme5_filtered",
        "monoenergetic_PAS_tiny_scheme11",
        "monoenergetic_PAS_tiny_scheme12",
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
    # The synthetic geometryScheme=12 fixture is slightly more sensitive numerically; allow a looser atol.
    atol_strict = 1e-6 if base == "monoenergetic_PAS_tiny_scheme12" else 5e-10
    for key, atol in (
        ("transportMatrix", atol_strict),
        ("FSABFlow", atol_strict),
        ("FSABjHat", atol_strict),
        ("FSABjHatOverRootFSAB2", atol_strict),
        ("FSABVelocityUsingFSADensity", atol_strict),
        ("particleFlux_vm_psiHat", atol_strict),
        ("heatFlux_vm_psiHat", atol_strict),
        ("particleFlux_vm0_psiHat", atol_strict),
        ("heatFlux_vm0_psiHat", atol_strict),
        ("particleFluxBeforeSurfaceIntegral_vm", atol_strict),
        ("heatFluxBeforeSurfaceIntegral_vm", atol_strict),
        ("particleFluxBeforeSurfaceIntegral_vm0", atol_strict),
        ("heatFluxBeforeSurfaceIntegral_vm0", atol_strict),
        ("particleFlux_vm_psiHat_vs_x", atol_strict),
        ("heatFlux_vm_psiHat_vs_x", atol_strict),
        ("sources", atol_strict),
        ("particleFlux_vm_psiN", atol_strict),
        ("particleFlux_vm_rHat", atol_strict),
        ("particleFlux_vm_rN", atol_strict),
        ("heatFlux_vm_psiN", atol_strict),
        ("heatFlux_vm_rHat", atol_strict),
        ("heatFlux_vm_rN", atol_strict),
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

    # Full-file parity (excluding timing and embedded input text).
    # Some scalar fields (e.g. Er) can differ at ~1e-8 for synthetic fixtures due to
    # radial-surface snapping and floating-point roundoff; keep a tight but forgiving atol.
    atol_full = 1e-6 if base == "monoenergetic_PAS_tiny_scheme12" else 5e-8
    rtol_full = 1e-12
    assert set(out.keys()) == set(ref.keys())
    for k in sorted(ref.keys()):
        if k in {"input.namelist", "elapsed time (s)"}:
            continue
        if not _is_numeric_dataset(ref[k]):
            continue
        np.testing.assert_allclose(
            np.asarray(out[k], dtype=np.float64),
            np.asarray(ref[k], dtype=np.float64),
            rtol=rtol_full,
            atol=atol_full,
        )
