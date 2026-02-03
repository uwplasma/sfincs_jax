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
        "pas_1species_PAS_noEr_tiny_scheme1",
        "pas_1species_PAS_noEr_tiny_scheme5",
        "pas_1species_PAS_noEr_tiny_scheme11",
        "pas_1species_PAS_noEr_tiny_scheme12",
    ),
)
def test_write_output_rhsmode1_solution_fields_match_fortran_fixture(base: str, tmp_path: Path) -> None:
    """End-to-end: from input.namelist, solve RHSMode=1 and write solution-derived fields."""
    here = Path(__file__).parent
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
    assert set(out.keys()) == set(ref.keys())

    # Full-file numeric parity (excluding embedded input text).
    # uHat involves long FFTs/reductions and is compared with a slightly looser atol.
    # GeometryScheme=12 uses non-stellarator-symmetric Boozer `.bc` inputs and can show slightly
    # larger floating-point differences in some geometry *derivative* arrays due to reduction
    # ordering and cancellation in large-magnitude terms.
    atol = 2e-9 if base.endswith("scheme12") else 5e-10
    atol_uhat = 1e-8
    scheme12_key_atol = {
        # These derivatives can accumulate small absolute differences ~1e-8.
        "dBHat_sup_theta_dpsiHat": 1e-7,
        "dBHat_sup_zeta_dpsiHat": 1e-7,
        # This metric coefficient can differ at the ~1e-3 level in absolute terms while still
        # matching to ~1e-10 relative on typical v3 fixtures (values are O(1e6)).
        "gpsiHatpsiHat": 1e-3,
    }

    for k in sorted(ref.keys()):
        if k == "input.namelist":
            continue
        if not _is_numeric_dataset(ref[k]):
            continue
        if base.endswith("scheme12") and k in scheme12_key_atol:
            this_atol = scheme12_key_atol[k]
        else:
            this_atol = atol_uhat if k == "uHat" else atol
        np.testing.assert_allclose(
            np.asarray(out[k], dtype=np.float64),
            np.asarray(ref[k], dtype=np.float64),
            rtol=0.0,
            atol=float(this_atol),
        )

    # Timings are expected to differ between Fortran and JAX runs, but we still write them for provenance.
    assert "elapsed time (s)" in out
    assert np.asarray(out["elapsed time (s)"]).shape == np.asarray(ref["elapsed time (s)"]).shape
    assert np.all(np.asarray(out["elapsed time (s)"], dtype=np.float64) >= 0.0)
