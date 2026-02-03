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
        "pas_1species_PAS_noEr_tiny_withPhi1_linear",
        "pas_1species_PAS_noEr_tiny_withPhi1_inKinetic_linear",
        "fp_1species_FPCollisions_noEr_tiny_withPhi1_inCollision",
        "pas_1species_PAS_noEr_tiny_scheme5_withPhi1_linear",
    ),
)
def test_write_output_rhsmode1_phi1_fixtures_match_fortran_end_to_end(base: str, tmp_path: Path) -> None:
    """End-to-end: solve RHSMode=1 includePhi1 fixtures and write a v3-style sfincsOutput.h5."""
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

    # Newton-based includePhi1 runs can differ at ~1e-9 due to floating-point and inner-solve details.
    # Keep a tight absolute tolerance consistent with other Phi1 fixture tests.
    atol = 5e-8

    for k in sorted(ref.keys()):
        if k == "input.namelist":
            continue
        if not _is_numeric_dataset(ref[k]):
            continue
        np.testing.assert_allclose(
            np.asarray(out[k], dtype=np.float64),
            np.asarray(ref[k], dtype=np.float64),
            rtol=0.0,
            atol=atol,
        )
