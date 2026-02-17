from __future__ import annotations

from pathlib import Path

import pytest

from sfincs_jax.compare import compare_sfincs_outputs
from sfincs_jax.io import write_sfincs_jax_output_h5


def test_state_recycle_preserves_solution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Recycling states should not change the converged solution."""
    here = Path(__file__).parent
    input_path = here / "ref" / "pas_1species_PAS_noEr_tiny.input.namelist"
    assert input_path.exists()

    out1 = tmp_path / "sfincsOutput_base.h5"
    out2 = tmp_path / "sfincsOutput_recycle.h5"
    state_path = tmp_path / "sfincs_jax_state.npz"

    monkeypatch.setenv("SFINCS_JAX_FORTRAN_STDOUT", "0")
    monkeypatch.setenv("SFINCS_JAX_STATE_OUT", str(state_path))
    monkeypatch.delenv("SFINCS_JAX_STATE_IN", raising=False)
    write_sfincs_jax_output_h5(input_namelist=input_path, output_path=out1, compute_solution=True)

    monkeypatch.setenv("SFINCS_JAX_STATE_IN", str(state_path))
    monkeypatch.delenv("SFINCS_JAX_STATE_OUT", raising=False)
    write_sfincs_jax_output_h5(input_namelist=input_path, output_path=out2, compute_solution=True)

    results = compare_sfincs_outputs(a_path=out1, b_path=out2, rtol=1e-12, atol=1e-12)
    assert all(result.ok for result in results)
