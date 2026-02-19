from __future__ import annotations

from pathlib import Path

from sfincs_jax.io import write_sfincs_jax_output_h5


def test_sparse_jax_preconditioner_runs(tmp_path, monkeypatch) -> None:
    """Ensure the JAX-native sparse preconditioner runs end-to-end."""
    here = Path(__file__).parent
    input_path = here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist"
    assert input_path.exists()

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_PRECOND", "jax")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_JAX_MAX_MB", "256")
    monkeypatch.setenv("SFINCS_JAX_IMPLICIT_SOLVE", "1")
    monkeypatch.setenv("SFINCS_JAX_FORTRAN_STDOUT", "0")
    monkeypatch.setenv("SFINCS_JAX_SOLVER_ITER_STATS", "0")

    out_path = tmp_path / "sfincsOutput.h5"
    write_sfincs_jax_output_h5(
        input_namelist=input_path,
        output_path=out_path,
        compute_solution=True,
    )
    assert out_path.exists()
