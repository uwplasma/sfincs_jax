from __future__ import annotations

from pathlib import Path

from sfincs_jax.io import write_sfincs_jax_output_h5


def test_pas_xblock_tz_precond_auto(monkeypatch, tmp_path: Path) -> None:
    """PAS cases with theta/zeta grids should prefer xblock_tz preconditioning when enabled."""
    input_path = Path(__file__).parent / "reduced_inputs" / "geometryScheme4_2species_PAS_noEr.input.namelist"
    out_path = tmp_path / "sfincsOutput_jax.h5"

    logs: list[str] = []

    def emit(_level: int, msg: str) -> None:
        logs.append(msg)

    monkeypatch.setenv("SFINCS_JAX_FORTRAN_STDOUT", "0")
    monkeypatch.setenv("SFINCS_JAX_SOLVER_ITER_STATS", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPECIES_BLOCK_MAX", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_TZ_MAX", "1000")

    write_sfincs_jax_output_h5(
        input_namelist=input_path,
        output_path=out_path,
        compute_solution=True,
        emit=emit,
        verbose=True,
    )

    joined = "\n".join(logs)
    assert "building RHSMode=1 preconditioner=xblock_tz" in joined
