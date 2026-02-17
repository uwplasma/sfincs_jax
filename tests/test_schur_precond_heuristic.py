from __future__ import annotations

from pathlib import Path

from sfincs_jax.io import write_sfincs_jax_output_h5


def test_full_precond_uses_schur_for_constraint_scheme2(tmp_path: Path, monkeypatch) -> None:
    input_path = Path(__file__).parent / "reduced_inputs" / "tokamak_1species_PASCollisions_noEr_Nx1.input.namelist"
    out_path = tmp_path / "sfincsOutput_jax.h5"

    logs: list[str] = []

    def emit(_level: int, msg: str) -> None:
        logs.append(msg)

    monkeypatch.setenv("SFINCS_JAX_FORTRAN_STDOUT", "0")
    monkeypatch.setenv("SFINCS_JAX_SOLVER_ITER_STATS", "0")

    write_sfincs_jax_output_h5(
        input_namelist=input_path,
        output_path=out_path,
        compute_solution=True,
        emit=emit,
        verbose=True,
    )

    joined = "\n".join(logs)
    assert "building RHSMode=1 preconditioner=schur" in joined
