from __future__ import annotations

from pathlib import Path
import re

from sfincs_jax.io import write_sfincs_jax_output_h5


def _patch_block_value(block: str, key: str, value: str) -> str:
    pat = re.compile(rf"(?im)^[ \t]*{re.escape(key)}[ \t]*=[ \t]*([^!\n\r]+)[ \t]*$")
    new_line = f"  {key} = {value}"
    if pat.search(block):
        return pat.sub(new_line, block)
    if not block.endswith("\n"):
        block += "\n"
    return block + new_line + "\n"


def _patch_resolution_block(txt: str, *, ntheta: int, nzeta: int, nxi: int, nx: int, solver_tol: float) -> str:
    start = re.search(r"(?im)^\s*&resolutionParameters\s*$", txt)
    if start is None:
        raise ValueError("Missing &resolutionParameters")
    end = re.search(r"(?m)^\s*/\s*$", txt[start.end() :])
    if end is None:
        raise ValueError("Missing / terminator for &resolutionParameters")
    end_pos = start.end() + end.start()
    block = txt[start.end() : end_pos]
    block = _patch_block_value(block, "Ntheta", str(int(ntheta)))
    block = _patch_block_value(block, "Nzeta", str(int(nzeta)))
    block = _patch_block_value(block, "Nxi", str(int(nxi)))
    block = _patch_block_value(block, "Nx", str(int(nx)))
    block = _patch_block_value(block, "solverTolerance", f"{solver_tol:.16g}")
    return txt[: start.end()] + block + txt[end_pos:]


def _patch_export_block(txt: str) -> str:
    start = re.search(r"(?im)^\s*&export_f\s*$", txt)
    if start is None:
        return txt
    end = re.search(r"(?m)^\s*/\s*$", txt[start.end() :])
    if end is None:
        raise ValueError("Missing / terminator for &export_f")
    end_pos = start.end() + end.start()
    block = txt[start.end() : end_pos]
    block = _patch_block_value(block, "export_full_f", ".false.")
    block = _patch_block_value(block, "export_delta_f", ".false.")
    block = _patch_block_value(block, "export_f_theta_option", "0")
    block = _patch_block_value(block, "export_f_zeta_option", "0")
    block = _patch_block_value(block, "export_f_xi_option", "0")
    block = _patch_block_value(block, "export_f_x_option", "0")
    return txt[: start.end()] + block + txt[end_pos:]


def test_full_precond_uses_schur_for_constraint_scheme2(tmp_path: Path, monkeypatch) -> None:
    input_path = Path(__file__).parent / "reduced_inputs" / "tokamak_1species_PASCollisions_noEr_Nx1.input.namelist"
    out_path = tmp_path / "sfincsOutput_jax.h5"

    txt = input_path.read_text()
    txt = _patch_resolution_block(txt, ntheta=7, nzeta=1, nxi=4, nx=1, solver_tol=1e-6)
    txt = _patch_export_block(txt)
    patched = tmp_path / "input_schur_tiny.namelist"
    patched.write_text(txt)

    logs: list[str] = []

    def emit(_level: int, msg: str) -> None:
        logs.append(msg)

    monkeypatch.setenv("SFINCS_JAX_FORTRAN_STDOUT", "0")
    monkeypatch.setenv("SFINCS_JAX_SOLVER_ITER_STATS", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SOLVE_METHOD", "incremental")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_DENSE_ACTIVE_CUTOFF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SCHUR_TOKAMAK", "1")

    write_sfincs_jax_output_h5(
        input_namelist=patched,
        output_path=out_path,
        compute_solution=True,
        emit=emit,
        verbose=True,
    )

    joined = "\n".join(logs)
    assert "building RHSMode=1 preconditioner=schur" in joined


def test_full_precond_tokamak_defaults_to_theta_line(tmp_path: Path, monkeypatch) -> None:
    input_path = Path(__file__).parent / "reduced_inputs" / "tokamak_1species_PASCollisions_noEr_Nx1.input.namelist"
    out_path = tmp_path / "sfincsOutput_jax.h5"

    txt = input_path.read_text()
    txt = _patch_resolution_block(txt, ntheta=7, nzeta=1, nxi=4, nx=1, solver_tol=1e-6)
    txt = _patch_export_block(txt)
    patched = tmp_path / "input_theta_line_tiny.namelist"
    patched.write_text(txt)

    logs: list[str] = []

    def emit(_level: int, msg: str) -> None:
        logs.append(msg)

    monkeypatch.setenv("SFINCS_JAX_FORTRAN_STDOUT", "0")
    monkeypatch.setenv("SFINCS_JAX_SOLVER_ITER_STATS", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SOLVE_METHOD", "incremental")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_DENSE_ACTIVE_CUTOFF", "0")

    write_sfincs_jax_output_h5(
        input_namelist=patched,
        output_path=out_path,
        compute_solution=True,
        emit=emit,
        verbose=True,
    )

    joined = "\n".join(logs)
    assert "building RHSMode=1 preconditioner=theta_line" in joined


def test_schur_auto_min_for_pas(tmp_path: Path, monkeypatch) -> None:
    """Auto Schur selection should respect SFINCS_JAX_RHSMODE1_SCHUR_AUTO_MIN."""
    input_path = Path(__file__).parent / "reduced_inputs" / "geometryScheme4_2species_PAS_noEr.input.namelist"
    out_path = tmp_path / "sfincsOutput_jax.h5"

    txt = input_path.read_text()
    txt = _patch_resolution_block(txt, ntheta=7, nzeta=7, nxi=4, nx=2, solver_tol=1e-6)
    txt = _patch_export_block(txt)
    patched = tmp_path / "input_schur_auto_tiny.namelist"
    patched.write_text(txt)

    logs: list[str] = []

    def emit(_level: int, msg: str) -> None:
        logs.append(msg)

    monkeypatch.setenv("SFINCS_JAX_FORTRAN_STDOUT", "0")
    monkeypatch.setenv("SFINCS_JAX_SOLVER_ITER_STATS", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SOLVE_METHOD", "incremental")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_DENSE_ACTIVE_CUTOFF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SCHUR_AUTO_MIN", "0")

    write_sfincs_jax_output_h5(
        input_namelist=patched,
        output_path=out_path,
        compute_solution=True,
        emit=emit,
        verbose=True,
    )

    joined = "\n".join(logs)
    assert "building RHSMode=1 preconditioner=schur" in joined
