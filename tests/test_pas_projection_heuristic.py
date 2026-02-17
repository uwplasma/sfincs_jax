from __future__ import annotations

from pathlib import Path
import re

import pytest

from sfincs_jax.io import write_sfincs_jax_output_h5


def _patch_group_block(txt: str, group: str, insert_lines: list[str]) -> str:
    start = re.search(rf"(?im)^\s*&{re.escape(group)}\s*$", txt)
    if start is None:
        raise ValueError(f"Missing &{group}")
    end = re.search(r"(?m)^\s*/\s*$", txt[start.end() :])
    if end is None:
        raise ValueError(f"Missing / terminator for &{group}")
    end_pos = start.end() + end.start()
    block = txt[start.end() : end_pos]
    if not block.endswith("\n"):
        block += "\n"
    block2 = block + "\n".join(insert_lines) + "\n"
    return txt[: start.end()] + block2 + txt[end_pos:]


def _patch_solver_tol(txt: str, value: float) -> str:
    start = re.search(r"(?im)^\s*&resolutionParameters\s*$", txt)
    if start is None:
        raise ValueError("Missing &resolutionParameters")
    end = re.search(r"(?m)^\s*/\s*$", txt[start.end() :])
    if end is None:
        raise ValueError("Missing / terminator for &resolutionParameters")
    end_pos = start.end() + end.start()
    block = txt[start.end() : end_pos]
    pat = re.compile(r"(?im)^[ \t]*solverTolerance[ \t]*=[ \t]*([^!\n\r]+)[ \t]*$")
    m = pat.search(block)
    new_line = f"  solverTolerance = {value:.16g}"
    if m is not None:
        block2 = block.replace(m.group(0), new_line)
    else:
        if not block.endswith("\n"):
            block += "\n"
        block2 = block + new_line + "\n"
    return txt[: start.end()] + block2 + txt[end_pos:]


def test_pas_projection_auto_disabled_for_full_precond(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Auto PAS projection must disable when full preconditioner is requested."""
    here = Path(__file__).parent
    input_path = here / "ref" / "pas_1species_PAS_noEr_tiny_scheme1.input.namelist"
    assert input_path.exists()

    txt = input_path.read_text()
    txt = _patch_solver_tol(txt, value=1e-6)
    txt = _patch_group_block(
        txt,
        "preconditionerOptions",
        [
            "  preconditioner_species = 0",
            "  preconditioner_x = 0",
            "  preconditioner_xi = 0",
        ],
    )

    patched = tmp_path / "input.namelist"
    patched.write_text(txt)

    logs: list[str] = []

    def emit(level: int, msg: str) -> None:
        logs.append(msg)

    monkeypatch.setenv("SFINCS_JAX_PAS_PROJECT_CONSTRAINTS", "auto")
    monkeypatch.setenv("SFINCS_JAX_FORTRAN_STDOUT", "0")

    out_path = tmp_path / "sfincsOutput.h5"
    write_sfincs_jax_output_h5(input_namelist=patched, output_path=out_path, compute_solution=True, emit=emit)

    assert not any("PAS constraint projection enabled" in line for line in logs)
