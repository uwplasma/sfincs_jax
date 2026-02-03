from __future__ import annotations

from pathlib import Path
import re

import pytest

from sfincs_jax.postprocess_upstream import run_upstream_util
from sfincs_jax.scans import run_er_scan


def _patch_solver_tolerance(txt: str, value: float) -> str:
    # Replace (or insert) solverTolerance inside &resolutionParameters.
    start = re.search(r"(?im)^\s*&resolutionParameters\s*$", txt)
    if start is None:
        raise ValueError("Missing &resolutionParameters")
    end = re.search(r"(?m)^\s*/\s*$", txt[start.end() :])
    if end is None:
        raise ValueError("Missing '/' terminator for &resolutionParameters")
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


def test_upstream_sfincsScanPlot_2_runs_on_jax_scan(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")

    repo_root = Path(__file__).resolve().parents[1]
    template_src = repo_root / "tests" / "ref" / "monoenergetic_PAS_tiny_scheme11.input.namelist"
    assert template_src.exists()

    # Write a modified template with a looser solverTolerance for speed.
    template = tmp_path / "template.input.namelist"
    txt = _patch_solver_tolerance(template_src.read_text(), value=1e-6)
    template.write_text(txt)

    scan_dir = tmp_path / "scan"
    run_er_scan(
        input_namelist=template,
        out_dir=scan_dir,
        values=(0.1, 0.0),
        compute_transport_matrix=True,
    )

    run_upstream_util(util="sfincsScanPlot_2", case_dir=scan_dir, args=("pdf",))
    pdf = scan_dir / "sfincsScanPlot_2.pdf"
    assert pdf.exists()
    assert pdf.stat().st_size > 0
