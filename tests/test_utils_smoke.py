from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _inject_group(text: str, group: str, lines: list[str]) -> str:
    out: list[str] = []
    inserted = False
    for line in text.splitlines():
        out.append(line)
        if line.strip().lower().startswith(f"&{group.lower()}"):
            out.extend(lines)
            inserted = True
    if not inserted:
        out.append(f"&{group}")
        out.extend(lines)
        out.append("/")
    return "\n".join(out) + "\n"


def _run_script(script: Path, cwd: Path, args: list[str]) -> None:
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    subprocess.run([sys.executable, str(script), *args], cwd=str(cwd), env=env, check=True)


def test_utils_sfincs_plot_smoke(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    script = repo / "utils" / "sfincsPlot"
    base = repo / "tests" / "ref" / "pas_1species_PAS_noEr_tiny.input.namelist"
    input_path = tmp_path / "input.namelist"
    input_path.write_text(base.read_text())

    out_prefix = tmp_path / "sfincsPlot"
    _run_script(script, tmp_path, ["--save-prefix", str(out_prefix)])

    assert (tmp_path / "sfincsOutput.h5").exists()
    assert (tmp_path / "sfincsPlot_fig1.png").exists()


def test_utils_sfincs_plotf_smoke(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    script = repo / "utils" / "sfincsPlotF"
    base = repo / "tests" / "ref" / "pas_1species_PAS_noEr_tiny.input.namelist"
    input_path = tmp_path / "input.namelist"

    text = _inject_group(
        base.read_text(),
        "export_f",
        [
            "  export_delta_f = .true.",
            "  export_full_f = .false.",
        ],
    )
    input_path.write_text(text)

    out_path = tmp_path / "sfincsPlotF.png"
    _run_script(script, tmp_path, ["--save", str(out_path)])

    assert (tmp_path / "sfincsOutput.h5").exists()
    assert out_path.exists()
