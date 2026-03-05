from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_scaled_example_suite.py"
sys.path.insert(0, str(_SCRIPT_PATH.parent))
_SPEC = importlib.util.spec_from_file_location("run_scaled_example_suite", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
_stage_reference_fortran_artifacts = _MODULE._stage_reference_fortran_artifacts


def test_stage_reference_fortran_artifacts_uses_last_success(tmp_path: Path) -> None:
    case_name = "tokamak_case"
    ref_root = tmp_path / "reference"
    case_dir = ref_root / case_name / "last_success"
    case_dir.mkdir(parents=True)
    (case_dir / "sfincsOutput_fortran.h5").write_text("fortran-h5", encoding="utf-8")
    (case_dir / "sfincs_fortran.log").write_text("fortran-log", encoding="utf-8")
    (ref_root / case_name / "input.namelist").write_text("&general\n  NTHETA = 21\n/\n", encoding="utf-8")

    case_input = tmp_path / "input.namelist"
    case_input.write_text("&general\n  NTHETA = 21\n/\n", encoding="utf-8")
    out_dir = tmp_path / "out"

    staged = _stage_reference_fortran_artifacts(
        case_name=case_name,
        case_input=case_input,
        case_out_dir=out_dir,
        reference_results_root=ref_root,
    )

    assert staged is True
    assert (out_dir / "fortran_run" / "input.namelist").read_text(encoding="utf-8") == case_input.read_text(encoding="utf-8")
    assert (out_dir / "fortran_run" / "sfincsOutput.h5").read_text(encoding="utf-8") == "fortran-h5"
    assert (out_dir / "fortran_run" / "sfincs.log").read_text(encoding="utf-8") == "fortran-log"


def test_stage_reference_fortran_artifacts_rejects_mismatched_input(tmp_path: Path) -> None:
    case_name = "tokamak_case"
    ref_root = tmp_path / "reference"
    case_dir = ref_root / case_name / "fortran_run"
    case_dir.mkdir(parents=True)
    (case_dir / "sfincsOutput.h5").write_text("fortran-h5", encoding="utf-8")
    (case_dir / "input.namelist").write_text("&general\n  NTHETA = 31\n/\n", encoding="utf-8")

    case_input = tmp_path / "input.namelist"
    case_input.write_text("&general\n  NTHETA = 21\n/\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Reference input mismatch"):
        _stage_reference_fortran_artifacts(
            case_name=case_name,
            case_input=case_input,
            case_out_dir=tmp_path / "out",
            reference_results_root=ref_root,
        )
