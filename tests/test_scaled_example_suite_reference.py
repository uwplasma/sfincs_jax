from __future__ import annotations

import importlib.util
import json
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
_write_suite_outputs = _MODULE._write_suite_outputs

from run_reduced_upstream_suite import CaseResult


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

    staged, effective_input = _stage_reference_fortran_artifacts(
        case_name=case_name,
        case_input=case_input,
        case_out_dir=out_dir,
        reference_results_root=ref_root,
    )

    assert staged is True
    assert effective_input == case_input
    assert (out_dir / "fortran_run" / "input.namelist").read_text(encoding="utf-8") == case_input.read_text(encoding="utf-8")
    assert (out_dir / "fortran_run" / "sfincsOutput.h5").read_text(encoding="utf-8") == "fortran-h5"
    assert (out_dir / "fortran_run" / "sfincs.log").read_text(encoding="utf-8") == "fortran-log"


def test_stage_reference_fortran_artifacts_reuses_reference_resolution(tmp_path: Path) -> None:
    case_name = "tokamak_case"
    ref_root = tmp_path / "reference"
    case_dir = ref_root / case_name / "fortran_run"
    case_dir.mkdir(parents=True)
    (case_dir / "sfincsOutput.h5").write_text("fortran-h5", encoding="utf-8")
    (case_dir / "input.namelist").write_text(
        "&general\n/\n&resolutionParameters\n  NTHETA = 31\n  NZETA = 9\n  NX = 2\n  NXI = 17\n/\n",
        encoding="utf-8",
    )

    case_input = tmp_path / "input.namelist"
    case_input.write_text(
        "&general\n/\n&geometryParameters\n  equilibriumFile = '/office/path/w7x.nc'\n/\n"
        "&resolutionParameters\n  NTHETA = 21\n  NZETA = 7\n  NX = 3\n  NXI = 18\n/\n",
        encoding="utf-8",
    )

    staged, effective_input = _stage_reference_fortran_artifacts(
        case_name=case_name,
        case_input=case_input,
        case_out_dir=tmp_path / "out",
        reference_results_root=ref_root,
    )

    assert staged is True
    assert effective_input != case_input
    text = effective_input.read_text(encoding="utf-8")
    assert "equilibriumFile = '/office/path/w7x.nc'" in text
    assert "  NTHETA = 31" in text
    assert "  NZETA = 9" in text
    assert "  NX = 2" in text
    assert "  NXI = 17" in text
    assert (tmp_path / "out" / "fortran_run" / "input.namelist").read_text(encoding="utf-8") == text


def _case_result(case: str, *, status: str = "parity_ok", strict_mismatches: int = 0) -> CaseResult:
    return CaseResult(
        case=case,
        status=status,
        blocker_type="",
        message="",
        attempts=1,
        reductions=0,
        fortran_runtime_s=1.0,
        jax_runtime_s=2.0,
        jax_runtime_s_cold=2.1,
        jax_runtime_s_warm=1.9,
        fortran_max_rss_mb=100.0,
        jax_max_rss_mb=200.0,
        jax_solver_iters_mean=5.0,
        jax_solver_iters_min=5,
        jax_solver_iters_max=5,
        jax_solver_iters_n=1,
        jax_solver_iters_detail=[5],
        jax_solver_kinds=["gmres"],
        print_parity_signals=9,
        print_parity_total=9,
        print_missing_signals=[],
        n_common_keys=10,
        n_mismatch_common=0,
        mismatch_keys_sample=[],
        n_mismatch_solver=0,
        n_mismatch_physics=0,
        mismatch_solver_sample=[],
        mismatch_physics_sample=[],
        max_abs_mismatch=0.0,
        strict_n_common_keys=10,
        strict_n_mismatch_common=strict_mismatches,
        strict_mismatch_keys_sample=["pressureAnisotropy"] if strict_mismatches else [],
        strict_n_mismatch_solver=0,
        strict_n_mismatch_physics=strict_mismatches,
        strict_mismatch_solver_sample=[],
        strict_mismatch_physics_sample=["pressureAnisotropy"] if strict_mismatches else [],
        strict_max_abs_mismatch=1.0e-6 if strict_mismatches else 0.0,
        final_resolution={"NTHETA": 21, "NZETA": 1, "NX": 8, "NXI": 31},
        input_path=f"{case}/input.namelist",
        promoted_input_path=None,
        fortran_h5=f"{case}/fortran.h5",
        jax_h5=f"{case}/jax.h5",
    )


def test_write_suite_outputs_writes_incremental_reports(tmp_path: Path) -> None:
    rows = [_case_result("b_case"), _case_result("a_case", strict_mismatches=1)]

    _write_suite_outputs(rows, tmp_path)

    report = tmp_path / "suite_report.json"
    report_strict = tmp_path / "suite_report_strict.json"
    report_rst = tmp_path / "suite_status.rst"
    report_rst_strict = tmp_path / "suite_status_strict.rst"
    summary = tmp_path / "summary.md"

    for path in (report, report_strict, report_rst, report_rst_strict, summary):
        assert path.exists(), f"missing {path.name}"

    report_rows = json.loads(report.read_text(encoding="utf-8"))
    strict_rows = json.loads(report_strict.read_text(encoding="utf-8"))

    assert [row["case"] for row in report_rows] == ["a_case", "b_case"]
    assert strict_rows[0]["status"] == "parity_mismatch"
    assert strict_rows[0]["n_mismatch_common"] == 1
    assert "Scaled Example Suite Summary" in summary.read_text(encoding="utf-8")
