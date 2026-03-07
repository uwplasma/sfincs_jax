from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_reduced_upstream_suite.py"
sys.path.insert(0, str(_SCRIPT_PATH.parent))
_SPEC = importlib.util.spec_from_file_location("run_reduced_upstream_suite", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
_canonicalize_fortran_v3_input_text = _MODULE._canonicalize_fortran_v3_input_text
_run_fortran_direct = _MODULE._run_fortran_direct


def test_run_fortran_direct_uses_stable_cs0_petsc_options_by_default(
    tmp_path: Path, monkeypatch
) -> None:
    input_path = tmp_path / "input.namelist"
    input_path.write_text(
        "&physicsParameters\n  constraintScheme = 0\n/\n",
        encoding="utf-8",
    )
    log_path = tmp_path / "sfincs.log"
    seen: dict[str, str] = {}

    def _fake_run_logged_subprocess(*, cmd, cwd, env, log_path, timeout_s):
        del cmd, cwd, timeout_s
        seen["petsc_options"] = env.get("PETSC_OPTIONS", "")
        log_path.write_text("", encoding="utf-8")
        (input_path.parent / "sfincsOutput.h5").write_text("dummy", encoding="utf-8")
        return 0

    monkeypatch.setattr(_MODULE, "_run_logged_subprocess", _fake_run_logged_subprocess)

    dt, out, rc, rss = _run_fortran_direct(
        input_path=input_path,
        exe=tmp_path / "sfincs",
        timeout_s=10.0,
        log_path=log_path,
    )

    assert rc == 0
    assert out == input_path.parent / "sfincsOutput.h5"
    assert rss is None
    assert dt >= 0.0
    assert seen["petsc_options"] == "-ksp_type gmres -pc_type none"


def test_run_fortran_direct_preserves_explicit_petsc_options(
    tmp_path: Path, monkeypatch
) -> None:
    input_path = tmp_path / "input.namelist"
    input_path.write_text(
        "&physicsParameters\n  constraintScheme = 0\n/\n",
        encoding="utf-8",
    )
    log_path = tmp_path / "sfincs.log"
    seen: dict[str, str] = {}

    def _fake_run_logged_subprocess(*, cmd, cwd, env, log_path, timeout_s):
        del cmd, cwd, timeout_s
        seen["petsc_options"] = env.get("PETSC_OPTIONS", "")
        log_path.write_text("", encoding="utf-8")
        (input_path.parent / "sfincsOutput.h5").write_text("dummy", encoding="utf-8")
        return 0

    monkeypatch.setattr(_MODULE, "_run_logged_subprocess", _fake_run_logged_subprocess)
    monkeypatch.setenv("PETSC_OPTIONS", "-ksp_type cg -pc_type jacobi")

    _run_fortran_direct(
        input_path=input_path,
        exe=tmp_path / "sfincs",
        timeout_s=10.0,
        log_path=log_path,
    )

    assert seen["petsc_options"] == "-ksp_type cg -pc_type jacobi"


def test_canonicalize_fortran_v3_input_text_rewrites_legacy_flowcontrol() -> None:
    text = (
        "&flowControl\n"
        "  programMode = 1\n"
        "  RHSMode = 1\n"
        "  outputScheme = 1\n"
        "  solveSystem = .true.\n"
        "/\n"
        "&geometryParameters\n"
        "  geometryScheme = 11\n"
        "  JGboozer_file = 'hsx.bc'\n"
        "  normradius_wish = 0.22\n"
        "/\n"
        "&physicsParameters\n"
        "  psiAHat = 0.03d0\n"
        "  Delta = 1d-3\n"
        "  dPhiHatdpsiN = 0.0d0\n"
        "  speciesMode = 0\n"
        "/\n"
        "&otherNumericalParameters\n"
        "  useIterativeSolver = .true.\n"
        "/\n"
    )
    rewritten = _canonicalize_fortran_v3_input_text(text)
    assert "&general" in rewritten
    assert "&flowControl" not in rewritten
    assert "RHSMode = 1" in rewritten
    assert "solveSystem = .true." in rewritten
    assert "programMode" not in rewritten
    assert "outputScheme" not in rewritten
    assert "equilibriumFile = 'hsx.bc'" in rewritten
    assert "inputRadialCoordinate = 3" in rewritten
    assert "inputRadialCoordinateForGradients = 1" in rewritten
    assert "rN_wish = 0.22" in rewritten
    assert "psiAHat = 0.03d0" in rewritten
    assert "speciesMode" not in rewritten
    assert "useIterativeLinearSolver = .true." in rewritten
    assert "useIterativeSolver = .true." not in rewritten
    assert "&export_f" in rewritten


def test_canonicalize_fortran_v3_input_text_preserves_trailing_newline_for_v3_input() -> None:
    text = "&general\n  RHSMode = 1\n/\n&export_f\n/\n"
    rewritten = _canonicalize_fortran_v3_input_text(text)
    assert rewritten == text
    assert rewritten.endswith("\n")


def test_canonicalize_fortran_v3_input_text_does_not_inject_mixed_gradient_coordinate() -> None:
    text = (
        "&geometryParameters\n"
        "  geometryScheme = 5\n"
        "/\n"
        "&speciesParameters\n"
        "  dNHatdrHats = -1.0d0\n"
        "  dTHatdrHats = -2.0d0\n"
        "/\n"
        "&physicsParameters\n"
        "  Er = -3.0d0\n"
        "/\n"
        "&export_f\n"
        "/\n"
    )
    rewritten = _canonicalize_fortran_v3_input_text(text)
    assert "inputRadialCoordinateForGradients" not in rewritten


def test_run_fortran_direct_canonicalizes_legacy_flowcontrol_for_fortran_v3(
    tmp_path: Path, monkeypatch
) -> None:
    input_path = tmp_path / "input.namelist"
    input_path.write_text(
        "&flowControl\n  RHSMode = 1\n/\n&physicsParameters\n  constraintScheme = -1\n/\n",
        encoding="utf-8",
    )
    log_path = tmp_path / "sfincs.log"
    seen: dict[str, str] = {}

    def _fake_run_logged_subprocess(*, cmd, cwd, env, log_path, timeout_s):
        del cmd, cwd, env, timeout_s
        seen["input_text"] = input_path.read_text(encoding="utf-8")
        log_path.write_text("", encoding="utf-8")
        (input_path.parent / "sfincsOutput.h5").write_text("dummy", encoding="utf-8")
        return 0

    monkeypatch.setattr(_MODULE, "_run_logged_subprocess", _fake_run_logged_subprocess)

    _run_fortran_direct(
        input_path=input_path,
        exe=tmp_path / "sfincs",
        timeout_s=10.0,
        log_path=log_path,
    )

    assert "&general" in seen["input_text"]
    assert "&flowControl" not in seen["input_text"]
