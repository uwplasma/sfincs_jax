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
