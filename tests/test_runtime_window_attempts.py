from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_reduced_upstream_suite.py"
sys.path.insert(0, str(_SCRIPT_PATH.parent))
_SPEC = importlib.util.spec_from_file_location("run_reduced_upstream_suite", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
suite = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = suite
_SPEC.loader.exec_module(suite)


def test_runtime_window_max_attempts_returns_last_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    case_input = tmp_path / "seed.namelist"
    reference_input = tmp_path / "reference.namelist"
    case_input.write_text(
        "&general\n/\n&resolutionParameters\n  NTHETA = 5\n  NZETA = 1\n  NX = 2\n  NXI = 4\n/\n",
        encoding="utf-8",
    )
    reference_input.write_text(
        "&general\n/\n&resolutionParameters\n  NTHETA = 11\n  NZETA = 7\n  NX = 5\n  NXI = 8\n/\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(suite, "localize_equilibrium_file_in_place", lambda *args, **kwargs: None)

    def fake_fortran(*, input_path: Path, exe: Path, timeout_s: float, log_path: Path):
        out = input_path.parent / "sfincsOutput.h5"
        out.write_text("fortran-h5", encoding="utf-8")
        log_path.write_text("Time to solve: 0.5 seconds\n", encoding="utf-8")
        return 0.5, out, 0, 100.0

    def fake_jax(
        *,
        input_path: Path,
        output_path: Path,
        timeout_s: float,
        log_path: Path,
        compute_solution: bool,
        compute_transport_matrix: bool,
        collect_iterations: bool = True,
        repeats: int = 1,
        cache_dir: Path | None = None,
    ):
        output_path.write_text("jax-h5", encoding="utf-8")
        log_path.write_text("elapsed_s=0.5\n", encoding="utf-8")
        return 0.5, None, 200.0

    monkeypatch.setattr(suite, "_run_fortran_direct", fake_fortran)
    monkeypatch.setattr(suite, "_run_jax_cli", fake_jax)
    monkeypatch.setattr(suite, "_compare_outputs", lambda *args, **kwargs: (10, 0, 0.0, []))
    monkeypatch.setattr(suite, "_compute_print_parity", lambda *args, **kwargs: (0, 0, []))
    monkeypatch.setattr(suite, "_parse_ksp_iterations", lambda *args, **kwargs: ([], []))

    result = suite._run_case(
        case_name="case",
        case_input=case_input,
        reference_input=reference_input,
        case_out_dir=tmp_path / "out",
        fortran_exe=tmp_path / "sfincs",
        timeout_s=1.0,
        rtol=5e-4,
        atol=1e-9,
        max_attempts=1,
        target_runtime_s=1.0,
        target_runtime_max_s=20.0,
        target_runtime_max_iters=1,
        target_runtime_basis="fortran",
        use_seed_resolution=True,
        reuse_fortran=False,
        collect_iterations=False,
        jax_repeats=1,
        jax_cache_dir=None,
    )

    assert result.status == "parity_ok"
    assert result.n_mismatch_common == 0
    assert result.fortran_runtime_s == pytest.approx(0.5)
