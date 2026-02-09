#!/usr/bin/env python
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
from sfincs_jax.compare import compare_sfincs_outputs
from sfincs_jax.io import localize_equilibrium_file_in_place
from sfincs_jax.namelist import read_sfincs_input

RES_KEYS: tuple[str, ...] = ("NTHETA", "NZETA", "NX", "NXI")
MIN_RES: dict[str, int] = {"NTHETA": 5, "NZETA": 1, "NX": 1, "NXI": 2}


@dataclass
class CaseResult:
    case: str
    status: str
    message: str
    attempts: int
    reductions: int
    fortran_runtime_s: float | None
    jax_runtime_s: float | None
    n_common_keys: int
    n_mismatch_common: int
    max_abs_mismatch: float | None
    final_resolution: dict[str, int]
    input_path: str
    fortran_h5: str | None
    jax_h5: str | None


def _iter_inputs(examples_root: Path) -> list[Path]:
    return sorted(examples_root.rglob("input.namelist"))


def _half_round_int(v: int, *, minimum: int = 1) -> int:
    return max(int(minimum), (int(v) + 1) // 2)


def _resolution_from_namelist(input_path: Path, keys: Sequence[str] = RES_KEYS) -> dict[str, int]:
    nml = read_sfincs_input(input_path)
    res = nml.group("resolutionParameters")
    out: dict[str, int] = {}
    for key in keys:
        if key in res:
            out[key] = int(res[key])
    return out


def _replace_resolution_values_in_text(text: str, *, updates: dict[str, int]) -> str:
    group_start = re.compile(r"^\s*&\s*resolutionParameters\s*$", flags=re.IGNORECASE)
    group_end = re.compile(r"^\s*/\s*$")
    key_patterns = {
        key: re.compile(rf"^(\s*{key}\s*=\s*)([^!\n\r]*)(.*)$", flags=re.IGNORECASE)
        for key in updates
    }

    lines = text.splitlines(keepends=True)
    out_lines: list[str] = []
    in_group = False
    for line in lines:
        if group_start.match(line):
            in_group = True
            out_lines.append(line)
            continue
        if in_group and group_end.match(line):
            in_group = False
            out_lines.append(line)
            continue
        if in_group:
            replaced = False
            for key, pat in key_patterns.items():
                m = pat.match(line)
                if m is not None:
                    prefix, _old, suffix = m.groups()
                    out_lines.append(f"{prefix}{int(updates[key])}{suffix}\n" if not suffix.endswith("\n") else f"{prefix}{int(updates[key])}{suffix}")
                    replaced = True
                    break
            if replaced:
                continue
        out_lines.append(line)
    return "".join(out_lines)


def _write_initial_reduced_input(*, source_input: Path, dst_input: Path) -> dict[str, int]:
    text = source_input.read_text()
    current = _resolution_from_namelist(source_input)
    updates = {k: _half_round_int(v, minimum=MIN_RES.get(k, 1)) for k, v in current.items() if v >= 1}
    dst_input.parent.mkdir(parents=True, exist_ok=True)
    dst_text = _replace_resolution_values_in_text(text, updates=updates)
    dst_input.write_text(dst_text)
    return _resolution_from_namelist(dst_input)


def _reduce_max_axis_in_place(input_path: Path) -> dict[str, int]:
    current = _resolution_from_namelist(input_path)
    if not current:
        return {}
    candidates = {k: v for k, v in current.items() if int(v) > int(MIN_RES.get(k, 1))}
    if not candidates:
        return current
    axis = max(candidates, key=lambda k: candidates[k])
    updates = {axis: _half_round_int(candidates[axis], minimum=MIN_RES.get(axis, 1))}
    text = input_path.read_text()
    input_path.write_text(_replace_resolution_values_in_text(text, updates=updates))
    return _resolution_from_namelist(input_path)


def _tail(path: Path, n: int = 40) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-n:])


def _run_jax_cli(*, input_path: Path, output_path: Path, timeout_s: float, log_path: Path) -> float:
    cmd = [
        "sfincs_jax",
        "-v",
        "write-output",
        "--input",
        str(input_path),
        "--out",
        str(output_path),
    ]
    t0 = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(cmd, cwd=str(REPO_ROOT), check=True, timeout=timeout_s, stdout=log, stderr=subprocess.STDOUT)
    if not output_path.exists():
        tail = _tail(log_path, n=40)
        raise RuntimeError(f"JAX run returned success but did not create output: {output_path}\n{tail}")
    return time.perf_counter() - t0


def _run_fortran_direct(*, input_path: Path, exe: Path, timeout_s: float, log_path: Path) -> tuple[float, Path, int]:
    cmd = [str(exe.resolve())]
    t0 = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=str(input_path.parent), check=False, timeout=timeout_s, stdout=log, stderr=subprocess.STDOUT)
    dt = time.perf_counter() - t0
    out = input_path.parent / "sfincsOutput.h5"
    if proc.returncode != 0:
        tail = _tail(log_path, n=40)
        raise RuntimeError(f"Fortran failed rc={proc.returncode}.\n{tail}")
    if not out.exists():
        tail = _tail(log_path, n=40)
        raise RuntimeError(f"Fortran did not produce output.\n{tail}")
    return dt, out, int(proc.returncode)


def _compare_outputs(fortran_h5: Path, jax_h5: Path, *, rtol: float, atol: float) -> tuple[int, int, float | None]:
    results = compare_sfincs_outputs(a_path=jax_h5, b_path=fortran_h5, rtol=rtol, atol=atol)
    bad = [r for r in results if not r.ok]
    max_abs = max((r.max_abs for r in bad), default=None)
    return len(results), len(bad), max_abs


def _write_rst(rows: list[CaseResult], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = len(rows)
    by_status: dict[str, int] = {}
    for row in rows:
        by_status[row.status] = by_status.get(row.status, 0) + 1
    summary = ", ".join(f"{k}={v}" for k, v in sorted(by_status.items()))

    lines: list[str] = []
    lines.append(":orphan:\n\n")
    lines.append(".. NOTE: Auto-generated by `scripts/run_reduced_upstream_suite.py`.\n\n")
    lines.append(f"- Cases: **{total}**\n")
    lines.append(f"- Status counts: {summary}\n")
    lines.append("- Timeout policy: 30s per Fortran/JAX run attempt, then halve largest axis and retry.\n\n")
    lines.append(".. list-table:: Reduced-resolution upstream suite parity status\n")
    lines.append("   :header-rows: 1\n")
    lines.append("   :widths: 25 9 10 8 8 11 11 12 16\n\n")
    lines.append("   * - Case\n")
    lines.append("     - Status\n")
    lines.append("     - Resolution\n")
    lines.append("     - Tries\n")
    lines.append("     - Reductions\n")
    lines.append("     - Fortran(s)\n")
    lines.append("     - JAX(s)\n")
    lines.append("     - Mismatches\n")
    lines.append("     - Note\n")
    for row in rows:
        res = ",".join(f"{k}={v}" for k, v in sorted(row.final_resolution.items()))
        ft = "" if row.fortran_runtime_s is None else f"{row.fortran_runtime_s:.3f}"
        jt = "" if row.jax_runtime_s is None else f"{row.jax_runtime_s:.3f}"
        mm = f"{row.n_mismatch_common}/{row.n_common_keys}" if row.n_common_keys > 0 else "-"
        lines.append(f"   * - {row.case}\n")
        lines.append(f"     - {row.status}\n")
        lines.append(f"     - {res}\n")
        lines.append(f"     - {row.attempts}\n")
        lines.append(f"     - {row.reductions}\n")
        lines.append(f"     - {ft}\n")
        lines.append(f"     - {jt}\n")
        lines.append(f"     - {mm}\n")
        lines.append(f"     - {row.message}\n")
    out_path.write_text("".join(lines), encoding="utf-8")


def _run_case(
    *,
    case_input: Path,
    case_out_dir: Path,
    fortran_exe: Path,
    timeout_s: float,
    rtol: float,
    atol: float,
    max_attempts: int,
) -> CaseResult:
    case = case_input.parent.name
    case_out_dir.mkdir(parents=True, exist_ok=True)
    dst_input = case_out_dir / "input.namelist"
    (case_out_dir / "input.original.namelist").write_text(case_input.read_text())
    _write_initial_reduced_input(source_input=case_input, dst_input=dst_input)
    localize_equilibrium_file_in_place(input_namelist=dst_input, overwrite=False)

    attempts = 0
    reductions = 0
    final_res = _resolution_from_namelist(dst_input)
    fortran_runtime = None
    jax_runtime = None
    note = ""
    status = "error"
    fortran_h5_path: Path | None = None
    jax_h5_path: Path | None = None
    n_common = 0
    n_bad = 0
    max_abs = None

    while attempts < max_attempts:
        attempts += 1
        final_res = _resolution_from_namelist(dst_input)
        fortran_dir = case_out_dir / "fortran_run"
        if fortran_dir.exists():
            shutil.rmtree(fortran_dir)
        fortran_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(dst_input, fortran_dir / "input.namelist")
        localize_equilibrium_file_in_place(input_namelist=fortran_dir / "input.namelist", overwrite=False)
        fortran_log = fortran_dir / "sfincs.log"

        try:
            fortran_runtime, out_fortran, fortran_rc = _run_fortran_direct(
                input_path=fortran_dir / "input.namelist",
                exe=fortran_exe,
                timeout_s=timeout_s,
                log_path=fortran_log,
            )
            fortran_h5_path = out_fortran
        except subprocess.TimeoutExpired:
            note = "Fortran timeout; reduced largest axis."
            new_res = _reduce_max_axis_in_place(dst_input)
            if new_res == final_res:
                status = "fortran_timeout"
                break
            reductions += 1
            continue
        except Exception as exc:  # noqa: BLE001
            note = f"Fortran error: {type(exc).__name__}: {exc}"
            new_res = _reduce_max_axis_in_place(dst_input)
            if new_res == final_res:
                status = "fortran_error"
                break
            reductions += 1
            continue

        jax_h5 = case_out_dir / "sfincsOutput_jax.h5"
        jax_log = case_out_dir / "sfincs_jax.log"
        try:
            jax_runtime = _run_jax_cli(input_path=dst_input, output_path=jax_h5, timeout_s=timeout_s, log_path=jax_log)
            jax_h5_path = jax_h5
        except subprocess.TimeoutExpired:
            note = "JAX timeout; reduced largest axis."
            new_res = _reduce_max_axis_in_place(dst_input)
            if new_res == final_res:
                status = "jax_timeout"
                break
            reductions += 1
            continue
        except Exception as exc:  # noqa: BLE001
            note = f"JAX error: {type(exc).__name__}: {exc}"
            status = "jax_error"
            break

        if fortran_h5_path is None or jax_h5_path is None:
            note = "Missing output file after successful run."
            status = "missing_output"
            break

        try:
            n_common, n_bad, max_abs = _compare_outputs(fortran_h5_path, jax_h5_path, rtol=rtol, atol=atol)
            if n_bad == 0:
                status = "parity_ok"
                note = "All common numeric datasets matched tolerance."
            else:
                status = "parity_mismatch"
                note = "Common numeric dataset mismatches present."
        except Exception as exc:  # noqa: BLE001
            status = "compare_error"
            note = f"Compare error: {type(exc).__name__}: {exc}"
        break

    else:
        status = "max_attempts"
        note = "Reached max attempts while reducing resolution."

    return CaseResult(
        case=case,
        status=status,
        message=note,
        attempts=attempts,
        reductions=reductions,
        fortran_runtime_s=fortran_runtime,
        jax_runtime_s=jax_runtime,
        n_common_keys=n_common,
        n_mismatch_common=n_bad,
        max_abs_mismatch=max_abs,
        final_resolution=final_res,
        input_path=str(dst_input),
        fortran_h5=str(fortran_h5_path) if fortran_h5_path is not None else None,
        jax_h5=str(jax_h5_path) if jax_h5_path is not None else None,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run reduced-resolution upstream suite one case at a time with 30s timeout/reduction policy.")
    parser.add_argument(
        "--examples-root",
        type=Path,
        default=REPO_ROOT / "examples" / "sfincs_examples",
        help="Path to upstream vendored examples.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=REPO_ROOT / "tests" / "reduced_upstream_examples",
        help="Output directory where reduced copied cases are written.",
    )
    parser.add_argument(
        "--fortran-exe",
        type=Path,
        default=Path("/Users/rogeriojorge/local/tests/sfincs/fortran/version3/sfincs"),
        help="Path to Fortran v3 executable.",
    )
    parser.add_argument("--pattern", type=str, default=None, help="Regex filter on case directory path.")
    parser.add_argument("--timeout-s", type=float, default=30.0, help="Per-run timeout in seconds.")
    parser.add_argument("--max-attempts", type=int, default=6, help="Maximum adaptive retries per case.")
    parser.add_argument("--rtol", type=float, default=1e-8)
    parser.add_argument("--atol", type=float, default=1e-8)
    args = parser.parse_args()

    examples_root = Path(args.examples_root)
    out_root = Path(args.out_root)
    fortran_exe = Path(args.fortran_exe)
    if not examples_root.exists():
        raise SystemExit(f"examples root does not exist: {examples_root}")
    if not fortran_exe.exists():
        raise SystemExit(f"Fortran executable does not exist: {fortran_exe}")

    out_root.mkdir(parents=True, exist_ok=True)
    inputs = _iter_inputs(examples_root)
    if args.pattern:
        rx = re.compile(str(args.pattern), flags=re.IGNORECASE)
        inputs = [p for p in inputs if rx.search(str(p.parent))]
    if not inputs:
        raise SystemExit("No input.namelist files matched.")
    results: list[CaseResult] = []

    for index, input_path in enumerate(inputs, start=1):
        case = input_path.parent.name
        print(f"[{index}/{len(inputs)}] {case}")
        case_out = out_root / case
        result = _run_case(
            case_input=input_path,
            case_out_dir=case_out,
            fortran_exe=fortran_exe,
            timeout_s=float(args.timeout_s),
            rtol=float(args.rtol),
            atol=float(args.atol),
            max_attempts=int(args.max_attempts),
        )
        results.append(result)
        print(
            f"  status={result.status} attempts={result.attempts} reductions={result.reductions} "
            f"res={result.final_resolution} mismatch={result.n_mismatch_common}/{result.n_common_keys}"
        )

    report_json = out_root / "suite_report.json"
    report_json.write_text(json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8")
    report_rst = REPO_ROOT / "docs" / "_generated" / "reduced_upstream_suite_status.rst"
    _write_rst(results, report_rst)
    print(f"Wrote {report_json}")
    print(f"Wrote {report_rst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
