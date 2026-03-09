#!/usr/bin/env python
from __future__ import annotations

import argparse
import concurrent.futures
from dataclasses import asdict
import json
import math
import os
import platform
import re
import shutil
import socket
import sys
from pathlib import Path

from sfincs_jax.io import localize_equilibrium_file_in_place

from run_reduced_upstream_suite import (
    CaseResult,
    REPO_ROOT,
    _estimate_active_size_from_namelist,
    _iter_inputs,
    _load_existing_results,
    _replace_resolution_values_in_text,
    _repo_rel,
    _resolution_from_namelist,
    _rhs_mode_from_namelist,
    _run_case,
    _sanitize_resolution,
    _status_for_mode,
    _write_rst,
)


def _gather_jax_env() -> dict[str, object]:
    info: dict[str, object] = {
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
    }
    try:
        import jax

        info["jax_version"] = getattr(jax, "__version__", "unknown")
        info["jax_backend"] = jax.default_backend()
        info["jax_devices"] = [f"{d.platform}:{getattr(d, 'device_kind', d)}" for d in jax.devices()]
    except Exception as exc:  # noqa: BLE001
        info["jax_error"] = f"{type(exc).__name__}: {exc}"
    return info


def _case_names_for_inputs(inputs: list[Path], *, base_root: Path | None = None) -> dict[Path, str]:
    parent_counts: dict[str, int] = {}
    for input_path in inputs:
        parent_counts[input_path.parent.name] = parent_counts.get(input_path.parent.name, 0) + 1

    names: dict[Path, str] = {}
    for input_path in inputs:
        parent_name = input_path.parent.name
        if parent_counts[parent_name] == 1:
            names[input_path] = parent_name
            continue
        if base_root is not None:
            try:
                rel = input_path.parent.resolve().relative_to(base_root.resolve())
                names[input_path] = "__".join(rel.parts)
                continue
            except Exception:  # noqa: BLE001
                pass
        names[input_path] = "__".join(input_path.parent.parts[-2:])
    return names


def _write_lane_summary(rows: list[CaseResult], out_path: Path) -> None:
    def _fmt(v: float | None, *, places: int = 3) -> str:
        if v is None:
            return "-"
        return f"{float(v):.{places}f}"

    practical_counts: dict[str, int] = {}
    strict_counts: dict[str, int] = {}
    for row in rows:
        practical_counts[row.status] = practical_counts.get(row.status, 0) + 1
        strict_status = _status_for_mode(row, strict=True)
        strict_counts[strict_status] = strict_counts.get(strict_status, 0) + 1

    offenders_runtime = sorted(
        (row for row in rows if row.jax_runtime_s is not None),
        key=lambda row: float(row.jax_runtime_s),
        reverse=True,
    )[:10]
    offenders_runtime_ratio = sorted(
        (
            (row, float(row.jax_runtime_s) / float(row.fortran_runtime_s))
            for row in rows
            if row.jax_runtime_s is not None and row.fortran_runtime_s not in (None, 0.0)
        ),
        key=lambda item: item[1],
        reverse=True,
    )[:10]
    offenders_memory = sorted(
        (row for row in rows if row.jax_max_rss_mb is not None),
        key=lambda row: float(row.jax_max_rss_mb),
        reverse=True,
    )[:10]
    offenders_memory_ratio = sorted(
        (
            (row, float(row.jax_max_rss_mb) / float(row.fortran_max_rss_mb))
            for row in rows
            if row.jax_max_rss_mb is not None and row.fortran_max_rss_mb not in (None, 0.0)
        ),
        key=lambda item: item[1],
        reverse=True,
    )[:10]
    mismatch_rows = [row for row in rows if row.n_mismatch_common > 0 or row.strict_n_mismatch_common > 0]
    print_gap_rows = [row for row in rows if row.print_parity_total > 0 and row.print_parity_signals < row.print_parity_total]
    failure_rows = [row for row in rows if row.status not in {"parity_ok", "parity_mismatch"}]

    lines: list[str] = []
    lines.append("# Scaled Example Suite Summary\n\n")
    lines.append(f"- Cases: {len(rows)}\n")
    lines.append(
        "- Practical status counts: "
        + ", ".join(f"{k}={v}" for k, v in sorted(practical_counts.items()))
        + "\n"
    )
    lines.append(
        "- Strict status counts: "
        + ", ".join(f"{k}={v}" for k, v in sorted(strict_counts.items()))
        + "\n\n"
    )

    lines.append("## Runtime offenders (absolute JAX time)\n\n")
    for row in offenders_runtime:
        lines.append(
            f"- {row.case}: jax={_fmt(row.jax_runtime_s)}s fortran={_fmt(row.fortran_runtime_s)}s "
            f"ratio={_fmt((float(row.jax_runtime_s) / float(row.fortran_runtime_s)) if row.fortran_runtime_s not in (None, 0.0) else None)} "
            f"res={row.final_resolution} status={row.status}\n"
        )
    lines.append("\n## Runtime offenders (JAX/Fortran ratio)\n\n")
    for row, ratio in offenders_runtime_ratio:
        lines.append(
            f"- {row.case}: ratio={_fmt(ratio)} jax={_fmt(row.jax_runtime_s)}s fortran={_fmt(row.fortran_runtime_s)}s "
            f"res={row.final_resolution} status={row.status}\n"
        )
    lines.append("\n## Memory offenders (absolute JAX RSS)\n\n")
    for row in offenders_memory:
        lines.append(
            f"- {row.case}: jax={_fmt(row.jax_max_rss_mb, places=1)}MB "
            f"fortran={_fmt(row.fortran_max_rss_mb, places=1)}MB "
            f"ratio={_fmt((float(row.jax_max_rss_mb) / float(row.fortran_max_rss_mb)) if row.fortran_max_rss_mb not in (None, 0.0) else None)} "
            f"res={row.final_resolution} status={row.status}\n"
        )
    lines.append("\n## Memory offenders (JAX/Fortran ratio)\n\n")
    for row, ratio in offenders_memory_ratio:
        lines.append(
            f"- {row.case}: ratio={_fmt(ratio)} jax={_fmt(row.jax_max_rss_mb, places=1)}MB "
            f"fortran={_fmt(row.fortran_max_rss_mb, places=1)}MB "
            f"res={row.final_resolution} status={row.status}\n"
        )
    lines.append("\n## Mismatches\n\n")
    if mismatch_rows:
        for row in sorted(
            mismatch_rows,
            key=lambda item: (item.n_mismatch_common, item.strict_n_mismatch_common),
            reverse=True,
        ):
            lines.append(
                f"- {row.case}: practical={row.n_mismatch_common}/{row.n_common_keys} "
                f"strict={row.strict_n_mismatch_common}/{row.strict_n_common_keys} "
                f"solver={row.n_mismatch_solver} physics={row.n_mismatch_physics} "
                f"sample={','.join(row.mismatch_keys_sample[:4]) or '-'}\n"
            )
    else:
        lines.append("- None\n")
    lines.append("\n## Print parity gaps\n\n")
    if print_gap_rows:
        for row in sorted(print_gap_rows, key=lambda item: (item.print_parity_total - item.print_parity_signals), reverse=True):
            lines.append(
                f"- {row.case}: {row.print_parity_signals}/{row.print_parity_total} "
                f"missing={','.join(row.print_missing_signals[:6]) or '-'}\n"
            )
    else:
        lines.append("- None\n")
    lines.append("\n## Failures and blockers\n\n")
    if failure_rows:
        for row in sorted(failure_rows, key=lambda item: (item.status, item.case)):
            lines.append(
                f"- {row.case}: status={row.status} blocker={row.blocker_type} "
                f"attempts={row.attempts} reductions={row.reductions} note={row.message}\n"
            )
    else:
        lines.append("- None\n")

    out_path.write_text("".join(lines), encoding="utf-8")


def _write_suite_outputs(rows: list[CaseResult], out_root: Path) -> None:
    ordered = [rows_by_case for rows_by_case in sorted(rows, key=lambda row: row.case)]
    report_json = out_root / "suite_report.json"
    report_json.write_text(json.dumps([asdict(row) for row in ordered], indent=2), encoding="utf-8")

    report_json_strict = out_root / "suite_report_strict.json"
    strict_rows = []
    for row in ordered:
        row_dict = asdict(row)
        row_dict["status"] = _status_for_mode(row, strict=True)
        row_dict["n_common_keys"] = row.strict_n_common_keys
        row_dict["n_mismatch_common"] = row.strict_n_mismatch_common
        row_dict["mismatch_keys_sample"] = list(row.strict_mismatch_keys_sample)
        row_dict["n_mismatch_solver"] = row.strict_n_mismatch_solver
        row_dict["n_mismatch_physics"] = row.strict_n_mismatch_physics
        row_dict["mismatch_solver_sample"] = list(row.strict_mismatch_solver_sample)
        row_dict["mismatch_physics_sample"] = list(row.strict_mismatch_physics_sample)
        row_dict["max_abs_mismatch"] = row.strict_max_abs_mismatch
        row_dict["compare_mode"] = "strict"
        strict_rows.append(row_dict)
    report_json_strict.write_text(json.dumps(strict_rows, indent=2), encoding="utf-8")

    report_rst = out_root / "suite_status.rst"
    report_rst_strict = out_root / "suite_status_strict.rst"
    _write_rst(ordered, report_rst, strict=False)
    _write_rst(ordered, report_rst_strict, strict=True)
    _write_lane_summary(ordered, out_root / "summary.md")


def _scaled_resolution_from_reference(*, reference_input: Path, runtime_input: Path, scale_factor: float) -> dict[str, int]:
    reference_res = _resolution_from_namelist(reference_input)
    rhs_mode = _rhs_mode_from_namelist(runtime_input)
    if not reference_res:
        return {}
    if float(scale_factor) <= 0.0:
        raise ValueError(f"scale_factor must be positive, got {scale_factor!r}")
    updates: dict[str, int] = {}
    for key, val in reference_res.items():
        if key == "NZETA" and int(val) <= 1 and float(scale_factor) > 1.0:
            updates[key] = int(val)
            continue
        scaled_float = float(val) * float(scale_factor)
        if float(scale_factor) >= 1.0:
            scaled = int(math.ceil(scaled_float))
        else:
            scaled = int(math.floor(scaled_float))
        updates[key] = scaled
    return _sanitize_resolution(updates, current=reference_res, rhs_mode=rhs_mode)


def _prepare_scaled_seed(
    *,
    source_input: Path,
    reference_input: Path,
    case_out_dir: Path,
    scale_factor: float,
) -> tuple[Path, dict[str, int], dict[str, int], dict[str, int]]:
    case_out_dir.mkdir(parents=True, exist_ok=True)
    original_copy = case_out_dir / "input.example.namelist"
    shutil.copyfile(source_input, original_copy)
    scaled_seed = case_out_dir / "input.scale_seed.namelist"
    shutil.copyfile(source_input, scaled_seed)
    source_res = _resolution_from_namelist(source_input)
    reference_res = _resolution_from_namelist(reference_input)
    scaled_res = _scaled_resolution_from_reference(
        reference_input=reference_input,
        runtime_input=source_input,
        scale_factor=float(scale_factor),
    )
    if scaled_res:
        scaled_seed.write_text(
            _replace_resolution_values_in_text(scaled_seed.read_text(), updates=scaled_res),
            encoding="utf-8",
        )
    prev_equilibria_dirs = os.environ.get("SFINCS_JAX_EQUILIBRIA_DIRS", "")
    os.environ["SFINCS_JAX_EQUILIBRIA_DIRS"] = (
        str(source_input.parent)
        if not prev_equilibria_dirs
        else os.pathsep.join((str(source_input.parent), prev_equilibria_dirs))
    )
    try:
        localize_equilibrium_file_in_place(input_namelist=scaled_seed, overwrite=True)
    finally:
        if prev_equilibria_dirs:
            os.environ["SFINCS_JAX_EQUILIBRIA_DIRS"] = prev_equilibria_dirs
        else:
            os.environ.pop("SFINCS_JAX_EQUILIBRIA_DIRS", None)
    scaled_res = _resolution_from_namelist(scaled_seed)
    return scaled_seed, source_res, reference_res, scaled_res


def _stage_reference_fortran_artifacts(
    *,
    case_name: str,
    case_input: Path,
    case_out_dir: Path,
    reference_results_root: Path | None,
) -> bool:
    if reference_results_root is None:
        return False
    ref_case_dir = reference_results_root / case_name
    h5_candidates = (
        ref_case_dir / "last_success" / "sfincsOutput_fortran.h5",
        ref_case_dir / "fortran_run" / "sfincsOutput.h5",
    )
    log_candidates = (
        ref_case_dir / "last_success" / "sfincs_fortran.log",
        ref_case_dir / "fortran_run" / "sfincs.log",
    )
    input_candidates = (
        ref_case_dir / "input.namelist",
        ref_case_dir / "fortran_run" / "input.namelist",
    )
    ref_h5 = next((path for path in h5_candidates if path.exists()), None)
    ref_log = next((path for path in log_candidates if path.exists()), None)
    ref_input = next((path for path in input_candidates if path.exists()), None)
    if ref_h5 is None:
        return False
    if ref_input is not None and ref_input.read_text(encoding="utf-8") != case_input.read_text(encoding="utf-8"):
        raise RuntimeError(f"Reference input mismatch for case {case_name}: {ref_input} does not match {case_input}")
    staged_dir = case_out_dir / "fortran_run"
    staged_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(case_input, staged_dir / "input.namelist")
    shutil.copyfile(ref_h5, staged_dir / "sfincsOutput.h5")
    if ref_log is not None:
        shutil.copyfile(ref_log, staged_dir / "sfincs.log")
    return True


def _run_prepared_case(
    *,
    case_name: str,
    case_input: Path,
    case_out_dir: Path,
    fortran_exe: Path | None,
    timeout_s: float,
    rtol: float,
    atol: float,
    max_attempts: int,
    reuse_fortran: bool,
    collect_iterations: bool,
    jax_repeats: int,
    jax_cache_dir: Path,
    equilibria_search_dir: Path | None,
    reference_results_root: Path | None,
) -> CaseResult:
    staged_reference = _stage_reference_fortran_artifacts(
        case_name=case_name,
        case_input=case_input,
        case_out_dir=case_out_dir,
        reference_results_root=reference_results_root,
    )
    if fortran_exe is None and not staged_reference:
        raise FileNotFoundError(
            f"No staged Fortran reference available for case {case_name}, and --fortran-exe was not provided."
        )
    prev_equilibria_dirs = os.environ.get("SFINCS_JAX_EQUILIBRIA_DIRS", "")
    if equilibria_search_dir is not None:
        os.environ["SFINCS_JAX_EQUILIBRIA_DIRS"] = (
            str(equilibria_search_dir)
            if not prev_equilibria_dirs
            else os.pathsep.join((str(equilibria_search_dir), prev_equilibria_dirs))
        )
    try:
        return _run_case(
            case_name=case_name,
            case_input=case_input,
            case_out_dir=case_out_dir,
            fortran_exe=fortran_exe if fortran_exe is not None else (case_out_dir / "__unused_sfincs__"),
            timeout_s=timeout_s,
            rtol=rtol,
            atol=atol,
            max_attempts=max_attempts,
            use_seed_resolution=True,
            reuse_fortran=bool(reuse_fortran or staged_reference),
            collect_iterations=collect_iterations,
            jax_repeats=jax_repeats,
            jax_cache_dir=jax_cache_dir,
        )
    finally:
        if prev_equilibria_dirs:
            os.environ["SFINCS_JAX_EQUILIBRIA_DIRS"] = prev_equilibria_dirs
        else:
            os.environ.pop("SFINCS_JAX_EQUILIBRIA_DIRS", None)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the vendored SFINCS example suite at scaled resolution.")
    parser.add_argument(
        "--examples-root",
        type=Path,
        default=Path("examples") / "sfincs_examples",
        help="Root containing upstream-style example directories.",
    )
    parser.add_argument(
        "--extra-input",
        action="append",
        default=[str(Path("examples") / "additional_examples" / "input.namelist")],
        help="Extra input.namelist to include outside --examples-root. Repeatable.",
    )
    parser.add_argument(
        "--scale-factor",
        type=float,
        default=1.0,
        help="Scale applied to upstream reference NTHETA/NZETA/NX/NXI before running.",
    )
    parser.add_argument(
        "--resolution-reference-root",
        type=Path,
        default=None,
        help=(
            "Optional root containing canonical upstream example inputs used only for "
            "NTHETA/NZETA/NX/NXI. Matching is by case directory name."
        ),
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("tests") / "scaled_example_suite",
        help="Output directory for per-case runs and reports.",
    )
    parser.add_argument(
        "--fortran-exe",
        type=Path,
        default=None,
        help="Path to the Fortran SFINCS v3 executable.",
    )
    parser.add_argument(
        "--reference-results-root",
        type=Path,
        default=None,
        help=(
            "Optional existing suite root containing per-case Fortran artifacts to reuse "
            "for comparison instead of re-running Fortran on this lane."
        ),
    )
    parser.add_argument("--timeout-s", type=float, default=900.0, help="Per-attempt timeout in seconds.")
    parser.add_argument("--rtol", type=float, default=5e-4, help="Relative tolerance for H5 comparison.")
    parser.add_argument("--atol", type=float, default=1e-9, help="Absolute tolerance for H5 comparison.")
    parser.add_argument("--max-attempts", type=int, default=3, help="Maximum attempts per case.")
    parser.add_argument(
        "--pattern",
        type=str,
        default=None,
        help="Regex filter applied to case name and source path.",
    )
    parser.add_argument(
        "--reuse-fortran",
        action="store_true",
        help="Reuse a matching per-case Fortran result when present.",
    )
    parser.add_argument(
        "--reset-report",
        action="store_true",
        help="Overwrite suite_report.json instead of merging with prior results.",
    )
    parser.add_argument(
        "--jax-cache-dir",
        type=Path,
        default=Path("tests") / "scaled_example_suite" / ".jax_compilation_cache",
        help="Persistent JAX compilation cache directory.",
    )
    parser.add_argument(
        "--jax-repeats",
        type=int,
        default=1,
        help="Number of sfincs_jax repeats per case.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of cases to run in parallel.",
    )
    parser.add_argument("--case-index", type=int, default=None, help="0-based job-array index.")
    parser.add_argument("--case-stride", type=int, default=1, help="Job-array stride.")
    parser.add_argument(
        "--no-collect-iterations",
        action="store_true",
        help="Disable solver-iteration parsing from sfincs_jax logs.",
    )
    args = parser.parse_args()

    examples_root = Path(args.examples_root)
    reference_root = Path(args.resolution_reference_root) if args.resolution_reference_root is not None else None
    out_root = Path(args.out_root)
    fortran_exe = Path(args.fortran_exe) if args.fortran_exe is not None else None
    reference_results_root = Path(args.reference_results_root) if args.reference_results_root is not None else None
    if not examples_root.exists():
        raise SystemExit(f"examples root does not exist: {examples_root}")
    if reference_root is not None and not reference_root.exists():
        raise SystemExit(f"resolution reference root does not exist: {reference_root}")
    if reference_results_root is not None and not reference_results_root.exists():
        raise SystemExit(f"reference results root does not exist: {reference_results_root}")
    if fortran_exe is None and reference_results_root is None:
        raise SystemExit("Either --fortran-exe or --reference-results-root must be provided.")
    if fortran_exe is not None and not fortran_exe.exists():
        raise SystemExit(f"Fortran executable does not exist: {fortran_exe}")
    out_root.mkdir(parents=True, exist_ok=True)

    inputs = _iter_inputs(examples_root)
    extra_inputs: list[Path] = []
    for raw in args.extra_input:
        path = Path(raw)
        if path.exists():
            extra_inputs.append(path)
    deduped_inputs: list[Path] = []
    seen_inputs: set[Path] = set()
    for input_path in [*inputs, *extra_inputs]:
        resolved = input_path.resolve()
        if resolved in seen_inputs:
            continue
        seen_inputs.add(resolved)
        deduped_inputs.append(input_path)
    inputs = deduped_inputs
    case_names = _case_names_for_inputs(inputs, base_root=examples_root.parent)

    if args.pattern:
        rx = re.compile(str(args.pattern), flags=re.IGNORECASE)
        inputs = [
            path
            for path in inputs
            if rx.search(case_names[path]) or rx.search(str(path.parent)) or rx.search(str(path))
        ]
    if not inputs:
        raise SystemExit("No input.namelist files matched.")

    stride_val = max(1, int(args.case_stride))
    if args.case_index is not None:
        idx = int(args.case_index)
        if idx < 0 or idx >= stride_val:
            raise SystemExit(f"--case-index={idx} out of range for --case-stride={stride_val}")
        inputs = [path for i, path in enumerate(inputs) if i % stride_val == idx]
        if not inputs:
            raise SystemExit("No inputs matched after case-index filtering.")

    report_json = out_root / "suite_report.json"
    merged_results: dict[str, CaseResult] = {} if args.reset_report else _load_existing_results(report_json)
    current_run_results: list[CaseResult] = []

    manifest_cases: list[dict[str, object]] = []
    prepared: list[tuple[int, str, Path, Path, Path, dict[str, int], dict[str, int], dict[str, int]]] = []
    for input_path in inputs:
        case = case_names[input_path]
        case_out = out_root / case
        reference_input = input_path
        if reference_root is not None:
            candidate = reference_root / case / "input.namelist"
            if candidate.exists():
                reference_input = candidate
        seed_input, source_res, reference_res, scaled_res = _prepare_scaled_seed(
            source_input=input_path,
            reference_input=reference_input,
            case_out_dir=case_out,
            scale_factor=float(args.scale_factor),
        )
        est_size = int(_estimate_active_size_from_namelist(seed_input) or 0)
        manifest_cases.append(
            {
                "case": case,
                "source_input": _repo_rel(input_path),
                "source_input_abs": str(input_path.resolve()),
                "reference_input": _repo_rel(reference_input),
                "reference_input_abs": str(reference_input.resolve()),
                "source_resolution": source_res,
                "reference_resolution": reference_res,
                "scaled_seed_resolution": scaled_res,
                "estimated_active_size": est_size,
                "scale_factor": float(args.scale_factor),
            }
        )
        prepared.append((est_size, case, input_path, reference_input, seed_input, source_res, reference_res, scaled_res))

    prepared.sort(key=lambda item: (item[0], item[1]))

    manifest = {
        "scale_factor": float(args.scale_factor),
        "timeout_s": float(args.timeout_s),
        "rtol": float(args.rtol),
        "atol": float(args.atol),
        "max_attempts": int(args.max_attempts),
        "jax_repeats": int(args.jax_repeats),
        "jobs": int(args.jobs),
        "resolution_reference_root": _repo_rel(reference_root) if reference_root is not None else None,
        "reference_results_root": _repo_rel(reference_results_root) if reference_results_root is not None else None,
        "fortran_exe": _repo_rel(fortran_exe) if fortran_exe is not None else None,
        "environment": _gather_jax_env(),
        "cases": manifest_cases,
    }
    (out_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def _handle_result(result: CaseResult) -> None:
        prev = merged_results.get(result.case)
        if prev is not None:
            for attr in (
                "fortran_runtime_s",
                "jax_runtime_s",
                "jax_runtime_s_cold",
                "jax_runtime_s_warm",
                "fortran_max_rss_mb",
                "jax_max_rss_mb",
            ):
                if getattr(result, attr) is None and getattr(prev, attr) is not None:
                    setattr(result, attr, getattr(prev, attr))
        current_run_results.append(result)
        merged_results[result.case] = result
        _write_suite_outputs(list(merged_results.values()), out_root)
        print(
            f"  status={result.status} attempts={result.attempts} reductions={result.reductions} "
            f"res={result.final_resolution} mismatch={result.n_mismatch_common}/{result.n_common_keys} "
            f"strict={result.strict_n_mismatch_common}/{result.strict_n_common_keys} "
            f"printParity={result.print_parity_signals}/{result.print_parity_total} blocker={result.blocker_type}"
        )

    jobs = max(1, int(args.jobs))
    if jobs <= 1:
        for index, (est_size, case, input_path, reference_input, seed_input, _source_res, _reference_res, scaled_res) in enumerate(prepared, start=1):
            print(f"[{index}/{len(prepared)}] {case}")
            print(f"  source={input_path}")
            print(f"  reference={reference_input}")
            print(f"  scaled_seed={scaled_res} est_size={est_size}")
            case_out = out_root / case
            result = _run_prepared_case(
                case_name=case,
                case_input=seed_input,
                case_out_dir=case_out,
                fortran_exe=fortran_exe,
                timeout_s=float(args.timeout_s),
                rtol=float(args.rtol),
                atol=float(args.atol),
                max_attempts=int(args.max_attempts),
                reuse_fortran=bool(args.reuse_fortran),
                collect_iterations=not bool(args.no_collect_iterations),
                jax_repeats=int(args.jax_repeats),
                jax_cache_dir=(REPO_ROOT / args.jax_cache_dir),
                equilibria_search_dir=seed_input.parent,
                reference_results_root=reference_results_root,
            )
            _handle_result(result)
    else:
        print(f"Running {len(prepared)} cases with --jobs={jobs}")
        futures = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as pool:
            for index, (est_size, case, input_path, reference_input, seed_input, _source_res, _reference_res, scaled_res) in enumerate(prepared, start=1):
                print(f"[{index}/{len(prepared)}] {case}")
                print(f"  source={input_path}")
                print(f"  reference={reference_input}")
                print(f"  scaled_seed={scaled_res} est_size={est_size}")
                case_out = out_root / case
                futures.append(
                    pool.submit(
                        _run_prepared_case,
                        case_name=case,
                        case_input=seed_input,
                        case_out_dir=case_out,
                        fortran_exe=fortran_exe,
                        timeout_s=float(args.timeout_s),
                        rtol=float(args.rtol),
                        atol=float(args.atol),
                        max_attempts=int(args.max_attempts),
                        reuse_fortran=bool(args.reuse_fortran),
                        collect_iterations=not bool(args.no_collect_iterations),
                        jax_repeats=int(args.jax_repeats),
                        jax_cache_dir=(REPO_ROOT / args.jax_cache_dir),
                        equilibria_search_dir=seed_input.parent,
                        reference_results_root=reference_results_root,
                    )
                )
            for fut in concurrent.futures.as_completed(futures):
                result = fut.result()
                _handle_result(result)

    ordered = [merged_results[key] for key in sorted(merged_results)]
    _write_suite_outputs(ordered, out_root)

    print(f"Wrote {report_json}")
    print(f"Wrote {out_root / 'suite_report_strict.json'}")
    print(f"Wrote {out_root / 'suite_status.rst'}")
    print(f"Wrote {out_root / 'suite_status_strict.rst'}")
    print(f"Wrote {out_root / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
