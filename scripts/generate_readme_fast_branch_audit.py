#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
README = REPO_ROOT / "README.md"
DEFAULT_OUT_ROOT = REPO_ROOT / "tests" / "scaled_example_suite_fast_cpu_v1"
BASELINE_REPORT = REPO_ROOT / "tests" / "scaled_example_suite_release_cpu_v4" / "suite_report.json"
EXAMPLES_ROOT = REPO_ROOT / "examples" / "sfincs_examples"
EXTRA_INPUT = REPO_ROOT / "examples" / "additional_examples" / "input.namelist"

BEGIN = "<!-- BEGIN FAST_BRANCH_AUDIT -->"
END = "<!-- END FAST_BRANCH_AUDIT -->"


def _load_json(path: Path) -> object:
    return json.loads(path.read_text())


def _repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except Exception:  # noqa: BLE001
        return str(path)


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


def _expected_cases() -> list[str]:
    inputs = sorted(EXAMPLES_ROOT.rglob("input.namelist"))
    if EXTRA_INPUT.exists():
        inputs.append(EXTRA_INPUT)
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in inputs:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(path)
    case_names = _case_names_for_inputs(deduped, base_root=EXAMPLES_ROOT.parent)
    return sorted(case_names.values())


def _fmt_float(value: object | None, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def _fmt_runtime_ratio(row: dict[str, object]) -> str:
    jax = row.get("jax_runtime_s")
    fort = row.get("fortran_runtime_s")
    if jax is None or fort in (None, 0):
        return "-"
    return f"{float(jax) / float(fort):.2f}x"


def _fmt_memory_ratio(row: dict[str, object]) -> str:
    jax = row.get("jax_max_rss_mb")
    fort = row.get("fortran_max_rss_mb")
    if jax is None or fort in (None, 0):
        return "-"
    return f"{float(jax) / float(fort):.2f}x"


def _status_counts(rows: list[dict[str, object]], prefix: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts[str(row.get(prefix, row.get("status", "unknown")))] += 1
    return counts


def _top_rows(
    rows: list[dict[str, object]],
    *,
    key: str,
    limit: int = 5,
) -> list[dict[str, object]]:
    valid = [row for row in rows if row.get(key) is not None]
    return sorted(valid, key=lambda row: float(row[key]), reverse=True)[:limit]


def _format_row_summary(row: dict[str, object], *, metric_key: str, digits: int = 1) -> str:
    value = _fmt_float(row.get(metric_key), digits)
    fort_key = "fortran_runtime_s" if metric_key == "jax_runtime_s" else "fortran_max_rss_mb"
    ratio = _fmt_runtime_ratio(row) if metric_key == "jax_runtime_s" else _fmt_memory_ratio(row)
    final_resolution = row.get("final_resolution")
    res_str = f", res={final_resolution}" if final_resolution else ""
    return (
        f"- `{row['case']}`: jax={value}"
        f"{'s' if metric_key == 'jax_runtime_s' else ' MB'} "
        f"fortran={_fmt_float(row.get(fort_key), digits)}"
        f"{'s' if metric_key == 'jax_runtime_s' else ' MB'} "
        f"ratio={ratio} status={row.get('status', '-')}{res_str}"
    )


def _format_mismatch(row: dict[str, object]) -> str:
    return (
        f"- `{row['case']}`: status={row.get('status', '-')}, "
        f"practical={row.get('n_mismatch_common', 0)}/{row.get('n_common_keys', 0)}, "
        f"strict={row.get('strict_n_mismatch_common', 0)}/{row.get('strict_n_common_keys', 0)}, "
        f"sample={','.join(row.get('mismatch_keys_sample', [])[:4]) or '-'}"
    )


def _format_improvement(
    current_row: dict[str, object],
    baseline_row: dict[str, object],
    *,
    metric_key: str,
    digits: int = 1,
) -> str:
    current = float(current_row[metric_key])
    baseline = float(baseline_row[metric_key])
    delta = baseline - current
    unit = "s" if metric_key == "jax_runtime_s" else " MB"
    return (
        f"- `{current_row['case']}`: "
        f"{_fmt_float(baseline, digits)}{unit} -> {_fmt_float(current, digits)}{unit} "
        f"(delta={_fmt_float(delta, digits)}{unit})"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Update README fast-branch audit block.")
    parser.add_argument(
        "--out-root",
        type=Path,
        default=DEFAULT_OUT_ROOT,
        help="Suite output root containing suite_report.json and run_manifest.json.",
    )
    parser.add_argument(
        "--baseline-report",
        type=Path,
        default=BASELINE_REPORT,
        help="Optional baseline report used for improvement summaries.",
    )
    args = parser.parse_args()

    out_root = Path(args.out_root)
    report_path = out_root / "suite_report.json"
    if not report_path.exists():
        raise SystemExit(f"Missing report: {report_path}")
    manifest_path = out_root / "run_manifest.json"
    manifest = _load_json(manifest_path) if manifest_path.exists() else {}

    rows = list(_load_json(report_path))
    case_order = _expected_cases()
    total_cases = len(case_order)
    rows_by_case = {str(row["case"]): row for row in rows}
    missing_cases = [case for case in case_order if case not in rows_by_case]

    strict_counts: Counter[str] = Counter()
    for row in rows:
        strict_mismatch = int(row.get("strict_n_mismatch_common", 0))
        strict_common = int(row.get("strict_n_common_keys", 0))
        strict_status = "parity_ok"
        if strict_common > 0 and strict_mismatch > 0:
            strict_status = "parity_mismatch"
        elif row.get("status") not in {"parity_ok", "parity_mismatch"}:
            strict_status = str(row.get("status"))
        strict_counts[str(strict_status)] += 1

    status_counts = Counter(str(row.get("status", "unknown")) for row in rows)
    mismatches = [row for row in rows if str(row.get("status")) != "parity_ok"]

    runtime_top = _top_rows(rows, key="jax_runtime_s")
    memory_top = _top_rows(rows, key="jax_max_rss_mb")

    improvements_runtime: list[str] = []
    improvements_memory: list[str] = []
    baseline_report = Path(args.baseline_report)
    if baseline_report.exists():
        baseline_rows = {str(row["case"]): row for row in _load_json(baseline_report)}
        paired_runtime = []
        paired_memory = []
        for case, row in rows_by_case.items():
            base = baseline_rows.get(case)
            if not base:
                continue
            if row.get("jax_runtime_s") is not None and base.get("jax_runtime_s") is not None:
                paired_runtime.append((float(base["jax_runtime_s"]) - float(row["jax_runtime_s"]), row, base))
            if row.get("jax_max_rss_mb") is not None and base.get("jax_max_rss_mb") is not None:
                paired_memory.append((float(base["jax_max_rss_mb"]) - float(row["jax_max_rss_mb"]), row, base))
        for _delta, row, base in sorted(paired_runtime, key=lambda item: item[0], reverse=True)[:5]:
            if _delta > 0:
                improvements_runtime.append(_format_improvement(row, base, metric_key="jax_runtime_s", digits=1))
        for _delta, row, base in sorted(paired_memory, key=lambda item: item[0], reverse=True)[:5]:
            if _delta > 0:
                improvements_memory.append(_format_improvement(row, base, metric_key="jax_max_rss_mb", digits=1))

    lines = [
        BEGIN,
        f"Current fast explicit CPU audit comes from `{_repo_rel(out_root)}`.",
        "",
        f"- Recorded cases: `{len(rows)}/{total_cases}`",
        f"- Practical status counts: `{', '.join(f'{k}={status_counts[k]}' for k in sorted(status_counts))}`",
        f"- Strict status counts: `{', '.join(f'{k}={strict_counts[k]}' for k in sorted(strict_counts))}`",
    ]
    if manifest:
        resolution_policy = manifest.get("resolution_policy")
        scale_factor = manifest.get("scale_factor")
        runtime_basis = manifest.get("runtime_target_basis")
        runtime_floor = manifest.get("fortran_min_runtime_s")
        runtime_cap = manifest.get("fortran_max_runtime_s")
        adjust_iters = manifest.get("runtime_adjustment_iters")
        lines.append(
            "- Resolution policy: "
            f"`{resolution_policy}, scale_factor={scale_factor}, runtime_basis={runtime_basis}, "
            f"fortran_min={runtime_floor}, fortran_max={runtime_cap}, adjust_iters={adjust_iters}`"
        )
    if missing_cases:
        lines.append(f"- Remaining cases: `{', '.join(missing_cases)}`")
    else:
        lines.append("- Remaining cases: none")

    lines.extend(
        [
            "",
            "Top CPU runtime offenders:",
            *[_format_row_summary(row, metric_key="jax_runtime_s", digits=3) for row in runtime_top],
            "",
            "Top CPU memory offenders:",
            *[_format_row_summary(row, metric_key="jax_max_rss_mb", digits=1) for row in memory_top],
        ]
    )

    if mismatches:
        lines.extend(
            [
                "",
                "Current mismatches:",
                *[_format_mismatch(row) for row in mismatches],
            ]
        )

    if improvements_runtime:
        lines.extend(
            [
                "",
                f"Largest CPU runtime improvements vs `{_repo_rel(baseline_report)}`:",
                *improvements_runtime,
            ]
        )
    if improvements_memory:
        lines.extend(
            [
                "",
                f"Largest CPU memory improvements vs `{_repo_rel(baseline_report)}`:",
                *improvements_memory,
            ]
        )

    lines.append(END)

    readme = README.read_text()
    if BEGIN not in readme or END not in readme:
        raise SystemExit("README fast-branch markers not found.")
    prefix, rest = readme.split(BEGIN, 1)
    _old, suffix = rest.split(END, 1)
    README.write_text(prefix + "\n".join(lines) + suffix)
    print("Updated README fast-branch audit block.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
