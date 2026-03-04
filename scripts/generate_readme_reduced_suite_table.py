#!/usr/bin/env python
from __future__ import annotations

import html
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
README = REPO_ROOT / "README.md"
REPORT = REPO_ROOT / "tests" / "reduced_upstream_examples" / "suite_report.json"
REPORT_STRICT = REPO_ROOT / "tests" / "reduced_upstream_examples" / "suite_report_strict.json"
REPORT_CPU = REPO_ROOT / "tests" / "reduced_upstream_examples" / "suite_report_cpu.json"
REPORT_GPU = REPO_ROOT / "tests" / "reduced_upstream_examples" / "suite_report_gpu.json"
REPORT_STRICT_CPU = REPO_ROOT / "tests" / "reduced_upstream_examples" / "suite_report_strict_cpu.json"
REPORT_STRICT_GPU = REPO_ROOT / "tests" / "reduced_upstream_examples" / "suite_report_strict_gpu.json"

BEGIN = "<!-- BEGIN REDUCED_SUITE_TABLE -->"
END = "<!-- END REDUCED_SUITE_TABLE -->"


def _load(path: Path) -> dict[str, dict]:
    data = json.loads(path.read_text())
    return {row["case"]: row for row in data}


def _format_row(case: str, row_cpu: dict, row_gpu: dict | None, row_strict: dict | None) -> str:
    n_common = int(row_cpu.get("n_common_keys", 0))
    n_bad = int(row_cpu.get("n_mismatch_common", 0))
    if row_strict is None:
        n_common_strict = n_common
        n_bad_strict = n_bad
    else:
        n_common_strict = int(row_strict.get("n_common_keys", n_common))
        n_bad_strict = int(row_strict.get("n_mismatch_common", n_bad))

    if n_common > 0:
        mismatch = f"{n_bad}/{n_common} (strict {n_bad_strict}/{n_common_strict})"
    else:
        mismatch = row_cpu.get("status", "-")

    pp_total = int(row_cpu.get("print_parity_total", 0))
    if pp_total > 0:
        pp = f"{row_cpu.get('print_parity_signals', 0)}/{pp_total}"
    else:
        pp = "-"

    ft = row_cpu.get("fortran_runtime_s")
    jt_cpu = row_cpu.get("jax_runtime_s")
    jt_gpu = row_gpu.get("jax_runtime_s") if row_gpu is not None else None
    ft_s = "-" if ft is None else f"{float(ft):.3f}"
    jt_cpu_s = "-" if jt_cpu is None else f"{float(jt_cpu):.3f}"
    jt_gpu_s = "-" if jt_gpu is None else f"{float(jt_gpu):.3f}"
    fm = row_cpu.get("fortran_max_rss_mb")
    jm_cpu = row_cpu.get("jax_max_rss_mb")
    jm_gpu = row_gpu.get("jax_max_rss_mb") if row_gpu is not None else None
    fm_s = "-" if fm is None else f"{float(fm):.1f}"
    jm_cpu_s = "-" if jm_cpu is None else f"{float(jm_cpu):.1f}"
    jm_gpu_s = "-" if jm_gpu is None else f"{float(jm_gpu):.1f}"

    case_cell = html.escape(case)
    return (
        f"| {case_cell} | {ft_s} | {jt_cpu_s} | {jt_gpu_s} | {fm_s} | {jm_cpu_s} | {jm_gpu_s} | {mismatch} | {pp} |"
    )


def main() -> int:
    if REPORT_CPU.exists():
        rows_cpu = _load(REPORT_CPU)
    else:
        if not REPORT.exists():
            raise SystemExit(f"Missing report: {REPORT}")
        rows_cpu = _load(REPORT)
    if REPORT_GPU.exists():
        rows_gpu = _load(REPORT_GPU)
    else:
        rows_gpu = {}
    if REPORT_STRICT_CPU.exists():
        rows_strict = _load(REPORT_STRICT_CPU)
    elif REPORT_STRICT.exists():
        rows_strict = _load(REPORT_STRICT)
    else:
        rows_strict = {}

    table_lines = [
        "| Case | Fortran CPU(s) | sfincs_jax CPU(s) | sfincs_jax GPU(s) | Fortran CPU MB | sfincs_jax CPU MB | sfincs_jax GPU MB | Mismatches (practical/strict) | Print comparison |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for case in sorted(rows_cpu):
        table_lines.append(_format_row(case, rows_cpu[case], rows_gpu.get(case), rows_strict.get(case)))

    readme = README.read_text()
    if BEGIN not in readme or END not in readme:
        raise SystemExit("README markers not found.")
    prefix, rest = readme.split(BEGIN, 1)
    _table, suffix = rest.split(END, 1)
    new_block = "\n".join([BEGIN, *table_lines, END])
    README.write_text(prefix + new_block + suffix)
    print("Updated README reduced-suite table.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
