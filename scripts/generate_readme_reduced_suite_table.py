#!/usr/bin/env python
from __future__ import annotations

import html
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
README = REPO_ROOT / "README.md"
REPORT = REPO_ROOT / "tests" / "reduced_upstream_examples" / "suite_report.json"
REPORT_STRICT = REPO_ROOT / "tests" / "reduced_upstream_examples" / "suite_report_strict.json"

BEGIN = "<!-- BEGIN REDUCED_SUITE_TABLE -->"
END = "<!-- END REDUCED_SUITE_TABLE -->"


def _load(path: Path) -> dict[str, dict]:
    data = json.loads(path.read_text())
    return {row["case"]: row for row in data}


def _format_row(case: str, row: dict, row_strict: dict | None) -> str:
    n_common = int(row.get("n_common_keys", 0))
    n_bad = int(row.get("n_mismatch_common", 0))
    if row_strict is None:
        n_common_strict = n_common
        n_bad_strict = n_bad
    else:
        n_common_strict = int(row_strict.get("n_common_keys", n_common))
        n_bad_strict = int(row_strict.get("n_mismatch_common", n_bad))

    if n_common > 0:
        mismatch = f"{n_bad}/{n_common} (strict {n_bad_strict}/{n_common_strict})"
    else:
        mismatch = row.get("status", "-")

    pp_total = int(row.get("print_parity_total", 0))
    if pp_total > 0:
        pp = f"{row.get('print_parity_signals', 0)}/{pp_total}"
    else:
        pp = "-"

    ft = row.get("fortran_runtime_s")
    jt = row.get("jax_runtime_s")
    ft_s = "-" if ft is None else f"{float(ft):.3f}"
    jt_s = "-" if jt is None else f"{float(jt):.3f}"
    fm = row.get("fortran_max_rss_mb")
    jm = row.get("jax_max_rss_mb")
    fm_s = "-" if fm is None else f"{float(fm):.1f}"
    jm_s = "-" if jm is None else f"{float(jm):.1f}"

    case_cell = html.escape(case)
    return (
        f"| {case_cell} | {ft_s} | {jt_s} | {fm_s} | {jm_s} | {mismatch} | {pp} |"
    )


def main() -> int:
    if not REPORT.exists():
        raise SystemExit(f"Missing report: {REPORT}")
    rows = _load(REPORT)
    rows_strict = _load(REPORT_STRICT) if REPORT_STRICT.exists() else {}

    table_lines = [
        "| Case | Fortran(s) | sfincs_jax(s) | Fortran MB | sfincs_jax MB | Mismatches (practical/strict) | Print parity |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for case in sorted(rows):
        table_lines.append(_format_row(case, rows[case], rows_strict.get(case)))

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
