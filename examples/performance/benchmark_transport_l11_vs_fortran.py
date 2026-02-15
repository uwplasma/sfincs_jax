"""Benchmark `L11` parity and runtime vs Fortran v3 on four reference cases.

This script generates a 2x2 panel figure. Each panel corresponds to one case and shows:
1) left: the figure of merit `L11 = transportMatrix[0,0]` across repeated runs
2) right (inset): mean runtime per run for Fortran vs `sfincs_jax`

For `sfincs_jax`, runtime excludes compilation by performing one warm-up run per case.

Usage
-----
From the repository root:

    # Reproducible mode (no Fortran runtime needed): uses frozen v3 fixture outputs/logs
    python examples/performance/benchmark_transport_l11_vs_fortran.py --repeats 4

    # Live mode (if Fortran v3 executable is runnable in your environment):
    python examples/performance/benchmark_transport_l11_vs_fortran.py \
        --fortran-exe ../sfincs/fortran/version3/sfincs --repeats 4
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

import h5py
import numpy as np

try:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "sfincs_jax_mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # noqa: BLE001
    raise SystemExit('This example requires matplotlib. Install with: pip install -e ".[viz]"') from exc

from sfincs_jax.io import localize_equilibrium_file_in_place, write_sfincs_jax_output_h5


def _l11_from_h5(path: Path) -> float:
    with h5py.File(path, "r") as f:
        tm = np.asarray(f["transportMatrix"], dtype=np.float64)
    if tm.shape != (2, 2):
        raise ValueError(f"Expected 2x2 transportMatrix in {path}, got {tm.shape}")
    return float(tm[0, 0])


def _run_fortran_once(*, input_template: Path, exe: Path, workdir: Path) -> tuple[float, float]:
    run_dir = workdir / "fortran_run"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    input_path = run_dir / "input.namelist"
    shutil.copy2(input_template, input_path)
    localize_equilibrium_file_in_place(input_namelist=input_path, overwrite=False)

    t0 = time.perf_counter()
    subprocess.run([str(exe.resolve())], cwd=run_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    dt = time.perf_counter() - t0
    l11 = _l11_from_h5(run_dir / "sfincsOutput.h5")
    return dt, l11


def _fortran_runtime_from_fixture_log(path: Path) -> float:
    txt = path.read_text(encoding="utf-8", errors="replace")
    marker = "Time to solve:"
    total = 0.0
    for line in txt.splitlines():
        if marker not in line:
            continue
        tail = line.split(marker, 1)[1]
        # Example line tail: "    2.5300000561401248E-004  seconds."
        token = tail.strip().split()[0]
        total += float(token.replace("D", "E").replace("d", "e"))
    if total <= 0.0:
        raise ValueError(f"Could not parse solver runtime from fixture log: {path}")
    return total


def _run_jax_once(*, input_template: Path, workdir: Path, warmup: bool) -> tuple[float, float]:
    run_dir = workdir / ("jax_warmup" if warmup else "jax_run")
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    input_path = run_dir / "input.namelist"
    shutil.copy2(input_template, input_path)
    localize_equilibrium_file_in_place(input_namelist=input_path, overwrite=False)

    out_path = run_dir / "sfincsOutput.h5"
    t0 = time.perf_counter()
    write_sfincs_jax_output_h5(
        input_namelist=input_path,
        output_path=out_path,
        overwrite=True,
        compute_transport_matrix=True,
    )
    dt = time.perf_counter() - t0
    l11 = _l11_from_h5(out_path)
    return dt, l11


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fortran-exe",
        type=Path,
        default=None,
        help="Optional path to compiled Fortran v3 sfincs executable. If omitted, uses frozen fixture outputs/logs.",
    )
    parser.add_argument("--repeats", type=int, default=4, help="Number of timed runs per case (default: 4).")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("examples") / "performance" / "output" / "transport_l11_vs_fortran",
        help="Directory to write figure and JSON summary.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    ref_dir = repo_root / "tests" / "ref"
    out_dir = (repo_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = [
        ("scheme1", ref_dir / "monoenergetic_PAS_tiny_scheme1.input.namelist"),
        ("scheme11", ref_dir / "monoenergetic_PAS_tiny_scheme11.input.namelist"),
        ("scheme12", ref_dir / "monoenergetic_PAS_tiny_scheme12.input.namelist"),
        ("scheme5_filtered", ref_dir / "monoenergetic_PAS_tiny_scheme5_filtered.input.namelist"),
    ]

    repeats = int(args.repeats)
    if repeats < 2:
        raise ValueError("--repeats must be >= 2")

    summary: dict[str, dict[str, object]] = {}

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5), constrained_layout=True)
    flat_axes = list(axes.reshape(-1))

    for ax, (label, input_template) in zip(flat_axes, cases, strict=True):
        if not input_template.exists():
            raise FileNotFoundError(str(input_template))

        case_tmp = Path(tempfile.mkdtemp(prefix=f"sfincs_jax_bench_{label}_"))
        try:
            fortran_times: list[float] = []
            fortran_l11: list[float] = []
            jax_times: list[float] = []
            jax_l11: list[float] = []

            # JAX warm-up compile (excluded from runtime metric).
            _run_jax_once(input_template=input_template, workdir=case_tmp, warmup=True)

            if args.fortran_exe is not None:
                for i in range(repeats):
                    dt_f, l11_f = _run_fortran_once(input_template=input_template, exe=args.fortran_exe, workdir=case_tmp / f"f{i}")
                    fortran_times.append(dt_f)
                    fortran_l11.append(l11_f)
                fortran_source = "live executable"
            else:
                base = input_template.name.replace(".input.namelist", "")
                fixture_h5 = ref_dir / f"{base}.sfincsOutput.h5"
                fixture_log = ref_dir / f"{base}.sfincs.log"
                if not fixture_h5.exists() or not fixture_log.exists():
                    raise FileNotFoundError(
                        f"Missing fixture files for {base}. Need both {fixture_h5.name} and {fixture_log.name}."
                    )
                l11_ref = _l11_from_h5(fixture_h5)
                time_ref = _fortran_runtime_from_fixture_log(fixture_log)
                fortran_l11 = [l11_ref] * repeats
                fortran_times = [time_ref] * repeats
                fortran_source = "frozen fixture"

            for i in range(repeats):
                dt_j, l11_j = _run_jax_once(input_template=input_template, workdir=case_tmp / f"j{i}", warmup=False)
                jax_times.append(dt_j)
                jax_l11.append(l11_j)

            x = np.arange(1, repeats + 1, dtype=np.int32)
            fortran_arr = np.asarray(fortran_l11, dtype=np.float64)
            jax_arr = np.asarray(jax_l11, dtype=np.float64)
            rel_diff = (jax_arr - fortran_arr) / fortran_arr
            ax.plot(x, rel_diff, marker="o", lw=1.8, label="(JAX − Fortran) / Fortran")
            ax.axhline(0.0, color="k", lw=1.0, alpha=0.3)
            ax.set_title(label)
            ax.set_xlabel("Run index")
            ax.set_ylabel("Relative ΔL11")
            ax.grid(alpha=0.3)

            inset = ax.inset_axes([0.57, 0.10, 0.40, 0.38])
            means = [float(np.mean(fortran_times)), float(np.mean(jax_times))]
            bar_x = np.arange(2, dtype=np.float64)
            inset.bar(bar_x, means, color=["#4C78A8", "#F58518"])
            inset.set_xticks(bar_x)
            inset.set_xticklabels(["Fortran", "JAX"])
            inset.set_ylabel("s/run")
            inset.set_title("Runtime")
            inset.grid(axis="y", alpha=0.25)

            speedup = means[0] / means[1] if means[1] > 0 else float("nan")
            summary[label] = {
                "input": str(input_template),
                "fortran_source": fortran_source,
                "fortran_mean_runtime_s": means[0],
                "jax_mean_runtime_s_excluding_compile": means[1],
                "jax_speedup_vs_fortran": speedup,
                "fortran_l11_runs": fortran_l11,
                "jax_l11_runs": jax_l11,
                "max_abs_l11_diff": float(np.max(np.abs(np.asarray(fortran_l11) - np.asarray(jax_l11)))),
            }
        finally:
            shutil.rmtree(case_tmp, ignore_errors=True)

    handles, labels = flat_axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("SFINCS vs sfincs_jax: relative L11 difference and runtime (JAX runtime excludes compilation)", y=1.02)

    png_path = out_dir / "sfincs_vs_sfincs_jax_l11_runtime_2x2.png"
    pdf_path = out_dir / "sfincs_vs_sfincs_jax_l11_runtime_2x2.pdf"
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)

    summary_path = out_dir / "sfincs_vs_sfincs_jax_l11_runtime_2x2.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
