"""Profile compile/runtime split for transport-matrix solves with persistent JAX cache.

This script runs each reference transport case in two isolated Python processes:

1) **cold cache**: empty cache directory, so first call includes compilation.
2) **warm cache**: same cache directory reused, so first call should reuse cached executables.

The script reports a compile estimate ``cold_first - warm_first`` and steady-state runtime
from repeated warm runs. It writes a 2x2 figure and JSON summary.

Usage
-----
From the repository root:

    python examples/performance/profile_transport_compile_runtime_cache.py --repeats 4
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

from sfincs_jax.io import localize_equilibrium_file_in_place


def _run_worker(*, script_path: Path, input_path: Path, cache_dir: Path, repeats: int) -> dict[str, object]:
    env = os.environ.copy()
    env.setdefault("JAX_ENABLE_X64", "True")
    env["JAX_COMPILATION_CACHE_DIR"] = str(cache_dir)
    env["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] = "0"
    env["JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES"] = "0"
    cmd = [
        sys.executable,
        str(script_path),
        "--worker",
        "--input",
        str(input_path),
        "--repeats",
        str(int(repeats)),
    ]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
    return json.loads(proc.stdout)


def _worker_main(*, input_path: Path, repeats: int) -> int:
    from sfincs_jax.namelist import read_sfincs_input
    from sfincs_jax.v3_driver import solve_v3_transport_matrix_linear_gmres

    nml = read_sfincs_input(input_path)
    first_t0 = time.perf_counter()
    first_result = solve_v3_transport_matrix_linear_gmres(nml=nml, tol=1e-10)
    _ = np.asarray(first_result.transport_matrix, dtype=np.float64)
    first_call_s = time.perf_counter() - first_t0

    steady_runs_s: list[float] = []
    for _ in range(max(1, int(repeats))):
        t0 = time.perf_counter()
        result = solve_v3_transport_matrix_linear_gmres(nml=nml, tol=1e-10)
        _ = np.asarray(result.transport_matrix, dtype=np.float64)
        steady_runs_s.append(time.perf_counter() - t0)

    payload = {
        "first_call_s": float(first_call_s),
        "steady_mean_s": float(np.mean(np.asarray(steady_runs_s, dtype=np.float64))),
        "steady_runs_s": [float(v) for v in steady_runs_s],
    }
    print(json.dumps(payload))
    return 0


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--input", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--repeats", type=int, default=4, help="Number of steady-state runs per case (default: 4).")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("examples") / "performance" / "output" / "compile_runtime_cache",
        help="Directory to write figure and JSON summary.",
    )
    args = parser.parse_args()

    if args.worker:
        if args.input is None:
            raise ValueError("--input is required in --worker mode.")
        return _worker_main(input_path=args.input, repeats=int(args.repeats))

    repo_root = Path(__file__).resolve().parents[2]
    script_path = Path(__file__).resolve()
    ref_dir = repo_root / "tests" / "ref"
    out_dir = (repo_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = [
        ("scheme1", ref_dir / "monoenergetic_PAS_tiny_scheme1.input.namelist"),
        ("scheme11", ref_dir / "monoenergetic_PAS_tiny_scheme11.input.namelist"),
        ("scheme12", ref_dir / "monoenergetic_PAS_tiny_scheme12.input.namelist"),
        ("scheme5_filtered", ref_dir / "monoenergetic_PAS_tiny_scheme5_filtered.input.namelist"),
    ]

    try:
        os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "sfincs_jax_mplconfig"))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        raise SystemExit('This example requires matplotlib. Install with: pip install -e ".[viz]"') from exc

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.3), constrained_layout=True)
    flat_axes = list(axes.reshape(-1))

    summary: dict[str, dict[str, float | list[float] | str]] = {}

    for ax, (label, template) in zip(flat_axes, cases, strict=True):
        if not template.exists():
            raise FileNotFoundError(str(template))
        case_tmp = Path(tempfile.mkdtemp(prefix=f"sfincs_jax_cache_prof_{label}_"))
        try:
            input_path = case_tmp / "input.namelist"
            shutil.copy2(template, input_path)
            localize_equilibrium_file_in_place(input_namelist=input_path, overwrite=False)

            cache_dir = case_tmp / "jax_cache"
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)

            cold = _run_worker(script_path=script_path, input_path=input_path, cache_dir=cache_dir, repeats=int(args.repeats))
            warm = _run_worker(script_path=script_path, input_path=input_path, cache_dir=cache_dir, repeats=int(args.repeats))

            cold_first = float(cold["first_call_s"])
            warm_first = float(warm["first_call_s"])
            warm_steady = float(warm["steady_mean_s"])
            compile_est = max(cold_first - warm_first, 0.0)

            bars = [compile_est, warm_steady]
            labels = ["Compile (est.)", "Solve (steady)"]
            colors = ["#54A24B", "#4C78A8"]
            x = np.arange(len(bars), dtype=np.float64)
            ax.bar(x, bars, color=colors)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=15)
            ax.set_ylabel("seconds")
            ax.set_title(label)
            ax.grid(axis="y", alpha=0.25)
            ax.text(
                0.03,
                0.96,
                f"cold first={cold_first:.3f}s\nwarm first={warm_first:.3f}s",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
            )

            summary[label] = {
                "input": str(template),
                "cold_first_call_s": cold_first,
                "warm_first_call_s": warm_first,
                "compile_estimate_s": compile_est,
                "warm_steady_mean_s": warm_steady,
                "warm_steady_runs_s": [float(v) for v in warm["steady_runs_s"]],
            }
        finally:
            shutil.rmtree(case_tmp, ignore_errors=True)

    fig.suptitle("sfincs_jax transport solves: persistent-cache compile/runtime split", y=1.02)
    png_path = out_dir / "transport_compile_runtime_cache_2x2.png"
    pdf_path = out_dir / "transport_compile_runtime_cache_2x2.pdf"
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)

    json_path = out_dir / "transport_compile_runtime_cache_2x2.json"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")
    print(f"Wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
