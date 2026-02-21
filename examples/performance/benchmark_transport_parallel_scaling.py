from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.v3_driver import solve_v3_transport_matrix_linear_gmres


def _run_once(
    input_path: Path,
    *,
    workers: int,
    cache_dir: Path | None,
    precond: str | None,
) -> float:
    os.environ["SFINCS_JAX_FORTRAN_STDOUT"] = "0"
    os.environ["SFINCS_JAX_SOLVER_ITER_STATS"] = "0"
    if precond:
        os.environ["SFINCS_JAX_TRANSPORT_PRECOND"] = precond
    if cache_dir is not None:
        os.environ["JAX_COMPILATION_CACHE_DIR"] = str(cache_dir)
    if workers > 1:
        os.environ["SFINCS_JAX_TRANSPORT_PARALLEL"] = "process"
        os.environ["SFINCS_JAX_TRANSPORT_PARALLEL_WORKERS"] = str(workers)
    else:
        os.environ["SFINCS_JAX_TRANSPORT_PARALLEL"] = "off"
        os.environ["SFINCS_JAX_TRANSPORT_PARALLEL_WORKERS"] = "1"

    nml = read_sfincs_input(input_path)
    t0 = time.perf_counter()
    solve_v3_transport_matrix_linear_gmres(
        nml=nml,
        tol=1e-10,
        input_namelist=input_path,
    )
    return time.perf_counter() - t0


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark parallel whichRHS scaling.")
    repo_root = Path(__file__).resolve().parents[2]
    default_input = repo_root / "examples" / "performance" / "transport_parallel_xxlarge.input.namelist"
    default_out = repo_root / "examples" / "performance" / "output" / "transport_parallel_scaling"
    default_cache = default_out / "jax_cache"

    parser.add_argument(
        "--input",
        type=Path,
        default=default_input,
        help="Path to input.namelist for RHSMode=2/3 transport matrix case.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        nargs="+",
        default=list(range(1, 5)),
        help="Worker counts to benchmark (default 1..4).",
    )
    parser.add_argument("--repeats", type=int, default=2, help="Repeats per worker count.")
    parser.add_argument("--warmup", type=int, default=0, help="Warmup runs per worker count.")
    parser.add_argument(
        "--global-warmup",
        type=int,
        default=1,
        help="Warmup runs before benchmarking (uses workers=1).",
    )
    parser.add_argument(
        "--precond",
        type=str,
        default="xmg",
        help="Transport preconditioner to use during the benchmark (default: xmg).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=default_out,
        help="Output directory for JSON and figure.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=default_cache,
        help="Persistent JAX cache directory.",
    )
    args = parser.parse_args()

    input_path = args.input
    if not input_path.exists():
        raise FileNotFoundError(str(input_path))

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    workers = sorted({int(w) for w in args.workers if int(w) >= 1})

    if args.global_warmup and args.global_warmup > 0:
        for _ in range(int(args.global_warmup)):
            _run_once(input_path, workers=1, cache_dir=cache_dir, precond=args.precond)

    results = []
    for w in workers:
        for _ in range(max(args.warmup, 0)):
            _run_once(input_path, workers=w, cache_dir=cache_dir, precond=args.precond)
        times = []
        for _ in range(max(args.repeats, 1)):
            dt = _run_once(input_path, workers=w, cache_dir=cache_dir, precond=args.precond)
            times.append(dt)
        times = np.asarray(times, dtype=float)
        results.append(
            {
                "workers": w,
                "mean_s": float(np.mean(times)),
                "std_s": float(np.std(times, ddof=1)) if times.size > 1 else 0.0,
                "samples": [float(v) for v in times],
            }
        )
        print(f"workers={w} mean_s={results[-1]['mean_s']:.3f} std_s={results[-1]['std_s']:.3f}", flush=True)

    # Normalize speedup to 1 worker.
    base = next((r for r in results if r["workers"] == 1), None)
    if base is not None and base["mean_s"] > 0:
        for r in results:
            r["speedup"] = float(base["mean_s"] / r["mean_s"])
    else:
        for r in results:
            r["speedup"] = None

    payload = {
        "input": input_path.name,
        "case": input_path.stem.replace(".input", ""),
        "workers": workers,
        "results": results,
        "precond": args.precond,
    }

    json_path = out_dir / "transport_parallel_scaling.json"
    json_path.write_text(json.dumps(payload, indent=2))

    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415

        w = np.array([r["workers"] for r in results], dtype=int)
        mean_s = np.array([r["mean_s"] for r in results], dtype=float)
        speedup = np.array([r.get("speedup", np.nan) for r in results], dtype=float)

        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
        axes[0].plot(w, mean_s, "o-", label="measured")
        axes[0].set_xlabel("workers")
        axes[0].set_ylabel("time (s)")
        axes[0].set_title("Runtime vs workers")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(w, speedup, "o-", label="measured")
        axes[1].plot(w, w, "--", label="ideal")
        axes[1].set_xlabel("workers")
        axes[1].set_ylabel("speedup")
        axes[1].set_title("Speedup vs workers")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(frameon=False)

        fig.suptitle(f"Parallel whichRHS scaling: {payload['case']}", y=1.02)
        fig.tight_layout()
        fig_path = out_dir / "transport_parallel_scaling.png"
        fig.savefig(fig_path, dpi=200)
        print(f"Saved figure -> {fig_path}")
    except Exception as exc:  # noqa: BLE001
        print(f"Matplotlib unavailable: {exc}")


if __name__ == "__main__":
    main()
