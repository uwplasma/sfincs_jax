from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import jax
import numpy as np

from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.v3_driver import solve_v3_full_system_linear_gmres


def _run_once(
    input_path: Path,
    *,
    shard_axis: str,
    gmres_distributed: str,
    distributed_krylov: str,
    periodic_stencil_on_sharded: str,
    nsolve: int,
    rhs1_precond: str,
) -> float:
    os.environ["SFINCS_JAX_FORTRAN_STDOUT"] = "0"
    os.environ["SFINCS_JAX_SOLVER_ITER_STATS"] = "0"
    os.environ["SFINCS_JAX_MATVEC_SHARD_AXIS"] = shard_axis
    os.environ["SFINCS_JAX_GMRES_DISTRIBUTED"] = gmres_distributed
    os.environ["SFINCS_JAX_DISTRIBUTED_KRYLOV"] = distributed_krylov
    os.environ["SFINCS_JAX_AUTO_SHARD"] = "0"
    os.environ["SFINCS_JAX_IMPLICIT_SOLVE"] = "0"
    os.environ["SFINCS_JAX_SHARD_PAD"] = "1"
    if rhs1_precond:
        os.environ["SFINCS_JAX_RHSMODE1_PRECONDITIONER"] = rhs1_precond
    else:
        os.environ.pop("SFINCS_JAX_RHSMODE1_PRECONDITIONER", None)
    if periodic_stencil_on_sharded == "off":
        os.environ["SFINCS_JAX_PERIODIC_STENCIL_ON_SHARDED"] = "0"
    elif periodic_stencil_on_sharded == "on":
        os.environ["SFINCS_JAX_PERIODIC_STENCIL_ON_SHARDED"] = "1"
    else:
        os.environ.pop("SFINCS_JAX_PERIODIC_STENCIL_ON_SHARDED", None)
    nml = read_sfincs_input(input_path)
    t0 = time.perf_counter()
    for _ in range(max(1, int(nsolve))):
        res = solve_v3_full_system_linear_gmres(
            nml=nml,
            tol=1e-10,
        )
        jax.block_until_ready(res.x)
    return time.perf_counter() - t0


def _run_once_subprocess(
    *,
    input_path: Path,
    devices: int,
    cache_dir: Path | None,
    shard_axis: str,
    gmres_distributed: str,
    distributed_krylov: str,
    periodic_stencil_on_sharded: str,
    nsolve: int,
    rhs1_precond: str,
) -> float:
    env = os.environ.copy()
    env["SFINCS_JAX_CPU_DEVICES"] = str(int(devices))
    env["SFINCS_JAX_MATVEC_SHARD_AXIS"] = shard_axis
    env["SFINCS_JAX_GMRES_DISTRIBUTED"] = gmres_distributed
    env["SFINCS_JAX_DISTRIBUTED_KRYLOV"] = distributed_krylov
    env["SFINCS_JAX_AUTO_SHARD"] = "0"
    env["SFINCS_JAX_IMPLICIT_SOLVE"] = "0"
    env["SFINCS_JAX_SHARD_PAD"] = "1"
    env["SFINCS_JAX_FORTRAN_STDOUT"] = "0"
    env["SFINCS_JAX_SOLVER_ITER_STATS"] = "0"
    if rhs1_precond:
        env["SFINCS_JAX_RHSMODE1_PRECONDITIONER"] = rhs1_precond
    else:
        env.pop("SFINCS_JAX_RHSMODE1_PRECONDITIONER", None)
    if periodic_stencil_on_sharded == "off":
        env["SFINCS_JAX_PERIODIC_STENCIL_ON_SHARDED"] = "0"
    elif periodic_stencil_on_sharded == "on":
        env["SFINCS_JAX_PERIODIC_STENCIL_ON_SHARDED"] = "1"
    else:
        env.pop("SFINCS_JAX_PERIODIC_STENCIL_ON_SHARDED", None)
    if cache_dir is not None:
        env["JAX_COMPILATION_CACHE_DIR"] = str(cache_dir)

    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--run-once",
        "--input",
        str(input_path),
        "--nsolve",
        str(int(nsolve)),
    ]
    out = subprocess.check_output(cmd, env=env, text=True)
    return float(out.strip().splitlines()[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark sharded RHSMode=1 solve scaling.")
    repo_root = Path(__file__).resolve().parents[2]
    default_input = repo_root / "examples" / "performance" / "rhsmode1_sharded.input.namelist"
    default_out = repo_root / "examples" / "performance" / "output" / "sharded_solve_scaling"
    default_cache = default_out / "jax_cache"

    parser.add_argument(
        "--input",
        type=Path,
        default=default_input,
        help="Path to RHSMode=1 input.namelist for sharded solve benchmarking.",
    )
    parser.add_argument(
        "--devices",
        type=int,
        nargs="+",
        default=list(range(1, 5)),
        help="CPU device counts to benchmark (default 1..4).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Warmup runs per device count.",
    )
    parser.add_argument("--repeats", type=int, default=1, help="Timing repeats per device count.")
    parser.add_argument(
        "--nsolve",
        type=int,
        default=1,
        help="Number of RHSMode=1 solves per timed sample.",
    )
    parser.add_argument(
        "--global-warmup",
        type=int,
        default=1,
        help="Warmup runs before benchmarking (uses devices=1).",
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
    parser.add_argument(
        "--run-once",
        action="store_true",
        help="Run once and print wall time (internal).",
    )
    parser.add_argument(
        "--shard-axis",
        type=str,
        default="theta",
        help="Matvec shard axis for distributed solve benchmark (theta/zeta/x/flat/off).",
    )
    parser.add_argument(
        "--gmres-distributed",
        type=str,
        default="1",
        help="Value for SFINCS_JAX_GMRES_DISTRIBUTED (default: 1).",
    )
    parser.add_argument(
        "--distributed-krylov",
        type=str,
        default="auto",
        help="Value for SFINCS_JAX_DISTRIBUTED_KRYLOV (default: auto).",
    )
    parser.add_argument(
        "--periodic-stencil-on-sharded",
        type=str,
        default="auto",
        choices=("auto", "on", "off"),
        help="Control SFINCS_JAX_PERIODIC_STENCIL_ON_SHARDED for this benchmark.",
    )
    parser.add_argument(
        "--rhs1-precond",
        type=str,
        default="",
        help="Optional SFINCS_JAX_RHSMODE1_PRECONDITIONER override for this benchmark.",
    )
    args = parser.parse_args()

    input_path = args.input
    if not input_path.exists():
        raise FileNotFoundError(str(input_path))

    if args.run_once:
        dt = _run_once(
            input_path,
            shard_axis=str(args.shard_axis),
            gmres_distributed=str(args.gmres_distributed),
            distributed_krylov=str(args.distributed_krylov),
            periodic_stencil_on_sharded=str(args.periodic_stencil_on_sharded),
            nsolve=int(args.nsolve),
            rhs1_precond=str(args.rhs1_precond),
        )
        print(f"{dt:.6f}")
        return

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    devices = sorted({int(d) for d in args.devices if int(d) >= 1})

    if args.global_warmup and args.global_warmup > 0:
        for _ in range(int(args.global_warmup)):
            _run_once_subprocess(
                input_path=input_path,
                devices=1,
                cache_dir=cache_dir,
                shard_axis=str(args.shard_axis),
                gmres_distributed=str(args.gmres_distributed),
                distributed_krylov=str(args.distributed_krylov),
                periodic_stencil_on_sharded=str(args.periodic_stencil_on_sharded),
                nsolve=int(args.nsolve),
                rhs1_precond=str(args.rhs1_precond),
            )

    results = []
    for d in devices:
        print(
            f"[device {d}] warmups={max(args.warmup, 0)} repeats={max(args.repeats, 1)} "
            f"shard_axis={args.shard_axis} gmres_distributed={args.gmres_distributed} "
            f"distributed_krylov={args.distributed_krylov} "
            f"stencil_on_sharded={args.periodic_stencil_on_sharded} "
            f"nsolve={int(args.nsolve)} rhs1_precond={args.rhs1_precond or 'auto'}",
            flush=True,
        )
        for i in range(max(args.warmup, 0)):
            print(f"[device {d}] warmup {i + 1}/{max(args.warmup, 0)} starting", flush=True)
            _run_once_subprocess(
                input_path=input_path,
                devices=d,
                cache_dir=cache_dir,
                shard_axis=str(args.shard_axis),
                gmres_distributed=str(args.gmres_distributed),
                distributed_krylov=str(args.distributed_krylov),
                periodic_stencil_on_sharded=str(args.periodic_stencil_on_sharded),
                nsolve=int(args.nsolve),
                rhs1_precond=str(args.rhs1_precond),
            )
            print(f"[device {d}] warmup {i + 1}/{max(args.warmup, 0)} done", flush=True)
        times = []
        for i in range(max(args.repeats, 1)):
            print(f"[device {d}] repeat {i + 1}/{max(args.repeats, 1)} starting", flush=True)
            dt = _run_once_subprocess(
                input_path=input_path,
                devices=d,
                cache_dir=cache_dir,
                shard_axis=str(args.shard_axis),
                gmres_distributed=str(args.gmres_distributed),
                distributed_krylov=str(args.distributed_krylov),
                periodic_stencil_on_sharded=str(args.periodic_stencil_on_sharded),
                nsolve=int(args.nsolve),
                rhs1_precond=str(args.rhs1_precond),
            )
            times.append(dt)
            print(f"[device {d}] repeat {i + 1}/{max(args.repeats, 1)} done in {dt:.3f}s", flush=True)
        times = np.asarray(times, dtype=float)
        results.append(
            {
                "devices": d,
                "mean_s": float(np.mean(times)),
                "std_s": float(np.std(times, ddof=1)) if times.size > 1 else 0.0,
                "samples": [float(v) for v in times],
            }
        )
        print(f"devices={d} mean_s={results[-1]['mean_s']:.3f} std_s={results[-1]['std_s']:.3f}", flush=True)

    base = next((r for r in results if r["devices"] == 1), None)
    if base is not None and base["mean_s"] > 0:
        for r in results:
            r["speedup"] = float(base["mean_s"] / r["mean_s"])
    else:
        for r in results:
            r["speedup"] = None

    payload = {
        "input": input_path.name,
        "case": input_path.stem.replace(".input", ""),
        "devices": devices,
        "results": results,
        "shard_axis": str(args.shard_axis),
        "gmres_distributed": str(args.gmres_distributed),
        "distributed_krylov": str(args.distributed_krylov),
        "periodic_stencil_on_sharded": str(args.periodic_stencil_on_sharded),
        "nsolve": int(args.nsolve),
        "rhs1_precond": str(args.rhs1_precond),
    }

    json_path = out_dir / "sharded_solve_scaling.json"
    json_path.write_text(json.dumps(payload, indent=2))

    try:
        import matplotlib  # noqa: PLC0415

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt  # noqa: PLC0415

        d = np.array([r["devices"] for r in results], dtype=int)
        mean_s = np.array([r["mean_s"] for r in results], dtype=float)
        speedup = np.array([r.get("speedup", np.nan) for r in results], dtype=float)

        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
        axes[0].plot(d, mean_s, "o-", label="measured")
        axes[0].set_xlabel("CPU devices")
        axes[0].set_ylabel("time (s)")
        axes[0].set_title("Runtime vs devices")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(d, speedup, "o-", label="measured")
        axes[1].plot(d, d, "--", label="ideal")
        axes[1].set_xlabel("CPU devices")
        axes[1].set_ylabel("speedup")
        axes[1].set_title("Speedup vs devices")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(frameon=False)

        fig.suptitle(f"Sharded solve scaling: {payload['case']}", y=1.02)
        fig.tight_layout()
        fig_path = out_dir / "sharded_solve_scaling.png"
        fig.savefig(fig_path, dpi=200)
        print(f"Saved figure -> {fig_path}")
    except Exception as exc:  # noqa: BLE001
        print(f"Matplotlib unavailable: {exc}")


if __name__ == "__main__":
    main()
