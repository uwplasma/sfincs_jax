#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import shutil
import tempfile
import time
from pathlib import Path

import jax
from jax import profiler as jax_profiler

from sfincs_jax.io import localize_equilibrium_file_in_place
from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.v3_driver import solve_v3_transport_matrix_linear_gmres


def _prepare_input(input_path: Path) -> Path:
    tmpdir = Path(tempfile.mkdtemp(prefix="sfincs_jax_transport_profile_"))
    dst = tmpdir / "input.namelist"
    shutil.copy2(input_path, dst)
    localize_equilibrium_file_in_place(input_namelist=dst, overwrite=False)
    return dst


def _run_transport(nml_path: Path, *, tol: float, warmup: int, repeats: int) -> None:
    nml = read_sfincs_input(nml_path)
    for _ in range(max(0, int(warmup))):
        res = solve_v3_transport_matrix_linear_gmres(nml=nml, tol=float(tol))
        _ = jax.block_until_ready(res.transport_matrix)
    for _ in range(max(1, int(repeats))):
        res = solve_v3_transport_matrix_linear_gmres(nml=nml, tol=float(tol))
        _ = jax.block_until_ready(res.transport_matrix)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["transportMatrix_geometryScheme11"],
        help="Reduced-suite case names to profile (default: transportMatrix_geometryScheme11).",
    )
    parser.add_argument(
        "--reduced-inputs",
        type=Path,
        default=Path("tests") / "reduced_inputs",
        help="Directory containing reduced input.namelist fixtures.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("examples") / "performance" / "output" / "jax_traces",
        help="Directory to write JAX profiler traces.",
    )
    parser.add_argument("--tol", type=float, default=1e-10, help="Transport solve tolerance.")
    parser.add_argument("--warmup", type=int, default=1, help="Number of warmup solves before tracing.")
    parser.add_argument("--repeats", type=int, default=1, help="Number of traced solves to capture.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    reduced_inputs = (repo_root / args.reduced_inputs).resolve()
    out_dir = (repo_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("JAX_ENABLE_X64", "True")
    os.environ.setdefault("JAX_ENABLE_COMPILATION_CACHE", "1")
    os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")
    os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "0")

    for case in args.cases:
        input_path = reduced_inputs / f"{case}.input.namelist"
        if not input_path.exists():
            raise FileNotFoundError(f"Missing reduced input for case '{case}': {input_path}")
        work_input = _prepare_input(input_path)
        trace_dir = out_dir / case
        trace_dir.mkdir(parents=True, exist_ok=True)

        # Warmup outside trace to avoid compile noise.
        _run_transport(work_input, tol=float(args.tol), warmup=int(args.warmup), repeats=0)

        jax_profiler.start_trace(str(trace_dir))
        t0 = time.perf_counter()
        _run_transport(work_input, tol=float(args.tol), warmup=0, repeats=int(args.repeats))
        jax_profiler.stop_trace()
        elapsed = time.perf_counter() - t0

        print(f"Wrote JAX trace for {case} -> {trace_dir} (elapsed {elapsed:.3f}s)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
