#!/usr/bin/env python
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_JAX_CACHE_DIR = REPO_ROOT / "tests" / "reduced_upstream_examples" / ".jax_compilation_cache"


def _repo_rel(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except Exception:  # noqa: BLE001
        return str(path)

def _ensure_jax_compilation_cache() -> None:
    disable_env = os.environ.get("SFINCS_JAX_DISABLE_COMPILATION_CACHE", "").strip().lower()
    if disable_env in {"1", "true", "yes", "on"}:
        return
    if not os.environ.get("JAX_COMPILATION_CACHE_DIR", "").strip():
        os.environ["JAX_COMPILATION_CACHE_DIR"] = str(_DEFAULT_JAX_CACHE_DIR)
    os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")
    os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "0")


_ensure_jax_compilation_cache()

from sfincs_jax.compare import compare_sfincs_outputs
from sfincs_jax.io import localize_equilibrium_file_in_place
from sfincs_jax.namelist import read_sfincs_input

RES_KEYS: tuple[str, ...] = ("NTHETA", "NZETA", "NX", "NXI")
MIN_RES: dict[str, int] = {"NTHETA": 5, "NZETA": 1, "NX": 1, "NXI": 2}


def _detect_mpi_vendor_for_exe(exe: Path) -> str | None:
    try:
        if sys.platform == "darwin":
            out = subprocess.check_output(["otool", "-L", str(exe)], text=True, stderr=subprocess.STDOUT)
        else:
            out = subprocess.check_output(["ldd", str(exe)], text=True, stderr=subprocess.STDOUT)
    except Exception:  # noqa: BLE001
        return None
    low = out.lower()
    if "mpich" in low:
        return "mpich"
    if "openmpi" in low or "open mpi" in low:
        return "openmpi"
    return None


def _detect_mpiexec_vendor(path: str) -> str | None:
    try:
        out = subprocess.check_output([path, "--version"], text=True, stderr=subprocess.STDOUT, timeout=2.0)
    except Exception:  # noqa: BLE001
        return None
    low = out.lower()
    if "open mpi" in low or "openmpi" in low:
        return "openmpi"
    if "mpich" in low or "hydra" in low:
        return "mpich"
    return None


@dataclass
class CaseResult:
    case: str
    status: str
    blocker_type: str
    message: str
    attempts: int
    reductions: int
    fortran_runtime_s: float | None
    jax_runtime_s: float | None
    jax_runtime_s_cold: float | None
    jax_runtime_s_warm: float | None
    jax_solver_iters_mean: float | None
    jax_solver_iters_min: int | None
    jax_solver_iters_max: int | None
    jax_solver_iters_n: int
    jax_solver_iters_detail: list[int]
    jax_solver_kinds: list[str]
    print_parity_signals: int
    print_parity_total: int
    print_missing_signals: list[str]
    n_common_keys: int
    n_mismatch_common: int
    mismatch_keys_sample: list[str]
    n_mismatch_solver: int
    n_mismatch_physics: int
    mismatch_solver_sample: list[str]
    mismatch_physics_sample: list[str]
    max_abs_mismatch: float | None
    strict_n_common_keys: int
    strict_n_mismatch_common: int
    strict_mismatch_keys_sample: list[str]
    strict_n_mismatch_solver: int
    strict_n_mismatch_physics: int
    strict_mismatch_solver_sample: list[str]
    strict_mismatch_physics_sample: list[str]
    strict_max_abs_mismatch: float | None
    final_resolution: dict[str, int]
    input_path: str
    promoted_input_path: str | None
    fortran_h5: str | None
    jax_h5: str | None


PRINT_SIGNALS: dict[str, tuple[str, str]] = {
    # These patterns are intentionally aligned with upstream v3 stdout so we can
    # track how close `sfincs_jax` is to being a drop-in replacement.
    "input_namelist": (r"input\.namelist|Successfully read parameters", r"input\.namelist|Successfully read parameters"),
    "geometry_summary": (r"Geometry scheme|Geometry parameters", r"Geometry scheme|Geometry parameters"),
    "resolution_summary": (r"Ntheta|Nzeta|Nxi|Nx", r"Ntheta|Nzeta|Nxi|Nx"),
    "x_grid": (r"\bx:\s", r"\bx:\s"),
    "residual": (r"Residual function norm|evaluateResidual called", r"Residual function norm|evaluateResidual called|residual_norm"),
    "jacobian": (r"evaluateJacobian called|populateMatrix", r"evaluateJacobian called|populateMatrix|whichMatrix"),
    "diagnostics": (r"Computing diagnostics|Results for species", r"Computing diagnostics|Results for species"),
    "output_write": (r"Saving diagnostics to h5 file|sfincsOutput\.h5", r"Saving diagnostics to h5 file|sfincsOutput\.h5"),
    "runtime": (r"Time to solve|seconds", r"Time to solve|seconds|elapsed_s="),
}

GEOMETRY_MISMATCH_HINTS = (
    "bhat",
    "dBHat",
    "ghat",
    "ihat",
    "iota",
    "sqrt_g",
    "gpsihat",
    "geometry",
    "gradpar",
)

SOLVER_MISMATCH_HINTS = (
    "niterations",
    "residual",
    "jacobian",
    "whichmatrix",
    "statevector",
    "transportmatrix",
    "flow",
    "fsa",
)

PHYSICS_MISMATCH_HINTS = (
    "bhat",
    "dbhat",
    "ghat",
    "ihat",
    "iota",
    "uhat",
    "gpsi",
    "sqrtg",
    "dps",
    "jacobianhat",
    "geometry",
)


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


def _run_jax_cli(
    *,
    input_path: Path,
    output_path: Path,
    timeout_s: float,
    log_path: Path,
    compute_solution: bool,
    compute_transport_matrix: bool,
    collect_iterations: bool = True,
    repeats: int = 1,
    cache_dir: Path | None = None,
) -> tuple[float, float | None]:
    cmd = [
        "python",
        "-m",
        "sfincs_jax",
        "-v",
        "write-output",
        "--input",
        str(input_path),
        "--out",
        str(output_path),
    ]
    if compute_solution:
        cmd.append("--compute-solution")
    if compute_transport_matrix:
        cmd.append("--compute-transport-matrix")
    env = dict(os.environ)
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        env.setdefault("JAX_COMPILATION_CACHE_DIR", str(cache_dir))
        env.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")
        env.setdefault("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "0")
    env["SFINCS_JAX_SOLVER_ITER_STATS"] = "1" if collect_iterations else "0"
    if collect_iterations:
        env.setdefault("SFINCS_JAX_SOLVER_ITER_STATS_MAX_SIZE", "800")
    env.setdefault(
        "SFINCS_JAX_KSP_HISTORY_MAX_SIZE",
        env.get("SFINCS_JAX_SOLVER_ITER_STATS_MAX_SIZE", "800"),
    )
    run_times: list[float] = []
    repeat_count = max(1, int(repeats))
    for idx in range(repeat_count):
        t0 = time.perf_counter()
        mode = "w" if idx == 0 else "a"
        with log_path.open(mode, encoding="utf-8") as log:
            if idx > 0:
                log.write(f"\n--- sfincs_jax repeat {idx + 1}/{repeat_count} ---\n")
            subprocess.run(
                cmd,
                cwd=str(REPO_ROOT),
                check=True,
                timeout=timeout_s,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
            )
        if not output_path.exists():
            tail = _tail(log_path, n=40)
            raise RuntimeError(f"JAX run returned success but did not create output: {output_path}\n{tail}")
        run_times.append(time.perf_counter() - t0)
    cold = float(run_times[0])
    warm = None
    if len(run_times) > 1:
        warm = float(np.mean(np.asarray(run_times[1:], dtype=np.float64)))
    return cold, warm


def _run_fortran_direct(*, input_path: Path, exe: Path, timeout_s: float, log_path: Path) -> tuple[float, Path, int]:
    cmd = [str(exe.resolve())]
    t0 = time.perf_counter()
    env = dict(os.environ)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            cmd,
            cwd=str(input_path.parent),
            check=False,
            timeout=timeout_s,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
        )
    dt = time.perf_counter() - t0
    out = input_path.parent / "sfincsOutput.h5"
    if proc.returncode != 0:
        tail = _tail(log_path, n=80)
        lower_tail = tail.lower()
        if "attempting to use an mpi routine" in lower_tail and "mpich" in lower_tail:
            fallback_opts = os.environ.get("SFINCS_JAX_FORTRAN_PETSC_OPTIONS_FALLBACK", "").strip()
            if not fallback_opts:
                fallback_opts = "-pc_type lu -pc_factor_mat_solver_type petsc -ksp_type preonly"
            retry_env = dict(env)
            retry_env["PETSC_OPTIONS"] = fallback_opts
            retry_log = log_path.with_suffix(".petsc_fallback.log")
            t1 = time.perf_counter()
            with retry_log.open("w", encoding="utf-8") as log:
                retry_proc = subprocess.run(
                    cmd,
                    cwd=str(input_path.parent),
                    check=False,
                    timeout=timeout_s,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=retry_env,
                )
            dt_retry = time.perf_counter() - t1
            out_retry = input_path.parent / "sfincsOutput.h5"
            if retry_proc.returncode == 0 and out_retry.exists():
                return dt_retry, out_retry, int(retry_proc.returncode)
            if out.exists() and retry_proc.returncode != 0:
                return dt_retry, out_retry if out_retry.exists() else out, int(retry_proc.returncode)
            return dt_retry, out_retry, int(retry_proc.returncode)
        # Some MPI-enabled builds error out on libfabric defaults. Retry once with a TCP provider.
        mpi_hint = any(s in lower_tail for s in ("mpi_init", "ofi call", "libfabric", "mpidi_ofi"))
        if mpi_hint:
            mpi_vendor = _detect_mpi_vendor_for_exe(exe)
            # In restricted/sandboxed environments, socket-based providers can fail
            # even for single-process jobs. Prefer a shared-memory provider first and
            # try a few common MPI/OFI fallbacks used on CI runners.
            retry_variants = [
                {"FI_PROVIDER": "shm"},
                {"FI_PROVIDER": "tcp"},
                {"FI_PROVIDER": "shm", "MPICH_OFI_PROVIDER": "shm"},
                {"FI_PROVIDER": "tcp", "MPICH_OFI_PROVIDER": "tcp"},
                {"I_MPI_FABRICS": "shm:tcp", "FI_PROVIDER": "tcp"},
                {"I_MPI_OFI_PROVIDER": "shm", "FI_PROVIDER": "shm"},
                {"MPIR_CVAR_CH4_OFI_ENABLE": "0"},
                {
                    "PMIX_MCA_ptl": "usock",
                    "PRTE_MCA_oob": "usock",
                    "OMPI_MCA_oob": "usock",
                },
                {
                    "PMIX_MCA_ptl": "usock",
                    "PMIX_MCA_ptl_tcp_if_include": "lo0",
                    "OMPI_MCA_oob_tcp_if_include": "lo0",
                    "PRTE_MCA_oob_tcp_if_include": "lo0",
                },
                {
                    "OMPI_MCA_btl": "self,vader",
                    "OMPI_MCA_btl_tcp_if_include": "lo0",
                },
                {
                    "PMIX_MCA_ptl": "usock",
                    "PMIX_MCA_gds": "hash",
                    "PMIX_MCA_psec": "none",
                    "PRTE_MCA_oob": "usock",
                    "OMPI_MCA_oob": "usock",
                },
                {
                    "OMPI_MCA_pml": "ob1",
                    "OMPI_MCA_btl": "self",
                    "OMPI_MCA_oob": "usock",
                    "PRTE_MCA_oob": "usock",
                    "PMIX_MCA_ptl": "usock",
                },
            ]
            for variant in retry_variants:
                env_retry = dict(env)
                env_retry.update(variant)
                env_retry.setdefault("FI_MR_CACHE_MAX_COUNT", "0")
                # Try to avoid touching non-loopback NICs on macOS runners.
                env_retry.setdefault("MPICH_OFI_INTERFACE_NAME", "lo0")
                env_retry.setdefault("FI_TCP_IFACE", "lo0")
                with log_path.open("w", encoding="utf-8") as log:
                    proc = subprocess.run(
                        cmd,
                        cwd=str(input_path.parent),
                        check=False,
                        timeout=timeout_s,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        env=env_retry,
                    )
                dt = time.perf_counter() - t0
                if proc.returncode == 0 and out.exists():
                    return dt, out, int(proc.returncode)
                tail = _tail(log_path, n=80)
            mpi_exec = None
            mpi_exec_vendor = None
            mpiexec_candidates = (
                "mpiexec.mpich",
                "mpiexec.hydra",
                "mpiexec-mpich-clang16",
                "mpiexec-mpich-gcc13",
                "mpiexec.hydra-mpich-clang16",
                "mpiexec.hydra-mpich-gcc13",
                "mpirun-mpich-clang16",
                "mpirun-mpich-gcc13",
                "mpiexec",
                "mpirun",
            )
            for candidate in mpiexec_candidates:
                candidate_path = shutil.which(candidate)
                if not candidate_path:
                    continue
                vendor = _detect_mpiexec_vendor(candidate_path)
                if mpi_vendor is None or vendor is None:
                    mpi_exec = candidate_path
                    mpi_exec_vendor = vendor
                    break
                if mpi_vendor == vendor:
                    mpi_exec = candidate_path
                    mpi_exec_vendor = vendor
                    break
            if mpi_exec is not None:
                if mpi_exec_vendor == "mpich":
                    cmd_mpi = [mpi_exec, "-launcher", "fork", "-iface", "lo0", "-n", "1", str(exe.resolve())]
                else:
                    cmd_mpi = [mpi_exec, "-n", "1", str(exe.resolve())]
                for variant in retry_variants:
                    env_retry = dict(env)
                    env_retry.update(variant)
                    env_retry.setdefault("FI_MR_CACHE_MAX_COUNT", "0")
                    env_retry.setdefault("MPICH_OFI_INTERFACE_NAME", "lo0")
                    env_retry.setdefault("FI_TCP_IFACE", "lo0")
                    env_retry.setdefault("OMPI_MCA_oob_tcp_if_include", "lo0")
                    env_retry.setdefault("PRTE_MCA_oob_tcp_if_include", "lo0")
                    env_retry.setdefault("HYDRA_IFACE", "lo0")
                    env_retry.setdefault("HYDRA_USE_LOCALHOST", "1")
                    with log_path.open("w", encoding="utf-8") as log:
                        proc = subprocess.run(
                            cmd_mpi,
                            cwd=str(input_path.parent),
                            check=False,
                            timeout=timeout_s,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            env=env_retry,
                        )
                    dt = time.perf_counter() - t0
                    if proc.returncode == 0 and out.exists():
                        return dt, out, int(proc.returncode)
                    tail = _tail(log_path, n=80)
        raise RuntimeError(f"Fortran failed rc={proc.returncode}.\n{tail}")
    if not out.exists():
        tail = _tail(log_path, n=40)
        raise RuntimeError(f"Fortran did not produce output.\n{tail}")
    return dt, out, int(proc.returncode)


def _compare_outputs(
    fortran_h5: Path,
    jax_h5: Path,
    *,
    rtol: float,
    atol: float,
    tolerances: dict[str, dict[str, float]] | None = None,
) -> tuple[int, int, float | None, list[str]]:
    results = compare_sfincs_outputs(a_path=jax_h5, b_path=fortran_h5, rtol=rtol, atol=atol, tolerances=tolerances)
    bad = [r for r in results if not r.ok]
    max_abs = max((r.max_abs for r in bad), default=None)
    bad_keys = sorted(r.key for r in bad)
    return len(results), len(bad), max_abs, bad_keys[:12]


def _compute_print_parity(*, fortran_log: Path, jax_log: Path) -> tuple[int, int, list[str]]:
    fortran_text = fortran_log.read_text(encoding="utf-8", errors="replace").lower() if fortran_log.exists() else ""
    jax_text = jax_log.read_text(encoding="utf-8", errors="replace").lower() if jax_log.exists() else ""

    matched = 0
    relevant = 0
    missing: list[str] = []
    for signal, (fortran_pat, jax_pat) in PRINT_SIGNALS.items():
        seen_fortran = bool(re.search(fortran_pat.lower(), fortran_text, flags=re.IGNORECASE | re.MULTILINE))
        seen_jax = bool(re.search(jax_pat.lower(), jax_text, flags=re.IGNORECASE | re.MULTILINE))
        if seen_fortran:
            relevant += 1
            if seen_jax:
                matched += 1
            else:
                missing.append(signal)
    return matched, relevant, missing


_KSP_ITER_RE = re.compile(r"ksp_iterations=(\d+)\s+solver=([a-zA-Z0-9_]+)")
_KSP_ITER_RHS_RE = re.compile(r"whichRHS=(\d+).*ksp_iterations=(\d+)\s+solver=([a-zA-Z0-9_]+)")


def _parse_ksp_iterations(log_path: Path) -> tuple[list[int], list[str]]:
    if not log_path.exists():
        return [], []
    text = log_path.read_text(encoding="utf-8", errors="replace")
    iter_by_rhs: dict[int, int] = {}
    kind_by_rhs: dict[int, str] = {}
    iter_general: list[int] = []
    kind_general: list[str] = []
    for line in text.splitlines():
        rhs_match = _KSP_ITER_RHS_RE.search(line)
        if rhs_match:
            rhs_idx = int(rhs_match.group(1))
            iter_by_rhs[rhs_idx] = int(rhs_match.group(2))
            kind_by_rhs[rhs_idx] = rhs_match.group(3).lower()
            continue
        match = _KSP_ITER_RE.search(line)
        if match:
            iter_general.append(int(match.group(1)))
            kind_general.append(match.group(2).lower())
    if iter_by_rhs:
        iters = [iter_by_rhs[k] for k in sorted(iter_by_rhs)]
        kinds = [kind_by_rhs[k] for k in sorted(iter_by_rhs)]
        return iters, kinds
    return iter_general, kind_general


def _classify_blocker(*, status: str, note: str, mismatch_keys: list[str], jax_log: Path | None) -> str:
    if status == "parity_ok":
        return "none"

    text_parts = [status, note]
    if jax_log is not None and jax_log.exists():
        text_parts.append(_tail(jax_log, n=80))
    text = "\n".join(text_parts).lower()

    if status in {"fortran_timeout", "jax_timeout", "max_attempts"}:
        return "solver branch mismatch"
    if status.startswith("fortran_"):
        return "unsupported physics/path"
    if status in {"parity_mismatch", "compare_error"}:
        lowered_keys = [k.lower() for k in mismatch_keys]
        if any(any(h in k for h in GEOMETRY_MISMATCH_HINTS) for k in lowered_keys):
            return "geometry parsing mismatch"
        if any(any(h in k for h in SOLVER_MISMATCH_HINTS) for k in lowered_keys):
            return "solver branch mismatch"
        return "output field mismatch"

    if "notimplemented" in text or "unsupported" in text or "todo" in text:
        return "unsupported physics/path"
    if "equilibrium" in text or "geometryscheme" in text or ".bc" in text or ".nc" in text or "netcdf" in text:
        return "geometry parsing mismatch"
    if "whichmatrix" in text or "rhsmode" in text or "transportmatrix" in text or "residual" in text or "jacobian" in text:
        return "solver branch mismatch"
    return "unsupported physics/path"


def _bucket_mismatch_keys(mismatch_keys: list[str]) -> tuple[list[str], list[str]]:
    """Split mismatches into solver-sensitive and physics-sensitive families."""
    solver: list[str] = []
    physics: list[str] = []
    for key in mismatch_keys:
        lk = key.lower()
        if any(h in lk for h in PHYSICS_MISMATCH_HINTS):
            physics.append(key)
        else:
            solver.append(key)
    return solver, physics


def _load_existing_results(report_json: Path) -> dict[str, CaseResult]:
    if not report_json.exists():
        return {}
    raw = json.loads(report_json.read_text(encoding="utf-8"))
    out: dict[str, CaseResult] = {}
    for item in raw:
        out[str(item["case"])] = CaseResult(
            case=str(item["case"]),
            status=str(item["status"]),
            blocker_type=str(item.get("blocker_type", "unsupported physics/path")),
            message=str(item.get("message", "")),
            attempts=int(item.get("attempts", 0)),
            reductions=int(item.get("reductions", 0)),
            fortran_runtime_s=item.get("fortran_runtime_s"),
            jax_runtime_s=item.get("jax_runtime_s"),
            jax_runtime_s_cold=item.get("jax_runtime_s_cold"),
            jax_runtime_s_warm=item.get("jax_runtime_s_warm"),
            jax_solver_iters_mean=item.get("jax_solver_iters_mean"),
            jax_solver_iters_min=item.get("jax_solver_iters_min"),
            jax_solver_iters_max=item.get("jax_solver_iters_max"),
            jax_solver_iters_n=int(item.get("jax_solver_iters_n", 0)),
            jax_solver_iters_detail=list(item.get("jax_solver_iters_detail", [])),
            jax_solver_kinds=list(item.get("jax_solver_kinds", [])),
            print_parity_signals=int(item.get("print_parity_signals", 0)),
            print_parity_total=int(item.get("print_parity_total", 0)),
            print_missing_signals=list(item.get("print_missing_signals", [])),
            n_common_keys=int(item.get("n_common_keys", 0)),
            n_mismatch_common=int(item.get("n_mismatch_common", 0)),
            mismatch_keys_sample=list(item.get("mismatch_keys_sample", [])),
            n_mismatch_solver=int(item.get("n_mismatch_solver", 0)),
            n_mismatch_physics=int(item.get("n_mismatch_physics", 0)),
            mismatch_solver_sample=list(item.get("mismatch_solver_sample", [])),
            mismatch_physics_sample=list(item.get("mismatch_physics_sample", [])),
            max_abs_mismatch=item.get("max_abs_mismatch"),
            strict_n_common_keys=int(item.get("strict_n_common_keys", int(item.get("n_common_keys", 0)))),
            strict_n_mismatch_common=int(item.get("strict_n_mismatch_common", int(item.get("n_mismatch_common", 0)))),
            strict_mismatch_keys_sample=list(item.get("strict_mismatch_keys_sample", item.get("mismatch_keys_sample", []))),
            strict_n_mismatch_solver=int(item.get("strict_n_mismatch_solver", int(item.get("n_mismatch_solver", 0)))),
            strict_n_mismatch_physics=int(item.get("strict_n_mismatch_physics", int(item.get("n_mismatch_physics", 0)))),
            strict_mismatch_solver_sample=list(
                item.get("strict_mismatch_solver_sample", item.get("mismatch_solver_sample", []))
            ),
            strict_mismatch_physics_sample=list(
                item.get("strict_mismatch_physics_sample", item.get("mismatch_physics_sample", []))
            ),
            strict_max_abs_mismatch=item.get("strict_max_abs_mismatch", item.get("max_abs_mismatch")),
            final_resolution={k: int(v) for k, v in dict(item.get("final_resolution", {})).items()},
            input_path=str(item.get("input_path", "")),
            promoted_input_path=item.get("promoted_input_path"),
            fortran_h5=item.get("fortran_h5"),
            jax_h5=item.get("jax_h5"),
        )
    return out


def _status_for_mode(row: CaseResult, *, strict: bool) -> str:
    if not strict:
        return row.status
    if row.status in {"parity_ok", "parity_mismatch"}:
        if row.strict_n_common_keys > 0 and row.strict_n_mismatch_common == 0:
            return "parity_ok"
        if row.strict_n_common_keys > 0:
            return "parity_mismatch"
    return row.status


def _write_rst(rows: list[CaseResult], out_path: Path, *, strict: bool) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = len(rows)
    by_status: dict[str, int] = {}
    for row in rows:
        mode_status = _status_for_mode(row, strict=strict)
        by_status[mode_status] = by_status.get(mode_status, 0) + 1
    summary = ", ".join(f"{k}={v}" for k, v in sorted(by_status.items()))
    mode = "strict" if strict else "practical"

    lines: list[str] = []
    lines.append(".. note::\n\n")
    lines.append("   Auto-generated by `scripts/run_reduced_upstream_suite.py`.\n\n")
    lines.append(f"- Comparison mode: **{mode}**\n")
    lines.append(f"- Cases: **{total}**\n")
    lines.append(f"- Status counts: {summary}\n")
    lines.append("- Timeout policy: 30s per Fortran/JAX run attempt, then halve largest axis and retry.\n\n")
    if strict:
        lines.append("- Tolerances: strict mode ignores all `*.compare_tolerances.json` overrides.\n\n")
    else:
        lines.append("- Tolerances: practical mode applies per-case `*.compare_tolerances.json` when present.\n\n")
    lines.append(".. list-table:: Reduced-resolution upstream suite parity status\n")
    lines.append("   :header-rows: 1\n")
    lines.append("   :widths: 23 9 16 10 7 7 10 10 10 11 13 12 16\n\n")
    lines.append("   * - Case\n")
    lines.append("     - Status\n")
    lines.append("     - Blocker\n")
    lines.append("     - Resolution\n")
    lines.append("     - Tries\n")
    lines.append("     - Reductions\n")
    lines.append("     - Fortran(s)\n")
    lines.append("     - JAX(s)\n")
    lines.append("     - JAX iters\n")
    lines.append("     - Mismatches\n")
    lines.append("     - Buckets\n")
    lines.append("     - Print parity\n")
    lines.append("     - Note\n")

    def _single_line(text: str | None) -> str:
        if text is None:
            return "-"
        collapsed = " ".join(str(text).split())
        return collapsed if collapsed else "-"

    for row in rows:
        mode_status = _status_for_mode(row, strict=strict)
        res = ",".join(f"{k}={v}" for k, v in sorted(row.final_resolution.items()))
        ft = "-" if row.fortran_runtime_s is None else f"{row.fortran_runtime_s:.3f}"
        jt = "-" if row.jax_runtime_s is None else f"{row.jax_runtime_s:.3f}"
        n_common = row.strict_n_common_keys if strict else row.n_common_keys
        n_bad = row.strict_n_mismatch_common if strict else row.n_mismatch_common
        n_solver = row.strict_n_mismatch_solver if strict else row.n_mismatch_solver
        n_physics = row.strict_n_mismatch_physics if strict else row.n_mismatch_physics
        mm = f"{n_bad}/{n_common}" if n_common > 0 else "-"
        buckets = f"S:{n_solver} P:{n_physics}"
        pp = f"{row.print_parity_signals}/{row.print_parity_total}" if row.print_parity_total > 0 else "-"
        iters = "-"
        if row.jax_solver_iters_n > 0 and row.jax_solver_iters_mean is not None:
            if row.jax_solver_iters_n == 1:
                iters = f"{int(round(row.jax_solver_iters_mean))}"
            else:
                min_iter = row.jax_solver_iters_min if row.jax_solver_iters_min is not None else 0
                max_iter = row.jax_solver_iters_max if row.jax_solver_iters_max is not None else 0
                iters = f"{row.jax_solver_iters_mean:.1f} ({min_iter}-{max_iter})"
        lines.append(f"   * - {row.case}\n")
        lines.append(f"     - {mode_status}\n")
        lines.append(f"     - {row.blocker_type}\n")
        lines.append(f"     - {res}\n")
        lines.append(f"     - {row.attempts}\n")
        lines.append(f"     - {row.reductions}\n")
        lines.append(f"     - {ft}\n")
        lines.append(f"     - {jt}\n")
        lines.append(f"     - {iters}\n")
        lines.append(f"     - {mm}\n")
        lines.append(f"     - {buckets}\n")
        lines.append(f"     - {pp}\n")
        lines.append(f"     - {_single_line(row.message)}\n")
    out_path.write_text("".join(lines), encoding="utf-8")


def _run_case(
    *,
    case_name: str,
    case_input: Path,
    case_out_dir: Path,
    fortran_exe: Path,
    timeout_s: float,
    rtol: float,
    atol: float,
    max_attempts: int,
    use_seed_resolution: bool = False,
    reuse_fortran: bool = False,
    collect_iterations: bool = True,
    jax_repeats: int = 1,
    jax_cache_dir: Path | None = None,
) -> CaseResult:
    case = str(case_name)
    case_out_dir.mkdir(parents=True, exist_ok=True)
    dst_input = case_out_dir / "input.namelist"
    (case_out_dir / "input.original.namelist").write_text(case_input.read_text())
    if use_seed_resolution:
        dst_input.write_text(case_input.read_text())
    else:
        _write_initial_reduced_input(source_input=case_input, dst_input=dst_input)
    localize_equilibrium_file_in_place(input_namelist=dst_input, overwrite=False)
    nml = read_sfincs_input(dst_input)
    rhs_mode = int(nml.group("general").get("RHSMODE", 1))
    compute_solution = rhs_mode == 1
    compute_transport_matrix = rhs_mode in {2, 3}

    attempts = 0
    reductions = 0
    final_res = _resolution_from_namelist(dst_input)
    fortran_runtime = None
    jax_runtime = None
    jax_runtime_cold = None
    jax_runtime_warm = None
    jax_solver_iters_mean = None
    jax_solver_iters_min = None
    jax_solver_iters_max = None
    jax_solver_iters_n = 0
    jax_solver_iters_detail: list[int] = []
    jax_solver_kinds: list[str] = []
    note = ""
    status = "error"
    blocker_type = "unsupported physics/path"
    fortran_h5_path: Path | None = None
    jax_h5_path: Path | None = None
    fortran_log_path: Path | None = None
    jax_log_path: Path | None = None
    print_signals = 0
    print_total = 0
    print_missing: list[str] = []
    n_common = 0
    n_bad = 0
    max_abs = None
    mismatch_keys: list[str] = []
    mismatch_solver_keys: list[str] = []
    mismatch_physics_keys: list[str] = []
    strict_n_common = 0
    strict_n_bad = 0
    strict_max_abs = None
    strict_mismatch_keys: list[str] = []
    strict_mismatch_solver_keys: list[str] = []
    strict_mismatch_physics_keys: list[str] = []

    while attempts < max_attempts:
        attempts += 1
        final_res = _resolution_from_namelist(dst_input)
        fortran_dir = case_out_dir / "fortran_run"
        fortran_log = fortran_dir / "sfincs.log"
        fortran_h5_this_attempt: Path | None = None
        out_fortran_existing = fortran_dir / "sfincsOutput.h5"
        # Only reuse Fortran outputs when the input resolution has not been reduced.
        # If a reduction happened, rerun Fortran so the outputs match the new input.
        if bool(reuse_fortran) and out_fortran_existing.exists() and attempts == 1 and reductions == 0:
            fortran_input = fortran_dir / "input.namelist"
            if fortran_input.exists() and fortran_input.read_text() == dst_input.read_text():
                fortran_h5_this_attempt = out_fortran_existing
                fortran_log_path = fortran_log if fortran_log.exists() else None
        else:
            if fortran_dir.exists():
                shutil.rmtree(fortran_dir)
            fortran_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(dst_input, fortran_dir / "input.namelist")
            localize_equilibrium_file_in_place(input_namelist=fortran_dir / "input.namelist", overwrite=False)
            fortran_log_path = fortran_log

        try:
            if fortran_h5_this_attempt is None:
                fortran_runtime, out_fortran, fortran_rc = _run_fortran_direct(
                    input_path=fortran_dir / "input.namelist",
                    exe=fortran_exe,
                    timeout_s=timeout_s,
                    log_path=fortran_log,
                )
                fortran_h5_this_attempt = out_fortran
                fortran_text = _tail(fortran_log, n=200).lower()
                if "snes_diverged" in fortran_text or "did not converge" in fortran_text:
                    note = "Fortran diverged in SNES; skipping JAX comparison."
                    status = "fortran_diverged"
                    break
        except subprocess.TimeoutExpired:
            note = "Fortran timeout; reduced largest axis."
            new_res = _reduce_max_axis_in_place(dst_input)
            if new_res == final_res:
                status = "fortran_timeout"
                break
            reductions += 1
            continue
        except Exception as exc:  # noqa: BLE001
            exc_text = str(exc)
            lower = exc_text.lower()
            if any(
                s in lower
                for s in (
                    "mpi_init",
                    "ofi call",
                    "libfabric",
                    "mpidi_ofi",
                    "openmpi",
                    "prte",
                    "pmix",
                    "oob_tcp",
                    "bind() failed",
                    "prterun",
                    "hydra",
                    "hydu_sock_listen",
                    "pmi port",
                )
            ):
                note = f"Fortran MPI init error: {exc_text}"
                status = "fortran_error"
                break
            note = f"Fortran error: {type(exc).__name__}: {exc}"
            new_res = _reduce_max_axis_in_place(dst_input)
            if new_res == final_res:
                status = "fortran_error"
                break
            reductions += 1
            continue

        jax_h5 = case_out_dir / "sfincsOutput_jax.h5"
        jax_log = case_out_dir / "sfincs_jax.log"
        jax_log_path = jax_log
        try:
            jax_runtime_cold, jax_runtime_warm = _run_jax_cli(
                input_path=dst_input,
                output_path=jax_h5,
                timeout_s=timeout_s,
                log_path=jax_log,
                compute_solution=compute_solution,
                compute_transport_matrix=compute_transport_matrix,
                collect_iterations=collect_iterations,
                repeats=jax_repeats,
                cache_dir=jax_cache_dir,
            )
            jax_runtime = jax_runtime_warm if jax_runtime_warm is not None else jax_runtime_cold
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

        if fortran_h5_this_attempt is None or jax_h5_path is None:
            note = "Missing output file after successful run."
            status = "missing_output"
            break
        fortran_h5_path = fortran_h5_this_attempt

        try:
            tolerances = None
            tol_path = case_out_dir / "compare_tolerances.json"
            if not tol_path.exists():
                reduced_tol = REPO_ROOT / "tests" / "reduced_inputs" / f"{case}.compare_tolerances.json"
                if reduced_tol.exists():
                    tol_path = reduced_tol
            if tol_path.exists():
                try:
                    tolerances = json.loads(tol_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    tolerances = None
            n_common, n_bad, max_abs, mismatch_keys = _compare_outputs(
                fortran_h5_path, jax_h5_path, rtol=rtol, atol=atol, tolerances=tolerances
            )
            mismatch_solver_keys, mismatch_physics_keys = _bucket_mismatch_keys(mismatch_keys)
            (
                strict_n_common,
                strict_n_bad,
                strict_max_abs,
                strict_mismatch_keys,
            ) = _compare_outputs(fortran_h5_path, jax_h5_path, rtol=rtol, atol=atol, tolerances=None)
            strict_mismatch_solver_keys, strict_mismatch_physics_keys = _bucket_mismatch_keys(strict_mismatch_keys)
            if n_bad == 0:
                status = "parity_ok"
                note = "All common numeric datasets matched tolerance."
            else:
                status = "parity_mismatch"
                note = (
                    "Common numeric dataset mismatches present. "
                    f"sample={','.join(mismatch_keys[:4])} "
                    f"buckets=solver:{len(mismatch_solver_keys)} physics:{len(mismatch_physics_keys)}"
                )
            if strict_n_common > 0:
                note = f"{note} strict={strict_n_bad}/{strict_n_common}"
        except Exception as exc:  # noqa: BLE001
            status = "compare_error"
            note = f"Compare error: {type(exc).__name__}: {exc}"
        if fortran_log_path is not None and jax_log_path is not None:
            print_signals, print_total, print_missing = _compute_print_parity(fortran_log=fortran_log_path, jax_log=jax_log_path)
            if print_total > 0 and print_signals < print_total:
                note = f"{note} printParity={print_signals}/{print_total} missing={','.join(print_missing[:3])}"
        if jax_log_path is not None and jax_log_path.exists():
            iters, kinds = _parse_ksp_iterations(jax_log_path)
            if iters:
                jax_solver_iters_detail = iters
                jax_solver_kinds = kinds
                jax_solver_iters_n = len(iters)
                jax_solver_iters_mean = float(np.mean(np.asarray(iters, dtype=np.float64)))
                jax_solver_iters_min = int(min(iters))
                jax_solver_iters_max = int(max(iters))
        break

    else:
        status = "max_attempts"
        note = "Reached max attempts while reducing resolution."

    blocker_type = _classify_blocker(status=status, note=note, mismatch_keys=mismatch_keys, jax_log=jax_log_path)

    return CaseResult(
        case=case,
        status=status,
        blocker_type=blocker_type,
        message=note,
        attempts=attempts,
        reductions=reductions,
        fortran_runtime_s=fortran_runtime,
        jax_runtime_s=jax_runtime,
        jax_runtime_s_cold=jax_runtime_cold,
        jax_runtime_s_warm=jax_runtime_warm,
        jax_solver_iters_mean=jax_solver_iters_mean,
        jax_solver_iters_min=jax_solver_iters_min,
        jax_solver_iters_max=jax_solver_iters_max,
        jax_solver_iters_n=jax_solver_iters_n,
        jax_solver_iters_detail=jax_solver_iters_detail,
        jax_solver_kinds=jax_solver_kinds,
        print_parity_signals=print_signals,
        print_parity_total=print_total,
        print_missing_signals=print_missing,
        n_common_keys=n_common,
        n_mismatch_common=n_bad,
        mismatch_keys_sample=mismatch_keys,
        n_mismatch_solver=len(mismatch_solver_keys),
        n_mismatch_physics=len(mismatch_physics_keys),
        mismatch_solver_sample=mismatch_solver_keys[:12],
        mismatch_physics_sample=mismatch_physics_keys[:12],
        max_abs_mismatch=max_abs,
        strict_n_common_keys=strict_n_common,
        strict_n_mismatch_common=strict_n_bad,
        strict_mismatch_keys_sample=strict_mismatch_keys,
        strict_n_mismatch_solver=len(strict_mismatch_solver_keys),
        strict_n_mismatch_physics=len(strict_mismatch_physics_keys),
        strict_mismatch_solver_sample=strict_mismatch_solver_keys[:12],
        strict_mismatch_physics_sample=strict_mismatch_physics_keys[:12],
        strict_max_abs_mismatch=strict_max_abs,
        final_resolution=final_res,
        input_path=_repo_rel(dst_input),
        promoted_input_path=None,
        fortran_h5=_repo_rel(fortran_h5_path),
        jax_h5=_repo_rel(jax_h5_path),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run reduced-resolution upstream suite one case at a time with an adaptive timeout/reduction policy."
        )
    )
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
        default=Path(os.environ.get("SFINCS_FORTRAN_EXE", "sfincs")),
        help="Path to Fortran v3 executable (or set SFINCS_FORTRAN_EXE).",
    )
    parser.add_argument("--pattern", type=str, default=None, help="Regex filter on case directory path.")
    parser.add_argument("--timeout-s", type=float, default=120.0, help="Per-run timeout in seconds.")
    parser.add_argument("--max-attempts", type=int, default=6, help="Maximum adaptive retries per case.")
    parser.add_argument("--rtol", type=float, default=5e-4)
    parser.add_argument("--atol", type=float, default=1e-10)
    parser.add_argument(
        "--reuse-fortran",
        action="store_true",
        help="Reuse an existing per-case fortran_run/sfincsOutput.h5 if present instead of rerunning Fortran.",
    )
    parser.add_argument(
        "--reset-report",
        action="store_true",
        help="Do not merge with existing suite_report.json; overwrite report with this run only.",
    )
    parser.add_argument(
        "--jax-cache-dir",
        type=Path,
        default=Path("tests") / "reduced_upstream_examples" / ".jax_compilation_cache",
        help="Persistent JAX compilation cache directory for sfincs_jax subprocess runs.",
    )
    parser.add_argument(
        "--jax-repeats",
        type=int,
        default=1,
        help="Number of sfincs_jax repeats per case (>=2 records warm runtime from repeats after the first).",
    )
    parser.add_argument(
        "--no-collect-iterations",
        action="store_true",
        help="Disable solver-iteration stats collection in sfincs_jax logs.",
    )
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
    report_json = out_root / "suite_report.json"
    merged_results: dict[str, CaseResult] = {} if args.reset_report else _load_existing_results(report_json)
    current_run_results: list[CaseResult] = []

    for index, input_path in enumerate(inputs, start=1):
        case = input_path.parent.name
        print(f"[{index}/{len(inputs)}] {case}")
        reduced_seed = REPO_ROOT / "tests" / "reduced_inputs" / f"{case}.input.namelist"
        case_input = reduced_seed if reduced_seed.exists() else input_path
        use_seed_resolution = case_input == reduced_seed
        if case_input == reduced_seed:
            print(f"  using reduced seed -> {reduced_seed}")
        case_out = out_root / case
        result = _run_case(
            case_name=case,
            case_input=case_input,
            case_out_dir=case_out,
            fortran_exe=fortran_exe,
            timeout_s=float(args.timeout_s),
            rtol=float(args.rtol),
            atol=float(args.atol),
            max_attempts=int(args.max_attempts),
            use_seed_resolution=use_seed_resolution,
            reuse_fortran=bool(args.reuse_fortran),
            collect_iterations=not bool(args.no_collect_iterations),
            jax_repeats=int(args.jax_repeats),
            jax_cache_dir=(REPO_ROOT / args.jax_cache_dir),
        )
        if result.status in {"parity_ok", "parity_mismatch"} and result.n_common_keys > 0:
            reduced_fixture = REPO_ROOT / "tests" / "reduced_inputs" / f"{case}.input.namelist"
            reduced_fixture.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(Path(result.input_path), reduced_fixture)
            result.promoted_input_path = _repo_rel(reduced_fixture)
            print(f"  saved reduced input -> {reduced_fixture}")
        current_run_results.append(result)
        merged_results[result.case] = result
        print(
            f"  status={result.status} attempts={result.attempts} reductions={result.reductions} "
            f"res={result.final_resolution} mismatch={result.n_mismatch_common}/{result.n_common_keys} "
            f"strict={result.strict_n_mismatch_common}/{result.strict_n_common_keys} "
            f"printParity={result.print_parity_signals}/{result.print_parity_total} blocker={result.blocker_type}"
        )

    ordered = [merged_results[k] for k in sorted(merged_results)]
    report_json.write_text(json.dumps([asdict(r) for r in ordered], indent=2), encoding="utf-8")
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
    report_rst = REPO_ROOT / "docs" / "_generated" / "reduced_upstream_suite_status.rst"
    _write_rst(ordered, report_rst, strict=False)
    report_rst_strict = REPO_ROOT / "docs" / "_generated" / "reduced_upstream_suite_status_strict.rst"
    _write_rst(ordered, report_rst_strict, strict=True)
    print(f"Wrote {report_json}")
    print(f"Wrote {report_json_strict}")
    print(f"Wrote {report_rst}")
    print(f"Wrote {report_rst_strict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
