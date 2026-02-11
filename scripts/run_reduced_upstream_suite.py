#!/usr/bin/env python
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
from sfincs_jax.compare import compare_sfincs_outputs
from sfincs_jax.io import localize_equilibrium_file_in_place
from sfincs_jax.namelist import read_sfincs_input

RES_KEYS: tuple[str, ...] = ("NTHETA", "NZETA", "NX", "NXI")
MIN_RES: dict[str, int] = {"NTHETA": 5, "NZETA": 1, "NX": 1, "NXI": 2}


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
    final_resolution: dict[str, int]
    input_path: str
    promoted_input_path: str | None
    fortran_h5: str | None
    jax_h5: str | None


PRINT_SIGNALS: dict[str, tuple[str, str]] = {
    "input_namelist": (r"input\.namelist|Successfully read parameters", r"input\.namelist summary|input="),
    "geometry_summary": (r"Geometry scheme|Geometry parameters", r"geometryScheme"),
    "resolution_summary": (r"Ntheta|Nzeta|Nxi|Nx", r"numerical resolution|resolution:"),
    "x_grid": (r"\bx:\s", r"x-grid="),
    "residual": (r"Residual function norm|evaluateResidual called", r"residual"),
    "jacobian": (r"evaluateJacobian called|populateMatrix", r"jacobian|whichMatrix"),
    "diagnostics": (r"Computing diagnostics|Results for species", r"diagnostic"),
    "output_write": (r"Saving diagnostics to h5 file|sfincsOutput\.h5", r"writing .*sfincsOutput|wrote sfincsOutput"),
    "runtime": (r"Time to solve|seconds", r"elapsed_s="),
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
) -> float:
    cmd = [
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
    t0 = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(cmd, cwd=str(REPO_ROOT), check=True, timeout=timeout_s, stdout=log, stderr=subprocess.STDOUT)
    if not output_path.exists():
        tail = _tail(log_path, n=40)
        raise RuntimeError(f"JAX run returned success but did not create output: {output_path}\n{tail}")
    return time.perf_counter() - t0


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
        # Some MPI-enabled builds error out on libfabric defaults. Retry once with a TCP provider.
        mpi_hint = any(s in tail.lower() for s in ("mpi_init", "ofi call", "libfabric", "mpidi_ofi"))
        if mpi_hint:
            # In restricted/sandboxed environments, socket-based providers can fail
            # even for single-process jobs. Prefer a shared-memory provider first.
            for provider in ("shm", "tcp"):
                env_retry = dict(env)
                env_retry.setdefault("FI_PROVIDER", provider)
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
            final_resolution={k: int(v) for k, v in dict(item.get("final_resolution", {})).items()},
            input_path=str(item.get("input_path", "")),
            promoted_input_path=item.get("promoted_input_path"),
            fortran_h5=item.get("fortran_h5"),
            jax_h5=item.get("jax_h5"),
        )
    return out


def _write_rst(rows: list[CaseResult], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = len(rows)
    by_status: dict[str, int] = {}
    for row in rows:
        by_status[row.status] = by_status.get(row.status, 0) + 1
    summary = ", ".join(f"{k}={v}" for k, v in sorted(by_status.items()))

    lines: list[str] = []
    lines.append(":orphan:\n\n")
    lines.append(".. NOTE: Auto-generated by `scripts/run_reduced_upstream_suite.py`.\n\n")
    lines.append(f"- Cases: **{total}**\n")
    lines.append(f"- Status counts: {summary}\n")
    lines.append("- Timeout policy: 30s per Fortran/JAX run attempt, then halve largest axis and retry.\n\n")
    lines.append(".. list-table:: Reduced-resolution upstream suite parity status\n")
    lines.append("   :header-rows: 1\n")
    lines.append("   :widths: 23 9 16 10 7 7 10 10 11 13 12 16\n\n")
    lines.append("   * - Case\n")
    lines.append("     - Status\n")
    lines.append("     - Blocker\n")
    lines.append("     - Resolution\n")
    lines.append("     - Tries\n")
    lines.append("     - Reductions\n")
    lines.append("     - Fortran(s)\n")
    lines.append("     - JAX(s)\n")
    lines.append("     - Mismatches\n")
    lines.append("     - Buckets\n")
    lines.append("     - Print parity\n")
    lines.append("     - Note\n")
    for row in rows:
        res = ",".join(f"{k}={v}" for k, v in sorted(row.final_resolution.items()))
        ft = "" if row.fortran_runtime_s is None else f"{row.fortran_runtime_s:.3f}"
        jt = "" if row.jax_runtime_s is None else f"{row.jax_runtime_s:.3f}"
        mm = f"{row.n_mismatch_common}/{row.n_common_keys}" if row.n_common_keys > 0 else "-"
        buckets = f"S:{row.n_mismatch_solver} P:{row.n_mismatch_physics}"
        pp = f"{row.print_parity_signals}/{row.print_parity_total}" if row.print_parity_total > 0 else "-"
        lines.append(f"   * - {row.case}\n")
        lines.append(f"     - {row.status}\n")
        lines.append(f"     - {row.blocker_type}\n")
        lines.append(f"     - {res}\n")
        lines.append(f"     - {row.attempts}\n")
        lines.append(f"     - {row.reductions}\n")
        lines.append(f"     - {ft}\n")
        lines.append(f"     - {jt}\n")
        lines.append(f"     - {mm}\n")
        lines.append(f"     - {buckets}\n")
        lines.append(f"     - {pp}\n")
        lines.append(f"     - {row.message}\n")
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

    while attempts < max_attempts:
        attempts += 1
        final_res = _resolution_from_namelist(dst_input)
        fortran_dir = case_out_dir / "fortran_run"
        fortran_log = fortran_dir / "sfincs.log"
        fortran_h5_this_attempt: Path | None = None
        out_fortran_existing = fortran_dir / "sfincsOutput.h5"
        if bool(reuse_fortran) and out_fortran_existing.exists():
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
            if any(s in lower for s in ("mpi_init", "ofi call", "libfabric", "mpidi_ofi")):
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
            jax_runtime = _run_jax_cli(
                input_path=dst_input,
                output_path=jax_h5,
                timeout_s=timeout_s,
                log_path=jax_log,
                compute_solution=compute_solution,
                compute_transport_matrix=compute_transport_matrix,
            )
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
        except Exception as exc:  # noqa: BLE001
            status = "compare_error"
            note = f"Compare error: {type(exc).__name__}: {exc}"
        if fortran_log_path is not None and jax_log_path is not None:
            print_signals, print_total, print_missing = _compute_print_parity(fortran_log=fortran_log_path, jax_log=jax_log_path)
            if print_total > 0 and print_signals < print_total:
                note = f"{note} printParity={print_signals}/{print_total} missing={','.join(print_missing[:3])}"
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
        final_resolution=final_res,
        input_path=str(dst_input),
        promoted_input_path=None,
        fortran_h5=str(fortran_h5_path) if fortran_h5_path is not None else None,
        jax_h5=str(jax_h5_path) if jax_h5_path is not None else None,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run reduced-resolution upstream suite one case at a time with 30s timeout/reduction policy.")
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
        default=Path("/Users/rogeriojorge/local/tests/sfincs/fortran/version3/sfincs"),
        help="Path to Fortran v3 executable.",
    )
    parser.add_argument("--pattern", type=str, default=None, help="Regex filter on case directory path.")
    parser.add_argument("--timeout-s", type=float, default=30.0, help="Per-run timeout in seconds.")
    parser.add_argument("--max-attempts", type=int, default=6, help="Maximum adaptive retries per case.")
    parser.add_argument("--rtol", type=float, default=1e-8)
    parser.add_argument("--atol", type=float, default=1e-8)
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
        )
        if result.status in {"parity_ok", "parity_mismatch"} and result.n_common_keys > 0:
            reduced_fixture = REPO_ROOT / "tests" / "reduced_inputs" / f"{case}.input.namelist"
            reduced_fixture.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(Path(result.input_path), reduced_fixture)
            result.promoted_input_path = str(reduced_fixture)
            print(f"  saved reduced input -> {reduced_fixture}")
        current_run_results.append(result)
        merged_results[result.case] = result
        print(
            f"  status={result.status} attempts={result.attempts} reductions={result.reductions} "
            f"res={result.final_resolution} mismatch={result.n_mismatch_common}/{result.n_common_keys} "
            f"printParity={result.print_parity_signals}/{result.print_parity_total} blocker={result.blocker_type}"
        )

    ordered = [merged_results[k] for k in sorted(merged_results)]
    report_json.write_text(json.dumps([asdict(r) for r in ordered], indent=2), encoding="utf-8")
    report_rst = REPO_ROOT / "docs" / "_generated" / "reduced_upstream_suite_status.rst"
    _write_rst(ordered, report_rst)
    print(f"Wrote {report_json}")
    print(f"Wrote {report_rst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
