#!/usr/bin/env python
"""Parity, runtime and peak-memory matrix: dkx vs SFINCS Fortran v3.

Runs *both* codes over a matrix of decks and records, per case, wall time,
peak resident memory, and the agreement of the physics outputs.  The matrix is
upstream's own example suite, which is what makes it a defensible sweep rather
than a curated one: it spans geometry schemes 1/2/4/5/11 plus the filtered
W7-X netCDF equilibria, pitch-angle and Fokker-Planck collisions, zero and
finite ``Er``, ``Phi1``/quasineutrality on and off, tangential magnetic
drifts, one to three species, and problem sizes from 651 to 1.9M unknowns.

Both codes run as isolated subprocesses under ``/usr/bin/time -l`` (macOS) or
``/usr/bin/time -v`` (GNU), so peak RSS is the operating system's number for a
whole process rather than an in-process estimate that misses allocator and
runtime overhead.  Each case gets a fresh copy of the example directory, so
equilibrium files resolve exactly as they do upstream and outputs never
collide.

``dkx`` is reported cold *and* warm.  Cold includes JIT compilation and is the
honest number for a single one-shot solve; warm is the honest number for the
scan/optimization workloads the code exists for.  Fortran has no equivalent
split.  Both are recorded so neither reading can be cherry-picked.

Results stream to JSONL as each case finishes, so a long sweep is resumable
and a single failing case never costs the rest of the run.

Usage::

    python tools/benchmarks/parity_performance_matrix.py \
        --examples /path/to/sfincs/fortran/version3/examples \
        --fortran-binary /path/to/sfincs \
        --out results.jsonl --max-dof 300000

    # resume: cases already present in the JSONL are skipped
    python tools/benchmarks/parity_performance_matrix.py ... --out results.jsonl
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

_MACOS_RSS = re.compile(r"^\s*(\d+)\s+maximum resident set size", re.MULTILINE)
_GNU_RSS = re.compile(r"Maximum resident set size \(kbytes\): (\d+)")

#: Physics outputs compared across the two codes.  Deliberately the headline
#: moments rather than every dataset: these are what a user quotes from a run,
#: and they contract the whole distribution function, so agreement here is a
#: strong statement about the solve rather than about one grid point.
COMPARE_KEYS = (
    "FSABFlow",
    "FSABjHat",
    "particleFlux_vm_psiHat",
    "heatFlux_vm_psiHat",
    "particleFlux_vd_psiHat",
    "heatFlux_vd_psiHat",
    "transportMatrix",
)

#: Keys that are NOT a monoenergetic run's output.  RHSMode=3 computes the
#: transport matrix; the species fluxes are not defined for it, and one code
#: writing zeros where the other writes a small residue yields a relative
#: difference of exactly 1.0 -- a total disagreement on a quantity neither code
#: was asked for.  That artifact set the campaign's headline "worst parity" and
#: hid the real outlier, so the comparison skips them rather than reporting them.
MONOENERGETIC_SKIP_KEYS = frozenset({
    "particleFlux_vm_psiHat",
    "heatFlux_vm_psiHat",
    "particleFlux_vd_psiHat",
    "heatFlux_vd_psiHat",
    "FSABFlow",
    "FSABjHat",
})


def _peak_rss_gb(timing_output: str) -> float | None:
    """Peak RSS in GB parsed from ``/usr/bin/time`` verbose output."""
    if match := _MACOS_RSS.search(timing_output):
        return int(match.group(1)) / 1024**3  # macOS reports bytes
    if match := _GNU_RSS.search(timing_output):
        return int(match.group(1)) / 1024**2  # GNU reports kbytes
    return None


def _time_flag() -> list[str]:
    return ["/usr/bin/time", "-l" if platform.system() == "Darwin" else "-v"]


def _run_measured(
    command: list[str], cwd: Path, timeout_s: float, env: dict | None = None
) -> dict:
    """Supervise a measured process group, retaining logs without buffering them.

    Cleanup also covers cancellation and descendants whose immediate parent has
    exited. The session leader PID is the group ID even after that leader exits.
    SIGTERM cancellation is translated into SystemExit on the main thread so
    the finally block runs; SIGINT/other exceptions already unwind normally.
    """
    start = time.perf_counter()
    proc = None
    previous = None

    def terminate(signum, frame):
        raise SystemExit(128 + signum)

    if threading.current_thread() is threading.main_thread():
        previous = signal.signal(signal.SIGTERM, terminate)
    result = {}
    try:
        with (cwd / "benchmark.stdout.log").open("wb") as stdout, (
            cwd / "benchmark.stderr.log"
        ).open("wb") as stderr:
            try:
                proc = subprocess.Popen(
                    _time_flag() + command, cwd=cwd, stdout=stdout, stderr=stderr,
                    env={**os.environ, **(env or {})}, start_new_session=True,
                )
                proc.wait(timeout=timeout_s)
                result["returncode"] = proc.returncode
            except subprocess.TimeoutExpired:
                result.update(returncode=None, error=f"timeout after {timeout_s:.0f}s")
            finally:
                if proc is not None:
                    _kill_group(proc)
    finally:
        if previous is not None:
            signal.signal(signal.SIGTERM, previous)
    for stream in ("stdout", "stderr"):
        with (cwd / f"benchmark.{stream}.log").open("rb") as log:
            log.seek(0, os.SEEK_END)
            log.seek(max(0, log.tell() - 8192))
            result[f"{stream}_tail"] = log.read().decode("utf-8", errors="replace")[-2000:]
    result.update(
        wall_s=round(time.perf_counter() - start, 2),
        peak_rss_gb=_peak_rss_gb(result["stderr_tail"]),
    )
    return result


def _kill_group(proc: subprocess.Popen) -> None:
    """Kill the owned session group and reap its leader, including after exit."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()


_DKX_DRIVER = """
import json, sys, time, statistics
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
from dkx.inputs import load_sfincs_input
from dkx.run import run_profile, run_transport_matrix

deck, out, reps = sys.argv[1], sys.argv[2], int(sys.argv[3])
inp = load_sfincs_input(deck)
rhs_mode = inp.general.rhs_mode
driver = run_profile if rhs_mode == 1 else run_transport_matrix
linear = not inp.physics.include_phi1
samples, residuals, converged = [], [], []
for _ in range(reps + 1):
    t0 = time.perf_counter()
    run = driver(deck, out_path=out, emit=None, tol=inp.resolution.solver_tolerance)
    states = [run.state_vector] if rhs_mode == 1 else run.state_vectors
    jax.block_until_ready(states)
    samples.append(time.perf_counter() - t0)
    converged.append(bool(run.solve_result.converged))
    if linear:
        for i, state in enumerate(states, 1):
            rhs = np.asarray(run.operator.rhs(i))
            r = np.asarray(run.operator.apply(state)) - rhs
            norm_b, norm_r = np.linalg.norm(rhs), np.linalg.norm(r)
            residuals.append(float(norm_r / norm_b) if norm_b else (0.0 if norm_r == 0 else None))
valid = residuals and all(r is not None and np.isfinite(r) for r in residuals)
json.dump({
    "cold_s": samples[0],
    "warm_s": statistics.median(samples[1:]) if reps else None,
    "warm_samples_s": samples[1:],
    "backend": jax.default_backend(),
    "method": str(run.solve_result.method),
    "converged": all(converged),
    "true_residual": max(residuals) if valid else None,
    "true_residuals": residuals if valid else None,
}, open("dkx_timing.json", "w"))
"""


_EQUILIBRIUM_KEY = re.compile(
    r'^(\s*equilibriumFile\s*=\s*["\'])([^"\']+)(["\'])', re.MULTILINE | re.IGNORECASE
)


# PETSc writes one fixed-width row per event. These are the events that
# decide a SFINCS run's cost, in the order they happen: build the matrix,
# order it, factor it, then back-substitute once per Krylov iteration.
_PETSC_EVENTS = (
    "MatAssemblyBegin",
    "MatAssemblyEnd",
    "MatLUFactorSym",
    "MatLUFactorNum",
    "MatSolve",
    "MatMult",
    "PCSetUp",
    "PCApply",
    "KSPSetUp",
    "KSPSolve",
    "SNESSolve",
    "VecMDot",
    "VecNorm",
)


def _parse_petsc_log(path: Path) -> dict:
    """Extract per-event time and call count from a ``-log_view`` dump.

    Returns ``{event: {"count": int, "time_s": float, "percent_time": float}}``
    plus ``total_time_s``. Absent events are simply missing rather than zero,
    so a case that never factors is distinguishable from one that factors in
    no time.
    """
    if not path.is_file():
        return {}
    events: dict = {}
    total = None
    for line in path.read_text(errors="replace").splitlines():
        stripped = line.strip()
        if stripped.startswith("Time (sec):"):
            parts = stripped.split()
            if len(parts) >= 3:
                try:
                    total = float(parts[2])
                except ValueError:
                    pass
        name = stripped.split(" ", 1)[0] if stripped else ""
        if name in _PETSC_EVENTS:
            fields = stripped.split()
            # PETSc's fixed 21-column event row:
            #   0 name  1 count  2 ratio  3 time  4 ratio  5 flop  6 ratio
            #   7..9 messages  10 %T (global)  ... 20 Mflop/s
            try:
                count = int(float(fields[1]))
                time_s = float(fields[3])
                percent_time = float(fields[10])
            except (IndexError, ValueError):
                continue
            events[name] = {
                "count": count,
                "time_s": time_s,
                "percent_time": percent_time,
            }
    if total is not None:
        events["total_time_s"] = total
    return events


def _absolutize_equilibrium(deck: Path, source_dir: Path) -> None:
    """Rewrite a relative ``equilibriumFile`` against the deck's original home.

    Each case runs in a scratch copy, which breaks the ``../../../..`` paths
    upstream's decks use.  SFINCS reports a missing equilibrium on stdout and
    still **exits zero**, so an unrewritten path does not fail loudly -- it
    produces a fast, tiny, meaningless "run".  Rewriting up front is what keeps
    such a case from being recorded as a Fortran win.
    """
    text = deck.read_text()

    def replace(match: re.Match) -> str:
        path = Path(match.group(2))
        if path.is_absolute():
            return match.group(0)
        return match.group(1) + str((source_dir / path).resolve()) + match.group(3)

    updated = _EQUILIBRIUM_KEY.sub(replace, text)
    if updated != text:
        deck.write_text(updated)


_BINARY_DUMP_KEY = "saveMatricesAndVectorsInBinary"


def _request_binary_dump(deck: Path) -> None:
    """Ask SFINCS to write its matrix, rhs and state vector alongside the run."""
    text = deck.read_text()
    # Only an *active* setting counts.  Several upstream decks ship the key
    # commented out ("!  saveMatricesAndVectorsInBinary = .t."), and treating
    # that as already-enabled silently skips the dump.
    active = re.search(
        rf"^\s*{_BINARY_DUMP_KEY}\s*=", text, re.MULTILINE | re.IGNORECASE
    )
    if active is not None:
        deck.write_text(re.sub(
            rf"(^\s*{_BINARY_DUMP_KEY}\s*=)\s*\.[A-Za-z]+\.",
            r"\1 .true.", text, flags=re.MULTILINE | re.IGNORECASE,
        ))
        return
    deck.write_text(re.sub(r"^(&general\s*)$", rf"\g<1>\n  {_BINARY_DUMP_KEY} = .true.",
                           text, count=1, flags=re.MULTILINE | re.IGNORECASE))


def fortran_true_residual(work: Path, *, linear: bool, rhs_mode: int = 1) -> float | None:
    """``||A x - b|| / ||b||`` from SFINCS's *own* matrix, state and rhs.

    Supported for linear RHSMode 1/2/3, checking every RHS. Nonlinear runs
    need a final-state coupled residual, not an initial Jacobian paired with a
    later state. The campaign conservatively excludes Phi1 from this pairing,
    including linear Phi1 configurations until their dump conventions are
    independently verified.

    For linear runs, a disagreement between the two codes is not evidence about
    dkx until this is known.  PETSc's Krylov convergence test measures the *preconditioned*
    residual, and SFINCS preconditions with a simplified matrix, so a run can
    report success at ``solverTolerance = 1d-12`` while leaving a true residual
    of several percent -- measured at 7.5e-2 on
    ``geometryScheme4_2species_PAS_noEr``, where it produced a 28% error in the
    bootstrap current that looked exactly like a dkx parity bug.

    Sparse throughout: the 83k-unknown case has 1.5M nonzeros and must never be
    densified.
    """
    if not linear:
        return None

    import numpy as np
    from scipy.sparse import csr_matrix

    from dkx.validation.fortran import read_petsc_mat_aij, read_petsc_vec

    if rhs_mode not in (1, 2, 3):
        return None
    matrix = work / "sfincsBinary_iteration_000_whichMatrix_3"
    if not matrix.exists():
        # Linear transport runs evaluate F only at zero; no residual matrix is
        # dumped. Their Jacobian is the same linear operator, not matrix 0 (P).
        matrix = work / "sfincsBinary_iteration_000_whichMatrix_1"
    try:
        aij = read_petsc_mat_aij(matrix)
        operator = csr_matrix((aij.data, aij.col_ind, aij.row_ptr), shape=aij.shape)
        residuals = []
        for i in range({1: 1, 2: 3, 3: 2}[rhs_mode]):
            prefix = work / f"sfincsBinary_iteration_{i:03d}"
            x = np.asarray(read_petsc_vec(Path(str(prefix) + "_stateVector")).values)
            b = -np.asarray(read_petsc_vec(Path(str(prefix) + "_residual")).values)
            norm_b = float(np.linalg.norm(b))
            norm_r = float(np.linalg.norm(operator @ x - b))
            relative = norm_r / norm_b if norm_b else (0.0 if norm_r == 0 else math.inf)
            if not math.isfinite(relative):
                return None
            residuals.append(relative)
        return max(residuals)
    except (OSError, ValueError, TypeError, IndexError):
        return None


def _algebraic_acceptance(result: dict, tolerance: float) -> str:
    """Original relative-residual acceptance, separate from execution success."""
    residual = result.get("true_residual")
    if residual is None:
        return "not_checked"
    return "passed" if math.isfinite(residual) and 0 <= residual <= tolerance else "failed"


def _fortran_succeeded(work: Path, result: dict) -> bool:
    """Require successful execution, convergence and complete finite moments.

    This is an execution gate, not an algebraic or phase-space certificate.
    SFINCS can exit zero and write moments after a failed SNES iteration.
    """
    import numpy as np

    def fail(reason: str) -> bool:
        result["execution_error"] = reason
        return False

    if result.get("returncode") != 0 or result.get("error"):
        return fail("process failed or timed out")
    for name in ("benchmark.stdout.log", "benchmark.stderr.log"):
        path = work / name
        if not path.is_file():
            return fail(f"missing execution log: {name}")
        with path.open(errors="replace") as log:
            for line in log:
                if "did not converge" in line.lower() or "DIVERGED_" in line:
                    return fail(line.strip()[:1000])
    try:
        output = _read_h5(
            work / "sfincsOutput.h5",
            (*COMPARE_KEYS, "RHSMode", "integerToRepresentTrue", "finished",
             "didNonlinearCalculationConverge"),
        )
        mode = int(np.asarray(output["RHSMode"]).item())
        required = (
            ("FSABFlow", "FSABjHat", "particleFlux_vm_psiHat", "heatFlux_vm_psiHat")
            if mode == 1 else ("transportMatrix",) if mode in (2, 3) else ()
        )
        if not required:
            return fail(f"unsupported reference RHSMode {mode}")
        true = int(np.asarray(output["integerToRepresentTrue"]).item())
        if int(np.asarray(output["finished"]).item()) != true:
            return fail("output is not marked finished")
        converged = output.get("didNonlinearCalculationConverge")
        if converged is not None and int(np.asarray(converged).ravel()[-1]) != true:
            return fail("nonlinear convergence flag is false")
        complete = all(
            key in output and np.asarray(output[key]).size > 0
            and np.isfinite(output[key]).all() for key in required
        )
        return complete or fail("missing, empty or nonfinite required moments")
    except (OSError, KeyError, TypeError, ValueError) as exc:
        return fail(f"invalid reference output: {exc}")


def _read_h5(path: Path, keys: tuple[str, ...] | None = None) -> dict:
    import h5py

    out: dict = {}
    with h5py.File(path, "r") as handle:
        handle.visititems(
            lambda name, obj: out.__setitem__(name.split("/")[-1], obj[...])
            if isinstance(obj, h5py.Dataset) and (keys is None or name.split("/")[-1] in keys)
            else None
        )
    return out


def compare_outputs(fortran_h5: Path, dkx_h5: Path, n_species: int = 1) -> dict:
    """Max relative difference per compared key, scaled by the larger magnitude.

    ``n_species`` is what makes nonlinear (``Phi1``) runs comparable: both
    codes write one row per Newton iteration and they need not take the same
    number of iterations, so the arrays differ in length even when the answers
    agree.  The converged answer is the last row of each.
    """
    import numpy as np

    if not (fortran_h5.exists() and dkx_h5.exists()):
        return {"error": "missing output file"}
    try:
        keys = ("RHSMode", *COMPARE_KEYS)
        reference, candidate = _read_h5(fortran_h5, keys), _read_h5(dkx_h5, keys)
    except Exception as exc:  # pragma: no cover - corrupt output
        return {"error": f"{type(exc).__name__}: {exc}"}

    # RHSMode is written by both codes; 3 is monoenergetic.
    def _mode(d: dict) -> int:
        v = d.get("RHSMode")
        try:
            return int(np.atleast_1d(np.asarray(v)).ravel()[0])
        except Exception:
            return 1

    monoenergetic = _mode(reference) == 3 or _mode(candidate) == 3

    report: dict = {}
    magnitudes: dict = {}
    for key in COMPARE_KEYS:
        if key not in reference or key not in candidate:
            continue
        if monoenergetic and key in MONOENERGETIC_SKIP_KEYS:
            continue
        a = np.atleast_1d(np.asarray(reference[key], dtype=np.float64)).ravel()
        b = np.atleast_1d(np.asarray(candidate[key], dtype=np.float64)).ravel()
        # Both codes write one row per Newton iteration for some keys, and a
        # nonlinear run need not converge in the same number of iterations in
        # both.  Compare the converged row: the last ``n_species`` entries.
        if a.size != b.size:
            # Per-species keys carry ``n_species`` values per iteration;
            # species-summed ones (FSABjHat) carry a single value.  Take the
            # first row width that divides both lengths.
            for row in (n_species, 1):
                if row and a.size % row == 0 and b.size % row == 0:
                    a, b = a[-row:], b[-row:]
                    break
        if a.size != b.size:
            report[key] = {"error": f"shape {a.size} vs {b.size}"}
            continue
        scale = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))), 1e-300)
        report[key] = round(float(np.max(np.abs(a - b))) / scale, 12)
        # The magnitude the relative difference was taken against. Without it a
        # relative-difference table cannot tell a regression from a physical
        # zero: on an axisymmetric single-species deck at Er = 0 the particle
        # flux is intrinsically ambipolar and lands at ~5e-12 against a
        # bootstrap current of ~3e-2, so a 1.4e-2 relative difference there is
        # an absolute difference of 7e-14 and means nothing. Recorded per key
        # so the report can scale each moment against the case's own largest.
        magnitudes[key] = float(np.max(np.abs(a)))
    return {"difference": report, "magnitude": magnitudes}


def deck_metadata(deck: Path) -> dict:
    """Grid sizes and physics switches, read with dkx's own namelist parser."""
    from dkx.namelist import read_sfincs_input

    def value(group: dict, key: str, default):
        for name, item in group.items():
            if name.lower() == key.lower():
                return item
        return default

    nml = read_sfincs_input(str(deck))
    res, phys = nml.group("resolutionParameters"), nml.group("physicsParameters")
    geo, spec = nml.group("geometryParameters"), nml.group("speciesParameters")
    general = nml.group("general")
    charges = value(spec, "Zs", [1])
    n_species = len(charges) if isinstance(charges, (list, tuple)) else 1
    n_theta, n_zeta = int(value(res, "Ntheta", 15)), int(value(res, "Nzeta", 15))
    n_xi, n_x = int(value(res, "Nxi", 16)), int(value(res, "Nx", 5))
    return {
        "dof": n_theta * n_zeta * n_xi * n_x * n_species,
        "solverTolerance": float(value(res, "solverTolerance", 1e-6)),
        "Ntheta": n_theta, "Nzeta": n_zeta, "Nxi": n_xi, "Nx": n_x,
        "n_species": n_species,
        "geometryScheme": int(value(geo, "geometryScheme", 1)),
        "collisionOperator": int(value(phys, "collisionOperator", 0)),
        "RHSMode": int(value(general, "RHSMode", value(phys, "RHSMode", 1))),
        "includePhi1": bool(value(phys, "includePhi1", False)),
        "Er": float(value(phys, "Er", 0.0)),
        "magneticDrifts": bool(value(phys, "includeXDotTerm", False))
        or "magneticDrift" in deck.parent.name,
    }


def run_case(
    example_dir: Path,
    fortran_binary: Path | None,
    petsc_profile: bool = False,
    *,
    ranks: list[int],
    reps: int,
    timeout_s: float,
    equilibria: str | None,
    launcher: list[str],
    fortran_residual: bool,
) -> dict:
    """One deck through both codes, in isolated copies of the example directory."""
    deck = example_dir / "input.namelist"
    record: dict = {"case": example_dir.name}
    try:
        record.update(deck_metadata(deck))
    except Exception as exc:
        record["metadata_error"] = f"{type(exc).__name__}: {exc}"
        return record

    with tempfile.TemporaryDirectory(prefix="dkx_matrix_") as scratch:
        root = Path(scratch)

        record["fortran"] = {}
        if fortran_binary is not None:
            for n_ranks in ranks:
                work = root / f"fortran_{n_ranks}"
                shutil.copytree(
                    example_dir, work, ignore=shutil.ignore_patterns(
                        "sfincsOutput.h5", "dkxOutput.h5", "dkx_timing.json", "sfincsBinary*"
                    ),
                )
                _absolutize_equilibrium(work / "input.namelist", example_dir)
                if fortran_residual:
                    _request_binary_dump(work / "input.namelist")
                binary = [str(fortran_binary), "-ksp_converged_reason", "-snes_converged_reason"]
                if petsc_profile:
                    # PETSc's own event log. This is the only way to see where
                    # the Fortran run actually spends its time -- assembly,
                    # symbolic factorization, numeric factorization, triangular
                    # solves -- rather than inferring it from wall time. Parsed
                    # below into ``petsc_events``.
                    binary += ["-log_view", ":" + str(work / "petsc_log.txt")]
                command = (
                    binary
                    if n_ranks == 1
                    else ["mpirun", "-n", str(n_ranks), *binary]
                )
                # The launcher (e.g. ``micromamba run -n sfincs-fortran``) keeps
                # the Fortran toolchain's libraries confined to this subprocess;
                # exporting them into the parent would shadow the BLAS that this
                # process's own numpy is linked against.
                result = _run_measured(launcher + command, work, timeout_s)
                result["succeeded"] = _fortran_succeeded(work, result)
                if fortran_residual and result["succeeded"]:
                    result["true_residual"] = fortran_true_residual(
                        work, linear=not record.get("includePhi1", False), rhs_mode=record["RHSMode"]
                    )
                result["algebraic_acceptance"] = _algebraic_acceptance(result, record["solverTolerance"])
                if petsc_profile:
                    result["petsc_events"] = _parse_petsc_log(work / "petsc_log.txt")
                if result["succeeded"]:
                    shutil.copy(work / "sfincsOutput.h5", root / f"fortran_{n_ranks}.h5")
                else:
                    result["failure_tail"] = result.get("stdout_tail", "")[-600:]
                record["fortran"][str(n_ranks)] = {
                    k: v for k, v in result.items() if k != "stdout_tail"
                }

        work = root / "dkx"
        shutil.copytree(
            example_dir, work, ignore=shutil.ignore_patterns(
                "sfincsOutput.h5", "dkxOutput.h5", "dkx_timing.json", "sfincsBinary*"
            ),
        )
        _absolutize_equilibrium(work / "input.namelist", example_dir)
        env = {"JAX_ENABLE_X64": "True"}
        if equilibria:
            env["DKX_EQUILIBRIA_DIRS"] = equilibria
        result = _run_measured(
            [sys.executable, "-c", _DKX_DRIVER, str(work / "input.namelist"),
             str(work / "dkxOutput.h5"), str(reps)],
            work, timeout_s, env=env,
        )
        timing_path = work / "dkx_timing.json"
        if timing_path.exists():
            result.update(json.loads(timing_path.read_text()))
        result["algebraic_acceptance"] = _algebraic_acceptance(result, record["solverTolerance"])
        record["dkx"] = {k: v for k, v in result.items() if k != "stdout_tail"}

        record["algebraic_pair_accepted"] = (
            result.get("returncode") == 0 and result.get("converged") is True
            and record["fortran"].get("1", {}).get("succeeded") is True
            and result["algebraic_acceptance"] == "passed"
            and record["fortran"].get("1", {}).get("algebraic_acceptance") == "passed"
        )
        reference = root / "fortran_1.h5"
        record["parity"] = (
            compare_outputs(
                reference, work / "dkxOutput.h5", int(record.get("n_species", 1))
            )
            if reference.exists() and result.get("returncode") == 0
            and result.get("converged") is True
            and (not fortran_residual or record["algebraic_pair_accepted"])
            else {"error": "no successful pair at the required original residual"}
        )
        record["parity"]["scope"] = "algebraically_accepted" if record["algebraic_pair_accepted"] else "diagnostic_only"
    return record


def preflight_fortran(binary: Path | None, launcher: list[str]) -> str | None:
    """Reason the Fortran reference cannot run, or ``None`` if it can.

    A sweep is hours long and its whole value is the comparison, so a reference
    that cannot start must stop the run *now* rather than at the end.  The
    failure this exists for is silent: ``micromamba run`` resolves its
    environment against ``MAMBA_ROOT_PREFIX``, which an interactive shell sets
    and a ``nohup``-ed one does not, so every case fails in 0.02 s against a
    nonexistent prefix and the sweep completes with 38 well-formed records and
    no reference in any of them.
    """
    if binary is None:
        return None  # dkx-only sweep: nothing to check
    if not binary.exists():
        return f"--fortran-binary {binary} does not exist"
    probe = [*launcher, str(binary.resolve()), "-help"]
    try:
        # SFINCS may continue initialization after printing help. Isolate its
        # files and supervise the launcher's descendants just like real runs.
        with tempfile.TemporaryDirectory(prefix="dkx-reference-preflight-") as scratch:
            work = Path(scratch)
            result = _run_measured(probe, work, 120)
            if result.get("error"):
                return f"cannot launch {' '.join(probe)}: {result['error']}"
            for stream in ("stdout", "stderr"):
                with (work / f"benchmark.{stream}.log").open(errors="replace") as log:
                    for line in log:
                        if "prefix does not exist" in line or "critical libmamba" in line:
                            return (
                                f"the launcher cannot resolve its environment: {line.strip()[:1000]}\n"
                                "  (set MAMBA_ROOT_PREFIX or use an explicit environment prefix)"
                            )
    except OSError as exc:
        return f"cannot launch {' '.join(probe)}: {exc}"
    return None


def _atomic_text(path: Path, text: str) -> None:
    """Publish a complete checkpoint on the same filesystem, or keep the old one."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _campaign_id(args, directories: list[Path]) -> str:
    """Bind resume to inputs, code, runtime settings and the selected executable.

    This detects changed campaigns; it is not a substitute for a pinned external
    compiler/MPI/library environment in a release benchmark.
    """
    files = {Path(__file__).resolve()}
    for directory in directories:
        files.update(p.resolve() for p in directory.rglob("*") if p.is_file())
        for match in _EQUILIBRIUM_KEY.finditer((directory / "input.namelist").read_text()):
            files.add((directory / match.group(2)).resolve())
    if args.fortran_binary is not None:
        files.add(args.fortran_binary.resolve())
    versions = {}
    for package in ("dkx", "solvax", "jax", "jaxlib", "numpy", "scipy", "h5py"):
        versions[package] = importlib.metadata.version(package)
        if package in ("dkx", "solvax"):
            spec = importlib.util.find_spec(package)
            if spec is not None and spec.origin is not None:
                files.update(Path(spec.origin).parent.rglob("*.py"))
    digest = hashlib.sha256(json.dumps({
        "schema": 1,
        "options": {k: str(v) for k, v in vars(args).items() if k != "out"},
        "versions": versions,
        "python": sys.version,
        "platform": platform.platform(),
        "environment": {k: v for k, v in os.environ.items() if k.startswith(
            ("DKX_", "JAX_", "XLA_", "CUDA_", "OMP_", "MKL_", "OPENBLAS_", "MAMBA_", "LD_", "DYLD_")
        )},
    }, sort_keys=True).encode())
    for path in sorted(files):
        digest.update(str(path).encode() + b"\0")
        if not path.is_file():
            digest.update(b"missing\0")
            continue
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _execution_complete(record: dict, args) -> bool:
    candidate = record.get("dkx", {})
    if candidate.get("returncode") != 0 or candidate.get("converged") is not True:
        return False
    if args.fortran_residual and candidate.get("algebraic_acceptance") != "passed":
        return False
    return args.fortran_binary is None or all(
        record.get("fortran", {}).get(str(rank), {}).get("succeeded") is True
        and (not args.fortran_residual or
             record["fortran"][str(rank)].get("algebraic_acceptance") == "passed")
        for rank in args.ranks
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--fortran-binary", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ranks", type=int, nargs="+", default=[1])
    parser.add_argument("--reps", type=int, default=1, help="warm repetitions")
    parser.add_argument("--timeout-s", type=float, default=1800.0)
    parser.add_argument("--max-dof", type=int, default=None)
    parser.add_argument("--min-dof", type=int, default=0)
    parser.add_argument("--only", nargs="*", default=None, help="case-name substrings")
    parser.add_argument(
        "--petsc-profile",
        action="store_true",
        help=(
            "run the Fortran side under PETSc -log_view and record per-event "
            "time and call counts (assembly, symbolic and numeric LU, "
            "triangular solves, Krylov). This is what says where SFINCS's time "
            "actually goes, and therefore what is worth attacking in dkx."
        ),
    )
    parser.add_argument("--equilibria", default=os.environ.get("DKX_EQUILIBRIA_DIRS"))
    parser.add_argument(
        "--fortran-residual", action=argparse.BooleanOptionalAction, default=True,
        help="dump SFINCS's matrix/state/rhs and record its own true residual, so a "
             "reference that converged only in the preconditioned norm is not mistaken "
             "for a dkx parity failure (costs disk: 1.5M nonzeros on an 83k deck)",
    )
    parser.add_argument(
        "--fortran-launcher", default="",
        help="command prefix isolating the Fortran toolchain, e.g. "
             "'micromamba run -n sfincs-fortran'",
    )
    args = parser.parse_args(argv)

    launcher = args.fortran_launcher.split() if args.fortran_launcher else []
    reason = preflight_fortran(args.fortran_binary, launcher)
    if reason is not None:
        print(f"refusing to start: {reason}", file=sys.stderr)
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.with_suffix(args.out.suffix + ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("refusing to start: another process owns this campaign", file=sys.stderr)
            return 2
        return _run_campaign(args)


def _run_campaign(args) -> int:
    directories = sorted(p.parent for p in args.examples.glob("*/input.namelist"))
    cases = []
    for directory in directories:
        if args.only and not any(token in directory.name for token in args.only):
            continue
        try:
            dof = deck_metadata(directory / "input.namelist")["dof"]
        except Exception:
            dof = 0
        if dof < args.min_dof or (args.max_dof is not None and dof > args.max_dof):
            continue
        cases.append((dof, directory))
    cases.sort()

    campaign_id = _campaign_id(args, [directory for _, directory in cases])
    records = []
    if args.out.exists():
        try:
            records = [json.loads(line) for line in args.out.read_text().splitlines() if line.strip()]
            if any(record.get("campaign_id") != campaign_id for record in records):
                raise ValueError("campaign inputs, code or settings differ (or legacy provenance is absent)")
        except (ValueError, AttributeError) as exc:
            print(f"refusing to resume: {exc}; choose a fresh --out path", file=sys.stderr)
            return 2
    latest = {record["case"]: record for record in records}
    done = {case for case, record in latest.items() if _execution_complete(record, args)}
    cases = [(dof, directory) for dof, directory in cases if directory.name not in done]
    sentinel = args.out.with_suffix(args.out.suffix + ".done")
    sentinel.unlink(missing_ok=True)
    print(f"{len(cases)} case(s) to run, {len(done)} successful executions retained", file=sys.stderr)
    for index, (dof, directory) in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {directory.name} ({dof} dof)", file=sys.stderr)
        try:
            record = run_case(
                directory, args.fortran_binary,
                petsc_profile=args.petsc_profile,
                ranks=args.ranks, reps=args.reps,
                timeout_s=args.timeout_s, equilibria=args.equilibria,
                launcher=args.fortran_launcher.split() if args.fortran_launcher else [],
                fortran_residual=args.fortran_residual,
            )
        except BaseException as exc:
            records.append({"case": directory.name, "campaign_id": campaign_id,
                            "error": f"{type(exc).__name__}: {exc}"})
            _atomic_text(args.out, "".join(json.dumps(row) + "\n" for row in records))
            raise
        record["campaign_id"] = campaign_id
        records.append(record)
        _atomic_text(args.out, "".join(json.dumps(row) + "\n" for row in records))
        latest[record["case"]] = record
    if not args.out.exists():
        _atomic_text(args.out, "")
    current = list(latest.values())
    summary = {
        "cases": len(current),
        "attempts": len(records),
        "algebraic_pairs": sum(record.get("algebraic_pair_accepted", False) for record in current),
        "execution_complete": sum(_execution_complete(record, args) for record in current),
        "campaign_id": campaign_id,
        "checkpoint_sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
        "fortran_ok": sum(
            1 for r in current
            if ((r.get("fortran") or {}).get("1") or {}).get("succeeded")
        ),
        "dkx_ok": sum(1 for r in current if (r.get("dkx") or {}).get("returncode") == 0
                      and (r.get("dkx") or {}).get("converged") is True),
        "comparable": sum(
            1 for r in current
            if any(
                isinstance(v, float)
                for v in ((r.get("parity") or {}).get("difference") or {}).values()
            )
        ),
    }
    _atomic_text(sentinel, json.dumps(summary, indent=2) + "\n")
    print(
        f"sweep complete: {summary['cases']} cases, "
        f"fortran ok {summary['fortran_ok']}, dkx ok {summary['dkx_ok']}, "
        f"comparable {summary['comparable']} -> {sentinel}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
