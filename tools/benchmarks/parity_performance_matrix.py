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
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
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
    """Run a command under /usr/bin/time; return wall seconds and peak RSS."""
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            _time_flag() + command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env={**os.environ, **(env or {})},
        )
    except subprocess.TimeoutExpired:
        return {"returncode": None, "wall_s": timeout_s, "peak_rss_gb": None,
                "error": f"timeout after {timeout_s:.0f}s"}
    return {
        "returncode": proc.returncode,
        "wall_s": round(time.perf_counter() - start, 2),
        "peak_rss_gb": _peak_rss_gb(proc.stderr),
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
    }


_DKX_DRIVER = """
import json, sys, time
import jax
jax.config.update("jax_enable_x64", True)
from dkx.inputs import load_sfincs_input
from dkx.run import run_profile, run_transport_matrix

deck, out, reps = sys.argv[1], sys.argv[2], int(sys.argv[3])
general = load_sfincs_input(deck).raw.group("general")
rhs_mode = int(next((v for k, v in general.items() if k.lower() == "rhsmode"), 1))
driver = run_profile if rhs_mode == 1 else run_transport_matrix

t0 = time.perf_counter()
run = driver(deck, out_path=out, emit=None)
cold = time.perf_counter() - t0
warm = None
for _ in range(reps):
    t0 = time.perf_counter()
    driver(deck, out_path=out, emit=None)
    elapsed = time.perf_counter() - t0
    warm = elapsed if warm is None else min(warm, elapsed)
json.dump({
    "cold_s": round(cold, 3),
    "warm_s": None if warm is None else round(warm, 3),
    "backend": jax.default_backend(),
    "method": str(run.solve_result.method),
    "converged": bool(run.solve_result.converged),
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
        return
    deck.write_text(re.sub(r"^(&general\s*)$", rf"\g<1>\n  {_BINARY_DUMP_KEY} = .true.",
                           text, count=1, flags=re.MULTILINE | re.IGNORECASE))


def fortran_true_residual(work: Path, *, linear: bool) -> float | None:
    """``||A x - b|| / ||b||`` from SFINCS's *own* matrix, state and rhs.

    Only meaningful for a **linear** run.  With ``Phi1``/quasineutrality the
    problem is a Newton iteration: the dumped ``iteration_000`` matrix is the
    Jacobian at the initial guess while the reported state comes from a later
    step, so the combination is not a residual of anything.  Left unguarded it
    reports values like 1.8 on decks whose outputs agree to 1e-12.

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

    matrix = work / "sfincsBinary_iteration_000_whichMatrix_3"
    state = work / "sfincsBinary_iteration_000_stateVector"
    residual = work / "sfincsBinary_iteration_000_residual"
    if not (matrix.exists() and state.exists() and residual.exists()):
        return None
    try:
        aij = read_petsc_mat_aij(matrix)
        operator = csr_matrix((aij.data, aij.col_ind, aij.row_ptr), shape=aij.shape)
        x = np.asarray(read_petsc_vec(state).values)
        # The dumped residual is evaluated at x = 0, so it is -b.
        b = -np.asarray(read_petsc_vec(residual).values)
        norm_b = float(np.linalg.norm(b))
        if norm_b == 0.0:
            return None
        return float(np.linalg.norm(operator @ x - b) / norm_b)
    except Exception:  # pragma: no cover - a malformed dump must not kill the sweep
        return None


def _fortran_succeeded(work: Path, result: dict) -> bool:
    """A Fortran run counts only if it produced output.

    The return code alone is not evidence: SFINCS exits zero after an
    equilibrium-file or geometry error.  The output file is the real signal.
    """
    return result.get("returncode") == 0 and (work / "sfincsOutput.h5").exists()


def _read_h5(path: Path) -> dict:
    import h5py

    out: dict = {}
    with h5py.File(path, "r") as handle:
        handle.visititems(lambda name, obj: out.__setitem__(name.split("/")[-1], obj[...]))
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
        reference, candidate = _read_h5(fortran_h5), _read_h5(dkx_h5)
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
                shutil.copytree(example_dir, work)
                _absolutize_equilibrium(work / "input.namelist", example_dir)
                if fortran_residual:
                    _request_binary_dump(work / "input.namelist")
                binary = [str(fortran_binary)]
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
                        work, linear=not record.get("includePhi1", False)
                    )
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
        shutil.copytree(example_dir, work)
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
        record["dkx"] = {k: v for k, v in result.items() if k != "stdout_tail"}

        reference = root / "fortran_1.h5"
        record["parity"] = (
            compare_outputs(
                reference, work / "dkxOutput.h5", int(record.get("n_species", 1))
            )
            if reference.exists()
            else {"error": "no fortran reference"}
        )
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
    probe = [*launcher, str(binary), "-help"]
    try:
        # PETSc's option banner is not valid UTF-8, so decode defensively: the
        # probe only needs to see whether the launcher itself failed.
        result = subprocess.run(
            probe, capture_output=True, timeout=120, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"cannot launch {' '.join(probe)}: {exc}"
    blob = (result.stdout + result.stderr).decode("utf-8", errors="replace")
    if "prefix does not exist" in blob or "critical libmamba" in blob:
        return (
            f"the launcher cannot resolve its environment: {blob.strip().splitlines()[0]}\n"
            "  (export MAMBA_ROOT_PREFIX=$HOME/micromamba before a non-interactive run)"
        )
    return None


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
        "--fortran-residual", action="store_true",
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

    # A sweep runs for hours, so something always ends up waiting on it.  Clear
    # the sentinel at the start and write it at the end, so a waiter can poll a
    # *file*: `until [ -f sweep.jsonl.done ]; do sleep 60; done`.  Waiting on a
    # process instead is a trap -- `pgrep -f parity_performance_matrix` matches
    # the waiting shell's own command line, so the loop waits on itself and
    # never exits.  That cost a day of wall clock on 2026-08-01, with the sweep
    # sitting complete the whole time.
    sentinel = args.out.with_suffix(args.out.suffix + ".done")
    sentinel.unlink(missing_ok=True)

    done = set()
    if args.out.exists():
        for line in args.out.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["case"])

    directories = sorted(p.parent for p in args.examples.glob("*/input.namelist"))
    cases = []
    for directory in directories:
        if directory.name in done:
            continue
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

    print(f"{len(cases)} case(s) to run, {len(done)} already recorded", file=sys.stderr)
    with args.out.open("a") as handle:
        for index, (dof, directory) in enumerate(cases, start=1):
            print(f"[{index}/{len(cases)}] {directory.name} ({dof} dof)", file=sys.stderr)
            record = run_case(
                directory, args.fortran_binary,
                petsc_profile=args.petsc_profile,
                ranks=args.ranks, reps=args.reps,
                timeout_s=args.timeout_s, equilibria=args.equilibria,
                launcher=args.fortran_launcher.split() if args.fortran_launcher else [],
                fortran_residual=args.fortran_residual,
            )
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            fortran = (record.get("fortran") or {}).get("1", {})
            dkx = record.get("dkx", {})
            print(
                f"    fortran {fortran.get('wall_s')}s/{fortran.get('peak_rss_gb')}GB"
                f"  dkx cold {dkx.get('cold_s')}s warm {dkx.get('warm_s')}s"
                f"/{dkx.get('peak_rss_gb')}GB",
                file=sys.stderr,
            )

    records = [
        json.loads(line)
        for line in args.out.read_text().splitlines()
        if line.strip()
    ]
    summary = {
        "cases": len(records),
        "fortran_ok": sum(
            1 for r in records
            if ((r.get("fortran") or {}).get("1") or {}).get("succeeded")
        ),
        "dkx_ok": sum(1 for r in records if (r.get("dkx") or {}).get("warm_s")),
        "comparable": sum(
            1 for r in records
            if any(
                isinstance(v, float)
                for v in ((r.get("parity") or {}).get("difference") or {}).values()
            )
        ),
    }
    sentinel.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"sweep complete: {summary['cases']} cases, "
        f"fortran ok {summary['fortran_ok']}, dkx ok {summary['dkx_ok']}, "
        f"comparable {summary['comparable']} -> {sentinel}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
