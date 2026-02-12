from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from .compare import compare_sfincs_outputs
from .fortran import run_sfincs_fortran
from .io import read_sfincs_h5, write_sfincs_jax_output_h5
from .namelist import read_sfincs_input
from .postprocess_upstream import run_upstream_util
from .scans import linspace_including_endpoints, run_er_scan
from .ambipolar import solve_ambipolar_from_scan_dir
from .v3_driver import solve_v3_full_system_linear_gmres, solve_v3_transport_matrix_linear_gmres


def _now() -> float:
    return time.perf_counter()


def _emit(msg: str, *, level: int, args: argparse.Namespace) -> None:
    """Simple structured stdout logging for the CLI.

    We intentionally avoid the stdlib `logging` module here to keep CLI output
    deterministic across platforms and to make it easy to compare with upstream
    SFINCS logs.
    """
    verbose = int(getattr(args, "verbose", 0) or 0)
    quiet = bool(getattr(args, "quiet", False))
    if quiet:
        return
    if verbose >= level:
        print(msg)


def _emit_namelist_summary(*, nml, args: argparse.Namespace) -> None:
    geom = nml.group("geometryParameters")
    phys = nml.group("physicsParameters")
    res = nml.group("resolutionParameters")
    general = nml.group("general")

    def _g(group: dict, key: str, default=None):
        return group.get(key.upper(), default)

    _emit("----------------------------------------------------------------", level=0, args=args)
    _emit(" input.namelist summary", level=0, args=args)
    _emit(f" geometryScheme={_g(geom, 'geometryScheme', '?')}", level=0, args=args)
    _emit(f" RHSMode={_g(general, 'RHSMode', '?')}", level=0, args=args)
    _emit(f" collisionOperator={_g(phys, 'collisionOperator', '?')}", level=0, args=args)
    _emit(f" includePhi1={bool(_g(phys, 'includePhi1', False))}", level=0, args=args)
    _emit(f" includePhi1InKineticEquation={bool(_g(phys, 'includePhi1InKineticEquation', False))}", level=2, args=args)
    _emit(f" includePhi1InCollisionOperator={bool(_g(phys, 'includePhi1InCollisionOperator', False))}", level=2, args=args)
    _emit(f" useDKESExBDrift={bool(_g(phys, 'useDKESExBDrift', False))}", level=2, args=args)
    _emit(
        " resolution:"
        f" Ntheta={_g(res, 'Ntheta', '?')}"
        f" Nzeta={_g(res, 'Nzeta', '?')}"
        f" Nxi={_g(res, 'Nxi', '?')}"
        f" NL={_g(res, 'NL', '?')}"
        f" Nx={_g(res, 'Nx', '?')}",
        level=0,
        args=args,
    )
    _emit(f" solverTolerance={_g(res, 'solverTolerance', '?')}", level=2, args=args)


def _emit_runtime_info(*, args: argparse.Namespace) -> None:
    """Emit basic runtime info helpful for benchmarking and bug reports."""
    try:
        import jax  # noqa: PLC0415
        import jax.numpy as _jnp  # noqa: PLC0415

        _emit(f" jax={jax.__version__} backend={jax.default_backend()} devices={jax.devices()}", level=2, args=args)
        _emit(f" jax_enable_x64={bool(_jnp.array(0.0).dtype == _jnp.float64)}", level=3, args=args)
    except Exception:  # noqa: BLE001
        return


def _cmd_solve_v3(args: argparse.Namespace) -> int:
    t0 = _now()
    nml = read_sfincs_input(Path(args.input))
    _emit("################################################################", level=0, args=args)
    _emit(" sfincs_jax solve-v3", level=0, args=args)
    _emit(f" input={Path(args.input).resolve()}", level=0, args=args)
    _emit_namelist_summary(nml=nml, args=args)
    _emit_runtime_info(args=args)
    _emit(f" tol={args.tol} atol={args.atol} restart={args.restart} maxiter={args.maxiter} solve_method={args.solve_method}", level=1, args=args)
    if args.which_rhs is not None:
        _emit(f" whichRHS={args.which_rhs}", level=0, args=args)
    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        which_rhs=int(args.which_rhs) if args.which_rhs is not None else None,
        tol=float(args.tol),
        atol=float(args.atol),
        restart=int(args.restart),
        maxiter=int(args.maxiter) if args.maxiter is not None else None,
        solve_method=str(args.solve_method),
        emit=lambda level, msg: _emit(msg, level=level, args=args),
    )
    out_state = Path(args.out_state)
    out_state.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_state, np.asarray(result.x))
    _emit(f" wrote stateVector -> {out_state.resolve()}", level=0, args=args)
    _emit(f" residual_norm={float(result.residual_norm):.6e}", level=0, args=args)
    _emit(f" elapsed_s={_now()-t0:.3f}", level=1, args=args)
    return 0


def _cmd_run_fortran(args: argparse.Namespace) -> int:
    t0 = _now()
    _emit("################################################################", level=0, args=args)
    _emit(" sfincs_jax run-fortran", level=0, args=args)
    _emit(f" input={Path(args.input).resolve()}", level=0, args=args)
    output_path = run_sfincs_fortran(
        input_namelist=Path(args.input),
        exe=Path(args.exe) if args.exe else None,
        workdir=Path(args.workdir) if args.workdir else None,
    )
    _emit(f" wrote sfincsOutput.h5 -> {output_path}", level=0, args=args)
    _emit(f" elapsed_s={_now()-t0:.3f}", level=1, args=args)
    return 0


def _cmd_write_output(args: argparse.Namespace) -> int:
    t0 = _now()
    nml = read_sfincs_input(Path(args.input))
    rhs_mode = int(nml.group("general").get("RHSMODE", 1))

    # Default to upstream v3 behavior: full solve/write appropriate to RHSMode.
    geometry_only = bool(getattr(args, "geometry_only", False))
    compute_solution = (not geometry_only) and (bool(getattr(args, "compute_solution", False)) or rhs_mode == 1)
    compute_transport_matrix = (not geometry_only) and (bool(args.compute_transport_matrix) or rhs_mode in (2, 3))

    out_path = write_sfincs_jax_output_h5(
        input_namelist=Path(args.input),
        output_path=Path(args.out),
        fortran_layout=bool(args.fortran_layout),
        overwrite=bool(args.overwrite),
        compute_transport_matrix=bool(compute_transport_matrix),
        compute_solution=bool(compute_solution),
        emit=lambda level, msg: _emit(msg, level=level, args=args),
        verbose=not bool(getattr(args, "quiet", False)),
    )
    _emit(f" elapsed_s={_now()-t0:.3f}", level=1, args=args)
    return 0


def _cmd_transport_matrix_v3(args: argparse.Namespace) -> int:
    t0 = _now()
    nml = read_sfincs_input(Path(args.input))
    _emit("################################################################", level=0, args=args)
    _emit(" sfincs_jax transport-matrix-v3", level=0, args=args)
    _emit(f" input={Path(args.input).resolve()}", level=0, args=args)
    _emit_namelist_summary(nml=nml, args=args)
    _emit_runtime_info(args=args)
    _emit(f" tol={args.tol} atol={args.atol} restart={args.restart} maxiter={args.maxiter} solve_method={args.solve_method}", level=1, args=args)
    result = solve_v3_transport_matrix_linear_gmres(
        nml=nml,
        tol=float(args.tol),
        atol=float(args.atol),
        restart=int(args.restart),
        maxiter=int(args.maxiter) if args.maxiter is not None else None,
        solve_method=str(args.solve_method),
        emit=lambda level, msg: _emit(msg, level=level, args=args),
    )

    out_tm = Path(args.out_matrix)
    out_tm.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_tm, np.asarray(result.transport_matrix))
    _emit(f" wrote transportMatrix -> {out_tm.resolve()}", level=0, args=args)

    if args.out_state_prefix is not None:
        pref = Path(args.out_state_prefix)
        pref.parent.mkdir(parents=True, exist_ok=True)
        for which_rhs, x in sorted(result.state_vectors_by_rhs.items()):
            p = pref.with_name(f"{pref.name}.whichRHS{which_rhs}.npy")
            np.save(p, np.asarray(x))
            _emit(f" wrote stateVector(whichRHS={which_rhs}) -> {p.resolve()}", level=1, args=args)

    for which_rhs, rn in sorted(result.residual_norms_by_rhs.items()):
        _emit(f" whichRHS={which_rhs} residual_norm={float(rn):.6e}", level=0, args=args)
    _emit(f" elapsed_s={_now()-t0:.3f}", level=1, args=args)
    return 0


def _cmd_dump_h5(args: argparse.Namespace) -> int:
    data = read_sfincs_h5(Path(args.sfincs_output))
    if args.keys_only:
        for k in sorted(data.keys()):
            print(k)
        return 0
    out = {k: v.tolist() if hasattr(v, "tolist") else v for k, v in data.items()}
    Path(args.out_json).write_text(json.dumps(out, indent=2, sort_keys=True))
    return 0


def _cmd_compare_h5(args: argparse.Namespace) -> int:
    tolerances = None
    if args.tolerances_json:
        with open(args.tolerances_json, "r", encoding="utf-8") as f:
            tolerances = json.load(f)
    results = compare_sfincs_outputs(
        a_path=Path(args.a),
        b_path=Path(args.b),
        rtol=float(args.rtol),
        atol=float(args.atol),
        tolerances=tolerances,
    )
    bad = [r for r in results if not r.ok]
    if args.show_all:
        for r in results:
            status = "OK" if r.ok else "FAIL"
            print(f"{status} {r.key}: max_abs={r.max_abs:.3e} max_rel={r.max_rel:.3e}")
    else:
        for r in bad[:50]:
            print(f"FAIL {r.key}: max_abs={r.max_abs:.3e} max_rel={r.max_rel:.3e}")
        if len(bad) > 50:
            print(f"... {len(bad) - 50} more failing keys omitted")
    return 0 if not bad else 2


def _cmd_scan_er(args: argparse.Namespace) -> int:
    t0 = _now()
    _emit("################################################################", level=0, args=args)
    _emit(" sfincs_jax scan-er", level=0, args=args)
    _emit(f" input={Path(args.input).resolve()}", level=0, args=args)
    _emit(f" out-dir={Path(args.out_dir).resolve()}", level=0, args=args)
    _emit_runtime_info(args=args)

    if args.values is not None:
        values = [float(x) for x in args.values]
    else:
        values = list(linspace_including_endpoints(float(args.min), float(args.max), int(args.n)))

    run_er_scan(
        input_namelist=Path(args.input),
        out_dir=Path(args.out_dir),
        values=values,
        compute_transport_matrix=bool(args.compute_transport_matrix),
        compute_solution=bool(getattr(args, "compute_solution", False)),
        emit=lambda level, msg: _emit(msg, level=level, args=args),
    )
    _emit(f" elapsed_s={_now()-t0:.3f}", level=1, args=args)
    return 0


def _cmd_ambipolar_solve(args: argparse.Namespace) -> int:
    t0 = _now()
    _emit("################################################################", level=0, args=args)
    _emit(" sfincs_jax ambipolar-solve", level=0, args=args)
    _emit(f" scan-dir={Path(args.scan_dir).resolve()}", level=0, args=args)
    _emit_runtime_info(args=args)

    res = solve_ambipolar_from_scan_dir(
        scan_dir=Path(args.scan_dir),
        write_pickle=True,
        write_json=True,
        n_fine=int(args.n_fine),
    )

    if res.roots_er.size == 0:
        _emit(" ambipolar-solve: no sign change found (no roots).", level=0, args=args)
    else:
        for i, (rv, re, rt) in enumerate(zip(res.roots_var, res.roots_er, res.root_types, strict=False), start=1):
            _emit(f" root[{i}] {res.var_name}={float(rv):.16g} Er={float(re):.16g} type={rt}", level=0, args=args)

    _emit(f" wrote {Path(args.scan_dir).resolve() / 'ambipolarSolutions.dat'}", level=1, args=args)
    _emit(f" wrote {Path(args.scan_dir).resolve() / 'ambipolarSolutions.json'}", level=2, args=args)
    _emit(f" elapsed_s={_now()-t0:.3f}", level=1, args=args)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sfincs_jax")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity (repeatable).")
    parser.add_argument("-q", "--quiet", action="store_true", help="Reduce output to a minimum.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_solve = sub.add_parser("solve-v3", help="Solve a supported v3 linear problem matrix-free and write stateVector.npy.")
    p_solve.add_argument("--input", required=True, help="Path to input.namelist")
    p_solve.add_argument("--out-state", default="stateVector.npy", help="Where to write the solution vector (NumPy .npy)")
    p_solve.add_argument("--tol", default="1e-10", help="GMRES relative tolerance")
    p_solve.add_argument("--atol", default="0.0", help="GMRES absolute tolerance")
    p_solve.add_argument("--restart", default="80", help="GMRES restart")
    p_solve.add_argument("--maxiter", default=None, help="GMRES maxiter (default: library default)")
    p_solve.add_argument("--solve-method", default="batched", help="JAX GMRES solve_method")
    p_solve.add_argument(
        "--which-rhs",
        default=None,
        help="For RHSMode=2/3 transport-matrix runs, select whichRHS (v3 loops over multiple RHS).",
    )
    p_solve.set_defaults(func=_cmd_solve_v3)

    p_scan = sub.add_parser(
        "scan-er",
        help="Run an Er (or dPhiHatd*) scan by writing sfincsOutput.h5 in multiple run directories.",
    )
    p_scan.add_argument("--input", required=True, help="Path to input.namelist (template).")
    p_scan.add_argument("--out-dir", required=True, help="Directory to create scan subdirectories inside.")
    p_scan.add_argument(
        "--compute-transport-matrix",
        action="store_true",
        help="Also compute RHSMode=2/3 transport-matrix outputs (slow).",
    )
    p_scan.add_argument(
        "--compute-solution",
        action="store_true",
        help="For RHSMode=1 runs, also solve and write solution-derived fields (may be slow).",
    )
    p_scan.add_argument("--min", default="-1.0", help="Minimum value (ignored if --values is provided).")
    p_scan.add_argument("--max", default="1.0", help="Maximum value (ignored if --values is provided).")
    p_scan.add_argument("--n", default="5", help="Number of points (ignored if --values is provided).")
    p_scan.add_argument("--values", default=None, nargs="+", help="Explicit list of values to use.")
    p_scan.set_defaults(func=_cmd_scan_er)

    p_ambi = sub.add_parser(
        "ambipolar-solve",
        help="Given an existing scan-er directory, solve for ambipolar Er roots and write ambipolarSolutions.dat.",
    )
    p_ambi.add_argument("--scan-dir", required=True, help="Scan directory produced by `sfincs_jax scan-er`.")
    p_ambi.add_argument("--n-fine", default="500", help="Number of fine-grid points for bracketing (default: 500).")
    p_ambi.set_defaults(func=_cmd_ambipolar_solve)

    p_run = sub.add_parser("run-fortran", help="Run the compiled Fortran SFINCS v3 executable.")
    p_run.add_argument("--input", required=True, help="Path to input.namelist")
    p_run.add_argument("--exe", default=None, help="Path to Fortran v3 sfincs executable")
    p_run.add_argument("--workdir", default=None, help="Directory to run in (default: temp dir)")
    p_run.set_defaults(func=_cmd_run_fortran)

    p_out = sub.add_parser("write-output", help="Write a SFINCS-style sfincsOutput.h5 using sfincs_jax.")
    p_out.add_argument("--input", required=True, help="Path to input.namelist")
    p_out.add_argument("--out", default="sfincsOutput.h5", help="Where to write sfincsOutput.h5")
    p_out.add_argument(
        "--no-fortran-layout",
        dest="fortran_layout",
        action="store_false",
        default=True,
        help="Disable Fortran-compatible array layout (not recommended for parity)",
    )
    p_out.add_argument(
        "--no-overwrite",
        dest="overwrite",
        action="store_false",
        default=True,
        help="Fail if output already exists",
    )
    p_out.add_argument(
        "--compute-transport-matrix",
        action="store_true",
        help="Force transport-matrix solves for RHSMode=2/3 (default: enabled when RHSMode=2/3).",
    )
    p_out.add_argument(
        "--compute-solution",
        action="store_true",
        help="Force RHSMode=1 solves (default: enabled when RHSMode=1).",
    )
    p_out.add_argument(
        "--geometry-only",
        action="store_true",
        help="Only write geometry/grid outputs (skip RHSMode=1 solve and RHSMode=2/3 transport-matrix loop).",
    )
    p_out.set_defaults(func=_cmd_write_output)

    p_tm = sub.add_parser("transport-matrix-v3", help="Solve RHSMode=2/3 transport-matrix systems and write transportMatrix.npy.")
    p_tm.add_argument("--input", required=True, help="Path to input.namelist (must have RHSMode=2 or 3)")
    p_tm.add_argument("--out-matrix", default="transportMatrix.npy", help="Where to write the transport matrix (NumPy .npy)")
    p_tm.add_argument(
        "--out-state-prefix",
        default=None,
        help="Optional prefix for saving solution vectors as <prefix>.whichRHS{k}.npy",
    )
    p_tm.add_argument("--tol", default="1e-10", help="GMRES relative tolerance")
    p_tm.add_argument("--atol", default="0.0", help="GMRES absolute tolerance")
    p_tm.add_argument("--restart", default="80", help="GMRES restart")
    p_tm.add_argument("--maxiter", default=None, help="GMRES maxiter (default: library default)")
    p_tm.add_argument("--solve-method", default="batched", help="JAX GMRES solve_method")
    p_tm.set_defaults(func=_cmd_transport_matrix_v3)

    p_dump = sub.add_parser("dump-h5", help="Dump SFINCS HDF5 output to JSON (small files only).")
    p_dump.add_argument("--sfincs-output", required=True, help="Path to sfincsOutput.h5")
    p_dump.add_argument("--out-json", required=True, help="Where to write JSON")
    p_dump.add_argument("--keys-only", action="store_true", help="Only print dataset names")
    p_dump.set_defaults(func=_cmd_dump_h5)

    p_cmp = sub.add_parser("compare-h5", help="Compare two SFINCS HDF5 output files.")
    p_cmp.add_argument("--a", required=True, help="First sfincsOutput.h5")
    p_cmp.add_argument("--b", required=True, help="Second sfincsOutput.h5")
    p_cmp.add_argument("--rtol", default="1e-12")
    p_cmp.add_argument("--atol", default="1e-12")
    p_cmp.add_argument("--tolerances-json", default=None, help="Optional JSON file of per-key tolerances")
    p_cmp.add_argument("--show-all", action="store_true", help="Print all keys (not just failures)")
    p_cmp.set_defaults(func=_cmd_compare_h5)

    p_pp = sub.add_parser(
        "postprocess-upstream",
        help="Run a vendored upstream v3 utils/ postprocessing script (best-effort, requires sfincsOutput.h5).",
    )
    p_pp.add_argument("--case-dir", required=True, help="Directory containing sfincsOutput.h5")
    p_pp.add_argument("--util", required=True, help="Upstream util script name (e.g. sfincsScanPlot_1)")
    p_pp.add_argument("--utils-dir", default=None, help="Override utils/ directory (else auto-detect / env var)")
    p_pp.add_argument("--interactive", action="store_true", help="Do not override input() (may hang in CI)")
    p_pp.add_argument(
        "util_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the upstream script (e.g. 'pdf'). Prefix with '--' to separate args.",
    )

    def _cmd_postprocess_upstream(args: argparse.Namespace) -> int:
        t0 = _now()
        _emit("################################################################", level=0, args=args)
        _emit(" sfincs_jax postprocess-upstream", level=0, args=args)
        _emit(f" case_dir={Path(args.case_dir).resolve()}", level=0, args=args)
        _emit(f" util={args.util}", level=0, args=args)
        if args.utils_dir is not None:
            _emit(f" utils_dir={Path(args.utils_dir).resolve()}", level=1, args=args)
        util_args = list(args.util_args or [])
        if util_args and util_args[0] == "--":
            util_args = util_args[1:]
        run_upstream_util(
            util=str(args.util),
            case_dir=Path(args.case_dir),
            args=util_args,
            utils_dir=Path(args.utils_dir) if args.utils_dir is not None else None,
            noninteractive=not bool(args.interactive),
            emit=lambda level, msg: _emit(msg, level=level, args=args),
        )
        _emit(f" elapsed_s={_now()-t0:.3f}", level=1, args=args)
        return 0

    p_pp.set_defaults(func=_cmd_postprocess_upstream)

    args = parser.parse_args(argv)
    return int(args.func(args))
