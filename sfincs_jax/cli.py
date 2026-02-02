from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .compare import compare_sfincs_outputs
from .fortran import run_sfincs_fortran
from .io import read_sfincs_h5, write_sfincs_jax_output_h5
from .namelist import read_sfincs_input
from .v3_driver import solve_v3_full_system_linear_gmres, solve_v3_transport_matrix_linear_gmres


def _cmd_solve_v3(args: argparse.Namespace) -> int:
    nml = read_sfincs_input(Path(args.input))
    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        which_rhs=int(args.which_rhs) if args.which_rhs is not None else None,
        tol=float(args.tol),
        atol=float(args.atol),
        restart=int(args.restart),
        maxiter=int(args.maxiter) if args.maxiter is not None else None,
        solve_method=str(args.solve_method),
    )
    out_state = Path(args.out_state)
    out_state.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_state, np.asarray(result.x))
    print(str(out_state.resolve()))
    print(f"residual_norm={float(result.residual_norm):.6e}")
    return 0


def _cmd_run_fortran(args: argparse.Namespace) -> int:
    output_path = run_sfincs_fortran(
        input_namelist=Path(args.input),
        exe=Path(args.exe) if args.exe else None,
        workdir=Path(args.workdir) if args.workdir else None,
    )
    print(str(output_path))
    return 0


def _cmd_write_output(args: argparse.Namespace) -> int:
    out_path = write_sfincs_jax_output_h5(
        input_namelist=Path(args.input),
        output_path=Path(args.out),
        fortran_layout=bool(args.fortran_layout),
        overwrite=bool(args.overwrite),
        compute_transport_matrix=bool(args.compute_transport_matrix),
    )
    print(str(out_path))
    return 0


def _cmd_transport_matrix_v3(args: argparse.Namespace) -> int:
    nml = read_sfincs_input(Path(args.input))
    result = solve_v3_transport_matrix_linear_gmres(
        nml=nml,
        tol=float(args.tol),
        atol=float(args.atol),
        restart=int(args.restart),
        maxiter=int(args.maxiter) if args.maxiter is not None else None,
        solve_method=str(args.solve_method),
    )

    out_tm = Path(args.out_matrix)
    out_tm.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_tm, np.asarray(result.transport_matrix))
    print(str(out_tm.resolve()))

    if args.out_state_prefix is not None:
        pref = Path(args.out_state_prefix)
        pref.parent.mkdir(parents=True, exist_ok=True)
        for which_rhs, x in sorted(result.state_vectors_by_rhs.items()):
            p = pref.with_name(f"{pref.name}.whichRHS{which_rhs}.npy")
            np.save(p, np.asarray(x))
            print(str(p.resolve()))

    for which_rhs, rn in sorted(result.residual_norms_by_rhs.items()):
        print(f"whichRHS={which_rhs} residual_norm={float(rn):.6e}")
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
    results = compare_sfincs_outputs(
        a_path=Path(args.a),
        b_path=Path(args.b),
        rtol=float(args.rtol),
        atol=float(args.atol),
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sfincs_jax")
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
        help="For RHSMode=2/3 runs, also perform the whichRHS solves and write `transportMatrix` (may be slow).",
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
    p_cmp.add_argument("--show-all", action="store_true", help="Print all keys (not just failures)")
    p_cmp.set_defaults(func=_cmd_compare_h5)

    args = parser.parse_args(argv)
    return int(args.func(args))
