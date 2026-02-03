from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import time

from sfincs_jax.compare import compare_sfincs_outputs
from sfincs_jax.fortran import run_sfincs_fortran
from sfincs_jax.io import write_sfincs_jax_output_h5
from sfincs_jax.namelist import read_sfincs_input


@dataclass(frozen=True)
class CaseResult:
    case_dir: Path
    input_namelist: Path
    ok: bool
    error: str | None = None
    sfincs_jax_output: Path | None = None
    fortran_output: Path | None = None


def _iter_input_namelists(root: Path) -> list[Path]:
    return sorted(root.rglob("input.namelist"))


def _matches(pattern: str | None, p: Path) -> bool:
    if not pattern:
        return True
    return re.search(pattern, str(p), flags=re.IGNORECASE) is not None


def main() -> int:
    parser = argparse.ArgumentParser(description="Run sfincs_jax on the vendored upstream Fortran v3 example suite.")
    parser.add_argument("--root", default=Path(__file__).resolve().parent, type=Path, help="Suite root directory")
    parser.add_argument("--pattern", default=None, help="Regex filter on case path (case-insensitive)")
    parser.add_argument("--limit", default=None, type=int, help="Stop after N cases")
    parser.add_argument("--write-output", action="store_true", help="Run `sfincs_jax write-output` for each case")
    parser.add_argument(
        "--compute-transport-matrix",
        action="store_true",
        help="If a case has RHSMode=2/3, also compute `transportMatrix` (can be slow).",
    )
    parser.add_argument("--compare-fortran", action="store_true", help="Also run the Fortran executable and compare outputs")
    parser.add_argument("--fortran-exe", default=None, type=Path, help="Path to the compiled Fortran v3 `sfincs` executable")
    parser.add_argument("--rtol", default=1e-10, type=float)
    parser.add_argument("--atol", default=1e-10, type=float)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    args = parser.parse_args()

    root: Path = args.root
    if not root.exists():
        raise SystemExit(f"--root does not exist: {root}")

    inputs = [p for p in _iter_input_namelists(root) if _matches(args.pattern, p)]
    if args.limit is not None:
        inputs = inputs[: int(args.limit)]

    if not args.write_output and not args.compare_fortran:
        raise SystemExit("Nothing to do: pass --write-output and/or --compare-fortran.")
    if args.compare_fortran and args.fortran_exe is None:
        raise SystemExit("--compare-fortran requires --fortran-exe")

    results: list[CaseResult] = []
    t0 = time.time()

    for i, input_path in enumerate(inputs, start=1):
        case_dir = input_path.parent
        if args.verbose:
            print(f"[{i}/{len(inputs)}] {case_dir}")

        try:
            nml = read_sfincs_input(input_path)
            rhs_mode = int(nml.group("general").get("RHSMODE", 1))

            jax_out = None
            if args.write_output:
                jax_out = case_dir / "sfincsOutput_jax.h5"
                if not args.dry_run:
                    write_sfincs_jax_output_h5(
                        input_namelist=input_path,
                        output_path=jax_out,
                        overwrite=True,
                        compute_transport_matrix=bool(args.compute_transport_matrix and rhs_mode in {2, 3}),
                    )

            f_out = None
            if args.compare_fortran:
                if args.dry_run:
                    f_out = case_dir / "sfincsOutput_fortran.h5"
                else:
                    # Run in the case directory to match upstream relative paths.
                    f_out = run_sfincs_fortran(input_namelist=input_path, exe=args.fortran_exe, workdir=case_dir)

            if args.compare_fortran and jax_out is not None and f_out is not None and not args.dry_run:
                compare_sfincs_outputs(a=jax_out, b=f_out, rtol=float(args.rtol), atol=float(args.atol))

            results.append(CaseResult(case_dir=case_dir, input_namelist=input_path, ok=True, sfincs_jax_output=jax_out, fortran_output=f_out))

        except Exception as e:  # noqa: BLE001
            results.append(CaseResult(case_dir=case_dir, input_namelist=input_path, ok=False, error=str(e)))
            if args.verbose:
                print(f"  FAIL: {e}")

    ok = [r for r in results if r.ok]
    bad = [r for r in results if not r.ok]
    dt = time.time() - t0

    print(f"cases_total={len(results)} ok={len(ok)} failed={len(bad)} elapsed_s={dt:.2f}")
    if bad:
        print("Failures (first 30):")
        for r in bad[:30]:
            print(f"- {r.case_dir}: {r.error}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

