#!/usr/bin/env python
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import shutil
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sfincs_jax.compare import compare_sfincs_outputs
from sfincs_jax.fortran import run_sfincs_fortran
from sfincs_jax.io import read_sfincs_h5, write_sfincs_jax_output_h5
from sfincs_jax.namelist import read_sfincs_input


@dataclass(frozen=True)
class CaseSummary:
    case: str
    case_dir: str
    rhs_mode: int
    geometry_scheme: int | None
    ok_write_output: bool
    ok_fortran: bool
    ok_compare_common: bool
    n_keys_fortran: int
    n_keys_jax: int
    n_keys_common: int
    n_missing_in_jax: int
    n_mismatch_common: int
    note: str | None = None


def _iter_inputs(examples_root: Path) -> list[Path]:
    return sorted(examples_root.rglob("input.namelist"))


def _patch_equilibrium_file_in_place(input_path: Path) -> None:
    """Patch `equilibriumFile` to be runnable from a copied workdir."""
    from sfincs_jax.io import localize_equilibrium_file_in_place  # noqa: PLC0415

    localize_equilibrium_file_in_place(input_namelist=input_path, overwrite=False)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run sfincs_jax vs Fortran on the upstream v3 example suite (best-effort).")
    ap.add_argument("--examples-root", type=Path, default=Path(__file__).resolve().parents[1] / "examples" / "sfincs_examples")
    ap.add_argument("--out-root", type=Path, default=Path("/tmp") / "sfincs_jax_suite_compare")
    ap.add_argument("--pattern", default=None, help="Regex filter on case path")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--fortran-exe", type=Path, default=None, help="Path to Fortran v3 `sfincs` executable (optional)")
    ap.add_argument("--rtol", type=float, default=1e-10)
    ap.add_argument("--atol", type=float, default=1e-10)
    ap.add_argument("--compute-transport-matrix", action="store_true", help="Enable for RHSMode=2/3 cases (slow)")
    ap.add_argument("-v", "--verbose", action="count", default=0)
    args = ap.parse_args()

    examples_root: Path = args.examples_root
    if not examples_root.exists():
        raise SystemExit(f"examples-root does not exist: {examples_root}")

    out_root: Path = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    inputs = _iter_inputs(examples_root)
    if args.pattern:
        rx = re.compile(str(args.pattern), flags=re.IGNORECASE)
        inputs = [p for p in inputs if rx.search(str(p))]
    if args.limit is not None:
        inputs = inputs[: int(args.limit)]

    summaries: list[CaseSummary] = []
    t0 = time.time()

    for i, input_path in enumerate(inputs, start=1):
        case_dir = input_path.parent
        case = case_dir.name
        workdir = out_root / case
        if workdir.exists():
            shutil.rmtree(workdir)
        shutil.copytree(case_dir, workdir)

        w_input = workdir / "input.namelist"
        _patch_equilibrium_file_in_place(w_input)

        nml = read_sfincs_input(w_input)
        general = nml.group("general")
        geom = nml.group("geometryParameters")
        rhs_mode = int(general.get("RHSMODE", 1))
        geometry_scheme = int(geom.get("GEOMETRYSCHEME")) if "GEOMETRYSCHEME" in geom else None

        if args.verbose:
            print(f"[{i}/{len(inputs)}] {case} (geometryScheme={geometry_scheme} RHSMode={rhs_mode})")

        jax_path = workdir / "sfincsOutput_jax.h5"
        try:
            write_sfincs_jax_output_h5(
                input_namelist=w_input,
                output_path=jax_path,
                compute_transport_matrix=bool(args.compute_transport_matrix and rhs_mode in {2, 3}),
            )
            ok_write = True
            note = None
        except Exception as e:  # noqa: BLE001
            ok_write = False
            note = f"sfincs_jax write-output failed: {type(e).__name__}: {e}"

        fortran_ok = False
        ok_compare_common = False
        n_keys_fortran = 0
        n_keys_jax = 0
        n_keys_common = 0
        n_missing = 0
        n_mismatch = 0

        if ok_write and args.fortran_exe is not None:
            try:
                out_fortran = run_sfincs_fortran(input_namelist=w_input, exe=args.fortran_exe, workdir=workdir)
                fortran_path = workdir / "sfincsOutput_fortran.h5"
                if fortran_path.exists():
                    fortran_path.unlink()
                out_fortran.rename(fortran_path)
                fortran_ok = True

                a = read_sfincs_h5(jax_path)
                b = read_sfincs_h5(fortran_path)
                keys_a = set(a.keys())
                keys_b = set(b.keys())
                n_keys_jax = len(keys_a)
                n_keys_fortran = len(keys_b)
                common = sorted(keys_a & keys_b)
                missing = sorted(keys_b - keys_a)
                n_keys_common = len(common)
                n_missing = len(missing)
                results = compare_sfincs_outputs(a_path=jax_path, b_path=fortran_path, keys=common, rtol=args.rtol, atol=args.atol)
                bad = [r for r in results if not r.ok]
                n_mismatch = len(bad)
                ok_compare_common = (n_mismatch == 0)
                if bad and args.verbose >= 2:
                    for r in bad[:20]:
                        print(f"  mismatch {r.key}: max_abs={r.max_abs:.3e} max_rel={r.max_rel:.3e}")
                if missing and args.verbose >= 2:
                    print(f"  missing_in_jax (first 20): {missing[:20]}")
            except Exception as e:  # noqa: BLE001
                note = (note + " | " if note else "") + f"Fortran run/compare failed: {type(e).__name__}: {e}"

        if ok_write and args.fortran_exe is None:
            a = read_sfincs_h5(jax_path)
            n_keys_jax = len(a.keys())

        summaries.append(
            CaseSummary(
                case=case,
                case_dir=str(case_dir),
                rhs_mode=rhs_mode,
                geometry_scheme=geometry_scheme,
                ok_write_output=ok_write,
                ok_fortran=fortran_ok,
                ok_compare_common=ok_compare_common,
                n_keys_fortran=n_keys_fortran,
                n_keys_jax=n_keys_jax,
                n_keys_common=n_keys_common,
                n_missing_in_jax=n_missing,
                n_mismatch_common=n_mismatch,
                note=note,
            )
        )

    dt = time.time() - t0
    ok = [s for s in summaries if s.ok_write_output]
    print(f"cases_total={len(summaries)} ok_write_output={len(ok)} elapsed_s={dt:.2f}")
    if args.fortran_exe is not None:
        ok_fortran = [s for s in summaries if s.ok_fortran]
        ok_common = [s for s in summaries if s.ok_fortran and s.ok_compare_common]
        print(f"ok_fortran={len(ok_fortran)} ok_compare_common={len(ok_common)}")

    # Emit a machine-readable summary for CI/log harvesting.
    out_json = out_root / "suite_summary.json"
    out_json.write_text(json.dumps([asdict(s) for s in summaries], indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
