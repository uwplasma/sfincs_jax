#!/usr/bin/env python
from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _localize_equilibrium(input_path: Path) -> None:
    # Avoid importing `sfincs_jax` at module import time so this script can be inspected without deps.
    import sys  # noqa: PLC0415

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from sfincs_jax.io import localize_equilibrium_file_in_place  # noqa: PLC0415

    localize_equilibrium_file_in_place(input_namelist=input_path, overwrite=False)


def _rename_sfincs_binary(src: Path, *, base: str, out_dir: Path) -> Path:
    name = src.name
    if "_whichMatrix_" in name:
        # e.g. sfincsBinary_iteration_000_whichMatrix_3
        which = name.split("_whichMatrix_")[-1]
        return out_dir / f"{base}.whichMatrix_{which}.petscbin"
    if name.endswith("_stateVector"):
        return out_dir / f"{base}.stateVector.petscbin"
    if name.endswith("_residual"):
        return out_dir / f"{base}.residual.petscbin"
    if name.endswith("_rhs"):
        return out_dir / f"{base}.rhs.petscbin"
    # Keep the original stem to avoid losing information for less common outputs.
    return out_dir / f"{base}.{name}.petscbin"


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate frozen Fortran v3 fixtures under tests/ref/ (maintainer tool).")
    ap.add_argument("--fortran-exe", type=Path, required=True, help="Path to the Fortran v3 `sfincs` executable.")
    ap.add_argument("--input", type=Path, required=True, help="Path to an input.namelist template.")
    ap.add_argument("--base", required=True, help="Fixture base name (e.g. pas_1species_PAS_noEr_tiny_scheme11).")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "tests" / "ref")
    ap.add_argument("--keep-workdir", action="store_true", help="Keep the temporary work directory.")
    args = ap.parse_args()

    fortran_exe: Path = args.fortran_exe
    if not fortran_exe.exists():
        raise SystemExit(f"Fortran executable does not exist: {fortran_exe}")

    input_src: Path = args.input
    if not input_src.exists():
        raise SystemExit(f"Input file does not exist: {input_src}")

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    tmpdir_obj = tempfile.TemporaryDirectory(prefix="sfincs_fortran_fixture_")
    workdir = Path(tmpdir_obj.name)
    try:
        w_input = workdir / "input.namelist"
        _copy_file(input_src, w_input)
        _localize_equilibrium(w_input)

        cmd = [str(fortran_exe)]
        proc = subprocess.run(
            cmd,
            cwd=str(workdir),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        log_path = out_dir / f"{args.base}.sfincs.log"
        log_path.write_text(proc.stdout, encoding="utf-8")
        if proc.returncode != 0:
            raise SystemExit(f"Fortran run failed (exit={proc.returncode}). See: {log_path}")

        out_h5 = workdir / "sfincsOutput.h5"
        if out_h5.exists():
            _copy_file(out_h5, out_dir / f"{args.base}.sfincsOutput.h5")

        # Copy PETSc binary outputs.
        for p in sorted(workdir.glob("sfincsBinary_iteration_*")):
            if p.name.endswith(".info"):
                continue
            _copy_file(p, _rename_sfincs_binary(p, base=str(args.base), out_dir=out_dir))

        print(f"Wrote fixtures for {args.base} to {out_dir}")
        return 0
    finally:
        if args.keep_workdir:
            print(f"Kept workdir: {workdir}")
            tmpdir_obj.cleanup = lambda: None  # type: ignore[assignment]
        else:
            tmpdir_obj.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
