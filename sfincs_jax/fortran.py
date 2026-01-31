from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path


def default_fortran_exe() -> Path | None:
    env = os.environ.get("SFINCS_FORTRAN_EXE")
    return Path(env) if env else None


def run_sfincs_fortran(
    *,
    input_namelist: Path,
    exe: Path | None = None,
    workdir: Path | None = None,
    env: dict[str, str] | None = None,
) -> Path:
    """Run the compiled Fortran SFINCS v3 executable.

    Notes
    -----
    - The Fortran executable is **not** shipped as part of this package.
    - The executable is expected to read `input.namelist` from the working directory and
      write `sfincsOutput.h5` there.
    """
    input_namelist = input_namelist.resolve()
    if not input_namelist.exists():
        raise FileNotFoundError(str(input_namelist))

    exe = (exe or default_fortran_exe())
    if exe is None:
        raise ValueError(
            "Fortran executable not specified. Pass `exe=...` or set SFINCS_FORTRAN_EXE."
        )
    exe = exe.resolve()
    if not exe.exists():
        raise FileNotFoundError(str(exe))

    if workdir is None:
        workdir = Path(tempfile.mkdtemp(prefix="sfincs_fortran_run_"))
    workdir = workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(input_namelist, workdir / "input.namelist")

    log_path = workdir / "sfincs.log"
    with log_path.open("w") as log:
        merged_env = os.environ.copy()
        if env:
            merged_env.update(env)
        subprocess.run(
            [str(exe)],
            cwd=str(workdir),
            stdout=log,
            stderr=subprocess.STDOUT,
            env=merged_env,
            check=True,
        )

    output_path = workdir / "sfincsOutput.h5"
    if not output_path.exists():
        raise RuntimeError(
            f"Fortran run finished but did not create {output_path}. See {log_path}."
        )
    return output_path
