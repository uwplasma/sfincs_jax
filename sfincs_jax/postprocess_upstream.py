from __future__ import annotations

from collections.abc import Callable, Sequence
import os
from pathlib import Path
import subprocess
import sys
import textwrap


def find_upstream_utils_dir(*, override: Path | None = None) -> Path:
    """Locate a directory containing the upstream v3 `utils/` postprocessing scripts.

    Search order:
    1) explicit `override`
    2) env var `SFINCS_JAX_UPSTREAM_UTILS_DIR`
    3) repo checkout layout: `<repo>/examples/sfincs_examples/utils`
    """
    if override is not None:
        p = Path(override)
        if not p.exists():
            raise FileNotFoundError(f"utils dir does not exist: {p}")
        return p

    env = os.environ.get("SFINCS_JAX_UPSTREAM_UTILS_DIR", "").strip()
    if env:
        p = Path(env)
        if not p.exists():
            raise FileNotFoundError(f"SFINCS_JAX_UPSTREAM_UTILS_DIR does not exist: {p}")
        return p

    repo_root = Path(__file__).resolve().parents[1]
    candidate = repo_root / "examples" / "sfincs_examples" / "utils"
    if candidate.exists():
        return candidate

    raise FileNotFoundError(
        "Could not locate upstream v3 utils/ scripts. Set SFINCS_JAX_UPSTREAM_UTILS_DIR or run from a repo checkout."
    )


def run_upstream_util(
    *,
    util: str,
    case_dir: Path,
    args: Sequence[str] = (),
    utils_dir: Path | None = None,
    noninteractive: bool = True,
    emit: Callable[[int, str], None] | None = None,
) -> None:
    """Run an upstream v3 `utils/<util>` script in a non-interactive, reproducible way.

    Notes
    -----
    Many upstream scripts assume:
    - output file name is `sfincsOutput.h5` in the current working directory
    - an interactive console (`input()` pauses)
    - a GUI matplotlib backend

    This wrapper sets `MPLBACKEND=Agg` and (optionally) overrides `builtins.input` to avoid hangs.
    """
    utils_dir = find_upstream_utils_dir(override=utils_dir)
    script_path = (utils_dir / util).resolve()
    if not script_path.exists():
        raise FileNotFoundError(f"Upstream util not found: {script_path}")

    case_dir = Path(case_dir).resolve()
    if not case_dir.exists():
        raise FileNotFoundError(f"case_dir does not exist: {case_dir}")

    # Execute using runpy in a clean subprocess so the script sees a normal `sys.argv`.
    harness = textwrap.dedent(
        """
        import builtins
        import runpy
        import sys

        noninteractive = bool(int(sys.argv[1]))
        script = sys.argv[2]
        argv = sys.argv[3:]
        if noninteractive:
            builtins.input = lambda *a, **k: ""
        sys.argv = [script] + argv
        runpy.run_path(script, run_name="__main__")
        """
    ).strip()

    cmd = [sys.executable, "-c", harness, "1" if noninteractive else "0", str(script_path), *list(args)]
    env = dict(os.environ)
    env.setdefault("MPLBACKEND", "Agg")

    if emit is not None:
        emit(0, f"postprocess-upstream: running {script_path.name} in {case_dir}")
    subprocess.run(cmd, cwd=str(case_dir), env=env, check=True)

