from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
import time
from pathlib import Path

import jax

from sfincs_jax.io import localize_equilibrium_file_in_place
from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.v3_driver import solve_v3_full_system_linear_gmres


_ROOT = Path(__file__).resolve().parents[2]
_REDUCED_INPUTS = _ROOT / "tests" / "reduced_inputs"


def _parse_profile_line(line: str) -> dict[str, float | str | None]:
    out: dict[str, float | str | None] = {"raw": line.strip()}
    if not line.startswith("profiling:"):
        return out
    try:
        _, rest = line.split("profiling:", 1)
        parts = rest.strip().split()
        label = parts[0]
        out["label"] = label
        for token in parts[1:]:
            if "=" not in token:
                continue
            k, v = token.split("=", 1)
            v = v.strip()
            if v.lower() in {"na", "n/a", "none"}:
                out[k] = None
                continue
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = None
    except Exception:
        return out
    return out


def _prep_input(input_path: Path) -> Path:
    tmpdir = Path(tempfile.mkdtemp(prefix="sfincs_jax_profile_"))
    dst = tmpdir / "input.namelist"
    shutil.copy2(input_path, dst)
    localize_equilibrium_file_in_place(input_namelist=dst, overwrite=False)
    return dst


def _rhs_count(rhs_mode: int) -> int:
    if rhs_mode == 2:
        return 3
    if rhs_mode == 3:
        return 2
    return 1


def _run_case(input_path: Path, *, source_input: Path) -> dict[str, object]:
    logs: list[str] = []

    def emit(level: int, msg: str) -> None:
        if msg.startswith("profiling:"):
            logs.append(msg)

    nml = read_sfincs_input(input_path)
    rhs_mode = int(nml.group("general").get("RHSMODE", 1))
    count = _rhs_count(rhs_mode)

    os.environ.setdefault("SFINCS_JAX_FORTRAN_STDOUT", "0")
    os.environ.setdefault("SFINCS_JAX_SOLVER_ITER_STATS", "0")

    t0 = time.perf_counter()
    for which_rhs in range(1, count + 1):
        res = solve_v3_full_system_linear_gmres(
            nml=nml,
            which_rhs=which_rhs if rhs_mode in {2, 3} else None,
            tol=1e-10,
            emit=emit,
        )
        _ = jax.block_until_ready(res.x)
    wall_s = time.perf_counter() - t0

    entries = [_parse_profile_line(line) for line in logs]
    return {
        "input": str(input_path),
        "source_input": str(source_input),
        "case": source_input.stem.replace(".input", ""),
        "rhs_mode": rhs_mode,
        "rhs_count": count,
        "wall_s": wall_s,
        "entries": entries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Solve-only profiling for reduced inputs.")
    parser.add_argument("--pattern", default=None, help="Regex to filter input filenames.")
    parser.add_argument("--limit", type=int, default=None, help="Max inputs to profile.")
    parser.add_argument(
        "--out-json",
        default=str(_ROOT / "examples" / "performance" / "output" / "reduced_profiles_solve_only.json"),
        help="JSON output path.",
    )
    args = parser.parse_args()

    os.environ["SFINCS_JAX_PROFILE"] = "1"

    inputs = sorted(_REDUCED_INPUTS.glob("*.input.namelist"))
    if args.pattern:
        pat = re.compile(args.pattern)
        inputs = [p for p in inputs if pat.search(p.name)]
    if args.limit is not None:
        inputs = inputs[: int(args.limit)]

    results: list[dict[str, object]] = []
    for idx, input_path in enumerate(inputs, start=1):
        print(f"[{idx}/{len(inputs)}] {input_path.name}")
        work_input = _prep_input(input_path)
        results.append(_run_case(work_input, source_input=input_path))

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
