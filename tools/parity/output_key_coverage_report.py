"""Inventory Fortran NAMELIST controls or report missing DKX output datasets.

This script compares:
  - a `sfincsOutput.h5` written by `dkx write-output`
  - a frozen Fortran v3 `sfincsOutput.h5` fixture

and prints keys that exist in the Fortran file but are not written by `dkx` yet.

Run:
  python tools/parity/output_key_coverage_report.py
  python tools/parity/output_key_coverage_report.py --namelist-source /path/to/readInput.F90
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

def namelist_controls(source: str) -> dict:
    """Inventory free-form NAMELIST member identifiers, not compiled support.

    Comments/blank lines do not end continuations. Preprocessor branch text is
    retained without evaluating macros. Unsupported declaration syntax and
    duplicate members fail rather than silently producing an incomplete count.
    """
    groups, guards = {}, []
    group = None
    for number, raw in enumerate(source.splitlines(), 1):
        line = raw.split("!", 1)[0].strip()
        if not line:
            continue
        if re.match(r"#\s*(if|ifdef|ifndef)\b", line):
            guards.append(line)
            continue
        if re.match(r"#\s*(else|elif)\b", line):
            if not guards:
                raise ValueError(f"unmatched preprocessor branch at line {number}")
            guards[-1] += " -> " + line
            continue
        if re.match(r"#\s*endif\b", line):
            if not guards:
                raise ValueError(f"unmatched #endif at line {number}")
            guards.pop()
            continue
        match = re.match(r"namelist\s*/\s*(\w+)\s*/(.*)", line, re.I)
        if match:
            if group is not None:
                raise ValueError(f"unfinished namelist before line {number}")
            group, line = match.groups()
            groups.setdefault(group, [])
        if group is None:
            continue
        continuation = line.rstrip().endswith("&")
        for member in line.strip().strip("&").split(","):
            name = member.strip()
            if not name:
                continue
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
                raise ValueError(f"unsupported namelist member at line {number}: {name}")
            if any(item["name"].lower() == name.lower() for item in groups[group]):
                raise ValueError(f"duplicate namelist member {group}.{name}")
            groups[group].append({"name": name, "line": number, "conditions": guards.copy()})
        if not continuation:
            group = None
    if group is not None or guards:
        raise ValueError("unfinished namelist or preprocessor conditional")
    if not groups:
        raise ValueError("no supported NAMELIST declarations found")
    return groups


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namelist-source", type=Path, help="Inventory a pinned readInput.F90 without running DKX")
    args = parser.parse_args()
    if args.namelist_source is not None:
        raw = args.namelist_source.read_bytes()
        groups = namelist_controls(raw.decode("utf-8"))
        print(json.dumps({"source_sha256": hashlib.sha256(raw).hexdigest(),
                          "scope": "declared members; preprocessor conditions are not evaluated",
                          "counts": {key: len(value) for key, value in groups.items()},
                          "groups": groups}, indent=2))
        return 0
    from dkx.api import write_output
    from dkx.io import read_sfincs_h5

    input_path = repo_root / "tests" / "ref" / "output_scheme4_1species_tiny.input.namelist"
    fortran_path = repo_root / "tests" / "ref" / "output_scheme4_1species_tiny.sfincsOutput.h5"

    out_dir = Path(__file__).with_suffix("").parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    jax_path = out_dir / "sfincsOutput_jax.h5"
    write_output(input_path, jax_path)

    a = read_sfincs_h5(jax_path)
    b = read_sfincs_h5(fortran_path)

    missing = sorted(set(b.keys()) - set(a.keys()))
    extra = sorted(set(a.keys()) - set(b.keys()))

    print(f"Fortran keys: {len(b)}")
    print(f"JAX keys:    {len(a)}")
    print(f"Missing in JAX: {len(missing)}")
    for k in missing:
        print(f"  {k}")
    if extra:
        print(f"Extra in JAX: {len(extra)}")
        for k in extra:
            print(f"  {k}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
