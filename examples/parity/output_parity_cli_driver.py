"""Compare a `sfincs_jax` output file (generated via CLI) against a Fortran fixture.

Run:
  python examples/parity/output_parity_cli_driver.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from sfincs_jax.compare import compare_sfincs_outputs


def main() -> int:
    input_path = repo_root / "tests" / "ref" / "output_scheme4_1species_tiny.input.namelist"
    fortran_path = repo_root / "tests" / "ref" / "output_scheme4_1species_tiny.sfincsOutput.h5"
    out_dir = Path(__file__).with_suffix("").parent / "output"
    out_path = out_dir / "sfincsOutput_jax_cli.h5"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "sfincs_jax",
        "write-output",
        "--input",
        str(input_path),
        "--out",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)

    keys = [
        "theta",
        "zeta",
        "x",
        "BHat",
        "DHat",
        "dBHatdtheta",
        "dBHatdzeta",
        "VPrimeHat",
        "FSABHat2",
    ]
    results = compare_sfincs_outputs(a_path=out_path, b_path=fortran_path, keys=keys, rtol=0, atol=1e-12)
    bad = [r for r in results if not r.ok]
    if bad:
        for r in bad:
            print(f"FAIL {r.key}: max_abs={r.max_abs:.3e} max_rel={r.max_rel:.3e}")
        return 2

    print(f"OK: {len(results)} keys match.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
