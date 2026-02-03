"""RHSMode=2 transport matrices for Boozer `.bc` and filtered VMEC netCDF geometries.

This example showcases end-to-end parity-tested support for:
- `RHSMode=2` (3×3 transport matrix) with the upstream v3 `whichRHS = 1..3` loop
- `geometryScheme=11` (Boozer `.bc`) and `geometryScheme=5` (VMEC `wout_*.nc`, filtered)
- classical transport fluxes computed per-whichRHS (v3 `classicalTransport.F90`)

It reuses the frozen Fortran v3 fixtures in `tests/ref/` as a benchmark.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sfincs_jax.compare import compare_sfincs_outputs
from sfincs_jax.io import write_sfincs_jax_output_h5


def _run_case(base: str, *, out_dir: Path) -> None:
    input_path = _REPO_ROOT / "tests" / "ref" / f"{base}.input.namelist"
    ref_path = _REPO_ROOT / "tests" / "ref" / f"{base}.sfincsOutput.h5"
    out_path = out_dir / f"{base}.sfincsOutput_jax.h5"

    write_sfincs_jax_output_h5(
        input_namelist=input_path,
        output_path=out_path,
        compute_transport_matrix=True,
        overwrite=True,
    )

    # Print a short parity summary for key arrays.
    results = compare_sfincs_outputs(a_path=ref_path, b_path=out_path, rtol=1e-12, atol=5e-8)
    key_set = {
        "transportMatrix",
        "classicalHeatFlux_psiHat",
        "classicalParticleFlux_psiHat",
        "particleFlux_vm_psiHat",
        "heatFlux_vm_psiHat",
        "FSABFlow",
    }
    selected = [r for r in results if r.key in key_set]
    print(f"\n[{base}]")
    for r in sorted(selected, key=lambda x: x.key):
        status = "OK" if r.ok else "FAIL"
        print(f"  {status:4s} {r.key:28s} max_abs={r.max_abs:.3e} max_rel={r.max_rel:.3e}")

    # A stronger check on the transport matrix itself:
    # (Fortran HDF5 layout is transposed as read by Python; compare as written.)
    # The fixture is already in Python-read order.
    from sfincs_jax.io import read_sfincs_h5

    a = read_sfincs_h5(ref_path)
    b = read_sfincs_h5(out_path)
    np.testing.assert_allclose(
        np.asarray(a["transportMatrix"], dtype=np.float64),
        np.asarray(b["transportMatrix"], dtype=np.float64),
        rtol=0.0,
        atol=5e-8,
    )

    print(f"  wrote {out_path}")


def main() -> None:
    out_dir = _REPO_ROOT / "examples" / "transport" / "output" / "19_transport_matrix_rhsmode2"
    out_dir.mkdir(parents=True, exist_ok=True)

    for base in (
        "transportMatrix_PAS_tiny_rhsMode2_scheme11",
        "transportMatrix_PAS_tiny_rhsMode2_scheme5_filtered",
    ):
        _run_case(base, out_dir=out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()

