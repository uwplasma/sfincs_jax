from __future__ import annotations

from pathlib import Path
import pickle

import numpy as np

from sfincs_jax.ambipolar import solve_ambipolar_from_scan_dir
from sfincs_jax.io import read_sfincs_h5
from sfincs_jax.scans import run_er_scan


def test_er_scan_writes_outputs_and_ambipolar_solve_runs(tmp_path: Path) -> None:
    """End-to-end smoke test for `scan-er` + `ambipolar-solve` workflow.

    This test is intentionally small but exercises:
      - scan directory layout + `!ss` metadata
      - per-run `sfincsOutput.h5` creation with RHSMode=1 solution-derived fields
      - ambipolar root postprocessing compatible with upstream `sfincsScanPlot_5`
    """
    here = Path(__file__).parent
    input_path = here / "ref" / "pas_1species_PAS_noEr_tiny_scheme11.input.namelist"
    scan_dir = tmp_path / "scan"

    values = [1.0e-3, 0.0, -1.0e-3]
    scan = run_er_scan(
        input_namelist=input_path,
        out_dir=scan_dir,
        values=values,
        compute_solution=True,
        compute_transport_matrix=False,
    )

    assert scan.scan_dir.exists()
    assert (scan.scan_dir / "input.namelist").exists()
    assert scan.variable == "Er"
    assert len(scan.run_dirs) == len(values)
    assert len(scan.outputs) == len(values)

    # Ensure outputs exist and contain the patched Er value.
    for v, out_h5 in zip(scan.values, scan.outputs, strict=True):
        assert out_h5.exists()
        d = read_sfincs_h5(out_h5)
        np.testing.assert_allclose(float(np.asarray(d["Er"]).reshape(())), float(v), rtol=0.0, atol=0.0)
        assert "particleFlux_vm_rHat" in d
        assert "Zs" in d

    res = solve_ambipolar_from_scan_dir(scan_dir=scan_dir, write_pickle=True, write_json=True, n_fine=200)

    pkl_path = scan_dir / "ambipolarSolutions.dat"
    json_path = scan_dir / "ambipolarSolutions.json"
    assert pkl_path.exists()
    assert json_path.exists()

    payload = pickle.loads(pkl_path.read_bytes())
    assert "roots" in payload
    assert "ylabels" in payload
    assert payload["numQuantities"] == len(payload["ylabels"])

    # Roots may or may not exist depending on this tiny fixture; ensure consistency if they do.
    if res.roots_var.size:
        np.testing.assert_allclose(np.asarray(payload["roots"], dtype=np.float64), np.asarray(res.roots_var, dtype=np.float64))

