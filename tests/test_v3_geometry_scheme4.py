from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.v3 import geometry_from_namelist, grids_from_namelist


def _load_ref() -> dict:
    ref_path = Path(__file__).parent / "ref" / "quick_2species_FPCollisions_noEr.json"
    return json.loads(ref_path.read_text())


def test_quick_example_grids_and_geometry_scheme4_parity() -> None:
    ref = _load_ref()
    input_path = Path(__file__).parent / "ref" / "quick_2species_FPCollisions_noEr.input.namelist"
    nml = read_sfincs_input(input_path)

    grids = grids_from_namelist(nml)
    geom = geometry_from_namelist(nml=nml, grids=grids)

    np.testing.assert_allclose(np.asarray(grids.theta), np.asarray(ref["theta"]), rtol=0, atol=1e-13)
    np.testing.assert_allclose(np.asarray(grids.zeta), np.asarray(ref["zeta"]), rtol=0, atol=1e-13)
    np.testing.assert_allclose(np.asarray(grids.x), np.asarray(ref["x"]), rtol=0, atol=1e-13)

    # Fortran HDF5 output is stored as (Nzeta, Ntheta); our internal layout is (Ntheta, Nzeta).
    np.testing.assert_allclose(np.asarray(geom.b_hat).T, np.asarray(ref["BHat"]), rtol=0, atol=1e-13)
    np.testing.assert_allclose(np.asarray(geom.db_hat_dtheta).T, np.asarray(ref["dBHatdtheta"]), rtol=0, atol=1e-13)
    np.testing.assert_allclose(np.asarray(geom.db_hat_dzeta).T, np.asarray(ref["dBHatdzeta"]), rtol=0, atol=1e-13)
    np.testing.assert_allclose(np.asarray(geom.d_hat).T, np.asarray(ref["DHat"]), rtol=0, atol=1e-13)
    np.testing.assert_allclose(np.asarray(geom.b_hat_sup_theta).T, np.asarray(ref["BHat_sup_theta"]), rtol=0, atol=1e-13)
    np.testing.assert_allclose(np.asarray(geom.b_hat_sup_zeta).T, np.asarray(ref["BHat_sup_zeta"]), rtol=0, atol=1e-13)
    np.testing.assert_allclose(np.asarray(geom.b_hat_sub_theta).T, np.asarray(ref["BHat_sub_theta"]), rtol=0, atol=1e-13)
    np.testing.assert_allclose(np.asarray(geom.b_hat_sub_zeta).T, np.asarray(ref["BHat_sub_zeta"]), rtol=0, atol=1e-13)

