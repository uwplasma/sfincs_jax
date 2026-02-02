from __future__ import annotations

from pathlib import Path

import numpy as np

from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.v3 import geometry_from_namelist, grids_from_namelist


def test_geometry_scheme11_matches_fortran_fixture() -> None:
    """Parity-check `geometryScheme=11` Boozer geometry against a frozen Fortran fixture.

    The fixture arrays were extracted from SFINCS v3 for a small (Ntheta=5, Nzeta=5) case.
    """
    here = Path(__file__).parent
    input_path = here / "ref" / "magdrift_1species_tiny.input.namelist"
    fixture_path = here / "ref" / "magdrift_1species_tiny.geometry.npz"

    nml = read_sfincs_input(input_path)
    grids = grids_from_namelist(nml)
    geom = geometry_from_namelist(nml=nml, grids=grids)

    mapping = {
        "BHat": np.asarray(geom.b_hat),
        "dBHatdtheta": np.asarray(geom.db_hat_dtheta),
        "dBHatdzeta": np.asarray(geom.db_hat_dzeta),
        "dBHatdpsiHat": np.asarray(geom.db_hat_dpsi_hat),
        "DHat": np.asarray(geom.d_hat),
        "BHat_sub_psi": np.asarray(geom.b_hat_sub_psi),
        "dBHat_sub_psi_dtheta": np.asarray(geom.db_hat_sub_psi_dtheta),
        "dBHat_sub_psi_dzeta": np.asarray(geom.db_hat_sub_psi_dzeta),
        "BHat_sub_theta": np.asarray(geom.b_hat_sub_theta),
        "BHat_sub_zeta": np.asarray(geom.b_hat_sub_zeta),
        "dBHat_sub_theta_dpsiHat": np.asarray(geom.db_hat_sub_theta_dpsi_hat),
        "dBHat_sub_zeta_dpsiHat": np.asarray(geom.db_hat_sub_zeta_dpsi_hat),
    }

    with np.load(fixture_path) as z:
        for key, arr in mapping.items():
            ref = np.asarray(z[key], dtype=np.float64)
            if ref.ndim == 2:
                ref = ref.T
            got = np.asarray(arr, dtype=np.float64)
            assert got.shape == ref.shape
            # Geometric quantities include many transcendental evaluations and long reductions,
            # so we allow small platform-dependent differences.
            np.testing.assert_allclose(got, ref, rtol=0.0, atol=1e-8, err_msg=f"Mismatch for {key}")
