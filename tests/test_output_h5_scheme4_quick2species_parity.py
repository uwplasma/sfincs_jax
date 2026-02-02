from __future__ import annotations

from pathlib import Path

import numpy as np

from sfincs_jax.compare import compare_sfincs_outputs
from sfincs_jax.io import read_sfincs_h5, write_sfincs_jax_output_h5


def test_output_scheme4_quick2species_matches_fortran_fixture(tmp_path: Path) -> None:
    """Parity test for scheme-4 output-writing on a 2-species v3 example output.

    This fixture is taken from the upstream v3 example `quick_2species_FPCollisions_noEr`,
    which exercises multi-species metadata and (small) grids.

    Only a subset of datasets are compared here: inputs + geometry + basic integrals.
    Full solver outputs (e.g. flows/fluxes, exported f, etc.) are not yet written by `sfincs_jax`.
    """
    here = Path(__file__).parent
    input_path = here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist"
    fortran_path = here / "ref" / "output_scheme4_2species_quick.sfincsOutput.h5"
    assert fortran_path.exists(), f"Missing Fortran fixture: {fortran_path}"

    out_path = tmp_path / "sfincsOutput.h5"
    write_sfincs_jax_output_h5(input_namelist=input_path, output_path=out_path)

    # Compare embedded input file text exactly.
    a = read_sfincs_h5(out_path)
    b = read_sfincs_h5(fortran_path)
    assert a["input.namelist"] == b["input.namelist"]

    keys = [
        "integerToRepresentFalse",
        "integerToRepresentTrue",
        "Nspecies",
        "Ntheta",
        "Nzeta",
        "Nxi",
        "NL",
        "Nx",
        "theta",
        "zeta",
        "x",
        "Nxi_for_x",
        "geometryScheme",
        "thetaDerivativeScheme",
        "zetaDerivativeScheme",
        "xGridScheme",
        "xGrid_k",
        "xMax",
        "Nxi_for_x_option",
        "solverTolerance",
        "Delta",
        "alpha",
        "nu_n",
        "Er",
        "dPhiHatdpsiHat",
        "dPhiHatdpsiN",
        "dPhiHatdrHat",
        "dPhiHatdrN",
        "psiAHat",
        "aHat",
        "psiHat",
        "psiN",
        "rHat",
        "rN",
        "inputRadialCoordinate",
        "inputRadialCoordinateForGradients",
        "coordinateSystem",
        "rippleScale",
        "gpsiHatpsiHat",
        "diotadpsiHat",
        "EParallelHat",
        "collisionOperator",
        "includeXDotTerm",
        "includeElectricFieldTermInXiDot",
        "useDKESExBDrift",
        "includePhi1",
        "NPeriods",
        "B0OverBBar",
        "iota",
        "GHat",
        "IHat",
        "VPrimeHat",
        "FSABHat2",
        "Zs",
        "mHats",
        "THats",
        "nHats",
        "dnHatdrHat",
        "dnHatdrN",
        "dnHatdpsiN",
        "dnHatdpsiHat",
        "dTHatdrHat",
        "dTHatdrN",
        "dTHatdpsiN",
        "dTHatdpsiHat",
        "DHat",
        "BHat",
        "dBHatdpsiHat",
        "dBHatdtheta",
        "dBHatdzeta",
        "BDotCurlB",
        "BHat_sub_psi",
        "dBHat_sub_psi_dtheta",
        "dBHat_sub_psi_dzeta",
        "BHat_sub_theta",
        "dBHat_sub_theta_dpsiHat",
        "dBHat_sub_theta_dzeta",
        "BHat_sub_zeta",
        "dBHat_sub_zeta_dpsiHat",
        "dBHat_sub_zeta_dtheta",
        "BHat_sup_theta",
        "dBHat_sup_theta_dpsiHat",
        "dBHat_sup_theta_dzeta",
        "BHat_sup_zeta",
        "dBHat_sup_zeta_dpsiHat",
        "dBHat_sup_zeta_dtheta",
    ]

    # Ensure the compared keys exist in both files (avoid silently skipping missing datasets).
    for k in keys:
        assert k in a, f"Missing key in sfincs_jax output: {k}"
        assert k in b, f"Missing key in Fortran fixture: {k}"

    results = compare_sfincs_outputs(a_path=out_path, b_path=fortran_path, keys=keys, rtol=0, atol=1e-12)
    bad = [r for r in results if not r.ok]
    assert not bad, f"Mismatched keys: {[b.key for b in bad]}"

    # `uHat` is sensitive to transcendental/reduction differences; compare with a looser tolerance.
    assert "uHat" in a and "uHat" in b
    uhat = compare_sfincs_outputs(a_path=out_path, b_path=fortran_path, keys=["uHat"], rtol=0, atol=1e-8)
    bad_uhat = [r for r in uhat if not r.ok]
    assert not bad_uhat, f"Mismatched keys (uHat): {[b.key for b in bad_uhat]}"

    # Basic sanity: this is truly a 2-species case.
    assert int(np.asarray(a["Nspecies"])) == 2

