from __future__ import annotations

from pathlib import Path

import numpy as np

from sfincs_jax.compare import compare_sfincs_outputs
from sfincs_jax.io import read_sfincs_h5, write_sfincs_jax_output_h5


def test_output_scheme1_matches_fortran_fixture(tmp_path: Path) -> None:
    here = Path(__file__).parent
    input_path = here / "ref" / "output_scheme1_tokamak_1species_tiny.input.namelist"
    fortran_path = here / "ref" / "output_scheme1_tokamak_1species_tiny.sfincsOutput.h5"
    assert fortran_path.exists(), f"Missing Fortran fixture: {fortran_path}"

    out_path = tmp_path / "sfincsOutput.h5"
    write_sfincs_jax_output_h5(input_namelist=input_path, output_path=out_path)

    a = read_sfincs_h5(out_path)
    b = read_sfincs_h5(fortran_path)
    assert a["input.namelist"] == b["input.namelist"]

    keys = [
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
        "RHSMode",
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
        "constraintScheme",
        "includeXDotTerm",
        "includeElectricFieldTermInXiDot",
        "useDKESExBDrift",
        "force0RadialCurrentInEquilibrium",
        "includePhi1",
        "includePhi1InKineticEquation",
        "includePhi1InCollisionOperator",
        "includeTemperatureEquilibrationTerm",
        "include_fDivVE_Term",
        "withAdiabatic",
        "withNBIspec",
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

    results = compare_sfincs_outputs(a_path=out_path, b_path=fortran_path, keys=keys, rtol=0, atol=1e-12)
    bad = [r for r in results if not r.ok]
    assert not bad, f"Mismatched keys: {[b.key for b in bad]}"

    uhat = compare_sfincs_outputs(a_path=out_path, b_path=fortran_path, keys=["uHat"], rtol=0, atol=1e-8)
    bad_uhat = [r for r in uhat if not r.ok]
    assert not bad_uhat, f"Mismatched keys (uHat): {[b.key for b in bad_uhat]}"


def test_output_scheme1_rhs_mode_is_preserved(tmp_path: Path) -> None:
    input_path = Path(__file__).parent / "ref" / "output_scheme1_tokamak_1species_tiny.input.namelist"
    out_path = tmp_path / "sfincsOutput.h5"
    write_sfincs_jax_output_h5(input_namelist=input_path, output_path=out_path)
    data = read_sfincs_h5(out_path)
    assert int(np.asarray(data["RHSMode"])) == 1

