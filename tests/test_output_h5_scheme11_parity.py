from __future__ import annotations

from pathlib import Path

from sfincs_jax.compare import compare_sfincs_outputs
from sfincs_jax.io import read_sfincs_h5, write_sfincs_jax_output_h5


def test_output_scheme11_matches_fortran_fixture(tmp_path: Path) -> None:
    here = Path(__file__).parent
    input_path = here / "ref" / "output_scheme11_1species_tiny.input.namelist"
    fortran_path = here / "ref" / "output_scheme11_1species_tiny.sfincsOutput.h5"
    assert fortran_path.exists(), f"Missing Fortran fixture: {fortran_path}"

    out_path = tmp_path / "sfincsOutput.h5"
    write_sfincs_jax_output_h5(input_namelist=input_path, output_path=out_path)

    # Compare the embedded input file text exactly.
    a = read_sfincs_h5(out_path)
    b = read_sfincs_h5(fortran_path)
    assert a["input.namelist"] == b["input.namelist"]

    keys_strict = [
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
        "xPotentialsGridScheme",
        "NxPotentialsPerVth",
        "pointAtX0",
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
        "constraintScheme",
        "reusePreconditioner",
        "preconditioner_species",
        "preconditioner_x",
        "preconditioner_x_min_L",
        "preconditioner_xi",
        "preconditioner_theta",
        "preconditioner_zeta",
        "preconditioner_magnetic_drifts_max_L",
        "RHSMode",
        "useIterativeLinearSolver",
        "NIterations",
        "finished",
        "includeXDotTerm",
        "includeElectricFieldTermInXiDot",
        "useDKESExBDrift",
        "export_full_f",
        "export_delta_f",
        "force0RadialCurrentInEquilibrium",
        "includePhi1",
        "includePhi1InKineticEquation",
        "includePhi1InCollisionOperator",
        "includeTemperatureEquilibrationTerm",
        "include_fDivVE_Term",
        "withAdiabatic",
        "withNBIspec",
        "classicalParticleFluxNoPhi1_psiHat",
        "classicalParticleFluxNoPhi1_psiN",
        "classicalParticleFluxNoPhi1_rHat",
        "classicalParticleFluxNoPhi1_rN",
        "classicalHeatFluxNoPhi1_psiHat",
        "classicalHeatFluxNoPhi1_psiN",
        "classicalHeatFluxNoPhi1_rHat",
        "classicalHeatFluxNoPhi1_rN",
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

    results = compare_sfincs_outputs(a_path=out_path, b_path=fortran_path, keys=keys_strict, rtol=0, atol=1e-12)
    bad = [r for r in results if not r.ok]
    assert not bad, f"Mismatched keys: {[b.key for b in bad]}"

    # `uHat` depends on many transcendental evaluations (cos/sin) and long reductions.
    uhat = compare_sfincs_outputs(a_path=out_path, b_path=fortran_path, keys=["uHat"], rtol=0, atol=1e-8)
    bad_uhat = [r for r in uhat if not r.ok]
    assert not bad_uhat, f"Mismatched keys (uHat): {[b.key for b in bad_uhat]}"

