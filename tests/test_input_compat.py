from __future__ import annotations

from pathlib import Path

import numpy as np

from sfincs_jax.input_compat import (
    effective_equilibrium_file,
    effective_r_n_wish,
    effective_use_iterative_linear_solver,
    infer_phi_input_radial_coordinate_for_gradients,
    infer_input_radial_coordinate_for_gradients,
    infer_species_input_radial_coordinate_for_gradients,
)
from sfincs_jax.io import localize_equilibrium_file_in_place, sfincs_jax_output_dict
from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.v3 import grids_from_namelist
from sfincs_jax.v3_system import full_system_operator_from_namelist


def test_infer_input_radial_coordinate_for_gradients_legacy_multispecies_psin() -> None:
    input_path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "upstream"
        / "fortran_multispecies"
        / "quick_2species_FPCollisions_noEr"
        / "input.namelist"
    )
    nml = read_sfincs_input(input_path)
    assert (
        infer_input_radial_coordinate_for_gradients(
            geom_params=nml.group("geometryParameters"),
            species_params=nml.group("speciesParameters"),
            phys_params=nml.group("physicsParameters"),
            default=4,
    )
        == 1
    )


def test_infer_gradient_coordinates_legacy_mixed_species_and_er() -> None:
    input_path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "sfincs_examples"
        / "geometryScheme5_3species_loRes"
        / "input.namelist"
    )
    nml = read_sfincs_input(input_path)
    geom = nml.group("geometryParameters")
    species = nml.group("speciesParameters")
    phys = nml.group("physicsParameters")
    assert infer_species_input_radial_coordinate_for_gradients(geom_params=geom, species_params=species, default=4) == 2
    assert infer_phi_input_radial_coordinate_for_gradients(geom_params=geom, phys_params=phys, default=4) == 4
    assert (
        infer_input_radial_coordinate_for_gradients(
            geom_params=geom,
            species_params=species,
            phys_params=phys,
            default=4,
        )
        == 4
    )


def test_effective_equilibrium_file_supports_legacy_jgboozer_alias() -> None:
    input_path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "upstream"
        / "fortran_multispecies"
        / "HSX_FPCollisions_DKESTrajectories"
        / "input.namelist"
    )
    nml = read_sfincs_input(input_path)
    equilibrium_file = effective_equilibrium_file(geom_params=nml.group("geometryParameters"))
    assert str(equilibrium_file).strip('"').strip("'").endswith("hsx3free.bc")


def test_effective_r_n_wish_supports_legacy_normradius_alias() -> None:
    input_path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "upstream"
        / "fortran_multispecies"
        / "HSX_FPCollisions_DKESTrajectories"
        / "input.namelist"
    )
    nml = read_sfincs_input(input_path)
    assert effective_r_n_wish(geom_params=nml.group("geometryParameters"), default=0.5) == 0.22


def test_effective_use_iterative_linear_solver_supports_legacy_alias() -> None:
    input_path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "upstream"
        / "fortran_multispecies"
        / "inductiveE_noEr"
        / "input.namelist"
    )
    nml = read_sfincs_input(input_path)
    assert effective_use_iterative_linear_solver(other_params=nml.group("otherNumericalParameters"), default=0) == 1


def test_sfincs_output_dict_uses_legacy_gradient_coordinate_inference() -> None:
    input_path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "upstream"
        / "fortran_multispecies"
        / "inductiveE_noEr"
        / "input.namelist"
    )
    nml = read_sfincs_input(input_path)
    grids = grids_from_namelist(nml)
    data = sfincs_jax_output_dict(nml=nml, grids=grids)
    assert int(np.asarray(data["inputRadialCoordinateForGradients"]).reshape(-1)[0]) == 1


def test_sfincs_output_dict_uses_legacy_normradius_wish_for_bc_geometry() -> None:
    input_path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "upstream"
        / "fortran_multispecies"
        / "HSX_FPCollisions_DKESTrajectories"
        / "input.namelist"
    )
    nml = read_sfincs_input(input_path)
    grids = grids_from_namelist(nml)
    data = sfincs_jax_output_dict(nml=nml, grids=grids)
    assert np.isclose(float(np.asarray(data["rN"]).reshape(-1)[0]), 0.22703830459418076)


def test_localize_equilibrium_file_in_place_patches_legacy_boozer_key(tmp_path: Path) -> None:
    source_input = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "upstream"
        / "fortran_multispecies"
        / "HSX_FPCollisions_DKESTrajectories"
        / "input.namelist"
    )
    dst_input = tmp_path / "input.namelist"
    dst_input.write_text(source_input.read_text(encoding="utf-8"), encoding="utf-8")
    localized = localize_equilibrium_file_in_place(input_namelist=dst_input, overwrite=True)
    assert localized is not None
    patched = dst_input.read_text(encoding="utf-8")
    assert f'JGboozer_file = "{localized.name}"' in patched


def test_sfincs_output_dict_preserves_legacy_er_coordinate_and_value() -> None:
    input_path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "sfincs_examples"
        / "geometryScheme5_3species_loRes"
        / "input.namelist"
    )
    nml = read_sfincs_input(input_path)
    grids = grids_from_namelist(nml)
    data = sfincs_jax_output_dict(nml=nml, grids=grids)
    psi_a_hat = float(np.asarray(data["psiAHat"]).reshape(-1)[0])
    a_hat = float(np.asarray(data["aHat"]).reshape(-1)[0])
    r_n = float(np.asarray(data["rN"]).reshape(-1)[0])
    er = float(np.asarray(data["Er"]).reshape(-1)[0])
    ddrhat2ddpsihat = a_hat / (2.0 * psi_a_hat * r_n)
    assert int(np.asarray(data["inputRadialCoordinateForGradients"]).reshape(-1)[0]) == 4
    assert np.isclose(er, -8.5897)
    assert np.isclose(float(np.asarray(data["dPhiHatdpsiHat"]).reshape(-1)[0]), ddrhat2ddpsihat * (-er))


def test_full_system_operator_uses_split_legacy_gradient_coordinates() -> None:
    input_path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "sfincs_examples"
        / "geometryScheme5_3species_loRes"
        / "input.namelist"
    )
    nml = read_sfincs_input(input_path)
    op = full_system_operator_from_namelist(nml=nml)
    grids = grids_from_namelist(nml)
    data = sfincs_jax_output_dict(nml=nml, grids=grids)
    psi_a_hat = float(np.asarray(data["psiAHat"]).reshape(-1)[0])
    a_hat = float(np.asarray(data["aHat"]).reshape(-1)[0])
    r_n = float(np.asarray(data["rN"]).reshape(-1)[0])
    ddrhat2ddpsihat = a_hat / (2.0 * psi_a_hat * r_n)
    assert np.isclose(float(op.dphi_hat_dpsi_hat), ddrhat2ddpsihat * 8.5897)
    assert np.allclose(
        np.asarray(op.dn_hat_dpsi_hat),
        ddrhat2ddpsihat * np.asarray([-15.0, -15.5, -0.025]),
    )
