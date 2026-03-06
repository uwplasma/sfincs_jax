from __future__ import annotations

from pathlib import Path

import numpy as np

from sfincs_jax.input_compat import (
    effective_equilibrium_file,
    effective_r_n_wish,
    effective_use_iterative_linear_solver,
    infer_input_radial_coordinate_for_gradients,
)
from sfincs_jax.io import localize_equilibrium_file_in_place, sfincs_jax_output_dict
from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.v3 import grids_from_namelist


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
