from __future__ import annotations

from pathlib import Path

import numpy as np
import jax.numpy as jnp

from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.petsc_binary import read_petsc_vec
from sfincs_jax.v3_system import full_system_operator_from_namelist, residual_v3_full_system


def _run_case(name: str) -> None:
    here = Path(__file__).parent
    input_path = here / "ref" / f"{name}.input.namelist"
    vec_path = here / "ref" / f"{name}.stateVector.petscbin"
    residual_path = here / "ref" / f"{name}.residual.petscbin"

    nml = read_sfincs_input(input_path)
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)

    x = read_petsc_vec(vec_path).values
    r_ref = read_petsc_vec(residual_path).values
    assert x.shape == (op.total_size,)
    assert r_ref.shape == (op.total_size,)

    r = np.asarray(residual_v3_full_system(op, jnp.asarray(x)))

    np.testing.assert_allclose(r, r_ref, rtol=0, atol=3e-12)


def test_full_system_residual_pas_tiny_matches_fortran() -> None:
    _run_case("pas_1species_PAS_noEr_tiny")

def test_full_system_residual_pas_tiny_scheme5_matches_fortran() -> None:
    _run_case("pas_1species_PAS_noEr_tiny_scheme5")

def test_full_system_residual_pas_tiny_scheme5_with_phi1_linear_matches_fortran() -> None:
    _run_case("pas_1species_PAS_noEr_tiny_scheme5_withPhi1_linear")

def test_full_system_residual_pas_tiny_scheme1_matches_fortran() -> None:
    _run_case("pas_1species_PAS_noEr_tiny_scheme1")

def test_full_system_residual_pas_tiny_scheme11_matches_fortran() -> None:
    _run_case("pas_1species_PAS_noEr_tiny_scheme11")

def test_full_system_residual_pas_tiny_scheme12_matches_fortran() -> None:
    _run_case("pas_1species_PAS_noEr_tiny_scheme12")


def test_full_system_residual_fp_2species_matches_fortran() -> None:
    _run_case("quick_2species_FPCollisions_noEr")


def test_full_system_residual_pas_tiny_with_phi1_linear_matches_fortran() -> None:
    _run_case("pas_1species_PAS_noEr_tiny_withPhi1_linear")


def test_full_system_residual_pas_tiny_with_phi1_in_kinetic_matches_fortran() -> None:
    _run_case("pas_1species_PAS_noEr_tiny_withPhi1_inKinetic_linear")


def test_full_system_residual_fp_tiny_with_phi1_in_collision_matches_fortran() -> None:
    _run_case("fp_1species_FPCollisions_noEr_tiny_withPhi1_inCollision")
