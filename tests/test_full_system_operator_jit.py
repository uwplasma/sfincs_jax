from __future__ import annotations

from pathlib import Path

import numpy as np
import jax.numpy as jnp

from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.petsc_binary import read_petsc_vec
from sfincs_jax.v3_system import apply_v3_full_system_operator, apply_v3_full_system_operator_jit, full_system_operator_from_namelist


def test_full_system_operator_can_jit_compile() -> None:
    """Regression test: V3FullSystemOperator must be a JAX PyTree usable under jit."""
    here = Path(__file__).parent
    input_path = here / "ref" / "pas_1species_PAS_noEr_tiny_scheme1.input.namelist"
    vec_path = here / "ref" / "pas_1species_PAS_noEr_tiny_scheme1.stateVector.petscbin"

    nml = read_sfincs_input(input_path)
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)
    x = jnp.asarray(read_petsc_vec(vec_path).values)

    y = np.asarray(apply_v3_full_system_operator(op, x))
    y_jit = np.asarray(apply_v3_full_system_operator_jit(op, x))
    np.testing.assert_allclose(y_jit, y, rtol=0, atol=1e-15)
