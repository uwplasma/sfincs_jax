from __future__ import annotations

from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.petsc_binary import read_petsc_vec
from sfincs_jax.residual import V3FullLinearSystem
from sfincs_jax.v3_system import apply_v3_full_system_operator, full_system_operator_from_namelist


def test_full_system_residual_and_jvp_pas_tiny() -> None:
    """Residual and JVP are consistent for the full operator in a tiny PAS case."""
    here = Path(__file__).parent
    input_path = here / "ref" / "pas_1species_PAS_noEr_tiny.input.namelist"
    vec_path = here / "ref" / "pas_1species_PAS_noEr_tiny.stateVector.petscbin"

    nml = read_sfincs_input(input_path)
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)

    x_ref = jnp.asarray(read_petsc_vec(vec_path).values)
    b = apply_v3_full_system_operator(op, x_ref)
    sys = V3FullLinearSystem(op=op, b_full=b)

    r = np.asarray(sys.residual(x_ref))
    np.testing.assert_allclose(r, 0.0, rtol=0, atol=1e-12)

    key = jax.random.key(0)
    v = jax.random.normal(key, shape=(op.total_size,), dtype=jnp.float64)
    r0, jvp = sys.jvp(x_ref, v)

    np.testing.assert_allclose(np.asarray(r0), 0.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(jvp), np.asarray(sys.jacobian_matvec(v)), rtol=0, atol=1e-12)

