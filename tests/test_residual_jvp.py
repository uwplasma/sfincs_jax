from __future__ import annotations

from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.residual import V3FBlockLinearSystem
from sfincs_jax.v3_fblock import fblock_operator_from_namelist, matvec_v3_fblock_flat


def test_v3_fblock_residual_jvp_matches_matvec() -> None:
    """Jacobian-vector products of the linear residual match the operator matvec."""
    here = Path(__file__).parent
    input_path = here / "ref" / "pas_1species_PAS_noEr_tiny.input.namelist"

    nml = read_sfincs_input(input_path)
    op = fblock_operator_from_namelist(nml=nml, identity_shift=0.0)

    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.normal(size=(op.flat_size,)).astype(np.float64))
    v = jnp.asarray(rng.normal(size=(op.flat_size,)).astype(np.float64))
    b = jnp.asarray(rng.normal(size=(op.flat_size,)).astype(np.float64))

    sys = V3FBlockLinearSystem(op=op, b_flat=b)

    r, jv = sys.jvp(x, v)
    np.testing.assert_allclose(np.asarray(r), np.asarray(matvec_v3_fblock_flat(op, x) - b), rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(jv), np.asarray(matvec_v3_fblock_flat(op, v)), rtol=0, atol=1e-12)

    # Also verify JIT compatibility.
    f_jit = jax.jit(sys.residual)
    jv_jit = jax.jit(sys.jacobian_matvec)
    np.testing.assert_allclose(np.asarray(f_jit(x)), np.asarray(r), rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(jv_jit(v)), np.asarray(jv), rtol=0, atol=1e-12)

