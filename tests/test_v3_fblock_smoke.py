from __future__ import annotations

from pathlib import Path

import numpy as np
import jax.numpy as jnp

from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.v3_fblock import fblock_operator_from_namelist, matvec_v3_fblock_flat, solve_v3_fblock_gmres


def test_v3_fblock_matvec_and_gmres_smoke() -> None:
    """Smoke-test a matrix-free solve for the (partial) v3 F-block operator."""
    input_path = Path(__file__).parent / "ref" / "pas_1species_PAS_noEr_tiny.input.namelist"
    nml = read_sfincs_input(input_path)

    op = fblock_operator_from_namelist(nml=nml, identity_shift=1.0)
    rng = np.random.default_rng(0)

    x_true = jnp.asarray(rng.normal(size=(op.flat_size,)).astype(np.float64))
    b = matvec_v3_fblock_flat(op, x_true)

    result = solve_v3_fblock_gmres(op=op, b_flat=b, tol=1e-12, restart=40, maxiter=60)
    x = np.asarray(result.x)

    np.testing.assert_allclose(x, np.asarray(x_true), rtol=1e-7, atol=1e-7)
    assert float(result.residual_norm) < 1e-7

