from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import jax.numpy as jnp

from sfincs_jax.v3_fblock import apply_v3_fblock_operator, fblock_operator_from_namelist
from sfincs_jax.namelist import read_sfincs_input


def test_fused_fblock_matvec_matches_unfused() -> None:
    """Fused scan path for collisionless/drift terms must match the unfused path."""
    here = Path(__file__).parent
    input_path = here / "ref" / "pas_1species_PAS_noEr_tiny.input.namelist"
    nml = read_sfincs_input(input_path)
    op = fblock_operator_from_namelist(nml=nml)

    f = jnp.arange(int(op.flat_size), dtype=jnp.float64).reshape(op.f_shape) * 1e-6

    prev = os.environ.get("SFINCS_JAX_FUSED_MATVEC")
    try:
        os.environ["SFINCS_JAX_FUSED_MATVEC"] = "0"
        out_ref = apply_v3_fblock_operator(op, f)

        os.environ["SFINCS_JAX_FUSED_MATVEC"] = "1"
        out_fused = apply_v3_fblock_operator(op, f)
    finally:
        if prev is None:
            os.environ.pop("SFINCS_JAX_FUSED_MATVEC", None)
        else:
            os.environ["SFINCS_JAX_FUSED_MATVEC"] = prev

    np.testing.assert_allclose(
        np.asarray(out_fused, dtype=np.float64),
        np.asarray(out_ref, dtype=np.float64),
        rtol=0.0,
        atol=1e-12,
    )
