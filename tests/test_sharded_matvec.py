from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.v3_driver import full_system_operator_from_namelist
from sfincs_jax.v3_system import apply_v3_full_system_operator_cached


def test_sharded_matvec_fallback_matches_unsharded(monkeypatch) -> None:
    """Sharded matvec should reduce to unsharded behavior on single-device hosts."""
    here = Path(__file__).parent
    input_path = here / "ref" / "transportMatrix_PAS_tiny_rhsMode2_scheme2.input.namelist"
    nml = read_sfincs_input(input_path)
    op = full_system_operator_from_namelist(nml=nml)
    x = np.arange(int(op.total_size), dtype=np.float64)

    monkeypatch.setenv("SFINCS_JAX_MATVEC_SHARD_AXIS", "zeta")
    y_shard = np.asarray(apply_v3_full_system_operator_cached(op, x))
    monkeypatch.setenv("SFINCS_JAX_MATVEC_SHARD_AXIS", "off")
    y_base = np.asarray(apply_v3_full_system_operator_cached(op, x))

    np.testing.assert_allclose(y_shard, y_base, rtol=1e-12, atol=1e-12)
