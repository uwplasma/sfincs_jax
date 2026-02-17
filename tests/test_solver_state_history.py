from __future__ import annotations

from pathlib import Path

import numpy as np
import jax.numpy as jnp

from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.v3_system import full_system_operator_from_namelist
from sfincs_jax.solver_state import save_krylov_state, load_krylov_state


def test_solver_state_history_roundtrip(tmp_path: Path) -> None:
    here = Path(__file__).parent
    input_path = here / "ref" / "pas_1species_PAS_noEr_tiny.input.namelist"
    nml = read_sfincs_input(input_path)
    op = full_system_operator_from_namelist(nml=nml)

    x1 = jnp.arange(int(op.total_size), dtype=jnp.float64) * 1e-6
    x2 = x1 + 1e-3
    out_path = tmp_path / "state.npz"

    save_krylov_state(path=out_path, op=op, x_full=x2, x_history=[x1, x2])
    state = load_krylov_state(path=out_path, op=op)

    assert state is not None
    assert "x_full" in state
    assert "x_history" in state
    np.testing.assert_allclose(state["x_full"], np.asarray(x2), rtol=0.0, atol=0.0)
    assert len(state["x_history"]) == 2
    np.testing.assert_allclose(state["x_history"][0], np.asarray(x1), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(state["x_history"][1], np.asarray(x2), rtol=0.0, atol=0.0)
