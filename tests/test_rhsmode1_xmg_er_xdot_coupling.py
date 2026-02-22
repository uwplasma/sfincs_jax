from __future__ import annotations

from pathlib import Path

import numpy as np

from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.v3_system import full_system_operator_from_namelist


def test_rhsmode1_xmg_includes_er_xdot_x_coupling_for_pas(monkeypatch) -> None:
    """RHSMode=1 xmg preconditioner must capture Er xDot dense-x coupling for PAS-only cases.

    Without including Er xDot in the coarse-x matrix, PAS-only systems are diagonal in x and
    xmg devolves to a pointwise scaling. With Er xDot enabled, the coarse inverse should
    become non-diagonal in x.
    """
    input_path = (
        Path(__file__).parent
        / "reduced_inputs"
        / "tokamak_1species_PASCollisions_withEr_fullTrajectories.input.namelist"
    )
    assert input_path.exists()
    nml = read_sfincs_input(input_path)
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)

    assert op.fblock.pas is not None
    assert op.fblock.fp is None
    assert op.fblock.er_xdot is not None

    import sfincs_jax.v3_driver as vd

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XMG_STRIDE", "2")
    vd._build_rhsmode1_xmg_preconditioner(op=op)

    # For PAS+Er, the xmg builder should route to the stable x-upwind preconditioner
    # (dense ddx-based x-block inversions can be extremely ill-conditioned).
    cache_key = vd._rhsmode1_precond_cache_key(op, "xupwind")
    cached = vd._RHSMODE1_XUPWIND_PRECOND_CACHE[cache_key]
    sub = np.asarray(cached.sub)[0, :, 0]  # (X,) for s=0, L=0
    assert float(np.max(np.abs(sub[1:]))) > 0.0
