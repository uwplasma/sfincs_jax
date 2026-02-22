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
    cache_key = vd._rhsmode1_precond_cache_key(op, "xmg_2")
    cached = vd._RHSMODE1_XMG_PRECOND_CACHE[cache_key]

    mat = np.asarray(cached.coarse_inv)[0, 0]  # (Xc,Xc) for s=0, L=0
    offdiag = mat - np.diag(np.diag(mat))
    assert float(np.max(np.abs(offdiag))) > 0.0

    # PAS+Er xDot includes ΔL=±2 couplings; the xmg preconditioner should build a small
    # coupled (L,x) coarse inverse for low-L modes so these couplings are captured.
    assert cached.coarse_inv_lblock is not None
    lblock = int(cached.lblock)
    assert lblock >= 3
    coarse_idx = np.asarray(cached.coarse_idx)
    n_coarse = int(coarse_idx.shape[0])
    mat_lblock = np.asarray(cached.coarse_inv_lblock)[0]  # (Lb*Xc,Lb*Xc)
    sub02 = mat_lblock[0:n_coarse, 2 * n_coarse : 3 * n_coarse]
    assert float(np.max(np.abs(sub02))) > 0.0
