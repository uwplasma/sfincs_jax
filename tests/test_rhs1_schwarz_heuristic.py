from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sfincs_jax.namelist import read_sfincs_input
import sfincs_jax.v3_driver as vd


def test_rhs1_auto_prefers_theta_schwarz_when_sharded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Auto RHSMode=1 preconditioner should pick Schwarz on sharded large-system path."""
    input_path = Path(__file__).parent / "ref" / "pas_1species_PAS_noEr_tiny_scheme1.input.namelist"
    assert input_path.exists()
    nml = read_sfincs_input(input_path)

    logs: list[str] = []

    def emit(level: int, msg: str) -> None:
        logs.append(msg)

    monkeypatch.setenv("SFINCS_JAX_FORTRAN_STDOUT", "0")
    monkeypatch.setenv("SFINCS_JAX_SOLVER_ITER_STATS", "0")
    monkeypatch.setenv("SFINCS_JAX_MATVEC_SHARD_AXIS", "theta")
    monkeypatch.setenv("SFINCS_JAX_AUTO_SHARD", "0")
    monkeypatch.setenv("SFINCS_JAX_GMRES_DISTRIBUTED", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SCHWARZ_AUTO_MIN", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_STRONG_PRECOND", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_DENSE_FALLBACK_MAX", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_TZ_MAX", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_TZ_PRECOND_MAX", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPECIES_BLOCK_MAX", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_COLLISION_PRECOND_MIN", "1000000000")
    monkeypatch.setattr(vd.jax, "device_count", lambda: 2)

    res = vd.solve_v3_full_system_linear_gmres(nml=nml, tol=1e-8, emit=emit)
    assert np.isfinite(float(res.residual_norm))
    assert any("building RHSMode=1 preconditioner=theta_schwarz" in msg for msg in logs)
