from __future__ import annotations

from types import SimpleNamespace

import pytest

import sfincs_jax.v3_driver as vd


def test_resolve_distributed_gmres_axis_allows_nondivisible_with_padding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SFINCS_JAX_GMRES_DISTRIBUTED", "theta")
    monkeypatch.setattr(vd.jax, "local_device_count", lambda: 8)
    op = SimpleNamespace(n_theta=65, n_zeta=65)
    assert vd._resolve_distributed_gmres_axis(op=op, emit=None) == "theta"

