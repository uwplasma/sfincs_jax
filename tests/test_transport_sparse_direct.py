from __future__ import annotations

from types import SimpleNamespace

from sfincs_jax.v3_driver import (
    _transport_dense_backend_allowed,
    _transport_sparse_direct_rescue_allowed,
    _transport_sparse_direct_rescue_first,
)


def _op(*, rhs_mode: int = 2, has_fp: bool = True, has_phi1: bool = False):
    return SimpleNamespace(
        rhs_mode=rhs_mode,
        include_phi1=has_phi1,
        fblock=SimpleNamespace(fp=object() if has_fp else None),
    )


def test_transport_sparse_direct_rescue_enabled_for_cpu_fp_transport(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_SPARSE_DIRECT", raising=False)
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_SPARSE_DIRECT_MAX", raising=False)
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_SPARSE_DIRECT_RATIO", raising=False)
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "cpu")
    assert _transport_sparse_direct_rescue_allowed(
        op=_op(rhs_mode=2),
        size=16382,
        residual_norm=1.0e-3,
        target=1.0e-9,
        use_implicit=False,
    )


def test_transport_sparse_direct_rescue_respects_guards(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_SPARSE_DIRECT", raising=False)
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "cpu")
    assert not _transport_sparse_direct_rescue_allowed(
        op=_op(rhs_mode=1),
        size=16382,
        residual_norm=1.0e-3,
        target=1.0e-9,
        use_implicit=False,
    )
    assert not _transport_sparse_direct_rescue_allowed(
        op=_op(rhs_mode=2, has_fp=False),
        size=16382,
        residual_norm=1.0e-3,
        target=1.0e-9,
        use_implicit=False,
    )
    assert not _transport_sparse_direct_rescue_allowed(
        op=_op(rhs_mode=2, has_phi1=True),
        size=16382,
        residual_norm=1.0e-3,
        target=1.0e-9,
        use_implicit=False,
    )
    assert not _transport_sparse_direct_rescue_allowed(
        op=_op(rhs_mode=2),
        size=50000,
        residual_norm=1.0e-3,
        target=1.0e-9,
        use_implicit=False,
    )
    assert not _transport_sparse_direct_rescue_allowed(
        op=_op(rhs_mode=2),
        size=16382,
        residual_norm=1.0e-3,
        target=1.0e-9,
        use_implicit=True,
    )


def test_transport_sparse_direct_rescue_can_be_disabled(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_SPARSE_DIRECT", "0")
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "cpu")
    assert not _transport_sparse_direct_rescue_allowed(
        op=_op(rhs_mode=2),
        size=16382,
        residual_norm=1.0e-3,
        target=1.0e-9,
        use_implicit=False,
    )


def test_transport_sparse_direct_rescue_respects_env_max(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_SPARSE_DIRECT", raising=False)
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_SPARSE_DIRECT_MAX", "12000")
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "cpu")
    assert not _transport_sparse_direct_rescue_allowed(
        op=_op(rhs_mode=2),
        size=12001,
        residual_norm=1.0e-3,
        target=1.0e-9,
        use_implicit=False,
    )


def test_transport_sparse_direct_rescue_enabled_for_gpu_explicit_transport(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_SPARSE_DIRECT", raising=False)
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "gpu")
    assert _transport_sparse_direct_rescue_allowed(
        op=_op(rhs_mode=2),
        size=16382,
        residual_norm=1.0e-3,
        target=1.0e-9,
        use_implicit=False,
    )


def test_transport_dense_backend_allowed_defaults_to_cpu(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_DENSE_ALLOW_ACCELERATOR", raising=False)
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "cpu")
    assert _transport_dense_backend_allowed()
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "gpu")
    assert not _transport_dense_backend_allowed()


def test_transport_dense_backend_allowed_respects_env(monkeypatch) -> None:
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "gpu")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_DENSE_ALLOW_ACCELERATOR", "1")
    assert _transport_dense_backend_allowed()
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_DENSE_ALLOW_ACCELERATOR", "0")
    assert not _transport_dense_backend_allowed()


def test_transport_sparse_direct_rescue_first_defaults_on(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_SPARSE_DIRECT_FIRST", raising=False)
    assert _transport_sparse_direct_rescue_first(sparse_direct_rescue=True)
    assert not _transport_sparse_direct_rescue_first(sparse_direct_rescue=False)


def test_transport_sparse_direct_rescue_first_can_be_disabled(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_SPARSE_DIRECT_FIRST", "0")
    assert not _transport_sparse_direct_rescue_first(sparse_direct_rescue=True)
