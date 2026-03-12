from __future__ import annotations

from types import SimpleNamespace

from sfincs_jax.v3_driver import (
    _host_sparse_factor_dtype,
    _transport_dense_backend_allowed,
    _transport_host_gmres_accepts_preconditioned_residual,
    _transport_host_gmres_first_attempt_allowed,
    _transport_sparse_direct_first_attempt_allowed,
    _transport_sparse_direct_rescue_allowed,
    _transport_sparse_direct_rescue_first,
    _transport_tzfft_backend_allowed,
)


def _op(*, rhs_mode: int = 2, has_fp: bool = True, has_phi1: bool = False, n_x: int = 4):
    return SimpleNamespace(
        rhs_mode=rhs_mode,
        include_phi1=has_phi1,
        n_x=n_x,
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


def test_transport_sparse_direct_rescue_enabled_for_cpu_collisionless_mono_medium_size(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_SPARSE_DIRECT", raising=False)
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_SPARSE_DIRECT_MAX", raising=False)
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "cpu")
    assert _transport_sparse_direct_rescue_allowed(
        op=_op(rhs_mode=3, has_fp=False, n_x=1),
        size=54811,
        residual_norm=1.0e-2,
        target=1.0e-6,
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


def test_transport_sparse_direct_rescue_enabled_for_gpu_collisionless_transport(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_SPARSE_DIRECT", raising=False)
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "gpu")
    assert _transport_sparse_direct_rescue_allowed(
        op=_op(rhs_mode=3, has_fp=False, n_x=1),
        size=5383,
        residual_norm=1.0e-3,
        target=1.0e-9,
        use_implicit=False,
    )


def test_transport_sparse_direct_rescue_enabled_for_nonfinite_residual(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_SPARSE_DIRECT", raising=False)
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "gpu")
    assert _transport_sparse_direct_rescue_allowed(
        op=_op(rhs_mode=3, has_fp=False, n_x=1),
        size=5383,
        residual_norm=float("nan"),
        target=1.0e-9,
        use_implicit=False,
    )


def test_transport_sparse_direct_first_attempt_allowed_for_gpu_explicit_transport(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_SPARSE_DIRECT", raising=False)
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "gpu")
    assert _transport_sparse_direct_first_attempt_allowed(
        op=_op(rhs_mode=3, has_fp=False, n_x=1),
        size=5383,
        use_implicit=False,
    )


def test_transport_sparse_direct_first_attempt_enabled_for_cpu_collisionless_mono_medium_size(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_SPARSE_DIRECT", raising=False)
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_SPARSE_DIRECT_FIRST_CPU_MIN", raising=False)
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "cpu")
    assert _transport_sparse_direct_first_attempt_allowed(
        op=_op(rhs_mode=3, has_fp=False, n_x=1),
        size=54811,
        use_implicit=False,
    )


def test_transport_sparse_direct_first_attempt_enabled_for_cpu_transport_fast_path(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_SPARSE_DIRECT", raising=False)
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_SPARSE_DIRECT_FIRST_CPU_MIN", raising=False)
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "cpu")
    assert _transport_sparse_direct_first_attempt_allowed(
        op=_op(rhs_mode=2),
        size=16382,
        use_implicit=False,
    )


def test_transport_sparse_direct_first_attempt_disabled_for_small_cpu_or_implicit(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_SPARSE_DIRECT", raising=False)
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_SPARSE_DIRECT_FIRST_CPU_MIN", raising=False)
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "cpu")
    assert not _transport_sparse_direct_first_attempt_allowed(
        op=_op(rhs_mode=2),
        size=8000,
        use_implicit=False,
    )
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "gpu")
    assert not _transport_sparse_direct_first_attempt_allowed(
        op=_op(rhs_mode=2),
        size=16382,
        use_implicit=True,
    )


def test_host_sparse_factor_dtype_defaults_to_float32_for_large_explicit_cpu_lu(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_HOST_SPARSE_FACTOR_DTYPE", raising=False)
    monkeypatch.delenv("SFINCS_JAX_HOST_SPARSE_FACTOR_FLOAT32_MIN", raising=False)
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "cpu")
    assert _host_sparse_factor_dtype(size=16382, factorization="lu", use_implicit=False).name == "float32"
    assert _host_sparse_factor_dtype(size=8000, factorization="lu", use_implicit=False).name == "float64"
    assert _host_sparse_factor_dtype(size=16382, factorization="ilu", use_implicit=False).name == "float64"
    assert _host_sparse_factor_dtype(size=16382, factorization="lu", use_implicit=True).name == "float64"


def test_transport_host_gmres_first_attempt_disabled_when_sparse_direct_first_is_available(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_HOST_GMRES_FIRST", raising=False)
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "cpu")
    assert not _transport_host_gmres_first_attempt_allowed(
        op=_op(rhs_mode=3, has_fp=False, n_x=1),
        size=54811,
        use_implicit=False,
    )


def test_transport_host_gmres_first_attempt_respects_guards(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_HOST_GMRES_FIRST", raising=False)
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "cpu")
    assert not _transport_host_gmres_first_attempt_allowed(
        op=_op(rhs_mode=2),
        size=16382,
        use_implicit=False,
    )
    assert not _transport_host_gmres_first_attempt_allowed(
        op=_op(rhs_mode=3, has_fp=False, n_x=3),
        size=54811,
        use_implicit=False,
    )
    assert not _transport_host_gmres_first_attempt_allowed(
        op=_op(rhs_mode=3, has_fp=False, n_x=1, has_phi1=True),
        size=54811,
        use_implicit=False,
    )
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "gpu")
    assert not _transport_host_gmres_first_attempt_allowed(
        op=_op(rhs_mode=3, has_fp=False, n_x=1),
        size=54811,
        use_implicit=False,
    )


def test_transport_host_gmres_accepts_preconditioned_residual_for_moderate_true_gap(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_HOST_GMRES_TRUE_RATIO", raising=False)
    assert _transport_host_gmres_accepts_preconditioned_residual(
        op=_op(rhs_mode=3, has_fp=False, n_x=1),
        true_residual_norm=5.0e-5,
        target_true=1.0e-6,
    )


def test_transport_host_gmres_rejects_preconditioned_residual_for_large_true_gap(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_HOST_GMRES_TRUE_RATIO", raising=False)
    assert not _transport_host_gmres_accepts_preconditioned_residual(
        op=_op(rhs_mode=3, has_fp=False, n_x=1),
        true_residual_norm=1.0e-2,
        target_true=1.0e-6,
    )


def test_transport_dense_backend_allowed_defaults_to_cpu(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_DENSE_ALLOW_ACCELERATOR", raising=False)
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "cpu")
    assert _transport_dense_backend_allowed()
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "gpu")
    assert not _transport_dense_backend_allowed()


def test_transport_tzfft_backend_allowed_defaults_to_cpu(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_TZFFT_ALLOW_ACCELERATOR", raising=False)
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "cpu")
    assert _transport_tzfft_backend_allowed()
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "gpu")
    assert not _transport_tzfft_backend_allowed()


def test_transport_dense_backend_allowed_respects_env(monkeypatch) -> None:
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "gpu")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_DENSE_ALLOW_ACCELERATOR", "1")
    assert _transport_dense_backend_allowed()
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_DENSE_ALLOW_ACCELERATOR", "0")
    assert not _transport_dense_backend_allowed()


def test_transport_tzfft_backend_allowed_respects_env(monkeypatch) -> None:
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "gpu")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_TZFFT_ALLOW_ACCELERATOR", "1")
    assert _transport_tzfft_backend_allowed()
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_TZFFT_ALLOW_ACCELERATOR", "0")
    assert not _transport_tzfft_backend_allowed()


def test_transport_sparse_direct_rescue_first_defaults_on(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_SPARSE_DIRECT_FIRST", raising=False)
    assert _transport_sparse_direct_rescue_first(sparse_direct_rescue=True)
    assert not _transport_sparse_direct_rescue_first(sparse_direct_rescue=False)


def test_transport_sparse_direct_rescue_first_can_be_disabled(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_SPARSE_DIRECT_FIRST", "0")
    assert not _transport_sparse_direct_rescue_first(sparse_direct_rescue=True)
