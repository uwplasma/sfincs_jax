from __future__ import annotations

from types import SimpleNamespace

from sfincs_jax.v3_driver import (
    _rhsmode1_constraint0_dense_fallback_allowed,
    _rhsmode1_constraint0_petsc_compat,
    _rhsmode1_host_sparse_direct_allowed,
    _rhsmode1_constraint0_sparse_first,
    _rhsmode1_sparse_exact_lu_requested,
)


def _op(*, constraint_scheme: int, has_fp: bool = True, has_phi1: bool = False, rhs_mode: int = 1):
    return SimpleNamespace(
        rhs_mode=rhs_mode,
        include_phi1=has_phi1,
        constraint_scheme=constraint_scheme,
        fblock=SimpleNamespace(fp=object() if has_fp else None),
    )


def test_constraint0_sparse_first_enabled_for_fp_auto_path(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_CS0_SPARSE_FIRST", raising=False)
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "gpu")
    assert _rhsmode1_constraint0_sparse_first(
        op=_op(constraint_scheme=0),
        solve_method_kind="incremental",
        sparse_precond_mode="auto",
        active_size=3276,
        sparse_max_size=6000,
    )


def test_constraint0_sparse_first_respects_guards(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_CS0_SPARSE_FIRST", raising=False)
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "gpu")
    assert not _rhsmode1_constraint0_sparse_first(
        op=_op(constraint_scheme=1),
        solve_method_kind="incremental",
        sparse_precond_mode="auto",
        active_size=3276,
        sparse_max_size=6000,
    )
    assert not _rhsmode1_constraint0_sparse_first(
        op=_op(constraint_scheme=0, has_fp=False),
        solve_method_kind="incremental",
        sparse_precond_mode="auto",
        active_size=3276,
        sparse_max_size=6000,
    )
    assert not _rhsmode1_constraint0_sparse_first(
        op=_op(constraint_scheme=0),
        solve_method_kind="dense",
        sparse_precond_mode="auto",
        active_size=3276,
        sparse_max_size=6000,
    )
    assert not _rhsmode1_constraint0_sparse_first(
        op=_op(constraint_scheme=0),
        solve_method_kind="incremental",
        sparse_precond_mode="off",
        active_size=3276,
        sparse_max_size=6000,
    )
    assert not _rhsmode1_constraint0_sparse_first(
        op=_op(constraint_scheme=0),
        solve_method_kind="incremental",
        sparse_precond_mode="auto",
        active_size=8000,
        sparse_max_size=6000,
    )


def test_constraint0_sparse_first_disabled_by_default_on_cpu(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_CS0_SPARSE_FIRST", raising=False)
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "cpu")
    assert not _rhsmode1_constraint0_sparse_first(
        op=_op(constraint_scheme=0),
        solve_method_kind="incremental",
        sparse_precond_mode="auto",
        active_size=3276,
        sparse_max_size=6000,
    )


def test_constraint0_sparse_first_can_be_disabled(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_CS0_SPARSE_FIRST", "0")
    assert not _rhsmode1_constraint0_sparse_first(
        op=_op(constraint_scheme=0),
        solve_method_kind="incremental",
        sparse_precond_mode="auto",
        active_size=3276,
        sparse_max_size=6000,
    )


def test_constraint0_petsc_compat_disabled_by_default_on_cpu(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_CS0_PETSC_COMPAT", raising=False)
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "cpu")
    assert not _rhsmode1_constraint0_petsc_compat(
        op=_op(constraint_scheme=0),
        solve_method_kind="incremental",
        sparse_precond_mode="auto",
        active_size=3276,
        sparse_max_size=6000,
    )


def test_constraint0_petsc_compat_disabled_by_default_on_gpu(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_CS0_PETSC_COMPAT", raising=False)
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "gpu")
    assert not _rhsmode1_constraint0_petsc_compat(
        op=_op(constraint_scheme=0),
        solve_method_kind="incremental",
        sparse_precond_mode="auto",
        active_size=3276,
        sparse_max_size=6000,
    )


def test_constraint0_petsc_compat_respects_guards(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_CS0_PETSC_COMPAT", raising=False)
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "cpu")
    assert not _rhsmode1_constraint0_petsc_compat(
        op=_op(constraint_scheme=1),
        solve_method_kind="incremental",
        sparse_precond_mode="auto",
        active_size=3276,
        sparse_max_size=6000,
    )
    assert not _rhsmode1_constraint0_petsc_compat(
        op=_op(constraint_scheme=0, has_fp=False),
        solve_method_kind="incremental",
        sparse_precond_mode="auto",
        active_size=3276,
        sparse_max_size=6000,
    )
    assert not _rhsmode1_constraint0_petsc_compat(
        op=_op(constraint_scheme=0),
        solve_method_kind="dense",
        sparse_precond_mode="auto",
        active_size=3276,
        sparse_max_size=6000,
    )
    assert not _rhsmode1_constraint0_petsc_compat(
        op=_op(constraint_scheme=0),
        solve_method_kind="incremental",
        sparse_precond_mode="off",
        active_size=3276,
        sparse_max_size=6000,
    )
    assert not _rhsmode1_constraint0_petsc_compat(
        op=_op(constraint_scheme=0),
        solve_method_kind="incremental",
        sparse_precond_mode="auto",
        active_size=8000,
        sparse_max_size=6000,
    )


def test_constraint0_petsc_compat_can_be_disabled(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_CS0_PETSC_COMPAT", "0")
    assert not _rhsmode1_constraint0_petsc_compat(
        op=_op(constraint_scheme=0),
        solve_method_kind="incremental",
        sparse_precond_mode="auto",
        active_size=3276,
        sparse_max_size=6000,
    )


def test_constraint0_petsc_compat_can_be_enabled(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_CS0_PETSC_COMPAT", "1")
    assert _rhsmode1_constraint0_petsc_compat(
        op=_op(constraint_scheme=0),
        solve_method_kind="incremental",
        sparse_precond_mode="auto",
        active_size=3276,
        sparse_max_size=6000,
    )


def test_constraint0_dense_fallback_disabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_CS0_DENSE_FALLBACK", raising=False)
    assert not _rhsmode1_constraint0_dense_fallback_allowed(_op(constraint_scheme=0))
    assert _rhsmode1_constraint0_dense_fallback_allowed(_op(constraint_scheme=1))


def test_constraint0_dense_fallback_can_be_enabled(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_CS0_DENSE_FALLBACK", "1")
    assert _rhsmode1_constraint0_dense_fallback_allowed(_op(constraint_scheme=0))


def test_sparse_exact_lu_auto_enables_on_small_gpu_fp_case(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_EXACT_LU", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_EXACT_LU_MAX", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_EXACT_LU_ACCEL_SMALL_MAX", raising=False)
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "gpu")
    assert _rhsmode1_sparse_exact_lu_requested(
        op=_op(constraint_scheme=1),
        solve_method_kind="incremental",
        active_size=2804,
        sparse_max_size=6000,
        preconditioner_x=1,
        use_dkes=False,
    )


def test_sparse_exact_lu_auto_respects_accel_small_cap(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_EXACT_LU", raising=False)
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_EXACT_LU_ACCEL_SMALL_MAX", "2000")
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "gpu")
    assert not _rhsmode1_sparse_exact_lu_requested(
        op=_op(constraint_scheme=1),
        solve_method_kind="incremental",
        active_size=2804,
        sparse_max_size=6000,
        preconditioner_x=1,
        use_dkes=False,
    )


def test_sparse_exact_lu_auto_stays_off_on_cpu_without_full_x(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_EXACT_LU", raising=False)
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "cpu")
    assert not _rhsmode1_sparse_exact_lu_requested(
        op=_op(constraint_scheme=1),
        solve_method_kind="incremental",
        active_size=2804,
        sparse_max_size=6000,
        preconditioner_x=1,
        use_dkes=False,
    )


def test_host_sparse_direct_default_on_gpu_for_exact_lu(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_DIRECT_HOST", raising=False)
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "gpu")
    assert _rhsmode1_host_sparse_direct_allowed(sparse_exact_lu=True)


def test_host_sparse_direct_off_without_exact_lu(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_DIRECT_HOST", raising=False)
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "gpu")
    assert not _rhsmode1_host_sparse_direct_allowed(sparse_exact_lu=False)


def test_host_sparse_direct_can_be_forced_on_cpu(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_DIRECT_HOST", "1")
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "cpu")
    assert _rhsmode1_host_sparse_direct_allowed(sparse_exact_lu=True)
