from __future__ import annotations

from types import SimpleNamespace

from sfincs_jax.v3_driver import _rhsmode1_constraint0_sparse_first


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
