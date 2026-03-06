from __future__ import annotations

from types import SimpleNamespace

from sfincs_jax.v3_driver import _rhsmode1_sparse_exact_lu_requested


def _op(*, has_fp: bool = True, has_phi1: bool = False, rhs_mode: int = 1):
    return SimpleNamespace(
        rhs_mode=rhs_mode,
        include_phi1=has_phi1,
        fblock=SimpleNamespace(fp=object() if has_fp else None),
    )


def test_sparse_exact_lu_enabled_for_full_x_coupling(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_EXACT_LU", raising=False)
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "cpu")
    assert _rhsmode1_sparse_exact_lu_requested(
        op=_op(),
        solve_method_kind="incremental",
        active_size=3276,
        sparse_max_size=6000,
        preconditioner_x=0,
        use_dkes=False,
    )


def test_sparse_exact_lu_enabled_for_gpu_dkes(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_EXACT_LU", raising=False)
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "gpu")
    assert _rhsmode1_sparse_exact_lu_requested(
        op=_op(),
        solve_method_kind="incremental",
        active_size=5000,
        sparse_max_size=12000,
        preconditioner_x=1,
        use_dkes=True,
    )


def test_sparse_exact_lu_respects_guards(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_EXACT_LU", raising=False)
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "cpu")
    assert not _rhsmode1_sparse_exact_lu_requested(
        op=_op(has_fp=False),
        solve_method_kind="incremental",
        active_size=3276,
        sparse_max_size=6000,
        preconditioner_x=0,
        use_dkes=False,
    )
    assert not _rhsmode1_sparse_exact_lu_requested(
        op=_op(),
        solve_method_kind="dense",
        active_size=3276,
        sparse_max_size=6000,
        preconditioner_x=0,
        use_dkes=False,
    )
    assert not _rhsmode1_sparse_exact_lu_requested(
        op=_op(),
        solve_method_kind="incremental",
        active_size=7000,
        sparse_max_size=6000,
        preconditioner_x=0,
        use_dkes=False,
    )


def test_sparse_exact_lu_can_be_forced_or_disabled(monkeypatch) -> None:
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "cpu")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_EXACT_LU", "0")
    assert not _rhsmode1_sparse_exact_lu_requested(
        op=_op(),
        solve_method_kind="incremental",
        active_size=3276,
        sparse_max_size=6000,
        preconditioner_x=0,
        use_dkes=False,
    )
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_EXACT_LU", "1")
    assert _rhsmode1_sparse_exact_lu_requested(
        op=_op(),
        solve_method_kind="incremental",
        active_size=3276,
        sparse_max_size=6000,
        preconditioner_x=1,
        use_dkes=False,
    )
