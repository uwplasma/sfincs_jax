from __future__ import annotations

from types import SimpleNamespace

from sfincs_jax.v3_driver import (
    _rhs1_pas_auto_large_base_kind,
    _rhsmode1_pas_fast_accept,
    _rhsmode1_constraint0_dense_fallback_allowed,
    _rhsmode1_constraint0_petsc_compat,
    _rhsmode1_host_sparse_direct_allowed,
    _rhsmode1_host_factor_probe_ok,
    _rhsmode1_constraint0_sparse_first,
    _rhsmode1_fp_xblock_assembled_host_allowed,
    _rhsmode1_large_cpu_xblock_skip_primary_allowed,
    _rhsmode1_large_cpu_sparse_rescue_allowed,
    _rhsmode1_large_cpu_sparse_exact_lu_allowed,
    _rhsmode1_large_cpu_sparse_rescue_first,
    _rhsmode1_prefer_sparse_over_dense_shortcut,
    _rhsmode1_sparse_prefer_skips_stage2,
    _resolve_use_implicit,
    _rhsmode1_sparse_sxblock_rescue_allowed,
    _rhsmode1_sparse_exact_lu_requested,
    _rhsmode1_sparse_xblock_rescue_allowed,
)


def _op(*, constraint_scheme: int, has_fp: bool = True, has_phi1: bool = False, rhs_mode: int = 1):
    return SimpleNamespace(
        rhs_mode=rhs_mode,
        include_phi1=has_phi1,
        constraint_scheme=constraint_scheme,
        point_at_x0=False,
        fblock=SimpleNamespace(fp=object() if has_fp else None, pas=None),
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


def test_large_pas_auto_prefers_pas_lite_above_threshold(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_PAS_LITE_MIN", raising=False)
    assert _rhs1_pas_auto_large_base_kind(active_size=25000) == "pas_lite"


def test_large_pas_auto_prefers_pas_hybrid_below_threshold(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_PAS_LITE_MIN", raising=False)
    assert _rhs1_pas_auto_large_base_kind(active_size=5000) == "pas_hybrid"


def test_pas_fast_accept_enabled_for_large_explicit_cpu_pas(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_PAS_FAST_ACCEPT", raising=False)
    monkeypatch.delenv("SFINCS_JAX_PAS_FAST_ACCEPT_MIN", raising=False)
    monkeypatch.delenv("SFINCS_JAX_PAS_FAST_ACCEPT_RATIO", raising=False)
    monkeypatch.delenv("SFINCS_JAX_PAS_FAST_ACCEPT_ABS", raising=False)
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "cpu")
    op = _op(constraint_scheme=1, has_fp=False)
    op.fblock.pas = object()
    assert _rhsmode1_pas_fast_accept(
        op=op,
        active_size=41561,
        residual_norm=6.6e-8,
        target=4.8e-10,
        use_implicit=False,
    )


def test_pas_fast_accept_respects_guards(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_PAS_FAST_ACCEPT", raising=False)
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "cpu")
    op = _op(constraint_scheme=1, has_fp=False)
    op.fblock.pas = object()
    assert not _rhsmode1_pas_fast_accept(
        op=op,
        active_size=5000,
        residual_norm=6.6e-8,
        target=4.8e-10,
        use_implicit=False,
    )
    assert not _rhsmode1_pas_fast_accept(
        op=op,
        active_size=41561,
        residual_norm=1.0e-5,
        target=4.8e-10,
        use_implicit=False,
    )
    assert not _rhsmode1_pas_fast_accept(
        op=op,
        active_size=41561,
        residual_norm=6.6e-8,
        target=4.8e-10,
        use_implicit=True,
    )
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "gpu")
    assert not _rhsmode1_pas_fast_accept(
        op=op,
        active_size=41561,
        residual_norm=6.6e-8,
        target=4.8e-10,
        use_implicit=False,
    )


def test_resolve_use_implicit_honors_explicit_flag(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_IMPLICIT_SOLVE", "1")
    assert _resolve_use_implicit(differentiable=False) is False
    assert _resolve_use_implicit(differentiable=True) is True


def test_resolve_use_implicit_falls_back_to_env(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_IMPLICIT_SOLVE", "0")
    assert _resolve_use_implicit(differentiable=None) is False
    monkeypatch.setenv("SFINCS_JAX_IMPLICIT_SOLVE", "1")
    assert _resolve_use_implicit(differentiable=None) is True


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
    assert _rhsmode1_host_sparse_direct_allowed(sparse_exact_lu=True, use_implicit=False)


def test_host_sparse_direct_default_on_cpu_for_explicit_exact_lu(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_DIRECT_HOST", raising=False)
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "cpu")
    assert _rhsmode1_host_sparse_direct_allowed(sparse_exact_lu=True, use_implicit=False)


def test_host_sparse_direct_stays_off_for_implicit_exact_lu(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_DIRECT_HOST", raising=False)
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "gpu")
    assert not _rhsmode1_host_sparse_direct_allowed(sparse_exact_lu=True, use_implicit=True)


class _Factor:
    def __init__(self, scale: float) -> None:
        self.scale = scale

    def solve(self, x):
        return self.scale * x


def test_host_factor_probe_accepts_bounded_factor(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_XBLOCK_FACTOR_PROBE_MAX", raising=False)
    assert _rhsmode1_host_factor_probe_ok(factor=_Factor(10.0), block_size=8)


def test_host_factor_probe_rejects_unbounded_factor(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_FACTOR_PROBE_MAX", "100")
    assert not _rhsmode1_host_factor_probe_ok(factor=_Factor(1.0e6), block_size=8)


def test_large_cpu_xblock_skip_primary_enabled_for_auto_cpu_lane(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_LARGE_CPU_XBLOCK_SKIP_PRIMARY", raising=False)
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "cpu")
    assert _rhsmode1_large_cpu_xblock_skip_primary_allowed(
        op=_op(constraint_scheme=1),
        solve_method_kind="incremental",
        active_size=20000,
        sparse_max_size=6000,
        preconditioner_species=1,
        preconditioner_x=1,
        preconditioner_xi=1,
        pre_theta=0,
        pre_zeta=0,
        use_implicit=False,
        rhs1_precond_env="",
    )


def test_large_cpu_xblock_skip_primary_respects_guards(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_LARGE_CPU_XBLOCK_SKIP_PRIMARY", raising=False)
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "cpu")
    assert not _rhsmode1_large_cpu_xblock_skip_primary_allowed(
        op=_op(constraint_scheme=1),
        solve_method_kind="dense",
        active_size=20000,
        sparse_max_size=6000,
        preconditioner_species=1,
        preconditioner_x=1,
        preconditioner_xi=1,
        pre_theta=0,
        pre_zeta=0,
        use_implicit=False,
        rhs1_precond_env="",
    )
    assert not _rhsmode1_large_cpu_xblock_skip_primary_allowed(
        op=_op(constraint_scheme=1),
        solve_method_kind="incremental",
        active_size=2000,
        sparse_max_size=6000,
        preconditioner_species=1,
        preconditioner_x=1,
        preconditioner_xi=1,
        pre_theta=0,
        pre_zeta=0,
        use_implicit=False,
        rhs1_precond_env="",
    )
    assert not _rhsmode1_large_cpu_xblock_skip_primary_allowed(
        op=_op(constraint_scheme=1),
        solve_method_kind="incremental",
        active_size=20000,
        sparse_max_size=6000,
        preconditioner_species=1,
        preconditioner_x=0,
        preconditioner_xi=1,
        pre_theta=0,
        pre_zeta=0,
        use_implicit=False,
        rhs1_precond_env="",
    )
    assert not _rhsmode1_large_cpu_xblock_skip_primary_allowed(
        op=_op(constraint_scheme=1),
        solve_method_kind="incremental",
        active_size=20000,
        sparse_max_size=6000,
        preconditioner_species=1,
        preconditioner_x=1,
        preconditioner_xi=1,
        pre_theta=1,
        pre_zeta=0,
        use_implicit=False,
        rhs1_precond_env="",
    )
    assert not _rhsmode1_large_cpu_xblock_skip_primary_allowed(
        op=_op(constraint_scheme=1),
        solve_method_kind="incremental",
        active_size=20000,
        sparse_max_size=6000,
        preconditioner_species=1,
        preconditioner_x=1,
        preconditioner_xi=1,
        pre_theta=0,
        pre_zeta=0,
        use_implicit=True,
        rhs1_precond_env="",
    )
    assert not _rhsmode1_large_cpu_xblock_skip_primary_allowed(
        op=_op(constraint_scheme=1),
        solve_method_kind="incremental",
        active_size=20000,
        sparse_max_size=6000,
        preconditioner_species=1,
        preconditioner_x=1,
        preconditioner_xi=1,
        pre_theta=0,
        pre_zeta=0,
        use_implicit=False,
        rhs1_precond_env="schur",
    )


def test_sparse_sxblock_rescue_enabled_for_large_cpu_fp_multispecies(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_SXBLOCK_RESCUE", "1")
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "cpu")
    op = _op(constraint_scheme=1)
    op.n_species = 2
    assert _rhsmode1_sparse_sxblock_rescue_allowed(
        op=op,
        solve_method_kind="incremental",
        active_size=20000,
        sparse_max_size=6000,
        preconditioner_x=1,
        pre_theta=0,
        pre_zeta=0,
        use_implicit=False,
    )


def test_sparse_sxblock_rescue_respects_guards(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_SXBLOCK_RESCUE", raising=False)
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "cpu")
    op = _op(constraint_scheme=1)
    op.n_species = 2
    assert not _rhsmode1_sparse_sxblock_rescue_allowed(
        op=op,
        solve_method_kind="incremental",
        active_size=20000,
        sparse_max_size=6000,
        preconditioner_x=1,
        pre_theta=0,
        pre_zeta=0,
        use_implicit=False,
    )
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_SXBLOCK_RESCUE", "1")
    op.n_species = 1
    assert not _rhsmode1_sparse_sxblock_rescue_allowed(
        op=op,
        solve_method_kind="incremental",
        active_size=20000,
        sparse_max_size=6000,
        preconditioner_x=1,
        pre_theta=0,
        pre_zeta=0,
        use_implicit=False,
    )
    op.n_species = 2
    assert not _rhsmode1_sparse_sxblock_rescue_allowed(
        op=op,
        solve_method_kind="dense",
        active_size=20000,
        sparse_max_size=6000,
        preconditioner_x=1,
        pre_theta=0,
        pre_zeta=0,
        use_implicit=False,
    )
    assert not _rhsmode1_sparse_sxblock_rescue_allowed(
        op=op,
        solve_method_kind="incremental",
        active_size=20000,
        sparse_max_size=6000,
        preconditioner_x=0,
        pre_theta=0,
        pre_zeta=0,
        use_implicit=False,
    )
    assert not _rhsmode1_sparse_sxblock_rescue_allowed(
        op=op,
        solve_method_kind="incremental",
        active_size=20000,
        sparse_max_size=6000,
        preconditioner_x=1,
        pre_theta=1,
        pre_zeta=0,
        use_implicit=False,
    )
    assert not _rhsmode1_sparse_sxblock_rescue_allowed(
        op=op,
        solve_method_kind="incremental",
        active_size=5000,
        sparse_max_size=6000,
        preconditioner_x=1,
        pre_theta=0,
        pre_zeta=0,
        use_implicit=False,
    )


def test_host_sparse_direct_off_without_exact_lu(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_DIRECT_HOST", raising=False)
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "gpu")
    assert not _rhsmode1_host_sparse_direct_allowed(sparse_exact_lu=False, use_implicit=False)


def test_host_sparse_direct_can_be_forced_on_cpu(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_DIRECT_HOST", "1")
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "cpu")
    assert _rhsmode1_host_sparse_direct_allowed(sparse_exact_lu=True, use_implicit=False)


def test_sparse_exact_lu_can_exceed_sparse_ilu_cap_on_gpu_dkes(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_EXACT_LU", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_EXACT_LU_MAX", raising=False)
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "gpu")
    assert _rhsmode1_sparse_exact_lu_requested(
        op=_op(constraint_scheme=1),
        solve_method_kind="incremental",
        active_size=6302,
        sparse_max_size=6000,
        preconditioner_x=0,
        use_dkes=True,
    )


def test_prefer_sparse_over_dense_shortcut_for_explicit_moderate_fp(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_PREFER_OVER_DENSE_SHORTCUT", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_PREFER_OVER_DENSE_SHORTCUT_MIN", raising=False)
    assert _rhsmode1_prefer_sparse_over_dense_shortcut(
        op=_op(constraint_scheme=1),
        active_size=4288,
        sparse_max_size=6000,
        use_implicit=False,
    )


def test_prefer_sparse_over_dense_shortcut_respects_guards(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_PREFER_OVER_DENSE_SHORTCUT", raising=False)
    assert not _rhsmode1_prefer_sparse_over_dense_shortcut(
        op=_op(constraint_scheme=1),
        active_size=1000,
        sparse_max_size=6000,
        use_implicit=False,
    )
    assert not _rhsmode1_prefer_sparse_over_dense_shortcut(
        op=_op(constraint_scheme=1),
        active_size=4288,
        sparse_max_size=4000,
        use_implicit=False,
    )
    assert not _rhsmode1_prefer_sparse_over_dense_shortcut(
        op=_op(constraint_scheme=1, has_fp=False),
        active_size=4288,
        sparse_max_size=6000,
        use_implicit=False,
    )
    assert not _rhsmode1_prefer_sparse_over_dense_shortcut(
        op=_op(constraint_scheme=1),
        active_size=4288,
        sparse_max_size=6000,
        use_implicit=True,
    )


def test_sparse_prefer_skips_stage2_by_default(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_SKIP_STAGE2", raising=False)
    assert _rhsmode1_sparse_prefer_skips_stage2(
        sparse_prefer_over_dense_shortcut=True,
        sparse_precond_mode="auto",
    )


def test_sparse_prefer_skips_stage2_respects_guards(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_SKIP_STAGE2", raising=False)
    assert not _rhsmode1_sparse_prefer_skips_stage2(
        sparse_prefer_over_dense_shortcut=False,
        sparse_precond_mode="auto",
    )
    assert not _rhsmode1_sparse_prefer_skips_stage2(
        sparse_prefer_over_dense_shortcut=True,
        sparse_precond_mode="off",
    )
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_SKIP_STAGE2", "0")
    assert not _rhsmode1_sparse_prefer_skips_stage2(
        sparse_prefer_over_dense_shortcut=True,
        sparse_precond_mode="auto",
    )


def test_large_cpu_sparse_rescue_enabled_for_large_fullx_fp_failures(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_LARGE_CPU_RESCUE", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_LARGE_CPU_RESCUE_MAX", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_LARGE_CPU_RESCUE_RATIO", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_LARGE_CPU_RESCUE_FULLX_MIN", raising=False)
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "cpu")
    assert _rhsmode1_large_cpu_sparse_rescue_allowed(
        op=_op(constraint_scheme=1),
        solve_method_kind="incremental",
        active_size=68670,
        sparse_max_size=6000,
        preconditioner_x=1,
        residual_norm=1.0,
        target=1.0e-6,
    )


def test_large_cpu_sparse_rescue_respects_guards(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_LARGE_CPU_RESCUE", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_LARGE_CPU_RESCUE_FULLX_MIN", raising=False)
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "cpu")
    assert not _rhsmode1_large_cpu_sparse_rescue_allowed(
        op=_op(constraint_scheme=1),
        solve_method_kind="incremental",
        active_size=5000,
        sparse_max_size=6000,
        preconditioner_x=0,
        residual_norm=1.0,
        target=1.0e-6,
    )
    assert not _rhsmode1_large_cpu_sparse_rescue_allowed(
        op=_op(constraint_scheme=1),
        solve_method_kind="incremental",
        active_size=18366,
        sparse_max_size=6000,
        preconditioner_x=1,
        residual_norm=1.0,
        target=1.0e-6,
    )
    assert not _rhsmode1_large_cpu_sparse_rescue_allowed(
        op=_op(constraint_scheme=1, has_phi1=True),
        solve_method_kind="incremental",
        active_size=18366,
        sparse_max_size=6000,
        preconditioner_x=0,
        residual_norm=1.0,
        target=1.0e-6,
    )
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "gpu")
    assert not _rhsmode1_large_cpu_sparse_rescue_allowed(
        op=_op(constraint_scheme=1),
        solve_method_kind="incremental",
        active_size=18366,
        sparse_max_size=6000,
        preconditioner_x=0,
        residual_norm=1.0,
        target=1.0e-6,
    )


def test_large_cpu_sparse_rescue_can_depend_on_active_dof_size(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_LARGE_CPU_RESCUE", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_LARGE_CPU_RESCUE_MAX", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_LARGE_CPU_RESCUE_FULLX_MIN", raising=False)
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "cpu")
    op = _op(constraint_scheme=1)
    assert _rhsmode1_large_cpu_sparse_rescue_allowed(
        op=op,
        solve_method_kind="incremental",
        active_size=18366,
        sparse_max_size=6000,
        preconditioner_x=0,
        residual_norm=1.0,
        target=1.0e-6,
    )
    assert not _rhsmode1_large_cpu_sparse_rescue_allowed(
        op=op,
        solve_method_kind="incremental",
        active_size=90001,
        sparse_max_size=6000,
        preconditioner_x=0,
        residual_norm=1.0,
        target=1.0e-6,
    )


def test_large_cpu_sparse_rescue_first_defaults_on_for_auto_strong_precond(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_LARGE_CPU_RESCUE_FIRST", raising=False)
    assert _rhsmode1_large_cpu_sparse_rescue_first(large_cpu_sparse_rescue=True, strong_precond_env="")
    assert _rhsmode1_large_cpu_sparse_rescue_first(large_cpu_sparse_rescue=True, strong_precond_env="auto")
    assert not _rhsmode1_large_cpu_sparse_rescue_first(large_cpu_sparse_rescue=False, strong_precond_env="")
    assert not _rhsmode1_large_cpu_sparse_rescue_first(large_cpu_sparse_rescue=True, strong_precond_env="theta_line")


def test_large_cpu_sparse_rescue_first_can_be_disabled(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_LARGE_CPU_RESCUE_FIRST", "0")
    assert not _rhsmode1_large_cpu_sparse_rescue_first(large_cpu_sparse_rescue=True, strong_precond_env="")


def test_large_cpu_sparse_exact_lu_stays_on_for_moderate_sizes(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_LARGE_CPU_RESCUE_EXACT_LU", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_LARGE_CPU_RESCUE_EXACT_LU_MAX", raising=False)
    assert _rhsmode1_large_cpu_sparse_exact_lu_allowed(active_size=18366)


def test_large_cpu_sparse_exact_lu_defaults_off_for_very_large_sizes(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_LARGE_CPU_RESCUE_EXACT_LU", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_LARGE_CPU_RESCUE_EXACT_LU_MAX", raising=False)
    assert not _rhsmode1_large_cpu_sparse_exact_lu_allowed(active_size=68670)


def test_sparse_xblock_rescue_enabled_for_large_fullx_fp_defaults(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_XBLOCK_RESCUE", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_XBLOCK_RESCUE_MIN", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_XBLOCK_RESCUE_MAX", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_XBLOCK_RESCUE_RATIO", raising=False)
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "cpu")
    assert _rhsmode1_sparse_xblock_rescue_allowed(
        op=_op(constraint_scheme=1),
        solve_method_kind="incremental",
        active_size=68670,
        sparse_max_size=6000,
        preconditioner_x=1,
        pre_theta=0,
        pre_zeta=0,
        residual_norm=1.0,
        target=1.0e-6,
    )


def test_sparse_xblock_rescue_respects_guards(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_XBLOCK_RESCUE", raising=False)
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "cpu")
    assert not _rhsmode1_sparse_xblock_rescue_allowed(
        op=_op(constraint_scheme=1),
        solve_method_kind="incremental",
        active_size=68670,
        sparse_max_size=6000,
        preconditioner_x=0,
        pre_theta=0,
        pre_zeta=0,
        residual_norm=1.0,
        target=1.0e-6,
    )


def test_fp_xblock_assembled_host_enabled_for_explicit_cpu_fp(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_FP_XBLOCK_ASSEMBLED_HOST", raising=False)
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "cpu")
    assert _rhsmode1_fp_xblock_assembled_host_allowed(
        op=_op(constraint_scheme=1),
        preconditioner_species=1,
        preconditioner_xi=1,
        use_implicit=False,
    )


def test_fp_xblock_assembled_host_respects_guards(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_FP_XBLOCK_ASSEMBLED_HOST", raising=False)
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "cpu")
    assert not _rhsmode1_fp_xblock_assembled_host_allowed(
        op=_op(constraint_scheme=1, has_fp=False),
        preconditioner_species=1,
        preconditioner_xi=1,
        use_implicit=False,
    )
    assert not _rhsmode1_fp_xblock_assembled_host_allowed(
        op=_op(constraint_scheme=1),
        preconditioner_species=0,
        preconditioner_xi=1,
        use_implicit=False,
    )
    assert not _rhsmode1_fp_xblock_assembled_host_allowed(
        op=_op(constraint_scheme=1),
        preconditioner_species=1,
        preconditioner_xi=0,
        use_implicit=False,
    )
    assert not _rhsmode1_fp_xblock_assembled_host_allowed(
        op=_op(constraint_scheme=1),
        preconditioner_species=1,
        preconditioner_xi=1,
        use_implicit=True,
    )
    op_x0 = _op(constraint_scheme=1)
    op_x0.point_at_x0 = True
    assert not _rhsmode1_fp_xblock_assembled_host_allowed(
        op=op_x0,
        preconditioner_species=1,
        preconditioner_xi=1,
        use_implicit=False,
    )


def test_fp_xblock_assembled_host_disabled_off_cpu(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_FP_XBLOCK_ASSEMBLED_HOST", raising=False)
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "gpu")
    assert not _rhsmode1_fp_xblock_assembled_host_allowed(
        op=_op(constraint_scheme=1),
        preconditioner_species=1,
        preconditioner_xi=1,
        use_implicit=False,
    )
    assert not _rhsmode1_sparse_xblock_rescue_allowed(
        op=_op(constraint_scheme=1),
        solve_method_kind="incremental",
        active_size=68670,
        sparse_max_size=6000,
        preconditioner_x=1,
        pre_theta=1,
        pre_zeta=0,
        residual_norm=1.0,
        target=1.0e-6,
    )
    monkeypatch.setattr("sfincs_jax.v3_driver.jax.default_backend", lambda: "gpu")
    assert not _rhsmode1_sparse_xblock_rescue_allowed(
        op=_op(constraint_scheme=1),
        solve_method_kind="incremental",
        active_size=68670,
        sparse_max_size=6000,
        preconditioner_x=1,
        pre_theta=0,
        pre_zeta=0,
        residual_norm=1.0,
        target=1.0e-6,
    )
