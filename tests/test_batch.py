"""Batched multi-surface / multi-``E_r`` vmap productization (:mod:`dkx.batch`).

Gates for the first-class batched-solve API:

- batched result == serial per-element result to ~1e-12 (``E_r`` scan and
  surface scan), for both the states and the moments;
- ``jax.grad`` through a batched scalar (sum of the bootstrap current over the
  batch) matches the per-element gradients and a central finite difference;
- ``jax.jit(batched_solve)`` compiles and runs, matching the eager result;
- the memory-budgeted auto-chunking respects the budget / ``max_batch`` and the
  chunked result is identical to the single-chunk result;
- batching a discretization leaf, or a surface batch that disagrees on the
  discretization, is rejected with a clear error;
- the :mod:`dkx.api` facades route to the same result;
- the multi-device split (``devices=``) is element-wise identical to the
  single-device path — verified in a fresh subprocess with two distinct
  forced host CPU devices — and anything short
  of two usable devices degrades to the single-device path unchanged.
"""

from __future__ import annotations

import dataclasses
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


def _pas_deck(
    er: float = 0.0,
    *,
    epsilon_h: float = 0.05,
    dndr: float = -0.5,
    n_theta: int = 7,
    n_zeta: int = 7,
    n_xi: int = 8,
    n_x: int = 3,
) -> str:
    """A tiny non-axisymmetric two-species PAS deck (RHSMode=1, ``Er`` knob)."""
    return f"""&general
/
&geometryParameters
  geometryScheme = 1
  inputRadialCoordinate = 3
  rN_wish = 0.3
  B0OverBBar = 1.0
  GHat = 1.0
  IHat = 0.0
  iota = 1.31
  epsilon_t = 0.1
  epsilon_h = {epsilon_h}
  helicity_l = 2
  helicity_n = 5
  psiAHat = 0.045
  aHat = 0.1
/
&speciesParameters
  Zs = 1 -1
  mHats = 1.0 0.000545509
  nHats = 1.0 1.0
  THats = 1.0 1.0
  dNHatdrHats = {dndr} {dndr}
  dTHatdrHats = -1.0 -1.0
/
&physicsParameters
  Delta = 4.5694d-3
  alpha = 1.0
  nu_n = 8.4774d-3
  Er = {er}
  collisionOperator = 1
  includeXDotTerm = .false.
  includeElectricFieldTermInXiDot = .false.
  useDKESExBDrift = .true.
  includePhi1 = .false.
/
&resolutionParameters
  Ntheta = {n_theta}
  Nzeta = {n_zeta}
  Nxi = {n_xi}
  NL = 4
  Nx = {n_x}
  solverTolerance = 1d-10
/
&otherNumericalParameters
  Nxi_for_x_option = 0
/
"""


def _write(tmp_path: Path, text: str, name: str = "input.namelist") -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


def _build_op(tmp_path: Path, *, epsilon_h: float, dndr: float, name: str):
    from dkx.drift_kinetic import kinetic_operator_from_namelist
    from dkx.inputs import load_sfincs_input

    deck = _pas_deck(epsilon_h=epsilon_h, dndr=dndr)
    raw = load_sfincs_input(_write(tmp_path, deck, name)).raw
    return kinetic_operator_from_namelist(raw)


# ---------------------------------------------------------------------------
# 1a. E_r scan: batched == serial per-element to ~1e-12
# ---------------------------------------------------------------------------


def test_batched_er_scan_matches_serial(tmp_path: Path) -> None:
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from dkx import batch as batch_mod
    from dkx import er as er_mod

    prob = er_mod.prepare(_write(tmp_path, _pas_deck()), er_bracket=(-5.0, 5.0))
    er_values = jnp.asarray([-2.0, -1.0, -0.5, 0.0, 0.5, 0.7], dtype=jnp.float64)

    result = batch_mod.batched_er_scan(prob, er_values)

    assert result.states.shape[0] == er_values.shape[0]
    assert result.radial_current.shape == er_values.shape
    assert result.residual_norms.shape == er_values.shape
    assert np.all(result.algebraic_converged)
    assert np.all(result.relative_residual_norms <= prob.tol)
    for i, er_value in enumerate(er_values):
        op_i = er_mod.operator_at_er(prob.operator, er_value, dphi_per_er=prob.dphi_per_er)
        rhs = op_i.rhs()
        relative = jnp.linalg.norm(op_i.apply(result.states[i]) - rhs) / jnp.linalg.norm(rhs)
        assert float(relative) < 1e-10, float(relative)

    for i, er in enumerate(np.asarray(er_values)):
        j_r, gamma, state = er_mod.radial_current(prob, float(er))
        np.testing.assert_allclose(
            np.asarray(result.radial_current[i]), float(j_r), rtol=0.0, atol=1e-12
        )
        np.testing.assert_allclose(
            np.asarray(result.moments["particleFlux_vm_psiHat"][i]),
            np.asarray(gamma, dtype=np.float64),
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            np.asarray(result.states[i]),
            np.asarray(jnp.reshape(state.result.x, (-1,))),
            rtol=0.0,
            atol=1e-12,
        )


# ---------------------------------------------------------------------------
# 1b. Surface scan: batched == serial per-element to ~1e-12
# ---------------------------------------------------------------------------


def test_batched_surface_scan_matches_serial(tmp_path: Path) -> None:
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from dkx import batch as batch_mod
    from dkx.run import profile_moments_from_operator
    from dkx.solve import solve

    specs = [(0.03, -0.5), (0.05, -0.6), (0.07, -0.4)]
    ops = [
        _build_op(tmp_path, epsilon_h=eh, dndr=dn, name=f"surface_{i}.namelist")
        for i, (eh, dn) in enumerate(specs)
    ]

    result = batch_mod.batched_surface_scan(ops)
    assert result.states.shape[0] == len(ops)

    for i, op in enumerate(ops):
        state = jnp.reshape(solve(op, op.rhs(), method="auto", tol=1e-10).x, (-1,))
        moments = profile_moments_from_operator(op, state)
        np.testing.assert_allclose(
            np.asarray(result.states[i]), np.asarray(state), rtol=0.0, atol=1e-12
        )
        for key in ("FSABjHat", "particleFlux_vm_psiHat", "heatFlux_vm_psiHat"):
            np.testing.assert_allclose(
                np.asarray(result.moments[key][i]),
                np.asarray(moments[key]),
                rtol=0.0,
                atol=1e-12,
            )


# ---------------------------------------------------------------------------
# 2. Differentiable: grad of a batched scalar == per-element grads == central FD
# ---------------------------------------------------------------------------


def test_batched_grad_matches_per_element_and_fd(tmp_path: Path) -> None:
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from dkx import batch as batch_mod
    from dkx import er as er_mod
    from dkx.run import profile_moments_from_operator
    from dkx.solve import solve

    prob = er_mod.prepare(_write(tmp_path, _pas_deck()), er_bracket=(-5.0, 5.0))
    er_values = jnp.asarray([-2.0, -0.5, 0.5], dtype=jnp.float64)
    dphi = jnp.asarray(prob.dphi_per_er, dtype=jnp.float64) * er_values
    batch_leaves = {"dphi_hat_dpsi_hat": dphi, "dphi_hat_dpsi_hat_kinetic": dphi}
    base_dn = prob.operator.dn_hat_dpsi_hat

    def batched_bootstrap_sum(scale):
        op = dataclasses.replace(prob.operator, dn_hat_dpsi_hat=base_dn * scale)
        res = batch_mod.batched_solve(op, batch_leaves, differentiable=True)
        return jnp.sum(res.moments["FSABjHat"])

    grad_batched = float(jax.grad(batched_bootstrap_sum)(1.0))

    # Per-element gradient sum (the batch scalar is a sum, so grads add).
    def element_bootstrap(scale, dphi_scalar):
        op = dataclasses.replace(
            prob.operator,
            dn_hat_dpsi_hat=base_dn * scale,
            dphi_hat_dpsi_hat=dphi_scalar,
            dphi_hat_dpsi_hat_kinetic=dphi_scalar,
        )
        state = jnp.reshape(
            solve(op, op.rhs(), method="auto", tol=1e-10, differentiable=True).x, (-1,)
        )
        return profile_moments_from_operator(op, state)["FSABjHat"]

    grad_elements = sum(
        float(jax.grad(element_bootstrap)(1.0, float(d))) for d in np.asarray(dphi)
    )

    # Central finite difference of the batched scalar.
    eps = 1e-4
    fd = float(
        (batched_bootstrap_sum(1.0 + eps) - batched_bootstrap_sum(1.0 - eps)) / (2 * eps)
    )

    assert abs(grad_batched - grad_elements) <= 1e-9
    assert abs(grad_batched - fd) / (abs(fd) + 1e-30) <= 1e-6


# ---------------------------------------------------------------------------
# 3. jit-safety: jax.jit(batched_solve) compiles and runs, matches eager
# ---------------------------------------------------------------------------


def test_jit_batched_solve_compiles_and_matches(tmp_path: Path) -> None:
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from dkx import batch as batch_mod
    from dkx import er as er_mod

    prob = er_mod.prepare(_write(tmp_path, _pas_deck()), er_bracket=(-5.0, 5.0))
    er_values = jnp.asarray([-1.5, 0.0, 0.6], dtype=jnp.float64)
    dphi = jnp.asarray(prob.dphi_per_er, dtype=jnp.float64) * er_values
    batch_leaves = {"dphi_hat_dpsi_hat": dphi, "dphi_hat_dpsi_hat_kinetic": dphi}

    eager = batch_mod.batched_solve(prob.operator, batch_leaves)
    jitted = jax.jit(lambda leaves: batch_mod.batched_solve(prob.operator, leaves))(
        batch_leaves
    )

    from dkx.solve import _auto_route_structural

    assert eager.method == "auto"
    assert eager.executed_method == _auto_route_structural(prob.operator)
    assert jitted.executed_method == eager.executed_method

    np.testing.assert_allclose(
        np.asarray(jitted.states), np.asarray(eager.states), rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(jitted.moments["FSABjHat"]),
        np.asarray(eager.moments["FSABjHat"]),
        rtol=0.0,
        atol=1e-12,
    )


# ---------------------------------------------------------------------------
# 4. Memory-budgeted auto-chunking: budget/max_batch honored, chunked == single
# ---------------------------------------------------------------------------


def test_auto_chunk_size_respects_budget_and_max_batch(tmp_path: Path) -> None:
    import jax

    jax.config.update("jax_enable_x64", True)

    from dkx import batch as batch_mod
    from dkx import er as er_mod

    prob = er_mod.prepare(_write(tmp_path, _pas_deck()), er_bracket=(-5.0, 5.0))
    op = prob.operator

    assert batch_mod.solve_footprint_bytes(op) > 0.0
    assert batch_mod.resolve_memory_budget_bytes(4.0) == pytest.approx(4.0 * 2**30)

    # A generous budget fits the whole batch in one chunk.
    assert batch_mod.auto_chunk_size(op, 8, memory_budget_gb=64.0) == 8
    # A tiny budget forces one solve at a time (never zero).
    assert batch_mod.auto_chunk_size(op, 8, memory_budget_gb=1e-6) == 1
    # max_batch caps the chunk; the batch length caps it too.
    assert batch_mod.auto_chunk_size(op, 8, max_batch=3, memory_budget_gb=64.0) == 3
    assert batch_mod.auto_chunk_size(op, 2, memory_budget_gb=64.0) == 2

    with pytest.raises(ValueError):
        batch_mod.auto_chunk_size(op, 0)
    with pytest.raises(ValueError):
        batch_mod.auto_chunk_size(op, 4, max_batch=0)


def test_batch_memory_budget_controls_route_and_preserves_moments(tmp_path: Path) -> None:
    """The batch budget must reach each solve, not only size the outer chunk."""

    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from dkx import batch as batch_mod
    from dkx import er as er_mod

    prob = er_mod.prepare(_write(tmp_path, _pas_deck()), er_bracket=(-5.0, 5.0))
    er_values = jnp.asarray([-0.5, 0.5], dtype=jnp.float64)

    full = batch_mod.batched_er_scan(prob, er_values, memory_budget_gb=64.0)
    bounded = batch_mod.batched_er_scan(prob, er_values, memory_budget_gb=1e-6)

    assert full.executed_method == "block_tridiagonal"
    assert bounded.executed_method == "block_tridiagonal_truncated"
    assert bounded.chunk_size == 1
    np.testing.assert_allclose(
        np.asarray(bounded.radial_current),
        np.asarray(full.radial_current),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(bounded.moments["particleFlux_vm_psiHat"]),
        np.asarray(full.moments["particleFlux_vm_psiHat"]),
        rtol=0.0,
        atol=1e-12,
    )


def test_truncated_batch_retains_selected_tail_upper_bound(tmp_path: Path) -> None:
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from dkx import batch as batch_mod
    from dkx import er as er_mod
    from dkx.moments import legendre_tail_relative_l2_batch
    from dkx.writer import operator_containers

    prob = er_mod.prepare(_write(tmp_path, _pas_deck()), er_bracket=(-5.0, 5.0))
    er_values = jnp.asarray([-0.5, 0.5], dtype=jnp.float64)
    full = batch_mod.batched_er_scan(prob, er_values, memory_budget_gb=64.0)
    omitted = batch_mod.batched_er_scan(prob, er_values, memory_budget_gb=1e-6)
    retained = batch_mod.batched_er_scan(
        prob,
        er_values,
        memory_budget_gb=1e-6,
        retain_legendre_tail=True,
    )

    assert omitted.legendre_tail_relative_l2_upper_bound is None
    bound = np.asarray(retained.legendre_tail_relative_l2_upper_bound)
    exact = np.asarray(
        legendre_tail_relative_l2_batch(
            *operator_containers(prob.operator)[:3], full.states
        )
    )
    assert bound.shape == exact.shape == (2, prob.operator.n_x, prob.operator.n_species)
    assert np.all(np.isfinite(bound))
    assert np.all(bound + 1e-13 >= exact)
    assert batch_mod.solve_footprint_bytes(
        prob.operator,
        memory_budget_gb=1e-6,
        retain_legendre_tail=True,
    ) > batch_mod.solve_footprint_bytes(prob.operator, memory_budget_gb=1e-6)


def test_solve_footprint_models_the_truncated_route(tmp_path: Path) -> None:
    """A production-shaped op is charged the truncated working set, not the bands.

    At (S=2, X=5, L=100, T=25, Z=51) — the 1.27M-DOF production shape — the
    full-band factorization peak is ~53 GB, far above the structured direct budget, so
    ``method="auto"`` routes to the truncated block-Thomas kernel whose
    measured working set is ~1.2 GB.  The footprint model must follow that
    route: charging the full-band peak caps batched scans at ``chunk=1`` and
    silently serializes them.
    """
    import jax

    jax.config.update("jax_enable_x64", True)

    from dkx import batch as batch_mod
    from dkx import solve as solve_mod
    from dkx.drift_kinetic import kinetic_operator_from_namelist
    from dkx.inputs import load_sfincs_input

    deck = _pas_deck(n_theta=25, n_zeta=51, n_xi=100, n_x=5)
    raw = load_sfincs_input(_write(tmp_path, deck, "production_shape.namelist")).raw
    op = kinetic_operator_from_namelist(raw)

    gb = 2.0**30
    # The auto-router would take the truncated kernel (full-band peak >> budget).
    assert solve_mod._auto_route_structural(op) == "block_tridiagonal_truncated"
    assert solve_mod.tier1_peak_memory_bytes(op) > 40.0 * gb
    # The footprint follows the truncated route: ~1.7 GB, not ~53 GB.
    assert batch_mod.solve_footprint_bytes(op) < 3.0 * gb
    # Chunk arithmetic at a 19.2 GB budget batches at least 8 solves per chunk.
    assert batch_mod.auto_chunk_size(op, 64, memory_budget_gb=19.2) >= 8


def test_chunked_batch_matches_single_chunk(tmp_path: Path) -> None:
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from dkx import batch as batch_mod
    from dkx import er as er_mod

    prob = er_mod.prepare(_write(tmp_path, _pas_deck()), er_bracket=(-5.0, 5.0))
    er_values = jnp.asarray([-2.0, -1.0, -0.5, 0.0, 0.5], dtype=jnp.float64)

    single = batch_mod.batched_er_scan(prob, er_values)
    chunked = batch_mod.batched_er_scan(prob, er_values, max_batch=2)

    assert single.chunk_size == er_values.shape[0]
    assert chunked.chunk_size == 2
    assert chunked.n_chunks == 3  # ceil(5 / 2)
    np.testing.assert_allclose(
        np.asarray(chunked.radial_current),
        np.asarray(single.radial_current),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(chunked.states), np.asarray(single.states), rtol=0.0, atol=1e-12
    )


# ---------------------------------------------------------------------------
# 5. Validation: discretization leaves may not be batched / must agree
# ---------------------------------------------------------------------------


def test_batching_discretization_leaf_is_rejected(tmp_path: Path) -> None:
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from dkx import batch as batch_mod
    from dkx import er as er_mod

    prob = er_mod.prepare(_write(tmp_path, _pas_deck()), er_bracket=(-5.0, 5.0))
    op = prob.operator

    with pytest.raises(ValueError, match="discretization leaf"):
        batch_mod.batched_solve(op, {"ddtheta": jnp.stack([op.ddtheta, op.ddtheta])})
    with pytest.raises(ValueError, match="not a batchable"):
        batch_mod.batched_solve(op, {"not_a_field": jnp.zeros((2,))})
    with pytest.raises(ValueError, match="empty"):
        batch_mod.batched_solve(op, {})


def test_surface_scan_requires_shared_discretization(tmp_path: Path) -> None:
    import jax

    jax.config.update("jax_enable_x64", True)

    from dkx import batch as batch_mod
    from dkx.drift_kinetic import kinetic_operator_from_namelist
    from dkx.inputs import load_sfincs_input

    op_a = _build_op(tmp_path, epsilon_h=0.05, dndr=-0.5, name="a.namelist")
    # A different Ntheta changes the discretization -> must be rejected.
    raw_b = load_sfincs_input(
        _write(tmp_path, _pas_deck(n_theta=9), "b.namelist")
    ).raw
    op_b = kinetic_operator_from_namelist(raw_b)

    with pytest.raises(ValueError, match="discretization|structure"):
        batch_mod.batched_surface_scan([op_a, op_b])

    with pytest.raises(ValueError, match="at least one"):
        batch_mod.batched_surface_scan([])


# ---------------------------------------------------------------------------
# 6. API facades route to the same result
# ---------------------------------------------------------------------------


def test_api_batched_facades_route(tmp_path: Path) -> None:
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from dkx import api
    from dkx import batch as batch_mod
    from dkx import er as er_mod

    deck_path = _write(tmp_path, _pas_deck())
    er_values = jnp.asarray([-1.5, -0.3, 0.4], dtype=jnp.float64)

    prob = er_mod.prepare(deck_path, er_bracket=(-5.0, 5.0))
    direct = batch_mod.batched_er_scan(prob, er_values)
    via_api = api.batched_er_scan(str(deck_path), er_values, er_bracket=(-5.0, 5.0))
    np.testing.assert_allclose(
        np.asarray(via_api.radial_current),
        np.asarray(direct.radial_current),
        rtol=0.0,
        atol=1e-12,
    )

    # Omitted options preserve the prepared problem's numerical policy.
    prepared = dataclasses.replace(prob, solve_method="gmres", tol=3e-9)
    inherited = api.batched_er_scan(prepared, er_values)
    assert inherited.method == inherited.executed_method == "gmres"
    expected = batch_mod.batched_er_scan(prepared, er_values)
    np.testing.assert_array_equal(inherited.states, expected.states)
    override = api.batched_er_scan(prepared, er_values, solve_method="auto", tol=1e-10)
    np.testing.assert_array_equal(override.states, direct.states)

    specs = [(0.04, -0.5), (0.06, -0.55)]
    ops = [
        _build_op(tmp_path, epsilon_h=eh, dndr=dn, name=f"api_surface_{i}.namelist")
        for i, (eh, dn) in enumerate(specs)
    ]
    direct_surf = batch_mod.batched_surface_scan(ops)
    via_api_surf = api.batched_surface_scan(ops)
    np.testing.assert_allclose(
        np.asarray(via_api_surf.moments["FSABjHat"]),
        np.asarray(direct_surf.moments["FSABjHat"]),
        rtol=0.0,
        atol=1e-12,
    )


# ---------------------------------------------------------------------------
# 7. Multi-device split: identity across device counts, degrade, per-device
#    chunk arithmetic
# ---------------------------------------------------------------------------


def _assert_batched_results_identical(a, b) -> None:
    """Element-wise (bitwise) identity of two ``BatchedSolveResult`` payloads."""
    assert np.array_equal(np.asarray(a.states), np.asarray(b.states))
    if a.radial_current is not None:
        assert np.array_equal(
            np.asarray(a.radial_current), np.asarray(b.radial_current)
        )
    assert set(a.moments) == set(b.moments)
    for key in a.moments:
        assert np.array_equal(np.asarray(a.moments[key]), np.asarray(b.moments[key])), key


def test_devices_degrade_and_validation(tmp_path: Path) -> None:
    """One usable device (or a too-small batch) degrades to the existing path."""
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from dkx import batch as batch_mod
    from dkx import er as er_mod

    d0 = jax.devices()[0]

    # Resolution rules (pure arithmetic, no dispatch).
    assert batch_mod._resolve_devices(None, 8) is None
    assert batch_mod._resolve_devices([d0], 8) is None  # one device -> degrade
    with pytest.raises(ValueError, match="distinct physical"):
        batch_mod._resolve_devices([d0, d0], 2)
    if len(jax.local_devices()) == 1:
        assert batch_mod._resolve_devices("auto", 8) is None
    with pytest.raises(ValueError, match="not recognised"):
        batch_mod._resolve_devices("all", 8)
    with pytest.raises(ValueError, match="non-empty"):
        batch_mod._resolve_devices([], 8)

    # End-to-end degrade: `devices` that resolve to one device produce the
    # identical result and chunk metadata as the default path.
    prob = er_mod.prepare(_write(tmp_path, _pas_deck()), er_bracket=(-5.0, 5.0))
    er_values = jnp.asarray([-1.0, 0.0, 0.5], dtype=jnp.float64)
    base = batch_mod.batched_er_scan(prob, er_values)
    for devices in ([d0], "auto") if len(jax.local_devices()) == 1 else ([d0],):
        degraded = batch_mod.batched_er_scan(prob, er_values, devices=devices)
        assert degraded.chunk_size == base.chunk_size
        assert degraded.n_chunks == base.n_chunks
        _assert_batched_results_identical(base, degraded)


def test_batch_status_rejects_bad_states_and_handles_zero_drive(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import jax
    import jax.numpy as jnp
    from dkx import batch as batch_mod
    from dkx import er as er_mod

    problem = er_mod.prepare(_write(tmp_path, _pas_deck()), er_bracket=(-5, 5))
    # Manufactured states test the gate independently of a solver's success flag.
    def manufactured(op, rhs, **kwargs):
        x = jnp.zeros(op.total_size)
        residual = jnp.linalg.norm(op.apply(x) - rhs.reshape(-1))
        x = jnp.where(op.dphi_hat_dpsi_hat > 0, jnp.nan, x)
        return SimpleNamespace(x=x, residual_norms=jnp.asarray([residual]), converged=True)
    monkeypatch.setattr(batch_mod, 'solve', manufactured)
    fields = jnp.asarray([-1.0, 0.0, 1.0])
    result = jax.jit(lambda e: batch_mod.batched_er_scan(problem, e))(fields)
    assert not np.any(result.algebraic_converged)
    np.testing.assert_allclose(result.relative_residual_norms, 1.0)
    # Manufacture a homogeneous equation; nonzero Er itself can drive the DKE
    # even when the prescribed density/temperature gradients vanish.
    monkeypatch.setattr(type(problem.operator), "rhs", lambda self: jnp.zeros(self.total_size))
    zero = problem
    exact = jax.jit(lambda e: batch_mod.batched_er_scan(zero, e))(fields)
    expected = np.asarray(zero.dphi_per_er * fields) <= 0
    np.testing.assert_array_equal(exact.algebraic_converged, expected)
    np.testing.assert_array_equal(exact.relative_residual_norms, 0.0)

    def nonzero(op, rhs, **kwargs):
        x = jnp.ones(op.total_size)
        return SimpleNamespace(x=x, residual_norms=jnp.asarray([jnp.linalg.norm(op.apply(x) - rhs.reshape(-1))]))
    monkeypatch.setattr(batch_mod, 'solve', nonzero)
    rejected = jax.jit(lambda e: batch_mod.batched_er_scan(zero, e))(fields)
    assert not np.any(rejected.algebraic_converged)
    assert np.all(np.isinf(rejected.relative_residual_norms))
    for tol in (float('nan'), float('inf'), -1.0):
        with pytest.raises(ValueError, match='finite and nonnegative'):
            batch_mod.batched_er_scan(problem, fields, tol=tol)


_TWO_DEVICE_SCRIPT = """
import sys

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from dkx import batch as batch_mod
from dkx import er as er_mod
from dkx import batched_er_scan, batched_surface_scan

assert len(jax.devices()) == 2, f"expected 2 forced CPU devices, got {jax.devices()}"

prob = er_mod.prepare(sys.argv[1], er_bracket=(-5.0, 5.0))
er_values = jnp.asarray([-2.0, -1.5, -1.0, -0.5, 0.0, 0.3, 0.5, 0.7])


def eq(a, b):
    return np.array_equal(np.asarray(a), np.asarray(b))


def close(a, b):
    # The default config executes different lax.map widths on the single- and
    # sharded paths (8 vs 4 per shard); XLA CPU may emit width-dependent code
    # whose results differ in the last ulp on some ISAs, so the default-config
    # gate uses the same tight tolerance as the chunked-equivalence test.
    return np.allclose(np.asarray(a), np.asarray(b), rtol=0.0, atol=1e-12)


# Default config: single device vs the two-device split.
single = batch_mod.batched_er_scan(prob, er_values)
multi = batch_mod.batched_er_scan(prob, er_values, devices="auto")
assert single.chunk_size == 8 and single.n_chunks == 1
assert multi.chunk_size == 4 and multi.n_chunks == 1  # proves the split ran
assert close(single.states, multi.states)
assert close(single.radial_current, multi.radial_current)
for key in single.moments:
    assert close(single.moments[key], multi.moments[key]), key

# Matched chunk width retains bitwise states/current on this fixture. Derived
# moment reductions/divisions can be fused differently across shard boundaries:
# x86 CPU measurements differ by <=8.7e-19 even with identical states.
single2 = batch_mod.batched_er_scan(prob, er_values, max_batch=2)
multi2 = batch_mod.batched_er_scan(prob, er_values, max_batch=2, devices="auto")
assert multi2.chunk_size == 2 and multi2.n_chunks == 2
assert eq(single2.states, multi2.states)
assert eq(single2.radial_current, multi2.radial_current)
for key in single2.moments:
    np.testing.assert_allclose(single2.moments[key], multi2.moments[key],
                               rtol=0.0, atol=1e-12, err_msg=key)

# A batch smaller than the device count degrades to the single-device path.
one = batch_mod.batched_er_scan(prob, er_values[:1], devices="auto")
ref = batch_mod.batched_er_scan(prob, er_values[:1])
assert one.chunk_size == ref.chunk_size == 1
assert eq(one.states, ref.states)

# Device placement is evidence of distribution; metadata alone is not.
assert len(multi2.states.addressable_shards) == 2
assert not multi2.states.is_fully_replicated
assert {s.device for s in multi2.states.addressable_shards} == set(jax.devices())

# Uneven batches retain order and discard padded cases, including in reverse mode.
values = er_values[:5]
def scan(values, devices):
    return batched_er_scan(
        prob, values, devices=devices, max_batch=2, differentiable=True)
for devices in (None, "auto"):
    result = jax.jit(lambda v: scan(v, devices))(values)
    assert close(result.states, single.states[:5])
    assert np.all(result.algebraic_converged)
    assert np.all(np.asarray(result.relative_residual_norms) < 1e-10)
    for i, er_value in enumerate(values):
        op_i = er_mod.operator_at_er(prob.operator, er_value, dphi_per_er=prob.dphi_per_er)
        rhs = op_i.rhs()
        relative = jnp.linalg.norm(op_i.apply(result.states[i]) - rhs) / jnp.linalg.norm(rhs)
        assert float(relative) < 1e-10, float(relative)
    assert result.chunk_size == 2
    assert result.n_chunks == (3 if devices is None else 2)

def objective(v, devices):
    return jnp.sum(scan(v, devices).moments["FSABjHat"])
value, grad = jax.jit(jax.value_and_grad(lambda v: objective(v, "auto")))(values)
ref_value, ref_grad = jax.jit(jax.value_and_grad(lambda v: objective(v, None)))(values)
np.testing.assert_allclose(value, ref_value, rtol=1e-10, atol=1e-12)
np.testing.assert_allclose(grad, ref_grad, rtol=1e-9, atol=1e-12)
direction = jnp.asarray([0.1, -0.2, 0.3, -0.1, 0.2])
step = 1e-3
fd = (objective(values + step * direction, None) -
      objective(values - step * direction, None)) / (2 * step)
np.testing.assert_allclose(jnp.vdot(grad, direction), fd, rtol=1e-4, atol=1e-12)

roomy = batch_mod.batched_er_scan(prob, er_values, memory_budget_gb=64, devices="auto")
tight = batch_mod.batched_er_scan(prob, er_values, memory_budget_gb=1e-6, devices="auto")
assert roomy.chunk_size == 4 and roomy.n_chunks == 1
assert tight.chunk_size == 1 and tight.n_chunks == 4
tight_ref = batch_mod.batched_er_scan(prob, er_values, memory_budget_gb=1e-6)
assert tight.executed_method == tight_ref.executed_method
assert close(tight.states, tight_ref.states)

# The public surface facade preserves actual placement and diagnostics too.
surface_ops = [er_mod.operator_at_er(prob.operator, e, dphi_per_er=prob.dphi_per_er)
               for e in er_values[:2]]
surface_result = batched_surface_scan(surface_ops, devices="auto")
assert not surface_result.states.is_fully_replicated
assert len(surface_result.states.addressable_shards) == 2
assert close(surface_result.states, single.states[:2])
assert close(surface_result.residual_norms, single.residual_norms[:2])

# Full-FP Krylov solves use the same independent-batch contract.
fp_prob = er_mod.prepare(sys.argv[2], er_bracket=(-1.0, 1.0), solve_method="gmres")
fp_values = jnp.asarray([-0.1, 0.2])
def fp_scan(v, devices):
    return batch_mod.batched_er_scan(
        fp_prob, v, devices=devices, differentiable=True, max_batch=1)
for devices in (None, "auto"):
    result = jax.jit(lambda v: fp_scan(v, devices))(fp_values)
    grad = jax.jit(jax.grad(lambda v: jnp.sum(fp_scan(v, devices).moments["FSABjHat"])))(fp_values)
    for i, er_value in enumerate(fp_values):
        op_i = er_mod.operator_at_er(fp_prob.operator, er_value, dphi_per_er=fp_prob.dphi_per_er)
        rhs = op_i.rhs()
        assert float(jnp.linalg.norm(op_i.apply(result.states[i]) - rhs) / jnp.linalg.norm(rhs)) < 1e-10
    if devices is None:
        fp_ref, grad_ref = result, grad
    else:
        np.testing.assert_allclose(result.states, fp_ref.states, rtol=1e-8, atol=1e-11)
        np.testing.assert_allclose(grad, grad_ref, rtol=1e-8, atol=1e-11)

print("MULTI_DEVICE_IDENTITY_OK")
"""


def test_two_forced_cpu_devices_identity(tmp_path: Path) -> None:
    """1-vs-2 forced host CPU devices: states/current and moments agree.

    Actual shard placement, uneven batches and reverse-mode sensitivities are
    checked independently of metadata. GPU validation remains separate. Device
    forcing must happen before JAX initializes, hence the fresh subprocess.
    """
    deck_path = _write(tmp_path, _pas_deck())
    fp_path = _write(tmp_path, _pas_deck(n_theta=5, n_zeta=5, n_xi=4).replace(
        "collisionOperator = 1", "collisionOperator = 0"), "fp.namelist")
    script_path = tmp_path / "two_device_identity.py"
    script_path.write_text(_TWO_DEVICE_SCRIPT)

    repo_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["DKX_CPU_DEVICES"] = "2"
    env.pop("XLA_FLAGS", None)  # derive the forced-device flag fresh
    env["JAX_ENABLE_X64"] = "True"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(repo_root)] + [p for p in [env.get("PYTHONPATH", "")] if p]
    )

    proc = subprocess.run(
        [sys.executable, str(script_path), str(deck_path), str(fp_path)],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "MULTI_DEVICE_IDENTITY_OK" in proc.stdout
