"""Referee tests for ``dkx.solve`` — the plan-§2.3 three-route auto-policy.

Tiny fixtures only (shared with ``tests/test_drift_kinetic.py``):

- structured direct (analytic block-Thomas) must match the sparse direct solve to 1e-10
  on the monoenergetic (RHSMode=3) and PAS (RHSMode=1) fixtures, and match the
  recorded Fortran v3 ``stateVector`` fixtures on the RHSMode=3 transport
  columns (the referee formerly provided by the retired probing-based
  ``solvers/block_tridiagonal_transport`` POC);
- recycled Krylov (GCROT + coarse-operator preconditioner) must converge on the
  Fokker-Planck two-species fixture, match sparse direct to 1e-8, and need
  strictly fewer iterations than the unpreconditioned solve;
- the auto-policy must pick structured direct for the PAS family and recycled Krylov for FP;
- recycling + warm start across an Er continuation must cut iterations;
- ``jax.grad`` through the differentiable structured direct solve must match finite
  differences.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import dataclasses

import numpy as np
import pytest
import scipy.linalg as sla

from dkx.drift_kinetic import KineticOperator
from dkx.namelist import parse_sfincs_input_text, read_sfincs_input
from dkx.coarse_precond import _COARSE_DIAGONAL_FLOOR, _l0_pin_gamma
from dkx.solve import (
    SolveResult,
    _resolve_subsystem_batch,
    build_coarse_preconditioner,
    materialize_dense,
    solve,
    tier1_available,
    tier1_full_band_bytes,
    tier1_peak_memory_bytes,
    tier1_truncated_peak_memory_bytes,
    tier1_truncated_subsystem_width,
    tier1_truncated_tail_blocks,
)

REF = Path(__file__).parent / "ref"


def _load_op(name: str) -> KineticOperator:
    return KineticOperator.from_namelist(read_sfincs_input(REF / f"{name}.input.namelist"))


def _load_text(name: str) -> str:
    return (REF / f"{name}.input.namelist").read_text()


def _load_sugama_op() -> KineticOperator:
    """A ``collisionOperator=3`` (improved Sugama) deck from the FP fixture text.

    Reuses the two-species Fokker-Planck namelist with ``collisionOperator``
    switched to 3 (no new golden fixture): the improved Sugama operator has the
    same dense ``(species, x)`` block layout as Fokker-Planck and defaults to
    ``constraintScheme=1``.
    """
    txt = _load_text("quick_2species_FPCollisions_noEr").replace(
        "collisionOperator = 0", "collisionOperator = 3"
    )
    return KineticOperator.from_namelist(parse_sfincs_input_text(txt))


def _dense_solve(op: KineticOperator, rhs2d: np.ndarray) -> np.ndarray:
    return sla.solve(materialize_dense(op), rhs2d)


def _rel_err(x: np.ndarray, ref: np.ndarray) -> float:
    scale = max(1.0, float(np.max(np.abs(ref))))
    return float(np.max(np.abs(x - ref))) / scale


# ---------------------------------------------------------------------------
# Structured direct == sparse direct on the structured direct family
# ---------------------------------------------------------------------------


def test_tier1_matches_dense_monoenergetic_rhsmode3() -> None:
    op = _load_op("monoenergetic_PAS_tiny_scheme1")
    rhs = jnp.stack([op.rhs(1), op.rhs(2)], axis=1)  # both transport drives
    result = solve(op, rhs, method="block_tridiagonal")
    assert result.method == "block_tridiagonal"
    assert result.converged
    x_ref = _dense_solve(op, np.asarray(rhs))
    assert _rel_err(np.asarray(result.x), x_ref) < 1e-10


def test_tier1_matches_dense_pas_rhsmode1() -> None:
    op = _load_op("pas_1species_PAS_noEr_tiny_scheme1")
    rhs = op.rhs()
    result = solve(op, rhs, method="block_tridiagonal")
    assert result.converged
    x_ref = _dense_solve(op, np.asarray(rhs)[:, None])[:, 0]
    assert _rel_err(np.asarray(result.x), x_ref) < 1e-10
    # rhs was 1-D: the solution must come back 1-D.
    assert result.x.shape == rhs.shape


@pytest.mark.parametrize("base", ("monoenergetic_PAS_tiny_scheme1", "monoenergetic_PAS_tiny_scheme11"))
def test_tier1_matches_recorded_fortran_state_vectors_rhsmode3(base: str) -> None:
    """Structured direct must reproduce the frozen v3 PETSc stateVector for both transport drives.

    This is the direct Fortran referee for the structured direct route on the
    RHSMode=3 transport columns; it replaces the equality test against the
    retired probing-based ``solvers/block_tridiagonal_transport`` POC.
    """
    from dkx.validation.fortran import read_petsc_vec

    op = _load_op(base)
    rhs = jnp.stack([op.rhs(1), op.rhs(2)], axis=1)
    result = solve(op, rhs, method="block_tridiagonal")
    assert result.converged
    x = np.asarray(result.x)
    for which_rhs in (1, 2):
        x_ref = read_petsc_vec(REF / f"{base}.whichRHS{which_rhs}.stateVector.petscbin").values
        assert _rel_err(x[:, which_rhs - 1], x_ref) < 1e-10, f"whichRHS={which_rhs}"


# ---------------------------------------------------------------------------
# Truncated structured direct: memory-driven routing, full parity, moments, gradients
# ---------------------------------------------------------------------------


def _vm_moments(op: KineticOperator, x_stack: np.ndarray) -> dict[str, np.ndarray]:
    """vm particle/heat fluxes + FSABFlow of solved states (moments module)."""
    from dkx.moments import (
        FluxSurface,
        SpeciesParams,
        StateLayout,
        VelocityGrid,
        vm_flux_moments_batch,
    )

    layout = StateLayout(
        n_species=op.n_species, n_x=op.n_x, n_xi=op.n_xi, n_theta=op.n_theta,
        n_zeta=op.n_zeta, include_phi1=False, constraint_scheme=op.constraint_scheme,
    )  # fmt: skip
    vgrid = VelocityGrid(x=op.x, x_weights=op.x_weights, n_xi_for_x=op.n_xi_for_x)
    surface = FluxSurface.from_operator(op)
    species = SpeciesParams.from_operator(op)
    m = vm_flux_moments_batch(
        layout, vgrid, surface, species, jnp.asarray(x_stack), delta=op.delta, alpha=op.alpha
    )
    return {
        "particleFlux": np.asarray(m.particle_flux_vm_psi_hat),
        "heatFlux": np.asarray(m.heat_flux_vm_psi_hat),
        "FSABFlow": np.asarray(m.fsab_flow),
    }


def test_tier1_memory_estimate_hand_computed() -> None:
    """The full-band byte formula must match a hand-computed value exactly.

    ``bytes = 3 * sum_x(Nxi_for_x) * n_species * (n_theta * n_zeta)**2 * 8`` —
    here n_theta=3, n_zeta=2 (m=6, m**2=36), n_xi=4, n_x=2, n_species=1 with
    uniform Nxi_for_x: 3 * (4+4) * 1 * 36 * 8 = 6912 bytes; the peak estimate
    is 2.5x that.  A ramped Nxi_for_x counts only the retained blocks.
    """
    fake = SimpleNamespace(
        n_theta=3, n_zeta=2, n_xi=4, n_x=2, n_species=1, n_xi_for_x=np.array([4, 4])
    )
    assert tier1_full_band_bytes(fake) == 3 * 4 * 1 * 2 * 36 * 8
    assert tier1_full_band_bytes(fake) == pytest.approx(6912.0)
    assert tier1_peak_memory_bytes(fake) == pytest.approx(2.5 * 6912.0)
    ramped = SimpleNamespace(
        n_theta=3, n_zeta=2, n_xi=4, n_x=2, n_species=1, n_xi_for_x=np.array([3, 4])
    )
    assert tier1_full_band_bytes(ramped) == 3 * (3 + 4) * 1 * 36 * 8


def test_auto_policy_selects_full_vs_truncated_by_budget() -> None:
    op = _load_op("pas_1species_PAS_noEr_tiny_scheme1")
    rhs = op.rhs()
    # Default (8 GB) budget dwarfs this tiny case: full factorization.
    r_full = solve(op, rhs, method="auto")
    assert r_full.method == "block_tridiagonal"
    # A deliberately tiny budget forces the truncated block-Thomas kernel.
    r_trunc = solve(op, rhs, method="auto", tier1_memory_budget_gb=1e-12)
    assert r_trunc.method == "block_tridiagonal_truncated"
    assert r_trunc.converged


def test_auto_policy_truncation_invalid_falls_through_to_tier2() -> None:
    # Structured-direct-eligible operator, but the RHS carries Legendre support at l>=keep:
    # the truncated kernel would be inexact, so auto must fall back to recycled Krylov.
    op = _load_op("pas_1species_PAS_noEr_tiny_scheme1")
    rhs = np.asarray(op.rhs())
    n_s, n_x, n_xi, n_t, n_z = op.f_shape
    f = rhs[: op.f_size].reshape(n_s, n_x, n_xi, n_t * n_z).copy()
    f[0, 0, 3, 0] = 1.0  # inject l=3 support (keep defaults to 3)
    rhs_bad = jnp.concatenate([jnp.asarray(f).reshape(-1), jnp.asarray(rhs[op.f_size :])])
    r = solve(op, rhs_bad, method="auto", tier1_memory_budget_gb=1e-12, tol=1e-9)
    assert r.method == "gcrot"
    assert r.converged


@pytest.mark.parametrize(
    "name,which",
    [
        ("pas_1species_PAS_noEr_tiny_scheme1", None),  # RHSMode=1 drive
        ("monoenergetic_PAS_tiny_scheme1", (1, 2)),  # RHSMode=3 transport drives
    ],
)
def test_truncated_matches_full_lowest_blocks_and_moments(name: str, which) -> None:
    op = _load_op(name)
    if which is None:
        rhs = op.rhs()
    else:
        rhs = jnp.stack([op.rhs(w) for w in which], axis=1)

    full = solve(op, rhs, method="block_tridiagonal")
    trunc = solve(op, rhs, method="block_tridiagonal_truncated")
    assert trunc.method == "block_tridiagonal_truncated"
    assert trunc.converged

    n_s, n_x, n_xi, n_t, n_z = op.f_shape
    n_tz = n_t * n_z
    xf, xt = np.asarray(full.x), np.asarray(trunc.x)
    ff = xf[: op.f_size].reshape(n_s, n_x, n_xi, n_tz, -1)
    ft = xt[: op.f_size].reshape(n_s, n_x, n_xi, n_tz, -1)
    # lowest-3 Legendre blocks are exact; blocks l>=3 are zero-padded.
    dl = np.linalg.norm(ft[:, :, :3] - ff[:, :, :3]) / np.linalg.norm(ff[:, :, :3])
    assert dl < 1e-10
    assert np.max(np.abs(ft[:, :, 3:])) == 0.0

    # Output moments (fluxes / FSABFlow) contract only l<=2, so they match.
    xf_stack = xf.T if xf.ndim == 2 else xf[None, :]
    xt_stack = xt.T if xt.ndim == 2 else xt[None, :]
    m_full, m_trunc = _vm_moments(op, xf_stack), _vm_moments(op, xt_stack)
    for key in ("particleFlux", "heatFlux", "FSABFlow"):
        a, b = m_full[key], m_trunc[key]
        rel = np.abs(a - b) / np.maximum(np.abs(a), 1e-300)
        assert rel.max() < 1e-10, f"{key}: {rel.max():.3e}"


def test_truncated_selected_tail_matches_full_constrained_solution() -> None:
    op = _load_op("pas_1species_PAS_noEr_tiny_scheme1")
    assert op.constraint_scheme == 2
    rhs = op.rhs()
    full = solve(op, rhs, method="block_tridiagonal")
    trunc = solve(op, rhs, method="block_tridiagonal_truncated")
    selected = np.asarray(tier1_truncated_tail_blocks(op, rhs, trunc.x))[..., 0]
    expected = np.asarray(full.x)[: op.f_size].reshape(op.f_shape)
    expected = expected[:, :, -2:].reshape(selected.shape)
    np.testing.assert_allclose(selected, expected, rtol=1e-11, atol=1e-14)


def test_ramped_truncated_selected_tail_matches_independent_full_state() -> None:
    op = _ramped_pas_op()
    rhs = op.rhs()
    trunc = solve(op, rhs, method="block_tridiagonal_truncated")
    selected = np.asarray(tier1_truncated_tail_blocks(op, rhs, trunc.x))[..., 0]
    reference = solve(op, rhs, method="gmres", tol=1e-12)
    f_ref = np.asarray(reference.x)[: op.f_size].reshape(op.f_shape)
    for ix, active in enumerate(np.asarray(op.n_xi_for_x)):
        np.testing.assert_allclose(
            selected[:, ix],
            f_ref[:, ix, int(active) - 2 : int(active)].reshape(selected[:, ix].shape),
            rtol=2e-10,
            atol=1e-14,
        )


def test_gradient_through_truncated_route_matches_finite_differences() -> None:
    op0 = _load_op("pas_1species_PAS_noEr_tiny_scheme1")

    def loss(t_hat_scalar: jnp.ndarray) -> jnp.ndarray:
        op = replace(op0, t_hat=jnp.reshape(t_hat_scalar, (1,)))
        # Tiny budget forces the truncated kernel; grad flows straight through
        # the block-Thomas sweeps (no full-operator IFT wrapper).
        result = solve(
            op, op.rhs(), method="auto", tier1_memory_budget_gb=1e-12, differentiable=True
        )
        return jnp.sum(result.x**2)

    t0 = float(op0.t_hat[0])
    g = float(jax.grad(loss)(jnp.asarray(t0)))
    eps = 1e-6
    fd = float((loss(jnp.asarray(t0 + eps)) - loss(jnp.asarray(t0 - eps))) / (2.0 * eps))
    assert np.isfinite(g) and np.isfinite(fd) and abs(fd) > 0.0
    np.testing.assert_allclose(g, fd, rtol=1e-4)


# ---------------------------------------------------------------------------
# Ramped (non-uniform Nxi_for_x) PAS decks: per-subsystem truncated structured direct
# ---------------------------------------------------------------------------


def _ramped_pas_op() -> KineticOperator:
    """The tiny PAS fixture rescaled so Nxi_for_x_option=1 gives a real ramp."""
    text = (
        _load_text("pas_1species_PAS_noEr_tiny_scheme1")
        .replace("Nxi = 4", "Nxi = 16")
        .replace("Nx = 3", "Nx = 5")
        .replace("Nxi_for_x_option = 0", "Nxi_for_x_option = 1")
    )
    op = KineticOperator.from_namelist(parse_sfincs_input_text(text))
    # The whole point: the production speed-dependent Legendre ramp.
    assert int(np.min(np.asarray(op.n_xi_for_x))) < op.n_xi
    return op


def test_ramped_pas_routes_truncated_and_matches_pinned_referees() -> None:
    """Auto must route ramped PAS decks to the per-subsystem truncated kernel.

    The solution (lowest-3 Legendre blocks and the vm flux/flow moments) must
    match both pinned referees — recycled Krylov and the dense pinned direct
    solve — to 1e-10; blocks l >= 3 are zero-padded (which covers every
    Nxi_for_x-truncated DOF, since keep <= min Nxi_for_x).
    """
    op = _ramped_pas_op()
    rhs = op.rhs()

    trunc = solve(op, rhs, method="auto")
    assert trunc.method == "block_tridiagonal_truncated"
    assert trunc.converged

    gcrot = solve(op, rhs, method="gmres", tol=1e-12)
    assert gcrot.converged
    dense = sla.solve(materialize_dense(op, pin_masked_dofs=True), np.asarray(rhs))

    n_s, n_x, n_xi, n_t, n_z = op.f_shape
    n_tz = n_t * n_z
    xt = np.asarray(trunc.x)
    ft = xt[: op.f_size].reshape(n_s, n_x, n_xi, n_tz)
    assert np.max(np.abs(ft[:, :, 3:])) == 0.0
    for x_ref in (np.asarray(gcrot.x), dense):
        f_ref = x_ref[: op.f_size].reshape(n_s, n_x, n_xi, n_tz)
        dl = np.linalg.norm(ft[:, :, :3] - f_ref[:, :, :3]) / np.linalg.norm(f_ref[:, :, :3])
        assert dl < 1e-10
        # The sources of this drive are numerically zero; agree absolutely.
        np.testing.assert_allclose(
            xt[op.f_size :], x_ref[op.f_size :], atol=1e-12 * np.linalg.norm(f_ref)
        )
        m_t = _vm_moments(op, xt[None, :])
        m_ref = _vm_moments(op, x_ref[None, :])
        for key in ("particleFlux", "heatFlux", "FSABFlow"):
            rel = np.abs(m_t[key] - m_ref[key]) / np.maximum(np.abs(m_ref[key]), 1e-300)
            assert rel.max() < 1e-10, f"{key}: {rel.max():.3e}"

    # The full-band factorization cannot carry the ramp and must refuse.
    with pytest.raises(NotImplementedError, match="uniform Nxi_for_x"):
        solve(op, rhs, method="block_tridiagonal")


@pytest.mark.parametrize("grouped", [False, True])
def test_ramped_full_recovery_matches_original_equation_and_dense(grouped):
    op = _ramped_pas_op()
    if grouped:
        layout = np.asarray(op.n_xi_for_x).copy()
        layout[1] = layout[0]
        op = replace(op, n_xi_for_x=jnp.asarray(layout))
    # Multiple RHSs include a manufactured active-state drive with high-L support.
    mask = op.active_dof_mask()
    manufactured = jnp.sin(jnp.arange(op.total_size, dtype=jnp.float64)) * mask
    rhs = jnp.stack([op.rhs().reshape(-1), op.apply(manufactured)], axis=1)
    reference = sla.solve(materialize_dense(op, pin_masked_dofs=True), np.asarray(rhs))
    for width in (1, "auto"):
        result = solve(op, rhs, method="auto", tier1_keep_lowest=op.n_xi,
                       subsystem_batch=width, tol=1e-11, emit=None)
        assert result.method == "block_tridiagonal_truncated" and result.converged
        residual = jnp.linalg.norm(jax.vmap(op.apply, in_axes=1, out_axes=1)(result.x) - rhs, axis=0)
        np.testing.assert_allclose(result.residual_norms, residual, rtol=1e-8, atol=1e-14)
        assert np.all(np.asarray(residual) <= 1e-11 * np.linalg.norm(rhs, axis=0))
        np.testing.assert_allclose(result.x, reference, rtol=1e-8, atol=1e-9)
        np.testing.assert_array_equal(np.asarray(result.x)[np.asarray(mask) == 0], 0.)


def test_ramped_full_recovery_windowed_gradient_matches_tape_and_fd():
    op = _ramped_pas_op()
    def loss(scale, window):
        varied = replace(op, t_hat=op.t_hat * scale)
        result = solve(varied, varied.rhs(), tier1_keep_lowest=op.n_xi,
                       differentiable=True, tier1_adjoint_window=window, emit=None)
        return jnp.sum(result.x**2)
    value, gradient = jax.jit(jax.value_and_grad(lambda t: loss(t, 2)))(1.)
    taped = jax.jit(jax.value_and_grad(lambda t: loss(t, None)))(1.)
    np.testing.assert_allclose([value, gradient], taped, rtol=1e-10, atol=1e-13)
    assert np.isfinite(gradient) and abs(float(gradient)) > 1e-12
    for h in (1e-3, 3e-4):
        fd = (loss(1.+h, None) - loss(1.-h, None))/(2*h)
        np.testing.assert_allclose(gradient, fd, rtol=1e-4, atol=1e-13)


def test_gradient_through_ramped_truncated_route_matches_finite_differences() -> None:
    op0 = _ramped_pas_op()

    def loss(t_hat_scalar: jnp.ndarray) -> jnp.ndarray:
        op = replace(op0, t_hat=jnp.reshape(t_hat_scalar, (1,)))
        # Auto routes the ramp to the truncated kernel; grad flows straight
        # through the per-subsystem block-Thomas sweeps.
        result = solve(op, op.rhs(), method="auto", differentiable=True)
        return jnp.sum(result.x**2)

    t0 = float(op0.t_hat[0])
    g = float(jax.grad(loss)(jnp.asarray(t0)))
    eps = 1e-6
    fd = float((loss(jnp.asarray(t0 + eps)) - loss(jnp.asarray(t0 - eps))) / (2.0 * eps))
    assert np.isfinite(g) and np.isfinite(fd) and abs(fd) > 0.0
    np.testing.assert_allclose(g, fd, rtol=1e-6)


# ---------------------------------------------------------------------------
# Subsystem batching: concurrent per-(species, x) elimination width
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("make_op", ["uniform", "ramped", "ramped_grouped"])
def test_truncated_subsystem_batch_any_width_is_bit_identical(make_op: str) -> None:
    """Every subsystem batch width must compute identical answers.

    The width only changes how many independent (species, x) eliminations run
    concurrently (grouped by equal ``Nxi_for_x`` block count on the ramped
    path); the per-subsystem arithmetic is unchanged, so the solutions must
    agree to the bit level.  ``ramped_grouped`` repeats ``Nxi_for_x`` values
    so the equal-``n_blocks`` groups hold more than one subsystem and the
    batched group path is genuinely exercised.
    """
    if make_op == "uniform":
        op = _load_op("pas_1species_PAS_noEr_tiny_scheme1")
    else:
        op = _ramped_pas_op()
        if make_op == "ramped_grouped":
            op = replace(
                op, n_xi_for_x=jnp.asarray([4, 4, 9, 16, 16], dtype=jnp.int32)
            )
    rhs = op.rhs()
    b = int(op.n_species) * int(op.n_x)
    x1 = np.asarray(
        solve(op, rhs, method="block_tridiagonal_truncated", subsystem_batch=1).x
    )
    for width in (2, b, "auto"):
        r = solve(op, rhs, method="block_tridiagonal_truncated", subsystem_batch=width)
        assert r.converged
        diff = np.max(np.abs(np.asarray(r.x) - x1))
        assert diff <= 1e-13, f"width={width}: max abs diff {diff:.3e}"


def test_truncated_subsystem_width_respects_memory_budget() -> None:
    op = _ramped_pas_op()
    b = int(op.n_species) * int(op.n_x)
    # A tiny budget forces the serial minimum-memory width.
    assert tier1_truncated_subsystem_width(op, memory_budget_gb=1e-12) == 1
    # An ample budget admits the full subsystem batch.
    assert tier1_truncated_subsystem_width(op, memory_budget_gb=1024.0) == b
    # The footprint model grows with the width it models (lockstep).
    lean = tier1_truncated_peak_memory_bytes(op, subsystem_batch=1)
    wide = tier1_truncated_peak_memory_bytes(op, subsystem_batch=b)
    assert wide > lean
    # "auto" is backend-aware: serial on CPU (measured best — XLA:CPU runs
    # batched LAPACK calls serially per element), memory-budgeted elsewhere.
    if jax.default_backend() == "cpu":
        assert _resolve_subsystem_batch(op, "auto", 3) == 1
    assert _resolve_subsystem_batch(op, 4, 3) == 4
    assert _resolve_subsystem_batch(op, 10**6, 3) == b
    with pytest.raises(ValueError):
        _resolve_subsystem_batch(op, 0, 3)
    with pytest.raises(ValueError):
        _resolve_subsystem_batch(op, "wide", 3)


# ---------------------------------------------------------------------------
# Recycled Krylov on the Fokker-Planck fixture: convergence, parity, preconditioning
# ---------------------------------------------------------------------------


def test_tier2_converges_and_matches_dense_fp() -> None:
    op = _load_op("quick_2species_FPCollisions_noEr")
    ok, _ = tier1_available(op)
    assert not ok  # FP couples (species, x): structured direct must refuse
    rhs = op.rhs()
    tol = 1e-10
    result = solve(op, rhs, method="gmres", tol=tol)
    assert result.method == "gcrot"
    assert result.converged
    assert float(result.residual_norms[0]) < tol * float(jnp.linalg.norm(rhs))
    x_ref = _dense_solve(op, np.asarray(rhs)[:, None])[:, 0]
    assert _rel_err(np.asarray(result.x), x_ref) < 1e-8


def test_tier2_coarse_preconditioner_reduces_iterations() -> None:
    op = _load_op("quick_2species_FPCollisions_noEr")
    rhs = op.rhs()
    r_pc = solve(op, rhs, method="gmres", tol=1e-8)
    r_nopc = solve(
        op, rhs, method="gmres", tol=1e-8, use_preconditioner=False, max_restarts=60
    )
    assert r_pc.converged
    # The unpreconditioned Krylov stalls on this FP system (it hits its
    # iteration cap far from tolerance); the coarse-operator preconditioner
    # must converge in strictly fewer iterations than the cap it burned.
    assert not r_nopc.converged or r_pc.iterations < r_nopc.iterations
    assert r_pc.iterations < r_nopc.iterations


# ---------------------------------------------------------------------------
# Recycled Krylov on the improved Sugama fixture (collisionOperator=3): routing, parity,
# and differentiability — the momentum/energy-restoring field coupling is
# dropped from the coarse preconditioner but kept in the full operator.
# ---------------------------------------------------------------------------


def test_sugama_collisionop3_routes_tier2_matches_tier3_and_differentiates() -> None:
    op = _load_sugama_op()
    assert op.sugama is not None and op.fp is None  # collisionOperator=3 built
    assert op.constraint_scheme == 1  # {density, temperature} speed null space
    ok, _reason = tier1_available(op)
    assert not ok  # dense (species, x) collisions + constraintScheme=1: structured direct refuses

    rhs = op.rhs()
    tol = 1e-10
    # The auto policy must pick the differentiable recycled Krylov, NOT the
    # non-differentiable sparse direct host fallback.
    r_auto = solve(op, rhs, method="auto", tol=tol)
    assert r_auto.method == "gcrot"
    assert r_auto.converged

    # Recycled Krylov == sparse direct (host SuperLU) element-wise.
    r_direct = solve(op, rhs, method="direct")
    assert r_direct.method == "direct"
    assert _rel_err(np.asarray(r_auto.x), np.asarray(r_direct.x)) < 1e-8

    # Gradient flows through the differentiable recycled Krylov solve.  The Sugama mat is
    # exactly proportional to nu_n, so a scalar multiplier on it models a nu_n
    # scan; AD (implicit function theorem) must match a central finite
    # difference on the differentiable segment.
    base_mat = op.sugama.mat
    g = jnp.asarray(np.random.default_rng(0).standard_normal(op.total_size))

    def moment(s: float) -> jnp.ndarray:
        op_s = replace(op, sugama=replace(op.sugama, mat=s * base_mat))
        sol = solve(op_s, rhs, method="gmres", tol=1e-11, differentiable=True)
        return jnp.dot(g, sol.x)

    # auto + differentiable stays on recycled Krylov (the production path).
    r_auto_diff = solve(
        replace(op, sugama=replace(op.sugama, mat=1.0 * base_mat)),
        rhs,
        method="auto",
        tol=1e-9,
        differentiable=True,
    )
    assert r_auto_diff.method == "gcrot"

    grad_ad = float(jax.grad(moment)(1.0))
    eps = 1e-4
    grad_fd = float((moment(1.0 + eps) - moment(1.0 - eps)) / (2 * eps))
    assert np.isfinite(grad_ad)
    assert abs(grad_ad - grad_fd) / max(1.0, abs(grad_fd)) < 1e-6


# ---------------------------------------------------------------------------
# Auto-policy selection
# ---------------------------------------------------------------------------


def test_auto_policy_selects_tier1_for_pas_and_tier2_for_fp() -> None:
    op_pas = _load_op("pas_1species_PAS_noEr_tiny_scheme1")
    r_pas = solve(op_pas, op_pas.rhs(), method="auto")
    assert r_pas.method == "block_tridiagonal"

    op_fp = _load_op("quick_2species_FPCollisions_noEr")
    r_fp = solve(op_fp, op_fp.rhs(), method="auto", tol=1e-8)
    assert r_fp.method == "gcrot"
    assert r_fp.converged


def test_tier1_refuses_er_xdot_l2_coupling() -> None:
    # Er xDot couples L±2: the analytic block extraction (and hence structured direct)
    # must refuse, leaving this family to the Krylov and sparse direct routes.
    op = _load_op("er_xdot_1species_tiny")
    ok, _reason = tier1_available(op)
    assert not ok


def test_auto_policy_recovers_a_starved_tier2_solve() -> None:
    # Starve recycled Krylov (no preconditioner, tiny restart budget) on the FP
    # fixture: the auto policy must recover and still return the right answer.
    #
    # This used to assert result.method == "direct", because a cap breach fell
    # straight to the sparse direct host solve.  That fallback is a dead end at any
    # real size -- sparse direct materializes the operator column by column, so a
    # 66004-DOF deck would need 66004 matvecs -- and a user hit exactly that,
    # losing a whole radius of an Er scan to the crash.  A stalled Krylov solve
    # is a preconditioner problem, so the policy now escalates the
    # preconditioner first and only reaches sparse direct where that route can run.  What
    # matters is that a starved solve still lands on the correct answer.
    op = _load_op("quick_2species_FPCollisions_noEr")
    rhs = op.rhs()
    result = solve(
        op, rhs, method="auto", tol=1e-10, use_preconditioner=False, max_restarts=2
    )
    assert result.converged
    x_ref = _dense_solve(op, np.asarray(rhs)[:, None])[:, 0]
    assert _rel_err(np.asarray(result.x), x_ref) < 1e-8


def test_explicit_tier1_request_raises_on_fp() -> None:
    op = _load_op("quick_2species_FPCollisions_noEr")
    with pytest.raises(NotImplementedError):
        solve(op, op.rhs(), method="block_tridiagonal")


# ---------------------------------------------------------------------------
# Sparse direct (host SuperLU) parity
# ---------------------------------------------------------------------------


def test_tier3_direct_solve_matches_dense() -> None:
    op = _load_op("pas_1species_PAS_noEr_tiny_scheme1")
    rhs = op.rhs()
    result = solve(op, rhs, method="direct")
    assert result.method == "direct"
    assert result.converged
    x_ref = _dense_solve(op, np.asarray(rhs)[:, None])[:, 0]
    assert _rel_err(np.asarray(result.x), x_ref) < 1e-10


# ---------------------------------------------------------------------------
# Recycling / warm start across an Er continuation
# ---------------------------------------------------------------------------


def _op_with_er(base_text: str, er: float) -> KineticOperator:
    assert "Er = 0" in base_text
    text = base_text.replace("Er = 0", f"Er = {er:.6f}")
    return KineticOperator.from_namelist(parse_sfincs_input_text(text))


def test_recycling_cuts_iterations_on_er_continuation() -> None:
    base = _load_text("quick_2species_FPCollisions_noEr")
    op1 = _op_with_er(base, 0.005)
    op2 = _op_with_er(base, 0.010)
    tol = 1e-9

    r1 = solve(op1, op1.rhs(), method="gmres", tol=tol)
    assert r1.converged

    cold = solve(op2, op2.rhs(), method="gmres", tol=tol)
    warm = solve(op2, op2.rhs(), method="gmres", tol=tol, x0=r1.x, recycle=r1.recycle)
    assert cold.converged and warm.converged
    assert warm.iterations < cold.iterations


# ---------------------------------------------------------------------------
# Differentiability: jax.grad through the structured direct solve vs finite differences
# ---------------------------------------------------------------------------


def test_gradient_through_tier1_solve_matches_finite_differences() -> None:
    op0 = _load_op("pas_1species_PAS_noEr_tiny_scheme1")

    def loss(t_hat_scalar: jnp.ndarray) -> jnp.ndarray:
        # Thread the scalar through the operator pytree (streaming/mirror
        # coefficients and the RHS drive depend on THat); the PAS collision
        # matrices stay frozen, so finite differences see the same function.
        op = replace(op0, t_hat=jnp.reshape(t_hat_scalar, (1,)))
        result = solve(op, op.rhs(), method="block_tridiagonal", differentiable=True)
        return jnp.sum(result.x**2)

    t0 = float(op0.t_hat[0])
    g = float(jax.grad(loss)(jnp.asarray(t0)))
    eps = 1e-6
    fd = float((loss(jnp.asarray(t0 + eps)) - loss(jnp.asarray(t0 - eps))) / (2.0 * eps))
    assert np.isfinite(g) and np.isfinite(fd) and abs(fd) > 0.0
    np.testing.assert_allclose(g, fd, rtol=1e-6)


# ---------------------------------------------------------------------------
# FP + constraintScheme=1 with Nxi_for_x truncation: the rectangular state
# layout embeds the packed Fortran system with exact zero rows on the
# truncated (x, L) DOFs, so the raw embedding is structurally singular and
# the naive adjoint (transposed) solve is inconsistent -> silently wrong
# gradients.  solve() must pin those DOFs (identity rows/columns) and the
# implicit-function-theorem gradient must then match finite differences.
# ---------------------------------------------------------------------------

FP_CS1_TRUNCATED_TEXT = """
&general
/
&geometryParameters
  geometryScheme = 1
/
&speciesParameters
  Zs = 1 6
  mHats = 1 6
  nHats = 0.6d+0 0.009d+0
  THats = 0.5d+0 0.8d+0
  dNHatdrHats = -0.587199 -0.00195733
  dTHatdrHats = -0.587199 -0.391466
/
&physicsParameters
  Delta = 4.5694d-3
  alpha = 1.0d+0
  nu_n = 8.4774d-3
  Er = 0
  collisionOperator = 0
  includePhi1 = .false.
/
&resolutionParameters
  Ntheta = 7
  Nzeta = 5
  Nxi = 6
  Nx = 4
/
"""


def _fp_cs1_truncated_op() -> KineticOperator:
    op = KineticOperator.from_namelist(parse_sfincs_input_text(FP_CS1_TRUNCATED_TEXT))
    # The whole point of this fixture: the default Nxi_for_x truncates L at
    # low x (constraintScheme=1 Fokker-Planck), so the rectangular embedding
    # is structurally singular.
    assert op.constraint_scheme == 1 and op.fp is not None
    assert int(np.min(np.asarray(op.n_xi_for_x))) < op.n_xi
    return op


def test_fp_cs1_truncated_embedding_is_singular_and_pinning_fixes_it() -> None:
    op = _fp_cs1_truncated_op()
    mask = np.asarray(op.active_dof_mask())
    assert mask is not None and mask.min() == 0.0

    raw = materialize_dense(op)
    # Truncated DOFs are exact zero rows of the raw embedding (Fortran v3
    # never carries them: packed indexing in indices.F90).
    assert np.max(np.abs(raw[mask == 0.0, :])) == 0.0

    pinned = materialize_dense(op, pin_masked_dofs=True)
    # Pinned rows/columns are exactly the identity...
    n = op.total_size
    eye = np.eye(n)
    assert np.array_equal(pinned[mask == 0.0, :], eye[mask == 0.0, :])
    assert np.array_equal(pinned[:, mask == 0.0], eye[:, mask == 0.0])
    # ...and the active block is untouched and nonsingular.
    act = mask == 1.0
    assert np.array_equal(pinned[np.ix_(act, act)], raw[np.ix_(act, act)])
    s = np.linalg.svd(pinned, compute_uv=False)
    assert s[-1] > 1e-8

    # The recycled Krylov solve on the physical RHS matches the pinned dense solve.
    rhs = op.rhs()
    assert float(np.max(np.abs(np.asarray(rhs)[mask == 0.0]))) == 0.0
    result = solve(op, rhs, method="gmres", tol=1e-10)
    assert result.converged
    x_ref = sla.solve(pinned, np.asarray(rhs))
    assert _rel_err(np.asarray(result.x), x_ref) < 1e-8


def test_fp_cs1_gradients_match_fd() -> None:
    """jax.grad through the differentiable recycled Krylov solve vs central FD.

    Before the truncated-DOF pinning this returned catastrophically wrong
    gradients (the adjoint system was inconsistent) while the forward solve
    converged fine — the historical silent-wrong-gradient failure.
    """
    op0 = _fp_cs1_truncated_op()

    def loss(scale: jnp.ndarray, differentiable: bool = True) -> jnp.ndarray:
        # Thread the scalar through the operator pytree (streaming/mirror and
        # the RHS drive depend on THat); the Fokker-Planck matrices stay
        # frozen, so finite differences see the same function.
        op = replace(op0, t_hat=op0.t_hat * scale)
        result = solve(
            op, op.rhs(), method="gmres", tol=1e-10, differentiable=differentiable
        )
        return jnp.sum(result.x**2)

    g = float(jax.grad(loss)(jnp.asarray(1.0)))
    eps = 1e-4
    fd = float(
        (loss(jnp.asarray(1.0 + eps), differentiable=False)
         - loss(jnp.asarray(1.0 - eps), differentiable=False)) / (2.0 * eps)
    )
    assert np.isfinite(g) and np.isfinite(fd) and abs(fd) > 0.0
    np.testing.assert_allclose(g, fd, rtol=1e-4)


def test_differentiable_solve_aborts_loudly_on_genuinely_singular_operator() -> None:
    """check_adjoint (default on) must abort instead of silently corrupting grads.

    Dropping the constraint scheme leaves the Fokker-Planck f-block with its
    physical (Maxwellian) null space, which pinning cannot fix; the stalled
    forward/adjoint GCROT solve must raise.
    """
    op0 = _fp_cs1_truncated_op()
    op_singular = replace(op0, constraint_scheme=0)
    # A generic linear functional of the solution: its cotangent is not in
    # range(A^T) of the singular operator, so the adjoint solve must stall.
    w = jnp.asarray(
        np.random.default_rng(7).standard_normal(op_singular.total_size)
    )

    def loss(scale: jnp.ndarray) -> jnp.ndarray:
        op = replace(op_singular, t_hat=op_singular.t_hat * scale)
        result = solve(
            op, op.rhs(), method="gmres", tol=1e-10, differentiable=True,
            max_restarts=10,
        )
        return jnp.dot(w, result.x)

    with pytest.raises(Exception, match="GCROT solve failed to converge"):
        jax.grad(loss)(jnp.asarray(1.0))


# ---------------------------------------------------------------------------
# The adjoint-residual guard.  The implicit-function-theorem VJP runs one
# *transposed* solve, and that solve is the only place in dkx where a wrong
# answer leaves no trace: the forward solution and every field of SolveResult
# stay right while jax.grad hands back garbage.  The contract is
#
#   (a) the true residual ||A^T y - g|| is recomputed from the operator (not
#       read off the Krylov method) and recorded on SolveResult.adjoint;
#   (b) a solve that misses both the requested tolerance and the float64
#       backward-error floor raises, by default, with an actionable message;
#   (c) no false positives: healthy decks must return their exact gradient
#       without raising, including near-singular ones whose adjoint solution
#       norm makes ``tol * ||g||`` unreachable in double precision.
# ---------------------------------------------------------------------------

# Two species with full Fokker-Planck collisions, constraintScheme=1, a finite
# Er with both the xDot and xiDot terms, and *uniform* Nxi_for_x — the shape of
# the flagship-optimization deck, at referee size.  Nothing here is truncated,
# so the active-DOF pinning is a no-op and the operator's conditioning is the
# only thing under test.
FP_CS1_ER_UNIFORM_TEXT = """
&general
  RHSMode = 1
/
&geometryParameters
  geometryScheme = 1
  helicity_n = 2
  psiAHat = 8.5714
  aHat = 1.70442623
/
&speciesParameters
  Zs = 1.0d+0 -1.0d+0
  mHats = 1.0d+0 5.446170214d-4
  nHats = 1.7 1.7
  THats = 5.0 5.0
  dNHatdrHats = -0.09 -0.09
  dTHatdrHats = -5.0 -5.0
/
&physicsParameters
  Delta = 4.5694d-3
  alpha = 1.0d+0
  nu_n = 0.00831565
  Er = 1.0
  collisionOperator = 0
  includeXDotTerm = .true.
  includeElectricFieldTermInXiDot = .true.
/
&resolutionParameters
  Ntheta = 5
  Nzeta = 5
  Nxi = 8
  NL = 4
  Nx = 4
/
&otherNumericalParameters
  xGridScheme = 5
  Nxi_for_x_option = 0
/
"""


def _tier2_grad_vs_fd(
    op0: KineticOperator, *, tol: float = 1e-10, seed: int = 11, **solve_kw
) -> tuple[float, float, SolveResult]:
    """``jax.grad`` through the differentiable recycled Krylov solve vs central FD.

    The scalar is threaded through ``THat`` (streaming/mirror and the RHS drive
    depend on it; the collision matrices stay frozen, so finite differences see
    the same function).  The cotangent is a fixed pseudo-random vector — a
    generic linear functional, the hardest case for the adjoint solve and the
    one a composed objective actually produces.
    """
    mask = op0.active_dof_mask()
    w = jnp.asarray(np.random.default_rng(seed).standard_normal(op0.total_size))
    if mask is not None:
        w = w * mask  # truncated DOFs carry no physics; keep the cotangent on the rest
    captured: dict[str, SolveResult] = {}

    def loss(scale: jnp.ndarray, differentiable: bool = True) -> jnp.ndarray:
        op = replace(op0, t_hat=op0.t_hat * scale)
        result = solve(
            op, op.rhs(), method="gmres", tol=tol,
            differentiable=differentiable, **solve_kw,
        )
        if differentiable:
            captured["result"] = result
        return jnp.dot(w, result.x)

    g = float(jax.grad(loss)(jnp.asarray(1.0)))
    eps = 1e-5
    fd = float(
        (loss(jnp.asarray(1.0 + eps), differentiable=False)
         - loss(jnp.asarray(1.0 - eps), differentiable=False)) / (2.0 * eps)
    )
    return g, fd, captured["result"]


def test_adjoint_residual_is_recorded_on_the_solve_result() -> None:
    """SolveResult.adjoint must carry the *true* transposed-solve residual.

    Empty when the result is handed back (the adjoint has not run yet), filled
    once the backward pass executes.  The recorded residual is recomputed from
    the operator, so it is the number a caller can act on.
    """
    op0 = KineticOperator.from_namelist(
        parse_sfincs_input_text(FP_CS1_ER_UNIFORM_TEXT)
    )
    g, fd, result = _tier2_grad_vs_fd(op0)
    np.testing.assert_allclose(g, fd, rtol=1e-6)

    diag = result.adjoint
    assert diag is not None and diag.checked
    assert diag.tol == 1e-10
    adj = diag.adjoint_records
    assert len(adj) == 1 and len(diag.forward_records) == 1
    rec = adj[0]
    assert rec.label.startswith("adjoint")
    assert rec.within_tolerance and diag.converged
    # The residual is the one the operator itself reports, and it is genuinely
    # above the requested tol*||g|| here: this near-singular deck is exactly
    # the case the backward-error floor exists to accept.
    assert rec.residual_norm > rec.target
    assert rec.residual_norm <= rec.limit
    assert diag.worst_relative_residual == rec.relative_residual
    assert 0.0 < rec.relative_residual < 1e-6

    diag.reset()
    assert diag.records == [] and diag.worst_relative_residual == 0.0


def test_singular_tier2_adjoint_raises_instead_of_returning_a_wrong_gradient() -> None:
    """A deck whose transposed solve diverges must abort, not hand back a number.

    ``er_xidot_1species_tiny`` pairs the Er xiDot term with the per-speed
    ``constraintScheme=2`` border, which does not span the resulting null space
    (smallest singular value ~4e-19 against ``||A|| ~ 15``).  The *forward*
    solve converges to 1e-15 — the drive stays in the range of ``A`` — while
    the transposed solve against a generic cotangent diverges by 50 orders of
    magnitude.  Nothing in SolveResult would show it.
    """
    op0 = _load_op("er_xidot_1species_tiny")
    with pytest.raises(Exception, match="GCROT solve failed to converge"):
        _tier2_grad_vs_fd(op0, max_restarts=20)


def test_singular_tier2_adjoint_message_names_cause_and_remedies() -> None:
    """The abort has to be actionable, not just loud."""
    op0 = _load_op("er_xidot_1species_tiny")
    with pytest.raises(Exception) as excinfo:
        _tier2_grad_vs_fd(op0, max_restarts=20)
    msg = str(excinfo.value)
    assert "adjoint (transposed)" in msg
    assert "||A^T y - g|| / ||g||" in msg  # the true residual, named
    for remedy in ("constraint scheme", "method='direct'", "check_adjoint=False",
                   "SolveResult.adjoint", "adjoint_residual_factor"):
        assert remedy in msg, remedy


def test_check_adjoint_false_is_the_documented_unchecked_opt_out() -> None:
    """Opting out must not raise — and must still record the residual."""
    op0 = _load_op("er_xidot_1species_tiny")
    g, fd, result = _tier2_grad_vs_fd(op0, max_restarts=20, check_adjoint=False)
    diag = result.adjoint
    assert diag is not None and not diag.checked
    rec = diag.adjoint_records[0]
    assert not rec.within_tolerance and not diag.converged
    # The unchecked path is unchecked: the caller gets the wrong gradient it
    # asked for, plus the residual that proves it wrong (the adjoint achieved
    # essentially nothing over the trivial y = 0, whose residual is ||g||).
    assert rec.relative_residual > 0.1
    assert rec.residual_norm > 1e3 * rec.limit
    # The unchecked gradient is not trustworthy here; the recorded residual
    # above is the evidence.  Its numerical value is not pinned: a stagnated
    # adjoint can land anywhere, including near the finite-difference value by
    # coincidence on a given backend, so asserting a specific error would be
    # asserting an accident.


@pytest.mark.parametrize(
    "deck",
    [
        "multispecies_quick_2species_FPCollisions_noEr",  # FP, constraintScheme=1
        "fp_1species_FPCollisions_noEr_tiny_cs3",  # FP, constraintScheme=3
        "fp_1species_FPCollisions_noEr_tiny_cs4",  # FP, constraintScheme=4
    ],
)
def test_healthy_tier2_gradients_are_exact_and_do_not_raise(deck: str) -> None:
    """No false positives: a guard that fires on good decks is worse than the bug.

    Each of these routes through the same differentiable recycled Krylov adjoint the
    guard watches, and each must come back matching finite differences with
    every recorded residual inside tolerance.
    """
    op0 = _load_op(deck)
    g, fd, result = _tier2_grad_vs_fd(op0)
    assert np.isfinite(g) and abs(fd) > 0.0
    np.testing.assert_allclose(g, fd, rtol=1e-5)
    diag = result.adjoint
    assert diag is not None and diag.converged
    assert all(r.within_tolerance for r in diag.records)


def test_flagship_shaped_fp_cs1_gradient_matches_fd_without_raising() -> None:
    """The documented failing configuration, at referee size.

    Full Fokker-Planck + constraintScheme=1 + finite Er on a uniform
    ``Nxi_for_x`` grid: the adjoint stagnates two decades above ``tol*||g||``
    because the cotangent excites an almost-null direction and ``||y||`` blows
    up, yet the resulting gradient is right.  Judging that solve failed would
    abort a healthy optimization; judging it silently fine is the original bug.
    """
    op0 = KineticOperator.from_namelist(
        parse_sfincs_input_text(FP_CS1_ER_UNIFORM_TEXT)
    )
    assert op0.constraint_scheme == 1 and op0.fp is not None
    assert op0.active_dof_mask() is None  # uniform: pinning is not what saves this
    g, fd, result = _tier2_grad_vs_fd(op0)
    np.testing.assert_allclose(g, fd, rtol=1e-6)
    assert result.adjoint is not None and result.adjoint.converged


def test_adjoint_guard_survives_jit() -> None:
    """Production gradients run under ``jit``; the guard has to fire there too.

    ``jax.debug.callback`` executes inside the compiled backward pass, so both
    the recording and the abort must survive compilation — the alternative is a
    check that quietly stops existing exactly where it matters.
    """
    op0 = _load_op("multispecies_quick_2species_FPCollisions_noEr")
    captured: dict[str, SolveResult] = {}
    w = jnp.asarray(np.random.default_rng(11).standard_normal(op0.total_size))

    def loss(scale: jnp.ndarray, op_base: KineticOperator = op0) -> jnp.ndarray:
        op = replace(op_base, t_hat=op_base.t_hat * scale)
        result = solve(op, op.rhs(), method="gmres", tol=1e-10, differentiable=True)
        captured["result"] = result
        return jnp.dot(w, result.x)

    g = float(jax.jit(jax.grad(loss))(jnp.asarray(1.0)))
    assert np.isfinite(g)
    diag = captured["result"].adjoint
    assert diag is not None and diag.converged
    assert [r.label for r in diag.records] == ["forward", "adjoint (transposed)"]

    op_singular = _load_op("er_xidot_1species_tiny")
    ws = jnp.asarray(np.random.default_rng(11).standard_normal(op_singular.total_size))

    def loss_singular(scale: jnp.ndarray) -> jnp.ndarray:
        op = replace(op_singular, t_hat=op_singular.t_hat * scale)
        result = solve(
            op, op.rhs(), method="gmres", tol=1e-10, differentiable=True,
            max_restarts=20,
        )
        return jnp.dot(ws, result.x)

    with pytest.raises(Exception, match="GCROT solve failed to converge"):
        jax.jit(jax.grad(loss_singular))(jnp.asarray(1.0))


def test_adjoint_guard_leaves_the_forward_solution_untouched() -> None:
    """No physics change: the check observes, it never alters an answer."""
    op0 = _load_op("multispecies_quick_2species_FPCollisions_noEr")
    rhs = op0.rhs()
    checked = solve(op0, rhs, method="gmres", tol=1e-10, differentiable=True)
    unchecked = solve(
        op0, rhs, method="gmres", tol=1e-10, differentiable=True, check_adjoint=False
    )
    plain = solve(op0, rhs, method="gmres", tol=1e-10)
    assert np.array_equal(np.asarray(checked.x), np.asarray(unchecked.x))
    assert np.array_equal(np.asarray(checked.x), np.asarray(plain.x))
    assert plain.adjoint is None  # non-differentiable solves run no adjoint


# ---------------------------------------------------------------------------
# jit-safety: the Nxi_for_x truncation mask must not host-materialize, so the
# coarse preconditioner and the differentiable solve stay traceable and reuse
# their compilation.  Regression guard for the two former
# ``np.asarray(op._mask())`` host round-trips (solve.py / drift_kinetic.py) that
# blocked jit-over-operator-leaves, vmap batching, and cross-eval jit reuse.
# ---------------------------------------------------------------------------


def test_coarse_preconditioner_is_jit_safe_over_traced_operator_leaves() -> None:
    """``build_coarse_preconditioner`` must build when the operator *leaves* are
    tracers (regression for ``mask = np.asarray(op._mask())`` at its top).

    A ramped ``Nxi_for_x`` deck makes the truncation mask non-uniform; the
    preconditioner action under jit-over-leaves must match the eager build.
    """
    op = _ramped_pas_op()
    leaves, treedef = jax.tree_util.tree_flatten(op)
    v = jnp.asarray(np.linspace(-1.0, 1.0, op.total_size), dtype=jnp.float64)

    def precond_action(ls: list) -> jnp.ndarray:
        precond, _ = build_coarse_preconditioner(jax.tree_util.tree_unflatten(treedef, ls))
        return precond(v)

    jitted = jax.jit(precond_action)(leaves)  # compiles (was a Tracer error)
    ref, _ = build_coarse_preconditioner(op)
    # The preconditioner is a dense factorization; XLA is free to fuse it
    # differently inside jit than out, and the two orderings differ in the last
    # few digits on some backends.  What must hold is that jit does not change
    # what the preconditioner *is* -- a Krylov method corrects any residual
    # error in its application -- so this compares the applied vectors in norm,
    # the quantity a Krylov method actually consumes.  An element-wise relative
    # check would instead be dominated by near-zero components, where a 1e-8
    # absolute wobble reads as a large relative one.
    reference = np.asarray(ref(v))
    difference = np.linalg.norm(np.asarray(jitted) - reference)
    assert difference <= 1e-6 * max(1.0, float(np.linalg.norm(reference))), difference


# ---------------------------------------------------------------------------
# The adaptive l=0 null-space pin.  The simplified l=0 diagonal block is
# annihilated by streaming, mirror, ExB and pitch-angle scattering on a
# distribution constant over the flux surface, so *something* must regularize
# it or the coarsest block-Thomas divides by zero.  Doing that unconditionally,
# at the mean |diagonal| over all L, made the pin dominate the very block it
# regularized: 87 GCROT iterations against 21 on the NCSX 11x21x41x5 recycled Krylov
# ladder.  The contract now is that the pin is sized by the invertibility floor
# and fires only where the block really is singular.
# ---------------------------------------------------------------------------


def test_l0_pin_is_floor_sized_and_off_on_a_regular_block() -> None:
    """The pin tops the null direction up to the floor, and stops there.

    ``(gamma * ones (x) c0) @ ones = gamma * sum(c0)`` is the row sum the pin
    adds to the ``l = 0`` block, so that product — not ``gamma`` itself, whose
    sign follows ``sum(c0)`` and hence the Jacobian sign convention — is the
    quantity with a contract.
    """
    band = jnp.asarray([[2.0]])
    c0 = jnp.asarray([0.25, 0.75])  # sum(c0) = 1.0
    scale = jnp.asarray([[7.0]])  # the legacy sizing; must not matter any more
    floor = _COARSE_DIAGONAL_FLOOR * 2.0  # floor * band

    # Exactly singular in the pinned direction -> pinned up to the floor.
    singular = _l0_pin_gamma(jnp.zeros((1, 1)), band, scale, c0)
    assert float(singular[0, 0] * jnp.sum(c0)) == pytest.approx(floor, rel=1e-12)
    # ... and the amount does NOT key on the mean |diagonal| over all L, which
    # the ``nu l(l+1)/2`` collision diagonal makes ~1e3x larger at high l.
    louder = _l0_pin_gamma(jnp.zeros((1, 1)), band, scale * 1e3, c0)
    assert float(louder[0, 0]) == float(singular[0, 0])

    # Half the floor missing -> exactly half the floor supplied (adaptive, not
    # a switch: the pin supplies the shortfall).
    half = _l0_pin_gamma(jnp.asarray([[0.5 * floor]]), band, scale, c0)
    assert float(half[0, 0] * jnp.sum(c0)) == pytest.approx(0.5 * floor, rel=1e-12)

    # A block whose own l=0 diagonal already clears the floor gets no pin.
    regular = _l0_pin_gamma(jnp.asarray([[1e-3]]), band, scale, c0)
    assert float(regular[0, 0]) == 0.0


def test_coarse_preconditioner_stays_finite_on_an_exactly_singular_l0_block() -> None:
    """A collisionless, drift-free f-block must still factor.

    ``nu_n=0`` with ``Er=0`` leaves the coarse f-block's diagonal EXACTLY zero —
    only streaming and the mirror force couple ``L`` — which is the case the pin
    exists for.  (Collisionless pitch-angle-scattering decks route to the
    structured direct solver, but the ``Phi1`` Newton inner solve forces the coarse
    preconditioner for every deck.)  Both the preconditioner and its transpose
    must come back finite rather than dividing by zero.
    """
    txt = _load_text("pas_1species_PAS_noEr_tiny").replace("nu_n = 8.4774d-3", "nu_n = 0d+0")
    op = KineticOperator.from_namelist(parse_sfincs_input_text(txt))
    precond, precond_t = build_coarse_preconditioner(op)
    v = jnp.asarray(np.linspace(-1.0, 1.0, op.total_size), dtype=jnp.float64)
    for apply in (precond, precond_t):
        assert np.all(np.isfinite(np.asarray(apply(v))))
    # The deck still solves through recycled Krylov on that preconditioner.
    result = solve(op, op.rhs(), method="gmres", tol=1e-10)
    assert result.converged


def test_jit_value_and_grad_through_differentiable_solve_compiles_once() -> None:
    """The differentiable solve must be jittable end-to-end and *reuse* its
    compilation.

    ``jax.jit(value_and_grad(...))`` around the differentiable structured direct solve must
    compile exactly once and NOT retrace when only the operator leaf VALUES
    change — the optimization inner loop that used to recompile per eval.  The
    jitted gradient still matches central finite differences.
    """
    op0 = _load_op("pas_1species_PAS_noEr_tiny_scheme1")
    traces = {"n": 0}

    def loss(t_vec: jnp.ndarray) -> jnp.ndarray:
        traces["n"] += 1  # increments once per trace (i.e. per compile)
        op = replace(op0, t_hat=t_vec)
        res = solve(op, op.rhs(), method="block_tridiagonal", differentiable=True)
        return jnp.sum(res.x**2)

    vg = jax.jit(jax.value_and_grad(loss))
    p0 = jnp.reshape(op0.t_hat[0], (1,))
    val0, g0 = vg(p0)
    n_after_first = traces["n"]
    val1, _ = vg(p0 * 1.3)  # different leaf VALUE, identical shape/dtype
    val2, _ = vg(p0 * 0.6)
    # compiled exactly once; new values reuse the executable (no retrace)
    assert n_after_first == 1
    assert traces["n"] == 1
    assert np.isfinite(float(g0[0]))
    assert float(val0) != float(val1) != float(val2)

    # the jitted AD gradient matches central finite differences
    def loss_plain(t: float) -> float:
        op = replace(op0, t_hat=jnp.reshape(jnp.asarray(t), (1,)))
        return float(jnp.sum(solve(op, op.rhs(), method="block_tridiagonal").x**2))

    t0 = float(op0.t_hat[0])
    eps = 1e-6
    fd = (loss_plain(t0 + eps) - loss_plain(t0 - eps)) / (2.0 * eps)
    np.testing.assert_allclose(float(g0[0]), fd, rtol=1e-5)


# ---------------------------------------------------------------------------
# Optional-dependency policy: solvax is optional until its PyPI release.
# ---------------------------------------------------------------------------


def test_solve_importable_without_solvax_and_fails_loudly_on_use() -> None:
    """``import dkx.solve`` must work without solvax; use must raise clearly.

    Runs in a subprocess (this session already imported solvax) and hides the
    package by poisoning ``sys.modules`` before the import.
    """
    import subprocess
    import sys

    code = "\n".join(
        [
            "import sys",
            "for m in ('solvax', 'solvax.direct', 'solvax.implicit', 'solvax.krylov',",
            "          'solvax.native', 'solvax.operators'):",
            "    sys.modules[m] = None  # poisoned: import raises ImportError",
            "import dkx.solve as solve_mod",
            # dkx.coarse_precond is the package's one solvax import guard;
            # dkx.solve re-exports _require_solvax from it rather than keeping
            # a second copy of the message.
            "import dkx.coarse_precond as precond_mod",
            "assert precond_mod._SOLVAX_IMPORT_ERROR is not None",
            "try:",
            "    solve_mod._require_solvax()",
            "except ImportError as exc:",
            "    assert 'solvax' in str(exc)",
            "else:",
            "    raise SystemExit('expected ImportError on solvax use')",
            "print('guarded-import-ok')",
        ]
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=300
    )
    assert proc.returncode == 0, proc.stderr
    assert "guarded-import-ok" in proc.stdout


# ---------------------------------------------------------------------------
# Size-aware device routing (solve(device=...))
# ---------------------------------------------------------------------------


def test_resolve_solve_device_semantics_on_cpu_host() -> None:
    """The routing knob on a CPU-default host: inert except explicit errors."""
    from dkx.solve import _resolve_solve_device

    op = _load_op("pas_1species_PAS_noEr_tiny_scheme1")
    assert jax.default_backend() == "cpu", "test assumes a CPU-default test host"
    # auto/default/cpu: no movement on a CPU host.
    assert _resolve_solve_device("auto", "block_tridiagonal", op, False) is None
    assert _resolve_solve_device("default", "gmres", op, False) is None
    assert _resolve_solve_device("cpu", "gmres", op, False) is None
    assert _resolve_solve_device(None, "gmres", op, False) is None
    # traced: always inert (jit-safety), whatever the request.
    cpu0 = jax.local_devices(backend="cpu")[0]
    assert _resolve_solve_device(cpu0, "gmres", op, True) is None
    # explicit device object passes through untraced.
    assert _resolve_solve_device(cpu0, "gmres", op, False) is cpu0
    # sparse direct is a host solve already: no movement.
    assert _resolve_solve_device(cpu0, "direct", op, False) is None
    # accelerator request on a CPU-only host is a loud error.
    with pytest.raises(ValueError, match="default JAX backend is CPU"):
        _resolve_solve_device("gpu", "gmres", op, False)
    with pytest.raises(ValueError, match="unknown solve device"):
        _resolve_solve_device("quantum", "gmres", op, False)


def test_solve_device_thresholds_read_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from dkx.solve import (
        _SOLVE_CPU_MAX_TIER1_DEFAULT,
        _SOLVE_CPU_MAX_TIER2_DEFAULT,
        _env_size,
    )

    # Auto-routing is OFF by default: the same-host measurements (see
    # docs/performance.rst) found the GPU faster at every practical size, so
    # a nonzero default is unsupported by data.  Users opt in via the envs.
    assert _SOLVE_CPU_MAX_TIER1_DEFAULT == 0
    assert _SOLVE_CPU_MAX_TIER2_DEFAULT == 0

    monkeypatch.setenv("DKX_SOLVE_CPU_MAX_SIZE_TIER1", "12345")
    assert _env_size("DKX_SOLVE_CPU_MAX_SIZE_TIER1", _SOLVE_CPU_MAX_TIER1_DEFAULT) == 12345
    monkeypatch.setenv("DKX_SOLVE_CPU_MAX_SIZE_TIER1", "not-a-number")
    assert (
        _env_size("DKX_SOLVE_CPU_MAX_SIZE_TIER1", _SOLVE_CPU_MAX_TIER1_DEFAULT)
        == _SOLVE_CPU_MAX_TIER1_DEFAULT
    )
    monkeypatch.delenv("DKX_SOLVE_CPU_MAX_SIZE_TIER1")
    assert (
        _env_size("DKX_SOLVE_CPU_MAX_SIZE_TIER2", _SOLVE_CPU_MAX_TIER2_DEFAULT)
        == _SOLVE_CPU_MAX_TIER2_DEFAULT
    )


def test_solve_on_explicit_second_cpu_device_matches_and_returns_home() -> None:
    """Real move/move-back exercise using two host CPU devices in a subprocess.

    ``--xla_force_host_platform_device_count=2`` must be set before jax import,
    so this runs in a fresh interpreter: solve on cpu:1 with inputs on cpu:0,
    check the answer matches the unrouted solve and comes back on cpu:0.
    """
    import subprocess
    import sys

    deck = REF / "pas_1species_PAS_noEr_tiny_scheme1.input.namelist"
    code = "\n".join(
        [
            "import os",
            "os.environ['XLA_FLAGS'] = (os.environ.get('XLA_FLAGS', '') +",
            "    ' --xla_force_host_platform_device_count=2').strip()",
            "import jax, numpy as np",
            "from dkx.drift_kinetic import KineticOperator",
            "from dkx.namelist import read_sfincs_input",
            "from dkx.solve import solve",
            f"op = KineticOperator.from_namelist(read_sfincs_input({str(deck)!r}))",
            "rhs = op.rhs()",
            "cpu0, cpu1 = jax.local_devices(backend='cpu')[:2]",
            "rhs0 = jax.device_put(rhs, cpu0)",
            "ref = solve(op, rhs0, device='default')",
            "routed = solve(op, rhs0, device=cpu1)",
            "assert routed.method == ref.method",
            "err = float(np.max(np.abs(np.asarray(routed.x) - np.asarray(ref.x))))",
            "assert err < 1e-12, f'routed answer differs: {err}'",
            "assert routed.x.devices() == {cpu0}, routed.x.devices()",
            "print('device-routing-ok')",
        ]
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=600
    )
    assert proc.returncode == 0, proc.stderr
    assert "device-routing-ok" in proc.stdout


def test_truncated_bounded_adjoint_matches_taped_gradient() -> None:
    # tier1_adjoint_window selects solvax's structure-preserving custom VJP for
    # the truncated kernel. The primal is unchanged, a full window reproduces
    # the taped gradient exactly (solvax pins full-window bitwise equality on
    # the generated path), and a small window agrees to the O(rho^{2w}) decay
    # of the block-dominant collisional operator.
    import inspect as _inspect

    from solvax.direct import block_thomas_truncated_fn as _fn

    if "params" not in _inspect.signature(_fn).parameters:
        pytest.skip("installed solvax predates params/adjoint_window (needs >= 0.8.7)")

    op = _load_op("pas_1species_PAS_noEr_tiny_scheme1")
    rhs = op.rhs()
    n_xi = int(op.n_xi)

    kwargs = dict(method="block_tridiagonal_truncated", tol=1e-10)
    r_taped = solve(op, rhs, **kwargs)
    r_bounded = solve(op, rhs, tier1_adjoint_window=n_xi, **kwargs)
    assert r_bounded.method == r_taped.method
    np.testing.assert_allclose(
        np.asarray(r_bounded.x), np.asarray(r_taped.x), rtol=0.0, atol=0.0
    )

    def loss(scale: jnp.ndarray, window: int | None) -> jnp.ndarray:
        scaled = replace(op, t_hat=op.t_hat * scale)
        return jnp.sum(solve(scaled, rhs, tier1_adjoint_window=window, **kwargs).x ** 2)

    one = jnp.asarray(1.0)
    g_taped = jax.grad(lambda s: loss(s, None))(one)
    g_full = jax.grad(lambda s: loss(s, n_xi))(one)
    np.testing.assert_allclose(float(g_full), float(g_taped), rtol=1e-12)
    g_small = jax.grad(lambda s: loss(s, 4))(one)
    np.testing.assert_allclose(float(g_small), float(g_taped), rtol=1e-6)


def test_truncated_bounded_adjoint_is_subsystem_batch_invariant() -> None:
    # The bounded adjoint (tier1_adjoint_window) and subsystem batching
    # (subsystem_batch) are orthogonal knobs that meet inside solve_one: on
    # accelerators subsystem_batch > 1 vmaps solve_one, and therefore vmaps the
    # generated-block custom VJP of block_thomas_truncated_fn. This pins that
    # the custom VJP is vmap-safe in both directions — the primal and the
    # gradient must be identical at width 1 (serial) and width B (fully
    # batched), on a genuinely multi-subsystem op (B = n_species * n_x = 3).
    import inspect as _inspect

    from solvax.direct import block_thomas_truncated_fn as _fn

    if "params" not in _inspect.signature(_fn).parameters:
        pytest.skip("installed solvax predates params/adjoint_window (needs >= 0.8.7)")

    op = _load_op("pas_1species_PAS_noEr_tiny_scheme1")
    b = int(op.n_species * op.n_x)
    assert b > 1  # guards the test's premise (would be vacuous at B=1)
    rhs = op.rhs()
    n_xi = int(op.n_xi)
    kwargs = dict(method="block_tridiagonal_truncated", tol=1e-10)

    x_serial = solve(op, rhs, subsystem_batch=1, tier1_adjoint_window=n_xi, **kwargs).x
    x_batched = solve(op, rhs, subsystem_batch=b, tier1_adjoint_window=n_xi, **kwargs).x
    np.testing.assert_allclose(np.asarray(x_batched), np.asarray(x_serial), rtol=0, atol=1e-18)

    def loss(scale: jnp.ndarray, width: int) -> jnp.ndarray:
        scaled = replace(op, t_hat=op.t_hat * scale)
        return jnp.sum(
            solve(scaled, rhs, subsystem_batch=width, tier1_adjoint_window=n_xi, **kwargs).x
            ** 2
        )

    one = jnp.asarray(1.0)
    g_serial = jax.grad(lambda s: loss(s, 1))(one)
    g_batched = jax.grad(lambda s: loss(s, b))(one)
    np.testing.assert_allclose(float(g_batched), float(g_serial), rtol=1e-12)



def test_tier1_adjoint_window_is_reverse_mode_only() -> None:
    """The bounded path is a custom_vjp, so forward mode raises through it.

    Not a defect to fix --- JAX cannot push a JVP through a ``custom_vjp`` ---
    but a restriction a caller has to know, because
    :func:`dkx.sensitivity.jvp_flux` is forward mode. A study mixing the two
    must leave ``tier1_adjoint_window`` unset on its forward-mode calls.
    Pinned here so the docstring cannot drift from the behaviour, and so we
    notice if JAX ever grows a fallback.
    """
    import inspect as _inspect

    from solvax.direct import block_thomas_truncated_fn as _fn

    if "params" not in _inspect.signature(_fn).parameters:
        pytest.skip("installed solvax predates params/adjoint_window")

    op = _load_op("pas_1species_PAS_noEr_tiny_scheme1")
    rhs = op.rhs()
    n_xi = int(op.n_xi)
    kwargs = dict(method="block_tridiagonal_truncated", tol=1e-10)

    def loss(scale: jnp.ndarray, window: int | None) -> jnp.ndarray:
        scaled = replace(op, t_hat=op.t_hat * scale)
        return jnp.sum(solve(scaled, rhs, tier1_adjoint_window=window, **kwargs).x ** 2)

    one = jnp.asarray(1.0)
    reverse = jax.grad(lambda s: loss(s, None))(one)
    forward = jax.jacfwd(lambda s: loss(s, None))(one)
    np.testing.assert_allclose(float(forward), float(reverse), rtol=1e-8)

    with pytest.raises(TypeError, match="forward-mode"):
        jax.jacfwd(lambda s: loss(s, n_xi))(one)


# ---------------------------------------------------------------------------
# Convergence is judged on a shared column scale
# ---------------------------------------------------------------------------
def test_a_small_norm_column_is_not_held_to_an_unachievable_target() -> None:
    """Columns share a factorization, so they share an accuracy scale.

    The monoenergetic decks carry two right-hand sides differing by 3000x in
    norm.  Judged per-column with atol=0 the small one got a target of 4.2e-14 --
    below what double precision delivers here -- came in at 7.7e-14, missed by
    1.8x, and vetoed an exact direct solve.  The auto policy then paid a full
    Krylov re-solve: 41 s where the direct answer was already in hand.
    """
    import numpy as np

    from dkx.solve import _converged_flag

    rhs = np.zeros((10, 2))
    rhs[:, 0] = 4.23e-04 / np.sqrt(10)   # the small column
    rhs[:, 1] = 1.277 / np.sqrt(10)
    assert _converged_flag(np.array([7.66e-14, 2.04e-13]), rhs, 1e-10, 0.0)


def test_a_genuinely_bad_solve_is_still_rejected() -> None:
    """The relaxation must not swallow a real failure.

    A wrong solve misses by orders of magnitude, not by 1.8x, so sharing the
    column scale costs no diagnostic power.
    """
    import numpy as np

    from dkx.solve import _converged_flag

    rhs = np.zeros((10, 2))
    rhs[:, 0] = 4.23e-04 / np.sqrt(10)
    rhs[:, 1] = 1.277 / np.sqrt(10)
    assert not _converged_flag(np.array([1e-3, 2e-13]), rhs, 1e-10, 0.0)
    assert not _converged_flag(np.array([np.nan, 2e-13]), rhs, 1e-10, 0.0)


def test_a_single_column_is_unaffected() -> None:
    """One column means max(||b||) is its own norm: the old behaviour exactly."""
    import numpy as np

    from dkx.solve import _converged_flag

    rhs = np.ones((10, 1))
    norm = float(np.linalg.norm(rhs))
    assert _converged_flag(np.array([0.9e-10 * norm]), rhs, 1e-10, 0.0)
    assert not _converged_flag(np.array([1.1e-10 * norm]), rhs, 1e-10, 0.0)


# ---------------------------------------------------------------------------
# Reusing a built preconditioner across a sweep (plan.md Phase G)
# ---------------------------------------------------------------------------


def test_a_reused_preconditioner_gives_the_same_answer() -> None:
    """The preconditioner must not change the converged solution, only the path.

    This is the property that makes reuse safe at all, and it is the same one
    SFINCS relies on to precondition with a deliberately different, cheaper
    matrix. If reuse moved the answer, the whole scheme would be unsound rather
    than merely slower.
    """
    from dkx.solve import build_tier2_preconditioner

    op = _load_sugama_op()
    rhs = op.rhs()
    built = solve(op, rhs, method="gmres", tol=1e-10)
    reused = solve(op, rhs, method="gmres", tol=1e-10,
                   precond=build_tier2_preconditioner(op, "coarse"))  # fmt: skip
    assert built.converged and reused.converged
    assert _rel_err(np.asarray(reused.x), np.asarray(built.x)) < 1e-9


def test_a_preconditioner_built_elsewhere_still_converges_on_a_neighbour() -> None:
    """A sweep reuses one build across nearby operators; convergence must hold.

    The collisionality is scaled by 15%, which is wider than the step an Er or
    nu scan takes between points. A preconditioner is an approximation to begin
    with, so the question is never whether it is exact for the neighbour but
    whether the iteration count stays sane -- and that is what is asserted,
    rather than a fixed count that would pin machine-specific noise.
    """
    from dataclasses import replace

    from dkx.solve import build_tier2_preconditioner

    op = _load_sugama_op()
    rhs = op.rhs()
    neighbour = replace(op, sugama=replace(op.sugama, mat=1.15 * op.sugama.mat))

    own = solve(neighbour, rhs, method="gmres", tol=1e-10)
    borrowed = solve(neighbour, rhs, method="gmres", tol=1e-10,
                     precond=build_tier2_preconditioner(op, "coarse"))  # fmt: skip

    assert borrowed.converged
    assert _rel_err(np.asarray(borrowed.x), np.asarray(own.x)) < 1e-9
    assert borrowed.iterations < 3 * own.iterations, (
        f"borrowed preconditioner took {borrowed.iterations} against "
        f"{own.iterations} for its own: reuse has stopped paying"
    )


def test_omitting_precond_still_builds_one() -> None:
    """The parameter is opt-in; the default path is unchanged."""
    op = _load_sugama_op()
    result = solve(op, op.rhs(), method="gmres", tol=1e-10)
    assert result.converged and result.method == "gcrot"


def test_recycle_dim_zero_is_refused_by_name() -> None:
    """solvax's GCROT cannot recycle zero directions; say so rather than crash.

    Left unguarded this dies inside jax indexing with "index is out of bounds
    for axis 1 with size 0" -- a traceback naming neither the parameter the
    caller set nor DKX. Plain restarted FGMRES is not reachable through this
    route, and the message says which alternatives are.
    """
    op = _load_sugama_op()
    with pytest.raises(ValueError, match="recycle_dim=0"):
        solve(op, op.rhs(), method="gmres", tol=1e-8, recycle_dim=0)


def test_one_recycled_direction_is_enough_to_run() -> None:
    """The guard's boundary: 1 must work, or the message would be wrong."""
    op = _load_sugama_op()
    assert solve(op, op.rhs(), method="gmres", tol=1e-8, recycle_dim=1).converged


def test_the_coarse_factor_dtype_knob_actually_changes_the_factors(monkeypatch) -> None:
    """DKX_COARSE_FACTOR_DTYPE=float32 was silently inert on the dense-band route.

    Its docstring promises the knob is honoured, and the GPU profile measured a
    bit-identical residual with it set -- proof that it did nothing. Operating
    rule 11 forbids a silent no-op, so this pins that the factors really change
    precision while the answer does not.
    """
    op = _load_sugama_op()
    rhs = op.rhs()
    base = solve(op, rhs, method="gmres", tol=1e-10)

    monkeypatch.setenv("DKX_COARSE_FACTOR_DTYPE", "float32")
    lowered = solve(op, rhs, method="gmres", tol=1e-10)

    assert lowered.converged
    # The factors differ, so the iteration path differs -- that is the proof it
    # took effect. A float32 Schur LU preconditions fine; it just is not free.
    assert (lowered.iterations, float(np.min(lowered.residual_norms))) != (
        base.iterations, float(np.min(base.residual_norms))
    ), "float32 factors produced an identical path: the knob is inert again"
    # The converged answer is a property of the operator, not the preconditioner.
    assert _rel_err(np.asarray(lowered.x), np.asarray(base.x)) < 1e-8


def test_only_the_krylov_route_returns_a_preconditioner() -> None:
    """The route decides, so the caller must not guess ahead of it.

    An earlier attempt at scan reuse built a preconditioner whenever the method
    was "auto", which meant a case landing on the structured-direct route paid
    for one it never used -- measurably slower. Returning what the solve
    actually built removes the guess: a direct point hands back None and the
    next point builds nothing.
    """
    op = _load_sugama_op()
    rhs = op.rhs()
    assert solve(op, rhs, method="gmres", tol=1e-9).precond is not None
    assert solve(op, rhs, method="direct").precond is None


def test_a_returned_preconditioner_can_be_fed_straight_back() -> None:
    """The round trip is the whole point: solve, then reuse without rebuilding."""
    from dataclasses import replace

    op = _load_sugama_op()
    rhs = op.rhs()
    first = solve(op, rhs, method="gmres", tol=1e-10)

    neighbour = replace(op, sugama=replace(op.sugama, mat=1.05 * op.sugama.mat))
    reused = solve(neighbour, rhs, method="gmres", tol=1e-10, precond=first.precond)
    own = solve(neighbour, rhs, method="gmres", tol=1e-10)

    assert reused.converged
    assert _rel_err(np.asarray(reused.x), np.asarray(own.x)) < 1e-9


def test_the_er_scan_threads_the_preconditioner_it_was_given() -> None:
    """ErSolveState carries it forward, and carries None when there is none."""
    from dkx.er import ErSolveState

    assert "precond" in {f.name for f in dataclasses.fields(ErSolveState)}
    assert ErSolveState(x=None, recycle=None, result=None).precond is None
