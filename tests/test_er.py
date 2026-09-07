"""Canonical ambipolar radial-electric-field slice (:mod:`dkx.er`).

Pins the ``er.py`` slice against the legacy path it replaces:

- :func:`dkx.er.radial_current` at three ``E_r`` reproduces a direct
  ``run_profile`` particle-flux computation to machine precision;
- :func:`dkx.er.find_ambipolar_er` returns the same root as the legacy
  Fortran-parity Brent solver (``problems/ambipolar.py``, captured before its
  deletion and hard-coded here);
- warm starts / GCROT recycling reduce the total Krylov iteration count on a
  Fokker-Planck (recycled Krylov) Er scan;
- ion / electron / unstable classification from the sign of ``dJr/dEr``;
- the differentiable :func:`dkx.er.ambipolar_er` gradient matches a
  central finite difference (implicit function theorem, not FD roots).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

# The legacy Fortran-parity Brent root of the two-species PAS deck below,
# captured from problems/ambipolar.brent_ambipolar_root (deleted in this slice)
# driven by er.radial_current on the identical physics.
LEGACY_BRENT_ROOT_ER = -0.4065766608975321


def _pas_deck(er: float = 0.0, *, collision_operator: int = 1,
              n_theta: int = 7, n_zeta: int = 7, n_xi: int = 8, n_x: int = 3) -> str:
    """A tiny non-axisymmetric two-species deck with an ambipolar root.

    Helical ripple (epsilon_h) makes the field non-intrinsically-ambipolar;
    ``inputRadialCoordinate=3`` selects the Er (coordinate 4) electric-field
    knob that ``ambipolarSolver.F90`` drives.
    """
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
  epsilon_h = 0.05
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
  dNHatdrHats = -0.5 -0.5
  dTHatdrHats = -1.0 -1.0
/
&physicsParameters
  Delta = 4.5694d-3
  alpha = 1.0
  nu_n = 8.4774d-3
  Er = {er}
  collisionOperator = {collision_operator}
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


# ---------------------------------------------------------------------------
# 1. radial_current == a direct run_profile particle-flux computation
# ---------------------------------------------------------------------------


def test_radial_current_matches_run_profile(tmp_path: Path) -> None:
    from dkx import er as er_mod
    from dkx.run import run_profile

    prob = er_mod.prepare(_write(tmp_path, _pas_deck()), er_bracket=(-3.0, 1.0))

    for er_val in (-2.0, -0.5, 0.7):
        run = run_profile(_write(tmp_path, _pas_deck(er_val)), solve_method="auto", emit=None)
        gamma_rp = np.asarray(run.moments["particleFlux_vm_psiHat"], dtype=np.float64)
        z_s = np.asarray(run.operator.z_s, dtype=np.float64)
        jr_rp = float(np.dot(z_s, gamma_rp))

        j_r, gamma, state = er_mod.radial_current(prob, er_val)
        gamma = np.asarray(gamma, dtype=np.float64)

        np.testing.assert_allclose(gamma, gamma_rp, rtol=0.0, atol=1e-12)
        assert abs(float(j_r) - jr_rp) < 1e-12
        assert state.result.converged


# ---------------------------------------------------------------------------
# 2. find_ambipolar_er == legacy Fortran-parity Brent root
# ---------------------------------------------------------------------------


def test_find_ambipolar_er_matches_legacy_brent(tmp_path: Path) -> None:
    from dkx import er as er_mod

    result = er_mod.find_ambipolar_er(
        _write(tmp_path, _pas_deck()),
        er_bracket=(-3.0, 1.0),
        er_initial=0.0,
        max_iter=20,
        current_tol=1e-10,
        all_roots=False,
        emit=None,
    )
    assert result.converged
    assert result.er is not None
    assert abs(result.er - LEGACY_BRENT_ROOT_ER) < 1e-6
    # J_r is driven to (near) zero at the reported root.
    assert abs(result.radial_current) < 1e-9
    assert result.per_species_flux is not None
    assert result.per_species_flux.shape == (2,)


def test_brent_expands_bracket_and_finds_analytic_root() -> None:
    """The Fortran option-2 zbrent (bracket expansion + NR update) on a cubic.

    ``J(Er) = (Er + 0.4)(Er^2 + 1)`` has the single real root ``Er = -0.4``.
    The initial bracket [0, 1] does not contain it, exercising the
    ``ambipolarSolver.F90`` sign-change expansion loop.
    """
    from dkx.er import _brent

    def eval_jr(er: float, _stage: str) -> float:
        return (er + 0.4) * (er * er + 1.0)

    root, converged, status, _msg = _brent(
        eval_jr, er_min=0.0, er_max=1.0, er_initial=0.5,
        max_iter=80, current_tol=1e-12, max_expansions=50, emit=None,
    )
    assert converged and status == "converged"
    assert abs(root - (-0.4)) < 1e-6


# ---------------------------------------------------------------------------
# 3. warm starts / recycling reduce total Krylov iterations (recycled Krylov FP)
# ---------------------------------------------------------------------------


def test_warm_start_reduces_solver_iterations(tmp_path: Path) -> None:
    from dkx import er as er_mod

    # Fokker-Planck collisions route the auto policy to the recycled
    # GCROT solver, where warm starts and recycling pay off.
    deck = _pas_deck(collision_operator=0, n_theta=5, n_zeta=5, n_xi=16, n_x=4)
    prob = er_mod.prepare(_write(tmp_path, deck), er_bracket=(-3.0, 1.0))
    er_seq = list(np.linspace(-0.6, -0.35, 5))

    def scan(warm: bool) -> int:
        total = 0
        state = None
        for er_val in er_seq:
            x0 = state.x if (warm and state is not None) else None
            recycle = state.recycle if (warm and state is not None) else None
            _j, _g, state = er_mod.radial_current(
                prob, float(er_val), x0=x0, recycle=recycle, solve_method="auto", tol=1e-9
            )
            assert state.result.converged
            total += int(state.result.iterations or 0)
        return total

    cold = scan(warm=False)
    warm = scan(warm=True)
    assert cold > 0
    assert warm < cold


# ---------------------------------------------------------------------------
# 4. ion / electron / unstable classification
# ---------------------------------------------------------------------------


def test_root_classification_ion(tmp_path: Path) -> None:
    from dkx import er as er_mod

    result = er_mod.find_ambipolar_er(
        _write(tmp_path, _pas_deck()), er_bracket=(-3.0, 1.0), all_roots=True, emit=None
    )
    # A single root at Er < 0 with dJr/dEr > 0 is the stable ion root.
    assert result.root_type == "ion"
    assert result.er is not None and result.er < 0.0
    assert len(result.roots) == 1
    assert result.roots[0].root_type == "ion"
    assert result.roots[0].slope > 0.0


def test_classify_unit_logic() -> None:
    from dkx.er import _classify

    assert _classify(-0.4, 1.0) == "ion"        # stable, Er < 0
    assert _classify(0.4, 1.0) == "electron"     # stable, Er > 0
    assert _classify(0.1, -1.0) == "unstable"    # dJr/dEr < 0 (middle branch)
    assert _classify(-0.1, -1.0) == "unstable"
    assert _classify(-0.1, 0.0) == "marginal"
    assert _classify(0.1, np.nan) == "unknown"
    assert _classify(np.inf, 1.0) == "unknown"


def test_brent_rejects_narrow_discontinuous_bracket():
    from dkx.er import _brent
    root, converged, status, _ = _brent(
        lambda e, _: -1. if e < 0.123 else 1.,
        er_min=-1., er_max=1., er_initial=0., max_iter=100,
        current_tol=1e-8, field_tol=1e-6, max_expansions=0, emit=None,
    )
    assert root is not None and not converged and status == "current_tolerance"


def test_brent_bracketing_does_not_multiply_tiny_currents():
    from dkx.er import _brent
    root, converged, status, _ = _brent(
        lambda e, _: 1e-200, er_min=-1., er_max=1., er_initial=0.,
        max_iter=100, current_tol=1e-250, max_expansions=2, emit=None,
    )
    assert root is None and not converged and status == "unbracketed"


@pytest.fixture
def host_current(monkeypatch):
    from dkx import er
    problem = er.ErProblem(None, 1., np.array([1.]), 0., -1., 1.)
    def install(function):
        def current(p, e, **kwargs):
            value = function(e)
            return value, np.array([value]), None
        monkeypatch.setattr(er, "radial_current", current)
        return lambda **kwargs: er.find_ambipolar_er(problem, emit=None, **kwargs)
    return install


def test_host_rejected_candidate_has_no_classified_roots(host_current):
    result = host_current(lambda e: -1. if e < .123 else 1.)(max_iter=100)
    assert not result.converged and result.status == "current_tolerance"
    assert result.roots == () and result.root_type == "unknown"
    assert abs(result.radial_current) == 1.


def test_host_rechecks_final_current(host_current):
    calls = 0
    def current(e):
        nonlocal calls
        calls += 1
        return e if calls <= 3 else 1.
    result = host_current(current)(all_roots=False)
    assert not result.converged and result.status == "current_tolerance"
    assert result.roots == ()


def test_host_final_acceptance_drops_warm_reuse(monkeypatch):
    from types import SimpleNamespace
    from dkx import er
    calls = []
    state = SimpleNamespace(x=np.ones(1), recycle=object(), precond=object())
    def current(problem, field, **kwargs):
        calls.append(kwargs)
        return field, np.array([field]), state
    monkeypatch.setattr(er, "radial_current", current)
    problem = er.ErProblem(None, 1., np.array([1.]), 0., -1., 1.)
    result = er.find_ambipolar_er(problem, all_roots=False, emit=None)
    assert result.converged
    assert calls[1]["x0"] is state.x and calls[2]["precond"] is state.precond
    assert all(calls[3][key] is None for key in ("x0", "recycle", "precond"))


def test_host_scan_keeps_endpoints_and_rejects_false_crossing(host_current):
    # Endpoint roots plus a jump at 0.123; the discontinuity is not a root.
    result = host_current(lambda e: (e**2 - 1) * (-1 if e < .123 else 1))(
        er_initial=-1., n_scan=9, max_iter=100,
    )
    assert result.converged
    assert [r.er for r in result.roots] == [-1., 1.]


def test_host_selected_classification_is_not_replaced_by_nearby_root(host_current):
    result = host_current(lambda e: (e + .001) * e * (e - .001))(
        er_bracket=(-.002, .002), er_initial=.001, n_scan=5,
        slope_step=.00075, current_tol=1e-16,
    )
    assert result.converged and result.er == .001
    assert result.root_type == "electron"
    assert [r.root_type for r in result.roots] == ["ion", "unstable", "electron"]


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_host_rejects_nonfinite_current(host_current, value):
    with pytest.raises(RuntimeError, match="Nonfinite ambipolar"):
        host_current(lambda e: value)()


@pytest.mark.parametrize("options", [
    {"current_tol": 0.}, {"current_tol": np.nan}, {"field_tol": -1.},
    {"field_tol": np.inf}, {"slope_step": 0.},
])
def test_host_tolerances_validated_before_preparation(options):
    from dkx.er import find_ambipolar_er
    with pytest.raises(ValueError, match="finite and positive"):
        find_ambipolar_er(None, **options)


# ---------------------------------------------------------------------------
# 5. differentiable ambipolar_er: jax.grad vs central finite difference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("collision,ramped", [(1, False), (1, True), (0, False), (0, True)])
def test_ambipolar_er_grad_matches_finite_difference(tmp_path: Path, collision, ramped) -> None:
    import jax

    jax.config.update("jax_enable_x64", True)

    from dkx import er as er_mod

    deck = _pas_deck(collision_operator=collision)
    if ramped:
        deck = deck.replace("Nxi_for_x_option = 0", "Nxi_for_x_option = 1")
    prob = er_mod.prepare(_write(tmp_path, deck), er_bracket=(-3.0, 1.0))

    # Seed the differentiable root near the true root (selects that branch).
    found = er_mod.find_ambipolar_er(
        _write(tmp_path, deck), er_bracket=(-3.0, 1.0), all_roots=False, emit=None
    )
    root = float(found.er)

    base_op = prob.operator
    base_dn = base_op.dn_hat_dpsi_hat

    def er_of_theta(theta):
        # theta scales both species' density gradient (drives the flux/root).
        op_theta = replace(base_op, dn_hat_dpsi_hat=base_dn * theta)
        return er_mod.ambipolar_er(
            op_theta, er0=root, dphi_per_er=prob.dphi_per_er, z_s=prob.z_s
        )

    # The routed differentiable solve reproduces the ambipolar root (the flat J_r
    # near the root makes the exact value tolerance-sensitive, hence the loose
    # bound; the implicit-function-theorem gradient below is the real check).
    assert abs(float(er_of_theta(1.0)) - root) < 1e-3

    grad = float(jax.grad(er_of_theta)(1.0))
    h = 1e-3
    fd = (float(er_of_theta(1.0 + h)) - float(er_of_theta(1.0 - h))) / (2.0 * h)

    assert np.isfinite(grad) and abs(fd) > 1e-6
    np.testing.assert_allclose(grad, fd, rtol=1e-4)


@pytest.mark.parametrize("collision,ramped", [(1, False), (1, True), (0, False), (0, True)])
def test_routed_radial_current_gradient_matches_cold_solves(tmp_path, monkeypatch, collision, ramped):
    import jax
    import jax.numpy as jnp
    from dkx import er as er_mod

    deck = _pas_deck(collision_operator=collision, n_theta=5, n_zeta=5, n_xi=6, n_x=3)
    if ramped:
        deck = deck.replace("Nxi_for_x_option = 0", "Nxi_for_x_option = 1")
    problem = er_mod.prepare(_write(tmp_path, deck), tol=1e-11)
    if ramped:
        assert problem.operator.active_dof_mask() is not None
    eye = jnp.eye
    def no_global_identity(n, *args, **kwargs):
        assert n != problem.operator.total_size, "ambipolar AD must not assemble the global dense matrix"
        return eye(n, *args, **kwargs)
    monkeypatch.setattr(jnp, "eye", no_global_identity)

    def current(field):
        return er_mod.radial_current(problem, field, differentiable=True)[0]
    field = 0.2
    value, gradient = jax.jit(jax.value_and_grad(current))(field)
    cold = lambda e: float(er_mod.radial_current(problem, e)[0])
    np.testing.assert_allclose(value, cold(field), rtol=1e-8, atol=1e-15)
    assert np.isfinite(gradient) and abs(gradient) > 1e-15
    differences = [(cold(field + h) - cold(field - h)) / (2 * h) for h in [1e-3, 3e-4, 1e-4]]
    np.testing.assert_allclose(differences, gradient, rtol=3e-4, atol=1e-13)


def test_ambipolar_root_preserves_prepared_solver_policy(tmp_path, monkeypatch):
    from dkx import er as er_mod
    problem = er_mod.prepare(_write(tmp_path, _pas_deck()), solve_method="direct", tol=2e-11)
    with pytest.raises(RuntimeError, match="non-differentiable"):
        er_mod.ambipolar_er(problem, er0=LEGACY_BRENT_ROOT_ER)
    original = er_mod.solve
    calls = []
    def record(op, rhs, **kwargs):
        calls.append((kwargs["method"], kwargs["tol"], kwargs["differentiable"]))
        return original(op, rhs, **kwargs)
    monkeypatch.setattr(er_mod, "solve", record)
    root = er_mod.ambipolar_er(problem, er0=LEGACY_BRENT_ROOT_ER, solve_method="auto")
    assert abs(float(root) - LEGACY_BRENT_ROOT_ER) < 1e-3
    assert calls and set(calls) == {("auto", 2e-11, True)}


@pytest.fixture
def scalar_current(monkeypatch):
    """Isolate root acceptance/IFT from kinetic discretization error."""
    from dkx import er as er_mod
    monkeypatch.setattr(er_mod, "_resolve_problem", lambda parameter, **kwargs: parameter)
    def install(function):
        monkeypatch.setattr(er_mod, "radial_current", lambda p, e, **kwargs: (function(e, p), None, None))
        return er_mod.ambipolar_er
    yield install
    # GPU callback failures can remain queued after block_until_ready raises.
    # Drain expected effects so negative tests do not emit an atexit traceback.
    import jax
    try:
        jax.effects_barrier()
    except Exception as exc:
        assert "Ambipolar root acceptance failed" in str(exc)
        # Also verify recovery: a successful callback replaces the failed
        # runtime token retained by JAX after effects_barrier raises.
        value = jax.jit(install(lambda e, p: e - p))(1.)
        np.testing.assert_allclose(value, 1.)
        jax.effects_barrier()


def test_ambipolar_acceptance_under_jit_grad_and_vmap(scalar_current):
    import jax
    import jax.numpy as jnp
    root = scalar_current(lambda e, p: e - p)
    values, gradients = jax.jit(jax.vmap(jax.value_and_grad(root)))(jnp.array([-2., 0., 3.]))
    np.testing.assert_allclose(values, [-2., 0., 3.], atol=1e-12)
    np.testing.assert_allclose(gradients, 1., atol=1e-12)


@pytest.mark.parametrize("case,options", [
    ("quadratic", {"er0": 1., "max_root_iter": 1}),
    ("linear", {"er0": 5., "root_tol": 1.}),
    ("constant", {}), ("flat", {}), ("nan", {}),
    ("sqrt", {}),
    ("shallow", {"min_abs_slope": 1e-10}),
    # Small current alone is insufficient to bound local field correction.
    ("shallow", {"er0": 5., "root_tol": 1.}),
])
@pytest.mark.parametrize("transform", ["eager", "jit", "grad"])
def test_ambipolar_acceptance_rejects_invalid_roots(scalar_current, case, options, transform):
    import jax
    import jax.numpy as jnp
    functions = {
        "quadratic": lambda e, p: e**2 - 2 * p,
        "linear": lambda e, p: e + p,
        "constant": lambda e, p: jnp.asarray(p),
        "flat": lambda e, p: jnp.zeros_like(e),
        "nan": lambda e, p: jnp.full_like(e, jnp.nan),
        "sqrt": lambda e, p: jnp.sqrt(e),
        "shallow": lambda e, p: 1e-14 * (e - p),
    }
    root = scalar_current(functions[case])
    call = lambda p: root(p, **options)
    if transform == "jit":
        call = jax.jit(call)
    elif transform == "grad":
        call = jax.grad(call)
    with pytest.raises(Exception, match="Ambipolar root acceptance failed"):
        jax.block_until_ready(call(1.))


def test_ambipolar_acceptance_rejects_bad_batch_member(scalar_current):
    import jax
    import jax.numpy as jnp
    root = scalar_current(lambda e, p: p * (e - 1.))
    with pytest.raises(Exception, match="Ambipolar root acceptance failed"):
        jax.jit(jax.vmap(root))(jnp.array([1., 0.])).block_until_ready()


@pytest.mark.parametrize("options", [
    {"root_tol": 0.}, {"root_tol": np.nan}, {"current_tol": -1.},
    {"current_tol": np.inf}, {"min_abs_slope": -1.}, {"min_abs_slope": np.nan},
    {"max_root_iter": 0}, {"max_root_iter": 1.5}, {"max_root_iter": True},
])
def test_ambipolar_acceptance_controls_checked_before_preparation(options):
    from dkx.er import ambipolar_er
    with pytest.raises(ValueError, match="must be"):
        ambipolar_er(None, **options)
