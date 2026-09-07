"""Tests for the phase-space convergence study behind ``dkx converge``.

The arithmetic is tested against a stub solve rather than real ones: a study
costs ``len(axes) + 2`` solves, and what needs pinning here is the refinement
schedule, the relative-change comparison and the verdict, none of which depend
on the physics. One real solve is exercised separately and marked slow.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from dkx.workflows import converge as cv


@dataclass(frozen=True)
class FakeResolution:
    theta: int
    zeta: int
    pitch: int
    speed: int


@dataclass(frozen=True)
class FakeCase:
    resolution: FakeResolution


class FakeResult:
    def __init__(self, arrays):
        self.arrays = arrays
        self.metadata = {"converged": True}


def study(monkeypatch, response, *, case=None, **kwargs):
    """Run a study where ``response`` maps a resolution to one flux value.

    Returns ``(report, calls)``; ``calls`` is every resolution the study asked
    for, in order, which is what pins the refinement schedule and the count.
    """
    calls: list[FakeResolution] = []

    def fake_run_case(case, **_):
        calls.append(case.resolution)
        return FakeResult({"particle_flux_m2_s": np.array([response(case.resolution)])})

    monkeypatch.setattr("dkx.execution.run_case", fake_run_case)
    kwargs.setdefault("observables", ("particle_flux_m2_s",))
    report = cv.converge_case(
        case or FakeCase(FakeResolution(theta=10, zeta=4, pitch=10, speed=10)), **kwargs
    )
    return report, calls


# --------------------------------------------------------------------------
# Refinement schedule
# --------------------------------------------------------------------------


def test_each_axis_is_refined_alone_then_all_together(monkeypatch) -> None:
    report, calls = study(monkeypatch, lambda r: 1.0)
    assert [r.label for r in report.refinements] == ["theta", "zeta", "pitch", "speed"]
    assert report.joint is not None
    # baseline + one per axis + one joint
    assert len(calls) == 6
    assert report.joint.resolution == {"theta": 15, "zeta": 6, "pitch": 15, "speed": 15}


def test_refining_an_axis_leaves_the_others_alone(monkeypatch) -> None:
    report, calls = study(monkeypatch, lambda r: 1.0)
    theta = next(r for r in report.refinements if r.label == "theta")
    assert theta.resolution == {"theta": 15, "zeta": 4, "pitch": 10, "speed": 10}


def test_an_axisymmetric_zeta_is_not_refined(monkeypatch) -> None:
    """``zeta = 1`` says the configuration has no toroidal variation.

    Scaling it would solve a different problem rather than the same one more
    accurately, so the axis is skipped and reported as skipped -- not silently
    refined, and not counted as a converged axis it never tested.
    """
    case = FakeCase(FakeResolution(theta=10, zeta=1, pitch=10, speed=10))
    report, calls = study(monkeypatch, lambda r: 1.0, case=case)
    assert [r.label for r in report.refinements] == ["theta", "pitch", "speed"]
    assert all(r.resolution["zeta"] == 1 for r in report.refinements)
    assert report.joint is not None and report.joint.resolution["zeta"] == 1


def test_refinement_always_advances_even_when_rounding_would_not(monkeypatch) -> None:
    """A small axis with a small factor must still grow by at least one node."""
    case = FakeCase(FakeResolution(theta=2, zeta=2, pitch=2, speed=2))
    report, calls = study(monkeypatch, lambda r: 1.0, case=case, factor=1.01)
    assert all(r.resolution[r.label] == 3 for r in report.refinements)


def test_a_factor_that_does_not_refine_is_refused(monkeypatch) -> None:
    with pytest.raises(ValueError, match="factor must exceed 1.0"):
        study(monkeypatch, lambda r: 1.0, factor=1.0)


def test_an_unknown_axis_is_refused(monkeypatch) -> None:
    with pytest.raises(ValueError, match="unknown refinement axes"):
        study(monkeypatch, lambda r: 1.0, axes=("theta", "radius"))


# --------------------------------------------------------------------------
# Verdict
# --------------------------------------------------------------------------


def test_a_solution_independent_of_resolution_is_converged(monkeypatch) -> None:
    report, calls = study(monkeypatch, lambda r: 3.0)
    assert report.per_axis_worst == 0.0
    assert report.converged


def test_a_single_unconverged_axis_fails_the_whole_study(monkeypatch) -> None:
    """One axis still moving is enough; the verdict is over the worst axis.

    Averaging would let three settled axes hide a fourth that is not, which is
    the failure this command exists to surface.
    """
    report, calls = study(
        monkeypatch, lambda r: 1.0 + (0.5 if r.speed > 10 else 0.0), tolerance=0.02
    )
    speed = next(r for r in report.refinements if r.label == "speed")
    assert speed.worst == pytest.approx(0.5)
    assert not report.converged


def test_the_joint_run_can_fail_a_study_every_axis_passed(monkeypatch) -> None:
    """The reason the joint run is not redundant.

    This response is flat unless *two* axes move together, so every single-axis
    refinement reports zero change and a per-axis-only study would call the case
    converged. This is not hypothetical: on the shipped analytic tokamak deck,
    theta refinement moves the outputs by 0.2% at pitch=8 and by 74% at
    pitch=40 -- the apparent theta convergence was an artifact of pitch being
    too coarse to expose it.
    """
    def response(r):
        return 2.0 if (r.theta > 10 and r.pitch > 10) else 1.0

    report, calls = study(monkeypatch, response, tolerance=0.02)
    assert report.per_axis_worst == 0.0
    assert report.joint is not None and report.joint.worst == pytest.approx(1.0)
    assert not report.converged
    assert report.axes_understate_the_joint_change


def test_skipping_the_joint_run_is_recorded_not_assumed_converged(monkeypatch) -> None:
    report, calls = study(monkeypatch, lambda r: 1.0, joint=False)
    assert report.joint is None
    assert not report.axes_understate_the_joint_change


def test_a_single_refinable_axis_needs_no_joint_run(monkeypatch) -> None:
    """With one axis there is nothing to refine jointly, so the run is skipped."""
    report, calls = study(monkeypatch, lambda r: 1.0, axes=("theta",))
    assert report.joint is None
    assert len(calls) == 2


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------


def test_a_zero_reference_requires_an_explicit_physical_absolute_budget(monkeypatch):
    def response(r):
        return 0.0 if r.theta == 10 else 1e-18
    report, _ = study(monkeypatch, response, axes=("theta",))
    assert not report.converged
    report, _ = study(monkeypatch, response, axes=("theta",),
                      absolute_tolerances={"particle_flux_m2_s": 2e-18})
    assert report.converged
    report, _ = study(monkeypatch, lambda r: 0.0, axes=("theta",))
    assert report.converged


def test_relative_change_is_used_for_ordinary_magnitudes() -> None:
    assert cv._relative_changes({"q": 2.0}, {"q": 3.0})["q"] == pytest.approx(0.5)


def test_an_observable_missing_from_a_refinement_fails_admission() -> None:
    changes = cv._relative_changes({"a": 1.0, "b": 2.0}, {"a": 1.0})
    assert changes["a"] == 0.0
    assert np.isinf(changes["b"])


def test_a_result_without_any_requested_observable_is_an_error(monkeypatch) -> None:
    """Silence here would report a vacuous 'converged' over an empty comparison."""
    monkeypatch.setattr(
        "dkx.execution.run_case", lambda case, **_: FakeResult({"other": np.array([1.0])})
    )
    with pytest.raises(ValueError, match="requested observables are missing"):
        cv.converge_case(FakeCase(FakeResolution(10, 4, 10, 10)))


def test_the_cli_axis_list_matches_the_workflow(monkeypatch) -> None:
    """The CLI mirrors AXES so building the parser does not import the solver."""
    from dkx import cli

    assert cli._CONVERGE_AXES == cv.AXES

@pytest.mark.parametrize('ref, got', [
    ([1., -1.], [-1., 1.]),
    ([[1., 2.], [3., 4.]], [[4., 3.], [2., 1.]]),
    ([1000., 1.], [1000., 2.]),
])
def test_refinement_compares_each_signed_species_and_surface(monkeypatch, ref, got):
    report, _ = study(monkeypatch, lambda r: ref if r.theta == 10 else got,
                      axes=('theta',))
    assert not report.converged
    assert report.refinements[0].worst >= 1.

@pytest.mark.parametrize('got', [[np.nan, 2.], [np.inf, 2.], [], [[1., 2.]]])
def test_invalid_or_misaligned_observables_cannot_pass(monkeypatch, got):
    report, _ = study(monkeypatch, lambda r: [1., 2.] if r.theta == 10 else got,
                      axes=('theta',))
    assert not report.converged


def test_no_refinable_axes_does_not_certify_resolution(monkeypatch):
    report, _ = study(monkeypatch, lambda r: 1., axes=())
    assert not report.converged
    report, _ = study(monkeypatch, lambda r: 1., axes=("zeta",),
                      case=FakeCase(FakeResolution(10, 1, 10, 10)))
    assert not report.converged


def test_a_failed_solve_cannot_certify_resolution(monkeypatch):
    result = FakeResult({"particle_flux_m2_s": np.ones(2)})
    result.metadata["converged"] = False
    monkeypatch.setattr("dkx.execution.run_case", lambda case: result)
    with pytest.raises(ValueError, match="failed solve"):
        cv.converge_case(FakeCase(FakeResolution(10, 4, 10, 10)))


@pytest.mark.parametrize("tolerance", [0., -1., np.inf, np.nan])
def test_invalid_convergence_tolerance_is_rejected(monkeypatch, tolerance):
    with pytest.raises(ValueError, match="tolerance"):
        study(monkeypatch, lambda r: 1., tolerance=tolerance)


@pytest.mark.parametrize("atols", [{"missing": 1.}, {"particle_flux_m2_s": -1.},
                                    {"particle_flux_m2_s": np.nan},
                                    {"particle_flux_m2_s": np.inf}])
def test_invalid_absolute_budgets_are_rejected(monkeypatch, atols):
    with pytest.raises(ValueError, match="absolute"):
        study(monkeypatch, lambda r: 1., absolute_tolerances=atols)


def test_entrywise_change_handles_extreme_finite_values():
    changes = cv._relative_changes({"q": np.array([1e308, 1e-308])},
                                   {"q": np.array([-1e308, 2e-308])})
    assert changes["q"] == pytest.approx(2.)


def test_large_absolute_budget_does_not_overflow_into_false_convergence():
    changes = cv._relative_changes({"q": 0.}, {"q": 1e308}, tolerance=.02,
                                   absolute_tolerances={"q": 5e307})
    assert changes["q"] == pytest.approx(.04)


def test_a_missing_requested_observable_cannot_hide_behind_an_available_one(monkeypatch):
    with pytest.raises(ValueError, match="missing"):
        study(monkeypatch, lambda r: 1., observables=("particle_flux_m2_s", "heat_flux_W_m2"))
