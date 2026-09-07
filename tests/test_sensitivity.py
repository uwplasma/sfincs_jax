from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from dkx.sensitivity import (
    LinearObservableSystem,
    MatrixFreeLinearObservableSystem,
    adjoint_dot_product_check,
    evaluate_linear_observable,
    evaluate_matrix_free_linear_observable,
    fortran_v3_adjoint_sensitivity_output_fields,
    fortran_v3_adjoint_sensitivity_output_ranks,
    implicit_linear_observable_derivative,
    implicit_linear_observable_derivative_from_builder,
    implicit_matrix_free_linear_observable_derivative,
    implicit_matrix_free_linear_observable_derivative_from_builder,
    jvp_flux,
    probe_linear_observable_vector,
    validate_fortran_v3_adjoint_sensitivity_constraints,
    validate_fortran_v3_adjoint_sensitivity_output_surface,
    vjp_flux,
)
from dkx.namelist import parse_sfincs_input_text, read_sfincs_input

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "fortran_v3_reference_fixture.json"
SENSITIVITY_REFERENCE_SUMMARY = "small_rhsmode45_summary_2026-06-25.json"
SENSITIVITY_DEBUG_REFERENCE_SUMMARY = "small_rhsmode4_debug_summary_2026-06-25.json"


@lru_cache(maxsize=1)
def _reference_fixture() -> dict:
    return json.loads(REFERENCE_FIXTURE.read_text(encoding="utf-8"))


def _sensitivity_summary(name: str) -> dict:
    return _reference_fixture()["sensitivity"]["summaries"][name]


def _sensitivity_namelist(name: str):
    text = _reference_fixture()["sensitivity"]["namelists"][name]
    return parse_sfincs_input_text(
        text,
        source_path=f"tests/fixtures/fortran_v3_reference_fixture.json:sensitivity/{name}",
    )


def _ambipolar_namelist(name: str):
    text = _reference_fixture()["ambipolar"]["namelists"][name]
    return parse_sfincs_input_text(
        text,
        source_path=f"tests/fixtures/fortran_v3_reference_fixture.json:ambipolar/{name}",
    )


def _linear_system_components():
    a0 = jnp.asarray(
        [
            [4.0, 0.7, -0.1],
            [-0.3, 3.2, 0.4],
            [0.2, -0.5, 2.7],
        ],
        dtype=jnp.float64,
    )
    ap = jnp.asarray(
        [
            [0.5, -0.05, 0.0],
            [0.1, -0.2, 0.03],
            [0.0, 0.08, 0.25],
        ],
        dtype=jnp.float64,
    )
    b0 = jnp.asarray([1.0, -0.25, 0.75], dtype=jnp.float64)
    bp = jnp.asarray([-0.2, 0.4, 0.1], dtype=jnp.float64)
    c0 = jnp.asarray([0.3, -0.7, 1.1], dtype=jnp.float64)
    cp = jnp.asarray([0.05, 0.0, -0.03], dtype=jnp.float64)
    offset0 = 0.125
    offsetp = -0.4
    p0 = 0.35
    return a0, ap, b0, bp, c0, cp, offset0, offsetp, p0


def _adjoint_config(**overrides):
    config = {
        "general": {"RHSMode": 4},
        "geometryParameters": {"geometryScheme": 4},
        "physicsParameters": {
            "includePhi1": False,
            "EParallelHat": 0.0,
            "magneticDriftScheme": 0,
            "collisionOperator": 0,
            "includeXDotTerm": True,
            "useDKESExBDrift": False,
            "includeElectricFieldTermInXiDot": True,
        },
        "resolutionParameters": {"constraintScheme": -1},
        "adjointOptions": {
            "discreteAdjointOption": True,
            "adjointBootstrapOption": False,
            "adjointRadialCurrentOption": False,
            "adjointTotalHeatFluxOption": False,
            "adjointHeatFluxOption": [False, False],
            "adjointParticleFluxOption": [False, False],
            "adjointParallelFlowOption": [False, False],
            "debugAdjoint": False,
        },
    }
    for group, values in overrides.items():
        config[group].update(values)
    return config


def test_jvp_and_vjp_flux_wrappers_match_jax_linearization() -> None:
    def flux(params):
        return jnp.asarray([params[0] ** 2 + params[1], params[0] - 3.0 * params[1]], dtype=jnp.float64)

    params = jnp.asarray([2.0, -1.0], dtype=jnp.float64)
    tangent = jnp.asarray([0.5, 2.0], dtype=jnp.float64)
    cotangent = jnp.asarray([1.5, -0.25], dtype=jnp.float64)

    value, jvp_value = jvp_flux(flux, params, tangent)
    value_vjp, vjp_value = vjp_flux(flux, params, cotangent)

    expected_jacobian = jnp.asarray([[4.0, 1.0], [1.0, -3.0]], dtype=jnp.float64)
    np.testing.assert_allclose(np.asarray(value), np.asarray(flux(params)), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(np.asarray(value_vjp), np.asarray(value), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(np.asarray(jvp_value), np.asarray(expected_jacobian @ tangent), rtol=1.0e-14)
    np.testing.assert_allclose(np.asarray(vjp_value), np.asarray(expected_jacobian.T @ cotangent), rtol=1.0e-14)


def test_fortran_v3_adjoint_sensitivity_constraints_and_output_fields() -> None:
    config = _adjoint_config(
        adjointOptions={
            "adjointBootstrapOption": True,
            "adjointRadialCurrentOption": True,
            "adjointTotalHeatFluxOption": True,
            "adjointHeatFluxOption": [False, True],
            "adjointParticleFluxOption": [True, False],
            "adjointParallelFlowOption": [False, True],
        }
    )

    assert validate_fortran_v3_adjoint_sensitivity_constraints(config) == ()
    assert fortran_v3_adjoint_sensitivity_output_fields(config) == (
        "dHeatFluxdLambda",
        "dParticleFluxdLambda",
        "dParallelFlowdLambda",
        "dTotalHeatFluxdLambda",
        "dRadialCurrentdLambda",
        "dBootstrapdLambda",
    )


def test_fortran_v3_adjoint_sensitivity_constraints_reject_invalid_source_combinations() -> None:
    config = _adjoint_config(
        general={"RHSMode": 5},
        geometryParameters={"geometryScheme": 5},
        physicsParameters={
            "includePhi1": True,
            "EParallelHat": 0.1,
            "magneticDriftScheme": 1,
            "collisionOperator": 1,
            "includeXDotTerm": False,
            "useDKESExBDrift": False,
            "includeElectricFieldTermInXiDot": True,
        },
        resolutionParameters={"constraintScheme": 2},
        adjointOptions={"adjointRadialCurrentOption": True, "discreteAdjointOption": False},
    )

    errors = validate_fortran_v3_adjoint_sensitivity_constraints(config)

    assert len(errors) == 8
    assert any("adjointRadialCurrentOption" in item for item in errors)
    assert any("Boozer-coordinate" in item for item in errors)
    assert any("includePhi1" in item for item in errors)
    assert any("EParallelHat" in item for item in errors)
    assert any("tangential magnetic drifts" in item for item in errors)
    assert any("constraintScheme" in item for item in errors)
    assert any("collisionOperator" in item for item in errors)
    assert any("discreteAdjointOption=false" in item for item in errors)


def test_fortran_v3_adjoint_output_fields_preserve_parallel_flow_source_gate() -> None:
    """Pin the writeHDF5Output.F90 gate before dkx adds its own fields."""

    parallel_only = _adjoint_config(
        adjointOptions={
            "adjointParallelFlowOption": [True, False],
            "adjointParticleFluxOption": [False, False],
        }
    )
    debug_rhs5 = _adjoint_config(
        general={"RHSMode": 5},
        adjointOptions={"debugAdjoint": True},
    )

    assert "dParallelFlowdLambda" not in fortran_v3_adjoint_sensitivity_output_fields(parallel_only)
    fields = fortran_v3_adjoint_sensitivity_output_fields(debug_rhs5)
    assert "dParallelFlowdLambda" in fields
    assert "dPhidPsidLambda" in fields
    assert "dPhidPsiPercentError" in fields
    assert "dPhidPsidLambda_finitediff" in fields


def test_fortran_v3_rhs4_reference_summary_pins_radial_current_sensitivity() -> None:
    summary = _sensitivity_summary(SENSITIVITY_REFERENCE_SUMMARY)
    case = next(item for item in summary["cases"] if item["case"] == "geometry4_w7x_like_small_rhs4_radial_current")
    nml = _sensitivity_namelist("geometry4_w7x_like_small_rhs4_radial_current.namelist")

    assert validate_fortran_v3_adjoint_sensitivity_constraints(nml) == ()
    assert fortran_v3_adjoint_sensitivity_output_fields(nml) == (
        "dParticleFluxdLambda",
        "dParallelFlowdLambda",
        "dRadialCurrentdLambda",
    )
    assert dict(fortran_v3_adjoint_sensitivity_output_ranks(nml)) == {
        "dParticleFluxdLambda": 4,
        "dParallelFlowdLambda": 4,
        "dRadialCurrentdLambda": 3,
    }
    assert validate_fortran_v3_adjoint_sensitivity_output_surface(nml, case["hdf5_fields"]) == ()
    assert case["wall_time_s"] < 1.0
    assert case["max_rss_bytes"] < 150_000_000
    assert case["hdf5_fields"]["dParticleFluxdLambda"]["shape"] == [1, 4, 2, 1]
    assert case["hdf5_fields"]["dRadialCurrentdLambda"]["shape"] == [1, 4, 1]

    particle = np.asarray(case["hdf5_fields"]["dParticleFluxdLambda"]["values"], dtype=np.float64)
    radial = np.asarray(case["hdf5_fields"]["dRadialCurrentdLambda"]["values"], dtype=np.float64)
    expected_radial = particle[:, :, 0, :] - particle[:, :, 1, :]
    np.testing.assert_allclose(radial, expected_radial, rtol=0.0, atol=5.0e-18)


def test_fortran_v3_rhs4_reference_summary_pins_heat_flux_sensitivity() -> None:
    summary = _sensitivity_summary(SENSITIVITY_REFERENCE_SUMMARY)
    case = next(item for item in summary["cases"] if item["case"] == "geometry4_w7x_like_small_rhs4_heat_flux")
    nml = _sensitivity_namelist("geometry4_w7x_like_small_rhs4_heat_flux.namelist")

    assert validate_fortran_v3_adjoint_sensitivity_constraints(nml) == ()
    assert fortran_v3_adjoint_sensitivity_output_fields(nml) == (
        "dHeatFluxdLambda",
        "dTotalHeatFluxdLambda",
    )
    assert dict(fortran_v3_adjoint_sensitivity_output_ranks(nml)) == {
        "dHeatFluxdLambda": 4,
        "dTotalHeatFluxdLambda": 3,
    }
    assert validate_fortran_v3_adjoint_sensitivity_output_surface(nml, case["hdf5_fields"]) == ()
    assert case["wall_time_s"] < 1.0
    assert case["max_rss_bytes"] < 150_000_000
    assert case["hdf5_fields"]["dHeatFluxdLambda"]["shape"] == [1, 4, 2, 1]
    assert case["hdf5_fields"]["dTotalHeatFluxdLambda"]["shape"] == [1, 4, 1]

    heat = np.asarray(case["hdf5_fields"]["dHeatFluxdLambda"]["values"], dtype=np.float64)
    total = np.asarray(case["hdf5_fields"]["dTotalHeatFluxdLambda"]["values"], dtype=np.float64)
    np.testing.assert_allclose(total, heat.sum(axis=2), rtol=0.0, atol=5.0e-18)


def test_fortran_v3_rhs4_reference_summary_pins_parallel_flow_and_bootstrap_sensitivity() -> None:
    summary = _sensitivity_summary(SENSITIVITY_REFERENCE_SUMMARY)
    case = next(item for item in summary["cases"] if item["case"] == "geometry4_w7x_like_small_rhs4_parallel_bootstrap")
    nml = _sensitivity_namelist("geometry4_w7x_like_small_rhs4_parallel_bootstrap.namelist")

    assert validate_fortran_v3_adjoint_sensitivity_constraints(nml) == ()
    assert fortran_v3_adjoint_sensitivity_output_fields(nml) == (
        "dParticleFluxdLambda",
        "dParallelFlowdLambda",
        "dBootstrapdLambda",
    )
    assert dict(fortran_v3_adjoint_sensitivity_output_ranks(nml)) == {
        "dParticleFluxdLambda": 4,
        "dParallelFlowdLambda": 4,
        "dBootstrapdLambda": 3,
    }
    assert validate_fortran_v3_adjoint_sensitivity_output_surface(nml, case["hdf5_fields"]) == ()
    assert case["wall_time_s"] < 1.0
    assert case["max_rss_bytes"] < 150_000_000
    assert case["hdf5_fields"]["dParticleFluxdLambda"]["shape"] == [1, 4, 2, 1]
    assert case["hdf5_fields"]["dParallelFlowdLambda"]["shape"] == [1, 4, 2, 1]
    assert case["hdf5_fields"]["dBootstrapdLambda"]["shape"] == [1, 4, 1]

    flow = np.asarray(case["hdf5_fields"]["dParallelFlowdLambda"]["values"], dtype=np.float64)
    bootstrap = np.asarray(case["hdf5_fields"]["dBootstrapdLambda"]["values"], dtype=np.float64)
    expected_bootstrap = flow[:, :, 0, :] - flow[:, :, 1, :]
    np.testing.assert_allclose(bootstrap, expected_bootstrap, rtol=0.0, atol=5.0e-17)


def test_fortran_v3_rhs5_reference_summary_pins_constant_current_heat_sensitivity() -> None:
    summary = _sensitivity_summary(SENSITIVITY_REFERENCE_SUMMARY)
    case = next(item for item in summary["cases"] if item["case"] == "geometry4_w7x_like_small_rhs5_heat_flux")
    nml = _sensitivity_namelist("geometry4_w7x_like_small_rhs5_heat_flux.namelist")

    assert validate_fortran_v3_adjoint_sensitivity_constraints(nml) == ()
    assert fortran_v3_adjoint_sensitivity_output_fields(nml) == (
        "dHeatFluxdLambda",
        "dTotalHeatFluxdLambda",
        "dPhidPsidLambda",
    )
    assert dict(fortran_v3_adjoint_sensitivity_output_ranks(nml)) == {
        "dHeatFluxdLambda": 4,
        "dTotalHeatFluxdLambda": 3,
        "dPhidPsidLambda": 3,
    }
    assert validate_fortran_v3_adjoint_sensitivity_output_surface(nml, case["hdf5_fields"]) == ()
    assert case["wall_time_s"] < 1.0
    assert case["max_rss_bytes"] < 160_000_000
    assert case["hdf5_fields"]["dHeatFluxdLambda"]["shape"] == [1, 4, 2, 1]
    assert case["hdf5_fields"]["dTotalHeatFluxdLambda"]["shape"] == [1, 4, 1]
    assert case["hdf5_fields"]["dPhidPsidLambda"]["shape"] == [1, 4, 1]

    heat = np.asarray(case["hdf5_fields"]["dHeatFluxdLambda"]["values"], dtype=np.float64)
    total = np.asarray(case["hdf5_fields"]["dTotalHeatFluxdLambda"]["values"], dtype=np.float64)
    dphi = np.asarray(case["hdf5_fields"]["dPhidPsidLambda"]["values"], dtype=np.float64)
    np.testing.assert_allclose(total, heat.sum(axis=2), rtol=0.0, atol=5.0e-18)
    assert np.isfinite(dphi).all()
    assert float(np.max(np.abs(dphi))) > 1.0


def test_fortran_v3_rhs4_debug_reference_summary_pins_finite_difference_outputs() -> None:
    summary = _sensitivity_summary(SENSITIVITY_DEBUG_REFERENCE_SUMMARY)
    nml = _sensitivity_namelist("geometry4_w7x_like_small_rhs4_debug_radial_current.namelist")
    fields = summary["hdf5_fields"]

    assert validate_fortran_v3_adjoint_sensitivity_constraints(nml) == ()
    expected_fields = fortran_v3_adjoint_sensitivity_output_fields(nml)
    assert "dParticleFluxdLambda_finitediff" in expected_fields
    assert "radialCurrentPercentError" in expected_fields
    assert validate_fortran_v3_adjoint_sensitivity_output_surface(nml, fields) == ()
    assert summary["wall_time_s"] < 1.0
    assert summary["max_rss_bytes"] < 160_000_000
    assert summary["finite_difference_times_s"][0] < 0.2

    particle = np.asarray(fields["dParticleFluxdLambda"]["values"], dtype=np.float64)
    particle_fd = np.asarray(fields["dParticleFluxdLambda_finitediff"]["values"], dtype=np.float64)
    radial = np.asarray(fields["dRadialCurrentdLambda"]["values"], dtype=np.float64)
    radial_fd = np.asarray(fields["dRadialCurrentdLambda_finitediff"]["values"], dtype=np.float64)
    particle_error = np.asarray(fields["particleFluxPercentError"]["values"], dtype=np.float64)
    radial_error = np.asarray(fields["radialCurrentPercentError"]["values"], dtype=np.float64)

    assert np.isnan(particle_fd[:, 1, :, :]).all()
    assert np.isnan(radial_fd[:, 1, :]).all()
    assert np.isfinite(np.delete(particle_fd, 1, axis=1)).all()
    assert np.isfinite(np.delete(radial_fd, 1, axis=1)).all()
    assert float(np.max(particle_error)) < 0.03
    assert float(np.max(radial_error)) < 0.02
    np.testing.assert_allclose(
        radial,
        particle[:, :, 0, :] - particle[:, :, 1, :],
        rtol=0.0,
        atol=5.0e-18,
    )


def test_fortran_v3_rhs45_reference_summaries_cover_all_public_sensitivity_families() -> None:
    """The checked fixtures cover every release-facing RHSMode 4/5 output family."""

    summary = _sensitivity_summary(SENSITIVITY_REFERENCE_SUMMARY)
    debug = _sensitivity_summary(SENSITIVITY_DEBUG_REFERENCE_SUMMARY)
    field_names = {
        field_name
        for case in summary["cases"]
        for field_name in case["hdf5_fields"]
    }
    field_names.update(debug["hdf5_fields"])

    expected_release_fields = {
        "dParticleFluxdLambda",
        "dParallelFlowdLambda",
        "dHeatFluxdLambda",
        "dTotalHeatFluxdLambda",
        "dRadialCurrentdLambda",
        "dBootstrapdLambda",
        "dPhidPsidLambda",
    }
    expected_debug_fields = {
        "dParticleFluxdLambda_finitediff",
        "dRadialCurrentdLambda_finitediff",
        "particleFluxPercentError",
        "radialCurrentPercentError",
    }

    assert summary["tier"] == "small"
    assert debug["tier"] == "small"
    assert len(summary["cases"]) == 4
    assert expected_release_fields <= field_names
    assert expected_debug_fields <= field_names
    assert all(case["wall_time_s"] < 1.0 for case in summary["cases"])
    assert all(case["max_rss_bytes"] < 160_000_000 for case in summary["cases"])


def test_fortran_v3_adjoint_sensitivity_output_surface_reports_missing_or_misranked_fields() -> None:
    config = _adjoint_config(adjointOptions={"adjointTotalHeatFluxOption": True})

    errors = validate_fortran_v3_adjoint_sensitivity_output_surface(
        config,
        {
            "dHeatFluxdLambda": {"shape": [1, 4, 2, 1]},
            "dTotalHeatFluxdLambda": {"shape": [1, 4, 2, 1]},
        },
    )

    assert errors == (
        "Sensitivity output field dTotalHeatFluxdLambda has rank 4; expected 3.",
    )


def test_fortran_v3_adjoint_sensitivity_surface_handles_non_rhs45_and_unshaped_outputs() -> None:
    """Guard the lightweight HDF5-summary validator used by frozen v3 fixtures."""

    non_adjoint = _adjoint_config(general={"RHSMode": 1})
    assert fortran_v3_adjoint_sensitivity_output_fields(non_adjoint) == ()
    assert dict(fortran_v3_adjoint_sensitivity_output_ranks(non_adjoint)) == {}
    assert validate_fortran_v3_adjoint_sensitivity_output_surface(non_adjoint, {"ignored": object()}) == (
        "RHSMode 4 or 5 is required for Fortran-v3 adjoint sensitivities.",
    )

    heat_and_total = _adjoint_config(
        adjointOptions={
            "adjointHeatFluxOption": [True, False],
            "adjointTotalHeatFluxOption": True,
        }
    )
    errors = validate_fortran_v3_adjoint_sensitivity_output_surface(
        heat_and_total,
        {"dHeatFluxdLambda": object()},
    )

    assert errors == (
        "Cannot determine shape for sensitivity output field: dHeatFluxdLambda.",
        "Missing Fortran-v3 RHSMode 4/5 sensitivity output field: dTotalHeatFluxdLambda.",
    )


def test_linear_observable_helpers_reject_malformed_shapes() -> None:
    """Shape failures should be explicit before optimization code sees bad derivatives."""

    a0, ap, b0, bp, c0, _cp, offset0, _offsetp, p0 = _linear_system_components()
    system = LinearObservableSystem(
        parameter=p0,
        matrix=a0,
        rhs=b0,
        matrix_derivative=ap,
        rhs_derivative=bp,
        observable_vector=c0,
        observable_offset=offset0,
    )

    with pytest.raises(ValueError, match="matrix must be a square 2D matrix"):
        evaluate_linear_observable(replace(system, matrix=jnp.ones((3,), dtype=jnp.float64)))
    with pytest.raises(ValueError, match="rhs length must match matrix size"):
        evaluate_linear_observable(replace(system, rhs=jnp.ones((2,), dtype=jnp.float64)))
    with pytest.raises(ValueError, match="observable_vector length must match matrix size"):
        evaluate_linear_observable(replace(system, observable_vector=jnp.ones((2,), dtype=jnp.float64)))

    with pytest.raises(ValueError, match="matrix and matrix_derivative must have the same shape"):
        implicit_linear_observable_derivative(
            matrix=a0,
            rhs=b0,
            matrix_derivative=jnp.eye(2, dtype=jnp.float64),
            rhs_derivative=bp,
            observable_vector=c0,
        )
    with pytest.raises(ValueError, match="rhs_derivative length must match matrix size"):
        implicit_linear_observable_derivative(
            matrix=a0,
            rhs=b0,
            matrix_derivative=ap,
            rhs_derivative=jnp.ones((2,), dtype=jnp.float64),
            observable_vector=c0,
        )
    with pytest.raises(ValueError, match="observable_vector_derivative length must match matrix size"):
        implicit_linear_observable_derivative(
            matrix=a0,
            rhs=b0,
            matrix_derivative=ap,
            rhs_derivative=bp,
            observable_vector=c0,
            observable_vector_derivative=jnp.ones((2,), dtype=jnp.float64),
        )
    with pytest.raises(ValueError, match="finite_difference_step must be positive"):
        implicit_linear_observable_derivative(
            matrix=a0,
            rhs=b0,
            matrix_derivative=ap,
            rhs_derivative=bp,
            observable_vector=c0,
            finite_difference_observable=lambda _p: 0.0,
            finite_difference_step=0.0,
        )


def test_matrix_free_observable_helpers_reject_malformed_callbacks() -> None:
    """Matrix-free derivatives must fail at the offending callback boundary."""

    a0, ap, b0, bp, c0, cp, offset0, offsetp, p0 = _linear_system_components()

    def make_system(**overrides) -> MatrixFreeLinearObservableSystem:
        values = {
            "parameter": p0,
            "size": int(a0.shape[0]),
            "rhs": b0,
            "rhs_derivative": bp,
            "apply": lambda x: a0 @ x,
            "transpose_apply": lambda x: a0.T @ x,
            "derivative_apply": lambda x: ap @ x,
            "solve": lambda rhs: jnp.linalg.solve(a0, rhs),
            "transpose_solve": lambda rhs: jnp.linalg.solve(a0.T, rhs),
            "observable_vector": c0,
            "observable_vector_derivative": cp,
            "observable_offset": offset0,
            "observable_offset_derivative": offsetp,
        }
        values.update(overrides)
        return MatrixFreeLinearObservableSystem(**values)

    with pytest.raises(ValueError, match="rhs length must match system.size"):
        evaluate_matrix_free_linear_observable(make_system(rhs=jnp.ones((2,), dtype=jnp.float64)))
    with pytest.raises(ValueError, match="observable_vector length must match system.size"):
        evaluate_matrix_free_linear_observable(
            make_system(observable_vector=jnp.ones((2,), dtype=jnp.float64))
        )
    with pytest.raises(ValueError, match="solve\\(rhs\\) length must match system.size"):
        evaluate_matrix_free_linear_observable(make_system(solve=lambda _rhs: jnp.ones((2,), dtype=jnp.float64)))

    with pytest.raises(ValueError, match="rhs_derivative length must match system.size"):
        implicit_matrix_free_linear_observable_derivative(
            make_system(rhs_derivative=jnp.ones((2,), dtype=jnp.float64))
        )
    with pytest.raises(ValueError, match="observable_vector_derivative length must match system.size"):
        implicit_matrix_free_linear_observable_derivative(
            make_system(observable_vector_derivative=jnp.ones((2,), dtype=jnp.float64))
        )
    with pytest.raises(ValueError, match="apply\\(solution\\) length must match system.size"):
        implicit_matrix_free_linear_observable_derivative(
            make_system(apply=lambda _x: jnp.ones((2,), dtype=jnp.float64))
        )
    with pytest.raises(ValueError, match="derivative_apply\\(solution\\) length must match system.size"):
        implicit_matrix_free_linear_observable_derivative(
            make_system(derivative_apply=lambda _x: jnp.ones((2,), dtype=jnp.float64))
        )
    with pytest.raises(ValueError, match="transpose_solve\\(observable_vector\\) length must match system.size"):
        implicit_matrix_free_linear_observable_derivative(
            make_system(transpose_solve=lambda _rhs: jnp.ones((2,), dtype=jnp.float64))
        )


def test_implicit_linear_observable_derivative_matches_tangent_adjoint_and_finite_difference() -> None:
    a0, ap, b0, bp, c0, cp, offset0, offsetp, p0 = _linear_system_components()

    def observable(p: float) -> float:
        a = a0 + float(p) * ap
        b = b0 + float(p) * bp
        c = c0 + float(p) * cp
        x = jnp.linalg.solve(a, b)
        return float(jnp.vdot(c, x) + offset0 + float(p) * offsetp)

    result = implicit_linear_observable_derivative(
        matrix=a0 + p0 * ap,
        rhs=b0 + p0 * bp,
        matrix_derivative=ap,
        rhs_derivative=bp,
        observable_vector=c0 + p0 * cp,
        observable_vector_derivative=cp,
        observable_offset=offset0 + p0 * offsetp,
        observable_offset_derivative=offsetp,
        parameter=p0,
        finite_difference_observable=observable,
        finite_difference_step=1.0e-6,
        metadata={"case": "nonsymmetric_dense"},
    )

    assert result.metadata["case"] == "nonsymmetric_dense"
    np.testing.assert_allclose(result.observable, observable(p0), rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(result.tangent_derivative, result.adjoint_derivative, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(result.derivative, result.finite_difference_derivative, rtol=0.0, atol=1.0e-8)
    assert result.primal_residual_norm < 1.0e-12
    assert result.tangent_residual_norm < 1.0e-12
    assert result.adjoint_residual_norm < 1.0e-12
    assert result.tangent_adjoint_abs_error < 1.0e-12
    assert result.finite_difference_abs_error is not None
    assert result.finite_difference_abs_error < 1.0e-8


def test_implicit_linear_observable_builder_path_matches_direct_path() -> None:
    a0, ap, b0, bp, c0, cp, offset0, offsetp, p0 = _linear_system_components()

    def build_system(p: float) -> LinearObservableSystem:
        return LinearObservableSystem(
            parameter=float(p),
            matrix=a0 + float(p) * ap,
            rhs=b0 + float(p) * bp,
            matrix_derivative=ap,
            rhs_derivative=bp,
            observable_vector=c0 + float(p) * cp,
            observable_vector_derivative=cp,
            observable_offset=offset0 + float(p) * offsetp,
            observable_offset_derivative=offsetp,
            metadata={"builder": "unit_test"},
        )

    result = implicit_linear_observable_derivative_from_builder(
        build_system,
        parameter=p0,
        finite_difference_step=1.0e-6,
        metadata={"caller": "ambipolar_lane"},
    )

    direct = implicit_linear_observable_derivative(
        matrix=a0 + p0 * ap,
        rhs=b0 + p0 * bp,
        matrix_derivative=ap,
        rhs_derivative=bp,
        observable_vector=c0 + p0 * cp,
        observable_vector_derivative=cp,
        observable_offset=offset0 + p0 * offsetp,
        observable_offset_derivative=offsetp,
        parameter=p0,
        finite_difference_step=None,
    )
    assert result.metadata["builder"] == "unit_test"
    assert result.metadata["caller"] == "ambipolar_lane"
    np.testing.assert_allclose(result.observable, direct.observable, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(result.derivative, direct.derivative, rtol=0.0, atol=1.0e-12)
    assert result.finite_difference_abs_error is not None
    assert result.finite_difference_abs_error < 1.0e-8


def test_matrix_free_linear_observable_builder_matches_dense_certificate() -> None:
    a0, ap, b0, bp, c0, cp, offset0, offsetp, p0 = _linear_system_components()

    def build_system(p: float) -> MatrixFreeLinearObservableSystem:
        a = a0 + float(p) * ap
        return MatrixFreeLinearObservableSystem(
            parameter=float(p),
            size=int(a.shape[0]),
            rhs=b0 + float(p) * bp,
            rhs_derivative=bp,
            apply=lambda x: a @ x,
            transpose_apply=lambda x: a.T @ x,
            derivative_apply=lambda x: ap @ x,
            solve=lambda rhs: jnp.linalg.solve(a, rhs),
            transpose_solve=lambda rhs: jnp.linalg.solve(a.T, rhs),
            observable_vector=c0 + float(p) * cp,
            observable_vector_derivative=cp,
            observable_offset=offset0 + float(p) * offsetp,
            observable_offset_derivative=offsetp,
            metadata={"builder": "matrix_free_unit_test"},
        )

    matrix_free = implicit_matrix_free_linear_observable_derivative_from_builder(
        build_system,
        parameter=p0,
        finite_difference_step=1.0e-6,
        metadata={"caller": "production_contract"},
    )
    dense = implicit_linear_observable_derivative(
        matrix=a0 + p0 * ap,
        rhs=b0 + p0 * bp,
        matrix_derivative=ap,
        rhs_derivative=bp,
        observable_vector=c0 + p0 * cp,
        observable_vector_derivative=cp,
        observable_offset=offset0 + p0 * offsetp,
        observable_offset_derivative=offsetp,
        parameter=p0,
        finite_difference_observable=lambda p: jnp.vdot(
            c0 + float(p) * cp,
            jnp.linalg.solve(a0 + float(p) * ap, b0 + float(p) * bp),
        )
        + offset0
        + float(p) * offsetp,
        finite_difference_step=1.0e-6,
    )

    assert matrix_free.metadata["builder"] == "matrix_free_unit_test"
    assert matrix_free.metadata["caller"] == "production_contract"
    assert matrix_free.metadata["system_kind"] == "matrix_free_linear_observable"
    np.testing.assert_allclose(matrix_free.observable, dense.observable, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(matrix_free.derivative, dense.derivative, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(
        matrix_free.finite_difference_derivative,
        dense.finite_difference_derivative,
        rtol=0.0,
        atol=1.0e-8,
    )
    assert matrix_free.primal_residual_norm < 1.0e-12
    assert matrix_free.tangent_residual_norm < 1.0e-12
    assert matrix_free.adjoint_residual_norm < 1.0e-12
    assert matrix_free.tangent_adjoint_abs_error < 1.0e-12
    assert matrix_free.finite_difference_abs_error is not None
    assert matrix_free.finite_difference_abs_error < 1.0e-8


def test_probe_linear_observable_vector_recovers_chunked_weights_and_offset() -> None:
    weights = jnp.asarray([0.3, -0.4, 1.7, 0.0, -2.0], dtype=jnp.float64)
    offset = 0.125

    def observable(state: jnp.ndarray) -> jnp.ndarray:
        return jnp.vdot(weights, state) + offset

    vector, probed_offset = probe_linear_observable_vector(
        observable,
        size=int(weights.size),
        chunk_size=2,
    )

    np.testing.assert_allclose(vector, weights, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(probed_offset, offset, rtol=0.0, atol=1.0e-12)


def test_rhs1_moment_observables_jvp_vjp_dot_product_gates() -> None:
    """Adjoint dot-product gates through the canonical moment observables.

    Replaces the retired legacy probing-observable machinery: the canonical
    per-species moment table (``run.profile_moments_from_operator``) is a pure
    differentiable function of the state, so forward/reverse sensitivities of
    the flux/flow observables must satisfy <cotangent, J tangent> ==
    <J^T cotangent, tangent>.
    """
    from dkx.drift_kinetic import kinetic_operator_from_namelist
    from dkx.run import profile_moments_from_operator

    input_path = Path(__file__).parent / "ref" / "pas_1species_PAS_noEr_tiny.input.namelist"
    op = kinetic_operator_from_namelist(read_sfincs_input(input_path))
    rng = np.random.default_rng(12)
    state = jnp.asarray(rng.normal(size=(int(op.total_size),)), dtype=jnp.float64)
    tangent = jnp.asarray(rng.normal(size=(int(op.total_size),)), dtype=jnp.float64)
    cotangent = jnp.asarray(0.75, dtype=jnp.float64)

    diagnostic_functions = {
        "particle_flux_vm_psi_hat": lambda x: profile_moments_from_operator(op, x)[
            "particleFlux_vm_psiHat"
        ][0],
        "heat_flux_vm_psi_hat": lambda x: profile_moments_from_operator(op, x)["heatFlux_vm_psiHat"][0],
        "fsab_flow": lambda x: profile_moments_from_operator(op, x)["FSABFlow"][0],
        "fsab_jhat": lambda x: jnp.vdot(op.z_s, profile_moments_from_operator(op, x)["FSABFlow"]),
    }

    for name, diagnostic in diagnostic_functions.items():
        result = adjoint_dot_product_check(diagnostic, state, tangent, cotangent)
        assert result.abs_error < 1.0e-10, name


@pytest.mark.parametrize("rhs_scale", [1., 1e-200, 1e200])
def test_algebraic_moment_error_recovers_manufactured_signed_error(rhs_scale, tmp_path):
    from dkx.sensitivity import linear_observable_algebraic_error

    # Nonnormal, nonsymmetric operator: transposing the wrong solve changes J.
    a = jnp.array([[2., 9., 0.], [0., 3., -4.], [0., 0., 5.]])
    exact = rhs_scale * jnp.array([1., -2., 3.])
    x = exact + rhs_scale * jnp.array([.002, -.001, .003])
    c = jnp.array([1., 2., -1.])
    out = linear_observable_algebraic_error(
        rhs=a @ exact, state=x, observable_vector=c, apply=lambda x: a @ x,
        transpose_apply=lambda x: a.T @ x,
        transpose_solve=lambda b: jnp.linalg.solve(a.T, b),
        primal_rtol=.01, adjoint_rtol=1e-12,
    )
    assert out["signed_correction"] / rhs_scale == pytest.approx(float(c @ ((exact-x)/rhs_scale)), abs=1e-13)
    assert out["corrected_observable"] / rhs_scale == pytest.approx(float(c @ (exact/rhs_scale)))
    assert out["primal_relative_residual"] > 0
    assert not out["is_bound"]
    assert json.loads(json.dumps(out)) == out

    from dkx.result import Result

    result = Result(case_id="manufactured", case_name="linear", workflow="profile",
                    arrays={"objective": np.array(out["observable"])},
                    dimensions={"objective": ()}, metadata={"algebraic_error": out})
    restored = Result.load(result.save(tmp_path / "audit.nc"))
    assert restored.metadata["algebraic_error"] == out


@pytest.mark.parametrize("failure", ["primal", "adjoint", "nonfinite", "shape", "zero_rhs"])
def test_algebraic_moment_error_rejects_invalid_equations(failure):
    from dkx.sensitivity import linear_observable_algebraic_error

    b = jnp.ones(2)
    x = b if failure != "primal" else b * 2
    if failure == "nonfinite":
        x = x.at[0].set(jnp.nan)
    if failure == "shape":
        x = jnp.ones(3)
    if failure == "zero_rhs":
        b, x = jnp.zeros(2), jnp.array([1e-200, 0.])
    with pytest.raises(ValueError):
        linear_observable_algebraic_error(
            rhs=b, state=x, observable_vector=jnp.ones(2), apply=lambda v: v,
            transpose_apply=lambda v: v,
            transpose_solve=lambda v: v * (2 if failure == "adjoint" else 1),
            primal_rtol=0., adjoint_rtol=0.,
        )
