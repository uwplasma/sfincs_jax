"""The first native Case -> solve -> Result route never passes through a deck."""

from __future__ import annotations

import hashlib
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import numpy as np
import pytest

import dkx
from dkx.constants import RadialCoordinates
from dkx.units import HEAT_FLUX, PARALLEL_CURRENT, PARTICLE_FLUX, flux_psi_hat_to_r_hat


def _case():
    return dkx.Case.from_mapping(
        {
            "schema": 1,
            "name": "native_tokamak_profile",
            "run": {"workflow": "profile", "progress": False},
            "geometry": {
                "format": "analytic",
                "file": "tokamak",
                "surfaces": [0.09, 0.16],
            },
            "species": [
                {
                    "name": "deuterium",
                    "charge": 1,
                    "mass_amu": 2.014,
                    "density_m3": [8.0e19, 7.0e19],
                    "temperature_keV": [1.0, 0.8],
                }
            ],
            "physics": {
                "collisions": "pitch_angle_scattering",
                "magnetic_drifts": "dkes",
                "phi1": "off",
            },
            "electric_field": {"mode": "prescribed", "value_kV_m": 0.0},
            "resolution": {"theta": 9, "zeta": 1, "pitch": 8, "speed": 4},
            "solver": {"method": "auto", "relative_tolerance": 1.0e-8},
            "output": {"file": "native-result.nc", "plots": False},
        }
    )


def _vmec_case():
    base = _case()
    return replace(
        base,
        name="native_vmec_profile",
        geometry=replace(
            base.geometry,
            format="vmec",
            file=Path("ref/wout_up_down_asymmetric_tokamak.nc"),
            surfaces=(0.16, 0.25),
        ),
        source_path=Path(__file__).resolve(),
    )


@pytest.mark.parametrize("collisions", ["pitch_angle_scattering", "linearized_fokker_planck"])
def test_native_profile_preparation_builds_only_selected_collisions(monkeypatch, collisions):
    import dkx.collisions as kernels

    base = _case()
    case = replace(base, physics=replace(base.physics, collisions=collisions))
    calls = {"make_fokker_planck_v3_operator": 0, "make_pitch_angle_scattering_v3_operator": 0}
    for name in calls:
        original = getattr(kernels, name)
        def count(*args, _name=name, _original=original, **kwargs):
            calls[_name] += 1
            return _original(*args, **kwargs)
        monkeypatch.setattr(kernels, name, count)
    problem = dkx.prepare_er_scan(case, differentiable_profiles=True)
    assert calls["make_fokker_planck_v3_operator"] == 0
    assert calls["make_pitch_angle_scattering_v3_operator"] == int(collisions == "pitch_angle_scattering")
    assert problem.operator.fp is not None or problem.operator.pas is not None


@pytest.mark.parametrize("surfaces,index", [((.09, .25), 0), ((.09, .25), 1),
                                          ((.04, .09, .2, .4, .7), 0),
                                          ((.04, .09, .2, .4, .7), 2),
                                          ((.04, .09, .2, .4, .7), 4)])
def test_native_profile_stencil_matches_fresh_boundaries_and_interior(surfaces, index):
    import jax
    import jax.numpy as jnp

    base = _case()
    radius = np.sqrt(surfaces)
    n, t = (1-radius/3+radius**2/5)*1e20, 1.5-radius/2+radius**3/4
    species = replace(base.species[0], density_m3=tuple(n), temperature_keV=tuple(t))
    case = replace(base, geometry=replace(base.geometry, surfaces=surfaces), species=(species,))
    problem = dkx.prepare_er_scan(case, surface_index=index, differentiable_profiles=True)
    fresh = dkx.prepare_er_scan(case, surface_index=index)
    def drives(density, temperature):
        op = problem.with_profiles(density_m3=density, temperature_keV=temperature).operator
        return op.dn_hat_dpsi_hat, op.dt_hat_dpsi_hat
    actual = jax.jit(drives)(jnp.asarray(n[:, None]), jnp.asarray(t[:, None]))
    for got, expected in zip(actual, (fresh.operator.dn_hat_dpsi_hat, fresh.operator.dt_hat_dpsi_hat)):
        np.testing.assert_allclose(got, expected, rtol=2e-13, atol=2e-13)
    # Even an invalid point outside the selected local stencil invalidates
    # a supplied native profile; it must not silently produce valid outputs.
    invalid = jnp.asarray(t[:, None]).at[-1, 0].set(-1.)
    assert np.all(np.isnan(jax.jit(drives)(jnp.asarray(n[:, None]), invalid)[1]))


@pytest.mark.parametrize("collisions", ["pitch_angle_scattering", "linearized_fokker_planck"])
@pytest.mark.parametrize("location", ["local", "neighbor"])
def test_native_profile_refresh_updates_stencil_collisions_and_current_gradient(collisions, location):
    import jax
    import jax.numpy as jnp

    base = _case()
    first = replace(base.species[0], density_m3=(8e19, 7e19, 6e19), temperature_keV=(1.2, 1., .7))
    second = replace(first, name="helium", charge=2, mass_amu=4.,
                     density_m3=(2e19, 1.8e19, 1.4e19), temperature_keV=(.9, .7, .5))
    case = replace(base, species=(first, second),
                   geometry=replace(base.geometry, surfaces=(.04, .1225, .36)),
                   physics=replace(base.physics, collisions=collisions),
                   solver=replace(base.solver, relative_tolerance=1e-11))
    problem = dkx.prepare_er_scan(case, surface_index=1, differentiable_profiles=True)
    n = jnp.asarray([s.density_m3 for s in case.species]).T
    t = jnp.asarray([s.temperature_keV for s in case.species]).T
    dn = jnp.asarray([[.1, -.05], [0., 0.], [-.1, .2]])
    dt = jnp.asarray([[.2, -.1], [0., 0.], [-.1, .1]])
    if location == "local":
        dn, dt = jnp.zeros_like(n).at[1].set(jnp.array([.1, -.2])), jnp.zeros_like(t).at[1].set(jnp.array([-.1, .2]))

    def profiles(scale):
        return n*(1+scale*dn), t*(1+scale*dt)

    def current(prepared, differentiable=False):
        scan = dkx.batched_er_scan(prepared, jnp.array([.2]), differentiable=differentiable,
                                   retain_full_state=True)
        value = scan.moments["FSABjHat"][0] * PARALLEL_CURRENT
        if differentiable:
            return value, scan.algebraic_converged
        assert np.all(np.asarray(scan.algebraic_converged))
        return value

    def loss(scale):
        density, temperature = profiles(scale)
        return current(problem.with_profiles(density_m3=density, temperature_keV=temperature), True)

    def fresh(scale):
        density, temperature = map(np.asarray, profiles(scale))
        species = tuple(replace(s, density_m3=tuple(density[:, i]), temperature_keV=tuple(temperature[:, i]))
                        for i, s in enumerate(case.species))
        return dkx.prepare_er_scan(replace(case, species=species), surface_index=1)

    updated = problem.with_profiles(density_m3=profiles(.1)[0], temperature_keV=profiles(.1)[1])
    cold = fresh(.1)
    assert updated.tol == problem.tol and updated.solve_method == problem.solve_method
    assert (updated.er_min, updated.er_max, updated.er_units) == (problem.er_min, problem.er_max, "kV/m")
    for name in ("n_hat", "t_hat", "dn_hat_dpsi_hat", "dt_hat_dpsi_hat"):
        np.testing.assert_allclose(getattr(updated.operator, name), getattr(cold.operator, name), rtol=2e-14, atol=2e-14)
    if location == "neighbor":
        np.testing.assert_array_equal(updated.operator.t_hat, problem.operator.t_hat)
        assert not np.array_equal(updated.operator.dt_hat_dpsi_hat, problem.operator.dt_hat_dpsi_hat)
    (value, accepted), derivative = jax.jit(jax.value_and_grad(loss, has_aux=True))(.1)
    assert np.all(np.asarray(accepted))
    np.testing.assert_allclose(value, current(cold), rtol=2e-8, atol=1e-8)
    assert np.isfinite(derivative) and abs(float(derivative)) > 1e-8
    for step in (1e-3, 3e-4):
        fd = (current(fresh(.1+step))-current(fresh(.1-step))) / (2*step)
        np.testing.assert_allclose(derivative, fd, rtol=2e-4, atol=abs(float(derivative))*1e-7)
    with pytest.raises(ValueError, match="native profiles must have shape"):
        problem.with_profiles(density_m3=n[1], temperature_keV=t[1])
    with pytest.raises(ValueError, match="differentiable_profiles=True"):
        fresh(0.).with_profiles(density_m3=n, temperature_keV=t)


def _boozer_case():
    return dkx.Case.from_file(
        Path(__file__).resolve().parents[1]
        / "examples"
        / "03_boozer_stellarator"
        / "case.toml"
    )


def _ambipolar_case():
    base = _case()
    return replace(
        base,
        name="native_ambipolar_profile",
        run=replace(base.run, workflow="ambipolar_profile"),
        electric_field=replace(
            base.electric_field,
            mode="ambipolar",
            value_kV_m=None,
            search_kV_m=(-5.0, 5.0),
            search_points=5,
            root_tolerance_kV_m=0.05,
            max_root_iterations=8,
        ),
        convergence=replace(
            base.convergence,
            enabled=True,
            observables=("particle_flux", "heat_flux", "electric_field"),
            max_refinements=1,
        ),
    )


def test_native_profile_retains_a_state_satisfying_the_original_equation(monkeypatch):
    import importlib
    # Resolve consumers before patching, so lazy imports cannot retain the spy.
    for name in ("dkx.batch", "dkx.er", "dkx.run"):
        importlib.import_module(name)
    solve_module = importlib.import_module("dkx.solve")
    original = solve_module.solve
    residuals = []
    def checked(op, rhs, **kwargs):
        result = original(op, rhs, **kwargs)
        state = np.asarray(result.x).reshape(-1)
        measured = float(np.linalg.norm(np.asarray(op.apply(state)) - np.asarray(rhs).reshape(-1)))
        assert measured <= kwargs["tol"] * np.linalg.norm(rhs)
        np.testing.assert_allclose(result.residual_norms, measured, atol=1e-14, rtol=1e-8)
        residuals.append(measured)
        return result
    monkeypatch.setattr(solve_module, "solve", checked)
    dkx.run(_case())
    assert len(residuals) == 2


def test_native_grid_honors_explicit_pitch_speed_ramp() -> None:
    from dkx.execution import _make_grids

    default = _case()
    uniform = replace(
        default,
        resolution=replace(default.resolution, pitch_speed_ramp=0),
    )

    default_grids = _make_grids(default, n_periods=1)
    uniform_grids = _make_grids(uniform, n_periods=1)

    assert np.any(np.asarray(default_grids.n_xi_for_x) < default.resolution.pitch)
    np.testing.assert_array_equal(
        uniform_grids.n_xi_for_x,
        np.full(uniform.resolution.speed, uniform.resolution.pitch),
    )


def test_native_grid_honors_explicit_pitch_modes_by_speed() -> None:
    from dkx.execution import _make_grids

    base = _case()
    explicit = replace(
        base,
        resolution=replace(
            base.resolution,
            pitch_modes_by_speed=(4, 5, 7, base.resolution.pitch),
        ),
    )

    grids = _make_grids(explicit, n_periods=1)

    np.testing.assert_array_equal(grids.n_xi_for_x, [4, 5, 7, 8])


def test_native_result_records_explicit_pitch_allocation(tmp_path) -> None:
    base = _case()
    case = replace(
        base,
        resolution=replace(
            base.resolution,
            pitch_modes_by_speed=(4, 5, 7, base.resolution.pitch),
        ),
    )

    result = dkx.run(case, out=tmp_path / "explicit-pitch.nc")

    assert result.metadata["phase_space"] == {
        "pitch_speed_ramp": None,
        "pitch_allocation_source": "explicit",
        "active_pitch_modes_by_speed": (4, 5, 7, 8),
        "active_pitch_mode_sum": 24,
    }


def test_explicit_default_pitch_allocation_preserves_science(tmp_path) -> None:
    base = _case()
    explicit = replace(
        base,
        name="native_tokamak_profile_explicit_default_pitch",
        resolution=replace(
            base.resolution,
            pitch_modes_by_speed=(4, 4, 5, base.resolution.pitch),
        ),
    )

    default_result = dkx.run(base, out=tmp_path / "default-pitch.nc")
    explicit_result = dkx.run(explicit, out=tmp_path / "explicit-default-pitch.nc")

    assert default_result.arrays.keys() == explicit_result.arrays.keys()
    for name in default_result.arrays:
        if name != "solve_time_s":
            np.testing.assert_array_equal(
                explicit_result.arrays[name], default_result.arrays[name]
            )


def test_native_case_solves_without_namelist_conversion(monkeypatch, tmp_path) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("native execution serialized or parsed a SFINCS namelist")

    monkeypatch.setattr("dkx.inputs.SfincsInput.to_namelist", forbidden)
    monkeypatch.setattr("dkx.run.parse_sfincs_input_text", forbidden)
    path = tmp_path / "result.nc"
    result = dkx.run(_case(), out=path)

    assert result.metadata["phase_space"] == {
        "pitch_speed_ramp": 1,
        "pitch_allocation_source": "pitch_speed_ramp",
        "active_pitch_modes_by_speed": (4, 4, 5, 8),
        "active_pitch_mode_sum": 21,
    }

    assert isinstance(result, dkx.Result)
    assert path.is_file()
    assert result.metadata["converged"] is True
    assert result.particle_flux_m2_s.shape == (2, 1)
    assert np.all(np.isfinite(result.particle_flux_m2_s))
    assert np.any(result.particle_flux_m2_s != 0.0)
    assert np.max(result.primal_residual) < 1.0e-8
    assert result.dimensions["particle_flux_m2_s"] == ("surface", "species")
    assert result.certificate()["case_id"] == _case().case_id

    loaded = dkx.Result.load(path)
    assert loaded.case_id == result.case_id
    np.testing.assert_array_equal(loaded.species, ["deuterium"])
    np.testing.assert_allclose(loaded.particle_flux_m2_s, result.particle_flux_m2_s)
    assert result.plot(tmp_path / "profile.png").is_file()


def test_native_ambipolar_result_preserves_scan_roots_and_selection(
    monkeypatch, tmp_path
) -> None:
    from dkx.workflows.ambipolar_native import (
        NativeAmbipolarRoot,
        NativeAmbipolarSurface,
        RootEvaluation,
    )

    base = _case()
    case = replace(
        base,
        name="native_ambipolar_profile",
        run=replace(base.run, workflow="ambipolar_profile"),
        electric_field=replace(
            base.electric_field,
            mode="ambipolar",
            value_kV_m=None,
            search_kV_m=(-4.0, 4.0),
            search_strategy="seeded_brackets",
            seed_brackets_kV_m=(((-2.0, 0.0),), ((-2.0, 0.0),)),
        ),
    )
    calls = []

    def fake_surface(_problem, *, previous_root_kv_m, **controls):
        surface_index = len(calls)
        root_field = -1.5 + 0.25 * surface_index
        evaluation = RootEvaluation(
            electric_field_kv_m=root_field,
            radial_current_a_m2=1.0e-10,
            particle_flux_m2_s=np.asarray([2.0 + surface_index]),
            heat_flux_w_m2=np.asarray([3.0 + surface_index]),
            parallel_current_a_t_m2=4.0 + surface_index,
            residual_norm=1.0e-12,
            stage="root_refinement",
            particle_flux_m2_s_vs_speed=(
                np.asarray([0.1, 0.2, 0.3, 0.4])[:, None]
                * (2.0 + surface_index)
            ),
            heat_flux_w_m2_vs_speed=(
                np.asarray([0.1, 0.2, 0.3, 0.4])[:, None]
                * (3.0 + surface_index)
            ),
        )
        coarse = replace(
            evaluation,
            electric_field_kv_m=-4.0,
            radial_current_a_m2=-2.0,
            stage="coarse_scan",
        )
        root = NativeAmbipolarRoot(
            electric_field_kv_m=root_field,
            radial_current_a_m2=evaluation.radial_current_a_m2,
            slope_a_m2_per_kv_m=2.0,
            root_type="ion",
            bracket_kv_m=(-2.0, 0.0),
            evaluation=evaluation,
        )
        calls.append((previous_root_kv_m, controls["seed_brackets_kv_m"]))
        return NativeAmbipolarSurface(
            evaluations=(coarse, evaluation),
            roots=(root,),
            selected_root=0,
            selected=evaluation,
            status="bracketed_root",
            solve_seconds=0.01,
            batch_chunk_size=2,
            batch_chunks=1,
            search_strategy="seeded_brackets",
            search_scope="explicit_seeded_intervals_only",
        )

    monkeypatch.setattr(
        "dkx.workflows.ambipolar_native.solve_native_ambipolar_surface",
        fake_surface,
    )
    result = dkx.run(case, out=tmp_path / "ambipolar.nc")

    assert calls == [(None, ((-2.0, 0.0),)), (-1.5, ((-2.0, 0.0),))]
    np.testing.assert_allclose(result.electric_field_kV_m, [-1.5, -1.25])
    np.testing.assert_allclose(result.particle_flux_m2_s[:, 0], [2.0, 3.0])
    np.testing.assert_array_equal(result.ambipolar_root_count, [1, 1])
    np.testing.assert_array_equal(result.ambipolar_status, ["bracketed_root"] * 2)
    np.testing.assert_array_equal(
        result.ambipolar_search_scope, ["explicit_seeded_intervals_only"] * 2
    )
    np.testing.assert_array_equal(
        result.ambipolar_root_branch_id[:, 0], ["ion-000", "ion-000"]
    )
    np.testing.assert_array_equal(
        result.selected_ambipolar_branch, ["ion-000", "ion-000"]
    )
    np.testing.assert_array_equal(
        result.ambipolar_selection_reason,
        ["nearest_zero_initial", "continued_selected_branch"],
    )
    np.testing.assert_array_equal(result.ambipolar_branch_event_count, [1, 0])
    assert result.metadata["ambipolar_branch_continuation"]["event_count"] == 1
    assert result.dimensions["radial_current_A_m2"] == ("surface", "evaluation")
    assert result.dimensions["evaluation_particle_flux_m2_s_vs_speed"] == (
        "surface",
        "evaluation",
        "speed",
        "species",
    )
    np.testing.assert_allclose(
        np.nansum(result.evaluation_particle_flux_m2_s_vs_speed, axis=2),
        result.evaluation_particle_flux_m2_s,
    )
    np.testing.assert_allclose(
        np.nansum(result.evaluation_heat_flux_W_m2_vs_speed, axis=2),
        result.evaluation_heat_flux_W_m2,
    )
    assert result.metadata["ambipolar_all_surfaces_bracketed"] is True
    assert result.metadata["ambipolar_search"] == {
        "strategy": "seeded_brackets",
        "scope": ("explicit_seeded_intervals_only",),
        "unsampled_crossings_excluded": False,
    }
    assert "unsampled crossings outside those intervals" in result.warnings[0]
    loaded = dkx.Result.load(tmp_path / "ambipolar.nc")
    np.testing.assert_allclose(loaded.ambipolar_root_kV_m[:, 0], [-1.5, -1.25])
    np.testing.assert_array_equal(
        loaded.selected_ambipolar_branch, result.selected_ambipolar_branch
    )
    np.testing.assert_allclose(loaded.speed_v_th, result.speed_v_th)
    np.testing.assert_allclose(
        loaded.evaluation_particle_flux_m2_s_vs_speed,
        result.evaluation_particle_flux_m2_s_vs_speed,
        equal_nan=True,
    )


def test_native_ambipolar_real_solver_brackets_and_roundtrips(tmp_path, capsys) -> None:
    case = _ambipolar_case()
    baseline = dkx.run(case)
    case = replace(
        case,
        convergence=replace(case.convergence, retain_legendre_tail=True),
    )
    result = dkx.run(case, out=tmp_path / "real-ambipolar.nc")

    assert "[dkx.solve]" not in capsys.readouterr().out
    assert result.metadata["legendre_tail_diagnostic"] == (
        "retained_full_state_relative_l2"
    )
    assert "evaluation_legendre_tail_relative_l2" in result.arrays
    assert "evaluation_legendre_tail_relative_l2_upper_bound" not in result.arrays
    # Full structured recovery avoids a failed moment-only solve and Krylov retry.
    assert result.metadata["ambipolar_solver_attempts"]["automatic_true_residual_recovery_count"] == 0
    assert set(result.metadata["ambipolar_solver_attempts"]["executed_route_counts"]) == {
        "block_tridiagonal_truncated"
    }
    assert "selected_tail_diagnostic_replay" not in set(
        result.evaluation_solver_attempt_reason.reshape(-1)
    )
    for name in (
        "electric_field_kV_m",
        "particle_flux_m2_s",
        "heat_flux_W_m2",
        "parallel_current_A_T_m2",
        "radial_current_A_m2",
        "primal_residual",
        "evaluation_particle_flux_m2_s_vs_speed",
        "evaluation_heat_flux_W_m2_vs_speed",
    ):
        np.testing.assert_array_equal(
            np.asarray(result.arrays[name]),
            np.asarray(baseline.arrays[name]),
        )
    retained_tail = result.evaluation_legendre_tail_relative_l2
    valid_evaluations = np.count_nonzero(np.isfinite(result.evaluation_electric_field_kV_m))
    assert np.count_nonzero(np.isfinite(retained_tail)) == valid_evaluations * 4
    for surface_index, selected_field in enumerate(result.electric_field_kV_m):
        evaluation_index = int(
            np.nanargmin(
                np.abs(
                    result.evaluation_electric_field_kV_m[surface_index]
                    - selected_field
                )
            )
        )
        assert np.all(np.isfinite(retained_tail[surface_index, evaluation_index]))
    np.testing.assert_array_equal(result.ambipolar_root_count, [1, 1])
    np.testing.assert_array_equal(result.ambipolar_status, ["bracketed_root"] * 2)
    np.testing.assert_array_equal(result.ambipolar_refinement_status, ["resolved"] * 2)
    assert np.all(result.ambipolar_refinement_converged[:, -1] == 1)
    assert np.all(result.ambipolar_refinement_max_bracket_width_kV_m[:, -1] <= 0.05)
    assert set(np.unique(result.evaluation_reason)) >= {
        "initial_uniform_grid",
        "interval_midpoint",
        "bracket_bisection",
    }
    assert np.all(np.isfinite(result.electric_field_kV_m))
    assert np.all(np.isfinite(result.particle_flux_m2_s))
    assert np.all(np.isfinite(result.heat_flux_W_m2))
    assert np.all(result.selected_ambipolar_root == 0)
    np.testing.assert_array_equal(
        result.selected_ambipolar_branch, ["ion-000", "ion-000"]
    )
    assert np.all(result.ambipolar_nonsmooth_event == 0)
    assert "ambipolar_branch_continuation" in result.certificate()
    scan_scale = np.nanmax(np.abs(result.radial_current_A_m2), axis=1)
    root_residual = np.abs(result.ambipolar_root_current_A_m2[:, 0])
    assert np.all(root_residual < 0.02 * scan_scale)
    loaded = dkx.Result.load(tmp_path / "real-ambipolar.nc")
    np.testing.assert_allclose(loaded.electric_field_kV_m, result.electric_field_kV_m)
    np.testing.assert_allclose(
        loaded.evaluation_legendre_tail_relative_l2,
        retained_tail,
        equal_nan=True,
    )


def test_result_arrays_and_contract_are_immutable() -> None:
    result = dkx.Result(
        case_id="a" * 64,
        case_name="small",
        workflow="profile",
        arrays={"surface": [0.25], "flux": [[1.0]]},
        dimensions={"surface": ("surface",), "flux": ("surface", "species")},
        metadata={"converged": True},
    )
    with pytest.raises(ValueError):
        result.flux[0, 0] = 2.0
    with pytest.raises(TypeError):
        result.metadata["converged"] = False
    with pytest.raises(FrozenInstanceError):
        result.case_name = "changed"

    nested = dkx.Result(
        case_id="b" * 64,
        case_name="nested",
        workflow="profile",
        arrays={"surface": [0.25]},
        dimensions={"surface": ("surface",)},
        metadata={"timings_s": {"total": 1.0}},
    )
    with pytest.raises(TypeError):
        nested.metadata["timings_s"]["total"] = 2.0


def test_physical_electric_field_normalization_is_explicit() -> None:
    """1 kV/m maps to ErHat=1 only because TBar=1 keV and RBar=1 m."""

    from dkx.execution import _electric_field_kv_m_to_er_hat

    assert _electric_field_kv_m_to_er_hat(1.0) == pytest.approx(1.0)
    assert _electric_field_kv_m_to_er_hat(-3.25) == pytest.approx(-3.25)


@pytest.mark.parametrize("pitch_speed_ramp", [0, 1])
def test_native_normalization_matches_the_accepted_kernel_path(
    pitch_speed_ramp: int,
) -> None:
    """The new boundary changes names/units, not the numerical answer."""

    base = _case()
    case = replace(
        base,
        resolution=replace(base.resolution, pitch_speed_ramp=pitch_speed_ramp),
    )
    native = dkx.run(case)
    r_hat = 0.5585 * np.sqrt(np.asarray(case.geometry.surfaces))
    n_hat = np.asarray(case.species[0].density_m3) / 1.0e20
    t_hat = np.asarray(case.species[0].temperature_keV)
    dn_dr_hat = np.gradient(n_hat, r_hat)[-1]
    dt_dr_hat = np.gradient(t_hat, r_hat)[-1]
    mass_hat = case.species[0].mass_amu * 1.66053906892e-27 / 1.67262192369e-27
    legacy = dkx.run(
        geometryScheme=1,
        inputRadialCoordinate=3,
        rN_wish=0.4,
        Zs=[1.0],
        mHats=[mass_hat],
        nHats=[n_hat[-1]],
        THats=[t_hat[-1]],
        dNHatdrHats=[dn_dr_hat],
        dTHatdrHats=[dt_dr_hat],
        Ntheta=9,
        Nzeta=1,
        Nxi=8,
        NL=4,
        Nx=4,
        collisionOperator=1,
        useDKESExBDrift=True,
        Nxi_for_x_option=case.resolution.pitch_speed_ramp,
        xGridScheme=5,
        solverTolerance=1.0e-8,
    )
    radial = RadialCoordinates(psi_a_hat=0.15596, a_hat=0.5585, r_n=0.4)
    factor = flux_psi_hat_to_r_hat(
        psi_a_hat=radial.psi_a_hat,
        a_hat=radial.a_hat,
        r_n=radial.r_n,
    )
    assert factor == radial.d_dr_hat_to_d_dpsi_hat
    assert factor != radial.d_dpsi_hat_to_d_dr_hat
    np.testing.assert_allclose(
        native.particle_flux_m2_s[-1],
        np.asarray(legacy.moments["particleFlux_vm_psiHat"]) * factor * PARTICLE_FLUX,
        rtol=2.0e-12,
    )
    np.testing.assert_allclose(
        native.heat_flux_W_m2[-1],
        np.asarray(legacy.moments["heatFlux_vm_psiHat"]) * factor * HEAT_FLUX,
        rtol=2.0e-12,
    )
    np.testing.assert_allclose(
        native.parallel_current_A_T_m2[-1],
        np.asarray(legacy.moments["FSABjHat"]) * PARALLEL_CURRENT,
        rtol=2.0e-12,
    )


def test_native_vmec_reuses_file_and_grids_and_matches_scheme5(
    monkeypatch,
) -> None:
    """A profile reads VMEC and constructs its shape-stable grids exactly once."""

    import dkx.magnetic_geometry as magnetic_geometry
    import dkx.phase_space as phase_space

    case = _vmec_case()
    original_read = magnetic_geometry.read_vmec_wout
    original_make_grids = phase_space.make_grids
    original_to_namelist = dkx.inputs.SfincsInput.to_namelist
    original_parse = dkx.run.parse_sfincs_input_text
    calls = {"read": 0, "grids": 0}

    def counted_read(*args, **kwargs):
        calls["read"] += 1
        return original_read(*args, **kwargs)

    def counted_make_grids(*args, **kwargs):
        calls["grids"] += 1
        return original_make_grids(*args, **kwargs)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("native VMEC execution used a SFINCS namelist adapter")

    monkeypatch.setattr(magnetic_geometry, "read_vmec_wout", counted_read)
    monkeypatch.setattr(phase_space, "make_grids", counted_make_grids)
    monkeypatch.setattr("dkx.inputs.SfincsInput.to_namelist", forbidden)
    monkeypatch.setattr("dkx.run.parse_sfincs_input_text", forbidden)
    native = dkx.run(case)

    assert calls == {"read": 1, "grids": 1}
    assert (
        native.metadata["geometry_sha256"]
        == hashlib.sha256(case.geometry_path.read_bytes()).hexdigest()
    )
    assert native.metadata["normalization"]["a_hat"] > 0.0
    assert native.metadata["normalization"]["psi_a_hat"] != 0.0

    # Restore the readers before exercising the established compatibility path.
    monkeypatch.setattr(magnetic_geometry, "read_vmec_wout", original_read)
    monkeypatch.setattr(phase_space, "make_grids", original_make_grids)
    monkeypatch.setattr(dkx.inputs.SfincsInput, "to_namelist", original_to_namelist)
    monkeypatch.setattr(dkx.run, "parse_sfincs_input_text", original_parse)
    r_n = np.sqrt(np.asarray(case.geometry.surfaces))
    r_hat = native.metadata["normalization"]["a_hat"] * r_n
    n_hat = np.asarray(case.species[0].density_m3) / 1.0e20
    t_hat = np.asarray(case.species[0].temperature_keV)
    mass_hat = case.species[0].mass_amu * 1.66053906892e-27 / 1.67262192369e-27
    legacy = dkx.run(
        geometryScheme=5,
        equilibriumFile=str(case.geometry_path),
        inputRadialCoordinate=3,
        rN_wish=float(r_n[-1]),
        VMECRadialOption=0,
        VMEC_Nyquist_option=1,
        Zs=[1.0],
        mHats=[mass_hat],
        nHats=[n_hat[-1]],
        THats=[t_hat[-1]],
        dNHatdrHats=[np.gradient(n_hat, r_hat)[-1]],
        dTHatdrHats=[np.gradient(t_hat, r_hat)[-1]],
        Ntheta=case.resolution.theta,
        Nzeta=case.resolution.zeta,
        Nxi=case.resolution.pitch,
        NL=min(4, case.resolution.pitch),
        Nx=case.resolution.speed,
        collisionOperator=1,
        useDKESExBDrift=True,
        Nxi_for_x_option=case.resolution.pitch_speed_ramp,
        xGridScheme=5,
        solverTolerance=case.solver.relative_tolerance,
    )
    radial = RadialCoordinates(
        psi_a_hat=native.metadata["normalization"]["psi_a_hat"],
        a_hat=native.metadata["normalization"]["a_hat"],
        r_n=float(r_n[-1]),
    )
    np.testing.assert_allclose(
        native.particle_flux_m2_s[-1],
        np.asarray(legacy.moments["particleFlux_vm_psiHat"])
        * radial.d_dr_hat_to_d_dpsi_hat
        * PARTICLE_FLUX,
        rtol=2.0e-12,
    )
    np.testing.assert_allclose(
        native.heat_flux_W_m2[-1],
        np.asarray(legacy.moments["heatFlux_vm_psiHat"])
        * radial.d_dr_hat_to_d_dpsi_hat
        * HEAT_FLUX,
        rtol=2.0e-12,
    )
    np.testing.assert_allclose(
        native.parallel_current_A_T_m2[-1],
        np.asarray(legacy.moments["FSABjHat"]) * PARALLEL_CURRENT,
        rtol=2.0e-12,
    )


def test_unsupported_native_route_names_the_field_and_correction() -> None:
    case = _case()
    case = replace(case, physics=replace(case.physics, magnetic_drifts="full"))
    with pytest.raises(dkx.CaseValidationError) as excinfo:
        dkx.run(case)
    message = str(excinfo.value)
    assert "physics.magnetic_drifts" in message
    assert "dkes" in message


def test_native_convergence_controls_are_ambipolar_and_observable_specific() -> None:
    prescribed = _case()
    prescribed = replace(
        prescribed,
        convergence=replace(prescribed.convergence, enabled=True),
    )
    with pytest.raises(dkx.CaseValidationError, match="ambipolar_profile"):
        dkx.run(prescribed)
    prescribed_tail = replace(
        _case(),
        convergence=replace(_case().convergence, retain_legendre_tail=True),
    )
    with pytest.raises(dkx.CaseValidationError, match="retain_legendre_tail"):
        dkx.run(prescribed_tail)

    ambipolar = _ambipolar_case()
    ambipolar = replace(
        ambipolar,
        convergence=replace(
            ambipolar.convergence,
            observables=("particle_flux", "not_an_observable"),
        ),
    )
    with pytest.raises(dkx.CaseValidationError, match="not_an_observable"):
        dkx.run(ambipolar)


def test_native_boozer_reuses_parsed_data_and_matches_scheme12(monkeypatch) -> None:
    """The native Boozer path reads once and agrees with the accepted kernel."""
    import dkx.magnetic_geometry as magnetic_geometry
    import dkx.phase_space as phase_space

    case = _boozer_case()
    original_read = magnetic_geometry.read_native_boozer
    original_make_grids = phase_space.make_grids
    original_to_namelist = dkx.inputs.SfincsInput.to_namelist
    original_parse = dkx.run.parse_sfincs_input_text
    calls = {"read": 0, "grids": 0}

    def counted_read(*args, **kwargs):
        calls["read"] += 1
        return original_read(*args, **kwargs)

    def counted_make_grids(*args, **kwargs):
        calls["grids"] += 1
        return original_make_grids(*args, **kwargs)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("native Boozer execution used a SFINCS namelist adapter")

    monkeypatch.setattr(magnetic_geometry, "read_native_boozer", counted_read)
    monkeypatch.setattr(phase_space, "make_grids", counted_make_grids)
    monkeypatch.setattr("dkx.inputs.SfincsInput.to_namelist", forbidden)
    monkeypatch.setattr("dkx.run.parse_sfincs_input_text", forbidden)
    native = dkx.run(case)

    assert calls == {"read": 1, "grids": 1}
    assert (
        native.metadata["geometry_sha256"]
        == hashlib.sha256(case.geometry_path.read_bytes()).hexdigest()
    )
    assert np.max(native.primal_residual) < case.solver.relative_tolerance

    monkeypatch.setattr(magnetic_geometry, "read_native_boozer", original_read)
    monkeypatch.setattr(phase_space, "make_grids", original_make_grids)
    monkeypatch.setattr(dkx.inputs.SfincsInput, "to_namelist", original_to_namelist)
    monkeypatch.setattr(dkx.run, "parse_sfincs_input_text", original_parse)
    r_n = np.sqrt(np.asarray(case.geometry.surfaces))
    r_hat = native.metadata["normalization"]["a_hat"] * r_n
    n_hat = np.asarray(case.species[0].density_m3) / 1.0e20
    t_hat = np.asarray(case.species[0].temperature_keV)
    mass_hat = case.species[0].mass_amu * 1.66053906892e-27 / 1.67262192369e-27
    legacy = dkx.run(
        geometryScheme=12,
        equilibriumFile=str(case.geometry_path),
        inputRadialCoordinate=3,
        rN_wish=float(r_n[-1]),
        VMECRadialOption=0,
        Zs=[1.0],
        mHats=[mass_hat],
        nHats=[n_hat[-1]],
        THats=[t_hat[-1]],
        dNHatdrHats=[np.gradient(n_hat, r_hat)[-1]],
        dTHatdrHats=[np.gradient(t_hat, r_hat)[-1]],
        Ntheta=case.resolution.theta,
        Nzeta=case.resolution.zeta,
        Nxi=case.resolution.pitch,
        NL=min(4, case.resolution.pitch),
        Nx=case.resolution.speed,
        collisionOperator=1,
        useDKESExBDrift=True,
        Nxi_for_x_option=1,
        xGridScheme=5,
        solverTolerance=case.solver.relative_tolerance,
    )
    radial = RadialCoordinates(
        psi_a_hat=native.metadata["normalization"]["psi_a_hat"],
        a_hat=native.metadata["normalization"]["a_hat"],
        r_n=float(r_n[-1]),
    )
    np.testing.assert_allclose(
        native.particle_flux_m2_s[-1],
        np.asarray(legacy.moments["particleFlux_vm_psiHat"])
        * radial.d_dr_hat_to_d_dpsi_hat
        * PARTICLE_FLUX,
        rtol=2.0e-12,
    )
    np.testing.assert_allclose(
        native.heat_flux_W_m2[-1],
        np.asarray(legacy.moments["heatFlux_vm_psiHat"])
        * radial.d_dr_hat_to_d_dpsi_hat
        * HEAT_FLUX,
        rtol=2.0e-12,
    )
    np.testing.assert_allclose(
        native.parallel_current_A_T_m2[-1],
        np.asarray(legacy.moments["FSABjHat"]) * PARALLEL_CURRENT,
        rtol=2.0e-12,
    )


@pytest.mark.parametrize("collisions", ["pitch_angle_scattering", "linearized_fokker_planck"])
def test_native_prepared_scan_matches_fresh_cases_and_field_derivative(tmp_path, monkeypatch, collisions):
    import importlib
    import jax
    import jax.numpy as jnp

    base = _case()
    electron = replace(base.species[0], name="electron", charge=-1, mass_amu=0.00054858)
    case = replace(base, species=(*base.species, electron),
                   physics=replace(base.physics, collisions=collisions),
                   solver=replace(base.solver, relative_tolerance=1e-11))
    def forbidden(*args, **kwargs):
        raise AssertionError('native preparation must neither solve nor parse a namelist')
    # Resolve consumers before replacing the provider, so lazy imports cannot
    # retain the temporary mock after this context exits.
    er_mod = importlib.import_module('dkx.er')
    with monkeypatch.context() as patch:
        patch.setattr(importlib.import_module('dkx.solve'), 'solve', forbidden)
        patch.setattr(er_mod, 'solve', forbidden)
        patch.setattr(importlib.import_module('dkx.namelist'), 'read_sfincs_input', forbidden)
        patch.setattr(importlib.import_module('dkx.inputs'), 'read_sfincs_input', forbidden)
        patch.setattr(importlib.import_module('dkx.inputs'), 'parse_sfincs_input_text', forbidden)
        problem = dkx.prepare_er_scan(case, surface_index=1)
    assert problem.er_units == 'kV/m'
    assert problem.tol == case.solver.relative_tolerance
    fields = jnp.asarray([0.0, 0.2])
    scan = jax.jit(lambda values: dkx.batched_er_scan(problem, values, devices="auto"))(fields)
    if len(jax.local_devices()) == 2:
        assert len(scan.states.addressable_shards) == 2
        assert not scan.states.is_fully_replicated
    for index, field in enumerate(fields):
        fresh = replace(case, electric_field=replace(case.electric_field, value_kV_m=float(field)))
        native = dkx.run(fresh, out=tmp_path / f'fresh-{index}.nc')
        np.testing.assert_allclose(
            scan.moments['FSABjHat'][index] * PARALLEL_CURRENT,
            native.parallel_current_A_T_m2[1], rtol=1e-8, atol=1e-8,
        )
    def current(field):
        return dkx.batched_er_scan(problem, jnp.reshape(field, (1,)),
                                   differentiable=True).moments['FSABjHat'][0]
    derivative = jax.jit(jax.grad(current))(0.2)
    fd = (current(0.201) - current(0.199)) / 0.002
    np.testing.assert_allclose(derivative, fd, rtol=1e-4, atol=1e-10)
    with pytest.raises(IndexError, match='surface_index'):
        dkx.prepare_er_scan(case, surface_index=-1)
    with pytest.raises(TypeError, match='native Case'):
        dkx.prepare_er_scan('case.toml')
