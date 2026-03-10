from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from sfincs_jax.compare import compare_sfincs_outputs


def _write_minimal_compare_h5(path: Path, *, gpsi: np.ndarray, dtheta: np.ndarray) -> None:
    with h5py.File(path, "w") as f:
        f["geometryScheme"] = np.asarray(5, dtype=np.int32)
        f["RHSMode"] = np.asarray(3, dtype=np.int32)
        f["constraintScheme"] = np.asarray(2, dtype=np.int32)
        f["gpsiHatpsiHat"] = np.asarray(gpsi, dtype=np.float64)
        f["dBHat_sup_theta_dzeta"] = np.asarray(dtheta, dtype=np.float64)
        f["NTVBeforeSurfaceIntegral"] = np.asarray(gpsi, dtype=np.float64)


def _write_compare_case_h5(
    path: Path,
    *,
    rhs_mode: int,
    constraint_scheme: int,
    geometry_scheme: int = 11,
    collision_operator: int = 1,
    fields: dict[str, np.ndarray],
) -> None:
    with h5py.File(path, "w") as f:
        f["geometryScheme"] = np.asarray(geometry_scheme, dtype=np.int32)
        f["RHSMode"] = np.asarray(rhs_mode, dtype=np.int32)
        f["constraintScheme"] = np.asarray(constraint_scheme, dtype=np.int32)
        f["collisionOperator"] = np.asarray(collision_operator, dtype=np.int32)
        for key, value in fields.items():
            f[key] = np.asarray(value, dtype=np.float64)


def _write_rhs1_compare_h5(path: Path, *, density: np.ndarray, constraint_scheme: int) -> None:
    _write_compare_case_h5(
        path,
        rhs_mode=1,
        constraint_scheme=constraint_scheme,
        fields={"densityPerturbation": density},
    )


def test_compare_masks_vmec_reference_corruption_outliers(tmp_path: Path) -> None:
    ref_path = tmp_path / "fortran.h5"
    jax_path = tmp_path / "jax.h5"

    finite = np.asarray([[0.5, 1.0], [1.5, 2.0]], dtype=np.float64)
    corrupted_gpsi = finite.copy()
    corrupted_gpsi[0, 1] = 1.125899906842624e15
    corrupted_dtheta = finite.copy()
    corrupted_dtheta[1, 0] = np.nan

    _write_minimal_compare_h5(ref_path, gpsi=corrupted_gpsi, dtheta=corrupted_dtheta)
    _write_minimal_compare_h5(jax_path, gpsi=finite, dtheta=finite)

    results = compare_sfincs_outputs(
        a_path=jax_path,
        b_path=ref_path,
        keys=["gpsiHatpsiHat", "dBHat_sup_theta_dzeta", "NTVBeforeSurfaceIntegral"],
        rtol=0.0,
        atol=1.0e-12,
    )

    assert all(result.ok for result in results), results


def test_compare_preserves_rhs1_model_floor_over_tighter_case_tolerance(tmp_path: Path) -> None:
    ref_path = tmp_path / "fortran_rhs1.h5"
    jax_path = tmp_path / "jax_rhs1.h5"

    ref_density = np.asarray([[[[-1.1250897588471625e-03]]]], dtype=np.float64)
    jax_density = np.asarray([[[[-1.1237663486148910e-03]]]], dtype=np.float64)

    _write_rhs1_compare_h5(ref_path, density=ref_density, constraint_scheme=2)
    _write_rhs1_compare_h5(jax_path, density=jax_density, constraint_scheme=2)

    results = compare_sfincs_outputs(
        a_path=jax_path,
        b_path=ref_path,
        keys=["densityPerturbation"],
        rtol=5.0e-4,
        atol=1.0e-9,
        tolerances={"densityPerturbation": {"atol": 1.0e-7}},
    )

    assert len(results) == 1
    assert results[0].ok, results


def test_compare_applies_rhs1_constraint0_center_fsa_and_flux_floors(tmp_path: Path) -> None:
    ref_path = tmp_path / "fortran_cs0.h5"
    jax_path = tmp_path / "jax_cs0.h5"

    ref_density = np.asarray([[[[1.0, 2.0]]]], dtype=np.float64)
    jax_density = ref_density + 1.0e-3
    ref_heat = np.asarray([1.0e-8], dtype=np.float64)
    jax_heat = np.asarray([1.9e-6], dtype=np.float64)
    ref_aniso = np.asarray([0.0, -1.5e-6], dtype=np.float64)
    jax_aniso = np.asarray([0.0, -1.985e-4], dtype=np.float64)

    _write_compare_case_h5(
        ref_path,
        rhs_mode=1,
        constraint_scheme=0,
        fields={
            "densityPerturbation": ref_density,
            "heatFlux_vm_psiHat": ref_heat,
            "pressureAnisotropy": ref_aniso,
            "FSADensityPerturbation": np.asarray([1.0e-2], dtype=np.float64),
        },
    )
    _write_compare_case_h5(
        jax_path,
        rhs_mode=1,
        constraint_scheme=0,
        fields={
            "densityPerturbation": jax_density,
            "heatFlux_vm_psiHat": jax_heat,
            "pressureAnisotropy": jax_aniso,
            "FSADensityPerturbation": np.asarray([0.0], dtype=np.float64),
        },
    )

    results = compare_sfincs_outputs(
        a_path=jax_path,
        b_path=ref_path,
        keys=["densityPerturbation", "heatFlux_vm_psiHat", "pressureAnisotropy", "FSADensityPerturbation"],
        rtol=5.0e-4,
        atol=1.0e-9,
        tolerances=None,
    )

    assert all(result.ok for result in results), results


def test_compare_applies_rhs1_constraint2_pressure_floors(tmp_path: Path) -> None:
    ref_path = tmp_path / "fortran_cs2.h5"
    jax_path = tmp_path / "jax_cs2.h5"

    _write_compare_case_h5(
        ref_path,
        rhs_mode=1,
        constraint_scheme=2,
        fields={
            "FSAPressurePerturbation": np.asarray([1.32401224e-05], dtype=np.float64),
            "pressurePerturbation": np.asarray([355.43718933], dtype=np.float64),
            "heatFlux_vm_psiHat_vs_x": np.asarray([1.88506326e-05], dtype=np.float64),
        },
    )
    _write_compare_case_h5(
        jax_path,
        rhs_mode=1,
        constraint_scheme=2,
        fields={
            "FSAPressurePerturbation": np.asarray([0.0], dtype=np.float64),
            "pressurePerturbation": np.asarray([355.42837155], dtype=np.float64),
            "heatFlux_vm_psiHat_vs_x": np.asarray([2.03554974e-05], dtype=np.float64),
        },
    )

    results = compare_sfincs_outputs(
        a_path=jax_path,
        b_path=ref_path,
        keys=["FSAPressurePerturbation", "pressurePerturbation", "heatFlux_vm_psiHat_vs_x"],
        rtol=5.0e-4,
        atol=1.0e-9,
        tolerances=None,
    )

    assert all(result.ok for result in results), results


def test_compare_applies_transport_momentum_flux_floor(tmp_path: Path) -> None:
    ref_path = tmp_path / "fortran_transport.h5"
    jax_path = tmp_path / "jax_transport.h5"

    _write_compare_case_h5(
        ref_path,
        rhs_mode=3,
        constraint_scheme=2,
        geometry_scheme=1,
        fields={"momentumFlux_vm_psiHat": np.asarray([-3.94630491e-08], dtype=np.float64)},
    )
    _write_compare_case_h5(
        jax_path,
        rhs_mode=3,
        constraint_scheme=2,
        geometry_scheme=1,
        fields={"momentumFlux_vm_psiHat": np.asarray([0.0], dtype=np.float64)},
    )

    results = compare_sfincs_outputs(
        a_path=jax_path,
        b_path=ref_path,
        keys=["momentumFlux_vm_psiHat"],
        rtol=5.0e-4,
        atol=1.0e-9,
        tolerances=None,
    )

    assert len(results) == 1
    assert results[0].ok, results


def test_compare_applies_rhs1_fsabflow_vs_x_floor(tmp_path: Path) -> None:
    ref_path = tmp_path / "fortran_rhs1_flow.h5"
    jax_path = tmp_path / "jax_rhs1_flow.h5"

    _write_compare_case_h5(
        ref_path,
        rhs_mode=1,
        constraint_scheme=1,
        fields={"FSABFlow_vs_x": np.asarray([-8.67663610e-08], dtype=np.float64)},
    )
    _write_compare_case_h5(
        jax_path,
        rhs_mode=1,
        constraint_scheme=1,
        fields={"FSABFlow_vs_x": np.asarray([7.26881129e-08], dtype=np.float64)},
    )

    results = compare_sfincs_outputs(
        a_path=jax_path,
        b_path=ref_path,
        keys=["FSABFlow_vs_x"],
        rtol=5.0e-4,
        atol=1.0e-9,
        tolerances=None,
    )

    assert len(results) == 1
    assert results[0].ok, results


def test_compare_applies_rhs1_constraint2_fsabflow_vs_x_floor(tmp_path: Path) -> None:
    ref_path = tmp_path / "fortran_rhs1_flow_cs2.h5"
    jax_path = tmp_path / "jax_rhs1_flow_cs2.h5"

    _write_compare_case_h5(
        ref_path,
        rhs_mode=1,
        constraint_scheme=2,
        fields={"FSABFlow_vs_x": np.asarray([-8.67663610e-08], dtype=np.float64)},
    )
    _write_compare_case_h5(
        jax_path,
        rhs_mode=1,
        constraint_scheme=2,
        fields={"FSABFlow_vs_x": np.asarray([7.26881129e-08], dtype=np.float64)},
    )

    results = compare_sfincs_outputs(
        a_path=jax_path,
        b_path=ref_path,
        keys=["FSABFlow_vs_x"],
        rtol=5.0e-4,
        atol=1.0e-9,
        tolerances=None,
    )

    assert len(results) == 1
    assert results[0].ok, results
