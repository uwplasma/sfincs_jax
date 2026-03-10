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


def _write_rhs1_compare_h5(path: Path, *, density: np.ndarray, constraint_scheme: int) -> None:
    with h5py.File(path, "w") as f:
        f["geometryScheme"] = np.asarray(11, dtype=np.int32)
        f["RHSMode"] = np.asarray(1, dtype=np.int32)
        f["constraintScheme"] = np.asarray(constraint_scheme, dtype=np.int32)
        f["collisionOperator"] = np.asarray(1, dtype=np.int32)
        f["densityPerturbation"] = np.asarray(density, dtype=np.float64)


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
