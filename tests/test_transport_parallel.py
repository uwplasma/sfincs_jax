from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from sfincs_jax.io import read_sfincs_h5, write_sfincs_jax_output_h5
from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax import cli
from sfincs_jax import v3_driver
from sfincs_jax.v3_driver import solve_v3_transport_matrix_linear_gmres


def test_transport_parallel_whichrhs_matches_sequential(tmp_path, monkeypatch) -> None:
    """Parallel whichRHS transport should match sequential outputs for a tiny case."""
    if (os.cpu_count() or 1) < 2:
        pytest.skip("need >=2 CPU cores for parallel whichRHS test")
    here = Path(__file__).parent
    input_path = here / "ref" / "transportMatrix_PAS_tiny_rhsMode2_scheme2.input.namelist"
    assert input_path.exists()

    # Sequential run
    seq_path = tmp_path / "seq.h5"
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_PARALLEL", "off")
    monkeypatch.setenv("SFINCS_JAX_FORTRAN_STDOUT", "0")
    monkeypatch.setenv("SFINCS_JAX_SOLVER_ITER_STATS", "0")
    write_sfincs_jax_output_h5(
        input_namelist=input_path,
        output_path=seq_path,
        compute_transport_matrix=True,
    )

    # Parallel whichRHS run
    par_path = tmp_path / "par.h5"
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_PARALLEL", "process")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_PARALLEL_WORKERS", "2")
    write_sfincs_jax_output_h5(
        input_namelist=input_path,
        output_path=par_path,
        compute_transport_matrix=True,
    )

    seq = read_sfincs_h5(seq_path)
    par = read_sfincs_h5(par_path)

    for key in (
        "transportMatrix",
        "particleFlux_vm_psiHat",
        "heatFlux_vm_psiHat",
        "FSABFlow",
        "sources",
        "densityPerturbation",
        "pressurePerturbation",
        "pressureAnisotropy",
    ):
        if key not in seq or key not in par:
            continue
        np.testing.assert_allclose(np.asarray(seq[key]), np.asarray(par[key]), rtol=5e-4, atol=1e-10)


def test_transport_scheme1_monoenergetic_write_output_regression(tmp_path, monkeypatch) -> None:
    """Scheme-1 monoenergetic transport output should not fail on an undefined distributed axis."""
    here = Path(__file__).parent
    input_path = here / "reduced_inputs" / "monoenergetic_geometryScheme1.input.namelist"
    if not input_path.exists():
        input_path = here.parent / "tests" / "reduced_inputs" / "monoenergetic_geometryScheme1.input.namelist"
    assert input_path.exists()

    scaled_input = tmp_path / "mono_scheme1.input.namelist"
    text = input_path.read_text()
    replacements = {
        "  Ntheta = 15": "  Ntheta = 6",
        "  Nzeta = 15": "  Nzeta = 6",
        "  Nxi = 18": "  Nxi = 8",
        "  Nx = 16": "  Nx = 4",
    }
    for old, new in replacements.items():
        assert old in text
        text = text.replace(old, new, 1)
    scaled_input.write_text(text)

    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_PARALLEL", "off")
    monkeypatch.setenv("SFINCS_JAX_FORTRAN_STDOUT", "0")
    monkeypatch.setenv("SFINCS_JAX_SOLVER_ITER_STATS", "0")

    out_path = tmp_path / "mono_scheme1.h5"
    write_sfincs_jax_output_h5(
        input_namelist=scaled_input,
        output_path=out_path,
        compute_transport_matrix=True,
    )

    data = read_sfincs_h5(out_path)
    assert int(np.asarray(data["RHSMode"]).item()) == 3
    assert "DHat" in data
    dhat = np.asarray(data["DHat"])
    assert dhat.size > 0
    assert np.all(np.isfinite(dhat))


def test_transport_scheme1_monoenergetic_gpu_heuristic_regression(tmp_path, monkeypatch) -> None:
    """Monoenergetic transport should avoid unsupported GPU-only tzfft paths in explicit mode."""
    here = Path(__file__).parent
    input_path = here / "reduced_inputs" / "monoenergetic_geometryScheme1.input.namelist"
    if not input_path.exists():
        input_path = here.parent / "tests" / "reduced_inputs" / "monoenergetic_geometryScheme1.input.namelist"
    assert input_path.exists()

    scaled_input = tmp_path / "mono_scheme1_gpu.input.namelist"
    text = input_path.read_text()
    replacements = {
        "  Ntheta = 15": "  Ntheta = 6",
        "  Nzeta = 15": "  Nzeta = 6",
        "  Nxi = 18": "  Nxi = 8",
        "  Nx = 16": "  Nx = 4",
    }
    for old, new in replacements.items():
        assert old in text
        text = text.replace(old, new, 1)
    scaled_input.write_text(text)

    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_PARALLEL", "off")
    monkeypatch.setenv("SFINCS_JAX_FORTRAN_STDOUT", "0")
    monkeypatch.setenv("SFINCS_JAX_SOLVER_ITER_STATS", "0")
    monkeypatch.setattr(v3_driver.jax, "default_backend", lambda: "gpu")

    out_path = tmp_path / "mono_scheme1_gpu.h5"
    write_sfincs_jax_output_h5(
        input_namelist=scaled_input,
        output_path=out_path,
        compute_transport_matrix=True,
    )

    data = read_sfincs_h5(out_path)
    assert int(np.asarray(data["RHSMode"]).item()) == 3
    assert "DHat" in data
    dhat = np.asarray(data["DHat"])
    assert dhat.size > 0
    assert np.all(np.isfinite(dhat))


def test_transport_solve_minimal_outputs_matches_full(monkeypatch: pytest.MonkeyPatch) -> None:
    """Minimal transport-output mode should preserve matrix/flux diagnostics."""
    here = Path(__file__).parent
    input_path = here / "ref" / "transportMatrix_PAS_tiny_rhsMode2_scheme2.input.namelist"
    assert input_path.exists()
    nml = read_sfincs_input(input_path)

    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_PARALLEL", "off")
    monkeypatch.setenv("SFINCS_JAX_FORTRAN_STDOUT", "0")
    monkeypatch.setenv("SFINCS_JAX_SOLVER_ITER_STATS", "0")

    full = solve_v3_transport_matrix_linear_gmres(
        nml=nml,
        tol=1e-10,
        input_namelist=input_path,
        collect_transport_output_fields=True,
    )
    minimal = solve_v3_transport_matrix_linear_gmres(
        nml=nml,
        tol=1e-10,
        input_namelist=input_path,
        collect_transport_output_fields=False,
    )

    np.testing.assert_allclose(
        np.asarray(minimal.transport_matrix),
        np.asarray(full.transport_matrix),
        rtol=5e-4,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        np.asarray(minimal.particle_flux_vm_psi_hat),
        np.asarray(full.particle_flux_vm_psi_hat),
        rtol=5e-4,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        np.asarray(minimal.heat_flux_vm_psi_hat),
        np.asarray(full.heat_flux_vm_psi_hat),
        rtol=5e-4,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        np.asarray(minimal.fsab_flow),
        np.asarray(full.fsab_flow),
        rtol=5e-4,
        atol=1e-10,
    )
    assert minimal.transport_output_fields is None


def test_transport_theta_dd_preconditioner_matches_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Theta-DD transport preconditioner should preserve transport outputs on tiny PAS cases."""
    here = Path(__file__).parent
    input_path = here / "ref" / "transportMatrix_PAS_tiny_rhsMode2_scheme2.input.namelist"
    assert input_path.exists()
    nml = read_sfincs_input(input_path)

    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_PARALLEL", "off")
    monkeypatch.setenv("SFINCS_JAX_FORTRAN_STDOUT", "0")
    monkeypatch.setenv("SFINCS_JAX_SOLVER_ITER_STATS", "0")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_DENSE_FALLBACK", "0")

    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_PRECOND", raising=False)
    base = solve_v3_transport_matrix_linear_gmres(
        nml=nml,
        tol=1e-10,
        input_namelist=input_path,
        collect_transport_output_fields=False,
    )

    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_PRECOND", "theta_dd")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_DD_BLOCK_T", "2")
    theta_dd = solve_v3_transport_matrix_linear_gmres(
        nml=nml,
        tol=1e-10,
        input_namelist=input_path,
        collect_transport_output_fields=False,
    )

    np.testing.assert_allclose(
        np.asarray(theta_dd.transport_matrix),
        np.asarray(base.transport_matrix),
        rtol=5e-4,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        np.asarray(theta_dd.particle_flux_vm_psi_hat),
        np.asarray(base.particle_flux_vm_psi_hat),
        rtol=5e-4,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        np.asarray(theta_dd.heat_flux_vm_psi_hat),
        np.asarray(base.heat_flux_vm_psi_hat),
        rtol=5e-4,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        np.asarray(theta_dd.fsab_flow),
        np.asarray(base.fsab_flow),
        rtol=5e-4,
        atol=1e-10,
    )


def test_transport_theta_schwarz_preconditioner_matches_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Theta-Schwarz transport preconditioner should preserve transport outputs on tiny PAS cases."""
    here = Path(__file__).parent
    input_path = here / "ref" / "transportMatrix_PAS_tiny_rhsMode2_scheme2.input.namelist"
    assert input_path.exists()
    nml = read_sfincs_input(input_path)

    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_PARALLEL", "off")
    monkeypatch.setenv("SFINCS_JAX_FORTRAN_STDOUT", "0")
    monkeypatch.setenv("SFINCS_JAX_SOLVER_ITER_STATS", "0")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_DENSE_FALLBACK", "0")

    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_PRECOND", raising=False)
    base = solve_v3_transport_matrix_linear_gmres(
        nml=nml,
        tol=1e-10,
        input_namelist=input_path,
        collect_transport_output_fields=False,
    )

    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_PRECOND", "theta_schwarz")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_DD_BLOCK_T", "2")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_DD_OVERLAP", "1")
    theta_schwarz = solve_v3_transport_matrix_linear_gmres(
        nml=nml,
        tol=1e-10,
        input_namelist=input_path,
        collect_transport_output_fields=False,
    )

    np.testing.assert_allclose(
        np.asarray(theta_schwarz.transport_matrix),
        np.asarray(base.transport_matrix),
        rtol=5e-4,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        np.asarray(theta_schwarz.particle_flux_vm_psi_hat),
        np.asarray(base.particle_flux_vm_psi_hat),
        rtol=5e-4,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        np.asarray(theta_schwarz.heat_flux_vm_psi_hat),
        np.asarray(base.heat_flux_vm_psi_hat),
        rtol=5e-4,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        np.asarray(theta_schwarz.fsab_flow),
        np.asarray(base.fsab_flow),
        rtol=5e-4,
        atol=1e-10,
    )


def test_transport_parallel_pool_reuses_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Persistent transport pool should be reused across repeated requests."""

    class _DummyPool:
        init_calls = 0
        shutdown_calls = 0

        def __init__(self, **_kwargs):
            type(self).init_calls += 1
            self._shutdown = False

        def shutdown(self, wait: bool = True, cancel_futures: bool = True) -> None:
            _ = (wait, cancel_futures)
            if not self._shutdown:
                self._shutdown = True
                type(self).shutdown_calls += 1

    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_POOL_PERSIST", "1")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_MP_START_METHOD", "spawn")
    monkeypatch.setattr(v3_driver.concurrent.futures, "ProcessPoolExecutor", _DummyPool)
    monkeypatch.setattr(v3_driver.mp, "get_context", lambda _name: object())

    v3_driver._shutdown_transport_parallel_pool()
    try:
        pool_1 = v3_driver._get_transport_parallel_pool(parallel_workers=2)
        pool_2 = v3_driver._get_transport_parallel_pool(parallel_workers=2)
        assert pool_1 is pool_2
        assert _DummyPool.init_calls == 1
    finally:
        v3_driver._shutdown_transport_parallel_pool()
    assert _DummyPool.shutdown_calls == 1


def test_transport_parallel_pool_rebuilds_on_worker_change(monkeypatch: pytest.MonkeyPatch) -> None:
    """Persistent transport pool should rebuild when parallel worker count changes."""

    class _DummyPool:
        init_calls = 0
        shutdown_calls = 0

        def __init__(self, **_kwargs):
            type(self).init_calls += 1
            self._shutdown = False

        def shutdown(self, wait: bool = True, cancel_futures: bool = True) -> None:
            _ = (wait, cancel_futures)
            if not self._shutdown:
                self._shutdown = True
                type(self).shutdown_calls += 1

    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_POOL_PERSIST", "1")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_MP_START_METHOD", "spawn")
    monkeypatch.setattr(v3_driver.concurrent.futures, "ProcessPoolExecutor", _DummyPool)
    monkeypatch.setattr(v3_driver.mp, "get_context", lambda _name: object())

    v3_driver._shutdown_transport_parallel_pool()
    try:
        pool_1 = v3_driver._get_transport_parallel_pool(parallel_workers=2)
        pool_2 = v3_driver._get_transport_parallel_pool(parallel_workers=3)
        assert pool_1 is not pool_2
        assert _DummyPool.init_calls == 2
        # First pool should be shut down on key change.
        assert _DummyPool.shutdown_calls == 1
    finally:
        v3_driver._shutdown_transport_parallel_pool()
    assert _DummyPool.shutdown_calls == 2


def test_apply_cores_setting_does_not_force_transport_parallel(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_PARALLEL", raising=False)
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_PARALLEL_WORKERS", raising=False)
    monkeypatch.delenv("SFINCS_JAX_GMRES_DISTRIBUTED", raising=False)
    monkeypatch.delenv("SFINCS_JAX_CORES", raising=False)
    monkeypatch.delenv("JAX_PLATFORM_NAME", raising=False)

    cli._apply_cores_setting(4)

    assert os.environ["SFINCS_JAX_CORES"] == "4"
    assert os.environ["SFINCS_JAX_GMRES_DISTRIBUTED"] == "auto"
    assert "SFINCS_JAX_TRANSPORT_PARALLEL" not in os.environ
    assert "SFINCS_JAX_TRANSPORT_PARALLEL_WORKERS" not in os.environ


def test_apply_cores_setting_skips_distributed_gmres_auto_on_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_PARALLEL", raising=False)
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_PARALLEL_WORKERS", raising=False)
    monkeypatch.delenv("SFINCS_JAX_GMRES_DISTRIBUTED", raising=False)
    monkeypatch.delenv("SFINCS_JAX_CORES", raising=False)
    monkeypatch.setenv("JAX_PLATFORM_NAME", "gpu")

    cli._apply_cores_setting(4)

    assert os.environ["SFINCS_JAX_CORES"] == "4"
    assert "SFINCS_JAX_GMRES_DISTRIBUTED" not in os.environ
