from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from sfincs_jax.io import read_sfincs_h5, write_sfincs_jax_output_h5
from sfincs_jax.namelist import read_sfincs_input
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

    for key in ("transportMatrix", "particleFlux_vm_psiHat", "heatFlux_vm_psiHat", "FSABFlow"):
        assert key in seq and key in par
        np.testing.assert_allclose(np.asarray(seq[key]), np.asarray(par[key]), rtol=5e-4, atol=1e-10)


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
