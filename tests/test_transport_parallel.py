from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from sfincs_jax.io import read_sfincs_h5, write_sfincs_jax_output_h5


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
