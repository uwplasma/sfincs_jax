from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from sfincs_jax import cli
from sfincs_jax.io import read_sfincs_h5, write_sfincs_jax_output_h5


class _FakeNamelist:
    def __init__(self, rhs_mode: int = 1) -> None:
        self._groups = {
            "general": {"RHSMODE": rhs_mode, "RHSMODE": rhs_mode},
            "geometryParameters": {},
            "physicsParameters": {},
            "resolutionParameters": {},
        }

    def group(self, name: str):
        return self._groups.get(name, {})


def test_cmd_write_output_forces_explicit_mode(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _fake_write_output_h5(**kwargs):
        captured.update(kwargs)
        out = Path(kwargs["output_path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"")
        return out

    monkeypatch.setattr("sfincs_jax.cli.read_sfincs_input", lambda _path: _FakeNamelist(rhs_mode=1))
    monkeypatch.setattr("sfincs_jax.io.write_sfincs_jax_output_h5", _fake_write_output_h5)

    args = Namespace(
        input=str(tmp_path / "input.namelist"),
        out=str(tmp_path / "sfincsOutput.h5"),
        fortran_layout=True,
        overwrite=True,
        compute_transport_matrix=False,
        compute_solution=False,
        geometry_only=False,
        quiet=True,
        verbose=0,
    )
    assert cli._cmd_write_output(args) == 0
    assert captured["differentiable"] is False


def test_cmd_solve_v3_forces_explicit_mode(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _fake_solve(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(x=np.zeros((2,), dtype=np.float64), residual_norm=np.float64(0.0))

    monkeypatch.setattr("sfincs_jax.cli.read_sfincs_input", lambda _path: _FakeNamelist(rhs_mode=1))
    monkeypatch.setattr("sfincs_jax.v3_driver.solve_v3_full_system_linear_gmres", _fake_solve)

    args = Namespace(
        input=str(tmp_path / "input.namelist"),
        out_state=str(tmp_path / "state.npy"),
        tol=1e-8,
        atol=0.0,
        restart=20,
        maxiter=40,
        solve_method="incremental",
        which_rhs=None,
        quiet=True,
        verbose=0,
    )
    assert cli._cmd_solve_v3(args) == 0
    assert captured["differentiable"] is False


def test_cmd_transport_matrix_v3_forces_explicit_mode(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _fake_transport(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            transport_matrix=np.zeros((2, 2), dtype=np.float64),
            state_vectors_by_rhs={},
            residual_norms_by_rhs={1: np.float64(0.0)},
        )

    monkeypatch.setattr("sfincs_jax.cli.read_sfincs_input", lambda _path: _FakeNamelist(rhs_mode=2))
    monkeypatch.setattr("sfincs_jax.v3_driver.solve_v3_transport_matrix_linear_gmres", _fake_transport)

    args = Namespace(
        input=str(tmp_path / "input.namelist"),
        out_matrix=str(tmp_path / "tm.npy"),
        out_state_prefix=None,
        tol=1e-8,
        atol=0.0,
        restart=20,
        maxiter=40,
        solve_method="incremental",
        quiet=True,
        verbose=0,
    )
    assert cli._cmd_transport_matrix_v3(args) == 0
    assert captured["differentiable"] is False


def test_write_output_full_system_regression(tmp_path: Path, monkeypatch) -> None:
    """Full-system write-output should not reference transport-only distributed state."""
    input_path = (
        Path(__file__).parent / "reduced_inputs" / "inductiveE_noEr.input.namelist"
    )
    assert input_path.exists()

    monkeypatch.setenv("SFINCS_JAX_FORTRAN_STDOUT", "0")
    monkeypatch.setenv("SFINCS_JAX_SOLVER_ITER_STATS", "0")

    out_path = tmp_path / "sfincsOutput.h5"
    write_sfincs_jax_output_h5(
        input_namelist=input_path,
        output_path=out_path,
    )

    data = read_sfincs_h5(out_path)
    assert int(np.asarray(data["RHSMode"]).item()) == 1
    assert "classicalParticleFluxNoPhi1_psiHat" in data
