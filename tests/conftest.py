from __future__ import annotations

import sys
import os
from pathlib import Path

import pytest


def pytest_configure() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    os.environ.setdefault("SFINCS_JAX_FORTRAN_STDOUT", "0")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Keep CI runtime under control by skipping the slowest integration tests."""
    if os.environ.get("SFINCS_JAX_CI", "0") != "1":
        return

    # Skip only the heaviest end-to-end tests in CI; keep unit + fast parity checks.
    slow_mark = pytest.mark.skip(reason="Skipped slow integration test in CI mode.")
    slow_patterns = (
        "test_transport_matrix_",
        "test_transport_parallel",
        "test_state_recycle_parity",
        "test_er_scan_and_ambipolar",
        "test_upstream_scanplot2_smoke",
        "test_rhsmode1_write_output_end_to_end",
        "test_rhsmode1_phi1_write_output_end_to_end",
        "test_output_h5_scheme",
        "test_full_system_newton_krylov",
        "test_full_system_gmres_solution_parity",
    )
    for item in items:
        nodeid = item.nodeid
        if any(pat in nodeid for pat in slow_patterns):
            item.add_marker(slow_mark)
