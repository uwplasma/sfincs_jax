from __future__ import annotations

import sys
import os
from pathlib import Path


def pytest_configure() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    os.environ.setdefault("SFINCS_JAX_FORTRAN_STDOUT", "0")
