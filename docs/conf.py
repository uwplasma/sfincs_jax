from __future__ import annotations

import sys
from pathlib import Path

project = "sfincs_jax"
copyright = "2026"
author = "sfincs_jax contributors"

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
]

templates_path = ["_templates"]
exclude_patterns: list[str] = []

try:
    import sphinx_rtd_theme  # type: ignore[import-not-found]
except Exception:
    html_theme = "alabaster"
else:
    html_theme = "sphinx_rtd_theme"
