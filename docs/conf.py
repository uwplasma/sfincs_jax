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
html_static_path = ["_static"]
html_css_files = ["custom.css"]

try:
    import sphinx_rtd_theme  # type: ignore[import-not-found]
except Exception:
    html_theme = "alabaster"
else:
    html_theme = "sphinx_rtd_theme"

# Read the Docs and some locked-down environments can block certain CDNs or inline styles.
# Pin MathJax to a widely mirrored CDN, and prefer the TeX-only bundle to avoid MathML
# fallbacks showing up as visible “math italic text” when CSS is restricted.
mathjax_path = "https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-chtml.min.js"

# Disable the assistive MathML render action (it can become visible if CSS is blocked).
mathjax3_config = {
    "options": {
        # Prefer to disable assistive MathML generation entirely. If it is generated but not hidden
        # (e.g. CSS stripped or theme quirks), it can show up as visible “math italic text” with
        # invisible operator glyphs (⁢, ⁡, …) on some hosted docs.
        "enableAssistiveMml": False,
        "renderActions": {
            # Properly disable assistive MathML output. If it is generated but not hidden (e.g. CSS stripped),
            # it can show up as “math italic text” with invisible operator glyphs (⁢, ⁡, …) on RTD pages.
            "assistiveMml": [0, "", ""],
        }
    }
}
