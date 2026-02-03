from __future__ import annotations

"""
Compute v3 transport matrices (RHSMode=2 and RHSMode=3) using the JAX matrix-free driver.

This example demonstrates:
  - The upstream v3 `whichRHS` loop for transport-matrix runs (RHSMode=2/3).
  - Assembling `transportMatrix` from the solved distributions using v3's formulas.
  - Running the same workflow via the CLI:

        sfincs_jax transport-matrix-v3 --input input.namelist --out-matrix transportMatrix.npy

Notes
-----
These are intentionally small toy runs (tiny grids) that complete quickly on a laptop.
They are meant for learning the workflow, not for production physics.
"""

from pathlib import Path
import sys

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.v3_driver import solve_v3_transport_matrix_linear_gmres


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _try_plot_matrix(matrix: np.ndarray, *, out_png: Path, title: str) -> None:
    try:
        import matplotlib.pyplot as plt  # noqa: WPS433
    except Exception:
        return

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.2, 4.2), constrained_layout=True)
    im = ax.imshow(matrix, cmap="coolwarm", interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("column (whichRHS)")
    ax.set_ylabel("row")
    fig.colorbar(im, ax=ax, shrink=0.9)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out_dir = root / "examples" / "transport" / "output" / "16_transport_matrix"

    # RHSMode=2 (3x3) energy-integrated transport matrix in the simplified LHD model.
    input_rhsmode2 = _write_text(
        out_dir / "rhsmode2_scheme2.input.namelist",
        """
&general
  RHSMode = 2
/

&geometryParameters
  geometryScheme = 2
/

&speciesParameters
  Zs = 1
  mHats = 1
  nHats = 1.0d+0
  THats = 1.0d+0
/

&physicsParameters
  Delta = 4.5694d-3
  alpha = 1.0d+0

  nu_n = 0.15d+0
  Er = 0.0d+0

  collisionOperator = 1
  includeXDotTerm = .false.
  includeElectricFieldTermInXiDot = .false.
  useDKESExBDrift = .true.
  includePhi1 = .false.
/

&resolutionParameters
  Ntheta = 9
  Nzeta = 9
  Nxi = 6
  NL = 3
  Nx = 3
  solverTolerance = 1d-10
/

&otherNumericalParameters
  Nxi_for_x_option = 0
/
""".lstrip(),
    )

    # RHSMode=3 (2x2) monoenergetic transport matrix in the 3-helicity analytic geometryScheme=1 model.
    input_rhsmode3 = _write_text(
        out_dir / "rhsmode3_scheme1.input.namelist",
        """
&general
  RHSMode = 3
/

&geometryParameters
  geometryScheme = 1
  epsilon_t = -0.07053d+0
  epsilon_h = 0.05067d+0
  iota = 0.4542d+0
  GHat = 3.7481d+0
  IHat = 0d+0
  helicity_l = 2
  helicity_n = 10
  B0OverBBar = 1d+0
/

&physicsParameters
  nuPrime = 1.0d+0
  EStar = 0.1d+0

  collisionOperator = 1
  includeXDotTerm = .false.
  includeElectricFieldTermInXiDot = .false.
  useDKESExBDrift = .true.
  includePhi1 = .false.
/

&resolutionParameters
  Ntheta = 9
  Nzeta = 9
  Nxi = 6
  NL = 3
  Nx = 1
  solverTolerance = 1d-10
/

&otherNumericalParameters
  Nxi_for_x_option = 0
/
""".lstrip(),
    )

    for label, path in (("RHSMode=2", input_rhsmode2), ("RHSMode=3", input_rhsmode3)):
        nml = read_sfincs_input(path)
        result = solve_v3_transport_matrix_linear_gmres(nml=nml, tol=1e-10, restart=80, maxiter=400)
        tm = np.asarray(result.transport_matrix)
        print(f"\n{label} transportMatrix (mathematical row/col order):\n{tm}\n")

        fig_path = root / "examples" / "transport" / "figures" / f"16_transport_matrix_{label.replace('=', '').replace(' ', '_')}.png"
        _try_plot_matrix(tm, out_png=fig_path, title=f"{label} transportMatrix")


if __name__ == "__main__":
    main()
