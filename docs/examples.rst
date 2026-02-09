Examples
========

The repository includes a structured `examples/` tree:

- `examples/getting_started/`: basic API usage (no Fortran required)
- `examples/parity/`: parity checks and validation vs frozen Fortran v3 fixtures
- `examples/transport/`: `RHSMode=2/3` transport-matrix workflows + upstream scanplot scripts
- `examples/autodiff/`: autodiff + implicit-diff demonstrations
- `examples/optimization/`: optimization patterns (may require extras)
- `examples/performance/`: JIT/performance microbenchmarks
- `examples/publication_figures/`: publication-style figure generation

Run from the repo root:

.. code-block:: bash

   cd sfincs_jax
   python examples/getting_started/build_grids_and_geometry.py

Writing `sfincsOutput.h5` (Python + CLI):

.. code-block:: bash

   python examples/getting_started/write_sfincs_output_python.py
   python examples/getting_started/write_sfincs_output_cli.py

Matrix-free linear solve demo (using frozen Fortran PETSc binaries):

.. code-block:: bash

   python examples/parity/solve_fortran_matrix_with_gmres.py
   python examples/autodiff/autodiff_gradient_nu_n_residual.py

Transport matrices (RHSMode=2/3)
--------------------------------

Upstream v3 uses ``RHSMode=2`` and ``RHSMode=3`` to compute transport matrices by looping over multiple
right-hand sides (``whichRHS``) and assembling a matrix from diagnostic moments of the solved distribution.

`sfincs_jax` provides both a Python driver and a CLI:

.. code-block:: bash

   python examples/transport/transport_matrix_rhsmode2_and_rhsmode3.py
   sfincs_jax transport-matrix-v3 --input input.namelist --out-matrix transportMatrix.npy

Upstream postprocessing (utils/)
--------------------------------

SFINCS Fortran v3 includes a set of plotting scripts under `utils/`. `sfincs_jax` vendors these scripts
in `examples/sfincs_examples/utils/` and can run them non-interactively:

.. code-block:: bash

   sfincs_jax postprocess-upstream --case-dir /path/to/case --util sfincsScanPlot_1 -- pdf

There is also a small end-to-end demo that generates PDF figures for a tiny transport-matrix case:

.. code-block:: bash

   pip install -e ".[viz]"
   python examples/transport/postprocess_upstream_scanplot_1_transport_matrix.py

Some advanced examples require optional dependencies:

.. code-block:: bash

   pip install -e ".[opt]"

Plotting examples require:

.. code-block:: bash

   pip install -e ".[viz]"

Optimization + figures
----------------------

Two examples that showcase autodiff-driven optimization (and write publication-style figures when `matplotlib`
is available):

.. code-block:: bash

   pip install -e ".[opt,viz]"
   python examples/optimization/optimize_scheme4_harmonics_publication_figures.py
   python examples/optimization/calibrate_nu_n_to_fortran_residual_fixture.py

Implicit differentiation through solves
---------------------------------------

An important differentiability capability is **implicit differentiation** through a linear solve
(``A x = b``) without backpropagating through Krylov iterations. `sfincs_jax` provides a small helper
based on `jax.lax.custom_linear_solve` and demonstrates it here:

.. code-block:: bash

   python examples/autodiff/implicit_diff_through_gmres_solve_scheme5.py

Upstream SFINCS example inputs
--------------------------------

For convenience, `sfincs_jax` also vendors the original SFINCS example inputs (Fortran v3, multi-species,
and MATLAB v3) in `examples/upstream/`. These files are intended as recognizable reference points for
SFINCS users; not all of them are runnable end-to-end in `sfincs_jax` yet.

The full upstream Fortran v3 example suite (plus the upstream postprocessing scripts) is also vendored in
`examples/sfincs_examples/`. A best-effort runner is provided:

.. code-block:: bash

   python examples/sfincs_examples/run_sfincs_jax.py --write-output
