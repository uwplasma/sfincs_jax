Examples
========

The repository includes a structured `examples/` tree:

- `examples/1_simple/`: basic API usage (no Fortran required)
- `examples/2_intermediate/`: parity checks and auto-diff demos
- `examples/3_advanced/`: optimization patterns (may require extras)

Run from the repo root:

.. code-block:: bash

   cd sfincs_jax
   python examples/1_simple/01_build_grids_and_geometry.py

Writing `sfincsOutput.h5` (Python + CLI):

.. code-block:: bash

   python examples/1_simple/03_write_sfincs_output_python.py
   python examples/1_simple/04_write_sfincs_output_cli.py

Matrix-free linear solve demo (using frozen Fortran PETSc binaries):

.. code-block:: bash

   python examples/2_intermediate/04_solve_fortran_matrix_with_gmres.py

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
   python examples/3_advanced/04_optimize_scheme4_harmonics_publication_figures.py
   python examples/3_advanced/05_calibrate_nu_n_to_fortran_residual_fixture.py

Implicit differentiation through solves
---------------------------------------

An important differentiability milestone is **implicit differentiation** through a linear solve
(``A x = b``) without backpropagating through Krylov iterations. `sfincs_jax` provides a small helper
based on `jax.lax.custom_linear_solve` and demonstrates it here:

.. code-block:: bash

   python examples/3_advanced/06_implicit_diff_through_gmres_solve_scheme5.py

Upstream SFINCS example inputs
--------------------------------

For convenience, `sfincs_jax` also vendors the original SFINCS example inputs (Fortran v3, multi-species,
and MATLAB v3) in `examples/upstream/`. These files are intended as recognizable reference points for
SFINCS users; not all of them are runnable end-to-end in `sfincs_jax` yet.
