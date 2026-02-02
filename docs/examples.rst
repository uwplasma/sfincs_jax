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
