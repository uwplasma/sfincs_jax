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

Some advanced examples require optional dependencies:

.. code-block:: bash

   pip install -e ".[opt]"

