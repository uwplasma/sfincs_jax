Release checklist
=================

This page is intended for maintainers preparing a tagged release (PyPI + Read the Docs).

What this project can and cannot claim
--------------------------------------

`sfincs_jax` is **not yet** a full end-to-end replacement for SFINCS Fortran v3 across the
entire upstream example suite. The project is intentionally **parity-first**:

- We only claim parity for features explicitly covered by tests/fixtures (see `Parity status`).
- The upstream example audit (`Fortran v3 example suite (audit)`) summarizes which example inputs
  currently reach grids/geometry/output writing without errors.

Before shipping a release, make sure `README.md` and `docs/parity.rst` accurately reflect the
current state of the port.

Local validation (recommended)
------------------------------

From the repository root:

.. code-block:: bash

   pytest -q
   sphinx-build -W -b html docs docs/_build/html

Smoke-run the examples that do not require optional dependencies:

.. code-block:: bash

   python examples/1_simple/01_build_grids_and_geometry.py
   python examples/1_simple/02_apply_collisionless_operator.py
   python examples/1_simple/03_write_sfincs_output_python.py
   python examples/1_simple/04_write_sfincs_output_cli.py
   python examples/2_intermediate/10_matrix_free_residual_and_jvp.py

Regenerate the upstream example audit table if upstream inputs or support levels change:

.. code-block:: bash

   python scripts/generate_fortran_example_audit.py

Packaging sanity check
----------------------

CI uses an isolated build environment. Locally, if you are in an offline/sandboxed environment,
an isolated build may fail due to missing network access. In that case, use:

.. code-block:: bash

   python -m build --no-isolation

Fixture generation note (Fortran v3)
------------------------------------

Many parity tests rely on **frozen Fortran v3 fixtures** (PETSc binaries and/or `sfincsOutput.h5`).
Generating new fixtures requires a working v3 executable and an MPI/PETSc runtime environment that
can complete `MPI_Init`. Some sandboxed CI environments can block network endpoints and cause the
Fortran executable to fail at startup; generate fixtures on a normal workstation/HPC environment
and commit the resulting reference files under `tests/ref/`.
