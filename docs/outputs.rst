Outputs (sfincsOutput.h5)
========================

SFINCS v3 writes results to an HDF5 file named ``sfincsOutput.h5``. `sfincs_jax` can now
write a **subset** of this file for the parts of the code that are implemented in JAX.

Writing output with `sfincs_jax`
-------------------------------

CLI
^^^

.. code-block:: bash

   sfincs_jax write-output --input input.namelist --out sfincsOutput.h5

The default output uses a **Fortran-compatible array layout**, which is recommended if
you intend to compare against Fortran v3 output using ``sfincs_jax compare-h5``.

Python
^^^^^^

.. code-block:: python

   from pathlib import Path
   from sfincs_jax.io import write_sfincs_jax_output_h5

   write_sfincs_jax_output_h5(
       input_namelist=Path("input.namelist"),
       output_path=Path("sfincsOutput.h5"),
   )

Current coverage
----------------

At the moment, `sfincs_jax` output writing supports:

- ``geometryScheme = 4`` (simplified W7-X Boozer model)
- v3 grids: ``theta``, ``zeta``, ``x`` and ``Nxi_for_x``
- core geometry fields: ``BHat``, ``DHat`` and derivatives available in `sfincs_jax.geometry`
- basic scalar integrals: ``VPrimeHat`` and ``FSABHat2`` (see `sfincs_jax.diagnostics`)
- selected run parameters, radial-coordinate conversions, and species arrays (e.g. ``Delta``, ``alpha``, ``Er``, ``dPhiHatdpsiHat``,
  ``psiAHat``, ``aHat``, ``rN``, ``Zs``, ``THats``)

Output parity tests live in ``tests/test_output_h5_scheme4_parity.py`` and compare the
datasets above against a frozen Fortran v3 fixture in ``tests/ref``.

Fortran vs Python array layout
------------------------------

Fortran writes arrays in column-major order. When those HDF5 datasets are read back in
Python, multi-dimensional arrays often appear with axes reversed relative to the
``(itheta, izeta, ...)`` indexing used in the Fortran source.

To make it easy to do *file-to-file* comparisons in Python, `sfincs_jax` writes arrays
using the same convention by default (see `sfincs_jax.io.write_sfincs_h5`).
