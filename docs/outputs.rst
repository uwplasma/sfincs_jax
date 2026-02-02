Outputs (sfincsOutput.h5)
=========================

SFINCS v3 writes results to an HDF5 file named ``sfincsOutput.h5``. `sfincs_jax` can now
write a v3-style output file for supported modes (currently ``geometryScheme in {4,5,11,12}``).

Writing output with `sfincs_jax`
--------------------------------

CLI
^^^

.. code-block:: bash

   sfincs_jax write-output --input input.namelist --out sfincsOutput.h5

For transport-matrix runs (``RHSMode=2`` or ``RHSMode=3``), the Fortran code loops over
multiple right-hand sides (``whichRHS``) and assembles a ``transportMatrix`` in the output.
To replicate that end-to-end behavior in `sfincs_jax`, enable:

.. code-block:: bash

   sfincs_jax write-output --input input.namelist --out sfincsOutput.h5 --compute-transport-matrix

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
- ``geometryScheme = 5`` (VMEC ``wout_*.nc`` netCDF workflow)
- ``geometryScheme = 11/12`` (Boozer `.bc` files for W7-X / general non-stellarator-symmetric equilibria)
- v3 grids: ``theta``, ``zeta``, ``x`` and ``Nxi_for_x``
- core geometry fields: ``BHat``, ``DHat`` and derivatives available in `sfincs_jax.geometry`
- basic scalar integrals: ``VPrimeHat`` and ``FSABHat2`` (see `sfincs_jax.diagnostics`)
- selected run parameters, radial-coordinate conversions, and species arrays (e.g. ``Delta``, ``alpha``, ``Er``, ``dPhiHatdpsiHat``,
  ``psiAHat``, ``aHat``, ``rN``, ``Zs``, ``THats``)
- `NTV`-related geometry diagnostic ``uHat`` (computed from harmonics of :math:`1/\hat B^2`)

Output parity tests live in:

- ``tests/test_output_h5_scheme4_parity.py`` (scheme 4)
- ``tests/test_output_h5_scheme1_parity.py`` (scheme 1)
- ``tests/test_output_h5_scheme2_parity.py`` (scheme 2)
- ``tests/test_output_h5_scheme11_parity.py`` (scheme 11)
- ``tests/test_output_h5_scheme5_parity.py`` (scheme 5)

and compare the datasets above against frozen Fortran v3 fixtures in ``tests/ref``.

There is also a multi-species parity test against the upstream v3 example output
(``quick_2species_FPCollisions_noEr``), implemented in
``tests/test_output_h5_scheme4_quick2species_parity.py``.

.. note::

   ``uHat`` depends on many transcendental evaluations (cos/sin) and long floating-point
   reductions. In practice we observe tiny platform-dependent differences vs the frozen
   Fortran fixture (absolute errors :math:`\sim 10^{-9}` in the small scheme-4 test case),
   so the parity test compares ``uHat`` with a slightly looser tolerance than most other
   datasets.

Fortran vs Python array layout
------------------------------

Fortran writes arrays in column-major order. When those HDF5 datasets are read back in
Python, multi-dimensional arrays often appear with axes reversed relative to the
``(itheta, izeta, ...)`` indexing used in the Fortran source.

To make it easy to do *file-to-file* comparisons in Python, `sfincs_jax` writes arrays
using the same convention by default (see `sfincs_jax.io.write_sfincs_h5`).
