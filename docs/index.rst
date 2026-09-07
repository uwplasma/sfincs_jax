DKX
===

`dkx` solves the radially local, linearized drift-kinetic equation on a flux
surface, in pure JAX. It returns neoclassical particle and heat fluxes,
parallel flows, bootstrap current, transport matrices and ambipolar electric
field roots for stellarators and tokamaks, on CPU or GPU. Expert JAX solve
paths support implicit differentiation within the domains described in
:doc:`differentiability`; file workflows and discrete root selection have
additional limits.

DKX implements a broad subset of SFINCS Fortran v3 physics and reads and
writes SFINCS files. Native case workflows expose a narrower set
of models; see :doc:`case_files` and :doc:`capabilities` for their scope.

Quickstart
----------

.. code-block:: bash

   pip install dkx

A complete run, with no equilibrium file to supply:

.. code-block:: python

   import dkx

   case = dkx.Case.from_mapping({
       "schema": 1,
       "name": "tokamak",
       "run": {"workflow": "profile", "progress": False},
       "geometry": {"format": "analytic", "file": "tokamak",
                    "surfaces": [0.16, 0.25, 0.36]},
       "species": [{"name": "deuterium", "charge": 1, "mass_amu": 2.014,
                    "density_m3": [8.0e19, 7.0e19, 6.0e19],
                    "temperature_keV": [1.0, 0.8, 0.6]}],
       "physics": {"model": "full_local", "collisions": "pitch_angle_scattering"},
       "electric_field": {"mode": "prescribed", "value_kV_m": 0.0},
       "resolution": {"theta": 9, "zeta": 1, "pitch": 8, "speed": 4},
       "solver": {"method": "auto", "relative_tolerance": 1e-8},
   })
   result = dkx.run(case)
   print(float(result.arrays["particle_flux_m2_s"][1, 0]))

Save that mapping as a ``.toml`` file and the command line does the same:

.. code-block:: bash

   dkx validate case.toml     # check it, and print its deterministic id
   dkx run case.toml --out result.nc
   dkx inspect result.nc

``dkx schema --format toml`` prints a template showing every field the schema
accepts. It is a reference, not a starting file: it names a VMEC equilibrium
you have to supply and enables options this quickstart does not need.

The resolution above is sized to run in seconds and is **not converged**. Run
``dkx converge case.toml`` before trusting any number from it.

A case is one TOML or JSON file. ``Case`` is immutable and carries a
deterministic ``case_id``; ``Result`` carries the arrays, the solver route,
the achieved residual and provenance. See :doc:`case_files` for the schema and
:doc:`cli` for the eleven commands.

Coming from SFINCS
------------------

Existing decks keep working, unchanged:

.. code-block:: bash

   dkx input.namelist --out sfincsOutput.h5   # solve a deck directly
   dkx convert input.namelist case.toml       # or turn it into a case file
   dkx sfincs --help                          # the compatibility commands

See :doc:`installation` for the ``solvax`` structured-solver core dependency,
GPU wheels, and the Fortran reference build.

Examples
--------

Nine numbered rungs in ``examples/``, each a ``run.py`` and, where the native
route supports it, an equivalent ``case.toml``. Every rung prints a physical
result with units, writes a NetCDF result and a figure, and runs in seconds;
:doc:`examples` walks through each one.

- ``01_tokamak_profile`` — the whole native loop: ``Case`` to ``dkx.run`` to
  SI moments, certificate, saved result, figure.
- ``02_vmec_stellarator`` — the same solve from a VMEC equilibrium.
- ``03_boozer_stellarator`` — Boozer ``.bc`` geometry, and the solver route
  the operator structure selects.
- ``04_monoenergetic_scan`` — ``D11*``/``D31*``/``D33*`` against collisionality.
- ``05_ambipolar_profile`` — every admitted root, classified, with the
  selection reason.
- ``06_convergence_certificate`` — per-axis and joint refinement, and why a
  small residual is not convergence.
- ``07_gradients`` — ``jax.grad`` through the solve against central differences.
- ``08_vmex_optimization`` — a kinetic shape gradient, and descent on it.
- ``09_phi1_and_impurities`` — a multi-species case with and without ``Phi1``.

Where a rung has a ``case.toml``, the script builds the same case in Python
and asserts the two share a ``case_id``, so the documented command line
provably solves what the script solves.

Performance and parity evidence
-------------------------------

:doc:`performance` records the measured canonical-stack evidence. On the
744k-unknown HSX PAS/DKES case:

.. list-table:: 744k-unknown HSX PAS/DKES case
   :header-rows: 1
   :widths: 46 20 20

   * - Configuration
     - Solve time
     - Peak RSS
   * - ``dkx`` structured direct solve, MacBook M4
     - ``27.2 s``
     - ``0.93 GB``
   * - SFINCS Fortran v3, 1 rank
     - ``463.6 s``
     - ``3.98 GB``
   * - SFINCS Fortran v3, measured 2-rank parallel floor
     - ``229.5 s``
     - ``2.86 GB``

Cross-check tests pin three envelopes against Fortran golden data: RHSMode=1
output tables to ``8e-14``, state vectors to ``1e-11``, and transport matrices
to ``6e-13 .. 9e-9``.

A broader benchmark covers more than that single case. It runs the full 39-case
CPU/GPU example suite against SFINCS Fortran v3, and plots every row whose
Fortran reference runtime clears a ``10 s`` reference-runtime-window, so
process-launch and JIT-amortization noise does not dominate the bars.

.. figure:: _static/figures/paper/dkx_fortran_suite_benchmark_summary.png
   :alt: Runtime and active-memory comparison for SFINCS Fortran v3 and dkx across the example suite.
   :align: center
   :width: 90%

   Example-suite benchmark for rows whose SFINCS Fortran v3 reference runtime is
   at least ``10 s``. Fortran memory is process maximum RSS; JAX memory uses
   profiler RSS deltas over the fixed runtime baseline. Reproduce with
   ``tools/publication_figures/generate_fortran_suite_benchmark_summary.py``.

Documentation map
-----------------

- getting started: :doc:`installation`, :doc:`usage`, :doc:`case_files`, :doc:`cli`, :doc:`examples`
- physics and numerics: :doc:`physics_models`, :doc:`system_equations`,
  :doc:`geometry`, :doc:`method`, :doc:`numerics`, :doc:`differentiability`,
  :doc:`capabilities`
- references: :doc:`inputs`, :doc:`outputs`, :doc:`normalizations`,
  :doc:`source_map`, :doc:`api`
- evidence: :doc:`performance`, :doc:`parity`, :doc:`feature_matrix`,
  :doc:`fortran_comparison`, :doc:`validation_matrix`
- workflows: :doc:`applications`, :doc:`optimization`, :doc:`parallelism`,
  :doc:`vmex_workflow`

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   applications
   optimization
   examples
   usage
   case_files
   cli
   inputs
   outputs
   normalizations
   geometry
   vmex_workflow
   method
   numerics
   differentiability
   capabilities
   source_map
   feature_matrix
   theory_from_upstream
   physics_models
   physics_reference
   system_equations
   parallelism
   research_lanes
   performance
   development_roadmap
   adaptive_speed_grid
   testing
   validation_matrix
   paper_figures
   upstream_docs
   fortran_examples
   utils
   api
   fortran_comparison
   references
   contributing
   release_notes
   release_checklist
