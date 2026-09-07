DKX documentation
=================

DKX computes neoclassical transport, flows, bootstrap current and stellarator
ambipolar electric fields on toroidal flux surfaces. Start with a physical-unit
case, then check its numerical resolution before using the result for research.

Quickstart
----------

.. code-block:: python

   import dkx

   case = dkx.Case.from_mapping({           # analytic tokamak, teaching grid: seconds, not converged
       "schema": 1, "name": "tokamak", "run": {"workflow": "profile", "progress": False},
       "geometry": {"format": "analytic", "file": "tokamak", "surfaces": [0.16, 0.25, 0.36]},
       "species": [{"name": "deuterium", "charge": 1, "mass_amu": 2.014,
                    "density_m3": [8.0e19, 7.0e19, 6.0e19], "temperature_keV": [1.0, 0.8, 0.6]}],
       "physics": {"model": "full_local", "collisions": "pitch_angle_scattering", "magnetic_drifts": "dkes", "phi1": "off"},
       "electric_field": {"mode": "prescribed", "value_kV_m": 0.0},
       "resolution": {"theta": 9, "zeta": 1, "pitch": 8, "speed": 4}, "solver": {"method": "auto", "relative_tolerance": 1e-8},
   })
   result = dkx.run(case)
   print("solver route:", result.metadata["solver_route"])
   print(float(result.arrays["particle_flux_m2_s"][1, 0]))   # particle flux, m^-2 s^-1

Then ``dkx converge examples/01_tokamak_profile/case.toml`` before using any number.

Performance evidence
--------------------

.. figure:: _static/figures/paper/dkx_fortran_suite_benchmark_summary.png

Fortran reference runtime clears a ``10 s`` reference-runtime-window, so
process-launch and JIT-amortization noise does not dominate the bars.

.. figure:: _static/figures/paper/dkx_fortran_suite_benchmark_summary.png

Choose a starting point
-----------------------

- **First calculation:** :doc:`installation`, then :doc:`examples` and
  :doc:`case_files`. Example 01 needs no external equilibrium.
- **A research calculation:** :doc:`physics_models`, :doc:`capabilities` and
  :doc:`validation_matrix` describe the equations and the supported evidence.
- **Scans and derivatives:** :doc:`parallelism` and :doc:`differentiability`
  explain prepared inputs, independent batches and implicit solve checks.
- **Optimization:** :doc:`optimization` and :doc:`vmex_workflow` distinguish
  the analytic geometry tutorial from a complete equilibrium-boundary workflow.

The numbered examples provide small teaching calculations. Research results
need separate and joint resolution studies for each observable. A successful
linear solve alone does not certify fluxes, bootstrap current or a root branch.
The compatibility/expert interface covers more physics than native Case execution;
see :doc:`capabilities` before selecting an advanced model.

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   installation
   examples
   usage

.. toctree::
   :maxdepth: 1
   :caption: How-to guides

   case_files
   applications
   optimization
   vmex_workflow
   parallelism
   fortran_examples

.. toctree::
   :maxdepth: 1
   :caption: Physics and numerical explanation

   physics_models
   system_equations
   geometry
   method
   numerics
   differentiability
   physics_reference
   theory_from_upstream
   adaptive_speed_grid

.. toctree::
   :maxdepth: 1
   :caption: Reference

   api
   cli
   inputs
   outputs
   normalizations
   capabilities
   feature_matrix
   source_map
   utils
   references
   upstream_docs

.. toctree::
   :maxdepth: 1
   :caption: Validation and development

   validation_matrix
   performance
   fortran_comparison
   research_lanes
   paper_figures
   testing
   contributing
   development_roadmap
   release_notes
   release_checklist
