DKX documentation
=================

DKX computes neoclassical transport, flows, bootstrap current and stellarator
ambipolar electric fields on toroidal flux surfaces. Start with a physical-unit
case, then check its numerical resolution before using the result for research.

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
