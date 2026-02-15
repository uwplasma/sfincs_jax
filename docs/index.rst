sfincs_jax
==========

`sfincs_jax` is a JAX implementation of **SFINCS Fortran v3** focused on
output compatibility, matrix-free performance, and differentiability.

.. figure:: _static/figures/sfincs_vs_sfincs_jax_l11_runtime_2x2.png
   :alt: Relative L11 difference and runtime comparison across four monoenergetic cases.
   :align: center
   :width: 90%

   Relative ``ΔL11`` (``(JAX − Fortran) / Fortran``) and runtime comparison for
   four monoenergetic fixtures. ``sfincs_jax`` runtime excludes JIT compilation
   (warm-up not timed). Reproduce with
   ``examples/performance/benchmark_transport_l11_vs_fortran.py``.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   normalizations
   method
   system_equations
   usage
   inputs
   outputs
   performance
   upstream_docs
   fortran_examples
   examples
   api
   parity
   references
   contributing
   release_checklist
