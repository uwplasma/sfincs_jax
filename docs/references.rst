References and related work
===========================

This page collects a few key references that inform the design and validation strategy of `sfincs_jax`.

JAX / autodiff tools
--------------------

For implicit differentiation through linear solves (and other solver-aware workflows), see:

- JAX docs: ``jax.lax.custom_linear_solve``.
- JAX docs: ``jax.linear_transpose``.
- JAX docs: ``jax.scipy.sparse.linalg.gmres`` and ``jax.scipy.sparse.linalg.cg``.

SFINCS (upstream v3)
--------------------

The upstream SFINCS v3 paper and technical notes are vendored in ``docs/upstream/`` and linked from
``docs/upstream_docs.rst``.

MONKES and optimization-focused neoclassical workflows
------------------------------------------------------

The MONKES ecosystem and related thesis/paper materials (external to this repository) are useful for:

- adjoint properties of drift-kinetic equations,
- derivative-aware workflows for optimization,
- and convergence/scaling studies that inform regression tests and benchmarks.

Recent applications (examples to prioritize)
--------------------------------------------

The following papers motivate parity targets and gradient-based examples:

- “Recent progress on neoclassical impurity transport in stellarators with implications for a stellarator reactor”
  (Nucl. Fusion / PPCF, 2021).
- “Electron root optimisation for stellarator reactor designs” (arXiv:2405.12058, 2024).
