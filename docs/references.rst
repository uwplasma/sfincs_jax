References and related work
===========================

This page collects a few key references that inform the design and validation strategy of `sfincs_jax`.

SFINCS (upstream v3)
--------------------

The upstream SFINCS v3 paper and technical notes are vendored in ``docs/upstream/`` and linked from
``docs/upstream_docs.rst``.

MONKES and optimization-focused neoclassical workflows
------------------------------------------------------

- Escoto (PhD thesis): ``Escoto_Thesis.pdf`` (repo root).
- MONKES paper materials: see ``MONKES/doc/NF-paper/``.

The MONKES materials are particularly useful for:

- adjoint properties of drift-kinetic equations,
- derivative-aware workflows for optimization,
- and convergence/scaling studies that inform regression tests and benchmarks.

Recent applications (examples to prioritize)
--------------------------------------------

The following papers motivate parity targets and gradient-based examples:

- “Recent progress on neoclassical impurity transport in stellarators with implications for a stellarator reactor”
  (Nucl. Fusion, 2021).
- “Electron root optimisation for stellarator reactor designs” (arXiv:2405.12058, 2024).

