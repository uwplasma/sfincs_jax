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

Linear algebra and preconditioning
----------------------------------

The solver stack in `sfincs_jax` draws on standard Krylov and preconditioning references:

- Y. Saad and M. Schultz, “GMRES: A generalized minimal residual algorithm for solving
  nonsymmetric linear systems,” *SIAM J. Sci. Stat. Comput.* 7(3), 1986.
- H. A. van der Vorst, “Bi-CGSTAB: A fast and smoothly converging variant of Bi-CG,”
  *SIAM J. Sci. Stat. Comput.* 13(2), 1992.
- P. Sonneveld and M. B. van Gijzen, “IDR(s): A family of simple and fast algorithms for
  solving large nonsymmetric systems of linear equations,” *SIAM J. Sci. Comput.* 31(2), 2008.
- M. A. Woodbury, “Inverting modified matrices,” *Statistical Research Group Memo Report*, 1950
  (Woodbury identity / low‑rank updates).
- G. H. Golub and C. F. Van Loan, *Matrix Computations*, 4th ed., Johns Hopkins Univ. Press, 2013
  (Schur complements, block factorization).
- M. de Sturler, “Truncation strategies for optimal Krylov subspace methods,”
  *SIAM J. Numer. Anal.* 36(3), 1999 (GCRO/deflation concepts).
- M. Benzi, “Preconditioning techniques for large linear systems: a survey,” *J. Comput. Phys.*
  182(2), 2002.

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
