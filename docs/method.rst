Method overview
===============

`sfincs_jax` is a parity-first port of **SFINCS Fortran v3** to **JAX**.

SFINCS (v3) computes neoclassical transport in stellarators by solving a drift-kinetic
equation (DKE) for the non-adiabatic part of the distribution function on a flux surface.

Discretization (v3)
-------------------

SFINCS v3 uses a structured discretization that is well-suited to a JAX port:

- **Angles**: periodic grids in :math:`\\theta` and :math:`\\zeta` with finite-difference
  derivative matrices.
- **Speed**: a polynomial/Stieltjes grid in :math:`x` (a normalized speed-like coordinate),
  with quadrature weights used in moments/constraints.
- **Pitch angle**: a Legendre-mode expansion in :math:`\\xi = v_\\parallel / v`.

The primary unknown (in many modes) can be viewed as a tensor

.. math::

   f = f(s, x, L, \\theta, \\zeta),

where :math:`s` is the species index and :math:`L` is the Legendre index.

Why JAX?
--------

Porting to JAX enables:

- **JIT compilation** (CPU/GPU) of the operator application and solver kernels.
- **Automatic differentiation** through geometry, collision operators, and eventually the
  full kinetic solve (useful for sensitivity studies and gradient-based optimization).
- **Matrix-free linear algebra**: express the v3 Jacobian as a matvec rather than assembling
  sparse matrices, enabling scalable iterative solvers.
- An ecosystem of tools that become natural once the compute graph is differentiable:

  - `jaxopt` for implicit differentiation and robust root/linear solvers.
  - `optax` for gradient-based optimization loops (calibration, inverse problems).
  - `equinox` for clean, testable module organization and parameter handling.

Parity-first strategy
---------------------

The v3 codebase is large, so `sfincs_jax` is built in small parity-checked slices:

1. Port a well-scoped subsystem (grid, geometry piece, operator term).
2. Add a test that compares against frozen Fortran v3 reference data.
3. Only then expand functionality.

This approach keeps the port correct and refactorable while still moving quickly.

