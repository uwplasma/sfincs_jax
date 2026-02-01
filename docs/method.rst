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

Selected operator terms (ported so far)
--------------------------------------

The full v3 operator is large. `sfincs_jax` currently ports and parity-tests the following
building blocks.

Collisionless streaming + mirror (|ΔL|=1)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The basic collisionless dynamics couple neighboring Legendre modes :math:`L \\leftrightarrow L\\pm 1`
through (i) streaming along the field line and (ii) the mirror force. These terms are
parity-tested against frozen PETSc binaries.

Non-standard Er term in xiDot (|ΔL|=2)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

SFINCS v3 contains an additional, non-standard :math:`\\partial/\\partial\\xi` term associated with
the radial electric field. In the Legendre basis, this term has a diagonal-in-:math:`L` piece
and couples :math:`L \\leftrightarrow L\\pm 2`.

In the Fortran code (`populateMatrix.F90`), the coefficient for this term is

.. math::

   F_{\\xi}(\\theta,\\zeta) = \\frac{\\alpha\\,\\Delta\\,\\partial_{\\psi}\\hat\\Phi}{4\\,\\hat B^3}
   \\;\\hat D\\;\\Big( \\hat B_{\\zeta}\\,\\partial_{\\theta}\\hat B - \\hat B_{\\theta}\\,\\partial_{\\zeta}\\hat B \\Big),

where the hats denote v3-normalized quantities.

In the `geometryScheme=4` model used in tests, :math:`\\hat B_{\\theta}` is constant (and
is zero in the default W7-X parameter set), so the expression simplifies.

Collisionless Er xDot term (x-coupling and |ΔL|=2)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When `includeXDotTerm = .true.`, v3 includes a collisionless radial-derivative term in the
kinetic equation, discretized as a dense differentiation matrix in the :math:`x` coordinate.

The v3 implementation includes:

- Dense **x-matvec** using :math:`x\\,\\partial/\\partial x` (with optional upwinding schemes).
- Legendre couplings with :math:`L \\leftrightarrow L\\pm 2` (and a diagonal-in-:math:`L` piece).

In `sfincs_jax` we currently implement the default `xDotDerivativeScheme = 0`, i.e.
the same polynomial-grid differentiation matrix is used for both upwind directions.

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
