System of equations (v3)
========================

In its most general configuration, SFINCS v3 solves a coupled system consisting of:

1) a drift-kinetic equation (DKE) for each kinetic species,
2) an optional quasineutrality equation for the flux-surface variation of the electrostatic potential
   :math:`\Phi_1(\theta,\zeta)`,
3) auxiliary constraints that remove nullspaces and enforce moment conditions.

This page summarizes the block structure and the equations most relevant to the *linear system* that
`sfincs_jax` is assembling matrix-free.

For full upstream context and derivations, see the vendored v3 manual and technical docs in
``docs/upstream/`` (linked from :doc:`upstream_docs`).

Unknown ordering (PETSc / indices.F90)
--------------------------------------

The Fortran v3 code defines a global ordering for the state vector and the rows/columns of the master
matrix in ``sfincs/fortran/version3/indices.F90``.

For the common ``readExternalPhi1 = .false.`` case:

- The **F-block** (distribution function) is ordered first, with indices nested as:

  ``species → x → xi/Legendre mode → theta → zeta``.

- If ``includePhi1 = .true.``, the **Phi1 block** (labeled ``BLOCK_QN`` in Fortran) contributes
  :math:`N_\theta N_\zeta` additional unknowns corresponding to :math:`\Phi_1(\theta,\zeta)` on the grid.

- A final scalar unknown ``lambda`` enforces the constraint :math:`\langle \Phi_1 \rangle = 0`.

- Constraint-scheme-dependent source unknowns (and their corresponding constraint rows) are appended last.

Quasineutrality and Phi1 constraint
-----------------------------------

In the upstream v3 manual (see ``docs/upstream/manual/version3/equations.tex``), the coupled system includes
a quasineutrality condition and the constraint :math:`\langle \Phi_1 \rangle = 0`:

.. math::

   \lambda + \sum_s Z_s \int d^3v\; (f_{s0} + f_{s1}) = 0,
   \qquad
   \langle \Phi_1 \rangle = 0.

In the fully nonlinear v3 configuration, the quasineutrality equation contains additional
nonlinear dependence through :math:`f_{s0}(\Phi_1)` and (depending on options) adiabatic responses.

Current `sfincs_jax` status
---------------------------

`sfincs_jax` currently supports the **linear** QN/lambda block needed for matrix-free operator application,
but does not yet include the full set of Phi1-coupling terms inside the kinetic equation and collision
operator.

This is sufficient to represent v3 configurations in which Phi1 is present as an auxiliary solve but
does not enter the kinetic operator (and to support block-wise fixture validation).
