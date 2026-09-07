Physics model and equations
===========================

`dkx` solves a **radially local, steady-state drift-kinetic problem** on a
single flux surface. This page summarizes the governing model, the main approximations,
and how the physics terms map onto the discretized operators documented in
:doc:`system_equations`, :doc:`geometry`, and :doc:`numerics`.
For full derivations and code-location detail, see :doc:`physics_reference` and
:doc:`source_map`.

Model overview
--------------

`dkx` evolves the non-adiabatic perturbation :math:`f_{s1}` about a Maxwellian
background :math:`f_{s0}` on a single flux surface:

.. math::

   f_s = f_{s0} + f_{s1}.

In normalized variables, the linearized drift-kinetic equation (DKE) can be written as

.. math::

   \mathcal{L}_s[f_{s1}] = S_s,

where the operator :math:`\mathcal{L}_s` includes streaming, mirror force,
:math:`E\times B` drifts, magnetic drifts, energy/pitch-angle drift terms, and the
linearized collision operator. The source :math:`S_s` contains thermodynamic drives,
the inductive electric field, and the RHSMode-specific forcing used in transport
matrix calculations. [#sfincs2015]_

Background distribution and normalized variables
------------------------------------------------

The code uses the normalized variables

.. math::

   x_s = \frac{v}{\sqrt{2T_s/m_s}},
   \qquad
   \xi = \frac{v_\parallel}{v},
   \qquad
   \mu = \frac{v_\perp^2}{2B},

with a background state

.. math::

   f_{s0} =
   n_s \left(\frac{m_s}{2\pi T_s}\right)^{3/2}
   \exp(-x_s^2)

or, when the flux-surface-varying potential is included,

.. math::

   f_{s0} =
   f_{sM}\exp\!\left(-\frac{Z_s e \Phi_1}{T_s}\right).

The normalization conventions used for hats and dimensionless drives are summarized in
:doc:`normalizations`. They set the coefficients assembled by the consolidated
:class:`dkx.drift_kinetic.KineticOperator` and the diagnostics written by
:mod:`dkx.writer`.

Geometry and guiding-center drifts
----------------------------------

The guiding-center drifts can be expressed in SI units, with signed
:math:`\Omega_s = Z_s e B/m_s`, as

.. math::

   \mathbf{v}_m
   =
   \frac{v_\parallel^2}{\Omega_s}\,\mathbf{b}\times(\mathbf{b}\cdot\nabla\mathbf{b})
   + \frac{\mu}{\Omega_s}\,\mathbf{b}\times\nabla B,

and the :math:`E\times B` drift as

.. math::

   \mathbf{v}_E = \frac{1}{B^2}\,\mathbf{E}\times\mathbf{B}.

SFINCS evaluates the geometric coefficients using Boozer-like straight-field-line
coordinates (especially for ``geometryScheme=11/12``), and the discrete operator
uses those coefficients to build the drift terms in the DKE. [#boozer1980]_

Collision operators
-------------------

SFINCS Fortran v3 supports two collision models, and `dkx` adds a third:

- **Pitch-angle scattering (PAS)** (``collisionOperator = 1``): a
  diagonal-in-:math:`L` operator used for reduced models and benchmark suites.
- **Full linearized Fokker–Planck (Landau)** (``collisionOperator = 0``):
  implemented via Rosenbluth potentials
  and dense coupling in the speed coordinate. This is the default for
  multispecies studies.
- **Improved Sugama model operator** (``collisionOperator = 3``): the momentum-
  and energy-conserving improved linearized model operator of Sugama *et al.*
  (2019), a `dkx` research extension beyond Fortran v3 that remains
  accurate into the highly collisional regime.

The linearized FP operator is also the basis for the collision-driven
preconditioners used in `dkx`. [#sfincs2015]_

Constraint closure and source/sink terms
----------------------------------------

The discrete linear system is closed by auxiliary constraints. In the common linear
formulation these are equivalent to enforcing

.. math::

   \left\langle \int d^3v \, f_{s1} \right\rangle = 0,
   \qquad
   \left\langle \int d^3v \, v^2 f_{s1} \right\rangle = 0,

with optional quasineutrality and gauge conditions when :math:`\Phi_1` is solved:

.. math::

   \lambda + \sum_s Z_s \int d^3v \, (f_{s0}+f_{s1}) = 0,
   \qquad
   \langle \Phi_1 \rangle = 0.

These conditions remove nullspaces associated with conservation laws and determine the
algebraic branch selected by the solve.

Phi1 and quasineutrality
------------------------

When ``includePhi1 = .true.``, SFINCS solves for the **flux-surface variation of the
electrostatic potential** :math:`\Phi_1(\theta,\zeta)` via a quasineutrality constraint.
The resulting potential modifies the kinetic equation and the collision operator
through poloidal density variations. This physics is important for impurity
transport and flows in stellarators. [#phi1_2018]_

Transport coefficients
----------------------

Transport-matrix modes (``RHSMode=2/3``) solve the same linearized DKE with multiple
right-hand sides and postprocess the solutions into particle/heat fluxes and FSAB
flows. These coefficients are the basis for neoclassical transport predictions and
bootstrap current calculations in SFINCS. [#sfincs2015]_

The main moments of interest are built from velocity-space and flux-surface integrals,
schematically

.. math::

   \Gamma_s = \left\langle \int d^3v \, f_{s1}\,\mathbf{v}_{d,s}\cdot\nabla r \right\rangle,

.. math::

   Q_s = \left\langle \int d^3v \, \frac{m_s v^2}{2} f_{s1}\,\mathbf{v}_{d,s}\cdot\nabla r \right\rangle,

.. math::

   V_{\parallel,s} = \left\langle \int d^3v \, v_\parallel f_{s1} \right\rangle,
   \qquad
   j_\parallel = \sum_s Z_s e V_{\parallel,s}.

`dkx` evaluates these moments in :mod:`dkx.moments` (with
geometry-derived scalars in :mod:`dkx.magnetic_geometry`).

Trajectory-model knobs
----------------------

The trajectory model is configurable through these switches:

- ``magneticDriftScheme`` for magnetic-drift terms,
- ``useDKESExBDrift`` for DKES-like vs full :math:`E\times B` advection,
- ``includeXDotTerm`` and the corresponding :math:`\dot \xi` electric-field term,
- ``collisionOperator`` for PAS vs full FP,
- ``includePhi1`` / ``includePhi1InKineticEquation`` /
  ``includePhi1InCollisionOperator`` for flux-surface-varying electrostatic physics.

The detailed switch-to-equation map is given in :doc:`system_equations`.

Related reference pages
-----------------------

- Term-by-term input switches: :doc:`system_equations`.
- Discretization details (Legendre modes, :math:`x` grid, angular finite
  differences): :doc:`method` and :doc:`numerics`.
- Normalizations for all hat variables: :doc:`normalizations`.
- Source-file locations for the main operators: :doc:`source_map`.

References
----------

.. [#sfincs2015] M. Mollén et al., “Implementation of a fully linearized Fokker–Planck
   collision operator in SFINCS,” arXiv:1504.04810 (2015).
.. [#boozer1980] A. H. Boozer, “Guiding center drift equations,” *Phys. Fluids* 23(5),
   904–908 (1980). OSTI: https://www.osti.gov/biblio/5655342
.. [#phi1_2018] M. Mollén et al., “Poloidal variation of impurity density and electric
   potential in stellarators,” *Plasma Phys. Control. Fusion* 60 (2018) 084001.
   OSTI: https://www.osti.gov/biblio/1473123
