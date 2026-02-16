Physics model and equations
===========================

`sfincs_jax` follows the **radially local drift-kinetic formulation** used in SFINCS v3.
This page summarizes the physics model, key terms, and how the model connects to the
discretized operators described in :doc:`system_equations` and :doc:`method`.
For full derivations, SFINCS v3 modeling notes, and a code-to-equation mapping, see
:doc:`physics_reference`.

Model overview
--------------

SFINCS evolves the non-adiabatic perturbation :math:`f_{s1}` about a Maxwellian
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

Geometry and guiding-center drifts
----------------------------------

The guiding-center drifts can be expressed (in physical variables) as

.. math::

   \mathbf{v}_m
   =
   \frac{v_\parallel^2}{\Omega_s}\,\mathbf{b}\times(\mathbf{b}\cdot\nabla\mathbf{b})
   + \frac{\mu}{\Omega_s}\,\mathbf{b}\times\nabla B,

and the :math:`E\times B` drift as

.. math::

   \mathbf{v}_E = \frac{c}{B^2}\,\mathbf{E}\times\mathbf{B}.

SFINCS evaluates the geometric coefficients using Boozer-like straight-field-line
coordinates (especially for ``geometryScheme=11/12``), and the discrete operator
uses those coefficients to build the drift terms in the DKE. [#boozer1980]_

Collision operators
-------------------

SFINCS supports two collision models:

- **Pitch-angle scattering (PAS)**: a diagonal-in-:math:`L` operator used for
  reduced models and benchmark suites.
- **Full linearized Fokker–Planck (Landau)**: implemented via Rosenbluth potentials
  and dense coupling in the speed coordinate. This is the default for high-fidelity
  multispecies studies.

The linearized FP operator is the most accurate model for neoclassical transport in
SFINCS and is the basis for the collision-driven preconditioners used in `sfincs_jax`.
[#sfincs2015]_

Phi1 and quasineutrality
------------------------

When ``includePhi1 = .true.``, SFINCS solves for the **flux-surface variation of the
electrostatic potential** :math:`\Phi_1(\theta,\zeta)` via a quasineutrality constraint.
The resulting potential modifies the kinetic equation and the collision operator
through poloidal density variations. This physics is especially important for impurity
transport and flows in stellarators. [#phi1_2018]_

Transport coefficients
----------------------

Transport-matrix modes (``RHSMode=2/3``) solve the same linearized DKE with multiple
right-hand sides and postprocess the solutions into particle/heat fluxes and FSAB
flows. These coefficients are the basis for neoclassical transport predictions and
bootstrap current calculations in SFINCS. [#sfincs2015]_

Implementation notes
--------------------

- Term-by-term input switches are documented in :doc:`system_equations`.
- Discretization details (Legendre modes, :math:`x` grid, angular finite differences)
  are summarized in :doc:`method`.
- Normalizations for all hat variables are listed in :doc:`normalizations`.

References
----------

.. [#sfincs2015] M. Mollen et al., “Implementation of a fully linearized Fokker–Planck
   collision operator in SFINCS,” arXiv:1504.04810 (2015).
.. [#boozer1980] A. H. Boozer, “Guiding center drift equations,” *Phys. Fluids* 23(5),
   904–908 (1980). OSTI: https://www.osti.gov/biblio/5655342
.. [#phi1_2018] M. Mollen et al., “Poloidal variation of impurity density and electric
   potential in stellarators,” *Plasma Phys. Control. Fusion* 60 (2018) 084001.
   OSTI: https://www.osti.gov/biblio/1473123
