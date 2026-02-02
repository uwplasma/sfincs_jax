Normalizations and units
========================

SFINCS (v3) uses a mixture of *physical* quantities (e.g. :math:`n_s`, :math:`T_s`) and
dimensionless *normalized* quantities (often written with hats in the code and documentation,
e.g. ``BHat`` or :math:`\hat B`). This page summarizes the key normalizations used in the v3
formulation that `sfincs_jax` is porting.

If you need the full upstream derivations and the complete set of equations, see the vendored
LaTeX sources and PDFs linked from :doc:`upstream_docs`.

Velocity-space coordinates
--------------------------

SFINCS solves the drift-kinetic equation on the 4D phase space
:math:`(\theta,\zeta,x,\xi)`:

- :math:`\theta`, :math:`\zeta` are Boozer angles (or appropriate substitutes depending on the
  geometry scheme).
- :math:`x` is a normalized speed.
- :math:`\xi = v_\parallel / v` is the pitch-angle cosine.

In the upstream v3 documentation (see ``docs/upstream/manual/version3/equations.tex``),
the normalized speed is defined as

.. math::

   x_s = \frac{v}{\sqrt{2 T_s / m_s}}.

In practice, SFINCS discretizes in :math:`x` on a finite interval, typically :math:`0 \le x \le x_{\max}`.

Distribution function split
---------------------------

SFINCS writes the distribution function as

.. math::

   f_s = f_{s0} + f_{s1},

where :math:`f_{s0}` is a Maxwellian (or a Maxwellian modified by :math:`\Phi_1` when enabled) and
:math:`f_{s1}` is the unknown first-order correction solved for by the drift-kinetic equation.

Flux-surface averages
---------------------

Many constraints and diagnostics use the flux-surface average

.. math::

   \langle g \rangle
   =
   \frac{\int d\theta\,d\zeta \; g(\theta,\zeta)\; \hat D(\theta,\zeta)^{-1}}
        {\int d\theta\,d\zeta \; \hat D(\theta,\zeta)^{-1}},

where the v3 discretization uses the quadrature weights ``thetaWeights`` and ``zetaWeights`` and the
geometry factor ``DHat``.

Radial electric field convention
--------------------------------

In v3, the radial electric field is specified via ``Er`` and is related to the flux-function potential
gradient by

.. math::

   E_r = -\frac{d\Phi_0}{dr}.

Several operator coefficients in the v3 collisionless terms are written in terms of
:math:`d\hat\Phi/d\hat\psi` (see :doc:`method`), and the exact conversion depends on the chosen radial
coordinate conventions. For `geometryScheme=4`, `sfincs_jax` matches the v3 defaults used in the
frozen parity fixtures.

