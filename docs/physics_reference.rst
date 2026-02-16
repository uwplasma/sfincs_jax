Physics reference and code map
==============================

This page consolidates the upstream SFINCS v3 notes (``docs/upstream``) into a single,
code-linked reference. It is intentionally long-form: the goal is to make the physics,
approximations, and numerics navigable without jumping between PDFs.

Governing drift-kinetic equation
--------------------------------

SFINCS v3 starts from the steady-state, radially local drift-kinetic equation for the
gyro-averaged distribution function :math:`f_s` (Hazeltine 1973; see the v3 technical note
``20150507-01 Technical documentation for version 3 of SFINCS.pdf``). We split

.. math::

   f_s = f_{s0} + f_{s1},

where :math:`f_{s0}` is a Maxwellian (possibly modified by :math:`\Phi_1`) and
:math:`f_{s1}` is the unknown correction.

In the variables used by v3,

.. math::

   x_s = \frac{v}{\sqrt{2 T_s / m_s}}, \qquad
   \xi = \frac{v_\parallel}{v},

the linearized DKE can be written schematically as

.. math::

   v_\parallel \mathbf{b}\cdot\nabla f_{s1}
   + \mathbf{v}_E\cdot\nabla f_{s1}
   + \mathbf{v}_m\cdot\nabla f_{s1}
   + \dot{x}\,\partial_x f_{s1}
   + \dot{\xi}\,\partial_\xi f_{s1}
   - \sum_b C^{\mathrm{lin}}_{sb}[f_{b1}]
   = S_s.

The source term :math:`S_s` contains thermodynamic drives, inductive-field drives
(``EParallelHat`` in v3), and (for transport-matrix modes) the internal ``whichRHS`` forcing.

Upstream v3 documentation defines the conserved quantities

.. math::

   W_s = \frac{m_s v^2}{2} + Z_s e \Phi,
   \qquad
   \mu = \frac{v_\perp^2}{2 B},
   \qquad
   \Omega_s = \frac{Z_s e B}{m_s},

and uses the standard :math:`E\times B` drift

.. math::

   \mathbf{v}_E = \frac{c}{B^2}\,\mathbf{E}\times\mathbf{B},
   \qquad
   \mathbf{E} = -\nabla\Phi_0.

Code links:
``sfincs_jax/v3_system.py`` (operator assembly),
``sfincs_jax/residual.py`` (source terms and residuals),
``sfincs_jax/transport_matrix.py`` (RHSMode=2/3 forcing).

Single-species baseline (20131220-04)
-------------------------------------

The single-species technical note derives the normalized variables and driving terms used
in SFINCS v3. With

.. math::

   v_{\mathrm{th}} = \sqrt{\frac{2 T}{m}}, \qquad
   x = \frac{v}{v_{\mathrm{th}}}, \qquad
   \xi = \frac{v_\parallel}{v}, \qquad
   \mu = \frac{v_\perp^2}{2 B},

the Maxwellian is written as

.. math::

   f_M = n \left(\frac{m}{2 \pi T}\right)^{3/2} \exp(-x^2).

The normalization parameters introduced in the note are

.. math::

   \Delta = \frac{v_{\mathrm{th}}}{\Omega R}, \qquad
   \alpha = \frac{e \Phi}{T}, \qquad
   \nu_n = \frac{\nu R}{v_{\mathrm{th}}},

where :math:`R` is a reference major radius and :math:`\Omega = Z e B/(m c)` is the
gyrofrequency. These correspond to ``Delta``, ``alpha``, and ``nu_n`` in the input
namelist and are stored in the output HDF5 file by `sfincs_jax/io.py`.

The same note writes the thermodynamic drive in the compact form

.. math::

   (\mathbf{v}_m + \mathbf{v}_E)\cdot\nabla r
   \left[\frac{1}{n}\frac{dn}{dr}
   + \frac{Z e}{T}\frac{d\Phi_0}{dr}
   + \left(x^2-\frac{3}{2}\right)\frac{1}{T}\frac{dT}{dr}\right] f_0,

with additional :math:`\Phi_1`-dependent pieces if flux-surface variation is enabled.
In `sfincs_jax`, these drive terms are assembled in
``sfincs_jax/residual.py`` and combined with the transport-matrix forcing in
``sfincs_jax/transport_matrix.py``.

Multi-species extension (20131219-01)
-------------------------------------

The multi-species note generalizes the above to species-indexed quantities
(:math:`m_s`, :math:`T_s`, :math:`Z_s`, :math:`n_s`) and introduces the
linearized collision operator

.. math::

   C_s^{\mathrm{lin}}[f_{1}] = \sum_b C_{sb}^{\mathrm{lin}}[f_{s1}, f_{b1}],

with test-particle and field-particle pieces that couple the Legendre modes across
species. This coupling is what makes the Fokker–Planck operator dense in speed space.
`sfincs_jax` mirrors the v3 block structure in
``sfincs_jax/collisions.py`` and assembles multi-species blocks in
``sfincs_jax/v3_system.py``.

Numerical implications:

- The dense species coupling in the full FP operator is the dominant memory and runtime
  cost in multi-species runs.
- The PAS operator keeps only the pitch-angle scattering term, resulting in a block-diagonal
  coupling that can be preconditioned independently for each species.

Guiding-center drifts and trajectory models
-------------------------------------------

SFINCS v3 supports multiple trajectory models, summarized in the SFINCS paper
(``docs/upstream/sfincsPaper/sfincsPaper.pdf``):

- **Full trajectories**: retain the complete :math:`\mathbf{v}_E` and magnetic-drift
  contributions in both :math:`\dot{\mathbf{r}}` and :math:`\dot{\xi}`.
- **Partial trajectories**: retain the dominant drift pieces but drop smaller terms
  that complicate the conservation properties.
- **DKES trajectories**: match the DKES form for benchmark comparisons and
  monoenergetic transport coefficients.

The magnetic drift used in v3 follows the two equivalent forms summarized in the v3
technical note (magnetic drift option ``\sigma_{\mathrm{mdo}}`` in
``20150507-01 Technical documentation for version 3 of SFINCS.pdf``):

.. math::

   \mathbf{v}_m
   =
   \frac{m v_\parallel^2}{Z e B}
   \mathbf{b}\times(\mathbf{b}\cdot\nabla\mathbf{b})
   + \frac{\mu}{Z e B}\,\mathbf{b}\times\nabla B
   + \sigma_{\mathrm{mdo}}
   \frac{m v_\parallel}{Z e B}\,(\mathbf{b}\cdot\nabla\times\mathbf{b})\,\mathbf{b}\times\mathbf{b}.

Numerically, `sfincs_jax` uses the same coefficient forms as v3 for
``magneticDriftScheme=1`` and builds the corresponding angular derivative operators in
``sfincs_jax/magnetic_drifts.py``.

Code links:
``sfincs_jax/magnetic_drifts.py``,
``sfincs_jax/collisionless.py``,
``sfincs_jax/collisionless_exb.py``.

Collision operators (PAS and full Fokker–Planck)
------------------------------------------------

SFINCS v3 implements two collision models:

- **Pitch-angle scattering (PAS)**, diagonal in Legendre mode :math:`L`, used for
  reduced benchmarks and DKES-like comparisons.
- **Full linearized Fokker–Planck (Landau)**, implemented with Rosenbluth potentials
  and dense coupling in the speed grid.

The v3 FP implementation is described in
``20150402-01 Implementation of the Fokker-Planck operator.pdf``. In brief:

1. The distribution on the collocation speed grid is transformed to a modal basis
   via the matrix

   .. math::

      Y_{mn} = \frac{w_n x_n^k M_m(x_n)}{A_m},

   where :math:`M_m` are the orthogonal polynomials, :math:`w_n` are quadrature weights,
   and :math:`k` is the ``xGrid_k`` exponent.
2. Rosenbluth potentials and their derivatives are expressed as definite integrals in
   the modal basis, evaluated for each species and Legendre mode.
3. The resulting field-particle terms are mapped back to the collocation grid and
   combined with the test-particle contributions.

The single- and multi-species notes emphasize that the linearized operator must conserve
particles, momentum, and energy. In practice this means the field-particle terms are
constructed to exactly cancel the moment losses of the test-particle operator, which is
why the FP block is dense in :math:`x`. `sfincs_jax` mirrors the v3 moment conservation
strategy in ``sfincs_jax/collisions.py`` (see the field-particle assembly helpers).

Code links:
``sfincs_jax/grids.py`` (polynomial grids and quadrature),
``sfincs_jax/collisions.py`` (PAS and FP operators),
``sfincs_jax/xgrid.py`` (collocation-to-modal transforms).

Phi1 and quasineutrality
------------------------

When ``includePhi1 = .true.``, the electrostatic potential is decomposed as

.. math::

   \Phi(\psi,\theta,\zeta) = \Phi_0(\psi) + \Phi_1(\theta,\zeta).

The Phi1 implementation notes (``Phi1_implementation_2016-01.tex`` and
``20150325-01 Effects on fluxes of including Phi_1.pdf``) show that the Maxwellian is
modified by

.. math::

   f_{s0} = f_{sM}\,\exp\!\left(-\frac{Z_s e \Phi_1}{T_s}\right),

and that the drift-kinetic RHS acquires additional :math:`\Phi_1`-dependent terms from
the :math:`\nabla\Phi_1` forces and from the background-gradient drives.

Numerical challenges include:

- the exponential dependence of :math:`f_{s0}` on :math:`\Phi_1`,
- the need to preserve flux-surface average constraints
  (:math:`\langle \Phi_1 \rangle = 0`),
- and the coupling of Phi1 into collision coefficients
  (``includePhi1InCollisionOperator``).

Code links:
``sfincs_jax/v3_system.py`` (Phi1 block),
``sfincs_jax/io.py`` (Phi1 input handling),
``sfincs_jax/collisions.py`` (Phi1-in-collisions),
``sfincs_jax/diagnostics.py`` (Phi1 output fields).

Phi1 impact on flux definitions (20150325-01)
---------------------------------------------

The Phi1 flux-impact note derives the relationship between the particle/energy fluxes
computed with and without :math:`\Phi_1`. Denote by :math:`\tilde{f}_{s1}` the solution
of the kinetic equation with the parallel :math:`\nabla_{||}\Phi_1` force included. The
note shows

.. math::

   \tilde{f}_{s1} = f_{s1} - \frac{Z_s e}{T_s}\Phi_1 f_{sM},

and derives the flux identities (schematically)

.. math::

   \Gamma_s^{(\mathrm{mag})} + \Gamma_s^{(E\times B)} = \tilde{\Gamma}_s,
   \qquad
   Q_{s,\mathrm{tot}} = Q_s + Z_s e \Phi_1 \Gamma_s,

which imply that the *physically relevant* particle and total-energy fluxes are unchanged
to leading order when :math:`\Phi_1` is included, provided the additional
:math:`E\times B` contributions are accounted for. This is the rationale for the
separate ``*_vm0`` and ``*_vE0`` diagnostic pieces in `sfincs_jax/diagnostics.py` and
the combined flux outputs stored in ``sfincsOutput.h5``.

Transport matrix and Beidler notation
--------------------------------------

For transport-matrix modes (``RHSMode=2/3``), SFINCS defines a matrix :math:`L_{ij}` that
relates fluxes to thermodynamic forces. The note
``20131206-02 Relating sfincs transport matrix to Beidler matrix.pdf`` gives explicit
relations, e.g.

.. math::

   \Gamma\cdot\nabla\psi
   =
   -L_{11}\left(\frac{1}{n}\frac{dn}{d\psi}
   + \frac{Z e}{T}\frac{d\Phi_0}{d\psi}
   - \frac{3}{2}\frac{1}{T}\frac{dT}{d\psi}\right)
   - L_{12}\frac{1}{T}\frac{dT}{d\psi}
   + L_{13}\frac{E_\parallel}{B},

and analogous expressions for the heat flux :math:`q\cdot\nabla\psi`.

Code links:
``sfincs_jax/transport_matrix.py`` (RHS generation),
``sfincs_jax/diagnostics.py`` (flux and flow diagnostics),
``sfincs_jax/compare.py`` (transport-matrix parity handling).

Single- vs multi-species normalization
--------------------------------------

The note ``20131003-02 Relating quantities in the 1-species and multi-species SFINCS.pdf``
derives conversion factors between the single-species transport coefficients and the
fully multi-species formulation. This is especially relevant for benchmarking against
legacy single-species results and for interpreting monoenergetic (``RHSMode=3``) runs.

Code links:
``sfincs_jax/diagnostics.py`` (normalization of flux outputs),
``sfincs_jax/transport_matrix.py`` (multi-RHS assembly).

Monoenergetic control parameters (nuPrime, EStar)
-------------------------------------------------

For ``RHSMode=3`` (monoenergetic/DKES-style runs), the SFINCS manual defines the
dimensionless collisionality and electric field as

.. math::

   \nu' = \frac{(G + \iota I)\,\nu}{v B_0},
   \qquad
   E^\* = \frac{c G}{\iota v B_0}\frac{d\Phi}{d\psi},

where :math:`G` and :math:`I` are Boozer covariant components, :math:`\iota` is the
rotational transform, and :math:`B_0` is the :math:`(0,0)` Fourier mode of :math:`B`.
The SFINCS manual further notes that, for these runs, the input parameters
``nu_n`` and ``dPhiHatdpsiHat`` are ignored and replaced by ``nuPrime``/``EStar``.

In `sfincs_jax`, the mapping between ``nuPrime``/``EStar`` and the internal
``nu_n``/``dPhiHatdpsiHat`` parameters is handled in ``sfincs_jax/io.py`` and
``sfincs_jax/v3_fblock.py``, while geometry-dependent factors
(:math:`B_0`, :math:`G`, :math:`I`) are computed in ``sfincs_jax/transport_matrix.py``
and ``sfincs_jax/diagnostics.py``.

Constraint schemes and source terms
-----------------------------------

SFINCS adds source terms and constraints to remove nullspaces and enforce solvability.
The v3 notes (``20150507-01`` and ``20131220-04``) describe several schemes, including:

- **constraintScheme=1**: explicit density/pressure constraints using moment conditions.
- **constraintScheme=2**: augment the system with source unknowns and enforce
  flux-surface-averaged constraints (common with PAS collisions).

Numerically, these constraints introduce near-nullspaces that require explicit projection
or block preconditioning to solve robustly.

Code links:
``sfincs_jax/v3_system.py`` (constraint rows/columns),
``sfincs_jax/v3_driver.py`` (constraint projections and preconditioners),
``sfincs_jax/solver.py`` (nullspace-aware diagnostics).

Classical radial fluxes
-----------------------

The classical transport notes (``classical_radial_fluxes_2019-01-17.pdf``) derive
classical particle and heat fluxes from the linearized collision operator.
Key ingredients include the gyrophase-dependent part of :math:`f_{a1}`:

.. math::

   \tilde{f}_{a1}
   = -\boldsymbol{\rho}_a\cdot\nabla f_{Ma},
   \qquad
   \boldsymbol{\rho}_a = \frac{1}{\Omega_a}\,\mathbf{b}\times\mathbf{v}_\perp,

and classical flux definitions

.. math::

   \Gamma_a^{\mathrm{C}} = \left\langle
   \frac{\mathbf{b}\times\nabla\psi}{Z_a e B}\cdot\mathbf{R}_a\right\rangle,
   \qquad
   Q_a^{\mathrm{C}} = \left\langle
   \frac{\mathbf{b}\times\nabla\psi}{Z_a e B}\cdot\mathbf{G}_a\right\rangle,

with :math:`\mathbf{R}_a` and :math:`\mathbf{G}_a` the friction and energy-weighted
friction forces defined from :math:`C[f_a]`.

Code links:
``sfincs_jax/classical_transport.py``.

DKES compatibility notes
------------------------

The DKES notes (``notes_dkes_sfincs.pdf``) write the DKES equation in the form

.. math::

   (\tilde{C} - \tilde{V})\,\hat{f}_i = S_i,

with pitch-angle scattering and Vlasov terms. SFINCS reproduces this model
by choosing the DKES trajectory model and PAS collisions. This is the basis for the
monoenergetic transport-matrix benchmarking in ``RHSMode=3``.

Code links:
``sfincs_jax/v3_system.py`` (trajectory switches),
``sfincs_jax/transport_matrix.py`` (monoenergetic RHS construction).

Equation-to-code map
--------------------

The table below summarizes where each term in the v3 drift-kinetic equation is implemented:

- Parallel streaming :math:`v_\parallel \mathbf{b}\cdot\nabla f_{s1}`:
  ``sfincs_jax/collisionless.py``.
- :math:`E\times B` advection and associated drive terms:
  ``sfincs_jax/collisionless_exb.py`` and ``sfincs_jax/collisionless_er.py``.
- Magnetic drifts (:math:`\mathbf{v}_m` terms and derivative couplings):
  ``sfincs_jax/magnetic_drifts.py``.
- Pitch-angle and speed derivatives (:math:`\partial_\xi f`, :math:`\partial_x f`):
  ``sfincs_jax/collisionless.py`` and ``sfincs_jax/collisionless_er.py``.
- Collision operators (PAS and full FP):
  ``sfincs_jax/collisions.py`` with modal transforms in ``sfincs_jax/xgrid.py``.
- Constraint rows/columns and Phi1 blocks:
  ``sfincs_jax/v3_system.py`` and ``sfincs_jax/v3_driver.py``.
- Diagnostics and flux assembly:
  ``sfincs_jax/diagnostics.py`` and ``sfincs_jax/transport_matrix.py``.

Numerical implementation notes
------------------------------

The SFINCS v3 discretization is stiff: it couples dense x-space operators with sparse
angular derivatives, and introduces near-nullspaces from constraints.
In `sfincs_jax`, the main numerical strategies are:

- **Matrix-free operator application** to avoid assembling the full dense system.
- **Short-recurrence Krylov solvers** (BiCGStab / IDR(s)) with GMRES fallback for
  challenging cases.
- **Block preconditioning** (collision-diagonal, constraint-aware Schur complements).
- **Explicit nullspace projections** for constraintScheme=1/2 to preserve solvability.

Additional implementation details relevant to stability and performance:

- Angular derivatives are discretized with centered finite-difference stencils in
  ``sfincs_jax/derivative_matrix.py`` and assembled into sparse operators.
- Pitch-angle dependence uses Legendre modes with sparse :math:`\Delta L=\pm 1, \pm 2`
  coupling, while the full FP operator introduces dense coupling in the speed grid.
- Constraint schemes introduce near-nullspaces; the code therefore applies explicit
  projection steps and (optionally) Schur-complement preconditioners to avoid solver
  stagnation.

The Krylov methods and preconditioning choices follow standard references
(GMRES: Saad & Schultz 1986, BiCGStab: van der Vorst 1992, IDR(s): Sonneveld & van Gijzen 2008,
preconditioning survey: Benzi 2002). See :doc:`references` for the citation list.

Code links:
``sfincs_jax/solver.py`` (Krylov wrappers),
``sfincs_jax/v3_driver.py`` (preconditioners and projections).

References (vendored)
---------------------

- ``docs/upstream/20150507-01 Technical documentation for version 3 of SFINCS.pdf``
- ``docs/upstream/20131220-04 Technical documentation for SFINCS with a single species.pdf``
- ``docs/upstream/20131219-01 Technical documentation for SFINCS with multiple species.pdf``
- ``docs/upstream/20150402-01 Implementation of the Fokker-Planck operator.pdf``
- ``docs/upstream/20150325-01 Effects on fluxes of including Phi_1.pdf``
- ``docs/upstream/Phi1_implementation_2016-01.tex``
- ``docs/upstream/20131206-02 Relating sfincs transport matrix to Beidler matrix.pdf``
- ``docs/upstream/20131003-02 Relating quantities in the 1-species and multi-species SFINCS.pdf``
- ``docs/upstream/classical_radial_fluxes_2019-01-17.pdf``
- ``docs/upstream/notes_dkes_sfincs.pdf``
