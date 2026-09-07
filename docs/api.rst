API reference
=============

The stable public facade is :mod:`dkx.api` (re-exported at the package
top level). The canonical modules that implement the physics and solver stack
are indexed below, with links to the pages that document them in depth and to
:doc:`source_map` for the full source catalogue.

Public facade
-------------

The high-level input contract is :class:`dkx.Case`; its typed
submodels and serializers live in :mod:`dkx.config`.

.. automodule:: dkx.config
   :members: Case, CaseValidationError, RunConfig, GeometryConfig, SpeciesConfig, PhysicsConfig, ElectricFieldConfig, ResolutionConfig, SolverConfig, ParallelConfig, ConvergenceConfig, OutputConfig, ScanAxis, ScanConfig, case_json_schema

The result contract is :class:`dkx.Result`.

.. automodule:: dkx.result
   :members: Result

.. automodule:: dkx.api
   :members:

High-level runners live in :mod:`dkx.run` (``run_profile``,
``run_transport_matrix``) and the CLI in :mod:`dkx.cli`; see :doc:`usage`.

Canonical modules
-----------------

For expert staged solves, ``KineticOperator`` is a JAX pytree. Discrete
dimensions, model switches and the ``n_xi_for_x`` pitch-truncation layout are
static compilation keys; physical coefficient arrays are dynamic leaves.
A change to active pitch layout requires a new trace even when the rectangular
state shape is unchanged. Same-layout operators with changed coefficients can
reuse an executable on supported staged solver routes.

Supply a consistently rebuilt operator when changing density, temperature or
geometry. Replacing only ``n_hat`` or ``t_hat`` on an existing operator does not
rebuild its collision coefficients. Compilation reuse is distinct from valid
physical-data or preconditioner reuse; it does not establish a prepared restart
contract or make the namelist/geometry builders differentiable.

For full-FP density scans at fixed temperatures, masses, charges and speed
grid, ``make_fokker_planck_v3_phi1_operator`` already stores unit-density
collision kernels. Its ``at_uniform_density(n_hats, n_xi=...)`` method
assembles a regular ``FokkerPlanckV3Operator`` using JAX tensor operations,
including energy scattering, pitch scattering and field-particle terms.
The refresh supports density JVPs/VJPs without repeating Rosenbluth integrals.
For an ordinary no-Phi1 full-FP operator and matching kernels:

.. code-block:: python

   from dataclasses import replace
   from dkx.solve import solve

   refreshed = replace(
       operator,
       n_hat=new_density,
       fp=kernels.at_uniform_density(new_density, n_xi=operator.n_xi),
   )
   # Recompute the drive as well as the collision response.
   result = solve(refreshed, refreshed.rhs(), differentiable=True)

Here ``kernels`` must have been built with the operator's collision/grid
settings. Keep profile-gradient inputs consistent with the intended scan;
the snippet holds them fixed. This refresh does not differentiate temperature,
change the Coulomb-logarithm prescription, or certify preconditioner reuse.
Changing a kernel dependency requires rebuilding the kernels.

The two-species regression checks normalized current and first-species heat-flux
density derivatives against independently rebuilt full-FP operators, central
differences and second-order Taylor remainders on CPU and GPU. This bounded
fixed-geometry, fixed-temperature test is not a grid-convergence certificate.
Example 07 separately teaches a single-species PAS temperature response,
rebuilding its pitch-angle coefficients at each temperature and checking a
finite-difference step window. Neither path establishes full-FP temperature,
geometry or ambipolar-root derivatives.

.. list-table::
   :header-rows: 1
   :widths: 30 45 25

   * - Module
     - Role
     - Reference
   * - ``dkx.run``
     - ``run_profile`` / ``run_transport_matrix`` orchestration and console flow
     - :doc:`usage`
   * - ``dkx.config``
     - Immutable ``Case``, schema validation, IDs, and scan preflight
     - :doc:`case_files`
   * - ``dkx.execution`` / ``dkx.result``
     - Direct normalization/execution and named-array NetCDF results
     - :doc:`case_files`
   * - ``dkx.inputs`` / ``dkx.namelist``
     - Typed ``SfincsInput`` parser and raw namelist reader
     - :doc:`inputs`
   * - ``dkx.magnetic_geometry``
     - ``FluxSurfaceGeometry`` and all geometry schemes
     - :doc:`geometry`
   * - ``dkx.species``
     - Charges, masses, profiles, deflection frequencies
     - :doc:`physics_reference`
   * - ``dkx.phase_space``
     - Legendre coupling, Landreman--Ernst speed grid, ``Nxi_for_x`` ramp
     - :doc:`numerics`
   * - ``dkx.drift_kinetic``
     - ``KineticOperator`` — the consolidated v3 drift-kinetic operator
     - :doc:`physics_reference`, :doc:`system_equations`
   * - ``dkx.collisions``
     - Pitch-angle scattering and full Fokker--Planck (Rosenbluth) operators
     - :doc:`physics_reference`
   * - ``dkx.solve``
     - Automatic route choice: structured direct, recycled Krylov, sparse direct
     - :doc:`numerics`
   * - ``dkx.moments``
     - Velocity-space moments, fluxes, ``FSABjHat``, transport matrix
     - :doc:`outputs`, :doc:`physics_reference`
   * - ``dkx.phi1``
     - Nonlinear :math:`\Phi_1` / quasineutrality Newton solve
     - :doc:`physics_reference`
   * - ``dkx.er``
     - Ambipolar radial-electric-field root solve (Brent + differentiable)
     - :doc:`physics_reference`
   * - ``dkx.writer`` / ``dkx.io``
     - ``sfincsOutput.h5`` / ``.nc`` / ``.npz`` writer and reader
     - :doc:`outputs`
   * - ``dkx.sensitivity``
     - Implicit-differentiation observable derivatives (RHSMode 4/5 spine)
     - :doc:`feature_matrix`
   * - ``dkx.compare``
     - ``compare-h5`` parity tooling against Fortran fixtures
     - :doc:`parity`

Namelist parsing and reader helpers
------------------------------------

.. automodule:: dkx.namelist
   :members:

Reduced-model and analysis modules
----------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 45 25

   * - Module
     - Role
     - Reference
   * - ``dkx.monoenergetic``
     - Monoenergetic-database mode: ``(nuPrime, EStar)`` scans and energy
       convolution to thermal ``L_ij``
     - :doc:`capabilities`
   * - ``dkx.variational``
     - Variational upper/lower bounds on the monoenergetic :math:`D_{11}`
       (an error bound needing no reference run)
     - :doc:`capabilities`
   * - ``dkx.shaing_callen``
     - Collisionless-limit bootstrap coefficient with an analytic axisymmetric
       cross-check
     - :doc:`capabilities`

The differentiable solve, the implicit adjoint, and the
``vmex -> booz_xform_jax -> dkx`` chain are documented in
:doc:`differentiability`; the full source catalogue is :doc:`source_map`.
