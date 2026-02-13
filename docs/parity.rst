Parity status
=============

High-level summary (parity-tested)
----------------------------------

.. list-table::
   :header-rows: 1
   :widths: 28 12 60

   * - Area
     - Status
     - Notes
   * - Grids (``theta``, ``zeta``, ``x``)
     - Yes
     - Includes monoenergetic ``x=1`` / ``xWeights=exp(1)`` special-case
   * - Geometry schemes ``1/2/4``
     - Yes
     - Output parity fixtures
   * - Geometry scheme ``5`` (VMEC ``wout_*.nc``)
     - Partial
     - Core arrays + output parity fixture for a small case
   * - Geometry schemes ``11/12`` (Boozer ``.bc``)
     - Yes
     - Geometry + transport-matrix end-to-end fixtures
   * - Linear runs (RHSMode=1)
     - Partial
     - Matrix-free GMRES parity on a growing set of tiny fixtures
   * - Transport matrices (RHSMode=2/3)
     - Yes
     - End-to-end ``sfincsOutput.h5`` parity for 2×2 and 3×3 cases
   * - Full upstream v3 example suite
     - Partial
     - Reduced-suite practical status is ``38/38 parity_ok``. Strict mode matches the same 38/38 split. For cases that emit stdout signals, print parity is 7/7.

Implemented (parity-tested)
---------------------------

- v3 grids: ``theta``, ``zeta``, ``x`` (including the v3 polynomial/Stieltjes ``x`` grid)
  and the monoenergetic (``RHSMode=3``) special-case ``x=1`` / ``xWeights=exp(1)`` grid used in v3 ``createGrids.F90``.
- Boozer geometryScheme=1 (three-helicity analytic model): ``BHat`` and derivatives (via output parity fixtures)
- Boozer geometryScheme=2 (simplified LHD model): ``BHat`` and derivatives (via output parity fixtures)
- Boozer geometryScheme=4 (simplified W7-X model): ``BHat`` and derivatives
- Boozer geometryScheme=11/12 from `.bc` file inputs: ``BHat``, ``DHat``, ``BHat_sub_psi``, and derivatives (parity vs frozen fixture)
- VMEC geometryScheme=5 from ``wout_*.nc`` inputs: core geometry arrays (``BHat``, ``DHat``, covariant/contravariant components)
  and ``gpsiHatpsiHat`` (parity vs frozen fixture)
- ``sfincsOutput.h5`` writing for ``geometryScheme in {1,2,4,5,11,12}`` with dataset-by-dataset parity against frozen
  Fortran v3 fixtures (see ``docs/outputs.rst``). ``uHat`` is compared with a looser tolerance due to tiny
  platform-dependent transcendental/reduction differences.
- Classical transport (`calculateClassicalFlux`) for geometries with `gpsiHatpsiHat` support:
  `geometryScheme=5` (VMEC) and `geometryScheme=11/12` (.bc) — parity-tested via frozen `sfincsOutput.h5` fixtures.
- Collisionless v3 operator slice: streaming + mirror (parity vs PETSc binaries for one example)
- Collisionless v3 Er terms:

  - non-standard ``d/dxi`` term (``includeElectricFieldTermInXiDot = .true.``): ΔL = ±2 parity vs Fortran Jacobian
  - collisionless ``d/dx`` term (``includeXDotTerm = .true.``): ΔL = ±2 parity vs Fortran Jacobian

- ExB drift term (``useDKESExBDrift = .false.``): ``d/dtheta`` parity vs Fortran Jacobian (geometryScheme=4)
- Magnetic drift terms (``magneticDriftScheme=1``): parity-tested as ΔL = ±2 slices vs Fortran Jacobian (geometryScheme=11)
- Pitch-angle scattering collisions (``collisionOperator=1`` without Phi1): diagonal parity vs PETSc binaries for one small example
- Full linearized Fokker-Planck collision operator (``collisionOperator=0`` without Phi1): F-block matvec parity vs a frozen
  PETSc matrix for a 2-species ``geometryScheme=4`` fixture (``tests/ref/quick_2species_FPCollisions_noEr.whichMatrix_3.petscbin``).
- Full-system matvec parity (includePhi1=false, constraint schemes 1/2) vs frozen PETSc matrices for:
  ``pas_1species_PAS_noEr_tiny`` and ``quick_2species_FPCollisions_noEr``.
- Full-system matvec + RHS + residual + GMRES-solution parity for VMEC ``geometryScheme=5`` (tiny PAS case):
  ``pas_1species_PAS_noEr_tiny_scheme5``.
- Full-system matvec + RHS + residual + GMRES-solution parity for VMEC ``geometryScheme=5`` with Phi1 QN/lambda blocks:
  ``pas_1species_PAS_noEr_tiny_scheme5_withPhi1_linear``.
- Full-system matvec + RHS + residual + GMRES-solution parity for ``geometryScheme=1`` (tokamak-like, Nzeta=1):
  ``pas_1species_PAS_noEr_tiny_scheme1``.
- Transport-matrix modes (``RHSMode=2/3``):

  - v3 internal ``whichRHS`` RHS settings parity (RHS/residual at ``f=0``) for ``RHSMode=2`` and ``RHSMode=3``.
  - Monoenergetic special-case ``x=1`` / ``xWeights=exp(1)`` (v3 ``createGrids.F90``) parity.
  - Full-system matvec parity vs v3 solver matrix (``whichMatrix=1``) for tiny monoenergetic fixtures in:
    ``monoenergetic_PAS_tiny_scheme1``, ``monoenergetic_PAS_tiny_scheme11``, and ``monoenergetic_PAS_tiny_scheme5_filtered``.
  - ``transportMatrix`` assembly parity vs frozen Fortran v3 ``sfincsOutput.h5`` for:
    ``monoenergetic_PAS_tiny_scheme{1,11,12,5_filtered}`` (``RHSMode=3``) and
    ``transportMatrix_PAS_tiny_rhsMode2_scheme2`` (``RHSMode=2``).
- Phi1/QN/lambda block parity (includePhi1=true, includePhi1InKineticEquation=false):
  full-system matvec + GMRES solution parity vs frozen PETSc binaries for
  ``pas_1species_PAS_noEr_tiny_withPhi1_linear``.
- Phi1 in kinetic equation parity (includePhi1=true, includePhi1InKineticEquation=true):
  full-system matvec + GMRES solution parity vs frozen PETSc binaries for
  ``pas_1species_PAS_noEr_tiny_withPhi1_inKinetic_linear``.
- Nonlinear end-to-end solve (experimental Newton–Krylov) parity for:
  ``pas_1species_PAS_noEr_tiny_withPhi1_inKinetic_linear``.
- Full-system RHS and residual assembly parity vs frozen Fortran v3 `evaluateResidual.F90` binaries for:
  ``pas_1species_PAS_noEr_tiny``, ``quick_2species_FPCollisions_noEr``,
  ``pas_1species_PAS_noEr_tiny_withPhi1_linear``, and ``pas_1species_PAS_noEr_tiny_withPhi1_inKinetic_linear``.
- Full linearized Fokker-Planck collisions with Phi1 in the collision operator
  (``collisionOperator=0``, ``includePhi1InCollisionOperator=true``) parity-tested as full-system matvec + residual for:
  ``fp_1species_FPCollisions_noEr_tiny_withPhi1_inCollision``.

Current scope limits
--------------------

- Strict parity (``docs/_generated/reduced_upstream_suite_status_strict.rst``) is currently clean on all
  reduced cases; no Fortran-run errors remain in the reduced suite on this machine.
- The unconstrained ``constraintScheme=0`` branch is rank-deficient, so different solvers can select different nullspace
  components. For comparisons, sfincs_jax treats a small set of density/pressure-like outputs as gauge-dependent and
  skips them when ``constraintScheme=0`` (see ``sfincs_jax/compare.py``).
- Full Phi1 coupling end-to-end (nonlinear residual assembly + collision operator contributions) is still being expanded beyond the currently parity-tested subset.
- VMEC-based geometry schemes beyond the current ``geometryScheme=5`` parity subset.
- Rosenbluth response matrices for FP cross-species coupling are computed with QUADPACK (matching v3). We added strict
  scalar-order accumulation for the collocation-to-modal projection, but the remaining ~1e-10 deltas appear dominated by
  quadrature rounding differences rather than matrix-ordering effects.
- A small number of RHSMode=1 reduced cases (notably filtered-W7X netCDF and sfincsPaper fig. 3) are ill-conditioned:
  PETSc's GMRES+LU preconditioning can converge to a slightly different solution than an exact linear solve. We capture
  these solver-branch differences via per-case tolerances on flow/jHat diagnostics in the reduced-suite reports, while
  maintaining full reduced-suite parity at the current tolerances.

Reduced-suite parity status (source of truth)
---------------------------------------------

The reduced upstream parity inventory is auto-generated and should be treated as the
authoritative status:

- ``docs/_generated/reduced_upstream_suite_status.rst``
- ``docs/_generated/reduced_upstream_suite_status_strict.rst``
- ``tests/reduced_upstream_examples/suite_report.json``
- ``tests/reduced_upstream_examples/suite_report_strict.json``

Regenerate these files:

.. code-block:: bash

   python scripts/run_reduced_upstream_suite.py --timeout-s 30 --max-attempts 1

Target a single case family:

.. code-block:: bash

   python scripts/run_reduced_upstream_suite.py \
     --pattern 'HSX_FPCollisions|filteredW7XNetCDF_2species_magneticDrifts|geometryScheme4_2species' \
     --timeout-s 30 --max-attempts 1

Matrix/operator parity diagnosis (Fortran PETSc matrix vs JAX matvec):

.. code-block:: bash

   python scripts/compare_fortran_matrix_to_jax_operator.py \
     --input /path/to/input.namelist \
     --fortran-matrix /path/to/sfincsBinary_iteration_000_whichMatrix_3 \
     --fortran-state /path/to/sfincsBinary_iteration_000_stateVector \
     --project-active-dofs \
     --out-json matrix_compare.json

Frozen-state diagnostics isolation (solver-vs-diagnostics for RHSMode=1 moment families):

.. code-block:: bash

   python scripts/compare_rhsmode1_diagnostics_from_state.py \
     --input /path/to/input.namelist \
     --state /path/to/sfincsBinary_iteration_000_stateVector \
     --fortran-h5 /path/to/sfincsOutput.h5 \
     --out-json diagnostics_from_frozen_state.json
