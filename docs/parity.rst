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
     - No
     - See ``docs/fortran_examples.rst`` for the current audit

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

- Full kinetic solve driver across the upstream example suite (run loop, solves, and HDF5 outputs)
- Full Phi1 coupling end-to-end (nonlinear residual assembly + collision operator contributions)
- VMEC-based geometry schemes beyond the current ``geometryScheme=5`` parity subset
- ``sfincsOutput.h5`` writing for geometries other than ``geometryScheme in {1,2,4,5,11,12}``

Highest-priority open parity cases (reduced-suite)
--------------------------------------------------

- ``tokamak_1species_FPCollisions_noEr_withPhi1InDKE``:
  currently ``126/263`` mismatches, concentrated in the nonlinear includePhi1 solver branch
  (``Phi1Hat``, ``dPhi1Hatdtheta``, and flow/current moment family).
- ``geometryScheme5_3species_loRes``:
  currently ``33/193`` mismatches, concentrated in RHSMode=1 solver branch diagnostics
  (``FSAB`` moment family, flows, and density/pressure perturbations).

Reproduce these two blocker cases only:

.. code-block:: bash

   python scripts/run_reduced_upstream_suite.py \
     --pattern 'tokamak_1species_FPCollisions_noEr_withPhi1InDKE|geometryScheme5_3species_loRes' \
     --timeout-s 30 --max-attempts 1 --reset-report
