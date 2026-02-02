Parity status
=============

Implemented (parity-tested)
---------------------------

- v3 grids: ``theta``, ``zeta``, ``x`` (including the v3 polynomial/Stieltjes ``x`` grid)
- Boozer geometryScheme=4 (simplified W7-X model): ``BHat`` and derivatives
- Boozer geometryScheme=11/12 from `.bc` file inputs: ``BHat``, ``DHat``, ``BHat_sub_psi``, and derivatives (parity vs frozen fixture)
- VMEC geometryScheme=5 from ``wout_*.nc`` inputs: core geometry arrays (``BHat``, ``DHat``, covariant/contravariant components)
  and ``gpsiHatpsiHat`` (parity vs frozen fixture)
- ``sfincsOutput.h5`` writing for ``geometryScheme in {4,5,11}`` with dataset-by-dataset parity against frozen
  Fortran v3 fixtures (see ``docs/outputs.rst``). ``uHat`` is compared with a looser tolerance due to tiny
  platform-dependent transcendental/reduction differences.
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

Not yet implemented
-------------------

- Full kinetic solve driver across the upstream example suite (run loop, solves, and HDF5 outputs)
- Full Phi1 coupling end-to-end (nonlinear residual assembly + collision operator contributions)
- VMEC-based geometry schemes beyond the current ``geometryScheme=5`` parity subset
- ``sfincsOutput.h5`` writing for geometries other than ``geometryScheme in {4,5,11,12}``
