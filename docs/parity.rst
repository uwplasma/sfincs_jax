Parity status
=============

Implemented (parity-tested)
---------------------------

- v3 grids: ``theta``, ``zeta``, ``x`` (including the v3 polynomial/Stieltjes ``x`` grid)
- Boozer geometryScheme=4 (simplified W7-X model): ``BHat`` and derivatives
- Boozer geometryScheme=11 from `.bc` file inputs: ``BHat``, ``DHat``, ``BHat_sub_psi``, and derivatives (parity vs frozen fixture)
- ``sfincsOutput.h5`` writing for ``geometryScheme=4`` with dataset-by-dataset parity against a frozen
  Fortran v3 fixture (see ``docs/outputs.rst``). ``uHat`` is compared with a looser tolerance due to
  tiny platform-dependent transcendental/reduction differences.
- Collisionless v3 operator slice: streaming + mirror (parity vs PETSc binaries for one example)
- Collisionless v3 Er terms:

  - non-standard ``d/dxi`` term (``includeElectricFieldTermInXiDot = .true.``): |ΔL|=2 parity vs Fortran Jacobian
  - collisionless ``d/dx`` term (``includeXDotTerm = .true.``): |ΔL|=2 parity vs Fortran Jacobian

- ExB drift term (``useDKESExBDrift = .false.``): ``d/dtheta`` parity vs Fortran Jacobian (geometryScheme=4)
- Magnetic drift terms (``magneticDriftScheme=1``): parity-tested as |ΔL|=2 slices vs Fortran Jacobian (geometryScheme=11)
- Pitch-angle scattering collisions (``collisionOperator=1`` without Phi1): diagonal parity vs PETSc binaries for one small example

Not yet implemented
-------------------

- Full kinetic solve (residual/Jacobian assembly, linear/nonlinear solve, Rosenbluth potentials)
- Full linearized Fokker-Planck collision operator (``collisionOperator=0``)
- VMEC-based geometry schemes and radial interpolation
- ``sfincsOutput.h5`` writing for geometries other than ``geometryScheme=4``
