Parity status
=============

Implemented (parity-tested)
---------------------------

- v3 grids: ``theta``, ``zeta``, ``x`` (including the v3 polynomial/Stieltjes ``x`` grid)
- Boozer geometryScheme=4 (simplified W7-X model): ``BHat`` and derivatives
- Collisionless v3 operator slice: streaming + mirror (parity vs PETSc binaries for one example)
- Pitch-angle scattering collisions (``collisionOperator=1`` without Phi1): diagonal parity vs PETSc binaries for one small example

Not yet implemented
-------------------

- Full kinetic solve (residual/Jacobian assembly, linear/nonlinear solve, Rosenbluth potentials)
- Full linearized Fokker-Planck collision operator (``collisionOperator=0``)
- VMEC-based geometry schemes and radial interpolation
- Writing JAX outputs to ``sfincsOutput.h5`` with full dataset coverage
