Parity status
=============

Implemented (parity-tested)
---------------------------

- v3 grids: ``theta``, ``zeta``, ``x`` (including the v3 polynomial/Stieltjes ``x`` grid)
- Boozer geometryScheme=4 (simplified W7-X model): ``BHat`` and derivatives

Not yet implemented
-------------------

- Full kinetic solve (matrix assembly, linear/nonlinear solve, Rosenbluth potentials, collision operators)
- VMEC-based geometry schemes and radial interpolation
- Writing JAX outputs to ``sfincsOutput.h5`` with full dataset coverage

