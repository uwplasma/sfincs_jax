# Parity & validation

These examples are focused on **parity-first development**:
- comparing against frozen Fortran v3 fixtures in `tests/ref/`
- reproducing upstream example inputs
- sanity-checking output key coverage

Most scripts run without the Fortran executable, but a few call it (or rely on its outputs) for comparisons.

Suggested starting points:
- `07_output_parity_vs_fortran_fixture.py` — dataset-by-dataset parity vs a frozen `sfincsOutput.h5`.
- `02_collisionless_operator_matvec_parity.py` — operator parity vs PETSc binaries.
- `13_solve_scheme5_tiny_parity.py` — end-to-end GMRES solve parity for a tiny VMEC case.

