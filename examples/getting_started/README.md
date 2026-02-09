# Getting started

These examples introduce the basic `sfincs_jax` workflow without requiring the Fortran v3 executable.

Suggested order:

1. `build_grids_and_geometry.py` — build v3 grids + geometry objects.
2. `apply_collisionless_operator.py` — apply a collisionless operator slice.
3. `write_sfincs_output_python.py` — write a v3-style `sfincsOutput.h5` from Python.
4. `write_sfincs_output_cli.py` — do the same via the CLI.

Run any script from the repo root, e.g.:

```bash
python examples/getting_started/build_grids_and_geometry.py
```

