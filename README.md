# `sfincs_jax`

`sfincs_jax` is a parity-first port of **SFINCS Fortran v3** to **JAX**, with the goal of:

- Matching the Fortran algorithms and outputs (example-by-example).
- Enabling **auto-differentiation** through the entire compute graph.
- Providing a clean, modular Python API with modern tooling (tests, docs, CI).

This repo is intentionally incremental: the Fortran v3 solver is large, so we port subsystems
in a way that always keeps a working, tested baseline.

## Install (editable)

```bash
pip install -e ".[dev]"
```

## Quick start

Parse a Fortran `input.namelist`, build the v3 grids, and compute the (simplified) Boozer
geometry for `geometryScheme = 4`:

```python
from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.v3 import grids_from_namelist, geometry_from_namelist

nml = read_sfincs_input("input.namelist")
grids = grids_from_namelist(nml)
geom = geometry_from_namelist(nml=nml, grids=grids)
print(geom.b_hat.shape)
```

## Running the Fortran v3 executable (benchmark)

The Fortran binary is not shipped with this package, but you can point to a local build:

```bash
export SFINCS_FORTRAN_EXE=/path/to/sfincs/fortran/version3/sfincs
sfincs_jax run-fortran --input /path/to/input.namelist
```

## Roadmap

- [x] Packaging + CLI scaffolding
- [x] Parse SFINCS `input.namelist` (minimal)
- [x] v3 theta/zeta/x grids and simplified Boozer geometryScheme=4
- [x] Collisionless v3 operator slice (streaming + mirror) with PETSc-binary parity test
- [x] Pitch-angle scattering collisions (collisionOperator=1 without Phi1) with PETSc-binary parity test
- [ ] Full linearized Fokker-Planck collisions (collisionOperator=0)
- [ ] Residual/Jacobian assembly in JAX (matrix-free where possible)
- [ ] Full solver parity across the v3 example suite

## Why JAX?

JAX makes it practical to:

- JIT-compile the operator application and solver kernels for performance.
- Differentiate through geometry, collisions, and (eventually) the full solve.
- Use matrix-free linear algebra and implicit-diff patterns (`jaxopt`) for scalable gradients.
- Build optimization loops (`optax`) once the compute graph is differentiable.

Optional ecosystem packages are available via `pip install -e ".[opt]"`.

## Examples

See `examples/README.md`. The examples are structured as:

- `examples/1_simple/`
- `examples/2_intermediate/`
- `examples/3_advanced/`
