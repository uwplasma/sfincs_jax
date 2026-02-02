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

## Writing `sfincsOutput.h5` with `sfincs_jax`

For supported modes (currently `geometryScheme=4`), `sfincs_jax` can write a SFINCS-style
`sfincsOutput.h5` file:

```bash
sfincs_jax write-output --input /path/to/input.namelist --out sfincsOutput.h5
```

You can compare a JAX output file with a Fortran v3 output file (dataset-by-dataset):

```bash
sfincs_jax compare-h5 --a sfincsOutput_jax.h5 --b sfincsOutput_fortran.h5
```

## Roadmap

- [x] Packaging + CLI scaffolding
- [x] Parse SFINCS `input.namelist` (minimal)
- [x] v3 theta/zeta/x grids and simplified Boozer geometryScheme=4
- [x] v3 grids for geometryScheme=11/12 (read `NPeriods` from `.bc` header)
- [x] Boozer geometryScheme=11/12 from `.bc` files (BHat and required covariant components)
- [x] Collisionless v3 operator slice (streaming + mirror) with PETSc-binary parity test
- [x] Collisionless v3 Er terms (`xiDot` + `xDot`) with PETSc-binary parity tests (ΔL = ±2)
- [x] ExB drift term (`d/dtheta` for geometryScheme=4) with PETSc-binary parity test
- [x] Magnetic drift terms (`d/dtheta`, `d/dzeta`, non-standard `d/dxi`) with PETSc-binary parity tests (ΔL = ±2 slices)
- [x] Pitch-angle scattering collisions (collisionOperator=1 without Phi1) with PETSc-binary parity test
- [x] Combined F-block matvec parity (collisionless + PAS) vs PETSc matrix (F-block slice)
- [x] Full linearized Fokker-Planck collisions (collisionOperator=0, no Phi1) with F-block matvec parity vs PETSc matrix
- [x] Residual/Jacobian (JVP) scaffolding for F-block (matrix-free)
- [x] Full-system matvec parity (includePhi1=false constraint schemes 1/2) vs PETSc matrices for two fixtures
- [x] Full-system GMRES solution parity vs PETSc stateVector (tiny PAS fixture)
- [ ] Full solver parity across the v3 example suite (includes includePhi1 and more geometries)

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

For a structured view of how much of the upstream Fortran v3 example suite is currently
supported, see the docs page `docs/fortran_examples.rst` (auto-generated audit table).
