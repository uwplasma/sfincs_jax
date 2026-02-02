# sfincs_jax

A parity-first **JAX** port of **SFINCS Fortran v3**, with a focus on:

- **Numerical parity** against the upstream v3 implementation (fixture-by-fixture).
- **Performance** via JIT + vectorization (matrix-free operator application).
- **End-to-end differentiability** to enable gradient-based sensitivity, calibration, and optimization.

Documentation: build locally (`sphinx-build -b html docs docs/_build/html`) or view on Read the Docs.

## What exists today

`sfincs_jax` is intentionally incremental. The upstream v3 codebase is large, so the port proceeds in
small, parity-tested slices:

- v3 grids (`theta`, `zeta`, `x`) including the polynomial/Stieltjes x-grid
- `sfincsOutput.h5` output parity for `geometryScheme in {4,5,11}` vs frozen v3 fixtures
- `geometryScheme=11/12` Boozer `.bc` parsing (B, D, covariant components) + drift-term parity fixtures
- `geometryScheme=5` VMEC `wout_*.nc` parsing + output parity fixture
- Collisionless operator terms (streaming/mirror, ExB, Er terms, magnetic drift slices) parity-tested
- Collision operators (PAS and full linearized FP, no-Phi1 modes) parity-tested at the F-block level
- Full linearized FP collisions with poloidally varying Phi1 (parity on a tiny fixture)
- Full-system **matrix-free** matvec parity for two fixtures (no-Phi1, constraint schemes 1/2)
- Full-system **matrix-free** matvec + RHS + residual + GMRES-solution parity for one VMEC `geometryScheme=5` fixture (tiny PAS)
- Full-system **RHS and residual** assembly parity vs frozen Fortran v3 `evaluateResidual.F90` binaries (subset)
- Experimental Newton–Krylov nonlinear solve (parity on a tiny Phi1-in-kinetic fixture)
- Matrix-free residual/JVP scaffolding for implicit-diff workflows

Current parity coverage is tracked in `docs/parity.rst` and via the v3 example audit in `docs/fortran_examples.rst`.

## Install

Editable install (development):

```bash
pip install -e ".[dev]"
```

Optional extras:

```bash
# Docs:
pip install -e ".[docs]"

# Plotting examples:
pip install -e ".[viz]"

# Optimization ecosystem:
pip install -e ".[opt]"
```

## Quick start (Python)

```python
from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.v3 import grids_from_namelist, geometry_from_namelist

nml = read_sfincs_input("input.namelist")
grids = grids_from_namelist(nml)
geom = geometry_from_namelist(nml=nml, grids=grids)
print(geom.b_hat.shape)  # (Ntheta, Nzeta)
```

## CLI

Solve a supported v3 linear run matrix-free and write the solution vector:

```bash
sfincs_jax solve-v3 --input /path/to/input.namelist --out-state stateVector.npy
```

Write a SFINCS-style `sfincsOutput.h5` using the JAX implementation (supported modes only):

```bash
sfincs_jax write-output --input /path/to/input.namelist --out sfincsOutput.h5
```

Compare two `sfincsOutput.h5` files dataset-by-dataset:

```bash
sfincs_jax compare-h5 --a sfincsOutput_jax.h5 --b sfincsOutput_fortran.h5
```

## Benchmarking against the Fortran v3 executable

If you have the upstream Fortran v3 binary available locally:

```bash
export SFINCS_FORTRAN_EXE=/path/to/sfincs/fortran/version3/sfincs
sfincs_jax run-fortran --input /path/to/input.namelist
```

Note: on some locked-down environments (including certain CI/sandboxed runtimes), MPI may be unable to
open network endpoints, causing the Fortran executable to fail at `MPI_Init`. In that case, run the
Fortran benchmark on a normal workstation/HPC environment and copy the resulting fixtures into `tests/ref/`.

## Examples

Examples are structured by difficulty:

- `examples/1_simple/`: basic API usage (no Fortran required)
- `examples/2_intermediate/`: parity checks + autodiff demos
- `examples/3_advanced/`: optimization/implicit-diff patterns (may require extras)

Start here:

```bash
python examples/1_simple/01_build_grids_and_geometry.py
python examples/2_intermediate/11_autodiff_er_xidot_term.py  # requires ".[viz]"
```

Quick performance sanity check (JIT vs no-JIT):

```bash
python examples/2_intermediate/12_benchmark_jit_matvec.py
```

Upstream example inputs (Fortran v3, multi-species, and MATLAB v3) are vendored in `examples/upstream/`
so existing SFINCS users can find familiar starting points.

## Why JAX?

JAX makes three things practical at the same time:

1) **Fast kernels**: JIT-compile the operator application and solver inner loops.
2) **Matrix-free linear algebra**: represent the Jacobian as a matvec rather than assembling sparse matrices.
3) **Differentiability**: obtain gradients through geometry/operators and eventually through the solve (implicit diff).

The optional JAX ecosystem becomes natural once the compute graph is differentiable:

- `jaxopt`: robust root/linear solvers + implicit differentiation patterns
- `optax`: gradient-based optimization loops
- `equinox`: clean, testable module structure and parameter handling (optional)

## Testing and parity fixtures

Tests live in `tests/` and include:

- Parity tests that compare matrix-free matvecs to frozen PETSc binaries (`*.petscbin`)
- Output parity tests for `sfincsOutput.h5` vs frozen Fortran v3 fixtures

Run:

```bash
pytest -q
```

## Upstream docs (vendored)

Selected upstream SFINCS technical notes and paper sources are vendored in `docs/upstream/` and linked
from `docs/upstream_docs.rst`. This makes the Read the Docs site self-contained for readers who want
the original derivations and parameter definitions.

## Roadmap to full v3 parity

High-level remaining milestones (see `docs/parity.rst` for the detailed, parity-tested inventory):

- Expand RHS/residual coverage to the full upstream physics model (all modes/options)
- Implement full Phi1 coupling in the kinetic equation + collision operator
- Implement additional geometry schemes (VMEC-based, filtered W7-X netCDF, etc.)
- Achieve end-to-end solver parity across the full upstream v3 example suite

## License

See `LICENSE`.
