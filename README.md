# sfincs_jax

A parity-first **JAX** port of **SFINCS Fortran v3**, with a focus on:

- **Numerical parity** against the upstream v3 implementation (fixture-by-fixture).
- **Performance** via JIT + vectorization (matrix-free operator application).
- **End-to-end differentiability** to enable gradient-based sensitivity, calibration, and optimization.

![BHat on a Boozer (θ, ζ) grid](docs/_static/figures/magdrift_bhat.png)

Documentation: build locally (`sphinx-build -b html docs docs/_build/html`) or view on Read the Docs.

## What exists today

`sfincs_jax` is intentionally incremental. The upstream v3 codebase is large, so the port proceeds in
small, parity-tested slices:

- v3 grids (`theta`, `zeta`, `x`) including the polynomial/Stieltjes x-grid
- `sfincsOutput.h5` output parity for `geometryScheme in {1,2,4,5,11,12}` vs frozen v3 fixtures
- `geometryScheme=11/12` Boozer `.bc` parsing (B, D, covariant components) + drift-term parity fixtures
- `geometryScheme=5` VMEC `wout_*.nc` parsing + output parity fixture
- Collisionless operator terms (streaming/mirror, ExB, Er terms, magnetic drift slices) parity-tested
- Collision operators (PAS and full linearized FP, no-Phi1 modes) parity-tested at the F-block level
- Full linearized FP collisions with poloidally varying Phi1 (parity on a tiny fixture)
- Full-system **matrix-free** matvec parity for two fixtures (no-Phi1, constraint schemes 1/2)
- Full-system **matrix-free** matvec + RHS + residual + GMRES-solution parity for VMEC `geometryScheme=5` fixtures (tiny PAS, with/without Phi1 QN blocks)
- Full-system **RHS and residual** assembly parity vs frozen Fortran v3 `evaluateResidual.F90` binaries (subset)
- Transport-matrix modes (`RHSMode=2/3`): v3 `whichRHS` loop RHS settings + `transportMatrix` assembly parity (including monoenergetic `x=1` / `xWeights=exp(1)` special-case)
- Experimental Newton–Krylov nonlinear solve (parity on a tiny Phi1-in-kinetic fixture)
- Matrix-free residual/JVP scaffolding for implicit-diff workflows
- Implicit-differentiation through linear GMRES solves (`sfincs_jax.implicit_solve`)

Current parity coverage is tracked in `docs/parity.rst` and via the v3 example audit in `docs/fortran_examples.rst`.

## Parity status (summary)

This table is a *high-level* view of what is currently parity-tested. See `docs/parity.rst` for the detailed inventory.

| Area | Status | Notes |
|---|---|---|
| Grids (`theta`, `zeta`, `x`) | ✅ | Includes monoenergetic `x=1`, `xWeights=exp(1)` special-case |
| Geometry (schemes 1/2/4) | ✅ | `sfincsOutput.h5` parity fixtures |
| Geometry (scheme 5 VMEC `wout_*.nc`) | ✅ (subset) | Core arrays + output parity fixture |
| Geometry (schemes 11/12 `.bc`) | ✅ | Includes transport-matrix end-to-end fixtures |
| Collisionless operator | ✅ (subset) | Streaming/mirror, ExB, Er slices, magnetic drift slices |
| Collision operators | ✅ (subset) | PAS + linearized FP, including a tiny Phi1-in-collision fixture |
| Classical transport fluxes | ✅ (subset) | `calculateClassicalFlux` parity for schemes with `gpsiHatpsiHat` support (5/11/12) |
| Linear solve (RHSMode=1) | ✅ (fixtures) | Matrix-free GMRES parity on tiny cases |
| Transport matrices (RHSMode=2/3) | ✅ (fixtures) | End-to-end `transportMatrix` parity (2×2 and 3×3 cases) |
| Nonlinear Newton–Krylov | ⚠️ experimental | Tiny parity fixture only |
| Full upstream v3 example suite | 🚧 | See `docs/fortran_examples.rst` |

## Install

Regular users (from PyPI):

```bash
pip install sfincs_jax
```

From source (recommended for parity work and examples):

```bash
git clone https://github.com/uwplasma/sfincs_jax.git
cd sfincs_jax
pip install .
```

Development install:

```bash
git clone https://github.com/uwplasma/sfincs_jax.git
cd sfincs_jax
pip install -e ".[dev]"
```

Optional extras (install as needed):

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

All subcommands support:

- `-v/--verbose` (repeatable): print more progress information
- `-q/--quiet`: suppress non-essential output

Solve a supported v3 linear run matrix-free and write the solution vector:

```bash
sfincs_jax solve-v3 --input /path/to/input.namelist --out-state stateVector.npy
```

Transport-matrix modes (``RHSMode=2/3``) require selecting which RHS to solve:

```bash
sfincs_jax solve-v3 --input /path/to/input.namelist --which-rhs 1
```

Compute the full transport matrix by looping ``whichRHS`` internally:

```bash
sfincs_jax transport-matrix-v3 --input /path/to/input.namelist --out-matrix transportMatrix.npy
```

Write a SFINCS-style `sfincsOutput.h5` using the JAX implementation (supported modes only):

```bash
sfincs_jax write-output --input /path/to/input.namelist --out sfincsOutput.h5
```

For ``RHSMode=1`` runs, you can optionally solve the linear system and write solution-derived fields
(flows, fluxes, constraints, etc.):

```bash
sfincs_jax write-output --input /path/to/input.namelist --out sfincsOutput.h5 --compute-solution
```

For ``RHSMode=2/3`` runs, you can optionally also compute and write ``transportMatrix``:

```bash
sfincs_jax write-output --input /path/to/input.namelist --out sfincsOutput.h5 --compute-transport-matrix
```

In this mode, `sfincs_jax` also writes a small set of RHSMode>1 diagnostics used by upstream plotting scripts:
`FSABFlow`, `FSABjHat`, `particleFlux_vm_psiHat`, `heatFlux_vm_psiHat`, before-surface-integral arrays,
and `*_vs_x` contributions (see `sfincs_jax.transport_matrix.v3_transport_output_fields_vm_only`).

Compare two `sfincsOutput.h5` files dataset-by-dataset:

```bash
sfincs_jax compare-h5 --a sfincsOutput_jax.h5 --b sfincsOutput_fortran.h5
```

## Postprocessing (upstream utils)

SFINCS Fortran v3 ships a set of plotting scripts under `utils/`. This repo vendors those scripts in
`examples/sfincs_examples/utils/`. To run one non-interactively:

```bash
sfincs_jax postprocess-upstream --case-dir /path/to/case --util sfincsScanPlot_1 -- pdf
```

For transport-matrix scans, you can also generate a scan directory compatible with upstream plotting:

```bash
sfincs_jax scan-er --input /path/to/input.namelist --out-dir /path/to/scan_dir --min -0.1 --max 0.1 --n 5 --compute-transport-matrix
sfincs_jax postprocess-upstream --case-dir /path/to/scan_dir --util sfincsScanPlot_2 -- pdf
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

Examples are structured by topic:

- `examples/getting_started/`: minimal “hello world” workflows (no Fortran required)
- `examples/parity/`: parity + validation vs frozen v3 fixtures
- `examples/transport/`: `RHSMode=2/3` transport-matrix workflows + upstream scanplot scripts
- `examples/autodiff/`: autodiff + implicit-diff demonstrations
- `examples/optimization/`: optimization patterns (may require extras)
- `examples/performance/`: JIT/performance microbenchmarks
- `examples/publication_figures/`: publication-style figure generation

Start here:

```bash
python examples/getting_started/01_build_grids_and_geometry.py
python examples/autodiff/11_autodiff_er_xidot_term.py  # requires ".[viz]"
python examples/transport/18_transport_matrix_er_scan_upstream_scanplot2.py  # requires ".[viz]"
```

Optimization + publication-ready figures (optional extras):

```bash
pip install -e ".[opt,viz]"
python examples/optimization/04_optimize_scheme4_harmonics_publication_figures.py
python examples/optimization/05_calibrate_nu_n_to_fortran_residual_fixture.py
```

Implicit differentiation through solves (advanced):

```bash
python examples/autodiff/06_implicit_diff_through_gmres_solve_scheme5.py
```

Quick performance sanity check (JIT vs no-JIT):

```bash
python examples/performance/12_benchmark_jit_matvec.py
```

Upstream inputs and scripts are vendored so existing SFINCS users can find familiar starting points:

- `examples/upstream/`: curated upstream inputs used by parity tests / docs
- `examples/sfincs_examples/`: full upstream Fortran v3 `examples/` suite (copy)
- `examples/sfincs_examples/utils/`: upstream postprocessing scripts (copy)

To run the full vendored suite on `sfincs_jax` (best-effort, not all examples supported yet):

```bash
python examples/sfincs_examples/run_sfincs_jax.py --write-output
```

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
