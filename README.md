# sfincs_jax

`sfincs_jax` is a JAX implementation of the SFINCS Fortran v3 workflow, with matrix-free operators,
JIT acceleration, and end-to-end differentiable components for sensitivity and optimization studies.
Default RHSMode=1 linear solves use GMRES (incremental, parity-first) with stage-2 fallback,
while RHSMode=2/3 transport solves default to BiCGStab with a collision-diagonal preconditioner
and GMRES fallback. Implicit differentiation is enabled by default for linear solves, and
preconditioner blocks use size-based mixed precision by default for memory efficiency.

![SFINCS vs sfincs_jax L11 relative difference and runtime](docs/_static/figures/sfincs_vs_sfincs_jax_l11_runtime_2x2.png)

Top figure: four monoenergetic test cases (`geometryScheme=1`, `11`, `12`, and filtered `5`) comparing
relative `ΔL11 = (JAX − Fortran) / Fortran` and per-run runtime.
For the JAX path, runtime excludes compilation (warm-up run excluded).

Reproduce the figure and JSON summary:

```bash
python examples/performance/benchmark_transport_l11_vs_fortran.py --repeats 4
```

Run against a live local Fortran executable:

```bash
python examples/performance/benchmark_transport_l11_vs_fortran.py \
  --fortran-exe /path/to/sfincs \
  --repeats 4
```

Current benchmark snapshot (Fortran source: live executable, 4 repeats; JAX runtime excludes compilation):

| Case | Fortran mean (s/run) | sfincs_jax mean (s/run) | Max \|ΔL11\| |
| --- | ---: | ---: | ---: |
| scheme1 | 0.2866 | 0.1774 | 2.16e-13 |
| scheme11 | 0.2382 | 0.2103 | 1.31e-15 |
| scheme12 | 0.0761 | 0.1896 | 8.82e-08 |
| scheme5_filtered | 0.0912 | 0.2015 | 5.30e-16 |

Snapshot note: when running with a local Fortran binary, PETSc MPIUNI (serial, no MUMPS) avoids MPI init issues in sandboxed runs.

Outputs are written to:

- `examples/performance/output/transport_l11_vs_fortran/sfincs_vs_sfincs_jax_l11_runtime_2x2.png`
- `examples/performance/output/transport_l11_vs_fortran/sfincs_vs_sfincs_jax_l11_runtime_2x2.json`

Persistent-cache compile/runtime split (same four cases):

```bash
python examples/performance/profile_transport_compile_runtime_cache.py --repeats 3
```

- Figure: `examples/performance/output/compile_runtime_cache/transport_compile_runtime_cache_2x2.png`
- JSON: `examples/performance/output/compile_runtime_cache/transport_compile_runtime_cache_2x2.json`

Small dense fallback assemblies now skip JIT for modest matrix sizes (`n<=800`) to
reduce overhead in tiny PAS/transport cases while keeping large systems compiled.

## Installation

Install from PyPI:

```bash
pip install sfincs_jax
```

Install from source:

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

Optional extras:

```bash
pip install -e ".[docs]"   # documentation build
pip install -e ".[viz]"    # plotting / figure scripts
pip install -e ".[opt]"    # optax / jaxopt workflows
```

## Quick start (Python)

```python
from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.v3 import grids_from_namelist, geometry_from_namelist

nml = read_sfincs_input("input.namelist")
grids = grids_from_namelist(nml)
geometry = geometry_from_namelist(nml=nml, grids=grids)
print(geometry.b_hat.shape)
```

## Parallel Scaling (Macbook M3 Max)

Parallel `whichRHS` scaling for an extra-large RHSMode=2 transport-matrix case
(`examples/performance/transport_parallel_xxlarge.input.namelist`, geometryScheme=2).

![Parallel whichRHS scaling](docs/_static/figures/parallel/transport_parallel_scaling.png)

Latest cache-warm run (1-4 workers): 1 worker 65.6s, 2 workers 46.0s,
3 workers 25.7s, 4 workers 26.0s.

Enable parallel execution in normal runs:

```bash
export SFINCS_JAX_CORES=4
```

Reproduce the 1-4 worker scaling figure and JSON summary:

```bash
python examples/performance/benchmark_transport_parallel_scaling.py \
  --input examples/performance/transport_parallel_xxlarge.input.namelist \
  --workers 1 2 3 4 \
  --repeats 1 \
  --warmup 0 \
  --global-warmup 0
```

For multi-node arrays and advanced parallel modes, see `docs/parallelism.rst`.

## What the code supports

- v3 grid construction (`theta`, `zeta`, Stieltjes/polynomial `x`, monoenergetic `x=1` path)
- Geometry pipelines for `geometryScheme in {1,2,4,5,11,12}`
- Matrix-free v3 full-system operator, RHS, and residual assembly in JAX
- Linear solves via GMRES (default for RHSMode=1) with BiCGStab optional; transport-matrix
  (`RHSMode=2/3`) loops default to BiCGStab with collision-diagonal preconditioning
- Implicit-diff linear solves via `jax.lax.custom_linear_solve` (default for RHSMode=1 + transport)
- Transport-matrix recycling warm starts (optional, `SFINCS_JAX_TRANSPORT_RECYCLE_K`)
- `sfincsOutput.h5` writing from Python and CLI
- Parity tests against frozen Fortran fixtures (PETSc binaries and `sfincsOutput.h5`)
- Differentiable operator and solve-adjacent workflows (including implicit-diff helper APIs)

Detailed parity inventory and dataset coverage:

- `docs/parity.rst`
- `docs/outputs.rst`
- `docs/fortran_examples.rst`

## Solver defaults (parity + performance)

- RHSMode=1 dense fallback now defaults to `SFINCS_JAX_RHSMODE1_DENSE_FALLBACK_MAX=400`, with higher
  ceilings `SFINCS_JAX_RHSMODE1_DENSE_FP_MAX=5000` (full FP) and
  `SFINCS_JAX_RHSMODE1_DENSE_PAS_MAX=5000` (PAS/constraintScheme=2) to recover Fortran convergence
  when Krylov stagnates.
- includePhi1 runs use full Newton updates by default (matching v3). Frozen linearization is
  opt‑in via `SFINCS_JAX_PHI1_USE_FROZEN_LINEARIZATION`, and small Phi1‑collision systems use a dense
  Newton step when `SFINCS_JAX_PHI1_NK_DENSE_CUTOFF` is met.
- Transport-matrix dense retries are capped by `SFINCS_JAX_TRANSPORT_DENSE_MAX_MB=128` to avoid
  excessive memory usage.
- PAS constraint projection auto-enables for tokamak-like `N_zeta=1` cases and DKES trajectories
  to eliminate nullspace drift (see `SFINCS_JAX_PAS_PROJECT_CONSTRAINTS`).
- `SFINCS_JAX_DENSE_MAX=8000` caps direct dense solves to avoid runaway memory use.
- VMEC geometryScheme=5 full‑FP parity uses dedicated near‑zero tolerances for flow/pressure
  diagnostics; strict tables still pass at reduced-suite tolerances (see `docs/parity.rst`).

Detailed performance profiling and advanced parallel modes are documented in
`docs/performance.rst` and `docs/parallelism.rst`.

## CLI

Write SFINCS-style output:

```bash
sfincs_jax write-output --input /path/to/input.namelist --out sfincsOutput.h5
```

Solve a supported v3 linear case (`RHSMode=1`) and save the state:

```bash
sfincs_jax solve-v3 --input /path/to/input.namelist --out-state stateVector.npy
```

Compute transport matrix (`RHSMode=2/3`):

```bash
sfincs_jax transport-matrix-v3 --input /path/to/input.namelist --out-matrix transportMatrix.npy
```

Compare two `sfincsOutput.h5` files key-by-key:

```bash
sfincs_jax compare-h5 --a sfincsOutput_jax.h5 --b sfincsOutput_fortran.h5
```

## Examples

`examples/` is organized by workflow category:

- `examples/getting_started/`: API/CLI fundamentals
- `examples/parity/`: fixture parity and validation runs
- `examples/transport/`: RHSMode=2/3 + postprocessing workflows
- `examples/autodiff/`: Jacobian-vector products, sensitivity, implicit differentiation
- `examples/optimization/`: optimization loops using JAX ecosystem tools
- `examples/performance/`: timing and JIT benchmarks
- `examples/publication_figures/`: polished figure generation
- `examples/sfincs_examples/`: vendored upstream Fortran v3 examples + helper runner

Try these first:

```bash
python examples/getting_started/build_grids_and_geometry.py
python examples/getting_started/write_sfincs_output_python.py
python examples/parity/output_parity_vs_fortran_fixture.py
python examples/autodiff/autodiff_er_xidot_term.py
python examples/transport/transport_matrix_rhsmode2_and_rhsmode3.py
python examples/transport/transport_matrix_recycle_demo.py
python examples/autodiff/implicit_diff_through_gmres_solve_scheme5.py --solver bicgstab
```

## Utils (ported upstream scripts)

The repository root `utils/` folder ports the full SFINCS v3 `utils` suite to
`sfincs_jax`. Every script runs via a Python driver (`utils/sfincs_jax_driver.py`)
and never calls the Fortran executable. Plotting and scan workflows produce the
same figure layouts as upstream.

Generate a small gallery:

```bash
python examples/utils/generate_utils_gallery.py
```

Add `--fast` for a quick pass or `--timeout-s <seconds>` to cap each step.

Reproduced SFINCS paper-style figures are documented in `docs/paper_figures.rst` and can be
generated with:

```bash
python examples/publication_figures/generate_sfincs_paper_figs.py --fast
```

The utilities honor the upstream `!ss` scan directives in `input.namelist`
(see `docs/utils.rst`) and produce the same scan layouts as the original
SFINCS v3 scripts.

## Upstream SFINCS compatibility and parity status

The repository vendors the upstream Fortran v3 example suite under `examples/sfincs_examples/`.
Key docs:

- `docs/fortran_examples.rst`
- `docs/parity.rst`

Reproduce the compatibility/parity artifacts:

```bash
python scripts/generate_fortran_example_output_status.py
python scripts/run_reduced_upstream_suite.py \
  --fortran-exe /Users/rogeriojorge/local/tests/sfincs/fortran/version3/sfincs \
  --reuse-fortran \
  --max-attempts 1 \
  --rtol 1e-4 \
  --atol 1e-9 \
  --jax-repeats 2
python scripts/generate_readme_reduced_suite_table.py
```

Latest reduced-suite snapshot (`rtol=1e-4`, `atol=1e-9`):

- **Practical parity:** 38/38 cases.
- **Strict parity:** 37/38 cases (strict deltas remain for `HSX_PASCollisions_DKESTrajectories`).
- **Print parity signals:** 38/38 cases.

Artifacts:

- `tests/reduced_upstream_examples/suite_report.json`
- `tests/reduced_upstream_examples/suite_report_strict.json`
- `docs/_generated/reduced_upstream_suite_status.rst`
- `docs/_generated/reduced_upstream_suite_status_strict.rst`

Reduced-suite comparison table (Fortran vs `sfincs_jax` runtimes, memory, and mismatches):

```bash
python scripts/generate_readme_reduced_suite_table.py
```

<!-- BEGIN REDUCED_SUITE_TABLE -->
| Case | Fortran(s) | sfincs_jax(s) | Fortran MB | sfincs_jax MB | Mismatches (practical/strict) | Print parity |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| HSX_FPCollisions_DKESTrajectories | 1.570 | 3.568 | 143.2 | 1853.7 | 0/193 (strict 0/193) | 9/9 |
| HSX_FPCollisions_fullTrajectories | 1.381 | 3.268 | 96.6 | 909.2 | 0/193 (strict 0/193) | 9/9 |
| HSX_PASCollisions_DKESTrajectories | 3.062 | - | - | - | 0/193 (strict 3/193) | 9/9 |
| HSX_PASCollisions_fullTrajectories | 0.122 | 3.020 | 103.3 | 608.7 | 0/193 (strict 0/193) | 9/9 |
| filteredW7XNetCDF_2species_magneticDrifts_noEr | 0.947 | 2.048 | 128.2 | 946.3 | 0/193 (strict 0/193) | 9/9 |
| filteredW7XNetCDF_2species_magneticDrifts_withEr | 0.774 | 2.277 | 122.4 | 548.2 | 0/193 (strict 0/193) | 9/9 |
| filteredW7XNetCDF_2species_noEr | 0.558 | 1.750 | 121.6 | 685.3 | 0/193 (strict 0/193) | 9/9 |
| geometryScheme4_1species_PAS_withEr_DKESTrajectories | 0.481 | - | 123.5 | - | 0/207 (strict 0/207) | 9/9 |
| geometryScheme4_2species_PAS_noEr | 1.645 | 2.962 | 114.3 | 901.2 | 0/207 (strict 0/207) | 9/9 |
| geometryScheme4_2species_noEr | 0.658 | 2.178 | 124.7 | 1557.8 | 0/207 (strict 0/207) | 9/9 |
| geometryScheme4_2species_noEr_withPhi1InDKE | 0.792 | 1.837 | 123.0 | 466.7 | 0/265 (strict 0/265) | 9/9 |
| geometryScheme4_2species_noEr_withQN | 0.772 | 1.645 | 115.1 | 440.6 | 0/265 (strict 0/265) | 9/9 |
| geometryScheme4_2species_withEr_fullTrajectories | 0.756 | 1.768 | 108.8 | 601.6 | 0/193 (strict 0/193) | 9/9 |
| geometryScheme4_2species_withEr_fullTrajectories_withQN | 1.324 | 1.943 | 117.1 | 508.2 | 0/251 (strict 0/251) | 9/9 |
| geometryScheme5_3species_loRes | 0.736 | 2.429 | 154.7 | 979.3 | 0/193 (strict 0/193) | 9/9 |
| inductiveE_noEr | 0.504 | 1.822 | 115.6 | 838.3 | 0/207 (strict 0/207) | 9/9 |
| monoenergetic_geometryScheme1 | 0.459 | 2.094 | 128.7 | 1522.7 | 0/203 (strict 0/203) | 9/9 |
| monoenergetic_geometryScheme11 | 2.812 | 3.138 | 205.2 | 538.4 | 0/208 (strict 0/208) | 9/9 |
| monoenergetic_geometryScheme5_ASCII | 0.684 | 1.973 | 164.0 | 1466.2 | 0/208 (strict 0/208) | 9/9 |
| monoenergetic_geometryScheme5_netCDF | 0.625 | 1.929 | 161.6 | 1461.2 | 0/208 (strict 0/208) | 9/9 |
| quick_2species_FPCollisions_noEr | 0.433 | 1.914 | 116.4 | 833.5 | 0/207 (strict 0/207) | 9/9 |
| sfincsPaperFigure3_geometryScheme11_FPCollisions_2Species_DKESTrajectories | 0.261 | 1.785 | 104.6 | 473.0 | 0/207 (strict 0/207) | 9/9 |
| sfincsPaperFigure3_geometryScheme11_FPCollisions_2Species_fullTrajectories | 0.446 | 2.282 | 115.6 | 814.4 | 0/207 (strict 0/207) | 9/9 |
| sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_DKESTrajectories | 0.125 | 3.027 | 111.6 | 820.2 | 0/207 (strict 0/207) | 9/9 |
| sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_fullTrajectories | 4.915 | 3.857 | - | 1169.8 | 0/207 (strict 0/207) | 9/9 |
| tokamak_1species_FPCollisions_noEr | 0.793 | 1.633 | 112.6 | 556.8 | 0/202 (strict 0/202) | 9/9 |
| tokamak_1species_FPCollisions_noEr_withPhi1InDKE | 1.547 | 1.997 | 143.6 | 463.9 | 0/275 (strict 0/275) | 9/9 |
| tokamak_1species_FPCollisions_noEr_withQN | 0.866 | 1.811 | 111.8 | 458.7 | 0/275 (strict 0/275) | 9/9 |
| tokamak_1species_FPCollisions_withEr_DKESTrajectories | 0.390 | 1.578 | 108.3 | 422.3 | 0/214 (strict 0/214) | 9/9 |
| tokamak_1species_FPCollisions_withEr_fullTrajectories | 1.398 | 2.413 | 206.6 | 1475.3 | 0/214 (strict 0/214) | 9/9 |
| tokamak_1species_PASCollisions_noEr | 1.642 | 3.152 | 553.0 | 1026.2 | 0/140 (strict 0/140) | 7/7 |
| tokamak_1species_PASCollisions_noEr_Nx1 | 0.124 | 2.768 | 110.9 | 737.6 | 0/212 (strict 0/212) | 9/9 |
| tokamak_1species_PASCollisions_noEr_withQN | 0.691 | 2.845 | 176.1 | 638.3 | 0/275 (strict 0/275) | 9/9 |
| tokamak_1species_PASCollisions_withEr_fullTrajectories | 0.434 | 2.025 | 124.0 | 836.7 | 0/212 (strict 0/212) | 9/9 |
| tokamak_2species_PASCollisions_noEr | 0.074 | 1.795 | 100.6 | 642.1 | 0/212 (strict 0/212) | 9/9 |
| tokamak_2species_PASCollisions_withEr_fullTrajectories | 0.887 | 3.783 | 169.6 | 1120.4 | 0/212 (strict 0/212) | 9/9 |
| transportMatrix_geometryScheme11 | 0.209 | 1.882 | 120.2 | 993.1 | 0/194 (strict 0/194) | 9/9 |
| transportMatrix_geometryScheme2 | 0.189 | 1.721 | 115.9 | 840.5 | 0/194 (strict 0/194) | 9/9 |
<!-- END REDUCED_SUITE_TABLE -->


## Documentation

Build locally:

```bash
sphinx-build -b html -W docs docs/_build/html
```

Core pages:

- `docs/normalizations.rst`
- `docs/system_equations.rst`
- `docs/method.rst`
- `docs/inputs.rst`
- `docs/outputs.rst`
- `docs/performance.rst`
- `docs/examples.rst`

## Testing

Run the full test suite:

```bash
pytest -q
```

## License

See `LICENSE`.
