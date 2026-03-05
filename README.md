# sfincs_jax

`sfincs_jax` is a JAX implementation of the SFINCS Fortran v3 workflow, with matrix-free operators,
JIT acceleration, and end-to-end differentiable components for sensitivity and optimization studies.
Default RHSMode=1 linear solves use GMRES (incremental, Fortran-comparison-first) with stage-2 fallback,
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

Parallel `whichRHS` scaling for a >2-minute RHSMode=2 transport-matrix case
(`examples/performance/transport_parallel_2min.input.namelist`, geometryScheme=2,
`Ntheta=21`, `Nzeta=21`, `Nxi=6`, `NL=6`, `Nx=6`), measured on 1–4 workers.

![Parallel whichRHS scaling](docs/_static/figures/parallel/transport_parallel_scaling.png)

Latest cache-warm run (1–4 workers): 1 worker 147.4s, 2 workers 122.3s,
3 workers 115.8s, 4 workers 114.8s.

Enable parallel execution in normal runs (no environment variables required):

```bash
sfincs_jax --cores 4 /path/to/input.namelist
```

`whichRHS` process pools are persistent by default (`SFINCS_JAX_TRANSPORT_POOL_PERSIST=1`)
to reduce warm-run overhead across repeated transport solves.

Reproduce the 1-5 worker scaling figure and JSON summary:

```bash
python examples/performance/benchmark_transport_parallel_scaling.py \
  --input examples/performance/transport_parallel_2min.input.namelist \
  --workers 1 2 3 4 \
  --repeats 1 \
  --warmup 0 \
  --global-warmup 1
```

The transport scaling benchmark uses the solve-only path
(`collect_transport_output_fields=False`) so runtime reflects linear-solve
scaling rather than H5-output diagnostics assembly.

Experimental transport domain-decomposition preconditioners are available via
`SFINCS_JAX_TRANSPORT_PRECOND=theta_dd` or `zeta_dd` (block sizes:
`SFINCS_JAX_TRANSPORT_DD_BLOCK_T`, `SFINCS_JAX_TRANSPORT_DD_BLOCK_Z`), with
overlap-RAS variants `theta_schwarz` / `zeta_schwarz`
(`SFINCS_JAX_TRANSPORT_DD_OVERLAP`).

Single-RHS sharded solves now default to a communication-reduced distributed
Krylov preference (`SFINCS_JAX_DISTRIBUTED_KRYLOV=auto`, BiCGStab-first with
GMRES available via `SFINCS_JAX_DISTRIBUTED_KRYLOV=gmres`), but still need
additional communication-avoiding Krylov work to show strong scaling on
5+ CPU devices. On the current long benchmark
(`examples/performance/rhsmode1_sharded_scaling.input.namelist`, `nsolve=240`,
baseline 136.25 s at 1 core), measured sharded solve times are 98.70 s (2 cores),
105.33 s (3 cores), and 110.78 s (4 cores).

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
- Fortran-comparison tests against frozen Fortran fixtures (PETSc binaries and `sfincsOutput.h5`)
- Differentiable operator and solve-adjacent workflows (including implicit-diff helper APIs)

Detailed Fortran-comparison inventory and dataset coverage:

- `docs/fortran_comparison.rst`
- `docs/outputs.rst`
- `docs/fortran_examples.rst`

## Solver defaults (comparison + performance)

- RHSMode=1 dense fallback now defaults to `SFINCS_JAX_RHSMODE1_DENSE_FALLBACK_MAX=400`, with higher
  ceilings `SFINCS_JAX_RHSMODE1_DENSE_FP_MAX=5000` (full FP) and
  `SFINCS_JAX_RHSMODE1_DENSE_PAS_MAX=5000` (PAS/constraintScheme=2) to recover Fortran convergence
  when Krylov stagnates.
- includePhi1 runs use full Newton updates by default (matching v3). Frozen linearization is
  opt‑in via `SFINCS_JAX_PHI1_USE_FROZEN_LINEARIZATION`, and small Phi1‑collision systems use a dense
  Newton step when `SFINCS_JAX_PHI1_NK_DENSE_CUTOFF` is met. Large qn-only includePhi1 systems
  auto-relax Newton absolute tolerance to avoid an extra nonlinear step
  (`SFINCS_JAX_PHI1_NEWTON_TOL` overrides).
- CLI auto-core selection now prefers 1 core for RHSMode=1 solves and up to 3 cores for
  RHSMode=2/3 transport runs when `--cores` / `SFINCS_JAX_CORES` are unset.
- Transport-matrix dense retries are capped by `SFINCS_JAX_TRANSPORT_DENSE_MAX_MB=128` to avoid
  excessive memory usage.
- PAS constraint projection auto-enables for tokamak-like `N_zeta=1` cases and DKES trajectories
  to eliminate nullspace drift (see `SFINCS_JAX_PAS_PROJECT_CONSTRAINTS`).
- `SFINCS_JAX_DENSE_MAX=8000` caps direct dense solves to avoid runaway memory use.
- VMEC geometryScheme=5 full‑FP comparisons use dedicated near‑zero tolerances for flow/pressure
  diagnostics; strict tables still pass at reduced-suite tolerances (see `docs/fortran_comparison.rst`).

Detailed performance profiling and advanced parallel modes are documented in
`docs/performance.rst` and `docs/parallelism.rst`.

## CLI

Run a SFINCS input file (default mode, matches Fortran v3 behavior):

```bash
sfincs_jax /path/to/input.namelist
```

Write SFINCS-style output explicitly:

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
- `examples/parity/`: fixture comparison and validation runs
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

## Upstream SFINCS compatibility and comparison status

The repository vendors the upstream Fortran v3 example suite under `examples/sfincs_examples/`.
Key docs:

- `docs/fortran_examples.rst`
- `docs/fortran_comparison.rst`

Reproduce the compatibility/comparison artifacts:

```bash
python scripts/generate_fortran_example_output_status.py
python scripts/run_reduced_upstream_suite.py \
  --fortran-exe /path/to/sfincs \
  --reuse-fortran \
  --max-attempts 1 \
  --rtol 5e-4 \
  --atol 1e-9 \
  --jax-repeats 2
python scripts/generate_readme_reduced_suite_table.py
```

Latest reduced-suite snapshot (`rtol=5e-4`, `atol=1e-9`):

- **Practical comparison:** 19/38 cases.
- **Strict comparison:** 19/38 cases.
- **Print comparison signals:** 29/38 cases.

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
| Case | Fortran CPU(s) | sfincs_jax CPU(s) | sfincs_jax GPU(s) | Fortran CPU MB | sfincs_jax CPU MB | sfincs_jax GPU MB | Mismatches (practical/strict) | Print comparison |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| HSX_FPCollisions_DKESTrajectories | 0.596 | 5.348 | 9.281 | 146.6 | 2932.1 | 2224.0 | 0/192 (strict 0/192) | 9/9 |
| HSX_FPCollisions_fullTrajectories | 2.702 | 5.110 | 10.901 | 99.9 | 1870.4 | 1743.2 | 0/192 (strict 0/192) | 9/9 |
| HSX_PASCollisions_DKESTrajectories | 2.778 | 61.428 | parity_ok | 346.0 | 5508.7 | parity_ok | 0/192 (strict 0/192) | 9/9 |
| HSX_PASCollisions_fullTrajectories | 32.903 | 196.472 | max_attempts | 467.0 | 3053.3 | max_attempts | 0/192 (strict 3/192) | 9/9 |
| filteredW7XNetCDF_2species_magneticDrifts_noEr | 3.642 | 3.797 | 4.901 | 138.0 | 558.7 | 1435.8 | 0/192 (strict 0/192) | 9/9 |
| filteredW7XNetCDF_2species_magneticDrifts_withEr | 4.337 | 3.782 | 5.457 | 137.5 | 595.7 | 1456.1 | 0/192 (strict 0/192) | 9/9 |
| filteredW7XNetCDF_2species_noEr | 5.021 | 2.761 | 3.993 | 136.0 | 819.5 | 1416.1 | 0/192 (strict 0/192) | 9/9 |
| geometryScheme4_1species_PAS_withEr_DKESTrajectories | 8.257 | 2.674 | max_attempts | 484.6 | 994.9 | max_attempts | 0/207 (strict 0/207) | 9/9 |
| geometryScheme4_2species_PAS_noEr | 0.355 | 3.498 | 7.063 | 139.0 | 878.9 | 1683.3 | 0/207 (strict 0/207) | 9/9 |
| geometryScheme4_2species_noEr | 0.300 | 3.475 | 6.010 | 134.5 | 1582.5 | 1999.0 | 0/206 (strict 0/206) | 9/9 |
| geometryScheme4_2species_noEr_withPhi1InDKE | 0.123 | 2.698 | 4.442 | 129.6 | 482.8 | 1417.9 | 0/264 (strict 0/264) | 9/9 |
| geometryScheme4_2species_noEr_withQN | 0.062 | 2.335 | 3.737 | 112.3 | 456.1 | 1400.2 | 0/264 (strict 0/264) | 9/9 |
| geometryScheme4_2species_withEr_fullTrajectories | 0.074 | 2.847 | 4.143 | 118.4 | 792.3 | 1417.6 | 0/192 (strict 0/192) | 9/9 |
| geometryScheme4_2species_withEr_fullTrajectories_withQN | 0.080 | 3.099 | 4.345 | 117.0 | 589.6 | 1438.4 | 0/250 (strict 0/250) | 9/9 |
| geometryScheme5_3species_loRes | 1.734 | 4.378 | 9.824 | 163.1 | 1648.6 | 1845.9 | 0/192 (strict 0/192) | 9/9 |
| inductiveE_noEr | 0.167 | 2.696 | 3.742 | 129.0 | 827.0 | 1439.3 | 0/206 (strict 0/206) | 9/9 |
| monoenergetic_geometryScheme1 | 0.612 | 4.465 | jax_error | 133.7 | 281.4 | jax_error | 0/203 (strict 4/203) | 7/9 |
| monoenergetic_geometryScheme11 | 2.665 | 6.405 | jax_error | 204.3 | 300.1 | jax_error | 0/207 (strict 0/207) | 7/9 |
| monoenergetic_geometryScheme5_ASCII | 0.978 | 5.228 | jax_error | 158.8 | 299.2 | jax_error | 0/205 (strict 2/206) | 7/9 |
| monoenergetic_geometryScheme5_netCDF | 0.971 | 5.048 | jax_error | 151.1 | 308.2 | jax_error | 0/205 (strict 2/206) | 7/9 |
| quick_2species_FPCollisions_noEr | 0.315 | 2.513 | 4.095 | 125.4 | 838.7 | 1438.6 | 0/206 (strict 0/206) | 9/9 |
| sfincsPaperFigure3_geometryScheme11_FPCollisions_2Species_DKESTrajectories | 0.065 | 1.926 | 3.743 | 113.9 | 598.8 | 1410.2 | 0/207 (strict 0/207) | 9/9 |
| sfincsPaperFigure3_geometryScheme11_FPCollisions_2Species_fullTrajectories | 0.173 | 3.077 | 4.649 | 126.8 | 879.4 | 1476.4 | 0/206 (strict 0/206) | 9/9 |
| sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_DKESTrajectories | 2.452 | 22.427 | 218.818 | 263.1 | 2407.6 | 2382.4 | 0/206 (strict 0/206) | 9/9 |
| sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_fullTrajectories | 4.769 | 8.444 | 59.438 | 375.5 | 1947.9 | 2374.1 | 0/206 (strict 0/206) | 9/9 |
| tokamak_1species_FPCollisions_noEr | 9.724 | 1.934 | 3.592 | 132.8 | 732.0 | 1386.7 | 0/187 (strict 12/187) | 9/9 |
| tokamak_1species_FPCollisions_noEr_withPhi1InDKE | 13.340 | 2.313 | 4.496 | 141.0 | 468.3 | 1410.7 | 0/274 (strict 0/274) | 9/9 |
| tokamak_1species_FPCollisions_noEr_withQN | 6.150 | 2.238 | 3.641 | 127.1 | 520.7 | 1415.7 | 0/274 (strict 0/274) | 9/9 |
| tokamak_1species_FPCollisions_withEr_DKESTrajectories | 4.400 | 2.062 | 3.239 | 116.8 | 527.0 | 1409.7 | 0/213 (strict 0/213) | 9/9 |
| tokamak_1species_FPCollisions_withEr_fullTrajectories | 55.728 | 4.194 | 8.477 | 334.2 | 1759.9 | 2046.6 | 0/142 (strict 0/142) | 7/7 |
| tokamak_1species_PASCollisions_noEr | 2.301 | 2.695 | 30.843 | 718.5 | 560.4 | 1684.3 | 0/140 (strict 0/140) | 7/7 |
| tokamak_1species_PASCollisions_noEr_Nx1 | 2.124 | 39.621 | 81.066 | 250.9 | 5260.5 | 3227.2 | 0/212 (strict 33/212) | 9/9 |
| tokamak_1species_PASCollisions_noEr_withQN | 4.851 | 106.650 | 127.487 | 389.6 | 556.9 | 1987.1 | 0/274 (strict 0/274) | 9/9 |
| tokamak_1species_PASCollisions_withEr_fullTrajectories | 49.530 | 11.348 | max_attempts | 574.4 | 1302.0 | max_attempts | 0/212 (strict 0/212) | 9/9 |
| tokamak_2species_PASCollisions_noEr | 5.667 | 3.922 | 35.621 | 478.3 | 3372.4 | 2465.3 | 0/212 (strict 0/212) | 9/9 |
| tokamak_2species_PASCollisions_withEr_fullTrajectories | 15.875 | 180.320 | 166.946 | 442.1 | 1732.1 | 1843.6 | 0/212 (strict 1/212) | 9/9 |
| transportMatrix_geometryScheme11 | 0.303 | 3.584 | jax_error | 129.2 | 256.3 | jax_error | 0/193 (strict 0/193) | 7/9 |
| transportMatrix_geometryScheme2 | 0.236 | 3.579 | jax_error | 118.8 | 251.6 | jax_error | 0/193 (strict 0/193) | 7/9 |
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
