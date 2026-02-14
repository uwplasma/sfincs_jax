# sfincs_jax

`sfincs_jax` is a JAX implementation of the SFINCS Fortran v3 workflow, with matrix-free operators,
JIT acceleration, and end-to-end differentiable components for sensitivity and optimization studies.
Default RHSMode=1 linear solves use a short-recurrence Krylov method (BiCGStab) with GMRES fallback,
while RHSMode=2/3 transport solves default to GMRES for robustness. Implicit differentiation is
enabled by default for linear solves.

![SFINCS vs sfincs_jax L11 parity and runtime](docs/_static/figures/sfincs_vs_sfincs_jax_l11_runtime_2x2.png)

Top figure: four monoenergetic test cases (`geometryScheme=1`, `11`, `12`, and filtered `5`) comparing
`L11 = transportMatrix[0,0]` and per-run runtime.
For the JAX path, runtime excludes compilation (warm-up run excluded).

Reproduce the figure and JSON summary:

```bash
python examples/performance/benchmark_transport_l11_vs_fortran.py --repeats 4
```

Run against a live local Fortran executable:

```bash
python examples/performance/benchmark_transport_l11_vs_fortran.py \
  --fortran-exe /Users/rogeriojorge/local/tests/sfincs/fortran/version3/sfincs \
  --repeats 4
```

Current benchmark snapshot (live Fortran executable, 4 repeats, JAX runtime excludes compilation):

| Case | Fortran mean (s/run) | sfincs_jax mean (s/run) | Max \|ΔL11\| |
| --- | ---: | ---: | ---: |
| scheme1 | 0.0797 | 0.1545 | 5.01e-14 |
| scheme11 | 0.2301 | 0.1839 | 1.46e-15 |
| scheme12 | 1.2328 | 0.1765 | 1.50e-07 |
| scheme5_filtered | 0.1041 | 0.1767 | 7.33e-16 |

Live snapshot notes: the local Fortran binary was built against a PETSc MPIUNI (serial, no MUMPS) configuration to avoid MPI init issues in sandboxed runs.

Outputs are written to:

- `examples/performance/output/transport_l11_vs_fortran/sfincs_vs_sfincs_jax_l11_runtime_2x2.png`
- `examples/performance/output/transport_l11_vs_fortran/sfincs_vs_sfincs_jax_l11_runtime_2x2.json`

Persistent-cache compile/runtime split (same four cases):

```bash
python examples/performance/profile_transport_compile_runtime_cache.py --repeats 3
```

- Figure: `examples/performance/output/compile_runtime_cache/transport_compile_runtime_cache_2x2.png`
- JSON: `examples/performance/output/compile_runtime_cache/transport_compile_runtime_cache_2x2.json`

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

## What the code supports

- v3 grid construction (`theta`, `zeta`, Stieltjes/polynomial `x`, monoenergetic `x=1` path)
- Geometry pipelines for `geometryScheme in {1,2,4,5,11,12}`
- Matrix-free v3 full-system operator, RHS, and residual assembly in JAX
- Linear solves via BiCGStab (default for RHSMode=1) with GMRES fallback; transport-matrix
  (`RHSMode=2/3`) loops default to GMRES
- Implicit-diff linear solves via `jax.lax.custom_linear_solve` (default for RHSMode=1 + transport)
- Transport-matrix recycling warm starts (optional, `SFINCS_JAX_TRANSPORT_RECYCLE_K`)
- `sfincsOutput.h5` writing from Python and CLI
- Parity tests against frozen Fortran fixtures (PETSc binaries and `sfincsOutput.h5`)
- Differentiable operator and solve-adjacent workflows (including implicit-diff helper APIs)

Detailed parity inventory and dataset coverage:

- `docs/parity.rst`
- `docs/outputs.rst`
- `docs/fortran_examples.rst`

## Quick start (Python)

```python
from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.v3 import grids_from_namelist, geometry_from_namelist

nml = read_sfincs_input("input.namelist")
grids = grids_from_namelist(nml)
geometry = geometry_from_namelist(nml=nml, grids=grids)
print(geometry.b_hat.shape)
```

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

## Upstream SFINCS compatibility and parity status

The repository vendors the upstream Fortran v3 example suite under `examples/sfincs_examples/`.
A generated status table reports, for every upstream example input:

1. whether `sfincs_jax` writes an output file for that input,
2. whether exact output parity for that exact input is verified in-repo,
3. and the reason when parity is not currently verified.

See:

- `docs/fortran_examples.rst`
- `docs/_generated/fortran_examples_output_status.rst`

Regenerate that table:

```bash
python scripts/generate_fortran_example_output_status.py
```

For fast parity iteration on reduced-resolution copies of the full upstream suite:

```bash
python scripts/run_reduced_upstream_suite.py --timeout-s 30 --max-attempts 1
```

The reduced-suite runner enables a persistent JAX compilation cache by default at
`tests/reduced_upstream_examples/.jax_compilation_cache` (override with `--jax-cache-dir`).

Target a specific case while preserving the 30s adaptive policy:

```bash
python scripts/run_reduced_upstream_suite.py --pattern 'geometryScheme5_3species_loRes' --timeout-s 30 --max-attempts 1
```

The latest reduced-suite status table is written to:

- `docs/_generated/reduced_upstream_suite_status.rst`
- `docs/_generated/reduced_upstream_suite_status_strict.rst`

And machine-readable reports are written to:

- `tests/reduced_upstream_examples/suite_report.json` (practical)
- `tests/reduced_upstream_examples/suite_report_strict.json` (strict)

Current reduced-suite snapshot (fixture baseline):

- **Practical:** 35/38 parity_ok (3 remaining mismatches).
- **Strict:** 35/38 parity_ok (see `docs/_generated/reduced_upstream_suite_status_strict.rst`).
- **Print parity:** 38/38.

Remaining mismatches (reduced suite, rtol=5e-4):

- `monoenergetic_geometryScheme1` (solver-branch mismatch)
- `transportMatrix_geometryScheme11` (output-field mismatch)
- `filteredW7XNetCDF_2species_noEr` (solver-branch mismatch)

### Reduced-suite outputs and mismatches (all upstream examples, reduced resolution)

Each reduced-resolution upstream example produces the following outputs for the listed input:

- Fortran: `sfincsOutput.h5`, `sfincs.log`
- sfincs_jax: `sfincsOutput_jax.h5`, `sfincs_jax.log`

The table below enumerates every upstream example in the reduced suite, the outputs produced,
the Fortran vs `sfincs_jax` runtimes, and the number of mismatches relative to Fortran output
(`bad/total`, rtol=5e-4, atol=1e-8). Stdout print parity signals are currently 38/38 across the reduced suite.

Regenerate the README table after a new reduced-suite run:

```bash
python scripts/generate_readme_reduced_suite_table.py
```

<!-- BEGIN REDUCED_SUITE_TABLE -->
| Case | Fortran outputs | sfincs_jax outputs | Fortran(s) | sfincs_jax(s) | Mismatches (practical/strict) | Print parity |
| --- | --- | --- | ---: | ---: | --- | --- |
| HSX_FPCollisions_DKESTrajectories | sfincsOutput.h5, sfincs.log | sfincsOutput_jax.h5, sfincs_jax.log | 1.100 | 23.517 | 0/123 (strict 0/123) | 7/7 |
| HSX_FPCollisions_fullTrajectories | sfincsOutput.h5, sfincs.log | sfincsOutput_jax.h5, sfincs_jax.log | 0.197 | 24.466 | 0/123 (strict 0/123) | 7/7 |
| HSX_PASCollisions_DKESTrajectories | sfincsOutput.h5, sfincs.log | sfincsOutput_jax.h5, sfincs_jax.log | 0.190 | 17.106 | 0/123 (strict 0/123) | 7/7 |
| HSX_PASCollisions_fullTrajectories | sfincsOutput.h5, sfincs.log | sfincsOutput_jax.h5, sfincs_jax.log | 0.143 | 27.789 | 0/123 (strict 0/123) | 7/7 |
| filteredW7XNetCDF_2species_magneticDrifts_noEr | sfincsOutput.h5, sfincs.log | sfincsOutput_jax.h5, sfincs_jax.log | 1.974 | 17.936 | 0/123 (strict 0/123) | 7/7 |
| filteredW7XNetCDF_2species_magneticDrifts_withEr | sfincsOutput.h5, sfincs.log | sfincsOutput_jax.h5, sfincs_jax.log | 3.542 | 15.489 | 0/123 (strict 0/123) | 7/7 |
| filteredW7XNetCDF_2species_noEr | sfincsOutput.h5, sfincs.log | sfincsOutput_jax.h5, sfincs_jax.log | - | 21.570 | 1/193 (strict 1/193) | 9/9 |
| geometryScheme4_1species_PAS_withEr_DKESTrajectories | sfincsOutput.h5, sfincs.log | sfincsOutput_jax.h5, sfincs_jax.log | 0.087 | 18.076 | 0/123 (strict 0/123) | 7/7 |
| geometryScheme4_2species_PAS_noEr | sfincsOutput.h5, sfincs.log | sfincsOutput_jax.h5, sfincs_jax.log | 0.074 | 22.088 | 0/123 (strict 0/123) | 7/7 |
| geometryScheme4_2species_noEr | sfincsOutput.h5, sfincs.log | sfincsOutput_jax.h5, sfincs_jax.log | 0.403 | 19.500 | 0/123 (strict 0/123) | 7/7 |
| geometryScheme4_2species_noEr_withPhi1InDKE | sfincsOutput.h5, sfincs.log | sfincsOutput_jax.h5, sfincs_jax.log | 0.084 | 2.083 | 0/124 (strict 0/124) | 7/7 |
| geometryScheme4_2species_noEr_withQN | sfincsOutput.h5, sfincs.log | sfincsOutput_jax.h5, sfincs_jax.log | 0.089 | 1.782 | 0/124 (strict 0/124) | 7/7 |
| geometryScheme4_2species_withEr_fullTrajectories | sfincsOutput.h5, sfincs.log | sfincsOutput_jax.h5, sfincs_jax.log | 0.074 | 15.613 | 0/123 (strict 0/123) | 7/7 |
| geometryScheme4_2species_withEr_fullTrajectories_withQN | sfincsOutput.h5, sfincs.log | sfincsOutput_jax.h5, sfincs_jax.log | 0.094 | 1.750 | 0/124 (strict 0/124) | 7/7 |
| geometryScheme5_3species_loRes | sfincsOutput.h5, sfincs.log | sfincsOutput_jax.h5, sfincs_jax.log | 0.133 | 13.014 | 0/123 (strict 0/123) | 7/7 |
| inductiveE_noEr | sfincsOutput.h5, sfincs.log | sfincsOutput_jax.h5, sfincs_jax.log | 0.077 | 6.642 | 0/123 (strict 0/123) | 7/7 |
| monoenergetic_geometryScheme1 | sfincsOutput.h5, sfincs.log | sfincsOutput_jax.h5, sfincs_jax.log | - | 1.880 | 39/203 (strict 39/203) | 9/9 |
| monoenergetic_geometryScheme11 | sfincsOutput.h5, sfincs.log | sfincsOutput_jax.h5, sfincs_jax.log | 0.237 | 2.833 | 0/125 (strict 0/125) | 7/7 |
| monoenergetic_geometryScheme5_ASCII | sfincsOutput.h5, sfincs.log | sfincsOutput_jax.h5, sfincs_jax.log | 0.076 | 2.138 | 0/124 (strict 0/125) | 7/7 |
| monoenergetic_geometryScheme5_netCDF | sfincsOutput.h5, sfincs.log | sfincsOutput_jax.h5, sfincs_jax.log | 0.134 | 2.196 | 0/124 (strict 0/125) | 7/7 |
| quick_2species_FPCollisions_noEr | sfincsOutput.h5, sfincs.log | sfincsOutput_jax.h5, sfincs_jax.log | 0.126 | 5.998 | 0/123 (strict 0/123) | 7/7 |
| sfincsPaperFigure3_geometryScheme11_FPCollisions_2Species_DKESTrajectories | sfincsOutput.h5, sfincs.log | sfincsOutput_jax.h5, sfincs_jax.log | 0.187 | 20.592 | 0/123 (strict 0/123) | 7/7 |
| sfincsPaperFigure3_geometryScheme11_FPCollisions_2Species_fullTrajectories | sfincsOutput.h5, sfincs.log | sfincsOutput_jax.h5, sfincs_jax.log | 0.134 | 17.729 | 0/123 (strict 0/123) | 7/7 |
| sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_DKESTrajectories | sfincsOutput.h5, sfincs.log | sfincsOutput_jax.h5, sfincs_jax.log | 0.136 | 19.916 | 0/123 (strict 0/123) | 7/7 |
| sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_fullTrajectories | sfincsOutput.h5, sfincs.log | sfincsOutput_jax.h5, sfincs_jax.log | 0.140 | 22.719 | 0/123 (strict 0/123) | 7/7 |
| tokamak_1species_FPCollisions_noEr | sfincsOutput.h5, sfincs.log | sfincsOutput_jax.h5, sfincs_jax.log | 0.082 | 1.599 | 0/130 (strict 0/130) | 7/7 |
| tokamak_1species_FPCollisions_noEr_withPhi1InDKE | sfincsOutput.h5, sfincs.log | sfincsOutput_jax.h5, sfincs_jax.log | 0.502 | 2.123 | 0/136 (strict 0/136) | 7/7 |
| tokamak_1species_FPCollisions_noEr_withQN | sfincsOutput.h5, sfincs.log | sfincsOutput_jax.h5, sfincs_jax.log | 0.085 | 1.693 | 0/136 (strict 0/136) | 7/7 |
| tokamak_1species_FPCollisions_withEr_DKESTrajectories | sfincsOutput.h5, sfincs.log | sfincsOutput_jax.h5, sfincs_jax.log | 0.084 | 2.296 | 0/130 (strict 0/130) | 7/7 |
| tokamak_1species_FPCollisions_withEr_fullTrajectories | sfincsOutput.h5, sfincs.log | sfincsOutput_jax.h5, sfincs_jax.log | 0.126 | 2.683 | 0/130 (strict 0/130) | 7/7 |
| tokamak_1species_PASCollisions_noEr | sfincsOutput.h5, sfincs.log | sfincsOutput_jax.h5, sfincs_jax.log | 0.072 | 1.872 | 0/130 (strict 0/130) | 7/7 |
| tokamak_1species_PASCollisions_noEr_Nx1 | sfincsOutput.h5, sfincs.log | sfincsOutput_jax.h5, sfincs_jax.log | 0.083 | 1.782 | 0/130 (strict 0/130) | 7/7 |
| tokamak_1species_PASCollisions_noEr_withQN | sfincsOutput.h5, sfincs.log | sfincsOutput_jax.h5, sfincs_jax.log | 0.059 | 1.960 | 0/136 (strict 0/136) | 7/7 |
| tokamak_1species_PASCollisions_withEr_fullTrajectories | sfincsOutput.h5, sfincs.log | sfincsOutput_jax.h5, sfincs_jax.log | 0.130 | 2.733 | 0/130 (strict 0/130) | 7/7 |
| tokamak_2species_PASCollisions_noEr | sfincsOutput.h5, sfincs.log | sfincsOutput_jax.h5, sfincs_jax.log | 0.077 | 2.033 | 0/130 (strict 0/130) | 7/7 |
| tokamak_2species_PASCollisions_withEr_fullTrajectories | sfincsOutput.h5, sfincs.log | sfincsOutput_jax.h5, sfincs_jax.log | 0.075 | 15.239 | 0/130 (strict 0/130) | 7/7 |
| transportMatrix_geometryScheme11 | sfincsOutput.h5, sfincs.log | sfincsOutput_jax.h5, sfincs_jax.log | - | 5.755 | 4/194 (strict 4/194) | 9/9 |
| transportMatrix_geometryScheme2 | sfincsOutput.h5, sfincs.log | sfincsOutput_jax.h5, sfincs_jax.log | 0.079 | 1.607 | 0/123 (strict 0/123) | 7/7 |
<!-- END REDUCED_SUITE_TABLE -->


For operator-level parity diagnosis against Fortran PETSc matrices:

```bash
python scripts/compare_fortran_matrix_to_jax_operator.py \
  --input /path/to/input.namelist \
  --fortran-matrix /path/to/sfincsBinary_iteration_000_whichMatrix_3 \
  --fortran-state /path/to/sfincsBinary_iteration_000_stateVector \
  --project-active-dofs \
  --out-json matrix_compare.json
```

For RHSMode=1 diagnostics isolation on a frozen state vector (to separate solver-branch
differences from postprocessing/diagnostic formulas):

```bash
python scripts/compare_rhsmode1_diagnostics_from_state.py \
  --input /path/to/input.namelist \
  --state /path/to/sfincsBinary_iteration_000_stateVector \
  --fortran-h5 /path/to/sfincsOutput.h5 \
  --out-json diagnostics_from_frozen_state.json
```

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
