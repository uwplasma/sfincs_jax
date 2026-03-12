# sfincs_jax

`sfincs_jax` is a JAX implementation of SFINCS v3 that solves the same neoclassical drift-kinetic problem with matching normalizations, geometry conventions, and output format (`sfincsOutput.h5`).

It is designed for:

- high-performance runs on CPU/GPU,
- memory-efficient large solves,
- end-to-end differentiable workflows.

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

## Quick Start (Python)

Read a namelist, run `sfincs_jax`, write `sfincsOutput.h5`, and inspect results directly in memory:

```python
from pathlib import Path

from sfincs_jax.io import write_sfincs_jax_output_h5

input_namelist = Path("input.namelist")
out_path, results = write_sfincs_jax_output_h5(
    input_namelist=input_namelist,
    output_path=Path("sfincsOutput.h5"),
    return_results=True,
)

print("Wrote:", out_path)
print("Available datasets:", len(results))
print("Example key:", "particleFlux_vm_psiHat" in results)
```

`sfincs_jax write-output` and `write_sfincs_jax_output_h5(...)` use the fast explicit
solve path by default. Request the implicit/differentiable linear-solve path only when
you need it:

```python
write_sfincs_jax_output_h5(
    input_namelist=input_namelist,
    output_path=Path("sfincsOutput.h5"),
    differentiable=True,
)
```

## Executable (CLI)

You can run `sfincs_jax` from anywhere in your terminal. You do not need to be inside the repository folder.

Run an input file (default behavior, same invocation style as Fortran SFINCS):

```bash
sfincs_jax /path/to/input.namelist
```

Write output explicitly:

```bash
sfincs_jax write-output --input /path/to/input.namelist --out /path/to/sfincsOutput.h5
```

Compare two outputs:

```bash
sfincs_jax compare-h5 --a sfincsOutput_jax.h5 --b sfincsOutput_fortran.h5
```

Advanced CLI/solver options are documented in `docs/usage.rst` and `docs/performance_techniques.rst`.

## Reduced-Suite Comparison (Fortran v3 vs sfincs_jax)

Reproduce the table:

```bash
python scripts/run_reduced_upstream_suite.py \
  --fortran-exe /path/to/sfincs \
  --reuse-fortran \
  --max-attempts 1 \
  --rtol 5e-4 \
  --atol 1e-9 \
  --jax-repeats 2
python scripts/generate_readme_reduced_suite_table.py
```

Artifacts:

- `tests/reduced_upstream_examples/suite_report.json`
- `tests/reduced_upstream_examples/suite_report_strict.json`
- `docs/_generated/reduced_upstream_suite_status.rst`
- `docs/_generated/reduced_upstream_suite_status_strict.rst`

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

Status labels in table cells:

- `max_attempts`: the suite runner retried/rescaled this case up to `--max-attempts` and still did not complete a successful comparison run.
- `jax_error`: the JAX run exited with an exception for that benchmark lane/case.

## Fast Explicit Branch Audit

Regenerate this block on the fast-path branch with:

```bash
python scripts/run_scaled_example_suite.py \
  --examples-root examples/sfincs_examples \
  --resolution-reference-root /Users/rogeriojorge/local/tests/sfincs_original/fortran/version3/examples \
  --fortran-exe /Users/rogeriojorge/local/tests/sfincs/fortran/version3/sfincs \
  --out-root tests/scaled_example_suite_fast_cpu_rtwindow_v1 \
  --scale-factor 1.0 \
  --runtime-target-basis fortran \
  --fortran-min-runtime-s 1.0 \
  --fortran-max-runtime-s 20.0 \
  --runtime-adjustment-iters 3
python scripts/generate_readme_fast_branch_audit.py \
  --out-root tests/scaled_example_suite_fast_cpu_rtwindow_v1
```

The benchmark policy on this branch is now:

- start from the original Fortran v3 example resolution,
- only downscale when a case is too expensive for a practical suite run,
- and never intentionally push a reduced case below about `1s` of Fortran wall time unless
  the original example is already that small.

That avoids the misleading sub-second Fortran rows that came from blind global downscaling.
The next full fast-branch README refresh should therefore come from the runtime-windowed
audit command above, not from the older fixed-scale partial reruns.

Since that partial audit, targeted fast-path reruns have already moved one of the listed
transport blockers: `transportMatrix_geometryScheme11` is now `parity_ok` on the branch
with explicit CPU sparse-LU factorization promoted to float64 on large transport solves
(`~185.3s`, `~5.17 GB` peak RSS on the stored scaled input). `geometryScheme4_2species_noEr`
has also moved materially: the default fast explicit CPU path now promotes the large sparse
rescue to exact host sparse-LU when the x-block seed is already strong on this x-coupled FP
case. On the stored scaled input that produces practical parity with only 4 tiny strict-only
velocity/Mach deltas, at about `456.7s` and about `8.7 GB` peak RSS. So the remaining known
fast-path mismatch is now concentrated primarily in `monoenergetic_geometryScheme1`; the
geometry4 CPU blocker is no longer on the old wrong-flow branch, though its memory cost is
still a real offender.

Additional targeted original-resolution work on `monoenergetic_geometryScheme1` has narrowed
that remaining mismatch substantially. The fast explicit branch now reproduces the dumped
Fortran Jacobian exactly on sampled columns and vectors, and the exact sparse solve of that
Jacobian lands on the same branch as the branch's host-GMRES fast path. In other words, the
remaining scheme-1 delta on this branch is no longer an operator-assembly bug. It is a real
solver-semantics divergence between:

- the fast explicit branch's true-residual solution of the dumped Jacobian system, and
- the original Fortran/PETSc lane's accepted preconditioned-residual iterate for this
  ill-conditioned monoenergetic transport case.

That means the next fast-branch step for `monoenergetic_geometryScheme1` is not more sparse
assembly work. It is an explicit policy decision about which solution concept the CLI/default
fast path should prefer for structurally singular or near-singular monoenergetic transport
systems.

<!-- BEGIN FAST_BRANCH_AUDIT -->
Current fast explicit CPU audit comes from `tests/scaled_example_suite_fast_cpu_rtwindow_v2`.

- Recorded cases: `39/39`
- Practical status counts: `parity_mismatch=19, parity_ok=20`
- Strict status counts: `parity_mismatch=19, parity_ok=20`
- Resolution policy: `reference_first_runtime_window, scale_factor=1.0, runtime_basis=fortran, fortran_min=1.0, fortran_max=20.0, adjust_iters=3`
- Remaining cases: none

Top CPU runtime offenders:
- `sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_DKESTrajectories`: jax=437.665s fortran=0.914s ratio=478.89x status=parity_mismatch, res={'NTHETA': 8, 'NZETA': 17, 'NX': 3, 'NXI': 17}
- `geometryScheme4_1species_PAS_withEr_DKESTrajectories`: jax=381.480s fortran=1.334s ratio=286.01x status=parity_mismatch, res={'NTHETA': 11, 'NZETA': 15, 'NX': 4, 'NXI': 32}
- `HSX_FPCollisions_fullTrajectories`: jax=252.609s fortran=6.600s ratio=38.27x status=parity_mismatch, res={'NTHETA': 5, 'NZETA': 39, 'NX': 2, 'NXI': 50}
- `filteredW7XNetCDF_2species_magneticDrifts_noEr`: jax=233.237s fortran=3.434s ratio=67.91x status=parity_mismatch, res={'NTHETA': 8, 'NZETA': 20, 'NX': 3, 'NXI': 38}
- `HSX_FPCollisions_DKESTrajectories`: jax=229.402s fortran=2.885s ratio=79.51x status=parity_mismatch, res={'NTHETA': 5, 'NZETA': 39, 'NX': 2, 'NXI': 50}

Top CPU memory offenders:
- `HSX_PASCollisions_fullTrajectories`: jax=5268.0 MB fortran=305.4 MB ratio=17.25x status=parity_mismatch, res={'NTHETA': 5, 'NZETA': 39, 'NX': 2, 'NXI': 50}
- `monoenergetic_geometryScheme11`: jax=4256.9 MB fortran=143.4 MB ratio=29.68x status=parity_ok, res={'NTHETA': 12, 'NZETA': 21, 'NX': 1, 'NXI': 17}
- `transportMatrix_geometryScheme11`: jax=4222.7 MB fortran=164.1 MB ratio=25.74x status=parity_ok, res={'NTHETA': 9, 'NZETA': 21, 'NX': 4, 'NXI': 17}
- `tokamak_1species_FPCollisions_withEr_fullTrajectories`: jax=3475.7 MB fortran=142.9 MB ratio=24.32x status=parity_ok, res={'NTHETA': 21, 'NZETA': 1, 'NX': 8, 'NXI': 31}
- `geometryScheme5_3species_loRes`: jax=3453.6 MB fortran=196.0 MB ratio=17.62x status=parity_ok, res={'NTHETA': 8, 'NZETA': 11, 'NX': 3, 'NXI': 12}

Current mismatches:
- `HSX_FPCollisions_DKESTrajectories`: status=parity_mismatch, practical=39/193, strict=39/193, sample=FSABFlow,FSABFlow_vs_x,FSABVelocityUsingFSADensity,FSABVelocityUsingFSADensityOverB0
- `HSX_FPCollisions_fullTrajectories`: status=parity_mismatch, practical=40/193, strict=40/193, sample=FSABFlow,FSABFlow_vs_x,FSABVelocityUsingFSADensity,FSABVelocityUsingFSADensityOverB0
- `HSX_PASCollisions_fullTrajectories`: status=parity_mismatch, practical=33/193, strict=33/193, sample=FSABFlow,FSABFlow_vs_x,FSABVelocityUsingFSADensity,FSABVelocityUsingFSADensityOverB0
- `additional_examples`: status=parity_mismatch, practical=32/193, strict=32/193, sample=FSABFlow,FSABFlow_vs_x,FSABVelocityUsingFSADensity,FSABVelocityUsingFSADensityOverB0
- `filteredW7XNetCDF_2species_magneticDrifts_noEr`: status=parity_mismatch, practical=37/193, strict=37/193, sample=FSABFlow,FSABFlow_vs_x,FSABVelocityUsingFSADensity,FSABVelocityUsingFSADensityOverB0
- `filteredW7XNetCDF_2species_magneticDrifts_withEr`: status=parity_mismatch, practical=37/193, strict=37/193, sample=FSABFlow,FSABFlow_vs_x,FSABVelocityUsingFSADensity,FSABVelocityUsingFSADensityOverB0
- `filteredW7XNetCDF_2species_noEr`: status=parity_mismatch, practical=33/193, strict=33/193, sample=FSABFlow,FSABFlow_vs_x,FSABVelocityUsingFSADensity,FSABVelocityUsingFSADensityOverB0
- `geometryScheme4_1species_PAS_withEr_DKESTrajectories`: status=parity_mismatch, practical=42/207, strict=42/207, sample=FSABFlow,FSABFlow_vs_x,FSABVelocityUsingFSADensity,FSABVelocityUsingFSADensityOverB0
- `geometryScheme4_2species_noEr`: status=parity_mismatch, practical=37/207, strict=37/207, sample=FSABFlow,FSABFlow_vs_x,FSABVelocityUsingFSADensity,FSABVelocityUsingFSADensityOverB0
- `geometryScheme4_2species_noEr_withPhi1InDKE`: status=parity_mismatch, practical=89/264, strict=90/264, sample=FSABFlow,FSABFlow_vs_x,FSABVelocityUsingFSADensity,FSABVelocityUsingFSADensityOverB0
- `geometryScheme4_2species_noEr_withQN`: status=parity_mismatch, practical=95/264, strict=95/264, sample=FSABFlow,FSABFlow_vs_x,FSABVelocityUsingFSADensity,FSABVelocityUsingFSADensityOverB0
- `geometryScheme4_2species_withEr_fullTrajectories`: status=parity_mismatch, practical=36/193, strict=36/193, sample=FSABFlow,FSABFlow_vs_x,FSABVelocityUsingFSADensity,FSABVelocityUsingFSADensityOverB0
- `geometryScheme4_2species_withEr_fullTrajectories_withQN`: status=parity_mismatch, practical=99/250, strict=99/250, sample=FSABFlow,FSABFlow_vs_x,FSABVelocityUsingFSADensity,FSABVelocityUsingFSADensityOverB0
- `monoenergetic_geometryScheme1`: status=parity_mismatch, practical=25/203, strict=29/203, sample=FSADensityPerturbation,FSAPressurePerturbation,NTV,NTVBeforeSurfaceIntegral
- `sfincsPaperFigure3_geometryScheme11_FPCollisions_2Species_DKESTrajectories`: status=parity_mismatch, practical=41/207, strict=41/207, sample=FSABFlow,FSABFlow_vs_x,FSABVelocityUsingFSADensity,FSABVelocityUsingFSADensityOverB0
- `sfincsPaperFigure3_geometryScheme11_FPCollisions_2Species_fullTrajectories`: status=parity_mismatch, practical=42/207, strict=42/207, sample=FSABFlow,FSABFlow_vs_x,FSABVelocityUsingFSADensity,FSABVelocityUsingFSADensityOverB0
- `sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_DKESTrajectories`: status=parity_mismatch, practical=37/207, strict=37/207, sample=FSABFlow,FSABFlow_vs_x,FSABVelocityUsingFSADensity,FSABVelocityUsingFSADensityOverB0
- `tokamak_1species_FPCollisions_noEr_withQN`: status=parity_mismatch, practical=86/274, strict=86/274, sample=FSABFlow,FSABFlow_vs_x,FSABVelocityUsingFSADensity,FSABVelocityUsingFSADensityOverB0
- `tokamak_1species_PASCollisions_noEr_withQN`: status=parity_mismatch, practical=91/274, strict=91/274, sample=FSABFlow,FSABFlow_vs_x,FSABVelocityUsingFSADensity,FSABVelocityUsingFSADensityOverB0

Largest CPU runtime improvements vs `tests/scaled_example_suite_release_cpu_v4/suite_report.json`:
- `monoenergetic_geometryScheme1`: 1956.1s -> 11.2s (delta=1944.9s)
- `transportMatrix_geometryScheme11`: 750.1s -> 65.7s (delta=684.3s)
- `transportMatrix_geometryScheme2`: 262.7s -> 89.8s (delta=172.8s)
- `geometryScheme5_3species_loRes`: 227.4s -> 104.5s (delta=122.8s)
- `geometryScheme4_1species_PAS_withEr_DKESTrajectories`: 496.0s -> 381.5s (delta=114.5s)

Largest CPU memory improvements vs `tests/scaled_example_suite_release_cpu_v4/suite_report.json`:
- `transportMatrix_geometryScheme11`: 6454.6 MB -> 4222.7 MB (delta=2231.9 MB)
- `tokamak_1species_PASCollisions_withEr_fullTrajectories`: 4153.6 MB -> 2042.1 MB (delta=2111.5 MB)
- `transportMatrix_geometryScheme2`: 3937.1 MB -> 2064.1 MB (delta=1873.0 MB)
- `geometryScheme5_3species_loRes`: 4789.5 MB -> 3453.6 MB (delta=1335.9 MB)
- `quick_2species_FPCollisions_noEr`: 1720.6 MB -> 952.6 MB (delta=768.0 MB)
<!-- END FAST_BRANCH_AUDIT -->

## Documentation

Build docs locally:

```bash
sphinx-build -b html -W docs docs/_build/html
```

Entry points:

- `docs/index.rst`
- `docs/system_equations.rst`
- `docs/method.rst`
- `docs/normalizations.rst`
- `docs/performance.rst`
- `docs/parallelism.rst`

## Testing

```bash
pytest -q
```

## License

See `LICENSE`.
