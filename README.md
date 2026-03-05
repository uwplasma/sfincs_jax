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
