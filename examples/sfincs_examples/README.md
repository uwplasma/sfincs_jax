# Upstream Fortran v3 example suite (vendored)

This folder is a **copy** of the upstream SFINCS Fortran v3 example suite
(`sfincs/fortran/version3/examples`) plus the upstream postprocessing scripts in
`sfincs/fortran/version3/utils`.

Goals:

- Let SFINCS users run the **same example inputs** with `sfincs_jax`.
- Provide a convenient place to benchmark against the compiled Fortran v3 executable.
- Keep the upstream plotting scripts available while `sfincs_jax` grows output parity.

## What works today

`sfincs_jax` is still parity-first and incremental: many examples here will not run end-to-end yet.

In general, you should expect:

- `sfincs_jax write-output` to work for a broad subset of inputs (geometry + `sfincsOutput.h5` fields).
- `sfincs_jax write-output --compute-transport-matrix` to work for the **RHSMode=2/3** examples used by the
  transport-matrix parity fixtures (small cases).

For the current support matrix, see:

- `docs/parity.rst`
- `docs/fortran_examples.rst` (auto-generated audit table)

## Running the vendored suite

From the `sfincs_jax` repository root:

```bash
python examples/sfincs_examples/run_sfincs_jax.py --write-output
```

To run a subset:

```bash
python examples/sfincs_examples/run_sfincs_jax.py --write-output --pattern monoenergetic_geometryScheme11
```

To also run the compiled Fortran v3 executable for comparison (slow):

```bash
python examples/sfincs_examples/run_sfincs_jax.py \\
  --write-output \\
  --compare-fortran \\
  --fortran-exe ../sfincs/fortran/version3/sfincs
```

## Upstream utils

The scripts in `utils/` are copied from upstream (e.g. `utils/sfincsScanPlot_1`).
They expect specific fields inside `sfincsOutput.h5`.

Many of these scripts also read default parameters from `globalVariables.F90`. In the upstream
layout, this file lives next to `utils/`, so this repo vendors it as:

- `examples/sfincs_examples/globalVariables.F90`

`sfincs_jax` currently writes the fields needed by the transport-matrix plotting scripts for RHSMode=2/3:
`transportMatrix`, `FSABFlow`, `particleFlux_vm_psiHat`, and `heatFlux_vm_psiHat`.

Over time, more datasets will be added for broader postprocessing parity.

To run an upstream `utils/` script in a non-interactive way:

```bash
sfincs_jax postprocess-upstream --case-dir /path/to/case --util sfincsScanPlot_1 -- pdf
```

If you are not running from a `sfincs_jax` repo checkout, set:

```bash
export SFINCS_JAX_UPSTREAM_UTILS_DIR=/path/to/utils
```
