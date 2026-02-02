# Upstream SFINCS example inputs

This folder vendors the example inputs that ship with **SFINCS Fortran v3**, so that users familiar
with SFINCS can find recognizable starting points in `sfincs_jax`.

These inputs are copied from the upstream repository locations:

- `sfincs/fortran/version3/examples` → `examples/upstream/fortran_v3/`
- `sfincs/fortran/multiSpecies/examples` → `examples/upstream/fortran_multispecies/`
- `sfincs/matlab/version3/examples` → `examples/upstream/matlab_v3/`

## Running an upstream input with the Fortran executable

From the `sfincs_jax` repo root:

```bash
sfincs_jax run-fortran --input examples/upstream/fortran_v3/quick_2species_FPCollisions_noEr/input.namelist
```

## Running supported parts with `sfincs_jax`

At the moment, `sfincs_jax` does **not** implement end-to-end solves for the full upstream suite.
However, several building blocks (grids, some geometries, selected operator terms, output writing)
are parity-tested.

Some inputs require equilibrium files (e.g. `geometryScheme=11` `.bc` files). You can point
`sfincs_jax` at one or more equilibrium directories using:

```bash
export SFINCS_JAX_EQUILIBRIA_DIRS="/path/to/equilibria1:/path/to/equilibria2"
```

Then you can, for example, write a `sfincsOutput.h5` for supported geometries:

```bash
sfincs_jax write-output --input tests/ref/output_scheme4_1species_tiny.input.namelist --out sfincsOutput.h5
```

