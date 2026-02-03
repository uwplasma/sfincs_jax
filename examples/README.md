## Examples

The examples are organized by **topic** (rather than “difficulty”), so you can jump directly to what you need.

- `examples/getting_started/`: minimal “hello world” workflows (no Fortran required)
- `examples/parity/`: parity + validation against frozen v3 fixtures
- `examples/transport/`: `RHSMode=2/3` transport-matrix workflows + upstream scanplot scripts
- `examples/autodiff/`: AD / implicit-diff examples
- `examples/optimization/`: optimization with Optax/JAX-native tooling
- `examples/performance/`: JIT + performance microbenchmarks
- `examples/publication_figures/`: publication-ready figure generation

Also included:

- `examples/sfincs_examples/`: a vendored copy of the upstream v3 example suite + helper scripts.
- `examples/upstream/`: curated upstream inputs used in tests and docs.

### Setup

From the repo root:

```bash
cd sfincs_jax
pip install -e ".[dev]"
```

For examples that use `optax` / `jaxopt` / `equinox`:

```bash
pip install -e ".[opt]"
```

For examples that generate publication-style figures:

```bash
pip install -e ".[viz]"
```

### Running

Each example is a standalone script:

```bash
python examples/getting_started/01_build_grids_and_geometry.py
```

New in this repo:

- Write `sfincsOutput.h5` via Python: `examples/getting_started/03_write_sfincs_output_python.py`
- Write `sfincsOutput.h5` via CLI: `examples/getting_started/04_write_sfincs_output_cli.py`
- Output parity vs Fortran fixture: `examples/parity/07_output_parity_vs_fortran_fixture.py`
- Transport matrices (RHSMode 2/3): `examples/transport/16_transport_matrix_rhsmode2_and_rhsmode3.py`
- Differentiate a residual norm w.r.t. `nu_n`: `examples/autodiff/18_autodiff_gradient_nu_n_residual.py`
