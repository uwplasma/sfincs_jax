## Examples

The examples are organized by difficulty:

- `examples/1_simple/`: basic API usage (no Fortran required)
- `examples/2_intermediate/`: parity checks and auto-diff demos
- `examples/3_advanced/`: optimization / implicit-diff patterns (may require extras)

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
python examples/1_simple/01_build_grids_and_geometry.py
```

New in this repo:

- Write `sfincsOutput.h5` via Python: `examples/1_simple/03_write_sfincs_output_python.py`
- Write `sfincsOutput.h5` via CLI: `examples/1_simple/04_write_sfincs_output_cli.py`
- Output parity vs Fortran fixture: `examples/2_intermediate/07_output_parity_vs_fortran_fixture.py`
