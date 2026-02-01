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
