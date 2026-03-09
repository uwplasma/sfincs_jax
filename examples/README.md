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
python examples/getting_started/build_grids_and_geometry.py
```

Common entry points:

- Write `sfincsOutput.h5` via Python: `examples/getting_started/write_sfincs_output_python.py`
- Write `sfincsOutput.h5` via CLI: `examples/getting_started/write_sfincs_output_cli.py`
- Output parity vs Fortran fixture: `examples/parity/output_parity_vs_fortran_fixture.py`
- Transport matrices (RHSMode 2/3): `examples/transport/transport_matrix_rhsmode2_and_rhsmode3.py`
- Transport matrices with Krylov recycling: `examples/transport/transport_matrix_recycle_demo.py`
- Differentiate a residual norm w.r.t. `nu_n`: `examples/autodiff/autodiff_gradient_nu_n_residual.py`
- Implicit differentiation through BiCGStab: `examples/autodiff/implicit_diff_through_gmres_solve_scheme5.py --solver bicgstab`

### Scaled upstream example sweep

The vendored `examples/sfincs_examples/` inputs currently match the original
Fortran v3 example inputs exactly. For reproducible benchmarking, use the
original upstream tree as the resolution reference and set `--scale-factor`
relative to that baseline. `1.0` means the original v3 example resolution, and
values below `1.0` reduce `NTHETA/NZETA/NX/NXI` consistently from that upstream
reference.

To compare Fortran vs `sfincs_jax` runtime, memory, output parity, and print
parity at the original upstream resolution, use:

```bash
cd /Users/rogeriojorge/local/tests/sfincs_jax
python scripts/run_scaled_example_suite.py \
  --examples-root examples/sfincs_examples \
  --resolution-reference-root /Users/rogeriojorge/local/tests/sfincs_original/fortran/version3/examples \
  --fortran-exe /Users/rogeriojorge/local/tests/sfincs/fortran/version3/sfincs \
  --out-root tests/scaled_example_suite_ref_cpu_local \
  --timeout-s 240 \
  --max-attempts 2 \
  --scale-factor 1.0
```

The script keeps the runnable input text from `--examples-root`, rewrites only
`NTHETA/NZETA/NX/NXI` from the matching case in
`--resolution-reference-root`, includes
`examples/additional_examples/input.namelist`, and writes per-case outputs plus
`suite_report.json`, `suite_report_strict.json`, `suite_status*.rst`,
`run_manifest.json`, and `summary.md` into the chosen `--out-root`.
These suite-level artifacts are checkpointed after every finished case, so an
interrupted long run still leaves a usable partial audit instead of only
per-case directories.
If you restart a long sweep after changing `sfincs_jax` code or after a bad
launch, reuse the same `--out-root` only with `--reset-report`, otherwise the
old case rows remain merged into the new `suite_report*.json` checkpoint files.
For the legacy `examples/upstream/fortran_multispecies` tree, the Fortran lane
also canonicalizes the old pre-v3 namelist groups and aliases into the v3
input shape expected by the reference executable, while the `sfincs_jax` lane
honors those same legacy aliases directly.

To separate reference generation from JAX benchmarking, first create a stable
CPU reference root, then benchmark CPU or GPU JAX runs against that fixed
reference without re-running Fortran:

```bash
cd /Users/rogeriojorge/local/tests/sfincs_jax
python scripts/run_scaled_example_suite.py \
  --examples-root examples/sfincs_examples \
  --resolution-reference-root /Users/rogeriojorge/local/tests/sfincs_original/fortran/version3/examples \
  --fortran-exe /Users/rogeriojorge/local/tests/sfincs/fortran/version3/sfincs \
  --out-root tests/gating_reference_cpu \
  --pattern '^(tokamak_1species_FPCollisions_noEr|inductiveE_noEr)$' \
  --scale-factor 1.0 \
  --max-attempts 1

python scripts/run_scaled_example_suite.py \
  --examples-root examples/sfincs_examples \
  --resolution-reference-root /Users/rogeriojorge/local/tests/sfincs_original/fortran/version3/examples \
  --reference-results-root tests/gating_reference_cpu \
  --out-root tests/gating_cpu_from_ref \
  --pattern '^(tokamak_1species_FPCollisions_noEr|inductiveE_noEr)$' \
  --scale-factor 1.0 \
  --max-attempts 1
```

This keeps the Fortran reference H5/log files fixed across lanes, which is
useful when comparing local CPU and remote GPU runs against the same baseline.

For a full-sweep audit on laptops or workstations where the original v3
resolution is too expensive, keep the upstream resolution ratios but reduce the
global scale factor instead of hand-editing individual examples:

```bash
cd /Users/rogeriojorge/local/tests/sfincs_jax
python scripts/run_scaled_example_suite.py \
  --examples-root examples/sfincs_examples \
  --extra-input examples/additional_examples/input.namelist \
  --resolution-reference-root /Users/rogeriojorge/local/tests/sfincs_original/fortran/version3/examples \
  --fortran-exe /Users/rogeriojorge/local/tests/sfincs/fortran/version3/sfincs \
  --out-root tests/scaled_example_suite_ref_cpu_full \
  --scale-factor 0.75 \
  --timeout-s 3600 \
  --max-attempts 1
```

This preserves the original example mix while reducing `NTHETA/NZETA/NX/NXI`
consistently from the upstream reference tree.

Use these reduced-scale full sweeps as an audit tool for runtime, memory, and
solver-branch fragility. They preserve the upstream resolution ratios, but they
can still shift conditioning enough to expose branch-sensitive mismatches that
do not appear at the original example resolution. The release gate remains the
standard reduced-suite comparisons plus targeted original-resolution examples.
