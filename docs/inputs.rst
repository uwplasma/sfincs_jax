Inputs (namelist) reference
===========================

`sfincs_jax` reads the same Fortran-style namelist files used by upstream SFINCS v3 (typically named
``input.namelist``).

Because upstream v3 supports a very large configuration space, this page focuses on:

1) **Where to find the complete upstream parameter definitions**, and
2) **What subset of inputs is currently implemented end-to-end in `sfincs_jax`**.

Full upstream parameter documentation
-------------------------------------

For the authoritative definitions of all namelist groups/parameters and their meaning, see the vendored
upstream technical documentation linked from ``docs/upstream_docs.rst`` (PDFs and TeX sources in
``docs/upstream/``).

Current `sfincs_jax` support (high level)
-----------------------------------------

`sfincs_jax` is parity-first and incremental. At a high level:

- **Geometry**: `geometryScheme` in `{1,2,4,5,11,12}` is supported for grid/geometry construction and for
  writing `sfincsOutput.h5` parity fixtures.
- **Full-system solve parity**: matrix-free matvec/RHS/residual/GMRES parity is available for a growing
  subset of fixtures (see ``docs/parity.rst``).

Practical notes for users
-------------------------

- If you are starting from an upstream example input, the quickest way to see whether it is supported
  end-to-end is:

  - Try `sfincs_jax write-output ...` and compare the resulting ``sfincsOutput.h5`` with upstream.
  - For solve parity status, consult ``docs/parity.rst`` (fixtures) and ``docs/fortran_examples.rst``
    (the example audit).

- If you want differentiability, prefer workflows that construct a `V3FullSystemOperator` once and then
  treat its fields as differentiable parameters (see ``docs/performance.rst``).
