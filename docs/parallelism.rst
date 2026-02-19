Parallelism
===========

This page explains how parallelism works in **SFINCS** and **sfincs_jax**, and how to
use it on a laptop (multi‑core CPU) and on clusters (multi‑CPU / multi‑GPU).

Why parallelism matters
-----------------------

Neoclassical transport solves are dominated by large, coupled linear systems. Even
with aggressive JIT and preconditioning, wall‑time scales roughly with the number
of Krylov iterations times the cost of each matvec. Parallelism is the primary way
to reduce time‑to‑solution once the single‑device kernels are efficient.

Two distinct patterns matter:

- **Embarrassingly parallel**: independent solves that can run concurrently
  (multiple `whichRHS`, multiple scan points, multiple cases).
- **Distributed linear algebra**: a single large linear system is split across
  devices to reduce per‑solve time.

Parallelism in JAX
------------------

JAX supports two broad modes of parallelism:

- **Multi‑process**: Run independent problems in separate Python processes. This is
  the simplest and most robust path on CPUs (and also works for GPUs if each
  process is pinned to a device).
- **SPMD / sharding**: Use `pjit` and sharded arrays to split a *single* linear
  system across multiple devices. This gives true per‑solve scaling but requires
  explicit sharding rules.

Key tradeoffs for `sfincs_jax`:

- Process parallelism is the easiest way to scale *independent* `whichRHS` solves
  and scan points on CPUs.
- Sharded matvec is the correct analogue to Fortran MPI for large single‑RHS
  solves, but it requires multi‑device setups and careful sharding constraints.

Parallelism in SFINCS (Fortran v3)
---------------------------------

SFINCS v3 uses **MPI + PETSc**:

- **Domain decomposition**: In `createGrids.F90`, PETSc DMDA splits **either**
  :math:`\theta` or :math:`\zeta` across MPI ranks (1‑D decomposition). Each rank
  owns a slab of the matrix rows for its local :math:`(\theta,\zeta)` range.
- **Distributed KSP**: In `solver.F90`, PETSc KSP solves the global linear system
  using distributed `Mat`/`Vec` objects. Direct solvers (MUMPS / SuperLU_DIST)
  handle the parallel factorization internally.

This is a classic MPI‑distributed linear‑algebra design: local matrix assembly,
parallel Krylov (or direct) solve.

Parallelism in sfincs_jax
-------------------------

`sfincs_jax` uses a layered approach that mirrors the Fortran design while
preserving differentiability:

1. **Parallel whichRHS (transport matrices)**

   RHSMode=2/3 solves are independent per `whichRHS`. We can solve multiple RHS
   in parallel across CPU processes or GPU devices.

   Implementation: `solve_v3_transport_matrix_linear_gmres` in
   `sfincs_jax.v3_driver`.

2. **Parallel cases / scan points**

   The reduced suite and scan workflows are embarrassingly parallel across
   cases or scan points. The suite runner now supports `--jobs` to execute
   multiple cases concurrently.

   Implementation: `scripts/run_reduced_upstream_suite.py`.

3. **Sharded matvec (SPMD)**

   For very large cases, a single RHS solve can be sharded across multiple
   devices by splitting the state vector along :math:`\theta` or :math:`\zeta`.

   Implementation: `apply_v3_full_system_operator_cached` in
   `sfincs_jax.v3_system` with `pjit` + `with_sharding_constraint`.

Design choices and parity
-------------------------

- **Parity first**: parallel paths call the same matrix‑free operators as the
  sequential path, so outputs remain bit‑compatible up to floating reduction
  order.
- **Deterministic merges**: results are merged by column index to avoid
  nondeterministic ordering in parallel `whichRHS`.
- **Differentiability**: each worker uses the same JAX operators, so the solve
  itself remains differentiable. Cross‑process aggregation is performed in
  Python, so end‑to‑end gradients across multiple processes are not automatic.
  If you need gradients for transport matrices, compute each RHS gradient in the
  worker and combine them explicitly.


Step (1): Parallel `whichRHS`
-----------------------------

Enable process‑parallel `whichRHS` solves with:

.. code-block:: bash

   export SFINCS_JAX_TRANSPORT_PARALLEL=process
   export SFINCS_JAX_TRANSPORT_PARALLEL_WORKERS=4

This parallelizes the RHSMode=2/3 transport matrix loop across CPU processes.
Parity is preserved because each `whichRHS` solve is identical to the sequential
path; outputs are merged deterministically by column.

**Relevant code paths**

- `sfincs_jax.v3_driver.solve_v3_transport_matrix_linear_gmres`
- `sfincs_jax.v3_driver._transport_parallel_worker`

**How it works**

- The master process partitions `whichRHS` indices and launches workers with
  `ProcessPoolExecutor`.
- Each worker reads the same input file, solves its RHS subset, and returns
  per‑RHS fluxes plus transport diagnostics.
- The master merges columns deterministically and reconstructs the transport
  matrix.

**Local sanity check**

On a tiny transport‑matrix fixture, process overhead dominates:

- `workers=1`: ~0.67 s
- `workers=2`: ~1.58 s

This is expected for very small problems. On large RHSMode=2/3 workloads (many
`whichRHS` solves, larger grids) we expect near‑linear speedup until memory or
compile overhead dominates.


Step (2): Parallel cases / scans
--------------------------------

The reduced suite runner can now execute multiple cases in parallel:

.. code-block:: bash

   python scripts/run_reduced_upstream_suite.py --jobs 4 --reuse-fortran

Each case runs in its own process, with independent Fortran and JAX runs.
This is the highest‑ROI parallel mode for large test campaigns.


Step (3): Sharded matvec (SPMD)
-------------------------------

Sharded matvec splits the *state vector* across devices for a **single solve**.
This is the closest analogue to the MPI / DMDA strategy in Fortran.

Enable sharding by selecting the axis:

.. code-block:: bash

   export SFINCS_JAX_MATVEC_SHARD_AXIS=zeta  # or theta

On CPUs, you can create multiple host devices with:

.. code-block:: bash

   export XLA_FLAGS=--xla_force_host_platform_device_count=4

On GPUs, JAX will automatically see all local devices.

**Notes**

- Sharding is currently **experimental** and only enabled when multiple devices
  are visible.
- When only one device is available, the code falls back to the standard JIT path.
- This mirrors Fortran DMDA splitting along :math:`\theta` or :math:`\zeta`,
  with the same intent: distribute matvec and preconditioner cost.

Verification
------------

- `tests/test_transport_parallel.py` compares sequential vs. parallel `whichRHS`
  outputs and confirms identical transport matrices.
- `tests/test_sharded_matvec.py` confirms sharded matvec falls back to standard
  JIT on single‑device hosts.


Recommended workflows
---------------------

**Macbook (multi‑core CPU)**

1. Use process parallelism for transport matrices:

   .. code-block:: bash

      export SFINCS_JAX_TRANSPORT_PARALLEL=process
      export SFINCS_JAX_TRANSPORT_PARALLEL_WORKERS=4

2. Use `--jobs` in the suite runner for concurrent cases.

**Perlmutter (multi‑CPU / multi‑GPU)**

- Multi‑CPU: set `--jobs` to the number of CPU tasks per node.
- Multi‑GPU: run a few `whichRHS` workers, one per GPU, or use
  `SFINCS_JAX_MATVEC_SHARD_AXIS=zeta` for single‑RHS sharding.


Parity and determinism
----------------------

- `whichRHS` parallelization preserves parity because each RHS is solved by the
  same matrix‑free algorithm and merged by column in deterministic order.
- Use `SFINCS_JAX_STRICT_SUM_ORDER=1` for stricter parity when combining
  reductions across devices.


Performance notes
-----------------

- Process parallelism helps most when `whichRHS` count is high (transport matrices)
  or when running many scan points.
- Sharded matvec helps most when a *single RHS* is large and dominates runtime.

See also:

- `docs/performance_techniques.rst`
- `sfincs_jax.v3_driver`
- `sfincs_jax.v3_system`
