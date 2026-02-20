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
----------------------------------

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

**Platform note (macOS)**

macOS uses `spawn` for multiprocessing. Run from a file/module (not `python - <<EOF`)
so worker processes can import the main module cleanly.

**Measured scaling (Macbook M3 Max, 14‑core)**

Benchmark case: `examples/performance/transport_parallel_xxlarge.input.namelist`
(RHSMode=2, geometryScheme=2, Ntheta=15, Nzeta=15, Nxi=6, NL=4, Nx=5).

Benchmark preconditioner: `SFINCS_JAX_TRANSPORT_PRECOND=xmg` to keep the
single‑worker runtime in the 1–2 minute range while preserving parity.
You can override this via `--precond` in the benchmark script.

Command:

.. code-block:: bash

   python examples/performance/benchmark_transport_parallel_scaling.py \
     --workers 1 2 3 4 \
     --repeats 1 \
     --warmup 1 \
     --global-warmup 1

Results (single run per worker count, JAX cache warm‑up enabled):

.. list-table::
   :header-rows: 1

   * - Workers
     - Mean time (s)
     - Speedup
   * - 1
     - 74.84
     - 1.00
   * - 2
     - 49.31
     - 1.52
   * - 3
     - 25.37
     - 2.95
   * - 4
     - 25.42
     - 2.95

.. figure:: _static/figures/parallel/transport_parallel_scaling.png
   :alt: Parallel whichRHS scaling on Macbook M3 Max
   :width: 90%

   Parallel whichRHS scaling (runtime + speedup vs workers).

For this larger case, scaling reaches ~3.0× by 3–4 workers before flattening.
The plateau reflects process overhead and shared‑resource contention on a
laptop‑class CPU. Larger multi‑RHS runs on server‑class nodes should show
stronger scaling.

Note: RHSMode=2 has only **3** right‑hand sides. A 4th worker has no extra RHS
to solve, so speedup naturally saturates near 3 workers.

Earlier runs (smaller grids)
----------------------------

We also benchmarked smaller RHSMode=2 cases (7–45 s single‑worker time). These
showed weaker scaling because process startup and JIT overheads dominate at
small problem sizes. The longer xxlarge case above is required to observe clear
speedup on laptop CPUs.

JIT/compilation notes
---------------------

To avoid skew from compilation:

- The results above were collected after a one‑off warm run (workers=1) to populate
  the persistent JAX cache, with ``--warmup 0 --global-warmup 0`` for the timing run.
- To reproduce, either run once with ``--workers 1`` before timing or set
  ``--global-warmup 1`` and keep ``--warmup 0`` for the timed measurements.
- A persistent `JAX_CACHE_DIR` is used so processes can reuse compiled kernels.


Step (2): Parallel cases / scans
--------------------------------

The reduced suite runner can now execute multiple cases in parallel:

.. code-block:: bash

   python scripts/run_reduced_upstream_suite.py --jobs 4 --reuse-fortran

Each case runs in its own process, with independent Fortran and JAX runs.
This is the highest‑ROI parallel mode for large test campaigns.

**Scan parallelism (E_r scans)**

For scans with many values, use `--jobs` to parallelize scan points:

.. code-block:: bash

   sfincs_jax scan-er \
     --input input.namelist \
     --out-dir scan_dir \
     --min -2 --max 2 --n 41 \
     --jobs 8

Parallel scan mode disables Krylov recycle between points. Use this when you
care more about throughput than per‑point warm‑start.

Scaling to dozens/hundreds (job arrays)
------------------------------------------------------------

For large ensembles, use job arrays on clusters and slice the work with
`--case-index`/`--case-stride` (suite) or `--index`/`--stride` (scan).

**Suite array (N cases across M array tasks)**

.. code-block:: bash

   #SBATCH --array=0-63
   python scripts/run_reduced_upstream_suite.py \
     --case-index ${SLURM_ARRAY_TASK_ID} \
     --case-stride 64 \
     --reuse-fortran

**Scan array (N scan points across M array tasks)**

.. code-block:: bash

   #SBATCH --array=0-63
   sfincs_jax scan-er \
     --input input.namelist \
     --out-dir scan_dir \
     --min -2 --max 2 --n 401 \
     --index ${SLURM_ARRAY_TASK_ID} \
     --stride 64

This gives near‑linear scaling to dozens or hundreds of workers, since each
task is independent.


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
- When only one device is available, the code falls back to the standard JIT path
  and skips sharding constraints (no functional change).
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
