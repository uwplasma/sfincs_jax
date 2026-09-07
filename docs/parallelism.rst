Parallelism
===========

`dkx` runs on a single node — a multi-core CPU or the node's local GPUs — and
gets its throughput from two places: batched ``jax.vmap`` over independent
solves (optionally split across the node's devices) and the structured
``solvax`` solve routes underneath. Parallel execution does not establish
SFINCS-v3 physics parity or complete parameter derivatives; see :doc:`numerics`
and :doc:`differentiability` for supported domains.

The lever to reach for first is **batching independent solves**. Scanning the
radial electric field, sweeping flux surfaces, or building a monoenergetic
database are all embarrassingly parallel — each point is its own solve — and a
single vmapped call amortizes dispatch on CPU and fills the device on GPU.

Two kinds of parallelism
------------------------

- **Across independent solves.** Many ``E_r`` values, many flux surfaces, many
  ``whichRHS`` right-hand sides, or many optimizer/scan points. They share a
  discretization and differ only in a few physics leaves, so one batched call
  solves them together. This is where the throughput is.
- **Across the node's devices.** When more than one accelerator is visible, the
  batch of independent solves splits across them (``devices=...``); each device
  runs the same memory-budgeted chunked solve on its shard, so multiple GPUs
  fill with scan-shaped work. A single solve runs whole on one device — there
  is no internal split of one linear system across devices.

Batching independent solves
---------------------------

The batched API in ``dkx.batch`` wraps ``jax.vmap`` over solves
that share a discretization:

.. code-block:: python

   from dkx import batched_er_scan, batched_surface_scan

   # Scan the radial electric field on one geometry.
   result = batched_er_scan(problem, er_values, devices="auto")
   radial_current = result.radial_current       # J_r for each E_r value

   # Sweep a set of flux surfaces (one KineticOperator each).
   result = batched_surface_scan(operators)

Both return a ``BatchedSolveResult`` carrying the stacked moments, both accept
``differentiable=True`` to keep the batch inside a ``jax.grad`` chain, and both
take optional ``max_batch`` / ``memory_budget_gb`` overrides. Independent solves
— a vector of ``E_r`` values, a set of surfaces, or the ``(nu*, E_r)`` grid of a
monoenergetic database (``dkx.monoenergetic``) — are exactly the
parallel-friendly shape.

Prepare the problem or surface operators outside JAX transformations. The public
``dkx.batched_er_scan`` retains a prepared problem's solver method and tolerance
unless explicitly overridden. For repeated differentiable scans:

.. code-block:: python

   import jax
   import jax.numpy as jnp
   from dkx.er import prepare

   problem = prepare("input.namelist", solve_method="gmres", tol=1e-10)
   current = jax.jit(jax.value_and_grad(
       lambda er: jnp.sum(batched_er_scan(
           problem, er, devices="auto", differentiable=True,
       ).moments["FSABjHat"])
   ))
   value, gradient = current(er_values)

``devices=None`` keeps the default single-device behavior. Both public scan
functions return the original residual norms along with states and moments;
use these diagnostics together with observable and resolution checks. These
operator/deck interfaces do not yet provide a prepared native ``Case`` API.

**Automatic memory budgeting.** There are no sharding environment variables on
this path. The batch runs in ``jax.lax.map`` chunks sized from two numbers: the
per-solve memory footprint of the route the ``auto`` policy actually takes, and
the resolved device (or host) memory budget. A solve that routes to the
memory-lean truncated structured direct kernel is charged its truncated working
set, not the full-band factorization peak that route never allocates. Forward
work is chunked, but reverse mode can retain residuals across chunks. The
budget is an estimate, not a hard allocation limit; measure the gradient peak
separately. ``memory_budget_gb`` overrides the
resolved budget and is forwarded to each element's solver-route decision as
well as the outer chunk planner. A tight budget therefore cannot size a small
chunk and then silently let each element choose an inadmissible full-factor
route. ``max_batch`` caps the chunk size; the defaults need neither override.

**Measured throughput.** Because ``vmap`` amortizes per-solve dispatch, batching
beats a serial Python loop even on CPU — about ``9.5x`` for an ``E_r`` scan and
``6.4x`` for a surface scan. The larger win is on the GPU, where a single solve
sits at CPU parity but a batch fills the device. Reproduce both with
``python tools/benchmarks/batched_scan.py``.

**Multiple devices.** ``devices="auto"`` uses every local device of the
selected backend; an explicit sequence selects distinct devices. JAX
``shard_map`` runs the memory-budgeted local map on each device, including
inside ``jax.jit`` and ``jax.grad``. SOLVAX still owns each local solve.
The current SOLVAX batch wrapper does not expose the varying-axis checker
option needed by the custom linear solve, so DKX places the physics map with
JAX directly. Numerical residual and derivative checks remain enabled.

- Fewer than two resolved devices, or fewer cases than devices, uses the
  single-device path. Duplicate device entries are rejected.
- Equal local shapes are obtained by repeating the final valid case for
  padding. Trimming restores batch order and excludes padding from objectives
  and their derivatives. There is no explicit host gather. Even batch outputs
  remain partitioned; trimming uneven outputs can require redistribution.
- ``tests/test_batch.py`` checks actual addressable shards, single-device
  agreement, uneven batches, JIT, and physical-current gradients on two forced
  CPU devices. Forced devices share host resources and do not prove CPU
  speedup. GPU scaling and production memory/throughput require separate
  synchronized measurements; device placement alone does not establish speedup.


A bounded installed-wheel probe on two RTX A4000s (eight two-species PAS
cases, 7×7 angular grid, eight pitch modes, three speed nodes, chunk size two)
measured synchronized warm medians over ten calls: forward 44.1 ms on one GPU
and 27.7 ms on two; current value plus gradient 71.8 ms and 43.4 ms. Compilation
and first execution were measured separately (4.7–8.4 s). Perfetto traces show
approximately half the device events on each GPU, rather than replicated
whole-batch work. XLA temporary-memory estimates were 5.84 MB per device for
forward execution and 23.67→19.22 MB for the gradient; these are compiled
estimates, not measured process/device peaks. Raw traces, HLO and timings are
archived outside Git under ``dkx-review-evidence-20260905/sharding-wheel``.
This teaching-grid probe does not qualify production scaling or resolution.

Solver routes and where the GPU helps
-------------------------------------

Every solve takes one of the three ``solvax``-backed routes selected by the
``auto`` policy (:doc:`numerics`):

- **Structured direct.** Block-tridiagonal elimination over the Legendre index.
  For the DKES-trajectory / pitch-angle family the system splits into
  independent block-tridiagonal systems (one per species and speed node) solved
  with ``vmap``. This route is GPU-viable and the one batching accelerates.
- **Recycled Krylov.** Matrix-free FGMRES with subspace recycling,
  right-preconditioned by an exact structured direct solve of a simplified
  coarse operator. It carries a recycle pair to warm-start neighbouring points
  in an ``E_r`` scan or a Newton iteration.
- **Sparse direct.** A host sparse factorization for cases the structured direct
  route cannot admit; non-differentiable, used only on ``method="direct"`` or
  when the recycled Krylov route breaches its iteration cap.

Measured in :doc:`performance`: the GPU reaches CPU parity on the **structured
direct** route and runs the **iterative** and small-system paths 2-5x *slower*,
because those are dominated by serial, dispatch-bound iterations. GPU wins
therefore come from **batched** structured direct work, meaning multi-``E_r`` or
multi-surface sweeps, not from single solves.

Subsystem batching within a structured direct solve
---------------------------------------------------

The truncated structured direct kernel eliminates ``B = n_species * n_x``
independent ``(species, x)`` subsystems. ``solve(subsystem_batch=...)`` sets how
many it eliminates concurrently. An integer fixes the width (clamped to ``[1, B]``;
``1`` is the fully serial, minimum-memory sweep), and any width computes
identical per-subsystem arithmetic — the knob trades memory for batched
parallel work, so the CPU path is byte-identical to the serial sweep.

``subsystem_batch="auto"`` (the default) is backend-aware:

- **CPU backend — width 1.** XLA:CPU runs the batch axis of the LAPACK
  factor/solve custom calls serially per element, so a wider sweep only adds
  memory and cache pressure, not parallelism. Measured on the 336,610-unknown
  mid HSX deck at 8 threads, every width above 1 is neutral-to-slower: the
  ramped deck is 10.3 s at width 1 versus 11.4 s at width 2, and the
  uniform-``Nxi`` variant 16.6 s at width 1 versus 20.5 s at width 10.
- **Accelerator backends — memory-budgeted width.** The widest width whose
  modeled footprint (:func:`dkx.solve.tier1_truncated_peak_memory_bytes`) fits
  the memory budget, because batching raises device occupancy there while the
  budget clamp bounds the working set.

The knob is ignored by the non-truncated routes.

CPU threads
-----------

The XLA host threadpool is sized once, when the CPU backend initializes, so
thread control must be in place **before JAX is imported**; the CLI
``--cores`` flag sets the environment for you:

.. code-block:: bash

   dkx --cores 4 input.namelist       # or: export DKX_CORES=4

``DKX_CORES=N`` pins the solver threadpool to ``N`` threads (applied as
``NPROC``, the variable XLA reads, plus the host BLAS OpenMP/OpenBLAS pools);
``DKX_CORES=0`` lets XLA size the threadpool itself; when unset the
threadpool is clamped to ``min(8, cpu_count)``.

The measured optimum is 4-8 threads on both a 10-core laptop and a 36-core
workstation: structured direct thread scaling saturates near 2-2.5x and
**inverts** beyond the optimum on wide machines. On the 36-core workstation the
mid HSX deck (336,610 unknowns) warm structured direct solve measures 9.7 s at
1 thread, 7.8 s at 2, 5.6 s at 4, and 4.87 s at 8 threads — the optimum, a 1.99x
speedup at 25% parallel efficiency — then rises back to 12.2 s at 16, 56.6 s at
32, and 29.3 s at the full 36. The operator build stays flat near 8 s at every
core count, so the inversion is entirely the XLA fork-join overhead over the
sequential Legendre-block sweep once the pool is too wide (the wide-pool tail
carries large run-to-run variance). The guidance on many-core hosts is to set
``--cores`` to roughly 4-8, not to ``nproc``.

``DKX_CPU_DEVICES`` is a separate, explicit opt-in that forces multiple host
*devices* for multi-device CPU tests; forced host devices share one threadpool,
so it is not a performance knob. Full semantics and defaults are in the
environment-variable reference (:doc:`usage`).

Multi-host execution
--------------------

For multi-host device pools, JAX distributed initialization is opt-in via
``DKX_DISTRIBUTED`` together with ``DKX_PROCESS_ID``,
``DKX_PROCESS_COUNT``, ``DKX_COORDINATOR_ADDRESS``, and
``DKX_COORDINATOR_PORT`` (or the matching ``--distributed``,
``--process-id``, ``--process-count``, ``--coordinator-address``, and
``--coordinator-port`` CLI flags). Independent transport right-hand sides can
additionally be spread across worker processes with
``DKX_TRANSPORT_PARALLEL`` / ``DKX_TRANSPORT_PARALLEL_WORKERS``
(CLI ``--transport-workers``). See :doc:`usage` for the full list.

Relation to SFINCS Fortran v3
-----------------------------

SFINCS Fortran v3 scales one solve across many nodes with MPI domain
decomposition. `dkx` targets a single node — a multi-core CPU or one GPU
— and recovers scan-level throughput a different way: batched ``vmap`` over
independent solves, subspace recycling across neighbouring points, and exact
gradients that replace finite-difference scans in optimization. Parallel paths
call the same matrix-free operators as the sequential path, so outputs stay
bit-compatible up to floating-point reduction order and a parallel run is an
independent cross-check of a serial one.
