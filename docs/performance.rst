Performance and differentiability
=================================

`sfincs_jax` is designed around a few principles that enable both speed and gradients:

1) **Matrix-free operators**: avoid assembling sparse matrices; apply the discrete operator as a pure function.
2) **JIT compilation**: compile hot kernels (matvecs, residuals, linear solves) with `jax.jit`.
3) **Vectorization**: prefer `vmap`, `einsum`, and batched linear algebra over Python loops.
4) **Explicit separations of concerns**: non-differentiable I/O (reading `.bc`/`wout_*.nc`) is isolated from
   the differentiable compute graph.

For a full, technique-by-technique breakdown (equations, derivations, knobs, and
implementation notes), see :doc:`performance_techniques`.


What is differentiable today?
-----------------------------

Within the current parity-tested subset, the following are differentiable with respect to JAX-array parameters:

- The **matrix-free operator** application (F-block and full-system operator blocks).
- The **linear residual** ``r(x) = A x - b`` and Jacobian-vector products via ``jax.jvp``.

The Fortran v3 inputs (namelist files, `.bc`, `wout_*.nc`) are *not* differentiable.
However, once a `V3FullSystemOperator` is constructed, you can treat its fields as parameters and
differentiate objectives with respect to them. For example:

- Differentiate a residual norm w.r.t. ``nu_n`` (see ``examples/autodiff/autodiff_sensitivity_nu_n_scheme5.py``).
- Differentiate a diagnostics functional w.r.t. a differentiable geometry parameter in ``geometryScheme=4`` optimization demos.
- Differentiate **through a linear solve** via implicit differentiation (see
  ``examples/autodiff/implicit_diff_through_gmres_solve_scheme5.py``).

By default, linear solves in RHSMode=1 and transport-matrix workflows use implicit
differentiation (`jax.lax.custom_linear_solve`). You can opt out by setting
``SFINCS_JAX_IMPLICIT_SOLVE=0`` if you need to debug or compare explicit Krylov
iteration gradients.


JAX-native performance patterns used in `sfincs_jax`
----------------------------------------------------

- **Keep arrays on-device**: build JAX arrays once and reuse them across matvec calls.
- **Use stable dtypes**: the v3 parity target requires 64-bit floats; `sfincs_jax` enables `jax_enable_x64`.
- **Avoid redundant dtype conversions**: collisionless and magnetic-drift operator kernels now cast
  `f` once per application (rather than per sub-term), reducing matvec overhead in PAS/FP hot cases.
- **Avoid Python loops in hot paths**:

  - For fixed-size recurrences (e.g. Legendre-coupled pitch-angle structure), prefer `jax.lax.scan` or
    banded updates via `at[].add(...)`.
  - For dense transforms, prefer `einsum`/batched `@` so XLA can fuse.
  - For Fourier-mode operations on uniform periodic grids, prefer `jax.numpy.fft` over explicit harmonic loops
    (e.g. `uHat` in `sfincs_jax.diagnostics`).

- **Exploit linearity**: for linear runs, the operator is constant; store and reuse the assembled RHS and
  re-run only GMRES when parameters change.
- **Batch transport RHS solves when possible**: for ``RHSMode=2/3`` dense branches, `sfincs_jax`
  now assembles the dense operator once and solves all ``whichRHS`` right-hand sides in one
  batched linear solve, reducing repeated operator assembly and retracing overhead.
- **Vectorized RHSMode=1 diagnostics**: vm-only moment/flux accumulation and output shaping are
  stacked/batched in JAX for non-``Phi1`` runs, reducing Python-loop overhead during
  ``write_sfincs_jax_output_h5(..., compute_solution=True)``.
- **Fast weighted reductions in diagnostics**: transport/rhsmode1 weighted sums now use
  fused ``einsum`` kernels by default (with an opt-in strict-order fallback), reducing
  diagnostic accumulation overhead in both RHSMode=1 and RHSMode=2/3 paths.
- **Vectorized transport-matrix assembly**: RHSMode=2/3 now builds
  ``transportMatrix`` directly from batched flux arrays, avoiding per-``whichRHS``
  Python loops and repeated diagnostic tree slicing.
- **Precomputed transport diagnostics**: geometry/species factors shared across ``whichRHS``
  solves are precomputed once and reused in batched diagnostics, reducing runtime and JIT work
  in transport-matrix modes.
- **Recycled transport solves**: optional warm-start recycling keeps a small number of
  recent solution vectors across ``whichRHS`` iterations (``SFINCS_JAX_TRANSPORT_RECYCLE_K``),
  reducing Krylov iterations on sequential RHS solves.
- **Cross-run Krylov recycling**: set ``SFINCS_JAX_STATE_OUT``/``SFINCS_JAX_STATE_IN`` (or
  ``SFINCS_JAX_TRANSPORT_RECYCLE_STATE=0`` to disable) to reuse transport solutions between
  adjacent scan points with matching operators. For built-in scans, enable
  ``SFINCS_JAX_SCAN_RECYCLE=1`` to wire these automatically.
- **Transport preconditioning (default)**: RHSMode=2/3 transport solves use a JAX-native
  preconditioner built analytically from the collision operator. By default (BiCGStab),
  a collision-diagonal preconditioner is used. For FP cases and GMRES-based transport solves,
  ``SFINCS_JAX_TRANSPORT_PRECOND=auto`` promotes a lightweight **species×x block-Jacobi**
  (per-L) preconditioner for modest system sizes. This cuts iterations without matvec-based
  assembly and preserves parity on the reduced suite.
- **Low-rank FP preconditioning**: optional Woodbury corrections approximate the dense
  FP species×x blocks with a low-rank update to reduce setup and apply costs.
- **Coarse x-grid preconditioning**: ``SFINCS_JAX_TRANSPORT_PRECOND=xmg`` adds a two-level
  x-grid correction (coarse solve + fine diagonal smoother) to reduce PAS/FP iterations.
- **Mixed-precision preconditioners**: ``SFINCS_JAX_PRECOND_DTYPE`` defaults to ``auto``
  (float32 for large systems, float64 otherwise) to reduce memory and preconditioner cost
  while keeping Krylov solves in float64. ``SFINCS_JAX_PRECOND_FP32_MIN_SIZE`` controls
  the auto threshold.
- **Cached Boozer `.bc` parsing**: scheme11/12 geometry loading now caches parsed
  surfaces by content digest (plus geometry scheme), so repeated localized/copy paths of
  the same equilibrium file reuse one parsed surface table.
- **Cached f-block operators**: reuse collisionless/collision/magnetic-drift operators
  across repeated runs with identical geometry and physics settings (e.g., scans that
  only change :math:`E_r`).
- **Vectorized NTV accumulation across nonlinear iterates**: RHSMode=1 output writing now
  computes NTV from stacked iterates in one batched JAX call instead of Python per-iterate loops.
- **Auto active-DOF reduction for RHSMode=1 (no Phi1)**: when ``Nxi_for_x`` truncates
  the pitch basis, the linear solve now reduces to active unknowns by default, cutting
  both matrix-free solve cost and JIT work on upstream-style reduced cases.
- **Persistent cache in reduced-suite automation**: ``scripts/run_reduced_upstream_suite.py``
  now runs `sfincs_jax` subprocesses with a persistent JAX compilation cache, reducing
  repeated-iteration benchmarking overhead.
- **Warm runtime reporting**: use ``--jax-repeats 2`` (or higher) in the reduced-suite runner
  to record steady-state `sfincs_jax` runtime from repeats after the first (cold-compile) run.
- **Remove dead Jacobian work in hot matvec paths**: direct-Phi1 ``factorJ`` kinetic-row terms
  that are absent in v3 ``whichMatrix=3`` are not assembled, improving parity and avoiding
  unnecessary FLOPs in includePhi1-in-kinetic matrix applications.
- **Use implicit differentiation for solve gradients**: for objectives that depend on the solution `x(p)` of
  a linear system `A(p) x = b(p)`, prefer `jax.lax.custom_linear_solve` (adjoint solve) over
  differentiating through Krylov iterations.
- **Default to short-recurrence Krylov for transport**: BiCGStab avoids storing a full GMRES basis and
  is therefore far more memory efficient for large RHSMode=2/3 systems. GMRES remains available and is
  used as a fallback when BiCGStab stagnates; transport-matrix solves default to BiCGStab with the
  collision-diagonal preconditioner for speed and memory efficiency. RHSMode=1 remains GMRES-first for
  parity. [#petsc-bcgs]_
- **JIT-compiled Krylov solves (default)**: `sfincs_jax` now JIT-compiles the GMRES/BiCGStab wrappers
  to reduce Python overhead for iterative solves; set ``SFINCS_JAX_SOLVER_JIT=0`` to disable.

Krylov solver strategy (memory + recycling)
-------------------------------------------

`sfincs_jax` defaults RHSMode=1 linear solves to GMRES (parity-first) and supports BiCGStab as an
opt-in low-memory option with GMRES fallback on stagnation. For RHSMode=2/3 transport-matrix solves we
default to BiCGStab and apply the collision-diagonal preconditioner by default, with GMRES as the
fallback. This keeps memory usage low while preserving reduced-suite parity. [#petsc-bcgs]_

For RHSMode=2/3 transport matrices, the ``whichRHS`` loop solves a sequence of linear systems with
nearly identical operators. We prototype a lightweight recycling hook that reuses the last ``k``
solution vectors as a warm start for the next solve. This is a small, practical approximation to
fully recycled Krylov methods such as GCRO-DR, which are designed explicitly for sequences of systems.
[#gcrodr]_ In practice it can reduce iterations without altering the linear operator or diagnostics.

For scans (e.g., ``sfincsScan``), you can enable on-disk Krylov recycling by writing a state file
after each run (``SFINCS_JAX_STATE_OUT``) and pointing the next run at it
(``SFINCS_JAX_STATE_IN``). When enabled, the transport solver will also seed its recycling basis
from the stored solutions to cut iterations between adjacent scan points.

Potential next solvers to explore (for further memory reductions or faster convergence on stiff cases):

- **IDR(s)**: short-recurrence, low-memory solvers for nonsymmetric systems with strong convergence
  properties on many practical problems. [#idrs]_
- **GCRO-DR / GMRES-DR / LGMRES**: recycled and deflated GMRES variants that explicitly reuse subspaces
  across sequences of linear systems. [#gcrodr]_

RHSMode=1 GMRES preconditioning (experimental)
----------------------------------------------

For RHSMode=1 linear solves that use matrix-free GMRES (as opposed to the dense assemble-from-matvec
path), you can enable an optional JAX-native preconditioner via an environment variable:

- ``SFINCS_JAX_RHSMODE1_PRECONDITIONER=point`` (or ``1``): point-block Jacobi on local (x,L) unknowns
  at each :math:`(\theta,\zeta)` (cheap, but can be too weak for stiff non-axisymmetric cases).
- ``SFINCS_JAX_RHSMODE1_PRECONDITIONER=collision``: collision-diagonal preconditioner using the
  analytic PAS/FP diagonal (cheap, effective for collision-dominated PAS/FP cases). For FP
  runs you can opt in to an x-block inverse per L via
  ``SFINCS_JAX_RHSMODE1_COLLISION_PRECOND_KIND=xblock`` or a full species×x block via
  ``SFINCS_JAX_RHSMODE1_COLLISION_PRECOND_KIND=sxblock``. Use
  ``SFINCS_JAX_RHSMODE1_FP_LOW_RANK_K`` (or ``SFINCS_JAX_FP_LOW_RANK_K``) to enable
  a low-rank Woodbury correction for the FP species×x blocks.
- ``SFINCS_JAX_RHSMODE1_PRECONDITIONER=theta_line``: theta-line block preconditioning that couples
  all theta points (at fixed zeta) for all local (x,L) unknowns (stronger, higher setup cost).
- ``SFINCS_JAX_RHSMODE1_PRECONDITIONER=zeta_line``: zeta-line block preconditioning that couples
  all zeta points (at fixed theta) for all local (x,L) unknowns (stronger, higher setup cost).
- ``SFINCS_JAX_RHSMODE1_PRECONDITIONER=adi``: apply theta-line then zeta-line preconditioning
  sequentially (strongest of the built-ins, highest setup + apply cost).
- ``SFINCS_JAX_RHSMODE1_PRECONDITIONER=0``: disable.

The regularization used when inverting preconditioner blocks can be tuned with:

- ``SFINCS_JAX_RHSMODE1_PRECOND_REG`` (default: ``1e-10``).
- ``SFINCS_JAX_RHSMODE1_COLLISION_PRECOND_MIN``: minimum ``total_size`` for default collision
  preconditioning when preconditioner options are not set (default: 600).

These options are most useful when you also select a Krylov solve method for RHSMode=1 via:

- ``SFINCS_JAX_RHSMODE1_SOLVE_METHOD=incremental`` (or ``batched``).

You can also control which side the preconditioner is applied on:

- ``SFINCS_JAX_GMRES_PRECONDITION_SIDE=left`` (default): left-preconditioned GMRES.
- ``SFINCS_JAX_GMRES_PRECONDITION_SIDE=right``: right-preconditioned GMRES (PETSc-like default).

.. note::

   Preconditioners change the Krylov iteration path and can therefore affect strict line-by-line
   parity with PETSc in near-singular branches. They are mainly intended to reduce runtime while
   preserving practical output parity.


Future optimization ideas (optional)
------------------------------------

Parity is now achieved at the reduced-suite tolerances, so remaining performance work is
profiling-driven and optional. High-ROI ideas to revisit if runtime becomes a bottleneck:

1. **Deeper Krylov recycling/deflation** (GCRO-DR / GMRES-DR) for long transport scans.
2. **Multilevel x-grid preconditioning** (coarse V-cycles) for stiff PAS/FP operators.
3. **Mixed-precision factorization** for FP block preconditioners (keep solves in float64).

.. [#petsc-bcgs] PETSc KSPBCGS manual page (BiCGStab solver notes, including memory behavior vs GMRES),
   https://petsc.gitlab.io/petsc/main/manualpages/KSP/KSPBCGS/
.. [#idrs] P. Sonneveld & M. van Gijzen, “IDR(s): a family of simple and fast algorithms for solving
   large nonsymmetric linear systems,” SIAM J. Sci. Comput. 31(2), 2008. TU Delft research portal:
   https://research.tudelft.nl/en/publications/idrs-a-family-of-simple-and-fast-algorithms-for-solving-large-non-2
.. [#gcrodr] E. de Sturler & M. L. Parks, “Analysis of Krylov subspace recycling for sequences of linear
   systems,” SAND2005-2794C, OSTI 970200, 2005.
   https://www.osti.gov/biblio/970200


Links to the JAX ecosystem (optional)
-------------------------------------

The package currently uses a lightweight in-repo GMRES implementation for parity control. For more advanced
workflows, the JAX ecosystem can be integrated cleanly once the residual is expressed in a differentiable way:

- `jaxopt`: implicit differentiation (root finding / linear solves) and robust optimization loops.
- `optax`: gradient-based optimization with schedules, constraints, and modern optimizers.
- `equinox`: structured parameter handling and module-style organization for larger refactors.


Parity tuning environment variables (developer)
-----------------------------------------------

For targeted parity debugging on difficult reduced fixtures, the solver exposes
opt-in environment variables:

- ``SFINCS_JAX_PHI1_NONLINEAR_RTOL``:
  override nonlinear relative stop for the includePhi1 frozen-linearization path.
- ``SFINCS_JAX_PHI1_NK_SOLVE_METHOD``:
  force Newton linear subsolve method (``dense``, ``incremental``, or ``batched``).
- ``SFINCS_JAX_PHI1_USE_FROZEN_LINEARIZATION``:
  force includePhi1 nonlinear branch to frozen/non-frozen Jacobian mode.
- ``SFINCS_JAX_PHI1_FROZEN_JAC_MODE``:
  select frozen-Jacobian variant for includePhi1 (``frozen``, ``frozen_rhs``, or ``frozen_op``; default ``frozen_rhs``).
- ``SFINCS_JAX_PHI1_GMRES_TOL``:
  override GMRES tolerance inside the includePhi1 nonlinear Newton–Krylov solves.
- ``SFINCS_JAX_PHI1_GMRES_MAXITER``:
  override GMRES max iterations inside includePhi1 Newton–Krylov solves.
- ``SFINCS_JAX_PHI1_LINESEARCH_FACTOR``:
  override the relative residual decrease required to accept a Newton step (legacy mode).
- ``SFINCS_JAX_PHI1_LINESEARCH_C1``:
  Armijo coefficient for the PETSc-style backtracking rule (default 1e-4).
- ``SFINCS_JAX_PHI1_LINESEARCH_MODE``:
  ``petsc`` (default for includePhi1 parity) uses the first step that satisfies the Armijo condition,
  ``best`` picks the step with the smallest residual among backtracking candidates,
  ``basic``/``full`` accept the full Newton step (no backtracking).
- ``SFINCS_JAX_PHI1_LINESEARCH_MAXITER``:
  override the maximum number of backtracking reductions (default 40 in ``petsc`` mode).
- ``SFINCS_JAX_PHI1_STEP_SCALE``:
  scale the Newton update step size (default 1.0); lower values damp iteration history.
- ``SFINCS_JAX_PHI1_QN_DIAG_SCALE``:
  scale the quasineutrality Phi1 diagonal stabilization (default 1.0).
- ``SFINCS_JAX_PHI1_MIN_ITERS``:
  minimum includePhi1 nonlinear iterations to record (default 4 in parity mode).
- ``SFINCS_JAX_TRANSPORT_MATVEC_MODE``:
  force transport-matrix matvec operator branch (``base`` or ``rhs``).
- ``SFINCS_JAX_TRANSPORT_DIAG_OP``:
  force diagnostics operator branch in transport-matrix runs (``base`` or ``rhs``).
- ``SFINCS_JAX_TRANSPORT_FORCE_KRYLOV``:
  disable the small-system dense fallback in RHSMode=2/3 and keep Krylov solves.
- ``SFINCS_JAX_TRANSPORT_EPAR_LOOSE``:
  opt-in looser GMRES tolerance for RHSMode=2 whichRHS=3 (E_parallel column), for parity experiments.
- ``SFINCS_JAX_TRANSPORT_EPAR_TOL``:
  override the tolerance used when ``SFINCS_JAX_TRANSPORT_EPAR_LOOSE`` is enabled (default 1e-8).
- ``SFINCS_JAX_TRANSPORT_EPAR_KRYLOV``:
  force Krylov (incremental GMRES) for RHSMode=2 whichRHS=3 regardless of dense fallback.
- ``SFINCS_JAX_TRANSPORT_PROJECT_NULLSPACE``:
  apply a constraint-space nullspace projection for RHSMode=2 whichRHS=3 (default on; set to 0 to disable).
- ``SFINCS_JAX_TRANSPORT_PROJECT_NULLSPACE_ATOL``:
  skip the projection when the constraint residual max-norm is below this threshold (default 1e-9).
- ``SFINCS_JAX_DENSE_REG``:
  override dense solve regularization strength for singular/near-singular systems.
- ``SFINCS_JAX_DENSE_SINGULAR_MODE``:
  choose singular branch handling in dense solves (default regularized mode; ``lstsq`` for minimum-norm).
- ``SFINCS_JAX_STRICT_SUM_ORDER``:
  force explicit loop-order weighted sums in diagnostics (debug/parity mode); by default
  fast fused ``einsum`` reductions are used.
- ``SFINCS_JAX_REMAT_COLLISIONS``:
  enable gradient checkpointing (``jax.checkpoint``) around collision operators to reduce peak memory
  during autodiff (``auto`` uses ``SFINCS_JAX_REMAT_COLLISIONS_MIN``; default 20000).
- ``SFINCS_JAX_REMAT_TRANSPORT_DIAGNOSTICS``:
  enable gradient checkpointing around transport diagnostics (``auto`` uses
  ``SFINCS_JAX_REMAT_TRANSPORT_DIAGNOSTICS_MIN``; default 20000).
- ``SFINCS_JAX_PRECOMPILE``:
  ahead-of-time compile core kernels when JAX persistent cache is enabled (``auto`` by default when
  ``JAX_COMPILATION_CACHE_DIR`` is set).


Reference benchmark figure (README/index)
-----------------------------------------

The repository includes a reproducible script that generates the top-level parity/runtime
comparison figure used in ``README.md`` and the docs index:

.. code-block:: bash

   python examples/performance/benchmark_transport_l11_vs_fortran.py --repeats 4

By default this uses frozen Fortran fixtures from ``tests/ref`` (no local Fortran runtime required).
If a local Fortran executable is available, pass ``--fortran-exe /path/to/sfincs`` for live runs.

Latest fixture-based snapshot (4 repeats, compile excluded for JAX):

.. list-table::
   :header-rows: 1
   :widths: 20 20 24 16

   * - Case
     - Fortran mean (s/run)
     - sfincs_jax mean (s/run)
     - max abs(ΔL11)
   * - ``scheme1``
     - 0.0275
     - 0.0937
     - 3.10e-13
   * - ``scheme11``
     - 3.6393
     - 0.1285
     - 1.39e-15
   * - ``scheme12``
     - 0.00888
     - 0.1073
     - 8.82e-08
   * - ``scheme5_filtered``
     - 2.9621
     - 0.1138
     - 5.30e-16

Fixture snapshot note: these values come from the frozen Fortran fixtures used by
``examples/performance/benchmark_transport_l11_vs_fortran.py`` when no local Fortran
executable is provided.

Persistent-cache compile/runtime split
--------------------------------------

To separate compile cost from steady solve time with the JAX persistent cache:

.. code-block:: bash

   python examples/performance/profile_transport_compile_runtime_cache.py --repeats 3

.. figure:: _static/figures/transport_compile_runtime_cache_2x2.png
   :alt: Compile/runtime split with persistent JAX cache across four reference cases.
   :align: center
   :width: 95%

   For each case, compile estimate = cold first call - warm first call; steady solve is the warm repeated runtime.

Latest snapshot (3 repeats):

.. list-table::
   :header-rows: 1
   :widths: 20 20 20

   * - Case
     - Compile estimate (s)
     - Warm steady solve (s/run)
   * - ``scheme1``
     - 1.5432
     - 0.0327
   * - ``scheme11``
     - 1.2959
     - 0.0232
   * - ``scheme12``
     - 1.3417
     - 0.0237
   * - ``scheme5_filtered``
     - 1.3672
     - 0.0364


Memory footprint and compilation-time optimization (literature-backed)
-----------------------------------------------------------------------

The main memory and compile-time levers for ``sfincs_jax`` map to standard JAX/XLA
mechanisms and Krylov-solver theory. The items below are the highest-ROI, literature-backed
paths we use to guide performance work:

- **Measure device + host allocation hotspots** using the JAX device memory profiler and
  XLA/trace timelines before changing algorithms. This pinpoints which buffers dominate the
  memory footprint and where JIT time is spent. [#jax-profiler]_
- **Use gradient checkpointing** (``jax.checkpoint`` / ``jax.remat``) to trade recomputation
  for lower peak memory during autodiff, especially for long transport chains. [#jax-checkpoint]_
- **Control GPU memory preallocation** and allocation strategy when GPU memory is the limiting
  factor (e.g., disable full preallocation or set a memory fraction). [#jax-gpu-mem]_
- **Persist and reuse compilation artifacts** with the JAX compilation cache to amortize
  expensive builds across repeated runs. [#jax-compile-cache]_
- **Use ahead-of-time (AOT) compilation** for stable-shape kernels that dominate wall time;
  this reduces JIT latency during interactive or production runs. [#jax-aot]_
- **Prefer short-recurrence Krylov methods** (e.g., BiCGStab/IDR(s)) when GMRES memory growth
  becomes dominant, since GMRES stores all previous Krylov vectors. [#gmres-memory]_

These sources inform our memory and compilation roadmap; any algorithmic change is still
validated against the reduced-suite parity and physics tests before it becomes a default.

.. [#jax-profiler] JAX profiling and device memory tools:
   https://docs.jax.dev/en/latest/device_memory_profiling.html
.. [#jax-checkpoint] JAX gradient checkpointing (``jax.checkpoint`` / ``jax.remat``):
   https://docs.jax.dev/en/latest/gradient-checkpointing.html
.. [#jax-gpu-mem] JAX GPU memory allocation and preallocation controls:
   https://docs.jax.dev/en/latest/gpu_memory_allocation.html
.. [#jax-compile-cache] JAX persistent compilation cache:
   https://docs.jax.dev/en/latest/persistent_compilation_cache.html
.. [#jax-aot] JAX ahead-of-time compilation:
   https://docs.jax.dev/en/latest/aot.html
.. [#gmres-memory] Iterative methods reference (GMRES storage growth vs short-recurrence methods):
   https://mathworld.wolfram.com/GeneralizedMinimalResidualMethod.html


Connection to MONKES / adjoint methods
--------------------------------------

The MONKES approach (see the MONKES code/paper materials and related thesis work, external to this repository) emphasizes:

- a **monoenergetic drift-kinetic equation** as a reduced model,
- a **Legendre basis** representation,
- and **adjoint properties** that enable efficient derivatives of transport coefficients.

In `sfincs_jax`, the same goals (fast derivatives for optimization) are achieved by keeping the discrete
operator/residual in JAX so that:

- Jacobian actions can be obtained via **automatic differentiation** (JVP/VJP),
- and gradients through a solve can be obtained via **implicit differentiation**
  without forming matrices.


Operator-level parity debugging utility
---------------------------------------

For difficult upstream mismatches, compare a Fortran PETSc matrix directly against the
JAX operator assembly:

.. code-block:: bash

   python scripts/compare_fortran_matrix_to_jax_operator.py \
     --input /path/to/input.namelist \
     --fortran-matrix /path/to/sfincsBinary_iteration_000_whichMatrix_3 \
     --fortran-state /path/to/sfincsBinary_iteration_000_stateVector \
     --project-active-dofs \
     --out-json matrix_compare.json

The report includes block-wise statistics (``f``/``phi``/``extra``) and top-entry deltas
to localize missing couplings quickly.
