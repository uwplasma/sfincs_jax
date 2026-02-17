Performance techniques (full detail)
====================================

This page documents **every performance enhancement currently implemented in `sfincs_jax`**,
including the mathematical model context, implementation strategy, tuning knobs, and
how each change differs from (or complements) the original Fortran v3 solver.

Where relevant we reference the upstream SFINCS documentation that defines the
physics and discretization being accelerated. The primary sources are the
vendored v3 manual and technical notes in ``docs/upstream`` (see references below).

Baseline model and linear system (v3)
-------------------------------------

For linear runs (``RHSMode=1``), v3 assembles a linear system of the form

.. math::

   A x = b,

where :math:`x` contains the distribution function unknowns (the **F-block**),
optionally the :math:`\Phi_1(\theta,\zeta)` block (QN) and a constraint scalar
:math:`\lambda`. The operator :math:`A` is the linearized drift-kinetic operator,
and :math:`b` is the drive from thermodynamic gradients, inductive fields, and
other source terms. See :doc:`system_equations` and the upstream manual for the
full normalized expressions and parameter definitions.

For transport-matrix runs (``RHSMode=2/3``), the operator is the same **linear**
operator, but the RHS is overwritten internally for each ``whichRHS`` (v3
``evaluateResidual(f=0)``), and the resulting solutions are postprocessed into
transport coefficients (particle/heat fluxes and FSAB flow).

`sfincs_jax` implements the same model and discretization, but replaces the
matrix assembly with **matrix-free operator application** and JAX-based kernels.

What SFINCS v3 does (for performance context)
---------------------------------------------

The Fortran v3 code uses PETSc/KSP for iterative solves:

- It **assembles sparse matrices** (and sometimes dense blocks) from the discretized
  operator in ``populateMatrix.F90``.
- PETSc performs GMRES/BCGS with user-selected preconditioners, and stores the
  Krylov basis explicitly (memory intensive for large restarts).
- Transport-matrix mode loops over ``whichRHS``, reuses the same matrix, and
  performs multiple solves.

This is a strong baseline for CPU-only runs, but:

- The matrix assembly cost can be large relative to matrix-free matvecs.
- Sparse storage inflates memory, especially in 3D grids or multispecies FP runs.
- Building preconditioners (even block Jacobi) can be expensive in wall time.

`sfincs_jax` retains the same physics but uses a different numerical strategy:
**matrix-free + JIT + JAX-native preconditioning**.

Comparison with Fortran v3 workflow
-----------------------------------

**Fortran v3**

- Assemble sparse matrices in PETSc format.
- Use PETSc/KSP GMRES or BiCGStab with PETSc preconditioners.
- Transport matrices solved by repeated RHS with the same assembled operator.
- Diagnostics evaluated in per-``whichRHS`` loops.

**sfincs_jax**

- Apply the operator matrix-free (no assembly).
- JIT compile matvecs and solver loops; reuse compilation cache across runs.
- Use lightweight JAX-native preconditioners that avoid matvec-assembled blocks.
- Batch transport diagnostics across all ``whichRHS`` and reuse precomputed factors.

These differences are purely algorithmic/performance-oriented; the physics and
normalization remain anchored to the same v3 equations.

Matrix-free operator application (A·x) and caching
--------------------------------------------------

**Technique.** Replace explicit matrix assembly with a matvec:

.. math::

   y = A x,

where the matvec is computed by composing collisionless, drift, and collision
operators directly on the state vector.

**Implementation.**

- Core operator apply: ``sfincs_jax.v3_system.apply_v3_full_system_operator_cached``.
- Per-operator **signature cache** prevents re-JITing the matvec when the operator
  shape and static fields are unchanged. See ``_operator_signature_cached``.

**Why it’s fast.**

Matrix-free matvec avoids assembling large sparse matrices and amortizes
operator evaluation over JIT-compiled kernels.

**Impact.**

Reduces matrix assembly overhead and enables kernel fusion in XLA. The effect is
largest in repeated transport solves and in large multispecies FP cases where
assembly dominates total runtime.

**Knobs.**

- ``SFINCS_JAX_SOLVER_JIT``: disable/enable JIT of Krylov solves.
- ``SFINCS_JAX_TRANSPORT_MATVEC_MODE``: choose base vs RHS operator for transport solves.

**Compared to Fortran.**

Fortran builds PETSc matrices once and multiplies by them; `sfincs_jax` applies
the operator directly, which is often cheaper than assembling matrices,
especially when JIT compilation fuses multiple operator sub-terms.

JIT compilation and persistent compilation cache
------------------------------------------------

**Technique.** Use `jax.jit` for hot kernels (matvecs, residuals, Krylov loops),
and enable persistent compilation caching for repeated runs.

**Implementation.**

- `sfincs_jax` JITs the matvec and solver wrappers (GMRES/BiCGStab).
- The reduced-suite runner (`scripts/run_reduced_upstream_suite.py`) supports a
  persistent cache via ``--jax-cache-dir``.

**Why it’s fast.**

JIT amortizes Python overhead and enables XLA fusion. Persistent cache reduces
cold-start overhead in batch runs.

**Impact.**

Removes repeated compilation costs in batch suites and improves throughput for
workflow-style runs (parameter scans, repeated transport matrices).

**Compared to Fortran.**

Fortran has no JIT overhead but also no fusion; JAX replaces repeated Python-side
dispatch with compiled kernels.

Active-DOF reduction (sparse pitch grid)
----------------------------------------

**Technique.** When the pitch-angle basis is truncated (`Nxi_for_x < Nxi`),
construct a reduced system that contains only active degrees of freedom.

**Implementation.**

- Transport solves: ``_transport_active_dof_indices`` in ``sfincs_jax.v3_driver``.
- RHSMode=1: similar logic for active DOFs to reduce matrix-free work.

**Mathematics.**

Let :math:`P` be the selection matrix that extracts active DOFs:

.. math::

   x_{\mathrm{act}} = P x, \qquad A_{\mathrm{act}} = P A P^\top, \qquad b_{\mathrm{act}} = P b.

Solve the reduced system and map back via :math:`x = P^\top x_{\mathrm{act}}`.

**Why it’s fast.**

Reduces problem size and Krylov memory by removing unused Legendre modes.

**Impact.**

Substantial reductions in both memory and Krylov iterations when
``Nxi_for_x`` truncation is active (common in reduced-resolution suites).

**Compared to Fortran.**

Fortran typically keeps the full layout; `sfincs_jax` can safely reduce
if the basis truncation is explicit in the input.

Krylov solver strategy (short recurrence + fallback)
----------------------------------------------------

**Technique.** Use GMRES as the default for RHSMode=1 (parity-first) and BiCGStab
as the default for transport, with GMRES fallback on stagnation or non-finite residuals.

**Motivation.**

GMRES stores a full Krylov basis, with memory ~ :math:`O(n \cdot \text{restart})`.
BiCGStab is short recurrence with memory ~ :math:`O(n)`.
IDR(s) is another short-recurrence family used for nonsymmetric systems and is a
candidate for future low-memory solves.

**Implementation.**

- ``_solve_linear`` and ``_solve_linear_with_residual`` in ``sfincs_jax.v3_driver``.
- Fallback controlled by ``SFINCS_JAX_BICGSTAB_FALLBACK``.

**Compared to Fortran.**

Fortran typically uses GMRES via PETSc; `sfincs_jax` keeps GMRES for RHSMode=1 parity
and switches to BiCGStab for transport to reduce memory, with GMRES fallback.

**Impact.**

Lower memory footprint for large systems and improved wall time in many transport
cases. GMRES remains available when BiCGStab stagnates.

Implicit differentiation through linear solves
----------------------------------------------

**Technique.** Use `jax.lax.custom_linear_solve` to differentiate
through linear solves without storing Krylov iterates.

**Math.**

For :math:`A(p)\,x(p) = b(p)`, implicit differentiation yields:

.. math::

   A \frac{dx}{dp} = \frac{db}{dp} - \frac{dA}{dp}x.

The adjoint solve reuses the same linear operator, so gradients are efficient
and memory-bounded.

**Implementation.**

- ``sfincs_jax.implicit_solve.linear_custom_solve`` and
  ``linear_custom_solve_with_residual``.
- Controlled by ``SFINCS_JAX_IMPLICIT_SOLVE``.

Transport preconditioning (RHSMode=2/3)
---------------------------------------

**Technique.** Use analytic, JAX-native preconditioners to reduce Krylov iterations
without matvec-based assembly.

**Collision-diagonal preconditioner (baseline).**

Approximate the operator with its collision diagonal:

.. math::

   P^{-1} \approx \left(\mathrm{diag}(\mathcal{C}) + \alpha I \right)^{-1}.

Includes PAS diagonal and FP self-collision diagonal (per L, per x).

**Species×x block-Jacobi (FP auto / opt-in).**

When the FP operator is available, build per-:math:`L` blocks across species and :math:`x`:

.. math::

   \mathsf{C}^{(L)}_{(a,i),(b,j)} \equiv \mathcal{C}^{\mathrm{FP}}_{ab,ij}(L),

and invert each :math:`\mathsf{C}^{(L)}` (with identity shift + PAS diagonal).
Inactive :math:`x` points (from `Nxi_for_x`) are masked to identity.

**Low-rank Woodbury correction (optional).**

For FP-heavy cases, approximate the dense species×x blocks with a low-rank update
and apply a Woodbury inverse:

.. math::

   \left(D + U V^\top\right)^{-1}
   = D^{-1} - D^{-1} U \left(I + V^\top D^{-1} U\right)^{-1} V^\top D^{-1}.

This reduces both setup and apply costs for the FP preconditioner when
``SFINCS_JAX_TRANSPORT_FP_LOW_RANK_K`` (or ``SFINCS_JAX_FP_LOW_RANK_K``) is set.

**Coarse x-grid additive preconditioner (xmg).**

``SFINCS_JAX_TRANSPORT_PRECOND=xmg`` adds a two-level correction:
fine-grid collision-diagonal smoothing plus a coarse x-grid solve per species/L.
Set ``SFINCS_JAX_XMG_STRIDE`` to control the coarsening.

**Implementation.**

- ``_build_rhsmode23_sxblock_preconditioner`` in ``sfincs_jax.v3_driver``.
- Controlled by ``SFINCS_JAX_TRANSPORT_PRECOND`` (``auto``, ``sxblock``, ``collision``, etc.).
  ``auto`` picks the collision-diagonal preconditioner for the default BiCGStab transport
  solver and upgrades to species×x blocks for modest FP systems when GMRES is selected.

**Compared to Fortran.**

Fortran often uses PETSc block preconditioners constructed from the assembled matrix.
`sfincs_jax` builds an analytic block preconditioner directly from the FP collision
matrix, avoiding matvec assembly and staying JAX-native.

**Impact.**

Reduces iteration counts in FP transport cases without a heavy preconditioner
build, improving performance in PAS/W7X and FP-heavy benchmarks.

RHSMode=1 preconditioning (matrix-free)
---------------------------------------

`sfincs_jax` includes a family of RHSMode=1 preconditioners to match and extend v3 options:

- **Point-block Jacobi**: local (x,L) blocks at each :math:`(\theta,\zeta)`.
- **Theta-line / Zeta-line / ADI**: 1D line solves across angular dimensions.
- **Collision diagonal / xblock / sxblock**: analytic blocks from PAS/FP collisions.
- **Constraint-aware Schur**: enforces constraintScheme=2 source constraints via a
  diagonal or dense Schur complement.

These are cached to avoid recomputation. Controls:

- ``SFINCS_JAX_RHSMODE1_PRECONDITIONER``
- ``SFINCS_JAX_RHSMODE1_COLLISION_PRECOND_KIND``
- ``SFINCS_JAX_RHSMODE1_SCHUR_MODE`` / ``SFINCS_JAX_RHSMODE1_SCHUR_FULL_MAX``
- ``SFINCS_JAX_PRECOND_MAX_MB`` / ``SFINCS_JAX_PRECOND_CHUNK`` (cap memory during block assembly)
- ``SFINCS_JAX_PRECOND_DTYPE`` (default ``auto``; ``float32`` or ``float64`` to override)
- ``SFINCS_JAX_PRECOND_FP32_MIN_SIZE`` (threshold for auto mixed precision)

Transport diagnostics: batched + precomputed
--------------------------------------------

**Technique.** For transport matrices, solve all ``whichRHS`` systems, stack solutions,
then compute diagnostics in one batched kernel.

**Transport flux formulas (vm-only).** From v3 diagnostics, the key outputs are:

.. math::

   \Gamma_s^{\mathrm{vm}} \propto \int d\theta\,d\zeta\; \mathcal{F}_{\mathrm{vm}}(\theta,\zeta)\,
   \left[\frac{8}{3}\,\sum_x w_x x^4 f_{s,L=0} + \frac{4}{15}\,\sum_x w_x x^4 f_{s,L=2}\right],

.. math::

   Q_s^{\mathrm{vm}} \propto \int d\theta\,d\zeta\; \mathcal{F}_{\mathrm{vm}}(\theta,\zeta)\,
   \left[\frac{8}{3}\,\sum_x w_x x^6 f_{s,L=0} + \frac{4}{15}\,\sum_x w_x x^6 f_{s,L=2}\right],

.. math::

   \mathrm{FSABFlow}_s \propto \int d\theta\,d\zeta\; \frac{B}{D}\,
   \sum_x w_x x^3 f_{s,L=1}.

These are implemented in ``sfincs_jax.transport_matrix`` with strict-order
reductions matching v3 when required.

**Precompute constants.**

Factors depending only on geometry, species normalization, and grids
(:math:`w_x`, :math:`B/D`, prefactors, etc.) are precomputed once per transport run
and reused for all ``whichRHS`` solves.

**Implementation.**

- ``v3_transport_diagnostics_vm_only_precompute`` and
  ``v3_transport_diagnostics_vm_only_batch_op0_precomputed``.
- Controlled by ``SFINCS_JAX_TRANSPORT_DIAG_PRECOMPUTE`` (default enabled).

**Compared to Fortran.**

Fortran computes diagnostics per ``whichRHS`` in loops. JAX batches this and
reuses precomputed constants to reduce overhead and JIT work.

**Impact.**

Lower diagnostic overhead in transport workflows and reduced JIT retracing from
repeated reconstruction of geometry-dependent factors.

Recycled Krylov initial guesses for transport
---------------------------------------------

**Technique.** Reuse a small basis from recent solves to warm-start the next RHS:

.. math::

   x_0 \approx U (A U)^{\dagger} b,

where :math:`U` contains recent solution vectors.

**Implementation.**

- ``SFINCS_JAX_TRANSPORT_RECYCLE_K`` in ``sfincs_jax.v3_driver``.
- ``SFINCS_JAX_STATE_IN``/``SFINCS_JAX_STATE_OUT`` (cross-run recycling).
- ``SFINCS_JAX_SCAN_RECYCLE`` (auto-wires state files between scan points).

**Compared to Fortran.**

Fortran does not reuse Krylov subspaces across ``whichRHS``. This reuse is
lightweight and stays matrix-free.

Weighted reductions and Fortran sum order
-----------------------------------------

**Technique.** Use fused `einsum` for weighted sums by default, but allow
Fortran-like deterministic accumulation order when parity demands it.

**Implementation.**

- ``_weighted_sum_x_fortran`` and ``_weighted_sum_tz_fortran`` in
  ``sfincs_jax.transport_matrix``.
- Strict order controlled by ``SFINCS_JAX_STRICT_SUM_ORDER``.

This reduces Python overhead and improves performance, while still preserving
Fortran parity when needed.

Dense fallbacks (RHSMode=1 vs transport)
----------------------------------------

Dense fallbacks can stabilize difficult RHSMode=1 systems, and dense *retries* can
rescue transport-matrix solves that stall.

**Current default:**

- RHSMode=1 dense fallback is **enabled for modest systems** (``total_size <= 3000``)
  when Krylov iterations stagnate.
- Transport dense fallback is **disabled** unless explicitly requested, but a
  dense retry is enabled for RHSMode=2/3 when the active system size is modest.

Controls:

- ``SFINCS_JAX_RHSMODE1_DENSE_FALLBACK_MAX`` (default: ``3000``).
- ``SFINCS_JAX_TRANSPORT_DENSE_RETRY_MAX`` (default: ``3000`` for RHSMode=2/3).
- ``SFINCS_JAX_TRANSPORT_DENSE_FALLBACK`` / ``SFINCS_JAX_TRANSPORT_DENSE_FALLBACK_MAX``.

**Impact.**

Keeps transport runs performance-first while improving stability for parity-sensitive
cases where Krylov solvers can stall.

Memory reduction: remat/checkpoint + short recurrence
-----------------------------------------------------

**Rematerialization.**

- ``SFINCS_JAX_REMAT_COLLISIONS`` and ``SFINCS_JAX_REMAT_TRANSPORT_DIAGNOSTICS``
  enable `jax.checkpoint` around large operator/diagnostic kernels to reduce
  peak memory during autodiff.

**Short recurrence.**

BiCGStab avoids storing a full GMRES basis, reducing memory pressure in large
transport and multispecies FP runs.

Geometry parsing cache
----------------------

`sfincs_jax` caches parsed Boozer geometry files (.bc) by content hash and
geometry scheme to avoid repeated parsing for multiple runs of the same equilibrium.

Implementation: ``sfincs_jax.geometry`` and ``sfincs_jax.v3``.

F-block operator cache
----------------------

`sfincs_jax` can reuse geometry- and physics-dependent operator blocks across
repeated runs with identical inputs (e.g., scans that only change :math:`E_r`).
This avoids rebuilding collisionless, collision, and magnetic-drift operators.

Controls:

- ``SFINCS_JAX_FBLOCK_CACHE`` (default: enabled)
- ``SFINCS_JAX_FBLOCK_CACHE_MAX`` (max cached entries; default: ``8``)

Implementation: ``sfincs_jax.v3_fblock``.

Performance deltas (where measured)
-----------------------------------

The project maintains benchmark scripts and figures in ``docs/_static/figures/``:

- ``transport_compile_runtime_cache_2x2.png``: runtime vs cache effects.
- ``sfincs_vs_sfincs_jax_l11_runtime_2x2.png``: v3 vs JAX runtime comparison.

These figures are updated as part of the benchmarking workflow
(``examples/performance/``).

For quick reproduction:

.. code-block:: bash

   python examples/performance/benchmark_transport_l11_vs_fortran.py --repeats 4
   python examples/performance/profile_transport_compile_runtime_cache.py --repeats 3

Implementation map (source code)
--------------------------------

Key modules and functions referenced above:

- **Operator apply + caching**:

  - ``sfincs_jax/v3_system.py``: ``apply_v3_full_system_operator_cached``,
    ``_operator_signature_cached``.

- **Transport solver + preconditioners**:

  - ``sfincs_jax/v3_driver.py``: ``solve_v3_transport_matrix_linear_gmres``,
    ``_build_rhsmode23_sxblock_preconditioner``,
    ``_build_rhsmode23_collision_preconditioner``.

- **Diagnostics and flux formulas**:

  - ``sfincs_jax/transport_matrix.py``: ``v3_transport_diagnostics_vm_only_precompute``,
    ``v3_transport_diagnostics_vm_only_batch_op0_precomputed``.

- **Solver backends**:

  - ``sfincs_jax/solver.py``: GMRES/BiCGStab wrappers, dense fallback options,
    memory-aware restart logic.


Summary of tuning knobs
-----------------------

See :doc:`usage` for the full environment variable reference. The most important
performance controls are:

- Solver selection and fallback: ``SFINCS_JAX_RHSMODE1_SOLVE_METHOD``,
  ``SFINCS_JAX_BICGSTAB_FALLBACK``.
- Transport preconditioning: ``SFINCS_JAX_TRANSPORT_PRECOND``.
- Diagnostics precompute: ``SFINCS_JAX_TRANSPORT_DIAG_PRECOMPUTE``.
- Remat thresholds: ``SFINCS_JAX_REMAT_COLLISIONS(_MIN)``,
  ``SFINCS_JAX_REMAT_TRANSPORT_DIAGNOSTICS(_MIN)``.
- Active DOF: ``SFINCS_JAX_ACTIVE_DOF`` and ``SFINCS_JAX_TRANSPORT_ACTIVE_DOF``.

References (vendored)
---------------------

- SFINCS v3 technical manual:
  :download:`20150507-01 Technical documentation for version 3 of SFINCS.pdf <upstream/20150507-01 Technical documentation for version 3 of SFINCS.pdf>`
- Landreman, Smith, Mollen, Helander (2014), PoP 21 042503:
  :download:`LandremanSmithMollenHelander_2014_PoP_v21_p042503_SFINCS.pdf <upstream/LandremanSmithMollenHelander_2014_PoP_v21_p042503_SFINCS.pdf>`
- Technical note on the FP operator:
  :download:`20150402-01 Implementation of the Fokker-Planck operator.pdf <upstream/20150402-01 Implementation of the Fokker-Planck operator.pdf>`
- “Generalized minimal residual method,” Wikipedia (accessed 2025), which cites the original
  GMRES method of Saad & Schultz (1986): https://en.wikipedia.org/wiki/Generalized_minimal_residual_method
- H. A. van der Vorst, “Bi-CGSTAB: A fast and smoothly converging variant of Bi-CG,”
  SIAM J. Sci. Stat. Comput. 13(2):631–644 (1992). DBLP: https://dblp.org/rec/journals/sisc/Vorst92.html
- P. Sonneveld and M. B. van Gijzen, “IDR(s): A family of simple and fast algorithms for
  solving large nonsymmetric linear systems,” SIAM J. Sci. Comput. 31(2):1035–1062 (2008).
  DBLP: https://dblp.org/rec/journals/sisc/SonneveldG08.html
