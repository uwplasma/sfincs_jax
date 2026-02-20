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
- The CLI defaults to a user cache directory (``~/.cache/sfincs_jax/jax_compilation_cache``)
  and enables ``jax.experimental.compilation_cache`` automatically unless disabled.
- Command-line subcommands lazily import heavy modules to reduce startup overhead.

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
``SFINCS_JAX_TRANSPORT_FP_LOW_RANK_K`` (or ``SFINCS_JAX_FP_LOW_RANK_K``) is set,
including the ``auto`` default for larger FP blocks.

**Measured impact (FP-heavy cases).** With ``SFINCS_JAX_TRANSPORT_FP_LOW_RANK_K=auto``
the low-rank update improved end-to-end wall time by ~9% for
``geometryScheme5_3species_loRes`` and ~21% for ``tokamak_1species_FPCollisions_noEr``
relative to ``SFINCS_JAX_TRANSPORT_FP_LOW_RANK_K=0`` (profiles in
``examples/performance/output/reduced_profiles_fp_*.json``).
See ``docs/references.rst`` for Woodbury/low-rank update references.

**Coarse x-grid additive preconditioner (xmg).**

``SFINCS_JAX_TRANSPORT_PRECOND=xmg`` adds a two-level correction:
fine-grid collision-diagonal smoothing plus a coarse x-grid solve per species/L.
Set ``SFINCS_JAX_XMG_STRIDE`` to control the coarsening.

**JAX sparse Jacobi (optional).**

``SFINCS_JAX_TRANSPORT_PRECOND=sparse_jax`` builds a sparsified operator and applies
a few weighted Jacobi sweeps in JAX. This can reduce memory relative to dense
preconditioners while staying differentiable. Controls mirror the RHSMode=1
``sparse_jax`` options (``SFINCS_JAX_TRANSPORT_SPARSE_JAX_*`` and
``SFINCS_JAX_TRANSPORT_SPARSE_DROP_*``).

**Implementation.**

- ``_build_rhsmode23_sxblock_preconditioner`` in ``sfincs_jax.v3_driver``.
- Controlled by ``SFINCS_JAX_TRANSPORT_PRECOND`` (``auto``, ``sxblock``, ``collision``, etc.).
  ``auto`` picks the collision-diagonal preconditioner for the default BiCGStab transport
  solver and upgrades to species×x blocks for modest FP systems when GMRES is selected.
  For larger FP systems (especially when dense fallbacks are blocked for memory),
  ``auto`` escalates to the matrix-free x-grid multigrid preconditioner (``xmg``).

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
- **Species-block (PAS)**: full (x,L,θ,ζ) block per species for strong PAS conditioning.
- **Collision diagonal / xblock / sxblock**: analytic blocks from PAS/FP collisions.
- **Constraint-aware Schur**: enforces constraintScheme=2 source constraints via a
  diagonal or dense Schur complement.

**PAS gauge/constraint projection.** PAS operators admit a nullspace drift in the
flux-surface average. We remove it before Krylov iterations:

.. math::

   f \leftarrow f - \langle f \rangle_{\mathrm{FS}},

applied per species and :math:`x` using the same :math:`(\theta,\zeta)` weights
as diagnostics. This stabilizes PAS tokamak-like cases and pairs with the
theta-line preconditioner default.

Implementation: ``sfincs_jax.v3_driver`` (``use_pas_projection`` and
``_project_pas_f``). Control: ``SFINCS_JAX_PAS_PROJECT_CONSTRAINTS`` (auto on for
``N_\zeta=1`` tokamak-like runs and DKES-trajectory PAS cases).

**Constraint-aware Schur (constraintScheme=2).** With constraint variables
:math:`c`, the linear system is partitioned as

.. math::

   \begin{bmatrix}
   A_{ff} & A_{fc} \\\\
   A_{cf} & A_{cc}
   \end{bmatrix}
   \begin{bmatrix}
   f \\\\
   c
   \end{bmatrix}
   =
   \begin{bmatrix}
   b_f \\\\
   b_c
   \end{bmatrix}.

The preconditioner uses a Schur complement approximation

.. math::

   S \approx A_{cc} - A_{cf} A_{ff}^{-1} A_{fc},

with diagonal or dense approximations for :math:`S`. This preserves constraint
coupling while improving conditioning in high‑ratio PAS cases.

Implementation: ``sfincs_jax.v3_driver`` (``_build_rhsmode1_schur_*``).
Controls: ``SFINCS_JAX_RHSMODE1_SCHUR_MODE`` and
``SFINCS_JAX_RHSMODE1_SCHUR_FULL_MAX``. The base preconditioner used inside the
Schur construction can be selected with ``SFINCS_JAX_RHSMODE1_SCHUR_BASE``; the
default ``auto`` path uses a PAS species-block base when the per‑species block
size is modest, then prefers the PAS x-block :math:`(\theta,\zeta)` variant when
the per‑:math:`x` block is still small, and falls back to theta/zeta line bases
otherwise. This avoids dense fallback on PAS stellarator cases without excessive
cost on larger systems.

For PAS cases with ``constraintScheme=2``, ``SFINCS_JAX_RHSMODE1_SCHUR_AUTO_MIN``
can trigger the Schur preconditioner automatically once ``total_size`` exceeds
the threshold (default: ``2500``), which helps HSX-like cases.
See ``docs/references.rst`` for Schur complement references.

For FP-heavy RHSMode=1 systems, the strong-preconditioner fallback is now enabled
automatically once the active system exceeds ``SFINCS_JAX_RHSMODE1_STRONG_PRECOND_MIN``,
so difficult FP cases attempt a stronger angular block preconditioner before dense fallback.
The fallback now uses a residual-ratio gate; tune
``SFINCS_JAX_RHSMODE1_STRONG_PRECOND_RATIO`` (default: ``1e2``) to avoid expensive
fallbacks when the residual is only slightly above target.

When the input requests a fully coupled preconditioner (``preconditioner_species = preconditioner_x = preconditioner_xi = 0``),
``sfincs_jax`` now defaults to the Schur preconditioner for ``constraintScheme=2`` to avoid dense fallbacks while
preserving the constraint coupling. For tokamak-like cases (``N_zeta=1``) with
``|Er|`` below ``SFINCS_JAX_RHSMODE1_SCHUR_ER_ABS_MIN`` (default: ``0``),
the default switches to the cheaper theta-line preconditioner to reduce setup time.
Set ``SFINCS_JAX_RHSMODE1_SCHUR_TOKAMAK=1`` to force Schur in these cases.

**Sparse ILU (FP-heavy RHSMode=1).** For FP-heavy RHSMode=1 systems, a PETSc‑like
incomplete factorization is available to avoid dense fallback while retaining
matrix‑free accuracy. We form a sparsified operator :math:`\tilde{A}` and build
an ILU preconditioner :math:`M \approx \tilde{L}\tilde{U}` so that GMRES solves

.. math::

   M^{-1} A x = M^{-1} b,

reducing iterations while keeping the exact operator :math:`A` in the matvec.
When ``SFINCS_JAX_IMPLICIT_SOLVE=1`` (default), the ILU factors are converted to
dense triangular factors and applied with JAX triangular solves to keep
end‑to‑end differentiability. For fully JAX‑native runs (no SciPy), a sparse
Jacobi preconditioner is available that builds a sparsified operator
(:math:`\tilde{A}`) in JAX and applies a few weighted Jacobi sweeps,

.. math::

   x^{(k+1)} = x^{(k)} + \omega D^{-1} (b - \tilde{A} x^{(k)}),

as a differentiable approximation to :math:`\tilde{A}^{-1}`. Explicit solves can
apply SciPy’s sparse ILU and optionally use the sparse operator for matvecs.
References: GMRES [#saad86]_, ILU/Preconditioning surveys [#benzi02]_.

Implementation: ``sfincs_jax.v3_driver`` (``_build_sparse_ilu_from_matvec`` and
the RHSMode=1 sparse fallback). Controls:

- ``SFINCS_JAX_RHSMODE1_SPARSE_PRECOND`` (auto/on/off/jax/scipy)
- ``SFINCS_JAX_RHSMODE1_SPARSE_OPERATOR`` (optional sparse matvec path)
- ``SFINCS_JAX_RHSMODE1_SPARSE_MATVEC`` (CSR matvec in explicit mode)
- ``SFINCS_JAX_RHSMODE1_SPARSE_DROP_TOL`` / ``SFINCS_JAX_RHSMODE1_SPARSE_DROP_REL``
- ``SFINCS_JAX_RHSMODE1_SPARSE_ILU_DROP_TOL`` / ``SFINCS_JAX_RHSMODE1_SPARSE_ILU_FILL_FACTOR``
- ``SFINCS_JAX_RHSMODE1_SPARSE_ILU_DENSE_MAX`` (max size for JAX triangular apply)
- ``SFINCS_JAX_RHSMODE1_SPARSE_DENSE_CACHE_MAX`` (reuse assembled dense operator for fallback solves)
- ``SFINCS_JAX_RHSMODE1_SPARSE_ALLOW_NONDIFF`` (explicit-only override)
- ``SFINCS_JAX_RHSMODE1_SPARSE_JAX_MAX_MB`` (memory guard for JAX sparse assembly)
- ``SFINCS_JAX_RHSMODE1_SPARSE_JAX_SWEEPS`` / ``SFINCS_JAX_RHSMODE1_SPARSE_JAX_OMEGA``
  (Jacobi sweep count and relaxation factor)
- ``SFINCS_JAX_RHSMODE1_SPARSE_JAX_REG`` (diagonal regularization for the sparse Jacobi preconditioner)

**PAS x-block :math:`(\theta,\zeta)` preconditioner.** For PAS cases with
angular grids, ``sfincs_jax`` can build per‑species, per‑:math:`x` blocks over
the full :math:`(L,\theta,\zeta)` space. This captures angular coupling without
forming the full species block, and it is selected automatically when
:math:`L \times N_\theta \times N_\zeta` stays below
``SFINCS_JAX_RHSMODE1_XBLOCK_TZ_MAX``.

Implementation: ``sfincs_jax.v3_driver`` (``_build_rhsmode1_xblock_tz_preconditioner``).

These are cached to avoid recomputation. RHS-only gradients are excluded from the cache key
so scan points can reuse the same preconditioner blocks. Controls:

- ``SFINCS_JAX_RHSMODE1_PRECONDITIONER``
- ``SFINCS_JAX_RHSMODE1_SPECIES_BLOCK_MAX`` (auto cap for PAS species-block preconditioning)
- ``SFINCS_JAX_RHSMODE1_XBLOCK_TZ_MAX`` (auto cap for PAS per‑x :math:`(\theta,\zeta)` preconditioning)
- ``SFINCS_JAX_RHSMODE1_SXBLOCK_MAX`` (auto cap for FP species×(x,L) blocks)
- ``SFINCS_JAX_RHSMODE1_COLLISION_PRECOND_KIND``
- ``SFINCS_JAX_RHSMODE1_COLLISION_SXBLOCK_MAX`` / ``SFINCS_JAX_RHSMODE1_COLLISION_XBLOCK_MAX``
- ``SFINCS_JAX_RHSMODE1_SCHUR_MODE`` / ``SFINCS_JAX_RHSMODE1_SCHUR_FULL_MAX``
- ``SFINCS_JAX_RHSMODE1_SCHUR_AUTO_MIN`` (auto Schur cutoff by total size)
- ``SFINCS_JAX_PRECOND_MAX_MB`` / ``SFINCS_JAX_PRECOND_CHUNK`` (cap memory during block assembly)
- ``SFINCS_JAX_PRECOND_DTYPE`` (default ``auto``; ``float32`` or ``float64`` to override)
- ``SFINCS_JAX_PRECOND_FP32_MIN_SIZE`` (threshold for auto mixed precision)

**KSP history cost.** PETSc-style KSP residual histories and iteration counts are
computed via an additional SciPy solve (to match the PETSc text). For large Krylov
counts this can dominate runtime, so the defaults now skip these when the estimated
iteration count exceeds ``SFINCS_JAX_KSP_HISTORY_MAX_ITER`` /
- ``SFINCS_JAX_RHSMODE1_PAS_XMG_MIN`` (auto switch to the lightweight PAS x‑multigrid
  preconditioner for large systems; default ``50000``)
- ``SFINCS_JAX_RHSMODE1_XMG_STRIDE`` (coarse‑x stride for the PAS x‑multigrid preconditioner)
- ``SFINCS_JAX_RHSMODE1_PAS_XDIAG_MIN`` (auto switch to point‑block x‑diagonal preconditioner for large PAS runs; default disabled)
- ``SFINCS_JAX_RHSMODE1_XBLOCK_TZ_LMAX`` (truncate L in PAS x‑block :math:`(\theta,\zeta)` preconditioning)
``SFINCS_JAX_SOLVER_ITER_STATS_MAX_ITER``. Raise those caps (or set to ``none``)
only when strict per-iteration history is required.

**Mixed-precision preconditioning.** With ``SFINCS_JAX_PRECOND_DTYPE=auto`` (default),
preconditioner blocks switch to float32 once the estimated system size exceeds
``SFINCS_JAX_PRECOND_FP32_MIN_SIZE`` (global) or the per-block size exceeds
``SFINCS_JAX_PRECOND_FP32_MIN_BLOCK`` (per-block), while Krylov iterations remain
in float64.

**Lightweight profiling.** Set ``SFINCS_JAX_PROFILE=1`` to emit coarse timing and
memory marks during RHSMode=1 solves (operator build, RHS assembly, preconditioner
construction, strong-preconditioner fallback). The output looks like:

.. code-block:: text

   profiling: operator_built dt_s=0.42 total_s=0.42 rss_mb=512.0 drss_mb=35.0 device_mb=na
   profiling: rhs_assembled dt_s=0.08 total_s=0.50 rss_mb=515.0 drss_mb=38.0 device_mb=na

.. [#saad86] Y. Saad and M. Schultz, “GMRES: A generalized minimal residual algorithm for
   solving nonsymmetric linear systems,” *SIAM J. Sci. Stat. Comput.* 7(3), 1986.
.. [#benzi02] M. Benzi, “Preconditioning techniques for large linear systems: a survey,”
   *J. Comput. Phys.* 182(2), 2002.
   profiling: rhs1_precond_build_start dt_s=0.00 total_s=0.50 ...
   profiling: rhs1_precond_build_done dt_s=1.25 total_s=1.75 ...

This is intentionally low overhead and does not require external profilers. For
detailed JAX tracing, use ``jax.profiler`` or standard tools, but keep them off
for parity runs.

**XLA dump profiling.** For kernel-level inspection, you can dump HLO/LLVM with
``XLA_FLAGS=--xla_dump_to=/tmp/sfincs_xla`` (optionally add
``--xla_dump_hlo_as_text``). This is heavier and should be used only for
targeted performance investigations.

Matvec fusion for collisionless + drift terms
---------------------------------------------

**Technique.** Accumulate collisionless streaming, ExB, magnetic-drift, and Er
drift contributions in a single static sum expression to reduce Python overhead
while keeping the matvec control‑flow free (required for JAX GMRES/BiCGStab).

**Implementation.**

- ``SFINCS_JAX_FUSED_MATVEC`` (default enabled) in ``sfincs_jax.v3_fblock``.

**Notes.** Avoid ``lax.scan``/``lax.fori_loop`` inside the matvec used by JAX
iterative solvers: they assert on control‑flow. The collision operators (PAS/FP)
remain separate so remat/checkpointing controls and Phi1‑dependent variants stay intact.

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

**Precompute constants + cache.**

Factors depending only on geometry, species normalization, and grids
(:math:`w_x`, :math:`B/D`, prefactors, etc.) are precomputed once per transport run
and reused for all ``whichRHS`` solves.

**Implementation.**

- ``v3_transport_diagnostics_vm_only_precompute`` and
  ``v3_transport_diagnostics_vm_only_batch_op0_precomputed``.
- Cached by operator signature in ``sfincs_jax.transport_matrix`` to reuse
  geometry/species factors across repeated transport solves (default cache size: ``4``;
  override with ``SFINCS_JAX_TRANSPORT_DIAG_CACHE_MAX``).
- For large transport solves, diagnostics can be processed in chunks to reduce peak
  memory. Use ``SFINCS_JAX_TRANSPORT_DIAG_CHUNK`` (default: auto for
  ``N * total_size > 2e5``) to set an explicit chunk size.
- Rematerialization for transport diagnostics is enabled automatically at the same
  threshold (override with ``SFINCS_JAX_TRANSPORT_DIAG_REMAT``).

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
- ``SFINCS_JAX_RHSMODE1_RECYCLE_K`` (RHSMode=1 scan reuse with least-squares deflation).

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
  when Krylov iterations stagnate. The trigger uses the **true (unpreconditioned)**
  residual norm so the fallback still fires even if a left-preconditioned norm
  appears small (parity-first behavior).
- For RHSMode=1 runs with ``includePhi1InCollisionOperator = .true.``, small systems
  bypass the Newton–Krylov inner GMRES step and take a dense Newton step instead.
  This avoids GMRES setup cost and matches Fortran parity for Phi1‑collision fixtures.
  The cutoff is controlled by ``SFINCS_JAX_PHI1_NK_DENSE_CUTOFF`` (default: ``5000``).
- Transport dense fallback is **disabled** unless explicitly requested, but a
  dense retry is enabled for RHSMode=2/3 when the active system size is modest.

Controls:

- ``SFINCS_JAX_RHSMODE1_DENSE_FALLBACK_MAX`` (default: ``400``).
- ``SFINCS_JAX_RHSMODE1_DENSE_FP_MAX`` (default: ``5000``) for full Fokker–Planck
  cases (``collisionOperator=0``).
- ``SFINCS_JAX_RHSMODE1_DENSE_PAS_MAX`` (default: ``5000``) for PAS/constraintScheme=2
  cases (notably DKES trajectories).
- ``SFINCS_JAX_RHSMODE1_DENSE_FALLBACK_RATIO`` (default: ``1e2``). Dense fallback
  only triggers when ``||r|| / target`` exceeds this ratio (set ``<= 0`` to always
  allow the fallback).
- ``SFINCS_JAX_RHSMODE1_DENSE_SHORTCUT_RATIO`` (default: ``1e6``). When the residual
  ratio exceeds this threshold, ``sfincs_jax`` skips sparse ILU and other heavy
  fallbacks and goes straight to the dense solve (if enabled). This avoids wasting
  time on ILU builds when dense fallback is inevitable.
- ``SFINCS_JAX_RHSMODE1_DENSE_PROBE`` (default: on). Perform a cheap single-step
  preconditioner probe (one matvec) and, if the residual ratio still exceeds
  ``SFINCS_JAX_RHSMODE1_DENSE_SHORTCUT_RATIO``, skip stage-2/strong Krylov attempts
  and proceed directly to the dense fallback.
- ``SFINCS_JAX_LINEAR_STAGE2_RATIO`` (default: ``1e2``). Stage-2 GMRES only runs
  when ``||r|| / target`` exceeds this ratio (set ``<= 0`` to always allow).
- ``SFINCS_JAX_TRANSPORT_DENSE_RETRY_MAX`` (default: ``3000`` for RHSMode=2/3).
- ``SFINCS_JAX_TRANSPORT_DENSE_FALLBACK`` / ``SFINCS_JAX_TRANSPORT_DENSE_FALLBACK_MAX``.
- ``SFINCS_JAX_TRANSPORT_DENSE_MAX_MB`` (default: ``128``). Disable dense transport
  fallbacks when the dense matrix would exceed this memory budget. If float64
  exceeds the limit but float32 does not, the fallback switches to float32 with
  one refinement step.
- ``SFINCS_JAX_TRANSPORT_DENSE_PRECOND_MAX_MB`` (default: ``min(32, dense_max_mb)``).
  Disables dense LU preconditioners when they would exceed the memory budget.
- ``SFINCS_JAX_DENSE_ASSEMBLE_JIT``: JIT-compile dense matrix assembly
  (auto by default: off for ``n<=800``, on for larger matrices).
- ``SFINCS_JAX_DENSE_MAX`` (default: ``8000``): guardrail for dense solves
  (max vector size before dense solve is disallowed).
- ``SFINCS_JAX_DENSE_BLOCK``: column block size for dense assembly (auto block size
  of ``128`` is used when ``n>=1000`` and no explicit block is set).
- ``SFINCS_JAX_DENSE_BLOCK``: assemble dense matrices in column blocks to cap peak memory.

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
