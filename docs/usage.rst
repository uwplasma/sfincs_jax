Usage
=====

Parsing an input file
---------------------

.. code-block:: python

   from sfincs_jax.namelist import read_sfincs_input

   nml = read_sfincs_input("input.namelist")
   print(nml.group("geometryParameters")["GEOMETRYSCHEME"])

Building v3 grids and geometry
------------------------------

.. code-block:: python

   from sfincs_jax.v3 import grids_from_namelist, geometry_from_namelist

   grids = grids_from_namelist(nml)
   geom = geometry_from_namelist(nml=nml, grids=grids)

Applying operator building blocks
---------------------------------

Collisionless v3 operator slice (streaming + mirror):

.. code-block:: python

   import jax.numpy as jnp
   import numpy as np

   from sfincs_jax.collisionless import CollisionlessV3Operator, apply_collisionless_v3

   species = nml.group("speciesParameters")
   t_hats = jnp.asarray(np.atleast_1d(np.asarray(species["THATS"], dtype=float)))
   m_hats = jnp.asarray(np.atleast_1d(np.asarray(species["MHATS"], dtype=float)))

   op = CollisionlessV3Operator(
       x=grids.x,
       ddtheta=grids.ddtheta,
       ddzeta=grids.ddzeta,
       b_hat=geom.b_hat,
       b_hat_sup_theta=geom.b_hat_sup_theta,
       b_hat_sup_zeta=geom.b_hat_sup_zeta,
       db_hat_dtheta=geom.db_hat_dtheta,
       db_hat_dzeta=geom.db_hat_dzeta,
       t_hats=t_hats,
       m_hats=m_hats,
       n_xi_for_x=grids.n_xi_for_x,
   )

   f = jnp.zeros((t_hats.size, grids.x.size, grids.n_xi, grids.theta.size, grids.zeta.size))
   y = apply_collisionless_v3(op, f)

Pitch-angle scattering collisions (``collisionOperator = 1`` without Phi1):

.. code-block:: python

   from sfincs_jax.collisions import make_pitch_angle_scattering_v3_operator, apply_pitch_angle_scattering_v3

   z_s = jnp.asarray(np.atleast_1d(np.asarray(species["ZS"], dtype=float)))
   n_hats = jnp.asarray(np.atleast_1d(np.asarray(species["NHATS"], dtype=float)))

   phys = nml.group("physicsParameters")
   nu_n = float(phys["NU_N"])

   cop = make_pitch_angle_scattering_v3_operator(
       x=grids.x,
       z_s=z_s,
       m_hats=m_hats,
       n_hats=n_hats,
       t_hats=t_hats,
       nu_n=nu_n,
       n_xi_for_x=grids.n_xi_for_x,
       n_xi=int(grids.n_xi),
   )

   y_col = apply_pitch_angle_scattering_v3(cop, f)

Running the Fortran v3 executable
---------------------------------

.. code-block:: bash

   export SFINCS_FORTRAN_EXE=/path/to/sfincs/fortran/version3/sfincs
   sfincs_jax run-fortran --input /path/to/input.namelist

.. tip::

   All CLI subcommands support ``-v/--verbose`` (repeatable), ``-q/--quiet``,
   and ``--fortran-stdout``/``--no-fortran-stdout`` for strict stdout mirroring.

If you are developing from a source checkout and have not installed the console script,
you can invoke the CLI module directly:

.. code-block:: bash

   python -m sfincs_jax run-fortran --input /path/to/input.namelist

Solving a supported v3 linear run (matrix-free)
------------------------------------------------------------

.. code-block:: bash

   sfincs_jax solve-v3 --input /path/to/input.namelist --out-state stateVector.npy

.. code-block:: bash

   python -m sfincs_jax solve-v3 --input /path/to/input.namelist --out-state stateVector.npy

.. note::

   The matrix-free solve path is parity-tested on a growing subset of v3 options.
   In particular, VMEC ``geometryScheme=5`` is now supported for the parity-tested tiny PAS case
   (see ``tests/ref/pas_1species_PAS_noEr_tiny_scheme5.input.namelist``).

.. note::

   For end-to-end differentiation, build inputs via the Python API and keep the computation in JAX.
   File I/O, VMEC/Boozer parsing, and SciPy-based solver-history logging use NumPy and are not
   differentiable. Disable history logging with ``SFINCS_JAX_FORTRAN_STDOUT=0`` and
   ``SFINCS_JAX_SOLVER_ITER_STATS=0`` when tracing gradients.

.. note::

   The default ``--solve-method auto`` uses GMRES for RHSMode=1 (parity-first) and BiCGStab for
   RHSMode=2/3 transport solves. BiCGStab remains available for low-memory RHSMode=1 runs via
   ``--solve-method bicgstab``. Transport solves apply a cheap collision-diagonal
   preconditioner by default, while RHSMode=1 preconditioning follows the v3 namelist defaults
   (point-block Jacobi unless line preconditioners are requested). For ``constraintScheme=2``,
   ``sfincs_jax`` will auto-try a Schur-complement strong preconditioner if the initial solve
   stalls, preserving the source constraints. For PAS tokamak-like ``N_zeta=1`` cases with
   constraint projection enabled, ``sfincs_jax`` upgrades to the ``xblock_tz`` preconditioner by
   default to reduce Krylov iterations. For strict PETSc-style iteration histories, use
   ``--solve-method incremental``.

Solver controls (environment variables)
---------------------------------------

Defaults are parity-first: running with the same input.namelist as v3 should produce
matching outputs and stdout without extra flags. The environment variables below are
advanced tuning knobs for performance or debugging.

Some solver options are intentionally exposed as environment variables so you can tune
performance without changing the input file:

- ``SFINCS_JAX_ACTIVE_DOF``: controls active-DOF reduction when ``Nxi_for_x`` truncation is present.

  - ``auto`` (default): enabled for RHSMode=2/3, and for RHSMode=1 when ``includePhi1=false``.
  - ``1``/``true``: always enable.
  - ``0``/``false``: always disable.

- ``SFINCS_JAX_RHSMODE1_SOLVE_METHOD``: choose the RHSMode=1 linear solve backend:

  - ``auto`` (default): GMRES (parity-first) with stage-2 fallback on stagnation.
  - ``bicgstab``: force BiCGStab for a low-memory Krylov solve (with GMRES fallback on stagnation).
  - ``dense``: assemble the dense operator from matvecs and solve directly (fast for tiny fixtures,
    but scales poorly).
  - ``incremental`` or ``batched``: matrix-free GMRES (higher memory, often robust).

- ``SFINCS_JAX_RHSMODE1_GMRES_SMALL_MAX``: force GMRES for RHSMode=1 when the total
  system size is below this threshold (default: ``600``). Set to ``0`` to disable.

- ``SFINCS_JAX_RHSMODE1_PRECONDITIONER`` (GMRES only): optional RHSMode=1 preconditioning.

  - ``point`` (or ``1``): point-block Jacobi on local (x,L) unknowns at each :math:`(\theta,\zeta)`.
  - ``collision``: collision-diagonal preconditioner (PAS/FP + identity shift).
  - ``xmg``: coarse x-grid correction built from PAS/FP diagonals (lightweight; reduces
    x‑coupling stiffness without full block setup).
  - ``sxblock``: species×(x,L) block at each :math:`(\theta,\zeta)` (includes inter-species coupling).
  - ``sxblock_tz``: per‑:math:`L` block over species×x×:math:`(\theta,\zeta)` (captures angular coupling).
  - ``xblock_tz``: PAS per‑:math:`x` block over :math:`(L,\theta,\zeta)` (captures angular coupling).
  - ``xblock_tz_lmax``: PAS per‑:math:`x` block over :math:`(L,\theta,\zeta)` using only the lowest
    ``L`` modes (see ``SFINCS_JAX_RHSMODE1_XBLOCK_TZ_LMAX``).
  - ``point_xdiag``: point-block Jacobi with **x‑diagonal** blocks (retains xi coupling, drops x coupling).
  - ``theta_line``: theta-line block preconditioner (stronger, higher setup cost).
  - ``zeta_line``: zeta-line block preconditioner (stronger, higher setup cost).
  - ``theta_dd``: block-diagonal theta preconditioner (domain-decomposition prototype).
  - ``zeta_dd``: block-diagonal zeta preconditioner (domain-decomposition prototype).
  - ``adi``: apply the theta-line and zeta-line preconditioners sequentially (strongest of the built-ins,
    but also the most expensive).
  - ``schur``: Schur-complement preconditioner for ``constraintScheme=2`` that keeps source constraints.
  - ``0``: disable.

- ``SFINCS_JAX_RHSMODE1_DD_BLOCK_T``: theta-block size for ``theta_dd`` (default: ``8``).
- ``SFINCS_JAX_RHSMODE1_DD_BLOCK_Z``: zeta-block size for ``zeta_dd`` (default: ``8``).

- ``SFINCS_JAX_RHSMODE1_COLLISION_PRECOND_MIN``: minimum ``total_size`` before the default
  RHSMode=1 preconditioner switches to the collision-diagonal option (default: ``600``).

- ``SFINCS_JAX_RHSMODE1_XBLOCK_TZ_MAX``: maximum per‑:math:`x` block size
  (:math:`L \times N_\theta \times N_\zeta`) before the PAS xblock_tz preconditioner
  is disabled in auto mode (default: ``1200``).

- ``SFINCS_JAX_RHSMODE1_PAS_XMG_MIN``: for large PAS systems that request full
  preconditioning, switch to the lightweight x‑multigrid preconditioner when
  ``total_size`` exceeds this threshold (default: ``50000``).

- ``SFINCS_JAX_RHSMODE1_XMG_STRIDE``: coarse‑grid stride for the RHSMode=1 x‑multigrid
  preconditioner (default: ``2``; falls back to ``SFINCS_JAX_XMG_STRIDE`` if unset).

- ``SFINCS_JAX_RHSMODE1_COLLISION_PRECOND_KIND``: choose the collision preconditioner flavor
  when ``SFINCS_JAX_RHSMODE1_PRECONDITIONER=collision`` or BiCGStab preconditioning is enabled.

  - ``xblock``: invert the per-species x-block for each L using the FP self-collision matrix
    (stronger for some FP cases, slightly higher apply cost).
  - ``sxblock``: invert the full species×x block for each L using the FP collision matrix
    (strongest option for FP cases; higher apply cost).
  - ``diag``: use the collision diagonal only (PAS/FP + identity shift).

- ``SFINCS_JAX_RHSMODE1_COLLISION_SXBLOCK_MAX``: auto-select the FP species×x block
  collision preconditioner when ``SFINCS_JAX_RHSMODE1_COLLISION_PRECOND_KIND`` is unset
  and ``S * X`` is below this threshold (default: ``64``). Set to ``-1`` to disable.

- ``SFINCS_JAX_RHSMODE1_COLLISION_XBLOCK_MAX``: if the FP species×x block is disabled,
  auto-select the per-species x-block collision preconditioner when ``N_x`` is below
  this threshold (default: ``256``). Set to ``-1`` to disable.

- ``SFINCS_JAX_RHSMODE1_FP_LOW_RANK_K``: use a low-rank Woodbury correction (rank ``K``)
  for the FP species×x collision preconditioner (``sxblock``). ``auto`` (default when
  unset) selects a small rank (up to 8) for larger FP blocks. Set to ``0`` to disable.
  ``SFINCS_JAX_FP_LOW_RANK_K`` provides a global fallback.

- ``SFINCS_JAX_RHSMODE1_SCHUR_EPS``: diagonal safeguard for the constraintScheme=2 Schur
  complement (default: ``1e-14``). Smaller values tighten the constraint solve but can
  amplify noise.

- ``SFINCS_JAX_RHSMODE1_SCHUR_AUTO_MIN``: when ``constraintScheme=2`` and PAS collisions
  are active, auto-select Schur preconditioning if ``total_size`` exceeds this threshold
  (default: ``2500``). Set to ``0`` to always allow auto Schur.

- ``SFINCS_JAX_RHSMODE1_SXBLOCK_MAX``: auto-select the RHSMode=1 species×(x,L) block
  preconditioner for FP cases when the per‑:math:`(\theta,\zeta)` block size
  (``S * sum_x N_{\xi,x}``) is below this threshold (default: ``64``).

- ``SFINCS_JAX_RHSMODE1_SXBLOCK_TZ_MAX``: auto-select the per‑:math:`L` species×x×:math:`(\theta,\zeta)`
  block preconditioner when the block size (``S * N_x * N_\theta * N_\zeta``) is below this threshold.
  Default ``0`` disables the auto-selection.

- ``SFINCS_JAX_PRECOND_DTYPE``: dtype for preconditioner blocks (default: ``auto`` uses
  float32 for large systems and float64 otherwise). ``SFINCS_JAX_PRECOND_FP32_MIN_SIZE``
  controls the global auto threshold; ``SFINCS_JAX_PRECOND_FP32_MIN_BLOCK`` controls
  the per-block threshold.

- ``SFINCS_JAX_RHSMODE1_BICGSTAB_PRECOND``: optional RHSMode=1 BiCGStab preconditioning.

  - ``collision`` (default): collision-diagonal preconditioner (PAS/FP + identity shift).
  - ``rhs1``/``same``: reuse the RHSMode=1 GMRES preconditioner for BiCGStab.
  - ``0``: disable.

- ``SFINCS_JAX_BICGSTAB_FALLBACK``: control when BiCGStab falls back to GMRES.

  - ``strict``/``1`` (default): fallback if the residual exceeds tolerance (parity-first).
  - ``0``/``loose``: fallback only on non-finite residuals (performance-first).

- ``SFINCS_JAX_TRANSPORT_PRECOND``: RHSMode=2/3 transport preconditioner.

  - ``auto`` (default): with the default BiCGStab transport solver, use the collision-diagonal
    preconditioner. When GMRES is selected and the FP collision operator is available, ``auto``
    upgrades to a lightweight **species×x block-Jacobi** preconditioner (per-L) for small systems.
  - ``block``/``block_jacobi``: local (x,L) block-Jacobi preconditioner built from a
    simplified transport operator (stronger, higher setup cost).
  - ``sxblock``/``block_sx``/``species_x``: lightweight species×x block-Jacobi built from
    the FP collision operator (no matvec assembly; stronger than diagonal for FP cases).
  - ``xmg``/``multigrid``: two-level additive x-grid preconditioner (coarse x solve +
    fine diagonal smoother).
  - ``collision``: collision-diagonal preconditioner (PAS/FP + identity shift).
  - ``0``/``none``: disable.

- ``SFINCS_JAX_TRANSPORT_PRECOND_BLOCK_MAX``: size threshold for ``auto`` to select
  block-Jacobi preconditioning (default: ``5000``).

- ``SFINCS_JAX_TRANSPORT_PRECOND_BLOCK_REG``: regularization added to transport block
  preconditioner diagonal blocks (default: ``1e-10``).

- ``SFINCS_JAX_TRANSPORT_FP_LOW_RANK_K``: low-rank Woodbury correction (rank ``K``)
  for the FP species×x transport preconditioner. ``auto`` (default when unset) selects
  a small rank (up to 8) for larger FP blocks. Set to ``0`` to disable.
  ``SFINCS_JAX_FP_LOW_RANK_K`` provides a global fallback.

- ``SFINCS_JAX_XMG_STRIDE``: coarse-grid stride for ``xmg`` transport preconditioning
  (default: ``2``).

- ``SFINCS_JAX_TRANSPORT_GMRES_RESTART``: GMRES restart length for transport fallback (default: 40).

- ``SFINCS_JAX_TRANSPORT_FORCE_DENSE``: force dense transport solves (debugging only; quadratic cost).

- ``SFINCS_JAX_TRANSPORT_DENSE_FALLBACK``: allow dense transport fallback for small ill-conditioned
  cases (disabled by default). When enabled, set ``SFINCS_JAX_TRANSPORT_DENSE_FALLBACK_MAX`` to
  bound the system size.

- ``SFINCS_JAX_TRANSPORT_DENSE_PRECOND_MAX``: enable a dense LU preconditioner for transport solves
  when the system size is below the specified threshold (default: ``1600`` for RHSMode=2,
  ``600`` for RHSMode=3).

- ``SFINCS_JAX_TRANSPORT_PARALLEL``: parallelize RHSMode=2/3 ``whichRHS`` solves
  across processes (``off``/``process``/``auto``).

- ``SFINCS_JAX_TRANSPORT_PARALLEL_WORKERS``: number of worker processes for parallel
  transport solves.

- ``SFINCS_JAX_CORES``: high‑level CPU parallelism knob. When set to ``N`` > 1,
  ``sfincs_jax`` enables process‑parallel ``whichRHS`` solves **and** exposes ``N``
  host devices for optional sharded matvecs. This gives a single user‑facing knob
  for "use N cores". Set ``SFINCS_JAX_SHARD=0`` to disable sharded matvecs while
  keeping process parallelism.
- ``SFINCS_JAX_XLA_THREADS``: opt‑in to setting the XLA CPU thread count based on
  ``SFINCS_JAX_CORES``. Some JAX builds do not recognize the
  ``--xla_cpu_parallelism_threads`` flag, so this is disabled by default.

- ``SFINCS_JAX_TRANSPORT_RECYCLE_STATE``: reuse saved Krylov recycle vectors across runs
  when ``SFINCS_JAX_STATE_IN`` is set (default: enabled; set to ``0`` to disable).

- ``SFINCS_JAX_MATVEC_SHARD_AXIS``: enable experimental sharded matvecs
  (``theta``, ``zeta``, ``x``, ``flat``, or ``auto``) when multiple devices are available.

- ``SFINCS_JAX_GMRES_PRECONDITION_SIDE``: side for applying the preconditioner in GMRES.

  - ``left`` (default): solve :math:`P^{-1} A x = P^{-1} b`.
  - ``right``: solve :math:`A P^{-1} y = b` and set :math:`x = P^{-1} y` (PETSc-like default for GMRES).
  - ``none``: ignore any preconditioner (debugging).

- ``SFINCS_JAX_PHI1_NK_DENSE_CUTOFF``: when ``includePhi1 = .true.``, use a dense Newton
  step instead of GMRES inside the Newton–Krylov solve for systems with ``total_size``
  below this cutoff (default: ``5000``). This improves parity and runtime for small
  Phi1 fixtures.

- ``SFINCS_JAX_LINEAR_STAGE2``: enable a second GMRES stage with a larger iteration budget when
  the first stage stagnates (default: auto-enabled for RHSMode=1 without Phi1 when GMRES is selected).
- ``SFINCS_JAX_LINEAR_STAGE2_RATIO``: only run stage-2 when ``||r|| / target`` exceeds the
  given ratio (default: ``1e2``; set ``<= 0`` to always allow stage-2).

- ``SFINCS_JAX_IMPLICIT_SOLVE``: control implicit differentiation through linear solves.

  - Default: enabled (implicit gradients via ``jax.lax.custom_linear_solve``).
  - ``0``/``false``: disable (differentiate through Krylov iterations; slower / higher memory).

- ``SFINCS_JAX_PRECOND_DTYPE``: preconditioner storage dtype (``float64`` default).
  Set to ``float32`` to reduce memory and speed up preconditioner application while
  keeping the Krylov solve in float64. ``auto``/``mixed`` switches to float32 when
  the estimated preconditioner size exceeds ``SFINCS_JAX_PRECOND_FP32_MIN_SIZE``.

- ``SFINCS_JAX_PRECOND_FP32_MIN_SIZE``: minimum preconditioner size (rough scalar count)
  before ``SFINCS_JAX_PRECOND_DTYPE=auto`` switches to float32 (default: ``20000``).

- ``SFINCS_JAX_STATE_IN``/``SFINCS_JAX_STATE_OUT``: path for reading/writing Krylov
  recycle states (used for scan warm-starting and multi-RHS reuse). RHSMode=1 states
  now store a short history of prior solutions for recycling.

- ``SFINCS_JAX_SCAN_RECYCLE``: enable automatic scan-level Krylov recycling in
  :func:`sfincs_jax.scans.run_er_scan` by wiring ``SFINCS_JAX_STATE_IN/OUT`` between
  adjacent scan points (default: disabled).

- ``SFINCS_JAX_FBLOCK_CACHE``: reuse geometry- and physics-dependent operator blocks
  across repeated runs with the same namelist settings (default: enabled).

- ``SFINCS_JAX_FBLOCK_CACHE_MAX``: maximum number of cached f-block operator entries
  (default: ``8``).

- ``SFINCS_JAX_FUSED_MATVEC``: fuse collisionless + drift contributions into a
  single static sum (control‑flow free so JAX GMRES/BiCGStab remain stable).
  Default: enabled. Set to ``0`` to use the unfused sequential path (debugging).

- ``SFINCS_JAX_REMAT_COLLISIONS``: enable gradient checkpointing around collision operators to
  reduce peak memory during autodiff (default: auto, based on size threshold).

- ``SFINCS_JAX_REMAT_COLLISIONS_MIN``: minimum ``f`` size before auto-remat triggers
  (default: ``20000``).

- ``SFINCS_JAX_REMAT_TRANSPORT_DIAGNOSTICS``: enable gradient checkpointing around transport
  diagnostics to reduce peak memory during autodiff (default: auto, based on size threshold).

- ``SFINCS_JAX_REMAT_TRANSPORT_DIAGNOSTICS_MIN``: minimum transport-stack size before auto-remat
  triggers (default: ``20000``).

- ``SFINCS_JAX_TRANSPORT_DIAG_PRECOMPUTE``: reuse geometry/species diagnostics factors across
  all ``whichRHS`` solves (default: enabled). Set to ``0``/``false`` to disable.

- ``SFINCS_JAX_PRECOMPILE``: ahead-of-time compile core kernels when JAX persistent compilation
  cache is enabled (default: auto when ``JAX_COMPILATION_CACHE_DIR`` is set).

- ``JAX_COMPILATION_CACHE_DIR``: set a persistent compilation cache directory to reuse compiled
  artifacts across runs (recommended for reduced-suite and batch runs).
- ``SFINCS_JAX_COMPILATION_CACHE_DIR``: convenience override for the default cache path when
  ``JAX_COMPILATION_CACHE_DIR`` is not set.

- ``SFINCS_JAX_CPU_DEVICES``: request multiple host CPU devices for JAX SPMD/pjit.
  Must be set **before** importing JAX (i.e., before running `python -m sfincs_jax`).

- ``SFINCS_JAX_MATVEC_SHARD_AXIS``: control SPMD sharding of the matvec along ``theta``,
  ``zeta``, ``x``, ``flat``, or ``auto``. ``auto`` chooses the larger of ``Ntheta``/``Nzeta``
  when multiple devices are present. ``x`` is a fallback for cases where odd
  ``Ntheta``/``Nzeta`` block theta/zeta sharding. ``flat`` shards the full state
  vector evenly across devices.
- ``SFINCS_JAX_MATVEC_SHARD_MIN_TZ``: minimum ``Ntheta * Nzeta`` before enabling
  auto sharding (default: ``128``).
- ``SFINCS_JAX_MATVEC_SHARD_MIN_X``: minimum ``Nx`` before auto selecting ``x``
  sharding (default: ``16``).
- ``SFINCS_JAX_MATVEC_SHARD_PREFER_X``: set to ``1`` to prefer ``x`` sharding when
  ``Nx`` exceeds the minimum.
- ``SFINCS_JAX_AUTO_SHARD``: set to ``0`` to disable auto sharding.
- ``SFINCS_JAX_SHARD``: shorthand to disable auto sharding even when
  ``SFINCS_JAX_CORES`` is set. Use ``0``/``false`` to keep single‑device matvecs.
- ``SFINCS_JAX_SHARD_PAD``: pad odd ``Ntheta``/``Nzeta`` internally so theta/zeta
  sharding can use even device counts, and pad ``Nx`` when x‑sharding is requested
  but ``Nx`` is not divisible by the device count (default: enabled). Padding adds
  ghost planes with zero weights and does not change outputs.

- ``SFINCS_JAX_GMRES_DISTRIBUTED``: enable distributed GMRES when using ``flat``
  sharding. Set to ``1`` to run the Krylov solver under `pjit`, keeping vectors
  sharded across devices. Default: off (fall back to single‑device GMRES).

- ``SFINCS_JAX_DISTRIBUTED``: enable JAX multi‑host initialization (default: off).
  When set, also provide:

  - ``SFINCS_JAX_PROCESS_ID``: this process rank (0‑based).
  - ``SFINCS_JAX_PROCESS_COUNT``: total number of processes.
  - ``SFINCS_JAX_COORDINATOR_ADDRESS``: host:port (or host) of the coordinator.
  - ``SFINCS_JAX_COORDINATOR_PORT``: port for the coordinator (default: 1234).

- ``SFINCS_JAX_GEOMETRY_CACHE``: enable/disable the geometry cache in ``geometry_from_namelist``
  (default: enabled).
- ``SFINCS_JAX_GEOMETRY_CACHE_PERSIST``: control persistent on‑disk geometry caching
  (default: enabled).
- ``SFINCS_JAX_GEOMETRY_CACHE_DIR``: override the geometry cache directory
  (default: ``~/.cache/sfincs_jax/geometry_cache``).

- ``SFINCS_JAX_OUTPUT_CACHE``: enable/disable caching of expensive output-only geometry fields
  (default: enabled).
- ``SFINCS_JAX_OUTPUT_CACHE_PERSIST``: control persistent on‑disk output caching
  (default: enabled).
- ``SFINCS_JAX_OUTPUT_CACHE_DIR``: override the output cache directory
  (default: ``~/.cache/sfincs_jax/output_cache``).

- ``SFINCS_JAX_TRANSPORT_RECYCLE_K``: recycle up to ``k`` previous Krylov solution vectors across
  successive ``whichRHS`` solves in transport-matrix runs. Set to ``0`` to disable.

- ``SFINCS_JAX_RHSMODE1_RECYCLE_K``: recycle up to ``k`` previous RHSMode=1 solution vectors
  (via least-squares deflation) when ``SFINCS_JAX_STATE_IN`` is provided. Set to ``0`` to
  disable (default: ``4``).

- ``SFINCS_JAX_TRANSPORT_DENSE_RETRY_MAX``: enable a dense retry when transport-matrix Krylov
  solves stagnate. The dense retry is applied only when the active system size is below the
  specified threshold (default: ``3000`` for RHSMode=2/3, ``0`` otherwise).
- ``SFINCS_JAX_TRANSPORT_DENSE_MAX_MB``: memory cap (MB) for dense transport retries. Dense
  transport solves are skipped once the estimated dense matrix exceeds this limit (default:
  ``128``).
- ``SFINCS_JAX_TRANSPORT_DENSE_BATCH_FALLBACK``: when a dense retry is triggered for any
  ``whichRHS`` in RHSMode=2/3 and the operator is identical across RHS, solve **all RHS in a
  single dense batch** (default: on). Disable with ``0``/``false`` if you want per‑RHS dense
  retries only.

- ``SFINCS_JAX_RHSMODE1_PROJECT_NULLSPACE``: control constraintScheme=1 nullspace projection
  for linear RHSMode=1 solves.

  - Default: enabled when ``constraintScheme=1`` and ``includePhi1=false``.
  - ``0``/``false``: disable (use raw GMRES solution).

- ``SFINCS_JAX_RHSMODE1_DENSE_FALLBACK_MAX``: enable a dense fallback solve for RHSMode=1
  when GMRES stagnates. This is only applied when the active system size is below the
  specified threshold (default: ``400``; see the FP-specific override below).
- ``SFINCS_JAX_RHSMODE1_DENSE_FP_CUTOFF``: for small full FP systems (``collisionOperator=0``),
  `sfincs_jax` now **defaults to a direct dense solve** instead of Krylov to match
  Fortran and avoid expensive fallback paths. This cutoff controls the active-size
  threshold for that default (default: same as
  ``SFINCS_JAX_RHSMODE1_DENSE_ACTIVE_CUTOFF``).
- ``SFINCS_JAX_RHSMODE1_DENSE_FP_MAX``: override the RHSMode=1 dense fallback ceiling for
  full Fokker–Planck (``collisionOperator=0``) cases (default: ``5000``).
- ``SFINCS_JAX_RHSMODE1_DENSE_PAS_MAX``: override the RHSMode=1 dense fallback ceiling for
  PAS/constraintScheme=2 cases. Dense PAS fallback is **disabled by default** to
  preserve parity; set this explicitly (e.g. ``5000``) to enable it.
- ``SFINCS_JAX_RHSMODE1_DENSE_FALLBACK_RATIO``: only run the dense fallback when
  ``||r|| / target`` exceeds the given ratio (default: ``1e2``; set ``<= 0`` to always allow).
- ``SFINCS_JAX_RHSMODE1_DENSE_SHORTCUT_RATIO``: skip sparse ILU and other expensive
  fallbacks and go directly to the dense solve when ``||r|| / target`` exceeds this
  ratio (default: ``1e6``; set ``<= 0`` to disable the shortcut).
- ``SFINCS_JAX_RHSMODE1_DENSE_PROBE``: before expensive Krylov fallbacks, run a
  one-step preconditioner probe (one matvec) and jump straight to the dense
  solve if the residual ratio still exceeds ``SFINCS_JAX_RHSMODE1_DENSE_SHORTCUT_RATIO``.
  Disable with ``0``/``false`` if you want to always attempt full GMRES first.
- ``SFINCS_JAX_DENSE_MAX``: guardrail for dense solves (max vector size, default: ``8000``).
- ``SFINCS_JAX_RHSMODE1_FORCE_KRYLOV``: force RHSMode=1 to stay in Krylov mode even when the
  small-system dense defaults (FP/PAS) would otherwise trigger.
- ``SFINCS_JAX_PRECOND_PAS_MAX_COLS``: cap the column chunk size used when assembling
  PAS RHSMode=1 block preconditioners from matvecs. Lowering this reduces peak
  RSS during preconditioner assembly at the cost of extra matvecs (default: ``64``).

- ``SFINCS_JAX_RHSMODE1_PAS_XDIAG_MIN``: for large PAS systems that request a full
  preconditioner (``preconditioner_species = preconditioner_x = preconditioner_xi = 0``),
  prefer a **point‑block x‑diagonal** preconditioner over collision‑only when
  ``total_size`` exceeds this threshold (default: ``1e9``; effectively disabled unless
  you opt in). This is an experimental cheaper alternative to full PAS block preconditioners.
- ``SFINCS_JAX_RHSMODE1_XBLOCK_TZ_LMAX``: truncate the L dimension used by the
  PAS ``xblock_tz`` preconditioner (or ``xblock_tz_lmax``), reducing block size.
  This is used automatically for large PAS runs when ``xblock_tz_lmax`` is selected.

- ``SFINCS_JAX_RHSMODE1_SCHUR_MODE``: constraintScheme=2 Schur preconditioner mode
  (``auto``/``diag``/``full``). ``auto`` selects a dense Schur complement when the
  constraint size is below ``SFINCS_JAX_RHSMODE1_SCHUR_FULL_MAX``.

- ``SFINCS_JAX_RHSMODE1_SCHUR_FULL_MAX``: max constraint size for the dense Schur
  complement in ``auto`` mode (default: ``256``).

- ``SFINCS_JAX_PHI1_PRECOND_KIND``: Newton–Krylov preconditioner for includePhi1 solves
  (active when ``SFINCS_JAX_PHI1_USE_PRECONDITIONER`` is enabled and frozen linearization is used;
  frozen linearization is now opt‑in via ``SFINCS_JAX_PHI1_USE_FROZEN_LINEARIZATION``).

  - ``collision`` (default for includePhi1): collision-diagonal preconditioner.
  - ``block``/``block_jacobi``: RHSMode=1 block-Jacobi preconditioner (stronger).

- ``SFINCS_JAX_PHI1_FROZEN_JAC_CACHE``: reuse the frozen-RHS linearized Jacobian across
  Newton steps (default: enabled).

- ``SFINCS_JAX_PHI1_FROZEN_JAC_CACHE_EVERY``: rebuild the frozen-RHS linearized Jacobian
  every ``k`` Newton steps (default: ``1``).

- ``SFINCS_JAX_GMRES_MAX_MB``: memory cap for GMRES basis storage; used to auto-limit the
  restart value when ``SFINCS_JAX_GMRES_AUTO_RESTART`` is enabled (default: ``2048``).

- ``SFINCS_JAX_GMRES_AUTO_RESTART``: enable memory-aware GMRES restarts (default: enabled).

- ``SFINCS_JAX_PRECOND_MAX_MB``: memory cap (in MB) for RHSMode=1 preconditioner assembly.
  The preconditioner block assembly is chunked to keep peak memory below this target.

- ``SFINCS_JAX_PRECOND_CHUNK``: explicit column chunk size for RHSMode=1 preconditioner assembly
  (overrides ``SFINCS_JAX_PRECOND_MAX_MB`` when set).

- ``SFINCS_JAX_FORTRAN_STDOUT``: control strict Fortran-style stdout mirroring.

- ``SFINCS_JAX_FORTRAN_PETSC_OPTIONS_FALLBACK``: PETSc options string used when the
  Fortran binary aborts with MPICH MPI-init errors in reduced-suite runs.

  - ``1``/``true``: emit PETSc-like SNES/KSP iteration lines in addition to the standard v3 text.
  - ``0``/``false``: skip the extra iteration logs (useful for speed in tests).

  .. note::

     For strict KSP iteration-line parity, force a GMRES solve method (``incremental``/``batched``);
     BiCGStab does not produce GMRES-style history lines.

- ``SFINCS_JAX_SOLVER_ITER_STATS``: emit ``ksp_iterations=...`` lines in stdout for suite reporting.

  - ``1``/``true``: run a SciPy Krylov solve after the JAX solve to estimate iteration counts.
  - ``0``/``false``: disable (default outside the reduced-suite runner).
  - Because this invokes an extra SciPy solve, keep the iteration caps below for performance.

- ``SFINCS_JAX_SOLVER_ITER_STATS_MAX_SIZE``: skip iteration counting when the linear system size
  exceeds the provided threshold (useful when stats collection becomes too costly).

- ``SFINCS_JAX_SOLVER_ITER_STATS_MAX_ITER``: skip iteration counting when the estimated iteration
  count (``restart * maxiter`` for GMRES) exceeds the provided threshold (default: ``2000``).

- ``SFINCS_JAX_KSP_HISTORY_MAX_SIZE``: skip PETSc-style KSP residual history output when the
  linear system size exceeds the provided threshold (default: ``800``; set to ``none`` to
  always emit).

- ``SFINCS_JAX_KSP_HISTORY_MAX_ITER``: skip PETSc-style KSP residual history output when the
  estimated iteration count (``restart * maxiter`` for GMRES) exceeds the provided threshold
  (default: ``2000``).

- ``SFINCS_JAX_RHSMODE1_STRONG_PRECOND``: strong RHSMode=1 fallback preconditioner
  (``theta_line``, ``zeta_line``, ``adi``, or ``auto``). Default: ``auto`` for
  ``constraintScheme=2`` when the environment variable is unset, otherwise disabled
  unless explicitly set.
- ``SFINCS_JAX_RHSMODE1_STRONG_PRECOND_RATIO``: only run strong-preconditioner fallbacks
  when ``||r|| / target`` exceeds the given ratio (default: ``1e2``; set ``<= 0`` to always allow).

- ``SFINCS_JAX_RHSMODE1_SCHUR_BASE``: choose the base preconditioner used inside the
  constraint-aware Schur preconditioner (``theta_line``, ``zeta_line``, ``adi``, or
  ``point``). Default: ``auto`` (uses line preconditioning when angular coupling is present).
- ``SFINCS_JAX_RHSMODE1_SCHUR_TOKAMAK``: force Schur preconditioning for tokamak-like
  cases with ``N_zeta=1`` even when a cheaper theta-line preconditioner would be
  selected by default (set to ``1`` to force Schur).
- ``SFINCS_JAX_RHSMODE1_SCHUR_ER_ABS_MIN``: minimum ``|Er|`` for which tokamak-like
  cases default to Schur. When ``|Er|`` is below this threshold (default: ``0``),
  ``sfincs_jax`` uses the cheaper theta-line preconditioner for ``N_zeta=1`` cases.

- ``SFINCS_JAX_PAS_PROJECT_CONSTRAINTS``: enable PAS-specific constraint projection for
  ``constraintScheme=2`` RHSMode=1 solves (drop explicit source unknowns and enforce the
  normalized flux-surface-average constraint on ``L=0``; sources are recovered from the
  projected residual).

  - ``auto`` (default): enable for tokamak-like cases with ``N_zeta=1`` (excluding
    ``geometryScheme=1`` analytic tokamak inputs) **and** for DKES-trajectory runs,
    unless a fully coupled preconditioner is requested
    (``preconditioner_species = preconditioner_x = preconditioner_xi = 0``), since those
    cases converge without projection and match Fortran more strictly.
  - ``1``/``true``: force enable for all PAS ``constraintScheme=2`` cases.
  - ``0``/``false``: disable.

- ``SFINCS_JAX_PAS_SOURCE_ZERO_TOL``: for ``constraintScheme=2`` solves, zero-out tiny
  recovered source terms when their max-abs value is below this tolerance (default:
  ``2e-9``). This tightens parity with Fortran when sources should be numerically zero.

- ``SFINCS_JAX_ROSENBLUTH_METHOD``: choose how the Rosenbluth potential response matrices
  are computed for ``collisionOperator=0`` with ``xGridScheme=5/6``.

  - ``quadpack`` (default): match the Fortran v3 QUADPACK-based implementation for parity.
  - ``analytic``: faster analytic integrals (may differ at strict parity level).

- ``SFINCS_JAX_FP_STRICT_PARITY``: for ``collisionOperator=0`` multispecies runs, force a
  scalar-ordered accumulation of the FP cross-species coupling to match v3 ordering.

  - Default: enabled automatically for RHSMode=1 multispecies cases.
  - ``0``/``false``: disable (use faster vectorized accumulation).

Writing `sfincsOutput.h5` with `sfincs_jax`
--------------------------------------------------

.. code-block:: bash

   sfincs_jax write-output --input /path/to/input.namelist --out sfincsOutput.h5

.. code-block:: bash

   python -m sfincs_jax write-output --input /path/to/input.namelist --out sfincsOutput.h5

.. code-block:: python

   from pathlib import Path
   from sfincs_jax.io import write_sfincs_jax_output_h5

   write_sfincs_jax_output_h5(
       input_namelist=Path("input.namelist"),
       output_path=Path("sfincsOutput.h5"),
   )

Silence stdout (useful for batch runs):

.. code-block:: python

   write_sfincs_jax_output_h5(
       input_namelist=Path("input.namelist"),
       output_path=Path("sfincsOutput.h5"),
       verbose=False,
   )

For transport-matrix runs (``RHSMode=2`` or ``RHSMode=3``), you can also request the
``whichRHS`` loop and write ``transportMatrix``:

.. code-block:: python

   write_sfincs_jax_output_h5(
       input_namelist=Path("input.namelist"),
       output_path=Path("sfincsOutput.h5"),
       compute_transport_matrix=True,
   )

Running an ``Er`` scan (transport-matrix mode)
----------------------------------------------

To generate a scan directory compatible with upstream plotting scripts like ``sfincsScanPlot_2``,
you can use the ``scan-er`` subcommand:

.. code-block:: bash

   sfincs_jax scan-er \
     --input /path/to/input.namelist \
     --out-dir /path/to/scan_dir \
     --min -0.1 --max 0.1 --n 5 \
     --compute-transport-matrix

This creates subdirectories like ``Er0.1/``, each containing ``input.namelist`` and ``sfincsOutput.h5``,
plus a scan-style ``input.namelist`` in the scan directory with ``!ss`` directives so the upstream
scan plotting scripts can infer the directory list.

For large scans, you can parallelize scan points:

.. code-block:: bash

   sfincs_jax scan-er \
     --input /path/to/input.namelist \
     --out-dir /path/to/scan_dir \
     --min -0.1 --max 0.1 --n 41 \
     --jobs 8

For job arrays, slice the scan values with ``--index`` and ``--stride``:

.. code-block:: bash

   sfincs_jax scan-er \
     --input /path/to/input.namelist \
     --out-dir /path/to/scan_dir \
     --min -0.1 --max 0.1 --n 401 \
     --index ${SLURM_ARRAY_TASK_ID} \
     --stride 64

Running upstream postprocessing scripts (utils/)
------------------------------------------------

The upstream Fortran v3 codebase ships a set of plotting scripts under `utils/`.
This repository vendors those scripts in `examples/sfincs_examples/utils/`.

If you have a directory containing `sfincsOutput.h5`, you can run one of these scripts non-interactively:

.. code-block:: bash

   sfincs_jax postprocess-upstream --case-dir /path/to/case --util sfincsScanPlot_1 -- pdf

For example, after running ``scan-er`` you can generate a PDF using the upstream script:

.. code-block:: bash

   sfincs_jax postprocess-upstream --case-dir /path/to/scan_dir --util sfincsScanPlot_2 -- pdf
