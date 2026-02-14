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

   The default ``--solve-method auto`` uses BiCGStab for RHSMode=1 and RHSMode=2/3 (short recurrence,
   low memory), with GMRES fallback on stagnation. Transport solves apply a cheap collision-diagonal
   preconditioner by default, while RHSMode=1 preconditioning follows the v3 namelist defaults
   (point-block Jacobi unless line preconditioners are requested). For strict PETSc-style iteration
   histories, use ``--solve-method incremental``.

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

  - ``auto`` (default): BiCGStab (short recurrence, low memory) with GMRES fallback on stagnation.
  - ``bicgstab``: force BiCGStab for a low-memory Krylov solve (with GMRES fallback on stagnation).
  - ``dense``: assemble the dense operator from matvecs and solve directly (fast for tiny fixtures,
    but scales poorly).
  - ``incremental`` or ``batched``: matrix-free GMRES (higher memory, often robust).

- ``SFINCS_JAX_RHSMODE1_PRECONDITIONER`` (GMRES only): optional RHSMode=1 preconditioning.

  - ``point`` (or ``1``): point-block Jacobi on local (x,L) unknowns at each :math:`(\theta,\zeta)`.
  - ``theta_line``: theta-line block preconditioner (stronger, higher setup cost).
  - ``zeta_line``: zeta-line block preconditioner (stronger, higher setup cost).
  - ``adi``: apply the theta-line and zeta-line preconditioners sequentially (strongest of the built-ins,
    but also the most expensive).
  - ``0``: disable.

- ``SFINCS_JAX_TRANSPORT_PRECOND``: RHSMode=2/3 transport preconditioner.

  - ``collision`` (default): collision-diagonal preconditioner (PAS/FP + identity shift).
  - ``0``/``none``: disable.

- ``SFINCS_JAX_TRANSPORT_GMRES_RESTART``: GMRES restart length for transport fallback (default: 40).

- ``SFINCS_JAX_TRANSPORT_FORCE_DENSE``: force dense transport solves (debugging only; quadratic cost).

- ``SFINCS_JAX_TRANSPORT_DENSE_FALLBACK``: allow dense transport fallback for small ill-conditioned
  cases (disabled by default).

- ``SFINCS_JAX_TRANSPORT_DENSE_PRECOND_MAX``: enable a dense LU preconditioner for RHSMode=3
  transport solves when the system size is below the specified threshold (default: ``600``).

- ``SFINCS_JAX_GMRES_PRECONDITION_SIDE``: side for applying the preconditioner in GMRES.

  - ``left`` (default): solve :math:`P^{-1} A x = P^{-1} b`.
  - ``right``: solve :math:`A P^{-1} y = b` and set :math:`x = P^{-1} y` (PETSc-like default for GMRES).
  - ``none``: ignore any preconditioner (debugging).

- ``SFINCS_JAX_LINEAR_STAGE2``: enable a second GMRES stage with a larger iteration budget when
  the first stage stagnates (default: auto-enabled for RHSMode=1 without Phi1).

- ``SFINCS_JAX_IMPLICIT_SOLVE``: control implicit differentiation through linear solves.

  - Default: enabled (implicit gradients via ``jax.lax.custom_linear_solve``).
  - ``0``/``false``: disable (differentiate through Krylov iterations; slower / higher memory).

- ``SFINCS_JAX_TRANSPORT_RECYCLE_K``: recycle up to ``k`` previous Krylov solution vectors across
  successive ``whichRHS`` solves in transport-matrix runs. Set to ``0`` to disable.

- ``SFINCS_JAX_RHSMODE1_PROJECT_NULLSPACE``: control constraintScheme=1 nullspace projection
  for linear RHSMode=1 solves.

  - Default: enabled when ``constraintScheme=1`` and ``includePhi1=false``.
  - ``0``/``false``: disable (use raw GMRES solution).

- ``SFINCS_JAX_RHSMODE1_DENSE_FALLBACK_MAX``: enable a dense fallback solve for RHSMode=1
  when GMRES stagnates. This is only applied when the active system size is below the
  specified threshold (default: ``0``, disabled).

- ``SFINCS_JAX_GMRES_MAX_MB``: memory cap for GMRES basis storage; used to auto-limit the
  restart value when ``SFINCS_JAX_GMRES_AUTO_RESTART`` is enabled (default: ``2048``).

- ``SFINCS_JAX_GMRES_AUTO_RESTART``: enable memory-aware GMRES restarts (default: enabled).

- ``SFINCS_JAX_PRECOND_MAX_MB``: memory cap (in MB) for RHSMode=1 preconditioner assembly.
  The preconditioner block assembly is chunked to keep peak memory below this target.

- ``SFINCS_JAX_PRECOND_CHUNK``: explicit column chunk size for RHSMode=1 preconditioner assembly
  (overrides ``SFINCS_JAX_PRECOND_MAX_MB`` when set).

- ``SFINCS_JAX_FORTRAN_STDOUT``: control strict Fortran-style stdout mirroring.

  - ``1``/``true``: emit PETSc-like SNES/KSP iteration lines in addition to the standard v3 text.
  - ``0``/``false``: skip the extra iteration logs (useful for speed in tests).

  .. note::

     For strict KSP iteration-line parity, force a GMRES solve method (``incremental``/``batched``);
     BiCGStab does not produce GMRES-style history lines.

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
