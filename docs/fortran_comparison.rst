:orphan:

Validation against reference implementations
============================================

`dkx` validates outputs and solver behavior against a mature Fortran SFINCS
implementation.

.. note::

   The suite figures below are historical reports, not certification under
   the current :doc:`development_roadmap`. Their ``parity_ok`` labels do not
   establish independently checked residuals for every RHS, phase-space
   convergence, or uncontended timings. Re-admit individual cases through
   the current benchmark gates before using these ratios in a publication.
   Additional historical runtime/memory evidence lives in :doc:`performance`.

The historical vendored example-suite audit merged production-floor tokamak
reruns into these frozen reference suites:

- CPU: ``tests/scaled_example_suite_release_cpu_2026-05-08_production_tokamak``
- GPU: ``tests/scaled_example_suite_gpu_bounded_default_2026-05-08_lu3000_pas``

Those artifacts report:

- ``39/39 parity_ok`` on CPU,
- ``39/39 parity_ok`` on GPU,
- no strict mismatches,
- no ``jax_error``,
- no ``max_attempts``.

The merged reports also generate a historical runtime and memory
comparison. The plotted rows are restricted to cases whose SFINCS Fortran v3
reference runtime is at least ``10 s``; shorter rows remain CI parity/smoke
checks unless they are rerun at production-comparison resolution.

.. code-block:: bash

   python tools/publication_figures/generate_fortran_suite_benchmark_summary.py

.. figure:: _static/figures/paper/dkx_fortran_suite_benchmark_summary.png
   :alt: dkx CPU/GPU suite benchmark against SFINCS Fortran v3
   :width: 92%

   Historical benchmark generated from the profiled CPU/GPU suite reports. Panel A
   compares wall-clock runtime and Panel B compares active solver memory for the
   reference-runtime-window subset, with separate ``dkx`` cold and warm bars for
   CPU and GPU. Cases are ordered by best warm ``dkx`` speedup over the
   Fortran v3 runtime.
   The benchmark artifacts have median cold JAX/Fortran wall-clock ratios of about
   ``0.021x`` on CPU and ``0.037x`` on GPU for the plotted reference-runtime
   subset. Median process maximum-RSS ratios remain available in the JSON audit
   fields. The public memory bars use profiler active RSS deltas over the
   fixed Python/JAX/XLA baseline, and the median active-memory ratios are about
   ``2.89x`` on CPU and ``3.71x`` on GPU. The top runtime and memory cases are recorded in
   ``tools/publication_figures/artifacts/dkx_fortran_suite_benchmark_summary.json``.

Use :doc:`parity` for the scope map and comparison policy, :doc:`performance` for CPU/GPU
runtime and memory context, and :doc:`fortran_examples` for the exact-input frozen-fixture audit.

Building an auditable SFINCS reference
--------------------------------------

Use an isolated build of a pinned SFINCS commit. Preserve local build edits
separately: a Git commit alone does not identify an executable compiled from
a dirty checkout. Keep the environment lock, compiler output, build flags,
source patches and binary checksum with the external benchmark evidence,
not as large repository artifacts.

The office review uses SFINCS ``8df5453472e982df0f6ae005243ce38d57a83711``
and an isolated conda-forge environment with PETSc ``3.23.6``, MPICH,
``gfortran_linux-64``, Make, HDF5, NetCDF and NetCDF Fortran. Export the
resolved package URLs with ``micromamba list --explicit``; package names
and the PETSc version alone are not a reproducible toolchain lock.
The environment's ``include/petscconf.h`` reports
``PETSC_USE_REAL_DOUBLE``, ``PETSC_HAVE_MUMPS`` and
``PETSC_HAVE_SUPERLU_DIST``. These macros establish build availability;
successful runtime use must be checked independently.

The initial PAS smoke on this build passed with MUMPS at one and two MPI
ranks (original relative residuals ``2.64e-14`` and ``3.92e-14`` against
``1e-10``). SuperLU_DIST failed at both rank counts with segmentation
violations and residuals of approximately ``0.068`` and ``0.053`` despite
a reported linear convergence reason. This environment is therefore not
a qualified SuperLU_DIST reference under its default settings. Neither this
tiny smoke nor compilation establishes production accuracy or scaling.

The checked ``-O0 -g -fcheck=all -fbacktrace`` SFINCS build reproduced the
crash. Source inspection found an unconditional ``MatMumpsGetInfog`` call in
``solver.F90`` after ``SNESSolve``, using a factor handle initialized only for
MUMPS. An isolated reference patch replaces that call with:

.. code-block:: fortran

   factor_err = 0
   if (actualSolverType == MATSOLVERMUMPS) then
      call MatMumpsGetInfog(factorMat, 1, factor_err, ierr)
   end if

This patch eliminates the crash at both rank counts and preserves the MUMPS
result, but leaves the SuperLU_DIST residual failure unchanged. Label the
patched source and executable separately from upstream; this is a reference
build correction, not a DKX physics change.

A separate C/PETSc replay loads the same dumped matrix and manufactures
``b = A * ones``. At one/two ranks, MUMPS obtains relative residuals below
``8e-16``; default SuperLU_DIST returns approximately ``0.041`` and maximum
state error ``3.32`` despite a positive solve reason. This reproduces an
accuracy problem without SFINCS orchestration, but does not identify a
specific library defect. In this replay, disabling equilibration, selecting
natural column ordering, or enabling iterative refinement each restores a
small residual. Removing row permutation instead produces a failed solve.

For the patched SFINCS PAS case, the explicitly selected variants
``-mat_superlu_dist_colperm NATURAL`` and
``-mat_superlu_dist_equil false`` pass execution and original-residual checks
at both rank counts (approximately ``1.3e-14`` and ``9.6e-15`` respectively).
These are smoke-qualified configurations only. Natural ordering may increase
fill on larger matrices; retain ordering, scaling and refinement settings
with every measurement and recheck each original residual. A favorable
setting for this case is not grounds for a global default change.

With that environment activated, the review's SFINCS makefile configuration is:

.. code-block:: make

   include ${PETSC_DIR}/lib/petsc/conf/variables
   FC = ${CONDA_PREFIX}/bin/mpifort
   FLINKER = ${CONDA_PREFIX}/bin/mpifort
   LIBSTELL_DIR = mini_libstell
   LIBSTELL_FOR_SFINCS = mini_libstell/mini_libstell.a
   EXTRA_COMPILE_FLAGS = -I${CONDA_PREFIX}/include -O3 -ffree-line-length-none -fallow-argument-mismatch
   EXTRA_LINK_FLAGS = -L${CONDA_PREFIX}/lib -Wl,-rpath,${CONDA_PREFIX}/lib -lnetcdff -lnetcdf -lhdf5_fortran -lhdf5 -lhdf5_hl -lhdf5hl_fortran
   SFINCS_IS_A_BATCH_SYSTEM_USED = no

Save this as ``fortran/version3/makefiles/makefile.dkx_review`` in the
isolated source tree, set ``PETSC_DIR`` to the environment prefix and
``SFINCS_SYSTEM=dkx_review``, then run ``make -C fortran/version3 -j4``.
Retain compiler warnings, including legacy argument-mismatch diagnostics;
successful compilation does not establish bounds safety or physical accuracy.

Before profiling, run bounded one- and two-rank smoke cases for each backend.
Record the selected factor backend and PETSc convergence reasons, then
independently check every RHS against the original operator and deck tolerance.
Compare direct factorization of the same matrix separately from a complete
preconditioned solve: factoring the preconditioner and factoring the kinetic
Jacobian are different workloads. Separate assembly, ordering/symbolic analysis,
numeric factorization, triangular application and Krylov costs. Preserve failed
attempts and never use their short runtimes as successful reference timings.

The campaign runner accepts repeatable PETSc argument tokens, for example:

.. code-block:: bash

   python tools/benchmarks/parity_performance_matrix.py \
     --examples /path/to/cases --out /path/to/campaign.jsonl \
     --fortran-binary /path/to/qualified/sfincs \
     --fortran-backend superlu_dist \
     --fortran-petsc-opt=-mat_superlu_dist_colperm \
     --fortran-petsc-opt=NATURAL --petsc-profile

Each record retains ``fortran_petsc_opts`` and each reference run reports
``observed_factor_backends`` from ``-ksp_view``. An empty observed list means
no factor-package name was captured; requested options alone do not prove
which implementation ran. SFINCS or PETSc can override an option.
``--fortran-backend PACKAGE`` selects the factor package in preflight and each
rank launch, taking precedence over conflicting option tokens. It also requires
the observed package set to equal the requested package; missing, mixed or
wrong observations reject that reference. The selected package and
``backend_acceptance`` are recorded independently of algebraic acceptance.
Backend-specific tuning tokens alone do not select a backend.

The matrix runner defaults to ``--fortran-threads 1`` per MPI rank. It sends
explicit OpenMP/OpenBLAS/MKL/BLIS/Accelerate thread requests through preflight
and every reference launch and disables OpenMP/MKL dynamic thread adjustment.
Use, for example, ``--ranks 1 2 --fortran-threads 2`` to request two threads per
rank. This setting is independent of DKX's CPU configuration and is recorded
in campaign identity and launch environment metadata. Changing it invalidates
resume. Quoted paths and arguments in ``--fortran-launcher`` are preserved.

``observed_mumps_threads`` reports values parsed from MUMPS diagnostics.
``mumps_thread_acceptance`` fails if any reported count exceeds the request or
is invalid; that reference cannot be accepted as a benchmark pair. Missing
reports remain ``not_checked``. These diagnostics do not establish CPU affinity,
active worker count or the behavior of every BLAS implementation. Measure and
record placement and utilization before comparing scaling or speed.

Changed options or recorded runtime environment settings invalidate resume.
The runner writes ``OUT.provenance.json`` with typed options, Python/package
versions, platform, selected environment variables and individual input/source/
executable hashes. Its checksum is included in ``OUT.done``. Refused resume
preserves the existing provenance and measurements.

Verify a retained campaign without running either solver:

.. code-block:: bash

   python tools/benchmarks/parity_performance_matrix.py \
     --verify --out /archive/campaign.jsonl --artifacts-dir /archive/attempts

``--artifacts-dir`` relocates the retained tree, preserving its
``CAMPAIGN_ID/ATTEMPT_DIRECTORY`` layout. Omit it to use recorded paths.
Verification follows ``OUT.done`` to the checkpoint/provenance checksums,
then binds every attempt record to its manifest and checks file sizes,
checksums and the complete file inventory. Missing retention, incomplete
campaigns, duplicate attempt directories and symlinks fail. Rejected scientific
runs can still have intact evidence and are included in the check.

Add ``--dependency-archive /archive/runtime`` to check exact source/library/input
bytes declared in provenance. The archive contains ``blobs/SHA256`` files and
``bound-files.json``: schema 1, a ``campaigns`` mapping whose entries contain
``campaign_id``, ``provenance_sha256`` and a ``files`` mapping from original path
to ``{blob, sha256, bytes}``. One archive can hold several campaigns and deduplicate
identical files. Each campaign must appear exactly once; missing or changed
declared files fail. Original host paths need not exist. This option verifies
an archive; it does not create one or discover undeclared dependencies.

The report establishes integrity relative to the supplied completion record.
It does not authenticate that record, rerun residual/observable comparisons,
establish a complete runtime environment, or certify scientific completeness.
Publish a trusted archive checksum alongside any publication; keep physics,
numerical-convergence and environment-reproducibility gates separate.

``ldd`` alone is insufficient for runtime archival: MPI/UCX can load plugins
and GPU-driver libraries dynamically. A bounded same-host SFINCS replay restored
the pinned executable, loader and libraries from archived bytes, reconstructed
SONAME aliases, suppressed embedded library search paths and set the UCX module
directory explicitly. All 100 traced library initializations used restored paths;
the small full-FP reference residual was 9.74e-11 and the checked moments matched
the qualified reference exactly. A fresh GPU full-FP replay captured resident
libraries, imported modules, installed metadata and compiler executables and
passed its 1e-10 residual gate. These are replay diagnostics, not clean-machine,
MPI scaling, complete GPU environment or performance certificates. Set and
record OpenMP/BLAS thread counts explicitly; a single MPI rank does not imply
one CPU thread.

Use repeatable ``--provenance-file PATH`` for environment locks, build records,
external PETSc option files or resolved shared libraries. Their contents become
part of campaign identity; missing explicit files fail before preflight.
Archive originals separately: a hash list does not supply the files, discover
all dynamically loaded libraries or prove the selected environment is complete.
Every configuration still requires the original-residual and observable gates.

``first_run_s`` records DKX's first invocation; ``cold_s`` remains its legacy
alias. ``compilation_cache_dir`` records the effective JAX cache directory.
These do not certify a fresh compilation: use an explicitly empty cache and
archive its initial state when measuring compilation, then distinguish later
warm calls. The default persistent cache may already contain compiled code.

Output comparison and transport reference limits
------------------------------------------------

The comparator requires matching explicit ``RHSMode`` values. Mode 2 compares
all entries of a 3-by-3 transport matrix; mode 3 requires a 2-by-2 matrix.
Profile mode compares final species columns and final current, allowing
different Newton-history lengths. Missing required outputs, incompatible
shapes and non-finite data are errors. Absolute and unrounded relative
differences are retained; irrelevant profile/transport datasets are excluded.
These schema checks do not replace original-residual or grid convergence gates.

The bounded office monoenergetic PAS fixture has ``Ntheta=Nzeta=9`` and
``Nxi=6,12``. Earlier notes misattributed the file named ``superlu.jsonl``:
its observed package was MUMPS, so its residuals were duplicate MUMPS evidence.
A new explicit selection verifies SuperLU_DIST separately. The original
relative-residual gate remains ``1e-12`` for every RHS at one/two MPI ranks.

.. list-table:: Bounded reference configurations
   :header-rows: 1

   * - Configuration
     - Largest original relative residual over both grids and rank counts
     - Algebraic gate
   * - MUMPS, default SFINCS GMRES
     - 1.59e-6
     - Failed
   * - Verified SuperLU_DIST, natural column ordering, GMRES
     - 7.41e-10
     - Failed
   * - Verified MUMPS, Richardson, unpreconditioned norm, max 20 iterations
     - 8.78e-13
     - Passed

For the accepted Richardson configuration, append these tokens to the runner:

.. code-block:: bash

   --fortran-backend mumps \
   --fortran-petsc-opt=-ksp_type --fortran-petsc-opt=richardson \
   --fortran-petsc-opt=-ksp_norm_type --fortran-petsc-opt=unpreconditioned \
   --fortran-petsc-opt=-ksp_max_it --fortran-petsc-opt=20

Both DKX solves also pass the original-residual gate. Complete transport
matrices agree with one-rank SFINCS within ``6.71e-12`` absolute and
``1.06e-14`` relative when scaled by the largest matrix entry. The individual
coefficients and differences are retained in HDF5; the aggregate metric does
not replace per-coefficient tolerances. These are discrete-model comparisons:
the coefficients change substantially between the two pitch grids, so neither
grid is physically certified. CPU allocations differ, precluding speed claims.
The dumped preconditioner equals the full kinetic matrix on these fixtures;
Richardson is not a general recommendation for approximate preconditioners.

A right-preconditioned MUMPS repeat with an unpreconditioned norm still fails:
the second RHS has estimated norm ``1.54e-16`` but explicit norm ``7.70e-8``
(relative ``1.28e-7``), confirmed by the independent dumped-matrix check.
PETSc's `true-residual monitor
<https://petsc.org/release/manualpages/KSP/KSPMonitorTrueResidual/>`_
distinguishes the explicitly evaluated norm from an estimated norm. Merely
changing norm semantics or orthogonalization did not resolve this gap.

Inputs, commands, JSON/logs, replay programs and retained matrices/states are
archived with checksums outside Git in
``dkx-review-evidence-20260905/transport-reference-pilot``. The original
campaign deleted temporary raw solve files; the explicitly selected backend
campaigns preserve them. A complete environment lock remains R0 work.

Retaining raw evidence
----------------------

Pass ``--artifacts-dir /path/outside/git`` to the campaign runner to retain
**every attempt**, successful or failed, under a campaign hash and a unique
case directory. Without this option, work directories remain temporary.
The retained directory contains copied inputs, generated matrix/state/output
files, complete stdout/stderr and PETSc logs, and each subprocess's argument
vector, timeout and explicit environment overrides. ``manifest.json`` records
per-file byte counts and SHA-256 checksums plus the case result; the JSONL
record links its directory and manifest checksum after normal completion.

Handled cancellation finalizes the partial manifest after subprocess cleanup,
and the campaign checkpoint points to the interrupted attempt. SIGKILL,
machine failure or disk exhaustion can leave an incomplete archive; a directory
alone is not proof of completion. Retried failures receive new directories.
Existing nonempty evidence directories are refused, and an archive inside the
copied example is rejected to prevent recursive copying.

The retained files are not pruned automatically. Hash verification, external
equilibrium files, library/compiler locks and implicit PETSc options still
belong in a publication archive; command overrides do not capture the whole
environment. Preserve failed attempts when publishing a performance envelope.
A one-case installed-wheel/office test retained and independently verified all
26 files, including the rejected reference's matrices and both RHS states.
It still reports no accepted pair.

The retained 487-by-487 monoenergetic Jacobian has a dense SVD condition
estimate of ``5.58e4``. An independent dense LU solve with extended-precision
residual refinement gives much smaller residuals than the failed PETSc run.
Modified Gram--Schmidt and classical Gram--Schmidt with reorthogonalization
also leave the reference rejected. These are bounded diagnostics, not a
causal explanation, production solver policy or a grid-convergence result.

A separate PETSc C replay loads the same matrix and each saved physical RHS,
without running SFINCS assembly. With MUMPS, GMRES and ``CNTL(1)=1e-6`` it
reproduces the two relative residuals ``1.89e-9`` and ``5.61e-8``. Direct
MUMPS with its default pivot threshold fails because of insufficient factor
workspace (``INFOG(1)=-9``, KSP reason ``-11``). Raising ``ICNTL(14)`` to 200
completes factorization but still misses the physical residual tolerance.
Forcing five refinement steps with ``ICNTL(10)=-5`` reduces both replay
residuals below ``1e-12``; positive 5 allows early stopping and does not suffice
here. Richardson with an explicitly evaluated residual also passes, including
the complete SFINCS runs above. This establishes bounded remedies; the source
of the factor/recurrence residual gap and their larger-case cost remain open.

The `MUMPS guide, section 5.8
<https://mumps-solver.org/doc/userguide_5.9.1.pdf#page=42>`_ distinguishes
fixed-count refinement from backward-error stopping and lists configurations
that disable internal refinement, including distributed right-hand sides or
solutions. Record the effective settings and verify the final residual at
every rank count; an option token alone does not demonstrate refinement.

Full-FP qualification with approximate preconditioners
----------------------------------------------------------

Two-species analytic scheme-4 fixtures at 2,804 and 15,844 total unknowns use
full linearized Fokker--Planck collisions, zero field and no Phi1. The larger
fixture has ``Ntheta=9, Nzeta=11, Nxi=16, Nx=5``; the smaller uses
``5,7,8,5``. Both derive from
``tests/ref/quick_2species_FPCollisions_noEr.input.namelist`` with
``solverTolerance=1e-10``. Their dumped preconditioners differ from A
(relative Frobenius differences 0.339 and 0.0953), unlike the monoenergetic
fixtures above.

Default SFINCS GMRES/MUMPS completes but fails original-residual acceptance
(5.72e-9--6.59e-9). Adding ``-ksp_pc_side right`` and
``-ksp_norm_type unpreconditioned`` passes at one/four MPI ranks on both grids
(maximum 9.75e-11). Both installed-wheel CPU DKX solves pass too; the largest
current difference is 2.44e-13 absolute, with flow/particle/heat moments also
compared. The accepted medium reference takes substantially longer than the
rejected default, so the earlier short timing is not a valid baseline.

The installed-wheel medium solve on one A4000 reports GPU execution and a
7.72e-11 original residual across first and two repeated invocations. Its
current differs from accepted one-rank SFINCS by 3.55e-14 absolute; particle
and heat fluxes differ by less than 4.28e-20. These are fixed-discretization
checks, not joint-grid convergence, multi-GPU scaling or idle-machine timings.
Host activity prevents a controlled speed comparison. First invocation used
the existing persistent cache, so it is not a fresh compilation measurement.

``dkx-review-evidence-20260905/full-fp-reference`` retains inputs, commands,
failed/accepted attempts and GPU comparisons. The CPU campaign binds an
explicit 107-package reference environment, build log, source patch and 91
resolved linked libraries. Readable provenance hashes 182 files for CPU and
86 for GPU; all 97 files in the accepted CPU/GPU attempt manifests and both
provenance sidecar checksums were independently verified locally. GPU driver/
runtime locking and comprehensive external-input archival remain separate
publication requirements.
