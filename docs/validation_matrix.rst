Validation Matrix
=================

This page tracks the publication-facing validation entries for ``dkx``. A
validation entry is one self-contained line of validation work: its scripts,
artifacts, tests, and pass/fail criteria. Each entry connects a physics claim or
benchmark figure to:

- a literature anchor,
- the script or workflow that generates it,
- the expected output artifact,
- and the claim status recorded in the release manifest.

Machine-readable manifest
-------------------------

The corresponding machine-readable manifest lives in:

- ``tools/publication_figures/validation_manifest.json``

It holds one record per validation entry, and is the stable spine for:

- future manuscript figure generation,
- reproducible benchmark reruns,
- and test/benchmark dashboards that separate implemented release entries from
  deferred post-release research topics.

Each manifest entry carries explicit research criteria:

- ``source_code``: the implementation files that define the entry,
- ``tests``: the tests that protect the entry or its scaffold,
- ``acceptance_gates``: the concrete criteria required before the entry can support a
  manuscript or release claim.
- ``release_gate``: the release-facing claim status, evidence level, nonblocking
  release decision, and ``promotion_gate`` for the entry.

The schema is enforced by ``tests/test_validation_manifest_schema.py``. Implemented
release entries must point to existing scripts, artifacts, source files, and tests.
Deferred post-release entries are closed for the tagged release but retain literature
anchors, implementation targets, tests, and acceptance criteria so follow-up research
work is not lost. ``python -m tools.release.release check-gates`` applies the same path
hygiene to deferred entries: listed source files, tests, scripts, and artifacts must
exist even when the claim status is ``closed_deferred``.

Release claim gate metadata
---------------------------

Every manifest record has a ``release_gate`` block checked by
``python -m tools.release.release check-gates`` and ``tests/test_release_gate_metadata.py``.
The allowed ``claim_status`` values are:

- ``release_ready``: checked-in artifacts support the documented release-scope
  claim, and the listed tests are the fast check for that claim.
- ``regression_scaffold``: checked-in bounded artifacts are useful for CI,
  branch validation, or manuscript layout, but a broader/full-resolution claim is
  intentionally not being made.
- ``bounded_proxy``: checked-in artifacts support a narrower proxy or
  normalization claim, while the corresponding full literature reproduction stays
  closed until its ``promotion_gate`` is met.
- ``closed_deferred``: the entry is explicitly closed for the tagged release as
  post-release or nightly research work.

No manifest entry may set ``blocks_current_release=true`` unless the release
process intentionally stops on that entry. An entry that is not ready must
therefore be either absent from the release manifest or recorded as
``closed_deferred`` with a concrete reason and ``promotion_gate``. This prevents
scaffold scripts, run plans, or proxy figures from being mistaken for closed
publication evidence.

Release decision
----------------

The release is shippable only for the documented release-ready and bounded-proxy
claims. Production-resolution QI CPU/GPU seed ladders, true differentiable
device-QI closure, and single-case multi-device strong scaling are not release
blockers because they are explicitly scoped as bounded or deferred research
topics. They should be promoted only after checked artifacts satisfy the listed
residual, output, trace, parity, and scaling checks.

Implemented literature reproductions
------------------------------------

These entries already have scripts and figure artifacts in the repository.

Matched full-kinetic SFINCS profile
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The first independent full-Fokker--Planck profile comparison uses the exact checked
``validation/inputs/tokamak_full_fp_{high,ultra}.input.namelist`` decks in both
DKX and pinned SFINCS v3. The case is an analytic axisymmetric tokamak surface
with physical density and temperature gradients, full trajectories, zero
electric field, and the recommended automatic full-FP constraint.

The machine-readable evidence is
``validation/full_kinetic_sfincs_v1.json``. Re-audit its decks, compact outputs,
convergence arithmetic, residual checks, and claim boundary with:

.. code-block:: console

   python tools/paper_benchmarks/audit_full_kinetic_sfincs_validation.py

.. list-table:: Matched full-kinetic tokamak profile, measured
   :header-rows: 1
   :widths: 56 28

   * - measurement
     - value
   * - accepted resolution
     - 6,887 -> 12,509 unknowns
   * - bootstrap/parallel-flow movement
     - ``0.042%``
   * - heat-flux movement
     - ``0.280%``
   * - largest scaled DKX/SFINCS difference at the finest refinement level,
       across nonzero scalar and speed-resolved observables
     - ``2.69e-10``
   * - completed true residuals
     - below ``1.82e-11``

Particle flux and NTV vanish by axisymmetric cancellation at the
checked accuracy, so their acceptance criterion is the recorded ``1e-12``
absolute scale rather than a meaningless relative error.

The reference build is the pinned SFINCS commit with PETSc 3.23.6 and MUMPS
5.8.1; both MUMPS and SuperLU_DIST are detected at runtime and MUMPS is selected.
No scientific SFINCS source edits or link stubs are used. Timing and memory are
retained for reproduction, not for a cross-code performance claim. Multispecies
and stellarator full-FP physics, finite electric field, Phi1, ambipolar roots,
and experimental agreement remain separate checks.

Matched finite-Er full-kinetic profile
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The companion ``validation/full_kinetic_sfincs_finite_er_v1.json`` artifact
uses the pinned upstream one-species full-FP tokamak case at normalized
``Er = -30``. Both codes use full trajectories, automatic constraint 1, a
``1e-13`` solver tolerance, and matched angular, pitch, and speed grids. Audit
the comparison by passing the artifact to the shared command above.

.. list-table:: Matched finite-``Er`` tokamak profile, measured
   :header-rows: 1
   :widths: 56 28

   * - measurement
     - value
   * - SFINCS matrices
     - 6,887 -> 12,509 unknowns
   * - DKX's distinct internal representation
     - 10,532 -> 18,614 unknowns
   * - finest flow/current, momentum flux, heat flux and speed-resolved
       outputs, scaled agreement
     - ``1.88e-9``
   * - high-to-ultra movement
     - at most ``0.326%``
   * - completed true residuals
     - below ``5.25e-11``

Axisymmetric intrinsic ambipolarity makes summed particle flux and NTV
cancellation-level quantities, so their acceptance criterion is ``2e-11``
absolute. This is a prescribed-field surface comparison, not an electric-field
scan or ambipolar-root validation. The separate zero-field stellarator full-FP
entry is recorded below.

Matched stellarator full-kinetic profile
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The companion ``validation/full_kinetic_sfincs_stellarator_v1.json`` artifact
uses exact relative-path decks on the checksummed W7-X SC1 Boozer surface at
``rN = 0.5``. Both live codes use physical density and temperature gradients,
full linearized Fokker--Planck collisions, full trajectories, zero electric
field, automatic constraint 1, and a ``1e-12`` solver tolerance.

.. list-table:: Matched stellarator full-kinetic profile, measured
   :header-rows: 1
   :widths: 56 28

   * - measurement
     - value
   * - SFINCS algebraic systems
     - 54,407 -> 98,126 unknowns
   * - DKX's distinct internal representation
     - 87,887 -> 155,994 unknowns
   * - largest high-to-ultra movement
     - ``0.444%``
   * - flow, particle and heat flux, NTV and retained speed spectra at the
       finest refinement level, scaled agreement
     - ``1.37e-8``
   * - completed true residuals
     - below ``1.82e-12``

That largest scaled difference is NTV, and corresponds to an absolute
difference of about ``8.31e-13``. Momentum flux is retained as a near-zero
absolute criterion rather than assigned an unstable relative error.

The artifact pins the SFINCS commit, MUMPS-enabled build, Boozer source path
and checksum, exact decks, raw and compact outputs, logs, solver traces,
cold/warm DKX timing, SFINCS timing, and process memory. These measurements are
reproduction provenance, not a cross-code performance claim. This one-species
surface comparison is not an Er scan, ambipolar-profile, Phi1, experimental,
or second-stellarator-family full-FP validation.

Whole-profile ambipolar evidence from a case file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``validation/native_ambipolar_profile_v1.json`` records the admitted case-file
workflow as a separate validation entry. The portable checked TOML drives five
W7-X standard-configuration surfaces with physical hydrogen/electron profiles,
PAS collisions, DKES drifts, bounded midpoint refinement, all-root search, and
radial branch continuation. Recompute every compact check with:

.. code-block:: console

   python tools/paper_benchmarks/audit_native_ambipolar_profile.py

The result retains root counts ``[1, 1, 3, 1, 1]``, seven discrete branch
events, selected ion/electron fields and SI particle/heat fluxes, and all 222
solver attempts. One structured evaluation at the outer-surface zero field
misses its unchanged target; the retained bounded GMRES retry reduces the true
residual from ``8.23e-13`` to ``1.93e-13``. Every final bracket is
``0.0048828125 kV/m`` wide and every refinement hierarchy is resolved.

The artifact pins the portable input, geometry, cold/warm DKX NetCDF,
compact profile, commit, environment, timing, and process-memory evidence.
Passing ``--results-root`` additionally verifies the external files and exact
cold/warm scientific-array identity. The warm process was 0.15% slower, so no
cache-speedup claim is made. This is a PAS/DKES workflow record, not
phase-space-convergence, continuously localized bifurcations, experiment,
full-Fokker--Planck or independent ambipolar validation, Phi1, or a second
stellarator-family claim.

Bounded ambipolar phase-space ladder
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``validation/ambipolar_phase_space_ladder_v1.json`` records a separate
coarse/reference/fine kinetic-grid ladder for the same five-surface W7-X
PAS/DKES profile. Recompute its checksums, every root movement, every selected
SI particle/heat-flux movement, topology decision, and admission status with:

.. code-block:: console

   python tools/paper_benchmarks/audit_ambipolar_phase_space_ladder.py

The three resolutions are ``(13, 31, 32, 5)``, ``(15, 37, 36, 6)``, and
``(17, 37, 40, 6)`` in theta, zeta, pitch, and speed. Root counts remain
``[1, 1, 3, 1, 1]`` with identical classifications, branch identities, and
selected branches. Reference-to-fine movement nevertheless fails the unchanged
root and observable criteria:

.. list-table:: Reference-to-fine movement against its thresholds
   :header-rows: 1
   :widths: 34 28 20 14

   * - quantity
     - movement
     - threshold
     - result
   * - ambipolar root
     - ``1.6259765625 kV/m``
     - ``0.005 kV/m``
     - fail
   * - selected particle flux
     - ``4.08%``
     - ``2%``
     - fail
   * - selected heat flux
     - ``7.81%``
     - ``2%``
     - fail
   * - maximum accepted true residual
     - ``3.92e-13``
     - ``1e-12``
     - pass

The auditable outcome is ``refinement_exhausted``. This negative result
prevents promotion of the workflow record to phase-space-converged
validation. The fine refinement level only refines theta and pitch beyond the
reference; zeta/speed convergence, independent full-Fokker--Planck ambipolar
comparison, experiment, and cross-code performance remain separate checks.

Theta/pitch resolution diagnosis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The follow-on ``validation/ambipolar_phase_space_axes_v1.json`` artifact
separates theta and pitch instead of combining them. It compares the reference
``(15, 37, 36, 6)`` resolution against theta-only ``(17, 37, 36, 6)`` and
pitch-only ``(15, 37, 40, 6)``, then extends the fixed-theta pitch sequence to
``(15, 37, 44, 6)``. Audit it with:

.. code-block:: console

   python tools/paper_benchmarks/audit_ambipolar_phase_space_axes.py

.. list-table:: Movement against the ``(15, 37, 36, 6)`` reference
   :header-rows: 1
   :widths: 32 26 20 20

   * - comparison
     - root movement
     - selected particle flux
     - selected heat flux
   * - theta-only ``(17, 37, 36, 6)``
     - ``0.1611328125 kV/m``
     - below ``2%``
     - below ``2%``
   * - pitch40 ``(15, 37, 40, 6)``
     - ``1.7333984375 kV/m``
     - —
     - ``9.47%``
   * - pitch40 to pitch44 ``(15, 37, 44, 6)``
     - ``0.205078125 kV/m``
     - ``13.52%``
     - ``14.07%``

Pitch40 is the dominant failed direction. Pitch40-to-pitch44 does not approach
the thresholds.

All topology and accepted true-residual checks pass, but the pitch44 process
reached a ``22,275,409,800 B`` footprint on the 24 GiB host. The artifact
therefore rejects a blind pitch48 escalation and retains
``refinement_exhausted`` status. It is a bounded diagnosis, not phase-space,
zeta, speed, independent-code, experiment, or performance validation.

Bounded uniform-pitch route and ladder
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``validation/ambipolar_pitch_budget_v1.json`` separates memory-route admission
from phase-space admission. Audit it with:

.. code-block:: console

   python tools/paper_benchmarks/audit_ambipolar_pitch_budget.py

For the exact uniform-pitch-22 profile, forwarding the DKX memory budget to
each batched solve changes all 139 evaluations from full factors to the
memory-bounded structured direct route. Roots and brackets are exact.

.. list-table:: Bounded route against the full-factor route
   :header-rows: 1
   :widths: 52 34

   * - measurement
     - value
   * - selected fluxes, relative agreement
     - ``3.58e-11``
   * - retained evaluation fluxes, relative agreement
     - ``1.53e-10``
   * - bounded residual
     - ``5.28e-14``
   * - retained bounded process footprint
     - below ``2,923,810,392 B``
   * - earlier full-factor process footprint
     - ``31,859,925,880 B``

The populated-cache run is slower than the cold run, so no cache-speedup claim
is made.

This admitted route makes a bounded uniform pitch-22/26/30 diagnostic possible,
but it does not make the physics converged. Root counts change
``[3, 1, 1] -> [1, 1, 1] -> [1, 3, 1]``.

.. list-table:: Adjacent-level movement, uniform pitch 22/26/30
   :header-rows: 1
   :widths: 38 26 26

   * - quantity
     - first adjacent pair
     - second adjacent pair
   * - selected field
     - ``9.599609375 kV/m``
     - ``7.7001953125 kV/m``
   * - selected heat flux
     - ``55.72%``
     - ``45.25%``

All accepted true residuals remain below ``5.65e-14``, so the artifact
truthfully records ``refinement_exhausted``,
rejects a uniform pitch-34-or-higher escalation, and leaves speed-node-local,
zeta, speed, independent-code, experiment, full-FP, and Phi1 checks open.

Fixed-work pitch-by-speed diagnosis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``validation/ambipolar_pitch_speed_groups_v1.json`` compares the supported
uniform-22, linear-ramp-36, and quadratic-ramp-44 rules on two common W7-X
surfaces.

.. list-table:: Active pitch counts by speed node
   :header-rows: 1
   :widths: 28 42 18

   * - rule
     - counts at the six speed nodes
     - total modes
   * - uniform-22
     - ``[22,22,22,22,22,22]``
     - 132
   * - linear-ramp-36
     - ``[4,9,17,27,36,36]``
     - 129
   * - quadratic-ramp-44
     - ``[4,5,11,25,44,44]``
     - 133

Audit the checksums, extraction, comparisons, and acceptance criteria with:

.. code-block:: console

   python tools/paper_benchmarks/audit_ambipolar_pitch_speed_groups.py

Root counts change ``[3,1] -> [1,3] -> [1,1]``.

.. list-table:: Movement between adjacent rules
   :header-rows: 1
   :widths: 32 30 26

   * - redistribution
     - selected electric field
     - selected heat flux
   * - uniform to linear
     - up to ``12.20703125 kV/m``
     - up to ``68.35%``
   * - linear to quadratic
     - ``2.177734375 kV/m``
     - ``17.93%``

Uniform-to-linear keeps the intermediate-speed group fixed at 44 modes. The
linear-to-quadratic redistribution also changes topology. All accepted
residuals stay below ``7.04e-14`` and all measured footprints below 4.14 GB.
The quadratic cold/warm scientific arrays are exact except for their timing
field; the warm run is slower and supports no cache-speedup claim.

The diagnostic is complete but phase-space convergence is false. The next
bounded pair must hold high-speed work fixed while separating low from
intermediate sensitivity. Zeta, speed, independent-code, experiment, full-FP,
Phi1, and performance checks remain open.

Explicit fixed-high-work pitch diagnosis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``validation/ambipolar_pitch_explicit_groups_v1.json`` compares three exact
six-speed-node allocations with the same 129 active modes and the same 72
high-speed modes.

.. list-table:: Explicit six-speed-node allocations
   :header-rows: 1
   :widths: 32 42

   * - allocation
     - active pitch counts
   * - supported linear-36
     - ``[4,9,17,27,36,36]``
   * - low-heavy
     - ``[12,12,16,17,36,36]``
   * - intermediate-heavy
     - ``[4,4,24,25,36,36]``

Audit the compact record and, when staged externally, all raw results and the
pinned geometry with:

.. code-block:: console

   python tools/paper_benchmarks/audit_ambipolar_pitch_explicit_groups.py

All three allocations preserve root counts ``[1,3]`` on the bounded surface
pair. Pairwise movements nevertheless reach the following:

.. list-table:: Largest pairwise movement across the three allocations
   :header-rows: 1
   :widths: 42 30

   * - quantity
     - movement
   * - selected electric field
     - ``1.064453125 kV/m``
   * - selected particle flux
     - ``9.89%``
   * - selected heat flux
     - ``9.08%``

Every new evaluation uses the bounded structured direct route, accepted residuals stay
below ``3.05e-14``, and measured footprints stay below 4.01 GB. The retained
intermediate-heavy replay is scientifically exact except for timing, but no
warm-speedup claim is supported.

The outcome is ``refinement_exhausted``, not phase-space convergence. The next
bounded diagnostic must raise low and intermediate work together while holding
the high-speed group fixed. Zeta, speed, independent-code, experiment,
full-FP, Phi1, and performance checks remain open.

Publication validation dashboard
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Literature anchor:

- `Landreman et al. 2014 <https://doi.org/10.1063/1.4870077>`__
- `Open PDF mirror <https://publications.lib.chalmers.se/records/fulltext/199559/local_199559.pdf>`_

Script:

- ``tools/publication_figures/generate_validation_dashboard.py``

Artifacts:

- ``tools/publication_figures/artifacts/dkx_publication_validation_dashboard_summary.json``
- ``docs/_static/figures/paper/dkx_publication_validation_dashboard.png``
- ``docs/_static/figures/paper/dkx_publication_validation_dashboard.pdf``

.. figure:: _static/figures/paper/dkx_publication_validation_dashboard.png
   :alt: Literature-anchored dkx validation dashboard
   :width: 92%

   Dashboard assembled from checked-in validation artifacts rather than hand-edited
   plot data. The acceptance tests assert that the collisionality scans contain both
   FP and PAS rows on the seven-point grid, that the high-collisionality ``L11``
   separation remains larger than the low-collisionality separation, and that the
   trajectory sweeps retain exact zero-field agreement while resolving finite-field
   model separation.

Fortran v3 CPU/GPU suite benchmark
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Literature and reference anchors:

- `Landreman et al. 2014 <https://doi.org/10.1063/1.4870077>`__
- `Open PDF mirror <https://publications.lib.chalmers.se/records/fulltext/199559/local_199559.pdf>`_
- `SFINCS Fortran repository <https://github.com/landreman/sfincs>`_

Script:

- ``tools/publication_figures/generate_fortran_suite_benchmark_summary.py``

Artifacts:

- ``tools/publication_figures/artifacts/dkx_fortran_suite_benchmark_summary.json``
- ``docs/_static/figures/paper/dkx_fortran_suite_benchmark_summary.png``
- ``docs/_static/figures/paper/dkx_fortran_suite_benchmark_summary.pdf``

.. figure:: _static/figures/paper/dkx_fortran_suite_benchmark_summary.png
   :alt: Frozen CPU and GPU suite benchmark against SFINCS Fortran v3
   :width: 92%

   Cross-code release benchmark generated from frozen CPU/GPU suite reports. The
   plotted bars show wall-clock runtime and active solver memory for SFINCS
   Fortran v3, ``dkx`` CPU cold/warm, and ``dkx`` GPU cold/warm
   across the reference-runtime-window rows whose Fortran v3 reference runtime
   is at least ``10 s``. Cases are ordered by best warm ``dkx`` speedup over the
   Fortran v3 runtime.

   JAX active memory subtracts the fixed Python/JAX/XLA runtime baseline using
   profiler RSS deltas while preserving full process RSS in the JSON audit
   fields. The acceptance tests require all 39 audited cases to remain
   ``parity_ok`` on both backends, with zero strict mismatches and no
   ``jax_error`` or ``max_attempts`` failures.

   The summary JSON records which frozen rows are excluded from public
   performance claims until production-resolution reruns exist. Absolute
   runtime, memory, ratios, top offenders, warm timing-source counts, and the
   excluded short-reference rows are recomputed from the checked-in reports and
   stored in the JSON summary for manuscript tables and regression triage. The
   excluded short-reference rows remain CI parity/smoke checks until rerun at
   production-comparison resolution.

SFINCS 2014 collisionality figures
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Literature anchor:

- `Landreman et al. 2014 <https://publications.lib.chalmers.se/records/fulltext/199559/local_199559.pdf>`__

Scripts:

- ``tools/publication_figures/generate_sfincs_paper_figs.py --case lhd``
- ``tools/publication_figures/generate_sfincs_paper_figs.py --case w7x``

Artifacts:

- ``docs/_static/figures/paper/dkx_fig1_lhd_collisionality.png``
- ``docs/_static/figures/paper/dkx_fig2_w7x_collisionality.png``
- ``docs/_static/figures/paper/dkx_fig3_simakov_helander.png``

The standard LHD and W7-X collisionality figures are generated from the
corrected scan-input writer and recorded as audited full-resolution validation
artifacts. They are regression and manuscript-scaffold figures, not a claim that
every plotted point reproduces the original paper image digit-for-digit.

Status note:

- the scan writer in ``generate_sfincs_paper_figs.py`` rejects duplicate
  namelist assignments that would otherwise override the intended
  ``collisionOperator`` and fast-resolution settings
- the generator emits machine-readable collisionality summaries with top-level
  metadata and sorted rows so full-resolution reruns have pinned provenance
  instead of relying only on figure files
- the checked-in full LHD and W7-X summaries each contain 14 rows: both FP and PAS
  labels on a seven-point collisionality ladder
- corrected bounded fast reruns are retained as branch-level regression scaffolds, but
  the main LHD/W7-X figure family points at the full audited artifacts

Audited full artifacts:

- full LHD summary:
  ``tools/publication_figures/artifacts/lhd_collisionality_summary.json``
- full LHD figure:
  ``docs/_static/figures/paper/dkx_fig1_lhd_collisionality.png``
- full W7-X summary:
  ``tools/publication_figures/artifacts/w7x_collisionality_summary.json``
- full W7-X figure:
  ``docs/_static/figures/paper/dkx_fig2_w7x_collisionality.png``

Corrected bounded branch artifacts:

- bounded corrected LHD summary:
  ``tools/publication_figures/artifacts/lhd_collisionality_reaudit_fast_summary.json``
- bounded corrected LHD figure:
  ``docs/_static/figures/paper/dkx_fig1_lhd_collisionality_reaudit_fast.png``

.. figure:: _static/figures/paper/dkx_fig1_lhd_collisionality_reaudit_fast.png
   :alt: Corrected bounded LHD collisionality scan for dkx
   :width: 85%

   Corrected bounded LHD collisionality rerun with the guarded scan-input writer.
   This artifact resolves the expected FP/PAS separation and is backed by direct
   JSON-based assertions, but it is a bounded fast branch artifact rather than the
   final audited paper figure.

- bounded corrected W7-X summary:
  ``tools/publication_figures/artifacts/w7x_collisionality_reaudit_fast_summary.json``
- bounded corrected W7-X figure:
  ``docs/_static/figures/paper/dkx_fig2_w7x_collisionality_reaudit_fast.png``

.. figure:: _static/figures/paper/dkx_fig2_w7x_collisionality_reaudit_fast.png
   :alt: Corrected bounded W7-X collisionality scan for dkx
   :width: 85%

   Corrected bounded W7-X collisionality rerun after fixing the scan-input writer.
   This rerun also resolves clean FP/PAS separation and is light enough for branch-level
   validation, but it remains a bounded fast artifact rather than the final audited
   paper figure.

Autodiff / sensitivity validation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Literature anchors:

- `Paul et al. 2019 adjoint optimization <https://arxiv.org/abs/1904.06430>`_
- `APS adjoint optimization abstract <https://meetings-archive.aps.org/dpp/2018/bp11/36/>`_

Script:

- ``tools/publication_figures/generate_autodiff_sensitivity_validation.py``

Artifacts:

- ``tools/publication_figures/artifacts/dkx_autodiff_sensitivity_validation_summary.json``
- ``docs/_static/figures/paper/dkx_autodiff_gradient_check.png``
- ``docs/_static/figures/paper/dkx_autodiff_gradient_check.pdf``
- ``docs/_static/figures/paper/dkx_autodiff_sensitivity_map.png``
- ``docs/_static/figures/paper/dkx_autodiff_sensitivity_map.pdf``

Fortran-v3 RHSMode 4/5 source-contract checks:

- ``dkx.sensitivity.validate_fortran_v3_adjoint_sensitivity_constraints``
  mirrors the source-code restrictions from ``validateInput.F90`` for adjoint
  sensitivity decks.
- ``dkx.sensitivity.fortran_v3_adjoint_sensitivity_output_fields`` pins
  the sensitivity HDF5 field names emitted by ``writeHDF5Output.F90`` before
  the numerical Fortran replay fixtures are promoted.
- ``dkx.sensitivity.fortran_v3_adjoint_sensitivity_output_ranks`` and
  ``validate_fortran_v3_adjoint_sensitivity_output_surface`` validate the
  required RHSMode=4/5 field names and tensor ranks against either HDF5-like
  arrays or lightweight JSON summaries.
- ``tests/test_sensitivity.py`` checks valid and invalid RHSMode 4/5 decks,
  including the Fortran source-code condition that writes ``dParallelFlowdLambda``
  from ``adjointParticleFluxOption`` or ``debugAdjoint``.
- ``tests/fixtures/fortran_v3_reference_fixture.json`` contains compact
  RHSMode=4/5 reference summaries and embedded namelist text. The checked
  W7-X-like analytic decks pin radial-current, heat-flux, total-heat-flux,
  parallel-flow, bootstrap, and RHSMode=5 ``dPhidPsidLambda`` sensitivity
  outputs from SFINCS Fortran v3 without committing generated HDF5 files,
  including
  ``dRadialCurrentdLambda = sum_s Z_s dParticleFlux_s/dLambda`` and
  ``dTotalHeatFluxdLambda = sum_s dHeatFluxdLambda_s`` and
  ``dBootstrapdLambda = sum_s Z_s dParallelFlowdLambda_s``.
- ``small_rhsmode4_debug_summary_2026-06-25.json`` records a bounded
  debug-adjoint finite-difference run. The regression validates every debug
  field name/rank, selected analytic/finite-difference values, finite percent
  errors below the checked tolerance, and the Fortran NaN mask for unfilled
  lambda/mode entries.

.. figure:: _static/figures/paper/dkx_autodiff_gradient_check.png
   :alt: Autodiff gradient validation for dkx
   :width: 92%

   Bounded manuscript-grade autodiff validation. The checked-in summary records
   centered finite-difference comparisons, primal residuals, and adjoint residuals
   for custom-linear-solve gradients. The SFINCS full-system panel uses a pinned
   tiny PAS fixture and validates the implicit-differentiation path without changing
   production solver defaults.

.. figure:: _static/figures/paper/dkx_autodiff_sensitivity_map.png
   :alt: Boozer harmonic sensitivity maps for dkx
   :width: 92%

   Differentiable ``geometryScheme=4`` Boozer-harmonic sensitivity maps. This
   artifact validates the public analytic-Boozer geometry path used by examples and
   optimization scaffolds; it does not claim full VMEC-boundary optimization.

Bounded integration work
------------------------

These lines of work are useful for integration review, but they are not
release-facing publication claims unless and until they are added to
``tools/publication_figures/validation_manifest.json`` with explicit
``release_gate`` metadata.

Open research topics
^^^^^^^^^^^^^^^^^^^^

- QI/device-QI solver research: QI seed-robustness, hard-seed GPU
  campaigns, and device-QI operator-reuse promotion evidence are preserved on
  the ``research/qi-device-hard-seed`` branch. They are not release-facing
  validation artifacts in the stable core. Any future QI/device-QI promotion
  must restore or regenerate compact artifacts from the candidate branch and
  pass residual, output, runtime, memory, CPU/GPU parity, solver-trace, and
  documentation checks before appearing in this matrix.
- PAS memory/runtime: guarded ``tzfft`` and weak-PAS fail-fast routes are bounded
  diagnostics. The byte-budgeted geometry4 and HSX real-solve probes are
  residual-clean and solver-path stable, but they are not promoted because they
  regress runtime, memory, or both against the checked baselines. Promotion still
  requires residual-clean CPU/GPU evidence with no parity loss and a measured
  runtime or memory win on geometry-rich PAS floors.
- Single-case scaling: transport-worker case/RHS throughput has its own checks,
  but single-case multi-device strong scaling remains experimental until a warm,
  compile-amortized, device-covered artifact shows a real speedup.
- Coverage/refactor: policy seams and solver helpers have focused tests, but the
  package-wide ``95%`` target still requires more owner-module tests for
  profile solves, transport solves, operator assembly, output writing, and a
  JAX-safe coverage environment.
- VMEC/Boozer workflow: validation checks cover workflow provenance, optional
  ecosystem checks, and proxy-gradient consistency. Full VMEC-boundary-to-SFINCS
  kinetic transport gradients remain deferred.
- Deferred validations: W7-X ambipolar validation, high-``nu`` analytic-limit
  extension, broader MONKES/KNOSOS overlap, production-resolution QI ladders, and
  large geometry-rich PAS claims remain deferred until checked-in artifacts with
  numerical acceptance criteria and ``release_gate`` metadata exist.
  Production-resolution QI ladders should not launch until the GPU hard-seed run
  writes output through a true device route.

Mapped x-grid PAS transport evidence (retired)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The mapped x-grid research owners and their tests were deleted with the
legacy pipeline (see :doc:`adaptive_speed_grid`). The bounded artifacts are
retained as a historical record:

- ``docs/_static/mapped_xgrid_transport_evidence_rhsmode2_tiny.json``
- ``docs/_static/mapped_xgrid_transport_evidence_rhsmode2_tiny.csv``
- ``docs/_static/mapped_xgrid_transport_evidence_reduced_pas_tokamak_rhsmode2.json``
- ``docs/_static/mapped_xgrid_transport_evidence_reduced_pas_tokamak_rhsmode2.csv``

Scope and status:

- The tiny artifact is a smoke comparison against a small RHSMode=2 PAS fixture.
- The reduced PAS tokamak artifact compares mapped ``Nx=7`` candidates against an
  ``Nx=13`` reference and records residuals, active-DOF counts, elapsed time,
  moment-objective diagnostics, and transport-matrix error.
- The best reduced PAS tokamak candidate by transport error is a bounded evidence
  point for the opt-in mapped-grid machinery, not a claim that mapped grids should
  replace default SFINCS-v3-compatible grids.
- Full-FP mapped-grid compatibility remains open because the full-FP collision
  precompute path makes assumptions that are not mapped-grid compatible.

Promotion criteria:

- add the entry to the manifest with ``claim_status`` no stronger than
  ``bounded_proxy`` until production-resolution evidence exists,
- compare against higher-resolution default-grid references, not only
  same-resolution smoke solves,
- demonstrate residual-clean CPU/GPU behavior on at least one representative PAS
  transport case,
- and keep default ``xGridScheme`` behavior unchanged unless full-suite parity and
  runtime/memory checks justify promotion.

QI/device-QI research boundary
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

QI seed-robustness scripts, hard-seed campaign artifacts, and device-QI
promotion tests are preserved on the ``research/qi-device-hard-seed`` branch.
The stable core intentionally keeps only general solver-policy and output-schema
contracts. It does not ship QI seed-robustness JSON artifacts, QI promotion
figures, or QI-only example inputs as release evidence.

Promotion criteria for any future QI/device-QI return to stable:

- regenerate compact artifacts from the candidate branch,
- pass strict true-residual and output-write checks on CPU and GPU,
- record solver traces, runtime, and peak-memory budgets,
- compare supported observables against SFINCS Fortran v3 where the models
  overlap,
- document the differentiability scope, and
- add only the minimal stable source/tests/docs needed for the admitted default.

Solver-path policy refactor
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Source and tests:

- ``dkx/solve.py`` (the automatic solver policy and its route-selection
  helpers)
- ``tests/test_solve.py`` and the solver-trace tests

Scope and status:

- Solver-path selection (route eligibility, memory-based auto-selection,
  preconditioner construction, residual checks, and recycling) is centralized
  in ``dkx/solve.py``; the standalone legacy policy module was retired
  with the legacy solver packages. Selection decisions are recorded in the
  versioned solver trace (``dkx/solver_trace.py``).
- This is a maintainability and reproducibility check for solver-path selection.
  It does not by itself support a new performance or physics claim.

Promotion criteria:

- keep policy tests green alongside the driver-wrapper tests,
- verify no solver-path branch change is promoted without residual-clean and
  parity-clean artifacts,
- and summarize solver-path provenance in release artifacts before using a new
  branch as a documented default.

Pinned independent monoenergetic device-family comparison
---------------------------------------------------------

The first independent device-family artifact is
``validation/independent_cross_code_v1.json``. It compares one axisymmetric
DSHAPE tokamak surface, NCSX, and W7-X EIM on the equations all participating
codes actually share: zero-field monoenergetic drift kinetics, Lorentz
pitch-angle scattering, and DKES trajectories. DSHAPE and NCSX are live YANCC
comparisons; W7-X EIM uses the pinned MONKES database row at its authored
``27 x 55 x 140`` resolution.

The comparison does not equate similarly named inputs. It maps physical
``nu/v`` through DKX's applied ``nuDHat(x0)`` factor, converts the dimensional
DKES coefficients to the Beidler ``D*`` convention with the *local* surface
radius, corrects the recorded reference-field convention, and applies the
explicit handedness map to ``D31*`` and ``D13*``. The checked audit recomputes
all of this from raw coefficients rather than trusting stored normalized
values::

   python tools/paper_benchmarks/audit_independent_cross_code_validation.py \
     --yancc-root ../YANCC

All four coefficients pass the 6% bounded acceptance criterion.

.. list-table:: Maximum relative difference by device
   :header-rows: 1
   :widths: 34 26

   * - device
     - maximum relative difference
   * - DSHAPE
     - 5.51%
   * - NCSX
     - 1.66%
   * - W7-X EIM
     - 5.52%

``D33*`` differs by at most 0.065%. The artifact pins external commits,
input and compact-output checksums, reference residuals, resolution, hardware,
wall time, and process peak RSS. The threshold covers the measured
cross-discretization spread, is fixed by the accepted artifact, and may not be
relaxed to admit a future regression.

This is deliberately not full-Fokker--Planck, finite-``Er``, ambipolar-profile,
experimental, or cross-code performance validation. Exact SFINCS Fortran-v3
discrete compatibility fixtures remain enforced separately; the checked
full-kinetic DSHAPE SFINCS table is recorded as context only because substituting
it here would compare different equations.

Closed post-release research topics
-----------------------------------

The following research topics are not release blockers. They are closed in the
manifest as ``deferred_post_release`` with explicit criteria for reopening them
in a later research/nightly cycle.

1. Electric-field sweeps
^^^^^^^^^^^^^^^^^^^^^^^^

Literature anchors:

- `Landreman et al. 2014 <https://publications.lib.chalmers.se/records/fulltext/199559/local_199559.pdf>`__

Publication target:

- one tokamak-like case,
- one stellarator case,
- fluxes, flows, and bootstrap current versus normalized radial electric field,
- clear comparison of partial, DKES-like, and full-trajectory models.

Scaffold:

- ``tools/publication_figures/generate_er_trajectory_sweep.py``

This script implements the correct upstream trajectory-model switches and
produces JSON summaries plus 2x2 publication-style figures.

Fixed artifacts:

- audited tokamak-like reference summary:
  ``tools/publication_figures/artifacts/er_sweep_tokamak_reference_summary.json``
- audited tokamak-like reference figure:
  ``docs/_static/figures/paper/dkx_er_trajectory_sweep_tokamak_reference.png``
- bounded stellarator-like fast summary:
  ``tools/publication_figures/artifacts/er_sweep_stellarator_fast_reference_summary.json``
- bounded stellarator-like fast figure:
  ``docs/_static/figures/paper/dkx_er_trajectory_sweep_stellarator_fast_reference.png``

.. figure:: _static/figures/paper/dkx_er_trajectory_sweep_tokamak_reference.png
   :alt: Tokamak-like electric-field trajectory-model sweep for dkx
   :width: 85%

   Fixed tokamak-like ``E_r`` sweep across DKES, partial, and full trajectory
   models. This entry is pinned to checked-in JSON and figure artifacts, and it
   is backed by direct numerical assertions on zero-field agreement and
   finite-field model separation.

.. figure:: _static/figures/paper/dkx_er_trajectory_sweep_stellarator_fast_reference.png
   :alt: Stellarator-like electric-field trajectory-model sweep for dkx
   :width: 85%

   Fixed stellarator-like fast branch scaffold across DKES, partial, and full
   trajectory models. This is intentionally a bounded branch-validation
   artifact: it resolves the expected model separation on the selected input,
   but the full-resolution stellarator sweep remains a heavier validation target.

Validation goal:

- verify small-field agreement and large-field separation behavior,
- make the ordering and crossover behavior explicit in both assertions and figures,
- promote the stellarator-like branch scaffold to a full-resolution audited entry
  only after the runtime/cost tradeoff is acceptable for the release/nightly workflow.

2. High-collisionality proxy after collisionality audit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Literature anchors:

- `Landreman et al. 2014 <https://publications.lib.chalmers.se/records/fulltext/199559/local_199559.pdf>`__

Closed branch evidence:

- bounded LHD and W7-X fast reruns resolve FP/PAS separation on the same
  four-point ``\\nu'`` ladders without label collapse in the stored outputs
- audited full LHD and W7-X collisionality summaries resolve both FP and PAS labels
  on seven-point ``\\nu'`` ladders
- a checked-in trend proxy records high-collisionality tail slopes from those
  corrected artifacts:
  ``tools/publication_figures/artifacts/dkx_high_collisionality_trend_proxy_summary.json``
- a checked-in Simakov-Helander normalization audit records the Appendix-B
  geometry ingredients, ``FSABHat2`` recomputation, inverse-``nu`` slope checks, and
  explicit readiness status:
  ``tools/publication_figures/artifacts/dkx_simakov_helander_limit_audit_summary.json``

.. figure:: _static/figures/paper/dkx_high_collisionality_trend_proxy.png
   :alt: High-collisionality trend proxy from checked-in collisionality artifacts
   :width: 92%

   Trend proxy for the ``L11`` and ``L12`` tails. The SFINCS 2014 paper states that
   PAS ``L11``/``L12`` scale like ``+nu`` at high collisionality, while
   momentum-conserving FP/model-operator results should approach inverse-``nu``
   scaling in the ``nu' >> 1`` limit. The checked-in LHD artifact satisfies the
   loose inverse-tail proxy, but the W7-X artifact does not yet. The stricter
   Simakov-Helander audit therefore keeps both geometries deferred until wider
   high-``nu`` scans are pinned, so this figure is kept as an implemented trend check
   rather than the final analytic-limit reproduction.

.. figure:: _static/figures/paper/dkx_simakov_helander_limit_audit.png
   :alt: Simakov-Helander high-collisionality readiness audit
   :width: 92%

   Normalization and readiness audit for the full Simakov-Helander entry. The audit
   confirms that checked-in ``sfincsOutput.h5`` files contain the geometry quantities
   needed for an Appendix-B comparison, but it keeps the full analytic-limit
   reproduction closed because the full collisionality summaries stop near
   ``nu'=10`` rather than a wider ``nu' >> 1`` range. The JSON summary also
   carries a recommended logarithmic high-``nu'`` extension grid for each case,
   ending near ``nu'`` of ``100``, so the next heavy run is pinned and reviewable.

Post-release acceptance criteria:

- keep machine-readable summary artifacts for each full scan,
- keep the Simakov-Helander audit artifact in CI as the parent check for future
  high-collisionality scan work,
- use
  ``tools/publication_figures/artifacts/dkx_simakov_helander_high_nu_run_plan.json``
  as the executable high-``nu'`` extension plan; it is generated from the audit
  and pins LHD and W7-X extension commands ending near ``nu'=100``,
- run each plan entry's ``pilot_command`` first; the first LHD FP pilot at
  ``nu'=17.78`` on the office GPU took about ``569 s`` for one transport point,
  so the complete FP/PAS LHD+W7-X extension is a nightly/workstation campaign,
- and only promote the deferred full analytic-limit reproduction after wider
  high-``nu`` LHD and W7-X scans are regenerated and the readiness check passes.

The run-plan artifact is explicitly labelled as a deferred executable plan. Its
machine-readable checks record that residual thresholds are wired into every command
and that ``ready_for_literature_claim`` remains false because no completed high-``nu``
scan artifact is present. Publication panel summaries use the same convention:
``publication_figure.claim_status`` is ``proxy_or_deferred`` unless the source JSON
carries numerical acceptance criteria and is checked in as a converged artifact.

A separate W7-X high-``nu`` preconditioner/performance figure is available for the
single first FP point, but it is not a physics-validation entry. Its recorded
acceptance criteria only support the bounded claim that sparse-helper factor reuse
is residual-clean, faster than no-reuse, uses fewer sparse factorizations, and
rejects the failed bounded Krylov route. The figure metadata therefore keeps
``ready_for_physics_validation_claim=false``.

3. W7-X ambipolar-field validation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Literature anchors:

- `Pablant et al. 2020 ion-root context <https://sites.fusion.ciemat.es/jlvelasco/files/papers/pablant2020ionroot.pdf>`_
- `Pablant et al. 2018 W7-X core radial electric field <https://sites.fusion.ciemat.es/jlvelasco/files/papers/pablant2018er.pdf>`_
- `Nature 2021 W7-X neoclassical validation context <https://www.nature.com/articles/s41586-021-03687-w>`_

Publication target:

- one figure comparing neoclassical ``E_r`` and/or heat-flux trends against the
  published W7-X validation context,
- one table documenting exactly which approximations and reconstructed inputs were used.

Validation goal:

- make any profile reconstruction assumptions explicit,
- use this entry only if the reconstructed input set is scientifically defensible.

Stable artifact check:

- ``tools.release.artifacts.build_w7x_ambipolar_root_provenance_panel``
- ``tools/publication_figures/provenance/w7x_ambipolar_provenance_template.json``

The stable branch keeps the ambipolar solver API, scan/readback tests, and a
fail-closed provenance panel builder. Long W7-X scan and figure generation is a
publication-audits research workflow until a defensible equilibrium/profile
reconstruction is supplied and the resulting source artifact is checked in.

The deferred panel includes explicit ``acceptance_gates``:

- finite distinct ``E_r``/current scan points,
- finite ambipolar roots,
- radial-current sign bracketing,
- roots inside the scanned ``E_r`` range,
- root consistency with a sign-change bracket,
- a resolved local current slope at the accepted root,
- an ion-root candidate,
- complete equilibrium/profile/discharge/literature provenance,
- checked-in source-artifact status,
- and the combined ``ready_for_literature_claim`` check.

Without a provenance JSON containing ``equilibrium_source``, ``profile_source``,
``configuration_or_shot``, and ``literature_reference``, generated artifacts remain
``w7x_like_scaffold`` rather than ``w7x_literature_validation``.
Even with complete provenance, generated summaries remain
``w7x_literature_candidate_deferred`` until the matching W7-X summary artifact is
checked in; this prevents an exploratory rerun from being labelled as a closed
literature comparison.
Start from
``tools/publication_figures/provenance/w7x_ambipolar_provenance_template.json``;
it is intentionally incomplete and should be copied/finalized for a specific
equilibrium/profile reconstruction before any literature-facing W7-X claim.

Closure note:

- the stable core keeps the ambipolar solver and provenance/artifact tests,
- the checked-in literature artifact and long generator are intentionally absent,
- this entry is classified as ``deferred_post_release`` until a
  defensible W7-X input reconstruction is run and its summary/figure are pinned in
  the repository.

4. MONKES / KNOSOS overlap
^^^^^^^^^^^^^^^^^^^^^^^^^^

Literature anchors:

- `MONKES paper <https://arxiv.org/abs/2312.12248>`_
- `KNOSOS paper <https://arxiv.org/abs/1908.11615>`_

Publication target:

- coefficient overlap on monoenergetic shared-model subsets,
- low-collisionality trend comparison where the models are not exactly identical.

Validation goal:

- separate exact overlap claims from qualitative trend/ordering claims,
- keep this entry focused on the model subset that is genuinely comparable.

Keeping this page in step with the code
---------------------------------------

Each time a new figure entry is implemented, update both:

- this page,
- and ``tools/publication_figures/validation_manifest.json``.

That keeps the manuscript-facing validation story synchronized with the code structure
and the test/benchmark infrastructure.
