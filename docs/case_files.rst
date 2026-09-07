Case files
==========

DKX schema version 1 defines an immutable, physically named ``Case``. TOML is
the primary human-authored format; JSON has exactly the same semantic fields
for generated inputs. Both formats pass through one validation boundary and
produce a deterministic SHA-256 case ID that is independent of table/key order
and the case file's location.

Start from the complete commented template, then run it:

.. code-block:: console

   dkx schema --format toml > case.toml
   dkx validate case.toml
   dkx run case.toml --out result.nc
   dkx inspect result.nc

``dkx run`` executes the case and prints the solver route, the achieved true
residual, and the wall time; ``--out`` writes the versioned NetCDF ``Result``,
and omitting it runs without saving rather than guessing a path. ``dkx inspect``
reads a saved ``Result`` back and lists its arrays without recomputing them.

``inspect`` prints no units column. A ``Result`` does not yet carry
per-variable units metadata, so an empty column would suggest the metadata is
present and blank rather than absent; array names carry the unit by convention
(``heat_flux_W_m2``).

Machine tooling can request JSON Schema instead:

.. code-block:: console

   dkx schema --format json > case-v1.schema.json

The checked full-schema example is
``examples/05_ambipolar_profile/w7x_case.toml``. Its field names carry
engineering units where a dimensional value appears, such as ``density_m3``,
``temperature_keV``, and ``search_kV_m``. Solver methods are
named for the route they take: ``structured_direct`` eliminates the Legendre
blocks in one direct sweep, ``recycled_krylov`` iterates with a coarse
preconditioner and a reused search subspace, and ``sparse_direct_referee``
factors the assembled operator as an independent cross-check on the other two.
:doc:`numerics` describes each route and the ``auto`` policy that chooses
between them.

The optional ``resolution.pitch_speed_ramp`` isolates a subtle convergence
choice without exposing a namelist route. Its values are the matching SFINCS
``Nxi_for_x_option`` rules:

.. list-table::
   :header-rows: 1
   :widths: 12 62

   * - value
     - rule
   * - 0
     - retain the declared maximum pitch order at every speed node
   * - 1
     - linear speed ramp (the default)
   * - 2
     - quadratic ramp

Because increasing ``resolution.pitch`` under a ramp changes several
speed-local truncations at once, resolution studies should record the option
and the resulting active mode counts rather than describe only the maximum
pitch order.

For a bounded allocation diagnosis, optional
``resolution.pitch_modes_by_speed`` replaces the ramp with one explicit active
mode count per speed node. Counts must be integers, nondecreasing with speed,
between 4 and ``resolution.pitch``, and must reach the declared pitch maximum
at the final node. ``pitch_speed_ramp`` must remain at its implicit default
when this advanced evidence control is present. Omitting the array preserves
the historical default and every existing case ID. The ``Result`` and NetCDF
metadata distinguish ``explicit`` from ``pitch_speed_ramp`` allocation and
retain the exact counts and their sum. The explicit control changes only the
active Legendre mask inside the same rectangular operator shape; it is not by
itself a convergence claim.

Validation and portability
--------------------------

Validation errors identify the full input path, supplied value, expected form,
and a correction. The checked rules are:

- profile arrays must have one entry per requested surface;
- surface values must be unique and lie from zero through one;
- all physical profile values must be finite and positive;
- an ambipolar search requires increasing finite bounds.

Schema validation does not require a geometry file to exist. This keeps a case
portable and permits validation before data staging. ``Case.geometry_path``
resolves a relative geometry path beside the loaded case when execution begins.
The source location is provenance and is excluded from the semantic case ID.

Declarative scan preflight
--------------------------

Schema-v1 accepts Cartesian and zipped explicit-value axes. ``Case.scan``
computes the case count before launch and rejects counts above ``max_cases``;
zipped axes must have equal lengths. Resume metadata and append-safe result
storage belong to the scan and result execution slice and are not simulated
by the validator.

Ambipolar work preflight
------------------------

For ``workflow = "ambipolar_profile"``, ``dkx validate`` also reports the
final adaptive hierarchy size, a conservative maximum number of retained
kinetic evaluations per surface and profile, and the corresponding retained
evidence bytes. This calculation reads only the versioned case: it does not
load geometry, initialize JAX, or launch a solve. Cases whose configured bound
exceeds 100,000 retained evaluations per surface fail validation with the same
actionable reduction advice as execution.

The evaluation count is a capacity bound, not a runtime estimate. It assumes
every hierarchy point could bracket a root on every refinement level, so a
normal run should use fewer evaluations. Conversely, no finite hierarchy can
prove the absence of an even number of hidden crossings. Use the preflight to
bound a campaign before launch, then retain the actual evaluation reasons,
refinement levels, brackets, and exhaustion status in the Result.

The checked
``validation/inputs/w7x_standard_native_ambipolar_admitted_flux_preflight.toml``
case applies this contract to the five-surface W7-X transport-flux grid admitted
by the fixed-field resolution cross-check: an independent study at a fixed
electric field that admits a phase-space grid only once refining each axis in
turn leaves the retained fluxes inside its declared tolerance.

.. list-table:: Preflight bound for that case
   :header-rows: 1
   :widths: 44 26

   * - bound
     - value
   * - evaluations per surface
     - 1,023
   * - evaluations for the profile
     - 5,115
   * - retained-evidence bytes
     - 4,746,720

This is a launch bound only: the input deliberately excludes bootstrap current
from its convergence observables because the high-zeta current grid is not yet
admitted.

Case execution and results
--------------------------

The directly executable route accepts built-in analytic geometry, a VMEC
``wout``, or a Boozer ``.bc`` file for prescribed-electric-field and ambipolar
profiles. It consumes ``Case`` fields directly: it does not serialize or parse
a SFINCS namelist while constructing grids, geometry, species, collisions, or
the operator. Run a checked example from Python:

.. code-block:: python

   import dkx

   case = dkx.Case.from_file("examples/01_tokamak_profile/case.toml")
   result = dkx.run(case)
   result.print_summary()
   result.save()                         # the case's [output].file

   result.plot("profile.png")
   particle_flux = result.particle_flux_m2_s
   certificate = result.certificate()

``result.certificate()`` returns the compact record used for review: solver
route, residual, iteration counts, geometry checksum, package and runtime
versions, and timings.

For an ambipolar profile, every retained electric-field evaluation also keeps
the already integrated speed-node contributions
``evaluation_particle_flux_m2_s_vs_speed`` and
``evaluation_heat_flux_W_m2_vs_speed`` on named
``(surface, evaluation, speed, species)`` axes. ``speed_v_th`` is the
dimensionless node coordinate :math:`v/v_{th}`. Summing either diagnostic over
``speed`` reproduces the corresponding retained species flux; these compact
arrays make speed-local convergence failures inspectable without retaining the
full distribution function. They are diagnostics, not a claim that the speed
or pitch discretization is converged.

The checked ``validation/ambipolar_speed_local_pitch_v1.json`` example shows
the intended use: compare identical physical fields before paying for another
root campaign, identify which speed nodes dominate an allocation change, and
keep a failed ceiling test as ``phase_space_converged=false``. Because the
collision operator couples speed nodes, a higher pitch ceiling can expose
changes at neighboring high-speed nodes; a single-node diagnosis therefore
does not license a single-axis convergence claim.

Full modal states can additionally retain
``evaluation_legendre_tail_relative_l2`` on the same named axes. The ratio is
the volume- and Legendre-orthogonality-weighted L2 norm of the final two active
modes divided by the norm of all active modes. It is supporting evidence, not
a replacement for observable movement under a resolution increase.

Native profile and ambipolar solves now request complete states, including
on the generated structured route with unequal pitch chains. Ambipolar
results therefore retain the full-state tail ratio at every evaluated field;
``convergence.retain_legendre_tail`` remains accepted for compatibility but
requires no selected-field replay or additional evaluation budget.

Expert moment-only batches can still retain only a low-order head. Their
zero-padded tail is not a full kinetic state and generally fails the original
residual check. ``retain_legendre_tail=True`` on that batch API computes a
separate selected-tail upper bound; ``retain_full_state=True`` instead recovers
all active blocks. See :doc:`parallelism`. Historical
``validation/ambipolar_joint_pitch_speed_v1.json`` evidence retains the older
selected-tail contract and unresolved speed/pitch checks; it is not a
certificate for the new complete-state workflow.

For a VMEC equilibrium, change only the geometry source:

.. code-block:: toml

   [geometry]
   format = "vmec"
   file = "wout_my_device.nc"
   surfaces = [0.16, 0.25, 0.36]

The file is resolved through ``Case.geometry_path``, read once per profile, and
its exact SHA-256 is stored in the result. One phase-space grid is reused across
all surfaces because their array shapes are identical; each surface still gets
its own radially interpolated magnetic geometry and operator coefficients.
``value_kV_m`` is explicitly normalized using the pinned 1 keV and 1 m SFINCS
reference set (for which the numerical conversion to ``ErHat`` is one).

For a Boozer equilibrium, use the physical source format rather than a numbered
SFINCS geometry route:

.. code-block:: toml

   [geometry]
   format = "boozer"
   file = "my_device.bc"
   surfaces = [0.20, 0.30]

The reader auto-detects the six-column cosine-only and ten-column
asymmetric v3 conventions at the file boundary. It reads and parses the source
once, then reuses the immutable Fourier tables for every surface. A DKX case
file therefore never requires ``geometryScheme = 11`` or ``12``, and never
converts the ``Case`` back into a SFINCS namelist. At least two tabulated
surfaces are required for the radial derivatives used by the kinetic operator.
The exact source SHA-256 is retained in the Result.

For ambipolar execution, select the workflow and give physical search
controls:

.. code-block:: toml

   [run]
   workflow = "ambipolar_profile"

   [electric_field]
   mode = "ambipolar"
   search_kV_m = [-5.0, 5.0]
   search_points = 5
   root_tolerance_kV_m = 0.05
   max_root_iterations = 8
   find_all_roots = true
   continue_branches = true

Each surface performs one memory-bounded coarse electric-field batch, refines
every sign-changing bracket with real kinetic solves, and preserves all
evaluated fields, radial currents, fluxes, residuals, brackets, slopes, and
root classifications in ``Result``. A profile selects the root nearest zero on
its first surface and then the root nearest the selected branch on the
preceding surface.

With ``solver.method = "auto"``, every batch records the structural route that
actually ran. If an individual field misses the declared true-residual target,
DKX retries only that field with one memory-bounded GMRES solve. The original
failed attempt and the recovery attempt remain separately visible through the
``solver_attempt`` dimension, including requested and executed methods,
residuals, acceptance flags, and reasons. A recovery that still misses the
unchanged target raises with both residuals. Explicitly selected solver methods
remain fail-closed and are never changed automatically.

When no root is bracketed, DKX retains the sampled point with the smallest
absolute radial current, labels it ``no_bracketed_root``, and never calls it
ambipolar.

Enable the existing convergence contract to insert every interval midpoint in
a deterministic bounded hierarchy:

.. code-block:: toml

   [convergence]
   enabled = true
   observables = ["particle_flux", "heat_flux", "electric_field"]
   relative_tolerance = 0.02
   max_refinements = 2

Every added kinetic solve records ``evaluation_reason`` and
``evaluation_refinement_level``. Each refinement level records its search and
total solve counts, discovered root count, root movement, requested-observable
movement, and maximum final bracket width. The preflight records a conservative
retained evaluation budget and rejects work or evidence storage beyond its
fixed work and requested memory bounds before allocating the hierarchy.

The bounded hierarchy runs through the configured ``max_refinements`` so an
early stable root cannot prevent a finer declared level from exposing another
pair of crossings. ``ambipolar_refinement_status`` is ``resolved`` only when
the final two levels retain the same nonzero root count and meet the declared
root, observable, and bracket-width tolerances. ``refinement_exhausted`` means
roots were observed but the final evidence did not stabilize.
``no_bracket_observed`` means the finite hierarchy observed no sign-changing
bracket. It is not a proof that no root exists: an even number of crossings can
remain hidden between the finest adjacent samples. ``find_all_roots`` therefore
means every bracket exposed by the declared finite hierarchy, not every
mathematically possible root. The discrete branch evidence below must still
pass independent dense-surface validation before it is promoted.

For an independently discovered set of sign-changing intervals, a second
stage can promote only those brackets at a more expensive phase-space grid:

.. code-block:: toml

   [electric_field]
   mode = "ambipolar"
   search_kV_m = [-5.0, 15.0]
   search_strategy = "seeded_brackets"
   seed_brackets_kV_m = [
     [[-2.0, -1.0], [7.0, 9.0]], # surface 1
     [[-2.0, -1.0]],             # surface 2
   ]
   find_all_roots = true

Every distinct endpoint is one real kinetic solve and every sign-changing
seed is refined by real bisection solves. The preflight bounds endpoint plus
bisection work per surface. Seeded promotion deliberately forbids the adaptive
global hierarchy: first converge and independently review the discovery grid,
then supply all of its candidate brackets. ``ambipolar_search_scope`` is
``explicit_seeded_intervals_only`` and unsampled crossings are never excluded.
A seed that does not bracket a root produces ``seeded_bracket_failed`` or
``seeded_bracket_partial_failure`` rather than global no-root evidence.

This split follows the useful sign-change/search-range discipline in PENTA's
``find_Er_roots``, while retaining DKX's stronger evidence boundary. PENTA's
routine requests a wider search for zero or even crossing counts, and refines
an interpolated radial-flux fit. DKX retains every kinetic endpoint and
bisection evaluation, and never presents interpolation as a solved root.

Radial branch evidence
----------------------

After every surface has completed root discovery, DKX assigns each retained
root a stable ``ambipolar_root_branch_id``. The tracker predicts each existing
branch from its two most recent radial points and uses a global minimum-cost
assignment, admitted only within one quarter of the declared electric-field
search span. The first observation at the profile boundary is labeled
``boundary_origin`` rather than a physical creation. Interior unmatched roots
and branches are labeled ``creation`` and ``loss``. A lost branch that
approaches a survivor within the continuation tolerance also retains a
``merger`` event whose detail explicitly calls it a *discrete merger
candidate*.

DKX additionally records branch-order ``crossing`` and
``classification_transition`` events between adjacent sampled surfaces. These
are discrete profile observations, not claims that the continuous bifurcation
location has been resolved. Every event retains its participating branch IDs,
root indices, electric field, explanatory detail, and nonsmooth flag. The
``ambipolar_nonsmooth_event`` surface mask and Result warning identify intervals
where branch-local derivatives are nonsmooth or undefined.

Selection is separate from discovery: every alternative root and branch stays
in the Result. With ``continue_branches = true``, the first available surface
selects the root nearest zero and later surfaces retain that branch ID. If the
selected branch is lost, the root nearest its electric field on the preceding
surface is selected and ``ambipolar_selection_reason`` records the fallback.
With continuation disabled, each surface selects its root nearest zero while
branch evidence remains visible. The electric-field plot overlays every branch on the
selected profile.

``Result`` copies its named arrays and makes them read-only. The
``dimensions`` map gives every array's named axes without requiring xarray;
``save`` writes schema-v1 NetCDF4, and ``Result.load`` reads it through the same
contract. Files contain the canonical case, normalization, geometry checksum,
package/runtime/device versions, selected route, residual, iteration and timing
evidence, and peak host memory.

The executable route supports these case fields and values:

.. list-table::
   :header-rows: 1
   :widths: 30 56

   * - case field
     - supported values
   * - ``workflow``
     - ``"profile"`` with a prescribed field, or ``"ambipolar_profile"``
       with a bounded search
   * - ``format``
     - ``"analytic"``, ``"vmec"``, or ``"boozer"`` geometry
   * - ``magnetic_drifts``
     - ``"dkes"``
   * - ``phi1``
     - ``"off"``
   * - profile surfaces
     - at least two

Unsupported combinations fail with the exact case field and a correction; they
are not silently downgraded. Resumable scan execution, phase-space convergence
levels, and SFINCS conversion are later work. Existing namelist workflows remain
available through ``dkx.run`` and the established CLI without a numerical-path
change.
