# DKX: a focused plan for verified transport and optimization

**Decision review: 2026-09-06. Planning only; no release is authorized by this plan.**
Reviewed default branch: [`44a2e864`](https://github.com/uwplasma/DKX/commit/44a2e864f640deb6d7d30834044d0044fcfa7b6d).
Reviewed implementation stack: [`be6506fe`](https://github.com/uwplasma/DKX/commit/be6506fedbf357ab7fbbc1c1eaa35195080afc9e),
the head of [PR #188](https://github.com/uwplasma/DKX/pull/188).
This file supersedes the R0–R11 work queue. The [preceding plan](https://github.com/uwplasma/DKX/blob/be6506fedbf357ab7fbbc1c1eaa35195080afc9e/plan.md)
retains detailed historical measurements, rejected experiments and references.
`docs/development_roadmap.rst` remains an entry point, not another plan.

## 1. The decision

Finish one dependable research workflow before expanding the solver portfolio:
**specified toroidal equilibrium and species profiles → resolved transport and
bootstrap current → a retained stellarator ambipolar branch → checked derivatives
→ fast repeated evaluations → one actual equilibrium-boundary optimization.**
For tokamaks, prescribe Er: an intrinsically ambipolar local model cannot determine
it uniquely. Keep density and temperature explicit; pressure alone fixes neither.

The code has made substantial progress in checking equations and derivatives.
Its principal remaining risk is promoting a collection of small demonstrations
into a general accuracy, performance or optimization claim. More solver choices,
coordinate systems, benchmark decks and documentation pages will not by themselves
close that gap. The next unit of progress is a useful calculation with a stated
model, error budget, repeatable runtime and a runnable example.

Three deliverables replace twelve parallel work packages:

| Order | Deliverable | Completion means |
| --- | --- | --- |
| **1. Trustworthy toroidal calculations** | Consolidate the PR stack; qualify a small PAS/full-FP reference set and a useful native profile/root workflow. | Every advertised observable has original-equation, unit, resolution and model checks; unresolved roots and failed references remain explicit. |
| **2. Fast repeated calculations** | In-memory prepared solves, state/preconditioner reuse, measured CPU/GPU batching and one selected preconditioner improvement. | A whole scan and value/gradient workload improves at the same accepted accuracy, with bounded memory, cold fallbacks and history-independent answers. |
| **3. A real design calculation** | VMEX boundary → equilibrium → geometry → DKX objective, first at prescribed Er, then on a regular stellarator root branch. | Independent full-chain derivatives, feasible improvement beyond numerical uncertainty, useful iteration cost and a cold, finer-grid final validation. |

Documentation, examples and removal of duplication are part of each deliverable.
SFINCS-v3 scientific completeness remains a long-term requirement; it is not
claimed by this first supported envelope. Native Phi1 and the wider drift models
follow explicit capability gates below. NEOPAX and ESSOS consume the validated
interfaces. Mirrors remain last. Do not begin a second active algorithm experiment
until the first has a measured decision or has been stopped.

## 2. What exists, what changed, and what remains unproved

### Source and review state

At the review snapshot there are **19 open DKX PRs**, no open non-PR DKX issues,
and one open SOLVAX PR. All checks reported for #188 are successful; this does not
merge its dependencies or replace required review. The stack has **53 commits,
44 changed files, 5,935 insertions and 2,820 deletions** relative to the reviewed
default branch. These are stack totals, not the size of this planning PR.

| Work to retain | Source / review | Remaining boundary |
| --- | --- | --- |
| Supervised, resumable benchmark attempts; original residual and complete-output checks; PETSc backend/provenance verification | #169–170, #176–177, #181, #183, #185; `tools/benchmarks/parity_performance_matrix.py` | Fresh installed production replay, valid references and observable/grid admission are still needed. |
| Static operator layouts, refreshed collision coefficients and native profile preparation | #173–174, #178, #182, #186–187; `drift_kinetic.py`, `collisions.py`, `execution.py`, `er.py` | Profile updates are opt-in and hold geometry, normalization, species, Coulomb log and discrete layout fixed. They do not make `Case.run` a JAX transformation. |
| Independent batch sharding through JIT and gradients, uneven-batch handling and per-input algebraic status | #179, #182, #187–188; `batch.py`, `api.py` | One whole system remains on one device. Native Case execution does not expose every expert batch option; memory budgets are estimates. |
| Original kinetic and transpose checks; qualified dense adjoint references; native profile-to-root Taylor tests | #180, #184, #188; `solve.py`, `er.py`, solver/root tests | Small fixed-grid evidence does not establish resolution, branch or geometry uncertainty. Partial state recovery has a different derivative contract. |
| Generated Schur-factor reuse across RHSs, forward/transpose solves and refinement | #188 `be6506fe`; existing SOLVAX generated-factor API | Factors belong to one solve execution. Persistent reuse across changed operators is not implemented by this change. Full-FP uses the Krylov route. |

Merge preparation must reconcile #168 with its included timeout fix and #172 with
its included roadmap review, rather than merging duplicate implementations. Review
the dependent chain #169 → #170 → #173 → #174 → #176 → #177 → #178 → #179 → #180
→ #181 → #182 → #183 → #184 → #185 → #186 → #187 → #188. Resolve/rebase in order,
rerun checks on the resulting integration commit, then close superseded PRs with
an explicit disposition. This is planned maintainer work, not permission to bypass
protection. SOLVAX [#100](https://github.com/uwplasma/SOLVAX/pull/100) is also
unmerged; the reviewed local DKX dependency is SOLVAX 0.20.0. Requalify its eventual
successor rather than silently depending on a branch.

The prior worktree and its paused root-timing edit are absent from their recorded
local paths. The pushed head was freshly cloned for this review; no recovery or
commit of that uncommitted fix is claimed. The pushed `find_ambipolar_er` timer
still needs an outer-clock audit including preparation, final cold admission,
slope estimation and optional root enumeration. Preserve its cold final solve.

### Useful results, with their limits

The #188 report records 244 distinct CPU cases and 22 GPU cases for its factor
integration, and native root Taylor orders of 2.00–2.01. On one **7,850-unknown,
three-field PAS objective**, twelve alternating synchronized pairs reduced warm
value/gradient medians from **67.59 to 37.09 ms on CPU** and **221.74 to 195.56 ms
on A4000**; GPU LU calls fell from 288 to 144. CPU used M3 Max/JAX 0.9.2; GPU used
JAX 0.10.2. These compare two checked routes
on each host. They are not full-FP, fresh compilation, allocator peak, whole-root,
whole-optimization or cross-host scaling results. The underlying external traces
were reported in the prior review; this review does not relabel them as new runs.

Earlier README claims about a 744k HSX case and a broad upstream speed ranking
are historical, with differing runtime/memory definitions and invalidated
reference campaigns elsewhere in the record. Retain their evidence in the
performance documentation; do not use them as an unqualified headline. In
particular, small algebraic residuals and agreement with a failed Fortran run
cannot certify a transport observable.

Fresh review evidence is outside Git in `dkx-plan-evidence-20260906`; exact commands,
inputs, source identities and logs accompany the review PR. Its role is to choose
work, not to declare a new production benchmark. The initial bounded local pilot
already reproduced a crucial distinction: a SFINCS full-FP solve reported success
but its original residual was **1.10e-9 at a requested 1e-10**; DKX's was 8.48e-11.
A separate direct reference also missed the gate. Neither failed-reference pair is admitted for a
performance comparison. An initial HSX copy omitted its equilibrium and was
rejected; its partial DKX state also failed the full residual check. Repair inputs
and explicitly choose complete-state versus moment-only evidence before rerunning.

A subsequent **right-preconditioned GMRES/MUMPS** run with an unpreconditioned
stopping norm and inner rtol=1e-13 passed the common original 1e-10 gate: SFINCS
1.20e-11 and DKX 8.48e-11. On this full-FP grid (5×5 angles, Nxi=6, Nx=3, two species, ramped
pitch; 654 rows in the constrained Fortran matrix), scaled differences
were 4.08e-6 for flow, 3.94e-6 for parallel current and below 8.64e-8 for fluxes.
The PETSc log records one symbolic factorization, one numeric factorization and
22 factor applications. This establishes a bounded algebraic comparison, not
grid convergence or a runtime ranking; concurrent review tests exclude performance
promotion. Solver/norm configuration must therefore be part of reference admission.

| Fresh check on the reviewed source | Result / limit |
| --- | --- |
| DKX solver, Er, batch, native execution and planning suites | 248 passed on M3 Max CPU/JAX 0.9.2/SOLVAX 0.20.0; not a full coverage campaign. |
| Office GPU selection | Four targeted batch/AD/full-state cases passed. The unchanged two-device harness also passed PAS/full-FP states, original residuals, uneven batches, actual placement, JIT gradients and an FD check on two A4000s, JAX 0.10.2. |
| YANCC at `6f399a21` | 59 preconditioner/solve/collision tests passed locally, including its SFINCS/MONKES fixture comparisons, coordinate representations, warm state and derivative tests. This does not establish comparative runtime or general warm-reuse correctness. |
| README examples | Native Python and CLI run/inspect pass; Python including the advanced JIT gradient also passes from an isolated DKX install inheriting host dependencies. |
| Existing independent comparison artifact | All three coefficient rows and external YANCC input hashes pass the offline audit; this is an artifact audit, not three fresh kinetic benchmarks. |
| Documentation | Five planning/wording checks and standard Sphinx `-W` pass. Extra `-n` reference checking finds 219 warnings in both baseline and revised docs; resolve this existing API-link debt during consolidation. |

### Capability priorities

| Capability | State at the reviewed head | Decision |
| --- | --- | --- |
| Analytic, VMEC and Boozer native profiles; PAS and full-FP; prescribed Er and stellarator roots | Implemented in a restricted DKES/no-Phi1 native domain | First supported research envelope; qualify multispecies fluxes **and** current separately. |
| Prepared native profile/Er sensitivities and expert batching | Implemented with explicit fixed dependencies | Productize and measure, rather than introduce another public interface. |
| RHSMode 1/2/3, richer trajectories, Phi1, magnetic drifts and distribution export | Broader compatibility/expert support | Preserve regression coverage; a checkmark must distinguish equation, native interface, derivative and validation support. |
| Native single-surface profiles, transport matrices, explicit Case sharding, Phi1/full drifts | Rejected or incomplete in `execution.py` | Single-surface explicit drives and prepared-scan access are useful near-term interface work; Phi1/full drifts require physics gates, not just removing a rejection. |
| Real VMEX boundary optimization | Example 08 is an analytic harmonic geometry proxy | Keep it as a teaching example; do not cite it as the design deliverable. |

Extend the existing `validation/capabilities.toml` and SFINCS control inventory
semantically: collision operators and backgrounds; trajectories/Er terms;
Phi1/quasineutrality and gauge; geometries/asymmetry/radial conventions;
RHS modes, sources/constraints and moments; grids/potentials; exports and solver
controls. The recorded 145 declared namelist controls are an inventory, not 145
validated scientific features. Map unsupported combinations to explicit reasons,
references and tests. Do not promise parity with every experimental SFINCS branch.

## 3. What the external codes and literature change

GitHub branch heads, all-state PR/issue bodies and repository-wide issue/review
comments were inventoried again. Branch comparisons identify divergent code;
technical review concentrates on equations, preconditioners, adjoints, geometry
and failures. This is not a claim to have built every branch or verified every
historical line. Fresh clones of YANCC, MONKES and KNOSOS supplement the existing
SFINCS-v3 and SOLVAX sources.

| Repository | Branches / PRs / non-PR issues, all states | Findings that affect the decision |
| --- | --- | --- |
| [SFINCS](https://github.com/landreman/sfincs) | 39 / 16 / 10 | v3 [`solver.F90`](https://github.com/landreman/sfincs/blob/8df5453472e982df0f6ae005243ce38d57a83711/fortran/version3/solver.F90) reuses preconditioners; `populateMatrix.F90` distinguishes simplified P (0), Jacobian (1), residual f1 action (3) and adjoint operators (4/5). #24 guards MUMPS-specific queries; #25–26 concern root defaults and field scaling. Adjoint, AMG and externalF branches diverge from master. |
| [YANCC](https://github.com/f0uriest/yancc) | 19 / 97 / 0 | Pinned `6f399a21`: full/monoenergetic DKE, multigrid, line smoothing and reusable initial/recycle state. Merged [#71](https://github.com/f0uriest/yancc/pull/71)/[#79](https://github.com/f0uriest/yancc/pull/79) reduce gauge/collision work, factor storage and smoother cost; open #8/#34/#62 remain proposals. The divergent smoothers branch adds frozen-plane coupling: a candidate to inspect, not a demonstrated DKX improvement. |
| [MONKES](https://github.com/JavierEscoto/MONKES) | 1 / 0 / 1 | Pinned `4e8281c9`; spectral monoenergetic block elimination and database/transport tooling. A fresh bounds-checked local build succeeded, then a 7×9×12 W7-X smoke run failed at `DKE_BTD_Solution_Legendre.f90:105`: S1 extent 7 versus vm extent 8. This is distinct from the reader issue #1; no new numerical reference is admitted. |
| [KNOSOS](https://github.com/joseluisvelasco/KNOSOS) | 3 / 0 / 4 | Orbit-averaged transport, tangential drifts and linearized surface potential; `amb_and_qn.f90` exposes the coupled workflow. Bounds issue #3 and build/usage requirements matter for an eventual reference. At `5134a9eb`, a serial bounds-checked build linked with local NetCDF/FFTW, but a bounded LHD monoenergetic smoke run stopped at `configuration.f90:91` (`bvco_b(0)` below lower bound 1). No transport/Phi1 comparison is admitted. |
| [SOLVAX](https://github.com/uwplasma/SOLVAX) | 15 / 86 / 14 | Generated factors/refinement, recycled Krylov and implicit primitives already exist. Read #63's refresh-calibration problem and #56–58's window limitations; do not schedule their assumptions as established certificates. |

Downstream inventories cover NEOPAX (19/11/3), ESSOS (53/60/5), VMEX (48/278/3)
and NTX (9/24/1). Their open interfaces are dependencies, not delivered DKX
features. Full inventories and relevant diffs remain external to keep the repo light.

The [YANCC paper](https://arxiv.org/html/2607.20861v1) makes differentiable GPU
neoclassics an existing comparison standard. Its important lesson is the joint
choice of discretization and preconditioner, not a transferable speedup factor.
Its general surface angles also show that Boozer coordinates are not obligatory.
[MONKES](https://arxiv.org/html/2312.12248v2) supports retaining DKX's efficient
Legendre structure where the physics permits it. [KNOSOS](https://arxiv.org/abs/1908.11615)
supports a reduced model for a specified low-collisionality regime, not wholesale
replacement of multispecies full-FP transport. The [SFINCS trajectory study](https://arxiv.org/abs/1312.6058)
requires explicit finite-Er and momentum-conservation limits; agreeing reduced
models cannot validate omitted physics.

## 4. Deliverable 1: establish a trustworthy calculation

First finish the stack review and fix evidence plumbing that prevents meaningful
comparisons. The campaign must request full recovery when checking the original
state equation; it must not confuse deliberately zero-filled Legendre tails with
solver failure or certify them as complete distributions. Moment-only algorithms
can remain useful, but require their own reduced-system/observable certificate.

Use **four representative families**, adding points only to resolve a concrete
uncertainty. Each has a fast verification grid and a separately converged research
grid; a smoke grid is never silently promoted.

| Family | Required physics and measurements | Independent anchor |
| --- | --- | --- |
| Analytic/axisymmetric tokamak | PAS and multispecies full-FP; prescribed Er; particle/heat flux, parallel flow, bootstrap/conductivity; NZeta=1 versus resolved symmetry | Collision invariants, Spitzer–Härm/full-FP and applicable tokamak limits; SFINCS v3 |
| One structured stellarator and one W7-X surface | PAS/DKES monoenergetic coefficients at zero and finite Er; sign/normalization/Onsager conventions; selected thermal convolution | MONKES/YANCC in matched equations; existing Beidler-normalized fixtures |
| Multispecies stellarator profile | Full-FP, independent n/T drives, finite Er, ion/electron currents and regular root branch | SFINCS/YANCC; native SI versus expert normalized path; independent cold roots |
| A bounded hard case | Existing finite-Er current sign-changing grid ladder and warm cross-surface discrepancy, then one Phi1/drift case when that model is admitted | Original inputs from #160–161, refined referee, appropriate trajectory/Phi1 literature |

For each published quantity Q, define a physical scale and an application tolerance
`atol_Q + rtol_Q * |Q|`. Budget algebraic, grid/quadrature, root and geometry errors
separately. A reasonable initial allocation is no more than 10% of the observable
budget to linear/nonlinear algebraic error; calibrate it, rather than choosing a
universal residual tolerance. Use absolute scales for near-zero current/flux.
For a simple root, propagate current uncertainty through `|dJr/dEr|`; if the slope
is too small or the error overlaps another branch, report unresolved/marginal.

Mathematical verification includes independently derived manufactured forcing and
moments, Fourier derivative symbols, Maxwell/Gamma quadrature identities, active
layout and border identities, nonsymmetric transpose dot tests, collision number,
combined momentum/energy conservation and nullspace/gauge checks. Preserve existing
proofs rather than rewrite them to mirror the implementation. Full-FP conductivity
is a different test from the existing Lorentz `8/sqrt(pi)` result. For linear
`Q=cᵀx`, `Aᵀlambda=c` and `r=b-Ax`, the exact discrete identity is
`Q_exact-Q_computed=lambdaᵀr`; approximate adjoints need an error allowance.
This neither proves grid convergence nor identifies the cause of every discrepancy.

Refine theta, zeta, pitch, speed, Rosenbluth resolution, geometry Fourier truncation
and relevant radial/Phi1 grids both separately and jointly. Compare current and
each species' flux independently. The historical Er=15 pitch ladder changes the
sign of current; it is not fixed by a tighter Krylov residual alone. Recover the
exact #161 source/target pair before attributing its 2.46% difference to conditioning.
If it cannot be recovered, keep it unresolved and construct a new identified
adversarial pair. Do not substitute the latter as a reproduction.

Source inventory at this head: 60 production Python files / 49,405 lines;
177 test Python files / 49,504 lines; 82 tool Python files / 29,329 lines.
A separate untouched full remote clone measures 33.43 MiB allocated on this
filesystem, including 16.26 MiB `.git`, above the 20 MiB target. The tested working
checkout also contains generated files and is not used for that clone measurement.
These counts identify consolidation opportunities, not a reason to delete physics.

**Exit:** the supported toroidal domain, representative decks and per-observable
tolerances are recorded; a valid reference and a convergence ladder support each
advertised result; root branch/slope and original primal/transpose checks accompany
derivatives. The installed native and compatibility paths agree in physical units.
Failed cases are retained in the denominator. A broad all-deck sweep is unnecessary
until this smaller set produces interpretable results.

## 5. Deliverable 2: make repeated solves cheap

### An in-memory reuse contract, not another restart subsystem

Extend the existing `ErProblem`, `ErSolveState`, `solve` and SOLVAX factor/recycle
objects. The intended expert interface is a prepared, immutable physical problem
plus explicit reusable state passed in and returned from Python/JAX. The following
is a **design sketch, not an implemented API**:

```python
prepared = prepare(case, layout=fixed_layout)
result, state = evaluate(prepared, profiles, er, state=state)
value, gradient = objective_and_grad(prepared, parameters, state=state)
```

File I/O is unnecessary for optimization. Disk checkpoints are an optional later
serialization of physical state and provenance, not the first implementation.
Keep reusable arrays as bounded pytrees; do not accumulate every iteration's
factors or diagnostics in a closure. State owns its device, dtype, layout,
physical dependency identity, constraints/gauge, previous iterate, recycle space
and preconditioner metadata. Rejected optimizer trial states must not overwrite
the accepted continuation state. A geometry/profile change can preserve layout
and compilation while still requiring all affected coefficients to refresh.

| Reused item | Validity / action |
| --- | --- |
| Compiled executable, quadrature and symbolic structure | Reuse for matching shapes, dtype, static model/layout and device contract. Changing n/T/Er is not automatically a retrace; changing active pitch topology is. |
| Numerical factors | Exact solver only for the same numerical operator, border and constraints, including changed collision/geometry dependencies. RHS-only changes can share factors. Otherwise use as an **approximate preconditioner**, or refactor. |
| Distribution / Phi1 / Er | Initial guess or branch predictor, never an accepted result by identity alone. Interpolate only through an explicit geometry/grid map and recheck physical constraints. |
| Recycle basis | Recompute its image under the new original operator; reorthogonalize and discard dependent/stale vectors. A small parameter step is not a validity certificate. |
| Lagged preconditioner | Permit nearby changed systems while true residuals converge; refresh on loss of convergence, measured extra work, memory pressure or changed constraints/layout. Keep forward/transpose use explicit. |

[PETSc's successive-system rules](https://petsc.org/release/manualpages/KSP/KSPSetReusePreconditioner/)
and [maintainer explanation](https://lists.mcs.anl.gov/pipermail/petsc-users/2022-October/047018.html)
make this distinction explicit. [GCRO-DR](https://doi.org/10.1137/040607277) motivates
recycling; SOLVAX's implementation still needs calibration on DKX sequences.
Compare four ablations on exactly the same sequence: compilation only; plus x0;
plus recycle; plus lagged P. Separately test repeated RHSs with an unchanged A.
Record setup/apply/matvec/orthogonalization/adjoint work, cache rebuilds and peak
storage, including large rejected steps, branch changes and species/grid changes.

Select refresh by measured economics: if rebuilding costs `T_build`, compare it
with the expected extra iterations times `T_apply + T_matvec + T_orth` over the
remaining reuse horizon. Use a bounded diagnostic and a cold fallback, not a
universal distance threshold. At least one full-FP n/T sequence and one stellarator
Er/root sequence must agree with independent cold solves within the observable
budget. Replay forward, reverse and permuted sequences: the answer must not depend
on history. Short memory ownership tests must include failed/rejected evaluations.

### Differentiation and coupled potential

Differentiate the **converged equations**, not cache decisions or a fixed number
of unconverged iterations. For `F(u,p)=0`, the adjoint solves
`F_uᵀ lambda=Q_uᵀ`, giving `dQ/dp=Q_p-lambdaᵀF_p`. Warm guesses and lagged P can
be detached from AD while physical coefficients remain differentiable, provided
the primal and adjoint solve the intended equations accurately. This is the
[JAX custom-linear-solve contract](https://docs.jax.dev/en/latest/_autosummary/jax.lax.custom_linear_solve.html),
not permission to ignore residuals. Test JVP/VJP, two or more FD steps, quadratic
Taylor remainders above the noise floor, and cold/warm derivative agreement.
[Paul et al.'s neoclassical adjoint work](https://arxiv.org/abs/1904.06430)
already includes root acceleration; evaluate safeguarded Newton continuation
against the existing Brent search, charging every slope evaluation and retaining
bracket fallback and cold final verification. Root selection switches are not smooth.

For Phi1, the eventual state is `u=(f, Phi1, source/gauge variables)` and, when
appropriate, Er. Reuse both f and Phi1, but certify the **coupled** kinetic,
quasineutrality and gauge residual. A kinetic solve with frozen Phi1 is not the
coupled derivative. Start with linearized quasineutrality and a block/Schur
preconditioner; qualify nonlinear Newton/line-search behavior and potential gauge
before native promotion. Reuse the existing compatibility implementation first.
[PETSc SNES lagging](https://petsc.org/release/manualpages/SNES/SNESSetLagPreconditioner/)
and [Eisenstat–Walker forcing](https://users.wpi.edu/~walker/Papers/forcing_terms%2CSISC_17%2C1996%2C16-32.pdf)
suggest avoiding oversolved early Newton steps. Use existing SOLVAX support where
available; tighten terminal primal/adjoint accuracy to the observable budget.
This coupled extension follows the no-Phi1 reuse contract, not a parallel rewrite.

### Preconditioners and MUMPS: adopt mechanisms, measure before implementing

DKX owns the physics approximation P and its constraints; SOLVAX owns elimination,
Krylov, refinement, recycling and reusable parallel algebra. The expensive
full-FP route drops species/speed and some trajectory couplings in P. At difficult
parameters this can require many iterations; dense angular blocks also make P
expensive to store and apply. Both hypotheses are measurable, not universal causes.

1. **First retain the implemented factor sharing**, then measure triangular solves,
   border/RHS batching, callbacks and transfers at the complete objective level.
   Avoid reconstructing a factor in every transpose or correction. Consider a
   narrower host diagnostic transfer only if failure information remains complete.
2. **Compare the same A and the same simplified P with PETSc/MUMPS/SuperLU_DIST.**
   Record ordering, nnz(A/P), nnz(L+U), scaling, pivot tolerance, symbolic analysis,
   numeric factorization, forward/transpose solve and multiple RHS cost. Include
   aggregate rank memory and startup. Do not compare factoring P with factoring A,
   or PETSc solve-only time with DKX compilation plus setup. Reuse symbolic structure
   separately from numeric factors. Repair the reference build when required.
3. **Choose one bounded improvement from the measured bottleneck.** If application
   dominates, test compact/batched factor application. If iteration growth dominates,
   retain the missing speed/species coupling in a coarse correction or test a
   line/block smoother on a coarse discretization. Preserve the fine physical
   operator and its nullspace; a coarse upwind P need not replace the fine scheme.
   Stop the experiment if setup-inclusive scan/gradient cost and accepted memory
   do not improve. No automatic escalation to a second discretization project.

The [MUMPS 5.9.1 guide](https://mumps-solver.org/doc/userguide_5.9.1.pdf) describes
separate analysis/factor/solve phases, pivoting, refinement and optional block
low-rank compression. [SuperLU_DIST](https://github.com/xiaoyeli/superlu_dist)
uses distributed supernodal methods with build-dependent accelerator support.
These are substantial sparse solver infrastructures. The fresh 654-row full-FP dump illustrates the distinction:
with the same COLAMD/SuperLU settings, simplified P has 10,070 nonzeros and
42,634 factor entries; the original A has 12,995 nonzeros and 199,337 factor
entries. Natural ordering increases the latter to 321,174. These are storage
counts on one identical layout, not MUMPS/DKX timings or an accuracy certificate.
A separate uniform-layout dump is excluded from this comparison.

Reimplementing MUMPS in
DKX would duplicate years of work and mix algorithms into the physics layer.
Use these backends as references and, if useful, optional SOLVAX integrations;
a CPU reference need not become a GPU dependency. BLR/mixed precision are deferred
until kinetic factor ranks and refinement convergence justify them. Elliptic-PDE
compression results do not establish compressibility of these nonsymmetric matrices.

### Measure the work users actually pay for

Time preparation → all kinetic/root/Newton evaluations → moments → backward pass
→ acceptance, synchronizing arrays **and diagnostic effects**. Also separate process
startup, empty-cache compile, persistent-cache load, warm solve and reuse modes.
Use at least five unprofiled repetitions/pairs after checking stable load; retain
samples, median and dispersion. Include failed solves, line searches and rebuilds.
Peak RSS, aggregate MPI memory, allocator peak VRAM and compiler temporary estimates
are separate quantities. Do not sum nested PETSc events or overlapping GPU intervals.

Follow [JAX profiling guidance](https://docs.jax.dev/en/latest/profiling.html):
coarse named regions first, then XPlane/Perfetto and HLO/XLA/kernel inspection of the
identified bottleneck. Check trace completeness/event caps; do not infer occupancy
or speed from a capped trace or HLO operation count. Keep TensorBoard/XProf/Perfetto
artifacts outside Git. Profiled runs diagnose; unprofiled runs establish runtime.

CPU policy compares thread counts and a small process pool with controlled BLAS/XLA
threads and per-process memory. Logical JAX CPU devices share resources; they are
not extra CPUs. GPU policy compares serial continuation against independent batches
on one/two physical A4000s, including compile and transfers, uneven sizes, gradient
placement and failure propagation. Nearby points may benefit more from serial warm
reuse than simultaneous cold solves; group related points into resident sequences
only if that measured tradeoff wins. Use strong/weak scaling for independent work;
state partitioning and multi-host collectives are deferred.

**Exit:** an installed native scan and a complete value/gradient/root workload
show a reproducible benefit at fixed accepted accuracy. As a decision target,
seek ≥20% end-to-end improvement or ≥2× lower measured peak memory on the identified
bottleneck, with no loss of admitted cases; otherwise keep the simpler baseline.
Targets are not promised results. Demonstrate cold fallback, bounded memory and
no unintended recompilation for supported parameter updates on CPU and GPU.

## 6. Coordinate choices and the actual optimization deliverable

A coordinate change can simplify streaming or geometry preparation, but it also
changes grids, Jacobians, collision representation and boundary conditions. It is
not an algebraic cure for missing physics or an unresolved trapped/passing layer.

| Option | Potential benefit | Cost / decision |
| --- | --- | --- |
| Existing Boozer/general surface-angle geometry with Legendre pitch | Preserves working block structure and compatibility; existing VMEC geometry avoids requiring every input to be Boozer transformed | **Default.** Audit the geometry tensor/weight contract and its derivatives. |
| Direct VMEX/VMEC or DESC surface angles | Could avoid an expensive coordinate transform and its derivative in optimization | Use the existing general-geometry path where valid. Compare the same equilibrium, surface moments and shape derivative in two representations before choosing the adapter. No new solver backend is required merely to change the input representation. |
| Field-aligned `(alpha,l)` | Simplifies parallel streaming, may aid a line preconditioner | Global toroidal periodicity, rational surfaces and cross-field ExB coupling remain. Defer a solver rewrite; reconsider only if measured angular coupling dominates after P improvements. |
| Pitch angle `alpha=acos(xi)` with finite differences | YANCC-style regular endpoint handling and line smoothing | Loses the present simple Legendre collision/block structure and introduces new resolution/error tradeoffs. At most test it in P if justified; defer replacement of the fine operator. |
| Bounce coordinates / orbit averaging | Removes a fast coordinate for selected low-collisionality objectives | Model/order restrictions, well creation/merging and singular quadrature require separate evidence. Use an external reduced objective for screening if useful, then verify with DKX. Do not claim full-FP/Phi1 parity from it. |

The [differentiable bounce-averaging study](https://arxiv.org/html/2412.01724v2)
demonstrates that useful reduced objectives can be differentiated; it does not
make a bounce-averaged model equivalent to DKX's full local problem. The
[NEO-2 bootstrap-limit study](https://arxiv.org/abs/2407.21599) also cautions against
using a universal low-collisionality asymptote without the stated precession and
ripple conditions. Both support a validity-based choice, not more active branches.

Deliver the real optimization in three steps within one existing example family:

1. Use a verified fixed equilibrium, explicit n/T profiles and prescribed Er.
   Validate profile and geometry derivatives independently, including radial
   coordinate, Fourier truncation and metric/Jacobian derivatives. Extend the
   adapter already used by VMEX; pin the actual dependency and equilibrium residual.
2. Optimize a small set of **boundary coefficients** through the equilibrium solve
   and DKX transport/current objective with explicit aspect-ratio, iota, field and
   geometric feasibility constraints. Report the full cost per accepted step,
   rejected steps and compilation. Compare AD with finite differences over several
   parameter counts at equal error; a harmonic field-amplitude descent is insufficient.
3. Add regular-branch stellarator ambipolar response, then self-consistent bootstrap
   current/equilibrium coupling. Differentiate the coupled fixed point or converge
   and validate its implicit Jacobian; freezing the equilibrium current omits part
   of the derivative. Keep a prescribed-Er tokamak example. Use
   [direct neoclassical optimization](https://arxiv.org/abs/2406.04147) and
   [bootstrap-consistent equilibrium optimization](https://arxiv.org/abs/2205.02914)
   as comparison designs, not claims that DKX already reproduces them.

**Exit:** objective improvement exceeds its numerical uncertainty, constraints are
satisfied, full-chain Taylor/FD tests pass on smooth branches, and the final design
is recomputed cold at finer resolution with an independent transport/current
reference. Charge the equilibrium and coordinate transformation to the timing.
Stop and narrow the parameter/physics domain if the derivative or model is invalid;
do not silently freeze it to preserve a descending objective.

NEOPAX integration then needs a small in-memory protocol for species order,
radial centers/faces, SI fluxes and Jacobians, boundary conditions, validity and
refresh. Check transport conservation and lagged-response error before claiming a
transport simulation. NTX is a candidate monoenergetic database producer; avoid a
second database framework until its normalization/interpolation/restart contract
is compared with DKX's existing one. ESSOS first realizes coils for an accepted
target with field-error, length, curvature, distance and current constraints.
Arbitrary coil fields need not possess nested surfaces. Joint plasma/coils and
open-field-line mirror optimization follow their own physical validity gates.

## 7. Documentation, examples and deliberate reduction

The README should contain one runnable start, a short feature/results summary and
an easy-to-advanced workflow map. Remove categorical SFINCS/AD/GPU claims and
multiple historical timing narratives. Keep scope beside each result. The docs
landing page should route readers by task; remove its duplicate performance table
and universal “runs in seconds” claim. Preserve existing URLs while consolidating.

| Existing material | Canonical destination / action |
| --- | --- |
| `installation`, `examples`, first-run parts of `usage` | Tutorials: first physical result, convergence, gradients; examples remain executable sources. |
| `case_files`, `applications`, `optimization`, `vmex_workflow`, `parallelism`, troubleshooting | How-to: one guide per user task; combine overlapping optimization instructions after the real adapter exists. |
| `physics_models`, `system_equations`, `physics_reference`, `theory_from_upstream`, `method`, `numerics` | Explanation: one model/units/constraints derivation and one numerical-method explanation; keep independent citations and applicability. |
| `api`, `cli`, `inputs`, `outputs`, `normalizations`, `capabilities`, `feature_matrix` | Reference: schemas/status generated from the existing definitions; link instead of copying a second capability matrix. |
| `performance`, `validation_matrix`, `parity`, `fortran_comparison`, `research_lanes` | Evidence: accepted results versus historical/experimental records; archive a superseded experiment, do not present it as a second roadmap. |

Reuse the nine numbered examples. Keep 01–03 for analytic/VMEC/Boozer profiles,
04 for the documented monoenergetic path, 05 for root evidence, 06 for convergence,
07 for gradients, 08 explicitly labeled geometry proxy until the real boundary
example replaces it, and 09 for qualified Phi1/impurity comparisons. Add warm/batch
options to the relevant examples and guides rather than another numbered gallery.
Each needs editable profiles/geometry/resolution, units, model scope, expected
qualitative result, a quick mode and a checksummed research case with measured
resource requirements. Do not imply all nine support native Case execution.

Students should reach a plot and a readable physical summary in one command, then
see why a small residual is not grid convergence. Researchers should be able to
replace equilibrium/profiles and run convergence, repeated solves and gradients
without copying internal code. Examples must expose rejected points, root scope,
error bars and SI conventions, not connect invalid entries as real data.

Audit duplication in `solve.py`, `coarse_precond.py`, `multigrid.py`, geometry and
workflow orchestration before creating helpers. Keep one physics assembly and one
solver policy; generic new algorithms belong in SOLVAX. Remove obsolete paths only
after preserving meaningful assertions, not by deleting difficult tests. Report
source/test/tool/doc file counts, physical lines, tracked bytes, fresh-clone size,
wheel and installed-owned size separately. Dependencies still count in user setup
cost. Preserve the <20 MiB owned-artifact/fresh-clone targets and soft 45k production
line target as visible debt where missed, without forcing unrelated modules into
one file or silently changing measurement definitions. No history rewrite is planned.

The 95% line/branch goal remains a ratchet for stable reachable code, not a reason
to manufacture tests or postpone an urgent scientific correction. Run focused
mathematical/physics tests per change, installed examples and warning-clean docs;
reserve large external/GPU campaigns for relevant changes. Keep compact inputs,
checksums, commands and result summaries in Git; raw states, traces and build trees
belong in an archive. Verify retrieval before pruning an artifact that underpins
an advertised result.

## 8. Deferred work, publications, and the next PRs

Defer a new fine-grid coordinate/discretization backend; MUMPS reimplementation;
BLR/mixed precision without measured kinetic evidence; learned/Nyström/PINN
preconditioners; unqualified truncated-adjoint windows; multi-host/state-decomposed
execution; new database frameworks; joint coils/plasma optimization; and mirrors.
Existing useful experimental APIs can remain clearly labeled. A deferred item
reopens only with a named user calculation the active deliverables cannot serve,
a bounded experiment and a measurable decision. Native Phi1/full drifts are
scientific completeness work after the corresponding coupled/error contracts,
not abandoned physics or an excuse to claim SFINCS parity early.

The next implementation PRs should be concrete and sequential:

1. **Integrate and qualify the baseline:** reconcile the existing PRs, close the
   root-timing scope defect, request appropriate full states in the benchmark
   runner, archive valid reference inputs/toolchains, and publish the four-family
   observable/error table. Avoid a full campaign while its pairs remain invalid.
2. **Expose safe reuse through existing prepared objects:** first unchanged A and
   changing RHS, then full-FP n/T and regular Er continuation; cold/warm values,
   derivatives, rejected-trial ownership, invalidation and memory are its tests.
3. **Optimize one measured bottleneck:** choose factor application or a stronger
   physics P from the ablations; include the complete scan/gradient benchmark,
   CPU thread/process policy and one/two-GPU results. Stop if it does not pay off.
4. **Replace the optimization proxy with the real dependency chain:** prescribed
   Er first, root and bootstrap coupling only after their respective checks.
   Improve the existing guides/examples in each PR, not in a final documentation dump.

Estimate implementation effort only after the first baseline identifies reference,
physics and dependency blockers. Do not promise a date or spend unlimited runtime
recovering one historical input. Bound attempts and retain failure information;
a smaller honest supported domain is more useful than a nominally complete matrix.

Prepare **one methods/software paper** from deliverables 1–2: stated equations,
independent mathematics/physics benchmarks, full-FP versus reduced-model scope,
error-controlled derivatives, warm reuse, CPU/GPU throughput, whole-workflow cost,
memory and failures. Differentiable GPU neoclassics alone is no longer a novelty
claim. Its strongest possible contribution is reliable reuse and measured design
throughput at accepted physical accuracy. A separate application paper is warranted
only when deliverable 3 yields a scientifically interesting independently validated
design, not merely because another example exists.

**No new release until the open PRs are merged or explicitly resolved and the most
important supported-scope goals above are achieved.** Then verify the exact installed
wheel/sdist/dependencies, all required checks/reviews, documentation commands,
scientific envelopes and archived figure provenance. Report remaining experimental
capabilities honestly. All commits use author and committer `rogeriojorge`; keep
user work and unrelated repositories intact. Completion updates this decision plan,
existing capability/evidence registries and canonical docs together; it does not
append another chronological execution diary.
