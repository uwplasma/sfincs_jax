# DKX research plan

**Authoritative plan, written 2026-09-06 by reconciling three predecessors: the R0–R11 work queue (`be6506fe`), [#189](https://github.com/uwplasma/DKX/pull/189) and [#190](https://github.com/uwplasma/DKX/pull/190). It is the only planning file in the repository, by governance test. Planning only; nothing here authorizes a release.**

## 0. Start here

This document is written so that an independent agent can pick it up cold and know what to do on Monday. Read section 0, then the phase you are assigned in section 4, then the working method in section 7. Everything else is reference.

**What DKX is.** A JAX reimplementation of the SFINCS Fortran v3 drift-kinetic neoclassical transport code: radially local, linearized drift-kinetic equation on a flux surface; fluxes, flows, bootstrap current, transport matrices and ambipolar radial electric fields; pitch-angle-scattering and full linearized Fokker–Planck collisions; Phi1; analytic, VMEC and Boozer geometry; SFINCS deck and HDF5 compatibility; differentiable end to end; CPU and GPU. Physics and solver policy live in `src/dkx`; reusable linear algebra, Krylov, factorization, recycling and implicit-differentiation primitives live in the sibling library [SOLVAX](https://github.com/uwplasma/SOLVAX). The closest external codes are SFINCS (the model), yancc (differentiable GPU full-DKE, no Phi1, no ambipolar root), MONKES and NTX (monoenergetic), and KNOSOS (bounce-averaged).

**Where things are.** `src/dkx/solve.py` owns the three linear-solve routes and the `method: auto` policy; `coarse_precond.py`, `multigrid.py` and `sparse_precond.py` are three inverses of one SFINCS-simplified operator used as the Krylov preconditioner; `er.py` owns the ambipolar root, differentiated by the implicit function theorem with original-equation admission; `batch.py` owns sharded independent batches; `api.py` exposes `prepare_er_scan`, `batched_er_scan` and `ErProblem.with_profiles`; `tools/benchmarks/parity_performance_matrix.py` is the supervised, resumable, verifiable benchmark runner; `validation/` holds the sealed cross-code artifacts and `baseline.toml`; `docs/` is Sphinx built with `-W`; `examples/01`–`09` is the example ladder. CI is thirteen coverage shards balanced by `.test_durations`, a wheel-install job that measures the size contract, a docs job, and a `commit-trailers` gate: every commit is authored by the maintainer alone.

**How to run things.**

```bash
pip install -e .                                                  # from a checkout; Python >= 3.11
JAX_ENABLE_X64=True DKX_CI=1 pytest -q -n 4 -m "not slow"         # the PR gate, about 10 min on a laptop
python -m sphinx -b html -W docs _build                            # docs must build warning-free
dkx run examples/01_tokamak_profile/case.toml --out r.nc && dkx converge examples/01_tokamak_profile/case.toml
```

**How progress is recorded.** This file holds phases, figures and criteria and never status prose. Decisions go to `docs/adr/NNNN-*.md`, one page each, immutable. Time-boxed experiments go to `docs/experiments/` as one-page records. What shipped goes to `CHANGELOG.md` per release. Execution detail lives in PR descriptions. Review reports are dated files under `docs/reviews/`. Section 9 records how this plan came to be and where its predecessors are.

**What to do first.** Phase 0 in section 4 is a checklist with commands.

**What never to do.** Do not silently change collisions, trajectories, precision, resolution, root-search scope or requested parallel execution. Do not call an approximation a full model. Do not promote a smoke grid to a research result. Do not compare DKX solve-only time with SFINCS whole-process time. Do not report a warm number without its cold companion. Do not open a stack of PRs without the tooling in section 7. Do not add evidence or provenance infrastructure until a figure or a reviewer asks for it. Do not begin a second algorithm experiment before the first has a written decision.

## 1. Mission, retained decisions, thesis

**Mission.** Deliver a research-grade, differentiable neoclassical code for stellarators and tokamaks with the scientifically supported functionality of SFINCS Fortran v3, reliable CPU/GPU execution, and derivatives that are verified rather than claimed. The flagship calculation is: specified toroidal equilibrium and species profiles → resolved transport and bootstrap current → a retained stellarator ambipolar branch → checked derivatives → fast repeated evaluations → one actual equilibrium-boundary optimization. For tokamaks `E_r` is prescribed; an intrinsically ambipolar local model cannot determine it. Density and temperature stay explicit; pressure alone fixes neither.

| Retained decision | Contract |
| --- | --- |
| Native interface | Immutable `Case`, TOML first and equivalent JSON; `Result` with versioned NetCDF. Physical units are normalized once. Source-relative paths and deterministic semantic IDs. |
| Compatibility | Permanent SFINCS namelist/HDF5 adapters, with explicit unsupported controls; DKX 3 may change the DKX 2 Python API. |
| Expert interface | Typed geometry, grids, operator, moments, prepared solve, and sensitivity contracts for composition with other codes. File I/O and Python orchestration need not be differentiable. |
| Runtime | Python ≥3.11, `src/dkx`, explicit runtime configuration; CLI uses argparse/Rich. Core emits structured progress and diagnostics. |
| Documentation | Sphinx/MyST/Furo; tutorials, how-to guides, explanation, reference. One source for schemas and capability status. |
| Algorithms | SOLVAX owns reusable linear algebra, differentiation primitives, and generic parallel primitives. DKX owns physics, discretization, and physics-dependent solver policy. |
| Platforms | Named M3 Max laptop CPU baseline and office NVIDIA accelerator lane. Performance results identify exact hardware and toolchain. |
| Size | Fresh full clone, wheel, sdist, and installed DKX-owned files each target <20 MiB. Report each measurement separately; dependencies are excluded from owned-file size, not from installation instructions. Measured 2026-09-01 on a fresh clone: 14.49 MiB tracked tree (3.07 MiB media), 14.83 MiB object store, 29.32 MiB total. This leaves about 5.5 MiB for Git metadata/history under the existing target; it does not prove that a rewrite could or could not achieve it. Measure reachable packed objects before considering another rewrite. Keep the <20 MiB target; do not silently restate it to make a gate pass. |
| Quality | 95% line **and** branch coverage of stable reachable code is the release target. Maintainer-deferred coverage work must not block urgent correctness fixes. Assertions must test behavior or science. |
| Workflow | One coherent implementation slice per PR; commits authored and committed by `rogeriojorge`, without assistant coauthor trailers. Preserve uncommitted work. |

Do not silently change collisions, trajectories, precision, resolution, root-search scope, or requested parallel execution. Do not call an approximation a full model. New evidence may change a policy, but must identify the affected contract and replace the superseded claim.

**Thesis.** The unit of progress is a figure, not a gate: each phase in section 4 is defined by a publishable figure or table, and code, tooling and documentation are written when a figure needs them. The differentiator is not "differentiable GPU neoclassics", which yancc has published; it is the complete SFINCS-v3 model including Phi1 and the ambipolar root, on GPU, with an error bar on every reported observable and derivatives verified through the root and through geometry. The algorithmic work is a diagnosis first and two bounded, kill-gated experiments second, because DKX already implements SFINCS's simplified-operator preconditioner as its default and the reasons the Krylov route still loses are specific and measurable.

## 2. State of the code and the evidence, 2026-09-06

### 2.1 What landed with the implementation stack (#169–#188)

| Work to retain | Source / review | Remaining boundary |
| --- | --- | --- |
| Supervised, resumable benchmark attempts; original residual and complete-output checks; PETSc backend/provenance verification | #169–170, #176–177, #181, #183, #185; `tools/benchmarks/parity_performance_matrix.py` | Fresh installed production replay, valid references and observable/grid admission are still needed. |
| Static operator layouts, refreshed collision coefficients and native profile preparation | #173–174, #178, #182, #186–187; `drift_kinetic.py`, `collisions.py`, `execution.py`, `er.py` | Profile updates are opt-in and hold geometry, normalization, species, Coulomb log and discrete layout fixed. They do not make `Case.run` a JAX transformation. |
| Independent batch sharding through JIT and gradients, uneven-batch handling and per-input algebraic status | #179, #182, #187–188; `batch.py`, `api.py` | One whole system remains on one device. Native Case execution does not expose every expert batch option; memory budgets are estimates. |
| Original kinetic and transpose checks; qualified dense adjoint references; native profile-to-root Taylor tests | #180, #184, #188; `solve.py`, `er.py`, solver/root tests | Small fixed-grid evidence does not establish resolution, branch or geometry uncertainty. Partial state recovery has a different derivative contract. |
| Generated Schur-factor reuse across RHSs, forward/transpose solves and refinement | #188 `be6506fe`; existing SOLVAX generated-factor API | Factors belong to one solve execution. Persistent reuse across changed operators is not implemented by this change. Full-FP uses the Krylov route. |

SOLVAX `>=0.19.0` is sufficient for everything the stack uses; requalify 0.20.x explicitly rather than by branch. `examples/08_vmex_optimization` is an analytic Boozer-spectrum proxy that does not import VMEX; real VMEX calls exist only in optional scripts under `examples/optimization` and `examples/autodiff`.

### 2.2 Useful results, with their limits

The #188 report records 244 distinct CPU cases and 22 GPU cases for its factor
integration, and native root Taylor orders of 2.00–2.01. On one **7,850-unknown,
three-field PAS objective**, twelve alternating synchronized pairs reduced warm
value/gradient medians from **67.59 to 37.09 ms on CPU** and **221.74 to 195.56 ms
on A4000**; GPU LU calls fell from 288 to 144. CPU used M3 Max/JAX 0.9.2; GPU used
JAX 0.10.2. These compare two checked routes
on each host. They are not full-FP, fresh compilation, allocator peak, whole-root,
whole-optimization or cross-host scaling results. The underlying external traces
were reported in the prior review; the 2026-09-06 review did not relabel them as new runs.

Earlier README claims about a 744k HSX case and a broad upstream speed ranking
are historical, with differing runtime/memory definitions and invalidated
reference campaigns elsewhere in the record. Retain their evidence in the
performance documentation; do not use them as an unqualified headline. In
particular, small algebraic residuals and agreement with a failed Fortran run
cannot certify a transport observable.

The 2026-09-06 review evidence is outside Git in `dkx-plan-evidence-20260906`, with exact commands,
inputs, source identities and logs. Its role is to choose
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

| Check on the reviewed source (2026-09-06) | Result / limit |
| --- | --- |
| DKX solver, Er, batch, native execution and planning suites | 248 passed on M3 Max CPU/JAX 0.9.2/SOLVAX 0.20.0; not a full coverage campaign. |
| Office GPU selection | Four targeted batch/AD/full-state cases passed. The unchanged two-device harness also passed PAS/full-FP states, original residuals, uneven batches, actual placement, JIT gradients and an FD check on two A4000s, JAX 0.10.2. |
| YANCC at `6f399a21` | 59 preconditioner/solve/collision tests passed locally, including its SFINCS/MONKES fixture comparisons, coordinate representations, warm state and derivative tests. This does not establish comparative runtime or general warm-reuse correctness. |
| README examples | Native Python and CLI run/inspect pass; Python including the advanced JIT gradient also passes from an isolated DKX install inheriting host dependencies. |
| Existing independent comparison artifact | All three coefficient rows and external YANCC input hashes pass the offline audit; this is an artifact audit, not three fresh kinetic benchmarks. |
| Documentation | Five planning/wording checks and standard Sphinx `-W` pass. Extra `-n` reference checking finds 219 warnings in both baseline and revised docs; resolve this existing API-link debt during consolidation. |

Facts established by the 2026-09-06 code survey that earlier plans got wrong or missed: the SFINCS-style simplified-operator preconditioner already exists and is DKX's default Krylov preconditioner (`coarse_precond.py:916`: self-species, x-diagonal collisions; L±2 Er and drift terms dropped; exact block-Thomas inverse over (species, x); Schur-eliminated border; Phi1-aware), and `multigrid.py` and `sparse_precond.py` are alternate inverses of the same operator that the escalation ladder already tries in order (`solve.py:1072–1248`). Of the six upstream decks that did not complete, five were killed by memory while the coarse preconditioner allocated its dense (N_θN_ζ)² bands (42.9 GB on `filteredW7XNetCDF_2species_magneticDrifts_noEr`) and one is a LIBSTELL `wout` reader case; memory-lean routes now exist and those decks run in 50 minutes to 2 hours 17 minutes. The ambipolar root is differentiated by the implicit function theorem (`solvax.implicit.root_solve` = `jax.lax.custom_root`, `er.py:909–1037`) with finite-slope and original-equation admission. Batched scans thread neither recycle spaces nor preconditioners across points; the root driver does when `warm_start=True`; the adjoint solve always cold-starts and has no recycle space of its own. There is no banana/plateau/Pfirsch–Schlüter flux-limit test in `tests/`.

### 2.3 Capability priorities

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

### 2.4 Position relative to yancc, MONKES, NTX and KNOSOS

A research-grade plan states what the code is expected to demonstrate that its neighbours do not, and where it must not overclaim. Reported numbers below are the other codes' own; none has been reproduced here.

| Code | Model and method | Reported | What DKX must show against it |
| --- | --- | --- | --- |
| [yancc](https://arxiv.org/abs/2607.20861) | Full 4D DKE, linearized FP with field-particle terms and speed/pitch electric-field terms; tangential magnetic-drift support requires a separate audit; `E_r` is an input, no Phi1 self-consistency; finite differences with a diagonally dominant upwind stencil, Maxwell collocation in speed, semi-coarsened multigrid V-cycle preconditioning GCROT; JAX. | The paper reports close SFINCS/MONKES agreement and roughly an order-of-magnitude per-scan speed/memory improvement. Those are author-reported comparisons, not a DKX benchmark. Audit derivative verification independently of the differentiability claim. | Certified observable error (§3.1), matched Phi1/quasineutrality scope and retained ambipolar-root evidence; verified derivatives through roots and geometry (B10). Historical DKX two-GPU batch scaling was below 1; the R4 checkpoint now records a bounded sharded PAS improvement and its structured direct route is exact only on PAS/DKES; do not claim the 10⁷-unknown GPU envelope until B12 measures it. |
| [MONKES](https://arxiv.org/abs/2312.12248), [thesis](https://arxiv.org/abs/2510.27513) | Monoenergetic DKE, Lorentz collisions, Legendre in pitch and Fourier collocation on the surface, block-tridiagonal direct elimination, O(N_ξ N_fs³). | 4–64× over DKES; about one minute per case on one core at ν̂ = 1e-5; converged with ≤180 Legendre modes and about 2,000 surface points; Onsager symmetry checked. Momentum conservation requires external correction. | The same block-tridiagonal structure DKX exploits on PAS decks; B3 must document its convergence criterion and meet DKX's observable tolerances while checking the required `N_ξ` resolution, and the [Boozer-reader off-by-one](https://github.com/JavierEscoto/MONKES/issues/1) must be excluded from any reference build. |
| [NTX](https://github.com/uwplasma/NTX) | JAX-native monoenergetic solver implementing the MONKES Legendre formulation with an embedded adjoint. | The README reports a 14× adjoint advantage over finite differences at 32 parameters. Pin the benchmark and audit gradient accuracy, normalization and timing boundaries before adoption. | The monoenergetic derivative reference for B3/B10, and a candidate owner of the monoenergetic database workflow rather than a duplicate in DKX; decide ownership explicitly in R5. |
| [KNOSOS](https://doi.org/10.1016/j.jcp.2020.109512) | Bounce-averaged, low-collisionality, radially local; includes the tangential magnetic drift and surface variation of the potential. | Fast enough for optimization loops; valid in its asymptotic regime. | Comparison only inside that regime (B5); a disagreement outside it is not a defect of either code. |
| [PENTA](https://ui.adsabs.harvard.edu/abs/2010APS..DPPTP9124L/abstract), [NEO-2](https://www.semanticscholar.org/paper/ef15bce085ab33694ac20af4ea1ce6591f3dbb0b) | Momentum-corrected transport from DKES-type coefficients; field-line-tracing full linearized collisions. | Standard references for momentum correction and parallel flows. | A momentum-conserving full-FP model should not add a DKES momentum correction; verify discrete conservation independently; these are the references for parallel-flow and bootstrap comparisons in B2/B4 and for the database-to-thermal path in B9. |

The differentiating claims are in the last column of the first row: certified observable error, the complete SFINCS-v3 model with root evidence, and verified derivatives of that model. Speed against yancc on the full DKE is not a differentiator until measured on matched grids.

Numbers from the 2026-09-06 literature survey, each traceable to a fetched source in section 10: yancc's largest reported problem is single-species NCSX at (n_x, n_α, n_θ, n_ζ) = (7, 121, 43, 65) in 6 GB on one A100, against SFINCS at (7, 141, 25, 81) on 128 cores needing over 50 GB, about 5× faster at moderate collisionality; its full text contains no finite-difference or adjoint check of any derivative, no Phi1, and lists the ambipolar `E_r` as future work. MONKES converges low-collisionality W7-X coefficients at N_ξ ≈ 140–180 with (N_θ, N_ζ) ≈ (23, 55–79) in about a minute on one core, and used 1.4 GB where yancc used 4 GB on the same monoenergetic case. NTX reports its adjoint at 14× the cost advantage over finite differences at 32 parameters with agreement to about 2e-14. Albert et al. 2024 show the `E_r = 0` off-set bootstrap current does not converge in the 1/ν regime and decays as `ν*^(3/5)` with finite `E_r`. Saxena et al. 2025 frame "how wrong is the analytic bootstrap closure off-symmetry" as the open question; DESC's own tutorial states the Redl isomorphism does not apply to non-quasisymmetric fields; Infinity Two iterates SFINCS with VMEC by hand; Stellaris is quasi-isodynamic and needs off-Redl verification.

## 3. Scientific contract and evidence hierarchy

### 3.1 SFINCS-v3 functionality to close

The Fortran source declares 145 namelist members in nine groups, inventoried by `tools/parity/output_key_coverage_report.py` at pinned commit `8df5453`. That is an inventory, not 145 validated features. A family is closed only when the source audit, the acceptance test and the documentation agree.

| Family to close | Source/API audit and acceptance requirement |
| --- | --- |
| Species and drives | Charges, masses, kinetic/adiabatic roles, density/temperature gradients, unequal temperatures, inductive drive, reference scales, Coulomb logarithm and quasineutrality assumptions. Independent density and temperature profiles are mandatory; pressure is not a substitute. |
| Collisions | Lorentz/PAS, full linearized multispecies Fokker–Planck and retained Sugama variants. Test field-particle terms, Rosenbluth potentials, conservation, Maxwellian limits, high-Z disparity and unequal-temperature domain. State where an H-theorem/equilibrium nullspace applies. |
| Trajectories | Full and DKES-like choices, compressible/incompressible electric drift, speed and pitch derivatives, `includeXDotTerm`/`includeElectricFieldTermInXiDot`, default combinations. Compare each operator term before comparing moments. |
| Magnetic drifts | Every supported `magneticDriftScheme` (including the 0–9 choices where implemented), radial/tangential terms, curvature/grad-B and electric-field conventions. Native full magnetic-drift selection is still a gap. |
| Phi1 and external distributions | Linear/nonlinear quasineutrality, kinetic species response, density/gauge constraints, external Phi1/distribution inputs, NBI conditional build behavior and temperature-equilibration limitations. Verify coupled residual/Jacobian and admissible physics. Native Phi1 is still a gap. |
| Nullspaces and sources | All retained `constraintScheme` choices, density/energy constraints and source coefficients. Demonstrate constrained rank, uniqueness and normalization; compare physical moments across equivalent constraints. |
| Full-kinetic outputs | RHSMode 1 particle/heat flux, parallel flow/current, bootstrap, classical terms, NTV/momentum diagnostics and electrostatic corrections where supported. Document heat versus energy flux and every physical-unit conversion. |
| Transport/database modes | RHSMode 2/3 matrices, monoenergetic coefficients, field/collisionality/speed scans, thermal convolution, interpolation validity and sign conventions. Compatibility exists; native workflow/Result admission remains incomplete. |
| Ambipolarity | Charge-weighted total radial flux, electric-field coordinate conversion, bracketing/refinement, all retained roots, slope/stability convention, continuation and selection. Compare with SFINCS root utilities and PENTA in matching models. |
| Sensitivities | Supported RHSMode 4/5 and adjoint controls: distinguish actual solves from input validation. Compare primal/adjoint equations, gradients of moments, geometry and profile parameters, and nonlinear Phi1/root derivatives where differentiable. |
| Geometry | Analytic and file-backed geometry schemes 1–5, external VMEC/Boozer families 11–13, asymmetric equilibria, signs/orientation, radial interpolation, field periods and Fourier truncation. Match source domains rather than treating equal integer settings as equal grids. |
| Resolution | `Ntheta`, `Nzeta`, `Nxi`, `Nx`, `NL`, speed maximum/grid scheme, interpolation/quadrature, active pitch layout, boundary conditions and Rosenbluth resolution. Joint refinement must close angular/pitch/speed coupling. |
| Execution and export | Iterative/direct routes, tolerances, preconditioners, nonlinear controls, all export fields, dump formats and restart semantics. Compare complete successful outputs; never accept a partial HDF5 simply because it opens. |
| Research workflows | Surface/profile scans, finite-Er databases, impurity transport, full-FP ambipolar calculations, bootstrap/equilibrium iteration and differentiable objectives. Document model validity and tested parameter envelope for each. |

### 3.2 Evidence hierarchy and certificates

Separate four questions: (1) does the code implement the stated equations, (2) are those equations solved accurately, (3) is the model appropriate and independently supported, and (4) does the workflow finish within its resource budget? Cross-code agreement answers none of these alone when both codes share a discretization error or incompatible normalization.

* **Mathematics/code verification:** manufactured distributions and forcing with analytic moments; polynomial/Gamma-function quadrature identities; periodic Fourier derivative symbols; active-grid indexing; adjoint dot products; block extraction/reconstruction and dense nonsymmetric referees; collision invariants and operator limits. Derive expected values independently of production helpers. Perturb a coefficient/sign/weight to confirm that important proofs fail.
* **Algebraic error:** retain original-system `r = b - A x`, absolute and relative norms and normwise/componentwise backward error. Check constraints separately. Establish rank and gauge before interpreting a condition estimate. A tiny global residual does not bound every small flux.
* **Observable error:** for a linear observable `Q = cᵀx`, solve `Aᵀλ = c`; `λᵀr` estimates the algebraic error, with adjoint error accounted for. Nonlinear observables need a linearization remainder or conservative refinement evidence. Do not call this a rigorous bound without its assumptions. Equilibration may improve numerical scaling; certification uses the original physical operator and units.
* **Numerical uncertainty:** refine each axis and jointly refine angular/pitch/speed/potential grids, Fourier truncation, radial interpolation, solver tolerances, and nonlinear/root tolerances. Track every published observable. Near-zero quantities use physically motivated absolute tolerances in their own units plus relative tolerances; never normalize current by a heat flux or discard an inconvenient species using another species' largest moment.
* **Model validation:** specify local ordering, collisionality, orbit-width and electric-field assumptions, collision model, magnetic drifts, Phi1, geometry regularity and boundary conditions. Model uncertainty is not a mesh error bar.
* **Differentiation:** JVP/VJP dot products, analytic sensitivities, central differences over a step-size window, and Taylor remainder rates for geometry, collisions, profiles, field and coupled objectives. Report primal and adjoint residuals, root branch, setup/gradient cost and peak memory. Stopping/reuse heuristics must not silently change the differentiated equations.

Existing `test_math.py`, `test_numerics.py`, `test_collision_physics_gates.py`, `test_transport_limits.py`, `test_shaing_callen.py` and solver/transport tests already close substantial proof work. In particular the thermal Lorentz coefficient `8/sqrt(pi)` in DKX normalization and manufactured Gamma-function thermal convolutions are proved; do not reopen them under a different normalization. Full-FP Spitzer–Härm and multispecies transport limits remain distinct from that Lorentz result. Onsager/Onsager–Casimir relations require the appropriate trajectory model, thermodynamic forces, magnetic-field reversal and sign conventions.

For a simple ambipolar root, differentiate `J_r(E_r,p)=0` using `dE_r/dp = -(∂J_r/∂p)/(∂J_r/∂E_r)`. A zero/uncertain slope is **marginal**, not stable; branch creation, tangency and selection switches are nonsmooth events. Sign samples cannot exclude even crossings or tangencies between samples. Axisymmetric local momentum-conserving neoclassical theory is intrinsically ambipolar: it does not select a unique tokamak Er. Use a prescribed field or an explicitly additional closure in that application.

Tier vocabulary, used everywhere: **Tier A code verification** (analytic limits, Onsager symmetry, adjoint versus finite difference); **Tier B solution verification** (convergence per axis with the criterion stated); **Tier C cross-code** (SFINCS parity with the same discretization is regression and is reported separately from MONKES/yancc independence at about 1 percent); **Tier D validation** (W7-X `E_r` against experiment with its uncertainty). Tier C is never called validation.

**Per-observable error budgets.** For each published quantity `Q`, define a physical scale and an application tolerance `atol_Q + rtol_Q |Q|`, and budget algebraic, grid/quadrature, root and geometry errors separately; start with at most 10 percent of the budget for algebraic error and calibrate rather than choosing a universal residual tolerance. For a linear observable `Q = cᵀx` with adjoint `Aᵀλ = c` and residual `r = b − Ax`, the exact discrete identity `Q_exact − Q_computed = λᵀr` gives the algebraic error bar at the cost of one transpose solve; approximate adjoints need an allowance. Richardson extrapolation over the convergence ladder gives the discretization bar. Use absolute scales for near-zero currents and fluxes. For a simple root, propagate current uncertainty through `|dJ_r/dE_r|`; a slope too small or an error overlapping another branch is reported as marginal, never as stable.

### 3.3 Benchmark families

Every family records pinned inputs, independent derivation/reference, model/normalization match, converged observable targets, tolerances with rationale, valid parameter range, CPU/GPU resource envelopes, and failure outcomes. Tiny smoke decks are not publication benchmarks. Extend the existing registry/runner instead of creating a script and JSON for each sweep point.

| ID | Case family | Evidence and use |
| --- | --- | --- |
| B0 | Manufactured operators/moments, grids and constrained linear systems | Analytic proof tests, backward error, conservation, derivatives; ordinary CI. |
| B1 | Uniform field, Lorentz conductivity, full-FP conductivity, high-collisionality limits | Independent normalized derivations; distinguish established Lorentz proofs from additional Spitzer–Härm and general-geometry transport work. |
| B2 | Large-aspect-ratio circular tokamak across collisionality | Banana/plateau/Pfirsch–Schlüter behavior, Shaing–Callen limits, flow/bootstrap and intrinsic ambipolarity; SFINCS plus appropriate analytic theory. Convergence toward the Shaing–Callen limit is slow and resolution-sensitive ([arXiv:2407.21599](https://arxiv.org/abs/2407.21599)); record the approach, not one point. |
| B3 | DSHAPE/NCSX/W7-X monoenergetic, finite Er | Existing MONKES/YANCC comparisons, matched DKES-like equations and Beidler conventions; extend resolution/collisionality coverage and compare full tables. Add [NTX](https://github.com/uwplasma/NTX), the JAX Legendre/block-tridiagonal monoenergetic solver with an embedded adjoint, as the sibling reference for monoenergetic derivatives. The external bar is yancc's reported agreement with MONKES within 1% across collisionality ([arXiv:2607.20861](https://arxiv.org/abs/2607.20861)). |
| B4 | LHD/HSX/W7-X full kinetic multispecies | SFINCS, impurities/high-Z/unequal temperatures, particle/heat/current outputs; finite-Er and collision terms separated. Include a direct yancc comparison on identical grids with a stated tolerance: it is open source, reports <1% against SFINCS on NCSX with about 5% on currents, and is the closest competitor (section 3.3). |
| B5 | Finite-Er trajectory and tangential-drift variants | SFINCS operator parity and converged moments; KNOSOS only inside its bounce-averaged asymptotic domain. |
| B6 | Linear/nonlinear Phi1 and impurity response | Quasineutrality/gauge proofs, full coupled residual and adjoint, independent SFINCS reference. |
| B7 | W7-X ambipolar profile and root events | Current uncertainty, seeded and discovery scopes, branch continuation, failed-point handling; broad-search completeness claimed only with an actual exclusion argument. |
| B8 | VMEC/Boozer conversion and asymmetric tokamak/stellarator | Coordinate/normalization and Fourier/radial errors; independently converged bootstrap rather than a tiny unresolved current. |
| B9 | Database → thermal response | Existing analytic convolution, withheld table points, interpolation/edge refusal and full-kinetic comparison under matched assumptions. |
| B10 | Linear, nonlinear and root sensitivities | Analytic/JVP/VJP/Taylor/finite-difference windows, SFINCS adjoints where supported, nonsmooth-event refusal. The gate is a derivative **through the ambipolar root and through geometry**, not a smooth PAS temperature derivative on a tiny deck: report the Taylor-remainder rate and the FD window for `dE_r/dp` and for a flux with respect to a boundary coefficient. Verify DKX's complete model and compare independently audited derivatives under matching model scope. |
| B11 | Restart and optimization hard cases | Cross-surface 2.46% regression, wide Er reuse, `Er=15` high-pitch case with `FSABjHat` = −3.77e-3 at `Nxi = 180` as a historical high-pitch comparison point pending joint refinement; cold equivalence in observables and bounded failure recovery. |
| B12 | CPU/GPU batch and state distribution | Strong/weak scaling, throughput/latency, real device occupancy, gradients, memory and communication; correctness before speed. |
| B13 | Actual VMEX and ESSOS design chain | Equilibrium → geometry → DKX objective → gradient → constrained design; final independently evaluated, refined design. |
| B14 | NEOPAX transport integration | Consistent profile/grid/flux units, conservation, ambipolar response and lagged-response refresh; prescribed versus evolved state. |

The four families Phase 1 actually runs, with their independent anchors; points are added only to resolve a concrete uncertainty:

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

## 4. Phases

Each phase names the figure it produces, its entry and exit criteria, numbered steps, a kill criterion where the work is an experiment, and an effort range in person-weeks. Phases are ordered by dependency; Phase 2 and Phase 3 can run in parallel once Phase 1's error bars exist.

### Phase 0: land, freeze, and set up the working method (1 week)

Entry: this plan merged. Steps, in order:

1. Confirm main carries the integration: the reverts of #171 and #175, the stack #169–#188 squash-merged in order, #191 (`.test_durations`), #189 and this plan. `git log --oneline origin/main | head -25` should show them; `pytest -q -n 4 -m "not slow"` must pass on main and `python -m sphinx -b html -W docs _build` must be clean.
2. Confirm the open-PR list is empty of superseded work: #168 and #172 closed with dispositions; #190 merged as this plan. Anything else open must map to a phase figure or a bug, or be closed.
3. Create `docs/adr/0001-figure-first-planning.md`, `docs/adr/0002-durations-balanced-shards.md` (the reversal of the 2026-07-17 archive decision) and `docs/adr/0003-attribution.md` (every commit authored by the maintainer alone; the `commit-trailers` gate enforces it), using the Nygard template: title, status, context, decision, consequences.
4. Create `docs/experiments/README.md` with the one-page record template (hypothesis, admission test, result, decision) and `CHANGELOG.md` with a `v2.4.0-rc1` entry drafted from the stack's PR descriptions. This is where the execution diary goes from now on.
5. Draft `v2.4.0-rc1` in the changelog, but defer tagging and publication until the maintainer authorizes the release after the important goals are achieved. Freeze evidence tooling: no changes under `tools/benchmarks/` except bug fixes until Phase 1 needs one.
6. Fetch current main and work from a clean branch or isolated worktree. Preserve existing branches and uncommitted work; do not hard-reset an unrelated checkout.

Exit: main green at the integrated tree; ADRs 1–3 and the experiment template exist; the changelog holds the stack's story. A deferred release tag does not block research phases. Effort: one person-week, mostly review.

### Phase 1: the positioning figure (3–4 weeks)

**Why first.** The yancc paper ([Conlin & Landreman 2026](https://arxiv.org/abs/2607.20861)) is now the reference point every reviewer will hold DKX against: full 4D DKE on one A100, NCSX single-species at (n_x, n_α, n_θ, n_ζ) = (7, 121, 43, 65) in 6 GB, about 5× faster than SFINCS on 128 cores at moderate collisionality, agreement within 1% with SFINCS and MONKES. It does not do Phi1, does not find the ambipolar root, and reports no numerical verification of any derivative. Until DKX has a figure on the same problem, every DKX speed claim is unanchored and every "differentiable" claim is undifferentiated from yancc's. This phase produces that anchor and, as a by-product, forces every check #189 lists.

**Figure 1 (methods paper, Fig. "positioning").** One NCSX full-Fokker-Planck single-species case and one two-species case at yancc's published resolutions, plus the W7-X monoenergetic MONKES/yancc case at (N_L, N_θ, N_ζ) = (180, 39, 99). Panels: (a) fluxes and bootstrap current from DKX, SFINCS v3 and yancc with DKX's algebraic error bar `λᵀr` and its Richardson grid estimate drawn on the DKX points; (b) time to accepted observable, cold and warm, CPU (M3 Max) and GPU (A4000), each code at its own converged resolution; (c) peak device/host memory. Discretizations differ (yancc: finite differences in pitch and angles; DKX/SFINCS: Legendre in pitch), so the comparison is at *converged observables*, never at "identical grids" in the literal sense; the figure says so in its caption.

Steps:
1. Install `yancc` from PyPI (0.0.1; depends on lineax, equinox, interpax, orthax) and pin its commit; reproduce its NCSX (7,121,43,65) and W7-X monoenergetic numbers on the A4000 before touching DKX. If its published numbers do not reproduce within 2× on our hardware, record that and proceed with our measurement only.
2. Build the DKX cases from the same VMEC/Boozer files; converge each observable separately (theta, zeta, pitch, speed, jointly) with `dkx converge`; record the accepted grid per observable.
3. Run SFINCS v3 on the office isolated toolchain (PETSc 3.23.6 / MUMPS 5.8.1) at the accepted grid with the right-preconditioned GMRES/MUMPS configuration that #189's evidence found necessary for a valid 1e-10 original residual.
4. Compute `λᵀr` for every reported linear observable (one transpose solve with the operator's transpose, which every differentiable route already exposes through `_implicit_solve`) and the Richardson estimate from the convergence ladder; these become the error bars. The adjoint solve exists; what is added is the ~50-line utility that forms `λᵀr` per moment and writes it into `Result`, plus its manufactured-solution test.
5. Measure with the existing supervised runner, pinned SHAs, idle machine, five repetitions, medians with dispersion; cold and warm separately; never DKX solve-only against SFINCS whole-process.

Exit: Figure 1 rendered from tracked inputs by one runner selection; every DKX point carries both error bars; the caption states scope. Kill: none, this phase is mandatory; if DKX loses on time or memory the figure still ships, because a loss with error bars is publishable and a win without them is not.

Effort: 3–4 weeks, one person, mostly measurement and yancc onboarding. No new DKX solver code. Evidence tooling frozen at what exists.

### Phase 2: why the Krylov route loses, and two bounded extensions (3 weeks, time-boxed)

**Correction to the first draft of this plan, and to a common assumption.** DKX already implements SFINCS's simplified-operator preconditioner, and it is the default on every deck that leaves the pitch-angle-scattering family: `build_coarse_preconditioner` (`coarse_precond.py:916`) mirrors the Fortran `preconditionerOptions` defaults (self-species, x-diagonal collisions; the Er and drift L±2 terms dropped), inverts that operator *exactly* with a batched block-Thomas factorization over (species, x), eliminates the constraint border exactly by a Schur complement, and is Phi1-aware. `multigrid.py` and `sparse_precond.py` are alternative inverses of the same simplified operator; the Krylov method is SOLVAX's GCROT with a recycle space; the adjoint solve is a cold-started GCROT on the transpose. So "use the exact PAS solve as the preconditioner" is not a proposal, it is the status quo, and the 7-of-23 record and six non-completions are *with* it. #189 was right to say measure first. What the numerics survey adds is a short list of specific, testable reasons the route still loses, and two extensions that follow from them.

**Step 1, one week: the diagnosis.** On the 23 Krylov-route decks and the six failures, log per solve: peak memory of the preconditioner bands and factors versus the operator apply (five of the six failures were OOM while `build_coarse_preconditioner` allocated dense (N_θN_ζ)² bands, 42.9 GB on `filteredW7XNetCDF_2species_magneticDrifts_noEr`; `docs/performance.rst:186–192`); GCROT iterations and restarts; the nonzero mass of `A − M` split by coupling type (field-particle x-coupling of the Fokker–Planck operator, inter-species blocks, |ΔL| = 2 terms from Er xiDot/xDot and tangential drifts, Phi1 border); whether the preconditioner was rebuilt or reused at that point; and the true residual at every restart. Group the decks by which coupling dominates `A − M`. This is instrumentation on data the solver already holds; no algorithm changes. Output: one table, decks × dominant coupling × iterations, which decides steps 2 and 3 and is itself a figure in the methods paper's solver section.

**Step 2, one week: block-pentadiagonal structured solve.** The Er xiDot/xDot terms and the tangential-drift terms couple only |ΔL| ≤ 2 (they carry ξ² factors); the structured route refuses them today (`drift_kinetic.py:1770–1787`) and the coarse preconditioner drops them. Generalizing SOLVAX's block-Thomas kernel from tridiagonal to pentadiagonal in L restores an *exact* direct solve on those decks at roughly four times the tridiagonal factor cost, which on the 744k HSX reference would be about 100 s against the Fortran build's 464 s. Decks whose dominant `A − M` term is |ΔL| = 2 leave the Krylov route entirely. Admission test: on the Er-xDot and magnetic-drift decks, exact residual ≤ 1e-12 and wall time ≤ 5× the tridiagonal solve at matched size; A/B/A/B, pinned SHA, idle machine. Kill: if step 1 shows |ΔL| = 2 is not the dominant dropped coupling on those decks, or the pentadiagonal factor exceeds 5× tridiagonal at the 66k deck, stop and record. Ownership: the generic kernel in SOLVAX, the admissibility check and band assembly in DKX.

**Step 3, one week: recycling discipline in sweeps.** Recycled harmonic Ritz vectors approximate invariant subspaces of the *preconditioned* operator; if M is rebuilt at every Er or ν point the recycle space refers to a different operator and mostly costs orthogonalizations ([Soodhalter, de Sturler & Kilmer 2020](https://arxiv.org/abs/2001.10347); [Parks et al. 2006](https://doi.org/10.1137/040607277)). Today the root driver threads `(x, recycle, precond)` across points when `warm_start=True` (`er.py:154–175, 659`), but `dkx.batch` scans thread nothing and rebuild M per point (`batch.py`), and the adjoint always cold-starts with no recycle space of its own. Changes: hold M fixed over a window of the sweep with a measured refresh rule (SFINCS's `reusePreconditioner`; #189's `T_build` versus expected extra iterations economics); restart the recycle space whenever M changes; keep a separate recycle space for the transpose (the operator is strongly nonnormal, so left and right invariant subspaces differ); solve all drives of a point as one block. An Er sweep is `A(E_r) = S + E_r E + ν C`, not a scalar shift, so shifted-Krylov shortcuts do not apply and validity is empirical: drop the recycle space when its first-cycle residual reduction does not beat no-recycling. Admission test: a 10-point Er sweep of one full-FP deck under four ablations, fixed M with recycling, fresh M with recycling, fixed M without, fresh M without; iterations and wall time per point, values within the observable budget of cold solves. Kill: fixed-M recycling does not reduce total iterations by 1.5× → stop and record.

**Step 4, two days, accuracy not speed: extended-precision refinement on the exact route.** The PAS decks sit at κ ≈ 3e12 (7e9 after Ruiz equilibration), so the forward-error bound is κu ≈ 3e-4 unscaled, 7e-7 scaled. One to three refinement sweeps with an fp64 factor and a compensated double-double residual recover full double accuracy for κu ≪ 1 ([Carson & Higham 2018](https://nhigham.com/2017/07/26/accelerating-the-solution-of-linear-systems-by-iterative-refinement-in-three-precisions/); [Amestoy et al. 2024](https://eprints.maths.manchester.ac.uk/)). About twenty lines on the existing `solvax.refine.iterative_refinement` path; it makes the 1e-10-level agreement claims against SFINCS and DKES defensible rather than lucky.

**Step 5, conditional on step 1, one week: memory-lean preconditioner by default where bands do not fit.** The memory-lean routes exist (`_coarse_factors_fit` keeps only the Schur LU, a third of the bands; `block_thomas_checkpointed_fn` regenerates rows on every application) and the coarse preconditioner has a `factor_dtype` switch. If step 1 confirms memory as the binding loss on the failing decks, make the route choice automatic against the device budget and add fp32 factors after Ruiz equilibration, admissible only for scaled κ ≪ 1e10 (Ruiz-scaled PAS at about 7e9 is borderline; unscaled 3e12 fails). Admission: the six decks complete within the office GPU's 16 GB with original residual ≤ 1e-10. Kill: fp32 factors do not converge under fp64 GMRES-IR on the 2,804 deck after Ruiz → keep fp64 and Schur-only.

**Explicitly not in this phase.** Half precision anywhere in a factor is inadmissible for these operators. The κ ≈ 1e18 Er-xDot decks are numerically singular in double (κu ≈ 1e2); no preconditioner or refinement rescues them, and the fix is formulation (constraint-row and source-column scaling per SFINCS's block structure), diagnosed with a double-double factorization on a small deck. Semi-coarsened multigrid with the exact-in-L solve as plane smoother is built only if step 1 shows iterations growing with resolution; DKX's `multigrid.py` is the starting point. cuDSS through an XLA FFI call (the spineax pattern) is the SFINCS-style general LU fallback on GPU; it is proprietary and NVIDIA-only, and it is deferred until a deck needs it.

Deliverable if any step passes: Figure 2 of the methods paper, iterations and time versus size for the ablation (none / coarse exact / multigrid / pentadiagonal direct / fixed-M recycling), with yancc's published multigrid numbers as the external bar. Deliverable if all fail: the same figure with the losses, and one-page experiment records under `docs/experiments/`.

### Phase 3: verified derivatives, the closure accuracy map, and one real optimization (5–6 weeks)

**Why this and not more speed.** Derivatives are the claim yancc makes without evidence and NTX proves only for the monoenergetic problem (14× over finite differences at 32 parameters, agreement ~2e-14). DKX's #184 and #188 routed the ambipolar-root and profile derivatives through the differentiable solver with original-equation admission; what does not exist is the *published* check on a real configuration, and the accuracy map that design teams have asked for in print.

**Figure 3.** For one QI configuration (CIEMAT-QI4X or a Goodman-type QI) and one QA with large bootstrap current (Helios-like), at a W7-X-like surface: `dJ_bs/dp_k` and `dE_r/dp_k` for profile parameters and for a handful of boundary Fourier coefficients, from `jax.grad` through the ambipolar root by the implicit function theorem, against central finite differences over a step window and a Taylor-remainder slope near 2; cost ratio AD/FD versus parameter count; cold versus warm agreement. Where Paul et al. ([2019](https://arxiv.org/abs/1904.06430)) published sensitivities for the same quantities, overlay them.

**Figure 4 (the accuracy map).** Bootstrap current from Redl/Sauter via the quasisymmetry isomorphism (what DESC and SIMSOPT use; DESC's own tutorial states it does not apply to non-quasisymmetric fields), from PENTA-style momentum-corrected monoenergetic coefficients, and from DKX full-operator multispecies kinetics, across ν* and E_r on the same three or four real designs (Infinity Two, Helios, Stellaris, W7-X). Saxena et al. ([2025](https://arxiv.org/abs/2507.05166)) frame "how wrong is the analytic closure off-symmetry" as the open question; Infinity Two's design paper iterates SFINCS with VMEC by hand; Stellaris is QI and needs off-Redl verification. This figure is the physics letter, and it uses nothing DKX does not already have except the phase-1 error bars.

Verification design constraint from [Albert et al. 2024](https://arxiv.org/abs/2407.21599): at E_r = 0 the 1/ν off-set current does not converge and oscillates in log ν*; with finite E_r it decays as ν*^(3/5). Every low-collisionality bootstrap ladder in this programme is therefore run at finite E_r, and the ν*^(3/5) decay is itself a code-independent target (Figure 4 inset).

**Coordinate choices for the optimization deliverable** (decided; the default is the first row):

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

**The optimization itself, in three steps within one example family:**

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

Exit: Figures 3 and 4 rendered from tracked inputs; one optimization with objective improvement exceeding its numerical uncertainty, constraints satisfied, full-chain Taylor and finite-difference tests passing on smooth branches, and the final design recomputed cold at finer resolution with an independent SFINCS check on the converged equilibrium. Kill: if the root derivative fails its Taylor test on a regular branch after two weeks of diagnosis, narrow to prescribed-`E_r` derivatives and say so; if the derivative or model is invalid, stop and narrow the parameter domain rather than freezing anything to preserve a descending objective. Effort: 5–6 person-weeks.

### Phase 4: native Phi1 and a W7-X impurity result (4–6 weeks; entry requires Phase 1 error bars)

#189 defers native Phi1 behind "coupled/error contracts". This plan puts it on the roadmap with a named result, because it is the one full-kinetic capability with a live experimental audience that neither yancc nor MONKES/NTX has: W7-X impurity transport is neoclassically dominated in NBI-heated and turbulence-suppressed scenarios with peaking scaling with Z (Nucl. Fusion 2023; PPCF 2025), and the flux-surface variation of the potential is known to change impurity fluxes at the order-unity level ([Mollén et al. 2018](https://iopscience.iop.org/article/10.1088/1361-6587/aac700); García-Regaña et al. 2017), with the classical channel mattering in optimized stellarators ([Buller et al.](https://arxiv.org/abs/1903.12511)). KNOSOS's Phi1 is low-collisionality only.

**Figure 5.** W7-X standard configuration, bulk ions + electrons + one impurity (C or Fe), impurity particle flux and its convective/diffusive decomposition versus ν* with and without Phi1, DKX native against SFINCS with Phi1, and the impurity `E_r` root shift. GPU, warm scans.

Steps: (1) expose the existing compatibility-path Phi1 through the prepared native objects with the *coupled* residual (kinetic + quasineutrality + gauge) as the admission check, linearized quasineutrality first, block/Schur preconditioner, Newton with Eisenstat–Walker forcing as #189 proposes; (2) reproduce Mollén 2018's impurity result; (3) the scan. Exit: Figure 5 with error bars from the coupled residual. Kill: if the coupled Newton solve does not converge on the W7-X case with the linearized quasineutrality after three weeks, ship the frozen-Phi1 comparison labeled as such.

NEOPAX integration then needs a small in-memory protocol for species order,
radial centers/faces, SI fluxes and Jacobians, boundary conditions, validity and
refresh. Check transport conservation and lagged-response error before claiming a
transport simulation. NTX is a candidate monoenergetic database producer; avoid a
second database framework until its normalization/interpolation/restart contract
is compared with DKX's existing one. ESSOS first realizes coils for an accepted
target with field-error, length, curvature, distance and current constraints.
Arbitrary coil fields need not possess nested surfaces. Joint plasma/coils and
open-field-line mirror optimization follow their own physical validity gates.

### Phase 5: the papers, defined by the figures above

**Methods/software paper (CPC or JCP).** Figures: 1 (positioning), 2 (preconditioner ablation, if Phase 2 passes; otherwise the factor-reuse ablation), 3 (derivative verification), plus the convergence-order figures already in the validation matrix, the SFINCS field-by-field parity table, and the MONKES/YANCC monoenergetic table. The contribution statement is: SFINCS-v3 physics including Phi1 and ambipolar roots, on GPU, with every observable carrying an algebraic and a discretization error bar, and derivatives verified through the root. Not "differentiable GPU neoclassics", which yancc already owns as a phrase.

**Physics letter (Nuclear Fusion or JPP Letters).** Figure 4, the accuracy map of Redl/PENTA/monoenergetic closures against full kinetics on real designs, with the ν*^(3/5) inset, and Figure 5 if Phase 4 lands in time. This is the result three design teams have said in print they need.

Results the community would take up immediately, in the order they become available here: verified `dJ_bs/d(boundary)` inside DESC or SIMSOPT for a QI and a large-J_bs QA; the closure accuracy map; neoclassical Jacobians `∂(Γ_s, Q_s)/∂(∇n, ∇T, E_r)` and the root derivative as a T3D-compatible module (Infinity Two's T3D-GX-SFINCS pipeline has SFINCS as its CPU-only, derivative-free component); GPU Phi1 impurity scans for W7-X. No published neural surrogate of core stellarator neoclassical transport exists as of this survey; a DKX-generated table is a cheap by-product once the scans run, and is deliberately *not* a phase.

## 5. Engineering contracts: reuse, differentiation, measurement

These are the specifications Phases 2 and 3 implement against. They are adopted from #189 with one change: the root is already differentiated by the implicit function theorem, so the "design sketch" below describes what to expose, not what to build.

### 5.1 In-memory reuse contract

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

### 5.2 Differentiation and the coupled potential

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

### 5.3 Measure the work users actually pay for

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

## 6. Documentation, examples and deliberate reduction

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

The README is governed by section 7 rule 12 and by the existing gates (`test_benchmark_doc_claims`, `test_readme_quickstart_runs`, `test_figure_provenance`): a self-contained quickstart that runs `dkx.run` and prints one flux and the solver route, one gradient example, the four-code capability table, the measured results table with every pinned number, both README figures and the cross-code figure, a BibTeX block, and no hedging sentences; hedges live in the docs quadrant Diátaxis assigns them.

## 7. Working method

The stack that produced #170–#188 was written in one day: 53 commits, 44 files, about 5,900 insertions. At the Cisco/SmartBear ceiling of roughly 500 reviewed lines per hour ([SmartBear](https://smartbear.com/learn/code-review/best-practices-for-peer-code-review/)) that is 12 to 18 reviewer-hours, ten to twenty times over Google's "100 lines is usually reasonable, 1,000 is usually too large" ([Google eng-practices](https://google.github.io/eng-practices/review/developer/small-cls.html)). The work was good; the process made it unreviewable, and an unreviewable stack is a release risk however green its CI. The following rules replace volume with direction.

1. **Figure-first.** The roadmap is the ordered figure list of the two papers (Whitesides: a good outline for the paper is also a good plan for the research programme, [Adv. Mater. 2004](https://www.gmwgroup.harvard.edu/publications/whitesides-group-writing-paper)). Each roadmap item is a figure or table with an owner, a status and an acceptance criterion. Work that maps to no figure and no bug is not scheduled. Every week at least one merged PR adds or upgrades a paper figure.
2. **PR size cap: at most 400 changed lines and 10 files, one idea per PR, refactor never shares a PR with behaviour.** Above 800 lines the PR is split before review. Generated data and pinned artifacts go in their own PR.
3. **No stacks without tooling.** Stacks are allowed only with depth at most 3, auto-rebase tooling, and each PR independently mergeable ([Graphite](https://graphite.com/guides/stacked-diffs)). Otherwise finish and merge PR n before opening n+1; at most three open PRs per author. Branch lifetime at most 48 hours ([trunk-based development](https://trunkbaseddevelopment.com/)); incomplete features land behind a flag.
4. **Agent output is budgeted by review capacity.** One reviewer, two 60-minute sessions a day at 400 lines each is about 800 reviewed lines a day. The agent stops opening PRs when the review queue reaches that, whatever its generation speed.
5. **Four CI tiers; the PR gate is T1 and takes at most 20 minutes.** T0 lint and unit on every push. T1 touched-module tests, one small SFINCS parity deck, the README example. T2 nightly: the full 38-deck matrix and GPU. T3 on release tags: cross-code, figure regeneration, wheel, Zenodo. A change that cannot be trusted after T1 is too large.
6. **Three documents, three jobs, hard caps.** `plan.md` holds phases, figures and criteria and never status prose (this document; deletions weekly). `docs/adr/NNNN-*.md` holds decisions, one page each, immutable, superseded by a new ADR ([adr.github.io](https://adr.github.io/)). `CHANGELOG.md` holds what shipped per release ([Wilson et al. 2017](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005510)). Execution detail lives in PR descriptions; review reports are dated write-once files under `docs/reviews/`.
7. **Experiments are time-boxed with a written kill criterion before they start.** The record is a one-page file under `docs/experiments/` with hypothesis, admission test, result, decision. Phase 2 is the template.
8. **Provenance is five fields, not a framework.** Every output records version, git SHA, JAX/jaxlib versions with the x64 flag, `case_id` and command line, device and host ([Taschuk & Wilson 2017](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005412), rule 10). Paper figures regenerate from `publications/<paper>/make_figures.py` in a pinned environment (the DESC pattern). The supervised runner, verifier, archive and provenance tooling that exist are frozen; nothing further until a reviewer asks.
9. **Benchmarks follow Hoefler and Belli** ([SC'15](https://htor.inf.ethz.ch/publications/img/hoefler-scientific-benchmarking.pdf)) and the [JAX benchmarking page](https://docs.jax.dev/en/latest/benchmarking.html): absolute times beside every ratio, all decks including losses, median of at least five warm solves after `block_until_ready()`, compile time once per resolution, spread stated, hardware and versions in the caption. Pinned SHA, idle machine, A/B/A/B.
10. **Verification is labelled by tier everywhere.** Tier A code verification: analytic limits, Onsager symmetry, adjoint-versus-finite-difference. Tier B solution verification: convergence per axis with the criterion stated. Tier C cross-code: SFINCS parity (same discretization, regression) reported separately from MONKES/yancc independence. Tier D validation: W7-X `E_r` with experimental uncertainty. Tier C is never called validation ([Oberkampf & Roy 2010](https://www.cambridge.org/core/books/abs/verification-and-validation-in-scientific-computing/index/EE029CB068531D27); ASME V&V 20).
11. **Prepare releases at paper milestones**, with tag, changelog, `CITATION.cff` and a Zenodo DOI when the maintainer authorizes publication after important goals are achieved. Papers cite the tag, not `main`.
12. **README budget enforced by a T1 test**: at most 120 lines and 650 words, one Python block of at most 12 lines that runs in under a minute on CPU and contains a gradient, one BibTeX block, and none of "has not", "cannot", "does not yet", "is not converged". Hedges live in the docs quadrant Diátaxis assigns them ([diataxis.fr](https://diataxis.fr/)). Measured exemplars: Diffrax 82 lines / 315 words, Optimistix 83 / 324, simsopt 85 / 403, DESC 137 / 623; DKX on main 250 / 1,508 with no citation block and no `jax.grad`.
13. **Definition of done for a PR, in at most ten lines:** what changed, why, which CI tier proves it, which figure, ADR or issue it serves; tests and docs in the same PR; a physics change regenerates the affected figure by script and attaches it.

## 8. Deferred work

Defer a new fine-grid coordinate/discretization backend; MUMPS reimplementation;
BLR/mixed precision without measured kinetic evidence; learned/Nyström/PINN
preconditioners; unqualified truncated-adjoint windows; multi-host/state-decomposed
execution; new database frameworks; joint coils/plasma optimization; and mirrors.
Existing useful experimental APIs can remain clearly labeled. A deferred item
reopens only with a named user calculation the active deliverables cannot serve,
a bounded experiment and a measurable decision. Native Phi1/full drifts are
scientific completeness work after the corresponding coupled/error contracts,
not abandoned physics or an excuse to claim SFINCS parity early.

Also deferred, from the 2026-09-06 numerics survey: fp32 factors before Ruiz equilibration or at scaled κ above about 1e10; half precision anywhere in a factor; differentiating through Krylov iterations; semi-coarsened multigrid with the exact-in-L plane smoother unless Phase 2 shows iterations growing with resolution; cuDSS through an XLA FFI call until a deck needs a general LU on GPU; a neural surrogate of core neoclassical transport (a cheap by-product of Phase 3 and 4 scans, not a phase).

## 9. History of this plan and disposition of its predecessors

| Predecessor | Where it is | Disposition |
| --- | --- | --- |
| Phase checklist and execution diary (to 2026-09-05) | git history before `be6506fe` | replaced by the R0–R11 queue |
| R0–R11 work queue with implementation checkpoint (`be6506fe`, #169–#188) | [`plan.md` at `be6506fe`](https://github.com/uwplasma/DKX/blob/be6506fedbf357ab7fbbc1c1eaa35195080afc9e/plan.md) | its contract, evidence hierarchy, benchmark families and retained decisions are sections 1 and 3 here; the checkpoint diary goes to `CHANGELOG.md` |
| Three-deliverable plan (#189) | merged; [its `plan.md`](https://github.com/uwplasma/DKX/pull/189/files) | its state tables, capability priorities, four families, reuse contract, differentiation and measurement specs, coordinate table, optimization steps, documentation table and deferred list are sections 2, 3.3, 4 and 5–8 here |
| Independent take (#190) | this branch's history | its figure-first phases, competitor positioning, Phase 2 diagnosis, working method and references are sections 1, 2.4, 4, 7 and 10 here |
| Survey reports behind #190 | branch `review/surveys-20260906` | audit trail for section 10 |
| Corrections accepted from #176 to #172 | `plan.md` at `be6506fe`, section 2.4 | an unknown-count threshold does not certify a reference; a saturated condition number attributes nothing; the Shaing–Callen paper concerns the collisionality limit, not a pitch ladder; clone-size arithmetic assumes an unchanged tree |

Merge order executed on 2026-09-06: #192 (reverts of #171 and #175) → #169 → #170 → #173 → #174 → #176 → #177 → #178 → #179 → #180 → #181 → #182 → #183 → #184 → #185 → #186 → #187 → #188 → #191 → #189 → this plan; #168 and #172 closed as superseded. Every commit is authored by the maintainer.

## 10. References

### Neoclassical codes, benchmarks and physics
- Landreman, Smith, Mollen, Helander, "Comparison of particle trajectories and collision operators for collisional transport in nonaxisymmetric plasmas", Phys. Plasmas 21, 042503 (2014), DOI 10.1063/1.4870077, arXiv 1312.6058 https://arxiv.org/abs/1312.6058
- SFINCS repository, landreman/sfincs; input.namelist symlink page https://github.com/landreman/sfincs/blob/master/fortran/version3/input.namelist ; issue #1 quoting preconditionerOptions (F) https://github.com/landreman/sfincs/issues/1
- Mollen, Landreman, Smith, Braun, Helander, "Impurities in a non-axisymmetric plasma: transport and effect on bootstrap current", arXiv 1504.04810 https://arxiv.org/abs/1504.04810
- Mollen, Landreman, Smith, Garcia-Regana, Nunami, "Flux-surface variations of the electrostatic potential in stellarators: impact on the radial electric field and neoclassical impurity transport", PPCF 60, 084001 (2018) https://www.osti.gov/biblio/1499870
- Garcia-Regana et al., "Electrostatic potential variation on the flux surface and its impact on impurity transport" (2017) https://www.researchgate.net/publication/271079728_Electrostatic_potential_variation_on_the_flux_surface_and_its_impact_on_impurity_tran
- Buller et al., "The importance of the classical channel in the impurity transport of optimized stellarators", J. Plasma Phys., arXiv 1903.12511 https://arxiv.org/html/1903.12511
- Paul, Abel, Landreman, Dorland, "An adjoint method for neoclassical stellarator optimization", arXiv 1904.06430 https://arxiv.org/abs/1904.06430
- Paul, Landreman, Antonsen, "Adjoint methods for stellarator shape optimization and sensitivity analysis", arXiv 2005.07633 https://arxiv.org/pdf/2005.07633
- Paul et al., "Adjoint approach to calculating shape gradients for three-dimensional magnetic confinement equilibria" https://www.osti.gov/pages/biblio/1597704
- Paul et al., "Adjoint methods for quasisymmetry of vacuum fields on a surface", arXiv 2108.11433 https://arxiv.org/pdf/2108.11433
- Conlin, Landreman, "yancc: A GPU-accelerated, differentiable solver for neoclassical transport in tokamaks and stellarators", arXiv 2607.20861 (F abstract and full text) https://arxiv.org/abs/2607.20861 ; https://arxiv.org/html/2607.20861
- yancc repository https://github.com/f0uriest/yancc
- Escoto, Velasco, Calvo, Landreman, Parra, "MONKES: a fast neoclassical code for the evaluation of monoenergetic transport coefficients", Nucl. Fusion, DOI 10.1088/1741-4326/ad3fc9, arXiv 2312.12248 (F abstract and full text) https://arxiv.org/abs/2312.12248 
- MONKES repository (F, from paper) https://github.com/JavierEscoto/MONKES/
- Escoto Lopez, PhD thesis, "Fast and accurate calculation of the bootstrap current and radial neoclassical transport in low collisionality stellarator plasmas", arXiv 2510.27513 https://arxiv.org/abs/2510.27513
- "Evaluation of neoclassical transport in nearly quasi-isodynamic stellarator magnetic fields using MONKES", arXiv 2410.17836 https://arxiv.org/pdf/2410.17836
- NTX repository, uwplasma/NTX https://github.com/uwplasma/NTX
- Velasco, Calvo, Parra, Garcia-Regana, "KNOSOS: a fast orbit-averaging neoclassical code for stellarator geometry", J. Comput. Phys. (2020), DOI 10.1016/j.jcp.2020.109512, arXiv 1908.11615 https://arxiv.org/abs/1908.11615
- "Fast simulations for large aspect ratio stellarators with the neoclassical code KNOSOS", arXiv 2106.01727 https://arxiv.org/pdf/2106.01727
- Hirshman, Shaing, van Rij, Beasley, Crume, "Plasma transport coefficients for nonsymmetric toroidal confinement systems", Phys. Fluids 29, 2951 (1986) https://pubs.aip.org/aip/pfl/article-abstract/29/9/2951/944354/ ; OSTI (S) https://www.osti.gov/servlets/pu
- van Rij, Hirshman, "Variational bounds for transport coefficients in three-dimensional toroidal plasmas", Phys. Fluids B 1, 563 (1989) https://pubs.aip.org/aip/pfb/article-abstract/1/3/563/940728/
- "Modelling of relativistic electron transport with non-relativistic DKES solver", J. Plasma Phys. (2024) https://www.cambridge.org/core/journals/journal-of-plasma-physics/article/modelling-of-relativistic-electron-transport-with-nonrelativistic-dkes-solver/A
- Spong, "Generation and damping of neoclassical plasma flows in stellarators", Phys. Plasmas 12, 056114 (2005) https://pubs.aip.org/aip/pop/article/12/5/056114/1015589/
- "Three-dimensional equilibria and transport in RFX-mod: A description using stellarator tools", Phys. Plasmas 18, 062505 (2011) https://pubs.aip.org/aip/pop/article-abstract/18/6/062505/387754/
- Kernbichler et al., "Recent progress in NEO-2 - a code for neoclassical transport computations based on field line tracing", Plasma Fusion Res. 3, S1061 (2008) https://www.jstage.jst.go.jp/article/pfr/3/0/3_0_S1061/_article/-char/en
- Kernbichler, Kasilov, Kapper, Martitsch, Nemov, Albert, Heyn, "Solution of drift kinetic equation in stellarators and tokamaks with broken symmetry using the code NEO-2", PPCF 58, 104001 (2016) https://iopscience.iop.org/article/10.1088/0741-3335/58/10/10400
- Beurskens et al., "Demonstration of reduced neoclassical energy transport in Wendelstein 7-X", Nature (2021) (S; author list U) https://www.nature.com/articles/s41586-021-03687-w
- W7-X power balance study (NEOTRANSP use), PPCF (2025) https://iopscience.iop.org/article/10.1088/1361-6587/ade824
- EUTERPE vs NEOTRANSP Er benchmarking figure https://www.researchgate.net/figure/Benchmarking-of-the-global-neoclassical-radial-electric-field-calculated-with-EUTERPE_fig1_378516498
- Beidler et al., "Benchmarking of the mono-energetic transport coefficients - results from the ICNTS", Nucl. Fusion 51, 076001 (2011) https://iopscience.iop.org/article/10.1088/0029-5515/51/7/076001
- Redl, Angioni, Belli, Sauter, "A new set of analytical formulae for the computation of the bootstrap current and the neoclassical conductivity in tokamaks", Phys. Plasmas 28, 022502 (2021) https://pubs.aip.org/aip/pop/article/28/2/022502/124727/ ; open copy 
- Sauter, Angioni, Lin-Liu, Phys. Plasmas 6, 2834 (1999) - U (no URL fetched)
- Landreman, Buller, Drevlak, "Optimization of quasisymmetric stellarators with self-consistent bootstrap current and energetic particle confinement", Phys. Plasmas 29, 082501 (2022), DOI 10.1063/5.0098166, arXiv 2205.02914 https://arxiv.org/abs/2205.02914 ; h
- Albert, Beidler, Kapper, Kasilov, Kernbichler, "On the convergence of bootstrap current to the Shaing-Callen limit in stellarators", arXiv 2407.21599 https://arxiv.org/abs/2407.21599
- Saxena, Ferraro, Martin, Wright, "Bootstrap current modeling in M3D-C1", J. Plasma Phys. 91, E141 (2025), DOI 10.1017/S0022377825100834, arXiv 2507.05166 https://arxiv.org/abs/2507.05166 ; https://www.cambridge.org/core/journals/journal-of-plasma-physics/art
- DESC tutorial "Bootstrap Current Self-Consistency" https://desc-docs.readthedocs.io/en/v0.15.0/notebooks/tutorials/bootstrap_current.html
- VMEX references page https://vmex.readthedocs.io/en/latest/project/references.html
- Goodman et al., "Quasi-isodynamic stellarators with low turbulence as fusion reactor candidates", PRX Energy 3, 023010 (2024), arXiv 2405.19860 https://arxiv.org/abs/2405.19860 ; https://link.aps.org/doi/10.1103/PRXEnergy.3.023010
- Goodman et al., "Constructing precisely quasi-isodynamic magnetic fields", J. Plasma Phys. (2023) https://www.cambridge.org/core/journals/journal-of-plasma-physics/article/constructing-precisely-quasiisodynamic-magnetic-fields/6601E449C8DD3B3FEB361DA2C5732EF
- Jorge et al., "A single-field-period quasi-isodynamic stellarator", J. Plasma Phys. https://www.cambridge.org/core/journals/journal-of-plasma-physics/article/singlefieldperiod-quasiisodynamic-stellarator/9B2A5FDCCD7774E4F91BE45E75FDC6B0
- "CIEMAT-QI4X: a reactor-relevant quasi-isodynamic stellarator configuration compatible with an island divertor", Nucl. Fusion, arXiv 2512.08825 https://arxiv.org/pdf/2512.08825 ; https://iopscience.iop.org/article/10.1088/1741-4326/ae54ad
- "Near-axis quasi-isodynamic database", arXiv 2601.08400 https://arxiv.org/pdf/2601.08400
- "Optimization of nonlinear turbulence in stellarators", J. Plasma Phys. 90, 905900210 (2024) https://www.cambridge.org/core/journals/journal-of-plasma-physics/article/optimization-of-nonlinear-turbulence-in-stellarators/916FCC56452B5B166C14868F56D99AF5
- Type One Energy, "A comprehensive, unified baseline physics design for the Type One Energy stellarator fusion pilot power plant, 'Infinity Two'", J. Plasma Phys. 91, E65 (2025) https://www.cambridge.org/core/journals/journal-of-plasma-physics/article/compreh
- "Predictions of core plasma performance for the Infinity Two fusion pilot plant", J. Plasma Phys. (2025) https://www.cambridge.org/core/journals/journal-of-plasma-physics/article/predictions-of-core-plasma-performance-for-the-infinity-two-fusion-pilot-plant/
- Thea Energy, "Overview of the Helios Design: A Practical Planar Coil Stellarator Fusion Power Plant", arXiv 2512.08027 https://arxiv.org/abs/2512.08027 ; PDF https://thea.energy/wp-content/uploads/2025/12/20251210_FPP_Helios_overview_paper.pdf ; Fusion Eng. 
- "Equilibrium optimization of the Helios planar coil stellarator power plant", Fusion Eng. Des. (2026) https://sciencedirect.com/science/article/pii/S0920379626002905
- "Stellarator fusion systems enabled by arrays of planar coils", Nucl. Fusion (2025) https://iopscience.iop.org/article/10.1088/1741-4326/ada56c
- Proxima Fusion, "Stellaris: A high-field quasi-isodynamic stellarator for a prototypical fusion power plant", Fusion Eng. Des. (2025) https://www.sciencedirect.com/science/article/pii/S0920379625000705 ; press release https://www.proximafusion.com/press-news
- "Quantitative comparison of impurity transport in turbulence reduced and enhanced scenarios at Wendelstein 7-X", Nucl. Fusion (2023) https://iopscience.iop.org/article/10.1088/1741-4326/aceb76
- "The suppression of anomalous impurity transport above a critical normalized density gradient scale length in Wendelstein 7-X", PPCF (2025) https://iopscience.iop.org/article/10.1088/1361-6587/add597
- "Neural network-based surrogate model for 3D edge-plasma transport in the standard configuration of W7-X", Nucl. Fusion 66 (2025) https://iopscience.iop.org/article/10.1088/1741-4326/ae203d
- IPP abstract "Neoclassical transport simulations for stellarators" (DCOM/NNW description) https://pure.mpg.de/rest/items/item_2139735_1/component/file_2139734/content
- MMMnet surrogate (NSTX-U) https://www6.lehigh.edu/~eus204/per/publications/journals/tps24_MMMnetNSTXU.pdf
- "5D Neural Surrogates for Nonlinear Gyrokinetic Simulations of Plasma Turbulence", arXiv 2502.07469 https://arxiv.org/pdf/2502.07469
- "Efficient dataset construction using active learning and uncertainty-aware neural networks for plasma turbulent transport surrogate models", arXiv 2507.15976 https://arxiv.org/pdf/2507.15976

### Numerical methods
- "yancc: A GPU-accelerated, differentiable solver for neoclassical transport in tokamaks and stellarators", arXiv:2607.20861 (2026). https://arxiv.org/abs/2607.20861 ; full text https://arxiv.org/html/2607.20861v1 (fetched). Author list not extracted — UNVERI
- Landreman, Smith, Mollen, Helander, "Comparison of particle trajectories and collision operators for collisional transport in nonaxisymmetric plasmas", Phys. Plasmas 21, 042503 (2014). https://arxiv.org/pdf/1312.6058 (fetched, text extracted); https://pubs.a
- SFINCS repository and v3 manual. https://github.com/landreman/sfincs ; https://raw.githubusercontent.com/landreman/sfincs/master/doc/manual/version3/runs.tex (read). input.tex not found (404) — namelist option names UNVERIFIED.
- Escoto et al., "MONKES: a fast neoclassical code for the evaluation of monoenergetic transport coefficients", arXiv:2312.12248. https://arxiv.org/pdf/2312.12248 (URL seen)
- Escoto Lopez, "Fast and accurate calculation of the bootstrap current and radial neoclassical transport in low collisionality stellarator plasmas" (thesis), arXiv:2510.27513 (2025). https://arxiv.org/abs/2510.27513 (fetched)
- Belli & Candy, "Full linearized Fokker-Planck collisions in neoclassical transport simulations", PPCF 54, 015015 (2012). https://iopscience.iop.org/article/10.1088/0741-3335/54/1/015015
- Landreman & Ernst, "New velocity-space discretization for continuum kinetic calculations and Fokker-Planck collisions", J. Comput. Phys. 243, 130-150 (2013). https://arxiv.org/abs/1210.5289 ; https://www.sciencedirect.com/science/article/abs/pii/S00219991130
- Velasco et al., KNOSOS. https://arxiv.org/pdf/2106.01727 ; https://github.com/joseluisvelasco/KNOSOS (URLs seen)
- PPPL-4775, "Numerical Calculation of Neoclassical Distribution Functions ..." https://bp-pub.pppl.gov/pub_report/2012/PPPL-4775.pdf (URL seen)
- DKX and SOLVAX repositories (given by the task; not fetched). https://github.com/uwplasma/DKX ; https://github.com/uwplasma/SOLVAX
- Dorf, Dorr, Ghosh, Umansky, Soukhanovskii, "Implicit full-F simulations of neoclassical ion transport", Phys. Plasmas 32(8) (2025). https://www.osti.gov/biblio/2588989 (fetched)
- "Axisymmetric Gyrokinetic Simulation of ASDEX-Upgrade Scrape-off Layer Using a Conservative Implicit BGK Collision Operator" (Gkeyll), arXiv:2507.22821. https://arxiv.org/abs/2507.22821
- Barnes, Abel, Dorland et al., "Linearized model Fokker-Planck collision operators for gyrokinetic simulations. II. Numerical implementation and tests", Phys. Plasmas 16, 072107 (2009). https://arxiv.org/abs/0809.3945
- GENE-X LBD collision operator (implementation/verification). https://www.researchgate.net/publication/357053643_Implementation_and_verification_of_a_conservative_multi-species_gyro-averaged_full-f_Lenard-Bernstein_Dougherty_collision_operator_in_the_gyrokine
- "An Angular Multigrid Preconditioner for the Radiation Transport Equation with Forward-Peaked Scatter", arXiv:2010.04559. https://arxiv.org/html/2010.04559 ; Fokker-Planck variant https://www.sciencedirect.com/science/article/pii/S0377042718306174
- "P-Multigrid Method for the Discontinuous Galerkin Discretization of Elliptic Problems", J. Sci. Comput. (2025). https://link.springer.com/article/10.1007/s10915-025-03105-7
- Parks, de Sturler, Mackey, Johnson, Maiti, "Recycling Krylov subspaces for sequences of linear systems", SIAM J. Sci. Comput. 28(5), 1651-1674 (2006), doi:10.1137/040607277. https://vtechworks.lib.vt.edu/items/590c07fe-a0c8-49b2-9494-be5061f5fbf7 ; https://w
- Soodhalter, de Sturler, Kilmer, "A survey of subspace recycling iterative methods", GAMM-Mitt. 43(4), e202000016 (2020), doi:10.1002/gamm.202000016. https://arxiv.org/abs/2001.10347 ; https://arxiv.org/pdf/2001.10347 (fetched, text extracted); https://online
- de Sturler, "Truncation strategies for optimal Krylov subspace methods", SIAM J. Numer. Anal. 36(3), 864-889 (1999), doi:10.1137/S0036142997315950 (DOI taken from the survey's reference list; not fetched separately)
- Morgan, "GMRES with deflated restarting", SIAM J. Sci. Comput. 24(1), 20-37 (2002) (from the survey's reference list; not fetched separately)
- Kilmer & de Sturler, "Recycling subspace information for diffuse optical tomography", SIAM J. Sci. Comput. (2006) (from the survey's reference list)
- "Recycling Krylov Subspaces and Truncating Deflation Subspaces for Solving Sequence of Linear Systems", ACM TOMS (2021). https://dl.acm.org/doi/10.1145/3439746
- Applications: https://arxiv.org/pdf/1501.03358 (CFD); https://arxiv.org/pdf/2309.09925 (aerostructural adjoints); https://arxiv.org/pdf/2401.09516 (neural-operator data generation)
- Carson & Higham, "Accelerating the Solution of Linear Systems by Iterative Refinement in Three Precisions", SIAM J. Sci. Comput. (2018); MIMS EPrint 2017.24. https://nhigham.com/2017/07/26/accelerating-the-solution-of-linear-systems-by-iterative-refinement-i
- Amestoy, Buttari, Higham, L'Excellent, Mary, Vieuble, "Five-precision GMRES-based iterative refinement", SIAM J. Matrix Anal. Appl. 45, 529-552 (2024); MIMS EPrint 2021.5. https://eprints.maths.manchester.ac.uk/2852/1/paper.pdf (fetched, text extracted); htt
- Higham & Mary, "Mixed precision algorithms in numerical linear algebra", Acta Numerica 31 (2022). https://eprints.maths.manchester.ac.uk/2841/ ; https://research.manchester.ac.uk/en/publications/mixed-precision-algorithms-in-numerical-linear-algebra/
- Abdelfattah et al., "A Survey of Numerical Methods Utilizing Mixed Precision Arithmetic". https://arxiv.org/pdf/2007.06674
- "Mixed Precision GMRES-based Iterative Refinement with Recycling". https://arxiv.org/pdf/2201.09827
- NVIDIA cuDSS documentation (v0.8.0, Preview). https://docs.nvidia.com/cuda/cudss/index.html (fetched); https://developer.nvidia.com/cudss
- nvmath-python sparse direct solver (cuDSS-backed). https://docs.nvidia.com/cuda/nvmath-python/0.5.0/host-apis/sparse/index.html ; https://github.com/NVIDIA/nvmath-python/tree/main/examples/sparse/advanced/direct_solver ; https://pypi.org/project/nvidia-cudss
- spineax (cuDSS in JAX via FFI). https://github.com/johnviljoen/spineax ; cudss_jax MWE https://github.com/stergiosba/cudss_jax ; JAX discussion https://github.com/jax-ml/jax/discussions/33205
- sparsax (SuiteSparse CHOLMOD/KLU via XLA FFI). https://github.com/knaaptime/sparsax/blob/main/README.md
- JAXMg (cuSOLVERMg multi-GPU dense via FFI). https://arxiv.org/pdf/2601.14466
- jax.experimental.sparse.linalg.spsolve docs. https://docs.jax.dev/en/latest/_autosummary/jax.experimental.sparse.linalg.spsolve.html
- Ghysels & Synk, "High performance sparse multifrontal solvers on modern GPUs", Parallel Computing 110 (2022). https://www.osti.gov/pages/biblio/1960514 (fetched)
- Claus, Ghysels, Boukaram, Li, "A graphics processing unit accelerated sparse direct solver and preconditioner with block low rank compression", Int. J. HPC Appl. (2025), doi:10.1177/10943420241288567. https://journals.sagepub.com/doi/10.1177/1094342024128856
- Li & Ghysels, ATPESC direct-solver lectures 2022/2023 (URLs seen). https://extremecomputingtraining.anl.gov/wp-content/uploads/sites/96/2023/08/ATPESC-2023-Track-5-Talk-3-Li-Ghysels-DirectSolvers.pdf
- Rader, Lyons, Kidger, "Lineax: unified linear solves and linear least-squares in JAX and Equinox", arXiv:2311.17283 (NeurIPS 2023 AI4Science). https://arxiv.org/abs/2311.17283 ; https://arxiv.org/pdf/2311.17283 (fetched, text extracted); https://github.com/p
- Rader et al., "Optimistix: modular optimisation in JAX and Equinox", arXiv:2402.09983. https://arxiv.org/pdf/2402.09983 ; adjoints doc https://docs.kidger.site/optimistix/api/adjoints/ (fetched); https://docs.kidger.site/optimistix/api/root_find/ ; https://g
- Blondel, Berthet, Cuturi, Frostig, Hoyer, Llinares-Lopez, Pedregosa, Vert, "Efficient and Modular Implicit Differentiation", NeurIPS 2022, arXiv:2105.15183. https://arxiv.org/pdf/2105.15183 ; https://ar5iv.labs.arxiv.org/html/2105.15183
- JAX issue #15837 "GMRES Fails Silently and Frequently from Stagnation". https://github.com/jax-ml/jax/issues/15837 ; gmres docs https://docs.jax.dev/en/latest/_autosummary/jax.scipy.sparse.linalg.gmres.html ; JEP 18137 https://docs.jax.dev/en/latest/jep/1813
- torch-sla, "Differentiable Sparse Linear Algebra with Adjoint Solvers ...", arXiv:2601.13994. https://arxiv.org/pdf/2601.13994
- "Differentiate the Solver, Not the Equation: Reverse-Sweep Adjoints for Block Implicit Simulation", arXiv:2608.08559. https://arxiv.org/html/2608.08559 (title only)
- "Automating Steady and Unsteady Adjoints: Efficiently Utilizing Implicit and Algorithmic Differentiation", arXiv:2306.15243. https://arxiv.org/html/2306.15243 (title only)
- Pierce & Giles, "Adjoint recovery of superconvergent functionals from PDE approximations", SIAM Review 42(2), 247-264 (2000) (metadata from search); Giles' error-analysis page https://people.maths.ox.ac.uk/gilesm/old/error.html
- Giles & Pierce, "Adjoint Error Correction for Integral Outputs", Springer (doi 10.1007/978-3-662-05189-4_2). https://link.springer.com/chapter/10.1007/978-3-662-05189-4_2
- Giles & Pierce, "Progress in adjoint error correction for integral functionals", Comput. Vis. Sci. https://people.maths.ox.ac.uk/~gilesm/files/cvs04.pdf ; https://link.springer.com/article/10.1007/s00791-003-0115-y
- Becker & Rannacher, "An optimal control approach to a posteriori error estimation in finite element methods", Acta Numerica (2001). https://www.cambridge.org/core/journals/acta-numerica/article/abs/an-optimal-control-approach-to-a-posteriori-error-estimation
- "Linearization Errors in Discrete Goal-Oriented Error Estimation", arXiv:2305.15285. https://arxiv.org/pdf/2305.15285 (title only)
- Roache, Grid Convergence Index (secondary sources). https://cfd.university/blog/how-to-manage-uncertainty-in-cfd-the-grid-convergence-index/ ; Roy, "Grid Convergence Error Analysis for Mixed-Order Numerical Schemes" https://www.aoe.vt.edu/content/dam/aoe_vt_
- Salari & Knupp, "Code Verification by the Method of Manufactured Solutions", SAND2000-1444 (2000). https://www.osti.gov/biblio/759450/
- Roache, "Code Verification by the Method of Manufactured Solutions", J. Fluids Eng. 124(1), 4 (2002). https://asmedigitalcollection.asme.org/fluidsengineering/article-abstract/124/1/4/462791/Code-Verification-by-the-Method-of-Manufactured
- ASME V&V 20-2009 (R2021), "Standard for Verification and Validation in Computational Fluid Dynamics and Heat Transfer". https://webstore.ansi.org/standards/asme/asme2020092021
- Oberkampf & Roy, "Verification and Validation in Scientific Computing", Cambridge University Press (2010), ISBN 9780521113601. https://books.google.com/books/about/Verification_and_Validation_in_Scientifi.html?id=7d26zLEJ1FUC
- "Accurate spectral numerical schemes for kinetic equations with energy diffusion", J. Comput. Phys. (2015). https://arxiv.org/pdf/1402.2971 ; https://www.sciencedirect.com/science/article/abs/pii/S0021999115001941
- "Pseudo spectral collocation with Maxwell polynomials for kinetic equations with energy diffusion". https://arxiv.org/pdf/1708.09031
- "A Spectral Transform Method for Singular Sturm-Liouville Problems with Applications to Energy Diffusion in Plasma Physics", SIAM J. Appl. Math. https://dx.doi.org/10.1137/130941948

### Research-software practice
- Google Engineering Practices, "Small CLs" — https://google.github.io/eng-practices/review/developer/small-cls.html
- SmartBear, "Best Practices for Peer Code Review" (Cisco study) — https://smartbear.com/learn/code-review/best-practices-for-peer-code-review/
- Hoefler & Belli, "Scientific Benchmarking of Parallel Computing Systems", SC '15, DOI 10.1145/2807591.2807644 — https://htor.inf.ethz.ch/publications/img/hoefler-scientific-benchmarking.pdf
- JAX documentation, "Benchmarking JAX code" — https://docs.jax.dev/en/latest/benchmarking.html (and FAQ https://docs.jax.dev/en/latest/faq.html)
- Wilson et al. 2017, "Good enough practices in scientific computing", DOI 10.1371/journal.pcbi.1005510 — https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005510
- Taschuk & Wilson 2017, "Ten simple rules for making research software more robust", DOI 10.1371/journal.pcbi.1005412 — https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005412
- JOSS review criteria — https://joss.readthedocs.io/en/latest/review_criteria.html
- Diátaxis — https://diataxis.fr/
- Graphite, "Stacked diffs" — https://graphite.com/guides/stacked-diffs
- Trunk Based Development — https://trunkbaseddevelopment.com/
- ADR GitHub organization — https://adr.github.io/
- FAIR4RS principles, RDA output page (metadata only), DOI 10.15497/RDA00068 — https://www.rd-alliance.org/group_output/fair-principles-for-research-software-fair4rs-principles/
- Conlin & Landreman, yancc, arXiv:2607.20861 — https://arxiv.org/html/2607.20861v1
- Escoto et al., MONKES, Nucl. Fusion 64, 076030 (2024), arXiv:2312.12248 — https://arxiv.org/html/2312.12248
- Panici et al., DESC Part I, JPP 2023, arXiv:2203.17173 — https://arxiv.org/abs/2203.17173
- READMEs (raw): DESC https://raw.githubusercontent.com/PlasmaControl/DESC/master/README.rst ; simsopt https://raw.githubusercontent.com/hiddenSymmetries/simsopt/master/README.md ; Diffrax https://raw.githubusercontent.com/patrick-kidger/diffrax/main/README.md
- Roache, "Code Verification by the Method of Manufactured Solutions", ASME J. Fluids Eng. 124, 4 (2002) — https://asmedigitalcollection.asme.org/fluidsengineering/article-abstract/124/1/4/462791
- Roy, "Review of Code and Solution Verification Procedures for Computational Simulation", JCP — https://www.aoe.vt.edu/content/dam/aoe_vt_edu/people/faculty/cjroy/Publications-Articles/cjr_jcp.revise.final-accepted.pdf
- Oberkampf & Roy, Verification and Validation in Scientific Computing, CUP 2010 — https://www.cambridge.org/core/books/abs/verification-and-validation-in-scientific-computing/index/EE029CB068531D278AB2631911F8BE42
- Velasco et al., KNOSOS, J. Comput. Phys. 418, 109512 (2020) — https://www.sciencedirect.com/science/article/abs/pii/S0021999120302862 ; code https://github.com/joseluisvelasco/KNOSOS
- Landreman, Smith, Mollén, Helander, Phys. Plasmas 21, 042503 (2014) (SFINCS) — https://pubs.aip.org/aip/pop/article-abstract/21/4/042503/818401 ; https://github.com/landreman/sfincs
- Whitesides, "Whitesides' Group: Writing a Paper", Adv. Mater. 16, 1375 (2004), DOI 10.1002/adma.200400767 — https://www.gmwgroup.harvard.edu/publications/whitesides-group-writing-paper
- The Turing Way, "Software Citation with CITATION.cff" — https://book.the-turing-way.org/communication/citable/citable-cff/ ; Citation File Format — https://citation-file-format.github.io/
- ACM Artifact Review and Badging v1.1 — https://www.acm.org/publications/policies/artifact-review-and-badging-current (HTTP 403)
- FAIR4RS principle wording; Chue Hong et al., Sci. Data 9, 622 (2022), DOI 10.1038/s41597-022-01710-x (Nature redirect loop)
- ASME V&V 20-2009 scope statement (snippet only)
- GENE / GS2 / COGENT verification suites; Google test-size taxonomy; Keep a Changelog; CPC "Program summary" requirement; `jax.test_util.check_grads`
