# SFINCS_JAX Master Handoff + Execution Plan

Last updated: 2026-03-07 (America/Chicago)
Owner: incoming agent

## 1) Prompt For A New Agent (copy/paste)

```text
You are taking over sfincs_jax, a JAX rewrite/extension of SFINCS v3.

Primary mission (phase 1):
- Reproduce SFINCS v3 functionality and numerics for supported geometries and physics,
- Match outputs/diagnostics/terminal behavior to Fortran SFINCS for the same input,
- Keep default behavior robust and general (no case-specific hard-coding),
- Maintain end-to-end differentiability for JAX-native solve paths,
- Deliver high performance and memory efficiency by default,
- Keep code easy to run, easy to maintain, thoroughly validated, and deeply documented.

Primary mission (phase 2+):
- Extend beyond strict SFINCS replication toward modern neoclassical workflows,
- Integrate/benchmark alternative numerical formulations and optimization-oriented methods,
- Borrow and generalize ideas from modern tools (e.g., MONKES, KNOSOS, NEO ecosystems),
- Preserve scientific correctness while improving throughput, scalability, and usability.

Non-negotiable engineering constraints:
1) No hidden dependence on colocated Fortran outputs for correctness.
2) No brittle per-case tuning as the default path.
3) New defaults must generalize to unseen inputs and still converge robustly.
4) Every numerical/performance change must be validated (unit + regression + physics + reduced-suite comparison).
5) Documentation must explain equations, normalization, discretization, solver/preconditioner design, and code locations.

Working directories and references:
- sfincs_jax repo: /Users/rogeriojorge/local/tests/sfincs_jax
- Fortran SFINCS v3 executable: /Users/rogeriojorge/local/tests/sfincs/fortran/version3/sfincs
- Original SFINCS source tree: /Users/rogeriojorge/local/tests/sfincs_original
- MONKES reference repo: /Users/rogeriojorge/local/tests/MONKES
- Main thesis/pdf refs: /Users/rogeriojorge/local/tests/Escoto_Thesis.pdf and /Users/rogeriojorge/local/tests/*.pdf
- sfincs_jax docs upstream refs: /Users/rogeriojorge/local/tests/sfincs_jax/docs/upstream

Immediate priorities:
- Keep reduced-suite comparison fully populated and reproducible,
- Keep defaults robust for all examples (including additional examples),
- Eliminate remaining solver branch fragility while preserving differentiability,
- Reduce worst runtime/memory offenders (especially PAS-heavy paths),
- Improve practical scaling strategy (CPU cores, GPU path, cluster portability).

Execution style:
- Always profile first, change second, validate third.
- Track performance/memory deltas before and after every significant change.
- Update docs/README/plan.md in lockstep with code.
- Commit small, coherent changes frequently.
```

---

## 2) Project Goal (explicit)

Build a production-grade neoclassical transport solver in JAX that:
- solves the drift-kinetic equation in tokamak and stellarator geometries,
- reproduces SFINCS v3 equation set and normalizations (phase 1),
- is differentiable end-to-end for optimization/sensitivity workflows,
- is performant and memory-efficient by default,
- is extensible to alternative numerical methods (phase 2+).

---

## 3) Physical/Numerical Scope (phase 1)

The code should replicate SFINCS v3 behavior for:
- Geometries: `geometryScheme in {1,2,4,5,11,12}`,
- Physics options used in reduced/upstream examples (FP/PAS, Er/noEr, Phi1 variants, DKES/full trajectories where supported),
- Diagnostics and H5 output fields in `sfincsOutput.h5`,
- CLI workflow comparable to Fortran invocation (`sfincs_jax input.namelist`).

Core requirement right now:
- same equations,
- same discretization intent,
- same normalization,
- same algorithmic behavior where practical.

---

## 4) Repository + Reference Map

### 4.1 Local roots
- Workspace root: `/Users/rogeriojorge/local/tests`
- Active repo: `/Users/rogeriojorge/local/tests/sfincs_jax`
- Fortran executable: `/Users/rogeriojorge/local/tests/sfincs/fortran/version3/sfincs`
- Fortran source: `/Users/rogeriojorge/local/tests/sfincs_original`
- MONKES: `/Users/rogeriojorge/local/tests/MONKES`
- Thesis/PDF refs: `/Users/rogeriojorge/local/tests/Escoto_Thesis.pdf`, `/Users/rogeriojorge/local/tests/*.pdf`

### 4.2 sfincs_jax key code files
- Operator/system: `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/v3_system.py`
- Driver/preconditioners: `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/v3_driver.py`
- Residual/Jacobian wrappers: `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/residual.py`
- Solver kernels: `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/solver.py`
- I/O + H5 writer: `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/io.py`
- Transport diagnostics: `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/transport_matrix.py`
- Output compare helper: `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/compare.py`

### 4.3 Validation and reporting
- Reduced suite runner: `/Users/rogeriojorge/local/tests/sfincs_jax/scripts/run_reduced_upstream_suite.py`
- README table generator: `/Users/rogeriojorge/local/tests/sfincs_jax/scripts/generate_readme_reduced_suite_table.py`
- Reduced inputs: `/Users/rogeriojorge/local/tests/sfincs_jax/tests/reduced_inputs`
- Reduced outputs/report dir: `/Users/rogeriojorge/local/tests/sfincs_jax/tests/reduced_upstream_examples`
- Tests root: `/Users/rogeriojorge/local/tests/sfincs_jax/tests`

### 4.4 Examples
- Main examples: `/Users/rogeriojorge/local/tests/sfincs_jax/examples`
- Additional high-res case: `/Users/rogeriojorge/local/tests/sfincs_jax/examples/additional_examples/input.namelist`
- Prior additional input: `/Users/rogeriojorge/local/tests/sfincs_jax/examples/additional_examples/input.namelist_old`

### 4.5 Documentation
- Docs root: `/Users/rogeriojorge/local/tests/sfincs_jax/docs`
- Upstream/reference material mirrored: `/Users/rogeriojorge/local/tests/sfincs_jax/docs/upstream`

---

## 5) Current State Snapshot (as of 2026-03-05)

### 5.1 Recent validated status
- Reduced suite practical status currently reports all cases as `parity_ok` at suite tolerances.
- README reduced-suite table now included as the primary comparison section.
- README reduced-suite table was regenerated from the lane-specific suite reports so the
  checked-in table now matches the stored report artifacts.
- `write_sfincs_jax_output_h5(..., return_results=True)` now returns in-memory result dictionary for immediate inspection.

### 5.2 Known pain points that still matter
- Runtime ratio is still high for several PAS-heavy and tiny-Fortran-time cases (some ratios are inflated by very short Fortran runtimes).
- Memory ratio remains high on select large PAS/FP cases.
- Parallel strong-scaling beyond a few cores is not yet consistently strong for single-RHS large solves.

### 5.3 Product posture
- Usable and scientifically functional,
- Still in active optimization/scaling hardening phase,
- Needs continued runtime/memory and distributed-solve work to reach “best-in-class” HPC behavior.

---

## 6) What Has Been Done (high-level execution history)

Mark completed milestones as `[x]`, active as `[~]`, pending as `[ ]`.

### 6.1 Core numerical/functionality work
- [x] Matrix-free JAX operator path and v3-compatible workflow implemented.
- [x] RHSMode=1 and RHSMode=2/3 solver branches present with multiple Krylov/preconditioner options.
- [x] PAS projection/preconditioner heuristics added and iterated.
- [x] Dense fallback controls added/capped for stability/memory.
- [x] IncludePhi1/Newton behavior tuned for practical convergence on larger cases.
- [x] Removed unsafe dependency on Fortran H5 overlays for core correctness (standalone output path preserved).

### 6.2 Validation/reporting infrastructure
- [x] Reduced-suite runner supports runtime/memory/parity/print diagnostics.
- [x] README table auto-generated from suite report.
- [x] Runtime + memory columns integrated for Fortran/JAX CPU/GPU lanes.
- [x] Iteration stats plumbing exists in suite scripts/log parsing.

### 6.3 Documentation and examples
- [x] Major docs expansion (equations, models, methods, performance notes, references).
- [x] Added examples for parity, transport, autodiff, optimization, performance.
- [x] README recently simplified to focus on install + quick start + CLI + reduced-suite table.
- [x] Python quick-start now includes in-memory result access via `return_results=True`.

### 6.4 CI/CD hardening
- [x] CI and docs pipelines exist (`.github/workflows/ci.yml`, `docs.yml`).
- [x] Examples smoke and docs builds are wired.
- [~] CI runtime remains a continuing optimization target (keep broad coverage but faster scheduling).

---

## 7) Required Behavior For New Work

1. Default behavior must generalize:
   - no case-name hacks,
   - no hidden fallback to external reference files.
2. Preserve differentiability for JAX-native solve paths.
3. Keep solver choices configurable, but defaults should “just work” for unseen cases.
4. Every performance change must report:
   - runtime delta,
   - memory delta,
   - validation delta.
5. Every algorithmic change must document:
   - equation/operator impact,
   - numerics/preconditioner rationale,
   - code location.

---

## 8) Validation Strategy (must run continuously)

### 8.1 Unit tests
- Operator blocks, geometry parsing, collision terms, diagnostics.

### 8.2 Regression tests
- For each reduced example, compare JAX output H5 against Fortran output H5.

### 8.3 Physics tests
- Verify expected asymptotic scalings/symmetries/conservation behavior where available.

### 8.4 Practical comparison threshold
- Default target: `rtol=5e-4`, `atol=1e-9` (or as currently standardized in suite scripts).

### 8.5 Strict comparison mode
- Also track strict mismatch counts without case-specific tolerance relaxations.

### 8.6 Repro commands

```bash
cd /Users/rogeriojorge/local/tests/sfincs_jax
python scripts/run_reduced_upstream_suite.py \
  --fortran-exe /Users/rogeriojorge/local/tests/sfincs/fortran/version3/sfincs \
  --reuse-fortran \
  --max-attempts 1 \
  --rtol 5e-4 \
  --atol 1e-9 \
  --jax-repeats 2
python scripts/generate_readme_reduced_suite_table.py
```

Single-case debug:

```bash
python scripts/run_reduced_upstream_suite.py \
  --fortran-exe /Users/rogeriojorge/local/tests/sfincs/fortran/version3/sfincs \
  --pattern "<CASE>$" \
  --reuse-fortran \
  --max-attempts 1 \
  --rtol 5e-4 \
  --atol 1e-9
```

---

## 9) CI/CD and Quality Gates

### 9.1 CI pipelines
- Tests matrix: `/Users/rogeriojorge/local/tests/sfincs_jax/.github/workflows/ci.yml`
- Docs build: `/Users/rogeriojorge/local/tests/sfincs_jax/.github/workflows/docs.yml`

### 9.2 Required pre-merge checks
- `pytest -q` (or CI split equivalent)
- `sphinx-build -W -b html docs docs/_build/html`
- Reduced-suite refresh for solver-affecting PRs (at least targeted cases; full sweep before release)
- README table regeneration when suite report changes

### 9.3 CI speed policy
- Keep scientific coverage while reducing wall-time via:
  - split scheduling,
  - fixture sizing discipline,
  - marked heavy tests separated from fast core path,
  - cached artifacts where safe.

---

## 10) Documentation Map + MD Update Protocol

### 10.1 Core docs to maintain
- `/Users/rogeriojorge/local/tests/sfincs_jax/README.md`
- `/Users/rogeriojorge/local/tests/sfincs_jax/docs/index.rst`
- `/Users/rogeriojorge/local/tests/sfincs_jax/docs/system_equations.rst`
- `/Users/rogeriojorge/local/tests/sfincs_jax/docs/method.rst`
- `/Users/rogeriojorge/local/tests/sfincs_jax/docs/normalizations.rst`
- `/Users/rogeriojorge/local/tests/sfincs_jax/docs/performance.rst`
- `/Users/rogeriojorge/local/tests/sfincs_jax/docs/performance_techniques.rst`
- `/Users/rogeriojorge/local/tests/sfincs_jax/docs/parallelism.rst`
- `/Users/rogeriojorge/local/tests/sfincs_jax/docs/usage.rst`
- `/Users/rogeriojorge/local/tests/sfincs_jax/docs/outputs.rst`

### 10.2 Markdown files to keep coherent
- `/Users/rogeriojorge/local/tests/sfincs_jax/README.md`
- `/Users/rogeriojorge/local/tests/sfincs_jax/examples/README.md`
- Example-specific READMEs under `/Users/rogeriojorge/local/tests/sfincs_jax/examples/*/README.md`
- `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md` (this file)

### 10.3 Update protocol for this `plan.md`
After every significant work block:
1. Update "Last updated" date.
2. Move checklist items from `[ ]` -> `[~]` -> `[x]`.
3. Add a short changelog entry under Section 16.
4. Record measured runtime/memory/parity deltas.
5. Add/refresh references if decisions used new literature/sources.

---

## 11) Competitor / Ecosystem Landscape

This project sits in a rapidly evolving fusion-computation ecosystem.

### 11.1 Relevant neoclassical or adjacent tools
- SFINCS (Fortran v3 baseline to replicate first)
- MONKES (fast monoenergetic coefficients, optimization-oriented)
- NEO (GACODE multispecies drift-kinetic solver ecosystem)
- KNOSOS (fast orbit-averaging stellarator neoclassical solver)
- STELLOPT tooling around stellarator optimization workflows

### 11.2 Why this matters for sfincs_jax
- Need robust, differentiable, optimization-friendly neoclassical kernels.
- Need interoperable, modern workflows (Python/JAX/HPC) while preserving first-principles fidelity.
- Need portability across laptop CPU, workstation GPU, and clusters (NERSC/Slurm).

---

## 12) Market Pull / Strategic Need (online snapshot)

The demand signal for production-grade fusion simulation software is rising due to:
- growth of private-sector fusion investment,
- national-scale public funding programs,
- open-source integrated modelling pushes,
- increasing HPC/GPU availability for high-fidelity predictive workflows.

Evidence (primary/public sources):
- IAEA World Fusion Outlook 2024 emphasizes global R&D growth, timelines, and public/private investment trends.
- U.S. DOE expanded FIRE + milestone-backed commercial fusion programs and reports milestone progress/funding leverage.
- Fusion Industry Association 2024 report indicates >$7.1B total private funding to date and 45 company responses.
- ITER released IMAS infrastructure/physics models as open source (2025), indicating a strong ecosystem trend toward open, interoperable modelling stacks.

Implication for sfincs_jax:
- there is clear pull for tools that are rigorous enough for physics validation and fast enough for iterative design/optimization.

---

## 13) Parallelization Target Context

### 13.1 Local target
- Efficient multi-core scaling on MacBook (user-level default usability).

### 13.2 Cluster target
- NERSC Perlmutter compatibility:
  - CPU-only and GPU node workflows,
  - Slurm-friendly execution,
  - robust scaling model for many-core / many-GPU execution.

Perlmutter references indicate heterogeneous CPU/GPU architecture and high-parallel-concurrency workflows.

---

## 14) Roadmap

### 14.1 Short-term (next 1-3 weeks)
- [ ] Ensure all reduced-suite rows are complete for CPU and GPU lanes (no missing runtime/memory cells).
- [ ] Re-run additional high-resolution example on CPU+GPU and integrate into comparison reporting.
- [ ] Close remaining worst runtime/memory offenders (especially PAS-heavy cases) while preserving tolerances.
- [~] Strengthen default PAS preconditioner path to avoid expensive fallback branches where possible.
- [ ] Keep docs and README synchronized with measured reality (no stale claims).
- [ ] Keep CI wall-time under control without reducing scientific coverage.

### 14.2 Medium-term (1-3 months)
- [ ] Implement stronger generalized domain-decomposition preconditioners for large RHSMode=1 systems.
- [ ] Improve communication-avoiding Krylov behavior for stronger multi-core/multi-device scaling.
- [ ] Stabilize one-node multi-GPU strategy for large-case throughput.
- [ ] Add benchmark suite for representative 2-4 minute cases (warm/cold timing and memory baselines).
- [ ] Add explicit solver-path provenance in logs/output metadata.

### 14.3 Long-term (3-12 months)
- [ ] Extend beyond strict SFINCS replication: broader equation/model options and modern numerical variants.
- [ ] Integrate MONKES-inspired fast monoenergetic pathways where scientifically consistent.
- [ ] Build coupled optimization workflows (profile/equilibrium loops) using implicit-diff where beneficial.
- [ ] Mature multi-node scaling strategy for Slurm (dozens/hundreds of workers) with robust defaults.
- [ ] Publish formal method/performance validation notes with reproducible artifacts.

---

## 15) Execution Checklist (live)

### 15.1 Always-on loop
- [ ] Use the original Fortran v3 example inputs as the resolution reference for example-suite benchmarking; do not use blind `2x` enlargement as the default benchmark mode.
- [ ] Benchmark CPU/GPU JAX lanes against a fixed CPU-generated Fortran reference root when machine-local Fortran outputs are not proven deterministic.
- [ ] For `constraintScheme=0` reference generation, force a stable Fortran Krylov solve (`PETSC_OPTIONS='-ksp_type gmres -pc_type none'`) unless an explicit PETSc override is requested.
- [ ] Pick top 1-2 offenders from latest report (runtime and memory separately).
- [ ] Profile (`SFINCS_JAX_PROFILE=1`) and isolate dominant phase.
- [ ] Implement smallest high-ROI change.
- [ ] Re-run targeted case(s), verify tolerances and print diagnostics.
- [ ] Re-run reduced-suite subset, then full suite when stable.
- [ ] Regenerate table + docs + this plan.

### 15.2 "Do not regress" list
- [ ] Differentiability on JAX-native solver paths.
- [ ] Standalone behavior (no hidden Fortran-output dependencies).
- [ ] Robust defaults for unseen inputs.
- [ ] CI/doc builds passing.

---

## 16) Changelog Entries For Future Agent Updates

Use this template and append newest at top:

```text
### YYYY-MM-DD
- Scope:
- Files changed:
- Validation run:
- Runtime/memory delta:
- Remaining risks:
- Next actions:
```

Current latest notable changes before this handoff:
- README simplified; quick-start now includes in-memory results API.
- `write_sfincs_jax_output_h5(..., return_results=True)` added.
- Reduced-suite runner now retries after JAX exceptions with resolution reduction before final `jax_error`.

### 2026-03-07
- Scope: Rework the large-CPU explicit RHSMode=1 FP fallback so the default CLI lane skips the wasteful initial/stage2 collision-GMRES on the geometry-4 blocker, assembles host x-block factors sparsely with cached operator pieces, and only enables the experimental per-L `sxblock_tz` rescue behind an explicit env opt-in.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/v3_driver.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_rhs1_sparse_first_heuristic.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_sparse_assembly.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: `python -m py_compile sfincs_jax/v3_driver.py tests/test_rhs1_sparse_first_heuristic.py`; `pytest -q tests/test_sparse_assembly.py tests/test_rhs1_sparse_first_heuristic.py tests/test_cli_solve_mode.py`
- Runtime/memory delta: on `examples/sfincs_examples/geometryScheme4_2species_withEr_fullTrajectories`, the default explicit CPU lane now skips the old `~156-209s` initial collision-preconditioned GMRES and reaches the explicit x-block seed in about `30-32s` total (`~8s` x-block build + `~22-24s` bounded solve), with peak RSS around `1.7-1.9 GB` before any later full sparse rescue. The experimental `sxblock_tz` seed path was reduced from about `9.9 GB` RSS to about `3.5-3.6 GB` by switching to sequential per-L factorization with smaller submatrix batches, but it still produced a poor seed (`residual≈1.95e+01`) and therefore remains off by default.
- Remaining risks: the geometry-4 large-FP default explicit lane still falls through to the full `68670x68670` sparse rescue because the explicit x-block factors remain too weak on nonzero-`x` blocks; simply adding more fallback branches is not closing the parity/performance gap.
- Next actions: inspect the rejected nonzero-`x` host factors directly and compare against the Fortran v3 matrix-preconditioner design, then replace the current per-`x` explicit rescue with a stronger x-coupled explicit block strategy before rerunning the full example suite.

### 2026-03-07
- Scope: Split explicit and differentiable solve modes so CLI/output generation can take a fast non-implicit path by default, while keeping the implicit-diff path available explicitly; add a host sparse x-block rescue implementation for explicit RHSMode=1 FP solves and use it only on the non-differentiable path.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/cli.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/io.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/v3_driver.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_cli_solve_mode.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_rhs1_sparse_first_heuristic.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: `python -m py_compile sfincs_jax/v3_driver.py sfincs_jax/io.py sfincs_jax/cli.py tests/test_rhs1_sparse_first_heuristic.py tests/test_cli_solve_mode.py`; `pytest -q tests/test_rhs1_sparse_first_heuristic.py tests/test_cli_solve_mode.py`
- Runtime/memory delta: on the original-resolution geometry-4 CPU blocker (`examples/sfincs_examples/geometryScheme4_2species_withEr_fullTrajectories`), the explicit host x-block preconditioner reduced peak RSS at x-block build completion from about `5648.7 MB` on the capped JAX-factor path to about `5050.4 MB` at comparable build time (`~117-118s`), but the explicit GMRES rescue still did not finish in a practical wall-clock window; a follow-up experiment that also switched the Krylov matvec to a host sparse operator drove CPU utilization to about `850%` but increased RSS to about `8.3 GB`, so that variant was not kept.
- Remaining risks: the CLI/default explicit lane is now correctly separated from the differentiable path, but the geometry-4 large-FP explicit rescue is still too slow and memory-heavy; the next fix should target a cheaper strong explicit rescue rather than growing the host sparse operator cache.
- Next actions: commit/push the explicit/differentiable split and test coverage on `main`, then continue the geometry-4 work by replacing the current explicit x-block GMRES rescue with a more memory-disciplined strong explicit solve path before rerunning the original-resolution CPU suite.

### 2026-03-06
- Scope: Fix legacy mixed-gradient handling by separating species-gradient and Phi-gradient coordinate inference in the JAX solve/output paths, so cases that specify `dNHatdrHats`/`dTHatdrHats` together with `Er` reproduce Fortran v3 instead of silently zeroing the electric field branch.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/input_compat.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/io.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/v3_system.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_input_compat.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: `python -m py_compile sfincs_jax/input_compat.py sfincs_jax/v3_system.py sfincs_jax/io.py tests/test_input_compat.py`; `pytest -q tests/test_input_compat.py tests/test_fortran_reference_solver_options.py tests/test_sparse_assembly.py`; targeted suite rerun in `/Users/rogeriojorge/local/tests/sfincs_jax/tests/debug_geometry5_3species_split_grad_v1`.
- Runtime/memory delta: `examples__sfincs_examples__geometryScheme5_3species_loRes` moved from `parity_mismatch` (`42/193` practical and strict, JAX `277.618s`, `4795.1 MB`) to `parity_ok` (`0/193` practical and strict, `9/9` print parity) at JAX `134.684s` and `4775.1 MB`; the Fortran reference lane on the corrected input is `21.506s`, `582.7 MB`.
- Remaining risks: the stale full CPU suite roots created before this mixed-gradient fix are invalid for any mixed legacy-gradient cases and should not be used as frozen references; runtime/memory on geometry5 remain materially above Fortran even though parity is restored.
- Next actions: commit/push this fix on `main`, rerun the full original-resolution CPU suite plus the additional example from a clean root, then use that frozen CPU reference root for the full office GPU suite before regenerating README tables.

### 2026-03-06
- Scope: Fix the Fortran v3 canonicalization path so modern v3 inputs keep their trailing newline, preventing false `&export_f` read failures in the scaled-suite reference lane after the legacy-input compatibility work.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/scripts/run_reduced_upstream_suite.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_fortran_reference_solver_options.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: `python -m py_compile scripts/run_reduced_upstream_suite.py`; `pytest -q tests/test_fortran_reference_solver_options.py tests/test_input_compat.py tests/test_scaled_example_suite_reference.py`; single-case suite rerun in `/Users/rogeriojorge/local/tests/sfincs_jax/tests/debug_case_runner_tokamak_pas_nx1_v2`.
- Runtime/memory delta: the representative original-resolution case `examples__sfincs_examples__tokamak_1species_PASCollisions_noEr_Nx1` moved from `max_attempts` with a Fortran `export_f` parse failure back to `parity_ok` at the original seed (`0/212` practical and strict, `9/9` print parity) with no resolution reduction.
- Remaining risks: the partial full-suite root `scaled_example_suite_ref_cpu_full_v6` is invalid because it was started on the broken reference lane and then interrupted; the full CPU sweep still needs to be restarted from scratch on current `main`.
- Next actions: commit/push the canonicalization fix on `main`, rerun the full original-resolution CPU suite plus the additional example from scratch, then continue to the frozen-reference GPU lane.

### 2026-03-06
- Scope: Add systematic legacy-input compatibility for the pre-v3 `examples/upstream/fortran_multispecies` tree by translating old namelist groups/keys for the Fortran reference lane, teaching `sfincs_jax` to infer non-default gradient-coordinate semantics from legacy inputs, and honoring legacy Boozer-file and `normradius_wish` aliases in the output, solve, and terminal-print paths.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/scripts/run_reduced_upstream_suite.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/input_compat.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/io.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/v3.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/v3_system.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/v3_fblock.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_fortran_reference_solver_options.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_input_compat.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/examples/README.md`, `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: `python -m py_compile sfincs_jax/input_compat.py sfincs_jax/io.py sfincs_jax/v3.py sfincs_jax/v3_system.py sfincs_jax/v3_fblock.py scripts/run_reduced_upstream_suite.py`; `pytest -q tests/test_input_compat.py tests/test_fortran_reference_solver_options.py tests/test_scaled_example_suite_reference.py`; targeted suite reruns in `/Users/rogeriojorge/local/tests/sfincs_jax/tests/debug_scaled_multispecies_inductive_suite_v7` and `/Users/rogeriojorge/local/tests/sfincs_jax/tests/debug_scaled_multispecies_fp_suite_v7`.
- Runtime/memory delta: the old multispecies `inductiveE_noEr` and `quick_2species_FPCollisions_noEr` cases moved from `max_attempts` reference-generation failure to `parity_ok` at the original-resolution seed (`0/193` practical and strict, `9/9` print parity); on local CPU the translated Fortran lane takes about `21.7s` and `119 MB` RSS on each case, while `sfincs_jax` takes about `4.9s` and `560 MB` RSS.
- Remaining risks: the large legacy geometryScheme=11 multispecies cases still need a full end-to-end parity pass on current `main`; the stale full original-resolution CPU suite was intentionally killed because these compatibility fixes changed both the runner and the JAX semantics underneath it.
- Next actions: commit/push this legacy-input compatibility block on `main`, restart the full original-resolution CPU suite plus the additional example from scratch, then use that frozen CPU reference root for the full office GPU suite before regenerating the README tables.

### 2026-03-06
- Scope: Tighten the CPU transport sparse-LU direct rescue by adding iterative residual refinement and raising the default rescue-size cap so the original-resolution LHD and low-collisionality W7-X transport-matrix examples converge on the sparse-direct parity branch instead of stalling in large Krylov retries.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/v3_driver.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_transport_sparse_direct.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: `python -m py_compile sfincs_jax/v3_driver.py tests/test_transport_sparse_direct.py`; `pytest -q tests/test_transport_sparse_direct.py tests/test_transport_parallel.py tests/test_transport_matrix_rhsmode2_parity.py`; targeted repros in `/Users/rogeriojorge/local/tests/sfincs_jax/tests/debug_lhd_co0_nu_0_5748_refine_v1` and `/Users/rogeriojorge/local/tests/sfincs_jax/tests/debug_w7x_co0_nu_0_01727_sparse_v1`.
- Runtime/memory delta: for `examples__publication_figures__output__lhd_co0__nu_n_0.5748`, transport residuals improved from the earlier sparse-direct lane at roughly `6e-06`-`4e-05` down to `4.62e-17`, `1.14e-15`, and `2.28e-12` after two LU-refinement steps, with transport elapsed about `267.98s` and RSS about `4228 MB`; for `examples__publication_figures__output__w7x_co0__nu_n_0.01727`, raising the sparse-direct cap from `30000` to `40000` moved the case off the stalled Krylov branch (`~1612.8s`, `37/194` mismatches) onto a machine-precision sparse-direct branch (`6.89e-19`, `1.64e-18`, `9.89e-14`) at about `748.19s`, with only metadata-only compare deltas remaining and RSS about `5149 MB`.
- Remaining risks: W7-X geometry-5 transport memory remains several GB above Fortran due to SuperLU fill; the full original-resolution CPU suite still needs to be rerun from scratch on this new default before freezing the reference root for the full office GPU lane.
- Next actions: commit/push this transport refinement block on `main`, rerun the full original-resolution CPU suite plus the additional example from scratch, then run the full office GPU suite against that frozen CPU reference root before regenerating README tables.

### 2026-03-06
- Scope: Add a CPU transport sparse-LU direct rescue with rescue-first ordering for large RHSMode=2/3 FP transport solves, so stalled transport Krylov branches can recover Fortran-like accuracy on the original geometry-scheme-2 transport example without spending most of the wall time in failed retry branches.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/v3_driver.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_transport_sparse_direct.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: `python -m py_compile sfincs_jax/v3_driver.py tests/test_transport_sparse_direct.py`; `pytest -q tests/test_transport_sparse_direct.py tests/test_transport_parallel.py tests/test_transport_matrix_rhsmode2_parity.py`; targeted transport repros in `tests/debug_transport_scheme2_default_v5` and `tests/debug_transport_scheme2_default_v6`.
- Runtime/memory delta: for `transportMatrix_geometryScheme2`, the original current-`main` sequential transport lane was about `773.9s` with large practical mismatches, the first sparse-LU rescue lane restored practical parity but still took about `720.2s`, and the rescue-first sparse-LU default dropped that to about `325.5s` while keeping transport-matrix max relative error at about `1.74e-5`; peak RSS rose from about `876 MB` on the inaccurate Krylov lane to about `4391 MB` on the accurate sparse-LU lane.
- Remaining risks: transport memory is still far above Fortran on this case because SuperLU fill remains large; the full original-resolution CPU suite and the office GPU suite still need to be rerun from scratch on this revision.
- Next actions: commit/push this transport sparse-LU rescue block on `main`, restart the full original-resolution CPU suite from scratch, then run the full office GPU suite against that frozen CPU reference root before regenerating README tables.

### 2026-03-06
- Scope: Fix transport `whichRHS` process-parallel diagnostics by merging parent-side state vectors through the common batched output path, stop auto-enabling transport process parallelism via the high-level cores knob, and add a chunked RHSMode=1 sparse-LU rescue path with rescue-first ordering for large catastrophic CPU FP cases.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/__init__.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/cli.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/v3_driver.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_transport_parallel.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_rhs1_sparse_first_heuristic.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_sparse_assembly.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: `python -m py_compile sfincs_jax/v3_driver.py tests/test_rhs1_sparse_first_heuristic.py`; `pytest -q tests/test_rhs1_sparse_first_heuristic.py tests/test_sparse_assembly.py tests/test_transport_parallel.py tests/test_transport_matrix_rhsmode2_parity.py`; targeted local repros for `geometryScheme5_3species_loRes` in `tests/debug_geometry5_3species_sparsefirst_v3` and `tests/debug_geometry5_3species_default_v4`.
- Runtime/memory delta: `geometryScheme5_3species_loRes` moved from the prior failing lane (`residual=5.860420e+04`, about `363.7s`, about `2143 MB`) to a converged sparse-LU rescue on the default path with residual `7.669738e-10`, about `204.6s`, and about `3697 MB` RSS; the same rescue path without redundant JAX sparse-factor materialization ran at about `182.0s` and about `3831 MB`. Practical parity stayed within the existing comparison tolerance, with representative transport/flow deltas below `~5.2e-7` relative.
- Remaining risks: `transportMatrix_geometryScheme2` is still being rerun on current `main`; the large CPU sparse rescue still allocates several GB on W7-X geometry-5 and needs further memory reduction to approach Fortran behavior.
- Next actions: finish the fresh `transportMatrix_geometryScheme2` rerun, commit/push this solver block on `main`, then restart the full original-resolution CPU suite from scratch and use that frozen root for the full office GPU rerun before regenerating the README tables.

### 2026-03-06
- Scope: Skip the expensive accelerator dense-polish branch after a successful host sparse-LU direct rescue when the remaining residual is already within a bounded ratio of the solve target, so small full-size GPU FP cases keep parity without paying for unnecessary dense Krylov cleanup.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/v3_driver.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: `python -m py_compile sfincs_jax/v3_driver.py`; `pytest -q tests/test_rhs1_sparse_first_heuristic.py tests/test_solver_gmres.py tests/test_fortran_reference_solver_options.py tests/test_scaled_example_suite_reference.py`; office GPU reruns of `inductiveE_noEr` into `tests/gating_gpu_inductive_v2` and `tests/gating_gpu_inductive_v2_nodense` against the frozen CPU reference root.
- Runtime/memory delta: on office GPU, `inductiveE_noEr` stays `0/207` practical and strict with full `9/9` print parity when the post-sparse dense fallback is skipped, while JAX runtime drops from `65.995s` to `34.366s`; RSS stays roughly flat at `1745.7 MB -> 1739.1 MB`.
- Remaining risks: this optimization is validated on the main full-size E_parallel FP blocker but still needs the wider GPU gate to confirm it does not hide useful dense-polish on other small accelerator FP cases.
- Next actions: commit/push this runtime optimization, rerun the full narrow GPU gate against the frozen CPU reference root, and then move to the remaining GPU/CPU runtime and memory offenders from the updated summaries.

### 2026-03-06
- Scope: Add a full-size RHSMode=1 sparse LU/ILU rescue path before dense fallback, widen exact sparse-LU auto selection for small accelerator FP cases, and add a host sparse-LU direct fallback for accelerator exact-LU rescues so full-size GPU FP solves no longer depend on the inaccurate explicit dense Krylov branch.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/v3_driver.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_rhs1_sparse_first_heuristic.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: `python -m py_compile sfincs_jax/v3_driver.py`; `pytest -q tests/test_rhs1_sparse_first_heuristic.py tests/test_solver_gmres.py tests/test_fortran_reference_solver_options.py tests/test_scaled_example_suite_reference.py`; targeted local `inductiveE_noEr` direct probes with forced sparse exact LU and forced host sparse direct rescue.
- Runtime/memory delta: on local CPU, the new full-size sparse exact-LU host-direct rescue returns `inductiveE_noEr` to `0/207` practical mismatches against the stable Fortran reference with residual `1.322041e-07` and about `19.9s` elapsed, replacing the earlier bad accelerator-style dense-Krylov branch that produced `41/207` mismatches on office GPU at about `69s`.
- Remaining risks: office GPU validation is still pending on this exact patch; the host sparse direct fallback is a robustness rescue path and should remain secondary to fully JAX-native solves where those already converge cleanly.
- Next actions: push this patch to `main`, rerun `inductiveE_noEr` and the narrow GPU gate from a fresh office checkout against the frozen CPU reference root, then use the updated gate report to decide whether the next performance/memory work should target PAS-heavy cases or remaining FP GPU branches.

### 2026-03-06
- Scope: Stabilize `constraintScheme=0` reference generation by forcing a reproducible Fortran Krylov policy in the suite runner, add an explicit left-preconditioned SciPy GMRES helper for solver debugging, and disable default RHSMode=1 dense shortcut/fallback paths for `constraintScheme=0` so the JAX lane stays on the physically correct sparse branch.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/scripts/run_reduced_upstream_suite.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/solver.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/v3_driver.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_rhs1_sparse_first_heuristic.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_solver_gmres.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_fortran_reference_solver_options.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: `python -m py_compile scripts/run_reduced_upstream_suite.py sfincs_jax/solver.py sfincs_jax/v3_driver.py`; `pytest -q tests/test_solver_gmres.py tests/test_rhs1_sparse_first_heuristic.py tests/test_fortran_reference_solver_options.py tests/test_scaled_example_suite_reference.py`; single-case stable reference compare in `/Users/rogeriojorge/local/tests/sfincs_jax/tests/gating_example_cpu_cs0_stable_ref_v3`.
- Runtime/memory delta: `tokamak_1species_FPCollisions_noEr` now follows the stable sparse branch against the forced-Fortran reference instead of the incorrect dense gauge-drift branch; the remaining delta is down to a single `pressureAnisotropy` mismatch (`1/188` practical and strict) rather than the earlier large density/pressure gauge errors from the dense shortcut.
- Remaining risks: `constraintScheme=0` still has a small residual branch difference in `pressureAnisotropy`; the full corrected CPU/GPU example suites have not yet been rerun from this new stable reference policy.
- Next actions: commit this solver/reference change on `main`, rerun the corrected CPU gate and full original-resolution reference lane, then rerun the office GPU lane against that frozen CPU reference root before widening back to the full examples plus additional examples.

### 2026-03-06
- Scope: Fix the GPU DKES sparse-shortcut trigger so it keys off the user-requested preconditioner setting rather than the later auto-mutated internal `rhs1_precond_env`, and confirm the office GPU log now skips the old `xblock_tz` plus stage-2 prefix.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/v3_driver.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: `python -m py_compile sfincs_jax/v3_driver.py`; office direct GPU DKES repro on clean checkout `0299b9c` with `CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 ~/venvs/sfincs_jax_gpu/bin/python -u -m sfincs_jax ...`.
- Runtime/memory delta: the live office GPU DKES log now enters `GPU DKES auto mode -> sparse ILU shortcut`, skips the initial Krylov solve entirely, and avoids the prior `xblock_tz` plus stage-2 prefix. At the same wall-clock point the process RSS dropped from about `1.58 GB` on the old path to about `1.37 GB` on the shortcut path while holding similar GPU memory (~`12.2 GB`).
- Remaining risks: the sparse-ILU solve itself still did not finish quickly enough to produce an H5/output comparison in the direct office rerun, so the new blocker is the sparse-ILU solve quality/runtime rather than the accelerator dense-fallback path.
- Next actions: instrument the sparse-ILU solve itself (residual/iteration/elapsed checkpoints), compare it against a direct dense-Krylov GPU rescue on this moderate-size DKES case, and only then rerun the GPU gate plus full examples suite.

### 2026-03-06
- Scope: Short-circuit the GPU FP DKES auto path directly to sparse ILU when that is already the intended rescue path, instead of first paying for `xblock_tz` plus stage-2 GMRES on accelerator backends.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/v3_driver.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: `python -m py_compile sfincs_jax/v3_driver.py`; `pytest -q tests/test_solver_gmres.py tests/test_small_regularized_lstsq.py tests/test_schur_precond_heuristic.py tests/test_pas_projection_heuristic.py tests/test_xblock_tz_precond_heuristic.py`; office direct GPU DKES repro on commit `5671004` confirmed the previous auto path was spending time in `xblock_tz`, then stage-2 GMRES, and only afterwards entering sparse ILU.
- Runtime/memory delta: before this patch the direct office GPU DKES repro built `xblock_tz`, reported a stage-2 residual of `1.231e-01`, then assembled sparse ILU and stayed resident at about `1.58 GB` host RSS / `12.2 GB` GPU memory before any output H5 was produced; the new shortcut is intended to remove that dead preconditioner/stage2 prefix entirely.
- Remaining risks: the actual office runtime/parity delta for the shortcut still needs to be measured on the rerun; `constraintScheme=0` remains an open nullspace/near-nullspace selection problem.
- Next actions: push this shortcut to `main`, rerun the direct office GPU DKES case from the clean checkout, and if the sparse-ILU-first path is still not parity-clean then tune the sparse ILU / dense-Krylov handoff rather than reintroducing accelerator dense-direct branches.

### 2026-03-06
- Scope: Add an accelerator-safe explicit dense-Krylov RHSMode=1 fallback path, keep dense fallback enabled on non-CPU backends without re-enabling CUDA direct solves, and validate that the CPU FP DKES lane stays parity-clean while the remaining `constraintScheme=0` FP mismatch remains isolated.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/solver.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/v3_driver.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_solver_gmres.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: `python -m py_compile sfincs_jax/solver.py sfincs_jax/v3_driver.py`; `pytest -q tests/test_solver_gmres.py tests/test_small_regularized_lstsq.py tests/test_scaled_example_suite_reference.py tests/test_schur_precond_heuristic.py tests/test_pas_projection_heuristic.py tests/test_xblock_tz_precond_heuristic.py`; targeted CPU gate into `/Users/rogeriojorge/local/tests/sfincs_jax/tests/gating_cpu_solverfix` for `tokamak_1species_FPCollisions_noEr` and `tokamak_1species_FPCollisions_withEr_DKESTrajectories` against `/Users/rogeriojorge/local/tests/sfincs_jax/tests/gating_reference_cpu`.
- Runtime/memory delta: on local CPU the FP DKES gate stayed parity-clean (`0/214`) while the `constraintScheme=0` FP case stayed at the same mismatch signature (`1/188` practical, `8/188` strict), confirming the new fallback path did not perturb the already-good CPU DKES lane and did not hide the remaining nullspace-selection problem.
- Remaining risks: the new dense-Krylov fallback still needs office GPU validation on `inductiveE_noEr` and `tokamak_1species_FPCollisions_withEr_DKESTrajectories`; `tokamak_1species_FPCollisions_noEr` still requires a principled `constraintScheme=0` solver/gauge selection change rather than more tolerance or fallback tuning.
- Next actions: sync `main` to a clean office GPU working copy and rerun the narrow GPU gate against the fixed CPU reference root, then use the resulting behavior to decide whether the remaining FP DKES issue is solved by dense-Krylov rescue alone or still needs stronger reduced-system preconditioning before returning to the `constraintScheme=0` branch.

### 2026-03-05
- Scope: Separate unsafe accelerator dense solves from the optional host-LU dense fallback so the GPU DKES path can be probed without re-enabling backend cuSOLVER calls, and verify whether the existing host-callback dense fallback is actually usable on office CUDA.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/v3_driver.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: `python -m py_compile sfincs_jax/v3_driver.py`; `pytest -q tests/test_schur_precond_heuristic.py tests/test_pas_projection_heuristic.py tests/test_xblock_tz_precond_heuristic.py`; office GPU direct reduced-resolution DKES probe with `SFINCS_JAX_RHSMODE1_DENSE_HOST_LU=1` against `tests/gating_gpu_from_ref_v3/tokamak_1species_FPCollisions_withEr_DKESTrajectories/input.namelist`.
- Runtime/memory delta: the explicit host-LU probe still finishes the reduced GPU DKES case in about `164.4s` with the same large residual (`6.107661e-02`) and similar RSS (`~946 MB` resident while running), so there is no practical runtime win yet.
- Remaining risks: the existing host-LU dense fallback path is not accelerator-safe on office CUDA either; it fails with `UNIMPLEMENTED: xla_ffi_python_gpu_callback for platform CUDA` once the reduced DKES solve reaches the dense fallback. This leaves the GPU DKES branch dependent on Krylov + sparse ILU alone, which is not yet parity-accurate.
- Next actions: either implement a new accelerator-safe host dense solve path that does not rely on `jax.pure_callback`/`custom_linear_solve` on CUDA, or improve the reduced RHSMode=1 FP Krylov path enough that the DKES branch no longer needs dense rescue at all.

### 2026-03-05
- Scope: Keep all active work on `main`, remove a full-size RHSMode=1 accelerator regression that skipped stage-2 GMRES without any real rescue path, preserve actual JAX subprocess failures in suite logs/max-attempts summaries, and disable the small full-preconditioner auto-dense path on accelerators after reproducing a CUDA `cusolver_getrf_ffi` failure on the FP DKES gate.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/v3_driver.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/scripts/run_reduced_upstream_suite.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: merged PR #1 into `main`; `python -m py_compile sfincs_jax/v3_driver.py scripts/run_reduced_upstream_suite.py`; `pytest -q tests/test_small_regularized_lstsq.py tests/test_scaled_example_suite_reference.py tests/test_schur_precond_heuristic.py tests/test_pas_projection_heuristic.py tests/test_xblock_tz_precond_heuristic.py`; `pytest -q tests/test_schur_precond_heuristic.py tests/test_pas_projection_heuristic.py tests/test_xblock_tz_precond_heuristic.py`; office GPU rerun of `inductiveE_noEr` into `/home/rjorge/sfincs_jax_codex_scaled_suite_20260305_lean/tests/gating_gpu_inductive_fix`; direct office GPU repro of `tokamak_1species_FPCollisions_withEr_DKESTrajectories` with `sfincs_jax -v write-output`.
- Runtime/memory delta: `inductiveE_noEr` on office GPU moved from the bad large-residual branch (`42/207` mismatches, `565.432s / 934.7 MB` in `gating_gpu_from_ref_v3`) back to the small residual-parity mismatch lane (`2/207` mismatches, `152.080s / 949.7 MB` in `gating_gpu_inductive_fix`). The direct DKES GPU repro no longer needs guesswork: before the latest patch it failed immediately on the small-system auto-dense full-preconditioner path with `UNIMPLEMENTED: cusolver_getrf_ffi for platform CUDA`.
- Remaining risks: `tokamak_1species_FPCollisions_withEr_DKESTrajectories` still needs a completed GPU rerun after the full-preconditioner dense-auto guard to confirm parity/performance on the Krylov path; `tokamak_1species_FPCollisions_noEr` remains a genuine `constraintScheme=0` nullspace-selection problem, not a convergence or dense-fallback issue. State-space analysis shows large unconstrained density/pressure/parallel-flow components, and removing those three expected null modes alone still leaves an additional local FP branch (`pressureAnisotropy` and local density/pressure errors remain too large).
- Next actions: finish the office GPU DKES rerun on the patched solver and then rerun the 3-case GPU gate from the fixed CPU reference root; continue the `constraintScheme=0` work by building a general nullspace-basis analysis/projection from the solved state rather than tuning solver tolerances or using Fortran-output-driven corrections.

### 2026-03-05
- Scope: Harden RHSMode=1 accelerator behavior by disabling non-CPU dense shortcut/fallback paths that still hit unsupported CUDA calls, fix the full-size strong-preconditioner fallback control flow for non-`point` FP solves, and re-run targeted CPU/GPU gate cases against a fixed CPU-generated Fortran reference root.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/v3_driver.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: `pytest -q tests/test_small_regularized_lstsq.py tests/test_scaled_example_suite_reference.py tests/test_schur_precond_heuristic.py tests/test_pas_projection_heuristic.py tests/test_xblock_tz_precond_heuristic.py`; `python -m py_compile sfincs_jax/v3_driver.py`; local CPU gate rerun into `/Users/rogeriojorge/local/tests/sfincs_jax/tests/gating_cpu_from_ref_v2`; targeted local debug reruns for `tokamak_1species_FPCollisions_noEr` with default CPU path, `SFINCS_JAX_ACTIVE_DOF=0`, and forced `SFINCS_JAX_RHSMODE1_STRONG_PRECOND=theta_line`.
- Runtime/memory delta: with the accelerator-safe sparse preference, `inductiveE_noEr` on office GPU no longer dies in CUDA dense/`lstsq` fallback and its solve log drops from the earlier `155.581s / 1033.2 MB` gate result to about `32s / 934.2 MB` for the completed standalone rerun before compare. On local CPU, `tokamak_1species_FPCollisions_withEr_DKESTrajectories` is now parity-clean against the fixed reference root (`0/214`) at `2.793s / 2282.5 MB`, while `tokamak_1species_FPCollisions_noEr` remains parity-mismatched even after a forced full-size strong fallback (`1/188` practical, `8/188` strict; about `28.6s / 1279.4 MB` with `theta_line`).
- Remaining risks: the office GPU `gating_gpu_from_ref_v2` rerun is stuck on the DKES case in a stale remote working copy and should be discarded; `tokamak_1species_FPCollisions_noEr` is now isolated as a `constraintScheme=0` FP nullspace-selection issue rather than a dense-fallback or generic convergence issue; the optional Fortran-gauge hook in `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/io.py` is only useful for debugging and must not become part of the correctness path.
- Next actions: sync a clean remote working copy with the latest local `v3_driver.py` before rerunning the GPU gate; add a narrow regression test for full-size non-`point` strong fallback reachability; debug `constraintScheme=0` FP state-vector/nullspace differences against Fortran (likely via exported state vectors or low-order-moment basis analysis) before changing default gauge behavior.

### 2026-03-05
- Scope: Remove the known GPU `lstsq` blocker with a backend-safe differentiable small least-squares path, add explicit reuse of fixed Fortran reference roots in the scaled example suite, and run a narrow local-CPU plus office-GPU gate against the same CPU-generated Fortran reference set.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/v3_driver.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/scripts/run_scaled_example_suite.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_small_regularized_lstsq.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_scaled_example_suite_reference.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/examples/README.md`, `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: `pytest -q tests/test_small_regularized_lstsq.py tests/test_scaled_example_suite_reference.py tests/test_schur_precond_heuristic.py tests/test_pas_projection_heuristic.py tests/test_xblock_tz_precond_heuristic.py`; local gate reference generation with `tests/gating_reference_cpu`; local CPU re-run against `--reference-results-root tests/gating_reference_cpu` into `tests/gating_cpu_from_ref`; office GPU re-run against the synced `tests/gating_reference_cpu` root into `tests/gating_gpu_from_ref`.
- Runtime/memory delta: the GPU gate now completes `inductiveE_noEr` instead of failing in CUDA dense/`lstsq` fallback paths, but it still takes `155.581s / 1033.2 MB` versus the local CPU reference lane `20.669s / 1098.0 MB` and Fortran `0.175s / 125.4 MB`. Local CPU against the fixed reference root stays aligned with the direct reference lane: `tokamak_1species_PASCollisions_noEr_Nx1` remains `0/212` practical and strict with `0.194s / 617.7 MB` JAX CPU versus `0.032s / 103.8 MB` Fortran, while `tokamak_1species_FPCollisions_noEr` remains the primary CPU mismatch at `1/188` practical and `8/188` strict.
- Remaining risks: office GPU still reports parity mismatches on three FP-heavy gate cases relative to the stable CPU Fortran reference (`inductiveE_noEr` `2/207`, `tokamak_1species_FPCollisions_noEr` `11/188`, `tokamak_1species_FPCollisions_withEr_DKESTrajectories` `38/214`); office still warns that `jax_cuda12_plugin 0.5.1` is incompatible with `jaxlib 0.6.2`; the clean `sfincs_original` reference branch could not be rebuilt locally because PETSc points at a missing Homebrew OpenMPI wrapper path.
- Next actions: inspect the GPU FP mismatch fields (`delta_f`, `sources`, `FSABFlow`, `particleFlux_vm_*`, `heatFlux_vm_*`, `pressureAnisotropy`) against the stable CPU reference lane, profile why GPU runtime regressed badly on `inductiveE_noEr` despite eliminating the crash, and either fix or explicitly gate the stale PETSc/OpenMPI path so the clean deterministic Fortran branch can be rebuilt reproducibly.

### 2026-03-05
- Scope: Replace the blind doubled-resolution example benchmark path with an upstream-reference resolution policy, preserve the partial `2x` profiling data as evidence, validate the corrected runner on local CPU and office GPU smoke cases, and start narrowing GPU-specific solver/backend blockers.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/scripts/run_scaled_example_suite.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/examples/README.md`, `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/io.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: confirmed all 38 vendored `examples/sfincs_examples/*/input.namelist` files are resolution-identical to `/Users/rogeriojorge/local/tests/sfincs_original/fortran/version3/examples/*/input.namelist`; local smoke with `tokamak_1species_FPCollisions_noEr` at `--scale-factor 1.0` against the original-example reference root; local and office smoke with `tokamak_1species_PASCollisions_noEr_Nx1` at `--scale-factor 1.0`; partial local CPU full-suite restart at the corrected original resolutions; office GPU smoke/restart on `inductiveE_noEr` after disabling dense auto-mode on non-CPU backends; `python -m py_compile sfincs_jax/io.py scripts/run_scaled_example_suite.py`.
- Runtime/memory delta: the aborted blind-`2x` partial run already showed why that default was wrong: `tokamak_1species_FPCollisions_noEr` at `42x1x16x62` took 183.747s / 2300.1 MB and still had `1/188` practical (`8/188` strict) mismatches, while the corrected upstream-reference run at the original `21x1x8x31` took 2.956s / 998.8 MB with the same mismatch signature. `tokamak_1species_PASCollisions_noEr_Nx1` at the corrected original resolution ran parity-clean on CPU (0.132s Fortran / 2.086s JAX / 630.2 MB) and in the initial office GPU smoke (1.576s Fortran / 5.001s JAX / 1293.3 MB). The partial corrected CPU full-suite already completed 13 tokamak/quick/inductive cases with full print parity and only one strict mismatch case so far (`tokamak_1species_FPCollisions_noEr`, `8/188`).
- Remaining risks: office GPU still reports a `jax_cuda12_plugin` / `jaxlib` version mismatch warning; office Fortran outputs appear nondeterministic on some classical-heat-flux fields (`classicalHeatFlux*`, `gpsiHatpsiHat`) for the same input, so office-generated Fortran H5s are not yet trustworthy as the GPU parity reference; the dense auto-mode patch for non-CPU backends moved `inductiveE_noEr` forward but the run still dies later in GPU-only dense-fallback / `jnp.linalg.lstsq` cuSOLVER calls.
- Next actions: finish parsing the partial corrected CPU suite into a report artifact, compare GPU JAX outputs against a stable CPU-generated Fortran reference instead of the unstable office Fortran H5s, and remove or host-fallback the remaining GPU dense-fallback / least-squares calls so `inductiveE_noEr` and related small FP cases can complete on CUDA.

### 2026-03-05
- Scope: Trim unnecessary PAS auto strong-preconditioner retries after already-strong base preconditioners, and resync README/docs with the stored suite artifacts.
- Files changed: `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/v3_driver.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/docs/usage.rst`, `/Users/rogeriojorge/local/tests/sfincs_jax/docs/performance_techniques.rst`, `/Users/rogeriojorge/local/tests/sfincs_jax/tests/test_schur_precond_heuristic.py`, `/Users/rogeriojorge/local/tests/sfincs_jax/plan.md`
- Validation run: direct CLI profiles on `tokamak_2species_PASCollisions_withEr_fullTrajectories` and `HSX_PASCollisions_fullTrajectories`, practical/strict H5 compares against stored Fortran outputs, an HSX gate-check confirming the larger-gap branch still enters the strong fallback path, `pytest -q tests/test_schur_precond_heuristic.py tests/test_pas_projection_heuristic.py tests/test_xblock_tz_precond_heuristic.py`, and `sphinx-build -W -b html docs docs/_build/html`.
- Runtime/memory delta: `tokamak_2species_PASCollisions_withEr_fullTrajectories` improved from 180.320s / 1732.1 MB (stored suite baseline) to 10.741s / 955.3 MB in the patched direct CLI run, with practical parity unchanged at `0/212` and strict parity unchanged at `1/212`. For `HSX_PASCollisions_fullTrajectories`, disabling the strong retry entirely still produced one practical mismatch (`densityPerturbation`), so the default keeps the fallback for larger residual gaps.
- Remaining risks: `HSX_PASCollisions_fullTrajectories` still needs a cheaper correction path than the full PAS strong retry; the full reduced suite has not yet been rerun after this solver change.
- Next actions: profile the HSX PAS full-trajectories branch again, isolate which part of the strong retry fixes `densityPerturbation`, and replace the expensive second Krylov cycle with a bounded PAS polish or equivalent constraint-aware correction.

---

## 17) Important Command Snippets

### 17.1 Docs + tests

```bash
cd /Users/rogeriojorge/local/tests/sfincs_jax
sphinx-build -W -b html docs docs/_build/html
pytest -q
```

### 17.2 Run one input like Fortran

```bash
sfincs_jax /path/to/input.namelist
```

### 17.3 Python run + in-memory results

```python
from pathlib import Path
from sfincs_jax.io import write_sfincs_jax_output_h5

out_path, results = write_sfincs_jax_output_h5(
    input_namelist=Path("input.namelist"),
    output_path=Path("sfincsOutput.h5"),
    return_results=True,
)
```

### 17.4 Upstream-reference example-suite benchmark

```bash
cd /Users/rogeriojorge/local/tests/sfincs_jax
python scripts/run_scaled_example_suite.py \
  --examples-root examples/sfincs_examples \
  --resolution-reference-root /Users/rogeriojorge/local/tests/sfincs_original/fortran/version3/examples \
  --fortran-exe /Users/rogeriojorge/local/tests/sfincs/fortran/version3/sfincs \
  --out-root tests/scaled_example_suite_ref_cpu_local \
  --timeout-s 240 \
  --max-attempts 2 \
  --scale-factor 1.0
```

---

## 18) References (online + local)

### 18.1 Online references used for strategy context
- IAEA World Fusion Outlook 2024: https://www.iaea.org/publications/15777/iaea-world-fusion-outlook-2024
- DOE FIRE + Milestone progress (Jan 16, 2025): https://www.energy.gov/articles/us-department-energy-announces-selectees-107-million-fusion-innovation-research-engine
- Fusion Industry Association 2024 report (PDF): https://sciencebusiness.net/sites/default/files/inline-files/FIA_annual%20report%202024.pdf
- ITER IMAS open-source release (Dec 8, 2025): https://www.iter.org/node/20687/release-imas-infrastructure-and-physics-models-open-source
- MONKES paper: https://arxiv.org/abs/2312.12248
- NEO docs (GACODE): https://gafusion.github.io/doc/neo.html
- NEO (STELLOPT page): https://princetonuniversity.github.io/STELLOPT/NEO.html
- KNOSOS paper: https://arxiv.org/abs/1908.11615
- NERSC Perlmutter architecture: https://docs.nersc.gov/systems/perlmutter/architecture/

### 18.2 Local references to mine and cite in docs
- `/Users/rogeriojorge/local/tests/sfincs_jax/docs/upstream/20131220-04 Technical documentation for SFINCS with a single species.pdf`
- `/Users/rogeriojorge/local/tests/sfincs_jax/docs/upstream/20131219-01 Technical documentation for SFINCS with multiple species.pdf`
- `/Users/rogeriojorge/local/tests/sfincs_jax/docs/upstream/20150325-01 Effects on fluxes of including Phi_1.pdf`
- `/Users/rogeriojorge/local/tests/sfincs_jax/docs/upstream/20150507-01 Technical documentation for version 3 of SFINCS.pdf`
- `/Users/rogeriojorge/local/tests/sfincs_jax/docs/upstream/sfincsPaper/sfincsPaper.pdf`
- `/Users/rogeriojorge/local/tests/Escoto_Thesis.pdf`
- `/Users/rogeriojorge/local/tests/Merkel_1987.pdf`
- `/Users/rogeriojorge/local/tests/hirshman_sigmar_1983.pdf`
- `/Users/rogeriojorge/local/tests/numerics_vmec.pdf`

---

## 19) Definition Of Done (current release gate)

Release-ready means:
1. Reduced-suite practical comparisons all pass and table is fully populated.
2. Additional high-resolution example(s) run successfully with validated outputs.
3. No hidden external-file dependence for correctness in default path.
4. CI/docs/tests are green.
5. Runtime/memory and solver defaults are documented with reproducible commands.
6. This `plan.md` reflects current truth and next executable steps.
