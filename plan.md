# SFINCS_JAX Master Handoff + Execution Plan

Last updated: 2026-03-05 (America/Chicago)
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
- [ ] Strengthen default PAS preconditioner path to avoid expensive fallback branches where possible.
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

