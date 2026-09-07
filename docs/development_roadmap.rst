Development roadmap
===================

``plan.md`` at the repository root is the single authoritative plan; this page
is an entry point, not a second plan. The plan is figure-first: each phase is
defined by a publishable figure or table with entry and exit criteria, numbered
steps, a kill criterion where the work is an experiment, and an effort range.

Phases
------

- **Phase 0, land and freeze (1 week):** the integrated main, ADRs for the
  working method, the experiment-record template, a changelog, a tag.
- **Phase 1, the positioning figure (3–4 weeks):** DKX against SFINCS v3 and
  yancc at matched resolution, with an algebraic and a discretization error bar
  on every DKX point, cold and warm, CPU and GPU.
- **Phase 2, why the Krylov route loses (3 weeks, time-boxed):** a one-week
  diagnosis of the dominant dropped coupling and resource per deck, then two
  kill-gated experiments (block-pentadiagonal exactness for Er and drift decks;
  recycling discipline across sweeps) and memory-lean preconditioning.
- **Phase 3, verified derivatives and one real optimization (5–6 weeks):**
  derivatives through the ambipolar root and geometry with Taylor and
  finite-difference evidence, the closure accuracy map on real designs, and a
  VMEX boundary optimization at prescribed Er first.
- **Phase 4, native Phi1 and a W7-X impurity result (4–6 weeks).**
- **Phase 5, the papers:** a methods paper and a physics letter, each defined
  by the figures the phases produce.

Working method, engineering contracts, deferred work and the history of the
plan's predecessors are sections 5 to 9 of ``plan.md``. Older phase letters and
experiment pages in these docs are historical records, not active plans.
