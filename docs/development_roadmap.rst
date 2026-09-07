Development roadmap
===================

The `repository plan <https://github.com/uwplasma/DKX/blob/main/plan.md>`_
is the single active roadmap. The September 2026 review replaces the previous
phase letters and chronological execution log with dependency-ordered work
packages, scientific acceptance gates, and a publication program.

Start with measurement integrity and observable accuracy, then prepared solves,
certified reuse, sparse preconditioners, native physics, and CPU/GPU sharding.
Real VMEX equilibrium optimization, NEOPAX transport coupling, and ESSOS coil
optimization require the corresponding scientific and derivative gates.
Mirror geometry and optimization are the final, deferred phase.

``validation/baseline.toml`` preserves the historical inventory and records the
latest review separately. ``validation/capabilities.toml`` records capability
scope; ``validation/registry.toml`` indexes retained scientific evidence.
``validation/hardware.toml`` identifies the available measurement hosts.
Older phase letters and PR descriptions describe historical work and do not
set the next action. Completing a work package updates the plan's current
checkpoint, capability status, and evidence links together.
