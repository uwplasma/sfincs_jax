Development roadmap
===================

The `repository plan <https://github.com/uwplasma/DKX/blob/main/plan.md>`_
is the single active roadmap. Its ordered deliverables are verified toroidal
calculations, fast in-memory repeated solves, and an independently checked
VMEX boundary optimization. Documentation, examples and removal of duplication
belong to each deliverable.

The plan records the reviewed PR stack, evidence boundaries, solver ownership,
reuse and coordinate decisions, and explicit stopping criteria for experiments.
Native Phi1/full-drift promotion requires its scientific gates. NEOPAX and ESSOS
consume validated interfaces; mirrors remain the final deferred extension.
Release follows resolution of the open PRs and the important supported-scope goals.

``validation/capabilities.toml`` records capability scope;
``validation/registry.toml`` indexes retained scientific evidence;
``validation/hardware.toml`` identifies measurement hosts;
``validation/baseline.toml`` preserves dated inventory measurements.
Older phase letters and experiment pages are historical context, not additional
active work queues. Update the plan, canonical documentation and existing evidence
registries together when a deliverable is completed.
