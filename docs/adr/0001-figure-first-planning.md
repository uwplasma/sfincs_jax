# 0001: Figure-first planning

## Status
Accepted, 2026-09-06, implementing [PR #190](https://github.com/uwplasma/DKX/pull/190).

## Context
The previous implementation stack mixed decisions, execution diaries and future
work in several competing plans. Review volume obscured scientific completion.

## Decision
`plan.md` is the single authoritative phase and acceptance specification. Schedule
work against a named paper figure/table or a correctness bug. Keep one idea per
PR, normally at most 400 changed lines and ten files. Without automated rebasing,
merge one PR before opening its dependent successor. Store immutable decisions
here, bounded experiment outcomes in `docs/experiments/`, shipped changes in the
changelog, and commands/results in PR descriptions. Freeze benchmark tooling
except for bugs or a concrete Phase 1 figure requirement.

## Consequences
A completed implementation does not certify a physical result. Admit figures only
against the plan's algebraic, observable, resolution and reference criteria;
report losses and rejected cases. Amend decisions through a superseding ADR.
Release timing remains subject to the maintainer's instruction to release later,
after important goals are achieved; this decision creates no release or tag.
