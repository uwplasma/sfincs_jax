# Experiment records

Use one page per time-boxed experiment authorized by `plan.md`. This directory
records outcomes; it is not another work queue. Name a record `YYYY-MM-DD-topic.md`.
Write admission and kill criteria before starting. Finish with a decision before
starting the next algorithm experiment. Link the PR for execution details and
keep raw states, traces and build trees outside Git.

```markdown
# Experiment title

## Hypothesis
Named plan phase and figure; physical mechanism or numerical bottleneck;
owner; time budget; expected measurable improvement.

## Admission test
Pinned case and model, observable/error budget, original primal/transpose checks,
reference and convergence requirements. Define baseline, ablations, repetitions,
cold/warm timing scope and memory limit. State the kill criterion in advance.

## Result
Version and git SHA; JAX/jaxlib and x64; case_id and command; device and host.
Absolute measurements and spread, accepted/rejected cases, uncertainty and limits.
Link reproducible inputs, logs and any retained external artifact with its checksum.
Distinguish a planned test from an executed test and a smoke grid from convergence.

## Decision
Continue, stop or narrow scope, with the admission/kill evidence that decides it.
Link the implementation PR or superseding decision. Record negative results.
```
