# 0002: Balance CI shards using measured test durations

## Status
Accepted, 2026-09-06, recording [PR #191](https://github.com/uwplasma/DKX/pull/191).
Supersedes the 2026-07-17 decision to archive the durations snapshot.

## Context
Thirteen `pytest-split` shards used `least_duration` without a durations file,
falling back to test counts. PR #191 reports a shard at 9m53s against a ten-minute
limit, followed by timeouts. Its measured balancing reduced shards to 3.8–6.2
minutes. Those measurements describe that PR's environment, not a universal CI
runtime guarantee.

## Decision
Track compact `.test_durations` and retain the existing thirteen shards. The
source-tree contract requires valid duration data no larger than 512 KiB.
Regenerate with `pytest --store-durations` on a representative complete run when
test costs change; review the resulting diff and shard times. Keep the measured
host and command in the PR description. Do not replace representative durations
with a partial test run.

## Consequences
A small tracked data file buys timeout margin without increasing shard count.
Unknown tests receive estimated durations until refreshed. Hardware, compilation
and test changes can make measurements stale; successful scheduling is not a
claim of application performance or complete scientific coverage.
