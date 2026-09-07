# 0003: Maintainer authorship for repository commits

## Status
Accepted, 2026-09-06, recording the maintainer's instruction and PR #190.

## Context
Assistant coauthor trailers can propagate into squash commits. The repository
already has a `commit-trailers` CI job to reject known assistant attribution.

## Decision
Set both author and committer to `rogeriojorge <rogerio.jorge@wisc.edu>` for work
performed under this instruction. Add no assistant coauthor trailers. Inspect
commit metadata and the final merge message before publication, including when
GitHub synthesizes a squash message.

## Consequences
The existing CI regex detects specified assistant identities and trailers; it
does not enforce an exact maintainer identity for every possible author. Explicit
metadata inspection remains necessary. Do not rewrite unrelated history or other
contributors' work to satisfy this rule for new commits.
