# Changelog

## Unreleased — candidate v2.4.0-rc1

Draft for the integrated #169–#188 stack, #191 and the reconciled #190 plan.
No release or candidate tag has been created by this change. Release remains
deferred until the maintainer's important-goal and verification requirements are met.

### Correctness and differentiation

- Keep discrete pitch layouts static under JIT; refresh full-FP density kernels
  and opt-in temperature-dependent coefficients for prepared profile derivatives
  (#173, #174, #178, #186).
- Prepare immutable native profile/field scans with per-input original-equation
  status and complete kinetic-state recovery. Differentiate supported regular
  ambipolar roots with original primal/transpose admission (#182, #184, #187, #188).
- Reject invalid kinetic states, roots and adjoint references; qualify physical
  profile sensitivities with independent cold solves and Taylor checks
  (#180, #184, #188).

### Execution

- Preserve independent-device batch sharding through JIT and gradients, including
  uneven batches (#179). Each complete system still resides on one device.
- Reuse generated SOLVAX Schur factors within supported PAS solve executions for
  multiple right-hand sides, transpose solves and refinement, subject to memory
  policy (#188). Persistent reuse across changed operators is still future work.

### Benchmarking and development

- Supervise benchmark process groups and cancellation; reject invalid reference
  outputs and stale resumed results; retain original-equation checks and evidence
  (#170, #183, #185).
- Preserve requested PETSc options and observed backends; make source inventories
  reproducible; correct sparse reference matrix interpretation (#169, #176, #177).
- Preserve the tested package tree when GitHub merge refs move (#181), and balance
  thirteen coverage shards using measured test durations (#191).
- Adopt one authoritative figure-first plan, a concise workflow-oriented README,
  grouped documentation and citation metadata (#189, #190). Establish decision
  records and an experiment template in Phase 0.

These changes do not certify converged production performance, general persistent
restart reuse, full native Phi1 support or complete equilibrium-boundary optimization.
Historical measurements and their limits remain in the performance documentation.
