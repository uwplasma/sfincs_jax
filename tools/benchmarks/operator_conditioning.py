#!/usr/bin/env python
"""Condition number of the kinetic operator, raw and equilibrated.

Dense SVD estimates conditioning of the pinned rectangular operator and its
Ruiz-equilibrated counterpart. Residual-to-state error bounds motivate this
probe, but do not establish the cause of a flux discrepancy or mesh error.
Near or beyond reciprocal machine precision, the smallest singular value and
condition estimate are not reliable quantitative certificates. Check the
constrained system, scaling and observable-weighted residual independently.

Build the operator without running a kinetic solve. The size cap applies
before dense materialization/SVD; construction still allocates grids and
collision coefficients. Failed construction is reported explicitly.

Run:
  python tools/benchmarks/operator_conditioning.py DECK.input.namelist [...]
  python tools/benchmarks/operator_conditioning.py TARGET --warm-from SOURCE

The optional warm audit executes bounded-size linear solves at a tolerance
ladder; it keeps the target coefficients fixed and recomputes preconditioners.
"""

from __future__ import annotations

import numpy as np

from dkx.drift_kinetic import KineticOperator
from dkx.namelist import read_sfincs_input
from dkx.solve import materialize_csr


def operator_for(namelist: str) -> KineticOperator:
    """Construct through the public builder, without solving or monkeypatching."""
    return KineticOperator.from_namelist(read_sfincs_input(namelist))


def ruiz(matrix: np.ndarray, iterations: int = 20) -> np.ndarray:
    """Ruiz row/column equilibration: scale both to unit max magnitude."""
    scaled = matrix.copy()
    for _ in range(iterations):
        rows = np.sqrt(np.maximum(np.abs(scaled).max(axis=1), 1e-300))
        cols = np.sqrt(np.maximum(np.abs(scaled).max(axis=0), 1e-300))
        scaled = scaled / rows[:, None] / cols[None, :]
    return scaled


def main(decks: list[str], max_size: int = 4000) -> None:
    print(
        f"{'deck':52s} {'n':>6s} {'cond(raw)':>11s} {'cond(equil)':>12s} {'gain':>7s}"
    )
    for deck in decks:
        name = deck.split("/")[-1][:52]
        try:
            op = operator_for(deck)
        except (OSError, ValueError, NotImplementedError) as exc:
            print(f"{name:52s} construction failed: {exc}")
            continue
        if op.include_phi1:
            print(f"{name:52s} coupled Phi1 needs an explicit linearization state")
            continue
        if op.total_size > max_size:
            print(f"{name:52s} {op.total_size:6d} {'too large':>11s}")
            continue
        dense = materialize_csr(op, pin_masked_dofs=True).toarray()
        raw = np.linalg.svd(dense, compute_uv=False)
        equil = np.linalg.svd(ruiz(dense), compute_uv=False)
        c_raw = raw[0] / raw[-1]
        c_eq = equil[0] / equil[-1]
        print(
            f"{name:52s} {op.total_size:6d} {c_raw:11.2e} {c_eq:12.2e} {c_raw / c_eq:6.0f}x",
            flush=True,
        )


def warm_audit(source_deck: str, target_deck: str, *, max_size: int = 4000,
               observable: str = "FSABjHat", tolerances=(1e-8, 1e-10, 1e-12)) -> dict:
    """Compare initial-state/recycle reuse on one fixed target equation.

    The adjoint identity attributes a *difference between computed states*,
    not their error against an exact solution or a resolved physical model.
    No factors or preconditioners are passed between solves.
    """
    import hashlib
    from pathlib import Path
    import jax
    import jax.numpy as jnp
    from scipy.linalg.lapack import dgesvx
    from dkx.batch import _check_shared_discretization
    from dkx.run import profile_moments_from_operator
    from dkx.solve import solve

    if not tolerances or any(not np.isfinite(t) or t <= 0 for t in tolerances):
        raise ValueError("tolerances must be positive and finite")
    source, target = operator_for(source_deck), operator_for(target_deck)
    if source.include_phi1 or target.include_phi1:
        raise ValueError("coupled Phi1 requires an explicit linearization state")
    if max(source.total_size, target.total_size) > max_size:
        raise ValueError("warm audit exceeds the dense size cap")
    _check_shared_discretization([source, target])
    def fingerprint(op):
        leaves, structure = jax.tree_util.tree_flatten(op)
        digest = hashlib.sha256(str(structure).encode())
        for leaf in leaves:
            array = np.asarray(leaf)
            digest.update(str((array.shape, array.dtype)).encode())
            digest.update(array.tobytes())
        return digest.hexdigest()

    identities = [fingerprint(op) for op in (source, target)]
    source_rhs, rhs = source.rhs(), target.rhs()
    source_norm, norm_b = (float(jnp.linalg.norm(b)) for b in (source_rhs, rhs))
    if not all(np.isfinite(np.asarray(b)).all() for b in (source_rhs, rhs)) or not all(
        np.isfinite(n) for n in (source_norm, norm_b)
    ):
        raise ValueError("warm audit requires finite drives and norms")
    if not norm_b:
        raise ValueError("warm audit requires a nonzero target drive")
    matrix = materialize_csr(target, pin_masked_dofs=True).toarray()

    def moment(x):
        return jnp.ravel(profile_moments_from_operator(target, x)[observable])[0]

    q = np.asarray(jax.grad(moment)(jnp.zeros(target.total_size)))
    # Factor A and solve its transpose with LAPACK's refinement/error estimates.
    # A plain solve of the explicitly transposed matrix was an inaccurate
    # referee on a checked full-FP case despite a small equation residual.
    *_, dual, rcond, ferr, berr, info = dgesvx(matrix, q[:, None], fact="N", trans="T")
    dual = dual[:, 0]
    if info != 0 or not all(np.isfinite(v).all() for v in (dual, rcond, ferr, berr)):
        raise RuntimeError(f"dense adjoint referee failed: DGESVX info={info}, rcond={rcond}")
    dual_residual = q - matrix.T @ dual
    seed = solve(source, source_rhs, method="gmres", tol=min(tolerances), emit=None)
    seed_residual = float(jnp.linalg.norm(source.apply(seed.x.reshape(-1)) - source_rhs.reshape(-1)))
    records = []
    for tol in tolerances:
        cold = None
        for label, initial, recycle in (("cold", None, None), ("state", seed.x, None),
                                        ("recycle", None, seed.recycle),
                                        ("state_and_recycle", seed.x, seed.recycle)):
            result = solve(target, rhs, method="gmres", tol=tol,
                           x0=initial, recycle=recycle, emit=None)
            x = np.asarray(result.x).reshape(-1)
            original_residual = np.asarray(rhs).reshape(-1) - np.asarray(target.apply(x))
            dense_residual = np.asarray(rhs).reshape(-1) - matrix @ x
            value = float(moment(x))
            if cold is None:
                cold = (x, value, dense_residual)
            delta_x = x - cold[0]
            delta = value - cold[1]
            predicted = float(dual @ (cold[2] - dense_residual))
            records.append(dict(
                tolerance=float(tol), reuse=label, method=result.method,
                initial_state_supplied=initial is not None, recycle_supplied=recycle is not None,
                iterations=result.iterations, solver_converged=bool(result.converged), observable=value,
                original_residual_pass=bool(np.linalg.norm(original_residual) <= tol * norm_b),
                original_relative_residual=float(np.linalg.norm(original_residual) / norm_b),
                dense_vs_original_residual_l2=float(np.linalg.norm(dense_residual - original_residual)),
                observable_difference=delta, dual_predicted_difference=predicted,
                identity_remainder=delta - predicted,
                linear_observable_difference=float(q @ delta_x),
                dual_residual_remainder_bound=float(np.linalg.norm(dual_residual) * np.linalg.norm(delta_x)),
            ))
    if identities != [fingerprint(op) for op in (source, target)]:
        raise ValueError("operator coefficients changed during warm audit")
    return dict(
        operator_sha256=dict(zip(("source", "target"), identities)),
        scope="computed-state difference only; no solution/grid-error certificate; remainder bound excludes roundoff",
        observable=observable, species_index=0, size=target.total_size,
        input_sha256={str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest()
                      for p in (source_deck, target_deck)},
        target_matrix_sha256=hashlib.sha256(matrix.tobytes()).hexdigest(),
        target_rhs_sha256=hashlib.sha256(np.asarray(rhs).tobytes()).hexdigest(),
        seed_solver_converged=bool(seed.converged),
        seed_absolute_residual=seed_residual,
        seed_original_residual_pass=bool(np.isfinite(seed_residual) and seed_residual <= min(tolerances) * source_norm),
        seed_relative_residual=seed_residual / source_norm if source_norm else None,
        dual_relative_residual=float(np.linalg.norm(dual_residual) / max(np.linalg.norm(q), 1e-300)),
        dual_reference=dict(driver="LAPACK DGESVX", transpose=True, equilibrated=False,
                            reciprocal_condition_estimate=float(rcond),
                            forward_error_norm="infinity",
                            relative_forward_error_estimate=float(ferr[0]),
                            componentwise_backward_error=float(berr[0])),
        records=records,
    )


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("decks", nargs="+")
    parser.add_argument("--max-size", type=int, default=4000)
    parser.add_argument("--warm-from", help="source deck for reuse into exactly one target deck")
    parser.add_argument("--observable", choices=("FSABjHat", "heatFlux_vm_psiHat"), default="FSABjHat")
    args = parser.parse_args()
    if args.warm_from:
        if len(args.decks) != 1:
            parser.error("--warm-from requires exactly one target deck")
        print(json.dumps(warm_audit(args.warm_from, args.decks[0], max_size=args.max_size,
                                   observable=args.observable), indent=2, allow_nan=False))
    else:
        main(args.decks, max_size=args.max_size)
