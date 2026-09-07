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
"""

from __future__ import annotations

import sys

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


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1:])
