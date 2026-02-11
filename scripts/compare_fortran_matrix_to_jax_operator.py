#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from scipy.sparse import csr_matrix

from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.petsc_binary import read_petsc_mat_aij, read_petsc_vec
from sfincs_jax.solver import assemble_dense_matrix_from_matvec
from sfincs_jax.v3_driver import _transport_active_dof_indices
from sfincs_jax.v3_system import apply_v3_full_system_operator, full_system_operator_from_namelist


def _block(name: str, start: int, end: int) -> dict[str, int | str]:
    return {"name": name, "start": int(start), "end": int(end)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare Fortran PETSc matrix vs JAX matrix-free operator.")
    parser.add_argument("--input", type=Path, required=True, help="input.namelist used for the Fortran matrix.")
    parser.add_argument("--fortran-matrix", type=Path, required=True, help="PETSc AIJ matrix file from Fortran.")
    parser.add_argument(
        "--fortran-state",
        type=Path,
        default=None,
        help="Optional PETSc state vector. If provided and includePhi1=true, its Phi1 is used as base linearization.",
    )
    parser.add_argument(
        "--project-active-dofs",
        action="store_true",
        help="Project JAX matrix to active DOFs (nXi_for_x-truncated system) before comparison.",
    )
    parser.add_argument("--threshold", type=float, default=1e-10, help="Absolute threshold for counting mismatches.")
    parser.add_argument("--out-json", type=Path, default=None, help="Optional JSON report path.")
    args = parser.parse_args()

    nml = read_sfincs_input(args.input)
    op0 = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)

    phi1_hat_base = None
    if bool(op0.include_phi1) and args.fortran_state is not None:
        x_ref = read_petsc_vec(args.fortran_state).values
        phi1_flat = x_ref[op0.f_size : op0.f_size + op0.n_theta * op0.n_zeta]
        phi1_hat_base = jnp.asarray(phi1_flat.reshape((op0.n_theta, op0.n_zeta)), dtype=jnp.float64)

    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0, phi1_hat_base=phi1_hat_base)

    mat = read_petsc_mat_aij(args.fortran_matrix)
    a_fortran = csr_matrix((mat.data, mat.col_ind, mat.row_ptr), shape=mat.shape).toarray()
    a_jax_full = np.asarray(
        assemble_dense_matrix_from_matvec(
            matvec=lambda v: apply_v3_full_system_operator(op, v),
            n=int(op.total_size),
            dtype=jnp.float64,
        ),
        dtype=np.float64,
    )

    if args.project_active_dofs:
        active_idx = np.asarray(_transport_active_dof_indices(op), dtype=np.int32)
        a_jax = a_jax_full[np.ix_(active_idx, active_idx)]
    else:
        active_idx = None
        a_jax = a_jax_full

    if a_jax.shape != a_fortran.shape:
        raise ValueError(f"Shape mismatch: JAX {a_jax.shape} vs Fortran {a_fortran.shape}")

    diff = a_jax - a_fortran
    abs_diff = np.abs(diff)
    threshold = float(args.threshold)

    f_end = int(op.f_size)
    phi_end = int(op.f_size + op.phi1_size)
    total = int(op.total_size)
    blocks = (
        _block("f", 0, f_end),
        _block("phi", f_end, phi_end),
        _block("extra", phi_end, total),
    )

    block_stats: list[dict[str, object]] = []
    for rb in blocks:
        for cb in blocks:
            rs, re = int(rb["start"]), int(rb["end"])
            cs, ce = int(cb["start"]), int(cb["end"])
            if rs >= a_jax.shape[0] or cs >= a_jax.shape[1]:
                continue
            re_use = min(re, a_jax.shape[0])
            ce_use = min(ce, a_jax.shape[1])
            if re_use <= rs or ce_use <= cs:
                continue
            blk = abs_diff[rs:re_use, cs:ce_use]
            block_stats.append(
                {
                    "row_block": rb["name"],
                    "col_block": cb["name"],
                    "shape": [int(blk.shape[0]), int(blk.shape[1])],
                    "count_gt_threshold": int(np.sum(blk > threshold)),
                    "max_abs": float(np.max(blk)),
                    "mean_abs": float(np.mean(blk)),
                }
            )

    top_n = 25
    order = np.argsort(abs_diff.ravel())[::-1]
    top_entries: list[dict[str, float | int]] = []
    for idx in order[:top_n]:
        r, c = np.unravel_index(idx, abs_diff.shape)
        if abs_diff[r, c] <= threshold:
            break
        top_entries.append(
            {
                "row": int(r),
                "col": int(c),
                "abs_diff": float(abs_diff[r, c]),
                "diff": float(diff[r, c]),
                "fortran": float(a_fortran[r, c]),
                "jax": float(a_jax[r, c]),
            }
        )

    report = {
        "input": str(args.input),
        "fortran_matrix": str(args.fortran_matrix),
        "fortran_state": str(args.fortran_state) if args.fortran_state is not None else None,
        "project_active_dofs": bool(args.project_active_dofs),
        "active_size": int(active_idx.shape[0]) if active_idx is not None else None,
        "jax_shape": [int(a_jax.shape[0]), int(a_jax.shape[1])],
        "fortran_shape": [int(a_fortran.shape[0]), int(a_fortran.shape[1])],
        "max_abs_diff": float(np.max(abs_diff)),
        "mean_abs_diff": float(np.mean(abs_diff)),
        "count_gt_threshold": int(np.sum(abs_diff > threshold)),
        "threshold": threshold,
        "block_stats": block_stats,
        "top_entries": top_entries,
    }

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote {args.out_json}")
    else:
        print(json.dumps(report, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
