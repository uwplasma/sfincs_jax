#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from sfincs_jax.io import read_sfincs_h5, sfincs_jax_output_dict
from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.petsc_binary import read_petsc_vec
from sfincs_jax.transport_matrix import v3_rhsmode1_output_fields_vm_only
from sfincs_jax.v3 import grids_from_namelist
from sfincs_jax.v3_driver import _transport_active_dof_indices
from sfincs_jax.v3_system import full_system_operator_from_namelist


def _load_state_vector(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.asarray(np.load(path), dtype=np.float64).reshape((-1,))
    else:
        arr = np.asarray(read_petsc_vec(path).values, dtype=np.float64).reshape((-1,))
    return arr


def _last_iter(arr: np.ndarray, n_iter: int) -> np.ndarray:
    if arr.ndim >= 1 and arr.shape[-1] == int(n_iter):
        return np.asarray(arr[..., -1], dtype=np.float64)
    return np.asarray(arr, dtype=np.float64)


def _to_fortran_read_layout(diag: dict[str, jnp.ndarray]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    out["densityPerturbation"] = np.transpose(np.asarray(diag["densityPerturbation"], dtype=np.float64), (2, 1, 0))
    out["pressurePerturbation"] = np.transpose(np.asarray(diag["pressurePerturbation"], dtype=np.float64), (2, 1, 0))
    fsad = np.asarray(diag["FSADensityPerturbation"], dtype=np.float64)
    fsap = np.asarray(diag["FSAPressurePerturbation"], dtype=np.float64)
    out["FSADensityPerturbation"] = np.transpose(fsad, (1, 0)) if fsad.ndim == 2 else fsad.reshape((-1,))
    out["FSAPressurePerturbation"] = np.transpose(fsap, (1, 0)) if fsap.ndim == 2 else fsap.reshape((-1,))
    out["flow"] = np.transpose(np.asarray(diag["flow"], dtype=np.float64), (2, 1, 0))
    out["velocityUsingTotalDensity"] = np.transpose(np.asarray(diag["velocityUsingTotalDensity"], dtype=np.float64), (2, 1, 0))
    out["MachUsingFSAThermalSpeed"] = np.transpose(np.asarray(diag["MachUsingFSAThermalSpeed"], dtype=np.float64), (2, 1, 0))
    out["FSABFlow"] = np.asarray(diag["FSABFlow"], dtype=np.float64).reshape((-1,))
    out["FSABFlow_vs_x"] = np.asarray(diag["FSABFlow_vs_x"], dtype=np.float64)
    out["FSABVelocityUsingFSADensity"] = np.asarray(diag["FSABVelocityUsingFSADensity"], dtype=np.float64).reshape((-1,))
    out["FSABVelocityUsingFSADensityOverB0"] = np.asarray(diag["FSABVelocityUsingFSADensityOverB0"], dtype=np.float64).reshape((-1,))
    out["FSABVelocityUsingFSADensityOverRootFSAB2"] = np.asarray(
        diag["FSABVelocityUsingFSADensityOverRootFSAB2"], dtype=np.float64
    ).reshape((-1,))
    out["FSABjHat"] = np.asarray(diag["FSABjHat"], dtype=np.float64).reshape(())
    out["FSABjHatOverB0"] = np.asarray(diag["FSABjHatOverB0"], dtype=np.float64).reshape(())
    out["FSABjHatOverRootFSAB2"] = np.asarray(diag["FSABjHatOverRootFSAB2"], dtype=np.float64).reshape(())
    return out


def _ntv_from_state(*, op, x_full: jnp.ndarray, base_data: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    geometry_scheme = int(np.asarray(base_data["geometryScheme"]))
    if geometry_scheme == 5 or int(op.n_xi) <= 2:
        ntv_before = np.zeros((int(op.n_zeta), int(op.n_theta), int(op.n_species)), dtype=np.float64)
        ntv = np.zeros((int(op.n_species),), dtype=np.float64)
        return ntv_before, ntv

    bh = jnp.asarray(base_data["BHat"], dtype=jnp.float64)
    dbt = jnp.asarray(base_data["dBHatdtheta"], dtype=jnp.float64)
    dbz = jnp.asarray(base_data["dBHatdzeta"], dtype=jnp.float64)
    uhat = jnp.asarray(base_data["uHat"], dtype=jnp.float64)
    inv_fsa_b2 = 1.0 / jnp.asarray(float(base_data["FSABHat2"]), dtype=jnp.float64)
    ghat = jnp.asarray(float(base_data["GHat"]), dtype=jnp.float64)
    ihat = jnp.asarray(float(base_data["IHat"]), dtype=jnp.float64)
    iota = jnp.asarray(float(base_data["iota"]), dtype=jnp.float64)
    ntv_kernel = (2.0 / 5.0) / bh * (
        (uhat - ghat * inv_fsa_b2) * (iota * dbt + dbz) + iota * (1.0 / (bh * bh)) * (ghat * dbt - ihat * dbz)
    )

    tw = jnp.asarray(op.theta_weights, dtype=jnp.float64)
    zw = jnp.asarray(op.zeta_weights, dtype=jnp.float64)
    w2d = tw[:, None] * zw[None, :]
    vprime_hat = jnp.sum(w2d / jnp.asarray(op.d_hat, dtype=jnp.float64))
    x = jnp.asarray(op.x, dtype=jnp.float64)
    xw = jnp.asarray(op.x_weights, dtype=jnp.float64)
    w_ntv = xw * (x**4)

    t_hat = jnp.asarray(op.t_hat, dtype=jnp.float64)
    m_hat = jnp.asarray(op.m_hat, dtype=jnp.float64)
    f_delta = x_full[: op.f_size].reshape(op.fblock.f_shape)
    sum_ntv = jnp.einsum("x,sxtz->stz", w_ntv, f_delta[:, :, 2, :, :])
    ntv_before_stz = (
        (4.0 * jnp.pi * (t_hat * t_hat) * jnp.sqrt(t_hat) / (m_hat * jnp.sqrt(m_hat) * vprime_hat))[:, None, None]
        * ntv_kernel[None, :, :]
        * sum_ntv
    )
    ntv_s = jnp.einsum("tz,stz->s", w2d, ntv_before_stz)
    ntv_before = np.transpose(np.asarray(ntv_before_stz, dtype=np.float64), (2, 1, 0))  # (Z,T,S)
    ntv = np.asarray(ntv_s, dtype=np.float64).reshape((-1,))
    return ntv_before, ntv


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare RHSMode=1 diagnostics computed from a frozen state vector against Fortran sfincsOutput.h5."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input namelist used for the run.")
    parser.add_argument("--state", type=Path, required=True, help="Frozen state vector (.npy or PETSc binary vec).")
    parser.add_argument("--fortran-h5", type=Path, required=True, help="Fortran sfincsOutput.h5 to compare against.")
    parser.add_argument("--out-json", type=Path, default=None, help="Optional JSON report path.")
    args = parser.parse_args()

    nml = read_sfincs_input(args.input)
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)
    x_in = _load_state_vector(args.state)
    if x_in.shape[0] == int(op.total_size):
        x_full_np = x_in
    else:
        active_idx = _transport_active_dof_indices(op)
        if x_in.shape[0] != int(active_idx.shape[0]):
            raise ValueError(
                f"State size mismatch: got {x_in.shape[0]}, expected {int(op.total_size)} "
                f"(full) or {int(active_idx.shape[0])} (active-DOF)."
            )
        x_full_np = np.zeros((int(op.total_size),), dtype=np.float64)
        x_full_np[np.asarray(active_idx, dtype=np.int32)] = x_in
    x_full = jnp.asarray(x_full_np, dtype=jnp.float64)

    diag = v3_rhsmode1_output_fields_vm_only(op, x_full=x_full)
    diag_fortran_layout = _to_fortran_read_layout(diag)

    grids = grids_from_namelist(nml)
    base_data = sfincs_jax_output_dict(nml=nml, grids=grids)
    ntv_before, ntv = _ntv_from_state(op=op, x_full=x_full, base_data=base_data)
    diag_fortran_layout["NTVBeforeSurfaceIntegral"] = ntv_before
    diag_fortran_layout["NTV"] = ntv

    ref = read_sfincs_h5(args.fortran_h5)
    n_iter = int(np.asarray(ref.get("NIterations", 1)).reshape(-1)[0])
    keys = [
        "densityPerturbation",
        "pressurePerturbation",
        "FSADensityPerturbation",
        "FSAPressurePerturbation",
        "FSABFlow",
        "FSABFlow_vs_x",
        "FSABVelocityUsingFSADensity",
        "FSABVelocityUsingFSADensityOverB0",
        "FSABVelocityUsingFSADensityOverRootFSAB2",
        "FSABjHat",
        "FSABjHatOverB0",
        "FSABjHatOverRootFSAB2",
        "velocityUsingTotalDensity",
        "MachUsingFSAThermalSpeed",
        "NTVBeforeSurfaceIntegral",
        "NTV",
    ]

    per_key: dict[str, dict[str, Any]] = {}
    for key in keys:
        jax_val = np.asarray(diag_fortran_layout[key], dtype=np.float64)
        ref_val = _last_iter(np.asarray(ref[key], dtype=np.float64), n_iter=n_iter)
        diff = np.asarray(jax_val - ref_val, dtype=np.float64)
        per_key[key] = {
            "shape": list(jax_val.shape),
            "max_abs_diff": float(np.max(np.abs(diff))),
            "mean_abs_diff": float(np.mean(np.abs(diff))),
        }

    worst_key = max(per_key.keys(), key=lambda k: float(per_key[k]["max_abs_diff"]))
    report = {
        "input": str(args.input),
        "state": str(args.state),
        "fortran_h5": str(args.fortran_h5),
        "n_iterations_fortran": int(n_iter),
        "worst_key": worst_key,
        "worst_key_max_abs_diff": float(per_key[worst_key]["max_abs_diff"]),
        "per_key": per_key,
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
