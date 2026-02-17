from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import jax.numpy as jnp


def _op_signature(op) -> np.ndarray:
    return np.asarray(
        [
            int(op.rhs_mode),
            int(op.total_size),
            int(op.n_species),
            int(op.n_x),
            int(op.n_xi),
            int(op.n_theta),
            int(op.n_zeta),
            int(op.constraint_scheme),
            int(bool(op.include_phi1)),
            int(bool(op.include_phi1_in_kinetic)),
            int(op.quasineutrality_option),
        ],
        dtype=np.int64,
    )


def save_krylov_state(
    *,
    path: str | Path,
    op,
    x_full: jnp.ndarray | None = None,
    x_by_rhs: dict[int, jnp.ndarray] | None = None,
    x_history: list[jnp.ndarray] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "signature": _op_signature(op),
    }
    if x_full is not None:
        payload["x_full"] = np.asarray(x_full, dtype=np.float64)
    if x_by_rhs is not None:
        which_rhs = np.asarray(sorted(x_by_rhs.keys()), dtype=np.int64)
        x_stack = np.stack([np.asarray(x_by_rhs[int(k)], dtype=np.float64) for k in which_rhs], axis=0)
        payload["which_rhs"] = which_rhs
        payload["x_by_rhs"] = x_stack
    if x_history is not None:
        if isinstance(x_history, (list, tuple)) and x_history:
            x_hist_stack = np.stack([np.asarray(v, dtype=np.float64) for v in x_history], axis=0)
            payload["x_history"] = x_hist_stack
    np.savez(path, **payload)


def load_krylov_state(
    *,
    path: str | Path,
    op,
) -> dict[str, Any] | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        data = np.load(path, allow_pickle=False)
    except Exception:
        return None
    try:
        sig = np.asarray(data["signature"], dtype=np.int64)
    except Exception:
        return None
    if sig.shape != _op_signature(op).shape:
        return None
    if not np.array_equal(sig, _op_signature(op)):
        return None
    out: dict[str, Any] = {}
    if "x_full" in data:
        out["x_full"] = np.asarray(data["x_full"], dtype=np.float64)
    if "x_by_rhs" in data and "which_rhs" in data:
        which_rhs = np.asarray(data["which_rhs"], dtype=np.int64)
        x_stack = np.asarray(data["x_by_rhs"], dtype=np.float64)
        if x_stack.ndim == 2 and which_rhs.ndim == 1 and x_stack.shape[0] == which_rhs.shape[0]:
            out["x_by_rhs"] = {int(k): x_stack[i, :] for i, k in enumerate(which_rhs)}
    if "x_history" in data:
        x_hist = np.asarray(data["x_history"], dtype=np.float64)
        if x_hist.ndim == 2:
            out["x_history"] = [x_hist[i, :] for i in range(x_hist.shape[0])]
    if not out:
        return None
    return out
