from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

from .io import read_sfincs_h5


@dataclass(frozen=True)
class CompareResult:
    key: str
    max_abs: float
    max_rel: float
    ok: bool


def _as_numpy(x: Any) -> np.ndarray | None:
    if isinstance(x, np.ndarray):
        if x.dtype.kind in {"S", "U", "O"}:
            return None
        return x
    if np.isscalar(x):
        arr = np.asarray(x)
        if arr.dtype.kind in {"S", "U", "O"}:
            return None
        return arr
    return None


def compare_sfincs_outputs(
    *,
    a_path: Path,
    b_path: Path,
    keys: Sequence[str] | None = None,
    ignore_keys: Iterable[str] = ("elapsed time (s)",),
    rtol: float = 1e-12,
    atol: float = 1e-12,
    tolerances: Dict[str, Dict[str, float]] | None = None,
) -> List[CompareResult]:
    """Compare two `sfincsOutput.h5` files dataset-by-dataset."""
    a = read_sfincs_h5(a_path)
    b = read_sfincs_h5(b_path)

    ignore = set(ignore_keys)
    if keys is None:
        keys = sorted(set(a.keys()) & set(b.keys()))

    results: List[CompareResult] = []
    for k in keys:
        if k in ignore:
            continue
        av = a.get(k)
        bv = b.get(k)
        if av is None or bv is None:
            continue
        an = _as_numpy(av)
        bn = _as_numpy(bv)
        if an is None or bn is None:
            continue
        if an.shape != bn.shape:
            results.append(CompareResult(key=k, max_abs=float("inf"), max_rel=float("inf"), ok=False))
            continue

        tol = tolerances.get(k, {}) if tolerances else {}
        rtol_k = float(tol.get("rtol", rtol))
        atol_k = float(tol.get("atol", atol))

        diff = np.abs(an - bn)
        max_abs = float(diff.max()) if diff.size else float(abs(float(an) - float(bn)))
        denom = np.maximum(np.abs(bn), np.asarray(atol_k))
        max_rel = float((diff / denom).max()) if diff.size else float(abs(float(an) - float(bn)) / max(abs(float(bn)), atol_k))
        ok = bool(np.allclose(an, bn, rtol=rtol_k, atol=atol_k))
        results.append(CompareResult(key=k, max_abs=max_abs, max_rel=max_rel, ok=ok))

    return results
