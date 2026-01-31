from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import h5py
import numpy as np


def _decode_if_bytes(x: Any) -> Any:
    if isinstance(x, (bytes, np.bytes_)):
        return x.decode("utf-8", errors="replace")
    if isinstance(x, np.ndarray) and x.dtype.kind in {"S", "O"}:
        # Common case in SFINCS: 1-element byte-string array.
        if x.size == 1:
            item = x.reshape(-1)[0]
            return _decode_if_bytes(item)
    return x


def read_sfincs_h5(path: Path) -> Dict[str, Any]:
    """Read a SFINCS `sfincsOutput.h5` file into memory.

    This is intended for small-to-moderate outputs used in tests and examples.
    """
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(str(path))

    out: Dict[str, Any] = {}
    with h5py.File(path, "r") as f:
        def visit(name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Dataset):
                v = obj[...]
                v = _decode_if_bytes(v)
                out[name] = v

        f.visititems(visit)
    return out

