from __future__ import annotations

from collections.abc import Callable
import sys
import time
from typing import TextIO


EmitFn = Callable[[int, str], None]


def make_emit(*, verbose: int = 0, quiet: bool = False, stream: TextIO | None = None, prefix: str = "") -> EmitFn:
    """Create a lightweight structured printer used throughout `sfincs_jax`.

    This is intentionally *not* the stdlib `logging` module:
    - deterministic output (useful for parity workflows)
    - trivial dependency surface
    - easy to pass into APIs that may be used in notebooks or scripts

    Parameters
    ----------
    verbose:
      Print messages with `level <= verbose`.
    quiet:
      Suppress all messages.
    stream:
      Destination stream (default: stdout).
    prefix:
      Optional line prefix (e.g. for indentation).
    """
    if stream is None:
        stream = sys.stdout

    v = int(verbose)
    q = bool(quiet)
    p = str(prefix)

    def emit(level: int, msg: str) -> None:
        if q:
            return
        if v >= int(level):
            print(f"{p}{msg}", file=stream, flush=True)

    return emit


class Timer:
    """Small helper for elapsed-time prints."""

    def __init__(self) -> None:
        self._t0 = time.perf_counter()

    def elapsed_s(self) -> float:
        return float(time.perf_counter() - self._t0)

