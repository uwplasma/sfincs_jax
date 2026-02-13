from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Callable


def _rss_mb() -> float | None:
    try:
        import psutil  # type: ignore

        return float(psutil.Process().memory_info().rss) / 1e6
    except Exception:
        pass
    try:
        import resource

        rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if sys.platform == "darwin":
            return rss / (1024.0 * 1024.0)
        return rss / 1024.0
    except Exception:
        return None


def _device_mem_mb() -> float | None:
    try:
        import jax  # noqa: PLC0415

        stats = jax.devices()[0].memory_stats() or {}
    except Exception:
        return None
    for key in ("bytes_in_use", "bytes_active", "bytes_limit", "peak_bytes_in_use"):
        if key in stats:
            try:
                return float(stats[key]) / 1e6
            except Exception:
                continue
    return None


@dataclass
class SimpleProfiler:
    emit: Callable[[int, str], None] | None = None
    t0: float = field(default_factory=time.perf_counter)
    last: float = field(default_factory=time.perf_counter)
    rss0_mb: float | None = field(default_factory=_rss_mb)
    entries: list[dict[str, float | str | None]] = field(default_factory=list)

    def mark(self, label: str) -> None:
        now = time.perf_counter()
        rss_mb = _rss_mb()
        dev_mb = _device_mem_mb()
        entry = {
            "label": label,
            "dt_s": now - self.last,
            "total_s": now - self.t0,
            "rss_mb": rss_mb,
            "drss_mb": (rss_mb - self.rss0_mb) if (rss_mb is not None and self.rss0_mb is not None) else None,
            "device_mb": dev_mb,
        }
        self.entries.append(entry)
        if self.emit is not None:
            rss_txt = f"{rss_mb:.1f}" if rss_mb is not None else "na"
            drss_txt = f"{entry['drss_mb']:.1f}" if entry["drss_mb"] is not None else "na"
            dev_txt = f"{dev_mb:.1f}" if dev_mb is not None else "na"
            self.emit(
                0,
                f"profiling: {label} dt_s={entry['dt_s']:.3f} total_s={entry['total_s']:.3f} "
                f"rss_mb={rss_txt} drss_mb={drss_txt} device_mb={dev_txt}",
            )
        self.last = now


def maybe_profiler(emit: Callable[[int, str], None] | None = None) -> SimpleProfiler | None:
    flag = os.environ.get("SFINCS_JAX_PROFILE", "").strip().lower()
    if flag in {"1", "true", "yes", "on"}:
        return SimpleProfiler(emit=emit)
    return None
