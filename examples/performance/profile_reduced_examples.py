from __future__ import annotations

import argparse
import json
import os
import re
import time
import signal
from pathlib import Path

from sfincs_jax.io import write_sfincs_jax_output_h5


_ROOT = Path(__file__).resolve().parents[2]
_REDUCED_INPUTS = _ROOT / "tests" / "reduced_inputs"


def _parse_profile_line(line: str) -> dict[str, float | str | None]:
    # profiling: label dt_s=0.123 total_s=0.456 rss_mb=123.4 drss_mb=2.3 device_mb=10.1
    out: dict[str, float | str | None] = {"raw": line.strip()}
    if not line.startswith("profiling:"):
        return out
    try:
        _, rest = line.split("profiling:", 1)
        parts = rest.strip().split()
        label = parts[0]
        out["label"] = label
        for token in parts[1:]:
            if "=" not in token:
                continue
            k, v = token.split("=", 1)
            v = v.strip()
            if v.lower() in {"na", "n/a", "none"}:
                out[k] = None
                continue
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = None
    except Exception:
        return out
    return out


def _collect_profiles(input_path: Path, out_path: Path, *, timeout_s: float | None = None) -> dict[str, object]:
    logs: list[str] = []

    def emit(level: int, msg: str) -> None:
        if msg.startswith("profiling:"):
            logs.append(msg)

    def _timeout_handler(signum, frame):  # noqa: ARG001
        raise TimeoutError(f"Timed out after {timeout_s}s")

    old_handler = None
    if timeout_s is not None and timeout_s > 0:
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, float(timeout_s))

    try:
        t0 = time.perf_counter()
        write_sfincs_jax_output_h5(
            input_namelist=input_path,
            output_path=out_path,
            fortran_layout=True,
            overwrite=True,
            compute_transport_matrix=True,
            compute_solution=True,
            emit=emit,
            verbose=True,
        )
        wall_s = time.perf_counter() - t0
    finally:
        if timeout_s is not None and timeout_s > 0:
            signal.setitimer(signal.ITIMER_REAL, 0)
            if old_handler is not None:
                signal.signal(signal.SIGALRM, old_handler)

    entries = [_parse_profile_line(line) for line in logs]
    max_rss = None
    max_dev = None
    for entry in entries:
        rss = entry.get("rss_mb")
        dev = entry.get("device_mb")
        if isinstance(rss, float):
            max_rss = rss if max_rss is None else max(max_rss, rss)
        if isinstance(dev, float):
            max_dev = dev if max_dev is None else max(max_dev, dev)

    return {
        "input": str(input_path),
        "output": str(out_path),
        "wall_s": wall_s,
        "entries": entries,
        "max_rss_mb": max_rss,
        "max_device_mb": max_dev,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile reduced upstream examples with per-phase timing.")
    parser.add_argument("--pattern", default=None, help="Regex pattern to filter input filenames.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of inputs profiled.")
    parser.add_argument(
        "--out-json",
        default=str(_ROOT / "examples" / "performance" / "output" / "reduced_profiles.json"),
        help="Where to write JSON summary.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(_ROOT / "examples" / "performance" / "output" / "reduced_profiles"),
        help="Directory for output sfincsOutput.h5 files.",
    )
    parser.add_argument("--timeout-s", type=float, default=None, help="Optional per-case timeout in seconds.")
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append/update an existing JSON file instead of overwriting.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip inputs already present in the output JSON (implies --append).",
    )
    args = parser.parse_args()

    os.environ["SFINCS_JAX_PROFILE"] = "1"

    inputs = sorted(_REDUCED_INPUTS.glob("*.input.namelist"))
    if args.pattern:
        pat = re.compile(args.pattern)
        inputs = [p for p in inputs if pat.search(p.name)]
    if args.limit is not None:
        inputs = inputs[: int(args.limit)]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    existing: dict[str, dict[str, object]] = {}
    if args.append or args.skip_existing:
        out_json = Path(args.out_json)
        if out_json.exists():
            try:
                prev = json.loads(out_json.read_text())
                if isinstance(prev, list):
                    for entry in prev:
                        if isinstance(entry, dict) and "input" in entry:
                            existing[str(entry["input"])] = entry
            except Exception:
                existing = {}

    results: list[dict[str, object]] = list(existing.values())
    for idx, input_path in enumerate(inputs, start=1):
        print(f"[{idx}/{len(inputs)}] {input_path.name}")
        out_path = out_dir / f"{input_path.stem}.sfincsOutput.h5"
        if args.skip_existing and str(input_path) in existing:
            continue
        try:
            results.append(_collect_profiles(input_path, out_path, timeout_s=args.timeout_s))
        except Exception as exc:  # noqa: BLE001
            results.append(
                {
                    "input": str(input_path),
                    "output": str(out_path),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    # Deduplicate by input path to keep the latest result.
    merged: dict[str, dict[str, object]] = {}
    for entry in results:
        if isinstance(entry, dict) and "input" in entry:
            merged[str(entry["input"])] = entry
    out_json.write_text(json.dumps(list(merged.values()), indent=2))
    print(f"Wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
