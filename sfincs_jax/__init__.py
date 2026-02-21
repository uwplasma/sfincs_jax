"""SFINCS (v3) port to JAX.

This repository is parity-first: we start by matching the Fortran v3 numerics and outputs
for selected examples, then expand coverage over time.
"""

from __future__ import annotations

# Enable host-device parallelism and a default JAX compilation cache for repeated
# CLI invocations unless the user explicitly disables it. This improves cold-start
# performance without requiring environment configuration.
import os
import tempfile

# Optional JAX multi-host bootstrap (must run before any JAX device use).
_distributed_env = os.environ.get("SFINCS_JAX_DISTRIBUTED", "").strip().lower()
if _distributed_env in {"1", "true", "yes", "on"}:
    try:
        import jax.distributed as _jax_distributed  # noqa: PLC0415

        _process_id_env = os.environ.get("SFINCS_JAX_PROCESS_ID", "").strip()
        _process_count_env = os.environ.get("SFINCS_JAX_PROCESS_COUNT", "").strip()
        _coord_addr = os.environ.get("SFINCS_JAX_COORDINATOR_ADDRESS", "").strip()
        _coord_port_env = os.environ.get("SFINCS_JAX_COORDINATOR_PORT", "").strip()

        _process_id = int(_process_id_env) if _process_id_env else 0
        _process_count = int(_process_count_env) if _process_count_env else 1
        _coord_port = int(_coord_port_env) if _coord_port_env else 1234

        if _coord_addr:
            _jax_distributed.initialize(
                coordinator_address=_coord_addr,
                coordinator_port=_coord_port,
                num_processes=_process_count,
                process_id=_process_id,
            )
    except Exception:
        # Best-effort: avoid hard failures when distributed runtime is unavailable.
        pass

# High-level cores knob: set this before importing JAX to request N CPU devices
# and enable parallel whichRHS + auto-sharding by default.
_cores_env = os.environ.get("SFINCS_JAX_CORES", "").strip()
if _cores_env:
    try:
        _cores_val = int(_cores_env)
    except ValueError:
        _cores_val = 0
    if _cores_val > 0:
        if _cores_val > 1:
            os.environ.setdefault("SFINCS_JAX_TRANSPORT_PARALLEL", "process")
            os.environ.setdefault("SFINCS_JAX_TRANSPORT_PARALLEL_WORKERS", str(_cores_val))
        _threads_env = os.environ.get("SFINCS_JAX_XLA_THREADS", "").strip().lower()
        if _threads_env in {"1", "true", "yes", "on"}:
            _xla_flags = os.environ.get("XLA_FLAGS", "")
            if "--xla_cpu_parallelism_threads" not in _xla_flags:
                flag = f"--xla_cpu_parallelism_threads={_cores_val}"
                os.environ["XLA_FLAGS"] = f"{_xla_flags} {flag}".strip()
        shard_env = os.environ.get("SFINCS_JAX_SHARD", "").strip().lower()
        if _cores_val > 1 and shard_env not in {"0", "false", "no", "off"}:
            os.environ.setdefault("SFINCS_JAX_CPU_DEVICES", str(_cores_val))
            os.environ.setdefault("SFINCS_JAX_MATVEC_SHARD_AXIS", "auto")
            os.environ.setdefault("SFINCS_JAX_AUTO_SHARD", "1")

# Allow users to request multiple CPU devices for JAX SPMD/pjit on host platforms.
# This must be set before importing JAX.
_cpu_devices_env = os.environ.get("SFINCS_JAX_CPU_DEVICES", "").strip()
if _cpu_devices_env:
    try:
        _cpu_devices = int(_cpu_devices_env)
    except ValueError:
        _cpu_devices = 0
    if _cpu_devices > 0:
        _xla_flags = os.environ.get("XLA_FLAGS", "")
        if "--xla_force_host_platform_device_count" not in _xla_flags:
            flag = f"--xla_force_host_platform_device_count={_cpu_devices}"
            os.environ["XLA_FLAGS"] = f"{_xla_flags} {flag}".strip()

_disable_cache = os.environ.get("SFINCS_JAX_DISABLE_COMPILATION_CACHE", "").strip().lower()
if _disable_cache not in {"1", "true", "yes", "on"}:
    if not os.environ.get("JAX_COMPILATION_CACHE_DIR", "").strip():
        cache_override = os.environ.get("SFINCS_JAX_COMPILATION_CACHE_DIR", "").strip()
        if cache_override:
            default_cache_dir = cache_override
        else:
            xdg_cache = os.environ.get("XDG_CACHE_HOME", "").strip()
            if xdg_cache:
                default_cache_dir = os.path.join(xdg_cache, "sfincs_jax", "jax_compilation_cache")
            else:
                default_cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "sfincs_jax", "jax_compilation_cache")
        try:
            os.makedirs(default_cache_dir, exist_ok=True)
        except OSError:
            default_cache_dir = os.path.join(tempfile.gettempdir(), "sfincs_jax", "jax_compilation_cache")
            try:
                os.makedirs(default_cache_dir, exist_ok=True)
            except OSError:
                default_cache_dir = ""
        if default_cache_dir:
            os.environ["JAX_COMPILATION_CACHE_DIR"] = default_cache_dir
        os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")
        os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "0")

# SFINCS parity fixtures and most scientific use-cases rely on float64 accuracy.
# Set this as early as possible on package import.
try:
    from jax import config as _jax_config  # noqa: PLC0415

    _jax_config.update("jax_enable_x64", True)
    _cache_dir = os.environ.get("JAX_COMPILATION_CACHE_DIR", "").strip()
    if _cache_dir:
        try:  # pragma: no cover - best-effort cache enable
            from jax.experimental import compilation_cache as _compilation_cache  # noqa: PLC0415

            _compilation_cache.set_cache_dir(_cache_dir)
        except Exception:
            pass
except Exception:
    # Keep import lightweight for tooling that inspects the package without JAX.
    pass

__all__ = [
    "__version__",
]

__version__ = "0.0.1"
