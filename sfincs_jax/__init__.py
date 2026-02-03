"""SFINCS (v3) port to JAX.

This repository is parity-first: we start by matching the Fortran v3 numerics and outputs
for selected examples, then expand coverage over time.
"""

from __future__ import annotations

# SFINCS parity fixtures and most scientific use-cases rely on float64 accuracy.
# Set this as early as possible on package import.
try:
    from jax import config as _jax_config  # noqa: PLC0415

    _jax_config.update("jax_enable_x64", True)
except Exception:
    # Keep import lightweight for tooling that inspects the package without JAX.
    pass

__all__ = [
    "__version__",
]

__version__ = "0.0.1"
