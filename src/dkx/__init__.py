"""Differentiable neoclassical transport solvers and SFINCS-style outputs in JAX.

The public CLI and Python APIs are maintained as standalone research tools while
retaining release-gated comparisons against SFINCS Fortran v3 for trust building.

Importing this package is inert: it reads no environment, writes none, creates
no directory, imports no JAX backend, and chooses no thread count. The runtime
those things configure lives in :mod:`dkx.runtime` and is applied by
:func:`dkx.runtime.configure`, which the CLI and every solve entry point call
for you. See plan.md section 6.4.
"""

from __future__ import annotations

from ._version import __version__
from . import runtime
from .runtime import configure, initialize_distributed_runtime_from_env

from .api import (  # noqa: E402
    BenchmarkReport,
    GeometryState,
    GridState,
    OperatorState,
    OutputSchema,
    PreconditionerState,
    SolveInputs,
    SolverOptions,
    SolverResult,
    TransportResult,
    batched_er_scan,
    batched_surface_scan,
    read_output,
    run_ambipolar_brent,
    run_monoenergetic_database,
    write_output,
)
from .inputs import SfincsInput, load_sfincs_input  # noqa: E402
from .config import (  # noqa: E402
    Case,
    CaseValidationError,
    ConvergenceConfig,
    ElectricFieldConfig,
    GeometryConfig,
    OutputConfig,
    ParallelConfig,
    PhysicsConfig,
    ResolutionConfig,
    RunConfig,
    ScanAxis,
    ScanConfig,
    SolverConfig,
    SpeciesConfig,
    case_json_schema,
)
from .result import RESULT_SCHEMA_VERSION, Result  # noqa: E402

# Heavy flagship entry points (they import the JAX solve stack) are exported
# lazily via PEP 562 module __getattr__ so `import dkx` stays cheap.
_LAZY_EXPORTS = {
    "plot": ("dkx.plotting", "plot"),
    "run_profile": ("dkx.run", "run_profile"),
    "run_transport_matrix": ("dkx.run", "run_transport_matrix"),
    "run_from_namelist": ("dkx.run", "run_from_namelist"),
    "batched_solve": ("dkx.batch", "batched_solve"),
    "monoenergetic_database": ("dkx.monoenergetic", "monoenergetic_database"),
    "ambipolar_er": ("dkx.er", "ambipolar_er"),
    "find_ambipolar_er": ("dkx.er", "find_ambipolar_er"),
    "classical_impurity_flux": ("dkx.impurity", "classical_impurity_flux"),
    "build_impurity_plasma": ("dkx.impurity", "build_impurity_plasma"),
}


def __getattr__(name: str):
    if name == "run":
        return _lazy_run_module()
    try:
        module_name, attr = _LAZY_EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    import importlib  # noqa: PLC0415

    value = getattr(importlib.import_module(module_name), attr)
    globals()[name] = value  # cache: subsequent lookups skip __getattr__
    return value


def __dir__() -> list[str]:
    # `annotations` is the __future__ import, not an export. Everything else
    # public is either in __all__ or a submodule the user reached for.
    return sorted((set(globals()) | set(_LAZY_EXPORTS)) - {"annotations"})


# ``dkx/run.py`` is a module and ``dkx.run(case)`` is a call.  Rather than pick
# one -- the function breaks ``dkx.run.run_profile`` and every monkeypatch that
# targets it by path, the module breaks the call -- ``dkx/run.py`` makes itself
# callable, so ``dkx.run`` is always the module and always invocable.  The name
# is resolved here rather than listed in _LAZY_EXPORTS because what it yields
# is the module itself, not an attribute of one.
def _lazy_run_module():
    import importlib  # noqa: PLC0415

    module = importlib.import_module(f"{__name__}.run")
    globals()["run"] = module
    return module



def require_float64() -> None:
    """Raise unless JAX is in float64 mode.

    :func:`dkx.runtime.configure` enables it, and every solve entry point calls
    that; a caller who set ``DKX_NO_X64_SETUP`` or passed ``jax_x64=False`` has
    taken the job on, and this is where they find out if they dropped it.  The
    check is a dtype probe rather than a config read because the config can be
    set and then overridden, and what matters is the dtype arrays actually get.
    """
    import jax.numpy as _jnp  # noqa: PLC0415

    if _jnp.zeros(1).dtype != _jnp.float64:
        raise RuntimeError(
            "dkx requires JAX float64: the block eliminations and every parity "
            "fixture depend on it, and single precision changes which results "
            "are trustworthy rather than merely how accurate they are. "
            "Enable it with dkx.runtime.configure() before the first array is "
            "created, or jax.config.update('jax_enable_x64', True), or "
            "JAX_ENABLE_X64=1 in the environment, or unset DKX_NO_X64_SETUP "
            "and let dkx set it."
        )

__all__ = [
    "configure",
    "runtime",
    "require_float64",
    "BenchmarkReport",
    "Case",
    "CaseValidationError",
    "ConvergenceConfig",
    "ElectricFieldConfig",
    "GeometryConfig",
    "GeometryState",
    "GridState",
    "OperatorState",
    "OutputSchema",
    "OutputConfig",
    "ParallelConfig",
    "PhysicsConfig",
    "PreconditionerState",
    "RESULT_SCHEMA_VERSION",
    "ResolutionConfig",
    "Result",
    "RunConfig",
    "SfincsInput",
    "SolveInputs",
    "SolverOptions",
    "SolverResult",
    "ScanAxis",
    "ScanConfig",
    "SolverConfig",
    "SpeciesConfig",
    "TransportResult",
    "__version__",
    "ambipolar_er",
    "batched_er_scan",
    "batched_surface_scan",
    "case_json_schema",
    "batched_solve",
    "build_impurity_plasma",
    "classical_impurity_flux",
    "find_ambipolar_er",
    "initialize_distributed_runtime_from_env",
    "load_sfincs_input",
    "monoenergetic_database",
    "read_output",
    "run_ambipolar_brent",
    "run_from_namelist",
    "run_monoenergetic_database",
    "plot",
    "run",
    "run_profile",
    "run_transport_matrix",
    "write_output",
]
