"""Stable public data contracts for high-level DKX workflows.

The classes in this module describe orchestration boundaries: user inputs,
geometry/grid/operator summaries, solver metadata, output schemas, and
benchmark reports. They intentionally avoid importing JAX so they stay cheap to
import from the CLI, documentation, tests, and downstream workflow code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .inputs import SfincsInput


def _immutable_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType(dict(value or {}))


def _path_or_none(value: str | Path | None) -> Path | None:
    return None if value is None else Path(value)


@dataclass(frozen=True, slots=True)
class SolverOptions:
    """Typed tuning knobs for :func:`dkx.solve.solve`, threaded by the run drivers.

    Pass to :func:`dkx.run.run_profile`, :func:`dkx.run.run_transport_matrix`,
    or :func:`dkx.run.run_from_namelist` as ``solver=SolverOptions(...)``; when
    given, it supersedes the drivers' quick ``solve_method``/``tol`` arguments
    and is expanded to :func:`dkx.solve.solve` keywords via
    :meth:`solve_kwargs`.  Environment variables keep working as overrides for
    knobs left at ``None`` (``memory_budget_gb=None`` reads
    ``DKX_TIER1_MEMORY_BUDGET_GB`` inside the solve).

    Attributes:
        method: ``"auto"`` | ``"block_tridiagonal"`` | ``"gmres"`` |
            ``"direct"`` (the route policy of :func:`dkx.solve.solve`).
        tol: relative residual tolerance (per RHS column).
        atol: absolute residual floor.
        restart: FGMRES cycle size ``m`` (recycled Krylov route).
        recycle_dim: GCROT recycle directions ``k`` (recycled Krylov route).
        max_restarts: recycled-Krylov outer-cycle cap; exceeding it is what
            makes ``auto`` fall through to the sparse direct route.
        differentiable: wrap the solution in an implicit-function-theorem
            ``linear_solve`` so ``jax.grad`` flows through (structured direct and
            recycled Krylov routes).
        use_preconditioner: recycled-Krylov coarse-operator preconditioner on/off.
        preconditioner: which recycled-Krylov preconditioner to build —
            ``"coarse"`` (dense block-Thomas over ``L``), ``"multigrid"``
            (:mod:`dkx.multigrid`), ``"sparse"`` (:mod:`dkx.sparse_precond`,
            the same operator inverted exactly in a fill-reducing order), or
            ``"none"``.  ``None`` keeps whatever ``use_preconditioner``
            selects, so existing callers are unaffected.  A preconditioner
            cannot change the answer — only the iteration count, the wall
            time, and whether the solve fits in memory at all.
        device: JAX device for the solve (a platform string such as ``"cpu"``
            or ``"gpu"``, or a concrete ``jax.Device``); ``None`` keeps the
            operator's placement.
        memory_budget_gb: budget above which ``method="auto"`` prefers the
            memory-lean truncated structured-direct kernel over the full-band
            factorization; ``None`` reads ``DKX_TIER1_MEMORY_BUDGET_GB``,
            else the solve's default applies.
        cores: host CPU threadpool width.  XLA sizes its threadpool once,
            before the first JAX device use, so a value stored here CANNOT
            change a process whose JAX backend is already initialized; it is
            carried for provenance and planning only.  To actually pin
            threads, set the ``DKX_CORES`` environment variable or the CLI
            ``--cores`` flag before ``import dkx`` (see
            ``docs/parallelism.rst``); :meth:`solve_kwargs` deliberately
            excludes this field.
    """

    method: str = "auto"
    tol: float = 1.0e-10
    atol: float = 0.0
    restart: int = 30
    recycle_dim: int = 8
    max_restarts: int = 200
    differentiable: bool = False
    use_preconditioner: bool = True
    preconditioner: str | None = None
    device: Any = None
    memory_budget_gb: float | None = None
    cores: int | None = None

    def solve_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for :func:`dkx.solve.solve` (``cores`` excluded)."""
        return {
            "method": str(self.method),
            "tol": float(self.tol),
            "atol": float(self.atol),
            "restart": int(self.restart),
            "recycle_dim": int(self.recycle_dim),
            "max_restarts": int(self.max_restarts),
            "differentiable": bool(self.differentiable),
            "use_preconditioner": bool(self.use_preconditioner),
            "preconditioner": (
                None if self.preconditioner is None else str(self.preconditioner)
            ),
            "device": self.device,
            "tier1_memory_budget_gb": (
                None if self.memory_budget_gb is None else float(self.memory_budget_gb)
            ),
        }


@dataclass(frozen=True, slots=True)
class SolveInputs:
    """Normalized high-level solve request passed across API/CLI boundaries."""

    input_path: Path | None = None
    wout_path: Path | None = None
    output_path: Path | None = None
    backend: str | None = None
    requires_autodiff: bool = False
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_path", _path_or_none(self.input_path))
        object.__setattr__(self, "wout_path", _path_or_none(self.wout_path))
        object.__setattr__(self, "output_path", _path_or_none(self.output_path))
        object.__setattr__(self, "options", _immutable_mapping(self.options))


@dataclass(frozen=True, slots=True)
class GeometryState:
    """Geometry contract exposed to problem setup and validation layers."""

    kind: str
    source_path: Path | None = None
    radial_coordinate: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_path", _path_or_none(self.source_path))
        object.__setattr__(self, "metadata", _immutable_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class GridState:
    """Phase-space grid sizes and layout metadata."""

    n_theta: int
    n_zeta: int
    n_xi: int
    n_x: int
    n_species: int = 1
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _immutable_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class OperatorState:
    """Operator summary shared by problem, solver, and diagnostics layers."""

    rhs_mode: int
    size: int
    collision_model: str
    include_phi1: bool = False
    matrix_free: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _immutable_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class PreconditionerState:
    """Selected preconditioner capability and setup metadata."""

    kind: str
    differentiable: bool
    device_safe: bool
    setup_memory_mb: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _immutable_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class SolverResult:
    """Backend-neutral solver summary for API-level callers."""

    residual_norm: float
    converged: bool
    iterations: int | None = None
    runtime_s: float | None = None
    solution: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _immutable_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class TransportResult:
    """Transport-output summary independent of a specific file format."""

    transport_matrix: Any = None
    particle_flux: Any = None
    heat_flux: Any = None
    bootstrap_current: Any = None
    solver: SolverResult | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _immutable_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class OutputSchema:
    """Versioned output-file schema and field-key contract."""

    format: str
    version: str
    keys: Sequence[str] = ()
    path: Path | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _path_or_none(self.path))
        object.__setattr__(self, "keys", tuple(self.keys))
        object.__setattr__(self, "metadata", _immutable_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    """Runtime, memory, and status summary for one benchmark case."""

    case: str
    backend: str
    runtime_s: float
    peak_memory_mb: float
    status: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _immutable_mapping(self.metadata))


def _solve_request_paths(
    request: SolveInputs | str | Path,
    output_path: str | Path | None = None,
) -> tuple[Path, Path | None, Path | None, str | None, bool, Mapping[str, Any]]:
    if isinstance(request, SolveInputs):
        if request.input_path is None:
            raise ValueError("SolveInputs.input_path is required for this API call.")
        resolved_output = _path_or_none(output_path) if output_path is not None else request.output_path
        return (
            request.input_path,
            request.wout_path,
            resolved_output,
            request.backend,
            bool(request.requires_autodiff),
            request.options,
        )
    return Path(request), None, _path_or_none(output_path), None, False, MappingProxyType({})


def write_output(
    request: SolveInputs | SfincsInput | str | Path,
    output_path: str | Path | None = None,
    **kwargs: Any,
) -> Path:
    """Run ``dkx`` from Python and write an output file.

    ``request`` may be a :class:`SolveInputs` object, an input namelist path,
    or an in-memory :class:`dkx.inputs.SfincsInput` (no file is read or
    written for the input; the output's ``input.namelist`` provenance dataset
    stores :meth:`~dkx.inputs.SfincsInput.to_namelist` text when the input
    was built programmatically).  Routes to
    :func:`dkx.run.run_from_namelist` (the canonical RHSMode
    dispatch behind ``dkx write-output``).  The implementation imports
    the heavy run stack lazily so importing ``dkx.api`` stays cheap for
    docs, CLI validation, and downstream workflow planning.
    """

    if isinstance(request, SfincsInput):
        resolved_output = _path_or_none(output_path)
        if resolved_output is None:
            raise ValueError("output_path is required when the request is an in-memory SfincsInput.")
        inp: SfincsInput = request
        wout_override = kwargs.pop("wout_path", None)
        if wout_override is not None:
            from .input_compat import with_equilibrium_override  # noqa: PLC0415
            from .inputs import parse_sfincs_input_text, sfincs_input_from_raw  # noqa: PLC0415

            raw = inp.raw if inp.raw is not None else parse_sfincs_input_text(inp.to_namelist())
            inp = sfincs_input_from_raw(with_equilibrium_override(nml=raw, wout_path=wout_override))

        from .run import run_from_namelist  # noqa: PLC0415

        run = run_from_namelist(inp, out_path=resolved_output, **kwargs)
        return Path(run.output_path)

    input_path, wout_path, resolved_output, _backend, _requires_autodiff, options = _solve_request_paths(
        request,
        output_path,
    )
    if resolved_output is None:
        raise ValueError("output_path is required when SolveInputs.output_path is not set.")
    for key, value in options.items():
        kwargs.setdefault(str(key), value)
    wout_path = kwargs.pop("wout_path", wout_path)

    import contextlib  # noqa: PLC0415
    import tempfile  # noqa: PLC0415

    from .run import run_from_namelist  # noqa: PLC0415

    @contextlib.contextmanager
    def _namelist_with_override():
        if wout_path is None:
            yield Path(input_path)
            return
        from .input_compat import with_equilibrium_override  # noqa: PLC0415
        from .namelist import read_sfincs_input  # noqa: PLC0415

        nml = with_equilibrium_override(nml=read_sfincs_input(Path(input_path)), wout_path=wout_path)
        if nml.source_text is None:
            raise ValueError("wout_path overrides require a readable input.namelist source text.")
        tmp = tempfile.NamedTemporaryFile(
            "w",
            dir=str(Path(input_path).resolve().parent),
            prefix=f".{Path(input_path).stem}.override.",
            suffix=".namelist",
            delete=False,
            encoding="utf-8",
        )
        try:
            tmp.write(nml.source_text)
            tmp.close()
            yield Path(tmp.name)
        finally:
            Path(tmp.name).unlink(missing_ok=True)

    with _namelist_with_override() as namelist_path:
        run = run_from_namelist(
            namelist_path,
            out_path=resolved_output,
            **kwargs,
        )
    return Path(run.output_path)


def read_output(path: str | Path) -> dict[str, Any]:
    """Read an HDF5, NetCDF, or NPZ ``dkx`` output file."""

    from .io import read_sfincs_output_file  # noqa: PLC0415

    return read_sfincs_output_file(Path(path))


def run_ambipolar_brent(
    request: SolveInputs | str | Path,
    *,
    work_dir: str | Path | None = None,
    er_min: float,
    er_max: float,
    er_initial: float = 0.0,
    max_evaluations: int = 20,
    current_tolerance: float = 1.0e-10,
    step_tolerance: float = 1.0e-8,
    solve_method: str = "auto",
    differentiable: bool | None = None,
    reuse_output_geometry_cache: bool = True,
    reuse_solver_state: bool = True,
    emit: Any = None,
    **kwargs: Any,
) -> Any:
    """Run the canonical-stack Brent ambipolar ``E_r`` solve (stable public facade).

    Routes to :func:`dkx.er.find_ambipolar_er` (the canonical
    ``inputs -> drift_kinetic -> solve -> moments`` slice, replacing the legacy
    in-process Brent owner).  ``work_dir``, ``step_tolerance``,
    ``reuse_output_geometry_cache`` and ``reuse_solver_state`` are accepted for
    backwards compatibility; warm starts / recycling are threaded internally.

    Returns:
        A :class:`dkx.er.AmbipolarResult` (``.er`` is the ambipolar
        ``E_r``; ``.roots`` lists every classified root).
    """
    input_path, _wout_path, _output_path, _backend, _requires_autodiff, _options = (
        _solve_request_paths(request)
    )
    del work_dir, step_tolerance, differentiable
    del reuse_output_geometry_cache, reuse_solver_state, kwargs

    from .er import find_ambipolar_er  # noqa: PLC0415

    return find_ambipolar_er(
        input_path,
        er_bracket=(float(er_min), float(er_max)),
        er_initial=float(er_initial),
        max_iter=int(max_evaluations),
        current_tol=float(current_tolerance),
        solve_method=str(solve_method),
        emit=emit,
    )


def run_monoenergetic_database(
    request: SolveInputs | str | Path,
    nu_prime_grid: Sequence[float],
    e_star_grid: Sequence[float] = (0.0,),
    *,
    output_path: str | Path | None = None,
    solve_method: str = "auto",
    tol: float = 1.0e-10,
    emit: Any = None,
) -> Any:
    """Scan (nuPrime, EStar) monoenergetic coefficients (stable public facade).

    Routes to :func:`dkx.monoenergetic.monoenergetic_database` (the
    canonical RHSMode=3 database scan) and optionally writes the compact
    ``.npz`` database via :func:`dkx.monoenergetic.save_database`.
    The heavy stack is imported lazily so ``dkx.api`` stays cheap to
    import.

    Returns:
        The :class:`dkx.monoenergetic.MonoenergeticDatabase`.
    """

    input_path, _wout_path, resolved_output, _backend, _requires_autodiff, _options = (
        _solve_request_paths(request, output_path)
    )
    if input_path is None:
        raise ValueError("run_monoenergetic_database requires an input namelist path.")

    from .monoenergetic import monoenergetic_database, save_database  # noqa: PLC0415

    database = monoenergetic_database(
        input_path,
        nu_prime_grid,
        e_star_grid,
        solve_method=str(solve_method),
        tol=float(tol),
        emit=emit,
    )
    if resolved_output is not None:
        save_database(resolved_output, database)
    return database


def momentum_corrected_bootstrap(
    database: Any,
    *,
    z_s: Any,
    m_hats: Any,
    n_hats: Any,
    t_hats: Any,
    nu_n: Any,
    dn_hat_dpsi_hat: Any,
    dt_hat_dpsi_hat: Any,
    dphi_hat_dpsi_hat: Any = 0.0,
    e_par_b: Any = 0.0,
    uncorrected_flows: Any = None,
    x: Any = None,
    x_weights: Any = None,
    n_x: int = 64,
    x_max: float = 5.0,
) -> Any:
    """Momentum-corrected bootstrap current (stable public facade).

    Routes to :func:`dkx.momentum_correction.momentum_corrected_bootstrap`,
    the Sugama-Nishimura moment-method parallel-momentum correction on the
    monoenergetic transport coefficients (H. Sugama and S. Nishimura, Phys.
    Plasmas 9, 4637 (2002); 15, 042502 (2008); H. Maassberg, C. D. Beidler,
    and Y. Turkin, Phys. Plasmas 16, 072504 (2009)).  The heavy JAX stack is
    imported lazily so ``dkx.api`` stays cheap to import.

    Args:
        database: a
            :class:`dkx.monoenergetic.MonoenergeticDatabase` (from
            :func:`run_monoenergetic_database`).
        z_s, m_hats, n_hats, t_hats: species parameters, shape ``(S,)``.
        nu_n: deck normalized collisionality.
        dn_hat_dpsi_hat, dt_hat_dpsi_hat: radial gradients ``dn/dpsiHat``,
            ``dT/dpsiHat`` per species (shape ``(S,)``).
        dphi_hat_dpsi_hat, e_par_b: radial-electric-field and inductive
            parallel-field drives (default 0).
        uncorrected_flows: optional ``(S,)`` override for the uncorrected
            parallel-flow moments (e.g. from a kinetic solve).
        x, x_weights, n_x, x_max: speed quadrature controls.

    Returns:
        A :class:`dkx.momentum_correction.MomentumCorrectionResult`
        (``.corrected_bootstrap``, ``.uncorrected_bootstrap``,
        ``.delta_bootstrap``, ``.corrected_flows``, ...).
    """

    from .momentum_correction import (  # noqa: PLC0415
        momentum_corrected_bootstrap as _momentum_corrected_bootstrap,
    )

    return _momentum_corrected_bootstrap(
        database,
        z_s=z_s,
        m_hats=m_hats,
        n_hats=n_hats,
        t_hats=t_hats,
        nu_n=nu_n,
        dn_hat_dpsi_hat=dn_hat_dpsi_hat,
        dt_hat_dpsi_hat=dt_hat_dpsi_hat,
        dphi_hat_dpsi_hat=dphi_hat_dpsi_hat,
        e_par_b=e_par_b,
        uncorrected_flows=uncorrected_flows,
        x=x,
        x_weights=x_weights,
        n_x=n_x,
        x_max=x_max,
    )


def bounce_averaged_transport(
    geometry: Any,
    *,
    r_eff: float | None = None,
    grad_psi_avg: float | None = None,
    n_field_periods: int = 160,
    points_per_period: int = 48,
    n_pitch: int = 128,
    n_quad: int = 14,
    max_wells: int = 224,
    n_field_lines: int = 1,
    m_keep: int | None = None,
    n_keep: int | None = None,
) -> Any:
    """Bounce-averaged ``1/nu`` effective-ripple transport (stable public facade).

    Routes to :func:`dkx.bounce_averaged.bounce_averaged_transport`: the
    differentiable, radially-local, bounce-averaged surrogate for the dominant
    low-collisionality (``1/nu``-regime) neoclassical radial transport of a flux
    surface -- the effective ripple ``epsilon_eff`` and the trapped-particle
    bounce integrals of the radial magnetic drift -- computed from the ``|B|``
    Boozer spectrum of ``geometry`` (J. L. Velasco et al., J. Comput. Phys. 418,
    109512 (2020); Nucl. Fusion 61, 116059 (2021); spectrally-accurate
    differentiable bounce points as in arXiv:2412.01724).  Pure JAX, jit/vmap
    safe, and differentiable in the Boozer amplitudes when ``geometry`` is built
    with :meth:`FluxSurfaceGeometry.from_fourier`.  The heavy JAX stack is
    imported lazily so ``dkx.api`` stays cheap to import.

    Args:
        geometry: a
            :class:`dkx.magnetic_geometry.FluxSurfaceGeometry`.
        r_eff, grad_psi_avg: the ``<|grad psi|>`` normalization of
            ``epsilon_eff`` (large-aspect-ratio ``B_0 r_eff`` or an explicit
            average); the ``|grad psi|``-free ``gamma_c`` core needs neither.
        n_field_periods, points_per_period, n_pitch, n_quad, max_wells, n_field_lines, m_keep, n_keep:
            quadrature/bandwidth controls (see the owning module).

    Returns:
        A :class:`dkx.bounce_averaged.BounceAveragedTransport`
        (``.epsilon_eff``, ``.epsilon_eff_32``, ``.gamma_c``, ...).
    """
    from .bounce_averaged import bounce_averaged_transport as _bounce_averaged_transport  # noqa: PLC0415

    return _bounce_averaged_transport(
        geometry,
        r_eff=r_eff,
        grad_psi_avg=grad_psi_avg,
        n_field_periods=n_field_periods,
        points_per_period=points_per_period,
        n_pitch=n_pitch,
        n_quad=n_quad,
        max_wells=max_wells,
        n_field_lines=n_field_lines,
        m_keep=m_keep,
        n_keep=n_keep,
    )


def prepare_er_scan(
    case: Any, *, surface_index: int = 0, differentiable_profiles: bool = False,
    quadrature_order: int = 128,
) -> Any:
    """Prepare one native Case surface for repeated scans, without a solve.

    Build outside JAX transformations. The returned ``ErProblem`` accepts
    electric fields in kV/m and retains the case's solver method/tolerance.
    Its scan moments and radial current retain the normalized expert-operator
    conventions. By default profiles are fixed. With ``differentiable_profiles``
    enabled, ``problem.with_profiles(density_m3=..., temperature_keV=...)``
    refreshes profiles, radial drives and collisions inside JAX transformations.
    Arrays cover all surfaces/species; geometry, normalization, Coulomb logarithm
    and layout stay fixed. Full-FP uses the opt-in differentiable quadrature
    builder, with explicit ``quadrature_order`` per panel; PAS is analytic.
    This does not enable factor reuse or differentiate geometry/Case parsing.
    """
    import operator  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    from .config import Case  # noqa: PLC0415
    from .er import ErProblem  # noqa: PLC0415
    from .execution import _make_operator, _prepare_profile, _prepare_profile_builder, _route_name, _validate_native_slice  # noqa: PLC0415

    if not isinstance(case, Case):
        raise TypeError("prepare_er_scan requires a native Case")
    _validate_native_slice(case)
    index = operator.index(surface_index)
    if not 0 <= index < len(case.geometry.surfaces):
        raise IndexError("surface_index is outside the Case's surfaces")
    geometry, grids, density, temperature, dn, dt = _prepare_profile(case)
    op, _, _, radial = _make_operator(
        case, surface_index=index, n_hat=density / 1e20, t_hat=temperature,
        dn_dr_hat=dn, dt_dr_hat=dt, grids=grids, geometry_state=geometry,
        electric_field_kv_m=1.0, force_exb_structure=True,
        build_collisions=not differentiable_profiles,
    )
    initial = case.electric_field.value_kV_m or 0.0
    bounds = case.electric_field.search_kV_m or (initial - 5.0, initial + 5.0)
    profile_builder = None
    if differentiable_profiles:
        profile_builder = _prepare_profile_builder(case, op, grids, geometry, radial, index, quadrature_order)
        op = profile_builder(density, temperature)
    return ErProblem(
        operator=op, dphi_per_er=float(op.dphi_hat_dpsi_hat_kinetic),
        z_s=np.asarray(op.z_s), er_initial=initial, er_min=bounds[0], er_max=bounds[1],
        solve_method=_route_name(case.solver.method), tol=case.solver.relative_tolerance,
        er_units="kV/m",
        _profile_builder=profile_builder,
    )


def batched_er_scan(
    request: "SolveInputs | str | Path | Any",
    er_values: Any,
    *,
    er_bracket: tuple[float, float] | None = None,
    er_initial: float | None = None,
    solve_method: str | None = None,
    tol: float | None = None,
    differentiable: bool = False,
    max_batch: int | None = None,
    memory_budget_gb: float | None = None,
    devices: Sequence[Any] | str | None = None,
    retain_legendre_tail: bool = False,
    retain_full_state: bool = False,
) -> Any:
    """Batched ``E_r`` scan on one geometry (stable public facade).

    Routes to :func:`dkx.batch.batched_er_scan`: one ``jax.vmap`` batched
    solve over a vector of radial-electric-field values sharing a single
    geometry, returning batched states, moments, and the radial current ``J_r``
    per ``E_r``.  Auto-chunked to a memory-budgeted batch size; differentiable
    and jit-safe with a prepared problem. Build deck inputs outside JAX
    transformations. The heavy JAX/batch stack is imported lazily so
    ``dkx.api`` stays cheap to import.

    Args:
        request: a prepared :class:`dkx.er.ErProblem`, or a deck
            (:class:`SolveInputs` / namelist path) that is prepared with
            :func:`dkx.er.prepare` (RHSMode=1, ``inputRadialCoordinate=4``
            ``Er`` knob).
        er_values: the ``E_r`` scan values, shape ``(batch,)``.
        er_bracket, er_initial: optional bracket / initial ``E_r`` forwarded to
            :func:`dkx.er.prepare` when ``request`` is a deck.
        solve_method, tol: optional overrides. A prepared problem retains its
            settings when omitted; a deck defaults to auto and 1e-10.
        differentiable: differentiable implicit solves (for ``jax.grad``).
        max_batch, memory_budget_gb: optional memory-budgeting overrides.
        devices: None for one device, "auto" for local devices, or a sequence
            of distinct devices. Forwarded to :func:`dkx.batch.batched_solve`.
        retain_full_state: recover all Legendre blocks for full-state residual
            certification/restart; includes the extra cost in chunk sizing.
        retain_legendre_tail: retain the selected-tail relative-L2 upper bound
            when the memory-lean truncated route executes.

    Returns:
        A :class:`dkx.batch.BatchedSolveResult` (``radial_current``
        populated).
    """
    from .batch import batched_er_scan as _batched_er_scan  # noqa: PLC0415
    from .er import ErProblem, prepare  # noqa: PLC0415

    if isinstance(request, ErProblem):
        problem = request
    else:
        input_path, _wout, _out, _backend, _autodiff, _options = _solve_request_paths(request)
        problem = prepare(
            input_path,
            solve_method=solve_method if solve_method is not None else "auto",
            tol=tol if tol is not None else 1.0e-10,
            er_bracket=er_bracket,
            er_initial=er_initial,
        )
    return _batched_er_scan(
        problem,
        er_values,
        solve_method=solve_method,
        tol=tol,
        differentiable=differentiable,
        max_batch=max_batch,
        memory_budget_gb=memory_budget_gb,
        devices=devices,
        retain_legendre_tail=retain_legendre_tail,
        retain_full_state=retain_full_state,
    )


def batched_surface_scan(
    surfaces: Sequence[Any],
    *,
    solve_method: str = "auto",
    tol: float = 1.0e-10,
    differentiable: bool = False,
    max_batch: int | None = None,
    memory_budget_gb: float | None = None,
    devices: Sequence[Any] | str | None = None,
    retain_legendre_tail: bool = False,
    retain_full_state: bool = False,
) -> Any:
    """Batched solve over a batch of flux surfaces (stable public facade).

    Routes to :func:`dkx.batch.batched_surface_scan`: a ``jax.vmap`` batched
    solve over a sequence of flux-surface operators that share discretization
    (grids/derivative matrices/layout) but differ in geometry, species,
    collision, and drive leaves.  Auto-chunked to a memory-budgeted batch size;
    differentiable and jit-safe with built operators. Build deck inputs outside
    JAX transformations.

    Args:
        surfaces: a sequence whose entries are each a built
            :class:`dkx.drift_kinetic.KineticOperator`, or a deck
            (:class:`SolveInputs` / namelist path) built into one per surface.
        solve_method, tol, differentiable: forwarded to the solve.
        max_batch, memory_budget_gb: optional memory-budgeting overrides.
        devices: None for one device, "auto" for local devices, or a sequence
            of distinct devices. Forwarded to :func:`dkx.batch.batched_solve`.
        retain_full_state: recover all Legendre blocks for full-state residual
            certification/restart; includes the extra cost in chunk sizing.
        retain_legendre_tail: retain the selected-tail relative-L2 upper bound
            when the memory-lean truncated route executes.

    Returns:
        A :class:`dkx.batch.BatchedSolveResult` with batched states and
        moments.
    """
    from .batch import batched_surface_scan as _batched_surface_scan  # noqa: PLC0415
    from .drift_kinetic import KineticOperator, kinetic_operator_from_namelist  # noqa: PLC0415
    from .inputs import load_sfincs_input  # noqa: PLC0415

    operators = []
    for surface in surfaces:
        if isinstance(surface, KineticOperator):
            operators.append(surface)
            continue
        input_path, _wout, _out, _backend, _autodiff, _options = _solve_request_paths(surface)
        operators.append(kinetic_operator_from_namelist(load_sfincs_input(input_path).raw))
    return _batched_surface_scan(
        operators,
        solve_method=solve_method,
        tol=tol,
        differentiable=differentiable,
        max_batch=max_batch,
        memory_budget_gb=memory_budget_gb,
        devices=devices,
        retain_legendre_tail=retain_legendre_tail,
        retain_full_state=retain_full_state,
    )


__all__ = [
    "BenchmarkReport",
    "GeometryState",
    "GridState",
    "OperatorState",
    "OutputSchema",
    "PreconditionerState",
    "SfincsInput",
    "SolveInputs",
    "SolverOptions",
    "SolverResult",
    "TransportResult",
    "prepare_er_scan",
    "batched_er_scan",
    "batched_surface_scan",
    "bounce_averaged_transport",
    "momentum_corrected_bootstrap",
    "read_output",
    "run_ambipolar_brent",
    "run_monoenergetic_database",
    "write_output",
]
