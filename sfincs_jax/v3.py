from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass
from pathlib import Path

from jax import config as _jax_config
_jax_config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from .geometry import BoozerGeometry, boozer_geometry_from_bc_file, boozer_geometry_scheme4
from .grids import uniform_diff_matrices
from .namelist import Namelist
from .paths import resolve_existing_path
from .vmec_geometry import vmec_geometry_from_wout_file
from .vmec_wout import read_vmec_wout
from .xgrid import XGrid, make_x_grid, make_x_polynomial_diff_matrices


def _n_periods_from_bc_file(path: str, *, base_dir: Path | None = None) -> int:
    """Read `NPeriods` from a Boozer `.bc` file header used by v3 geometryScheme=11/12.

    In upstream Fortran v3 runs, ``equilibriumFile`` is commonly specified as a **relative**
    path, resolved relative to the run directory (i.e. the directory containing
    ``input.namelist``).
    """
    repo_root = Path(__file__).resolve().parents[1]
    extra = (repo_root / "tests" / "ref", repo_root / "sfincs_jax" / "data" / "equilibria")
    p = resolve_existing_path(path, base_dir=base_dir, extra_search_dirs=extra).path
    with open(p, "r") as f:
        for line in f:
            if line.startswith("CC"):
                continue
            # First non-comment line contains:
            # m0b n0b nsurf nper flux a R ...
            parts = line.split()
            if len(parts) < 4:
                raise ValueError(f"Unexpected .bc header line: {line!r}")
            try:
                return int(parts[3])
            except ValueError:
                # Some files include an extra header line with column names.
                continue
    raise ValueError(f"Unable to find header line in {str(p)!r}")


def _resolve_vmec_equilibrium_file(
    path: str,
    *,
    base_dir: Path | None,
    extra_search_dirs: tuple[Path, ...],
) -> Path:
    """Resolve `equilibriumFile` for `geometryScheme=5`, with a `.txt -> .nc` fallback.

    Upstream v3 examples sometimes point to a VMEC ASCII `wout_*.txt` file, but many repositories
    also ship an equivalent netCDF `wout_*.nc`. `sfincs_jax` currently reads netCDF and will
    automatically use the `.nc` sibling if the `.txt` path cannot be resolved.
    """
    try:
        return resolve_existing_path(path, base_dir=base_dir, extra_search_dirs=extra_search_dirs).path
    except FileNotFoundError:
        p = Path(str(path).strip().strip('"').strip("'"))
        if p.suffix.lower() not in {".txt", ".dat"}:
            raise
        p2 = p.with_suffix(".nc")
        return resolve_existing_path(str(p2), base_dir=base_dir, extra_search_dirs=extra_search_dirs).path


def _hashable_value(val) -> object:
    if isinstance(val, list):
        return tuple(_hashable_value(v) for v in val)
    if isinstance(val, dict):
        return tuple(sorted((str(k), _hashable_value(v)) for k, v in val.items()))
    return val


def _group_key(group: dict, keys: list[str]) -> tuple[tuple[str, object], ...]:
    items = []
    for key in keys:
        items.append((key.upper(), _hashable_value(group.get(key.upper(), None))))
    return tuple(items)


def _equilibrium_file_key(*, nml: Namelist, geometry_scheme: int, geom_group: dict) -> tuple[str, float] | None:
    equilibrium_file = geom_group.get("EQUILIBRIUMFILE", None)
    if equilibrium_file is None:
        return None
    base_dir = nml.source_path.parent if nml.source_path is not None else None
    repo_root = Path(__file__).resolve().parents[1]
    extra = (repo_root / "tests" / "ref", repo_root / "sfincs_jax" / "data" / "equilibria")
    if geometry_scheme == 5:
        path = _resolve_vmec_equilibrium_file(str(equilibrium_file), base_dir=base_dir, extra_search_dirs=extra)
    else:
        path = resolve_existing_path(str(equilibrium_file), base_dir=base_dir, extra_search_dirs=extra).path
    try:
        mtime = float(path.stat().st_mtime)
    except FileNotFoundError:
        mtime = -1.0
    return (str(path), mtime)


_GRIDS_CACHE: dict[tuple[object, ...], V3Grids] = {}
_GEOMETRY_CACHE: dict[tuple[object, ...], BoozerGeometry] = {}

_GEOMETRY_CACHE_FIELDS = (
    "b_hat",
    "db_hat_dtheta",
    "db_hat_dzeta",
    "d_hat",
    "b_hat_sup_theta",
    "b_hat_sup_zeta",
    "b_hat_sub_theta",
    "b_hat_sub_zeta",
    "b_hat_sub_psi",
    "db_hat_dpsi_hat",
    "db_hat_sub_psi_dtheta",
    "db_hat_sub_psi_dzeta",
    "db_hat_sub_theta_dpsi_hat",
    "db_hat_sub_zeta_dpsi_hat",
    "db_hat_sub_theta_dzeta",
    "db_hat_sub_zeta_dtheta",
    "db_hat_sup_theta_dpsi_hat",
    "db_hat_sup_theta_dzeta",
    "db_hat_sup_zeta_dpsi_hat",
    "db_hat_sup_zeta_dtheta",
)


def _geometry_cache_dir() -> Path | None:
    cache_dir_env = os.environ.get("SFINCS_JAX_GEOMETRY_CACHE_DIR", "").strip()
    if cache_dir_env:
        cache_dir = Path(cache_dir_env).expanduser()
    else:
        xdg_cache = os.environ.get("XDG_CACHE_HOME", "").strip()
        if xdg_cache:
            cache_dir = Path(xdg_cache) / "sfincs_jax" / "geometry_cache"
        else:
            cache_dir = Path.home() / ".cache" / "sfincs_jax" / "geometry_cache"
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    return cache_dir


def _geometry_cache_enabled() -> bool:
    cache_env = os.environ.get("SFINCS_JAX_GEOMETRY_CACHE", "").strip().lower()
    if cache_env in {"0", "false", "no", "off"}:
        return False
    persist_env = os.environ.get("SFINCS_JAX_GEOMETRY_CACHE_PERSIST", "").strip().lower()
    if persist_env in {"0", "false", "no", "off"}:
        return False
    return True


def _geometry_cache_path(cache_key: tuple[object, ...]) -> Path | None:
    cache_dir = _geometry_cache_dir()
    if cache_dir is None:
        return None
    digest = hashlib.blake2b(repr(cache_key).encode("utf-8"), digest_size=16).hexdigest()
    return cache_dir / f"geom_{digest}.npz"


def _geometry_to_cache_payload(geom: BoozerGeometry, cache_key: tuple[object, ...]) -> dict[str, np.ndarray]:
    payload: dict[str, np.ndarray] = {
        "cache_version": np.asarray(1, dtype=np.int32),
        "cache_key": np.asarray(repr(cache_key)),
        "n_periods": np.asarray(int(geom.n_periods), dtype=np.int32),
        "b0_over_bbar": np.asarray(float(geom.b0_over_bbar), dtype=np.float64),
        "iota": np.asarray(float(geom.iota), dtype=np.float64),
        "g_hat": np.asarray(float(geom.g_hat), dtype=np.float64),
        "i_hat": np.asarray(float(geom.i_hat), dtype=np.float64),
    }
    for field in _GEOMETRY_CACHE_FIELDS:
        payload[field] = np.asarray(getattr(geom, field), dtype=np.float64)
    return payload


def _geometry_from_cache_payload(data: dict[str, np.ndarray]) -> BoozerGeometry | None:
    if int(np.asarray(data.get("cache_version", 0)).reshape(())) != 1:
        return None
    try:
        geom_kwargs = {
            "n_periods": int(np.asarray(data["n_periods"]).reshape(())),
            "b0_over_bbar": float(np.asarray(data["b0_over_bbar"]).reshape(())),
            "iota": float(np.asarray(data["iota"]).reshape(())),
            "g_hat": float(np.asarray(data["g_hat"]).reshape(())),
            "i_hat": float(np.asarray(data["i_hat"]).reshape(())),
        }
    except Exception:  # noqa: BLE001
        return None
    for field in _GEOMETRY_CACHE_FIELDS:
        if field not in data:
            return None
        geom_kwargs[field] = jnp.asarray(data[field], dtype=jnp.float64)
    return BoozerGeometry(**geom_kwargs)


def _load_geometry_cache(cache_key: tuple[object, ...]) -> BoozerGeometry | None:
    if not _geometry_cache_enabled():
        return None
    path = _geometry_cache_path(cache_key)
    if path is None or not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            return _geometry_from_cache_payload({k: data[k] for k in data.files})
    except Exception:  # noqa: BLE001
        return None


def _save_geometry_cache(cache_key: tuple[object, ...], geom: BoozerGeometry) -> None:
    if not _geometry_cache_enabled():
        return
    path = _geometry_cache_path(cache_key)
    if path is None:
        return
    try:
        payload = _geometry_to_cache_payload(geom, cache_key)
        np.savez_compressed(path, **payload)
    except Exception:
        return


@dataclass(frozen=True)
class V3Grids:
    theta: jnp.ndarray
    zeta: jnp.ndarray
    x: jnp.ndarray

    theta_weights: jnp.ndarray
    zeta_weights: jnp.ndarray
    x_weights: jnp.ndarray

    ddtheta: jnp.ndarray
    ddzeta: jnp.ndarray
    ddx: jnp.ndarray
    d2dx2: jnp.ndarray
    ddtheta_magdrift_plus: jnp.ndarray
    ddtheta_magdrift_minus: jnp.ndarray
    ddzeta_magdrift_plus: jnp.ndarray
    ddzeta_magdrift_minus: jnp.ndarray

    n_xi: int
    n_l: int
    n_xi_for_x: jnp.ndarray  # (Nx,) int32


def _get_int(group: dict, key: str, default: int) -> int:
    v = group.get(key.upper(), default)
    if isinstance(v, list):
        v = v[0] if v else default
    return int(v)


def _get_float(group: dict, key: str, default: float) -> float:
    v = group.get(key.upper(), default)
    if isinstance(v, list):
        v = v[0] if v else default
    return float(v)


def grids_from_namelist(nml: Namelist) -> V3Grids:
    """Construct v3 grids using the same defaults as the Fortran code (where implemented)."""
    cache_env = os.environ.get("SFINCS_JAX_GRIDS_CACHE", "").strip().lower()
    use_cache = cache_env not in {"0", "false", "no", "off"}
    res = nml.group("resolutionParameters")
    other = nml.group("otherNumericalParameters")
    geom = nml.group("geometryParameters")
    general = nml.group("general")

    ntheta = _get_int(res, "Ntheta", 15)
    nzeta = _get_int(res, "Nzeta", 15)
    nx = _get_int(res, "Nx", 5)
    nxi = _get_int(res, "Nxi", 16)
    nl = _get_int(res, "NL", 4)

    # SFINCS v3 defaults:
    force_odd = True
    if force_odd:
        if ntheta % 2 == 0:
            ntheta += 1
        if nzeta % 2 == 0:
            nzeta += 1

    theta_derivative_scheme = _get_int(other, "thetaDerivativeScheme", 2)
    zeta_derivative_scheme = _get_int(other, "zetaDerivativeScheme", 2)
    magnetic_drift_derivative_scheme = _get_int(other, "magneticDriftDerivativeScheme", 3)
    x_grid_scheme = _get_int(other, "xGridScheme", 5)
    x_grid_k = _get_float(other, "xGrid_k", 0.0)
    nxi_for_x_option = _get_int(other, "Nxi_for_x_option", 1)
    xdot_derivative_scheme = _get_int(other, "xDotDerivativeScheme", 0)
    rhs_mode = _get_int(general, "RHSMode", 1)

    # v3 validateInput() hard-overrides several settings for RHSMode=3.
    # Keep grid construction/output metadata consistent with upstream behavior.
    if rhs_mode == 3:
        nx = 1
        nxi_for_x_option = 0

    geometry_scheme = _get_int(geom, "geometryScheme", -1)
    if use_cache:
        res_key = _group_key(res, ["Ntheta", "Nzeta", "Nx", "Nxi", "NL"])
        other_key = _group_key(
            other,
            [
                "thetaDerivativeScheme",
                "zetaDerivativeScheme",
                "magneticDriftDerivativeScheme",
                "xGridScheme",
                "xGrid_k",
                "Nxi_for_x_option",
                "xDotDerivativeScheme",
            ],
        )
        general_key = _group_key(general, ["RHSMode"])
        geom_key = _group_key(
            geom,
            [
                "geometryScheme",
                "helicity_n",
            ],
        )
        eq_key = _equilibrium_file_key(nml=nml, geometry_scheme=geometry_scheme, geom_group=geom)
        cache_key = ("grids", res_key, other_key, general_key, geom_key, eq_key)
        cached = _GRIDS_CACHE.get(cache_key)
        if cached is not None:
            return cached
    if geometry_scheme == 4:
        n_periods = 5
    elif geometry_scheme == 1:
        # v3: NPeriods = max(1, helicity_n)
        helicity_n = _get_int(geom, "helicity_n", 10)
        n_periods = max(1, int(helicity_n))
    elif geometry_scheme == 2:
        # v3: fixed simplified LHD model
        n_periods = 10
    elif geometry_scheme == 3:
        # v3: fixed LHD inward-shifted model
        n_periods = 10
    elif geometry_scheme in {11, 12}:
        equilibrium_file = geom.get("EQUILIBRIUMFILE", None)
        if equilibrium_file is None:
            raise ValueError("geometryScheme=11/12 requires equilibriumFile in geometryParameters.")
        base_dir = nml.source_path.parent if nml.source_path is not None else None
        n_periods = _n_periods_from_bc_file(str(equilibrium_file), base_dir=base_dir)
    elif geometry_scheme == 5:
        equilibrium_file = geom.get("EQUILIBRIUMFILE", None)
        if equilibrium_file is None:
            raise ValueError("geometryScheme=5 requires equilibriumFile in geometryParameters.")
        base_dir = nml.source_path.parent if nml.source_path is not None else None
        repo_root = Path(__file__).resolve().parents[1]
        extra = (repo_root / "tests" / "ref", repo_root / "sfincs_jax" / "data" / "equilibria")
        p = _resolve_vmec_equilibrium_file(str(equilibrium_file), base_dir=base_dir, extra_search_dirs=extra)
        n_periods = int(read_vmec_wout(p).nfp)
    else:
        raise NotImplementedError(
            "Only geometryScheme in {1,2,3,4,5,11,12} is supported for grid construction so far."
        )

    # theta grid
    theta_scheme_map = {0: 20, 1: 0, 2: 10}
    theta_scheme = theta_scheme_map.get(theta_derivative_scheme)
    if theta_scheme is None:
        raise ValueError(f"Invalid thetaDerivativeScheme={theta_derivative_scheme}")
    theta, theta_weights, ddtheta, _ = uniform_diff_matrices(
        n=ntheta, x_min=0.0, x_max=2 * math.pi, scheme=theta_scheme
    )

    # Upwinded theta matrices for magnetic drifts (ddtheta_magneticDrift_plus/minus in v3).
    if magnetic_drift_derivative_scheme == 0:
        ddtheta_magdrift_plus = ddtheta
        ddtheta_magdrift_minus = ddtheta
    else:
        drift_scheme_map = {
            1: (80, 90),
            2: (100, 110),
            3: (120, 130),
            -1: (90, 80),
            -2: (110, 100),
            -3: (130, 120),
        }
        schemes = drift_scheme_map.get(magnetic_drift_derivative_scheme)
        if schemes is None:
            raise ValueError(f"Invalid magneticDriftDerivativeScheme={magnetic_drift_derivative_scheme}")
        scheme_plus, scheme_minus = schemes
        _, _, ddtheta_magdrift_plus, _ = uniform_diff_matrices(
            n=ntheta, x_min=0.0, x_max=2 * math.pi, scheme=scheme_plus
        )
        _, _, ddtheta_magdrift_minus, _ = uniform_diff_matrices(
            n=ntheta, x_min=0.0, x_max=2 * math.pi, scheme=scheme_minus
        )

    # zeta grid
    zeta_max = 2 * math.pi / n_periods
    zeta_scheme_map = {0: 20, 1: 0, 2: 10}
    zeta_scheme = zeta_scheme_map.get(zeta_derivative_scheme)
    if zeta_scheme is None:
        raise ValueError(f"Invalid zetaDerivativeScheme={zeta_derivative_scheme}")
    if nzeta == 1:
        zeta = jnp.asarray(np.array([0.0], dtype=np.float64))
        zeta_weights = jnp.asarray(np.array([2 * math.pi * n_periods], dtype=np.float64))
    else:
        zeta, zeta_weights, ddzeta, _ = uniform_diff_matrices(
            n=nzeta, x_min=0.0, x_max=zeta_max, scheme=zeta_scheme
        )
        zeta_weights = zeta_weights * n_periods
    # If axisymmetric, ddzeta is unused but keep it defined.
    if nzeta == 1:
        ddzeta = jnp.zeros((1, 1), dtype=jnp.float64)
        ddzeta_magdrift_plus = ddzeta
        ddzeta_magdrift_minus = ddzeta
    else:
        # Upwinded zeta matrices for magnetic drifts (ddzeta_magneticDrift_plus/minus in v3).
        if magnetic_drift_derivative_scheme == 0:
            ddzeta_magdrift_plus = ddzeta
            ddzeta_magdrift_minus = ddzeta
        else:
            drift_scheme_map = {
                1: (80, 90),
                2: (100, 110),
                3: (120, 130),
                -1: (90, 80),
                -2: (110, 100),
                -3: (130, 120),
            }
            schemes = drift_scheme_map.get(magnetic_drift_derivative_scheme)
            if schemes is None:
                raise ValueError(f"Invalid magneticDriftDerivativeScheme={magnetic_drift_derivative_scheme}")
            scheme_plus, scheme_minus = schemes
            _, _, ddzeta_magdrift_plus, _ = uniform_diff_matrices(
                n=nzeta, x_min=0.0, x_max=zeta_max, scheme=scheme_plus
            )
            _, _, ddzeta_magdrift_minus, _ = uniform_diff_matrices(
                n=nzeta, x_min=0.0, x_max=zeta_max, scheme=scheme_minus
            )

    # x grid
    #
    # v3 special-case: for monoenergetic transport coefficients (RHSMode=3), `createGrids.F90`
    # overwrites the x-grid and weights to a single point at x=1, regardless of xGridScheme:
    #   x = 1; xWeights = exp(1); pointAtX0 = .false.
    if rhs_mode == 3:
        x = jnp.asarray(np.full((nx,), 1.0, dtype=np.float64))
        x_weights = jnp.asarray(np.full((nx,), math.exp(1.0), dtype=np.float64))
        ddx = jnp.zeros((nx, nx), dtype=jnp.float64)
        d2dx2 = jnp.zeros((nx, nx), dtype=jnp.float64)
    else:
        if x_grid_scheme in {1, 5}:
            include_x0 = False
        elif x_grid_scheme in {2, 6}:
            include_x0 = True
        else:
            raise NotImplementedError(
                f"Only xGridScheme in {{1,2,5,6}} is implemented (got {x_grid_scheme})."
            )

        xg: XGrid = make_x_grid(n=nx, k=x_grid_k, include_point_at_x0=include_x0)
        x = jnp.asarray(xg.x)
        x_weights = jnp.asarray(xg.dx_weights(x_grid_k))

        # x differentiation matrix (used by the Er xDot term and by collisions).
        ddx_np, d2dx2_np = make_x_polynomial_diff_matrices(np.asarray(xg.x, dtype=np.float64), k=x_grid_k)
        ddx = jnp.asarray(ddx_np)
        d2dx2 = jnp.asarray(d2dx2_np)

        if xdot_derivative_scheme != 0:
            raise NotImplementedError(
                "Only xDotDerivativeScheme=0 is implemented (ddx_xDot_plus/minus = ddx)."
            )

    # Nxi_for_x logic (see createGrids.F90).
    x_np = np.asarray(x, dtype=float)
    nxi_for_x = np.zeros((nx,), dtype=int)
    if nxi_for_x_option == 0:
        nxi_for_x[:] = nxi
    elif nxi_for_x_option == 1:
        for j in range(nx):
            temp = nxi * (0.1 + 0.9 * x_np[j] / 2.0)
            nxi_for_x[j] = max(4, nl, min(int(temp), nxi))
    elif nxi_for_x_option == 2:
        for j in range(nx):
            temp = nxi * (0.1 + 0.9 * ((x_np[j] / 2.0) ** 2))
            nxi_for_x[j] = max(4, nl, min(int(temp), nxi))
    elif nxi_for_x_option == 3:
        for j in range(nx):
            temp = nxi * (0.1 + 0.9 * x_np[j] / 2.0)
            nxi_for_x[j] = max(3, nl, int(temp))
    else:
        raise ValueError(f"Invalid Nxi_for_x_option={nxi_for_x_option}")

    grids = V3Grids(
        theta=theta,
        zeta=zeta,
        x=x,
        theta_weights=theta_weights,
        zeta_weights=zeta_weights,
        x_weights=x_weights,
        ddtheta=ddtheta,
        ddzeta=ddzeta,
        ddx=ddx,
        d2dx2=d2dx2,
        ddtheta_magdrift_plus=ddtheta_magdrift_plus,
        ddtheta_magdrift_minus=ddtheta_magdrift_minus,
        ddzeta_magdrift_plus=ddzeta_magdrift_plus,
        ddzeta_magdrift_minus=ddzeta_magdrift_minus,
        n_xi=nxi,
        n_l=nl,
        n_xi_for_x=jnp.asarray(nxi_for_x, dtype=jnp.int32),
    )
    if use_cache:
        _GRIDS_CACHE[cache_key] = grids
    return grids


def geometry_from_namelist(*, nml: Namelist, grids: V3Grids) -> BoozerGeometry:
    geom = nml.group("geometryParameters")
    geometry_scheme = _get_int(geom, "geometryScheme", -1)
    cache_env = os.environ.get("SFINCS_JAX_GEOMETRY_CACHE", "").strip().lower()
    use_cache = cache_env not in {"0", "false", "no", "off"}
    if use_cache:
        geom_key = _group_key(
            geom,
            [
                "geometryScheme",
                "epsilon_t",
                "epsilon_h",
                "epsilon_antisymm",
                "iota",
                "GHat",
                "IHat",
                "B0OverBBar",
                "helicity_l",
                "helicity_n",
                "helicity_antisymm_l",
                "helicity_antisymm_n",
                "RN_WISH",
                "VMECRadialOption",
                "VMEC_NYQUIST_OPTION",
                "MIN_BMN_TO_LOAD",
                "RIPPLESCALE",
                "HELICITY_L",
                "HELICITY_N",
            ],
        )
        eq_key = _equilibrium_file_key(nml=nml, geometry_scheme=geometry_scheme, geom_group=geom)
        grid_key = (int(grids.theta.size), int(grids.zeta.size))
        cache_key = ("geometry", geometry_scheme, geom_key, eq_key, grid_key)
        cached = _GEOMETRY_CACHE.get(cache_key)
        if cached is not None:
            return cached
        disk_cached = _load_geometry_cache(cache_key)
        if disk_cached is not None:
            _GEOMETRY_CACHE[cache_key] = disk_cached
            return disk_cached
    if geometry_scheme == 1:
        from .geometry import boozer_geometry_scheme1

        geom_out = boozer_geometry_scheme1(
            theta=grids.theta,
            zeta=grids.zeta,
            epsilon_t=_get_float(geom, "epsilon_t", -0.07053),
            epsilon_h=_get_float(geom, "epsilon_h", 0.05067),
            epsilon_antisymm=_get_float(geom, "epsilon_antisymm", 0.0),
            iota=_get_float(geom, "iota", 0.4542),
            g_hat=_get_float(geom, "GHat", 3.7481),
            i_hat=_get_float(geom, "IHat", 0.0),
            b0_over_bbar=_get_float(geom, "B0OverBBar", 1.0),
            helicity_l=_get_int(geom, "helicity_l", 2),
            helicity_n=_get_int(geom, "helicity_n", 10),
            helicity_antisymm_l=_get_int(geom, "helicity_antisymm_l", 1),
            helicity_antisymm_n=_get_int(geom, "helicity_antisymm_n", 0),
        )
        if use_cache:
            _GEOMETRY_CACHE[cache_key] = geom_out
            _save_geometry_cache(cache_key, geom_out)
        return geom_out
    if geometry_scheme == 2:
        from .geometry import boozer_geometry_scheme2

        geom_out = boozer_geometry_scheme2(theta=grids.theta, zeta=grids.zeta)
        if use_cache:
            _GEOMETRY_CACHE[cache_key] = geom_out
            _save_geometry_cache(cache_key, geom_out)
        return geom_out
    if geometry_scheme == 4:
        geom_out = boozer_geometry_scheme4(theta=grids.theta, zeta=grids.zeta)
        if use_cache:
            _GEOMETRY_CACHE[cache_key] = geom_out
            _save_geometry_cache(cache_key, geom_out)
        return geom_out
    if geometry_scheme in {11, 12}:
        equilibrium_file = geom.get("EQUILIBRIUMFILE", None)
        if equilibrium_file is None:
            raise ValueError("geometryScheme=11/12 requires equilibriumFile in geometryParameters.")
        base_dir = nml.source_path.parent if nml.source_path is not None else None
        repo_root = Path(__file__).resolve().parents[1]
        extra = (repo_root / "tests" / "ref", repo_root / "sfincs_jax" / "data" / "equilibria")
        p = resolve_existing_path(str(equilibrium_file), base_dir=base_dir, extra_search_dirs=extra).path

        r_n_wish = float(geom.get("RN_WISH", 0.5))
        vmecradial_option = int(_get_int(geom, "VMECRadialOption", 1))
        geom_out = boozer_geometry_from_bc_file(
            path=str(p),
            theta=grids.theta,
            zeta=grids.zeta,
            r_n_wish=r_n_wish,
            vmecradial_option=vmecradial_option,
            geometry_scheme=int(geometry_scheme),
        )
        if use_cache:
            _GEOMETRY_CACHE[cache_key] = geom_out
            _save_geometry_cache(cache_key, geom_out)
        return geom_out
    if geometry_scheme == 5:
        equilibrium_file = geom.get("EQUILIBRIUMFILE", None)
        if equilibrium_file is None:
            raise ValueError("geometryScheme=5 requires equilibriumFile in geometryParameters.")
        base_dir = nml.source_path.parent if nml.source_path is not None else None
        repo_root = Path(__file__).resolve().parents[1]
        extra = (repo_root / "tests" / "ref", repo_root / "sfincs_jax" / "data" / "equilibria")
        p = _resolve_vmec_equilibrium_file(str(equilibrium_file), base_dir=base_dir, extra_search_dirs=extra)

        r_n_wish = float(geom.get("RN_WISH", 0.5))
        psi_n_wish = float(r_n_wish) * float(r_n_wish)
        vmecradial_option = int(_get_int(geom, "VMECRadialOption", 1))
        vmec_nyq_opt = int(geom.get("VMEC_NYQUIST_OPTION", 1))
        min_bmn_to_load = float(geom.get("MIN_BMN_TO_LOAD", 0.0))
        ripple_scale = float(geom.get("RIPPLESCALE", 1.0))
        helicity_n = int(geom.get("HELICITY_N", 0))
        helicity_l = int(geom.get("HELICITY_L", 0))

        geom_out = vmec_geometry_from_wout_file(
            path=str(p),
            theta=grids.theta,
            zeta=grids.zeta,
            psi_n_wish=psi_n_wish,
            vmec_radial_option=vmecradial_option,
            vmec_nyquist_option=vmec_nyq_opt,
            min_bmn_to_load=min_bmn_to_load,
            ripple_scale=ripple_scale,
            helicity_n=helicity_n,
            helicity_l=helicity_l,
        )
        if use_cache:
            _GEOMETRY_CACHE[cache_key] = geom_out
            _save_geometry_cache(cache_key, geom_out)
        return geom_out
    raise NotImplementedError("Only geometryScheme in {1,2,4,5,11,12} is implemented so far.")
