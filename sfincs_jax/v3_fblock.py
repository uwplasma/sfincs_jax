from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
import numpy as np
from jax import tree_util as jtu

from .collisionless import CollisionlessV3Operator, apply_collisionless_v3
from .collisionless_exb import ExBThetaV3Operator, ExBZetaV3Operator, apply_exb_theta_v3, apply_exb_zeta_v3
from .collisionless_er import ErXDotV3Operator, ErXiDotV3Operator, apply_er_xdot_v3, apply_er_xidot_v3
from .collisions import (
    FokkerPlanckV3Operator,
    FokkerPlanckV3Phi1Operator,
    PitchAngleScatteringV3Operator,
    apply_fokker_planck_v3,
    apply_fokker_planck_v3_phi1,
    apply_pitch_angle_scattering_v3,
    make_fokker_planck_v3_operator,
    make_fokker_planck_v3_phi1_operator,
    make_pitch_angle_scattering_v3_operator,
)
from .diagnostics import fsab_hat2 as fsab_hat2_jax
from .geometry import BoozerGeometry
from .magnetic_drifts import (
    MagneticDriftThetaV3Operator,
    MagneticDriftXiDotV3Operator,
    MagneticDriftZetaV3Operator,
    apply_magnetic_drift_theta_v3,
    apply_magnetic_drift_xidot_v3,
    apply_magnetic_drift_zeta_v3,
)
from .namelist import Namelist
from .solver import GMRESSolveResult, gmres_solve
from .boozer_bc import read_boozer_bc_header
from .paths import resolve_existing_path
from .v3 import V3Grids, geometry_from_namelist, grids_from_namelist
from .vmec_wout import psi_a_hat_from_wout, read_vmec_wout, vmec_interpolation


def _as_1d_float(group: dict, key: str) -> np.ndarray:
    """Read a namelist value as a 1D float64 numpy array."""
    v = group[key.upper()]
    return np.atleast_1d(np.asarray(v, dtype=np.float64))


def _get_float(group: dict, key: str, default: float) -> float:
    v = group.get(key.upper(), default)
    if isinstance(v, list):
        v = v[0] if v else default
    return float(v)


def _get_int(group: dict, key: str, default: int) -> int:
    v = group.get(key.upper(), default)
    if isinstance(v, list):
        v = v[0] if v else default
    return int(v)


def _as_1d_float_default(group: dict, key: str, *, default: float) -> np.ndarray:
    """Read a namelist value as a 1D float64 numpy array, with a scalar default."""
    k = key.upper()
    if k not in group:
        return np.atleast_1d(np.asarray(default, dtype=np.float64))
    return np.atleast_1d(np.asarray(group[k], dtype=np.float64))


# Defaults from v3 `globalVariables.F90`, used when upstream inputs omit these values.
_V3_DEFAULT_DELTA = 4.5694e-3
_V3_DEFAULT_NU_N = 8.330e-3


def _dphi_hat_dpsi_hat_from_er(*, nml: Namelist, er: float) -> float:
    """Compute dPhiHat/dpsiHat from Er using v3 defaults (subset).

    Notes
    -----
    This conversion depends on the input radial coordinate conventions. For now we support
    the common defaults used throughout the v3 examples:

    - inputRadialCoordinate = 3 (rN)
    - inputRadialCoordinateForGradients = 4 (rHat)
    - Er interpreted as :math:`E_r = - d\\hat\\Phi / d\\hat r`.

    For `geometryScheme=4`, we use the v3 hard-coded W7-X constants. For `.bc` geometries
    (geometryScheme 11/12), we read `psiAHat` and `aHat` from the `.bc` header.
    """
    if float(er) == 0.0:
        return 0.0

    geom_params = nml.group("geometryParameters")
    geometry_scheme = _get_int(geom_params, "geometryScheme", -1)

    input_radial = _get_int(geom_params, "inputRadialCoordinate", 3)
    input_radial_grad = _get_int(geom_params, "inputRadialCoordinateForGradients", 4)
    if input_radial != 3 or input_radial_grad != 4:
        raise NotImplementedError(
            "sfincs_jax currently only converts Er->dPhiHatdpsiHat for "
            "inputRadialCoordinate=3 (rN) and inputRadialCoordinateForGradients=4 (rHat)."
        )

    if geometry_scheme == 4:
        psi_a_hat = -0.384935
        a_hat = 0.5109
        r_n = 0.5  # v3 forces rN=0.5 for geometryScheme=4.
    elif geometry_scheme in {11, 12}:
        eq = geom_params.get("EQUILIBRIUMFILE", None)
        if eq is None:
            raise ValueError("geometryScheme=11/12 requires equilibriumFile in geometryParameters.")
        base_dir = nml.source_path.parent if nml.source_path is not None else None
        repo_root = Path(__file__).resolve().parents[1]
        extra = (repo_root / "tests" / "ref", repo_root / "sfincs_jax" / "data" / "equilibria")
        p = resolve_existing_path(str(eq), base_dir=base_dir, extra_search_dirs=extra).path

        header = read_boozer_bc_header(path=str(p), geometry_scheme=int(geometry_scheme))
        psi_a_hat = float(header.psi_a_hat)
        a_hat = float(header.a_hat)
        r_n = float(geom_params.get("RN_WISH", 0.5))
    elif geometry_scheme == 5:
        eq = geom_params.get("EQUILIBRIUMFILE", None)
        if eq is None:
            raise ValueError("geometryScheme=5 requires equilibriumFile in geometryParameters.")
        base_dir = nml.source_path.parent if nml.source_path is not None else None
        repo_root = Path(__file__).resolve().parents[1]
        extra = (repo_root / "tests" / "ref", repo_root / "sfincs_jax" / "data" / "equilibria")
        p = resolve_existing_path(str(eq), base_dir=base_dir, extra_search_dirs=extra).path

        w = read_vmec_wout(p)
        psi_a_hat = float(psi_a_hat_from_wout(w))
        a_hat = float(w.aminor_p)

        r_n_wish = float(geom_params.get("RN_WISH", 0.5))
        psi_n_wish = float(r_n_wish) * float(r_n_wish)
        vmecradial_option = int(geom_params.get("VMECRADIALOPTION", 0))
        interp = vmec_interpolation(w=w, psi_n_wish=psi_n_wish, vmec_radial_option=vmecradial_option)
        r_n = float(interp.psi_n) ** 0.5
    else:
        raise NotImplementedError(f"Er->dPhiHatdpsiHat conversion not implemented for geometryScheme={geometry_scheme}.")

    # With rHat = aHat * rN and psiHat = psiAHat * (rN^2):
    # dpsiHat/drHat = 2*psiAHat*rN/aHat -> drHat/dpsiHat = aHat/(2*psiAHat*rN).
    drhat_dpsihat = float(a_hat) / (2.0 * float(psi_a_hat) * float(r_n))
    return float((-float(er)) * drhat_dpsihat)


def _fsab_hat2(*, grids: V3Grids, geom: BoozerGeometry) -> float:
    """Compute FSABHat2 as in v3 `geometry.F90:computeBIntegrals`."""
    return float(np.asarray(fsab_hat2_jax(grids=grids, geom=geom), dtype=np.float64))


def collisionless_operator_from_namelist(
    *,
    nml: Namelist,
    grids: V3Grids,
    geom: BoozerGeometry,
) -> CollisionlessV3Operator:
    species = nml.group("speciesParameters")
    # Monoenergetic runs (RHSMode=3) can omit speciesParameters entirely in upstream examples.
    t_hats = _as_1d_float_default(species, "THats", default=1.0)
    m_hats = _as_1d_float_default(species, "mHats", default=1.0)
    return CollisionlessV3Operator(
        x=grids.x,
        ddtheta=grids.ddtheta,
        ddzeta=grids.ddzeta,
        b_hat=geom.b_hat,
        b_hat_sup_theta=geom.b_hat_sup_theta,
        b_hat_sup_zeta=geom.b_hat_sup_zeta,
        db_hat_dtheta=geom.db_hat_dtheta,
        db_hat_dzeta=geom.db_hat_dzeta,
        t_hats=jnp.asarray(t_hats),
        m_hats=jnp.asarray(m_hats),
        n_xi_for_x=grids.n_xi_for_x,
    )


def pas_collision_operator_from_namelist(
    *, nml: Namelist, grids: V3Grids, nu_n_override: float | None = None
) -> PitchAngleScatteringV3Operator:
    species = nml.group("speciesParameters")
    phys = nml.group("physicsParameters")

    collision_operator = _get_int(phys, "collisionOperator", 0)
    if collision_operator != 1:
        raise NotImplementedError(
            "sfincs_jax currently only builds a collision operator for collisionOperator=1 "
            "(pitch-angle scattering)."
        )

    z_s = _as_1d_float_default(species, "Zs", default=1.0)
    m_hats = _as_1d_float_default(species, "mHats", default=1.0)
    n_hats = _as_1d_float_default(species, "nHats", default=1.0)
    t_hats = _as_1d_float_default(species, "THats", default=1.0)
    nu_n = float(nu_n_override) if nu_n_override is not None else _get_float(phys, "nu_n", _V3_DEFAULT_NU_N)
    krook = _get_float(phys, "Krook", 0.0)

    return make_pitch_angle_scattering_v3_operator(
        x=grids.x,
        z_s=jnp.asarray(z_s),
        m_hats=jnp.asarray(m_hats),
        n_hats=jnp.asarray(n_hats),
        t_hats=jnp.asarray(t_hats),
        nu_n=nu_n,
        krook=krook,
        n_xi_for_x=grids.n_xi_for_x,
    )


def fokker_planck_collision_operator_from_namelist(*, nml: Namelist, grids: V3Grids) -> FokkerPlanckV3Operator:
    species = nml.group("speciesParameters")
    phys = nml.group("physicsParameters")
    other = nml.group("otherNumericalParameters")

    collision_operator = _get_int(phys, "collisionOperator", 0)
    if collision_operator != 0:
        raise NotImplementedError("collisionOperator must be 0 for the Fokker-Planck collision operator builder.")

    if bool(phys.get("INCLUDEPHI1", False)):
        raise NotImplementedError(
            "sfincs_jax currently implements collisionOperator=0 only for includePhi1 = .false. "
            "unless includePhi1InCollisionOperator is enabled."
        )

    x_grid_scheme = _get_int(other, "xGridScheme", 5)
    if x_grid_scheme not in {5, 6}:
        raise NotImplementedError(
            f"sfincs_jax currently only supports collisionOperator=0 for xGridScheme in {{5,6}} (got {x_grid_scheme})."
        )

    x_grid_k = _get_float(other, "xGrid_k", 0.0)
    z_s = _as_1d_float(species, "Zs")
    m_hats = _as_1d_float(species, "mHats")
    n_hats = _as_1d_float(species, "nHats")
    t_hats = _as_1d_float(species, "THats")
    nu_n = _get_float(phys, "nu_n", _V3_DEFAULT_NU_N)
    krook = _get_float(phys, "Krook", 0.0)

    return make_fokker_planck_v3_operator(
        x=np.asarray(grids.x, dtype=np.float64),
        x_weights=np.asarray(grids.x_weights, dtype=np.float64),
        ddx=np.asarray(grids.ddx, dtype=np.float64),
        d2dx2=np.asarray(grids.d2dx2, dtype=np.float64),
        x_grid_k=float(x_grid_k),
        z_s=z_s,
        m_hats=m_hats,
        n_hats=n_hats,
        t_hats=t_hats,
        nu_n=float(nu_n),
        krook=float(krook),
        n_xi=int(grids.n_xi),
        nl=int(grids.n_l),
        n_xi_for_x=np.asarray(grids.n_xi_for_x, dtype=np.int32),
    )


def fokker_planck_collision_operator_with_phi1_from_namelist(
    *, nml: Namelist, grids: V3Grids, alpha: float
) -> FokkerPlanckV3Phi1Operator:
    """Build the poloidally varying v3 FP collision operator (includePhi1InCollisionOperator=true)."""
    species = nml.group("speciesParameters")
    phys = nml.group("physicsParameters")
    other = nml.group("otherNumericalParameters")

    collision_operator = _get_int(phys, "collisionOperator", 0)
    if collision_operator != 0:
        raise NotImplementedError("collisionOperator must be 0 for the Fokker-Planck collision operator builder.")

    if not bool(phys.get("INCLUDEPHI1", False)):
        raise ValueError("includePhi1 must be true for includePhi1InCollisionOperator=true.")
    if not bool(phys.get("INCLUDEPHI1INKINETICEQUATION", False)):
        raise ValueError("includePhi1InKineticEquation must be true for includePhi1InCollisionOperator=true.")

    x_grid_scheme = _get_int(other, "xGridScheme", 5)
    if x_grid_scheme not in {5, 6}:
        raise NotImplementedError(
            f"sfincs_jax currently only supports collisionOperator=0 for xGridScheme in {{5,6}} (got {x_grid_scheme})."
        )

    x_grid_k = _get_float(other, "xGrid_k", 0.0)
    z_s = _as_1d_float(species, "Zs")
    m_hats = _as_1d_float(species, "mHats")
    n_hats = _as_1d_float(species, "nHats")
    t_hats = _as_1d_float(species, "THats")
    nu_n = _get_float(phys, "nu_n", _V3_DEFAULT_NU_N)
    krook = _get_float(phys, "Krook", 0.0)

    return make_fokker_planck_v3_phi1_operator(
        x=np.asarray(grids.x, dtype=np.float64),
        x_weights=np.asarray(grids.x_weights, dtype=np.float64),
        ddx=np.asarray(grids.ddx, dtype=np.float64),
        d2dx2=np.asarray(grids.d2dx2, dtype=np.float64),
        x_grid_k=float(x_grid_k),
        z_s=z_s,
        m_hats=m_hats,
        n_hats=n_hats,
        t_hats=t_hats,
        nu_n=float(nu_n),
        krook=float(krook),
        n_xi=int(grids.n_xi),
        nl=int(grids.n_l),
        alpha=float(alpha),
        n_xi_for_x=np.asarray(grids.n_xi_for_x, dtype=np.int32),
    )


@jtu.register_pytree_node_class
@dataclass(frozen=True)
class V3FBlockOperator:
    """Matrix-free operator for the v3 distribution-function block (BLOCK_F).

    This is intentionally incomplete. Today it includes:
    - collisionless streaming + mirror (±1 couplings in L)
    - ExB drift terms (d/dtheta and d/dzeta)
    - magnetic drifts (d/dtheta, d/dzeta, and the non-standard d/dxi term), when enabled
    - pitch-angle scattering collisions (collisionOperator=1, diagonal in L)
    - full linearized Fokker-Planck collisions (collisionOperator=0, no Phi1), when enabled
    - collisionless Er terms (xiDot and xDot), when enabled in the namelist

    As more v3 terms are ported, they will be composed here.
    """

    collisionless: CollisionlessV3Operator
    exb_theta: ExBThetaV3Operator
    exb_zeta: ExBZetaV3Operator
    magdrift_theta: MagneticDriftThetaV3Operator | None
    magdrift_zeta: MagneticDriftZetaV3Operator | None
    magdrift_xidot: MagneticDriftXiDotV3Operator | None
    pas: PitchAngleScatteringV3Operator | None
    fp: FokkerPlanckV3Operator | None
    fp_phi1: FokkerPlanckV3Phi1Operator | None
    er_xidot: ErXiDotV3Operator | None
    er_xdot: ErXDotV3Operator | None
    identity_shift: jnp.ndarray  # scalar, helps make toy solves well-conditioned

    n_species: int
    n_x: int
    n_xi: int
    n_theta: int
    n_zeta: int

    @property
    def f_shape(self) -> tuple[int, int, int, int, int]:
        return (self.n_species, self.n_x, self.n_xi, self.n_theta, self.n_zeta)

    @property
    def flat_size(self) -> int:
        s, x, l, t, z = self.f_shape
        return int(s * x * l * t * z)

    def tree_flatten(self):
        children = (
            self.collisionless,
            self.exb_theta,
            self.exb_zeta,
            self.magdrift_theta,
            self.magdrift_zeta,
            self.magdrift_xidot,
            self.pas,
            self.fp,
            self.fp_phi1,
            self.er_xidot,
            self.er_xdot,
            self.identity_shift,
        )
        aux = (self.n_species, self.n_x, self.n_xi, self.n_theta, self.n_zeta)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        collisionless, exb_theta, exb_zeta, magdrift_theta, magdrift_zeta, magdrift_xidot, pas, fp, fp_phi1, er_xidot, er_xdot, identity_shift = children
        n_species, n_x, n_xi, n_theta, n_zeta = aux
        return cls(
            collisionless=collisionless,
            exb_theta=exb_theta,
            exb_zeta=exb_zeta,
            magdrift_theta=magdrift_theta,
            magdrift_zeta=magdrift_zeta,
            magdrift_xidot=magdrift_xidot,
            pas=pas,
            fp=fp,
            fp_phi1=fp_phi1,
            er_xidot=er_xidot,
            er_xdot=er_xdot,
            identity_shift=identity_shift,
            n_species=n_species,
            n_x=n_x,
            n_xi=n_xi,
            n_theta=n_theta,
            n_zeta=n_zeta,
        )


def fblock_operator_from_namelist(*, nml: Namelist, identity_shift: float = 0.0) -> V3FBlockOperator:
    grids = grids_from_namelist(nml)
    geom = geometry_from_namelist(nml=nml, grids=grids)
    colless = collisionless_operator_from_namelist(nml=nml, grids=grids, geom=geom)
    general = nml.group("general")
    phys = nml.group("physicsParameters")
    rhs_mode = _get_int(general, "RHSMode", 1)
    collision_operator = _get_int(phys, "collisionOperator", 0)
    include_phi1 = bool(phys.get("INCLUDEPHI1", False))
    include_phi1_in_collisions = bool(phys.get("INCLUDEPHI1INCOLLISIONOPERATOR", False))
    if collision_operator == 1:
        nu_n_override = None
        if rhs_mode == 3:
            nu_prime = _get_float(phys, "nuPrime", 1.0)
            denom = float(geom.g_hat) + float(geom.iota) * float(geom.i_hat)
            nu_n_override = float(nu_prime) * float(geom.b0_over_bbar) / float(denom)
        pas = pas_collision_operator_from_namelist(nml=nml, grids=grids, nu_n_override=nu_n_override)
        fp = None
        fp_phi1 = None
    elif collision_operator == 0:
        pas = None
        if include_phi1 and include_phi1_in_collisions:
            fp = None
            fp_phi1 = fokker_planck_collision_operator_with_phi1_from_namelist(
                nml=nml, grids=grids, alpha=float(phys.get("ALPHA", 1.0))
            )
        else:
            fp = fokker_planck_collision_operator_from_namelist(nml=nml, grids=grids)
            fp_phi1 = None
    else:
        raise NotImplementedError(f"collisionOperator={collision_operator} is not supported.")
    include_xdot = bool(phys.get("INCLUDEXDOTTERM", False))
    include_er_xidot = bool(phys.get("INCLUDEELECTRICFIELDTERMINXIDOT", False))
    use_dkes_exb = bool(phys.get("USEDKESEXBDRIFT", False))
    magnetic_drift_scheme = _get_int(phys, "magneticDriftScheme", 0)
    er = float(phys.get("ER", 0.0))
    alpha = float(phys.get("ALPHA", 1.0))
    delta = float(phys.get("DELTA", _V3_DEFAULT_DELTA))

    if rhs_mode == 3:
        e_star = _get_float(phys, "EStar", 0.0)
        dphi = (
            2.0
            / (float(alpha) * float(delta))
            * float(e_star)
            * float(geom.iota)
            * float(geom.b0_over_bbar)
            / float(geom.g_hat)
        )
    else:
        dphi = _dphi_hat_dpsi_hat_from_er(nml=nml, er=er)
    fsab_hat2 = _fsab_hat2(grids=grids, geom=geom)

    exb_theta = ExBThetaV3Operator(
        alpha=jnp.asarray(alpha, dtype=jnp.float64),
        delta=jnp.asarray(delta, dtype=jnp.float64),
        dphi_hat_dpsi_hat=jnp.asarray(dphi, dtype=jnp.float64),
        ddtheta=grids.ddtheta,
        d_hat=geom.d_hat,
        b_hat=geom.b_hat,
        b_hat_sub_zeta=geom.b_hat_sub_zeta,
        use_dkes_exb_drift=bool(use_dkes_exb),
        fsab_hat2=jnp.asarray(fsab_hat2, dtype=jnp.float64),
        n_xi_for_x=grids.n_xi_for_x,
    )
    exb_zeta = ExBZetaV3Operator(
        alpha=jnp.asarray(alpha, dtype=jnp.float64),
        delta=jnp.asarray(delta, dtype=jnp.float64),
        dphi_hat_dpsi_hat=jnp.asarray(dphi, dtype=jnp.float64),
        ddzeta=grids.ddzeta,
        d_hat=geom.d_hat,
        b_hat=geom.b_hat,
        b_hat_sub_theta=geom.b_hat_sub_theta,
        use_dkes_exb_drift=bool(use_dkes_exb),
        fsab_hat2=jnp.asarray(fsab_hat2, dtype=jnp.float64),
        n_xi_for_x=grids.n_xi_for_x,
    )

    magdrift_theta = None
    magdrift_zeta = None
    magdrift_xidot = None
    if magnetic_drift_scheme != 0:
        if magnetic_drift_scheme != 1:
            raise NotImplementedError("sfincs_jax currently only builds magnetic drifts for magneticDriftScheme=1.")

        species = nml.group("speciesParameters")
        t_hat = float(_as_1d_float_default(species, "THats", default=1.0)[0])
        z = float(_as_1d_float_default(species, "Zs", default=1.0)[0])

        magdrift_theta = MagneticDriftThetaV3Operator(
            delta=jnp.asarray(delta, dtype=jnp.float64),
            t_hat=jnp.asarray(t_hat, dtype=jnp.float64),
            z=jnp.asarray(z, dtype=jnp.float64),
            x=grids.x,
            ddtheta_plus=grids.ddtheta_magdrift_plus,
            ddtheta_minus=grids.ddtheta_magdrift_minus,
            d_hat=geom.d_hat,
            b_hat=geom.b_hat,
            b_hat_sub_zeta=geom.b_hat_sub_zeta,
            b_hat_sub_psi=geom.b_hat_sub_psi,
            db_hat_dzeta=geom.db_hat_dzeta,
            db_hat_dpsi_hat=geom.db_hat_dpsi_hat,
            db_hat_sub_psi_dzeta=geom.db_hat_sub_psi_dzeta,
            db_hat_sub_zeta_dpsi_hat=geom.db_hat_sub_zeta_dpsi_hat,
            n_xi_for_x=grids.n_xi_for_x,
        )
        magdrift_zeta = MagneticDriftZetaV3Operator(
            delta=jnp.asarray(delta, dtype=jnp.float64),
            t_hat=jnp.asarray(t_hat, dtype=jnp.float64),
            z=jnp.asarray(z, dtype=jnp.float64),
            x=grids.x,
            ddzeta_plus=grids.ddzeta_magdrift_plus,
            ddzeta_minus=grids.ddzeta_magdrift_minus,
            d_hat=geom.d_hat,
            b_hat=geom.b_hat,
            b_hat_sub_theta=geom.b_hat_sub_theta,
            b_hat_sub_psi=geom.b_hat_sub_psi,
            db_hat_dtheta=geom.db_hat_dtheta,
            db_hat_dpsi_hat=geom.db_hat_dpsi_hat,
            db_hat_sub_theta_dpsi_hat=geom.db_hat_sub_theta_dpsi_hat,
            db_hat_sub_psi_dtheta=geom.db_hat_sub_psi_dtheta,
            n_xi_for_x=grids.n_xi_for_x,
        )
        magdrift_xidot = MagneticDriftXiDotV3Operator(
            delta=jnp.asarray(delta, dtype=jnp.float64),
            t_hat=jnp.asarray(t_hat, dtype=jnp.float64),
            z=jnp.asarray(z, dtype=jnp.float64),
            x=grids.x,
            d_hat=geom.d_hat,
            b_hat=geom.b_hat,
            db_hat_dtheta=geom.db_hat_dtheta,
            db_hat_dzeta=geom.db_hat_dzeta,
            db_hat_sub_psi_dzeta=geom.db_hat_sub_psi_dzeta,
            db_hat_sub_zeta_dpsi_hat=geom.db_hat_sub_zeta_dpsi_hat,
            db_hat_sub_theta_dpsi_hat=geom.db_hat_sub_theta_dpsi_hat,
            db_hat_sub_psi_dtheta=geom.db_hat_sub_psi_dtheta,
            n_xi_for_x=grids.n_xi_for_x,
        )

    er_xidot = None
    if include_er_xidot:
        er_xidot = ErXiDotV3Operator(
            alpha=jnp.asarray(alpha, dtype=jnp.float64),
            delta=jnp.asarray(delta, dtype=jnp.float64),
            dphi_hat_dpsi_hat=jnp.asarray(dphi, dtype=jnp.float64),
            d_hat=geom.d_hat,
            b_hat=geom.b_hat,
            b_hat_sub_theta=geom.b_hat_sub_theta,
            b_hat_sub_zeta=geom.b_hat_sub_zeta,
            db_hat_dtheta=geom.db_hat_dtheta,
            db_hat_dzeta=geom.db_hat_dzeta,
            force0_radial_current=jnp.asarray(True),
            n_xi_for_x=grids.n_xi_for_x,
        )

    er_xdot = None
    if include_xdot:
        er_xdot = ErXDotV3Operator(
            alpha=jnp.asarray(alpha, dtype=jnp.float64),
            delta=jnp.asarray(delta, dtype=jnp.float64),
            dphi_hat_dpsi_hat=jnp.asarray(dphi, dtype=jnp.float64),
            x=grids.x,
            ddx_plus=grids.ddx,
            ddx_minus=grids.ddx,
            d_hat=geom.d_hat,
            b_hat=geom.b_hat,
            b_hat_sub_theta=geom.b_hat_sub_theta,
            b_hat_sub_zeta=geom.b_hat_sub_zeta,
            db_hat_dtheta=geom.db_hat_dtheta,
            db_hat_dzeta=geom.db_hat_dzeta,
            force0_radial_current=jnp.asarray(True),
            n_xi_for_x=grids.n_xi_for_x,
        )

    return V3FBlockOperator(
        collisionless=colless,
        exb_theta=exb_theta,
        exb_zeta=exb_zeta,
        magdrift_theta=magdrift_theta,
        magdrift_zeta=magdrift_zeta,
        magdrift_xidot=magdrift_xidot,
        pas=pas,
        fp=fp,
        fp_phi1=fp_phi1,
        er_xidot=er_xidot,
        er_xdot=er_xdot,
        identity_shift=jnp.asarray(identity_shift, dtype=jnp.float64),
        n_species=int(colless.n_species),
        n_x=int(colless.n_x),
        n_xi=int(grids.n_xi),
        n_theta=int(colless.n_theta),
        n_zeta=int(colless.n_zeta),
    )


def apply_v3_fblock_operator(op: V3FBlockOperator, f: jnp.ndarray, *, phi1_hat_base: jnp.ndarray | None = None) -> jnp.ndarray:
    out = op.identity_shift * f
    out = out + apply_collisionless_v3(op.collisionless, f)
    out = out + apply_exb_theta_v3(op.exb_theta, f)
    out = out + apply_exb_zeta_v3(op.exb_zeta, f)
    if op.magdrift_theta is not None:
        out = out + apply_magnetic_drift_theta_v3(op.magdrift_theta, f)
    if op.magdrift_zeta is not None:
        out = out + apply_magnetic_drift_zeta_v3(op.magdrift_zeta, f)
    if op.magdrift_xidot is not None:
        out = out + apply_magnetic_drift_xidot_v3(op.magdrift_xidot, f)
    if op.er_xidot is not None:
        out = out + apply_er_xidot_v3(op.er_xidot, f)
    if op.er_xdot is not None:
        out = out + apply_er_xdot_v3(op.er_xdot, f)
    if op.pas is not None:
        out = out + apply_pitch_angle_scattering_v3(op.pas, f)
    if op.fp is not None:
        out = out + apply_fokker_planck_v3(op.fp, f)
    if op.fp_phi1 is not None:
        if phi1_hat_base is None:
            raise ValueError("phi1_hat_base is required when includePhi1InCollisionOperator is enabled.")
        out = out + apply_fokker_planck_v3_phi1(op.fp_phi1, f, phi1_hat=phi1_hat_base)
    return out


def matvec_v3_fblock_flat(op: V3FBlockOperator, x_flat: jnp.ndarray) -> jnp.ndarray:
    x_flat = jnp.asarray(x_flat)
    f = x_flat.reshape(op.f_shape)
    y = apply_v3_fblock_operator(op, f)
    return y.reshape((-1,))


def solve_v3_fblock_gmres(
    *,
    op: V3FBlockOperator,
    b_flat: jnp.ndarray,
    x0_flat: jnp.ndarray | None = None,
    tol: float = 1e-10,
    atol: float = 0.0,
    restart: int = 50,
    maxiter: int | None = None,
    solve_method: str = "batched",
) -> GMRESSolveResult:
    b_flat = jnp.asarray(b_flat)
    if b_flat.shape != (op.flat_size,):
        raise ValueError(f"b_flat must have shape {(op.flat_size,)}, got {b_flat.shape}")

    def mv(x):
        return matvec_v3_fblock_flat(op, x)

    return gmres_solve(
        matvec=mv,
        b=b_flat,
        x0=x0_flat,
        tol=tol,
        atol=atol,
        restart=restart,
        maxiter=maxiter,
        solve_method=solve_method,
    )


apply_v3_fblock_operator_jit = jax.jit(apply_v3_fblock_operator, static_argnums=())
matvec_v3_fblock_flat_jit = jax.jit(matvec_v3_fblock_flat, static_argnums=())
