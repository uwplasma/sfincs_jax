"""The v3 drift-kinetic operator as a single consolidated ``KineticOperator``.

This module is the Phase-3.2 consolidation target for the drift-kinetic-equation
(DKE) physics that previously lived across the retired ``operators`` package
(collisionless streaming/mirror, ExB, radial-electric-field xDot/xiDot terms,
collision wiring, constraint/source bordering, and the RHS drives).  It becomes
``dkx/dke.py`` at the v2 purge.

Fortran correspondence (paths relative to ``sfincs/fortran/version3``):

- ``populateMatrix.F90`` — every operator term applied by :meth:`KineticOperator.apply`:
  streaming (``ddtheta``/``ddzeta`` blocks coupling L±1), the mirror term (L±1),
  the ExB drift d/dtheta and d/dzeta terms, the non-standard d/dxi term
  associated with E_r (L, L±2), the collisionless d/dx term associated with E_r
  (L, L±2, dense in x), pitch-angle-scattering and Fokker–Planck collisions
  (diagonal in L), and the source/constraint rows and columns
  (``constraintScheme`` 1 and 2).
- ``evaluateResidual.F90`` — the inhomogeneous drives produced by
  :meth:`KineticOperator.rhs` (the ``dot(psi) dfM/dpsi`` gradient drive on
  L=0/L=2 and the inductive ``EParallelHat`` drive on L=1).
- ``indices.F90`` — the state-vector layout (see below).
- ``sfincs_main.F90`` lines 151-154 — the RHSMode=3 (monoenergetic)
  ``nuPrime``/``EStar`` overwrites of ``nu_n`` and ``dPhiHatdpsiHat``.

State-vector layout (matches v3 ``indices.F90`` with ``BLOCK_F`` first)::

    [ f(species, x, L, theta, zeta) row-major  |  constraint unknowns ]

- The distribution block flattens ``(n_species, n_x, n_xi, n_theta, n_zeta)``
  in C order.
- ``constraintScheme=1`` (default for Fokker-Planck): two source unknowns per
  species (particle, energy), constraint rows enforce zero density and pressure
  moments.
- ``constraintScheme=2`` (default for pitch-angle scattering): one L=0 source
  unknown per (species, x), constraint rows enforce ``<f1>=0`` at each x.
- ``includePhi1`` extends the state with ``Ntheta*Nzeta`` quasineutrality rows
  and one ``<Phi1>=0`` (lambda) row, laid out ``[f | Phi1 | lambda | sources]``
  (indices.F90).  The system becomes nonlinear (Phi1 couples back through
  quasineutrality and, with ``includePhi1InKineticEquation``, the kinetic
  equation); :meth:`KineticOperator.residual_phi1` is the nonlinear residual and
  the Newton outer solve lives in :mod:`dkx.phi1`.  Supported:
  ``quasineutralityOption`` 1 (full) and 2 (EUTERPE, incl. an adiabatic
  species), ``includePhi1InKineticEquation``, and
  ``includePhi1InCollisionOperator`` (the collisional densities become
  poloidally varying, ``n_pol = nHat*exp(-Z*alpha*Phi1Hat/THat)``).
  ``readExternalPhi1`` reads a FIXED external Phi1(theta,zeta) field and holds it
  constant (NO quasineutrality block, NO Phi1 unknown, NO lambda row): the DKE is
  LINEAR again and the external field enters the same Phi1 terms via
  ``external_phi1_hat`` (see :meth:`KineticOperator._apply_external_phi1`).

Coefficient provenance: everything is built from the committed consolidated
modules — :mod:`dkx.phase_space` (grids, differentiation matrices, Legendre
couplings), :mod:`dkx.magnetic_geometry` (flux-surface geometry for
geometrySchemes 1/2/3/4/5/11/12/13), :mod:`dkx.species` (charges, profiles,
psiHat-gradients), and :mod:`dkx.constants` (normalizations and radial
conversions).  Collision matrices are built by the stable
:mod:`dkx.collisions` (the canonical collision-operator owner).

Three consumers, one source of truth (plan §2.2):

1. :meth:`KineticOperator.apply` — matrix-free matvec for Krylov solvers,
   bit-compatible with ``operators.profile_system.apply_v3_full_system_operator``.
2. :meth:`KineticOperator.legendre_blocks` / :meth:`to_block_tridiagonal` — the
   analytic (probing-free) block-tridiagonal-in-L representation used by the
   structured direct solver and preconditioners.  The terms know their own
   L-coupling: streaming/mirror couple L±1 with the
   :func:`dkx.phase_space.legendre_coupling_lower` /
   :func:`~dkx.phase_space.legendre_coupling_upper` factors, ExB is diagonal
   in L, and pitch-angle scattering is diagonal in L (eigenvalues
   ``l(l+1)/2``).
3. :meth:`KineticOperator.rhs` — the v3 drives, including the internal
   ``whichRHS`` gradient/E_parallel overwrites used by RHSMode 2 and 3
   transport-matrix loops.

Every ``magneticDriftScheme`` 0-9 is canonical (the tangential magnetic drifts
couple L, L±2 so :meth:`to_block_tridiagonal` refuses drift decks and recycled
Krylov GCROT owns them).  With that, the last deferred physics family is consolidated;
the remaining refusals below are numerical-surface gaps, not physics:

- ``collisionOperator`` other than 0 (Fokker-Planck) and 1 (pitch-angle
  scattering);
- ``collisionOperator=0`` with the uniform/Chebyshev speed grids
  (``xGridScheme`` 3/4/7/8): the Fokker-Planck Rosenbluth-potential
  interpolation matrices for those grids (``interpolationMatrix.F90`` /
  ``ChebyshevInterpolationMatrix.F90``) are not ported.

All ``xGridScheme`` values 1-8 and every valid ``xDotDerivativeScheme`` (-2..11
except the Fortran-buggy -1, whose ``do i=i,Nx`` loop reads an undefined start)
are canonical; see :func:`dkx.phase_space.make_grids` and
:func:`dkx.phase_space.xdot_diff_matrices`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, NamedTuple

# The JAX backend is imported below; dkx/runtime.py explains why this is here.
from .runtime import configure as _configure_runtime

_configure_runtime()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax import tree_util as jtu  # noqa: E402

from dkx.constants import (  # noqa: E402
    DEFAULT_DELTA,
    DEFAULT_NU_N,
    RadialCoordinates,
    d_phi_hat_d_psi_hat_from_e_star,
    nu_n_from_nu_prime,
)
from dkx.magnetic_geometry import (  # noqa: E402
    FluxSurfaceGeometry,
    psi_a_hat_from_wout,
    read_boozer_bc,
    read_vmec_wout,
    selected_r_n_from_bc,
    vmec_radial_interpolation,
)
from dkx.phase_space import (  # noqa: E402
    Grids,
    legendre_coupling_lower,
    legendre_coupling_upper,
    make_grids,
)
from dkx.input_compat import (  # noqa: E402
    effective_equilibrium_file,
    effective_psi_a_hat,
    effective_psi_n_wish,
    infer_phi_input_radial_coordinate_for_gradients,
    infer_species_input_radial_coordinate_for_gradients,
)
from dkx.paths import resolve_existing_path  # noqa: E402

# Collision matrices: the stable, committed implementation.  This import moves
# to `dkx.collisions` when that consolidation lands (same public API).
from dkx.collisions import (  # noqa: E402
    FokkerPlanckV3Operator,
    FokkerPlanckV3Phi1Operator,
    ImprovedSugamaV3Operator,
    PitchAngleScatteringV3Operator,
    apply_fokker_planck_v3,
    apply_fokker_planck_v3_phi1,
    apply_improved_sugama_v3,
    apply_pitch_angle_scattering_v3,
    make_fokker_planck_v3_operator,
    make_fokker_planck_v3_phi1_operator,
    make_improved_sugama_v3_operator,
    make_pitch_angle_scattering_v3_operator,
    resolve_rosenbluth_method,
)
from dkx.species import SpeciesSet, species_set_from_namelist  # noqa: E402
from .paths import repository_root

__all__ = [
    "KineticOperator",
    "KineticOperatorBuild",
    "LegendreBlocks",
    "kinetic_operator_build_from_namelist",
    "kinetic_operator_from_namelist",
]

# =============================================================================
# Small namelist-access helpers (same conventions as readInput.F90 defaults).
# =============================================================================

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

def _get_bool(group: dict, key: str, default: bool = False) -> bool:
    v = group.get(key.upper(), default)
    if isinstance(v, list):
        v = v[0] if v else default
    return bool(v)

def _get_str_or_none(group: dict, key: str) -> str | None:
    """A namelist string value, or ``None`` when the key is absent or blank."""
    v = group.get(key.upper(), None)
    if isinstance(v, list):
        v = v[0] if v else None
    if v is None:
        return None
    text = str(v).strip()
    return text or None

def _mask_xi(n_xi_for_x: jnp.ndarray, n_xi: int) -> jnp.ndarray:
    """(X, L) float mask of retained Legendre modes (createGrids.F90 Nxi_for_x)."""
    ell = jnp.arange(int(n_xi), dtype=jnp.int32)[None, :]
    return (ell < n_xi_for_x.astype(jnp.int32)[:, None]).astype(jnp.float64)

def _ix_min(point_at_x0: bool) -> int:
    """First speed row carrying DKE equations (populateMatrix.F90 ixMin)."""
    return 1 if point_at_x0 else 0

class LegendreBlocks(NamedTuple):
    """Dense (theta*zeta) blocks of one Legendre row of the f-block operator.

    ``lower`` maps mode ``l-1`` to ``l``, ``diag`` maps ``l`` to ``l``, and
    ``upper`` maps ``l+1`` to ``l``.  All have shape
    ``(n_species, n_x, n_theta*n_zeta, n_theta*n_zeta)``; ``lower`` is zero for
    ``l=0`` and ``upper`` is zero for ``l=n_xi-1``.
    """

    lower: jnp.ndarray
    diag: jnp.ndarray
    upper: jnp.ndarray

# =============================================================================
# The operator
# =============================================================================

@jtu.register_pytree_node_class
@dataclass(frozen=True)
class KineticOperator:
    """Matrix-free v3 drift-kinetic operator with structured-block extractors.

    One instance holds the per-term coefficient arrays for a single flux
    surface and applies the linear operator of the v3 system
    ``populateMatrix.F90`` (whichMatrix=3 content for the supported subset:
    no Phi1, no tangential magnetic drifts).

    The supported term inventory (see module docstring for the Fortran map):

    ====================  =========  ==========================================
    term                  L-coupling  notes
    ====================  =========  ==========================================
    streaming             L±1        ``x_j (l±-coupling) v_|| b·∇``; diagonal in x
    mirror                L±1        ``-x_j (b·∇B)/(2B)``; diagonal in x
    ExB (d/dtheta,dzeta)  diag       present iff the kinetic dPhiHat/dpsiHat ≠ 0
    magnetic drift        L, L±2     ``magneticDriftScheme`` 1-9; d/dtheta, d/dzeta
                                     (upwinded) + non-standard d/dxi; per-species
    Er xiDot              L, L±2     ``includeElectricFieldTermInXiDot``
    Er xDot               L, L±2     ``includeXDotTerm``; dense in x (x ddx)
    PAS collisions        diag       ``collisionOperator=1``; ``nu_n nuD l(l+1)/2``
    FP collisions         diag       ``collisionOperator=0``; dense in (species,x)
    sources/constraints   n/a        bordered rows/cols, ``constraintScheme`` 1/2
    ====================  =========  ==========================================
    """

    # ---- static layout / option flags (pytree aux data) ----
    n_species: int
    n_x: int
    n_xi: int
    n_theta: int
    n_zeta: int
    rhs_mode: int
    constraint_scheme: int
    point_at_x0: bool
    use_dkes_exb: bool
    with_exb: bool
    with_er_xidot: bool
    with_er_xdot: bool

    # ---- speed / angle grids (phase_space) ----
    x: jnp.ndarray  # (X,)
    x_weights: jnp.ndarray  # (X,)
    ddx: jnp.ndarray  # (X,X)
    ddtheta: jnp.ndarray  # (T,T)
    ddzeta: jnp.ndarray  # (Z,Z)
    theta_weights: jnp.ndarray  # (T,)
    zeta_weights: jnp.ndarray  # (Z,)
    n_xi_for_x: jnp.ndarray  # (X,) int32; static pitch layout in the pytree
    xi_coupling_lower: jnp.ndarray  # (L,) l/(2l-1)
    xi_coupling_upper: jnp.ndarray  # (L,) (l+1)/(2l+3)

    # ---- flux-surface geometry (magnetic_geometry) ----
    b_hat: jnp.ndarray  # (T,Z)
    db_hat_dtheta: jnp.ndarray  # (T,Z)
    db_hat_dzeta: jnp.ndarray  # (T,Z)
    d_hat: jnp.ndarray  # (T,Z)
    b_hat_sup_theta: jnp.ndarray  # (T,Z)
    b_hat_sup_zeta: jnp.ndarray  # (T,Z)
    b_hat_sub_theta: jnp.ndarray  # (T,Z)
    b_hat_sub_zeta: jnp.ndarray  # (T,Z)
    fsab_hat2: jnp.ndarray  # scalar <BHat^2>

    # ---- species (species) ----
    z_s: jnp.ndarray  # (S,)
    m_hat: jnp.ndarray  # (S,)
    t_hat: jnp.ndarray  # (S,)
    n_hat: jnp.ndarray  # (S,)
    dn_hat_dpsi_hat: jnp.ndarray  # (S,)
    dt_hat_dpsi_hat: jnp.ndarray  # (S,)

    # ---- normalization / drive scalars (constants conventions) ----
    alpha: jnp.ndarray  # scalar e*phiBar/TBar
    delta: jnp.ndarray  # scalar rho*_ref
    dphi_hat_dpsi_hat: jnp.ndarray  # scalar; RHS-drive value (phi gradient coordinate)
    dphi_hat_dpsi_hat_kinetic: jnp.ndarray  # scalar; value multiplying ExB/Er terms
    e_parallel_hat: jnp.ndarray  # scalar
    e_parallel_hat_spec: jnp.ndarray  # (S,)

    # ---- collisions (dkx.collisions) ----
    pas: PitchAngleScatteringV3Operator | None
    fp: FokkerPlanckV3Operator | None
    # ``collisionOperator = 3``: the improved Sugama momentum/energy-conserving
    # model operator (research extension beyond v3).  Mutually exclusive with
    # ``pas``/``fp``; a collisionOperator=3 deck builds this one.  Same block
    # layout as ``fp`` (dense in x, block in species, diagonal in Legendre L).
    sugama: ImprovedSugamaV3Operator | None = None
    # ``includePhi1InCollisionOperator``: the poloidally varying Fokker-Planck
    # operator whose collisional densities are shifted by the Boltzmann factor
    # ``exp(-Z*alpha*Phi1Hat/THat)`` (``populateMatrix.F90`` Phi1-in-collision
    # branch).  Mutually exclusive with ``fp``: a collisionOperator=0 deck builds
    # one or the other.  Applied with the current ``Phi1Hat`` field so it enters
    # both the nonlinear residual and (via autodiff of the residual) the Jacobian.
    fp_phi1: FokkerPlanckV3Phi1Operator | None = None

    # ---- Phi1 / quasineutrality (populateMatrix.F90 QN block + lambda row;
    #      evaluateResidual.F90 nonlinear QN drive; the includePhi1 vertical
    #      slice). All default to the no-Phi1 configuration so the base RHSMode
    #      1/2/3 operators are unchanged. ----
    include_phi1: bool = False
    quasineutrality_option: int = 1
    include_phi1_in_kinetic: bool = False
    with_adiabatic: bool = False
    adiabatic_z: jnp.ndarray = 1.0  # scalar
    adiabatic_n_hat: jnp.ndarray = 0.0  # scalar
    adiabatic_t_hat: jnp.ndarray = 1.0  # scalar
    # Base Phi1(theta,zeta) linearization point for the QN/kinetic exp couplings.
    phi1_hat_base: jnp.ndarray | None = None  # (T,Z) or None
    # Full-state Newton linearization point: when set (Phi1 runs) ``apply`` is
    # the Jacobian-vector product of :meth:`residual_phi1` at this state, so the
    # nonlinear solve reuses :func:`dkx.solve.solve` as the inner linear
    # solver.  ``None`` for the base linear operators.
    phi1_lin_state: jnp.ndarray | None = None  # (total_size,) or None
    # ``readExternalPhi1``: a FIXED external Phi1(theta,zeta) field read from an
    # HDF5 file (``externalPhi1Filename``).  When set, ``include_phi1`` is False
    # (the state stays f-only: NO quasineutrality block, NO Phi1 unknown, NO
    # lambda row -- indices.F90 for readExternalPhi1) and the DKE is LINEAR
    # again: :meth:`apply` and :meth:`rhs` evaluate the Phi1-in-kinetic /
    # Phi1-in-collision term coefficients at this fixed field instead of a Newton
    # iterate (evaluateResidual.F90 readExternalPhi1 branch).
    external_phi1_hat: jnp.ndarray | None = None  # (T,Z) or None

    # ---- ``xDotDerivativeScheme != 0``: upwinded d/dx pair for the E_r xDot
    #      term (createGrids.F90 ``ddx_xDot_plus``/``ddx_xDot_minus``;
    #      populateMatrix.F90 selects by the sign of the local xDotFactor).
    #      ``None`` (scheme 0) keeps the centered ``ddx`` for both signs. ----
    ddx_xdot_plus: jnp.ndarray | None = None  # (X,X) or None
    ddx_xdot_minus: jnp.ndarray | None = None  # (X,X) or None

    # ---- tangential magnetic drifts (``magneticDriftScheme`` 1-9;
    #      populateMatrix.F90 d/dtheta, d/dzeta, and non-standard d/dxi drift
    #      terms with ``force0RadialCurrentInEquilibrium=.true.``, the hardcoded
    #      v3 value). All default to the no-drift configuration so the base
    #      operators are unchanged.  The d/dtheta and d/dzeta terms couple L,
    #      L±2 through the upwinded ``ddtheta/ddzeta_magneticDrift_plus/minus``
    #      stencils; the geometry arrays are the radial (psi) B-field
    #      derivatives that only the Boozer/VMEC geometries (schemes 5/11/12)
    #      populate.  ``magnetic_drift_scheme`` selects the geometricFactor1/2/3
    #      variant (see :meth:`_magnetic_drifts`); the flux-function scalars
    #      ``iota``/``g_hat``/``diota_dpsi_hat``/``p_prime_hat`` feed the
    #      scheme 3/4/8 shear terms and scheme 6 pressure term, and
    #      ``grad_psi_dot_grad_b_over_gpsipsi`` is the Sugama normal-curvature
    #      factor of schemes 5/6. ----
    with_magnetic_drifts: bool = False
    magnetic_drift_scheme: int = 0
    b_hat_sub_psi: jnp.ndarray | None = None  # (T,Z)
    db_hat_dpsi_hat: jnp.ndarray | None = None  # (T,Z)
    db_hat_sub_psi_dtheta: jnp.ndarray | None = None  # (T,Z)
    db_hat_sub_psi_dzeta: jnp.ndarray | None = None  # (T,Z)
    db_hat_sub_theta_dpsi_hat: jnp.ndarray | None = None  # (T,Z)
    db_hat_sub_zeta_dpsi_hat: jnp.ndarray | None = None  # (T,Z)
    ddtheta_magdrift_plus: jnp.ndarray | None = None  # (T,T) upwinded d/dtheta
    ddtheta_magdrift_minus: jnp.ndarray | None = None  # (T,T)
    ddzeta_magdrift_plus: jnp.ndarray | None = None  # (Z,Z) upwinded d/dzeta
    ddzeta_magdrift_minus: jnp.ndarray | None = None  # (Z,Z)
    iota: jnp.ndarray | None = None  # scalar (schemes 3/4/8)
    g_hat: jnp.ndarray | None = None  # scalar GHat (scheme 8 shear term)
    diota_dpsi_hat: jnp.ndarray | None = None  # scalar (schemes 4/8)
    p_prime_hat: jnp.ndarray | None = None  # scalar (scheme 6)
    grad_psi_dot_grad_b_over_gpsipsi: jnp.ndarray | None = None  # (T,Z) (schemes 5/6)

    # ------------------------------------------------------------------
    # pytree protocol
    # ------------------------------------------------------------------

    _AUX_FIELDS = (
        "n_species",
        "n_x",
        "n_xi",
        "n_theta",
        "n_zeta",
        "rhs_mode",
        "constraint_scheme",
        "point_at_x0",
        "use_dkes_exb",
        "with_exb",
        "with_er_xidot",
        "with_er_xdot",
        "include_phi1",
        "quasineutrality_option",
        "include_phi1_in_kinetic",
        "with_adiabatic",
        "with_magnetic_drifts",
        "magnetic_drift_scheme",
    )
    _CHILD_FIELDS = (
        "x",
        "x_weights",
        "ddx",
        "ddtheta",
        "ddzeta",
        "theta_weights",
        "zeta_weights",
        "xi_coupling_lower",
        "xi_coupling_upper",
        "b_hat",
        "db_hat_dtheta",
        "db_hat_dzeta",
        "d_hat",
        "b_hat_sup_theta",
        "b_hat_sup_zeta",
        "b_hat_sub_theta",
        "b_hat_sub_zeta",
        "fsab_hat2",
        "z_s",
        "m_hat",
        "t_hat",
        "n_hat",
        "dn_hat_dpsi_hat",
        "dt_hat_dpsi_hat",
        "alpha",
        "delta",
        "dphi_hat_dpsi_hat",
        "dphi_hat_dpsi_hat_kinetic",
        "e_parallel_hat",
        "e_parallel_hat_spec",
        "pas",
        "fp",
        "sugama",
        "fp_phi1",
        "adiabatic_z",
        "adiabatic_n_hat",
        "adiabatic_t_hat",
        "phi1_hat_base",
        "phi1_lin_state",
        "external_phi1_hat",
        "b_hat_sub_psi",
        "db_hat_dpsi_hat",
        "db_hat_sub_psi_dtheta",
        "db_hat_sub_psi_dzeta",
        "db_hat_sub_theta_dpsi_hat",
        "db_hat_sub_zeta_dpsi_hat",
        "ddtheta_magdrift_plus",
        "ddtheta_magdrift_minus",
        "ddzeta_magdrift_plus",
        "ddzeta_magdrift_minus",
        "iota",
        "g_hat",
        "diota_dpsi_hat",
        "p_prime_hat",
        "grad_psi_dot_grad_b_over_gpsipsi",
        "ddx_xdot_plus",
        "ddx_xdot_minus",
    )

    def tree_flatten(self):
        children = tuple(getattr(self, name) for name in self._CHILD_FIELDS)
        # Pitch truncation defines the discrete layout and solver route. Keep
        # it hashable and static; physical coefficients remain dynamic leaves.
        layout = tuple(int(n) for n in np.asarray(self.n_xi_for_x))
        aux = tuple(getattr(self, name) for name in self._AUX_FIELDS) + (layout,)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        kwargs = dict(zip(cls._AUX_FIELDS, aux[:-1]))
        kwargs.update(dict(zip(cls._CHILD_FIELDS, children)))
        layout = np.asarray(aux[-1], dtype=np.int32)
        layout.setflags(write=False)
        kwargs["n_xi_for_x"] = layout
        return cls(**kwargs)

    # ------------------------------------------------------------------
    # layout
    # ------------------------------------------------------------------

    @property
    def f_shape(self) -> tuple[int, int, int, int, int]:
        """Distribution-block shape ``(n_species, n_x, n_xi, n_theta, n_zeta)``."""
        return (self.n_species, self.n_x, self.n_xi, self.n_theta, self.n_zeta)

    @property
    def f_size(self) -> int:
        return int(np.prod(self.f_shape))

    @property
    def extra_size(self) -> int:
        """Number of bordered source unknowns (= number of constraint rows)."""
        if self.constraint_scheme == 0:
            return 0
        if self.constraint_scheme in (1, 3, 4):
            return 2 * self.n_species
        if self.constraint_scheme == 2:
            return self.n_species * self.n_x
        raise NotImplementedError(f"constraintScheme={self.constraint_scheme} is not supported.")

    @property
    def phi1_size(self) -> int:
        """Phi1(theta,zeta) unknowns plus the ``<Phi1>=0`` lambda row (indices.F90)."""
        if self.include_phi1:
            return self.n_theta * self.n_zeta + 1
        return 0

    @property
    def total_size(self) -> int:
        return self.f_size + self.phi1_size + self.extra_size

    # ------------------------------------------------------------------
    # matrix-free apply
    # ------------------------------------------------------------------

    def _mask(self) -> jnp.ndarray:
        return _mask_xi(self.n_xi_for_x, self.n_xi)  # (X,L)

    def active_dof_mask(self) -> jnp.ndarray | None:
        """Flat 0/1 mask of the DOFs retained by the ``Nxi_for_x`` truncation.

        The rectangular ``(species, x, L, theta, zeta)`` state layout carries
        entries for Legendre modes ``l >= Nxi_for_x(ix)`` that Fortran v3
        excludes from the matrix entirely (packed indexing, ``indices.F90``
        ``DKE_size``).  Those rows of :meth:`apply` are identically zero, so
        the rectangular embedding of the operator is structurally singular:
        one exact zero singular value per truncated DOF.  Solvers must pin
        them (``A M + (I - M)`` with ``M = diag(mask)``) to recover the
        nonsingular packed system — see :func:`dkx.solve.solve`.

        Returns:
            ``None`` when every speed node keeps the full Legendre resolution
            (no truncation, operator nonsingular as embedded); otherwise a
            ``(total_size,)`` float64 vector with 1.0 on active DOFs (all
            bordered source unknowns are active) and 0.0 on truncated ones.
        """
        if int(np.min(np.asarray(self.n_xi_for_x))) >= self.n_xi:
            return None
        m = jnp.broadcast_to(self._mask()[None, :, :, None, None], self.f_shape)
        return jnp.concatenate(
            [m.reshape((-1,)), jnp.ones((self.phi1_size + self.extra_size,), dtype=jnp.float64)]
        )

    def _streaming_mirror(self, f: jnp.ndarray) -> jnp.ndarray:
        """Parallel streaming + mirror terms (populateMatrix.F90 ddtheta/ddzeta/ddxi blocks).

        Couples Legendre rows L to columns L±1 with the
        ``legendre_coupling_lower/upper`` factors; diagonal in x with an overall
        factor ``x sqrt(THat/mHat)``.
        """
        mask = self._mask()
        # Fortran excludes the truncated-L DOFs entirely; mask columns first so
        # the L±1 coupling cannot pull from them.
        fm = f * mask[None, :, :, None, None]

        sqrt_t_over_m = jnp.sqrt(self.t_hat / self.m_hat)  # (S,)
        v_theta_s = sqrt_t_over_m[:, None, None] * (self.b_hat_sup_theta / self.b_hat)[None, :, :]
        v_zeta_s = sqrt_t_over_m[:, None, None] * (self.b_hat_sup_zeta / self.b_hat)[None, :, :]

        dtheta_f = jnp.einsum("ij,sxljz->sxliz", self.ddtheta, fm) * v_theta_s[:, None, None, :, :]
        dzeta_f = jnp.einsum("ij,sxltj->sxlti", self.ddzeta, fm) * v_zeta_s[:, None, None, :, :]

        coef_plus_x = self.x[:, None] * self.xi_coupling_upper[None, :]  # (X,L)
        coef_minus_x = self.x[:, None] * self.xi_coupling_lower[None, :]  # (X,L)

        def couple_l(g: jnp.ndarray) -> jnp.ndarray:
            term_plus = coef_plus_x[None, :, :-1, None, None] * g[:, :, 1:, :, :]
            term_plus = jnp.pad(term_plus, ((0, 0), (0, 0), (0, 1), (0, 0), (0, 0)))
            term_minus = coef_minus_x[None, :, 1:, None, None] * g[:, :, :-1, :, :]
            term_minus = jnp.pad(term_minus, ((0, 0), (0, 0), (1, 0), (0, 0), (0, 0)))
            return term_plus + term_minus

        out = couple_l(dtheta_f) + couple_l(dzeta_f)

        # Mirror term: -x sqrt(T/m) (b·∇B)/(2B^2) with (l+2)(l+1)/(2l+3) and
        # -l(l-1)/(2l-1) factors (= the streaming couplings times (l+2), -(l-1)).
        mirror_geom = self.b_hat_sup_theta * self.db_hat_dtheta + self.b_hat_sup_zeta * self.db_hat_dzeta
        mirror_factor = -sqrt_t_over_m[:, None, None] * mirror_geom[None, :, :] / (2.0 * self.b_hat**2)

        ell = jnp.arange(self.n_xi, dtype=jnp.float64)
        coef_mirror_plus_x = self.x[:, None] * (self.xi_coupling_upper * (ell + 2.0))[None, :]
        coef_mirror_minus_x = self.x[:, None] * (-self.xi_coupling_lower * (ell - 1.0))[None, :]

        term_plus = coef_mirror_plus_x[None, :, :-1, None, None] * fm[:, :, 1:, :, :]
        term_plus = jnp.pad(term_plus, ((0, 0), (0, 0), (0, 1), (0, 0), (0, 0)))
        term_minus = coef_mirror_minus_x[None, :, 1:, None, None] * fm[:, :, :-1, :, :]
        term_minus = jnp.pad(term_minus, ((0, 0), (0, 0), (1, 0), (0, 0), (0, 0)))
        out = out + (term_plus + term_minus) * mirror_factor[:, None, None, :, :]

        return out * mask[None, :, :, None, None]

    def _exb_coefficients(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Row-scaling ``(T,Z)`` coefficients of the ExB d/dtheta and d/dzeta terms."""
        denom = self.fsab_hat2 if self.use_dkes_exb else self.b_hat**2
        factor = self.alpha * self.delta * 0.5 * self.dphi_hat_dpsi_hat_kinetic
        coef_theta = factor * self.d_hat * self.b_hat_sub_zeta / denom
        coef_zeta = -factor * self.d_hat * self.b_hat_sub_theta / denom
        return coef_theta, coef_zeta

    def _exb(self, f: jnp.ndarray) -> jnp.ndarray:
        """ExB drift terms (diagonal in L and x; ``useDKESExBDrift`` selects <B^2>)."""
        coef_theta, coef_zeta = self._exb_coefficients()
        dtheta_f = jnp.einsum("ij,sxljz->sxliz", self.ddtheta, f)
        dzeta_f = jnp.einsum("ij,sxltj->sxlti", self.ddzeta, f)
        out = dtheta_f * coef_theta[None, None, None, :, :] + dzeta_f * coef_zeta[None, None, None, :, :]
        return out * self._mask()[None, :, :, None, None]

    def _er_xidot(self, f: jnp.ndarray) -> jnp.ndarray:
        """Non-standard d/dxi term associated with E_r (L diagonal and L±2)."""
        temp = self.b_hat_sub_zeta * self.db_hat_dtheta - self.b_hat_sub_theta * self.db_hat_dzeta
        factor = (
            self.alpha
            * self.delta
            * self.dphi_hat_dpsi_hat_kinetic
            / (4.0 * self.b_hat**3)
            * self.d_hat
            * temp
        )  # (T,Z)

        ell = jnp.arange(self.n_xi, dtype=jnp.float64)
        denom0 = (2.0 * ell - 1.0) * (2.0 * ell + 3.0)
        diag_coef = (ell + 1.0) * ell / denom0

        out = (factor[None, None, None, :, :] * diag_coef[None, None, :, None, None]) * f

        sup2 = (ell + 3.0) * (ell + 2.0) * (ell + 1.0) / ((2.0 * ell + 5.0) * (2.0 * ell + 3.0))
        sub2 = -ell * (ell - 1.0) * (ell - 2.0) / ((2.0 * ell - 3.0) * (2.0 * ell - 1.0))

        term_sup2 = sup2[None, None, :-2, None, None] * f[:, :, 2:, :, :]
        term_sup2 = jnp.pad(term_sup2, ((0, 0), (0, 0), (0, 2), (0, 0), (0, 0)))
        out = out + factor[None, None, None, :, :] * term_sup2

        term_sub2 = sub2[None, None, 2:, None, None] * f[:, :, :-2, :, :]
        term_sub2 = jnp.pad(term_sub2, ((0, 0), (0, 0), (2, 0), (0, 0), (0, 0)))
        out = out + factor[None, None, None, :, :] * term_sub2

        return out * self._mask()[None, :, :, None, None]

    def _er_xdot(self, f: jnp.ndarray) -> jnp.ndarray:
        """Collisionless d/dx term associated with E_r (dense in x; L and L±2).

        ``force0RadialCurrentInEquilibrium=.true.`` (v3 default) so
        xDotFactor2=0.  ``xDotDerivativeScheme=0`` uses the centered ``ddx``
        for both upwind directions; nonzero schemes select the
        ``ddx_xdot_plus``/``ddx_xdot_minus`` pair by the sign of the local
        xDotFactor, exactly as ``populateMatrix.F90`` does per (theta, zeta).
        """
        factor0 = -(self.alpha * self.delta * self.dphi_hat_dpsi_hat_kinetic) / 4.0
        xdot_factor = (
            factor0
            * self.d_hat
            / self.b_hat**3
            * (self.b_hat_sub_theta * self.db_hat_dzeta - self.b_hat_sub_zeta * self.db_hat_dtheta)
        )  # (T,Z)

        if self.ddx_xdot_plus is None and not self.point_at_x0:
            x_ddx = self.x[:, None] * self.ddx  # (X,X)

            def x_apply(g: jnp.ndarray) -> jnp.ndarray:
                # g (S,X,L',T,Z) -> x d/dx along the X axis.
                g_xlast = jnp.transpose(g, (0, 2, 3, 4, 1))
                y = jnp.einsum("ij,...j->...i", x_ddx, g_xlast)
                return jnp.transpose(y, (0, 4, 1, 2, 3))

            ell = jnp.arange(self.n_xi, dtype=jnp.float64)
            denom = (2.0 * ell + 3.0) * (2.0 * ell - 1.0)
            diag_coef = 2.0 * (3.0 * ell * ell + 3.0 * ell - 2.0) / denom  # (L,)

            out = x_apply(f) * (diag_coef[:, None, None] * xdot_factor[None, :, :])[None, None, :, :, :]

            if self.n_xi >= 3:
                l0 = ell[:-2]
                sup = (l0 + 1.0) * (l0 + 2.0) / ((2.0 * l0 + 5.0) * (2.0 * l0 + 3.0))
                y_sup = x_apply(f[:, :, 2:, :, :])
                y_sup = jnp.pad(y_sup, ((0, 0), (0, 0), (0, 2), (0, 0), (0, 0)))
                sup_coef = jnp.pad(sup, (0, 2))
                out = out + y_sup * (sup_coef[:, None, None] * xdot_factor[None, :, :])[None, None, :, :, :]

                l2 = ell[2:]
                sub = l2 * (l2 - 1.0) / ((2.0 * l2 - 3.0) * (2.0 * l2 - 1.0))
                y_sub = x_apply(f[:, :, :-2, :, :])
                y_sub = jnp.pad(y_sub, ((0, 0), (0, 0), (2, 0), (0, 0), (0, 0)))
                sub_coef = jnp.pad(sub, (2, 0))
                out = out + y_sub * (sub_coef[:, None, None] * xdot_factor[None, :, :])[None, None, :, :, :]

            return out * self._mask()[None, :, :, None, None]

        # Upwinded and/or point-at-x0 path.
        ddx_plus = self.ddx if self.ddx_xdot_plus is None else self.ddx_xdot_plus
        h = self._er_xdot_l_coupled(f, ddx_plus)
        if self.ddx_xdot_plus is not None:
            h_minus = self._er_xdot_l_coupled(f, self.ddx_xdot_minus)
            h = jnp.where((xdot_factor > 0.0)[None, None, None, :, :], h, h_minus)
        out = h * xdot_factor[None, None, None, :, :]
        return out * self._mask()[None, :, :, None, None]

    def _er_xdot_l_coupled(self, f: jnp.ndarray, ddx_mat: jnp.ndarray) -> jnp.ndarray:
        """L-coupled ``x d/dx`` combination of the xDot term for one upwind matrix.

        The sum of the diagonal-in-L and L±2 pieces WITHOUT the (theta, zeta)
        xDotFactor, which the caller applies after the upwind selection.  With
        a grid point at x=0 the x=0 column is dropped for L>0 rows but kept
        for L=0 rows (populateMatrix.F90 ``ixMinCol``).
        """
        x_ddx = self.x[:, None] * ddx_mat  # (X,X)
        x_ddx_nox0 = x_ddx.at[:, 0].set(0.0) if self.point_at_x0 else x_ddx

        def x_apply(g: jnp.ndarray, m: jnp.ndarray) -> jnp.ndarray:
            # g (S,X,L',T,Z) -> x d/dx along the X axis.
            g_xlast = jnp.transpose(g, (0, 2, 3, 4, 1))
            y = jnp.einsum("ij,...j->...i", m, g_xlast)
            return jnp.transpose(y, (0, 4, 1, 2, 3))

        ell = jnp.arange(self.n_xi, dtype=jnp.float64)
        denom = (2.0 * ell + 3.0) * (2.0 * ell - 1.0)
        diag_coef = 2.0 * (3.0 * ell * ell + 3.0 * ell - 2.0) / denom  # (L,)

        y_diag = x_apply(f, x_ddx_nox0)
        if self.point_at_x0:
            # Rows with L=0 keep the x=0 column (ixMinCol=1 for L=0).
            y_diag = y_diag.at[:, :, 0, :, :].set(x_apply(f[:, :, :1, :, :], x_ddx)[:, :, 0, :, :])
        out = y_diag * diag_coef[None, None, :, None, None]

        if self.n_xi >= 3:
            l0 = ell[:-2]
            sup = (l0 + 1.0) * (l0 + 2.0) / ((2.0 * l0 + 5.0) * (2.0 * l0 + 3.0))
            y_sup = x_apply(f[:, :, 2:, :, :], x_ddx_nox0)
            if self.point_at_x0:
                y_sup = y_sup.at[:, :, 0, :, :].set(
                    x_apply(f[:, :, 2:3, :, :], x_ddx)[:, :, 0, :, :]
                )
            y_sup = jnp.pad(y_sup, ((0, 0), (0, 0), (0, 2), (0, 0), (0, 0)))
            sup_coef = jnp.pad(sup, (0, 2))
            out = out + y_sup * sup_coef[None, None, :, None, None]

            l2 = ell[2:]
            sub = l2 * (l2 - 1.0) / ((2.0 * l2 - 3.0) * (2.0 * l2 - 1.0))
            y_sub = x_apply(f[:, :, :-2, :, :], x_ddx_nox0)
            y_sub = jnp.pad(y_sub, ((0, 0), (0, 0), (2, 0), (0, 0), (0, 0)))
            sub_coef = jnp.pad(sub, (2, 0))
            out = out + y_sub * sub_coef[None, None, :, None, None]

        return out

    def magnetic_drift_diagonal_parts(self) -> dict[str, jnp.ndarray] | None:
        """The ``L``-diagonal part of the magnetic drifts, as ``L``-invariant matrices.

        The drift terms couple Legendre row ``L`` to ``L`` and to ``L +- 2``, never
        to ``L +- 1`` (see :meth:`_magnetic_drifts`), which is why
        :meth:`to_block_tridiagonal` refuses them outright.  But the ``L``-diagonal
        half of them *does* fit a block-tridiagonal chain: it adds to ``D_L`` and
        nothing else.  That half is what the coarse preconditioner is missing
        against Fortran, whose ``preconditioner_magnetic_drifts_max_L`` keeps drifts
        in the preconditioner rather than dropping them
        (``populateMatrix.F90`` lines 544 and 671, ``whichMatrix==0``).

Measured through the production recycled Krylov path.  On the tiny drift
        fixtures the gain is modest (51 -> 30 GCROT iterations on ``magdrift_1species_tiny``,
        0-1 elsewhere), but it grows with resolution: on a reduced-resolution
        W7-X deck carrying the production physics --- Fokker-Planck collisions,
        magnetic drifts, ``geometryScheme`` 5, 194,404 unknowns --- it is
        **5838 -> 2163 iterations and 900 s -> 337 s**, a factor of 2.7 on both.

        An earlier 6000 -> 7 figure was a harness artifact --- it came from a
        dense pseudo-inverse of the *unpinned* stripped operator, so it measured
        the missing ``l = 0`` and mask pins, not the missing drifts.

        (Keeping the ``L +- 2`` part too needs a block-pentadiagonal solve.)

        The ``L`` dependence is separable: every term is a scalar function of ``L``
        times a matrix that does not depend on ``L``.  So this returns the matrices
        once, and :func:`dkx.coarse_precond` combines them per row with the scalar
        coefficients from :meth:`magnetic_drift_diagonal_coefficients`, instead of
        materializing an ``(S, X, L, TZ, TZ)`` array the size of a whole band.

        Returns:
            ``None`` when the operator carries no drifts, else a dict of
            ``(S, TZ, TZ)`` matrices ``mt1/mt2/mt3`` and ``mz1/mz2/mz3`` (the
            ``d/dtheta`` and ``d/dzeta`` parts against ``geometricFactor1/2/3``)
            and an ``(S, TZ)`` vector ``xi`` (the ``d/dxi`` diagonal), all carrying
            the per-species ``base`` factor but **not** the ``x^2`` speed factor,
            which the caller applies per ``x``.
        """
        if not self.with_magnetic_drifts:
            return None
        gf1_t, gf2_t, gf3_t, gf1_z, gf2_z, gf3_z, xi_temp = self._drift_geometric_factors()
        n_t, n_z = self.n_theta, self.n_zeta
        tz = n_t * n_z
        b = self.b_hat
        base = (
            self.delta * self.t_hat[:, None, None] * self.d_hat[None, :, :]
            / (2.0 * self.z_s[:, None, None] * b[None, :, :] ** 3)
        )  # (S,T,Z)
        dhat11 = self.d_hat[0, 0]
        eye_t, eye_z = jnp.eye(n_t), jnp.eye(n_z)
        out: dict[str, jnp.ndarray] = {}

        def _theta_mat(gf):
            """(S,TZ,TZ): d/dtheta at fixed zeta, upwind-selected per (s,theta,zeta).

            The upwind selector is built from ``gf1_t`` for *every* factor, not
            from the factor being multiplied --- that is what
            :meth:`_magnetic_drifts` does, and using ``gf2``/``gf3`` here instead
            silently changes the operator wherever they differ in sign.
            """
            if gf is None:
                return jnp.zeros((self.n_species, tz, tz), dtype=jnp.float64)
            use_plus = (gf1_t[None, :, :] * dhat11 / self.z_s[:, None, None]) > 0
            d_plus = self.ddtheta_magdrift_plus[None, :, None, :]   # (1,T,1,T)
            d_minus = self.ddtheta_magdrift_minus[None, :, None, :]
            d_sel = jnp.where(use_plus[:, :, :, None], d_plus, d_minus)  # (S,T,Z,T)
            scale = (base * gf[None, :, :])[:, :, :, None]               # (S,T,Z,1)
            # (S,T,Z,T) -> (S,T,Z,T,Z) diagonal in zeta -> (S,TZ,TZ)
            full = (scale * d_sel)[:, :, :, :, None] * eye_z[None, None, :, None, :]
            return full.reshape(self.n_species, tz, tz)

        def _zeta_mat(gf):
            """(S,TZ,TZ): d/dzeta at fixed theta, upwind-selected per (s,theta,zeta)."""
            if gf is None:
                return jnp.zeros((self.n_species, tz, tz), dtype=jnp.float64)
            use_plus = (gf1_z[None, :, :] * dhat11 / self.z_s[:, None, None]) > 0
            d_plus = self.ddzeta_magdrift_plus[None, None, :, :]   # (1,1,Z,Z)
            d_minus = self.ddzeta_magdrift_minus[None, None, :, :]
            d_sel = jnp.where(use_plus[:, :, :, None], d_plus, d_minus)  # (S,T,Z,Z)
            scale = (base * gf[None, :, :])[:, :, :, None]
            # (S,T,Z,Z) = [s, t, z_out, z_in] -> [s, t_out, z_out, t_in, z_in],
            # NOT [s, t_out, t_in, z_out, z_in]: the flat index is (t, z), so the
            # theta delta belongs between the two zeta axes, not before them.
            full = (scale * d_sel)[:, :, :, None, :] * eye_t[None, :, None, :, None]
            return full.reshape(self.n_species, tz, tz)

        for key, gf in (("mt1", gf1_t), ("mt2", gf2_t), ("mt3", gf3_t)):
            out[key] = _theta_mat(gf)
        for key, gf in (("mz1", gf1_z), ("mz2", gf2_z), ("mz3", gf3_z)):
            out[key] = _zeta_mat(gf)
        out["xi"] = (
            jnp.zeros((self.n_species, tz), dtype=jnp.float64)
            if xi_temp is None
            else (-base * xi_temp[None, :, :]).reshape(self.n_species, tz)
        )
        return out

    @staticmethod
    def magnetic_drift_diagonal_coefficients(ell: jnp.ndarray) -> dict[str, jnp.ndarray]:
        """Scalar ``L`` coefficients multiplying :meth:`magnetic_drift_diagonal_parts`.

        ``c1/c2/c3`` are the ``geometricFactor1/2/3`` diagonal coefficients shared by
        the ``d/dtheta`` and ``d/dzeta`` terms; ``xi`` is the ``d/dxi`` diagonal.
        Traced ``ell`` is fine --- these are arithmetic, not indexing.
        """
        ell = jnp.asarray(ell, dtype=jnp.float64)
        denom = (2.0 * ell + 3.0) * (2.0 * ell - 1.0)
        return {
            "c1": 2.0 * (3.0 * ell * ell + 3.0 * ell - 2.0) / denom,
            "c2": (2.0 * ell * ell + 2.0 * ell - 1.0) / denom,
            "c3": -2.0 * ell * (ell + 1.0) / denom,
            "xi": jnp.where(ell > 0, (ell + 1.0) * ell / ((2.0 * ell - 1.0) * (2.0 * ell + 3.0)), 0.0),
        }

    def _drift_geometric_factors(
        self,
    ) -> tuple[
        jnp.ndarray | None,
        jnp.ndarray | None,
        jnp.ndarray | None,
        jnp.ndarray | None,
        jnp.ndarray | None,
        jnp.ndarray | None,
        jnp.ndarray | None,
    ]:
        """``(gf1_t, gf2_t, gf3_t, gf1_z, gf2_z, gf3_z, xi_temp)`` per scheme.

        Direct transcription of the three ``select case (magneticDriftScheme)``
        blocks of populateMatrix.F90 (d/dtheta, d/dzeta, d/dxi) with the
        hardcoded v3 ``force0RadialCurrentInEquilibrium=.true.``.  ``None``
        marks a term that scheme drops entirely (schemes 3/4 have no d/dtheta
        drift; schemes 3/4/7 have no d/dxi drift; ``gf3`` is scheme-2 only).

        What distinguishes the scheme-1 family:

        * scheme 2 = scheme 1 plus the ``geometricFactor3`` terms built from
          ``BDotCurlB`` (geometry.F90),
        * scheme 7 = scheme 1 without the non-standard d/dxi drift term,
        * scheme 8 = scheme 1 with the magnetic-shear correction
          ``-diotadpsiHat GHat BHat^theta/(DHat iota^3)`` inside the d/dzeta
          ``geometricFactor2``,
        * scheme 9 = scheme 1 exactly: its intended d/dtheta shear term indexes
          ``BHat_sup_theta(itheta,izetaRow)`` where both variables are the -1
          sentinels of that block (populateMatrix.F90:574), an out-of-bounds
          read observed to contribute exactly 0 in the reference build (the
          scheme-9 golden matrix is bit-identical to scheme 1's), so the
          faithful port adds nothing.

        Schemes 3/4 fold the poloidal drift into the d/dzeta term via the
        field-line-following combination ``d/dzeta + (1/iota) d/dtheta``
        (scheme 4 adds the ``-diotadpsiHat BHat_sub_zeta/iota^2`` shear piece);
        schemes 5/6 are the Sugama forms built from
        ``gradpsidotgradB_overgpsipsi`` (scheme 5 regularized, scheme 6 the
        unregularized ``2 pPrimeHat/BHat`` pressure form).
        """
        scheme = int(self.magnetic_drift_scheme)
        b = self.b_hat
        gpg = self.grad_psi_dot_grad_b_over_gpsipsi

        # ---- d/dtheta block (populateMatrix.F90 case (…) at the drift-theta term) ----
        if scheme in (1, 2, 7, 8, 9):
            gf1_t = self.b_hat_sub_zeta * self.db_hat_dpsi_hat - self.b_hat_sub_psi * self.db_hat_dzeta
            gf2_t = 2.0 * b * (self.db_hat_sub_psi_dzeta - self.db_hat_sub_zeta_dpsi_hat)
        elif scheme in (3, 4):
            gf1_t = None
            gf2_t = None
        elif scheme == 5:
            gf1_t = self.b_hat_sub_zeta * gpg
            gf2_t = -2.0 * gf1_t
        elif scheme == 6:
            gf1_t = self.b_hat_sub_zeta * gpg
            gf2_t = self.b_hat_sub_zeta * 2.0 * self.p_prime_hat / b  # unregularized
        else:
            raise ValueError(f"Invalid magneticDriftScheme={scheme} in d/dtheta term")

        # ---- d/dzeta block ----
        if scheme in (1, 2, 7, 9):
            gf1_z = self.b_hat_sub_psi * self.db_hat_dtheta - self.b_hat_sub_theta * self.db_hat_dpsi_hat
            gf2_z = 2.0 * b * (self.db_hat_sub_theta_dpsi_hat - self.db_hat_sub_psi_dtheta)
        elif scheme in (3, 4):
            iota = self.iota
            gf1_z = (
                self.b_hat_sub_psi * (self.db_hat_dtheta + self.db_hat_dzeta / iota)
                - (self.b_hat_sub_theta + self.b_hat_sub_zeta / iota) * self.db_hat_dpsi_hat
            )
            shear = (
                -self.diota_dpsi_hat / (iota * iota) * self.b_hat_sub_zeta if scheme == 4 else 0.0
            )
            gf2_z = 2.0 * b * (
                self.db_hat_sub_theta_dpsi_hat
                + self.db_hat_sub_zeta_dpsi_hat / iota
                + shear
                - (self.db_hat_sub_psi_dtheta + self.db_hat_sub_psi_dzeta / iota)
            )
        elif scheme == 5:
            gf1_z = -self.b_hat_sub_theta * gpg
            gf2_z = -2.0 * gf1_z
        elif scheme == 6:
            gf1_z = -self.b_hat_sub_theta * gpg
            gf2_z = -self.b_hat_sub_theta * 2.0 * self.p_prime_hat / b  # unregularized
        elif scheme == 8:
            iota = self.iota
            gf1_z = self.b_hat_sub_psi * self.db_hat_dtheta - self.b_hat_sub_theta * self.db_hat_dpsi_hat
            gf2_z = 2.0 * b * (
                self.db_hat_sub_theta_dpsi_hat
                - self.db_hat_sub_psi_dtheta
                - self.diota_dpsi_hat
                * self.g_hat
                * self.b_hat_sup_theta
                / (self.d_hat * iota * iota * iota)
            )
        else:
            raise ValueError(f"Invalid magneticDriftScheme={scheme} in d/dzeta term")

        # ---- geometricFactor3 (scheme 2 only): BDotCurlB terms.  geometry.F90
        #      computes BDotCurlB with force0RadialCurrentInEquilibrium=.true.,
        #      i.e. without the BHat_sub_psi curl piece. ----
        if scheme == 2:
            b_dot_curl_b = self.d_hat * (
                self.b_hat_sub_theta * (self.db_hat_sub_psi_dzeta - self.db_hat_sub_zeta_dpsi_hat)
                + self.b_hat_sub_zeta * (self.db_hat_sub_theta_dpsi_hat - self.db_hat_sub_psi_dtheta)
            )
            gf3_t = b_dot_curl_b * self.b_hat_sup_theta / (b * self.d_hat)
            gf3_z = b_dot_curl_b * self.b_hat_sup_zeta / (b * self.d_hat)
        else:
            gf3_t = None
            gf3_z = None

        # ---- non-standard d/dxi block ----
        if scheme in (1, 2, 8, 9):
            xi_temp = (
                self.db_hat_sub_psi_dzeta - self.db_hat_sub_zeta_dpsi_hat
            ) * self.db_hat_dtheta + (
                self.db_hat_sub_theta_dpsi_hat - self.db_hat_sub_psi_dtheta
            ) * self.db_hat_dzeta
        elif scheme in (3, 4, 7):
            xi_temp = None
        elif scheme in (5, 6):
            xi_temp = (
                -(
                    self.b_hat_sub_zeta * self.db_hat_dtheta
                    - self.b_hat_sub_theta * self.db_hat_dzeta
                )
                / b
                * gpg
            )
        else:
            raise ValueError(f"Invalid magneticDriftScheme={scheme} in d/dxi term")

        return gf1_t, gf2_t, gf3_t, gf1_z, gf2_z, gf3_z, xi_temp

    def _magnetic_drifts(self, f: jnp.ndarray) -> jnp.ndarray:
        """Tangential (poloidal+toroidal) magnetic-drift streaming terms.

        The three ``populateMatrix.F90`` magnetic-drift blocks for
        ``magneticDriftScheme`` 1-9 with the hardcoded v3
        ``force0RadialCurrentInEquilibrium=.true.``:

        * the d/dtheta drift term,
        * the d/dzeta drift term, and
        * the non-standard d/dxi drift term,

        with the per-scheme ``geometricFactor1/2/3`` variants supplied by
        :meth:`_drift_geometric_factors`.

        The d/dtheta and d/dzeta terms couple Legendre rows L to L (the
        ``2(3L^2+3L-2)`` / ``(2L^2+2L-1)`` / ``-2L(L+1)`` diagonal coefficients
        of geometricFactor1/2/3) and to L±2 (the ``(L+2)(L+1)/…`` /
        ``(L-1)L/…`` couplings of ``gf1 + gf2 - 3 gf3``); the d/dxi term
        couples L to L and L±2 (the ``(L+1)L/…`` diagonal and
        ``(L+3)(L+2)(L+1)/…`` / ``-L(L-1)(L-2)/…`` couplings).  Because of the
        L±2 coupling the operator is not block-tridiagonal in L, so
        :meth:`to_block_tridiagonal` refuses it and recycled Krylov GCROT owns
        these decks.

        Per-species ``THat``/``Z`` enter via ``factor = Delta THat DHat x^2 /
        (2 Z BHat^3)`` and the upwind selector ``sign(geometricFactor1 DHat(1,1)
        / Z)`` picks ``ddtheta/ddzeta_magneticDrift_plus`` vs ``…_minus``
        (``magneticDriftDerivativeScheme != 0``).
        """
        mask = self._mask()  # (X,L)
        fm = f * mask[None, :, :, None, None]

        b = self.b_hat
        # factor = Delta THat DHat x^2 / (2 Z BHat^3): the per-(species, theta,
        # zeta) part, kept separate from the x^2 speed part.
        base_s = (
            self.delta
            * self.t_hat[:, None, None]
            * self.d_hat[None, :, :]
            / (2.0 * self.z_s[:, None, None] * b[None, :, :] ** 3)
        )  # (S,T,Z)
        x2 = self.x * self.x  # (X,)
        dhat11 = self.d_hat[0, 0]

        ell = jnp.arange(self.n_xi, dtype=jnp.float64)
        denom0 = (2.0 * ell + 3.0) * (2.0 * ell - 1.0)
        # d/dtheta, d/dzeta diagonal-in-L coefficients for geometricFactor1/2/3.
        c1 = 2.0 * (3.0 * ell * ell + 3.0 * ell - 2.0) / denom0
        c2 = (2.0 * ell * ell + 2.0 * ell - 1.0) / denom0
        c3 = -2.0 * ell * (ell + 1.0) / denom0
        # L±2 couplings shared by the d/dtheta and d/dzeta terms.
        off_plus = (ell + 2.0) * (ell + 1.0) / ((2.0 * ell + 5.0) * (2.0 * ell + 3.0))
        off_minus = jnp.where(ell > 1, (ell - 1.0) * ell / ((2.0 * ell - 3.0) * (2.0 * ell - 1.0)), 0.0)

        def _lpm2(g: jnp.ndarray, cp: jnp.ndarray, cm: jnp.ndarray) -> jnp.ndarray:
            term_plus = cp[None, None, :-2, None, None] * g[:, :, 2:, :, :]
            term_plus = jnp.pad(term_plus, ((0, 0), (0, 0), (0, 2), (0, 0), (0, 0)))
            term_minus = cm[None, None, 2:, None, None] * g[:, :, :-2, :, :]
            term_minus = jnp.pad(term_minus, ((0, 0), (0, 0), (2, 0), (0, 0), (0, 0)))
            return term_plus + term_minus

        gf1_t, gf2_t, gf3_t, gf1_z, gf2_z, gf3_z, xi_temp = self._drift_geometric_factors()
        out = jnp.zeros_like(f)

        # ---- d/dtheta magnetic-drift term ----
        if gf1_t is not None:
            gf3t = gf3_t if gf3_t is not None else jnp.zeros_like(gf1_t)
            gf123_t = gf1_t + gf2_t - 3.0 * gf3t
            dtheta_plus = jnp.einsum("ij,sxljz->sxliz", self.ddtheta_magdrift_plus, fm)
            dtheta_minus = jnp.einsum("ij,sxljz->sxliz", self.ddtheta_magdrift_minus, fm)
            use_plus_t = (gf1_t[None, :, :] * dhat11 / self.z_s[:, None, None]) > 0  # (S,T,Z)
            dtheta_f = jnp.where(use_plus_t[:, None, None, :, :], dtheta_plus, dtheta_minus)
            diag_t = (
                c1[:, None, None] * gf1_t[None, :, :]
                + c2[:, None, None] * gf2_t[None, :, :]
                + c3[:, None, None] * gf3t[None, :, :]
            )  # (L,T,Z)
            theta = diag_t[None, None, :, :, :] * dtheta_f + gf123_t[None, None, None, :, :] * _lpm2(
                dtheta_f, off_plus, off_minus
            )
            out = out + base_s[:, None, None, :, :] * x2[None, :, None, None, None] * theta

        # ---- d/dzeta magnetic-drift term ----
        if gf1_z is not None:
            gf3z = gf3_z if gf3_z is not None else jnp.zeros_like(gf1_z)
            gf123_z = gf1_z + gf2_z - 3.0 * gf3z
            dzeta_plus = jnp.einsum("ij,sxltj->sxlti", self.ddzeta_magdrift_plus, fm)
            dzeta_minus = jnp.einsum("ij,sxltj->sxlti", self.ddzeta_magdrift_minus, fm)
            use_plus_z = (gf1_z[None, :, :] * dhat11 / self.z_s[:, None, None]) > 0  # (S,T,Z)
            dzeta_f = jnp.where(use_plus_z[:, None, None, :, :], dzeta_plus, dzeta_minus)
            diag_z = (
                c1[:, None, None] * gf1_z[None, :, :]
                + c2[:, None, None] * gf2_z[None, :, :]
                + c3[:, None, None] * gf3z[None, :, :]
            )  # (L,T,Z)
            zeta = diag_z[None, None, :, :, :] * dzeta_f + gf123_z[None, None, None, :, :] * _lpm2(
                dzeta_f, off_plus, off_minus
            )
            out = out + base_s[:, None, None, :, :] * x2[None, :, None, None, None] * zeta

        # ---- non-standard d/dxi magnetic-drift term ----
        if xi_temp is not None:
            xidot_diag = jnp.where(ell > 0, (ell + 1.0) * ell / ((2.0 * ell - 1.0) * (2.0 * ell + 3.0)), 0.0)
            xidot_plus = (ell + 3.0) * (ell + 2.0) * (ell + 1.0) / ((2.0 * ell + 5.0) * (2.0 * ell + 3.0))
            xidot_minus = jnp.where(
                ell > 1, -ell * (ell - 1.0) * (ell - 2.0) / ((2.0 * ell - 3.0) * (2.0 * ell - 1.0)), 0.0
            )
            xidot_parts = xidot_diag[None, None, :, None, None] * fm + _lpm2(fm, xidot_plus, xidot_minus)
            # xiDot factor = -Delta THat DHat x^2 / (2 Z BHat^3) * temp = -base_s * temp * x^2.
            out = out - (base_s * xi_temp[None, :, :])[:, None, None, :, :] * x2[
                None, :, None, None, None
            ] * xidot_parts

        return out * mask[None, :, :, None, None]

    def apply_f(self, f: jnp.ndarray, phi1_hat: jnp.ndarray | None = None) -> jnp.ndarray:
        """Apply the f-block (kinetic) part of the operator to a 5-D ``f``.

        ``phi1_hat`` is the ``Phi1Hat(theta, zeta)`` field the collision operator
        uses when ``includePhi1InCollisionOperator`` is active (``fp_phi1``): the
        collisional densities become poloidally varying,
        ``n_pol = nHat * exp(-Z*alpha*Phi1Hat/THat)``.  For the non-Phi1
        collision operators (``pas``/``fp``) it is ignored; it defaults to the
        frozen linearization field ``phi1_hat_base`` (zeros for the base linear
        operators).
        """
        f = jnp.asarray(f, dtype=jnp.float64)
        if f.shape != self.f_shape:
            raise ValueError(f"f must have shape {self.f_shape}, got {f.shape}")
        out = self._streaming_mirror(f)
        if self.with_exb:
            out = out + self._exb(f)
        if self.with_magnetic_drifts:
            out = out + self._magnetic_drifts(f)
        if self.with_er_xidot:
            out = out + self._er_xidot(f)
        if self.with_er_xdot:
            out = out + self._er_xdot(f)
        if self.pas is not None:
            out = out + apply_pitch_angle_scattering_v3(self.pas, f)
        if self.fp is not None:
            out = out + apply_fokker_planck_v3(self.fp, f)
        if self.sugama is not None:
            out = out + apply_improved_sugama_v3(self.sugama, f)
        if self.fp_phi1 is not None:
            ph = phi1_hat if phi1_hat is not None else self.phi1_hat_base
            if ph is None:
                ph = jnp.zeros((self.n_theta, self.n_zeta), dtype=jnp.float64)
            out = out + apply_fokker_planck_v3_phi1(self.fp_phi1, f, phi1_hat=ph)
        if self.point_at_x0:
            # populateMatrix.F90: with a grid point at x=0 every DKE term skips
            # the ix=1 row (ixMin=2); that row instead carries the x=0 boundary
            # conditions: f(x=0)=0 for L>0 (identity rows, up to the Nxi_for_x
            # truncation) and the regularity condition df/dx(x=0)=0 (the first
            # ddx row) for L=0.
            mask0 = self._mask()[0]  # (L,)
            bc = f[:, 0, :, :, :] * mask0[None, :, None, None]  # (S,L,T,Z)
            bc0 = jnp.einsum("j,sjtz->stz", self.ddx[0, :], f[:, :, 0, :, :])
            bc = bc.at[:, 0, :, :].set(bc0)
            out = out.at[:, 0, :, :, :].set(bc)
        return out

    def _fs_average_factor(self) -> jnp.ndarray:
        """(T,Z) weights of the flux-surface average (w_theta w_zeta / DHat)."""
        return (self.theta_weights[:, None] * self.zeta_weights[None, :]) / self.d_hat

    def _source_basis(self, scheme: int) -> tuple[jnp.ndarray, jnp.ndarray]:
        """(xPartOfSource1, xPartOfSource2) source x-shapes for constraintScheme 1/3/4.

        ``populateMatrix.F90`` lines 2915-2938 (the ``whichMatrix != 4,5`` branch):
        the particle source ``S_p`` provides particles but no heat and the energy
        source ``S_e`` provides heat but no particles, both normalized to the same
        density/pressure moments.  The three schemes differ only in which two
        Laguerre-like polynomial terms carry the sources (the density/pressure
        constraint rows built in :meth:`apply` are identical across 1/3/4):

        * scheme 1 — constant + quadratic;
        * scheme 3 — constant + quartic;
        * scheme 4 — quadratic + quartic.
        """
        x2 = self.x * self.x
        x4 = x2 * x2
        coef = jnp.exp(-x2) / (jnp.pi * jnp.sqrt(jnp.pi))
        if scheme == 1:
            return (-x2 + 2.5) * coef, ((2.0 / 3.0) * x2 - 1.0) * coef
        if scheme == 3:
            return (-(1.0 / 5.0) * x4 + 7.0 / 4.0) * coef, ((2.0 / 15.0) * x4 - 0.5) * coef
        if scheme == 4:
            return (
                (-(2.0 / 3.0) * x4 + (7.0 / 3.0) * x2) * coef,
                ((4.0 / 15.0) * x4 - (2.0 / 3.0) * x2) * coef,
            )
        raise NotImplementedError(f"constraintScheme={scheme} has no source basis.")

    def _source_and_constraint_rows(
        self, y_f: jnp.ndarray, f: jnp.ndarray, extra: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Border ``y_f`` with the source injection and evaluate the constraint rows.

        ``B`` injects the constraint-scheme source shapes into the L=0 DKE rows
        of ``y_f`` and ``C`` evaluates the flux-surface-average moment rows of
        ``f`` (populateMatrix.F90 source/constraint blocks).  Shared by the base
        linear :meth:`apply` and the fixed-external-Phi1 :meth:`_apply_external_phi1`.
        """
        factor = self._fs_average_factor()
        ix0 = _ix_min(self.point_at_x0)

        if self.constraint_scheme == 0:
            return y_f, jnp.zeros((0,), dtype=jnp.float64)

        if self.constraint_scheme in (1, 3, 4):
            src = extra.reshape((self.n_species, 2))
            xpart1, xpart2 = self._source_basis(self.constraint_scheme)
            y_f = y_f.at[:, ix0:, 0, :, :].add(
                xpart1[ix0:][None, :, None, None] * src[:, 0, None, None, None]
                + xpart2[ix0:][None, :, None, None] * src[:, 1, None, None, None]
            )
            x2 = self.x * self.x
            w2 = x2 * self.x_weights
            w4 = x2 * x2 * self.x_weights
            y_dens = jnp.einsum("x,tz,sxtz->s", w2, factor, f[:, :, 0, :, :])
            y_pres = jnp.einsum("x,tz,sxtz->s", w4, factor, f[:, :, 0, :, :])
            return y_f, jnp.stack([y_dens, y_pres], axis=1).reshape((-1,))

        if self.constraint_scheme == 2:
            src = extra.reshape((self.n_species, self.n_x))
            y_f = y_f.at[:, ix0:, 0, :, :].add(src[:, ix0:, None, None])
            y_avg = jnp.einsum("tz,sxtz->sx", factor, f[:, :, 0, :, :])
            if self.point_at_x0:
                y_avg = y_avg.at[:, 0].set(src[:, 0])
            return y_f, y_avg.reshape((-1,))

        raise NotImplementedError(f"constraintScheme={self.constraint_scheme} is not supported.")

    def apply(self, v: jnp.ndarray) -> jnp.ndarray:
        """Apply the full bordered operator ``[[A, B], [C, 0]]`` to a flat state.

        ``A`` is the kinetic f-block, ``B`` injects the constraint-scheme source
        shapes into the L=0 DKE rows, and ``C`` evaluates the flux-surface-average
        moments (populateMatrix.F90 source/constraint blocks).
        """
        v = jnp.asarray(v, dtype=jnp.float64)
        if v.shape != (self.total_size,):
            raise ValueError(f"v must have shape {(self.total_size,)}, got {v.shape}")

        if self.external_phi1_hat is not None:
            # readExternalPhi1: the state is f-only and the DKE is LINEAR, with the
            # Phi1 terms evaluated at the fixed external field (no Newton, no QN).
            return self._apply_external_phi1(v)

        if self.include_phi1:
            # Phi1 makes the DKE nonlinear (quasineutrality couples Phi1 back into
            # the kinetic equation).  ``apply`` is the linear operator consumed by
            # :func:`dkx.solve.solve`, so for Phi1 runs it is the
            # Jacobian-vector product of :meth:`residual_phi1` at
            # ``phi1_lin_state`` (the current Newton iterate) — the exact
            # linearization that the parity oracle takes with ``jax.linearize``.
            base = self.phi1_lin_state
            if base is None:
                base = jnp.zeros((self.total_size,), dtype=jnp.float64)
            return jax.jvp(self.residual_phi1, (base,), (v,))[1]

        f = v[: self.f_size].reshape(self.f_shape)
        extra = v[self.f_size :]
        y_f, y_extra = self._source_and_constraint_rows(self.apply_f(f), f, extra)
        return jnp.concatenate([y_f.reshape((-1,)), y_extra], axis=0)

    # ------------------------------------------------------------------
    # Phi1 / quasineutrality (includePhi1 vertical slice)
    #
    # State layout with Phi1 (indices.F90 ordering):
    #     [ f(species,x,L,theta,zeta) | Phi1(theta,zeta) | lambda | sources ]
    #
    # ``residual_phi1`` is the nonlinear residual ``A(x) - b(x)`` (Newton is the
    # outer solve in ``dkx.phi1``); it is element-wise bit-comparable to
    # ``operators.profile_system.residual_v3_full_system``.
    # ------------------------------------------------------------------

    def _nonlinear_temp_vector_phi1(self, f: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """``tempVector2`` L-coupling of ``d f/dx`` for the Phi1-in-kinetic terms.

        Mirrors ``populateMatrix.F90``'s tempVector2 assembly (see the legacy
        ``operators.profile_system._nonlinear_temp_vector``): a speed-derivative
        (``x ddx``) combined with the L±1 Legendre couplings, masked to the
        retained (x, L) DOFs.
        """
        n_xi = int(self.n_xi)
        inv_x = 1.0 / self.x  # (X,)
        ddx_to_use = jnp.where(jnp.abs(self.ddx) > 1e-12, self.ddx, 0.0)  # sparsify.F90 threshold
        ddx_f = jnp.einsum("ij,sltzj->sltzi", ddx_to_use, jnp.transpose(f, (0, 2, 3, 4, 1)))
        ddx_f = jnp.transpose(ddx_f, (0, 4, 1, 2, 3))  # (S,X,L,T,Z)

        out_nl = jnp.zeros_like(f, dtype=jnp.float64)
        ell = jnp.arange(n_xi, dtype=jnp.float64)
        if n_xi > 1:
            lp1 = ell[:-1]
            coef = (lp1 + 1.0) / (2.0 * lp1 + 3.0)
            diag_xl = (((lp1 + 1.0) * (lp1 + 2.0) / (2.0 * lp1 + 3.0))[:, None] * inv_x[None, :]).T
            src = f[:, :, 1:, :, :]
            ddx_src = ddx_f[:, :, 1:, :, :]
            term = coef[None, None, :, None, None] * ddx_src + diag_xl[None, :, :, None, None] * src
            out_nl = out_nl.at[:, :, :-1, :, :].add(term)

            lm1 = ell[1:]
            coef = lm1 / (2.0 * lm1 - 1.0)
            diag_xl = ((-(lm1 - 1.0) * lm1 / (2.0 * lm1 - 1.0))[:, None] * inv_x[None, :]).T
            src = f[:, :, :-1, :, :]
            ddx_src = ddx_f[:, :, :-1, :, :]
            term = coef[None, None, :, None, None] * ddx_src + diag_xl[None, :, :, None, None] * src
            out_nl = out_nl.at[:, :, 1:, :, :].add(term)

        ix0 = _ix_min(bool(self.point_at_x0))
        mask_x = (jnp.arange(self.n_x) >= ix0).astype(jnp.float64)  # (X,)
        mask_l = _mask_xi(self.n_xi_for_x, self.n_xi)  # (X,L)
        return out_nl, mask_l, mask_x

    def _phi1_in_kinetic_flinear(self, f: jnp.ndarray, phi1: jnp.ndarray) -> jnp.ndarray:
        """The ``f``-linear part of the Phi1-in-kinetic coupling (``populateMatrix.F90``).

        The tempVector2 speed-derivative term ``x d f/dx`` combined with the
        L±1 Legendre couplings and the ``E·b`` Phi1-gradient factor.  Linear in
        ``f`` at fixed ``phi1``; used both by the self-consistent Newton residual
        (via :meth:`_add_phi1_in_kinetic`) and the fixed-external-Phi1 operator.
        """
        dphi1_dtheta = self.ddtheta @ phi1  # (T,Z)
        dphi1_dzeta = phi1 @ self.ddzeta.T  # (T,Z)
        e_term = self.b_hat_sup_theta * dphi1_dtheta + self.b_hat_sup_zeta * dphi1_dzeta  # (T,Z)
        nonlinear_factor = (
            -(self.alpha * self.z_s)[:, None, None]
            / (2.0 * self.b_hat[None, :, :] * jnp.sqrt(self.t_hat)[:, None, None] * jnp.sqrt(self.m_hat)[:, None, None])
            * e_term[None, :, :]
        )  # (S,T,Z)
        out_nl, mask_l, mask_x = self._nonlinear_temp_vector_phi1(f)
        return (
            out_nl
            * nonlinear_factor[:, None, None, :, :]
            * mask_l[None, :, :, None, None]
            * mask_x[None, :, None, None, None]
        )

    def _phi1_in_kinetic_source(self, phi1: jnp.ndarray) -> jnp.ndarray:
        """The Phi1-only (``f``-independent) part of the Phi1-in-kinetic coupling.

        The ``coeff1``/``coeff2`` × ``dPhi1`` pieces added to the L=0 DKE rows;
        depends on ``phi1`` (and the species/geometry) but not on ``f``.  For the
        self-consistent Newton residual it is a Jacobian-frozen contribution; for
        the fixed-external-Phi1 linear system it is a constant that moves to the
        right-hand side.  Returns the ``(S,X,T,Z)`` array added to ``y_f[:,:,0]``.
        """
        dphi1_dtheta = self.ddtheta @ phi1  # (T,Z)
        dphi1_dzeta = phi1 @ self.ddzeta.T  # (T,Z)
        x2 = self.x * self.x
        expx2 = jnp.exp(-x2)
        sqrt_pi = jnp.sqrt(jnp.pi)
        norm = jnp.pi * sqrt_pi

        exp_phi = jnp.exp(-(self.z_s[:, None, None] * self.alpha / self.t_hat[:, None, None]) * phi1[None, :, :])
        sp_pref1 = self.n_hat * (self.m_hat * jnp.sqrt(self.m_hat)) / (self.t_hat * jnp.sqrt(self.t_hat) * norm)
        bracket = (self.dn_hat_dpsi_hat / self.n_hat)[:, None] + (x2[None, :] - 1.5) * (
            self.dt_hat_dpsi_hat / self.t_hat
        )[:, None]  # (S,X)
        fm = sp_pref1[:, None] * expx2[None, :] * bracket  # (S,X)

        geom_theta = -self.alpha * self.delta * self.d_hat * self.b_hat_sub_zeta / (2.0 * (self.b_hat * self.b_hat))
        geom_zeta = self.alpha * self.delta * self.d_hat * self.b_hat_sub_theta / (2.0 * (self.b_hat * self.b_hat))
        coeff1_theta = fm[:, :, None, None] * geom_theta[None, None, :, :] * exp_phi[:, None, :, :]
        coeff1_zeta = fm[:, :, None, None] * geom_zeta[None, None, :, :] * exp_phi[:, None, :, :]

        sp_pref2 = self.z_s * self.n_hat * (self.m_hat * jnp.sqrt(self.m_hat)) / (
            self.t_hat * self.t_hat * jnp.sqrt(self.t_hat)
        )
        phi_term = self.dphi_hat_dpsi_hat + phi1[None, :, :] * (self.dt_hat_dpsi_hat / self.t_hat)[:, None, None]
        geom2_theta = -(self.alpha * self.alpha) * self.delta * self.d_hat * self.b_hat_sub_zeta / (
            2.0 * norm * (self.b_hat * self.b_hat)
        )
        geom2_zeta = (self.alpha * self.alpha) * self.delta * self.d_hat * self.b_hat_sub_theta / (
            2.0 * norm * (self.b_hat * self.b_hat)
        )
        coeff2_theta = (
            sp_pref2[:, None, None, None]
            * expx2[None, :, None, None]
            * exp_phi[:, None, :, :]
            * phi_term[:, None, :, :]
            * geom2_theta[None, None, :, :]
        )
        coeff2_zeta = (
            sp_pref2[:, None, None, None]
            * expx2[None, :, None, None]
            * exp_phi[:, None, :, :]
            * phi_term[:, None, :, :]
            * geom2_zeta[None, None, :, :]
        )
        return (coeff1_theta + coeff2_theta) * dphi1_dtheta[None, None, :, :] + (
            coeff1_zeta + coeff2_zeta
        ) * dphi1_dzeta[None, None, :, :]

    def _add_phi1_in_kinetic(self, y_f: jnp.ndarray, f: jnp.ndarray, phi1: jnp.ndarray) -> jnp.ndarray:
        """Add the Phi1-in-kinetic-equation coupling to the L=0 DKE rows.

        Residual-mode (``includeJacobianTerms=False``) counterpart of the
        ``populateMatrix.F90`` Phi1-gradient blocks: uses the *current* Phi1
        (not the frozen base state).  The ``f``-linear part
        (:meth:`_phi1_in_kinetic_flinear`) and the Phi1-only part
        (:meth:`_phi1_in_kinetic_source`) together reproduce the original block.
        """
        y_f = y_f + self._phi1_in_kinetic_flinear(f, phi1)
        return y_f.at[:, :, 0, :, :].add(self._phi1_in_kinetic_source(phi1))

    def _quasineutrality_rows(self, f: jnp.ndarray, phi1: jnp.ndarray, lam: jnp.ndarray) -> jnp.ndarray:
        """The quasineutrality rows + ``<Phi1>=0`` lambda row (residual mode).

        quasineutralityOption 1 (full): the nonlinear Boltzmann response lives in
        :meth:`rhs_phi1` so the operator row is ``qn_from_f + lambda``.
        quasineutralityOption 2 (EUTERPE): only the first kinetic species enters
        the charge density and Phi1 has a linear (adiabatic) diagonal.
        """
        x2w = (self.x * self.x) * self.x_weights  # (X,)
        species_factor = 4.0 * jnp.pi * self.z_s * self.t_hat / self.m_hat * jnp.sqrt(self.t_hat / self.m_hat)
        if int(self.quasineutrality_option) == 2:
            qn_from_f = species_factor[0] * jnp.einsum("x,xtz->tz", x2w, f[0, :, 0, :, :])
            phi1_diag = jnp.asarray(0.0, dtype=jnp.float64)
            if self.with_adiabatic and self.n_species > 0:
                phi1_diag = -self.alpha * (
                    (self.z_s[0] * self.z_s[0]) * self.n_hat[0] / self.t_hat[0]
                    + (self.adiabatic_z * self.adiabatic_z) * self.adiabatic_n_hat / self.adiabatic_t_hat
                )
            qn = qn_from_f + phi1_diag * phi1
        elif int(self.quasineutrality_option) == 1:
            qn = jnp.einsum("s,x,sxtz->tz", species_factor, x2w, f[:, :, 0, :, :])
        else:
            raise NotImplementedError(
                f"quasineutralityOption={self.quasineutrality_option} is not supported "
                "(only 1 (full) and 2 (EUTERPE) are consolidated)."
            )
        qn = qn + lam
        factor = self._fs_average_factor()
        y_lam = jnp.sum(factor * phi1)
        return jnp.concatenate([qn.reshape((-1,)), jnp.asarray([y_lam], dtype=jnp.float64)], axis=0)

    def _apply_phi1_operator(self, v: jnp.ndarray) -> jnp.ndarray:
        """Nonlinear operator ``A(x)`` for the Phi1 system (residual assembly).

        Element-wise counterpart of
        ``apply_v3_full_system_operator(..., include_jacobian_terms=False)``.
        """
        f = v[: self.f_size].reshape(self.f_shape)
        rest = v[self.f_size :]
        phi1 = rest[: self.n_theta * self.n_zeta].reshape((self.n_theta, self.n_zeta))
        lam = rest[self.n_theta * self.n_zeta]
        extra = rest[self.phi1_size :]

        # includePhi1InCollisionOperator: the collision operator uses the current
        # Phi1 field (residual-mode ``whichMatrix=3``); the autodiff of this
        # residual then supplies the exact Jacobian coupling.
        y_f = self.apply_f(f, phi1_hat=phi1)
        y_phi1 = self._quasineutrality_rows(f, phi1, lam)
        if self.include_phi1_in_kinetic:
            y_f = self._add_phi1_in_kinetic(y_f, f, phi1)

        factor = self._fs_average_factor()
        ix0 = _ix_min(self.point_at_x0)
        if self.constraint_scheme == 0:
            y_extra = jnp.zeros((0,), dtype=jnp.float64)
        elif self.constraint_scheme in (1, 3, 4):
            src = extra.reshape((self.n_species, 2))
            xpart1, xpart2 = self._source_basis(self.constraint_scheme)
            y_f = y_f.at[:, ix0:, 0, :, :].add(
                xpart1[ix0:][None, :, None, None] * src[:, 0, None, None, None]
                + xpart2[ix0:][None, :, None, None] * src[:, 1, None, None, None]
            )
            x2 = self.x * self.x
            w2 = x2 * self.x_weights
            w4 = x2 * x2 * self.x_weights
            y_dens = jnp.einsum("x,tz,sxtz->s", w2, factor, f[:, :, 0, :, :])
            y_pres = jnp.einsum("x,tz,sxtz->s", w4, factor, f[:, :, 0, :, :])
            y_extra = jnp.stack([y_dens, y_pres], axis=1).reshape((-1,))
        elif self.constraint_scheme == 2:
            src = extra.reshape((self.n_species, self.n_x))
            y_f = y_f.at[:, ix0:, 0, :, :].add(src[:, ix0:, None, None])
            y_avg = jnp.einsum("tz,sxtz->sx", factor, f[:, :, 0, :, :])
            if self.point_at_x0:
                y_avg = y_avg.at[:, 0].set(src[:, 0])
            y_extra = y_avg.reshape((-1,))
        else:
            raise NotImplementedError(f"constraintScheme={self.constraint_scheme} is not supported.")

        return jnp.concatenate([y_f.reshape((-1,)), y_phi1, y_extra], axis=0)

    def rhs_phi1(self) -> jnp.ndarray:
        """Inhomogeneous drive of the Phi1 system (evaluateResidual.F90, f=0).

        The radial-gradient / inductive drive is multiplied by
        ``exp(-Z*alpha*Phi1Hat/THat)`` (and carries the extra
        ``Phi1Hat*dTHat/dpsiHat`` piece) when Phi1 enters the kinetic equation;
        quasineutralityOption 1 additionally drives the QN rows with the
        nonlinear Boltzmann response ``-sum_s Z_s n_s exp(...)``.  Depends on
        ``phi1_hat_base`` (the linearization point).
        """
        phi1 = self.phi1_hat_base
        if phi1 is None:
            phi1 = jnp.zeros((self.n_theta, self.n_zeta), dtype=jnp.float64)

        f_rhs = jnp.zeros(self.f_shape, dtype=jnp.float64)
        ix0 = _ix_min(self.point_at_x0)
        x2 = self.x * self.x
        expx2 = jnp.exp(-x2)
        x2_expx2 = x2 * expx2
        sqrt_pi = jnp.sqrt(jnp.pi)
        two_pi = jnp.asarray(2.0 * jnp.pi, dtype=jnp.float64)

        geom2 = (
            (self.b_hat_sub_zeta * self.db_hat_dtheta - self.b_hat_sub_theta * self.db_hat_dzeta)
            * self.d_hat
            / self.b_hat**3
        )  # (T,Z)
        mask_x = (jnp.arange(self.n_x) >= ix0).astype(jnp.float64)

        x_part = x2_expx2[None, :] * (
            (self.dn_hat_dpsi_hat / self.n_hat)[:, None]
            + (self.alpha * self.z_s / self.t_hat)[:, None] * self.dphi_hat_dpsi_hat
            + (x2[None, :] - 1.5) * (self.dt_hat_dpsi_hat / self.t_hat)[:, None]
        )  # (S,X)

        if self.include_phi1_in_kinetic:
            x_part2 = x2_expx2[None, :] * (self.dt_hat_dpsi_hat / (self.t_hat * self.t_hat))[:, None]
            exp_phi1 = jnp.exp(-(self.z_s[:, None, None] * self.alpha / self.t_hat[:, None, None]) * phi1[None, :, :])
            x_part_total = x_part[:, :, None, None] + (
                x_part2[:, :, None, None] * (self.z_s * self.alpha)[:, None, None, None] * phi1[None, None, :, :]
            )
            x_part_total = x_part_total * exp_phi1[:, None, :, :]  # (S,X,T,Z)
        else:
            x_part_total = x_part[:, :, None, None]  # (S,X,1,1)

        pref = self.delta * self.n_hat * self.m_hat * jnp.sqrt(self.m_hat) / (
            two_pi * sqrt_pi * self.z_s * jnp.sqrt(self.t_hat)
        )  # (S,)
        factor = pref[:, None, None, None] * geom2[None, None, :, :] * x_part_total
        factor = factor * mask_x[None, :, None, None]

        if self.n_xi > 0:
            mask_l0 = (self.n_xi_for_x > 0).astype(jnp.float64) * mask_x
            f_rhs = f_rhs.at[:, :, 0, :, :].add((4.0 / 3.0) * factor * mask_l0[None, :, None, None])
        if self.n_xi > 2:
            mask_l2 = (self.n_xi_for_x > 2).astype(jnp.float64) * mask_x
            f_rhs = f_rhs.at[:, :, 2, :, :].add((2.0 / 3.0) * factor * mask_l2[None, :, None, None])

        if self.n_xi > 1:
            epar = self.e_parallel_hat + self.e_parallel_hat_spec  # (S,)
            factor_e = (
                self.alpha
                * self.z_s[:, None]
                * self.x[None, :]
                * expx2[None, :]
                * epar[:, None]
                * self.n_hat[:, None]
                * self.m_hat[:, None]
                / (jnp.pi * sqrt_pi * (self.t_hat * self.t_hat)[:, None] * self.fsab_hat2)
            )  # (S,X)
            factor_e = factor_e * mask_x[None, :]
            f_rhs = f_rhs.at[:, :, 1, :, :].add(factor_e[:, :, None, None] * self.b_hat[None, None, :, :])

        rhs_phi1 = jnp.zeros((self.phi1_size,), dtype=jnp.float64)
        if int(self.quasineutrality_option) == 1:
            exp_phi = jnp.exp(-(self.z_s[:, None, None] * self.alpha / self.t_hat[:, None, None]) * phi1[None, :, :])
            qn_nonlin = -jnp.sum((self.z_s * self.n_hat)[:, None, None] * exp_phi, axis=0)
            if self.with_adiabatic:
                qn_nonlin = qn_nonlin - self.adiabatic_z * self.adiabatic_n_hat * jnp.exp(
                    -(self.adiabatic_z * self.alpha / self.adiabatic_t_hat) * phi1
                )
            rhs_phi1 = jnp.concatenate([qn_nonlin.reshape((-1,)), jnp.asarray([0.0], dtype=jnp.float64)], axis=0)

        rhs_extra = jnp.zeros((self.extra_size,), dtype=jnp.float64)
        return jnp.concatenate([f_rhs.reshape((-1,)), rhs_phi1, rhs_extra], axis=0)

    def residual_phi1(self, x_full: jnp.ndarray) -> jnp.ndarray:
        """Nonlinear residual ``A(x) - b(x)`` of the Phi1 system.

        Relinearizes ``phi1_hat_base`` at the current Phi1 field (v3 ``SNES``
        residual evaluation); element-wise comparable to
        ``operators.profile_system.residual_v3_full_system``.
        """
        x_full = jnp.asarray(x_full, dtype=jnp.float64)
        phi1 = x_full[self.f_size : self.f_size + self.n_theta * self.n_zeta].reshape(
            (self.n_theta, self.n_zeta)
        )
        op_use = replace(self, phi1_hat_base=phi1)
        return op_use._apply_phi1_operator(x_full) - op_use.rhs_phi1()

    # ------------------------------------------------------------------
    # readExternalPhi1: fixed external Phi1 field, LINEAR f-only system
    #
    # State layout = f-only (like a non-Phi1 run): [ f | sources ] with NO
    # quasineutrality block, NO Phi1 unknown, NO lambda row (indices.F90 errors
    # out if those exist with readExternalPhi1).  The external Phi1 enters the
    # SAME Phi1-in-kinetic / Phi1-in-collision terms as the self-consistent path,
    # only evaluated at a given field.  Because the residual on the f + source
    # rows is affine in [f | sources] at fixed Phi1, ``apply`` is its linear part
    # (source injection + moment rows around the Phi1-shifted f-block) and ``rhs``
    # is the constant part negated (drive minus the Phi1-only source term).
    # ------------------------------------------------------------------

    def _apply_external_phi1(self, v: jnp.ndarray) -> jnp.ndarray:
        """Linear f-only operator with the fixed external Phi1 (readExternalPhi1)."""
        phi1 = self.external_phi1_hat
        f = v[: self.f_size].reshape(self.f_shape)
        extra = v[self.f_size :]
        y_f = self.apply_f(f, phi1_hat=phi1)
        if self.include_phi1_in_kinetic:
            y_f = y_f + self._phi1_in_kinetic_flinear(f, phi1)
        y_f, y_extra = self._source_and_constraint_rows(y_f, f, extra)
        return jnp.concatenate([y_f.reshape((-1,)), y_extra], axis=0)

    def _rhs_external_phi1(self) -> jnp.ndarray:
        """RHS drive of the fixed-external-Phi1 system (drive minus the Phi1-only term).

        The gradient / inductive drive is the ``rhs_phi1`` f-block evaluated at the
        external field (exp(-Z*alpha*Phi1Hat/THat) factors included when Phi1 is in
        the kinetic equation); the Phi1-only :meth:`_phi1_in_kinetic_source` term is
        f-independent so it moves from the operator to the right-hand side.
        """
        phi1 = self.external_phi1_hat
        # ``rhs_phi1`` needs the include_phi1 layout to assemble its f-block drive;
        # take that block (unchanged by the QN option) evaluated at the external field.
        aux = replace(self, include_phi1=True, phi1_hat_base=phi1, external_phi1_hat=None)
        f_rhs = aux.rhs_phi1()[: self.f_size].reshape(self.f_shape)
        if self.include_phi1_in_kinetic:
            f_rhs = f_rhs.at[:, :, 0, :, :].add(-self._phi1_in_kinetic_source(phi1))
        rhs_extra = jnp.zeros((self.extra_size,), dtype=jnp.float64)
        return jnp.concatenate([f_rhs.reshape((-1,)), rhs_extra], axis=0)

    # ------------------------------------------------------------------
    # RHS drives (evaluateResidual.F90 with f = 0)
    # ------------------------------------------------------------------

    def _with_rhs_settings(self, which_rhs: int | None) -> "KineticOperator":
        """Apply v3's internal whichRHS gradient/E_parallel overwrites.

        RHSMode=2: whichRHS 1..3 (dn drive, dT drive at fixed pressure drive,
        E_parallel drive); RHSMode=3: whichRHS 1..2 (radial drive, parallel
        drive).  Matches ``solver.F90``'s overwrites before evaluateResidual.
        """
        if which_rhs is None:
            return self
        w = int(which_rhs)
        if self.rhs_mode == 3:
            if w == 1:
                dn, dt, epar = jnp.ones_like(self.dn_hat_dpsi_hat), jnp.zeros_like(self.dt_hat_dpsi_hat), 0.0
            elif w == 2:
                dn, dt, epar = jnp.zeros_like(self.dn_hat_dpsi_hat), jnp.zeros_like(self.dt_hat_dpsi_hat), 1.0
            else:
                raise ValueError("RHSMode=3 expects which_rhs in {1,2}.")
        elif self.rhs_mode == 2:
            if w == 1:
                dn, dt, epar = jnp.ones_like(self.dn_hat_dpsi_hat), jnp.zeros_like(self.dt_hat_dpsi_hat), 0.0
            elif w == 2:
                # (1/n) dn/dpsi + (3/2) dT/dpsi = 0 with dT/dpsi = 1.
                dn_val = 1.5 * self.n_hat[0] * self.t_hat[0]
                dn = jnp.broadcast_to(dn_val, self.dn_hat_dpsi_hat.shape)
                dt, epar = jnp.ones_like(self.dt_hat_dpsi_hat), 0.0
            elif w == 3:
                dn, dt, epar = jnp.zeros_like(self.dn_hat_dpsi_hat), jnp.zeros_like(self.dt_hat_dpsi_hat), 1.0
            else:
                raise ValueError("RHSMode=2 expects which_rhs in {1,2,3}.")
        else:
            if w != 1:
                raise ValueError("RHSMode=1 has a single RHS (which_rhs=1 or None).")
            return self
        return replace(
            self,
            dn_hat_dpsi_hat=dn,
            dt_hat_dpsi_hat=dt,
            e_parallel_hat=jnp.asarray(epar, dtype=jnp.float64),
        )

    def rhs(self, which_rhs: int | None = None) -> jnp.ndarray:
        """Assemble the v3 RHS vector (evaluateResidual.F90 drives, f-independent).

        - the ``dot(psi) dfM/dpsi`` gradient drive (L=0 with weight 4/3 and L=2
          with weight 2/3);
        - the inductive ``EParallelHat`` drive (L=1);
        - zeros on the constraint rows.

        ``which_rhs`` selects the RHSMode 2/3 transport-matrix drive column.
        """
        if self.external_phi1_hat is not None:
            # readExternalPhi1 is a single-RHS (RHSMode=1) linear system whose drive
            # is evaluated at the fixed external Phi1 field.
            return self._rhs_external_phi1()
        op = self._with_rhs_settings(which_rhs)

        f_rhs = jnp.zeros(op.f_shape, dtype=jnp.float64)
        ix0 = _ix_min(op.point_at_x0)
        x2 = op.x * op.x
        expx2 = jnp.exp(-x2)
        x2_expx2 = x2 * expx2
        sqrt_pi = jnp.sqrt(jnp.pi)
        two_pi = jnp.asarray(2.0 * jnp.pi, dtype=jnp.float64)

        # For RHSMode 2/3 the electrostatic term is excluded from the drive.
        dphi_to_use = jnp.where(
            (op.rhs_mode == 1) | (op.rhs_mode > 3),
            op.dphi_hat_dpsi_hat,
            jnp.asarray(0.0, dtype=jnp.float64),
        )

        geom2 = (
            (op.b_hat_sub_zeta * op.db_hat_dtheta - op.b_hat_sub_theta * op.db_hat_dzeta)
            * op.d_hat
            / op.b_hat**3
        )  # (T,Z)

        mask_x = (jnp.arange(op.n_x) >= ix0).astype(jnp.float64)

        x_part = x2_expx2[None, :] * (
            (op.dn_hat_dpsi_hat / op.n_hat)[:, None]
            + (op.alpha * op.z_s / op.t_hat)[:, None] * dphi_to_use
            + (x2[None, :] - 1.5) * (op.dt_hat_dpsi_hat / op.t_hat)[:, None]
        )  # (S,X)

        pref = (
            op.delta
            * op.n_hat
            * op.m_hat
            * jnp.sqrt(op.m_hat)
            / (two_pi * sqrt_pi * op.z_s * jnp.sqrt(op.t_hat))
        )  # (S,)

        factor = pref[:, None, None, None] * geom2[None, None, :, :] * x_part[:, :, None, None]
        factor = factor * mask_x[None, :, None, None]

        if op.n_xi > 0:
            mask_l0 = (op.n_xi_for_x > 0).astype(jnp.float64) * mask_x
            f_rhs = f_rhs.at[:, :, 0, :, :].add((4.0 / 3.0) * factor * mask_l0[None, :, None, None])
        if op.n_xi > 2:
            mask_l2 = (op.n_xi_for_x > 2).astype(jnp.float64) * mask_x
            f_rhs = f_rhs.at[:, :, 2, :, :].add((2.0 / 3.0) * factor * mask_l2[None, :, None, None])

        if op.n_xi > 1:
            epar = op.e_parallel_hat + op.e_parallel_hat_spec  # (S,)
            factor_e = (
                op.alpha
                * op.z_s[:, None]
                * op.x[None, :]
                * expx2[None, :]
                * epar[:, None]
                * op.n_hat[:, None]
                * op.m_hat[:, None]
                / (jnp.pi * sqrt_pi * (op.t_hat * op.t_hat)[:, None] * op.fsab_hat2)
            )  # (S,X)
            factor_e = factor_e * mask_x[None, :]
            f_rhs = f_rhs.at[:, :, 1, :, :].add(factor_e[:, :, None, None] * op.b_hat[None, None, :, :])

        rhs_extra = jnp.zeros((op.extra_size,), dtype=jnp.float64)
        return jnp.concatenate([f_rhs.reshape((-1,)), rhs_extra], axis=0)

    # ------------------------------------------------------------------
    # analytic Legendre-block extraction (probing-free)
    # ------------------------------------------------------------------

    def _check_block_extraction_supported(self) -> None:
        if self.with_er_xidot or self.with_er_xdot:
            raise NotImplementedError(
                "legendre_blocks requires DKES trajectories: the Er xiDot/xDot terms "
                "couple L±2 and break the block-tridiagonal-in-L structure."
            )
        # NOTE: point_at_x0 grids are accepted here because the blocks also
        # serve as the Krylov coarse preconditioner, where approximating the
        # x=0 boundary rows (f=0 for L>0, the df/dx=0 regularity row for L=0)
        # by the plain DKE blocks is fine.  The exact structured direct routes
        # additionally refuse point_at_x0 in dkx.solve.
        if self.with_magnetic_drifts:
            raise NotImplementedError(
                "legendre_blocks does not support tangential magnetic drifts: the "
                "magneticDriftScheme d/dtheta, d/dzeta, and d/dxi terms couple L±2 "
                "and break the block-tridiagonal-in-L structure (the recycled "
                "Krylov route owns these decks)."
            )
        if self.fp is not None or self.fp_phi1 is not None or self.sugama is not None:
            raise NotImplementedError(
                "legendre_blocks currently supports pitch-angle-scattering collisions only; "
                "Fokker-Planck and the improved Sugama model operator couple (species, x) "
                "densely within each L (their per-L blocks live in KineticOperator.fp.mat / "
                "KineticOperator.sugama.mat). The coarse preconditioner reduces them "
                "to their PAS-like self-species x-diagonal in dkx.solve."
            )
        if self.external_phi1_hat is not None and self.include_phi1_in_kinetic:
            raise NotImplementedError(
                "legendre_blocks does not support the readExternalPhi1 Phi1-in-kinetic "
                "coupling: the fixed-external-Phi1 speed-derivative term couples x densely "
                "(the recycled Krylov route owns these decks)."
            )

    def legendre_blocks(self, ell: int) -> LegendreBlocks:
        """Dense (theta*zeta) blocks of Legendre row ``ell`` of the f-block.

        Built analytically from the term coefficients — no operator probing:
        streaming/mirror provide ``lower``/``upper`` (L±1 with the phase_space
        coupling factors), ExB and pitch-angle scattering provide ``diag``.
        Supported for the DKES-trajectory PAS family (the structured direct
        solver family of plan §2.3); raises otherwise.

        Returns blocks of shape ``(n_species, n_x, T*Z, T*Z)``; the truncated-L
        masks are baked in, so the block-tridiagonal matvec reproduces
        :meth:`apply_f` exactly.
        """
        self._check_block_extraction_supported()
        if not (0 <= int(ell) < self.n_xi):
            raise ValueError(f"ell must be in [0, {self.n_xi}), got {ell}")
        ell = int(ell)

        n_tz = self.n_theta * self.n_zeta
        eye_t = jnp.eye(self.n_theta, dtype=jnp.float64)
        eye_z = jnp.eye(self.n_zeta, dtype=jnp.float64)
        d_theta_tz = jnp.kron(self.ddtheta, eye_z)  # (TZ,TZ)
        d_zeta_tz = jnp.kron(eye_t, self.ddzeta)  # (TZ,TZ)

        # Keep the mask a jnp array (no host materialization) so the block
        # extraction stays traceable when the operator leaves are tracers
        # (jit-over-leaves / vmap).  ``ell`` is a static Python int, so the
        # column slices below are static-index selects on a traced (X,L) array.
        mask = self._mask()  # (X,L)
        row_mask = mask[:, ell]  # (X,)

        sqrt_t_over_m = jnp.sqrt(self.t_hat / self.m_hat)  # (S,)
        v_theta = (self.b_hat_sup_theta / self.b_hat).reshape((-1,))  # (TZ,)
        v_zeta = (self.b_hat_sup_zeta / self.b_hat).reshape((-1,))
        # Streaming operator per species: rowscale(v)·D, summed over both angles.
        stream_tz = (
            sqrt_t_over_m[:, None, None]
            * (v_theta[None, :, None] * d_theta_tz[None, :, :] + v_zeta[None, :, None] * d_zeta_tz[None, :, :])
        )  # (S,TZ,TZ)

        mirror_geom = self.b_hat_sup_theta * self.db_hat_dtheta + self.b_hat_sup_zeta * self.db_hat_dzeta
        mirror_diag = (
            -sqrt_t_over_m[:, None] * (mirror_geom / (2.0 * self.b_hat**2)).reshape((-1,))[None, :]
        )  # (S,TZ)
        mirror_tz = mirror_diag[:, :, None] * jnp.eye(n_tz, dtype=jnp.float64)[None, :, :]  # (S,TZ,TZ)

        def _shaped(block_s: jnp.ndarray, x_factor: jnp.ndarray, col_mask: jnp.ndarray) -> jnp.ndarray:
            # block_s (S,TZ,TZ); x_factor (X,); col_mask (X,) -> (S,X,TZ,TZ)
            scale = x_factor * row_mask * col_mask  # (X,)
            return block_s[:, None, :, :] * scale[None, :, None, None]

        # ---- lower: row ell receives column ell-1 ----
        if ell >= 1:
            coef_stream = self.xi_coupling_lower[ell]  # traced 0-d scalar (static ell)
            coef_mirror = -coef_stream * (ell - 1.0)
            col_mask = mask[:, ell - 1]
            lower = _shaped(coef_stream * stream_tz + coef_mirror * mirror_tz, self.x, col_mask)
        else:
            lower = jnp.zeros((self.n_species, self.n_x, n_tz, n_tz), dtype=jnp.float64)

        # ---- upper: row ell receives column ell+1 ----
        if ell + 1 < self.n_xi:
            coef_stream = self.xi_coupling_upper[ell]  # traced 0-d scalar (static ell)
            coef_mirror = coef_stream * (ell + 2.0)
            col_mask = mask[:, ell + 1]
            upper = _shaped(coef_stream * stream_tz + coef_mirror * mirror_tz, self.x, col_mask)
        else:
            upper = jnp.zeros((self.n_species, self.n_x, n_tz, n_tz), dtype=jnp.float64)

        # ---- diag: ExB (species-independent) + PAS (diagonal in theta/zeta) ----
        diag = jnp.zeros((self.n_species, self.n_x, n_tz, n_tz), dtype=jnp.float64)
        if self.with_exb:
            coef_theta, coef_zeta = self._exb_coefficients()
            exb_tz = (
                coef_theta.reshape((-1,))[:, None] * d_theta_tz
                + coef_zeta.reshape((-1,))[:, None] * d_zeta_tz
            )  # (TZ,TZ)
            diag = diag + exb_tz[None, None, :, :] * row_mask[None, :, None, None]
        if self.pas is not None:
            pas_coef = self.pas.coef[:, :, ell]  # (S,X)
            diag = diag + (pas_coef * row_mask[None, :])[:, :, None, None] * jnp.eye(
                n_tz, dtype=jnp.float64
            )[None, None, :, :]

        return LegendreBlocks(lower=lower, diag=diag, upper=upper)

    def to_block_tridiagonal(self) -> LegendreBlocks:
        """Stack :meth:`legendre_blocks` over all L.

        Returns arrays of shape ``(n_xi, n_species, n_x, T*Z, T*Z)`` ready for a
        block-Thomas recursion over L (SOLVAX structured direct kernel).
        """
        self._check_block_extraction_supported()
        blocks = [self.legendre_blocks(ell) for ell in range(self.n_xi)]
        return LegendreBlocks(
            lower=jnp.stack([b.lower for b in blocks]),
            diag=jnp.stack([b.diag for b in blocks]),
            upper=jnp.stack([b.upper for b in blocks]),
        )

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    @classmethod
    def from_namelist(cls, nml: Any) -> "KineticOperator":
        """Build the operator from a parsed SFINCS input namelist.

        See :func:`kinetic_operator_from_namelist` for details.
        """
        return kinetic_operator_from_namelist(nml)

# =============================================================================
# Namelist construction (readInput.F90 defaults; createGrids.F90 overrides)
# =============================================================================

def _flux_surface_averages_effective(
    *, grids: Grids, geom: FluxSurfaceGeometry
) -> tuple[float, float, float]:
    """Effective ``(B0OverBBar, GHat, IHat)`` from surface averages.

    For VMEC geometry (scheme 5) the flux functions are stored as 0.0
    placeholders and v3 recomputes them in ``computeBIntegrals``:
    ``B0OverBBar = <B^3>/<B^2>``, ``GHat = <B_sub_zeta>``, ``IHat = <B_sub_theta>``
    (the latter two averaged over 4*pi^2 with the quadrature weights).
    """
    w = np.asarray(grids.theta_weights)[:, None] * np.asarray(grids.zeta_weights)[None, :]
    d_hat = np.asarray(geom.d_hat)
    b_hat = np.asarray(geom.b_hat)
    vprime = float(np.sum(w / d_hat))
    fsab2 = float(np.sum(w * b_hat**2 / d_hat) / vprime)
    b0_eff = float(np.sum(w * b_hat**3 / d_hat) / (vprime * fsab2))
    denom = 4.0 * math.pi * math.pi
    g_eff = float(np.sum(w * np.asarray(geom.b_hat_sub_zeta)) / denom)
    i_eff = float(np.sum(w * np.asarray(geom.b_hat_sub_theta)) / denom)
    return b0_eff, g_eff, i_eff

def _geometry_and_radial(
    *, nml: Any, grids: Grids, compute_grad_psi_dot_grad_b: bool = False
) -> tuple[FluxSurfaceGeometry, RadialCoordinates]:
    """Geometry + radial-coordinate conversions for the supported geometrySchemes.

    ``compute_grad_psi_dot_grad_b`` requests the Sugama normal-curvature factor
    ``gradpsidotgradB_overgpsipsi`` (magneticDriftScheme 5/6), available for the
    file-based geometries (5/11/12) only.
    """
    geom_params = nml.group("geometryParameters")
    phys = nml.group("physicsParameters")
    scheme = _get_int(geom_params, "geometryScheme", -1)

    if scheme == 1:
        psi_a_hat = effective_psi_a_hat(geom_params=geom_params, phys_params=phys, default=0.15596)
        a_hat = _get_float(geom_params, "aHat", 0.5585)
        psi_n_wish = effective_psi_n_wish(
            geom_params=geom_params, default_r_n=0.5, psi_a_hat=psi_a_hat, a_hat=a_hat
        )
        r_n = math.sqrt(float(psi_n_wish))
        geom = FluxSurfaceGeometry.from_scheme(
            1,
            theta=grids.theta,
            zeta=grids.zeta,
            epsilon_t=_get_float(geom_params, "epsilon_t", -0.07053),
            epsilon_h=_get_float(geom_params, "epsilon_h", 0.05067),
            epsilon_antisymm=_get_float(geom_params, "epsilon_antisymm", 0.0),
            iota=_get_float(geom_params, "iota", 0.4542),
            g_hat=_get_float(geom_params, "GHat", 3.7481),
            i_hat=_get_float(geom_params, "IHat", 0.0),
            b0_over_bbar=_get_float(geom_params, "B0OverBBar", 1.0),
            helicity_l=_get_int(geom_params, "helicity_l", 2),
            helicity_n=_get_int(geom_params, "helicity_n", 10),
            helicity_antisymm_l=_get_int(geom_params, "helicity_antisymm_l", 1),
            helicity_antisymm_n=_get_int(geom_params, "helicity_antisymm_n", 0),
        )
        return geom, RadialCoordinates(psi_a_hat=float(psi_a_hat), a_hat=float(a_hat), r_n=r_n)

    if scheme == 2:
        # v3 fixed simplified LHD model; rN forced to 0.5.
        a_hat = 0.5585
        return (
            FluxSurfaceGeometry.from_scheme(2, theta=grids.theta, zeta=grids.zeta),
            RadialCoordinates(psi_a_hat=(a_hat * a_hat) / 2.0, a_hat=a_hat, r_n=0.5),
        )

    if scheme == 3:
        # v3 fixed LHD inward-shifted analytic Boozer model (geometry.F90 case 3);
        # aHat=0.5400, psiAHat=aHat^2/2, rN forced to 0.5 (rN_wish ignored).  The
        # B-field harmonics/flux functions come from from_scheme(3).
        a_hat = 0.5400
        return (
            FluxSurfaceGeometry.from_scheme(3, theta=grids.theta, zeta=grids.zeta),
            RadialCoordinates(psi_a_hat=(a_hat * a_hat) / 2.0, a_hat=a_hat, r_n=0.5),
        )

    if scheme == 4:
        # v3 built-in W7-X standard model; rN forced to 0.5.
        return (
            FluxSurfaceGeometry.from_scheme(4, theta=grids.theta, zeta=grids.zeta),
            RadialCoordinates(psi_a_hat=-0.384935, a_hat=0.5109, r_n=0.5),
        )

    if scheme in {11, 12}:
        path = _resolve_equilibrium_path(nml=nml, geom_params=geom_params)
        header, _surfaces = read_boozer_bc(path, geometry_scheme=scheme)
        psi_n_wish = effective_psi_n_wish(geom_params=geom_params, default_r_n=0.5)
        r_n_wish = math.sqrt(float(psi_n_wish))
        vmec_radial_option = _get_int(geom_params, "VMECRadialOption", 1)
        r_n = float(
            selected_r_n_from_bc(
                path=str(path),
                geometry_scheme=scheme,
                r_n_wish=r_n_wish,
                vmec_radial_option=vmec_radial_option,
            )
        )
        geom = FluxSurfaceGeometry.from_boozer(
            path,
            theta=grids.theta,
            zeta=grids.zeta,
            r_n_wish=r_n_wish,
            vmec_radial_option=vmec_radial_option,
            geometry_scheme=scheme,
            compute_grad_psi_dot_grad_b=compute_grad_psi_dot_grad_b,
        )
        return geom, RadialCoordinates(psi_a_hat=float(header.psi_a_hat), a_hat=float(header.a_hat), r_n=r_n)

    if scheme == 5:
        path = _resolve_equilibrium_path(nml=nml, geom_params=geom_params, vmec=True)
        w = read_vmec_wout(path)
        psi_a_hat = float(psi_a_hat_from_wout(w))
        a_hat = float(w.aminor_p)
        psi_n_wish = effective_psi_n_wish(
            geom_params=geom_params, default_r_n=0.5, psi_a_hat=psi_a_hat, a_hat=a_hat
        )
        vmec_radial_option = _get_int(geom_params, "VMECRadialOption", 1)
        interp = vmec_radial_interpolation(
            w=w, psi_n_wish=float(psi_n_wish), vmec_radial_option=vmec_radial_option
        )
        r_n = float(interp.psi_n) ** 0.5
        geom = FluxSurfaceGeometry.from_vmec(
            w,
            theta=grids.theta,
            zeta=grids.zeta,
            psi_n_wish=float(psi_n_wish),
            vmec_radial_option=vmec_radial_option,
            vmec_nyquist_option=_get_int(geom_params, "VMEC_NYQUIST_OPTION", 1),
            min_bmn_to_load=_get_float(geom_params, "MIN_BMN_TO_LOAD", 0.0),
            ripple_scale=_get_float(geom_params, "RIPPLESCALE", 1.0),
            helicity_n=_get_int(geom_params, "HELICITY_N", 0),
            helicity_l=_get_int(geom_params, "HELICITY_L", 0),
            compute_grad_psi_dot_grad_b=compute_grad_psi_dot_grad_b,
        )
        return geom, RadialCoordinates(psi_a_hat=psi_a_hat, a_hat=a_hat, r_n=r_n)

    if scheme == 13:
        # Namelist Boozer |B| spectrum (geometry.F90 case 13, the STELLOPT/BMNC
        # optimization path).  ``boozer_bmnc(m,n)`` / ``boozer_bmns(m,n)`` are 2-D
        # indexed arrays in geometryParameters; NPeriods, psiAHat, aHat, iota,
        # GHat, IHat are read from the namelist.  The field is analytic
        # (nearbyRadiiGiven=.false., radial derivatives 0, rN = rN_wish), so the
        # radial coordinate mirrors the scheme-1 analytic pattern.
        psi_a_hat = effective_psi_a_hat(geom_params=geom_params, phys_params=phys, default=0.15596)
        a_hat = _get_float(geom_params, "aHat", 0.5585)
        psi_n_wish = effective_psi_n_wish(
            geom_params=geom_params, default_r_n=0.5, psi_a_hat=psi_a_hat, a_hat=a_hat
        )
        r_n = math.sqrt(float(psi_n_wish))
        idx = nml.indexed.get("geometryparameters", {})
        bmnc_map = idx.get("BOOZER_BMNC", {})
        bmns_map = idx.get("BOOZER_BMNS", {})
        if not bmnc_map and not bmns_map:
            raise ValueError(
                "geometryScheme=13 requires at least one boozer_bmnc/boozer_bmns amplitude "
                "in geometryParameters (validateInput.F90)."
            )
        # from_fourier takes 1-D (amp, m, n) arrays with the (0,0) mode included
        # (it extracts B0OverBBar = bmnc(0,0) internally).  When a sine spectrum is
        # present, align it to the same (m,n) order as the cosine spectrum by
        # unioning the mode keys and zero-filling both sides.
        if bmns_map:
            keys = sorted(set(bmnc_map) | set(bmns_map))
            bmnc = jnp.asarray([float(bmnc_map.get(k, 0.0)) for k in keys], dtype=jnp.float64)
            bmns = jnp.asarray([float(bmns_map.get(k, 0.0)) for k in keys], dtype=jnp.float64)
        else:
            keys = sorted(bmnc_map)
            bmnc = jnp.asarray([float(bmnc_map[k]) for k in keys], dtype=jnp.float64)
            bmns = None
        m_arr = jnp.asarray([int(k[0]) for k in keys])
        n_arr = jnp.asarray([int(k[1]) for k in keys])
        geom = FluxSurfaceGeometry.from_fourier(
            theta=grids.theta,
            zeta=grids.zeta,
            bmnc=bmnc,
            m=m_arr,
            n=n_arr,
            bmns=bmns,
            n_periods=max(1, _get_int(geom_params, "Nperiods", 1)),
            iota=_get_float(geom_params, "iota", 0.4542),
            g_hat=_get_float(geom_params, "GHat", 3.7481),
            i_hat=_get_float(geom_params, "IHat", 0.0),
        )
        return geom, RadialCoordinates(psi_a_hat=float(psi_a_hat), a_hat=float(a_hat), r_n=r_n)

    raise NotImplementedError(
        f"KineticOperator.from_namelist supports geometryScheme in {{1,2,3,4,5,11,12,13}}; got {scheme}."
    )

def _resolve_equilibrium_path(*, nml: Any, geom_params: dict, vmec: bool = False) -> Path:
    eq = effective_equilibrium_file(geom_params=geom_params)
    if eq is None:
        raise ValueError("This geometryScheme requires equilibriumFile in geometryParameters.")
    base_dir = nml.source_path.parent if nml.source_path is not None else None
    repo_root = repository_root()
    extra = (
        (repo_root / "tests" / "ref", repo_root / "src" / "dkx" / "data" / "equilibria")
        if repo_root is not None
        else ()
    )
    try:
        return resolve_existing_path(str(eq), base_dir=base_dir, extra_search_dirs=extra).path
    except FileNotFoundError:
        if vmec:
            # Allow `.txt -> .nc` fallback for VMEC wout files (matches the old reader).
            p2 = Path(str(eq).strip().strip('"').strip("'")).with_suffix(".nc")
            return resolve_existing_path(str(p2), base_dir=base_dir, extra_search_dirs=extra).path
        raise

def _n_periods_from_namelist(*, nml: Any) -> int:
    """NPeriods for grid construction (createGrids.F90 / geometry.F90)."""
    geom_params = nml.group("geometryParameters")
    scheme = _get_int(geom_params, "geometryScheme", -1)
    if scheme == 1:
        return max(1, _get_int(geom_params, "helicity_n", 10))
    if scheme in {2, 3}:
        return 10
    if scheme == 4:
        return 5
    if scheme in {11, 12}:
        path = _resolve_equilibrium_path(nml=nml, geom_params=geom_params)
        header, _ = read_boozer_bc(path, geometry_scheme=scheme)
        return int(header.n_periods)
    if scheme == 5:
        path = _resolve_equilibrium_path(nml=nml, geom_params=geom_params, vmec=True)
        return int(read_vmec_wout(path).nfp)
    if scheme == 13:
        # geometry.F90 case 13: NPeriods is read from the namelist.
        return max(1, _get_int(geom_params, "Nperiods", 1))
    raise NotImplementedError(
        f"KineticOperator.from_namelist supports geometryScheme in {{1,2,3,4,5,11,12,13}}; got {scheme}."
    )

def _load_external_phi1(*, nml: Any, phys: dict, grids: Grids) -> jnp.ndarray:
    """Read the FIXED external Phi1(theta,zeta) field for ``readExternalPhi1``.

    Reuses :func:`dkx.io.read_sfincs_h5` (the ``sfincsOutput.h5`` reader) on
    ``externalPhi1Filename`` (default ``externalPhi1.h5``, resolved relative to the
    input deck's directory), takes the LAST ``NIterations`` slice, and returns it
    as a ``(Ntheta, Nzeta)`` array (readHDF5Input.F90:264-320; SFINCS stores
    ``Phi1Hat`` as ``(Nzeta, Ntheta, NIterations)``).  The external (theta, zeta)
    grid must equal the run grid; external-to-run grid interpolation is a
    documented follow-up.
    """
    from dkx.io import read_sfincs_h5  # noqa: PLC0415

    raw = phys.get("EXTERNALPHI1FILENAME", "externalPhi1.h5")
    if isinstance(raw, list):
        raw = raw[0] if raw else "externalPhi1.h5"
    filename = str(raw).strip().strip('"').strip("'")
    base_dir = nml.source_path.parent if nml.source_path is not None else None
    repo_root = repository_root()
    extra = (repo_root / "tests" / "ref",) if repo_root is not None else ()
    path = resolve_existing_path(filename, base_dir=base_dir, extra_search_dirs=extra).path

    data = read_sfincs_h5(path)
    if "Phi1Hat" not in data:
        raise ValueError(f"external Phi1 file {path} has no Phi1Hat dataset.")
    phi1 = np.asarray(data["Phi1Hat"], dtype=np.float64)
    if phi1.ndim == 3:  # (Nzeta, Ntheta, NIterations) -> last iteration
        phi1 = phi1[..., -1]
    phi1 = phi1.T  # (Ntheta, Nzeta)
    if phi1.shape != (grids.n_theta, grids.n_zeta):
        raise ValueError(
            f"external Phi1Hat has shape {phi1.shape}; run grid is "
            f"{(grids.n_theta, grids.n_zeta)}.  External-to-run grid interpolation is a "
            "documented follow-up; make the external grid equal the run grid."
        )
    for name, ext_key, run in (("theta", "theta", grids.theta), ("zeta", "zeta", grids.zeta)):
        if ext_key in data and not np.allclose(
            np.asarray(data[ext_key], dtype=np.float64), np.asarray(run, dtype=np.float64), rtol=0.0, atol=1e-10
        ):
            raise NotImplementedError(
                f"external Phi1 {name} grid differs from the run grid; external-to-run grid "
                "interpolation is a documented follow-up (make the external grid equal the run grid)."
            )
    return jnp.asarray(phi1, dtype=jnp.float64)

class KineticOperatorBuild(NamedTuple):
    """One operator build together with the grids/geometry it was derived from.

    The single-build context of :func:`kinetic_operator_build_from_namelist`:
    downstream consumers (the run drivers, the writer, diagnostics) share these
    objects instead of re-deriving them from the namelist.

    Attributes:
        operator: the assembled :class:`KineticOperator`.
        grids: the :class:`dkx.phase_space.Grids` the operator uses
            (theta/zeta nodes and quadrature weights are not stored on the
            operator itself).
        geometry: the full :class:`dkx.magnetic_geometry.FluxSurfaceGeometry`
            (radial-derivative fields included).
        radial: the :class:`dkx.constants.RadialCoordinates` conversions
            for the selected flux surface.
    """

    operator: KineticOperator
    grids: Grids
    geometry: FluxSurfaceGeometry
    radial: RadialCoordinates

def kinetic_operator_from_namelist(nml: Any) -> KineticOperator:
    """Build a :class:`KineticOperator` from a parsed SFINCS input namelist.

    Thin facade over :func:`kinetic_operator_build_from_namelist` returning the
    operator only; callers that also need the grids, geometry, or radial
    conversions use the build function to avoid re-deriving them.
    """
    return kinetic_operator_build_from_namelist(nml).operator

def kinetic_operator_build_from_namelist(nml: Any) -> KineticOperatorBuild:
    """Build a :class:`KineticOperator` plus its grids/geometry/radial context.

    ``nml`` is a :class:`dkx.namelist.Namelist`.  Grids come from
    :func:`dkx.phase_space.make_grids`, geometry from
    :class:`dkx.magnetic_geometry.FluxSurfaceGeometry`, species from
    :func:`dkx.species.species_set_from_namelist`, radial-coordinate
    and monoenergetic conversions from :mod:`dkx.constants`, and
    collision matrices from :mod:`dkx.collisions`.  The returned
    :class:`KineticOperatorBuild` carries the grids, flux-surface geometry, and
    radial conversions alongside the operator so one build serves the solve,
    diagnostics, and output-writing consumers.

    Supports ``geometryScheme`` in {1, 2, 3, 4, 5, 11, 12, 13} (analytic schemes
    1/2/3/4, file-based 5/11/12, and the namelist Boozer |B| spectrum 13) and
    every ``magneticDriftScheme`` 0-9 (the tangential magnetic drifts need a
    geometryScheme in {5, 11, 12} that carries the radial B-field derivatives;
    scheme 4 is geometryScheme 11/12 only, as in validateInput.F90).  Raises
    ``NotImplementedError`` for the deferred features listed in the module
    docstring (Fokker-Planck collisions on the uniform/Chebyshev x grids).
    """
    general = nml.group("general")
    phys = nml.group("physicsParameters")
    other = nml.group("otherNumericalParameters")
    res = nml.group("resolutionParameters")
    geom_params = nml.group("geometryParameters")

    # ---- Phi1 / quasineutrality configuration (includePhi1 vertical slice) ----
    species_params = nml.group("speciesParameters")
    include_phi1_input = _get_bool(phys, "includePhi1", False)
    read_external_phi1 = include_phi1_input and _get_bool(phys, "readExternalPhi1", False)
    # ``readExternalPhi1`` holds Phi1 fixed (read from an external file), so the
    # DKE stays LINEAR and the state is f-only: the operator's ``include_phi1``
    # (the QN block + Phi1 unknown + lambda row) is off, but the Phi1-in-kinetic /
    # Phi1-in-collision term coefficients still evaluate at the external field.
    include_phi1 = include_phi1_input and not read_external_phi1
    # Default TRUE, matching version3/globalVariables.F90:152 (and dkx's own
    # SfincsInput field, which already carried the right default).  Defaulting
    # it off silently dropped the Phi1-in-kinetic coupling from every deck that
    # enabled Phi1 without naming this switch, which costs O(Phi1^2): the term
    # is Phi1 times f, and f itself carries an O(Phi1) response.
    include_phi1_in_kinetic = include_phi1_input and _get_bool(
        phys, "includePhi1InKineticEquation", True
    )
    include_phi1_in_collision = include_phi1_input and _get_bool(phys, "includePhi1InCollisionOperator", False)
    if include_phi1_in_collision and not include_phi1_in_kinetic:
        raise NotImplementedError(
            "includePhi1InCollisionOperator=.true. requires includePhi1InKineticEquation=.true. "
            "(populateMatrix.F90 assembles the Phi1-in-collision densities alongside the "
            "Phi1-in-kinetic couplings)."
        )
    quasineutrality_option = _get_int(phys, "quasineutralityOption", 1)
    if include_phi1 and quasineutrality_option not in {1, 2}:
        raise NotImplementedError(
            f"quasineutralityOption={quasineutrality_option} is not supported "
            "(only 1 (full) and 2 (EUTERPE) are consolidated)."
        )
    # withAdiabatic only enters quasineutrality; for readExternalPhi1 (no QN) it is
    # physically inert but still echoed to output, so track the namelist flag.
    with_adiabatic = include_phi1_input and _get_bool(species_params, "withAdiabatic", False)
    adiabatic_z = _get_float(species_params, "adiabaticZ", 1.0)
    adiabatic_n_hat = _get_float(species_params, "adiabaticNHat", 0.0)
    adiabatic_t_hat = _get_float(species_params, "adiabaticTHat", 1.0)

    magnetic_drift_scheme = _get_int(phys, "magneticDriftScheme", 0)
    if not 0 <= magnetic_drift_scheme <= 9:
        # validateInput.F90: "magneticDriftScheme must be >= 0" / "<= 9".
        raise ValueError(f"magneticDriftScheme must be between 0 and 9 (got {magnetic_drift_scheme}).")
    with_magnetic_drifts = magnetic_drift_scheme > 0
    if with_magnetic_drifts:
        md_geometry_scheme = _get_int(geom_params, "geometryScheme", -1)
        if md_geometry_scheme not in {5, 11, 12}:
            raise NotImplementedError(
                f"magneticDriftScheme={magnetic_drift_scheme} requires a geometryScheme carrying "
                "the radial (psi) B-field derivatives; only 5 (VMEC), 11, and 12 (Boozer .bc) are "
                f"consolidated (Fortran validateInput allows 5/6/7/11/12), got {md_geometry_scheme}."
            )
        if magnetic_drift_scheme == 4 and md_geometry_scheme not in {11, 12}:
            # validateInput.F90: "magneticDriftScheme 4 has only been implemented
            # for geometryScheme 11 and 12."
            raise ValueError(
                "magneticDriftScheme 4 has only been implemented for geometryScheme 11 and 12 "
                f"(got geometryScheme={md_geometry_scheme})."
            )
        # validateInput.F90 allows geometryScheme 5/6/11/12 for schemes 5 and 6;
        # the {5, 11, 12} gate above already enforces the consolidated subset.
    collision_operator = _get_int(phys, "collisionOperator", 0)
    if collision_operator not in {0, 1, 3}:
        raise NotImplementedError(f"collisionOperator={collision_operator} is not supported.")

    rhs_mode = _get_int(general, "RHSMode", 1)
    constraint_scheme = _get_int(phys, "constraintScheme", -1)
    if constraint_scheme < 0:
        # collisionOperator=3 (improved Sugama) acts on the L=0 block through its
        # energy-diffusion + energy/particle field terms, so its speed null space
        # is {density, temperature} per species (2 sources) like the Fokker-Planck
        # operator -- not the per-speed PAS null space (constraintScheme=2).
        constraint_scheme = 1 if collision_operator in {0, 3} else 2
    if constraint_scheme not in {0, 1, 2, 3, 4}:
        raise NotImplementedError(f"constraintScheme={constraint_scheme} is not supported.")

    x_grid_scheme = _get_int(other, "xGridScheme", 5)
    if not 1 <= x_grid_scheme <= 8:
        raise ValueError(f"xGridScheme must be between 1 and 8 (got {x_grid_scheme}).")
    if collision_operator == 0 and x_grid_scheme in {3, 4, 7, 8}:
        raise NotImplementedError(
            f"collisionOperator=0 with xGridScheme={x_grid_scheme} is not supported: the "
            "Fokker-Planck Rosenbluth-potential interpolation matrices for the uniform/"
            "Chebyshev speed grids (interpolationMatrix.F90 / "
            "ChebyshevInterpolationMatrix.F90) are not ported."
        )
    x_dot_derivative_scheme = _get_int(other, "xDotDerivativeScheme", 0)
    point_at_x0 = x_grid_scheme in {2, 3, 4, 6, 7, 8}

    # ---- grids (phase_space) ----
    grids = make_grids(
        n_theta=_get_int(res, "Ntheta", 15),
        n_zeta=_get_int(res, "Nzeta", 15),
        n_xi=_get_int(res, "Nxi", 16),
        n_x=_get_int(res, "Nx", 5),
        n_l=_get_int(res, "NL", 4),
        n_periods=_n_periods_from_namelist(nml=nml),
        theta_derivative_scheme=_get_int(other, "thetaDerivativeScheme", 2),
        zeta_derivative_scheme=_get_int(other, "zetaDerivativeScheme", 2),
        magnetic_drift_derivative_scheme=_get_int(other, "magneticDriftDerivativeScheme", 3),
        x_grid_scheme=x_grid_scheme,
        x_grid_k=_get_float(other, "xGrid_k", 0.0),
        x_max=_get_float(res, "xMax", 5.0),
        x_dot_derivative_scheme=x_dot_derivative_scheme,
        n_xi_for_x_option=_get_int(other, "Nxi_for_x_option", 1),
        monoenergetic=(rhs_mode == 3),
    )

    # ---- readExternalPhi1: read the FIXED external Phi1(theta,zeta) field ----
    external_phi1_hat = _load_external_phi1(nml=nml, phys=phys, grids=grids) if read_external_phi1 else None

    # ---- geometry + radial conversions (magnetic_geometry / constants) ----
    geom, radial = _geometry_and_radial(
        nml=nml,
        grids=grids,
        compute_grad_psi_dot_grad_b=magnetic_drift_scheme in {5, 6},
    )
    fsab_hat2 = float(geom.fsab_hat2(theta_weights=grids.theta_weights, zeta_weights=grids.zeta_weights))

    # ---- species (species) ----
    species_grad_coord = infer_species_input_radial_coordinate_for_gradients(
        geom_params=geom_params, species_params=nml.group("speciesParameters"), default=4
    )
    species: SpeciesSet = species_set_from_namelist(
        nml, radial=radial, input_radial_coordinate_for_gradients=species_grad_coord
    )
    n_species = species.n_species

    alpha = _get_float(phys, "alpha", 1.0)
    delta = _get_float(phys, "Delta", DEFAULT_DELTA)
    nu_n = _get_float(phys, "nu_n", DEFAULT_NU_N)
    krook = _get_float(phys, "Krook", 0.0)
    er = _get_float(phys, "Er", 0.0)

    # ---- flux functions (effective values for VMEC placeholders) ----
    b0_eff = float(geom.b0_over_bbar)
    g_eff = float(geom.g_hat)
    i_eff = float(geom.i_hat)
    if abs(g_eff) < 1e-30 or abs(b0_eff) < 1e-30:
        b0_eff, g_eff, i_eff = _flux_surface_averages_effective(grids=grids, geom=geom)

    # ---- dPhiHat/dpsiHat: drive value (phi gradient coordinate) and kinetic value ----
    phi_grad_coord = infer_phi_input_radial_coordinate_for_gradients(
        geom_params=geom_params, phys_params=phys, default=4
    )
    if phi_grad_coord == 0:
        dphi_rhs = _get_float(phys, "dPhiHatdpsiHat", 0.0)
    elif phi_grad_coord == 1:
        dphi_rhs = radial.d_dpsi_n_to_d_dpsi_hat * _get_float(phys, "dPhiHatdpsiN", 0.0)
    elif phi_grad_coord == 2:
        dphi_rhs = radial.d_dr_hat_to_d_dpsi_hat * _get_float(phys, "dPhiHatdrHat", 0.0)
    elif phi_grad_coord == 3:
        dphi_rhs = radial.d_dr_n_to_d_dpsi_hat * _get_float(phys, "dPhiHatdrN", 0.0)
    elif phi_grad_coord == 4:
        dphi_rhs = radial.d_dr_hat_to_d_dpsi_hat * (-er)
    else:
        raise NotImplementedError(f"Unsupported inputRadialCoordinateForGradients={phi_grad_coord} for Phi.")

    if rhs_mode == 3:
        # sfincs_main.F90: EStar overwrites dPhiHatdpsiHat for monoenergetic runs.
        e_star = _get_float(phys, "EStar", 0.0)
        dphi_kinetic = d_phi_hat_d_psi_hat_from_e_star(
            e_star=e_star, alpha=alpha, delta=delta, iota=float(geom.iota), b0_over_bbar=b0_eff, g_hat=g_eff
        )
        dphi_rhs = dphi_kinetic
    else:
        # The ExB/Er kinetic terms follow the Er input (v3 examples convention).
        if er != 0.0 and phi_grad_coord != 4:
            raise NotImplementedError(
                "Er != 0 with a non-Er inputRadialCoordinateForGradients is not supported."
            )
        dphi_kinetic = radial.d_dr_hat_to_d_dpsi_hat * (-er)

    # These three defaults must match globalVariables.F90:144-146, because a
    # deck is free to omit the key and inherit it. includeXDotTerm and
    # includeElectricFieldTermInXiDot default to .true. upstream; defaulting
    # them to False here silently dropped both E_r trajectory terms from any
    # compact deck with a finite E_r, which is a wrong answer rather than a
    # missing feature. useDKESExBDrift really is .false. upstream.
    include_xdot = _get_bool(phys, "includeXDotTerm", True)
    include_er_xidot = _get_bool(phys, "includeElectricFieldTermInXiDot", True)
    use_dkes_exb = _get_bool(phys, "useDKESExBDrift", False)
    has_er = float(dphi_kinetic) != 0.0

    # ---- collisions (dkx.collisions builders) ----
    pas = None
    fp = None
    fp_phi1 = None
    sugama = None
    # &otherNumericalParameters RosenbluthMethod selects how the Fokker-Planck
    # Rosenbluth response matrices are integrated ('quadpack' for Fortran
    # parity, 'hybrid', or 'analytic').  Resolved eagerly so a mistyped deck
    # fails on any collisionOperator, not only the branches that consume it.
    rosenbluth_method = resolve_rosenbluth_method(_get_str_or_none(other, "RosenbluthMethod"))
    if include_phi1_in_collision and collision_operator != 0:
        raise NotImplementedError(
            "includePhi1InCollisionOperator=.true. requires collisionOperator=0 "
            "(the linearized Fokker-Planck operator carries the Phi1-shifted densities)."
        )
    if collision_operator == 3:
        # Improved Sugama momentum/energy-conserving model operator (research
        # extension beyond v3; Sugama et al. Phys. Plasmas 26, 102108 (2019);
        # Frei et al. arXiv:2202.06293).  Same block layout as the FP operator.
        sugama = make_improved_sugama_v3_operator(
            x=np.asarray(grids.x, dtype=np.float64),
            x_weights=np.asarray(grids.x_weights, dtype=np.float64),
            ddx=np.asarray(grids.ddx, dtype=np.float64),
            d2dx2=np.asarray(grids.d2dx2, dtype=np.float64),
            z_s=np.asarray(species.z, dtype=np.float64),
            m_hats=np.asarray(species.m_hat, dtype=np.float64),
            n_hats=np.asarray(species.n_hat, dtype=np.float64),
            t_hats=np.asarray(species.t_hat, dtype=np.float64),
            nu_n=nu_n,
            krook=krook,
            n_xi=grids.n_xi,
            n_xi_for_x=np.asarray(grids.n_xi_for_x, dtype=np.int32),
        )
    elif collision_operator == 1:
        nu_n_use = nu_n
        if rhs_mode == 3:
            nu_n_use = nu_n_from_nu_prime(
                nu_prime=_get_float(phys, "nuPrime", 1.0),
                b0_over_bbar=b0_eff,
                g_hat=g_eff,
                i_hat=i_eff,
                iota=float(geom.iota),
            )
        pas = make_pitch_angle_scattering_v3_operator(
            x=grids.x,
            z_s=species.z,
            m_hats=species.m_hat,
            n_hats=species.n_hat,
            t_hats=species.t_hat,
            nu_n=float(nu_n_use),
            krook=krook,
            n_xi_for_x=grids.n_xi_for_x,
            n_xi=grids.n_xi,
        )
    elif include_phi1_in_collision:
        # includePhi1InCollisionOperator=.true.: the poloidally varying FP
        # operator (collisions.FokkerPlanckV3Phi1Operator); the collision
        # densities are shifted by exp(-Z*alpha*Phi1Hat/THat) at apply time.
        fp_phi1 = make_fokker_planck_v3_phi1_operator(
            rosenbluth_method=rosenbluth_method,
            x=np.asarray(grids.x, dtype=np.float64),
            x_weights=np.asarray(grids.x_weights, dtype=np.float64),
            ddx=np.asarray(grids.ddx, dtype=np.float64),
            d2dx2=np.asarray(grids.d2dx2, dtype=np.float64),
            x_grid_k=_get_float(other, "xGrid_k", 0.0),
            z_s=np.asarray(species.z, dtype=np.float64),
            m_hats=np.asarray(species.m_hat, dtype=np.float64),
            n_hats=np.asarray(species.n_hat, dtype=np.float64),
            t_hats=np.asarray(species.t_hat, dtype=np.float64),
            nu_n=nu_n,
            krook=krook,
            n_xi=grids.n_xi,
            nl=grids.n_l,
            alpha=alpha,
            n_xi_for_x=np.asarray(grids.n_xi_for_x, dtype=np.int32),
        )
    else:
        import os  # noqa: PLC0415

        strict_env = os.environ.get("DKX_FP_STRICT_PARITY", "").strip().lower()
        if strict_env in {"0", "false", "no", "off"}:
            strict_parity = False
        elif strict_env in {"1", "true", "yes", "on"}:
            strict_parity = True
        else:
            strict_parity = bool(rhs_mode == 1 and n_species > 1)
        fp = make_fokker_planck_v3_operator(
            rosenbluth_method=rosenbluth_method,
            x=np.asarray(grids.x, dtype=np.float64),
            x_weights=np.asarray(grids.x_weights, dtype=np.float64),
            ddx=np.asarray(grids.ddx, dtype=np.float64),
            d2dx2=np.asarray(grids.d2dx2, dtype=np.float64),
            x_grid_k=_get_float(other, "xGrid_k", 0.0),
            z_s=np.asarray(species.z, dtype=np.float64),
            m_hats=np.asarray(species.m_hat, dtype=np.float64),
            n_hats=np.asarray(species.n_hat, dtype=np.float64),
            t_hats=np.asarray(species.t_hat, dtype=np.float64),
            nu_n=nu_n,
            krook=krook,
            n_xi=grids.n_xi,
            nl=grids.n_l,
            n_xi_for_x=np.asarray(grids.n_xi_for_x, dtype=np.int32),
            strict_parity=strict_parity,
        )

    # ---- inductive E_parallel ----
    e_parallel_hat = _get_float(phys, "EParallelHat", 0.0)
    epar_spec_raw = phys.get("EPARALLELHATSPEC", None)
    if epar_spec_raw is None:
        e_parallel_hat_spec = jnp.zeros((n_species,), dtype=jnp.float64)
    else:
        e_parallel_hat_spec = jnp.atleast_1d(jnp.asarray(epar_spec_raw, dtype=jnp.float64))
        if e_parallel_hat_spec.shape == (1,) and n_species > 1:
            e_parallel_hat_spec = jnp.broadcast_to(e_parallel_hat_spec, (n_species,))
        if e_parallel_hat_spec.shape != (n_species,):
            raise ValueError(
                f"EParallelHatSpec must have {n_species} entries, got {e_parallel_hat_spec.shape}"
            )

    # Tangential magnetic-drift geometry + upwinded stencils (magneticDriftScheme 1-9).
    # The radial (psi) B-field derivatives come straight from the Boozer/VMEC
    # geometry; the plus/minus stencils implement magneticDriftDerivativeScheme.
    # The flux-function scalars feed the scheme 3/4/8 shear terms and the scheme 6
    # pressure term; the Sugama normal-curvature factor feeds schemes 5/6.
    if with_magnetic_drifts:
        magnetic_drift_arrays: dict[str, Any] = {
            "b_hat_sub_psi": geom.b_hat_sub_psi,
            "db_hat_dpsi_hat": geom.db_hat_dpsi_hat,
            "db_hat_sub_psi_dtheta": geom.db_hat_sub_psi_dtheta,
            "db_hat_sub_psi_dzeta": geom.db_hat_sub_psi_dzeta,
            "db_hat_sub_theta_dpsi_hat": geom.db_hat_sub_theta_dpsi_hat,
            "db_hat_sub_zeta_dpsi_hat": geom.db_hat_sub_zeta_dpsi_hat,
            "ddtheta_magdrift_plus": grids.ddtheta_magdrift_plus,
            "ddtheta_magdrift_minus": grids.ddtheta_magdrift_minus,
            "ddzeta_magdrift_plus": grids.ddzeta_magdrift_plus,
            "ddzeta_magdrift_minus": grids.ddzeta_magdrift_minus,
            "iota": jnp.asarray(geom.iota, dtype=jnp.float64),
            "g_hat": jnp.asarray(g_eff, dtype=jnp.float64),
            "diota_dpsi_hat": jnp.asarray(geom.diota_dpsi_hat, dtype=jnp.float64),
            "p_prime_hat": jnp.asarray(geom.p_prime_hat, dtype=jnp.float64),
            "grad_psi_dot_grad_b_over_gpsipsi": geom.grad_psi_dot_grad_b_over_gpsipsi,
        }
    else:
        magnetic_drift_arrays = {}

    op = KineticOperator(
        n_species=n_species,
        n_x=grids.n_x,
        n_xi=grids.n_xi,
        n_theta=grids.n_theta,
        n_zeta=grids.n_zeta,
        rhs_mode=rhs_mode,
        constraint_scheme=constraint_scheme,
        point_at_x0=point_at_x0,
        use_dkes_exb=use_dkes_exb,
        with_exb=has_er,
        with_er_xidot=bool(include_er_xidot and has_er),
        with_er_xdot=bool(include_xdot and has_er),
        x=grids.x,
        x_weights=grids.x_weights,
        ddx=grids.ddx,
        ddtheta=grids.ddtheta,
        ddzeta=grids.ddzeta,
        theta_weights=grids.theta_weights,
        zeta_weights=grids.zeta_weights,
        n_xi_for_x=grids.n_xi_for_x,
        xi_coupling_lower=jnp.asarray(legendre_coupling_lower(grids.n_xi)),
        xi_coupling_upper=jnp.asarray(legendre_coupling_upper(grids.n_xi)),
        b_hat=geom.b_hat,
        db_hat_dtheta=geom.db_hat_dtheta,
        db_hat_dzeta=geom.db_hat_dzeta,
        d_hat=geom.d_hat,
        b_hat_sup_theta=geom.b_hat_sup_theta,
        b_hat_sup_zeta=geom.b_hat_sup_zeta,
        b_hat_sub_theta=geom.b_hat_sub_theta,
        b_hat_sub_zeta=geom.b_hat_sub_zeta,
        fsab_hat2=jnp.asarray(fsab_hat2, dtype=jnp.float64),
        z_s=species.z,
        m_hat=species.m_hat,
        t_hat=species.t_hat,
        n_hat=species.n_hat,
        dn_hat_dpsi_hat=species.dn_hat_dpsi_hat,
        dt_hat_dpsi_hat=species.dt_hat_dpsi_hat,
        alpha=jnp.asarray(alpha, dtype=jnp.float64),
        delta=jnp.asarray(delta, dtype=jnp.float64),
        dphi_hat_dpsi_hat=jnp.asarray(dphi_rhs, dtype=jnp.float64),
        dphi_hat_dpsi_hat_kinetic=jnp.asarray(dphi_kinetic, dtype=jnp.float64),
        e_parallel_hat=jnp.asarray(e_parallel_hat, dtype=jnp.float64),
        e_parallel_hat_spec=e_parallel_hat_spec,
        pas=pas,
        fp=fp,
        sugama=sugama,
        fp_phi1=fp_phi1,
        include_phi1=bool(include_phi1),
        quasineutrality_option=int(quasineutrality_option),
        include_phi1_in_kinetic=bool(include_phi1_in_kinetic),
        with_adiabatic=bool(with_adiabatic),
        adiabatic_z=jnp.asarray(adiabatic_z, dtype=jnp.float64),
        adiabatic_n_hat=jnp.asarray(adiabatic_n_hat, dtype=jnp.float64),
        adiabatic_t_hat=jnp.asarray(adiabatic_t_hat, dtype=jnp.float64),
        phi1_hat_base=(
            jnp.zeros((grids.n_theta, grids.n_zeta), dtype=jnp.float64) if include_phi1 else None
        ),
        phi1_lin_state=None,
        external_phi1_hat=external_phi1_hat,
        ddx_xdot_plus=grids.ddx_xdot_plus,
        ddx_xdot_minus=grids.ddx_xdot_minus,
        with_magnetic_drifts=with_magnetic_drifts,
        magnetic_drift_scheme=magnetic_drift_scheme if with_magnetic_drifts else 0,
        **magnetic_drift_arrays,
    )
    return KineticOperatorBuild(operator=op, grids=grids, geometry=geom, radial=radial)

apply_kinetic_operator_jit = jax.jit(lambda op, v: op.apply(v))
