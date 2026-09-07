"""Velocity-space moments and transport diagnostics of solved SFINCS states.

This module consolidates every "given a solved state vector, produce physical
outputs" computation previously scattered across
``dkx/problems/transport_diagnostics.py`` (vm-drift flux moments,
RHSMode=1 output tables, RHSMode=2/3 transport matrices),
``dkx/outputs/rhsmode1.py`` (NTV recomputation, classical-flux and
radial-coordinate output plumbing), ``dkx/outputs/writer.py`` (the
inline electric-drift vE/vE0 flux block of the Phi1 path), and
``dkx/physics/classical_transport.py``.  It becomes
``dkx/moments.py`` at the v2 purge.

Fortran correspondence (paths relative to ``sfincs/fortran/version3``):

- ``diagnostics.F90`` — all f1 velocity moments: density/pressure
  perturbations, parallel flows, FSABFlow, particle/heat/momentum fluxes for
  the vm (magnetic-drift) and vE (ExB) families, NTV, sources, and the
  ``transportMatrix`` assembly for RHSMode 2/3.  See also the SFINCS technical
  documentation, "Moments of the distribution function" and "Transport
  matrix" sections.
- ``classicalTransport.F90:calculateClassicalFlux`` — classical fluxes.
- ``geometry.F90:computeBIntegrals`` — VPrimeHat, FSABHat2, B0OverBBar and
  the NTV kernel ingredients (uHat).
- ``radialCoordinates.F90`` — the psiHat/psiN/rHat/rN flux variants (via
  :class:`dkx.constants.RadialCoordinates`).

Units and normalizations (plan section 2.5): every quantity is the
dimensionless v3 "Hat" quantity — fluxes are normalized to
``nBar*vBar*RBar^2`` projected on the ``psiHat`` gradient, flows to
``nBar*vBar``, ``Delta = mBar*vBar/(e*BBar*RBar)`` and ``alpha = e*phiBar/TBar``
enter exactly as in ``diagnostics.F90``.  All functions here are pure,
jit/vmap friendly, and take explicit arrays: no namelist parsing, no file IO.

Array-axis conventions: S = species, X = speed grid, L = Legendre mode,
T = theta, Z = zeta, N = whichRHS (or iteration) index.  The distribution
block of a state vector reshapes to ``(S, X, L, T, Z)`` (v3 ``indices.F90``
ordering).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, NamedTuple

# The JAX backend is imported below; dkx/runtime.py explains why this is here.
from .runtime import configure as _configure_runtime

_configure_runtime()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax import lax  # noqa: E402
from jax import vmap  # noqa: E402

from .constants import RadialCoordinates  # noqa: E402

__all__ = [
    "StateLayout", "VelocityGrid", "FluxSurface", "SpeciesParams",
    "VmFluxMoments", "ElectricDriftFluxMoments", "RadialFluxVariants",
    "vprime_hat", "flux_surface_b_integrals", "maxwellian_f0_l0",
    "vm_flux_moments", "vm_flux_moments_batch", "rhsmode1_moments",
    "electric_drift_flux_moments", "combined_drift_fluxes", "heat_flux_without_phi1",
    "ntv_kernel", "ntv_moments",
    "transport_matrix_size", "transport_matrix_from_flux_arrays",
    "transport_matrix_from_state_vectors", "transport_moments_table",
    "classical_fluxes", "flux_coordinate_variants",
    "legendre_tail_relative_l2", "legendre_tail_relative_l2_batch",
    "legendre_tail_relative_l2_upper_bound_batch",
]  # fmt: skip

# ---- Input containers: state-vector layout, grids, geometry, species -------

@dataclass(frozen=True)
class StateLayout:
    """Static layout of one v3 state vector (v3 ``indices.F90`` ordering).

    The vector is ``[f1 block, Phi1(theta,zeta), lambda, constraint unknowns]``
    with the f1 block reshaping to ``(S, X, L, T, Z)``.  All fields are Python
    ints/bools so the object is hashable and can be closed over by jitted
    functions.
    """

    n_species: int
    n_x: int
    n_xi: int
    n_theta: int
    n_zeta: int
    include_phi1: bool = False
    constraint_scheme: int = 0

    @property
    def f_shape(self) -> tuple[int, int, int, int, int]:
        return (self.n_species, self.n_x, self.n_xi, self.n_theta, self.n_zeta)

    @property
    def f_size(self) -> int:
        return self.n_species * self.n_x * self.n_xi * self.n_theta * self.n_zeta

    @property
    def phi1_size(self) -> int:
        # Phi1(theta,zeta) unknowns plus the <Phi1>=0 Lagrange multiplier row.
        return self.n_theta * self.n_zeta + 1 if self.include_phi1 else 0

    @property
    def extra_size(self) -> int:
        if self.constraint_scheme == 2:
            return self.n_species * self.n_x
        if self.constraint_scheme in {1, 3, 4}:
            return 2 * self.n_species
        return 0

    @property
    def total_size(self) -> int:
        return self.f_size + self.phi1_size + self.extra_size

    def f_delta(self, x_full: jnp.ndarray) -> jnp.ndarray:
        """Distribution-function block reshaped to ``(S, X, L, T, Z)``."""
        return jnp.asarray(x_full, dtype=jnp.float64)[: self.f_size].reshape(self.f_shape)

    def phi1_hat(self, x_full: jnp.ndarray) -> jnp.ndarray | None:
        """``Phi1Hat(theta,zeta)`` block, or ``None`` when not present."""
        if not self.include_phi1:
            return None
        flat = jnp.asarray(x_full, dtype=jnp.float64)[self.f_size : self.f_size + self.n_theta * self.n_zeta]
        return flat.reshape((self.n_theta, self.n_zeta))

    def sources(self, x_full: jnp.ndarray) -> jnp.ndarray | None:
        """Constraint-scheme source unknowns as written to ``sources`` (X,S) or (2,S)."""
        extra = jnp.asarray(x_full, dtype=jnp.float64)[self.f_size + self.phi1_size :].reshape((-1,))
        if self.constraint_scheme == 2:
            return jnp.transpose(extra.reshape((self.n_species, self.n_x)), (1, 0))
        if self.constraint_scheme in {1, 3, 4}:
            return jnp.transpose(extra.reshape((self.n_species, 2)), (1, 0))
        return None

    @staticmethod
    def from_operator(op: Any) -> "StateLayout":
        """Duck-typed constructor from a legacy ``V3FullSystemOperator``."""
        return StateLayout(
            n_species=int(op.n_species), n_x=int(op.n_x), n_xi=int(op.n_xi),
            n_theta=int(op.n_theta), n_zeta=int(op.n_zeta),
            include_phi1=bool(op.include_phi1), constraint_scheme=int(op.constraint_scheme),
        )  # fmt: skip

class VelocityGrid(NamedTuple):
    """Speed-grid nodes/weights and per-x Legendre mode counts (``Nxi_for_x``)."""

    x: jnp.ndarray  # (X,) speed nodes
    x_weights: jnp.ndarray  # (X,) quadrature weights (include exp(-x^2) factor)
    n_xi_for_x: jnp.ndarray  # (X,) int, number of Legendre modes retained at each x

    @staticmethod
    def from_operator(op: Any) -> "VelocityGrid":
        return VelocityGrid(
            x=jnp.asarray(op.x, dtype=jnp.float64),
            x_weights=jnp.asarray(op.x_weights, dtype=jnp.float64),
            n_xi_for_x=jnp.asarray(op.fblock.collisionless.n_xi_for_x, dtype=jnp.int32),
        )

_FLUX_SURFACE_OP_FIELDS = (
    "theta_weights", "zeta_weights", "b_hat", "d_hat", "db_hat_dtheta", "db_hat_dzeta",
    "b_hat_sub_theta", "b_hat_sub_zeta", "fsab_hat2",
)  # fmt: skip

class FluxSurface(NamedTuple):
    """Flux-surface geometry arrays used by the moment integrals."""

    theta_weights: jnp.ndarray  # (T,)
    zeta_weights: jnp.ndarray  # (Z,)
    b_hat: jnp.ndarray  # (T,Z)
    d_hat: jnp.ndarray  # (T,Z) Jacobian factor: dV ~ dtheta dzeta / DHat
    db_hat_dtheta: jnp.ndarray  # (T,Z)
    db_hat_dzeta: jnp.ndarray  # (T,Z)
    b_hat_sub_theta: jnp.ndarray  # (T,Z)
    b_hat_sub_zeta: jnp.ndarray  # (T,Z)
    fsab_hat2: jnp.ndarray  # scalar <BHat^2>

    @staticmethod
    def from_operator(op: Any) -> "FluxSurface":
        return FluxSurface(*(jnp.asarray(getattr(op, name), dtype=jnp.float64) for name in _FLUX_SURFACE_OP_FIELDS))

class SpeciesParams(NamedTuple):
    """Per-species charge, mass, density, temperature ("Hat" normalized)."""

    z_s: jnp.ndarray  # (S,)
    m_hat: jnp.ndarray  # (S,)
    t_hat: jnp.ndarray  # (S,)
    n_hat: jnp.ndarray  # (S,)

    @staticmethod
    def from_operator(op: Any) -> "SpeciesParams":
        return SpeciesParams(
            *(jnp.asarray(getattr(op, n), dtype=jnp.float64) for n in ("z_s", "m_hat", "t_hat", "n_hat"))
        )

# ---- Quadrature helpers (diagnostics.F90 accumulation orders) --------------

def _sum_x(w_x: jnp.ndarray, values_sxtz: jnp.ndarray, *, strict: bool = False) -> jnp.ndarray:
    """``sum_x w_x[x] * values[:, x, :, :]``; ``strict`` uses the Fortran x-loop order."""
    w_x = jnp.asarray(w_x, dtype=jnp.float64).reshape((-1,))
    values_sxtz = jnp.asarray(values_sxtz, dtype=jnp.float64)
    if strict:
        return _sum_x_strict(w_x, values_sxtz)
    return jnp.einsum("x,sxtz->stz", w_x, values_sxtz, precision=lax.Precision.HIGHEST)


@jax.jit
def _sum_x_strict(w_x: jnp.ndarray, values_sxtz: jnp.ndarray) -> jnp.ndarray:
    """The Fortran x-loop accumulation, jitted once at module level.

    The loop body is a closure. Defined inside the caller it is a *new* Python
    object on every call, so the eager dispatch cache misses and XLA lowers the
    same module again -- 42 identical lowerings per solve on the shipped
    tokamak deck, which was most of a warm solve's wall time. Hoisting it under
    a module-level ``jax.jit`` builds the closure once. The accumulation order
    is unchanged, so the Fortran-parity results this branch exists for are
    bit-identical.
    """
    acc0 = jnp.zeros_like(values_sxtz[:, 0, :, :], dtype=jnp.float64)

    def body(ix: int, acc: jnp.ndarray) -> jnp.ndarray:
        return acc + w_x[ix] * values_sxtz[:, ix, :, :]

    return lax.fori_loop(0, int(values_sxtz.shape[1]), body, acc0)

def _sum_tz(w_t: jnp.ndarray, w_z: jnp.ndarray, values_stz: jnp.ndarray) -> jnp.ndarray:
    """Flux-surface quadrature ``sum_t sum_z w_t w_z values[:, t, z]`` -> (S,)."""
    return jnp.einsum(
        "t,z,stz->s",
        jnp.asarray(w_t, dtype=jnp.float64).reshape((-1,)),
        jnp.asarray(w_z, dtype=jnp.float64).reshape((-1,)),
        jnp.asarray(values_stz, dtype=jnp.float64),
        precision=lax.Precision.HIGHEST,
    )

def _sum_tz_sx(w_t: jnp.ndarray, w_z: jnp.ndarray, values_sxtz: jnp.ndarray, *, strict: bool = False) -> jnp.ndarray:
    """``sum_t sum_z w_t w_z values[:, x, t, z]`` -> (S, X)."""
    w_t = jnp.asarray(w_t, dtype=jnp.float64).reshape((-1,))
    w_z = jnp.asarray(w_z, dtype=jnp.float64).reshape((-1,))
    values_sxtz = jnp.asarray(values_sxtz, dtype=jnp.float64)
    if strict:
        return _sum_tz_sx_strict(w_t, w_z, values_sxtz)
    return jnp.einsum("t,z,sxtz->sx", w_t, w_z, values_sxtz, precision=lax.Precision.HIGHEST)


@jax.jit
def _sum_tz_sx_strict(
    w_t: jnp.ndarray, w_z: jnp.ndarray, values_sxtz: jnp.ndarray
) -> jnp.ndarray:
    """The Fortran (theta, zeta) loop order, jitted once. See :func:`_sum_x_strict`."""
    acc0 = jnp.zeros_like(values_sxtz[:, :, 0, 0], dtype=jnp.float64)

    def body_t(it: int, acc_t: jnp.ndarray) -> jnp.ndarray:
        def body_z(iz: int, acc_z: jnp.ndarray) -> jnp.ndarray:
            return acc_z + (w_t[it] * w_z[iz]) * values_sxtz[:, :, it, iz]

        return lax.fori_loop(0, int(values_sxtz.shape[3]), body_z, acc_t)

    return lax.fori_loop(0, int(values_sxtz.shape[2]), body_t, acc0)

# ---- Flux-surface scalars and the leading-order Maxwellian -----------------

def vprime_hat(surface: FluxSurface) -> jnp.ndarray:
    """``VPrimeHat = sum_ij w_theta_i w_zeta_j / DHat_ij`` (geometry.F90 computeBIntegrals)."""
    inv_d = jnp.asarray(1.0 / surface.d_hat, dtype=jnp.float64)
    return _sum_tz(surface.theta_weights, surface.zeta_weights, inv_d[None, :, :])[0]

def flux_surface_b_integrals(surface: FluxSurface) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return ``(B0OverBBar, GHat, IHat)`` computed from arrays (computeBIntegrals).

    Needed for geometries (notably VMEC ``geometryScheme=5``) whose scalar flux
    functions are stored as placeholders but still enter the transport-matrix
    formulas and monoenergetic normalizations.
    """
    w2d = surface.theta_weights[:, None] * surface.zeta_weights[None, :]
    vp = vprime_hat(surface)
    b0 = jnp.sum(w2d * (surface.b_hat**3) / surface.d_hat) / (vp * jnp.asarray(surface.fsab_hat2, dtype=jnp.float64))
    denom = jnp.asarray(4.0 * jnp.pi * jnp.pi, dtype=jnp.float64)
    g_hat = jnp.sum(w2d * surface.b_hat_sub_zeta) / denom
    i_hat = jnp.sum(w2d * surface.b_hat_sub_theta) / denom
    return b0, g_hat, i_hat

def maxwellian_f0_l0(
    species: SpeciesParams,
    x: jnp.ndarray,
    *,
    alpha: jnp.ndarray | float,
    phi1_hat: jnp.ndarray | None,
    n_theta: int,
    n_zeta: int,
) -> jnp.ndarray:
    """L=0 Maxwellian ``f0`` on the (S,X,T,Z) grid (populateMatrix.F90 ``init_f0``).

    ``f0 = exp(-Z*alpha*Phi1Hat/THat) * nHat*mHat/(pi*THat) * sqrt(mHat/(pi*THat)) * exp(-x^2)``;
    all L>0 components of f0 vanish.
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    expx2 = jnp.exp(-(x * x))  # (X,)
    pref = species.n_hat[:, None] * species.m_hat[:, None] / (jnp.pi * species.t_hat[:, None])
    pref = pref * jnp.sqrt(species.m_hat[:, None] / (jnp.pi * species.t_hat[:, None]))
    pref = pref * expx2[None, :]  # (S,X)
    if phi1_hat is None:
        phi1 = jnp.zeros((int(n_theta), int(n_zeta)), dtype=jnp.float64)
    else:
        phi1 = jnp.asarray(phi1_hat, dtype=jnp.float64)
    exp_phi1 = jnp.exp(
        -(species.z_s[:, None, None] * jnp.asarray(alpha, dtype=jnp.float64) / species.t_hat[:, None, None])
        * phi1[None, :, :]
    )  # (S,T,Z)
    return pref[:, :, None, None] * exp_phi1[:, None, :, :]

def _factor_vm(surface: FluxSurface) -> jnp.ndarray:
    """Magnetic-drift geometric factor ``(B_theta dB/dzeta - B_zeta dB/dtheta)/BHat^3``."""
    return (surface.b_hat_sub_theta * surface.db_hat_dzeta - surface.b_hat_sub_zeta * surface.db_hat_dtheta) / (
        surface.b_hat * surface.b_hat * surface.b_hat
    )

def _l_masks(n_xi_for_x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    n = jnp.asarray(n_xi_for_x, dtype=jnp.int32)
    return (
        (n > 0).astype(jnp.float64),
        (n > 1).astype(jnp.float64),
        (n > 2).astype(jnp.float64),
        (n > 3).astype(jnp.float64),
    )


def legendre_tail_relative_l2_batch(
    layout: StateLayout,
    vgrid: VelocityGrid,
    surface: FluxSurface,
    x_full_stack: jnp.ndarray,
    *,
    tail_modes: int = 2,
) -> jnp.ndarray:
    """Relative spatial-L2 energy in the last active Legendre modes.

    For each solved state, speed node, and species, the numerator contains the
    final ``tail_modes`` active Legendre coefficients and the denominator all
    active coefficients. The norm uses the Legendre orthogonality factor
    ``2/(2*l+1)`` and the absolute flux-surface volume element. The result
    has axes ``(state, speed, species)`` and is zero when the complete
    coefficient block is exactly zero.

    This compact quantity is a truncation indicator, not a convergence proof:
    observable movement under a resolution increase remains the admission
    gate. Retaining two modes also makes even/odd under-resolution behavior
    visible, as recommended by the SFINCS convergence guidance.
    """
    if int(tail_modes) < 1:
        raise ValueError("tail_modes must be at least 1")
    states = jnp.asarray(x_full_stack, dtype=jnp.float64)
    if states.ndim != 2 or states.shape[1] != layout.total_size:
        raise ValueError(
            f"x_full_stack must have shape (N,{layout.total_size}), got {states.shape}"
        )
    f = states[:, : layout.f_size].reshape((states.shape[0], *layout.f_shape))
    ell = jnp.arange(layout.n_xi, dtype=jnp.int32)[None, :]
    active_count = jnp.asarray(vgrid.n_xi_for_x, dtype=jnp.int32)[:, None]
    active = ell < active_count
    tail = active & (ell >= jnp.maximum(active_count - int(tail_modes), 0))
    legendre_weight = 2.0 / (
        2.0 * jnp.arange(layout.n_xi, dtype=jnp.float64) + 1.0
    )
    surface_weight = (
        surface.theta_weights[:, None] * surface.zeta_weights[None, :]
    ) / jnp.abs(surface.d_hat)
    scale = jnp.max(jnp.abs(f), axis=(3, 4, 5), keepdims=True)
    scaled = jnp.where(scale > 0.0, f / scale, 0.0)
    energy = jnp.einsum(
        "nsxltz,l,tz->nsxl",
        scaled * scaled,
        legendre_weight,
        surface_weight,
    )
    total = jnp.sum(jnp.where(active[None, None, :, :], energy, 0.0), axis=-1)
    tail_energy = jnp.sum(
        jnp.where(tail[None, None, :, :], energy, 0.0), axis=-1
    )
    ratio = jnp.where(total > 0.0, jnp.sqrt(tail_energy / total), 0.0)
    return jnp.transpose(ratio, (0, 2, 1))


def legendre_tail_relative_l2(
    layout: StateLayout,
    vgrid: VelocityGrid,
    surface: FluxSurface,
    x_full: jnp.ndarray,
    *,
    tail_modes: int = 2,
) -> jnp.ndarray:
    """Single-state wrapper for :func:`legendre_tail_relative_l2_batch`."""
    state = jnp.asarray(x_full, dtype=jnp.float64)
    if state.ndim != 1:
        raise ValueError(f"x_full must be one-dimensional, got {state.shape}")
    return legendre_tail_relative_l2_batch(
        layout, vgrid, surface, state[None, :], tail_modes=tail_modes
    )[0]


def legendre_tail_relative_l2_upper_bound_batch(
    layout: StateLayout,
    vgrid: VelocityGrid,
    surface: FluxSurface,
    low_state_stack: jnp.ndarray,
    tail_blocks_stack: jnp.ndarray,
    *,
    keep_lowest: int = 3,
) -> jnp.ndarray:
    """Bound the full-state Legendre-tail ratio from selected exact blocks.

    ``low_state_stack`` contains exact low modes and zero padding from the
    truncated structured solver. ``tail_blocks_stack`` contains the exact
    final active modes reconstructed by its selected-tail sweep, with shape
    ``(state, species, speed, tail, theta*zeta)``. The denominator uses the
    union of these known low and tail modes. Since the omitted middle-mode
    energy is nonnegative, the returned ratio is an upper bound on the same
    tail-over-full-state ratio computed by :func:`legendre_tail_relative_l2_batch`.

    This remains supporting truncation evidence, not a convergence proof;
    observable movement under a resolution increase is the admission gate.
    """
    keep = int(keep_lowest)
    if keep < 1 or keep > layout.n_xi:
        raise ValueError(f"keep_lowest must be in [1,{layout.n_xi}]")
    states = jnp.asarray(low_state_stack, dtype=jnp.float64)
    tails = jnp.asarray(tail_blocks_stack, dtype=jnp.float64)
    expected_tail_prefix = (states.shape[0], layout.n_species, layout.n_x)
    expected_tz = layout.n_theta * layout.n_zeta
    if states.ndim != 2 or states.shape[1] != layout.total_size:
        raise ValueError(
            f"low_state_stack must have shape (N,{layout.total_size}), got {states.shape}"
        )
    if (
        tails.ndim != 5
        or tails.shape[:3] != expected_tail_prefix
        or tails.shape[4] != expected_tz
    ):
        raise ValueError(
            "tail_blocks_stack must have shape "
            f"(N,{layout.n_species},{layout.n_x},K,{expected_tz}), got {tails.shape}"
        )
    tail_modes = int(tails.shape[3])
    if tail_modes < 1:
        raise ValueError("tail_blocks_stack must retain at least one tail mode")
    active_count = jnp.asarray(vgrid.n_xi_for_x, dtype=jnp.int32)

    f = states[:, : layout.f_size].reshape((states.shape[0], *layout.f_shape))
    low = f[:, :, :, :keep].reshape(
        states.shape[0], layout.n_species, layout.n_x, keep, expected_tz
    )
    scale = jnp.maximum(
        jnp.max(jnp.abs(low), axis=(3, 4), keepdims=True),
        jnp.max(jnp.abs(tails), axis=(3, 4), keepdims=True),
    )
    low_scaled = jnp.where(scale > 0.0, low / scale, 0.0)
    tail_scaled = jnp.where(scale > 0.0, tails / scale, 0.0)
    surface_weight = (
        surface.theta_weights[:, None] * surface.zeta_weights[None, :]
    ).reshape((-1,)) / jnp.abs(surface.d_hat).reshape((-1,))

    low_ell = jnp.arange(keep, dtype=jnp.int32)[None, :]
    low_active = low_ell < active_count[:, None]
    low_legendre_weight = 2.0 / (2.0 * low_ell.astype(jnp.float64) + 1.0)
    low_energy = jnp.einsum(
        "nsxlt,t,xl->nsxl",
        low_scaled * low_scaled,
        surface_weight,
        low_legendre_weight * low_active,
    )

    tail_ell = active_count[:, None] - tail_modes + jnp.arange(
        tail_modes, dtype=jnp.int32
    )[None, :]
    tail_legendre_weight = 2.0 / (2.0 * tail_ell.astype(jnp.float64) + 1.0)
    tail_energy_by_mode = jnp.einsum(
        "nsxlt,t,xl->nsxl",
        tail_scaled * tail_scaled,
        surface_weight,
        tail_legendre_weight,
    )
    numerator = jnp.sum(tail_energy_by_mode, axis=-1)
    nonoverlap_tail = tail_ell >= keep
    known_total = jnp.sum(low_energy, axis=-1) + jnp.sum(
        jnp.where(nonoverlap_tail[None, None, :, :], tail_energy_by_mode, 0.0),
        axis=-1,
    )
    ratio = jnp.where(known_total > 0.0, jnp.sqrt(numerator / known_total), 0.0)
    return jnp.transpose(ratio, (0, 2, 1))

# ---- vm (magnetic-drift) flux moments — the RHSMode=2/3 diagnostics subset ----

class VmFluxMoments(NamedTuple):
    """Magnetic-drift flux moments of one solved state (diagnostics.F90 subset).

    Same content as the legacy ``V3TransportDiagnostics``: the vm particle/heat
    fluxes and FSABFlow that the RHSMode=2/3 transport matrices depend on, plus
    the before-surface-integral and per-x decompositions written to
    ``sfincsOutput.h5``.
    """

    vprime_hat: jnp.ndarray  # scalar
    particle_flux_vm_psi_hat: jnp.ndarray  # (S,)
    heat_flux_vm_psi_hat: jnp.ndarray  # (S,)
    fsab_flow: jnp.ndarray  # (S,)
    particle_flux_before_surface_integral_vm: jnp.ndarray  # (S,T,Z)
    heat_flux_before_surface_integral_vm: jnp.ndarray  # (S,T,Z)
    particle_flux_before_surface_integral_vm0: jnp.ndarray  # (S,T,Z)
    heat_flux_before_surface_integral_vm0: jnp.ndarray  # (S,T,Z)
    particle_flux_vm_psi_hat_vs_x: jnp.ndarray  # (X,S)
    heat_flux_vm_psi_hat_vs_x: jnp.ndarray  # (X,S)
    fsab_flow_vs_x: jnp.ndarray  # (X,S)

def vm_flux_moments(
    layout: StateLayout,
    vgrid: VelocityGrid,
    surface: FluxSurface,
    species: SpeciesParams,
    x_full: jnp.ndarray,
    *,
    delta: jnp.ndarray | float,
    alpha: jnp.ndarray | float,
    phi1_hat: jnp.ndarray | None = None,
) -> VmFluxMoments:
    """vm particle/heat fluxes and FSABFlow of one state (diagnostics.F90).

    ``phi1_hat`` is the Phi1 field entering the leading-order Maxwellian f0
    (the linearization point); pass ``None`` for runs without Phi1.
    """
    pre = _vm_precompute(vgrid, surface, species, delta)
    f0_l0 = maxwellian_f0_l0(
        species, vgrid.x, alpha=alpha, phi1_hat=phi1_hat, n_theta=layout.n_theta, n_zeta=layout.n_zeta
    )
    return _vm_core(layout, pre, surface, f0_l0, x_full, flow_via_b_over_d=False)

def vm_flux_moments_batch(
    layout: StateLayout,
    vgrid: VelocityGrid,
    surface: FluxSurface,
    species: SpeciesParams,
    x_full_stack: jnp.ndarray,
    *,
    delta: jnp.ndarray | float,
    alpha: jnp.ndarray | float,
    phi1_hat: jnp.ndarray | None = None,
) -> VmFluxMoments:
    """Vectorized :func:`vm_flux_moments` over a leading whichRHS/iteration axis."""
    x_full_stack = jnp.asarray(x_full_stack, dtype=jnp.float64)
    if x_full_stack.ndim != 2 or x_full_stack.shape[1] != layout.total_size:
        raise ValueError(f"x_full_stack must have shape (N,{layout.total_size}), got {x_full_stack.shape}")

    def _one(x_state: jnp.ndarray) -> VmFluxMoments:
        return vm_flux_moments(layout, vgrid, surface, species, x_state, delta=delta, alpha=alpha, phi1_hat=phi1_hat)

    return vmap(_one, in_axes=0, out_axes=0)(x_full_stack)

class _VmPrecomputed(NamedTuple):
    """Geometry/species factors precomputed eagerly for the batched vm diagnostics.

    Mirrors the legacy ``V3TransportDiagnosticsPrecomputed`` structure so the
    jitted whichRHS batch is numerically identical to the legacy jitted path.
    """

    vprime_hat: jnp.ndarray
    factor_vm: jnp.ndarray  # (T,Z)
    wpf0: jnp.ndarray  # (X,)
    wpf2: jnp.ndarray  # (X,)
    whf0: jnp.ndarray  # (X,)
    whf2: jnp.ndarray  # (X,)
    wf1: jnp.ndarray  # (X,)
    particle_flux_factor_vm: jnp.ndarray  # (S,)
    heat_flux_factor_vm: jnp.ndarray  # (S,)
    flow_factor: jnp.ndarray  # (S,)
    b_over_d: jnp.ndarray  # (T,Z)

def _vm_precompute(
    vgrid: VelocityGrid, surface: FluxSurface, species: SpeciesParams, delta: jnp.ndarray | float
) -> _VmPrecomputed:
    vp = vprime_hat(surface)
    w_pf = vgrid.x_weights * (vgrid.x**4)
    w_hf = vgrid.x_weights * (vgrid.x**6)
    w_flow = vgrid.x_weights * (vgrid.x**3)
    mask0, mask1, mask2, _ = _l_masks(vgrid.n_xi_for_x)
    z, m, t = species.z_s, species.m_hat, species.t_hat
    sqrt_t = jnp.sqrt(t)
    sqrt_m = jnp.sqrt(m)
    delta = jnp.asarray(delta, dtype=jnp.float64)
    return _VmPrecomputed(
        vprime_hat=vp,
        factor_vm=_factor_vm(surface),
        wpf0=w_pf * mask0,
        wpf2=w_pf * mask2,
        whf0=w_hf * mask0,
        whf2=w_hf * mask2,
        wf1=w_flow * mask1,
        particle_flux_factor_vm=jnp.pi * delta * (t * t) * sqrt_t / (z * vp * m * sqrt_m),
        heat_flux_factor_vm=jnp.pi * delta * (t * t * t) * sqrt_t / (2.0 * z * vp * m * sqrt_m),
        flow_factor=4.0 * jnp.pi * (t * t) / (3.0 * m * m),
        b_over_d=jnp.asarray(surface.b_hat / surface.d_hat, dtype=jnp.float64),
    )

def _vm_core(
    layout: StateLayout,
    pre: _VmPrecomputed,
    surface: FluxSurface,
    f0_l0: jnp.ndarray,
    x_full: jnp.ndarray,
    *,
    flow_via_b_over_d: bool,
) -> VmFluxMoments:
    """Shared vm-moment kernel.

    ``flow_via_b_over_d`` selects between the two floating-point associations
    of the legacy code paths: ``flow * (BHat/DHat)`` (the jitted precomputed
    batch used by the RHSMode=2/3 output table) versus ``(flow * BHat)/DHat``
    (the per-state and plain-batch paths).  The results differ only by
    rounding, but strict-parity tests distinguish them.
    """
    f_delta = jnp.asarray(x_full, dtype=jnp.float64)[: layout.f_size].reshape(layout.f_shape)
    f_l0 = f_delta[:, :, 0, :, :] + jnp.asarray(f0_l0, dtype=jnp.float64)
    f_l1 = f_delta[:, :, 1, :, :] if layout.n_xi > 1 else jnp.zeros_like(f_l0)
    f_l2 = f_delta[:, :, 2, :, :] if layout.n_xi > 2 else jnp.zeros_like(f_l0)
    f0_l2 = jnp.zeros_like(f0_l0)
    tw, zw = surface.theta_weights, surface.zeta_weights

    def _before(factor_s: jnp.ndarray, sum_l0: jnp.ndarray, sum_l2: jnp.ndarray) -> jnp.ndarray:
        return factor_s[:, None, None] * (
            (8.0 / 3.0) * pre.factor_vm[None, :, :] * sum_l0 + (4.0 / 15.0) * pre.factor_vm[None, :, :] * sum_l2
        )

    pf_before = _before(pre.particle_flux_factor_vm, _sum_x(pre.wpf0, f_l0), _sum_x(pre.wpf2, f_l2))
    hf_before = _before(pre.heat_flux_factor_vm, _sum_x(pre.whf0, f_l0), _sum_x(pre.whf2, f_l2))
    pf_before_vm0 = _before(pre.particle_flux_factor_vm, _sum_x(pre.wpf0, f0_l0), _sum_x(pre.wpf2, f0_l2))
    hf_before_vm0 = _before(pre.heat_flux_factor_vm, _sum_x(pre.whf0, f0_l0), _sum_x(pre.whf2, f0_l2))

    flow = pre.flow_factor[:, None, None] * _sum_x(pre.wf1, f_l1)
    if flow_via_b_over_d:
        flow_weight_tz = flow * pre.b_over_d[None, :, :]
    else:
        flow_weight_tz = flow * surface.b_hat[None, :, :] / surface.d_hat[None, :, :]
    fsab_flow = _sum_tz(tw, zw, flow_weight_tz) / pre.vprime_hat

    def _before_x(factor_s: jnp.ndarray, w0: jnp.ndarray, w2: jnp.ndarray) -> jnp.ndarray:
        return factor_s[:, None, None, None] * (
            (8.0 / 3.0) * pre.factor_vm[None, None, :, :] * (f_l0 * w0[None, :, None, None])
            + (4.0 / 15.0) * pre.factor_vm[None, None, :, :] * (f_l2 * w2[None, :, None, None])
        )

    pf_vs_x = _sum_tz_sx(tw, zw, _before_x(pre.particle_flux_factor_vm, pre.wpf0, pre.wpf2))
    hf_vs_x = _sum_tz_sx(tw, zw, _before_x(pre.heat_flux_factor_vm, pre.whf0, pre.whf2))
    flow_x = pre.flow_factor[:, None, None, None] * (f_l1 * pre.wf1[None, :, None, None])
    if flow_via_b_over_d:
        flow_weight_x = flow_x * pre.b_over_d[None, None, :, :]
    else:
        flow_weight_x = flow_x * surface.b_hat[None, None, :, :] / surface.d_hat[None, None, :, :]
    fsab_flow_vs_x = _sum_tz_sx(tw, zw, flow_weight_x) / pre.vprime_hat

    return VmFluxMoments(
        vprime_hat=pre.vprime_hat,
        particle_flux_vm_psi_hat=_sum_tz(tw, zw, pf_before),
        heat_flux_vm_psi_hat=_sum_tz(tw, zw, hf_before),
        fsab_flow=fsab_flow,
        particle_flux_before_surface_integral_vm=pf_before,
        heat_flux_before_surface_integral_vm=hf_before,
        particle_flux_before_surface_integral_vm0=pf_before_vm0,
        heat_flux_before_surface_integral_vm0=hf_before_vm0,
        particle_flux_vm_psi_hat_vs_x=jnp.transpose(pf_vs_x, (1, 0)),
        heat_flux_vm_psi_hat_vs_x=jnp.transpose(hf_vs_x, (1, 0)),
        fsab_flow_vs_x=jnp.transpose(fsab_flow_vs_x, (1, 0)),
    )

@partial(jax.jit, static_argnums=(0,))
def _vm_flux_moments_batch_precomputed_jit(
    layout: StateLayout,
    pre: _VmPrecomputed,
    vgrid: VelocityGrid,
    surface: FluxSurface,
    species: SpeciesParams,
    x_full_stack: jnp.ndarray,
    alpha: jnp.ndarray,
    phi1_hat: jnp.ndarray | None,
) -> VmFluxMoments:
    """Jitted whichRHS batch (mirrors the legacy precomputed jitted path bit-for-bit)."""
    f0_l0 = maxwellian_f0_l0(
        species, vgrid.x, alpha=alpha, phi1_hat=phi1_hat, n_theta=layout.n_theta, n_zeta=layout.n_zeta
    )

    def _one(x_state: jnp.ndarray) -> VmFluxMoments:
        return _vm_core(layout, pre, surface, f0_l0, x_state, flow_via_b_over_d=True)

    return vmap(_one, in_axes=0, out_axes=0)(x_full_stack)

def _vm_flux_moments_batch_precomputed(
    layout: StateLayout,
    vgrid: VelocityGrid,
    surface: FluxSurface,
    species: SpeciesParams,
    x_full_stack: jnp.ndarray,
    delta: jnp.ndarray,
    alpha: jnp.ndarray,
    phi1_hat: jnp.ndarray | None,
) -> VmFluxMoments:
    """Batch entry for the output-field table: eager precompute + jitted vmap."""
    pre = _vm_precompute(vgrid, surface, species, delta)
    return _vm_flux_moments_batch_precomputed_jit(layout, pre, vgrid, surface, species, x_full_stack, alpha, phi1_hat)

@partial(jax.jit, static_argnums=(0,))
def _vm_flux_moments_batch_plain_jit(
    layout: StateLayout,
    vgrid: VelocityGrid,
    surface: FluxSurface,
    species: SpeciesParams,
    x_full_stack: jnp.ndarray,
    delta: jnp.ndarray,
    alpha: jnp.ndarray,
    phi1_hat: jnp.ndarray | None,
) -> VmFluxMoments:
    """Jitted plain batch for the transport-matrix path (mirrors the legacy jitted path)."""
    return vm_flux_moments_batch(
        layout, vgrid, surface, species, x_full_stack, delta=delta, alpha=alpha, phi1_hat=phi1_hat
    )

# ---- RHSMode=1 per-species output table ------------------------------------

def rhsmode1_moments(
    layout: StateLayout,
    vgrid: VelocityGrid,
    surface: FluxSurface,
    species: SpeciesParams,
    x_full: jnp.ndarray,
    *,
    delta: jnp.ndarray | float,
    alpha: jnp.ndarray | float,
    phi1_hat: jnp.ndarray | None = None,
    phi1_from_state: bool = False,
) -> dict[str, jnp.ndarray]:
    """Full RHSMode=1 per-species moment table of one solved state.

    Mirrors ``diagnostics.F90`` including the ``.../VPrimeHat`` flux-surface
    averages: density/pressure perturbations (+FSA), flows and velocity
    variants (OverB0, OverRootFSAB2), bootstrap current FSABjHat family, and
    the vm/vm0 particle/heat/momentum flux families in the psiHat coordinate.
    Keys use the ``sfincsOutput.h5`` variable names (the io/ name-map contract);
    vE and NTV fields are zero placeholders here — see
    :func:`electric_drift_flux_moments` and :func:`ntv_moments` for the Phi1
    and NTV families.

    With ``phi1_from_state=True`` (requires ``layout.include_phi1``), Phi1 is
    extracted from the state vector and overrides ``phi1_hat`` in f0 and in the
    total density/pressure.
    """
    x_full = jnp.asarray(x_full, dtype=jnp.float64)
    if x_full.shape != (layout.total_size,):
        raise ValueError(f"x_full must have shape {(layout.total_size,)}, got {x_full.shape}")
    if phi1_from_state:
        if not layout.include_phi1:
            raise ValueError("phi1_from_state=True requires layout.include_phi1=True.")
        phi1_hat = layout.phi1_hat(x_full)

    f_delta = layout.f_delta(x_full)
    f0_l0 = maxwellian_f0_l0(
        species, vgrid.x, alpha=alpha, phi1_hat=phi1_hat, n_theta=layout.n_theta, n_zeta=layout.n_zeta
    )
    f_full_l0 = f_delta[:, :, 0, :, :] + f0_l0

    vp = vprime_hat(surface)
    tw, zw = surface.theta_weights, surface.zeta_weights
    fac_vm = _factor_vm(surface)
    delta = jnp.asarray(delta, dtype=jnp.float64)
    alpha = jnp.asarray(alpha, dtype=jnp.float64)

    x = vgrid.x
    xw = vgrid.x_weights
    w_x2, w_x3, w_x4, w_x5, w_x6 = (xw * (x**k) for k in (2, 3, 4, 5, 6))
    mask0, mask1, mask2, mask3 = _l_masks(vgrid.n_xi_for_x)

    z_s, n_hat, t_hat, m_hat = species.z_s, species.n_hat, species.t_hat, species.m_hat
    sqrt_t = jnp.sqrt(t_hat)
    sqrt_m = jnp.sqrt(m_hat)

    density_factor = 4.0 * jnp.pi * t_hat * sqrt_t / (m_hat * sqrt_m)
    flow_factor = 4.0 * jnp.pi * (t_hat * t_hat) / (3.0 * m_hat * m_hat)
    pressure_factor = 8.0 * jnp.pi * (t_hat * t_hat) * sqrt_t / (3.0 * m_hat * sqrt_m)
    pf_factor_vm = jnp.pi * delta * (t_hat * t_hat) * sqrt_t / (z_s * vp * m_hat * sqrt_m)
    hf_factor_vm = jnp.pi * delta * (t_hat * t_hat * t_hat) * sqrt_t / (2.0 * z_s * vp * m_hat * sqrt_m)
    mf_factor_vm = jnp.pi * delta * (t_hat * t_hat * t_hat) / (z_s * vp * m_hat)

    # Moments of the delta-f (strict Fortran x-accumulation order):
    dens = density_factor[:, None, None] * _sum_x(w_x2 * mask0, f_delta[:, :, 0, :, :], strict=True)
    pres = pressure_factor[:, None, None] * _sum_x(w_x4 * mask0, f_delta[:, :, 0, :, :], strict=True)
    if layout.n_xi > 1:
        flow = flow_factor[:, None, None] * _sum_x(w_x3 * mask1, f_delta[:, :, 1, :, :], strict=True)
    else:
        flow = jnp.zeros_like(dens)
    if layout.n_xi > 2:
        pres_aniso = pressure_factor[:, None, None] * (-3.0 / 5.0) * _sum_x(
            w_x4 * mask2, f_delta[:, :, 2, :, :], strict=True
        )
    else:
        pres_aniso = jnp.zeros_like(dens)

    fsadens = _sum_tz(tw, zw, dens / surface.d_hat[None, :, :]) / vp
    fsapres = _sum_tz(tw, zw, pres / surface.d_hat[None, :, :]) / vp
    fsabflow = _sum_tz(tw, zw, flow * surface.b_hat[None, :, :] / surface.d_hat[None, :, :]) / vp

    # vm particle/heat fluxes from the full f (L=0 and L=2):
    f_full_l2 = f_delta[:, :, 2, :, :] if layout.n_xi > 2 else jnp.zeros_like(f_full_l0)
    f0_l2 = jnp.zeros_like(f0_l0)

    def _before(factor_s: jnp.ndarray, sum_l0: jnp.ndarray, sum_l2: jnp.ndarray) -> jnp.ndarray:
        return factor_s[:, None, None] * (
            (8.0 / 3.0) * fac_vm[None, :, :] * sum_l0 + (4.0 / 15.0) * fac_vm[None, :, :] * sum_l2
        )

    pf_before_vm = _before(
        pf_factor_vm, _sum_x(w_x4 * mask0, f_full_l0, strict=True), _sum_x(w_x4 * mask2, f_full_l2, strict=True)
    )
    hf_before_vm = _before(
        hf_factor_vm, _sum_x(w_x6 * mask0, f_full_l0, strict=True), _sum_x(w_x6 * mask2, f_full_l2, strict=True)
    )
    pf_before_vm0 = _before(
        pf_factor_vm, _sum_x(w_x4 * mask0, f0_l0, strict=True), _sum_x(w_x4 * mask2, f0_l2, strict=True)
    )
    hf_before_vm0 = _before(
        hf_factor_vm, _sum_x(w_x6 * mask0, f0_l0, strict=True), _sum_x(w_x6 * mask2, f0_l2, strict=True)
    )
    pf_vm_psi_hat = _sum_tz(tw, zw, pf_before_vm)
    hf_vm_psi_hat = _sum_tz(tw, zw, hf_before_vm)
    pf_vm0_psi_hat = _sum_tz(tw, zw, pf_before_vm0)
    hf_vm0_psi_hat = _sum_tz(tw, zw, hf_before_vm0)

    # Momentum flux (vm): L=1 and L=3 of the delta-f:
    if layout.n_xi > 1:
        sum_mf_l1 = _sum_x(w_x5 * mask1, f_delta[:, :, 1, :, :], strict=True)
    else:
        sum_mf_l1 = jnp.zeros_like(dens)
    if layout.n_xi > 3:
        sum_mf_l3 = _sum_x(w_x5 * mask3, f_delta[:, :, 3, :, :], strict=True)
    else:
        sum_mf_l3 = jnp.zeros_like(dens)
    mf_before_vm = mf_factor_vm[:, None, None] * surface.b_hat[None, :, :] * (
        (16.0 / 15.0) * fac_vm[None, :, :] * sum_mf_l1 + (4.0 / 35.0) * fac_vm[None, :, :] * sum_mf_l3
    )
    mf_vm_psi_hat = _sum_tz(tw, zw, mf_before_vm)
    # Momentum vm0 vanishes identically for the L=0-only Maxwellian f0:
    mf_before_vm0 = jnp.zeros_like(mf_before_vm)
    mf_vm0_psi_hat = jnp.zeros_like(mf_vm_psi_hat)

    # Per-x decompositions (sum over x to the surface-integrated values):
    pf_before_x = pf_factor_vm[:, None, None, None] * (
        (8.0 / 3.0) * fac_vm[None, None, :, :] * (f_full_l0 * (w_x4 * mask0)[None, :, None, None])
        + (4.0 / 15.0) * fac_vm[None, None, :, :] * (f_full_l2 * (w_x4 * mask2)[None, :, None, None])
    )
    pf_vs_x = _sum_tz_sx(tw, zw, pf_before_x)
    hf_before_x = hf_factor_vm[:, None, None, None] * (
        (8.0 / 3.0) * fac_vm[None, None, :, :] * (f_full_l0 * (w_x6 * mask0)[None, :, None, None])
        + (4.0 / 15.0) * fac_vm[None, None, :, :] * (f_full_l2 * (w_x6 * mask2)[None, :, None, None])
    )
    hf_vs_x = _sum_tz_sx(tw, zw, hf_before_x)
    if layout.n_xi > 1:
        flow_x = flow_factor[:, None, None, None] * (f_delta[:, :, 1, :, :] * (w_x3 * mask1)[None, :, None, None])
        flow_vs_x = (
            _sum_tz_sx(
                tw, zw, flow_x * surface.b_hat[None, None, :, :] / surface.d_hat[None, None, :, :], strict=True
            )
            / vp
        )
    else:
        flow_vs_x = jnp.zeros((layout.n_species, layout.n_x), dtype=jnp.float64)

    # Totals, velocities, and current-like diagnostics:
    phi1_use = (
        jnp.zeros((layout.n_theta, layout.n_zeta), dtype=jnp.float64)
        if phi1_hat is None
        else jnp.asarray(phi1_hat, dtype=jnp.float64)
    )
    exp_phi1 = jnp.exp(-(z_s[:, None, None] * alpha / t_hat[:, None, None]) * phi1_use[None, :, :])
    total_density = n_hat[:, None, None] * exp_phi1 + dens
    total_pressure = n_hat[:, None, None] * exp_phi1 * t_hat[:, None, None] + pres
    vel_fsadens = flow / n_hat[:, None, None]
    mach = vel_fsadens * (sqrt_m[:, None, None] / sqrt_t[:, None, None])

    j_hat_tz = jnp.einsum("s,stz->tz", z_s, flow)
    b0, _g, _i = flux_surface_b_integrals(surface)
    fsab2 = jnp.asarray(surface.fsab_hat2, dtype=jnp.float64)
    fsab_j = jnp.einsum("s,s->", z_s, fsabflow)

    out: dict[str, jnp.ndarray] = {
        "densityPerturbation": dens,
        "pressurePerturbation": pres,
        "pressureAnisotropy": pres_aniso,
        "flow": flow,
        "FSADensityPerturbation": fsadens,
        "FSAPressurePerturbation": fsapres,
        "FSABFlow": fsabflow,
        "FSABFlow_vs_x": jnp.transpose(flow_vs_x, (1, 0)),
        "FSABVelocityUsingFSADensity": fsabflow / n_hat,
        "FSABVelocityUsingFSADensityOverB0": (fsabflow / n_hat) / b0,
        "FSABVelocityUsingFSADensityOverRootFSAB2": (fsabflow / n_hat) / jnp.sqrt(fsab2),
        "FSABjHat": fsab_j,
        "FSABjHatOverB0": fsab_j / b0,
        "FSABjHatOverRootFSAB2": fsab_j / jnp.sqrt(fsab2),
        "totalDensity": total_density,
        "totalPressure": total_pressure,
        "velocityUsingFSADensity": vel_fsadens,
        "velocityUsingTotalDensity": flow / total_density,
        "MachUsingFSAThermalSpeed": mach,
        "jHat": j_hat_tz,
        "particleFluxBeforeSurfaceIntegral_vm": pf_before_vm,
        "particleFluxBeforeSurfaceIntegral_vm0": pf_before_vm0,
        "heatFluxBeforeSurfaceIntegral_vm": hf_before_vm,
        "heatFluxBeforeSurfaceIntegral_vm0": hf_before_vm0,
        "momentumFluxBeforeSurfaceIntegral_vm": mf_before_vm,
        "momentumFluxBeforeSurfaceIntegral_vm0": mf_before_vm0,
        "particleFlux_vm_psiHat": pf_vm_psi_hat,
        "particleFlux_vm0_psiHat": pf_vm0_psi_hat,
        "heatFlux_vm_psiHat": hf_vm_psi_hat,
        "heatFlux_vm0_psiHat": hf_vm0_psi_hat,
        "momentumFlux_vm_psiHat": mf_vm_psi_hat,
        "momentumFlux_vm0_psiHat": mf_vm0_psi_hat,
        "particleFlux_vm_psiHat_vs_x": jnp.transpose(pf_vs_x, (1, 0)),
        "heatFlux_vm_psiHat_vs_x": jnp.transpose(hf_vs_x, (1, 0)),
    }
    sources = layout.sources(x_full)
    if sources is not None:
        out["sources"] = sources

    # vE / NTV placeholders (computed by the dedicated functions below):
    for key in (
        "particleFluxBeforeSurfaceIntegral_vE",
        "particleFluxBeforeSurfaceIntegral_vE0",
        "heatFluxBeforeSurfaceIntegral_vE",
        "heatFluxBeforeSurfaceIntegral_vE0",
        "momentumFluxBeforeSurfaceIntegral_vE",
        "momentumFluxBeforeSurfaceIntegral_vE0",
        "NTVBeforeSurfaceIntegral",
    ):
        out[key] = jnp.zeros_like(pf_before_vm)
    out["NTV"] = jnp.zeros((layout.n_species,), dtype=jnp.float64)
    return out

# ---- vE (ExB-drift) flux moments — Phi1 runs (diagnostics.F90 vE family) ----

class ElectricDriftFluxMoments(NamedTuple):
    """vE/vE0 flux moments of one Phi1 iterate (diagnostics.F90 ExB family)."""

    particle_flux_before_surface_integral_ve: jnp.ndarray  # (S,T,Z)
    particle_flux_before_surface_integral_ve0: jnp.ndarray  # (S,T,Z)
    heat_flux_before_surface_integral_ve: jnp.ndarray  # (S,T,Z)
    heat_flux_before_surface_integral_ve0: jnp.ndarray  # (S,T,Z)
    momentum_flux_before_surface_integral_ve: jnp.ndarray  # (S,T,Z)
    momentum_flux_before_surface_integral_ve0: jnp.ndarray  # (S,T,Z)
    particle_flux_ve_psi_hat: jnp.ndarray  # (S,)
    particle_flux_ve0_psi_hat: jnp.ndarray  # (S,)
    heat_flux_ve_psi_hat: jnp.ndarray  # (S,)
    heat_flux_ve0_psi_hat: jnp.ndarray  # (S,)
    momentum_flux_ve_psi_hat: jnp.ndarray  # (S,)
    momentum_flux_ve0_psi_hat: jnp.ndarray  # (S,)

def electric_drift_flux_moments(
    layout: StateLayout,
    vgrid: VelocityGrid,
    surface: FluxSurface,
    species: SpeciesParams,
    x_full: jnp.ndarray,
    *,
    delta: jnp.ndarray | float,
    alpha: jnp.ndarray | float,
    phi1_hat: jnp.ndarray,
    dphi1_hat_dtheta: jnp.ndarray,
    dphi1_hat_dzeta: jnp.ndarray,
) -> ElectricDriftFluxMoments:
    """ExB (vE) particle/heat/momentum flux moments for one Phi1 iterate.

    The geometric factor is ``(B_theta dPhi1/dzeta - B_zeta dPhi1/dtheta)/BHat^2``
    and f0 is evaluated at the same Phi1 iterate (diagnostics.F90 vE terms;
    previously inline in ``outputs/writer.py``).  The vE0 variants use f0 only;
    the momentum vE0 flux vanishes identically for the L=0-only f0.
    """
    f_delta = layout.f_delta(x_full)
    phi1 = jnp.asarray(phi1_hat, dtype=jnp.float64)
    dpt = jnp.asarray(dphi1_hat_dtheta, dtype=jnp.float64)
    dpz = jnp.asarray(dphi1_hat_dzeta, dtype=jnp.float64)
    f0_l0 = maxwellian_f0_l0(
        species, vgrid.x, alpha=alpha, phi1_hat=phi1, n_theta=layout.n_theta, n_zeta=layout.n_zeta
    )
    f_full_l0 = f_delta[:, :, 0, :, :] + f0_l0

    w2d = surface.theta_weights[:, None] * surface.zeta_weights[None, :]
    vp = jnp.sum(w2d / surface.d_hat)
    delta = jnp.asarray(delta, dtype=jnp.float64)
    alpha = jnp.asarray(alpha, dtype=jnp.float64)
    t_hat, m_hat = species.t_hat, species.m_hat
    sqrt_t = jnp.sqrt(t_hat)
    sqrt_m = jnp.sqrt(m_hat)

    w_pf = vgrid.x_weights * (vgrid.x**2)
    w_hf = vgrid.x_weights * (vgrid.x**4)
    w_mf = vgrid.x_weights * (vgrid.x**3)
    pf_factor = 2.0 * alpha * jnp.pi * delta * t_hat * sqrt_t / (vp * m_hat * sqrt_m)
    hf_factor = alpha * jnp.pi * delta * (t_hat * t_hat) * sqrt_t / (vp * m_hat * sqrt_m)
    mf_factor = 2.0 * alpha * jnp.pi * delta * (t_hat * t_hat) / (vp * m_hat)

    factor_ve = (surface.b_hat_sub_theta * dpz - surface.b_hat_sub_zeta * dpt) / (surface.b_hat * surface.b_hat)

    pf_before = pf_factor[:, None, None] * factor_ve[None, :, :] * jnp.einsum("x,sxtz->stz", w_pf, f_full_l0)
    pf_before0 = pf_factor[:, None, None] * factor_ve[None, :, :] * jnp.einsum("x,sxtz->stz", w_pf, f0_l0)
    hf_before = hf_factor[:, None, None] * factor_ve[None, :, :] * jnp.einsum("x,sxtz->stz", w_hf, f_full_l0)
    hf_before0 = hf_factor[:, None, None] * factor_ve[None, :, :] * jnp.einsum("x,sxtz->stz", w_hf, f0_l0)
    mf_before = (
        (2.0 / 3.0)
        * mf_factor[:, None, None]
        * factor_ve[None, :, :]
        * surface.b_hat[None, :, :]
        * jnp.einsum("x,sxtz->stz", w_mf, f_delta[:, :, 1, :, :])
    )
    mf_before0 = jnp.zeros_like(mf_before)

    return ElectricDriftFluxMoments(
        particle_flux_before_surface_integral_ve=pf_before,
        particle_flux_before_surface_integral_ve0=pf_before0,
        heat_flux_before_surface_integral_ve=hf_before,
        heat_flux_before_surface_integral_ve0=hf_before0,
        momentum_flux_before_surface_integral_ve=mf_before,
        momentum_flux_before_surface_integral_ve0=mf_before0,
        particle_flux_ve_psi_hat=jnp.einsum("tz,stz->s", w2d, pf_before),
        particle_flux_ve0_psi_hat=jnp.einsum("tz,stz->s", w2d, pf_before0),
        heat_flux_ve_psi_hat=jnp.einsum("tz,stz->s", w2d, hf_before),
        heat_flux_ve0_psi_hat=jnp.einsum("tz,stz->s", w2d, hf_before0),
        momentum_flux_ve_psi_hat=jnp.einsum("tz,stz->s", w2d, mf_before),
        momentum_flux_ve0_psi_hat=jnp.einsum("tz,stz->s", w2d, mf_before0),
    )

def combined_drift_fluxes(
    *,
    flux_vm_psi_hat: jnp.ndarray,
    flux_ve0_psi_hat: jnp.ndarray,
    flux_ve_psi_hat: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Total-drift flux variants ``(vd1, vd) = (vm + vE0, vm + vE)`` (diagnostics.F90)."""
    return flux_vm_psi_hat + flux_ve0_psi_hat, flux_vm_psi_hat + flux_ve_psi_hat

def heat_flux_without_phi1(*, heat_flux_vm_psi_hat: jnp.ndarray, heat_flux_ve0_psi_hat: jnp.ndarray) -> jnp.ndarray:
    """``heatFlux_withoutPhi1 = heatFlux_vm + (5/3) heatFlux_vE0`` (diagnostics.F90)."""
    return heat_flux_vm_psi_hat + (5.0 / 3.0) * heat_flux_ve0_psi_hat

# ---- NTV torque ------------------------------------------------------------

def ntv_kernel(
    surface: FluxSurface,
    *,
    u_hat: jnp.ndarray,
    g_hat: jnp.ndarray | float,
    i_hat: jnp.ndarray | float,
    iota: jnp.ndarray | float,
) -> jnp.ndarray:
    """NTV geometric kernel (v3 geometry.F90; ``invFSA_BHat2 = 1/FSABHat2``)."""
    b = surface.b_hat
    inv_fsa_b2 = 1.0 / jnp.asarray(surface.fsab_hat2, dtype=jnp.float64)
    g_hat = jnp.asarray(g_hat, dtype=jnp.float64)
    i_hat = jnp.asarray(i_hat, dtype=jnp.float64)
    iota = jnp.asarray(iota, dtype=jnp.float64)
    return (2.0 / 5.0) / b * (
        (jnp.asarray(u_hat, dtype=jnp.float64) - g_hat * inv_fsa_b2) * (iota * surface.db_hat_dtheta + surface.db_hat_dzeta)
        + iota * (1.0 / (b * b)) * (g_hat * surface.db_hat_dtheta - i_hat * surface.db_hat_dzeta)
    )

def ntv_moments(
    layout: StateLayout,
    vgrid: VelocityGrid,
    surface: FluxSurface,
    species: SpeciesParams,
    x_full: jnp.ndarray,
    *,
    kernel: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """NTV torque from the L=2 moment of f1 (diagnostics.F90).

    Returns ``(NTVBeforeSurfaceIntegral (S,T,Z), NTV (S,))``.  Pass the kernel
    from :func:`ntv_kernel`; for geometries without uHat (VMEC scheme 5) v3
    writes NTV = 0 — do so by passing a zero kernel or skipping the call.
    """
    weights_2d = surface.theta_weights[:, None] * surface.zeta_weights[None, :]
    vp = jnp.sum(weights_2d / surface.d_hat)
    t_hat, m_hat = species.t_hat, species.m_hat
    sqrt_t = jnp.sqrt(t_hat)
    sqrt_m = jnp.sqrt(m_hat)
    if layout.n_xi <= 2:
        before = jnp.zeros((layout.n_species, layout.n_theta, layout.n_zeta), dtype=jnp.float64)
        return before, jnp.zeros((layout.n_species,), dtype=jnp.float64)
    f_delta = layout.f_delta(x_full)
    w_ntv = vgrid.x_weights * (vgrid.x**4)
    sum_ntv = jnp.einsum("x,sxtz->stz", w_ntv, f_delta[:, :, 2, :, :])
    before = (
        (4.0 * jnp.pi * (t_hat * t_hat) * sqrt_t / (m_hat * sqrt_m * vp))[:, None, None]
        * jnp.asarray(kernel, dtype=jnp.float64)[None, :, :]
        * sum_ntv
    )
    return before, jnp.einsum("tz,stz->s", weights_2d, before)

# ---- RHSMode=2/3 transport matrix (monoenergetic and Onsager forms) --------

def transport_matrix_size(rhs_mode: int) -> int:
    """3 for RHSMode=2 (Onsager), 2 for RHSMode=3 (monoenergetic/DKES)."""
    if int(rhs_mode) == 2:
        return 3
    if int(rhs_mode) == 3:
        return 2
    raise ValueError("transport matrix is only defined for RHSMode=2 or RHSMode=3.")

def _effective_flux_functions(
    surface: FluxSurface,
    *,
    g_hat: jnp.ndarray,
    i_hat: jnp.ndarray,
    b0_over_bbar: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Replace ~0 placeholder flux functions by the computeBIntegrals values."""
    b0_eff, g_eff, i_eff = flux_surface_b_integrals(surface)
    b0 = jnp.where(jnp.abs(b0_over_bbar) < 1e-30, b0_eff, b0_over_bbar)
    g = jnp.where(jnp.abs(g_hat) < 1e-30, g_eff, g_hat)
    i = jnp.where(jnp.abs(i_hat) < 1e-30, i_eff, i_hat)
    return b0, g, i

def transport_matrix_from_flux_arrays(
    *,
    rhs_mode: int,
    surface: FluxSurface,
    species: SpeciesParams,
    delta: jnp.ndarray | float,
    alpha: jnp.ndarray | float,
    g_hat: jnp.ndarray | float,
    i_hat: jnp.ndarray | float,
    iota: jnp.ndarray | float,
    b0_over_bbar: jnp.ndarray | float,
    particle_flux_vm_psi_hat: jnp.ndarray,  # (S,N)
    heat_flux_vm_psi_hat: jnp.ndarray,  # (S,N)
    fsab_flow: jnp.ndarray,  # (S,N)
) -> jnp.ndarray:
    """Assemble the RHSMode=2/3 transport matrix from per-whichRHS flux arrays.

    Implements the ``transportMatrix`` entries of ``diagnostics.F90`` (v3 uses
    ``ispecies=1``).  RHSMode=3 gives the 2x2 monoenergetic (DKES-normalized)
    matrix; RHSMode=2 the 3x3 Onsager matrix.  Placeholder ``g_hat`` or
    ``b0_over_bbar`` values (|.| < 1e-30, VMEC scheme 5) are replaced by the
    computeBIntegrals values.
    """
    n = transport_matrix_size(rhs_mode)
    s_count = int(species.z_s.shape[0])
    for name, arr in (
        ("particle_flux_vm_psi_hat", particle_flux_vm_psi_hat),
        ("heat_flux_vm_psi_hat", heat_flux_vm_psi_hat),
        ("fsab_flow", fsab_flow),
    ):
        if arr.shape != (s_count, n):
            raise ValueError(f"{name} expected shape {(s_count, n)}, got {arr.shape}")

    n_hat = jnp.asarray(species.n_hat[0], dtype=jnp.float64)
    t_hat = jnp.asarray(species.t_hat[0], dtype=jnp.float64)
    m_hat = jnp.asarray(species.m_hat[0], dtype=jnp.float64)
    z = jnp.asarray(species.z_s[0], dtype=jnp.float64)
    sqrt_t = jnp.sqrt(t_hat)
    sqrt_m = jnp.sqrt(m_hat)
    delta = jnp.asarray(delta, dtype=jnp.float64)
    alpha = jnp.asarray(alpha, dtype=jnp.float64)
    iota = jnp.asarray(iota, dtype=jnp.float64)
    fsab_hat2 = jnp.asarray(surface.fsab_hat2, dtype=jnp.float64)

    g_hat = jnp.asarray(g_hat, dtype=jnp.float64)
    i_hat = jnp.asarray(i_hat, dtype=jnp.float64)
    b0_over_bbar = jnp.asarray(b0_over_bbar, dtype=jnp.float64)
    # Placeholder substitution as a trace-safe select (jit-compatible); the
    # values are bit-identical to the former Python branch: substitution is
    # applied only when GHat or B0OverBBar carry the ~0 VMEC placeholders.
    placeholder = (jnp.abs(g_hat) < 1e-30) | (jnp.abs(b0_over_bbar) < 1e-30)
    b0_eff, g_eff, i_eff = _effective_flux_functions(
        surface, g_hat=g_hat, i_hat=i_hat, b0_over_bbar=b0_over_bbar
    )
    g_hat = jnp.where(placeholder, g_eff, g_hat)
    i_hat = jnp.where(placeholder, i_eff, i_hat)
    b0_over_bbar = jnp.where(placeholder, b0_eff, b0_over_bbar)
    g_plus = g_hat + iota * i_hat

    pf = jnp.asarray(particle_flux_vm_psi_hat[0, :], dtype=jnp.float64)
    hf = jnp.asarray(heat_flux_vm_psi_hat[0, :], dtype=jnp.float64)
    flow = jnp.asarray(fsab_flow[0, :], dtype=jnp.float64)

    def _pf_gradient_entry(pf_w: jnp.ndarray, extra_denom: jnp.ndarray) -> jnp.ndarray:
        return (4.0 / (delta * delta)) * (sqrt_t / sqrt_m) * (z * z) * g_plus * pf_w * b0_over_bbar / (
            extra_denom * g_hat * g_hat
        )

    def _hf_gradient_entry(hf_w: jnp.ndarray, extra_denom: jnp.ndarray) -> jnp.ndarray:
        return (8.0 / (delta * delta)) * (sqrt_t / sqrt_m) * (z * z) * g_plus * hf_w * b0_over_bbar / (
            extra_denom * g_hat * g_hat
        )

    if int(rhs_mode) == 3:
        col1 = jnp.array(
            [_pf_gradient_entry(pf[0], t_hat * t_hat), 2.0 * z * flow[0] / (delta * g_hat * t_hat)],
            dtype=jnp.float64,
        )
        col2 = jnp.array(
            [
                pf[1] * 2.0 * fsab_hat2 / (n_hat * alpha * delta * g_hat),
                flow[1] * sqrt_t * sqrt_m * fsab_hat2 / (g_plus * alpha * z * n_hat * b0_over_bbar),
            ],
            dtype=jnp.float64,
        )
        return jnp.stack([col1, col2], axis=1)

    col1 = jnp.array(
        [
            _pf_gradient_entry(pf[0], t_hat * t_hat),
            _hf_gradient_entry(hf[0], t_hat * t_hat * t_hat),
            2.0 * z * flow[0] / (delta * g_hat * t_hat),
        ],
        dtype=jnp.float64,
    )
    col2 = jnp.array(
        [
            _pf_gradient_entry(pf[1], n_hat * t_hat),
            _hf_gradient_entry(hf[1], n_hat * t_hat * t_hat),
            2.0 * z * flow[1] / (delta * g_hat * n_hat),
        ],
        dtype=jnp.float64,
    )
    col3 = jnp.array(
        [
            pf[2] * 2.0 * fsab_hat2 / (n_hat * alpha * delta * g_hat),
            hf[2] * 4.0 * fsab_hat2 / (n_hat * t_hat * alpha * delta * g_hat),
            flow[2] * sqrt_t * sqrt_m * fsab_hat2 / (g_plus * alpha * z * n_hat * b0_over_bbar),
        ],
        dtype=jnp.float64,
    )
    return jnp.stack([col1, col2, col3], axis=1)

def _states_to_stack(layout: StateLayout, state_vectors: Any, n: int) -> jnp.ndarray:
    """Accept ``{whichRHS: vec}`` (1-based) or an ``(N,total)`` array."""
    if isinstance(state_vectors, dict):
        for which_rhs in range(1, n + 1):
            if which_rhs not in state_vectors:
                raise ValueError(f"Missing state vector for which_rhs={which_rhs}.")
        stack = jnp.stack(
            [jnp.asarray(state_vectors[w], dtype=jnp.float64) for w in range(1, n + 1)], axis=0
        )
    else:
        stack = jnp.asarray(state_vectors, dtype=jnp.float64)
    if stack.shape != (n, layout.total_size):
        raise ValueError(f"expected state stack of shape {(n, layout.total_size)}, got {stack.shape}")
    return stack

def transport_matrix_from_state_vectors(
    layout: StateLayout,
    vgrid: VelocityGrid,
    surface: FluxSurface,
    species: SpeciesParams,
    state_vectors: Any,
    *,
    rhs_mode: int,
    delta: jnp.ndarray | float,
    alpha: jnp.ndarray | float,
    g_hat: jnp.ndarray | float,
    i_hat: jnp.ndarray | float,
    iota: jnp.ndarray | float,
    b0_over_bbar: jnp.ndarray | float,
    phi1_hat: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Assemble the RHSMode=2/3 transport matrix from solved whichRHS states."""
    n = transport_matrix_size(rhs_mode)
    stack = _states_to_stack(layout, state_vectors, n)
    diag = _vm_flux_moments_batch_plain_jit(
        layout, vgrid, surface, species, stack, jnp.asarray(delta), jnp.asarray(alpha), phi1_hat
    )
    return transport_matrix_from_flux_arrays(
        rhs_mode=rhs_mode,
        surface=surface,
        species=species,
        delta=delta,
        alpha=alpha,
        g_hat=g_hat,
        i_hat=i_hat,
        iota=iota,
        b0_over_bbar=b0_over_bbar,
        particle_flux_vm_psi_hat=jnp.transpose(diag.particle_flux_vm_psi_hat, (1, 0)),
        heat_flux_vm_psi_hat=jnp.transpose(diag.heat_flux_vm_psi_hat, (1, 0)),
        fsab_flow=jnp.transpose(diag.fsab_flow, (1, 0)),
    )

def transport_moments_table(
    layout: StateLayout,
    vgrid: VelocityGrid,
    surface: FluxSurface,
    species: SpeciesParams,
    state_vectors: Any,
    *,
    rhs_mode: int,
    delta: jnp.ndarray | float,
    alpha: jnp.ndarray | float,
    phi1_hat: jnp.ndarray | None = None,
) -> dict[str, jnp.ndarray]:
    """RHSMode=2/3 diagnostic-field table used by scan postprocessing scripts.

    Returns the same keys and array orders as reading a Fortran
    ``sfincsOutput.h5`` in Python (flows/fluxes ``(S,N)``, before-surface
    integrals ``(Z,T,S,N)``, per-x arrays ``(X,S,N)``), matching the legacy
    ``v3_transport_output_fields_vm_only``.  vE fields are zero (the
    RHSMode=2/3 fixtures have no Phi1/Er drive).
    """
    n = transport_matrix_size(rhs_mode)
    stack = _states_to_stack(layout, state_vectors, n)
    diag = _vm_flux_moments_batch_precomputed(
        layout, vgrid, surface, species, stack, jnp.asarray(delta), jnp.asarray(alpha), phi1_hat
    )

    flow = jnp.transpose(diag.fsab_flow, (1, 0))  # (S,N)
    w2d = surface.theta_weights[:, None] * surface.zeta_weights[None, :]
    pf_before_vm0_nstz = diag.particle_flux_before_surface_integral_vm0
    hf_before_vm0_nstz = diag.heat_flux_before_surface_integral_vm0
    b0, _g, _i = flux_surface_b_integrals(surface)
    fsab2 = jnp.asarray(surface.fsab_hat2, dtype=jnp.float64)
    jhat = jnp.einsum("s,sn->n", species.z_s, flow)

    out: dict[str, jnp.ndarray] = {
        "FSABFlow": flow,
        "FSABFlow_vs_x": jnp.transpose(diag.fsab_flow_vs_x, (1, 2, 0)),
        "FSABVelocityUsingFSADensity": flow / species.n_hat[:, None],
        "FSABVelocityUsingFSADensityOverB0": (flow / species.n_hat[:, None]) / b0,
        "FSABVelocityUsingFSADensityOverRootFSAB2": (flow / species.n_hat[:, None]) / jnp.sqrt(fsab2),
        "FSABjHat": jhat,
        "FSABjHatOverB0": jhat / b0,
        "FSABjHatOverRootFSAB2": jhat / jnp.sqrt(fsab2),
        "particleFlux_vm_psiHat": jnp.transpose(diag.particle_flux_vm_psi_hat, (1, 0)),
        "heatFlux_vm_psiHat": jnp.transpose(diag.heat_flux_vm_psi_hat, (1, 0)),
        "particleFlux_vm0_psiHat": jnp.einsum("tz,nstz->sn", w2d, pf_before_vm0_nstz),
        "heatFlux_vm0_psiHat": jnp.einsum("tz,nstz->sn", w2d, hf_before_vm0_nstz),
        "particleFluxBeforeSurfaceIntegral_vm": jnp.transpose(diag.particle_flux_before_surface_integral_vm, (3, 2, 1, 0)),
        "heatFluxBeforeSurfaceIntegral_vm": jnp.transpose(diag.heat_flux_before_surface_integral_vm, (3, 2, 1, 0)),
        "particleFluxBeforeSurfaceIntegral_vm0": jnp.transpose(pf_before_vm0_nstz, (3, 2, 1, 0)),
        "heatFluxBeforeSurfaceIntegral_vm0": jnp.transpose(hf_before_vm0_nstz, (3, 2, 1, 0)),
        "particleFlux_vm_psiHat_vs_x": jnp.transpose(diag.particle_flux_vm_psi_hat_vs_x, (1, 2, 0)),
        "heatFlux_vm_psiHat_vs_x": jnp.transpose(diag.heat_flux_vm_psi_hat_vs_x, (1, 2, 0)),
    }
    zeros_ztsn = jnp.zeros((layout.n_zeta, layout.n_theta, layout.n_species, n), dtype=jnp.float64)
    for key in (
        "particleFluxBeforeSurfaceIntegral_vE",
        "heatFluxBeforeSurfaceIntegral_vE",
        "particleFluxBeforeSurfaceIntegral_vE0",
        "heatFluxBeforeSurfaceIntegral_vE0",
    ):
        out[key] = zeros_ztsn
    if layout.constraint_scheme in {1, 2, 3, 4}:
        sources = jnp.stack([layout.sources(stack[i]) for i in range(n)], axis=-1)
        out["sources"] = sources
    return out

# ---- Classical transport (classicalTransport.F90) --------------------------

def classical_fluxes(
    *,
    use_phi1: bool,
    surface: FluxSurface,
    species: SpeciesParams,
    gpsipsi: jnp.ndarray,  # (T,Z), the |grad psiHat|^2 metric (gpsiHatpsiHat)
    phi1_hat: jnp.ndarray,  # (T,Z), ignored when use_phi1=False
    alpha: jnp.ndarray | float,
    delta: jnp.ndarray | float,
    nu_n: jnp.ndarray | float,
    dn_hat_dpsi_hat: jnp.ndarray,  # (S,)
    dt_hat_dpsi_hat: jnp.ndarray,  # (S,)
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Classical particle/heat fluxes projected on ``psiHat``.

    Matches ``classicalTransport.F90:calculateClassicalFlux`` (Braginskii-type
    friction moments summed over species pairs).  Returns ``(S,)`` arrays; the
    heat flux is the "total" definition including the ``5/2 T Gamma``-like
    convective piece exactly as in v3.
    """
    gpsipsi = jnp.asarray(gpsipsi, dtype=jnp.float64)
    vp = vprime_hat(surface)
    alpha = jnp.asarray(alpha, dtype=jnp.float64)
    phi1_hat = jnp.asarray(phi1_hat, dtype=jnp.float64)
    delta = jnp.asarray(delta, dtype=jnp.float64)
    nu_n = jnp.asarray(nu_n, dtype=jnp.float64)
    z_s, m_hat, t_hat, n_hat = species.z_s, species.m_hat, species.t_hat, species.n_hat
    dn_hat_dpsi_hat = jnp.asarray(dn_hat_dpsi_hat, dtype=jnp.float64)
    dt_hat_dpsi_hat = jnp.asarray(dt_hat_dpsi_hat, dtype=jnp.float64)

    w = (surface.theta_weights[:, None] * surface.zeta_weights[None, :]) / surface.d_hat
    integrand = w * (gpsipsi / (surface.b_hat * surface.b_hat))

    s = int(z_s.shape[0])
    xab2 = (m_hat[:, None] * t_hat[None, :]) / (m_hat[None, :] * t_hat[:, None])
    m_ratio = m_hat[:, None] / m_hat[None, :]
    one_plus_x = 1.0 + xab2
    denom = one_plus_x ** (2.5)

    # Braginskii friction matrix elements M^{ab}_{jk} (classicalTransport.F90):
    mab00 = -((1.0 + m_ratio) * one_plus_x) / denom
    mab01 = -(1.5) * (1.0 + m_ratio) / denom
    mab11 = -(13.0 + 16.0 * xab2 + 30.0 * (xab2**2)) / 4.0 / denom
    nab11 = (27.0 * m_ratio) / 4.0 / denom

    if use_phi1:
        coef_ab = alpha * ((z_s / t_hat)[:, None] + (z_s / t_hat)[None, :])
        exp_ab = jnp.exp(-coef_ab[:, :, None, None] * phi1_hat[None, None, :, :])
        geom1 = jnp.einsum("tz,abtz->ab", integrand, exp_ab) / vp
        geom2 = jnp.einsum("tz,abtz->ab", integrand * phi1_hat, exp_ab) / vp
    else:
        geom1 = jnp.broadcast_to(jnp.sum(integrand) / vp, (s, s))
        geom2 = jnp.zeros((s, s), dtype=jnp.float64)

    geom1 = geom1 * (n_hat[:, None] * n_hat[None, :])
    geom2 = geom2 * (n_hat[:, None] * n_hat[None, :])

    u_dn = (t_hat * dn_hat_dpsi_hat) / (n_hat * z_s)
    u_dt = dt_hat_dpsi_hat / t_hat
    u_dt_over_z = dt_hat_dpsi_hat / z_s
    term_dn = u_dn[:, None] - u_dn[None, :]
    term_dt = u_dt[:, None] - u_dt[None, :]

    pf_ab = (
        geom1 * mab00 * term_dn
        + geom2 * alpha * mab00 * term_dt
        + geom1 * ((mab00 - mab01) * u_dt_over_z[:, None] - (mab00 - xab2 * mab01) * u_dt_over_z[None, :])
    )
    hf_ab = (
        geom1 * mab01 * term_dn
        + geom2 * alpha * mab01 * term_dt
        + geom1 * ((mab01 - mab11) * u_dt_over_z[:, None] - (mab01 + nab11) * u_dt_over_z[None, :])
    )

    z2_b = z_s[None, :] ** 2
    pf_a = jnp.sum(z2_b * pf_ab, axis=1)
    hf_a = jnp.sum(z2_b * hf_ab, axis=1)

    pf_a = z_s * (delta**2) * nu_n * jnp.sqrt(m_hat) * pf_a / (2.0 * (t_hat**1.5))
    hf_a = -z_s * (delta**2) * nu_n * jnp.sqrt(m_hat) * hf_a / (4.0 * jnp.sqrt(t_hat))
    hf_a = hf_a + 1.25 * t_hat * pf_a
    return pf_a, hf_a

# ---- Radial-coordinate flux variants (radialCoordinates.F90) ---------------


