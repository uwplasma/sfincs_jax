from __future__ import annotations

from dataclasses import dataclass

from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import lax

from .geometry import BoozerGeometry
from .v3_system import V3FullSystemOperator, with_transport_rhs_settings


@dataclass(frozen=True)
class V3TransportDiagnostics:
    """Subset of v3 diagnostics needed for RHSMode=2/3 transport matrices.

    All quantities are in the same normalized units used by v3 `diagnostics.F90`.
    """

    vprime_hat: jnp.ndarray  # scalar
    particle_flux_vm_psi_hat: jnp.ndarray  # (S,)
    heat_flux_vm_psi_hat: jnp.ndarray  # (S,)
    fsab_flow: jnp.ndarray  # (S,)
    # Additional fields used for `sfincsOutput.h5` parity and upstream postprocessing scripts:
    particle_flux_before_surface_integral_vm: jnp.ndarray  # (S,T,Z)
    heat_flux_before_surface_integral_vm: jnp.ndarray  # (S,T,Z)
    particle_flux_before_surface_integral_vm0: jnp.ndarray  # (S,T,Z)
    heat_flux_before_surface_integral_vm0: jnp.ndarray  # (S,T,Z)
    particle_flux_vm_psi_hat_vs_x: jnp.ndarray  # (X,S) contributions that sum to particle_flux_vm_psi_hat
    heat_flux_vm_psi_hat_vs_x: jnp.ndarray  # (X,S)
    fsab_flow_vs_x: jnp.ndarray  # (X,S) contributions that sum to fsab_flow


def _weighted_sum_x_fortran(w_x: jnp.ndarray, values_sxtz: jnp.ndarray) -> jnp.ndarray:
    """Compute Σ_x w_x[x] * values[:,x,:,:] using a deterministic x-loop order."""
    w_x = jnp.asarray(w_x, dtype=jnp.float64).reshape((-1,))
    values_sxtz = jnp.asarray(values_sxtz, dtype=jnp.float64)
    n_x = int(values_sxtz.shape[1])
    if w_x.shape[0] != n_x:
        raise ValueError(f"w_x has length {w_x.shape[0]}, expected {n_x}.")
    acc0 = jnp.zeros((values_sxtz.shape[0], values_sxtz.shape[2], values_sxtz.shape[3]), dtype=jnp.float64)

    def body(ix: int, acc: jnp.ndarray) -> jnp.ndarray:
        return acc + w_x[ix] * values_sxtz[:, ix, :, :]

    return lax.fori_loop(0, n_x, body, acc0)


def _weighted_sum_tz_fortran(w_t: jnp.ndarray, w_z: jnp.ndarray, values_stz: jnp.ndarray) -> jnp.ndarray:
    """Compute Σ_t Σ_z w_t[t] w_z[z] values[:,t,z] in explicit Fortran-like order."""
    w_t = jnp.asarray(w_t, dtype=jnp.float64).reshape((-1,))
    w_z = jnp.asarray(w_z, dtype=jnp.float64).reshape((-1,))
    values_stz = jnp.asarray(values_stz, dtype=jnp.float64)
    n_t = int(values_stz.shape[1])
    n_z = int(values_stz.shape[2])
    if w_t.shape[0] != n_t or w_z.shape[0] != n_z:
        raise ValueError(f"Weight shapes {(w_t.shape[0], w_z.shape[0])} do not match values {values_stz.shape}.")
    acc0 = jnp.zeros((values_stz.shape[0],), dtype=jnp.float64)

    def body_t(it: int, acc_t: jnp.ndarray) -> jnp.ndarray:
        def body_z(iz: int, acc_z: jnp.ndarray) -> jnp.ndarray:
            return acc_z + (w_t[it] * w_z[iz]) * values_stz[:, it, iz]

        return lax.fori_loop(0, n_z, body_z, acc_t)

    return lax.fori_loop(0, n_t, body_t, acc0)


def _weighted_sum_tz_fortran_sx(w_t: jnp.ndarray, w_z: jnp.ndarray, values_sxtz: jnp.ndarray) -> jnp.ndarray:
    """Compute Σ_t Σ_z w_t[t] w_z[z] values[:,x,t,z] in explicit order -> (S,X)."""
    w_t = jnp.asarray(w_t, dtype=jnp.float64).reshape((-1,))
    w_z = jnp.asarray(w_z, dtype=jnp.float64).reshape((-1,))
    values_sxtz = jnp.asarray(values_sxtz, dtype=jnp.float64)
    n_t = int(values_sxtz.shape[2])
    n_z = int(values_sxtz.shape[3])
    if w_t.shape[0] != n_t or w_z.shape[0] != n_z:
        raise ValueError(f"Weight shapes {(w_t.shape[0], w_z.shape[0])} do not match values {values_sxtz.shape}.")
    acc0 = jnp.zeros((values_sxtz.shape[0], values_sxtz.shape[1]), dtype=jnp.float64)

    def body_t(it: int, acc_t: jnp.ndarray) -> jnp.ndarray:
        def body_z(iz: int, acc_z: jnp.ndarray) -> jnp.ndarray:
            return acc_z + (w_t[it] * w_z[iz]) * values_sxtz[:, :, it, iz]

        return lax.fori_loop(0, n_z, body_z, acc_t)

    return lax.fori_loop(0, n_t, body_t, acc0)


def _vprime_hat_from_op(op: V3FullSystemOperator) -> jnp.ndarray:
    inv_d = jnp.asarray(1.0 / op.d_hat, dtype=jnp.float64)
    return _weighted_sum_tz_fortran(
        jnp.asarray(op.theta_weights, dtype=jnp.float64),
        jnp.asarray(op.zeta_weights, dtype=jnp.float64),
        inv_d[None, :, :],
    )[0]


def _flux_functions_from_op(op: V3FullSystemOperator) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return (B0OverBBar, GHat, IHat) computed from arrays, matching v3 `computeBIntegrals`.

    This is needed for geometries (notably VMEC `geometryScheme=5`) in which `sfincs_jax` stores
    the scalar flux functions as placeholders in the geometry struct but still needs the
    effective values for transport-matrix formulas and RHSMode=3 overwrites.
    """
    w2d = op.theta_weights[:, None] * op.zeta_weights[None, :]
    vprime_hat = _vprime_hat_from_op(op)
    fsab2 = jnp.asarray(op.fsab_hat2, dtype=jnp.float64)

    b0 = jnp.sum(w2d * (op.b_hat**3) / op.d_hat) / (vprime_hat * fsab2)
    denom = jnp.asarray(4.0 * jnp.pi * jnp.pi, dtype=jnp.float64)
    g_hat = jnp.sum(w2d * op.b_hat_sub_zeta) / denom
    i_hat = jnp.sum(w2d * op.b_hat_sub_theta) / denom
    return b0, g_hat, i_hat


def f0_v3_from_operator(op: V3FullSystemOperator) -> jnp.ndarray:
    """Compute v3 `f0` (Maxwellian) in the BLOCK_F layout.

    This matches v3 `populateMatrix.F90:init_f0`:

      f0(L=0) = exp(-Z*alpha*Phi1Hat/THat) * nHat*mHat/(pi*THat) * sqrt(mHat/(pi*THat)) * exp(-x^2)

    with all L>0 entries set to 0.
    """
    # Shape: (S, X, L, T, Z)
    out = jnp.zeros(op.fblock.f_shape, dtype=jnp.float64)

    x = jnp.asarray(op.x, dtype=jnp.float64)
    expx2 = jnp.exp(-(x * x))  # (X,)

    z = jnp.asarray(op.z_s, dtype=jnp.float64)  # (S,)
    n_hat = jnp.asarray(op.n_hat, dtype=jnp.float64)
    t_hat = jnp.asarray(op.t_hat, dtype=jnp.float64)
    m_hat = jnp.asarray(op.m_hat, dtype=jnp.float64)

    # (S, X)
    pref = n_hat[:, None] * m_hat[:, None] / (jnp.pi * t_hat[:, None])
    pref = pref * jnp.sqrt(m_hat[:, None] / (jnp.pi * t_hat[:, None]))
    pref = pref * expx2[None, :]

    phi1 = jnp.asarray(op.phi1_hat_base, dtype=jnp.float64)  # (T,Z)
    exp_phi1 = jnp.exp(-(z[:, None, None] * op.alpha / t_hat[:, None, None]) * phi1[None, :, :])  # (S,T,Z)

    out = out.at[:, :, 0, :, :].set(pref[:, :, None, None] * exp_phi1[:, None, :, :])
    return out


def v3_transport_diagnostics_vm_only(op: V3FullSystemOperator, *, x_full: jnp.ndarray) -> V3TransportDiagnostics:
    """Compute the subset of `diagnostics.F90` needed for RHSMode=2/3 transport matrices.

    Notes
    -----
    This implementation currently includes:
    - `particleFlux_vm_psiHat`
    - `heatFlux_vm_psiHat`
    - `FSABFlow`

    It deliberately omits vE terms, momentum flux, NTV, and classical terms since the RHSMode=2/3
    transport matrices in v3 only depend on the vm (magnetic drift) particle/heat fluxes and FSAB flow.
    """
    x_full = jnp.asarray(x_full, dtype=jnp.float64)
    if x_full.shape != (op.total_size,):
        raise ValueError(f"x_full must have shape {(op.total_size,)}, got {x_full.shape}")

    f_delta = x_full[: op.f_size].reshape(op.fblock.f_shape)  # (S,X,L,T,Z)
    f0 = f0_v3_from_operator(op)
    f_full = f_delta + f0

    vprime_hat = _vprime_hat_from_op(op)  # scalar

    theta_w = jnp.asarray(op.theta_weights, dtype=jnp.float64)
    zeta_w = jnp.asarray(op.zeta_weights, dtype=jnp.float64)
    w2d = theta_w[:, None] * zeta_w[None, :]  # (T,Z)
    factor_vm = (op.b_hat_sub_theta * op.db_hat_dzeta - op.b_hat_sub_zeta * op.db_hat_dtheta) / (op.b_hat * op.b_hat * op.b_hat)

    # x integral weights (diagnostics.F90):
    x = jnp.asarray(op.x, dtype=jnp.float64)
    xw = jnp.asarray(op.x_weights, dtype=jnp.float64)
    w_pf = xw * (x**4)
    w_hf = xw * (x**6)
    w_flow = xw * (x**3)

    # Per-species factors (diagnostics.F90):
    z = jnp.asarray(op.z_s, dtype=jnp.float64)
    t_hat = jnp.asarray(op.t_hat, dtype=jnp.float64)
    m_hat = jnp.asarray(op.m_hat, dtype=jnp.float64)

    sqrt_t = jnp.sqrt(t_hat)
    sqrt_m = jnp.sqrt(m_hat)

    particle_flux_factor_vm = jnp.pi * op.delta * (t_hat * t_hat) * sqrt_t / (z * vprime_hat * m_hat * sqrt_m)
    heat_flux_factor_vm = (
        jnp.pi * op.delta * (t_hat * t_hat * t_hat) * sqrt_t / (2.0 * z * vprime_hat * m_hat * sqrt_m)
    )
    flow_factor = 4.0 * jnp.pi * (t_hat * t_hat) / (3.0 * m_hat * m_hat)

    # L=0 and L=2 contributions:
    f_l0 = f_full[:, :, 0, :, :]  # (S,X,T,Z)
    if op.n_xi > 2:
        f_l2 = f_full[:, :, 2, :, :]
    else:
        f_l2 = jnp.zeros_like(f_l0)

    # Mask x points that don't include a given Legendre mode (Nxi_for_x):
    n_xi_for_x = jnp.asarray(op.fblock.collisionless.n_xi_for_x, dtype=jnp.int32)  # (X,)
    mask_l0 = (n_xi_for_x > 0).astype(jnp.float64)
    mask_l1 = (n_xi_for_x > 1).astype(jnp.float64)
    mask_l2 = (n_xi_for_x > 2).astype(jnp.float64)

    # Particle flux (vm):
    wpf0 = (w_pf * mask_l0).astype(jnp.float64)  # (X,)
    wpf2 = (w_pf * mask_l2).astype(jnp.float64)
    sum_pf_l0 = _weighted_sum_x_fortran(wpf0, f_l0)
    sum_pf_l2 = _weighted_sum_x_fortran(wpf2, f_l2)
    pf_before = particle_flux_factor_vm[:, None, None] * (
        (8.0 / 3.0) * factor_vm[None, :, :] * sum_pf_l0 + (4.0 / 15.0) * factor_vm[None, :, :] * sum_pf_l2
    )  # (S,T,Z)
    particle_flux_vm_psi_hat = _weighted_sum_tz_fortran(theta_w, zeta_w, pf_before)

    # Heat flux (vm):
    whf0 = (w_hf * mask_l0).astype(jnp.float64)
    whf2 = (w_hf * mask_l2).astype(jnp.float64)
    sum_hf_l0 = _weighted_sum_x_fortran(whf0, f_l0)
    sum_hf_l2 = _weighted_sum_x_fortran(whf2, f_l2)
    hf_before = heat_flux_factor_vm[:, None, None] * (
        (8.0 / 3.0) * factor_vm[None, :, :] * sum_hf_l0 + (4.0 / 15.0) * factor_vm[None, :, :] * sum_hf_l2
    )  # (S,T,Z)
    heat_flux_vm_psi_hat = _weighted_sum_tz_fortran(theta_w, zeta_w, hf_before)

    # FSABFlow:
    if op.n_xi > 1:
        f_l1 = f_delta[:, :, 1, :, :]
    else:
        f_l1 = jnp.zeros_like(f_l0)

    wf1 = (w_flow * mask_l1).astype(jnp.float64)
    sum_flow = _weighted_sum_x_fortran(wf1, f_l1)
    flow = flow_factor[:, None, None] * sum_flow
    fsab_flow = _weighted_sum_tz_fortran(theta_w, zeta_w, flow * op.b_hat[None, :, :] / op.d_hat[None, :, :]) / vprime_hat

    # vm0 contributions (use f0 only):
    f0_l0 = f0[:, :, 0, :, :]
    f0_l2 = f0[:, :, 2, :, :] if op.n_xi > 2 else jnp.zeros_like(f0_l0)
    sum_pf0_l0 = _weighted_sum_x_fortran(wpf0, f0_l0)
    sum_pf0_l2 = _weighted_sum_x_fortran(wpf2, f0_l2)
    pf_before_vm0 = particle_flux_factor_vm[:, None, None] * (
        (8.0 / 3.0) * factor_vm[None, :, :] * sum_pf0_l0 + (4.0 / 15.0) * factor_vm[None, :, :] * sum_pf0_l2
    )

    sum_hf0_l0 = _weighted_sum_x_fortran(whf0, f0_l0)
    sum_hf0_l2 = _weighted_sum_x_fortran(whf2, f0_l2)
    hf_before_vm0 = heat_flux_factor_vm[:, None, None] * (
        (8.0 / 3.0) * factor_vm[None, :, :] * sum_hf0_l0 + (4.0 / 15.0) * factor_vm[None, :, :] * sum_hf0_l2
    )

    # Contributions vs x (these sum over x to the surface-integrated fluxes):
    # Particle flux:
    pf_x_l0 = f_l0 * wpf0[None, :, None, None]
    pf_x_l2 = f_l2 * wpf2[None, :, None, None]
    pf_before_x = particle_flux_factor_vm[:, None, None, None] * (
        (8.0 / 3.0) * factor_vm[None, None, :, :] * pf_x_l0 + (4.0 / 15.0) * factor_vm[None, None, :, :] * pf_x_l2
    )  # (S,X,T,Z)
    pf_vs_x = _weighted_sum_tz_fortran_sx(theta_w, zeta_w, pf_before_x)  # (S,X)

    # Heat flux:
    hf_x_l0 = f_l0 * whf0[None, :, None, None]
    hf_x_l2 = f_l2 * whf2[None, :, None, None]
    hf_before_x = heat_flux_factor_vm[:, None, None, None] * (
        (8.0 / 3.0) * factor_vm[None, None, :, :] * hf_x_l0 + (4.0 / 15.0) * factor_vm[None, None, :, :] * hf_x_l2
    )  # (S,X,T,Z)
    hf_vs_x = _weighted_sum_tz_fortran_sx(theta_w, zeta_w, hf_before_x)  # (S,X)

    # Flow:
    flow_x = flow_factor[:, None, None, None] * (f_l1 * wf1[None, :, None, None])  # (S,X,T,Z)
    fsab_flow_vs_x = (
        _weighted_sum_tz_fortran_sx(theta_w, zeta_w, flow_x * op.b_hat[None, None, :, :] / op.d_hat[None, None, :, :])
        / vprime_hat
    )

    return V3TransportDiagnostics(
        vprime_hat=vprime_hat,
        particle_flux_vm_psi_hat=particle_flux_vm_psi_hat,
        heat_flux_vm_psi_hat=heat_flux_vm_psi_hat,
        fsab_flow=fsab_flow,
        particle_flux_before_surface_integral_vm=pf_before,
        heat_flux_before_surface_integral_vm=hf_before,
        particle_flux_before_surface_integral_vm0=pf_before_vm0,
        heat_flux_before_surface_integral_vm0=hf_before_vm0,
        particle_flux_vm_psi_hat_vs_x=jnp.transpose(pf_vs_x, (1, 0)),  # (X,S)
        heat_flux_vm_psi_hat_vs_x=jnp.transpose(hf_vs_x, (1, 0)),  # (X,S)
        fsab_flow_vs_x=jnp.transpose(fsab_flow_vs_x, (1, 0)),  # (X,S)
    )


def v3_rhsmode1_output_fields_vm_only(op: V3FullSystemOperator, *, x_full: jnp.ndarray) -> dict[str, jnp.ndarray]:
    """Compute a RHSMode=1 output subset from a solved state vector.

    This helper targets end-to-end `sfincsOutput.h5` parity for the smallest RHSMode=1
    fixtures. It currently includes the "vm-only" (magnetic drift) neoclassical
    contributions and writes vE / NTV-related quantities as 0 placeholders.

    The formulas mirror `sfincs/fortran/version3/diagnostics.F90`, including the
    `... / VPrimeHat` normalization of flux-surface averages.
    """
    x_full = jnp.asarray(x_full, dtype=jnp.float64)
    if x_full.shape != (op.total_size,):
        raise ValueError(f"x_full must have shape {(op.total_size,)}, got {x_full.shape}")

    f_delta = x_full[: op.f_size].reshape(op.fblock.f_shape)  # (S,X,L,T,Z)
    f0 = f0_v3_from_operator(op)
    f_full = f_delta + f0

    vprime_hat = _vprime_hat_from_op(op)  # scalar
    theta_w = jnp.asarray(op.theta_weights, dtype=jnp.float64)
    zeta_w = jnp.asarray(op.zeta_weights, dtype=jnp.float64)
    w2d = theta_w[:, None] * zeta_w[None, :]  # (T,Z)

    # Geometry factors:
    factor_vm = (op.b_hat_sub_theta * op.db_hat_dzeta - op.b_hat_sub_zeta * op.db_hat_dtheta) / (
        op.b_hat * op.b_hat * op.b_hat
    )  # (T,Z)

    # x integral weights:
    x = jnp.asarray(op.x, dtype=jnp.float64)  # (X,)
    xw = jnp.asarray(op.x_weights, dtype=jnp.float64)  # (X,)
    w_x2 = xw * (x**2)
    w_x3 = xw * (x**3)
    w_x4 = xw * (x**4)
    w_x5 = xw * (x**5)
    w_x6 = xw * (x**6)

    # Mask x points that don't include a given Legendre mode (Nxi_for_x):
    n_xi_for_x = jnp.asarray(op.fblock.collisionless.n_xi_for_x, dtype=jnp.int32)  # (X,)
    mask_l0 = (n_xi_for_x > 0).astype(jnp.float64)
    mask_l1 = (n_xi_for_x > 1).astype(jnp.float64)
    mask_l2 = (n_xi_for_x > 2).astype(jnp.float64)
    mask_l3 = (n_xi_for_x > 3).astype(jnp.float64)

    # Per-species factors:
    z = jnp.asarray(op.z_s, dtype=jnp.float64)  # (S,)
    n_hat = jnp.asarray(op.n_hat, dtype=jnp.float64)
    t_hat = jnp.asarray(op.t_hat, dtype=jnp.float64)
    m_hat = jnp.asarray(op.m_hat, dtype=jnp.float64)

    sqrt_t = jnp.sqrt(t_hat)
    sqrt_m = jnp.sqrt(m_hat)

    density_factor = 4.0 * jnp.pi * t_hat * sqrt_t / (m_hat * sqrt_m)
    flow_factor = 4.0 * jnp.pi * (t_hat * t_hat) / (3.0 * m_hat * m_hat)
    pressure_factor = 8.0 * jnp.pi * (t_hat * t_hat) * sqrt_t / (3.0 * m_hat * sqrt_m)

    particle_flux_factor_vm = jnp.pi * op.delta * (t_hat * t_hat) * sqrt_t / (z * vprime_hat * m_hat * sqrt_m)
    heat_flux_factor_vm = (
        jnp.pi * op.delta * (t_hat * t_hat * t_hat) * sqrt_t / (2.0 * z * vprime_hat * m_hat * sqrt_m)
    )
    momentum_flux_factor_vm = jnp.pi * op.delta * (t_hat * t_hat * t_hat) / (z * vprime_hat * m_hat)

    # Moments from delta-f:
    dens = density_factor[:, None, None] * _weighted_sum_x_fortran(w_x2 * mask_l0, f_delta[:, :, 0, :, :])
    pres = pressure_factor[:, None, None] * _weighted_sum_x_fortran(w_x4 * mask_l0, f_delta[:, :, 0, :, :])

    if op.n_xi > 1:
        flow = flow_factor[:, None, None] * _weighted_sum_x_fortran(w_x3 * mask_l1, f_delta[:, :, 1, :, :])
    else:
        flow = jnp.zeros_like(dens)

    if op.n_xi > 2:
        pres_aniso = pressure_factor[:, None, None] * (-3.0 / 5.0) * _weighted_sum_x_fortran(w_x4 * mask_l2, f_delta[:, :, 2, :, :])
    else:
        pres_aniso = jnp.zeros_like(dens)

    # Flux-surface averages (divide by VPrimeHat):
    fsadens = _weighted_sum_tz_fortran(theta_w, zeta_w, dens / op.d_hat[None, :, :]) / vprime_hat
    fsapres = _weighted_sum_tz_fortran(theta_w, zeta_w, pres / op.d_hat[None, :, :]) / vprime_hat
    fsabflow = _weighted_sum_tz_fortran(theta_w, zeta_w, flow * op.b_hat[None, :, :] / op.d_hat[None, :, :]) / vprime_hat

    # Particle / heat flux (vm): L=0 and L=2 using full-f.
    f_full_l0 = f_full[:, :, 0, :, :]
    f_full_l2 = f_full[:, :, 2, :, :] if op.n_xi > 2 else jnp.zeros_like(f_full_l0)

    sum_pf_l0 = _weighted_sum_x_fortran(w_x4 * mask_l0, f_full_l0)
    sum_pf_l2 = _weighted_sum_x_fortran(w_x4 * mask_l2, f_full_l2)
    pf_before_vm = particle_flux_factor_vm[:, None, None] * (
        (8.0 / 3.0) * factor_vm[None, :, :] * sum_pf_l0 + (4.0 / 15.0) * factor_vm[None, :, :] * sum_pf_l2
    )
    pf_vm_psi_hat = _weighted_sum_tz_fortran(theta_w, zeta_w, pf_before_vm)

    sum_hf_l0 = _weighted_sum_x_fortran(w_x6 * mask_l0, f_full_l0)
    sum_hf_l2 = _weighted_sum_x_fortran(w_x6 * mask_l2, f_full_l2)
    hf_before_vm = heat_flux_factor_vm[:, None, None] * (
        (8.0 / 3.0) * factor_vm[None, :, :] * sum_hf_l0 + (4.0 / 15.0) * factor_vm[None, :, :] * sum_hf_l2
    )
    hf_vm_psi_hat = _weighted_sum_tz_fortran(theta_w, zeta_w, hf_before_vm)

    # vm0 contributions (f0 only):
    f0_l0 = f0[:, :, 0, :, :]
    f0_l2 = f0[:, :, 2, :, :] if op.n_xi > 2 else jnp.zeros_like(f0_l0)

    sum_pf0_l0 = _weighted_sum_x_fortran(w_x4 * mask_l0, f0_l0)
    sum_pf0_l2 = _weighted_sum_x_fortran(w_x4 * mask_l2, f0_l2)
    pf_before_vm0 = particle_flux_factor_vm[:, None, None] * (
        (8.0 / 3.0) * factor_vm[None, :, :] * sum_pf0_l0 + (4.0 / 15.0) * factor_vm[None, :, :] * sum_pf0_l2
    )
    pf_vm0_psi_hat = _weighted_sum_tz_fortran(theta_w, zeta_w, pf_before_vm0)

    sum_hf0_l0 = _weighted_sum_x_fortran(w_x6 * mask_l0, f0_l0)
    sum_hf0_l2 = _weighted_sum_x_fortran(w_x6 * mask_l2, f0_l2)
    hf_before_vm0 = heat_flux_factor_vm[:, None, None] * (
        (8.0 / 3.0) * factor_vm[None, :, :] * sum_hf0_l0 + (4.0 / 15.0) * factor_vm[None, :, :] * sum_hf0_l2
    )
    hf_vm0_psi_hat = _weighted_sum_tz_fortran(theta_w, zeta_w, hf_before_vm0)

    # Momentum flux (vm): L=1 and L=3 using full-f.
    if op.n_xi > 1:
        sum_mf_l1 = _weighted_sum_x_fortran(w_x5 * mask_l1, f_full[:, :, 1, :, :])
    else:
        sum_mf_l1 = jnp.zeros_like(dens)
    if op.n_xi > 3:
        sum_mf_l3 = _weighted_sum_x_fortran(w_x5 * mask_l3, f_full[:, :, 3, :, :])
    else:
        sum_mf_l3 = jnp.zeros_like(dens)

    mf_before_vm = momentum_flux_factor_vm[:, None, None] * op.b_hat[None, :, :] * (
        (16.0 / 15.0) * factor_vm[None, :, :] * sum_mf_l1 + (4.0 / 35.0) * factor_vm[None, :, :] * sum_mf_l3
    )
    mf_vm_psi_hat = _weighted_sum_tz_fortran(theta_w, zeta_w, mf_before_vm)

    # Momentum vm0 is identically 0 for the current f0 definition (L>0=0), but keep it explicit.
    mf_before_vm0 = jnp.zeros_like(mf_before_vm)
    mf_vm0_psi_hat = jnp.zeros_like(mf_vm_psi_hat)

    # vs-x contributions:
    pf_x_l0 = f_full_l0 * (w_x4 * mask_l0)[None, :, None, None]
    pf_x_l2 = f_full_l2 * (w_x4 * mask_l2)[None, :, None, None]
    pf_before_x = particle_flux_factor_vm[:, None, None, None] * (
        (8.0 / 3.0) * factor_vm[None, None, :, :] * pf_x_l0 + (4.0 / 15.0) * factor_vm[None, None, :, :] * pf_x_l2
    )  # (S,X,T,Z)
    pf_vs_x = _weighted_sum_tz_fortran_sx(theta_w, zeta_w, pf_before_x)  # (S,X)

    hf_x_l0 = f_full_l0 * (w_x6 * mask_l0)[None, :, None, None]
    hf_x_l2 = f_full_l2 * (w_x6 * mask_l2)[None, :, None, None]
    hf_before_x = heat_flux_factor_vm[:, None, None, None] * (
        (8.0 / 3.0) * factor_vm[None, None, :, :] * hf_x_l0 + (4.0 / 15.0) * factor_vm[None, None, :, :] * hf_x_l2
    )
    hf_vs_x = _weighted_sum_tz_fortran_sx(theta_w, zeta_w, hf_before_x)  # (S,X)

    if op.n_xi > 1:
        flow_x = flow_factor[:, None, None, None] * (f_delta[:, :, 1, :, :] * (w_x3 * mask_l1)[None, :, None, None])
        flow_vs_x = _weighted_sum_tz_fortran_sx(
            theta_w,
            zeta_w,
            flow_x * op.b_hat[None, None, :, :] / op.d_hat[None, None, :, :],
        ) / vprime_hat
    else:
        flow_vs_x = jnp.zeros((op.n_species, op.n_x), dtype=jnp.float64)

    # Total density/pressure and velocities:
    exp_phi1 = jnp.exp(-(z[:, None, None] * op.alpha / t_hat[:, None, None]) * op.phi1_hat_base[None, :, :])
    total_density = n_hat[:, None, None] * exp_phi1 + dens
    total_pressure = n_hat[:, None, None] * exp_phi1 * t_hat[:, None, None] + pres
    vel_fsadens = flow / n_hat[:, None, None]
    vel_total = flow / total_density
    mach = vel_fsadens * (sqrt_m[:, None, None] / sqrt_t[:, None, None])

    # Current-like diagnostics:
    j_hat_tz = jnp.einsum("s,stz->tz", z, flow)  # (T,Z)
    b0, _g, _i = _flux_functions_from_op(op)
    fsab2 = jnp.asarray(op.fsab_hat2, dtype=jnp.float64)
    fsab_j = jnp.einsum("s,s->", z, fsabflow)  # scalar

    out: dict[str, jnp.ndarray] = {
        "densityPerturbation": dens,
        "pressurePerturbation": pres,
        "pressureAnisotropy": pres_aniso,
        "flow": flow,
        "FSADensityPerturbation": fsadens,
        "FSAPressurePerturbation": fsapres,
        "FSABFlow": fsabflow,
        "FSABFlow_vs_x": jnp.transpose(flow_vs_x, (1, 0)),  # (X,S)
        "FSABVelocityUsingFSADensity": fsabflow / n_hat,
        "FSABVelocityUsingFSADensityOverB0": (fsabflow / n_hat) / b0,
        "FSABVelocityUsingFSADensityOverRootFSAB2": (fsabflow / n_hat) / jnp.sqrt(fsab2),
        "FSABjHat": fsab_j,
        "FSABjHatOverB0": fsab_j / b0,
        "FSABjHatOverRootFSAB2": fsab_j / jnp.sqrt(fsab2),
        "totalDensity": total_density,
        "totalPressure": total_pressure,
        "velocityUsingFSADensity": vel_fsadens,
        "velocityUsingTotalDensity": vel_total,
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
        "particleFlux_vm_psiHat_vs_x": jnp.transpose(pf_vs_x, (1, 0)),  # (X,S)
        "heatFlux_vm_psiHat_vs_x": jnp.transpose(hf_vs_x, (1, 0)),  # (X,S)
    }

    # Sources from constraint schemes (as read in Python):
    extra = x_full[op.f_size + op.phi1_size :].reshape((-1,))
    if int(op.constraint_scheme) == 2:
        src = extra.reshape((op.n_species, op.n_x))  # (S,X)
        out["sources"] = jnp.transpose(src, (1, 0))  # (X,S)
    elif int(op.constraint_scheme) in {1, 3, 4}:
        src = extra.reshape((op.n_species, 2))  # (S,2)
        out["sources"] = jnp.transpose(src, (1, 0))  # (2,S)

    # vE / NTV placeholders:
    out["particleFluxBeforeSurfaceIntegral_vE"] = jnp.zeros_like(pf_before_vm)
    out["particleFluxBeforeSurfaceIntegral_vE0"] = jnp.zeros_like(pf_before_vm)
    out["heatFluxBeforeSurfaceIntegral_vE"] = jnp.zeros_like(hf_before_vm)
    out["heatFluxBeforeSurfaceIntegral_vE0"] = jnp.zeros_like(hf_before_vm)
    out["momentumFluxBeforeSurfaceIntegral_vE"] = jnp.zeros_like(mf_before_vm)
    out["momentumFluxBeforeSurfaceIntegral_vE0"] = jnp.zeros_like(mf_before_vm)
    out["NTVBeforeSurfaceIntegral"] = jnp.zeros_like(pf_before_vm)
    out["NTV"] = jnp.zeros((op.n_species,), dtype=jnp.float64)

    return out


def v3_transport_output_fields_vm_only(
    *,
    op0: V3FullSystemOperator,
    state_vectors_by_rhs: dict[int, jnp.ndarray],
) -> dict[str, jnp.ndarray]:
    """Compute a larger set of RHSMode=2/3 output fields used by upstream postprocessing scripts.

    Returns arrays in the same shape/order as they appear when reading a Fortran v3 `sfincsOutput.h5`
    file in Python (i.e. the shapes in `tests/ref/*.sfincsOutput.h5`).
    """
    rhs_mode = int(op0.rhs_mode)
    n = transport_matrix_size_from_rhs_mode(rhs_mode)
    s = int(op0.n_species)
    x = int(op0.n_x)
    t = int(op0.n_theta)
    z = int(op0.n_zeta)

    rhs_values = list(range(1, n + 1))
    for which_rhs in rhs_values:
        if which_rhs not in state_vectors_by_rhs:
            raise ValueError(f"Missing state vector for which_rhs={which_rhs}.")

    x_stack = jnp.stack([jnp.asarray(state_vectors_by_rhs[which_rhs], dtype=jnp.float64) for which_rhs in rhs_values], axis=0)  # (N,total)
    op_rhs_list = [with_transport_rhs_settings(op0, which_rhs=which_rhs) for which_rhs in rhs_values]
    diag_list = [v3_transport_diagnostics_vm_only(op_rhs, x_full=x_stack[i]) for i, op_rhs in enumerate(op_rhs_list)]

    pf_vm_psi_hat = jnp.stack([d.particle_flux_vm_psi_hat for d in diag_list], axis=1)  # (S,N)
    hf_vm_psi_hat = jnp.stack([d.heat_flux_vm_psi_hat for d in diag_list], axis=1)  # (S,N)
    flow = jnp.stack([d.fsab_flow for d in diag_list], axis=1)  # (S,N)

    pf_before_vm_stzn = jnp.stack([d.particle_flux_before_surface_integral_vm for d in diag_list], axis=0)  # (N,S,T,Z)
    hf_before_vm_stzn = jnp.stack([d.heat_flux_before_surface_integral_vm for d in diag_list], axis=0)  # (N,S,T,Z)
    pf_before_vm0_stzn = jnp.stack([d.particle_flux_before_surface_integral_vm0 for d in diag_list], axis=0)  # (N,S,T,Z)
    hf_before_vm0_stzn = jnp.stack([d.heat_flux_before_surface_integral_vm0 for d in diag_list], axis=0)  # (N,S,T,Z)

    # Convert to Python-read order (Z,T,S,N):
    pf_before_vm = jnp.transpose(pf_before_vm_stzn, (3, 2, 1, 0))
    hf_before_vm = jnp.transpose(hf_before_vm_stzn, (3, 2, 1, 0))
    pf_before_vm0 = jnp.transpose(pf_before_vm0_stzn, (3, 2, 1, 0))
    hf_before_vm0 = jnp.transpose(hf_before_vm0_stzn, (3, 2, 1, 0))

    w2d_stack = jnp.stack([op_rhs.theta_weights[:, None] * op_rhs.zeta_weights[None, :] for op_rhs in op_rhs_list], axis=0)  # (N,T,Z)
    pf_vm0_psi_hat = jnp.einsum("ntz,nstz->sn", w2d_stack, pf_before_vm0_stzn)
    hf_vm0_psi_hat = jnp.einsum("ntz,nstz->sn", w2d_stack, hf_before_vm0_stzn)

    pf_vs_x = jnp.stack([d.particle_flux_vm_psi_hat_vs_x for d in diag_list], axis=2)  # (X,S,N)
    hf_vs_x = jnp.stack([d.heat_flux_vm_psi_hat_vs_x for d in diag_list], axis=2)  # (X,S,N)
    flow_vs_x = jnp.stack([d.fsab_flow_vs_x for d in diag_list], axis=2)  # (X,S,N)

    # vE terms are 0 in the parity-tested RHSMode=2/3 fixtures without Phi1/Er.
    pf_before_ve = jnp.zeros((z, t, s, n), dtype=jnp.float64)
    hf_before_ve = jnp.zeros((z, t, s, n), dtype=jnp.float64)
    pf_before_ve0 = jnp.zeros((z, t, s, n), dtype=jnp.float64)
    hf_before_ve0 = jnp.zeros((z, t, s, n), dtype=jnp.float64)

    sources = None
    extra_stack = x_stack[:, op0.f_size + op0.phi1_size :]  # (N,extra)
    if int(op0.constraint_scheme) == 2:
        src = extra_stack.reshape((n, s, x))  # (N,S,X)
        sources = jnp.transpose(src, (2, 1, 0))  # (X,S,N)
    elif int(op0.constraint_scheme) in {1, 3, 4}:
        src = extra_stack.reshape((n, s, 2))  # (N,S,2)
        sources = jnp.transpose(src, (2, 1, 0))  # (2,S,N)

    fsab2 = jnp.asarray(op0.fsab_hat2, dtype=jnp.float64)
    b0, _g, _i = _flux_functions_from_op(op0)

    # Current-like diagnostics used by scan plot scripts:
    jhat = jnp.einsum("s,sn->n", op0.z_s, flow)  # (N,)
    out: dict[str, jnp.ndarray] = {
        "FSABFlow": flow,
        "FSABFlow_vs_x": flow_vs_x,
        "FSABVelocityUsingFSADensity": flow / op0.n_hat[:, None],
        "FSABVelocityUsingFSADensityOverB0": (flow / op0.n_hat[:, None]) / b0,
        "FSABVelocityUsingFSADensityOverRootFSAB2": (flow / op0.n_hat[:, None]) / jnp.sqrt(fsab2),
        "FSABjHat": jhat,
        "FSABjHatOverB0": jhat / b0,
        "FSABjHatOverRootFSAB2": jhat / jnp.sqrt(fsab2),
        "particleFlux_vm_psiHat": pf_vm_psi_hat,
        "heatFlux_vm_psiHat": hf_vm_psi_hat,
        "particleFlux_vm0_psiHat": pf_vm0_psi_hat,
        "heatFlux_vm0_psiHat": hf_vm0_psi_hat,
        "particleFluxBeforeSurfaceIntegral_vm": pf_before_vm,
        "heatFluxBeforeSurfaceIntegral_vm": hf_before_vm,
        "particleFluxBeforeSurfaceIntegral_vm0": pf_before_vm0,
        "heatFluxBeforeSurfaceIntegral_vm0": hf_before_vm0,
        "particleFluxBeforeSurfaceIntegral_vE": pf_before_ve,
        "heatFluxBeforeSurfaceIntegral_vE": hf_before_ve,
        "particleFluxBeforeSurfaceIntegral_vE0": pf_before_ve0,
        "heatFluxBeforeSurfaceIntegral_vE0": hf_before_ve0,
        "particleFlux_vm_psiHat_vs_x": pf_vs_x,
        "heatFlux_vm_psiHat_vs_x": hf_vs_x,
    }
    if sources is not None:
        out["sources"] = sources
    return out


def transport_matrix_size_from_rhs_mode(rhs_mode: int) -> int:
    if int(rhs_mode) == 2:
        return 3
    if int(rhs_mode) == 3:
        return 2
    raise ValueError("transport matrix is only defined for RHSMode=2 or RHSMode=3.")


def v3_transport_matrix_column(
    *,
    op: V3FullSystemOperator,
    geom: BoozerGeometry,
    which_rhs: int,
    diag: V3TransportDiagnostics,
) -> jnp.ndarray:
    """Compute one transport-matrix column from the diagnostics of a solved whichRHS system."""
    rhs_mode = int(op.rhs_mode)
    w = int(which_rhs)
    nrow = transport_matrix_size_from_rhs_mode(rhs_mode)

    # v3 uses ispecies=1 for RHSMode=2/3.
    s = 0
    n_hat = jnp.asarray(op.n_hat[s], dtype=jnp.float64)
    t_hat = jnp.asarray(op.t_hat[s], dtype=jnp.float64)
    m_hat = jnp.asarray(op.m_hat[s], dtype=jnp.float64)
    z = jnp.asarray(op.z_s[s], dtype=jnp.float64)
    sqrt_t = jnp.sqrt(t_hat)
    sqrt_m = jnp.sqrt(m_hat)

    delta = jnp.asarray(op.delta, dtype=jnp.float64)
    alpha = jnp.asarray(op.alpha, dtype=jnp.float64)

    g_hat = jnp.asarray(float(geom.g_hat), dtype=jnp.float64)
    i_hat = jnp.asarray(float(geom.i_hat), dtype=jnp.float64)
    iota = jnp.asarray(float(geom.iota), dtype=jnp.float64)
    b0_over_bbar = jnp.asarray(float(geom.b0_over_bbar), dtype=jnp.float64)
    fsab_hat2 = jnp.asarray(op.fsab_hat2, dtype=jnp.float64)

    if (jnp.abs(g_hat) < 1e-30) | (jnp.abs(b0_over_bbar) < 1e-30):
        b0_eff, g_eff, i_eff = _flux_functions_from_op(op)
        b0_over_bbar = jnp.where(jnp.abs(b0_over_bbar) < 1e-30, b0_eff, b0_over_bbar)
        g_hat = jnp.where(jnp.abs(g_hat) < 1e-30, g_eff, g_hat)
        i_hat = jnp.where(jnp.abs(i_hat) < 1e-30, i_eff, i_hat)

    g_plus = g_hat + iota * i_hat

    particle_flux = jnp.asarray(diag.particle_flux_vm_psi_hat[s], dtype=jnp.float64)
    heat_flux = jnp.asarray(diag.heat_flux_vm_psi_hat[s], dtype=jnp.float64)
    fsab_flow = jnp.asarray(diag.fsab_flow[s], dtype=jnp.float64)

    out = jnp.zeros((nrow,), dtype=jnp.float64)

    if rhs_mode == 3:
        if w == 1:
            out = out.at[0].set(
                (4.0 / (delta * delta))
                * (sqrt_t / sqrt_m)
                * (z * z)
                * g_plus
                * particle_flux
                * b0_over_bbar
                / (t_hat * t_hat * g_hat * g_hat)
            )
            out = out.at[1].set(2.0 * z * fsab_flow / (delta * g_hat * t_hat))
            return out
        if w == 2:
            out = out.at[0].set(particle_flux * 2.0 * fsab_hat2 / (n_hat * alpha * delta * g_hat))
            out = out.at[1].set(
                fsab_flow * sqrt_t * sqrt_m * fsab_hat2 / (g_plus * alpha * z * n_hat * b0_over_bbar)
            )
            return out
        raise ValueError("RHSMode=3 expects which_rhs in {1,2}.")

    if rhs_mode == 2:
        if w == 1:
            out = out.at[0].set(
                (4.0 / (delta * delta))
                * (sqrt_t / sqrt_m)
                * (z * z)
                * g_plus
                * particle_flux
                * b0_over_bbar
                / (t_hat * t_hat * g_hat * g_hat)
            )
            out = out.at[1].set(
                (8.0 / (delta * delta))
                * (sqrt_t / sqrt_m)
                * (z * z)
                * g_plus
                * heat_flux
                * b0_over_bbar
                / (t_hat * t_hat * t_hat * g_hat * g_hat)
            )
            out = out.at[2].set(2.0 * z * fsab_flow / (delta * g_hat * t_hat))
            return out
        if w == 2:
            out = out.at[0].set(
                (4.0 / (delta * delta))
                * (sqrt_t / sqrt_m)
                * (z * z)
                * g_plus
                * particle_flux
                * b0_over_bbar
                / (n_hat * t_hat * g_hat * g_hat)
            )
            out = out.at[1].set(
                (8.0 / (delta * delta))
                * (sqrt_t / sqrt_m)
                * (z * z)
                * g_plus
                * heat_flux
                * b0_over_bbar
                / (n_hat * t_hat * t_hat * g_hat * g_hat)
            )
            out = out.at[2].set(2.0 * z * fsab_flow / (delta * g_hat * n_hat))
            return out
        if w == 3:
            out = out.at[0].set(particle_flux * 2.0 * fsab_hat2 / (n_hat * alpha * delta * g_hat))
            out = out.at[1].set(heat_flux * 4.0 * fsab_hat2 / (n_hat * t_hat * alpha * delta * g_hat))
            out = out.at[2].set(
                fsab_flow * sqrt_t * sqrt_m * fsab_hat2 / (g_plus * alpha * z * n_hat * b0_over_bbar)
            )
            return out
        raise ValueError("RHSMode=2 expects which_rhs in {1,2,3}.")

    raise AssertionError("unreachable")


def v3_transport_matrix_from_state_vectors(
    *,
    op0: V3FullSystemOperator,
    geom: BoozerGeometry,
    state_vectors_by_rhs: dict[int, jnp.ndarray],
) -> jnp.ndarray:
    """Assemble the RHSMode=2/3 transport matrix from solved whichRHS state vectors."""
    rhs_mode = int(op0.rhs_mode)
    n = transport_matrix_size_from_rhs_mode(rhs_mode)
    rhs_values = list(range(1, n + 1))
    for which_rhs in rhs_values:
        if which_rhs not in state_vectors_by_rhs:
            raise ValueError(f"Missing state vector for which_rhs={which_rhs}.")
    x_stack = jnp.stack([jnp.asarray(state_vectors_by_rhs[which_rhs], dtype=jnp.float64) for which_rhs in rhs_values], axis=0)  # (N,total)
    op_rhs_list = [with_transport_rhs_settings(op0, which_rhs=which_rhs) for which_rhs in rhs_values]
    diag_list = [v3_transport_diagnostics_vm_only(op_rhs, x_full=x_stack[i]) for i, op_rhs in enumerate(op_rhs_list)]
    cols = [
        v3_transport_matrix_column(op=op_rhs_list[i], geom=geom, which_rhs=which_rhs, diag=diag_list[i])
        for i, which_rhs in enumerate(rhs_values)
    ]
    return jnp.stack(cols, axis=1)
