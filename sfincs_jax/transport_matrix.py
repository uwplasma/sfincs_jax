from __future__ import annotations

from dataclasses import dataclass

from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)

import jax.numpy as jnp

from .geometry import BoozerGeometry
from .v3_system import V3FullSystemOperator


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


def _vprime_hat_from_op(op: V3FullSystemOperator) -> jnp.ndarray:
    w2d = op.theta_weights[:, None] * op.zeta_weights[None, :]
    return jnp.sum(w2d / op.d_hat)


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

    w2d = op.theta_weights[:, None] * op.zeta_weights[None, :]  # (T,Z)
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
    sum_pf_l0 = jnp.einsum("x,sxtz->stz", wpf0, f_l0)
    sum_pf_l2 = jnp.einsum("x,sxtz->stz", wpf2, f_l2)
    pf_before = particle_flux_factor_vm[:, None, None] * (
        (8.0 / 3.0) * factor_vm[None, :, :] * sum_pf_l0 + (4.0 / 15.0) * factor_vm[None, :, :] * sum_pf_l2
    )  # (S,T,Z)
    particle_flux_vm_psi_hat = jnp.einsum("tz,stz->s", w2d, pf_before)

    # Heat flux (vm):
    whf0 = (w_hf * mask_l0).astype(jnp.float64)
    whf2 = (w_hf * mask_l2).astype(jnp.float64)
    sum_hf_l0 = jnp.einsum("x,sxtz->stz", whf0, f_l0)
    sum_hf_l2 = jnp.einsum("x,sxtz->stz", whf2, f_l2)
    hf_before = heat_flux_factor_vm[:, None, None] * (
        (8.0 / 3.0) * factor_vm[None, :, :] * sum_hf_l0 + (4.0 / 15.0) * factor_vm[None, :, :] * sum_hf_l2
    )  # (S,T,Z)
    heat_flux_vm_psi_hat = jnp.einsum("tz,stz->s", w2d, hf_before)

    # FSABFlow:
    if op.n_xi > 1:
        f_l1 = f_delta[:, :, 1, :, :]
    else:
        f_l1 = jnp.zeros_like(f_l0)

    wf1 = (w_flow * mask_l1).astype(jnp.float64)
    sum_flow = jnp.einsum("x,sxtz->stz", wf1, f_l1)
    flow = flow_factor[:, None, None] * sum_flow
    fsab_flow = jnp.einsum("tz,stz->s", w2d, flow * op.b_hat[None, :, :] / op.d_hat[None, :, :]) / vprime_hat

    # vm0 contributions (use f0 only):
    f0_l0 = f0[:, :, 0, :, :]
    f0_l2 = f0[:, :, 2, :, :] if op.n_xi > 2 else jnp.zeros_like(f0_l0)
    sum_pf0_l0 = jnp.einsum("x,sxtz->stz", wpf0, f0_l0)
    sum_pf0_l2 = jnp.einsum("x,sxtz->stz", wpf2, f0_l2)
    pf_before_vm0 = particle_flux_factor_vm[:, None, None] * (
        (8.0 / 3.0) * factor_vm[None, :, :] * sum_pf0_l0 + (4.0 / 15.0) * factor_vm[None, :, :] * sum_pf0_l2
    )

    sum_hf0_l0 = jnp.einsum("x,sxtz->stz", whf0, f0_l0)
    sum_hf0_l2 = jnp.einsum("x,sxtz->stz", whf2, f0_l2)
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
    pf_vs_x = jnp.einsum("tz,sxtz->sx", w2d, pf_before_x)  # (S,X)

    # Heat flux:
    hf_x_l0 = f_l0 * whf0[None, :, None, None]
    hf_x_l2 = f_l2 * whf2[None, :, None, None]
    hf_before_x = heat_flux_factor_vm[:, None, None, None] * (
        (8.0 / 3.0) * factor_vm[None, None, :, :] * hf_x_l0 + (4.0 / 15.0) * factor_vm[None, None, :, :] * hf_x_l2
    )  # (S,X,T,Z)
    hf_vs_x = jnp.einsum("tz,sxtz->sx", w2d, hf_before_x)  # (S,X)

    # Flow:
    flow_x = flow_factor[:, None, None, None] * (f_l1 * wf1[None, :, None, None])  # (S,X,T,Z)
    fsab_flow_vs_x = jnp.einsum("tz,sxtz->sx", w2d, flow_x * op.b_hat[None, None, :, :] / op.d_hat[None, None, :, :]) / vprime_hat

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

    # Outputs in Python-read order:
    pf_vm_psi_hat = jnp.zeros((s, n), dtype=jnp.float64)
    hf_vm_psi_hat = jnp.zeros((s, n), dtype=jnp.float64)
    flow = jnp.zeros((s, n), dtype=jnp.float64)

    pf_vm0_psi_hat = jnp.zeros((s, n), dtype=jnp.float64)
    hf_vm0_psi_hat = jnp.zeros((s, n), dtype=jnp.float64)

    # Note: In Fortran v3, these arrays are indexed as (itheta, izeta, ispecies, whichRHS),
    # but when written from Fortran (column-major) and read back in Python (row-major),
    # the first two axes appear swapped. Therefore the Python-read order is (izeta, itheta, ispecies, whichRHS).
    pf_before_vm = jnp.zeros((z, t, s, n), dtype=jnp.float64)
    hf_before_vm = jnp.zeros((z, t, s, n), dtype=jnp.float64)
    pf_before_vm0 = jnp.zeros((z, t, s, n), dtype=jnp.float64)
    hf_before_vm0 = jnp.zeros((z, t, s, n), dtype=jnp.float64)

    # vE terms are 0 in the parity-tested RHSMode=2/3 fixtures without Phi1/Er.
    pf_before_ve = jnp.zeros((z, t, s, n), dtype=jnp.float64)
    hf_before_ve = jnp.zeros((z, t, s, n), dtype=jnp.float64)
    pf_before_ve0 = jnp.zeros((z, t, s, n), dtype=jnp.float64)
    hf_before_ve0 = jnp.zeros((z, t, s, n), dtype=jnp.float64)

    pf_vs_x = jnp.zeros((x, s, n), dtype=jnp.float64)
    hf_vs_x = jnp.zeros((x, s, n), dtype=jnp.float64)
    flow_vs_x = jnp.zeros((x, s, n), dtype=jnp.float64)

    sources = None
    if int(op0.constraint_scheme) == 2:
        sources = jnp.zeros((x, s, n), dtype=jnp.float64)
    elif int(op0.constraint_scheme) in {1, 3, 4}:
        sources = jnp.zeros((2, s, n), dtype=jnp.float64)

    for which_rhs in range(1, n + 1):
        x_full = state_vectors_by_rhs[which_rhs]
        diag = v3_transport_diagnostics_vm_only(op0, x_full=x_full)

        pf_vm_psi_hat = pf_vm_psi_hat.at[:, which_rhs - 1].set(diag.particle_flux_vm_psi_hat)
        hf_vm_psi_hat = hf_vm_psi_hat.at[:, which_rhs - 1].set(diag.heat_flux_vm_psi_hat)
        flow = flow.at[:, which_rhs - 1].set(diag.fsab_flow)

        # vm0 from "before surface integral" arrays:
        pf_vm0_psi_hat = pf_vm0_psi_hat.at[:, which_rhs - 1].set(
            jnp.einsum("tz,stz->s", op0.theta_weights[:, None] * op0.zeta_weights[None, :], diag.particle_flux_before_surface_integral_vm0)
        )
        hf_vm0_psi_hat = hf_vm0_psi_hat.at[:, which_rhs - 1].set(
            jnp.einsum("tz,stz->s", op0.theta_weights[:, None] * op0.zeta_weights[None, :], diag.heat_flux_before_surface_integral_vm0)
        )

        # Before-surface-integral arrays in Python-read order (Z,T,S,N):
        pf_before_vm = pf_before_vm.at[:, :, :, which_rhs - 1].set(jnp.transpose(diag.particle_flux_before_surface_integral_vm, (2, 1, 0)))
        hf_before_vm = hf_before_vm.at[:, :, :, which_rhs - 1].set(jnp.transpose(diag.heat_flux_before_surface_integral_vm, (2, 1, 0)))
        pf_before_vm0 = pf_before_vm0.at[:, :, :, which_rhs - 1].set(jnp.transpose(diag.particle_flux_before_surface_integral_vm0, (2, 1, 0)))
        hf_before_vm0 = hf_before_vm0.at[:, :, :, which_rhs - 1].set(jnp.transpose(diag.heat_flux_before_surface_integral_vm0, (2, 1, 0)))

        # vs_x arrays (X,S,N):
        pf_vs_x = pf_vs_x.at[:, :, which_rhs - 1].set(diag.particle_flux_vm_psi_hat_vs_x)
        hf_vs_x = hf_vs_x.at[:, :, which_rhs - 1].set(diag.heat_flux_vm_psi_hat_vs_x)
        flow_vs_x = flow_vs_x.at[:, :, which_rhs - 1].set(diag.fsab_flow_vs_x)

        if sources is not None:
            # Extract the extra unknowns for the constraint scheme.
            extra = x_full[op0.f_size + op0.phi1_size :].reshape((-1,))
            if int(op0.constraint_scheme) == 2:
                src = extra.reshape((op0.n_species, op0.n_x))  # (S,X)
                sources = sources.at[:, :, which_rhs - 1].set(jnp.transpose(src, (1, 0)))  # (X,S)
            elif int(op0.constraint_scheme) in {1, 3, 4}:
                src = extra.reshape((op0.n_species, 2))  # (S,2)
                sources = sources.at[:, :, which_rhs - 1].set(jnp.transpose(src, (1, 0)))  # (2,S)

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
    out = jnp.zeros((n, n), dtype=jnp.float64)

    for which_rhs in range(1, n + 1):
        if which_rhs not in state_vectors_by_rhs:
            raise ValueError(f"Missing state vector for which_rhs={which_rhs}.")
        x = state_vectors_by_rhs[which_rhs]
        diag = v3_transport_diagnostics_vm_only(op0, x_full=x)
        col = v3_transport_matrix_column(op=op0, geom=geom, which_rhs=which_rhs, diag=diag)
        out = out.at[:, which_rhs - 1].set(col)

    return out
