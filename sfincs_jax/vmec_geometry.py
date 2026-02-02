from __future__ import annotations

import math
from pathlib import Path

from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from .geometry import BoozerGeometry
from .vmec_wout import VmecWout, psi_a_hat_from_wout, read_vmec_wout, vmec_interpolation, _set_scale_factor


def _finite_diff_on_full_mesh_from_half_mesh(arr_half: np.ndarray, dpsi: float) -> np.ndarray:
    """Replicate v3's finite-difference pattern for half-mesh quantities.

    v3 uses, for j=2..ns-1 (1-based):
      dQdpsiHat(j) = (Q(j+1) - Q(j)) / dpsi
    and copies endpoints from adjacent interior values.
    """
    arr_half = np.asarray(arr_half, dtype=np.float64)
    if arr_half.ndim == 1:
        ns = int(arr_half.shape[0])
        out = np.zeros((ns,), dtype=np.float64)
        out[1 : ns - 1] = (arr_half[2:ns] - arr_half[1 : ns - 1]) / float(dpsi)
        if ns >= 3:
            out[0] = out[1]
            out[ns - 1] = out[ns - 2]
        return out
    if arr_half.ndim == 2:
        n_mode, ns = arr_half.shape
        out = np.zeros((n_mode, ns), dtype=np.float64)
        out[:, 1 : ns - 1] = (arr_half[:, 2:ns] - arr_half[:, 1 : ns - 1]) / float(dpsi)
        if ns >= 3:
            out[:, 0] = out[:, 1]
            out[:, ns - 1] = out[:, ns - 2]
        return out
    raise ValueError(f"arr_half must be 1D or 2D, got shape {arr_half.shape}")


def vmec_geometry_from_wout_file(
    *,
    path: str | Path,
    theta: jnp.ndarray,
    zeta: jnp.ndarray,
    psi_n_wish: float,
    vmec_radial_option: int = 0,
    vmec_nyquist_option: int = 1,
    min_bmn_to_load: float = 0.0,
    ripple_scale: float = 1.0,
    helicity_n: int = 0,
    helicity_l: int = 0,
    chunk: int = 256,
) -> BoozerGeometry:
    """Compute geometry arrays for v3 `geometryScheme=5` (VMEC wout file).

    This is a direct, minimal translation of the core Fourier-sum logic in v3
    `geometry.F90::computeBHat_VMEC` for stellarator-symmetric equilibria (lasym=false).

    Notes
    -----
    - The VMEC file read is not differentiable. Once loaded and evaluated, the resulting
      arrays are standard JAX arrays usable in JIT-compiled kernels.
    - This implementation aims to support output parity and operator kernels on the same
      (theta,zeta) grid used by v3.
    """
    w: VmecWout = read_vmec_wout(path)
    psi_a_hat = psi_a_hat_from_wout(w)
    a_hat = float(w.aminor_p)
    n_periods = int(w.nfp)

    interp = vmec_interpolation(w=w, psi_n_wish=float(psi_n_wish), vmec_radial_option=int(vmec_radial_option))
    (i_full0, i_full1) = interp.index_full
    (w_full0, w_full1) = interp.weight_full
    (i_half0, i_half1) = interp.index_half
    (w_half0, w_half1) = interp.weight_half

    # Mode-0 b00 (m=n=0) on half mesh is index 0 in the nyquist arrays.
    b00 = float(w.bmnc[0, i_half0] * w_half0 + w.bmnc[0, i_half1] * w_half1)
    if b00 == 0.0:
        raise ValueError("VMEC bmnc(0,0) is zero; cannot apply min_Bmn_to_load filter.")

    # VMEC spacing in psiHat: dpsi = phi(2)/(2*pi) in v3, which equals psiAHat/(ns-1).
    dpsi = float(w.phi[1]) / (2.0 * math.pi)

    theta_np = np.asarray(theta, dtype=np.float64)
    zeta_np = np.asarray(zeta, dtype=np.float64)
    theta1 = theta_np[None, :, None]
    zeta1 = zeta_np[None, None, :]

    ntheta = int(theta_np.shape[0])
    nzeta = int(zeta_np.shape[0])

    # Allocate outputs:
    b_hat = np.zeros((ntheta, nzeta), dtype=np.float64)
    d_hat = np.zeros_like(b_hat)  # Will accumulate Jacobian then invert.
    db_dtheta = np.zeros_like(b_hat)
    db_dzeta = np.zeros_like(b_hat)
    db_dpsi = np.zeros_like(b_hat)

    b_sub_theta = np.zeros_like(b_hat)
    b_sub_zeta = np.zeros_like(b_hat)
    b_sub_psi = np.zeros_like(b_hat)
    db_sub_theta_dpsi = np.zeros_like(b_hat)
    db_sub_zeta_dpsi = np.zeros_like(b_hat)
    db_sub_theta_dzeta = np.zeros_like(b_hat)
    db_sub_zeta_dtheta = np.zeros_like(b_hat)
    db_sub_psi_dtheta = np.zeros_like(b_hat)
    db_sub_psi_dzeta = np.zeros_like(b_hat)

    b_sup_theta = np.zeros_like(b_hat)
    b_sup_zeta = np.zeros_like(b_hat)
    db_sup_theta_dzeta = np.zeros_like(b_hat)
    db_sup_zeta_dtheta = np.zeros_like(b_hat)

    # v3 sets these to 0 for VMEC:
    db_sup_theta_dpsi = np.zeros_like(b_hat)
    db_sup_zeta_dpsi = np.zeros_like(b_hat)

    # Loop over nyquist modes in chunks.
    xm = np.asarray(w.xm_nyq, dtype=np.float64)
    xn = np.asarray(w.xn_nyq, dtype=np.float64)  # This is n * nfp in VMEC conventions.

    # Decide which modes to include based on |b/b00| >= min_Bmn_to_load, using the half-mesh interpolation.
    b_mode = w.bmnc[:, i_half0] * w_half0 + w.bmnc[:, i_half1] * w_half1  # (mnmax_nyq,)
    scale_all = np.array(
        [
            _set_scale_factor(
                n=int(round(float(xn[k]) / float(n_periods))),
                m=int(round(float(xm[k]))),
                helicity_n=int(helicity_n),
                helicity_l=int(helicity_l),
                ripple_scale=float(ripple_scale),
            )
            for k in range(int(xm.shape[0]))
        ],
        dtype=np.float64,
    )
    # v3 applies the scale factor before checking `min_Bmn_to_load`.
    include = np.abs((b_mode * scale_all) / float(b00)) >= float(min_bmn_to_load)

    vmec_nyquist_option = int(vmec_nyquist_option)
    if vmec_nyquist_option == 0:
        # Backward-compatibility with early sfincs_jax prototypes: treat 0 as v3 default (1).
        vmec_nyquist_option = 1
    if vmec_nyquist_option not in {1, 2}:
        raise ValueError("VMEC_Nyquist_option must be 1 (skip Nyquist) or 2 (include Nyquist).")

    # Apply optional VMEC Nyquist truncation:
    # v3 skips modes with |m|>=mpol or |n|>ntor when VMEC_Nyquist_option==1.
    if vmec_nyquist_option == 1:
        n_eff = xn / float(n_periods)
        include = include & (np.abs(xm) < float(w.mpol)) & (np.abs(n_eff) <= float(w.ntor))

    idx = np.nonzero(include)[0].astype(np.int32)
    if idx.size == 0:
        raise ValueError("No VMEC modes were included (min_Bmn_to_load too large?).")

    for i0 in range(0, int(idx.size), int(chunk)):
        i1 = min(int(idx.size), i0 + int(chunk))
        sel = idx[i0:i1]

        m = xm[sel][:, None, None]
        n_nyq = xn[sel][:, None, None]  # equals n*NPeriods

        # Per-mode scale factors (rippleScale / quasisymmetry selection):
        scale = np.array(
            [
                _set_scale_factor(
                    n=int(round(float(xn[k]) / float(n_periods))),
                    m=int(round(float(xm[k]))),
                    helicity_n=int(helicity_n),
                    helicity_l=int(helicity_l),
                    ripple_scale=float(ripple_scale),
                )
                for k in sel.tolist()
            ],
            dtype=np.float64,
        )[:, None, None]

        angle = m * theta1 - n_nyq * zeta1
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        # Coefficients at the requested radius:
        # Half-mesh quantities:
        b = (w.bmnc[sel, i_half0] * w_half0 + w.bmnc[sel, i_half1] * w_half1)[:, None, None] * scale
        jac = (w.gmnc[sel, i_half0] * w_half0 + w.gmnc[sel, i_half1] * w_half1)[:, None, None] * scale / float(psi_a_hat)
        bsupu = (w.bsupumnc[sel, i_half0] * w_half0 + w.bsupumnc[sel, i_half1] * w_half1)[:, None, None] * scale
        bsupv = (w.bsupvmnc[sel, i_half0] * w_half0 + w.bsupvmnc[sel, i_half1] * w_half1)[:, None, None] * scale
        bsubu = (w.bsubumnc[sel, i_half0] * w_half0 + w.bsubumnc[sel, i_half1] * w_half1)[:, None, None] * scale
        bsubv = (w.bsubvmnc[sel, i_half0] * w_half0 + w.bsubvmnc[sel, i_half1] * w_half1)[:, None, None] * scale

        # Full-mesh quantities:
        bsubs = (w.bsubsmns[sel, i_full0] * w_full0 + w.bsubsmns[sel, i_full1] * w_full1)[:, None, None] * scale / float(
            psi_a_hat
        )

        # Radial derivatives for half-mesh quantities (on full mesh):
        d_b_dpsi_full = _finite_diff_on_full_mesh_from_half_mesh(w.bmnc[sel, :], dpsi=dpsi)
        d_bsubu_dpsi_full = _finite_diff_on_full_mesh_from_half_mesh(w.bsubumnc[sel, :], dpsi=dpsi)
        d_bsubv_dpsi_full = _finite_diff_on_full_mesh_from_half_mesh(w.bsubvmnc[sel, :], dpsi=dpsi)

        d_b_dpsi = (
            d_b_dpsi_full[:, i_full0] * w_full0 + d_b_dpsi_full[:, i_full1] * w_full1
        )[:, None, None] * scale
        d_bsubu_dpsi = (
            d_bsubu_dpsi_full[:, i_full0] * w_full0 + d_bsubu_dpsi_full[:, i_full1] * w_full1
        )[:, None, None] * scale
        d_bsubv_dpsi = (
            d_bsubv_dpsi_full[:, i_full0] * w_full0 + d_bsubv_dpsi_full[:, i_full1] * w_full1
        )[:, None, None] * scale

        # Accumulate symmetric (cosine) contributions:
        b_hat += np.sum(b * cos_a, axis=0)
        db_dtheta += np.sum(-m * b * sin_a, axis=0)
        db_dzeta += np.sum(n_nyq * b * sin_a, axis=0)

        d_hat += np.sum(jac * cos_a, axis=0)

        b_sup_theta += np.sum(bsupu * cos_a, axis=0)
        db_sup_theta_dzeta += np.sum(n_nyq * bsupu * sin_a, axis=0)
        b_sup_zeta += np.sum(bsupv * cos_a, axis=0)
        db_sup_zeta_dtheta += np.sum(-m * bsupv * sin_a, axis=0)

        b_sub_theta += np.sum(bsubu * cos_a, axis=0)
        db_sub_theta_dzeta += np.sum(n_nyq * bsubu * sin_a, axis=0)
        b_sub_zeta += np.sum(bsubv * cos_a, axis=0)
        db_sub_zeta_dtheta += np.sum(-m * bsubv * sin_a, axis=0)

        b_sub_psi += np.sum(bsubs * sin_a, axis=0)
        db_sub_psi_dtheta += np.sum(m * bsubs * cos_a, axis=0)
        db_sub_psi_dzeta += np.sum(-n_nyq * bsubs * cos_a, axis=0)

        db_dpsi += np.sum(d_b_dpsi * cos_a, axis=0)
        db_sub_theta_dpsi += np.sum(d_bsubu_dpsi * cos_a, axis=0)
        db_sub_zeta_dpsi += np.sum(d_bsubv_dpsi * cos_a, axis=0)

    # Convert Jacobian to inverse Jacobian:
    d_hat = 1.0 / d_hat

    # Zeros for quantities not yet computed for VMEC in sfincs_jax:
    zeros = np.zeros_like(b_hat)

    # iota for `sfincsOutput.h5` is read separately as a flux function; here we store it in the geometry struct
    # as the half-mesh interpolated value (consistent with v3).
    iota = float(w.iotas[i_half0] * w_half0 + w.iotas[i_half1] * w_half1)

    # v3 computes `B0OverBBar`, `GHat`, and `IHat` for VMEC geometries later in
    # `geometry.F90:computeBIntegrals` for output/reporting. For the subset of geometry arrays
    # we port here, keep the scalar placeholders in the geometry struct.
    #
    # Callers that need these flux functions (e.g. RHSMode=3 monoenergetic overwrites) should
    # compute them from the arrays, matching v3 output.
    b0_over_bbar = 0.0
    g_hat = 0.0
    i_hat = 0.0

    return BoozerGeometry(
        n_periods=n_periods,
        b0_over_bbar=b0_over_bbar,
        iota=iota,
        g_hat=g_hat,
        i_hat=i_hat,
        b_hat=jnp.asarray(b_hat),
        db_hat_dtheta=jnp.asarray(db_dtheta),
        db_hat_dzeta=jnp.asarray(db_dzeta),
        d_hat=jnp.asarray(d_hat),
        b_hat_sup_theta=jnp.asarray(b_sup_theta),
        b_hat_sup_zeta=jnp.asarray(b_sup_zeta),
        b_hat_sub_theta=jnp.asarray(b_sub_theta),
        b_hat_sub_zeta=jnp.asarray(b_sub_zeta),
        b_hat_sub_psi=jnp.asarray(b_sub_psi),
        db_hat_dpsi_hat=jnp.asarray(db_dpsi),
        db_hat_sub_psi_dtheta=jnp.asarray(db_sub_psi_dtheta),
        db_hat_sub_psi_dzeta=jnp.asarray(db_sub_psi_dzeta),
        db_hat_sub_theta_dpsi_hat=jnp.asarray(db_sub_theta_dpsi),
        db_hat_sub_zeta_dpsi_hat=jnp.asarray(db_sub_zeta_dpsi),
        db_hat_sub_theta_dzeta=jnp.asarray(db_sub_theta_dzeta),
        db_hat_sub_zeta_dtheta=jnp.asarray(db_sub_zeta_dtheta),
        db_hat_sup_theta_dpsi_hat=jnp.asarray(db_sup_theta_dpsi),
        db_hat_sup_theta_dzeta=jnp.asarray(db_sup_theta_dzeta),
        db_hat_sup_zeta_dpsi_hat=jnp.asarray(db_sup_zeta_dpsi),
        db_hat_sup_zeta_dtheta=jnp.asarray(db_sup_zeta_dtheta),
    )
