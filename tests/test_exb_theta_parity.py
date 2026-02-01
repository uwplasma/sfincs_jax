from __future__ import annotations

from pathlib import Path

import numpy as np
import jax.numpy as jnp

from sfincs_jax.collisionless_exb import ExBThetaV3Operator, apply_exb_theta_v3
from sfincs_jax.geometry import boozer_geometry_scheme4
from sfincs_jax.indices import V3Indexing
from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.petsc_binary import read_petsc_mat_aij
from sfincs_jax.v3 import grids_from_namelist


def _dphi_hat_dpsi_hat_from_er_scheme4(er: float) -> float:
    # Matches geometry.F90 for scheme=4 and radialCoordinates.F90 with
    # inputRadialCoordinateForGradients=4 (default): dPhiHatdpsiHat = ddrHat2ddpsiHat * (-Er).
    psi_a_hat = -0.384935
    a_hat = 0.5109
    psi_n = 0.25  # rN=0.5 is forced for geometryScheme=4
    ddrhat2ddpsihat = a_hat / (2.0 * psi_a_hat * np.sqrt(psi_n))
    return float(ddrhat2ddpsihat * (-er))


def _fsab_hat2(*, theta_weights: np.ndarray, zeta_weights: np.ndarray, d_hat: np.ndarray, b_hat: np.ndarray) -> float:
    # Matches geometry.F90:computeBIntegrals.
    w = theta_weights[:, None] * zeta_weights[None, :]
    vprime_hat = float(np.sum(w / d_hat))
    return float(np.sum(w * (b_hat**2) / d_hat) / vprime_hat)


def test_exb_theta_matvec_matches_fortran_matrix() -> None:
    """Parity test for the ExB d/dtheta term against a Fortran v3 Jacobian matrix."""
    here = Path(__file__).parent
    input_path = here / "ref" / "exb_theta_1species_tiny.input.namelist"
    mat_path = here / "ref" / "exb_theta_1species_tiny.whichMatrix_1.petscbin"

    nml = read_sfincs_input(input_path)
    grids = grids_from_namelist(nml)
    geom = boozer_geometry_scheme4(theta=grids.theta, zeta=grids.zeta)

    phys = nml.group("physicsParameters")
    alpha = float(phys.get("ALPHA", 1.0))
    delta = float(phys.get("DELTA", 0.0))
    er = float(phys.get("ER", 0.0))
    use_dkes_exb = bool(phys.get("USEDKESEXBDRIFT", False))
    dphi = _dphi_hat_dpsi_hat_from_er_scheme4(er)
    fsab_hat2 = _fsab_hat2(
        theta_weights=np.asarray(grids.theta_weights, dtype=np.float64),
        zeta_weights=np.asarray(grids.zeta_weights, dtype=np.float64),
        d_hat=np.asarray(geom.d_hat, dtype=np.float64),
        b_hat=np.asarray(geom.b_hat, dtype=np.float64),
    )

    op = ExBThetaV3Operator(
        alpha=jnp.asarray(alpha, dtype=jnp.float64),
        delta=jnp.asarray(delta, dtype=jnp.float64),
        dphi_hat_dpsi_hat=jnp.asarray(dphi, dtype=jnp.float64),
        ddtheta=grids.ddtheta,
        d_hat=geom.d_hat,
        b_hat=geom.b_hat,
        b_hat_sub_zeta=geom.b_hat_sub_zeta,
        use_dkes_exb_drift=jnp.asarray(use_dkes_exb),
        fsab_hat2=jnp.asarray(fsab_hat2, dtype=jnp.float64),
        n_xi_for_x=grids.n_xi_for_x,
    )

    a = read_petsc_mat_aij(mat_path)

    indexing = V3Indexing(
        n_species=1,
        n_x=int(grids.x.shape[0]),
        n_theta=int(grids.theta.shape[0]),
        n_zeta=int(grids.zeta.shape[0]),
        n_xi_max=int(grids.n_xi),
        n_xi_for_x=np.asarray(grids.n_xi_for_x, dtype=int),
    )
    inv = indexing.build_inverse_f_map()
    n_f = len(inv)

    rng = np.random.default_rng(0)
    x_vec = rng.normal(size=(n_f,)).astype(np.float64)

    f = np.zeros((1, indexing.n_x, indexing.n_xi_max, indexing.n_theta, indexing.n_zeta), dtype=np.float64)
    for g, (s, ix, l, it, iz) in enumerate(inv):
        f[s, ix, l, it, iz] = x_vec[g]

    y_jax = np.asarray(apply_exb_theta_v3(op, jnp.asarray(f)))

    # Fortran reference: restrict A@x to entries that are diagonal in (species,x,L,zeta).
    y_ref = np.zeros((n_f,), dtype=np.float64)
    for row in range(n_f):
        s_r, ix_r, l_r, it_r, iz_r = inv[row]
        start = int(a.row_ptr[row])
        end = int(a.row_ptr[row + 1])
        cols = a.col_ind[start:end]
        vals = a.data[start:end]
        acc = 0.0
        for c, v in zip(cols.tolist(), vals.tolist()):
            if c < 0 or c >= n_f:
                continue
            s_c, ix_c, l_c, it_c, iz_c = inv[c]
            if (s_c, ix_c, l_c, iz_c) != (s_r, ix_r, l_r, iz_r):
                continue
            acc += float(v) * float(x_vec[c])
        y_ref[row] = acc

    y_jax_vec = np.zeros((n_f,), dtype=np.float64)
    for g, (s, ix, l, it, iz) in enumerate(inv):
        y_jax_vec[g] = y_jax[s, ix, l, it, iz]

    np.testing.assert_allclose(y_jax_vec, y_ref, rtol=0, atol=1e-12)

