from __future__ import annotations

from pathlib import Path

import numpy as np
import jax.numpy as jnp

from sfincs_jax.collisionless_er import ErXiDotV3Operator, apply_er_xidot_v3_offdiag2
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


def test_er_xidot_offdiag2_matvec_matches_fortran_matrix() -> None:
    """Parity test for the Er xiDot term (|ΔL|=2) against a Fortran v3 Jacobian matrix."""
    here = Path(__file__).parent
    input_path = here / "ref" / "er_xidot_1species_tiny.input.namelist"
    mat_path = here / "ref" / "er_xidot_1species_tiny.whichMatrix_1.petscbin"

    nml = read_sfincs_input(input_path)
    grids = grids_from_namelist(nml)

    geom = boozer_geometry_scheme4(theta=grids.theta, zeta=grids.zeta)
    phys = nml.group("physicsParameters")
    alpha = float(phys.get("ALPHA", 1.0))
    delta = float(phys.get("DELTA", 0.0))
    er = float(phys.get("ER", 0.0))
    dphi = _dphi_hat_dpsi_hat_from_er_scheme4(er)

    op = ErXiDotV3Operator(
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

    # Pack into padded f tensor.
    f = np.zeros((1, indexing.n_x, indexing.n_xi_max, indexing.n_theta, indexing.n_zeta), dtype=np.float64)
    for g, (s, ix, l, it, iz) in enumerate(inv):
        f[s, ix, l, it, iz] = x_vec[g]

    y_jax = np.asarray(apply_er_xidot_v3_offdiag2(op, jnp.asarray(f)))

    # Fortran reference: restrict A@x to |ΔL|=2 *and* same x,theta,zeta within the F block.
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
            if (s_c, ix_c, it_c, iz_c) != (s_r, ix_r, it_r, iz_r):
                continue
            if abs(l_c - l_r) != 2:
                continue
            acc += float(v) * float(x_vec[c])
        y_ref[row] = acc

    y_jax_vec = np.zeros((n_f,), dtype=np.float64)
    for g, (s, ix, l, it, iz) in enumerate(inv):
        y_jax_vec[g] = y_jax[s, ix, l, it, iz]

    np.testing.assert_allclose(y_jax_vec, y_ref, rtol=0, atol=1e-12)
