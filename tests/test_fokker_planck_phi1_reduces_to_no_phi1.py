from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from sfincs_jax.collisions import (
    apply_fokker_planck_v3,
    apply_fokker_planck_v3_phi1,
    make_fokker_planck_v3_operator,
    make_fokker_planck_v3_phi1_operator,
)
from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.v3 import grids_from_namelist


def test_phi1_in_collisions_reduces_to_no_phi1_when_phi1_zero() -> None:
    nml = read_sfincs_input("tests/ref/quick_2species_FPCollisions_noEr.input.namelist")
    grids = grids_from_namelist(nml)
    species = nml.group("speciesParameters")
    phys = nml.group("physicsParameters")
    other = nml.group("otherNumericalParameters")

    x_grid_k = float(other.get("XGRID_K", 0.0))
    z_s = np.atleast_1d(np.asarray(species["ZS"], dtype=np.float64))
    m_hats = np.atleast_1d(np.asarray(species["MHATS"], dtype=np.float64))
    n_hats = np.atleast_1d(np.asarray(species["NHATS"], dtype=np.float64))
    t_hats = np.atleast_1d(np.asarray(species["THATS"], dtype=np.float64))
    nu_n = float(phys["NU_N"])
    krook = float(phys.get("KROOK", 0.0))
    alpha = float(phys.get("ALPHA", 1.0))

    op0 = make_fokker_planck_v3_operator(
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
    op_phi1 = make_fokker_planck_v3_phi1_operator(
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

    rng = np.random.default_rng(0)
    f = jnp.asarray(rng.standard_normal((z_s.size, grids.x.size, grids.n_xi, grids.theta.size, grids.zeta.size)))
    phi1_hat = jnp.zeros((grids.theta.size, grids.zeta.size), dtype=jnp.float64)

    y0 = apply_fokker_planck_v3(op0, f)
    y1 = apply_fokker_planck_v3_phi1(op_phi1, f, phi1_hat=phi1_hat)

    np.testing.assert_allclose(np.asarray(y1), np.asarray(y0), rtol=0, atol=2e-11)

