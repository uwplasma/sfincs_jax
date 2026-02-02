from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from sfincs_jax.diagnostics import u_hat, u_hat_np
from sfincs_jax.geometry import boozer_geometry_scheme4
from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.v3 import geometry_from_namelist, grids_from_namelist


def test_u_hat_fft_matches_numpy_reference_for_scheme4_fixture() -> None:
    input_path = Path(__file__).parent / "ref" / "quick_2species_FPCollisions_noEr.input.namelist"
    nml = read_sfincs_input(input_path)
    grids = grids_from_namelist(nml)
    geom = geometry_from_namelist(nml=nml, grids=grids)

    u_fft = np.asarray(u_hat(grids=grids, geom=geom))
    u_np = u_hat_np(grids=grids, geom=geom)

    np.testing.assert_allclose(u_fft, u_np, rtol=0, atol=1e-11)


def test_u_hat_is_differentiable_wrt_scheme4_harmonics() -> None:
    input_path = Path(__file__).parent / "ref" / "quick_2species_FPCollisions_noEr.input.namelist"
    nml = read_sfincs_input(input_path)
    grids = grids_from_namelist(nml)

    amp0 = jnp.asarray([0.04645, -0.04351, -0.01902], dtype=jnp.float64)

    def objective(a: jnp.ndarray) -> jnp.ndarray:
        geom = boozer_geometry_scheme4(theta=grids.theta, zeta=grids.zeta, harmonics_amp0=a)
        u = u_hat(grids=grids, geom=geom)
        return jnp.sum(u * u)

    g = jax.grad(objective)(amp0)
    assert g.shape == amp0.shape
    assert bool(jnp.all(jnp.isfinite(g)))

