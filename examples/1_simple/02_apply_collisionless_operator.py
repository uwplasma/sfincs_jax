"""Apply the v3 collisionless operator (streaming + mirror) in JAX.

This example shows:
- Building the operator from a namelist-derived grid + geometry
- Running a non-jitted and jitted matvec and confirming they match

The operator is only a *subset* of v3 so far. See docs for the parity-first roadmap.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sfincs_jax.collisionless import CollisionlessV3Operator, apply_collisionless_v3, apply_collisionless_v3_jit
from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.v3 import geometry_from_namelist, grids_from_namelist


def _default_input() -> Path:
    return Path(__file__).parents[1] / "data" / "geometryScheme4_quick_2species.input.namelist"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=str(_default_input()))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    nml = read_sfincs_input(Path(args.input))
    grids = grids_from_namelist(nml)
    geom = geometry_from_namelist(nml=nml, grids=grids)

    species = nml.group("speciesParameters")
    t_hats = np.asarray(species["THATS"], dtype=np.float64)
    m_hats = np.asarray(species["MHATS"], dtype=np.float64)

    op = CollisionlessV3Operator(
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

    key = jax.random.key(args.seed)
    f = jax.random.normal(
        key,
        shape=(t_hats.size, grids.x.size, grids.n_xi, grids.theta.size, grids.zeta.size),
        dtype=jnp.float64,
    )

    y = apply_collisionless_v3(op, f)
    y_jit = apply_collisionless_v3_jit(op, f).block_until_ready()
    err = np.max(np.abs(np.asarray(y_jit) - np.asarray(y)))

    print(f"out.shape = {tuple(y.shape)}")
    print(f"max |jit - nojit| = {err:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
