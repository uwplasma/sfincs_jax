"""Micro-benchmark: JIT speedup for a matrix-free operator apply.

This script is intentionally simple and avoids optional dependencies. It is not a
scientific benchmark; it just demonstrates the typical benefit of JIT compilation
for repeated operator applications.
"""

from __future__ import annotations

import time
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sfincs_jax.collisionless import CollisionlessV3Operator, apply_collisionless_v3
from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.v3 import geometry_from_namelist, grids_from_namelist


def main() -> int:
    input_path = _REPO_ROOT / "tests" / "ref" / "quick_2species_FPCollisions_noEr.input.namelist"
    nml = read_sfincs_input(input_path)
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

    rng = np.random.default_rng(0)
    f0 = jnp.asarray(
        rng.normal(size=(t_hats.size, int(grids.x.shape[0]), int(grids.n_xi), int(grids.theta.shape[0]), int(grids.zeta.shape[0])))
        .astype(np.float64)
    )

    f0.block_until_ready()

    def apply(f):
        return apply_collisionless_v3(op, f)

    apply_jit = jax.jit(apply)

    # Warmup (compile):
    y = apply_jit(f0)
    y.block_until_ready()

    nrep = 50

    t0 = time.perf_counter()
    for _ in range(nrep):
        y = apply(f0)
    y.block_until_ready()
    t1 = time.perf_counter()
    t_nojit = (t1 - t0) / nrep

    t0 = time.perf_counter()
    for _ in range(nrep):
        y = apply_jit(f0)
    y.block_until_ready()
    t1 = time.perf_counter()
    t_jit = (t1 - t0) / nrep

    print(f"nrep={nrep}")
    print(f"mean time/apply (nojit): {t_nojit*1e3:.3f} ms")
    print(f"mean time/apply (jit):   {t_jit*1e3:.3f} ms")
    if t_jit > 0:
        print(f"speedup: {t_nojit/t_jit:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

