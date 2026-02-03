"""Parity-check the collisionless operator (streaming + mirror) against Fortran PETSc binaries.

This script mirrors `tests/test_collisionless_operator_parity.py`, but prints a short
summary instead of asserting.

By default it uses the repository fixture in `tests/ref/quick_2species_FPCollisions_noEr.*`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import jax.numpy as jnp

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sfincs_jax.collisionless import CollisionlessV3Operator, apply_collisionless_v3
from sfincs_jax.indices import V3Indexing
from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.petsc_binary import read_petsc_mat_aij, read_petsc_vec
from sfincs_jax.v3 import geometry_from_namelist, grids_from_namelist


def _default_prefix() -> Path:
    return Path(__file__).parents[2] / "tests" / "ref" / "quick_2species_FPCollisions_noEr"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=None, help="Path to input.namelist (default: fixture)")
    p.add_argument("--mat", default=None, help="Path to whichMatrix_3 petscbin (default: fixture)")
    p.add_argument("--vec", default=None, help="Path to stateVector petscbin (default: fixture)")
    args = p.parse_args()

    prefix = _default_prefix()
    input_path = Path(args.input) if args.input else Path(str(prefix) + ".input.namelist")
    mat_path = Path(args.mat) if args.mat else Path(str(prefix) + ".whichMatrix_3.petscbin")
    vec_path = Path(args.vec) if args.vec else Path(str(prefix) + ".stateVector.petscbin")

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

    a = read_petsc_mat_aij(mat_path)
    x_full = read_petsc_vec(vec_path).values

    indexing = V3Indexing(
        n_species=int(t_hats.size),
        n_x=int(grids.x.shape[0]),
        n_theta=int(grids.theta.shape[0]),
        n_zeta=int(grids.zeta.shape[0]),
        n_xi_max=int(grids.n_xi),
        n_xi_for_x=np.asarray(grids.n_xi_for_x, dtype=int),
    )
    inv = indexing.build_inverse_f_map()
    n_f = len(inv)

    f = np.zeros(
        (
            indexing.n_species,
            indexing.n_x,
            indexing.n_xi_max,
            indexing.n_theta,
            indexing.n_zeta,
        ),
        dtype=np.float64,
    )
    for g, (s, ix, l, it, iz) in enumerate(inv):
        f[s, ix, l, it, iz] = x_full[g]

    y_jax = np.asarray(apply_collisionless_v3(op, jnp.asarray(f)))

    # Restrict A@x to columns with |ΔL|=1 within the distribution block.
    y_ref = np.zeros((n_f,), dtype=np.float64)
    for row in range(n_f):
        s_r, ix_r, l_r, _, _ = inv[row]
        start = int(a.row_ptr[row])
        end = int(a.row_ptr[row + 1])
        cols = a.col_ind[start:end]
        vals = a.data[start:end]
        acc = 0.0
        for c, v in zip(cols.tolist(), vals.tolist()):
            if c < 0 or c >= n_f:
                continue
            s_c, ix_c, l_c, _, _ = inv[c]
            if s_c != s_r or ix_c != ix_r:
                continue
            if abs(l_c - l_r) != 1:
                continue
            acc += float(v) * float(x_full[c])
        y_ref[row] = acc

    y_jax_vec = np.zeros((n_f,), dtype=np.float64)
    for g, (s, ix, l, it, iz) in enumerate(inv):
        y_jax_vec[g] = y_jax[s, ix, l, it, iz]

    max_abs = float(np.max(np.abs(y_jax_vec - y_ref)))
    print(f"max |jax - fortran| = {max_abs:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
