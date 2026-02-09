"""Autodiff sensitivity demo for the full-system residual (VMEC `geometryScheme=5`).

This example shows a core advantage of a JAX port: automatic differentiation through the
discrete operator/residual without forming sparse matrices.

We treat `nu_n` (the normalized collisionality parameter) as a differentiable scalar and compute:

  d/dnu_n  ( 0.5 * || r(nu_n; x0) ||^2 )

where `x0` is a perturbed state vector built from a frozen Fortran v3 `stateVector` fixture.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.petsc_binary import read_petsc_vec
from sfincs_jax.v3_system import full_system_operator_from_namelist, residual_v3_full_system


def _default_input() -> Path:
    return Path(__file__).parents[2] / "tests" / "ref" / "pas_1species_PAS_noEr_tiny_scheme5.input.namelist"


def _default_statevector() -> Path:
    return Path(__file__).parents[2] / "tests" / "ref" / "pas_1species_PAS_noEr_tiny_scheme5.stateVector.petscbin"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=str(_default_input()))
    p.add_argument("--statevector", default=str(_default_statevector()))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--noise", type=float, default=1e-3, help="additive perturbation scale applied to x_ref")
    p.add_argument("--eps", type=float, default=1e-5, help="finite-difference step for a quick check")
    args = p.parse_args()

    nml = read_sfincs_input(Path(args.input))
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)
    x_ref = jnp.asarray(read_petsc_vec(Path(args.statevector)).values)
    rng = np.random.default_rng(int(args.seed))
    noise = float(args.noise)
    x0 = x_ref + jnp.asarray(noise * rng.normal(size=(x_ref.size,)).astype(np.float64))

    if op.fblock.pas is None:
        raise SystemExit("This example expects collisionOperator=1 (PAS), but op.fblock.pas is None.")

    def objective(nu_n: jnp.ndarray) -> jnp.ndarray:
        pas2 = replace(op.fblock.pas, nu_n=jnp.asarray(nu_n, dtype=jnp.float64))
        fblock2 = replace(op.fblock, pas=pas2)
        op2 = replace(op, fblock=fblock2)
        r = residual_v3_full_system(op2, x0)
        return 0.5 * jnp.vdot(r, r)

    nu0 = jnp.asarray(op.fblock.pas.nu_n, dtype=jnp.float64)
    g = jax.grad(objective)(nu0)

    eps = float(args.eps)
    fd = (float(objective(nu0 + eps)) - float(objective(nu0 - eps))) / (2.0 * eps)

    print(f"nu_n0 = {float(nu0):.6g}")
    print(f"objective(nu_n0) = {float(objective(nu0)):.6e}")
    print(f"d objective / d nu_n  (autodiff) = {float(g):.6e}")
    print(f"d objective / d nu_n  (finite-diff) = {fd:.6e}")
    print(f"abs_err = {abs(float(g) - fd):.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
