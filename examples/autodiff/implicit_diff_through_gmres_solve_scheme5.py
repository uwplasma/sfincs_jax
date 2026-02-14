"""Implicit-differentiation through a full-system Krylov solve (VMEC `geometryScheme=5`).

This example demonstrates *end-to-end differentiability through a linear solve* using
implicit differentiation via `jax.lax.custom_linear_solve`.

Instead of backpropagating through GMRES iterations (slow and numerically brittle), we use
JAX's `custom_linear_solve` to define the gradient via the adjoint/transpose solve:

  A x = b,    dL/dp = λᵀ (db/dp - (dA/dp) x),    where Aᵀ λ = dL/dx.

This is a key ingredient for gradient-based stellarator optimization workflows (e.g. electron-root
design studies) once the physics model is fully ported.

We treat `nu_n` (normalized collisionality) as a differentiable scalar and compute:

  d/dnu_n  ( 0.5 * || x(nu_n) ||^2 ),  where  A(nu_n) x = rhs(nu_n).

Requirements: none beyond the base `sfincs_jax` install.

You can switch between GMRES and BiCGStab with `--solver` (BiCGStab is lower-memory and
is the default for RHSMode=1 solves in `sfincs_jax`).
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

from sfincs_jax.implicit_solve import gmres_custom_linear_solve, linear_custom_solve
from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.v3_system import (
    apply_v3_full_system_operator,
    full_system_operator_from_namelist,
    rhs_v3_full_system,
)


def _default_input() -> Path:
    return Path(__file__).parents[2] / "tests" / "ref" / "pas_1species_PAS_noEr_tiny_scheme5.input.namelist"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=str(_default_input()))
    p.add_argument("--solver", choices=("gmres", "bicgstab"), default="gmres")
    p.add_argument("--eps", type=float, default=1e-5, help="finite-difference step for a quick check")
    args = p.parse_args()

    nml = read_sfincs_input(Path(args.input))
    op0 = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)

    if op0.fblock.pas is None:
        raise SystemExit("This example expects collisionOperator=1 (PAS), but op.fblock.pas is None.")

    nu0 = jnp.asarray(op0.fblock.pas.nu_n, dtype=jnp.float64)

    def objective(nu_n: jnp.ndarray) -> jnp.ndarray:
        pas2 = replace(op0.fblock.pas, nu_n=jnp.asarray(nu_n, dtype=jnp.float64))
        op = replace(op0, fblock=replace(op0.fblock, pas=pas2))

        b = rhs_v3_full_system(op)

        def mv(x: jnp.ndarray) -> jnp.ndarray:
            return apply_v3_full_system_operator(op, x)

        if args.solver == "bicgstab":
            x = linear_custom_solve(
                matvec=mv,
                b=b,
                tol=1e-12,
                maxiter=400,
                solver="bicgstab",
            ).x
        else:
            x = gmres_custom_linear_solve(matvec=mv, b=b, tol=1e-12, restart=80, maxiter=400).x
        return 0.5 * jnp.vdot(x, x)

    g = jax.grad(objective)(nu0)

    eps = float(args.eps)
    fd = (float(objective(nu0 + eps)) - float(objective(nu0 - eps))) / (2.0 * eps)

    print(f"nu_n0 = {float(nu0):.6g}")
    print(f"objective(nu_n0) = {float(objective(nu0)):.6e}")
    print(f"d objective / d nu_n  (implicit-diff) = {float(g):.6e}")
    print(f"d objective / d nu_n  (finite-diff)   = {fd:.6e}")
    print(f"abs_err = {abs(float(g) - fd):.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
