"""JIT-compiled gradient-based optimization with implicit differentiation.

This example shows a fully JIT-compiled objective that:
  * builds the full-system RHS and matvec from a cached operator,
  * solves A x = b with implicit differentiation (custom_linear_solve),
  * optimizes a scalar parameter (nu_n) using gradient descent.

We treat nu_n (normalized collisionality) as a differentiable parameter and
minimize 0.5 * ||x(nu_n)||^2, where x is the implicit solution of the linear system.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp

from sfincs_jax.implicit_solve import gmres_custom_linear_solve
from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.v3_system import (
    apply_v3_full_system_operator,
    full_system_operator_from_namelist,
    rhs_v3_full_system,
)


def _default_input() -> Path:
    return Path(__file__).parents[2] / "tests" / "ref" / "pas_1species_PAS_noEr_tiny_scheme5.input.namelist"


def main() -> int:
    nml = read_sfincs_input(_default_input())
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

        x = gmres_custom_linear_solve(matvec=mv, b=b, tol=1e-12, restart=80, maxiter=400).x
        return 0.5 * jnp.vdot(x, x)

    value_and_grad = jax.jit(jax.value_and_grad(objective))

    nu = nu0
    step = 0.1
    for k in range(5):
        loss, grad = value_and_grad(nu)
        nu = nu - step * grad
        print(f"iter={k:02d} nu_n={float(nu):.6g} loss={float(loss):.6e} grad={float(grad):.6e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
