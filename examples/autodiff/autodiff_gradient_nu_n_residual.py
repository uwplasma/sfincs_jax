from __future__ import annotations

"""
Autodiff demo: gradient of a parity objective w.r.t. the collision frequency `nu_n`.

This example is a lightweight version of the optax-based calibration example in
`examples/optimization/calibrate_nu_n_to_fortran_residual_fixture.py`:

- no optimization loop
- no optional dependencies

It demonstrates a key `sfincs_jax` capability for "design/optimization-style" workflows:
differentiate a physics objective through the (matrix-free) residual evaluation.
"""

from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp

from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.petsc_binary import read_petsc_vec
from sfincs_jax.v3_system import full_system_operator_from_namelist, residual_v3_full_system


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    input_path = repo_root / "tests" / "ref" / "pas_1species_PAS_noEr_tiny_scheme5.input.namelist"
    x_path = repo_root / "tests" / "ref" / "pas_1species_PAS_noEr_tiny_scheme5.stateVector.petscbin"
    r_path = repo_root / "tests" / "ref" / "pas_1species_PAS_noEr_tiny_scheme5.residual.petscbin"

    nml = read_sfincs_input(input_path)
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)
    if op.fblock.pas is None:
        raise RuntimeError("Expected collisionOperator=1 (PAS) fixture.")

    x_ref = jnp.asarray(read_petsc_vec(x_path).values)
    r_ref = jnp.asarray(read_petsc_vec(r_path).values)
    nu0 = jnp.asarray(op.fblock.pas.nu_n, dtype=jnp.float64)

    def loss(nu_n: jnp.ndarray) -> jnp.ndarray:
        pas2 = replace(op.fblock.pas, nu_n=jnp.asarray(nu_n, dtype=jnp.float64))
        op2 = replace(op, fblock=replace(op.fblock, pas=pas2))
        r = residual_v3_full_system(op2, x_ref)
        d = r - r_ref
        return 0.5 * jnp.vdot(d, d)

    val, g = jax.value_and_grad(loss)(nu0)
    print("nu_n:", float(nu0))
    print("loss:", float(val))
    print("d(loss)/d(nu_n):", float(g))


if __name__ == "__main__":
    main()
