from __future__ import annotations

from pathlib import Path

import numpy as np
import jax.numpy as jnp

from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.v3_system import apply_v3_full_system_operator, full_system_operator_from_namelist


def test_phi1_qn_lambda_blocks_have_expected_size_and_action() -> None:
    """Unit-test the Phi1/QN/lambda block wiring (no Fortran parity fixture required)."""
    here = Path(__file__).parent
    input_path = here / "ref" / "include_phi1_linear_subset_tiny.input.namelist"
    nml = read_sfincs_input(input_path)

    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)
    assert bool(op.include_phi1)
    assert op.phi1_size == op.n_theta * op.n_zeta + 1

    # Build a state with f=0 and sources=0 so only the QN/lambda blocks contribute.
    f_size = op.f_size
    phi1_n = op.n_theta * op.n_zeta
    extra_n = op.extra_size

    phi1 = jnp.ones((phi1_n,), dtype=jnp.float64)
    lam = jnp.asarray([2.0], dtype=jnp.float64)
    x = jnp.concatenate([jnp.zeros((f_size,), dtype=jnp.float64), phi1, lam, jnp.zeros((extra_n,), dtype=jnp.float64)])

    y = np.asarray(apply_v3_full_system_operator(op, x))
    y_f = y[:f_size]
    y_qn = y[f_size : f_size + phi1_n]
    y_lam = y[f_size + phi1_n]

    # DKE block should be zero since f is zero and no sources were set.
    np.testing.assert_allclose(y_f, 0.0, rtol=0, atol=1e-12)

    # QN block should be spatially uniform for phi1=1 and lambda=2, since the diagonal coefficient is constant
    # (geometryScheme=4 DHat varies but only appears in the lambda constraint, not in this QN diagonal term).
    np.testing.assert_allclose(y_qn, y_qn[0], rtol=0, atol=1e-12)
    # With the parameters in the fixture namelist, the QN diagonal is:
    #   phi1_diag = -alpha * (Z^2 n/T + Za^2 na/Ta) = -(1 + 1) = -2,
    # so qn = phi1_diag*phi1 + lambda = -2*1 + 2 = 0.
    np.testing.assert_allclose(y_qn[0], 0.0, rtol=0, atol=1e-12)

    # Lambda row is the flux-surface average of Phi1. For phi1=1, this is just the sum of weights/DHat.
    factor = np.asarray(op.theta_weights)[:, None] * np.asarray(op.zeta_weights)[None, :] / np.asarray(op.d_hat)
    expected_lam = float(np.sum(factor))
    np.testing.assert_allclose(y_lam, expected_lam, rtol=0, atol=1e-12)
