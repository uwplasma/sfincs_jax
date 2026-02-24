from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp
import numpy as np

from sfincs_jax.collisionless_exb import (
    ExBThetaV3Operator,
    ExBZetaV3Operator,
    apply_exb_theta_v3,
    apply_exb_zeta_v3,
)
from sfincs_jax.magnetic_drifts import (
    MagneticDriftThetaV3Operator,
    MagneticDriftZetaV3Operator,
    apply_magnetic_drift_theta_v3,
    apply_magnetic_drift_zeta_v3,
)
from sfincs_jax.periodic_stencil import extract_sparse_row_stencil


def _periodic_first_derivative_matrix(n: int) -> np.ndarray:
    d = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        d[i, (i + 1) % n] = 0.5
        d[i, (i - 1) % n] = -0.5
    return d


def _random_f(*, n_x: int, n_xi: int, n_theta: int, n_zeta: int) -> jnp.ndarray:
    rng = np.random.default_rng(0)
    f = rng.normal(size=(1, n_x, n_xi, n_theta, n_zeta)).astype(np.float64)
    return jnp.asarray(f)


def test_exb_sparse_theta_matches_dense() -> None:
    n_theta, n_zeta = 9, 7
    ddtheta = _periodic_first_derivative_matrix(n_theta)
    cols, vals = extract_sparse_row_stencil(ddtheta)
    f = _random_f(n_x=3, n_xi=5, n_theta=n_theta, n_zeta=n_zeta)

    op_dense = ExBThetaV3Operator(
        alpha=jnp.asarray(1.0),
        delta=jnp.asarray(0.2),
        dphi_hat_dpsi_hat=jnp.asarray(0.1),
        ddtheta=jnp.asarray(ddtheta),
        d_hat=jnp.ones((n_theta, n_zeta), dtype=jnp.float64),
        b_hat=jnp.ones((n_theta, n_zeta), dtype=jnp.float64),
        b_hat_sub_zeta=jnp.ones((n_theta, n_zeta), dtype=jnp.float64) * 0.3,
        use_dkes_exb_drift=jnp.asarray(False),
        fsab_hat2=jnp.asarray(1.0),
        n_xi_for_x=jnp.asarray([5, 4, 3], dtype=jnp.int32),
    )
    op_sparse = replace(
        op_dense,
        ddtheta_sparse_cols=jnp.asarray(cols, dtype=jnp.int32),
        ddtheta_sparse_vals=jnp.asarray(vals, dtype=jnp.float64),
    )

    y_dense = np.asarray(apply_exb_theta_v3(op_dense, f))
    y_sparse = np.asarray(apply_exb_theta_v3(op_sparse, f))
    np.testing.assert_allclose(y_sparse, y_dense, rtol=0, atol=1e-12)


def test_exb_sparse_zeta_matches_dense() -> None:
    n_theta, n_zeta = 7, 9
    ddzeta = _periodic_first_derivative_matrix(n_zeta)
    cols, vals = extract_sparse_row_stencil(ddzeta)
    f = _random_f(n_x=2, n_xi=4, n_theta=n_theta, n_zeta=n_zeta)

    op_dense = ExBZetaV3Operator(
        alpha=jnp.asarray(1.0),
        delta=jnp.asarray(0.2),
        dphi_hat_dpsi_hat=jnp.asarray(0.1),
        ddzeta=jnp.asarray(ddzeta),
        d_hat=jnp.ones((n_theta, n_zeta), dtype=jnp.float64),
        b_hat=jnp.ones((n_theta, n_zeta), dtype=jnp.float64),
        b_hat_sub_theta=jnp.ones((n_theta, n_zeta), dtype=jnp.float64) * -0.2,
        use_dkes_exb_drift=jnp.asarray(False),
        fsab_hat2=jnp.asarray(1.0),
        n_xi_for_x=jnp.asarray([4, 3], dtype=jnp.int32),
    )
    op_sparse = replace(
        op_dense,
        ddzeta_sparse_cols=jnp.asarray(cols, dtype=jnp.int32),
        ddzeta_sparse_vals=jnp.asarray(vals, dtype=jnp.float64),
    )

    y_dense = np.asarray(apply_exb_zeta_v3(op_dense, f))
    y_sparse = np.asarray(apply_exb_zeta_v3(op_sparse, f))
    np.testing.assert_allclose(y_sparse, y_dense, rtol=0, atol=1e-12)


def test_magnetic_drift_sparse_theta_matches_dense() -> None:
    n_theta, n_zeta = 9, 7
    dd_plus = _periodic_first_derivative_matrix(n_theta)
    dd_minus = -0.7 * dd_plus
    cols_plus, vals_plus = extract_sparse_row_stencil(dd_plus)
    cols_minus, vals_minus = extract_sparse_row_stencil(dd_minus)
    f = _random_f(n_x=3, n_xi=5, n_theta=n_theta, n_zeta=n_zeta)
    base_2d = jnp.ones((n_theta, n_zeta), dtype=jnp.float64)

    op_dense = MagneticDriftThetaV3Operator(
        delta=jnp.asarray(0.1),
        t_hat=jnp.asarray(1.0),
        z=jnp.asarray(1.0),
        x=jnp.asarray(np.linspace(0.1, 2.0, 3), dtype=jnp.float64),
        ddtheta_plus=jnp.asarray(dd_plus),
        ddtheta_minus=jnp.asarray(dd_minus),
        d_hat=base_2d,
        b_hat=base_2d,
        b_hat_sub_zeta=base_2d * 0.2,
        b_hat_sub_psi=base_2d * 0.1,
        db_hat_dzeta=base_2d * 0.4,
        db_hat_dpsi_hat=base_2d * -0.3,
        db_hat_sub_psi_dzeta=base_2d * 0.6,
        db_hat_sub_zeta_dpsi_hat=base_2d * -0.2,
        n_xi_for_x=jnp.asarray([5, 4, 3], dtype=jnp.int32),
    )
    op_sparse = replace(
        op_dense,
        ddtheta_plus_sparse_cols=jnp.asarray(cols_plus, dtype=jnp.int32),
        ddtheta_plus_sparse_vals=jnp.asarray(vals_plus, dtype=jnp.float64),
        ddtheta_minus_sparse_cols=jnp.asarray(cols_minus, dtype=jnp.int32),
        ddtheta_minus_sparse_vals=jnp.asarray(vals_minus, dtype=jnp.float64),
    )

    y_dense = np.asarray(apply_magnetic_drift_theta_v3(op_dense, f))
    y_sparse = np.asarray(apply_magnetic_drift_theta_v3(op_sparse, f))
    np.testing.assert_allclose(y_sparse, y_dense, rtol=0, atol=1e-12)


def test_magnetic_drift_sparse_zeta_matches_dense() -> None:
    n_theta, n_zeta = 7, 9
    dd_plus = _periodic_first_derivative_matrix(n_zeta)
    dd_minus = -0.5 * dd_plus
    cols_plus, vals_plus = extract_sparse_row_stencil(dd_plus)
    cols_minus, vals_minus = extract_sparse_row_stencil(dd_minus)
    f = _random_f(n_x=2, n_xi=4, n_theta=n_theta, n_zeta=n_zeta)
    base_2d = jnp.ones((n_theta, n_zeta), dtype=jnp.float64)

    op_dense = MagneticDriftZetaV3Operator(
        delta=jnp.asarray(0.1),
        t_hat=jnp.asarray(1.0),
        z=jnp.asarray(1.0),
        x=jnp.asarray(np.linspace(0.1, 2.0, 2), dtype=jnp.float64),
        ddzeta_plus=jnp.asarray(dd_plus),
        ddzeta_minus=jnp.asarray(dd_minus),
        d_hat=base_2d,
        b_hat=base_2d,
        b_hat_sub_theta=base_2d * -0.3,
        b_hat_sub_psi=base_2d * 0.2,
        db_hat_dtheta=base_2d * 0.7,
        db_hat_dpsi_hat=base_2d * -0.1,
        db_hat_sub_theta_dpsi_hat=base_2d * 0.4,
        db_hat_sub_psi_dtheta=base_2d * -0.8,
        n_xi_for_x=jnp.asarray([4, 3], dtype=jnp.int32),
    )
    op_sparse = replace(
        op_dense,
        ddzeta_plus_sparse_cols=jnp.asarray(cols_plus, dtype=jnp.int32),
        ddzeta_plus_sparse_vals=jnp.asarray(vals_plus, dtype=jnp.float64),
        ddzeta_minus_sparse_cols=jnp.asarray(cols_minus, dtype=jnp.int32),
        ddzeta_minus_sparse_vals=jnp.asarray(vals_minus, dtype=jnp.float64),
    )

    y_dense = np.asarray(apply_magnetic_drift_zeta_v3(op_dense, f))
    y_sparse = np.asarray(apply_magnetic_drift_zeta_v3(op_sparse, f))
    np.testing.assert_allclose(y_sparse, y_dense, rtol=0, atol=1e-12)
