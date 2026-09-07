from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from dkx.collisions import (
    _V3_SQRTPI,
    _psi_chandra,
    apply_pitch_angle_scattering_v3,
    apply_fokker_planck_v3,
    apply_fokker_planck_v3_phi1,
    FokkerPlanckV3Operator,
    FokkerPlanckV3Phi1Operator,
    make_fokker_planck_v3_operator,
    make_pitch_angle_scattering_v3_operator,
    nu_d_hat_pitch_angle_scattering_v3,
    polynomial_interpolation_matrix_np,
    rosenbluth_potential_terms_v3_np,
    prepare_fokker_planck_v3_profiles,
)
from dkx.xgrid import make_x_grid
from dkx.phase_space import make_speed_grid, speed_grid_diff_matrices


def _pas_operator():
    return make_pitch_angle_scattering_v3_operator(
        x=jnp.asarray([0.35, 0.9, 1.7], dtype=jnp.float64),
        z_s=jnp.asarray([1.0], dtype=jnp.float64),
        m_hats=jnp.asarray([1.0], dtype=jnp.float64),
        n_hats=jnp.asarray([1.0], dtype=jnp.float64),
        t_hats=jnp.asarray([1.0], dtype=jnp.float64),
        nu_n=0.7,
        krook=0.0,
        n_xi_for_x=jnp.asarray([4, 3, 2], dtype=jnp.int32),
        n_xi=5,
    )


def test_pas_l0_is_null_and_inactive_legendre_slots_are_masked() -> None:
    op = _pas_operator()
    f = np.ones((1, 3, 5, 2, 2), dtype=np.float64)
    out = np.asarray(apply_pitch_angle_scattering_v3(op, jnp.asarray(f)))

    np.testing.assert_allclose(out[:, :, 0, :, :], 0.0, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(out[:, 0, 4, :, :], 0.0, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(out[:, 1, 3:, :, :], 0.0, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(out[:, 2, 2:, :, :], 0.0, atol=0.0, rtol=0.0)
    assert np.all(out[:, :, 1, :, :] > 0.0)


def test_pas_arbitrary_l0_is_null_and_anisotropic_modes_are_dissipative() -> None:
    op = _pas_operator()
    f = np.zeros((1, 3, 5, 2, 2), dtype=np.float64)
    f[:, :, 0, :, :] = np.asarray(
        [
            [
                [[1.0, -0.5], [0.25, 2.0]],
                [[-1.5, 0.7], [3.0, -2.0]],
                [[0.1, 0.3], [-0.4, 0.9]],
            ]
        ],
        dtype=np.float64,
    )
    out_l0 = np.asarray(apply_pitch_angle_scattering_v3(op, jnp.asarray(f)))
    np.testing.assert_allclose(out_l0, 0.0, rtol=0.0, atol=0.0)

    anisotropic = np.arange(1, 1 + np.prod(f.shape), dtype=np.float64).reshape(f.shape) / 11.0
    anisotropic[:, :, 0, :, :] = 0.0
    out = np.asarray(apply_pitch_angle_scattering_v3(op, jnp.asarray(anisotropic)))

    coef = np.asarray(op.coef[0], dtype=np.float64)
    ell_mask = np.arange(anisotropic.shape[2])[None, :] < np.asarray(op.n_xi_for_x)[:, None]
    expected_quadratic_form = np.sum(
        coef[None, :, :, None, None]
        * ell_mask[None, :, :, None, None]
        * anisotropic
        * anisotropic
    )
    actual_quadratic_form = np.sum(anisotropic * out)
    assert expected_quadratic_form > 0.0
    np.testing.assert_allclose(actual_quadratic_form, expected_quadratic_form, rtol=0.0, atol=1.0e-12)


def test_pas_legendre_eigenvalues_follow_l_lplus1_over_two() -> None:
    op = _pas_operator()
    coef = np.asarray(op.coef[0])

    # With krook=0, L=1 has factor 1, so higher active-L coefficients should
    # follow L(L+1)/2 relative to L=1 at each x.
    for ix, n_l_active in enumerate([4, 3, 2]):
        base = coef[ix, 1]
        assert base > 0.0
        for ell in range(1, n_l_active):
            expected = 0.5 * ell * (ell + 1.0)
            np.testing.assert_allclose(coef[ix, ell] / base, expected, rtol=2e-15, atol=2e-15)


def test_chandrasekhar_function_matches_small_x_limit() -> None:
    x = jnp.asarray([1.0e-12, 1.0e-10, 1.0e-8], dtype=jnp.float64)
    psi = np.asarray(_psi_chandra(x))
    expected_ratio = 2.0 / (3.0 * float(_V3_SQRTPI))
    np.testing.assert_allclose(psi / np.asarray(x), expected_ratio, rtol=1.0e-8, atol=1.0e-12)
    assert np.all(psi > 0.0)


def test_polynomial_interpolation_matrix_is_identity_on_matching_nodes() -> None:
    xk = np.asarray([0.2, 0.7, 1.4, 2.2], dtype=np.float64)
    alpxk = np.exp(-(xk * xk))
    mat = polynomial_interpolation_matrix_np(xk=xk, x=xk.copy(), alpxk=alpxk, alpx=alpxk.copy())
    np.testing.assert_allclose(mat, np.eye(xk.size), rtol=0.0, atol=1.0e-13)


def test_pas_deflection_frequency_has_coulomb_scaling() -> None:
    x = jnp.asarray([0.4, 1.0, 2.0], dtype=jnp.float64)
    common = {
        "x": x,
        "m_hats": jnp.asarray([1.0], dtype=jnp.float64),
        "t_hats": jnp.asarray([1.0], dtype=jnp.float64),
    }

    base = np.asarray(
        nu_d_hat_pitch_angle_scattering_v3(
            **common,
            z_s=jnp.asarray([1.0], dtype=jnp.float64),
            n_hats=jnp.asarray([1.0], dtype=jnp.float64),
        )
    )
    double_density = np.asarray(
        nu_d_hat_pitch_angle_scattering_v3(
            **common,
            z_s=jnp.asarray([1.0], dtype=jnp.float64),
            n_hats=jnp.asarray([2.0], dtype=jnp.float64),
        )
    )
    double_charge = np.asarray(
        nu_d_hat_pitch_angle_scattering_v3(
            **common,
            z_s=jnp.asarray([2.0], dtype=jnp.float64),
            n_hats=jnp.asarray([1.0], dtype=jnp.float64),
        )
    )

    assert np.all(np.isfinite(base))
    assert np.all(base > 0.0)
    np.testing.assert_allclose(double_density / base, 2.0, rtol=2.0e-15, atol=2.0e-15)
    np.testing.assert_allclose(double_charge / base, 16.0, rtol=2.0e-15, atol=2.0e-15)


def test_weighted_barycentric_interpolation_is_exact_for_polynomials() -> None:
    xk = np.asarray([0.15, 0.7, 1.4, 2.3], dtype=np.float64)
    x = np.asarray([0.2, 0.9, 1.8], dtype=np.float64)
    alpxk = np.exp(-(xk * xk))
    alpx = np.exp(-(x * x))
    mat = polynomial_interpolation_matrix_np(xk=xk, x=x, alpxk=alpxk, alpx=alpx)

    def polynomial(y: np.ndarray) -> np.ndarray:
        return 1.0 - 2.0 * y + 0.5 * y * y + 0.1 * y * y * y

    source = alpxk * polynomial(xk)
    expected = alpx * polynomial(x)
    np.testing.assert_allclose(mat @ source, expected, rtol=2.0e-15, atol=2.0e-15)


def test_rosenbluth_analytic_terms_match_quadpack_reference() -> None:
    xg = make_x_grid(n=3, k=0.0, include_point_at_x0=False)
    kwargs = {
        "x": xg.x,
        "x_weights": xg.dx_weights(),
        "x_grid_k": 0.0,
        "xg": xg,
        "z_s": np.asarray([1.0], dtype=np.float64),
        "m_hats": np.asarray([1.0], dtype=np.float64),
        "n_hats": np.asarray([1.0], dtype=np.float64),
        "t_hats": np.asarray([1.0], dtype=np.float64),
        "nl": 2,
    }

    quadpack = rosenbluth_potential_terms_v3_np(**kwargs, method="quadpack")
    analytic = rosenbluth_potential_terms_v3_np(**kwargs, method="analytic")

    assert quadpack.shape == (1, 1, 2, 3, 3)
    assert np.all(np.isfinite(quadpack))
    assert np.all(np.isfinite(analytic))
    np.testing.assert_allclose(analytic, quadpack, rtol=2.0e-13, atol=2.0e-13)


def test_fokker_planck_apply_matches_dense_x_matvec_and_masks_inactive_l() -> None:
    mat = np.zeros((1, 1, 3, 2, 2), dtype=np.float64)
    mat[0, 0, 0] = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    mat[0, 0, 1] = np.asarray([[0.5, -1.0], [2.0, 0.25]])
    mat[0, 0, 2] = np.asarray([[2.0, 0.0], [0.0, -1.0]])
    op = FokkerPlanckV3Operator(
        mat=jnp.asarray(mat),
        n_xi_for_x=jnp.asarray([3, 1], dtype=jnp.int32),
        # Deliberately use the wrong L-width so apply_fokker_planck_v3 must rebuild
        # the mask for the runtime n_xi.
        mask_xi=jnp.ones((2, 1), dtype=jnp.float64),
    )
    f = np.arange(1 * 2 * 3 * 2 * 1, dtype=np.float64).reshape(1, 2, 3, 2, 1)

    out = np.asarray(apply_fokker_planck_v3(op, jnp.asarray(f)))
    expected = np.zeros_like(f)
    for ell in range(3):
        for itheta in range(2):
            expected[0, :, ell, itheta, 0] = mat[0, 0, ell] @ f[0, :, ell, itheta, 0]
    expected[0, 1, 1:, :, :] = 0.0

    np.testing.assert_allclose(out, expected, rtol=0.0, atol=0.0)


def test_fokker_planck_apply_rejects_bad_shapes() -> None:
    op = FokkerPlanckV3Operator(
        mat=jnp.zeros((1, 1, 1, 1, 1), dtype=jnp.float64),
        n_xi_for_x=jnp.asarray([1], dtype=jnp.int32),
        mask_xi=jnp.ones((1, 1), dtype=jnp.float64),
    )
    with pytest.raises(ValueError, match="f must have shape"):
        apply_fokker_planck_v3(op, jnp.ones((1, 1, 1, 1), dtype=jnp.float64))

    bad_op = FokkerPlanckV3Operator(
        mat=jnp.zeros((1, 1, 2, 1, 1), dtype=jnp.float64),
        n_xi_for_x=jnp.asarray([1], dtype=jnp.int32),
        mask_xi=jnp.ones((1, 1), dtype=jnp.float64),
    )
    with pytest.raises(ValueError, match="op.mat has shape"):
        apply_fokker_planck_v3(bad_op, jnp.ones((1, 1, 1, 1, 1), dtype=jnp.float64))


def test_phi1_fokker_planck_apply_uses_boltzmann_density_factor_and_mask() -> None:
    op = FokkerPlanckV3Phi1Operator(
        nu_n=jnp.asarray(1.0, dtype=jnp.float64),
        krook=jnp.asarray(0.0, dtype=jnp.float64),
        alpha=jnp.asarray(2.0, dtype=jnp.float64),
        z_s=jnp.asarray([1.0], dtype=jnp.float64),
        n_hats=jnp.asarray([3.0], dtype=jnp.float64),
        t_hats=jnp.asarray([4.0], dtype=jnp.float64),
        nl=0,
        k_nu=jnp.zeros((1, 1, 1), dtype=jnp.float64),
        k_cd=jnp.asarray([[[[2.0]]]], dtype=jnp.float64),
        k_ce=jnp.zeros((1, 1, 1, 1), dtype=jnp.float64),
        k_rosen=jnp.zeros((1, 1, 0, 1, 1), dtype=jnp.float64),
        n_xi_for_x=jnp.asarray([1], dtype=jnp.int32),
    )
    f = jnp.ones((1, 1, 2, 2, 1), dtype=jnp.float64)
    phi1_hat = jnp.asarray([[0.0], [np.log(4.0)]], dtype=jnp.float64)

    out = np.asarray(apply_fokker_planck_v3_phi1(op, f, phi1_hat=phi1_hat))

    expected = np.zeros((1, 1, 2, 2, 1), dtype=np.float64)
    n_pol = np.asarray([3.0, 1.5], dtype=np.float64)
    expected[0, 0, 0, :, 0] = -2.0 * n_pol
    np.testing.assert_allclose(out, expected, rtol=0.0, atol=2.0e-15)


def test_phi1_fokker_planck_apply_rejects_bad_shapes() -> None:
    op = FokkerPlanckV3Phi1Operator(
        nu_n=jnp.asarray(1.0, dtype=jnp.float64),
        krook=jnp.asarray(0.0, dtype=jnp.float64),
        alpha=jnp.asarray(1.0, dtype=jnp.float64),
        z_s=jnp.asarray([1.0], dtype=jnp.float64),
        n_hats=jnp.asarray([1.0], dtype=jnp.float64),
        t_hats=jnp.asarray([1.0], dtype=jnp.float64),
        nl=0,
        k_nu=jnp.zeros((1, 1, 1), dtype=jnp.float64),
        k_cd=jnp.zeros((1, 1, 1, 1), dtype=jnp.float64),
        k_ce=jnp.zeros((1, 1, 1, 1), dtype=jnp.float64),
        k_rosen=jnp.zeros((1, 1, 0, 1, 1), dtype=jnp.float64),
        n_xi_for_x=jnp.asarray([1], dtype=jnp.int32),
    )
    f = jnp.ones((1, 1, 1, 1, 1), dtype=jnp.float64)
    with pytest.raises(ValueError, match="phi1_hat must have shape"):
        apply_fokker_planck_v3_phi1(op, f, phi1_hat=jnp.ones((2, 1), dtype=jnp.float64))

    bad_op = FokkerPlanckV3Phi1Operator(
        nu_n=op.nu_n,
        krook=op.krook,
        alpha=op.alpha,
        z_s=op.z_s,
        n_hats=op.n_hats,
        t_hats=op.t_hats,
        nl=0,
        k_nu=jnp.zeros((1, 2, 1), dtype=jnp.float64),
        k_cd=op.k_cd,
        k_ce=op.k_ce,
        k_rosen=op.k_rosen,
        n_xi_for_x=op.n_xi_for_x,
    )
    with pytest.raises(ValueError, match="op.k_nu has shape"):
        apply_fokker_planck_v3_phi1(bad_op, f, phi1_hat=jnp.zeros((1, 1), dtype=jnp.float64))


def test_pas_apply_rejects_bad_shapes() -> None:
    op = _pas_operator()
    with pytest.raises(ValueError, match="f must have shape"):
        apply_pitch_angle_scattering_v3(op, jnp.ones((1, 3, 5, 2), dtype=jnp.float64))


# ----------------------------------------------------------------------------
# Discrete conservation: the linearized Fokker-Planck blocks annihilate the
# Maxwellian null vectors (particle number, parallel momentum incl. interspecies
# exchange, energy) at the collocation level, to machine precision.
# ----------------------------------------------------------------------------

_FP_XGRID_K = 0.0


def _fp_blocks_v3(z, m, n, t, *, n_x: int = 8, nl: int = 4, n_xi: int = 5, prepared=False) -> np.ndarray:
    sg = make_speed_grid(n_x=n_x, k=_FP_XGRID_K)
    x = np.asarray(sg.x, dtype=np.float64)
    x_weights = np.asarray(sg.dx_weights(_FP_XGRID_K), dtype=np.float64)
    ddx, d2dx2 = speed_grid_diff_matrices(x, k=_FP_XGRID_K)
    kwargs = dict(
        x=x,
        x_weights=x_weights,
        ddx=ddx,
        d2dx2=d2dx2,
        x_grid_k=_FP_XGRID_K,
        z_s=np.asarray(z, dtype=np.float64),
        m_hats=np.asarray(m, dtype=np.float64),
        n_hats=np.asarray(n, dtype=np.float64),
        t_hats=np.asarray(t, dtype=np.float64),
        nu_n=1.0,
        krook=0.0,
        n_xi=n_xi,
        nl=nl,
        n_xi_for_x=np.full((n_x,), n_xi, dtype=np.int32),
    )
    if prepared:
        builder = prepare_fokker_planck_v3_profiles(
            **{k: v for k, v in kwargs.items() if k not in ("n_hats", "t_hats", "n_xi")})
        op = builder(kwargs["n_hats"], kwargs["t_hats"]).at_uniform_density(kwargs["n_hats"], n_xi=n_xi)
    else:
        op = make_fokker_planck_v3_operator(**kwargs)
    return np.asarray(op.mat)


@pytest.mark.parametrize("prepared", [False, True])
def test_fokker_planck_annihilates_maxwellian_null_vectors_single_species(prepared) -> None:
    """C[F_M] = 0 at L=0 (density AND energy) and C[x F_M] = 0 at L=1 (momentum).

    The linearized self-collision operator annihilates the perturbations that
    correspond to shifting the background Maxwellian's density, temperature,
    and mean velocity; the discretization preserves this to machine precision.
    """
    mat = _fp_blocks_v3([1.0], [1.0], [1.0], [1.0], prepared=prepared)
    x = np.asarray(make_speed_grid(n_x=8, k=_FP_XGRID_K).x)
    f_m = np.exp(-(x * x))
    scale = float(np.max(np.abs(mat[0, 0, :2])))

    assert np.max(np.abs(mat[0, 0, 0] @ f_m)) <= 1e-15 * scale  # particle number
    assert np.max(np.abs(mat[0, 0, 0] @ ((x * x - 1.5) * f_m))) <= 1e-15 * scale  # energy
    assert np.max(np.abs(mat[0, 0, 1] @ (x * f_m))) <= 1e-15 * scale  # momentum
    # L=2 has no conservation law: the same Maxwellian-weighted vector is NOT null.
    assert np.max(np.abs(mat[0, 0, 2] @ (x * x * f_m))) > 1e-6 * scale


@pytest.mark.parametrize("prepared", [False, True])
def test_fokker_planck_interspecies_conservation_equal_temperature(prepared) -> None:
    """Cross-species null vectors: per-species density at L=0 and a common flow at L=1.

    For equal temperatures, C_ab[F_Ma, F_Mb] = 0, so per-species density
    perturbations (nHat_a e^{-x^2}) are annihilated at L=0.  A common mean
    velocity u gives f1_a = u (m_a v / T) F_Ma, i.e. collocation values
    proportional to nHat_a mHat_a^2 x e^{-x^2}; interspecies momentum exchange
    must cancel it at L=1.  Both hold to machine precision discretely.
    """
    n_hat = np.asarray([0.6, 0.009], dtype=np.float64)
    m_hat = np.asarray([1.0, 6.0], dtype=np.float64)
    mat = _fp_blocks_v3([1.0, 6.0], m_hat, n_hat, [1.0, 1.0], prepared=prepared)
    x = np.asarray(make_speed_grid(n_x=8, k=_FP_XGRID_K).x)
    f_m = np.exp(-(x * x))
    scale = float(np.max(np.abs(mat[:, :, :2])))

    density = n_hat[:, None] * f_m[None, :]
    r0 = np.einsum("abij,bj->ai", mat[:, :, 0, :, :], density)
    assert np.max(np.abs(r0)) <= 1e-15 * scale

    flow = (n_hat * m_hat**2)[:, None] * (x * f_m)[None, :]
    r1 = np.einsum("abij,bj->ai", mat[:, :, 1, :, :], flow)
    assert np.max(np.abs(r1)) <= 1e-15 * scale
