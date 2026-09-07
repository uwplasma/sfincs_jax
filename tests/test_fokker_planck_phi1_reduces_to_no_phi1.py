from __future__ import annotations

import numpy as np
import jax
import pytest
import jax.numpy as jnp

from dkx.collisions import (
    apply_fokker_planck_v3,
    apply_fokker_planck_v3_phi1,
    make_fokker_planck_v3_operator,
    make_fokker_planck_v3_phi1_operator,
    prepare_fokker_planck_v3_profiles,
)
from dkx.inputs import load_sfincs_input
from dkx.run import _grids_from_input


@pytest.fixture(scope="module")
def fp_case():
    inp = load_sfincs_input("tests/ref/quick_2species_FPCollisions_noEr.input.namelist")
    nml = inp.raw
    grids = _grids_from_input(inp, nml)
    species = nml.group("speciesParameters")
    phys = nml.group("physicsParameters")
    other = nml.group("otherNumericalParameters")

    x_grid_k = float(other.get("XGRID_K", 0.0))
    z_s = np.atleast_1d(np.asarray(species["ZS"], dtype=np.float64))
    m_hats = np.atleast_1d(np.asarray(species["MHATS"], dtype=np.float64))
    n_hats = np.atleast_1d(np.asarray(species["NHATS"], dtype=np.float64))
    t_hats = np.atleast_1d(np.asarray(species["THATS"], dtype=np.float64))
    nu_n = float(phys["NU_N"])
    krook = float(phys.get("KROOK", 0.0))
    alpha = float(phys.get("ALPHA", 1.0))

    kwargs = dict(
        x=np.asarray(grids.x, dtype=np.float64),
        x_weights=np.asarray(grids.x_weights, dtype=np.float64),
        ddx=np.asarray(grids.ddx, dtype=np.float64),
        d2dx2=np.asarray(grids.d2dx2, dtype=np.float64),
        x_grid_k=float(x_grid_k),
        z_s=z_s,
        m_hats=m_hats,
        n_hats=n_hats,
        t_hats=t_hats,
        nu_n=float(nu_n),
        krook=float(krook),
        n_xi=int(grids.n_xi),
        nl=int(grids.n_l),
        n_xi_for_x=np.asarray(grids.n_xi_for_x, dtype=np.int32),
    )
    op0 = make_fokker_planck_v3_operator(**kwargs)
    op_phi1 = make_fokker_planck_v3_phi1_operator(**kwargs, alpha=alpha)
    return kwargs, op0, op_phi1, grids


def test_phi1_in_collisions_reduces_to_no_phi1_when_phi1_zero(fp_case) -> None:
    kwargs, op0, op_phi1, grids = fp_case
    z_s = kwargs["z_s"]

    rng = np.random.default_rng(0)
    f = jnp.asarray(rng.standard_normal((z_s.size, grids.x.size, grids.n_xi, grids.theta.size, grids.zeta.size)))
    phi1_hat = jnp.zeros((grids.theta.size, grids.zeta.size), dtype=jnp.float64)

    y0 = apply_fokker_planck_v3(op0, f)
    y1 = apply_fokker_planck_v3_phi1(op_phi1, f, phi1_hat=phi1_hat)

    np.testing.assert_allclose(np.asarray(y1), np.asarray(y0), rtol=0, atol=2e-11)


@pytest.fixture(scope="module")
def profile_builder(fp_case):
    kwargs, _, kernels, _ = fp_case
    return prepare_fokker_planck_v3_profiles(
        **{k: v for k, v in kwargs.items() if k not in ("n_hats", "t_hats", "n_xi")},
        alpha=float(kernels.alpha),
    )


def test_independent_temperature_refresh_matches_all_kernels_and_phi1(fp_case, profile_builder):
    kwargs, _, kernels, grids = fp_case
    n, t = jnp.asarray(kwargs["n_hats"]), jnp.asarray(kwargs["t_hats"])
    compiled = jax.jit(profile_builder)
    f = jnp.asarray(np.random.default_rng(31).normal(
        size=(2, grids.x.size, grids.n_xi, grids.theta.size, grids.zeta.size)))
    phi = jnp.full((grids.theta.size, grids.zeta.size), .07)
    for factors in ([1., 1.], [.6, 1.4], [1.7, .8]):
        temp = t*jnp.asarray(factors)
        got = compiled(n, temp)
        cold = make_fokker_planck_v3_phi1_operator(
            **{**kwargs, "t_hats": np.asarray(temp)}, alpha=float(kernels.alpha))
        for name in ("k_nu", "k_cd", "k_ce", "k_rosen"):
            np.testing.assert_allclose(getattr(got, name), getattr(cold, name), rtol=3e-10, atol=3e-10)
        np.testing.assert_allclose(apply_fokker_planck_v3_phi1(got, f, phi1_hat=phi),
                                   apply_fokker_planck_v3_phi1(cold, f, phi1_hat=phi), rtol=3e-10, atol=3e-10)
    with pytest.raises(ValueError, match="must have shape"):
        compiled(n[:1], t)
    assert np.all(np.isnan(compiled(n, -t).k_rosen))
    assert np.all(np.isnan(compiled(-n, t).n_hats))


def test_profile_interpolation_derivative_at_coincident_species_nodes(fp_case, profile_builder):
    kwargs, _, _, grids = fp_case
    x = jnp.asarray(grids.x)
    n, m, z = map(jnp.asarray, (kwargs["n_hats"], kwargs["m_hats"], kwargs["z_s"]))
    polynomial = lambda y: 1 - .2*y + .3*y*y
    source = jnp.exp(-x*x)*polynomial(x)
    temperature = .5*m  # Both species speed grids coincide, although masses differ.

    def actual(t):
        return profile_builder(n, t).k_cd[0, 1] @ source

    def expected(t):
        xb = x*jnp.sqrt(t[0]*m[1]/(t[1]*m[0]))
        prefactor = 3*m[0]/m[1]*z[0]**2*z[1]**2/(t[0]*jnp.sqrt(t[0]*m[0]))
        return prefactor*jnp.exp(-x*x)*jnp.exp(-xb*xb)*polynomial(xb)

    direction = jnp.asarray([.07, -.11])
    for got, want in zip(jax.jit(lambda t: jax.jvp(actual, (t,), (direction,)))(temperature),
                         jax.jvp(expected, (temperature,), (direction,))):
        np.testing.assert_allclose(got, want, rtol=2e-12, atol=2e-12)


def test_uniform_density_refresh_matches_rebuild_and_density_derivatives(fp_case):
    kwargs, _, kernels, _ = fp_case
    density = jnp.asarray(kwargs["n_hats"])
    traces = []

    def refresh(n):
        traces.append(1)
        return kernels.at_uniform_density(n, n_xi=kwargs["n_xi"]).mat

    compiled = jax.jit(refresh)
    # Independent species changes at unequal temperatures, including a
    # zero-density algebraic limit; no ratios to a reference density are used.
    for factors in ([1.0, 1.0], [0.7, 1.8], [0.0, 1.0]):
        n = density * jnp.asarray(factors)
        cold = make_fokker_planck_v3_operator(**{**kwargs, "n_hats": np.asarray(n)})
        np.testing.assert_allclose(compiled(n), cold.mat, rtol=2e-12, atol=2e-12)
    assert len(traces) == 1
    for n_xi in (1, kwargs["n_xi"] + 2):
        cold = make_fokker_planck_v3_operator(**{**kwargs, "n_xi": n_xi})
        refreshed = kernels.at_uniform_density(density, n_xi=n_xi)
        np.testing.assert_allclose(refreshed.mat, cold.mat, rtol=2e-12, atol=2e-12)
        np.testing.assert_array_equal(refreshed.mask_xi, cold.mask_xi)
    direction = density * jnp.asarray([0.3, -0.2])
    _, tangent = jax.jvp(compiled, (density,), (direction,))
    eps = 1e-4
    plus = make_fokker_planck_v3_operator(**{**kwargs, "n_hats": np.asarray(density + eps * direction)}).mat
    minus = make_fokker_planck_v3_operator(**{**kwargs, "n_hats": np.asarray(density - eps * direction)}).mat
    np.testing.assert_allclose(tangent, (plus - minus) / (2 * eps), rtol=2e-8, atol=2e-8)
    weights = jnp.reshape(jnp.linspace(-1, 1, tangent.size), tangent.shape)
    _, pullback = jax.vjp(compiled, density)
    np.testing.assert_allclose(jnp.vdot(weights, tangent), jnp.vdot(pullback(weights)[0], direction), rtol=1e-12)
    # Exact linearity in density is a stronger check than one FD step alone.
    np.testing.assert_allclose(tangent, compiled(direction), rtol=2e-12, atol=2e-12)
    with pytest.raises(ValueError, match="n_hats must have shape"):
        kernels.at_uniform_density(density[:1], n_xi=kwargs["n_xi"])
    with pytest.raises(ValueError, match="n_xi must be positive"):
        kernels.at_uniform_density(density, n_xi=0)


def test_common_temperature_refresh_matches_rebuild_and_phi1_derivative(fp_case):
    kwargs, _, kernels, grids = fp_case
    f = jnp.asarray(np.random.default_rng(17).normal(
        size=(2, grids.x.size, grids.n_xi, grids.theta.size, grids.zeta.size)))
    phi = jnp.full((grids.theta.size, grids.zeta.size), 0.07)

    def action(scale):
        return apply_fokker_planck_v3_phi1(
            kernels.rescale_temperature(scale), f, phi1_hat=phi)

    compiled = jax.jit(action)

    def cold(scale):
        rebuilt = make_fokker_planck_v3_phi1_operator(
            **{**kwargs, "t_hats": kwargs["t_hats"] * scale}, alpha=float(kernels.alpha))
        return apply_fokker_planck_v3_phi1(rebuilt, f, phi1_hat=phi)

    for scale in (0.7, 1.0, 1.8):
        np.testing.assert_allclose(compiled(scale), cold(scale), rtol=3e-11, atol=3e-11)
    _, tangent = jax.jvp(compiled, (1.2,), (1.0,))
    h = 1e-4
    np.testing.assert_allclose(tangent, (cold(1.2 + h) - cold(1.2 - h)) / (2 * h),
                               rtol=2e-6, atol=2e-7)
    for invalid in (0.0, -1.0, np.inf, np.nan):
        assert np.all(np.isnan(compiled(invalid)))
    with pytest.raises(ValueError, match="scale must be scalar"):
        kernels.rescale_temperature(jnp.ones(2))


@pytest.mark.parametrize("parameter", ["density", "temperature", "independent_temperature"])
@pytest.mark.parametrize("observable", ["FSABjHat", "heatFlux_vm_psiHat"])
def test_profile_gradient_through_full_fp_solve_matches_fresh_builds(fp_case, profile_builder, observable, parameter):
    from dataclasses import replace
    from dkx.drift_kinetic import KineticOperator
    from dkx.namelist import read_sfincs_input
    from dkx.solve import solve
    from dkx.run import profile_moments_from_operator

    kwargs, _, kernels, _ = fp_case
    base = KineticOperator.from_namelist(read_sfincs_input(
        "tests/ref/quick_2species_FPCollisions_noEr.input.namelist"
    ))
    point = (base.n_hat if parameter == "density" else base.t_hat
             if parameter == "independent_temperature" else jnp.asarray(1.0))

    def updated(n, *, fresh=False):
        if parameter == "density":
            values = {"n_hat": n}
            builder = {**kwargs, "n_hats": np.asarray(n)} if fresh else None
            fp = kernels.at_uniform_density(n, n_xi=base.n_xi) if not fresh else None
        elif parameter == "temperature":
            values = {"t_hat": base.t_hat * n}
            builder = {**kwargs, "t_hats": np.asarray(values["t_hat"])} if fresh else None
            fp = (kernels.rescale_temperature(n).at_uniform_density(base.n_hat, n_xi=base.n_xi)
                  if not fresh else None)
        else:
            values = {"t_hat": n}
            builder = {**kwargs, "t_hats": np.asarray(n)} if fresh else None
            fp = (profile_builder(base.n_hat, n).at_uniform_density(base.n_hat, n_xi=base.n_xi)
                  if not fresh else None)
        return replace(base, **values, fp=make_fokker_planck_v3_operator(**builder) if fresh else fp)

    def loss(n):
        op = updated(n)
        state = solve(op, op.rhs(), method="gmres", tol=1e-11, differentiable=True).x
        return jnp.ravel(profile_moments_from_operator(op, state)[observable])[0]

    value, grad = jax.jit(jax.value_and_grad(loss))(point)
    assert np.isfinite(value) and np.all(np.isfinite(grad))
    direction = point * jnp.asarray(0.2 if parameter == "temperature" else [0.1, -0.2])
    ad = float(jnp.vdot(grad, direction))
    assert abs(ad) > 1e-20

    def cold(n):
        op = updated(n, fresh=True)
        rhs = op.rhs()
        state = solve(op, rhs, method="gmres", tol=1e-11).x
        assert float(jnp.linalg.norm(op.apply(state) - rhs)) <= 1e-11 * float(jnp.linalg.norm(rhs))
        return float(jnp.ravel(profile_moments_from_operator(op, state)[observable])[0])

    for eps in (1e-3, 3e-4):
        fd = (cold(point + eps * direction) - cold(point - eps * direction)) / (2 * eps)
        np.testing.assert_allclose(ad, fd, rtol=2e-5, atol=abs(ad) * 1e-8)

    # A first-order derivative must leave a second-order Taylor remainder.
    baseline = cold(point)
    remainders = [abs(cold(point + h * direction) - baseline - h * ad)
                  for h in (0.04, 0.02, 0.01)]
    for coarse, fine in zip(remainders, remainders[1:]):
        assert 3.0 < coarse / fine < 5.0, (observable, remainders)
