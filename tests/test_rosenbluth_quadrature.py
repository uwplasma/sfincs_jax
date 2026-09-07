import warnings
from pathlib import Path

import numpy as np
import pytest
import jax
import jax.numpy as jnp
from scipy.integrate import IntegrationWarning

from dkx.collisions import (
    ROSENBLUTH_METHODS,
    _monomial_int_upper,
    resolve_rosenbluth_method,
    rosenbluth_potential_terms_v3_np,
    prepare_fokker_planck_v3_profiles,
)
from dkx.drift_kinetic import KineticOperator
from dkx.namelist import parse_sfincs_input_text
from dkx.xgrid import make_x_grid
from dkx.phase_space import speed_grid_diff_matrices

REF = Path(__file__).parent / "ref"
_FP_DECK = "fp_1species_FPCollisions_noEr_tiny_cs3"


def _east_three_species_case(nl: int) -> tuple[dict[str, object], object]:
    xg = make_x_grid(n=12, k=0.0, include_point_at_x0=False)
    kwargs = {
        "x": xg.x,
        "x_weights": xg.dx_weights(),
        "x_grid_k": 0.0,
        "xg": xg,
        "z_s": np.array([-1.0, 1.0, 6.0]),
        "m_hats": np.array([5.4461702149014566e-4, 2.0, 12.0]),
        "n_hats": np.array([0.17326127575229972, 0.13860902060183977, 0.005775375858409991]),
        "t_hats": np.full(3, 1.7221796790605068),
        "nl": nl,
    }
    return kwargs, xg


def test_negative_power_upper_moments_cover_small_and_large_species_speed() -> None:
    # 80-digit mpmath references for Gamma((n+1)/2, xb**2)/2.  The first
    # case exercises the sharply peaked small-x electron/ion integral and the
    # second the exponentially small large-x continuation.
    cases = (
        (4.929789984040181e-4, -14, 7.573333596728916e41),
        (22.88985968775449, -14, 5.642165067724352e-249),
    )
    for xb, power, expected in cases:
        got = _monomial_int_upper(xb, power)
        assert np.isfinite(got)
        assert got > 0.0
        assert np.isclose(got, expected, rtol=2e-13, atol=0.0)


def test_hybrid_rosenbluth_is_warning_free_and_keeps_low_l_quadpack_parity() -> None:
    kwargs, _ = _east_three_species_case(nl=5)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        hybrid = rosenbluth_potential_terms_v3_np(**kwargs, method="hybrid")

    assert np.isfinite(hybrid).all()
    assert not any(isinstance(item.message, IntegrationWarning) for item in caught)

    low_l_kwargs = dict(kwargs)
    low_l_kwargs["nl"] = 4
    quadpack = rosenbluth_potential_terms_v3_np(**low_l_kwargs, method="quadpack")
    assert np.array_equal(hybrid[:, :, :4], quadpack)


def test_prepared_electron_ion_responses_converge_and_differentiate_against_quadpack():
    kwargs, xg = _east_three_species_case(nl=4)
    kwargs["t_hats"] = np.array([1.2, .8, 1.4])
    kwargs["n_hats"] = np.ones(3)  # Stored kernels factor out density.
    ddx, d2dx2 = speed_grid_diff_matrices(xg.x, k=0.)

    def prepared(order):
        build = prepare_fokker_planck_v3_profiles(
            **{k: v for k, v in kwargs.items() if k not in ("xg", "n_hats", "t_hats")},
            ddx=ddx, d2dx2=d2dx2, nu_n=1., krook=0., n_xi_for_x=np.full(12, 4),
            quadrature_order=order)
        return jax.jit(build)

    t = jnp.asarray(kwargs["t_hats"])
    reference = rosenbluth_potential_terms_v3_np(**kwargs, method="quadpack")
    scales = np.max(np.abs(reference), axis=(-1, -2), keepdims=True)
    errors = []
    for order in (32, 64, 128):
        build = prepared(order)
        response = lambda t: build(jnp.ones(3), t).k_rosen
        errors.append(float(np.max(np.abs(np.asarray(response(t))-reference)/scales)))
    assert errors[-1] < 5e-11, errors
    assert errors[-1] < errors[0]/100, errors
    # 80-digit mathematical erf/sqrt(pi) evaluation of the full CE/nu_D
    # formulas at these float64 grid, profile and derivative-matrix inputs.
    # Rounded v3 sqrt(pi) plus subtractive cancellation misses these values
    # by ~1.5e-7 relative; do not use that host result as the accuracy oracle.
    kernels = build(jnp.ones(3), t)
    np.testing.assert_allclose(kernels.k_ce[2, 0, 0, 11], 378407.02215702748336043, rtol=2e-12)
    np.testing.assert_allclose(kernels.k_nu[2, 0, 0], 51.098136568481772455399, rtol=2e-12)
    direction = jnp.asarray([.1, -.2, .05])
    tangent = jax.jvp(response, (t,), (direction,))[1]
    fd_errors = []
    for step in (1e-3, 3e-4, 1e-4):
        plus = rosenbluth_potential_terms_v3_np(**{**kwargs, "t_hats": np.asarray(t+step*direction)}, method="quadpack")
        minus = rosenbluth_potential_terms_v3_np(**{**kwargs, "t_hats": np.asarray(t-step*direction)}, method="quadpack")
        fd_errors.append(float(np.max(np.abs(np.asarray(tangent)-(plus-minus)/(2*step))/scales)))
    assert fd_errors[-1] < 2e-8, fd_errors
    # Independent QUADPACK central differences must approach the tangent
    # quadratically; a single coarse step has measurable truncation error.
    for coarse, fine in zip(fd_errors, fd_errors[1:]):
        assert 5 < coarse/fine < 15, fd_errors


@pytest.mark.parametrize("order", [True, 1, 2.5])
def test_prepared_quadrature_rejects_invalid_static_order(order):
    kwargs, xg = _east_three_species_case(nl=4)
    ddx, d2dx2 = speed_grid_diff_matrices(xg.x, k=0.)
    with pytest.raises(ValueError, match="quadrature_order"):
        prepare_fokker_planck_v3_profiles(
            **{k: v for k, v in kwargs.items() if k not in ("xg", "n_hats", "t_hats")},
            ddx=ddx, d2dx2=d2dx2, nu_n=1., krook=0., n_xi_for_x=np.full(12, 4),
            quadrature_order=order)


def test_high_l_prepared_response_matches_80_digit_integral_reference():
    xg = make_x_grid(n=12, k=0., include_point_at_x0=False)
    ddx, d2dx2 = speed_grid_diff_matrices(xg.x, k=0.)
    responses = []
    for order in (128, 256):
        build = prepare_fokker_planck_v3_profiles(
            x=xg.x, x_weights=xg.dx_weights(), x_grid_k=0., ddx=ddx, d2dx2=d2dx2,
            z_s=np.array([6.]), m_hats=np.array([12.]), nu_n=1., krook=0., nl=16,
            n_xi_for_x=np.full(12, 16), quadrature_order=order)
        responses.append(np.asarray(jax.jit(build)(jnp.ones(1), jnp.array([1.4])).k_rosen))
    # Independent mpmath 80-digit integrals of t^p P_j(t) exp(-t²),
    # using the stored float64 grid/recurrence and v3 pi constant. The upper
    # integrals are split at max(10,2*xb) and extend to infinity; accumulate
    # H and G'' and the modal projection at 80 digits. The existing analytic
    # monomial route misses this entry by 1.28e-4 due to cancellation.
    reference = -1772.5765410647286686749207961405177489297929584892556157086
    for response in responses:
        np.testing.assert_allclose(response[0, 0, 11, 4, 11], reference, rtol=2e-12, atol=0)
    scale = np.max(np.abs(responses[-1]), axis=(-1, -2), keepdims=True)
    assert np.max(np.abs(responses[0]-responses[1])/scale) < 3e-11


# --- selection routes -------------------------------------------------------
#
# The repo rule is that a solver route is reachable from a namelist key or an
# API argument; the environment variable is an override, never the only way in.


def _fp_operator(rosenbluth_line: str = "") -> KineticOperator:
    # NL = Nxi = 6 so the assembled operator actually reaches the hybrid
    # route's analytic L >= 4 blocks (below that hybrid is QUADPACK by
    # construction and the comparison would be vacuous).
    text = (
        (REF / f"{_FP_DECK}.input.namelist")
        .read_text()
        .replace("NL = 3", "NL = 6")
        .replace("Nxi = 4", "Nxi = 6")
    )
    if rosenbluth_line:
        text = text.replace(
            "  Nxi_for_x_option = 0", f"  Nxi_for_x_option = 0\n  {rosenbluth_line}"
        )
    return KineticOperator.from_namelist(parse_sfincs_input_text(text))


def test_rosenbluth_method_resolution_prefers_the_explicit_route() -> None:
    assert resolve_rosenbluth_method(None) == "quadpack"
    for name in ROSENBLUTH_METHODS:
        assert resolve_rosenbluth_method(name.upper()) == name
        assert resolve_rosenbluth_method(f"  {name}  ") == name
    with pytest.raises(ValueError, match="RosenbluthMethod"):
        resolve_rosenbluth_method("quadpak")


def test_env_var_overrides_only_the_unset_route(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DKX_ROSENBLUTH_METHOD", "hybrid")
    assert resolve_rosenbluth_method(None) == "hybrid"
    # An explicit namelist/API selection wins over the environment.
    assert resolve_rosenbluth_method("quadpack") == "quadpack"
    monkeypatch.setenv("DKX_ROSENBLUTH_METHOD", "not-a-method")
    with pytest.raises(ValueError, match="RosenbluthMethod"):
        resolve_rosenbluth_method(None)


def test_builder_api_argument_selects_the_hybrid_rosenbluth_path() -> None:
    from dkx.collisions import make_fokker_planck_v3_operator
    from dkx.phase_space import make_speed_grid, speed_grid_diff_matrices

    sg = make_speed_grid(n_x=4, k=0.0)
    x = np.asarray(sg.x, dtype=np.float64)
    ddx, d2dx2 = speed_grid_diff_matrices(x, k=0.0)
    common = {
        "x": x,
        "x_weights": np.asarray(sg.dx_weights(0.0), dtype=np.float64),
        "ddx": ddx,
        "d2dx2": d2dx2,
        "x_grid_k": 0.0,
        "z_s": np.array([1.0]),
        "m_hats": np.array([1.0]),
        "n_hats": np.array([1.0]),
        "t_hats": np.array([1.0]),
        "nu_n": 0.01,
        "krook": 0.0,
        "n_xi": 6,
        "nl": 6,
        "n_xi_for_x": np.full(4, 6, dtype=np.int32),
    }
    base = make_fokker_planck_v3_operator(**common)
    same = make_fokker_planck_v3_operator(**common, rosenbluth_method="quadpack")
    hybrid = make_fokker_planck_v3_operator(**common, rosenbluth_method="hybrid")

    # The default and an explicit 'quadpack' are the same operator; 'hybrid'
    # reaches a different quadrature -- with nl = 6 the L >= 4 blocks take the
    # analytic moments -- without the environment variable ever being set.
    np.testing.assert_array_equal(np.asarray(same.mat), np.asarray(base.mat))
    assert not np.array_equal(np.asarray(hybrid.mat), np.asarray(base.mat))
    np.testing.assert_allclose(
        np.asarray(hybrid.mat), np.asarray(base.mat), rtol=1e-6, atol=1e-10
    )

    with pytest.raises(ValueError, match="RosenbluthMethod"):
        make_fokker_planck_v3_operator(**common, rosenbluth_method="quadpak")


def test_namelist_key_selects_the_hybrid_rosenbluth_path() -> None:
    base = _fp_operator()
    hybrid = _fp_operator("RosenbluthMethod = 'hybrid'")
    quadpack = _fp_operator("RosenbluthMethod = 'quadpack'")

    np.testing.assert_array_equal(np.asarray(quadpack.fp.mat), np.asarray(base.fp.mat))
    assert not np.array_equal(np.asarray(hybrid.fp.mat), np.asarray(base.fp.mat))
    np.testing.assert_allclose(
        np.asarray(hybrid.fp.mat), np.asarray(base.fp.mat), rtol=1e-6, atol=1e-10
    )


def test_namelist_key_rejects_an_unknown_method() -> None:
    with pytest.raises(ValueError, match="RosenbluthMethod"):
        _fp_operator("RosenbluthMethod = 'quadpak'")
