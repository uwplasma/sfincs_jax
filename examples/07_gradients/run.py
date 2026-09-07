"""Temperature sensitivity of PAS bootstrap current with refreshed collisions.

Differentiate the kinetic solve, its drive and the current moment together.
The pitch-angle collision coefficients are rebuilt in JAX for each temperature;
this is a fixed-density, fixed-gradient, fixed-geometry response at fixed nu_n.
The finite-difference window checks the same physical parameter dependence.

This expert-operator example does not differentiate the host Case runner or
claim full-FP temperature derivatives. Its grid is a teaching configuration;
research use requires resolution and observable-error checks.

Physics: single-species PAS in a circular tokamak; normalized bootstrap current.
Expected runtime: tens of seconds on a laptop CPU, depending on compilation.
"""

# 1. Imports
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from netCDF4 import Dataset  # noqa: E402

import dkx  # noqa: E402
from dkx.collisions import make_pitch_angle_scattering_v3_operator  # noqa: E402
from dkx.run import profile_moments_from_operator  # noqa: E402
from dkx.solve import solve  # noqa: E402

# 2. User-editable parameters
HERE = Path(__file__).resolve().parent
OUT_DIR = HERE.parent / "output" / "07_gradients"
RESULT_FILE = OUT_DIR / "gradients.nc"
PLOT_FILE = OUT_DIR / "gradients.png"

# The step for the finite-difference cross-check.  Finite differences are not
# ground truth here: they carry their own truncation and round-off error, so
# this establishes that the two agree, not that autodiff has been "validated".
FD_STEP = 1.0e-5
# Temperatures at which to evaluate <j.B> for the figure, as multiples of the
# base THat.  Each is one extra solve.
SWEEP_FRACTIONS = (0.80, 0.90, 1.00, 1.10, 1.20)

# 3. Geometry and species construction
CASE = dict(
    geometryScheme=1, inputRadialCoordinate=3, rN_wish=0.3,
    B0OverBBar=1.0, epsilon_t=-0.07, epsilon_h=0.0,
    iota=0.4542, GHat=3.7481, IHat=0.0, psiAHat=0.15596, aHat=0.5585,
    Zs=[1.0], mHats=[1.0], nHats=[1.0], THats=[1.0],
    dNHatdrHats=[-0.5], dTHatdrHats=[-1.0],
)

# 4. Physics and numerical configuration
NUMERICS = dict(
    Ntheta=11, Nzeta=1, Nxi=12, NL=4, Nx=5,
    collisionOperator=1, Delta=4.5694e-3, alpha=1.0, nu_n=8.330e-3,
)
# end of parameters

OUT_DIR.mkdir(parents=True, exist_ok=True)
operator = dkx.run(**CASE, **NUMERICS, emit=None).operator
print(f"operator: {operator.n_species} species, matrix size {operator.total_size}")
if operator.pas is None or operator.n_species != 1:
    raise ValueError("This tutorial requires one kinetic species and collisionOperator=1.")


@jax.jit
def bootstrap_current(t_hat):
    """<j.B> as a function of the species temperature.  Differentiable."""
    temperature = jnp.reshape(t_hat, (1,))
    pas = make_pitch_angle_scattering_v3_operator(
        x=operator.x, z_s=operator.z_s, m_hats=operator.m_hat,
        n_hats=operator.n_hat, t_hats=temperature,
        nu_n=operator.pas.nu_n, krook=operator.pas.krook,
        n_xi_for_x=operator.n_xi_for_x, n_xi=operator.n_xi,
    )
    perturbed = replace(operator, t_hat=temperature, pas=pas)
    # method="auto" picks the route this operator needs.  Naming a route by
    # hand is how you meet "the full-band structured direct factorization
    # requires uniform Nxi_for_x": this deck ramps Nxi with speed, so it
    # belongs on the truncated kernel, and "auto" knows that.
    solved = solve(perturbed, perturbed.rhs(), method="auto", differentiable=True)
    return profile_moments_from_operator(perturbed, solved.x)["FSABjHat"]


# 5. Run
t_hat_0 = float(operator.t_hat[0])
value = float(bootstrap_current(jnp.asarray(t_hat_0)))
gradient = float(jax.grad(bootstrap_current)(jnp.asarray(t_hat_0)))

steps = FD_STEP * max(1.0, abs(t_hat_0)) * np.array([1.0, 3.0, 10.0])
finite_differences = np.array([
    float((bootstrap_current(jnp.asarray(t_hat_0 + step))
           - bootstrap_current(jnp.asarray(t_hat_0 - step))) / (2.0 * step))
    for step in steps
])
finite_difference = finite_differences[0]

sweep_t_hat = np.array([fraction * t_hat_0 for fraction in SWEEP_FRACTIONS], dtype=float)
sweep_current = np.array(
    [float(bootstrap_current(jnp.asarray(t))) for t in sweep_t_hat], dtype=float
)

# 6. Print a scientific summary and certificate
relative_difference = float(np.max(np.abs(gradient - finite_differences) /
                                     np.maximum(np.maximum(abs(gradient), np.abs(finite_differences)), 1e-30)))
print("\n=== Final results ===")
print(f"  <j.B> at THat = {t_hat_0:.4f}      = {value:+.8e} (normalized)")
print(f"  jax.grad           d<j.B>/dTHat = {gradient:+.10e}")
print(f"  central difference d<j.B>/dTHat = {finite_difference:+.10e}")
print(f"  relative difference             = {relative_difference:.3e}")
assert relative_difference < 1.0e-5, "autodiff and central differences disagree"
print("  all gradients verified against central finite differences")
# Report the sign for this case; it is not a universal monotonicity law.
print(f"  sign check: d<j.B>/dTHat is {'positive' if gradient > 0 else 'negative'}, "
      f"and <j.B> is {'positive' if value > 0 else 'negative'}")
print("  cost: reverse mode is one transposed solve regardless of how many "
      "parameters are differentiated, plus coefficient/moment derivatives and checks")

# 7. Save native result
with Dataset(RESULT_FILE, "w", format="NETCDF4") as dataset:
    dataset.createDimension("sweep", sweep_t_hat.size)
    dataset.createDimension("fd_step", steps.size)
    dataset.createVariable("fd_step", "f8", ("fd_step",))[:] = steps
    dataset.createVariable("fd_gradient", "f8", ("fd_step",))[:] = finite_differences
    dataset.createVariable("THat", "f8", ("sweep",))[:] = sweep_t_hat
    dataset.createVariable("FSABjHat", "f8", ("sweep",))[:] = sweep_current
    dataset.createVariable("dFSABjHat_dTHat", "f8")[...] = gradient
    dataset.createVariable("dFSABjHat_dTHat_central_difference", "f8")[...] = finite_difference
    dataset.dkx_version = dkx.__version__
    dataset.autodiff = "jax.grad through refreshed PAS collisions, solve and current moment"
    dataset.derivative_scope = "temperature; fixed density, gradients, geometry and nu_n"
print(f"  Wrote result: {RESULT_FILE}")

# 8. Plot publication-ready outputs
figure, (left, right) = plt.subplots(1, 2, figsize=(11.0, 4.2), constrained_layout=True)
left.plot(sweep_t_hat, sweep_current, "o-", color="tab:blue", label=r"$\langle j\cdot B\rangle$")
tangent_t = np.linspace(sweep_t_hat.min(), sweep_t_hat.max(), 2)
left.plot(tangent_t, value + gradient * (tangent_t - t_hat_0), "--", color="tab:red",
          label="tangent from jax.grad")  # fmt: skip
left.plot([t_hat_0], [value], "x", color="tab:red", ms=11, mew=2.5)
left.set(xlabel=r"$\hat T$", ylabel=r"$\langle j\cdot B\rangle$ (normalized)")
left.set_title("current and its temperature derivative", fontsize=10)
left.grid(alpha=0.3)
left.legend(fontsize=9)

right.bar(["jax.grad", "central difference"], [gradient, finite_difference],
          color=["tab:red", "tab:grey"])  # fmt: skip
right.set_ylabel(r"$d\langle j\cdot B\rangle/d\hat T$")
right.set_title(f"agreement: {relative_difference:.1e} relative", fontsize=10)
right.grid(alpha=0.3, axis="y")
figure.suptitle("Temperature response with refreshed pitch-angle collisions")
figure.savefig(PLOT_FILE, dpi=150)
plt.close(figure)
print(f"  Saved plot: {PLOT_FILE}")
print("Done: examples/07_gradients/run.py")
