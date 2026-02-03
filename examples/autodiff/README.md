# Automatic differentiation (AD) & implicit differentiation

These examples showcase what you get from a JAX port:
- gradients of geometry integrals / diagnostics
- Jacobian–vector products (JVPs) of the full residual
- differentiating through linear solves (where supported)

Examples:
- `10_matrix_free_residual_and_jvp.py` — matrix-free residual + JVP.
- `14_autodiff_sensitivity_nu_n_scheme5.py` — sensitivity of a residual norm w.r.t. collisionality.
- `06_implicit_diff_through_gmres_solve_scheme5.py` — implicit differentiation through a GMRES solve.

