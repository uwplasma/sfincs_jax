Differentiability
=================

`dkx` is differentiable end to end. Because the drift-kinetic operator,
its right-hand side, and the moment diagnostics are all pure JAX functions, a
scalar built from a solved distribution — a flux, a bootstrap current, an
ambipolar :math:`E_r`, a transport coefficient — can be handed straight to
``jax.grad`` and returns an exact derivative with respect to geometry harmonics,
plasma profiles, or the collisionality. There is no divided-difference
stencil in the loop and no differentiation through solver iterations.

Covered here: the gradient through the linear solve, the catalogue of
differentiable targets, the measured gradient-vs-finite-difference agreement, and
the differentiable geometry chain ``vmex -> booz_xform_jax -> dkx`` used for
stellarator optimization.

.. figure:: _static/figures/paper/dkx_autodiff_gradient_check.png
   :alt: Autodiff gradients of dkx observables overlaid on centered finite differences.
   :align: center
   :width: 88%

   Reverse-mode ``jax.grad`` derivatives of kinetic observables (points) plotted
   against centered finite differences (line). Reproduce with
   ``examples/autodiff/gradients_tour.py``.

Implicit differentiation through the solve
------------------------------------------

Every solver route returns the solution of a linear system
:math:`A(p)\,u = b(p)`, where :math:`p` collects the differentiable parameters
(geometry, profiles, drives). For a scalar objective :math:`J = g(u, p)` the
chain rule needs :math:`\mathrm{d}u/\mathrm{d}p`, which satisfies the *tangent*
system

.. math::

   A\,\frac{\mathrm{d}u}{\mathrm{d}p}
   = \frac{\partial b}{\partial p} - \frac{\partial A}{\partial p}\,u .

Differentiating through the Krylov or block-elimination iterations to get this
would be expensive and numerically noisy. Instead the solve is wrapped with the
**implicit function theorem**: the reverse-mode adjoint of :math:`A u = b` is the
single *transposed* solve

.. math::

   A^{\mathsf T}\,\lambda = \left(\frac{\partial g}{\partial u}\right)^{\!\mathsf T},
   \qquad
   \frac{\mathrm{d}J}{\mathrm{d}p}
   = \frac{\partial g}{\partial p}
   + \lambda^{\mathsf T}\!\left(\frac{\partial b}{\partial p}
   - \frac{\partial A}{\partial p}\,u\right).

The transposed solve :math:`A^{\mathsf T}\lambda = \cdot` **reuses the
forward factorization**. On the structured direct route the adjoint is the same
block-Thomas sweep run with ``transpose=True`` on the factors already computed
for the forward solve; on the recycled Krylov route it is the
transposed-preconditioner solve seeded from the same coarse operator. A gradient
therefore costs *one extra solve*, independent of how many iterations the forward
solve took.

The wrappers come from the standalone ``solvax`` package: linear solves route
through ``solvax.implicit.linear_solve`` (``jax.lax.custom_linear_solve``), and
the outer root problems — the ambipolar :math:`E_r` and the nonlinear
:math:`\Phi_1` Newton solve — route through ``solvax.implicit.root_solve``
(``jax.lax.custom_root``), so their derivatives also fall out of the implicit
function theorem rather than unrolled iterations.

.. admonition:: Where in the code

   ``solve(op, rhs, differentiable=True)`` (:func:`dkx.solve.solve`) wraps
   the structured direct and recycled Krylov routes with the implicit adjoint.
   The scalar ``ambipolar_er`` (:func:`dkx.er.ambipolar_er`) and the
   ``phi1_state`` (:func:`dkx.phi1.phi1_state`) helpers return
   differentiable JAX arrays directly. The sparse direct host solve is *not*
   differentiable and raises if ``differentiable=True`` is requested.

Ambipolar solves use the routed kinetic solver
----------------------------------------------

``radial_current(..., differentiable=True)`` and ``ambipolar_er`` use the
same structured/recycled-Krylov routes as :func:`dkx.solve.solve`, with SOLVAX
owning the implicit linear and root differentiation. These paths avoid
assembly and dense factorization of the global kinetic matrix.
The selected route's layout and differentiation restrictions still apply;
there is no longer a blanket rejection of ``Nxi_for_x_option=1``.
Prepared ``ErProblem`` method/tolerance settings are preserved unless explicitly
overridden. An explicit sparse ``direct`` request raises because that route
is non-differentiable. Root and initial-field units follow the prepared problem.

CPU and installed-wheel GPU tests compare routed current derivatives to cold finite differences and
root derivatives to finite differences of independently solved roots, for PAS
and full-FP collisions with uniform and ramped pitch layouts. They also reject
construction of the global dense identity. These are bounded discretization
checks, not joint-grid or marginal-root certificates. Seed the unbracketed
secant near the desired isolated root. Its final acceptance requires finite
field/current/slope, ``abs(Jr) <= current_tol`` (default 1e-12),
``abs(dJr/dEr) > min_abs_slope`` (default zero), and the local Newton correction
``abs(Jr/(dJr/dEr)) <= root_tol`` (default 1e-11). The field tolerance uses the
prepared problem's units; current is normalized and the slope is normalized
current per field unit. These controls are static under JIT.

Failure raises an exception, including under JIT, AD and vmap; the runtime
callback needs an available CPU backend on GPU hosts (``JAX_PLATFORMS=cuda,cpu``).
Acceptance adds a final current/field-tangent evaluation. A zero default slope
threshold rejects exactly flat roots but does not certify a nearly marginal
root: choose a positive threshold from the application's current uncertainty
and acceptable field uncertainty. The local correction is not a rigorous
root-error bound, branch-continuity guarantee, or phase-space convergence test.
An intrinsically ambipolar model cannot determine a unique field this way.

A paired installed-wheel A4000 probe on 2,358 PAS unknowns compares the former
dense expression with the routed current at identical parameters. Twelve
alternating, synchronized samples give median forward times 53.2 -> 10.7 ms
and value/gradient times 54.7 -> 17.8 ms, with matching values/derivatives.
HLO replaces the global 2358-by-2358 LU with batches of 49-by-49 factors.
XLA temporary-buffer estimates decrease from about 266 MB to 4.3 MB; these
are not allocator peak measurements. The trace confirms GPU execution of
both expressions; the routed expression launches more, smaller kernels.
Inputs, wheel checksum, HLO, trace and raw timings remain outside Git in
``dkx-review-evidence-20260905/routed-ambipolar-ad``. This measures an inner
current evaluation, not a full optimizer iteration or production scaling.
The paired local CPU probe gives forward 51.6 -> 2.63 ms and value/gradient
58.6 -> 3.01 ms. The GPU regression selection takes 419 seconds overall;
full-root setup/compilation and repeated execution costs remain to be separated.
A separate 2,358-unknown root value/gradient probe isolates the acceptance
check: CPU median 11.36 -> 12.12 ms and A4000 52.32 -> 53.32 ms, with matching
roots/derivatives. Twelve alternating samples synchronize both device work and
host callback effects; the persistent compilation cache is retained. These
fixed-seed PAS timings do not characterize branch searches or optimizer runs.

Bounded reverse mode for the truncated structured direct kernel
---------------------------------------------------------------

One path is deliberately outside the implicit-adjoint wrapper. The memory-lean
truncated structured direct kernel
(``solve(op, rhs, method="block_tridiagonal_truncated")``) inverts the *reduced*
Schur-complemented operator on the lowest ``tier1_keep_lowest`` Legendre blocks
rather than the full band, so a full-operator :math:`A^{\mathsf T}` adjoint would
be inconsistent and would silently corrupt the gradient. Its blocks are instead
assembled on the fly, which keeps the **forward** working set at
:math:`O(\text{keep}\cdot m^2)` per ``(species, x)`` subsystem, independent of
:math:`N_\xi` — the property that lets large ramped PAS/DKES decks route through
the structured direct kernel at all.

Plain ``jax.grad`` through that kernel tapes the generated sweeps, so the
**reverse** pass costs :math:`O(N_\xi\cdot m^2)` per subsystem and surrenders
the block-count independence exactly where gradient-based transport and profile
inversion need it. ``solve(..., tier1_adjoint_window=w)`` restores it by routing
through ``solvax``'s structure-preserving custom VJP for generated blocks: the
right-hand-side gradient is an exactly *generated* truncated solve of the
transposed operator, and the coefficient
gradients are pulled back through the block assembly's own derivative on the
leading ``keep + w`` blocks. Reverse mode then runs at
:math:`O((\text{keep}+w)\cdot m^2)` per subsystem, matching the forward sweep.

``tier1_adjoint_window=None`` is the default and keeps the taped gradient, so
behavior is unchanged unless the option is passed. The right-hand-side gradient
carries no window error at any :math:`w`; the coefficient-gradient error decays
as :math:`O(\rho^{2w})` for the block-dominant collisional operators this kernel
targets, and :math:`w \ge N_\xi` reproduces the taped gradient exactly. The knob
composes with ``subsystem_batch`` — the custom VJP is ``vmap``-safe, and the
gradient is identical at any batch width.

Compiled peak temporary working set of a ``jax.grad`` through the truncated
solve on the tiny scheme-1 PAS fixture, sweeping :math:`N_\xi` at a fixed
window ``w = 4`` (XLA ``memory_analysis().temp_size_in_bytes``):

.. list-table::
   :header-rows: 1
   :widths: 20 28 28 24

   * - :math:`N_\xi`
     - taped (MiB)
     - ``tier1_adjoint_window=4`` (MiB)
     - reduction
   * - 8
     - 0.161
     - 0.068
     - 2.4x
   * - 32
     - 0.528
     - 0.068
     - 7.7x
   * - 128
     - 1.994
     - 0.068
     - 29.2x
   * - 256
     - 3.949
     - 0.068
     - 57.9x

The taped column grows linearly in :math:`N_\xi`; the windowed column is flat,
which is the whole point of the option.

What is differentiable
----------------------

.. list-table::
   :header-rows: 1
   :widths: 26 40 34

   * - Target
     - What flows
     - Entry point
   * - **Geometry**
     - Boozer harmonics :math:`\hat B_{mn}` and derived metric coefficients, for
       analytic schemes and for JAX-native geometry producers
     - :meth:`dkx.drift_kinetic.KineticOperator.apply` /
       ``booz_xform_jax``
   * - **Profiles**
     - densities, temperatures, their radial gradients, ``nu_n``, and the
       :math:`E_r`/potential-gradient drive
     - ``KineticOperator.rhs`` and the operator coefficients
   * - **Ambipolar** :math:`E_r`
     - the scalar root of :math:`J_r(E_r)=0` and any downstream function of it
     - :func:`dkx.er.ambipolar_er`
   * - :math:`\Phi_1` **state**
     - the solved flux-surface potential :math:`\Phi_1(\theta,\zeta)` from the
       nonlinear quasineutrality Newton solve
     - :func:`dkx.phi1.phi1_state`
   * - **Monoenergetic transport matrix**
     - the RHSMode=3 coefficients and the energy-convolved thermal
       :math:`L_{ij}`, differentiated w.r.t. geometry
     - :func:`dkx.monoenergetic.monoenergetic_database_from_operator`
       (``differentiable=True``)

The file-based readers (``input.namelist``, ``.bc`` Boozer files, ``wout_*.nc``)
are provenance and parity tools and are **not** differentiable. Geometry
sensitivities flow through JAX-native producers instead — the analytic geometry
schemes, or the ``vmex -> booz_xform_jax`` transform below.

.. note::

   The differentiable ambipolar :math:`E_r` and :math:`\Phi_1` helpers require the
   untruncated pitch embedding (``Nxi_for_x_option = 0``); with an active
   :math:`N_\xi`-for-:math:`x` ramp they raise ``NotImplementedError`` rather than
   return an approximate gradient. RHSMode=1 outputs remain differentiable through
   the ramped structured direct route.

Measured gradient accuracy
--------------------------

Every differentiable path is checked against centered finite differences. The
recorded agreements are:

.. list-table::
   :header-rows: 1
   :widths: 40 34 26

   * - Differentiable target
     - Reverse-mode route
     - ``grad`` vs finite difference
   * - PAS + :math:`E_r` kinetic outputs
     - recycled Krylov transposed solve
     - ``2.9e-6``
   * - Ramped-PAS RHSMode=1 output
     - truncated block-Thomas, taped reverse mode (the
       ``tier1_adjoint_window=None`` default; no implicit adjoint)
     - agree at rtol ``1e-6``
   * - Monoenergetic :math:`L_{11}` w.r.t. :math:`\hat B_{mn}`
     - differentiable structured direct + energy convolution
     - ``5.5e-10``
   * - Monoenergetic energy convolution to thermal :math:`L_{ij}`
     - closed-form convolution vs a full RHSMode=2 kinetic solve
     - ``5.8e-14``

As a throughput reference, a ``value_and_grad`` of a 39,318-unknown PAS
scheme-1 objective through the differentiable structured direct route costs
about ``5.3 s`` cold and ``2.1 s`` warm on the development MacBook — the adjoint adds
roughly one forward solve, as predicted.

.. warning::

   **A recycled Krylov gradient on a singular operator aborts rather than
   answers.** The implicit-function-theorem adjoint is a *transposed* solve, and
   on an operator whose null space the constraint scheme does not span it is the
   only thing that fails: the physical drive stays in the range of
   :math:`A`, so the forward solve converges to ``1e-15`` and every field of
   ``SolveResult`` looks healthy while the vector-Jacobian product returns
   garbage. ``dkx`` recomputes the true residual :math:`\|A^T y - g\|` from
   the operator after the transposed solve — never the Krylov method's own
   estimate — records it in ``SolveResult.adjoint``, and raises with the
   residual and the remedies unless you pass ``check_adjoint=False``. Read
   ``result.adjoint`` after the backward pass to see the number behind the
   decision; the check is silent on near-singular decks whose adjoint
   residual is at the double-precision backward-error floor, where the
   gradient is right even though the requested tolerance is unreachable.

The differentiable optimization chain
-------------------------------------

Stellarator optimization with a *kinetic* objective closes the loop from the
plasma boundary to a neoclassical figure of merit and back, entirely under
automatic differentiation:

.. math::

   \text{boundary } \partial\Omega
   \;\xrightarrow[\text{equilibrium}]{\texttt{vmec\_jax}}\;
   \{ \hat B_{mn} \}
   \;\xrightarrow[\text{Boozer transform}]{\texttt{booz\_xform\_jax}}\;
   \text{geometry}
   \;\xrightarrow[\text{kinetic solve}]{\texttt{sfincs\_jax}}\;
   \langle \mathbf{j}\cdot\mathbf{B}\rangle,\ D_{ij},\ \Gamma_s .

Each arrow is a JAX transformation, so ``jax.grad`` of the bootstrap current
:math:`\langle \mathbf{j}\cdot\mathbf{B}\rangle` (or a transport coefficient)
with respect to the boundary Fourier modes propagates through the equilibrium
solve, the Boozer transform, and the drift-kinetic solve without any finite
differences. ``examples/optimization/optimize_QA_bootstrap.py`` drives a
quasi-axisymmetric, low-bootstrap optimization on exactly this chain with warm
starts and finite-difference-verified gradients; the geometry link on its own is
demonstrated in ``examples/autodiff/vmex_to_boozer_sfincs_pipeline.py``. See
:doc:`optimization` and :doc:`vmex_workflow` for the full workflow.

Cost against a non-differentiable reference
-------------------------------------------

A finite-difference gradient of :math:`N` parameters costs :math:`2N`
converged solves with central differences.  Implicit differentiation costs one
transposed solve whatever :math:`N` is, because the adjoint is defined by the
linear equation at the converged solution rather than by differentiating the
iteration.

.. figure:: _static/figures/paper_benchmarks/gradient_cost_scaling.png
   :alt: Gradient wall time against parameter count, and agreement across four configurations.
   :align: center
   :width: 95%

   Four upstream decks spanning one and two species, pitch-angle and
   Fokker-Planck collisions, and zero and finite ``Er``.  The objective is
   ``FSABjHat``; the parameters are the per-species density and temperature
   gradient drives.  Regenerate with
   ``python tools/paper_benchmarks/gradient_cost_scaling.py --results ...``.

The cost is measured at *every* :math:`k`, not extrapolated from one point:
the harness records the wall time of the two solves each parameter needs, so
the cost of a :math:`k`-parameter gradient is a partial sum.  On the
two-species decks that gives four measured points, and they are linear --
``2.86, 5.71, 8.56, 11.37`` seconds -- against a flat one-adjoint cost.  The
fitted slope is ``3.1`` s/parameter.

The absolute ratio at these :math:`k` is modest, and deliberately so: at
:math:`k = 4` a finite difference needs eight solves, the same order as one
forward plus one adjoint, so the measured wall-time ratio is only 1.4x to
7.1x.  The claim is the slope rather than the intercept.  Profile and geometry
optimization run at :math:`k` in the tens, where the slope dominates; the
figure shades that range and marks the line there as fitted rather than
measured.

Agreement with the finite-difference gradient is ``4.7e-10`` to ``4.8e-07``
across the four configurations.  Finite differences have no exact answer to
converge to -- the step size trades truncation error against solver noise --
so ``tools/benchmarks/ad_vs_fortran_fd.py`` sweeps the step and reports the
error floor rather than one lucky value.

The decks are restricted to those whose *reference* solve is well converged in
the true residual (see :doc:`performance`): a finite-difference gradient built
from an under-converged reference measures that reference's noise rather than
its derivative.

Worked examples
---------------

- ``examples/autodiff/gradients_tour.py`` — ``jax.grad`` of kinetic outputs through the
  implicit solve, checked against finite differences (the figure above).
- ``examples/autodiff/matrix_free_residual_and_jvp.py`` — matrix-free residual
  and Jacobian-vector products for the F-block.
- ``examples/autodiff/implicit_diff_through_gmres_solve_scheme5.py`` — implicit
  differentiation through a full-system Krylov solve on a VMEC geometry.
- ``examples/autodiff/differentiable_geometry_gradients.py`` — a geometry scalar
  differentiated with respect to harmonic amplitudes.
- ``examples/optimization/optimize_QA_bootstrap.py`` — gradient-based
  optimization with kinetic :math:`\langle \mathbf{j}\cdot\mathbf{B}\rangle` in
  the objective.

See :doc:`numerics` for the solver routes behind the adjoint and :doc:`performance`
for runtime and memory of the differentiable paths.
