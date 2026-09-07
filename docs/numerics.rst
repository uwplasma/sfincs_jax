Numerics and algorithms
=======================

`dkx` solves a large structured linear (or, with :math:`\Phi_1`,
nonlinear) system that comes from discretizing the radially local drift-kinetic
equation of :doc:`physics_reference` on a single flux surface. This page covers
the discretization, the three solver routes and how one is chosen, and the
implicit-differentiation adjoint.

Discrete unknowns
-----------------

For each kinetic species the first-order distribution is represented on the
tensor grid :math:`(x_i, L, \theta_j, \zeta_k)`:

- :math:`x = v/v_\mathrm{th}` — normalized speed (collocation nodes);
- :math:`L` — Legendre index in :math:`\xi = v_\parallel/v`;
- :math:`\theta,\zeta` — periodic straight-field-line angles.

The total number of degrees of freedom scales as

.. math::

   N_\mathrm{dof} \sim
   N_\mathrm{species}\,N_x\,N_\xi\,N_\theta\,N_\zeta
   + N_\theta N_\zeta + N_\mathrm{constraints},

where the last two terms are the optional :math:`\Phi_1(\theta,\zeta)` unknowns
and the constraint/source coefficients. This scaling is why memory layout,
truncation, and preconditioning matter — a production HSX case is
:math:`\sim 7.4\times10^5` unknowns.

Angular discretization
----------------------

The angles use periodic grids with dense differentiation matrices acting along
each axis, :math:`\partial_\theta f \approx D_\theta f`,
:math:`\partial_\zeta f \approx D_\zeta f` (``thetaDerivativeScheme`` /
``zetaDerivativeScheme``). For the advection-dominated magnetic-drift terms the
code can use directional **upwind** derivative matrices selected pointwise from
the sign of the local drift coefficient, which stabilizes the trapped-passing
boundary layer at low collisionality.

.. _widened-upwind:

Widened upwind stencils
~~~~~~~~~~~~~~~~~~~~~~~

A centered difference puts *no* weight on the diagonal
(:math:`c_0 = 0`), so the advection operator it generates has an empty diagonal
and neither a relaxation smoother nor a diagonal preconditioner can act on it.
`dkx` therefore offers opt-in **widened upwind** angular stencils chosen for
diagonal dominance rather than for the smallest truncation constant. Both are
periodic, first-derivative only, and come as a mirrored pair whose orientation
is set by the sign of the local wind:

.. list-table::
   :header-rows: 1
   :widths: 12 10 30 20 14 14

   * - Namelist
     - Order
     - Offsets (positive wind)
     - Coefficients :math:`\times\,\Delta`
     - :math:`|c_0|/\sum_{j\neq0}|c_j|`
     - Truncation
   * - ``±103``
     - 3
     - :math:`(-3,-1,0,+2)`
     - :math:`(1/15,\,-1,\,5/6,\,1/10)`
     - :math:`5/7 \approx 0.714`
     - :math:`+\tfrac14 \Delta^3 f^{(4)}`
   * - ``±104``
     - 4
     - :math:`(-4,-3,-1,0,+2)`
     - :math:`(-1/12,\,4/15,\,-4/3,\,13/12,\,1/15)`
     - :math:`13/21 \approx 0.619`
     - :math:`+\tfrac15 \Delta^4 f^{(5)}`

For comparison the same measure is :math:`0` for both centered schemes and for
spectral collocation, and :math:`1/3`, :math:`5/14`, :math:`2/11` for the
compact upwind-biased stencils of ``magneticDriftDerivativeScheme`` 1, 2, 3. The
offsets are not free choices: each is the unique maximizer of
:math:`|c_0|/\sum_{j\neq0}|c_j|` over every :math:`N`-point offset set inside a
window of width :math:`N+1` that is exact to order :math:`N-1` and dissipative
for the given wind (:math:`\mathrm{Re}\,\hat c(k)\ge 0` for the Fourier symbol
:math:`\hat c(k) = \sum_j c_j e^{ik o_j}`). Reversing the wind mirrors the
stencil, which preserves both the order and the diagonal weight; using the wrong
orientation flips the sign of :math:`\mathrm{Re}\,\hat c` and is unstable. See
Fromm, *J. Comput. Phys.* **3**, 176 (1968); Warming & Beam, *AIAA J.* **14**,
1241 (1976); Tam & Webb, *J. Comput. Phys.* **107**, 262 (1993); and Fornberg,
*Math. Comput.* **51**, 699 (1988) for the arbitrary-offset weights.

These stencils are a `dkx`-only extension: they are **not** bit-parity with the
Fortran code, they are never a default, and the Fortran-parity suites pin the
centered schemes. The namelist numbering uses a 100 block precisely because
upstream numbers every scheme knob with small integers, so no future upstream
value can collide. ``thetaDerivativeScheme`` / ``zetaDerivativeScheme`` take
``±103`` and ``±104`` with the sign giving the wind direction;
``magneticDriftDerivativeScheme`` takes the same codes with its usual
pair-swapping sign convention, which routes the stencils through the existing
pointwise upwind selector.

Velocity-space discretization
-----------------------------

**Pitch angle** is expanded in Legendre modes,
:math:`f = \sum_{L=0}^{N_\xi-1} f_L(x,\theta,\zeta)\,P_L(\xi)`. The structural
consequences that drive the solver design are:

- streaming and mirror couple :math:`L\leftrightarrow L\pm1`;
- the :math:`E_r` energy/pitch drifts couple :math:`L\leftrightarrow L\pm2`;
- collisions are diagonal in :math:`L` for pitch-angle scattering and dense in
  :math:`x` for the full Fokker--Planck operator.

**Speed** uses the Landreman--Ernst grid: collocation nodes of the non-classical
orthogonal polynomials for the weight :math:`e^{-x^2}x^{k}` (``xGrid_k``),
constructed by a Stieltjes three-term recurrence and Golub--Welsch
eigendecomposition. This gives spectral accuracy for Maxwellian-weighted moments
with few nodes, and the matching spectral differentiation matrices
:math:`d/dx`, :math:`d^2/dx^2` for the energy-drift and Fokker--Planck terms
(`Landreman & Ernst, J. Comput. Phys. 243 (2013) <https://arxiv.org/abs/1210.5289>`_).

**The** :math:`N_\xi`-**for-**:math:`x` **ramp** keeps fewer Legendre modes at
high speed, where the distribution is smoother in pitch: ``Nxi_for_x_option``
sets :math:`N_\xi(x)` to ramp from a floor (the Rosenbluth :math:`N_L`) up to
:math:`N_\xi`. On the 744k-unknown HSX case this ramp is the difference between a
warm solve at ``0.93 GB`` (ramp) and ``1.16 GB`` (uniform :math:`N_\xi`) — see
:doc:`performance`.

.. admonition:: Where in the code

   Legendre couplings and Lorentz eigenvalues:
   :func:`dkx.phase_space.legendre_coupling_upper` /
   ``legendre_coupling_lower`` / ``lorentz_eigenvalues``. Speed grid:
   :func:`dkx.phase_space.make_speed_grid` and
   ``speed_grid_diff_matrices``. Ramp:
   :func:`dkx.phase_space.n_xi_for_x_ramp`. All are collected in
   :class:`dkx.phase_space.Grids` via ``make_grids``.

Linear-system structure
-----------------------

After discretization the problem is :math:`A u = b` with the block structure

.. math::

   A =
   \begin{bmatrix}
     A_{ff} & A_{f\Phi} & A_{fc} \\
     A_{\Phi f} & A_{\Phi\Phi} & A_{\Phi c} \\
     A_{cf} & A_{c\Phi} & A_{cc}
   \end{bmatrix},

with :math:`A_{ff}` the kinetic block, :math:`A_{\Phi\Phi}` the quasineutrality
block (when :math:`\Phi_1` is active), and the :math:`c` rows/columns imposing
density, energy, and gauge constraints. The operator is applied **matrix-free**
as a composition of tensor contractions and directional derivatives rather than
an assembled sparse matrix, so it JIT-compiles for CPU or GPU and differentiates
cleanly.

.. admonition:: Where in the code

   The matrix-free action is :meth:`dkx.drift_kinetic.KineticOperator.apply`;
   the right-hand side is ``KineticOperator.rhs``. The analytic
   block-tridiagonal-in-:math:`L` extraction is
   ``KineticOperator.to_block_tridiagonal``.

The three solver routes
-----------------------

The solve policy (:func:`dkx.solve.solve`, ``solve_method="auto"``) picks the
cheapest adequate route over a :class:`~dkx.drift_kinetic.KineticOperator`.
The three routes are named the same way in a case file's ``[solver] method``
field: ``structured_direct``, ``recycled_krylov``, and
``sparse_direct_referee``.

Code identifiers still spell these routes with the retired tier numbering:
``tier1`` names belong to the structured direct route
(``tier1_keep_lowest``, ``DKX_TIER1_MEMORY_BUDGET_GB``), ``tier2`` to recycled
Krylov (``DKX_TIER2_MEMORY_GUARD``), and ``tier3`` to the sparse direct route.
The prose on every page uses the route names.

Structured direct (block-tridiagonal Legendre elimination)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When the operator reduces to the **DKES-trajectory / pitch-angle-scattering
family** — streaming and mirror couple :math:`L\pm1`, :math:`E\times B` and PAS
are diagonal in :math:`L`, no :math:`E_r` :math:`L\pm2` terms, and no
Fokker--Planck :math:`(\text{species},x)` coupling — the Legendre-mode
representation of the drift-kinetic operator is **block tridiagonal** in
:math:`L`. In that case the :math:`(\text{species}, x)` axes are mutually
uncoupled, so the full system splits into :math:`N_\mathrm{species}\times N_x`
independent block-tridiagonal systems of :math:`N_\xi` dense
:math:`(N_\theta N_\zeta)` blocks, each with a rank-one constraint border. The
border is absorbed exactly with a rank-one update, and the batch is solved by a
``vmap``-ed block-Thomas factor/solve; multiple right-hand sides share one
elimination.

The tridiagonal structure and its block elimination follow the Legendre
analysis of `Escoto, PhD thesis (2025), arXiv:2510.27513
<https://arxiv.org/abs/2510.27513>`_, and rest on the classical block-tridiagonal
and variational treatment of the monoenergetic drift-kinetic equation by
`Hirshman, Shaing, van Rij, Beasley & Crume, Phys. Fluids 29, 2951 (1986)
<https://doi.org/10.1063/1.865495>`_ (with monoenergetic normalizations as in
`Beidler et al., Nucl. Fusion 51, 076001 (2011)
<https://doi.org/10.1088/0029-5515/51/7/076001>`_). The implementation adds a
**truncated-storage** back-substitution: the forward elimination visits all
:math:`N_\xi` blocks, but only the lowest ``keep`` blocks — the ones the
right-hand side and the physical moments actually touch — are retained, so peak
memory is bounded by the truncation depth instead of the full Legendre chain.

Concretely, the structured direct peak memory is

.. math::

   \mathcal{O}\!\left(K\,m^2\right),
   \qquad m = N_\theta N_\zeta,\quad K = \texttt{tier1\_keep\_lowest},

i.e. it scales with the retained keep-depth :math:`K` (default 3) times the
square of the dense angular block dimension :math:`m`, and is **independent of**
:math:`N_\xi` and :math:`N_x`. One :math:`2875^2` block at the 744k-unknown HSX
resolution is about 66 MB, so the truncated route needs :math:`\sim 0.3` GB where
a full-band factorization of the same operator would need :math:`\sim 91` GB
(:doc:`performance`). This is the origin of the structured direct memory
advantage over an assembled sparse factorization. The same discrete operator
also yields an error bound computed from the solution itself, with no reference
run needed: the variational transport-coefficient bounds (:doc:`capabilities`)
bracket the monoenergetic :math:`D_{11}` from above and below.

Recycled Krylov (preconditioned, with subspace recycling)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When the structured direct route does not apply (full Fokker--Planck, or the
full-trajectory :math:`E_r` terms), the code runs matrix-free FGMRES with
subspace recycling (GCROT) on ``KineticOperator.apply``, right-preconditioned by
an **exact structured direct solve of a SFINCS-simplified coarse operator**.
The coarse operator uses the
Fortran ``preconditionerOptions`` idiom — ``preconditioner_species=1``
(self-collisions only) and ``preconditioner_x=1`` (:math:`x`-diagonal
collisions) reduce Fokker--Planck to a PAS-like :math:`L`-diagonal coefficient,
and the :math:`E_r` :math:`L\pm2` terms are dropped — so the preconditioner is
itself a structured direct solve. The recycle pair :math:`(C,U)` is returned for
warm-starting continuation, which makes neighbouring points in an :math:`E_r`
scan or Newton :math:`\Phi_1` iteration converge in a handful of iterations.

Sparse direct (host fallback and independent cross-check)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As an escape hatch the operator is materialized (vmapped unit vectors, guarded
by ``max_dense_size``) into CSR and factored by SuperLU on the host. This route
is non-differentiable and non-jittable and prints a one-line notice; it is used
on explicit request (``method="direct"``) or when the recycled Krylov route
breaches its iteration cap under ``method="auto"``. Because it inverts the
assembled operator with a general-purpose factorization, it also serves as the
independent cross-check on answers from the other two routes; the case-file
value ``sparse_direct_referee`` names that role.

.. admonition:: Where in the code

   :func:`dkx.solve.solve` (auto policy, solve.py); the structured direct build
   ``build_tier1_solver`` and the truncated variant ``_solve_tier1_truncated``;
   the recycled Krylov ``_solve_tier2`` with ``build_coarse_preconditioner``;
   the sparse direct ``_solve_tier3``. The structured factorization, recycled
   Krylov, and host direct solves are provided by the ``solvax`` library
   (`github.com/uwplasma/SOLVAX <https://github.com/uwplasma/SOLVAX>`_,
   `PyPI <https://pypi.org/project/solvax/>`_).

Implicit differentiation (IFT adjoint)
--------------------------------------

For gradient-based workflows the structured direct and recycled Krylov routes
are wrapped with the implicit function theorem
(``jax.lax.custom_linear_solve`` via ``solvax.implicit.linear_solve``). Rather
than differentiating through the solver iterations, the adjoint of a linear
solve :math:`Au=b` is one **transposed solve**, which reuses the same
block-Thomas factors
(``block_thomas_solve(transpose=True)``) or a transposed-preconditioner GCROT
solve. The cost of a gradient is therefore one additional solve, independent of
the iteration count of the forward solve. The ambipolar :math:`E_r` root and the
nonlinear :math:`\Phi_1` Newton solve are differentiated the same way at the
outer (root) level (:func:`dkx.er.ambipolar_er`,
:func:`dkx.phi1.phi1_state`).

Numerical building blocks — the structured factorizations, the recycled Krylov,
the mixed-precision block-Thomas, and the implicit-solve wrappers — live in the
standalone ``solvax`` package so they can be tested and reused independently.
The mixed-precision block-Thomas path runs on GPU only (it is faster on GPU FP64
but slower on CPU), so the CPU path uses the plain block-Thomas factorization.

When to use which route
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 26 20 30 24

   * - Case
     - Auto route
     - Why
     - Differentiable
   * - DKES trajectories + PAS (RHSMode 3, monoenergetic)
     - Structured direct
     - Block-tridiagonal in :math:`L`; :math:`N_s N_x` independent chains
     - yes (transposed factors)
   * - PAS, full profile solve (RHSMode 1)
     - Structured direct
     - Same structure; multi-RHS shares one elimination
     - yes
   * - Full Fokker--Planck collisions
     - Recycled Krylov
     - Dense :math:`x`/species coupling breaks tridiagonality
     - yes (transposed preconditioner)
   * - Full-trajectory :math:`E_r` (:math:`L\pm2` terms)
     - Recycled Krylov
     - :math:`L\pm2` coupling breaks tridiagonality
     - yes
   * - Ill-conditioned / small, or a recycled Krylov stall
     - Sparse direct
     - Host SuperLU direct
     - no (loud escape hatch)

Resolution guidance
-------------------

The practical knobs are :math:`N_\theta, N_\zeta, N_\xi, N_x`. Low-collisionality
runs are especially sensitive to :math:`N_\zeta` and :math:`N_\xi` because of the
trapped-passing boundary layer, while :math:`N_x` changes more slowly with
collisionality. Convergence is therefore best checked by refining one axis at a
time rather than by a blind global scale factor; the examples and audited suite
choose resolution changes per axis. For measured runtime/memory and parity
evidence see :doc:`performance` and :doc:`parity`.

Auditing cold and warm observable differences
---------------------------------------------

The bounded diagnostic in ``tools/benchmarks/operator_conditioning.py`` can
compare cold solves, initial-state reuse, recycle-space reuse, and both:

.. code-block:: bash

   python tools/benchmarks/operator_conditioning.py target.namelist \
       --warm-from source.namelist --observable heatFlux_vm_psiHat --max-size 4000

The source and target must share discretization and model structure. The tool
builds their coefficients independently, passes no cached preconditioner or
factorization, fingerprints both complete operators before/after the audit,
and hashes the target matrix and drive. It checks original residuals at
relative tolerances 1e-8, 1e-10 and 1e-12, retaining failures and executed methods.
Each row records whether an initial state and recycle data were actually
supplied, separately from the requested trial label. The seed has its own
convergence flag, absolute residual and original-residual gate. A zero-drive
seed is allowed: its relative residual is null, and its absolute residual
must vanish to pass. Non-finite drives or overflowing drive norms are rejected
before dense materialization. The selected moment is normalized current or
first-species heat flux.

For a linear observable with coefficient vector :math:`q`, the diagnostic
solves :math:`A^T\lambda=q` and compares
:math:`J(x_w)-J(x_c)` with :math:`\lambda^T(r_c-r_w)`, where
:math:`r=b-Ax`. The report retains the adjoint residual, identity remainder,
and the difference between dense and original-operator residual evaluations.
The adjoint-residual remainder bound excludes floating-point evaluation error.
This attributes differences between computed states; it is not an exact-solution
or grid-error certificate. An ill-conditioned adjoint can itself be unreliable.
Coupled Phi1 requires an explicit linearization and is rejected here.

The historical 2.46% report in PR #161 has no identified input pair in its PR
body. A different supplied pair is a new audit, not a reproduction of that
report; warm-restart promotion still requires the R1 acceptance evidence.
