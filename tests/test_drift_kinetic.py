"""Structural and physics tests for ``dkx.drift_kinetic.KineticOperator``.

Element-wise Fortran matvec/RHS/residual parity lives in
``tests/test_kinetic_operator_fortran_parity.py`` (whichMatrix petscbin
referees).  This file pins the operator's structure: dense materialization vs
the frozen Fortran matrix, the analytic Legendre-block extraction, transport
RHS column validation, Phi1 layouts, magnetic-drift scheme wiring, and the
constraintScheme source bases.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dkx.drift_kinetic import KineticOperator
from dkx.namelist import parse_sfincs_input_text, read_sfincs_input

REF = Path(__file__).parent / "ref"

# Tiny fixtures spanning: RHSMode 1/3, PAS + FP collisions, 1 and 2 species,
# Er=0 and Er!=0 (ExB + Er xDot / xiDot terms), geometry schemes 1, 4, 12,
# constraintScheme 1 and 2, Nzeta=1 (axisymmetric) and Nzeta>1, and the
# tangential magnetic drifts (magneticDriftScheme=1, geometryScheme 11).
CASES = [
    "monoenergetic_PAS_tiny_scheme1",
    "pas_1species_PAS_noEr_tiny_scheme1",
    "quick_2species_FPCollisions_noEr",
    "er_xdot_1species_tiny",
    "er_xidot_1species_tiny",
    "pas_1species_PAS_noEr_tiny_scheme12",
    "magdrift_1species_tiny",
]

RHSMODE2_TEXT = """
&general
  RHSMode = 2
/
&geometryParameters
  geometryScheme = 1
  inputRadialCoordinate = 3
  rN_wish = 0.3
  B0OverBBar = 1.0d+0
  GHat = 1.0d+0
  IHat = 0.0d+0
  iota = 1.31d+0
  epsilon_t = 0.1d+0
  epsilon_h = 0.0d+0
  helicity_l = 1
  helicity_n = 1
  psiAHat = 0.045d+0
  aHat = 0.1
/
&speciesParameters
  Zs = 1
  mHats = 1.0d+0
  nHats = 1.0d+0
  THats = 1.0d+0
/
&physicsParameters
  Delta = 4.5694d-3
  alpha = 1.0d+0
  nu_n = 0.15d+0
  Er = 0.0d+0
  collisionOperator = 1
  includeXDotTerm = .false.
  includeElectricFieldTermInXiDot = .false.
  useDKESExBDrift = .true.
  includePhi1 = .false.
/
&resolutionParameters
  Ntheta = 7
  Nzeta = 5
  Nxi = 5
  NL = 2
  Nx = 3
/
&otherNumericalParameters
  Nxi_for_x_option = 0
/
"""

# PAS + DKES trajectories with Er != 0: streaming/mirror (L±1), ExB (diagonal
# in L), and PAS collisions (diagonal in L) — the structured direct block-tridiagonal
# family used to validate the analytic Legendre-block extraction.
PAS_DKES_ER_TEXT = """
&general
/
&geometryParameters
  geometryScheme = 1
  inputRadialCoordinate = 3
  rN_wish = 0.3
  B0OverBBar = 1.0d+0
  GHat = 1.0d+0
  IHat = 0.0d+0
  iota = 1.31d+0
  epsilon_t = 0.1d+0
  epsilon_h = 0.05d+0
  helicity_l = 2
  helicity_n = 5
  psiAHat = 0.045d+0
  aHat = 0.1
/
&speciesParameters
  Zs = 1
  mHats = 1.0d+0
  nHats = 1.0d+0
  THats = 0.5d+0
  dNHatdrHats = -0.5d+0
  dTHatdrHats = -1.0d+0
/
&physicsParameters
  Delta = 4.5694d-3
  alpha = 1.0d+0
  nu_n = 8.4774d-3
  Er = 0.4d+0
  collisionOperator = 1
  includeXDotTerm = .false.
  includeElectricFieldTermInXiDot = .false.
  useDKESExBDrift = .true.
  includePhi1 = .false.
/
&resolutionParameters
  Ntheta = 7
  Nzeta = 5
  Nxi = 5
  NL = 2
  Nx = 3
/
&otherNumericalParameters
  Nxi_for_x_option = 0
/
"""


_FP_KRONECKER_TEXT = """
&general
/
&geometryParameters
  geometryScheme = 4
/
&speciesParameters
  Zs = 1
  mHats = 1
  nHats = 1.0d+0
  THats = 1.0d+0
  dNHatdrHats = -0.5
  dTHatdrHats = -2.0
/
&physicsParameters
  Delta = 4.5694d-3
  alpha = 1d+0
  nu_n = 8.3d-3
  Er = 0.0
  collisionOperator = 0
  includeXDotTerm = .false.
  includeElectricFieldTermInXiDot = .false.
/
&resolutionParameters
  Ntheta = 5
  Nzeta = 5
  Nxi = 4
  Nx = 4
  solverTolerance = 1d-11
/
&otherNumericalParameters
  Nxi_for_x_option = 0
/
&preconditionerOptions
/
&export_f
/
"""


def _load(name: str):
    return read_sfincs_input(REF / f"{name}.input.namelist")


def _assert_close(new: np.ndarray, old: np.ndarray) -> None:
    """1e-13 relative agreement (atol tied to the reference vector scale)."""
    old = np.asarray(old)
    scale = max(1.0, float(np.max(np.abs(old))) if old.size else 1.0)
    np.testing.assert_allclose(np.asarray(new), old, rtol=1e-13, atol=1e-13 * scale)


# ---------------------------------------------------------------------------
# Transport RHS columns: structure + validation (Fortran value parity lives in
# tests/test_kinetic_operator_fortran_parity.py)
# ---------------------------------------------------------------------------


def test_rhs_transport_columns_rhsmode3() -> None:
    op_new = KineticOperator.from_namelist(_load("monoenergetic_PAS_tiny_scheme1"))
    for which_rhs in (1, 2):
        ref = op_new.rhs(which_rhs)
        assert float(np.max(np.abs(np.asarray(ref)))) > 0.0
    with pytest.raises(ValueError):
        op_new.rhs(3)


def test_rhs_transport_columns_rhsmode2() -> None:
    op_new = KineticOperator.from_namelist(parse_sfincs_input_text(RHSMODE2_TEXT))
    for which_rhs in (1, 2, 3):
        ref = op_new.rhs(which_rhs)
        assert float(np.max(np.abs(np.asarray(ref)))) > 0.0
    with pytest.raises(ValueError):
        op_new.rhs(4)


# ---------------------------------------------------------------------------
# Matrix-free apply == materialized matrix == frozen Fortran matrix
# (tiniest fixture: 111 unknowns)
# ---------------------------------------------------------------------------


def test_matrix_free_apply_equals_materialized() -> None:
    from scipy.sparse import csr_matrix

    from dkx.validation.fortran import read_petsc_mat_aij

    base = "pas_1species_PAS_noEr_tiny_scheme1"
    op_new = KineticOperator.from_namelist(_load(base))
    n = op_new.total_size
    assert n <= 256, "materialization test is meant for the tiniest fixture"

    eye = jnp.eye(n, dtype=jnp.float64)
    mat_new = np.asarray(jax.vmap(op_new.apply)(eye)).T  # columns = apply(e_j)

    a = read_petsc_mat_aij(REF / f"{base}.whichMatrix_3.petscbin")
    mat_fortran = csr_matrix((a.data, a.col_ind, a.row_ptr), shape=a.shape).toarray()
    _assert_close(mat_new, mat_fortran)

    rng = np.random.default_rng(2)
    for _ in range(3):
        v = rng.standard_normal(n)
        _assert_close(op_new.apply(jnp.asarray(v)), mat_new @ v)


# ---------------------------------------------------------------------------
# Analytic Legendre-block extraction (the probing-free structured direct route)
# ---------------------------------------------------------------------------


def _block_tridiagonal_matvec(op_new: KineticOperator, f: np.ndarray) -> np.ndarray:
    """Reference block-tridiagonal matvec over L using the extracted blocks."""
    blocks = op_new.to_block_tridiagonal()
    lower = np.asarray(blocks.lower)  # (L,S,X,TZ,TZ)
    diag = np.asarray(blocks.diag)
    upper = np.asarray(blocks.upper)

    n_s, n_x, n_xi = op_new.n_species, op_new.n_x, op_new.n_xi
    n_tz = op_new.n_theta * op_new.n_zeta
    g = f.reshape(n_s, n_x, n_xi, n_tz)
    y = np.zeros_like(g)
    for s in range(n_s):
        for ix in range(n_x):
            for ell in range(n_xi):
                acc = diag[ell, s, ix] @ g[s, ix, ell]
                if ell > 0:
                    acc = acc + lower[ell, s, ix] @ g[s, ix, ell - 1]
                if ell + 1 < n_xi:
                    acc = acc + upper[ell, s, ix] @ g[s, ix, ell + 1]
                y[s, ix, ell] = acc
    return y.reshape(op_new.f_shape)


@pytest.mark.parametrize(
    "source",
    ["monoenergetic_PAS_tiny_scheme1", "pas_dkes_with_er_inline"],
)
def test_legendre_blocks_reproduce_fblock_apply(source: str) -> None:
    if source == "pas_dkes_with_er_inline":
        nml = parse_sfincs_input_text(PAS_DKES_ER_TEXT)
    else:
        nml = _load(source)
    op_new = KineticOperator.from_namelist(nml)

    if source == "pas_dkes_with_er_inline":
        # The point of this case: the ExB diagonal-in-L term must be present.
        assert op_new.with_exb

    rng = np.random.default_rng(4)
    for _ in range(3):
        f = rng.standard_normal(op_new.f_shape)
        y_apply = np.asarray(op_new.apply_f(jnp.asarray(f)))
        y_blocks = _block_tridiagonal_matvec(op_new, f)
        assert np.max(np.abs(y_apply)) > 0.0
        _assert_close(y_blocks, y_apply)


def test_legendre_blocks_reject_l2_coupled_terms() -> None:
    # Er xDot couples L±2: the block-tridiagonal extraction must refuse.
    op_new = KineticOperator.from_namelist(_load("er_xdot_1species_tiny"))
    with pytest.raises(NotImplementedError):
        op_new.legendre_blocks(0)
    # Fokker-Planck (dense species/x blocks) is likewise not extracted here.
    op_fp = KineticOperator.from_namelist(_load("quick_2species_FPCollisions_noEr"))
    with pytest.raises(NotImplementedError):
        op_fp.to_block_tridiagonal()
    # Tangential magnetic drifts couple L±2: also not block-tridiagonal.
    op_md = KineticOperator.from_namelist(_load("magdrift_1species_tiny"))
    with pytest.raises(NotImplementedError, match="magnetic drift"):
        op_md.legendre_blocks(0)


def test_block_extraction_is_jit_safe_over_traced_operator_leaves() -> None:
    """``to_block_tridiagonal`` must stay traceable when the operator *leaves*
    are tracers (jit-over-leaves / vmap / the differentiable kernel).

    Regression for the former ``mask = np.asarray(self._mask())`` host
    round-trip in :meth:`KineticOperator.legendre_blocks`, which raised
    ``TracerArrayConversionError`` once ``n_xi_for_x`` entered the trace.  A
    ramped ``Nxi_for_x`` deck makes the truncation mask non-uniform (the
    interesting, non-all-ones case); the extracted bands under jit-over-leaves
    must match the eager bands to round-off (they differ only by ~1 float64 ULP
    of XLA FMA contraction), so switching the mask to a traced jnp array leaves
    the numerics unchanged.
    """
    text = (
        PAS_DKES_ER_TEXT.replace("Nxi = 5", "Nxi = 16")
        .replace("Nx = 3", "Nx = 5")
        .replace("Nxi_for_x_option = 0", "Nxi_for_x_option = 1")
    )
    op = KineticOperator.from_namelist(parse_sfincs_input_text(text))
    # the point of this fixture: a genuine speed-dependent Legendre ramp
    assert int(np.min(np.asarray(op.n_xi_for_x))) < op.n_xi

    leaves, treedef = jax.tree_util.tree_flatten(op)

    def blocks_from_leaves(ls):
        return jax.tree_util.tree_unflatten(treedef, ls).to_block_tridiagonal()

    eager = op.to_block_tridiagonal()
    jitted = jax.jit(blocks_from_leaves)(leaves)  # compiles (was a Tracer error)
    for name in ("lower", "diag", "upper"):
        np.testing.assert_allclose(
            np.asarray(getattr(jitted, name)),
            np.asarray(getattr(eager, name)),
            rtol=1e-13,
            atol=1e-13,
            err_msg=f"{name} band changed under jit-over-leaves",
        )


# ---------------------------------------------------------------------------
# Deferred features fail loudly at construction
# ---------------------------------------------------------------------------


def test_include_phi1_is_canonical_with_collision_coupling() -> None:
    # includePhi1 (quasineutrality + kinetic coupling) is consolidated: the
    # operator builds and carries the Phi1 rows/lambda layout.
    nml = _load("pas_1species_PAS_noEr_tiny_withPhi1_linear")
    op = KineticOperator.from_namelist(nml)
    assert op.include_phi1
    assert op.phi1_size == op.n_theta * op.n_zeta + 1
    assert op.total_size == op.f_size + op.phi1_size + op.extra_size
    assert op.fp_phi1 is None

    # includePhi1InCollisionOperator is canonical: the operator builds the
    # poloidally varying Fokker-Planck collision operator (fp_phi1, not fp) and
    # reproduces the frozen Fortran residual at the recorded Newton state.
    from dkx.validation.fortran import read_petsc_vec

    coll_base = "fp_1species_FPCollisions_noEr_tiny_withPhi1_inCollision"
    coll = _load(coll_base)
    op_c = KineticOperator.from_namelist(coll)
    assert op_c.fp_phi1 is not None and op_c.fp is None
    x_ref = read_petsc_vec(REF / f"{coll_base}.stateVector.petscbin").values
    r_ref = read_petsc_vec(REF / f"{coll_base}.residual.petscbin").values
    r = np.asarray(op_c.residual_phi1(jnp.asarray(x_ref)))
    scale = max(1.0, float(np.max(np.abs(x_ref))))
    np.testing.assert_allclose(r, r_ref, rtol=0.0, atol=1e-11 * scale)


def test_magnetic_drifts_are_canonical_and_reject_block_extraction() -> None:
    # magneticDriftScheme=1 (tangential magnetic drift) is consolidated: the
    # operator builds and reproduces the frozen Fortran whichMatrix=1 matvec
    # element-wise.
    from scipy.sparse import csr_matrix

    from dkx.validation.fortran import read_petsc_mat_aij

    base = "magdrift_1species_tiny"
    nml = _load(base)
    op_new = KineticOperator.from_namelist(nml)
    assert op_new.with_magnetic_drifts
    a = read_petsc_mat_aij(REF / f"{base}.whichMatrix_1.petscbin")
    mat = csr_matrix((a.data, a.col_ind, a.row_ptr), shape=a.shape)

    rng = np.random.default_rng(11)
    for _ in range(5):
        v = rng.standard_normal(op_new.total_size)
        y_ref = mat.dot(v)
        y_new = np.asarray(op_new.apply(jnp.asarray(v)))
        assert np.max(np.abs(y_ref)) > 0.0
        # 5 random vectors; the frozen Fortran assembly carries ~1e-12
        # summation-order roundoff per entry, so gate at 1e-11 absolute
        # (term-wise element parity is pinned in test_magnetic_drifts_parity).
        np.testing.assert_allclose(y_new, y_ref, rtol=0.0, atol=1e-11)

    # The d/dtheta, d/dzeta, and d/dxi drift terms couple L±2, so the
    # block-tridiagonal extraction must refuse (solve.py routes to recycled Krylov).
    with pytest.raises(NotImplementedError, match="magnetic drift"):
        op_new.legendre_blocks(0)
    with pytest.raises(NotImplementedError, match="magnetic drift"):
        op_new.to_block_tridiagonal()


def test_magnetic_drift_subschemes_canonical_with_fortran_restrictions() -> None:
    # All schemes 0-9 are canonical now.  Out-of-range values mirror
    # validateInput.F90 ("magneticDriftScheme must be >= 0" / "<= 9") as
    # ValueError, and the geometryScheme restrictions mirror the Fortran ones:
    # drifts need the radial B-field derivatives (5/11/12 canonically), and
    # scheme 4 is geometryScheme 11/12 only (validateInput.F90:507).
    text = (REF / "magdrift_1species_tiny.input.namelist").read_text()
    for bad in (-1, 10):
        nml_bad = parse_sfincs_input_text(
            text.replace("magneticDriftScheme = 1", f"magneticDriftScheme = {bad}")
        )
        with pytest.raises(ValueError, match="magneticDriftScheme"):
            KineticOperator.from_namelist(nml_bad)
    # Drift schemes with an analytic geometry (no radial derivatives) refuse:
    nml_geo = parse_sfincs_input_text(
        text.replace("magneticDriftScheme = 1", "magneticDriftScheme = 2").replace(
            "geometryScheme = 11", "geometryScheme = 4"
        )
    )
    with pytest.raises(NotImplementedError, match="magneticDriftScheme=2"):
        KineticOperator.from_namelist(nml_geo)
    # Scheme 4 with a VMEC geometry mirrors the Fortran geometryScheme 11/12 gate:
    nml_s4 = parse_sfincs_input_text(
        text.replace("magneticDriftScheme = 1", "magneticDriftScheme = 4").replace(
            "geometryScheme = 11", "geometryScheme = 5"
        )
    )
    with pytest.raises(ValueError, match="magneticDriftScheme 4"):
        KineticOperator.from_namelist(nml_s4)


# ---------------------------------------------------------------------------
# geometryScheme 3 (LHD inward-shifted analytic Boozer model)
# ---------------------------------------------------------------------------


def test_geometry_scheme3_bfield_parity_and_end_to_end() -> None:
    """geometryScheme=3 is wired into the canonical operator and runs end to end.

    The analytic LHD inward-shifted geometry already lives in
    :meth:`dkx.magnetic_geometry.FluxSurfaceGeometry.from_scheme` (scheme
    3); the operator builder used to raise ``NotImplementedError`` for it.  This
    pins the now-wired path: the operator B-field must equal ``from_scheme(3)``
    exactly, and a tiny monoenergetic (RHSMode=3) transport-matrix run must
    converge to finite fluxes through the canonical ``run_transport_matrix``.
    """
    from dkx.magnetic_geometry import FluxSurfaceGeometry
    from dkx.phase_space import make_grids
    from dkx.run import run_transport_matrix

    name = "monoenergetic_PAS_tiny_scheme3"
    op = KineticOperator.from_namelist(_load(name))

    # B-field parity: the operator's geometry must match the from_scheme source
    # of truth on the same theta/zeta grids (NPeriods=10 for scheme 3).
    grids = make_grids(
        n_theta=op.n_theta, n_zeta=op.n_zeta, n_xi=op.n_xi, n_x=op.n_x,
        n_l=3, n_periods=10, monoenergetic=True,
    )
    geom = FluxSurfaceGeometry.from_scheme(3, theta=grids.theta, zeta=grids.zeta)
    assert np.array_equal(np.asarray(op.b_hat), np.asarray(geom.b_hat))

    # End-to-end monoenergetic run: converged, finite 2x2 transport matrix.
    run = run_transport_matrix(REF / f"{name}.input.namelist", emit=None)
    assert run.solve_result.converged
    tm = np.asarray(run.transport_matrix)
    assert tm.shape == (2, 2)
    assert np.all(np.isfinite(tm))
    assert np.all(np.isfinite(np.asarray(run.state_vectors)))


# ---------------------------------------------------------------------------
# constraintScheme 3 and 4 (constant+quartic / quadratic+quartic sources)
#
# cs3/4 differ from cs1 ONLY in the two source x-shapes injected into the L=0
# DKE rows (populateMatrix.F90 lines 2915-2938); the flux-surface-averaged
# density/pressure constraint rows are shared.  Canonical-vs-Fortran output
# parity lives in tests/test_output_h5_constraintscheme34_parity.py (the legacy
# stack reuses the cs1 basis for cs3/4, so it is NOT a valid oracle here).
# ---------------------------------------------------------------------------


def test_constraint_scheme_3_4_source_basis_matches_fortran() -> None:
    """``_source_basis`` equals populateMatrix.F90's xPartOfSource1/2 for 1/3/4."""
    op = KineticOperator.from_namelist(_load("fp_1species_FPCollisions_noEr_tiny_cs3"))
    x2 = np.asarray(op.x) ** 2
    coef = np.exp(-x2) / (np.pi * np.sqrt(np.pi))
    expected = {
        1: ((-x2 + 2.5) * coef, (2.0 / 3.0 * x2 - 1.0) * coef),
        3: ((-1.0 / 5.0 * x2 * x2 + 7.0 / 4.0) * coef, (2.0 / 15.0 * x2 * x2 - 0.5) * coef),
        4: (
            (-2.0 / 3.0 * x2 * x2 + 7.0 / 3.0 * x2) * coef,
            (4.0 / 15.0 * x2 * x2 - 2.0 / 3.0 * x2) * coef,
        ),
    }
    for scheme, (e1, e2) in expected.items():
        s1, s2 = op._source_basis(scheme)
        np.testing.assert_allclose(np.asarray(s1), e1, rtol=0, atol=1e-14)
        np.testing.assert_allclose(np.asarray(s2), e2, rtol=0, atol=1e-14)


def test_constraint_scheme_3_4_operator_differs_from_scheme1() -> None:
    """Canonical cs3/cs4 apply differs from cs1 on the same deck (basis took effect).

    The bordered source columns carry the scheme-specific x-shapes, so the full
    apply must differ from cs1 — the guard against the legacy fall-through that
    silently reused the cs1 basis.  ``extra_size`` and the shared density/pressure
    constraint rows are unchanged (both carry ``2*Nspecies`` source unknowns).
    """
    base = (REF / "fp_1species_FPCollisions_noEr_tiny_cs3.input.namelist").read_text()
    ops = {
        cs: KineticOperator.from_namelist(
            parse_sfincs_input_text(base.replace("constraintScheme = 3", f"constraintScheme = {cs}"))
        )
        for cs in (1, 3, 4)
    }
    for cs in (1, 3, 4):
        assert ops[cs].constraint_scheme == cs
        assert ops[cs].extra_size == 2 * ops[cs].n_species

    rng = np.random.default_rng(0)
    v = jnp.asarray(rng.standard_normal(ops[1].total_size))
    y = {cs: np.asarray(ops[cs].apply(v)) for cs in (1, 3, 4)}
    # The three source bases are genuinely distinct, so the applies must separate.
    assert np.max(np.abs(y[3] - y[1])) > 1e-3
    assert np.max(np.abs(y[4] - y[1])) > 1e-3
    assert np.max(np.abs(y[4] - y[3])) > 1e-3


# ---------------------------------------------------------------------------
# Fokker-Planck: the Kronecker structure a direct route needs
# ---------------------------------------------------------------------------


def _dense_f_block(op) -> np.ndarray:
    """The f-block of a one-species operator as ``(X, L, TZ, X, L, TZ)``."""
    n_species, n_x, n_xi, n_theta, n_zeta = op.f_shape
    assert n_species == 1
    size = n_x * n_xi * n_theta * n_zeta
    basis = jnp.eye(size).reshape((size, *op.f_shape))
    columns = jax.vmap(lambda u: op.apply_f(u).reshape(-1))(basis)
    return np.asarray(columns).T.reshape(
        n_x, n_xi, n_theta * n_zeta, n_x, n_xi, n_theta * n_zeta
    )


@pytest.mark.parametrize("er", ("0.0", "-1.5"))
def test_fokker_planck_block_is_a_kronecker_sum_in_speed_and_angle(er: str) -> None:
    r"""``diag_L = C_{x,L} (x) I_TZ`` and ``offdiag_L = diag(x) (x) S_L``, exactly.

    This is the structure that decides whether a *direct* route is affordable
    for Fokker-Planck decks.  ``legendre_blocks`` refuses them because the
    collision operator couples ``(species, x)`` densely inside each Legendre
    index, which would turn the block-Thomas block size from ``TZ`` into
    ``Nx*TZ`` -- an ``Nx^2`` cost multiplier.

    But the coupling is not arbitrarily dense.  Two facts hold to machine
    precision, and together they say the operator is a Kronecker sum:

    1. Every *speed-off-diagonal* coupling is a multiple of the identity in
       ``(theta, zeta)``: collisions act on ``x`` alone, so they contribute
       ``C_x (x) I``.
    2. Every *Legendre-off-diagonal* block (streaming and the mirror force)
       is the same ``theta/zeta`` matrix scaled by the speed ``x_i``: those
       terms are proportional to the particle speed, so they contribute
       ``diag(x) (x) S``.

    Consequence: scaling the system by ``diag(x)^{-1}`` makes the
    Legendre coupling independent of ``x``, which is what lets the dense
    speed coupling be carried at the cost of the pitch-angle-scattering
    factorization instead of ``Nx^2`` times it.

    Both facts survive a finite ``Er``: the ``ExB`` drift is independent of
    speed and diagonal in the Legendre index, so it lands entirely on the
    part of the block that neither statement constrains.
    """
    text = _FP_KRONECKER_TEXT.replace("Er = 0.0", f"Er = {er}")
    op = KineticOperator.from_namelist(parse_sfincs_input_text(text))
    assert op.fp is not None  # the whole point: a Fokker-Planck deck
    block = _dense_f_block(op)
    n_x, n_xi = block.shape[0], block.shape[1]
    speeds = np.asarray(op.x)

    for ell in range(n_xi):
        for i in range(n_x):
            for j in range(n_x):
                if i == j:
                    continue
                coupling = block[i, ell, :, j, ell, :]
                scale = max(float(np.max(np.abs(coupling))), 1e-300)
                off_diagonal = coupling - np.diag(np.diag(coupling))
                assert float(np.max(np.abs(off_diagonal))) / scale < 1e-13, (
                    f"speed coupling at L={ell}, x {i}<-{j} is not a multiple of I"
                )

    for ell in range(n_xi - 1):
        reference = block[0, ell, :, 0, ell + 1, :]
        support = np.abs(reference) > 1e-12 * max(float(np.max(np.abs(reference))), 1e-300)
        if not support.any():
            continue
        for i in range(n_x):
            candidate = block[i, ell, :, i, ell + 1, :]
            expected = (speeds[i] / speeds[0]) * reference
            scale = max(float(np.max(np.abs(candidate))), 1e-300)
            assert float(np.max(np.abs(candidate - expected))) / scale < 1e-13, (
                f"streaming block at L={ell}->{ell + 1}, x={i} is not x-proportional"
            )

@pytest.mark.parametrize("truncated", [False, True])
def test_staged_solve_reuses_layout_with_fresh_physics(truncated):
    """Rebuilt collision coefficients must pass through a cached executable."""
    from dkx.solve import solve

    text = PAS_DKES_ER_TEXT
    if truncated:
        text = text.replace("Nxi = 5", "Nxi = 8").replace(
            "Nxi_for_x_option = 0", "Nxi_for_x_option = 1"
        )
    decks = [
        text,
        text.replace("THats = 0.5d+0", "THats = 0.7d+0").replace(
            "nHats = 1.0d+0", "nHats = 1.3d+0"
        ),
    ]
    if truncated:
        # Same rectangular shapes, different active layout: must retrace.
        decks.append(text.replace("Nxi_for_x_option = 1", "Nxi_for_x_option = 0"))
    operators = [KineticOperator.from_namelist(parse_sfincs_input_text(deck)) for deck in decks]
    assert not np.array_equal(operators[0].t_hat, operators[1].t_hat)
    assert (operators[0].active_dof_mask() is not None) == truncated
    traces = []

    def staged(op):
        traces.append(1)
        return solve(op, op.rhs(), method="gmres", tol=1e-10).x

    compiled = jax.jit(staged)
    states = []
    for op in operators:
        state = compiled(op)
        rhs = op.rhs()
        np.testing.assert_allclose(op.apply(state), rhs, rtol=0, atol=1e-10 * np.linalg.norm(rhs))
        cold = solve(op, rhs, method="gmres", tol=1e-10).x
        np.testing.assert_allclose(state, cold, rtol=1e-7, atol=1e-10)
        states.append(np.asarray(state))
    assert len(traces) == (2 if truncated else 1)
    assert not np.allclose(states[0], states[1])
