from __future__ import annotations

from pathlib import Path

import numpy as np
import jax.numpy as jnp

from sfincs_jax.indices import V3Indexing
from sfincs_jax.magnetic_drifts import (
    MagneticDriftThetaV3Operator,
    MagneticDriftZetaV3Operator,
    MagneticDriftXiDotV3Operator,
    apply_magnetic_drift_theta_v3_offdiag2,
    apply_magnetic_drift_zeta_v3_offdiag2,
    apply_magnetic_drift_xidot_v3_offdiag2,
)
from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.petsc_binary import read_petsc_mat_aij
from sfincs_jax.v3 import grids_from_namelist


def _load_geom_npz(path: Path) -> dict[str, np.ndarray]:
    """Load geometry fixture saved from Fortran/HDF5.

    Fortran arrays are column-major. When they are written to HDF5 and then read
    back in Python, the leading 2D axes often appear transposed relative to the
    (itheta, izeta) indexing used in SFINCS source.

    The `.npz` fixture follows that raw Fortran/HDF5 ordering, so we transpose all
    2D arrays to get (Ntheta, Nzeta) arrays consistent with the JAX operators.
    """
    data = np.load(path)
    out: dict[str, np.ndarray] = {}
    for k in data.files:
        arr = np.asarray(data[k])
        if arr.ndim == 2:
            arr = arr.T
        out[k] = arr
    return out


def _pack_f(inv: list[tuple[int, int, int, int, int]], indexing: V3Indexing, x_vec: np.ndarray) -> np.ndarray:
    f = np.zeros((indexing.n_species, indexing.n_x, indexing.n_xi_max, indexing.n_theta, indexing.n_zeta), dtype=np.float64)
    for g, (s, ix, l, it, iz) in enumerate(inv):
        f[s, ix, l, it, iz] = x_vec[g]
    return f


def _unpack_f(inv: list[tuple[int, int, int, int, int]], y: np.ndarray) -> np.ndarray:
    y_vec = np.zeros((len(inv),), dtype=np.float64)
    for g, (s, ix, l, it, iz) in enumerate(inv):
        y_vec[g] = y[s, ix, l, it, iz]
    return y_vec


def _csr_matvec_filtered(
    *,
    a,
    inv: list[tuple[int, int, int, int, int]],
    x_vec: np.ndarray,
    keep_entry,
) -> np.ndarray:
    n_f = len(inv)
    y = np.zeros((n_f,), dtype=np.float64)
    for row in range(n_f):
        s_r, ix_r, l_r, it_r, iz_r = inv[row]
        start = int(a.row_ptr[row])
        end = int(a.row_ptr[row + 1])
        cols = a.col_ind[start:end]
        vals = a.data[start:end]
        acc = 0.0
        for c, v in zip(cols.tolist(), vals.tolist()):
            if c < 0 or c >= n_f:
                continue
            s_c, ix_c, l_c, it_c, iz_c = inv[c]
            if not keep_entry((s_r, ix_r, l_r, it_r, iz_r), (s_c, ix_c, l_c, it_c, iz_c)):
                continue
            acc += float(v) * float(x_vec[c])
        y[row] = acc
    return y


def test_magnetic_drift_theta_offdiag2_offdiag_theta_matches_fortran() -> None:
    """Parity for magnetic-drift d/dtheta term: |ΔL|=2, off-diagonal in theta.

    We zero out the diagonal of ddtheta on the JAX side to isolate the off-diagonal-in-theta
    contribution, which avoids contamination from the magnetic-drift xiDot term (diagonal in theta).
    """
    here = Path(__file__).parent
    input_path = here / "ref" / "magdrift_1species_tiny.input.namelist"
    mat_path = here / "ref" / "magdrift_1species_tiny.whichMatrix_1.petscbin"
    geom_path = here / "ref" / "magdrift_1species_tiny.geometry.npz"

    nml = read_sfincs_input(input_path)
    grids = grids_from_namelist(nml)
    geom = _load_geom_npz(geom_path)

    species = nml.group("speciesParameters")
    phys = nml.group("physicsParameters")
    delta = float(phys.get("DELTA", 0.0))
    t_hat = float(np.atleast_1d(np.asarray(species["THATS"], dtype=np.float64))[0])
    z = float(np.atleast_1d(np.asarray(species["ZS"], dtype=np.float64))[0])

    ddtheta_plus = np.array(grids.ddtheta_magdrift_plus, dtype=np.float64, copy=True)
    ddtheta_minus = np.array(grids.ddtheta_magdrift_minus, dtype=np.float64, copy=True)
    np.fill_diagonal(ddtheta_plus, 0.0)
    np.fill_diagonal(ddtheta_minus, 0.0)

    op = MagneticDriftThetaV3Operator(
        delta=jnp.asarray(delta, dtype=jnp.float64),
        t_hat=jnp.asarray(t_hat, dtype=jnp.float64),
        z=jnp.asarray(z, dtype=jnp.float64),
        x=grids.x,
        ddtheta_plus=jnp.asarray(ddtheta_plus),
        ddtheta_minus=jnp.asarray(ddtheta_minus),
        d_hat=jnp.asarray(geom["DHat"]),
        b_hat=jnp.asarray(geom["BHat"]),
        b_hat_sub_zeta=jnp.asarray(geom["BHat_sub_zeta"]),
        b_hat_sub_psi=jnp.asarray(geom["BHat_sub_psi"]),
        db_hat_dzeta=jnp.asarray(geom["dBHatdzeta"]),
        db_hat_dpsi_hat=jnp.asarray(geom["dBHatdpsiHat"]),
        db_hat_sub_psi_dzeta=jnp.asarray(geom["dBHat_sub_psi_dzeta"]),
        db_hat_sub_zeta_dpsi_hat=jnp.asarray(geom["dBHat_sub_zeta_dpsiHat"]),
        n_xi_for_x=grids.n_xi_for_x,
    )

    a = read_petsc_mat_aij(mat_path)

    indexing = V3Indexing(
        n_species=1,
        n_x=int(grids.x.shape[0]),
        n_theta=int(grids.theta.shape[0]),
        n_zeta=int(grids.zeta.shape[0]),
        n_xi_max=int(grids.n_xi),
        n_xi_for_x=np.asarray(grids.n_xi_for_x, dtype=int),
    )
    inv = indexing.build_inverse_f_map()
    n_f = len(inv)

    rng = np.random.default_rng(0)
    x_vec = rng.normal(size=(n_f,)).astype(np.float64)
    f = _pack_f(inv, indexing, x_vec)

    y_jax = np.asarray(apply_magnetic_drift_theta_v3_offdiag2(op, jnp.asarray(f)))
    y_jax_vec = _unpack_f(inv, y_jax)

    def keep(row, col) -> bool:
        s_r, ix_r, l_r, it_r, iz_r = row
        s_c, ix_c, l_c, it_c, iz_c = col
        if (s_c, ix_c, iz_c) != (s_r, ix_r, iz_r):
            return False
        if abs(l_c - l_r) != 2:
            return False
        return it_c != it_r  # off-diagonal-in-theta only

    y_ref = _csr_matvec_filtered(a=a, inv=inv, x_vec=x_vec, keep_entry=keep)
    # JAX (XLA) and the Python reference sum in a different order, so we allow a
    # tiny absolute tolerance above 1e-12 here.
    np.testing.assert_allclose(y_jax_vec, y_ref, rtol=0, atol=3e-12)


def test_magnetic_drift_zeta_offdiag2_offdiag_zeta_matches_fortran() -> None:
    """Parity for magnetic-drift d/dzeta term: |ΔL|=2, off-diagonal in zeta."""
    here = Path(__file__).parent
    input_path = here / "ref" / "magdrift_1species_tiny.input.namelist"
    mat_path = here / "ref" / "magdrift_1species_tiny.whichMatrix_1.petscbin"
    geom_path = here / "ref" / "magdrift_1species_tiny.geometry.npz"

    nml = read_sfincs_input(input_path)
    grids = grids_from_namelist(nml)
    geom = _load_geom_npz(geom_path)

    species = nml.group("speciesParameters")
    phys = nml.group("physicsParameters")
    delta = float(phys.get("DELTA", 0.0))
    t_hat = float(np.atleast_1d(np.asarray(species["THATS"], dtype=np.float64))[0])
    z = float(np.atleast_1d(np.asarray(species["ZS"], dtype=np.float64))[0])

    ddzeta_plus = np.array(grids.ddzeta_magdrift_plus, dtype=np.float64, copy=True)
    ddzeta_minus = np.array(grids.ddzeta_magdrift_minus, dtype=np.float64, copy=True)
    np.fill_diagonal(ddzeta_plus, 0.0)
    np.fill_diagonal(ddzeta_minus, 0.0)

    op = MagneticDriftZetaV3Operator(
        delta=jnp.asarray(delta, dtype=jnp.float64),
        t_hat=jnp.asarray(t_hat, dtype=jnp.float64),
        z=jnp.asarray(z, dtype=jnp.float64),
        x=grids.x,
        ddzeta_plus=jnp.asarray(ddzeta_plus),
        ddzeta_minus=jnp.asarray(ddzeta_minus),
        d_hat=jnp.asarray(geom["DHat"]),
        b_hat=jnp.asarray(geom["BHat"]),
        b_hat_sub_theta=jnp.asarray(geom["BHat_sub_theta"]),
        b_hat_sub_psi=jnp.asarray(geom["BHat_sub_psi"]),
        db_hat_dtheta=jnp.asarray(geom["dBHatdtheta"]),
        db_hat_dpsi_hat=jnp.asarray(geom["dBHatdpsiHat"]),
        db_hat_sub_theta_dpsi_hat=jnp.asarray(geom["dBHat_sub_theta_dpsiHat"]),
        db_hat_sub_psi_dtheta=jnp.asarray(geom["dBHat_sub_psi_dtheta"]),
        n_xi_for_x=grids.n_xi_for_x,
    )

    a = read_petsc_mat_aij(mat_path)

    indexing = V3Indexing(
        n_species=1,
        n_x=int(grids.x.shape[0]),
        n_theta=int(grids.theta.shape[0]),
        n_zeta=int(grids.zeta.shape[0]),
        n_xi_max=int(grids.n_xi),
        n_xi_for_x=np.asarray(grids.n_xi_for_x, dtype=int),
    )
    inv = indexing.build_inverse_f_map()
    n_f = len(inv)

    rng = np.random.default_rng(1)
    x_vec = rng.normal(size=(n_f,)).astype(np.float64)
    f = _pack_f(inv, indexing, x_vec)

    y_jax = np.asarray(apply_magnetic_drift_zeta_v3_offdiag2(op, jnp.asarray(f)))
    y_jax_vec = _unpack_f(inv, y_jax)

    def keep(row, col) -> bool:
        s_r, ix_r, l_r, it_r, iz_r = row
        s_c, ix_c, l_c, it_c, iz_c = col
        if (s_c, ix_c, it_c) != (s_r, ix_r, it_r):
            return False
        if abs(l_c - l_r) != 2:
            return False
        return iz_c != iz_r  # off-diagonal-in-zeta only

    y_ref = _csr_matvec_filtered(a=a, inv=inv, x_vec=x_vec, keep_entry=keep)
    # JAX (XLA) and the Python reference sum in a different order, so we allow a
    # tiny absolute tolerance above 1e-12 here.
    np.testing.assert_allclose(y_jax_vec, y_ref, rtol=0, atol=3e-12)


def test_magnetic_drift_diag_theta_zeta_offdiag2_matches_fortran() -> None:
    """Parity for the diagonal-in-(theta,zeta) part of the |ΔL|=2 magnetic-drift contributions.

    This slice includes:
      - magnetic-drift d/dtheta term using only diag(ddtheta)
      - magnetic-drift d/dzeta term using only diag(ddzeta)
      - non-standard magnetic-drift d/dxi term (diagonal in theta,zeta)
    """
    here = Path(__file__).parent
    input_path = here / "ref" / "magdrift_1species_tiny.input.namelist"
    mat_path = here / "ref" / "magdrift_1species_tiny.whichMatrix_1.petscbin"
    geom_path = here / "ref" / "magdrift_1species_tiny.geometry.npz"

    nml = read_sfincs_input(input_path)
    grids = grids_from_namelist(nml)
    geom = _load_geom_npz(geom_path)

    species = nml.group("speciesParameters")
    phys = nml.group("physicsParameters")
    delta = float(phys.get("DELTA", 0.0))
    t_hat = float(np.atleast_1d(np.asarray(species["THATS"], dtype=np.float64))[0])
    z = float(np.atleast_1d(np.asarray(species["ZS"], dtype=np.float64))[0])

    ddtheta = np.asarray(grids.ddtheta, dtype=np.float64)
    ddzeta = np.asarray(grids.ddzeta, dtype=np.float64)
    ddtheta_plus = np.asarray(grids.ddtheta_magdrift_plus, dtype=np.float64)
    ddtheta_minus = np.asarray(grids.ddtheta_magdrift_minus, dtype=np.float64)
    ddzeta_plus = np.asarray(grids.ddzeta_magdrift_plus, dtype=np.float64)
    ddzeta_minus = np.asarray(grids.ddzeta_magdrift_minus, dtype=np.float64)
    ddtheta_plus_diag = np.diag(np.diag(ddtheta_plus))
    ddtheta_minus_diag = np.diag(np.diag(ddtheta_minus))
    ddzeta_plus_diag = np.diag(np.diag(ddzeta_plus))
    ddzeta_minus_diag = np.diag(np.diag(ddzeta_minus))

    op_theta = MagneticDriftThetaV3Operator(
        delta=jnp.asarray(delta, dtype=jnp.float64),
        t_hat=jnp.asarray(t_hat, dtype=jnp.float64),
        z=jnp.asarray(z, dtype=jnp.float64),
        x=grids.x,
        ddtheta_plus=jnp.asarray(ddtheta_plus_diag),
        ddtheta_minus=jnp.asarray(ddtheta_minus_diag),
        d_hat=jnp.asarray(geom["DHat"]),
        b_hat=jnp.asarray(geom["BHat"]),
        b_hat_sub_zeta=jnp.asarray(geom["BHat_sub_zeta"]),
        b_hat_sub_psi=jnp.asarray(geom["BHat_sub_psi"]),
        db_hat_dzeta=jnp.asarray(geom["dBHatdzeta"]),
        db_hat_dpsi_hat=jnp.asarray(geom["dBHatdpsiHat"]),
        db_hat_sub_psi_dzeta=jnp.asarray(geom["dBHat_sub_psi_dzeta"]),
        db_hat_sub_zeta_dpsi_hat=jnp.asarray(geom["dBHat_sub_zeta_dpsiHat"]),
        n_xi_for_x=grids.n_xi_for_x,
    )
    op_zeta = MagneticDriftZetaV3Operator(
        delta=jnp.asarray(delta, dtype=jnp.float64),
        t_hat=jnp.asarray(t_hat, dtype=jnp.float64),
        z=jnp.asarray(z, dtype=jnp.float64),
        x=grids.x,
        ddzeta_plus=jnp.asarray(ddzeta_plus_diag),
        ddzeta_minus=jnp.asarray(ddzeta_minus_diag),
        d_hat=jnp.asarray(geom["DHat"]),
        b_hat=jnp.asarray(geom["BHat"]),
        b_hat_sub_theta=jnp.asarray(geom["BHat_sub_theta"]),
        b_hat_sub_psi=jnp.asarray(geom["BHat_sub_psi"]),
        db_hat_dtheta=jnp.asarray(geom["dBHatdtheta"]),
        db_hat_dpsi_hat=jnp.asarray(geom["dBHatdpsiHat"]),
        db_hat_sub_theta_dpsi_hat=jnp.asarray(geom["dBHat_sub_theta_dpsiHat"]),
        db_hat_sub_psi_dtheta=jnp.asarray(geom["dBHat_sub_psi_dtheta"]),
        n_xi_for_x=grids.n_xi_for_x,
    )
    op_xidot = MagneticDriftXiDotV3Operator(
        delta=jnp.asarray(delta, dtype=jnp.float64),
        t_hat=jnp.asarray(t_hat, dtype=jnp.float64),
        z=jnp.asarray(z, dtype=jnp.float64),
        x=grids.x,
        d_hat=jnp.asarray(geom["DHat"]),
        b_hat=jnp.asarray(geom["BHat"]),
        db_hat_dtheta=jnp.asarray(geom["dBHatdtheta"]),
        db_hat_dzeta=jnp.asarray(geom["dBHatdzeta"]),
        db_hat_sub_psi_dzeta=jnp.asarray(geom["dBHat_sub_psi_dzeta"]),
        db_hat_sub_zeta_dpsi_hat=jnp.asarray(geom["dBHat_sub_zeta_dpsiHat"]),
        db_hat_sub_theta_dpsi_hat=jnp.asarray(geom["dBHat_sub_theta_dpsiHat"]),
        db_hat_sub_psi_dtheta=jnp.asarray(geom["dBHat_sub_psi_dtheta"]),
        n_xi_for_x=grids.n_xi_for_x,
    )

    a = read_petsc_mat_aij(mat_path)

    indexing = V3Indexing(
        n_species=1,
        n_x=int(grids.x.shape[0]),
        n_theta=int(grids.theta.shape[0]),
        n_zeta=int(grids.zeta.shape[0]),
        n_xi_max=int(grids.n_xi),
        n_xi_for_x=np.asarray(grids.n_xi_for_x, dtype=int),
    )
    inv = indexing.build_inverse_f_map()
    n_f = len(inv)

    rng = np.random.default_rng(2)
    x_vec = rng.normal(size=(n_f,)).astype(np.float64)
    f = _pack_f(inv, indexing, x_vec)

    y_theta = np.asarray(apply_magnetic_drift_theta_v3_offdiag2(op_theta, jnp.asarray(f)))
    y_zeta = np.asarray(apply_magnetic_drift_zeta_v3_offdiag2(op_zeta, jnp.asarray(f)))
    y_xidot = np.asarray(apply_magnetic_drift_xidot_v3_offdiag2(op_xidot, jnp.asarray(f)))
    y_jax_vec = _unpack_f(inv, (y_theta + y_zeta + y_xidot))

    def keep(row, col) -> bool:
        s_r, ix_r, l_r, it_r, iz_r = row
        s_c, ix_c, l_c, it_c, iz_c = col
        if (s_c, ix_c, it_c, iz_c) != (s_r, ix_r, it_r, iz_r):
            return False
        return abs(l_c - l_r) == 2

    y_ref = _csr_matvec_filtered(a=a, inv=inv, x_vec=x_vec, keep_entry=keep)
    np.testing.assert_allclose(y_jax_vec, y_ref, rtol=0, atol=1e-12)
