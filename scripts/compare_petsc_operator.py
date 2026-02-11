#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np

from sfincs_jax.indices import V3Indexing
from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.petsc_binary import read_petsc_mat_aij, read_petsc_vec
from sfincs_jax.v3_driver import _transport_active_dof_indices
from sfincs_jax.v3_system import apply_v3_full_system_operator, full_system_operator_from_namelist


def _csr_matvec(row_ptr: np.ndarray, col_ind: np.ndarray, data: np.ndarray, x: np.ndarray) -> np.ndarray:
    y = np.zeros((row_ptr.size - 1,), dtype=np.float64)
    for i in range(row_ptr.size - 1):
        start = int(row_ptr[i])
        end = int(row_ptr[i + 1])
        if start == end:
            continue
        cols = col_ind[start:end]
        vals = data[start:end]
        y[i] = np.dot(vals, x[cols])
    return y


def _find_binary(parent: Path, stem: str) -> Path:
    matches = sorted(parent.glob(stem))
    if not matches:
        raise FileNotFoundError(f"Missing PETSc binary matching {stem} in {parent}")
    return matches[0]


def _build_fblock_l_map(op) -> tuple[np.ndarray, np.ndarray]:
    n_xi_for_x = np.asarray(op.fblock.collisionless.n_xi_for_x, dtype=int)
    idx = V3Indexing(
        n_species=int(op.n_species),
        n_x=int(op.n_x),
        n_theta=int(op.n_theta),
        n_zeta=int(op.n_zeta),
        n_xi_max=int(op.n_xi),
        n_xi_for_x=n_xi_for_x,
    )
    inv = idx.build_inverse_f_map()
    l_map = np.fromiter((it[2] for it in inv), dtype=int, count=len(inv))
    s_map = np.fromiter((it[0] for it in inv), dtype=int, count=len(inv))
    return l_map, s_map


def _build_fullshape_l_map(op) -> tuple[np.ndarray, np.ndarray]:
    s, x, l, t, z = op.fblock.f_shape
    grid = np.indices((s, x, l, t, z), dtype=np.int32)
    s_map = grid[0].ravel()
    l_map = grid[2].ravel()
    return l_map, s_map


def _summarize_by_l(label: str, diff: np.ndarray, l_map: np.ndarray, n_xi: int) -> None:
    print(f"{label}:")
    for l in range(n_xi):
        mask = l_map == l
        if not np.any(mask):
            continue
        vals = diff[mask]
        max_abs = float(np.max(np.abs(vals)))
        l2 = float(np.linalg.norm(vals))
        print(f"  L={l}: max_abs={max_abs:.6e} l2={l2:.6e}")


def _summarize_by_species(label: str, diff: np.ndarray, s_map: np.ndarray, n_species: int) -> None:
    print(f"{label}:")
    for s in range(n_species):
        mask = s_map == s
        if not np.any(mask):
            continue
        vals = diff[mask]
        max_abs = float(np.max(np.abs(vals)))
        l2 = float(np.linalg.norm(vals))
        print(f"  species={s}: max_abs={max_abs:.6e} l2={l2:.6e}")


def _summary_block(label: str, diff: np.ndarray) -> None:
    max_abs = float(np.max(np.abs(diff))) if diff.size else 0.0
    l2 = float(np.linalg.norm(diff)) if diff.size else 0.0
    print(f"{label}: max_abs={max_abs:.6e} l2={l2:.6e}")


def _vector_with_l(op, l_target: int, *, species: int | None = None) -> np.ndarray:
    x = np.zeros((op.total_size,), dtype=np.float64)
    n_xi_for_x = np.asarray(op.fblock.collisionless.n_xi_for_x, dtype=int)
    idx = V3Indexing(
        n_species=int(op.n_species),
        n_x=int(op.n_x),
        n_theta=int(op.n_theta),
        n_zeta=int(op.n_zeta),
        n_xi_max=int(op.n_xi),
        n_xi_for_x=n_xi_for_x,
    )
    species_indices = range(int(op.n_species)) if species is None else (int(species),)
    for s in species_indices:
        for ix in range(int(op.n_x)):
            if l_target >= int(n_xi_for_x[ix]):
                continue
            for itheta in range(int(op.n_theta)):
                for izeta in range(int(op.n_zeta)):
                    gi = idx.f_index(
                        i_species=s,
                        i_x=ix,
                        i_xi=l_target,
                        i_theta=itheta,
                        i_zeta=izeta,
                    )
                    x[gi] = 1.0
    return x


def _summary(
    label: str,
    a_ref: np.ndarray,
    a_jax: np.ndarray,
    op,
    *,
    active_f_size: int | None = None,
    l_map_active: np.ndarray | None = None,
    s_map_active: np.ndarray | None = None,
) -> None:
    diff = a_jax - a_ref
    f_size = int(op.f_size)
    phi1_size = int(op.phi1_size)
    extra_size = int(op.extra_size)

    _summary_block(f"{label} (full)", diff)
    if active_f_size is not None:
        if active_f_size:
            _summary_block(f"{label} (f-block)", diff[:active_f_size])
        if phi1_size:
            _summary_block(f"{label} (phi1-block)", diff[active_f_size : active_f_size + phi1_size])
        if extra_size:
            _summary_block(f"{label} (extra-block)", diff[active_f_size + phi1_size :])
        if l_map_active is not None:
            _summarize_by_l(f"{label} (f-block by L)", diff[:active_f_size], l_map_active, int(op.n_xi))
        if s_map_active is not None:
            _summarize_by_species(f"{label} (f-block by species)", diff[:active_f_size], s_map_active, int(op.n_species))
    else:
        if f_size:
            _summary_block(f"{label} (f-block)", diff[:f_size])
        if phi1_size:
            _summary_block(f"{label} (phi1-block)", diff[f_size : f_size + phi1_size])
        if extra_size:
            _summary_block(f"{label} (extra-block)", diff[f_size + phi1_size :])
        if f_size:
            l_map, s_map = _build_fblock_l_map(op)
            _summarize_by_l(f"{label} (f-block by L)", diff[:f_size], l_map, int(op.n_xi))
            _summarize_by_species(f"{label} (f-block by species)", diff[:f_size], s_map, int(op.n_species))


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare Fortran PETSc matrix vs JAX operator.")
    parser.add_argument("--input", required=True, type=Path, help="Input namelist path.")
    parser.add_argument(
        "--fortran-dir",
        required=True,
        type=Path,
        help="Directory containing Fortran PETSc binaries (sfincsBinary_iteration_000_*).",
    )
    parser.add_argument(
        "--which-matrix",
        default=3,
        type=int,
        help="whichMatrix index to compare (default: 3).",
    )
    parser.add_argument(
        "--basis-species",
        type=int,
        default=None,
        help="Restrict basis vectors to a single species index (0-based).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    input_path = args.input
    fortran_dir = args.fortran_dir
    which = int(args.which_matrix)

    mat_path = _find_binary(fortran_dir, f"sfincsBinary_iteration_*_whichMatrix_{which}")
    vec_path = None
    vec_candidates = sorted(fortran_dir.glob("sfincsBinary_iteration_*_stateVector"))
    if vec_candidates:
        vec_path = vec_candidates[0]

    print(f"Loading PETSc matrix: {mat_path}")
    mat = read_petsc_mat_aij(mat_path)

    if vec_path is not None:
        print(f"Loading PETSc state vector: {vec_path}")
        x_ref = read_petsc_vec(vec_path).values
    else:
        x_ref = np.zeros((mat.shape[1],), dtype=np.float64)

    nml = read_sfincs_input(input_path)
    op = full_system_operator_from_namelist(nml=nml)

    def matvec_fortran(x: np.ndarray) -> np.ndarray:
        return _csr_matvec(mat.row_ptr, mat.col_ind, mat.data, x)

    active_idx = None
    full_to_active = None
    active_f_size = None
    l_map_active = None
    s_map_active = None
    if mat.shape[0] != int(op.total_size):
        active_idx = _transport_active_dof_indices(op)
        full_to_active = np.zeros((int(op.total_size),), dtype=np.int32)
        full_to_active[np.asarray(active_idx, dtype=np.int32)] = np.arange(1, int(active_idx.shape[0]) + 1, dtype=np.int32)
        active_f_mask = active_idx < int(op.f_size)
        active_f_size = int(np.sum(active_f_mask))
        l_map_full, s_map_full = _build_fullshape_l_map(op)
        l_map_active = l_map_full[active_idx[:active_f_size]]
        s_map_active = s_map_full[active_idx[:active_f_size]]

    def matvec_jax(x: np.ndarray) -> np.ndarray:
        if active_idx is None:
            return np.asarray(apply_v3_full_system_operator(op, x), dtype=np.float64)
        # Expand reduced -> full, apply operator, then reduce back.
        z0 = np.zeros((1,), dtype=x.dtype)
        padded = np.concatenate([z0, x], axis=0)
        x_full = padded[full_to_active]
        y_full = np.asarray(apply_v3_full_system_operator(op, x_full), dtype=np.float64)
        return y_full[active_idx]

    print("Comparing with Fortran state vector...")
    y_fortran = matvec_fortran(x_ref)
    y_jax = matvec_jax(x_ref)
    _summary(
        "stateVector",
        y_fortran,
        y_jax,
        op,
        active_f_size=active_f_size,
        l_map_active=l_map_active,
        s_map_active=s_map_active,
    )

    basis_species = args.basis_species
    for l in (0, 1, 2):
        print(f"Comparing with L={l} unit vector...")
        x_l = _vector_with_l(op, l, species=basis_species)
        y_fortran = matvec_fortran(x_l)
        y_jax = matvec_jax(x_l)
        _summary(
            f"L{l}",
            y_fortran,
            y_jax,
            op,
            active_f_size=active_f_size,
            l_map_active=l_map_active,
            s_map_active=s_map_active,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
