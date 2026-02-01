from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class V3Indexing:
    """Indexing utilities matching SFINCS v3 `indices.F90`.

    Notes
    -----
    - All global indices are 0-based (PETSc convention).
    - Local indices passed to `f_index()` are 0-based.
    """

    n_species: int
    n_x: int
    n_theta: int
    n_zeta: int
    n_xi_max: int
    n_xi_for_x: np.ndarray  # shape (n_x,), int

    def __post_init__(self) -> None:
        n_xi_for_x = np.asarray(self.n_xi_for_x, dtype=int)
        if n_xi_for_x.shape != (self.n_x,):
            raise ValueError(f"n_xi_for_x must have shape ({self.n_x},), got {n_xi_for_x.shape}")
        if np.any(n_xi_for_x < 1):
            raise ValueError("n_xi_for_x entries must be >= 1")
        object.__setattr__(self, "n_xi_for_x", n_xi_for_x)

    @property
    def dke_size(self) -> int:
        return int(np.sum(self.n_xi_for_x) * self.n_theta * self.n_zeta)

    @property
    def first_index_for_x(self) -> np.ndarray:
        # Fortran uses 1-based ix; here ix is 0-based.
        out = np.zeros((self.n_x,), dtype=int)
        out[0] = 0
        for ix in range(1, self.n_x):
            out[ix] = out[ix - 1] + int(self.n_xi_for_x[ix - 1])
        return out

    def f_index(self, *, i_species: int, i_x: int, i_xi: int, i_theta: int, i_zeta: int) -> int:
        """Global index for the distribution function block (BLOCK_F)."""
        if not (0 <= i_species < self.n_species):
            raise ValueError("i_species out of range")
        if not (0 <= i_x < self.n_x):
            raise ValueError("i_x out of range")
        if not (0 <= i_xi < int(self.n_xi_for_x[i_x])):
            raise ValueError("i_xi out of range for this i_x")
        if not (0 <= i_theta < self.n_theta):
            raise ValueError("i_theta out of range")
        if not (0 <= i_zeta < self.n_zeta):
            raise ValueError("i_zeta out of range")

        first = int(self.first_index_for_x[i_x])
        return (
            i_species * self.dke_size
            + first * self.n_theta * self.n_zeta
            + i_xi * self.n_theta * self.n_zeta
            + i_theta * self.n_zeta
            + i_zeta
        )

    def build_inverse_f_map(self) -> list[tuple[int, int, int, int, int]]:
        """Return a dense map from global indices -> (species, x, xi, theta, zeta).

        Only covers the distribution-function block.
        """
        out: list[tuple[int, int, int, int, int]] = []
        for i_species in range(self.n_species):
            for i_x in range(self.n_x):
                for i_xi in range(int(self.n_xi_for_x[i_x])):
                    for i_theta in range(self.n_theta):
                        for i_zeta in range(self.n_zeta):
                            out.append((i_species, i_x, i_xi, i_theta, i_zeta))
        if len(out) != self.n_species * self.dke_size:
            raise AssertionError("Internal error building inverse map.")
        return out

