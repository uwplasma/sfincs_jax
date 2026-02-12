from __future__ import annotations

import hashlib
import math
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import IO, Iterable, Iterator, List, Tuple

import numpy as np


@dataclass(frozen=True)
class BoozerBCHeader:
    n_periods: int
    psi_a_hat: float
    a_hat: float
    turkin_sign: int


@dataclass(frozen=True)
class BoozerBCSurface:
    r_n: float
    iota: float
    g_hat: float
    i_hat: float
    p_prime_hat: float
    b0_over_bbar: float
    r0: float

    m: np.ndarray  # (H,) int32
    n: np.ndarray  # (H,) int32
    parity: np.ndarray  # (H,) bool, True=cos, False=sin

    b_amp: np.ndarray  # (H,) float64
    r_amp: np.ndarray  # (H,) float64
    z_amp: np.ndarray  # (H,) float64
    dz_amp: np.ndarray  # (H,) float64


def _parse_header(first_non_cc_line: str) -> Tuple[List[int], List[float]]:
    parts = first_non_cc_line.split()
    if len(parts) < 6:
        raise ValueError(f"Unexpected .bc header line (too short): {first_non_cc_line!r}")
    header_ints = [int(x) for x in parts[:4]]
    header_reals = [float(x.replace("D", "E").replace("d", "E")) for x in parts[4:]]
    return header_ints, header_reals


def read_boozer_bc_header(*, path: str | Path, geometry_scheme: int) -> BoozerBCHeader:
    """Read the header of a SFINCS v3 `.bc` file (geometryScheme 11/12)."""
    header, _surfaces = _read_boozer_bc_all_surfaces(path=path, geometry_scheme=geometry_scheme)
    return header


def _read_boozer_bc_header_uncached(*, path: str | Path, geometry_scheme: int) -> BoozerBCHeader:
    """Uncached header reader used by the file-level cache parser."""
    if geometry_scheme not in {11, 12}:
        raise ValueError(f"geometry_scheme must be 11 or 12, got {geometry_scheme}")

    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(str(p))

    turkin_sign = 1
    with p.open("r") as f:
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Unexpected EOF while reading {str(p)!r}")
            if line.startswith("CC"):
                # In v3, this substring indicates the file was saved by Yuriy Turkin and
                # uses an additional sign convention that must be flipped.
                if geometry_scheme == 11 and "CStconfig" in line:
                    turkin_sign = -1
                continue
            try:
                header_ints, header_reals = _parse_header(line)
            except Exception:  # noqa: BLE001
                # Some files include a non-comment column-name line before the numeric header.
                continue
            else:
                n_periods = int(header_ints[3])
                psi_a_hat = float(header_reals[0]) / (2.0 * math.pi)
                a_hat = float(header_reals[1])
                break

    # Switch from left-handed to right-handed (radial,poloidal,toroidal) system.
    if geometry_scheme == 11:
        psi_a_hat = psi_a_hat * (-1.0) * float(turkin_sign)
    else:
        psi_a_hat = psi_a_hat * (-1.0)

    return BoozerBCHeader(n_periods=n_periods, psi_a_hat=psi_a_hat, a_hat=a_hat, turkin_sign=turkin_sign)


def _bc_cache_key(path: str | Path) -> tuple[str, int, int]:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(str(p))
    st = p.stat()
    return str(p), int(st.st_mtime_ns), int(st.st_size)


_BOOZER_CONTENT_CACHE_MAX = 32
_boozer_content_cache: "OrderedDict[tuple[str, int], tuple[BoozerBCHeader, tuple[BoozerBCSurface, ...]]]" = OrderedDict()


@lru_cache(maxsize=256)
def _bc_content_digest(path_resolved: str, mtime_ns: int, file_size: int) -> str:
    """Stable content digest used to reuse parsed `.bc` data across copied files.

    The hash is keyed by `(path, mtime_ns, size)` so unchanged paths are zero-copy cache hits.
    Across copied files (different paths) with identical contents, the digest collides by design.
    """
    h = hashlib.blake2b(digest_size=16)
    with Path(path_resolved).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    h.update(str(file_size).encode("ascii"))
    return h.hexdigest()


def _read_boozer_bc_all_surfaces_uncached(
    *,
    path_resolved: str,
    geometry_scheme: int,
) -> tuple[BoozerBCHeader, tuple[BoozerBCSurface, ...]]:
    p = Path(path_resolved)
    header = _read_boozer_bc_header_uncached(path=p, geometry_scheme=geometry_scheme)

    with p.open("r") as f:
        # Skip CC lines and the (possibly multi-line) header section.
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Unexpected EOF while reading {str(p)!r}")
            if line.startswith("CC"):
                continue
            try:
                _ = _parse_header(line)
            except Exception:  # noqa: BLE001
                continue
            else:
                break
        surfaces = tuple(
            _iter_surfaces(
                fh=f,
                geometry_scheme=geometry_scheme,
                n_periods=header.n_periods,
                psi_a_hat=header.psi_a_hat,
            )
        )
    return header, surfaces


def _read_boozer_bc_all_surfaces(*, path: str | Path, geometry_scheme: int) -> tuple[BoozerBCHeader, tuple[BoozerBCSurface, ...]]:
    path_resolved, mtime_ns, file_size = _bc_cache_key(path)
    geom_scheme = int(geometry_scheme)

    # Cache on file content (not absolute path) so repeated temporary/localized copies
    # of the same `.bc` file reuse the parsed surface table.
    digest = _bc_content_digest(path_resolved, mtime_ns, file_size)
    cache_key = (digest, geom_scheme)
    hit = _boozer_content_cache.get(cache_key)
    if hit is not None:
        _boozer_content_cache.move_to_end(cache_key)
        return hit

    parsed = _read_boozer_bc_all_surfaces_uncached(path_resolved=path_resolved, geometry_scheme=geom_scheme)
    _boozer_content_cache[cache_key] = parsed
    _boozer_content_cache.move_to_end(cache_key)
    if len(_boozer_content_cache) > _BOOZER_CONTENT_CACHE_MAX:
        _boozer_content_cache.popitem(last=False)
    return parsed


def _try_parse_floats(tokens: List[str], n: int) -> List[float] | None:
    if len(tokens) < n:
        return None
    out: List[float] = []
    for t in tokens[:n]:
        try:
            out.append(float(t.replace("D", "E").replace("d", "E")))
        except ValueError:
            return None
    return out


def _try_parse_ints(tokens: List[str], n: int) -> List[int] | None:
    if len(tokens) < n:
        return None
    out: List[int] = []
    for t in tokens[:n]:
        try:
            out.append(int(t))
        except ValueError:
            return None
    return out


def _iter_surfaces(
    *,
    fh: IO[str],
    geometry_scheme: int,
    n_periods: int,
    psi_a_hat: float,
) -> Iterator[BoozerBCSurface]:
    """Yield surfaces in order from a v3 `.bc` file.

    This implements the list-directed read patterns used in v3 `geometry.F90` for
    geometryScheme 11/12.
    """
    mu0 = 4.0 * math.pi * 1e-7

    # After the header line, v3 reads (and discards) one extra line.
    _ = fh.readline()

    while True:
        # Scan forward to a line that contains the marker "s" (new surface).
        line = fh.readline()
        if not line:
            return
        if "s" not in line:
            continue

        # After the "s" line, v3's formats differ slightly by scheme; for robustness we
        # just keep consuming lines until we can parse the 5-number surfHeader.
        surf_header: List[float] | None = None
        while surf_header is None:
            line = fh.readline()
            if not line:
                return
            tokens = line.split()
            surf_header = _try_parse_floats(tokens, 5)

        s, iota, g_raw, i_raw, pprime_raw = surf_header[:5]
        r_n = math.sqrt(float(s))

        # v3 uses these conventions and units:
        # - G and I are read with a minus sign due to Ampere's law in a left-handed system.
        # - They are then later flipped again by the toroidal direction sign switch.
        g_hat = -float(g_raw) * float(n_periods) / (2.0 * math.pi) * mu0
        i_hat = -float(i_raw) / (2.0 * math.pi) * mu0
        p_prime_hat = float(pprime_raw) / float(psi_a_hat) * mu0  # dp/dpsi = pPrimeHat / mu0

        # Skip units line.
        _ = fh.readline()

        m_list: List[int] = []
        n_list: List[int] = []
        parity_list: List[bool] = []
        b_list: List[float] = []
        r_list: List[float] = []
        z_list: List[float] = []
        dz_list: List[float] = []

        b0_over_bbar = 0.0
        r0 = 0.0
        found_b00 = False

        while True:
            pos = fh.tell()
            line = fh.readline()
            if not line:
                break
            if "s" in line:
                # Push back for the next outer iteration.
                fh.seek(pos)
                break
            tokens = line.split()
            ij = _try_parse_ints(tokens, 2)
            if ij is None:
                continue
            m, n = int(ij[0]), int(ij[1])
            if geometry_scheme == 11:
                vals = _try_parse_floats(tokens[2:], 4)
                if vals is None:
                    continue
                rmn, zmn, dzmn, bmn = vals
                if m == 0 and n == 0:
                    b0_over_bbar = float(bmn)
                    r0 = float(rmn)
                    found_b00 = True
                else:
                    m_list.append(m)
                    n_list.append(n)
                    parity_list.append(True)  # cosine-only in scheme 11
                    r_list.append(float(rmn))
                    z_list.append(float(zmn))
                    dz_list.append(float(dzmn))
                    b_list.append(float(bmn))
            else:
                vals8 = _try_parse_floats(tokens[2:], 8)
                if vals8 is None:
                    continue
                if m == 0 and n == 0:
                    b0_over_bbar = float(vals8[6])
                    r0 = float(vals8[0])
                    found_b00 = True
                else:
                    # v3 expands each (m,n) into two entries: cosine then sine.
                    # See geometry.F90 around the geometryScheme=12 reader.
                    rcos, rsin, zcos, zsin, dzcos, dzsin, bcos, bsin = vals8

                    # Cosine component:
                    m_list.append(m)
                    n_list.append(n)
                    parity_list.append(True)
                    r_list.append(float(rcos))
                    z_list.append(float(zsin))
                    dz_list.append(float(dzsin))
                    b_list.append(float(bcos))

                    # Sine component:
                    m_list.append(m)
                    n_list.append(n)
                    parity_list.append(False)
                    r_list.append(float(rsin))
                    z_list.append(float(zcos))
                    dz_list.append(float(dzcos))
                    b_list.append(float(bsin))

        if not found_b00:
            raise ValueError("Error: no (0,0) mode found in Boozer .bc file surface block.")

        yield BoozerBCSurface(
            r_n=float(r_n),
            iota=float(iota),
            g_hat=float(g_hat),
            i_hat=float(i_hat),
            p_prime_hat=float(p_prime_hat),
            b0_over_bbar=float(b0_over_bbar),
            r0=float(r0),
            m=np.asarray(m_list, dtype=np.int32),
            n=np.asarray(n_list, dtype=np.int32),
            parity=np.asarray(parity_list, dtype=bool),
            b_amp=np.asarray(b_list, dtype=np.float64),
            r_amp=np.asarray(r_list, dtype=np.float64),
            z_amp=np.asarray(z_list, dtype=np.float64),
            dz_amp=np.asarray(dz_list, dtype=np.float64),
        )


def read_boozer_bc_bracketing_surfaces(
    *,
    path: str | Path,
    geometry_scheme: int,
    r_n_wish: float,
) -> Tuple[BoozerBCHeader, BoozerBCSurface, BoozerBCSurface]:
    """Read and return (header, surface_old, surface_new) bracketing `rN_wish`."""
    header, surfaces = _read_boozer_bc_all_surfaces(path=path, geometry_scheme=geometry_scheme)
    p = Path(path).expanduser()

    old: BoozerBCSurface | None = None
    new: BoozerBCSurface | None = None
    for s in surfaces:
        if new is None:
            new = s
        if new.r_n < float(r_n_wish):
            old = new
            new = None
            continue
        # new.r_n >= r_n_wish: bracket found if old exists, else use new for both.
        if old is None:
            old = s
        new = s
        break

    if old is None or new is None:
        raise ValueError(f"Failed to locate surfaces bracketing rN_wish={r_n_wish} in {str(p)!r}")

    return header, old, new


def selected_r_n_from_bc(
    *,
    path: str | Path,
    geometry_scheme: int,
    r_n_wish: float,
    vmecradial_option: int = 1,
) -> float:
    """Return the effective radial location used by v3 for geometryScheme 11/12.

    Notes
    -----
    - For ``vmecradial_option=1`` (nearest surface), this returns whichever bracketing
      surface is closest to ``r_n_wish`` (matching v3 tie-breaking).
    - For interpolation options, v3 interpolates linearly in ``s = rN^2``. This function
      returns the corresponding effective ``rN``.
    """
    _header, surf_old, surf_new = read_boozer_bc_bracketing_surfaces(
        path=path,
        geometry_scheme=geometry_scheme,
        r_n_wish=float(r_n_wish),
    )
    r_old = float(surf_old.r_n)
    r_new = float(surf_new.r_n)
    if r_new == r_old:
        return r_old

    if int(vmecradial_option) == 1:
        return r_old if abs(r_old - float(r_n_wish)) < abs(r_new - float(r_n_wish)) else r_new

    s_old = r_old * r_old
    s_new = r_new * r_new
    s_wish = float(r_n_wish) * float(r_n_wish)
    radial_weight = (s_new - s_wish) / (s_new - s_old)
    s_eff = radial_weight * s_old + (1.0 - radial_weight) * s_new
    return float(math.sqrt(max(s_eff, 0.0)))
