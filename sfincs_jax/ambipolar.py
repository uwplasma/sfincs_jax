from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np

from .io import read_sfincs_h5


@dataclass(frozen=True)
class AmbipolarSolveResult:
    var_name: str
    var_values: np.ndarray  # (N,)
    er_values: np.ndarray  # (N,)
    radial_currents: np.ndarray  # (N,)
    roots_var: np.ndarray  # (Nr,)
    roots_er: np.ndarray  # (Nr,)
    root_types: list[str]
    outputs_labels: list[str]
    outputs_by_run: np.ndarray  # (N, Q)
    outputs_at_roots: list[np.ndarray]  # length Q, each (Nr,)
    radius_wish: float | None = None
    radius_actual: float | None = None


def _fortran_bool_to_py(v) -> bool:
    # `sfincsOutput.h5` uses v3 integer-to-represent-true/false convention.
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    try:
        return int(np.asarray(v).reshape(())[()]) > 0
    except Exception:  # noqa: BLE001
        return bool(v)


def _as_float(v) -> float:
    return float(np.asarray(v, dtype=np.float64).reshape(()))


def radial_current_from_output(data: dict) -> float:
    """Compute the net radial current used by upstream `sfincsScanPlot_2`.

    Upstream convention:
      - if includePhi1:  j_psi = sum_s Z_s * particleFlux_vd_rHat[s, -1]
      - else:            j_psi = sum_s Z_s * particleFlux_vm_rHat[s, -1]
    """
    include_phi1 = _fortran_bool_to_py(data.get("includePhi1", False))
    z_s = np.asarray(data["Zs"], dtype=np.float64).reshape((-1,))
    if include_phi1 and "particleFlux_vd_rHat" in data:
        pf = np.asarray(data["particleFlux_vd_rHat"], dtype=np.float64)
    else:
        pf = np.asarray(data["particleFlux_vm_rHat"], dtype=np.float64)
    # Expect shape (S, NIterations) (or (S,) for some static outputs); use last iteration if needed.
    if pf.ndim == 2:
        pf_last = pf[:, -1]
    else:
        pf_last = pf.reshape((-1,))
    return float(np.sum(z_s * pf_last))


def _infer_var_name_from_scan_input(scan_input: Path) -> str:
    # `run_er_scan` writes comment lines like: "!ss ErMin = ..." and "!ss ErMax = ..."
    txt = scan_input.read_text()
    for k in ("Er", "dPhiHatdpsiHat", "dPhiHatdpsiN", "dPhiHatdrHat", "dPhiHatdrN"):
        if f"!ss {k}Min" in txt or f"!ss {k}Max" in txt:
            return k
    # Fallback: assume Er.
    return "Er"


def _scanplot2_labels(*, n_species: int, include_phi1: bool) -> list[str]:
    if n_species == 1:
        if include_phi1:
            return [
                "FSABFlow",
                "particleFlux_vm_rHat",
                "particleFlux_vd_rHat",
                "heatFlux_vm_rHat",
                "heatFlux_withoutPhi1_rHat",
                "FSABjHat",
                "radial current",
            ]
        return [
            "FSABFlow",
            "particleFlux_vm_rHat",
            "heatFlux_vm_rHat",
            "source 1",
            "source 2",
            "FSABjHat",
            "radial current",
        ]

    labels: list[str] = []
    for i in range(1, n_species + 1):
        labels.append(f"FSABFlow (species {i})")
        labels.append(f"particleFlux rHat (species {i})")
        labels.append(f"heatFlux rHat (species {i})")
    labels.append("FSABjHat")
    labels.append("radial current")
    return labels


def _scanplot2_outputs_for_run(data: dict) -> np.ndarray:
    n_species = int(np.asarray(data["Nspecies"]))
    include_phi1 = _fortran_bool_to_py(data.get("includePhi1", False))
    # Many datasets are (S, NIterations), some are (NIterations,), some are scalars.
    def _last(v):
        arr = np.asarray(v, dtype=np.float64)
        if arr.ndim == 0:
            return arr.reshape((1,))
        if arr.ndim == 1:
            return arr[-1:]
        if arr.ndim == 2:
            return arr[:, -1]
        raise ValueError(f"Unexpected dataset rank {arr.ndim}")

    out: list[float] = []
    if n_species == 1:
        fsab_flow = float(_last(data["FSABFlow"])[0])
        pf_vm = float(_last(data["particleFlux_vm_rHat"])[0])
        hf_vm = float(_last(data["heatFlux_vm_rHat"])[0])
        fsab_j = float(_last(data["FSABjHat"])[0]) if "FSABjHat" in data else float(_as_float(data.get("FSABjHat", 0.0)))
        if include_phi1 and "particleFlux_vd_rHat" in data:
            pf_vd = float(_last(data["particleFlux_vd_rHat"])[0])
            hf_wo = float(_last(data["heatFlux_withoutPhi1_rHat"])[0])
            out += [fsab_flow, pf_vm, pf_vd, hf_vm, hf_wo, fsab_j, radial_current_from_output(data)]
        else:
            src = np.asarray(data.get("sources", np.zeros((2,), dtype=np.float64)), dtype=np.float64)
            # sources shape varies across fixtures; take last iteration if present.
            if src.ndim == 1:
                s0, s1 = (float(src[0]), float(src[1])) if src.size >= 2 else (float(src[0]), 0.0)
            elif src.ndim == 2:
                s0 = float(src[0, -1])
                s1 = float(src[1, -1])
            else:
                s0 = s1 = 0.0
            out += [fsab_flow, pf_vm, hf_vm, s0, s1, fsab_j, radial_current_from_output(data)]
        return np.asarray(out, dtype=np.float64)

    # Multi-species:
    fsab_flow = np.asarray(_last(data["FSABFlow"]), dtype=np.float64).reshape((n_species,))
    if include_phi1 and "particleFlux_vd_rHat" in data:
        pf = np.asarray(_last(data["particleFlux_vd_rHat"]), dtype=np.float64).reshape((n_species,))
        hf = np.asarray(_last(data.get("heatFlux_vd_rHat", data.get("heatFlux_vm_rHat"))), dtype=np.float64).reshape((n_species,))
    else:
        pf = np.asarray(_last(data["particleFlux_vm_rHat"]), dtype=np.float64).reshape((n_species,))
        hf = np.asarray(_last(data["heatFlux_vm_rHat"]), dtype=np.float64).reshape((n_species,))
    for i in range(n_species):
        out += [float(fsab_flow[i]), float(pf[i]), float(hf[i])]
    fsab_j = float(_last(data["FSABjHat"])[0]) if "FSABjHat" in data and np.asarray(data["FSABjHat"]).ndim == 1 else float(_as_float(data.get("FSABjHat", 0.0)))
    out += [fsab_j, radial_current_from_output(data)]
    return np.asarray(out, dtype=np.float64)


def solve_ambipolar_from_scan_dir(
    *,
    scan_dir: Path,
    write_pickle: bool = True,
    write_json: bool = True,
    n_fine: int = 500,
) -> AmbipolarSolveResult:
    """Compute ambipolar roots from an existing Er scan directory.

    The directory must look like an upstream scanType=2 directory:
    - `scan_dir/input.namelist` exists and includes the `!ss` metadata written by `run_er_scan`.
    - subdirectories contain `sfincsOutput.h5`.

    This routine writes `ambipolarSolutions.dat` in a format compatible with upstream `sfincsScanPlot_5`.
    """
    scan_dir = Path(scan_dir).resolve()
    scan_input = scan_dir / "input.namelist"
    if not scan_input.exists():
        raise FileNotFoundError(f"Missing scan input.namelist: {scan_input}")

    var_name = _infer_var_name_from_scan_input(scan_input)

    run_dirs = sorted([p for p in scan_dir.iterdir() if p.is_dir() and p.name.startswith(var_name)])
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found under {scan_dir} with prefix {var_name}")

    records: list[tuple[float, float, float, dict]] = []
    for rd in run_dirs:
        out_h5 = rd / "sfincsOutput.h5"
        if not out_h5.exists():
            continue
        d = read_sfincs_h5(out_h5)
        if int(np.asarray(d.get("RHSMode", 1))) != 1:
            continue
        v = _as_float(d[var_name]) if var_name in d else float("nan")
        er = _as_float(d["Er"]) if "Er" in d else v
        jpsi = radial_current_from_output(d)
        records.append((v, er, jpsi, d))

    if len(records) < 2:
        raise RuntimeError("Need at least 2 completed runs to attempt an ambipolarity solve.")

    # Sort by scan variable (as upstream does).
    records.sort(key=lambda t: t[0])
    var_vals = np.asarray([r[0] for r in records], dtype=np.float64)
    er_vals = np.asarray([r[1] for r in records], dtype=np.float64)
    jpsi_vals = np.asarray([r[2] for r in records], dtype=np.float64)

    # Determine which flux variant was used (for labels), based on includePhi1 in the last run.
    include_phi1 = _fortran_bool_to_py(records[-1][3].get("includePhi1", False))
    n_species = int(np.asarray(records[-1][3]["Nspecies"]))
    labels = _scanplot2_labels(n_species=n_species, include_phi1=include_phi1)
    outputs_by_run = np.stack([_scanplot2_outputs_for_run(r[3]) for r in records], axis=0)  # (N, Q)

    # Interpolator choice:
    from scipy.interpolate import PchipInterpolator, interp1d  # noqa: PLC0415
    from scipy.optimize import brentq  # noqa: PLC0415

    if var_vals.size < 3:
        interpolator = interp1d(var_vals, jpsi_vals, kind="linear")
        quantity_interp = lambda y: interp1d(var_vals, y, kind="linear")
    else:
        try:
            interpolator = PchipInterpolator(var_vals, jpsi_vals)
            quantity_interp = lambda y: PchipInterpolator(var_vals, y)
        except Exception:
            interpolator = interp1d(var_vals, jpsi_vals, kind="linear")
            quantity_interp = lambda y: interp1d(var_vals, y, kind="linear")

    roots_var: list[float] = []
    if float(np.max(jpsi_vals)) > 0.0 and float(np.min(jpsi_vals)) < 0.0:
        fine = np.linspace(float(np.min(var_vals)), float(np.max(var_vals)), num=int(n_fine), dtype=np.float64)
        j_fine = np.asarray(interpolator(fine), dtype=np.float64)
        pos = j_fine > 0
        flips = pos[:-1] != pos[1:]
        for idx, v in enumerate(flips):
            if not v:
                continue
            a = float(fine[idx])
            b = float(fine[idx + 1])
            try:
                roots_var.append(float(brentq(interpolator, a, b)))
            except Exception:
                continue

    roots_var_arr = np.sort(np.asarray(roots_var, dtype=np.float64))
    # Conversion to Er: Er = var * (Er/var).
    conversion_to_er = 1.0
    if var_name != "Er":
        # Use the ratio from the last run (they are constant for a given setup).
        v_last = float(var_vals[-1])
        er_last = float(er_vals[-1])
        conversion_to_er = float(er_last / v_last)
    roots_er_arr = roots_var_arr * float(conversion_to_er)

    # Root typing matches upstream heuristic.
    if roots_er_arr.size == 1:
        root_types = ["electron" if float(roots_er_arr[0]) > 0.0 else "ion"]
    elif roots_er_arr.size == 3:
        root_types = ["ion", "unstable", "electron"]
    else:
        root_types = ["unknown"] * int(roots_er_arr.size)

    outputs_at_roots: list[np.ndarray] = []
    if roots_var_arr.size:
        for q in range(outputs_by_run.shape[1]):
            qi = quantity_interp(outputs_by_run[:, q])
            outputs_at_roots.append(np.asarray(qi(roots_var_arr), dtype=np.float64))
    else:
        outputs_at_roots = [np.zeros((0,), dtype=np.float64) for _ in range(outputs_by_run.shape[1])]

    # Estimate radii from the last run:
    last = records[-1][3]
    radius_wish = None
    radius_actual = None
    for k in ("rN", "rHat", "psiN", "psiHat"):
        if k in last:
            radius_actual = _as_float(last[k])
            break
    # `*_wish` is not always written; keep None if unavailable.

    result = AmbipolarSolveResult(
        var_name=var_name,
        var_values=var_vals,
        er_values=er_vals,
        radial_currents=jpsi_vals,
        roots_var=roots_var_arr,
        roots_er=roots_er_arr,
        root_types=root_types,
        outputs_labels=labels,
        outputs_by_run=outputs_by_run,
        outputs_at_roots=outputs_at_roots,
        radius_wish=radius_wish,
        radius_actual=radius_actual,
    )

    if write_pickle:
        # Minimal pickle format compatible with upstream `sfincsScanPlot_5`.
        payload = {
            "numQuantities": int(outputs_by_run.shape[1]),
            "ylabels": labels,
            "roots": roots_var_arr,
            "root_types": root_types,
            "outputs_ambipolar": outputs_at_roots,
            "radius_wish": radius_wish,
            "radius_actual": radius_actual,
            "nHats": np.asarray(last.get("nHats", np.zeros((0,), dtype=np.float64)), dtype=np.float64).reshape((-1,)),
            "THats": np.asarray(last.get("THats", np.zeros((0,), dtype=np.float64)), dtype=np.float64).reshape((-1,)),
        }
        (scan_dir / "ambipolarSolutions.dat").write_bytes(pickle.dumps(payload))

    if write_json:
        import json  # noqa: PLC0415

        (scan_dir / "ambipolarSolutions.json").write_text(
            json.dumps(
                {
                    "var_name": var_name,
                    "roots_var": roots_var_arr.tolist(),
                    "roots_er": roots_er_arr.tolist(),
                    "root_types": root_types,
                    "radial_currents": jpsi_vals.tolist(),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    return result
