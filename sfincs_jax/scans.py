from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import os
from pathlib import Path
import re

import numpy as np

from .io import localize_equilibrium_file_in_place, write_sfincs_jax_output_h5
from .namelist import Namelist, read_sfincs_input


EmitFn = Callable[[int, str], None]


@dataclass(frozen=True)
class ScanResult:
    scan_dir: Path
    run_dirs: tuple[Path, ...]
    outputs: tuple[Path, ...]
    variable: str
    values: tuple[float, ...]


def _er_scan_var_name(*, nml: Namelist) -> str:
    geom = nml.group("geometryParameters")
    v = geom.get("INPUTRADIALCOORDINATEFORGRADIENTS", None)
    if v is None:
        # v3 default in many examples is 4 (Er).
        igrad = 4
    else:
        igrad = int(v if not isinstance(v, list) else v[0])

    if igrad == 0:
        return "dPhiHatdpsiHat"
    if igrad == 1:
        return "dPhiHatdpsiN"
    if igrad == 2:
        return "dPhiHatdrHat"
    if igrad == 3:
        return "dPhiHatdrN"
    if igrad == 4:
        return "Er"
    raise ValueError(f"Invalid inputRadialCoordinateForGradients={igrad}")


def _patch_scalar_in_group(*, txt: str, group: str, key: str, value: float) -> str:
    """Patch a scalar assignment inside a Fortran namelist group.

    If the key is not present, it is appended before the group terminator `/`.
    """
    g = str(group)
    k = str(key)

    start = re.search(rf"(?im)^\s*&{re.escape(g)}\s*$", txt)
    if start is None:
        raise ValueError(f"Missing namelist group &{g}")

    # Find the group terminator "/" after the group start.
    end = re.search(r"(?m)^\s*/\s*$", txt[start.end() :])
    if end is None:
        raise ValueError(f"Missing '/' terminator for &{g}")
    end_pos = start.end() + end.start()
    group_txt = txt[start.end() : end_pos]

    # Replace if present (handle quoted/unquoted, spacing, and fortran D exponents).
    pat = re.compile(rf"(?im)^[ \t]*{re.escape(k)}[ \t]*=[ \t]*([^!\n\r]+)[ \t]*$")
    m = pat.search(group_txt)
    new_line = f"  {k} = {value:.16g}"
    if m is not None:
        group_txt2 = group_txt.replace(m.group(0), new_line)
    else:
        # Insert just before the "/" line.
        if not group_txt.endswith("\n"):
            group_txt = group_txt + "\n"
        group_txt2 = group_txt + new_line + "\n"

    return txt[: start.end()] + group_txt2 + txt[end_pos:]


def run_er_scan(
    *,
    input_namelist: Path,
    out_dir: Path,
    values: Sequence[float],
    compute_transport_matrix: bool = False,
    compute_solution: bool = False,
    emit: EmitFn | None = None,
) -> ScanResult:
    """Run an E_r (or dPhiHatd*) scan using `sfincs_jax` and write `sfincsOutput.h5` in each run dir.

    The directory naming convention follows upstream `utils/sfincsScan_2`:
    - the varied variable name is determined by `inputRadialCoordinateForGradients`
    - directories are named like `Er{:.4g}` / `dPhiHatdpsiHat{:.4g}` etc.
    """
    input_namelist = Path(input_namelist).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    template_txt = input_namelist.read_text()

    nml0 = read_sfincs_input(input_namelist)
    var = _er_scan_var_name(nml=nml0)
    # Use a deterministic order that matches upstream `sfincsScan_2`, which generates values
    # via linspace(max, min, N).
    vals = sorted([float(v) for v in values], reverse=True)
    if emit is not None:
        emit(0, f"scan-er: input={input_namelist}")
        emit(0, f"scan-er: out_dir={out_dir}")
        emit(
            0,
            f"scan-er: variable={var} n={len(vals)} compute_solution={bool(compute_solution)} compute_transport_matrix={bool(compute_transport_matrix)}",
        )

    # Write a scan-style `input.namelist` in the scan directory so vendored upstream
    # `utils/sfincsScanPlot_*` scripts can infer the directory list.
    scan_txt = template_txt
    if not scan_txt.endswith("\n"):
        scan_txt += "\n"
    scan_txt += f"!ss NErs = {len(vals)}\n"
    scan_txt += f"!ss {var}Min = {min(vals):.16g}\n"
    scan_txt += f"!ss {var}Max = {max(vals):.16g}\n"
    (out_dir / "input.namelist").write_text(scan_txt)

    run_dirs: list[Path] = []
    outputs: list[Path] = []
    scan_recycle_env = os.environ.get("SFINCS_JAX_SCAN_RECYCLE", "").strip().lower()
    scan_recycle_enabled = scan_recycle_env in {"1", "true", "yes", "on"}
    prev_run_dir: Path | None = None
    for i, v in enumerate(vals, start=1):
        run_dir = out_dir / f"{var}{v:.4g}"
        run_dir.mkdir(parents=True, exist_ok=True)
        run_dirs.append(run_dir)
        if emit is not None:
            emit(0, f"scan-er: [{i}/{len(vals)}] {run_dir.name} {var}={v:.16g}")

        # Patch input.namelist for this run:
        txt2 = _patch_scalar_in_group(txt=template_txt, group="physicsParameters", key=var, value=float(v))
        w_input = run_dir / "input.namelist"
        w_input.write_text(txt2)

        # Ensure equilibriumFile is runnable from the run dir:
        localize_equilibrium_file_in_place(input_namelist=w_input, overwrite=False)

        if scan_recycle_enabled:
            state_out = run_dir / "sfincs_jax_state.npz"
            os.environ["SFINCS_JAX_STATE_OUT"] = str(state_out)
            if prev_run_dir is None:
                os.environ.pop("SFINCS_JAX_STATE_IN", None)
            else:
                prev_state = prev_run_dir / "sfincs_jax_state.npz"
                if prev_state.exists():
                    os.environ["SFINCS_JAX_STATE_IN"] = str(prev_state)
                else:
                    os.environ.pop("SFINCS_JAX_STATE_IN", None)

        out_path = run_dir / "sfincsOutput.h5"
        write_sfincs_jax_output_h5(
            input_namelist=w_input,
            output_path=out_path,
            overwrite=True,
            compute_transport_matrix=bool(compute_transport_matrix),
            compute_solution=bool(compute_solution),
            emit=emit,
        )
        outputs.append(out_path)
        prev_run_dir = run_dir

    return ScanResult(
        scan_dir=out_dir,
        run_dirs=tuple(run_dirs),
        outputs=tuple(outputs),
        variable=var,
        values=tuple(vals),
    )


def linspace_including_endpoints(min_value: float, max_value: float, n: int) -> np.ndarray:
    if int(n) < 2:
        raise ValueError("n must be >= 2")
    return np.linspace(float(min_value), float(max_value), int(n), dtype=np.float64)
