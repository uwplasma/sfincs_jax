#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import time
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
UTILS = REPO_ROOT / "utils"
FIG_DIR = REPO_ROOT / "docs" / "_static" / "figures" / "utils"
WORK_DIR = REPO_ROOT / "examples" / "utils" / "output"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="generate_utils_gallery",
        description="Run a compact gallery of sfincs_jax utils figures.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use reduced resolutions and fewer scan points for quick runs.",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=None,
        help="Per-step timeout in seconds (applied to each sfincsScan/sfincsPlot call).",
    )
    return parser.parse_args()


def _run(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    stdin: str | None = None,
    label: str | None = None,
    timeout_s: float | None = None,
) -> None:
    label_text = f"[{label}] " if label else ""
    print(f"{label_text}cwd={cwd}")
    print(f"{label_text}cmd={' '.join(cmd)}")
    print(f"{label_text}start={time.strftime('%Y-%m-%d %H:%M:%S')}")
    sys.stdout.flush()
    e = os.environ.copy()
    e["MPLBACKEND"] = "Agg"
    if env:
        e.update(env)
    try:
        subprocess.run(cmd, cwd=str(cwd), input=stdin, text=True, check=True, env=e, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        print(f"{label_text}timeout after {timeout_s}s")
        sys.stdout.flush()
        raise
    print(f"{label_text}done={time.strftime('%Y-%m-%d %H:%M:%S')}")
    sys.stdout.flush()


def _stage(msg: str) -> None:
    print(f"==> {msg}")
    sys.stdout.flush()


def _inject_group(text: str, group: str, lines: list[str]) -> str:
    out: list[str] = []
    inserted = False
    for line in text.splitlines():
        out.append(line)
        if line.strip().lower().startswith(f"&{group.lower()}"):
            out.extend(lines)
            inserted = True
    if not inserted:
        out.append(f"&{group}")
        out.extend(lines)
        out.append("/")
    return "\n".join(out) + "\n"


def _prepare_input(
    base: Path,
    dest: Path,
    *,
    extra_lines: list[str] | None = None,
    resolution_overrides: list[str] | None = None,
    group_overrides: dict[str, list[str]] | None = None,
) -> None:
    text = base.read_text()
    # Keep resolutions tiny for speed.
    overrides = resolution_overrides or [
        "  Ntheta = 5",
        "  Nzeta = 5",
        "  Nxi = 4",
        "  NL = 3",
        "  Nx = 4",
    ]
    if resolution_overrides:
        override_vars = {
            line.split("=", 1)[0].strip().lower()
            for line in overrides
            if "=" in line
        }
        in_resolution = False
        filtered: list[str] = []
        for line in text.splitlines():
            stripped = line.strip().lower()
            if stripped.startswith("&resolutionparameters"):
                in_resolution = True
                filtered.append(line)
                continue
            if in_resolution and stripped.startswith("/"):
                in_resolution = False
                filtered.append(line)
                continue
            if in_resolution:
                if any(stripped.startswith(var) for var in override_vars):
                    continue
            filtered.append(line)
        text = "\n".join(filtered) + "\n"
    text = _inject_group(
        text,
        "resolutionParameters",
        overrides,
    )
    if group_overrides:
        for group, lines in group_overrides.items():
            text = _inject_group(text, group, lines)
    if extra_lines:
        text += "\n" + "\n".join(extra_lines) + "\n"
    dest.write_text(text)


def _copy_profiles(dest: Path) -> None:
    dest.write_text(
        "\n".join(
            [
                "# profiles for radial scans (rN coordinate)",
                "3",
                "0.20  3  -1e-3  1e-3  1.0  1.0",
                "0.30  3  -1e-3  1e-3  1.0  1.0",
                "0.40  3  -1e-3  1e-3  1.0  1.0",
                "0.50  3  -1e-3  1e-3  1.0  1.0",
            ]
        )
        + "\n"
    )


def _copy_equilibrium(dest_dir: Path, filename: str = "w7x_standardConfig.bc") -> None:
    src = REPO_ROOT / "sfincs_jax" / "data" / "equilibria" / filename
    if not src.exists():
        src = REPO_ROOT / "tests" / "ref" / filename
    shutil.copyfile(src, dest_dir / filename)


def main() -> None:
    args = _parse_args()
    fast = bool(args.fast)
    timeout_s = args.timeout_s
    base_res = [
        "  Ntheta = 7",
        "  Nzeta = 5",
        "  Nxi = 4",
        "  NL = 3",
        "  Nx = 4",
    ]
    fast_res = [
        "  Ntheta = 5",
        "  Nzeta = 3",
        "  Nxi = 3",
        "  NL = 2",
        "  Nx = 2",
        "  solverTolerance = 1e-4",
    ]
    res_overrides = fast_res if fast else base_res

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    base_rhs1 = REPO_ROOT / "tests" / "ref" / "pas_1species_PAS_noEr_tiny.input.namelist"
    base_scheme11 = REPO_ROOT / "tests" / "ref" / "output_scheme11_1species_tiny.input.namelist"
    base_mono = REPO_ROOT / "tests" / "ref" / "monoenergetic_PAS_tiny_scheme11.input.namelist"
    base_bootstrap = REPO_ROOT / "tests" / "reduced_inputs" / "geometryScheme4_2species_noEr.input.namelist"

    _stage("Starting utils gallery generation")

    # sfincsPlot
    _stage("Running sfincsPlot")
    work = WORK_DIR / "sfincsPlot"
    work.mkdir(parents=True, exist_ok=True)
    _prepare_input(base_rhs1, work / "input.namelist", resolution_overrides=res_overrides)
    _run(
        [sys.executable, str(UTILS / "sfincsPlot"), "--save-prefix", str(FIG_DIR / "sfincsPlot")],
        cwd=work,
        label="sfincsPlot",
        timeout_s=timeout_s,
    )

    # sfincsPlotF (enable export_delta_f)
    _stage("Running sfincsPlotF")
    work = WORK_DIR / "sfincsPlotF"
    work.mkdir(parents=True, exist_ok=True)
    extra = [
        "&export_f",
        "  export_delta_f = .true.",
        "  export_full_f = .false.",
        "  export_f_theta_option = 0",
        "  export_f_zeta_option = 0",
        "  export_f_x_option = 0",
        "  export_f_xi_option = 0",
        "/",
    ]
    _prepare_input(base_rhs1, work / "input.namelist", extra_lines=extra, resolution_overrides=res_overrides)
    _run(
        [sys.executable, str(UTILS / "sfincsPlotF"), "--save", str(FIG_DIR / "sfincsPlotF.png")],
        cwd=work,
        label="sfincsPlotF",
        timeout_s=timeout_s,
    )

    # scanType=1 (convergence)
    _stage("Running scanType=1 (convergence)")
    work = WORK_DIR / "scan_type1"
    work.mkdir(parents=True, exist_ok=True)
    extra = [
        "!ss scanType = 1",
        "!ss NthetaMinFactor = 1.0",
        "!ss NthetaMaxFactor = 2.0",
        f\"!ss NthetaNumRuns = {2 if fast else 3}\",
    ]
    _prepare_input(
        base_rhs1,
        work / "input.namelist",
        extra_lines=extra,
        resolution_overrides=res_overrides,
    )
    _run([sys.executable, str(UTILS / "sfincsScan"), "--yes"], cwd=work, label="scan1-run", timeout_s=timeout_s)
    _run(
        [sys.executable, str(UTILS / "sfincsScanPlot_1"), "--save", str(FIG_DIR / "sfincsScanPlot_1.png")],
        cwd=work,
        label="scan1-plot",
        timeout_s=timeout_s,
    )

    # scanType=2 (Er scan)
    _stage("Running scanType=2 (Er scan)")
    work = WORK_DIR / "scan_type2"
    work.mkdir(parents=True, exist_ok=True)
    extra = [
        "!ss scanType = 2",
        f\"!ss NErs = {2 if fast else 3}\",
        "!ss ErMin = -1e-3",
        "!ss ErMax = 1e-3",
    ]
    _prepare_input(base_rhs1, work / "input.namelist", extra_lines=extra, resolution_overrides=res_overrides)
    _run([sys.executable, str(UTILS / "sfincsScan"), "--yes"], cwd=work, label="scan2-run", timeout_s=timeout_s)
    _run(
        [sys.executable, str(UTILS / "sfincsScanPlot_2"), "--save", str(FIG_DIR / "sfincsScanPlot_2.png")],
        cwd=work,
        label="scan2-plot",
        timeout_s=timeout_s,
    )

    # scanType=3 (parameter scan)
    _stage("Running scanType=3 (parameter scan)")
    work = WORK_DIR / "scan_type3"
    work.mkdir(parents=True, exist_ok=True)
    extra = [
        "!ss scanType = 3",
        "!ss scanVariable = nu_n",
        "!ss scanVariableMin = 1e-3",
        "!ss scanVariableMax = 2e-3",
        f\"!ss scanVariableN = {2 if fast else 3}\",
        "!ss scanVariableScale = lin",
    ]
    _prepare_input(base_rhs1, work / "input.namelist", extra_lines=extra, resolution_overrides=res_overrides)
    _run([sys.executable, str(UTILS / "sfincsScan"), "--yes"], cwd=work, label="scan3-run", timeout_s=timeout_s)
    _run(
        [sys.executable, str(UTILS / "sfincsScanPlot_3"), "--save", str(FIG_DIR / "sfincsScanPlot_3.png")],
        cwd=work,
        label="scan3-plot",
        timeout_s=timeout_s,
    )

    # scanType=4 (radial scan)
    _stage("Running scanType=4 (radial scan)")
    work = WORK_DIR / "scan_type4"
    work.mkdir(parents=True, exist_ok=True)
    extra = [
        "!ss scanType = 4",
        "!ss profilesScheme = 1",
        f\"!ss Nradius = {2 if fast else 3}\",
        "!ss rN_min = 0.2",
        "!ss rN_max = 0.4",
    ]
    _prepare_input(base_scheme11, work / "input.namelist", extra_lines=extra, resolution_overrides=res_overrides)
    _copy_equilibrium(work)
    _copy_profiles(work / "profiles")
    _run([sys.executable, str(UTILS / "sfincsScan"), "--yes"], cwd=work, label="scan4-run", timeout_s=timeout_s)
    _run(
        [sys.executable, str(UTILS / "sfincsScanPlot_4"), "--save", str(FIG_DIR / "sfincsScanPlot_4.png")],
        cwd=work,
        label="scan4-plot",
        timeout_s=timeout_s,
    )

    # scanType=5 (radial + Er)
    _stage("Running scanType=5 (radial + Er)")
    work = WORK_DIR / "scan_type5"
    work.mkdir(parents=True, exist_ok=True)
    extra = [
        "!ss scanType = 5",
        "!ss profilesScheme = 1",
        f\"!ss Nradius = {2 if fast else 3}\",
        "!ss rN_min = 0.2",
        "!ss rN_max = 0.4",
    ]
    _prepare_input(base_scheme11, work / "input.namelist", extra_lines=extra, resolution_overrides=res_overrides)
    _copy_equilibrium(work)
    _copy_profiles(work / "profiles")
    _run([sys.executable, str(UTILS / "sfincsScan"), "--yes"], cwd=work, label="scan5-run", timeout_s=timeout_s)
    _run(
        [sys.executable, str(UTILS / "sfincsScanPlot_5"), "--save", str(FIG_DIR / "sfincsScanPlot_5.png")],
        cwd=work,
        label="scan5-plot",
        timeout_s=timeout_s,
    )

    # scanType=21 (runspec)
    _stage("Running scanType=21 (runspec)")
    work = WORK_DIR / "scan_type21"
    work.mkdir(parents=True, exist_ok=True)
    extra = [
        "!ss scanType = 21",
        "!ss runSpecFile = 'runspec.dat'",
    ]
    _prepare_input(base_rhs1, work / "input.namelist", extra_lines=extra, resolution_overrides=res_overrides)
    (work / "runspec.dat").write_text("! nu_n\n1.0e-3\n2.0e-3\n")
    _run([sys.executable, str(UTILS / "sfincsScan"), "--yes"], cwd=work, label="scan21-run", timeout_s=timeout_s)
    _run(
        [
            sys.executable,
            str(UTILS / "sfincsScanPlot_21"),
            "--save",
            str(FIG_DIR / "sfincsScanPlot_21.png"),
            "--x",
            "nu_n",
            "--y",
            "particleFlux_vm_psiHat_1",
            "--xscale",
            "linear",
            "--yscale",
            "linear",
        ],
        cwd=work,
        label="scan21-plot",
        timeout_s=timeout_s,
    )

    # scanType=2 combined (reuse scan_type2 twice)
    _stage("Combining scanType=2 plots")
    work = WORK_DIR / "scan_type2b"
    work.mkdir(parents=True, exist_ok=True)
    shutil.copytree(WORK_DIR / "scan_type2", work, dirs_exist_ok=True)
    _run(
        [
            sys.executable,
            str(UTILS / "sfincsScanPlot_combine"),
            str(WORK_DIR / "scan_type2"),
            str(WORK_DIR / "scan_type2b"),
            "--save",
            str(FIG_DIR / "sfincsScanPlot_combine.png"),
        ],
        cwd=WORK_DIR,
        label="scan2-combine",
        timeout_s=timeout_s,
    )

    # ModelTest_AI (uses sfincsOutput.h5 from sfincsPlotF run)
    _stage("Running ModelTest_AI")
    model_dir = WORK_DIR / "sfincsPlotF"
    _run(
        [
            sys.executable,
            str(UTILS / "ModelTest_AI" / "testModel.py"),
            str(model_dir) + "/",
            "epsilont",
            "data0",
            "--save",
            str(FIG_DIR / "ModelTest_AI.png"),
        ],
        cwd=model_dir,
        label="modeltest",
        timeout_s=timeout_s,
    )

    # Monoenergetic transport coefficients vs collisionality for multiple EStar
    _stage("Running monoenergetic collisionality scans")
    mono_dirs = []
    for idx, estar in enumerate([-0.2, 0.0, 0.2]):
        work = WORK_DIR / f"mono_collisionality_E{idx}"
        work.mkdir(parents=True, exist_ok=True)
        extra = [
            "!ss scanType = 3",
            "!ss scanVariable = nuPrime",
            "!ss scanVariableMin = 0.1",
            "!ss scanVariableMax = 1.0",
            "!ss scanVariableN = 3",
            "!ss scanVariableScale = log",
        ]
        _prepare_input(
            base_mono,
            work / "input.namelist",
            extra_lines=extra,
            group_overrides={
                "physicsParameters": [f"  EStar = {estar}"],
            },
            resolution_overrides=[
                "  Ntheta = 5",
                "  Nzeta = 5",
                "  Nxi = 4",
                "  NL = 3",
                "  Nx = 1",
            ],
        )
        _copy_equilibrium(work)
        _run(
            [sys.executable, str(UTILS / "sfincsScan"), "--yes"],
            cwd=work,
            label=f\"mono-scan{idx}-run\",
            timeout_s=timeout_s,
        )
        _run(
            [sys.executable, str(UTILS / "sfincsScanPlot_3"), "--save", str(FIG_DIR / f"mono_collisionality_E{idx}.png")],
            cwd=work,
            label=f"mono-scan{idx}-plot",
            timeout_s=timeout_s,
        )
        mono_dirs.append(work)
    _run(
        [
            sys.executable,
            str(UTILS / "sfincsScanPlot_combine"),
            *[str(d) for d in mono_dirs],
            "--save",
            str(FIG_DIR / "monoenergetic_transport_coeffs.png"),
        ],
        cwd=WORK_DIR,
        label="mono-combine",
        timeout_s=timeout_s,
    )

    # Bootstrap current vs collisionality (2-species PAS, geometryScheme=4)
    _stage("Running bootstrap current vs collisionality scan")
    work = WORK_DIR / "bootstrap_collisionality"
    work.mkdir(parents=True, exist_ok=True)
    extra = [
        "!ss scanType = 3",
        "!ss scanVariable = nu_n",
        "!ss scanVariableMin = 1e-3",
        "!ss scanVariableMax = 1e-2",
        "!ss scanVariableN = 3",
        "!ss scanVariableScale = log",
    ]
    _prepare_input(
        base_bootstrap,
        work / "input.namelist",
        extra_lines=extra,
        group_overrides={
            "export_f": [
                "  export_full_f = .false.",
                "  export_delta_f = .false.",
            ],
        },
        resolution_overrides=res_overrides,
    )
    _run([sys.executable, str(UTILS / "sfincsScan"), "--yes"], cwd=work, label="bootstrap-run", timeout_s=timeout_s)
    _run(
        [
            sys.executable,
            str(UTILS / "sfincsScanPlot_21"),
            "--save",
            str(FIG_DIR / "bootstrap_current_vs_collisionality.png"),
            "--x",
            "nu_n",
            "--y",
            "FSABjHat_1",
            "--xscale",
            "log",
            "--yscale",
            "linear",
        ],
        cwd=work,
        label="bootstrap-plot",
        timeout_s=timeout_s,
    )
    _stage("Utils gallery generation complete")


if __name__ == "__main__":
    main()
