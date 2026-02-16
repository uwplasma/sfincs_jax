#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "mplconfig"))

import h5py
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
UTILS = REPO_ROOT / "utils"
EXAMPLES = REPO_ROOT / "examples" / "sfincs_examples"


@dataclass(frozen=True)
class ScanConfig:
    name: str
    base_input: Path
    nuprime_factor: float
    collision_operator: int
    label: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="generate_sfincs_paper_figs",
        description="Reproduce low-resolution SFINCS paper figures with sfincs_jax runs.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "docs" / "_static" / "figures" / "paper",
        help="Directory for output figures.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=REPO_ROOT / "examples" / "publication_figures" / "output",
        help="Scratch directory for scan runs.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use reduced resolution and fewer scan points.",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=None,
        help="Per-step timeout in seconds (applied to each scan run).",
    )
    parser.add_argument(
        "--case",
        choices=("lhd", "w7x", "all"),
        default="all",
        help="Which geometry scans to run/plot.",
    )
    parser.add_argument(
        "--scan-only",
        action="store_true",
        help="Run scans only (skip plotting).",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Plot only (reuse existing scan output; do not run scans).",
    )
    return parser.parse_args()


def _run(cmd: list[str], *, cwd: Path, timeout_s: float | None, label: str) -> None:
    print(f"[{label}] cwd={cwd}")
    print(f"[{label}] cmd={' '.join(cmd)}")
    log_path = cwd / f"{label}.log"
    print(f"[{label}] log={log_path}")
    sys.stdout.flush()
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env.setdefault("MPLCONFIGDIR", str(cwd / ".mplconfig"))
    with log_path.open("w") as log:
        subprocess.run(
            cmd,
            cwd=str(cwd),
            check=True,
            env=env,
            timeout=timeout_s,
            stdout=log,
            stderr=log,
        )


def _strip_ss_lines(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if line.strip().lower().startswith("!ss"):
            continue
        lines.append(line)
    return "\n".join(lines) + "\n"


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


def _write_scan_input(
    *,
    base_input: Path,
    dest: Path,
    nu_n_min: float,
    nu_n_max: float,
    n_points: int,
    collision_operator: int,
    fast: bool,
) -> None:
    text = _strip_ss_lines(base_input.read_text())
    text = text + "\n".join(
        [
            "!ss scanType = 3",
            "!ss scanVariable = nu_n",
            f"!ss scanVariableMin = {nu_n_min:.6e}",
            f"!ss scanVariableMax = {nu_n_max:.6e}",
            f"!ss scanVariableN = {n_points}",
            "!ss scanVariableScale = log",
            "",
        ]
    )
    text = _inject_group(
        text,
        "physicsParameters",
        [
            f"  collisionOperator = {collision_operator}",
        ],
    )
    if fast:
        text = _inject_group(
            text,
            "resolutionParameters",
            [
                "  Ntheta = 5",
                "  Nzeta = 3",
                "  Nxi = 3",
                "  NL = 3",
                "  Nx = 3",
                "  solverTolerance = 1e-4",
            ],
        )
    dest.write_text(text)


def _collect_transport_matrix(work_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    for sub in sorted(work_dir.iterdir()):
        if not sub.is_dir():
            continue
        h5_path = sub / "sfincsOutput.h5"
        if not h5_path.exists():
            continue
        with h5py.File(h5_path, "r") as f:
            nu_n = float(np.asarray(f["nu_n"][()]))
            g_hat = float(np.asarray(f["GHat"][()]))
            i_hat = float(np.asarray(f["IHat"][()]))
            iota = float(np.asarray(f["iota"][()]))
            b0_over_bbar = float(np.asarray(f["B0OverBBar"][()]))
            nuprime = nu_n * (g_hat + iota * i_hat) / b0_over_bbar
            tm = np.asarray(f["transportMatrix"][()], dtype=float)
        rows.append((nuprime, tm))
    rows.sort(key=lambda x: x[0])
    nu = np.asarray([r[0] for r in rows])
    tm = np.asarray([r[1] for r in rows])
    return nu, tm


def _fit_high_collisionality(nu: np.ndarray, y: np.ndarray, n_fit: int = 2) -> np.ndarray:
    if nu.size < n_fit + 1:
        return y
    x_fit = np.log(nu[-n_fit:])
    y_fit = np.log(np.abs(y[-n_fit:]) + 1e-30)
    slope, intercept = np.polyfit(x_fit, y_fit, 1)
    sign = np.sign(y[-1]) if np.sign(y[-1]) != 0 else 1.0
    return sign * np.exp(slope * np.log(nu) + intercept)


def _plot_matrix_elements(
    *,
    out_path: Path,
    title: str,
    datasets: dict[str, tuple[np.ndarray, np.ndarray]],
    y_label: str = "transportMatrix element",
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), constrained_layout=True, sharex=True)
    elements = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for ax, (i, j) in zip(axes.flat, elements, strict=False):
        for label, (nu, tm) in datasets.items():
            ax.plot(nu, tm[:, i, j], marker="o", label=label)
        ax.set_xscale("log")
        ax.set_title(f"L{i+1}{j+1}")
        ax.grid(True, which="both", alpha=0.3)
    axes[1, 0].set_xlabel(r"$\nu'$")
    axes[1, 1].set_xlabel(r"$\nu'$")
    axes[0, 0].set_ylabel(y_label)
    axes[1, 0].set_ylabel(y_label)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle(title)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_simakov_helander_proxy(
    *,
    out_path: Path,
    title: str,
    datasets: dict[str, tuple[np.ndarray, np.ndarray]],
    element: tuple[int, int] = (0, 0),
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    i, j = element
    for label, (nu, tm) in datasets.items():
        ax.plot(nu, tm[:, i, j], marker="o", label=label)
        ax.plot(nu, _fit_high_collisionality(nu, tm[:, i, j]), linestyle="--", alpha=0.7)
    ax.set_xscale("log")
    ax.set_title(f"{title} (L{i+1}{j+1})")
    ax.set_xlabel(r"$\nu'$")
    ax.set_ylabel("transportMatrix element")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    out_dir = args.out_dir
    work_dir = args.work_dir
    fast = bool(args.fast)
    timeout_s = args.timeout_s
    case = args.case
    scan_only = bool(args.scan_only)
    plot_only = bool(args.plot_only)

    if scan_only and plot_only:
        raise ValueError("Cannot combine --scan-only and --plot-only.")

    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(work_dir / ".mplconfig"))

    if not plot_only:
        if case in ("lhd", "all"):
            shutil.rmtree(work_dir / "lhd_co0", ignore_errors=True)
            shutil.rmtree(work_dir / "lhd_co1", ignore_errors=True)
        if case in ("w7x", "all"):
            shutil.rmtree(work_dir / "w7x_co0", ignore_errors=True)
            shutil.rmtree(work_dir / "w7x_co1", ignore_errors=True)

    n_points = 4 if fast else 7
    nuprime_min = 0.1
    nuprime_max = 10.0

    lhd = ScanConfig(
        name="lhd",
        base_input=EXAMPLES / "transportMatrix_geometryScheme2" / "input.namelist",
        nuprime_factor=0.2668018,
        collision_operator=0,
        label="Fokker-Planck",
    )
    w7x = ScanConfig(
        name="w7x",
        base_input=EXAMPLES / "transportMatrix_geometryScheme11" / "input.namelist",
        nuprime_factor=0.172714565,
        collision_operator=0,
        label="Fokker-Planck",
    )

    fig1_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    if case in ("lhd", "all"):
        scan_models = [
            (lhd, 0, "Fokker-Planck"),
            (lhd, 1, "PAS"),
        ]
        for cfg, collision_operator, label in scan_models:
            nu_n_min = nuprime_min * cfg.nuprime_factor
            nu_n_max = nuprime_max * cfg.nuprime_factor
            case_dir = work_dir / f"{cfg.name}_co{collision_operator}"
            case_dir.mkdir(parents=True, exist_ok=True)
            if not plot_only:
                _write_scan_input(
                    base_input=cfg.base_input,
                    dest=case_dir / "input.namelist",
                    nu_n_min=nu_n_min,
                    nu_n_max=nu_n_max,
                    n_points=n_points,
                    collision_operator=collision_operator,
                    fast=fast,
                )
                _run(
                    [sys.executable, str(UTILS / "sfincsScan"), "--yes", "--input", "input.namelist"],
                    cwd=case_dir,
                    timeout_s=timeout_s,
                    label=f"scan-{cfg.name}-co{collision_operator}",
                )
            fig1_data[label] = _collect_transport_matrix(case_dir)

        if not scan_only:
            _plot_matrix_elements(
                out_path=out_dir / "sfincs_jax_fig1_lhd_collisionality.png",
                title="LHD collisionality scan (sfincs_jax)",
                datasets=fig1_data,
            )

    # Figure 2 (W7-X)
    fig2_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if case in ("w7x", "all"):
        for collision_operator, label in [(0, "Fokker-Planck"), (1, "PAS")]:
            nu_n_min = nuprime_min * w7x.nuprime_factor
            nu_n_max = nuprime_max * w7x.nuprime_factor
            case_dir = work_dir / f"w7x_co{collision_operator}"
            case_dir.mkdir(parents=True, exist_ok=True)
            if not plot_only:
                _write_scan_input(
                    base_input=w7x.base_input,
                    dest=case_dir / "input.namelist",
                    nu_n_min=nu_n_min,
                    nu_n_max=nu_n_max,
                    n_points=n_points,
                    collision_operator=collision_operator,
                    fast=fast,
                )
                _run(
                    [sys.executable, str(UTILS / "sfincsScan"), "--yes", "--input", "input.namelist"],
                    cwd=case_dir,
                    timeout_s=timeout_s,
                    label=f"scan-w7x-co{collision_operator}",
                )
            fig2_data[label] = _collect_transport_matrix(case_dir)

        if not scan_only:
            _plot_matrix_elements(
                out_path=out_dir / "sfincs_jax_fig2_w7x_collisionality.png",
                title="W7-X collisionality scan (sfincs_jax)",
                datasets=fig2_data,
            )

    # Figure 3 proxy: high-collisionality fit for FP data
    if not scan_only and fig1_data and fig2_data:
        fig3_data = {
            "LHD (FP)": fig1_data["Fokker-Planck"],
            "W7-X (FP)": fig2_data["Fokker-Planck"],
        }
        _plot_simakov_helander_proxy(
            out_path=out_dir / "sfincs_jax_fig3_simakov_helander.png",
            title="High-collisionality proxy",
            datasets=fig3_data,
            element=(0, 0),
        )


if __name__ == "__main__":
    main()
