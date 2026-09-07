"""The one test module for registered DKX validation evidence.

This replaces nineteen per-campaign modules. Each of them re-implemented the
same generic half -- load an audit script by path, run it, assert it passed,
assert the claim scope and exclusions, and in eight cases tamper with the
artifact and assert the auditor rejected it. That half is now
`tools.release.registry`, driven from `validation/registry.toml` and
parametrized below over every entry.

What is *not* generic stays written out: the resolutions, root topologies,
movement values, and gate lists that make each campaign a scientific claim
rather than a file that parses. Those assertions are carried over unchanged
from the modules this file replaces, grouped by capability.

Adding a campaign means adding a registry entry, not a new test module.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tools.release.artifacts import (
    coefficient_relative_errors,
    dkes_to_beidler,
    nu_prime_for_nu_over_v,
)
from tools.release.registry import (
    check_entry,
    load_capability_ids,
    load_registry,
    run_corruption_probe,
    run_entry,
)

# The checkout is resolved from this file, not from the package. dkx.validation
# .registry can find a checkout when one sits above the package, but under an
# installed wheel there is none and it says so precisely. A test knows where the
# repository is; asking the package to guess is the wrong direction, and it is
# what made this whole module fail collection when coverage first ran against
# the installed artifact.
ROOT = Path(__file__).resolve().parents[1]
REGISTRY = load_registry(ROOT)
CAPABILITIES = load_capability_ids(ROOT)
ENTRY_IDS = list(REGISTRY.ids)
CORRUPTIBLE_IDS = [entry.id for entry in REGISTRY.entries if entry.corruption]


def test_direct_backend_referee_preserves_a_nonsymmetric_petsc_operator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Factoring A.T instead of A can change sparse fill and benchmark conclusions."""
    import numpy as np
    import scipy.sparse.linalg as sla
    from scipy.sparse import csr_matrix
    from tools.benchmarks.direct_solver_backends import superlu_seconds

    operator = csr_matrix([[4.0, 2.0, 0.0], [0.0, 3.0, 1.0], [2.0, 0.0, 5.0]])
    matrix = tmp_path / "matrix.petscbin"
    matrix.write_bytes(
        np.asarray([1211216, 3, 3, operator.nnz], dtype=">i4").tobytes()
        + np.diff(operator.indptr).astype(">i4").tobytes()
        + operator.indices.astype(">i4").tobytes()
        + operator.data.astype(">f8").tobytes()
    )
    factor = sla.splu
    checked = []

    def checked_factor(actual):
        np.testing.assert_array_equal(actual.toarray(), operator.toarray())
        factors = factor(actual)
        rhs = np.asarray([1.0, -2.0, 3.0])
        np.testing.assert_allclose(operator @ factors.solve(rhs), rhs, atol=1e-14)
        checked.append(True)
        return factors

    monkeypatch.setattr(sla, "splu", checked_factor)
    elapsed, size, fill = superlu_seconds(matrix)
    assert checked and elapsed >= 0 and size == 3 and fill >= operator.nnz


def _assert_process_stopped(pid: int) -> None:
    import subprocess
    import time

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        state = subprocess.run(
            ["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True,
        ).stdout.strip()
        if not state or state.startswith("Z"):
            return
        time.sleep(0.02)
    pytest.fail(f"measured descendant {pid} survived cleanup ({state})")


@pytest.mark.parametrize("parent_exits", [False, True])
def test_measurement_reaps_descendants_after_timeout_or_leader_exit(
    tmp_path: Path, parent_exits: bool,
) -> None:
    import os
    import signal
    import sys
    from tools.benchmarks.parity_performance_matrix import _run_measured

    if os.name != "posix":
        pytest.skip("measurement runner uses POSIX process groups")
    script = (
        "import subprocess,sys,time,pathlib; "
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
        "pathlib.Path('descendant.pid').write_text(str(p.pid)); "
        + ("sys.exit(0)" if parent_exits else "time.sleep(60)")
    )
    pid = None
    try:
        result = _run_measured([sys.executable, "-c", script], tmp_path, 1)
        pid = int((tmp_path / "descendant.pid").read_text())
        assert ("error" in result) is not parent_exits
        _assert_process_stopped(pid)
    finally:
        if pid is not None:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.parametrize("cancel_signal", ["SIGINT", "SIGTERM"])
def test_measurement_cancellation_reaps_the_solver_group(
    tmp_path: Path, cancel_signal: str,
) -> None:
    import os
    import signal
    import subprocess
    import sys
    import time

    if os.name != "posix":
        pytest.skip("measurement runner uses POSIX process groups")
    script = (
        "import os,pathlib,time; "
        "pathlib.Path('solver.pid').write_text(str(os.getpid())); time.sleep(60)"
    )
    worker = (
        "import sys; from pathlib import Path; "
        f"sys.path.insert(0, {str(ROOT)!r}); "
        "from tools.benchmarks.parity_performance_matrix import _run_measured; "
        f"_run_measured([{sys.executable!r}, '-c', {script!r}], Path('.'), 60)"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", worker], cwd=tmp_path,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
    )
    pid = None
    try:
        deadline = time.monotonic() + 10
        while not (tmp_path / "solver.pid").exists() and time.monotonic() < deadline:
            assert proc.poll() is None
            time.sleep(0.02)
        pid = int((tmp_path / "solver.pid").read_text())
        proc.send_signal(getattr(signal, cancel_signal))
        assert proc.wait(timeout=5) != 0
        _assert_process_stopped(pid)
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()
        if pid is not None:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.parametrize("mode", [1, 2, 3])
@pytest.mark.parametrize("defect", [None, "partial", "nan", "unfinished", "diverged", "nonlinear_false", "exit"])
def test_fortran_execution_gate_rejects_invalid_outputs(tmp_path: Path, defect, mode) -> None:
    import h5py
    import numpy as np
    from tools.benchmarks.parity_performance_matrix import _fortran_succeeded

    for stream in ("stdout", "stderr"):
        (tmp_path / f"benchmark.{stream}.log").write_text(
            "Nonlinear solve did not converge due to DIVERGED_MAX_IT\n" + "ok\n" * 4000
            if defect == "diverged" and stream == "stdout" else ""
        )
    with h5py.File(tmp_path / "sfincsOutput.h5", "w") as f:
        f["RHSMode"] = mode
        f["integerToRepresentTrue"] = 1
        f["finished"] = 0 if defect == "unfinished" else 1
        if defect == "nonlinear_false":
            f["didNonlinearCalculationConverge"] = -1
        keys = ("FSABFlow", "FSABjHat", "particleFlux_vm_psiHat", "heatFlux_vm_psiHat") if mode == 1 else ("transportMatrix",)
        for key in keys:
            if defect == "partial" and key == keys[0]:
                continue
            f[key] = [np.nan if defect == "nan" else 1.0]
        f.create_group("optional_group")
    result = {"returncode": 1 if defect == "exit" else 0}
    assert _fortran_succeeded(tmp_path, result) is (defect is None)
    assert ("execution_error" in result) is (defect is not None)


def test_sweep_does_not_reuse_outputs_copied_with_an_example(tmp_path: Path, monkeypatch) -> None:
    import shutil
    from tools.benchmarks import parity_performance_matrix as matrix

    example = tmp_path / "example"
    example.mkdir()
    shutil.copy(ROOT / "tests/ref/pas_1species_PAS_noEr_tiny_scheme1.input.namelist", example / "input.namelist")
    stale = ("sfincsOutput.h5", "dkxOutput.h5", "dkx_timing.json", "sfincsBinary_iteration_000_stateVector")
    for name in stale:
        (example / name).write_text('{"converged": true, "cold_s": 0.01}')
    calls = []

    def failed_run(command, work, timeout_s, env=None):
        assert not any((work / name).exists() for name in stale)
        calls.append(command)
        return {"returncode": 1}

    monkeypatch.setattr(matrix, "_run_measured", failed_run)
    record = matrix.run_case(
        example, Path("unused-sfincs"), ranks=[1], reps=0, timeout_s=1,
        equilibria=None, launcher=[], fortran_residual=False,
    )
    assert len(calls) == 2
    assert record["fortran"]["1"]["succeeded"] is False
    assert "cold_s" not in record["dkx"]
    assert "error" in record["parity"]


def test_campaign_checkpoint_is_atomic_on_publish_failure(tmp_path: Path, monkeypatch) -> None:
    from tools.benchmarks import parity_performance_matrix as matrix

    checkpoint = tmp_path / "results.jsonl"
    checkpoint.write_text("original\n")
    def interrupted_replace(source, target):
        raise OSError("interrupted publication")
    monkeypatch.setattr(matrix.os, "replace", interrupted_replace)
    with pytest.raises(OSError, match="interrupted"):
        matrix._atomic_text(checkpoint, "replacement\n")
    assert checkpoint.read_text() == "original\n"
    assert list(tmp_path.iterdir()) == [checkpoint]


@pytest.mark.parametrize("mutation", ["input", "equilibrium", "settings", "petsc", "petsc_env"])
@pytest.mark.parametrize("interrupt", [False, True])
def test_campaign_resume_retries_failures_and_checks_provenance(tmp_path: Path, monkeypatch, mutation, interrupt) -> None:
    from tools.benchmarks import parity_performance_matrix as matrix

    examples = tmp_path / "examples"
    for case in ("good", "retry"):
        work = examples / case
        work.mkdir(parents=True)
        (work / "input.namelist").write_text("&general\n/\n")
    equilibrium = tmp_path / "geometry.bc"
    equilibrium.write_text("original geometry")
    (examples / "good/input.namelist").write_text(f"equilibriumFile = '{equilibrium}'\n")
    out = tmp_path / "results.jsonl"
    calls = []
    def run(directory, *args, **kwargs):
        calls.append(directory.name)
        success = directory.name == "good" or calls.count("retry") > 1
        if interrupt and not success:
            raise KeyboardInterrupt("cancelled pilot")
        return {"case": directory.name, "dkx": {"returncode": 0 if success else 1, "converged": success, "algebraic_acceptance": "passed" if success else "failed"}}
    monkeypatch.setattr(matrix, "run_case", run)
    monkeypatch.setattr(matrix, "deck_metadata", lambda path: {"dof": 1})
    argv = ["--examples", str(examples), "--out", str(out)]
    if interrupt:
        with pytest.raises(KeyboardInterrupt):
            matrix.main(argv)
        assert not out.with_suffix(".jsonl.done").exists()
    else:
        assert matrix.main(argv) == 0
    assert matrix.main(argv) == 0
    assert calls == ["good", "retry", "retry"]
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == 3
    assert "KeyboardInterrupt" in rows[1]["error"] if interrupt else rows[1]["dkx"]["returncode"] == 1
    summary = json.loads(out.with_suffix(".jsonl.done").read_text())
    assert summary["cases"] == 2 and summary["attempts"] == 3
    assert summary["execution_complete"] == 2
    snapshot = out.read_bytes()
    if mutation == "input":
        (examples / "good/input.namelist").write_text("&general\n RHSMode=2\n/\n")
    elif mutation == "equilibrium":
        equilibrium.write_text("changed geometry")
    elif mutation == "petsc":
        argv += ["--fortran-petsc-opt=-mat_superlu_dist_equil", "--fortran-petsc-opt=false"]
    elif mutation == "petsc_env":
        monkeypatch.setenv("PETSC_OPTIONS", "-ksp_rtol 1e-8")
    else:
        argv += ["--reps", "2"]
    assert matrix.main(argv) == 2
    assert out.read_bytes() == snapshot and len(calls) == 3


def test_campaign_lock_refuses_a_concurrent_writer(tmp_path: Path) -> None:
    import fcntl
    from tools.benchmarks import parity_performance_matrix as matrix

    out = tmp_path / "results.jsonl"
    with out.with_suffix(".jsonl.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert matrix.main(["--examples", str(tmp_path), "--out", str(out)]) == 2
    assert not out.exists()


@pytest.mark.parametrize("mode", [1, 2, 3])
@pytest.mark.parametrize("perturb", [False, True])
def test_original_residual_checks_every_rhs_of_a_nonsymmetric_system(tmp_path: Path, mode, perturb) -> None:
    import numpy as np
    from scipy.sparse import csr_matrix
    from tools.benchmarks.parity_performance_matrix import fortran_true_residual

    a = csr_matrix([[3.0, 2.0], [0.0, 4.0]])
    matrix = tmp_path / "sfincsBinary_iteration_000_whichMatrix_1"
    matrix.write_bytes(
        np.asarray([1211216, 2, 2, a.nnz], dtype=">i4").tobytes()
        + np.diff(a.indptr).astype(">i4").tobytes()
        + a.indices.astype(">i4").tobytes() + a.data.astype(">f8").tobytes()
    )
    count = {1: 1, 2: 3, 3: 2}[mode]
    for i in range(count):
        x = np.asarray([1.0, i + 1.0])
        b = a @ x
        if perturb and i == count - 1:
            x[0] += 0.1
        for suffix, values in (("stateVector", x), ("residual", -b)):
            (tmp_path / f"sfincsBinary_iteration_{i:03d}_{suffix}").write_bytes(
                np.asarray([1211214, 2], dtype=">i4").tobytes() + values.astype(">f8").tobytes()
            )
    error = fortran_true_residual(tmp_path, linear=True, rhs_mode=mode)
    assert error > 1e-3 if perturb else error == 0.0
    assert fortran_true_residual(tmp_path, linear=False, rhs_mode=mode) is None
    (tmp_path / f"sfincsBinary_iteration_{count-1:03d}_stateVector").unlink()
    assert fortran_true_residual(tmp_path, linear=True, rhs_mode=mode) is None


@pytest.mark.parametrize("value,status", [(None, "not_checked"), (float("nan"), "failed"), (1e-4, "failed"), (1e-8, "passed")])
def test_original_residual_acceptance_is_distinct_from_convergence(value, status) -> None:
    from tools.benchmarks.parity_performance_matrix import _algebraic_acceptance
    assert _algebraic_acceptance({"converged": True, "true_residual": value}, 1e-6) == status


def test_requested_binary_dump_overrides_a_disabled_setting(tmp_path: Path) -> None:
    from tools.benchmarks.parity_performance_matrix import _request_binary_dump
    deck = tmp_path / "input.namelist"
    deck.write_text("&general\n saveMatricesAndVectorsInBinary = .false. ! retain this note\n/\n")
    _request_binary_dump(deck)
    assert "= .true. ! retain this note" in deck.read_text()


def payload(entry_id: str) -> dict[str, Any]:
    """Return the registered artifact for ``entry_id``."""
    return json.loads(
        (ROOT / REGISTRY[entry_id].artifact).read_text(encoding="utf-8")
    )


# --------------------------------------------------------------------------
# Generic registry gates
# --------------------------------------------------------------------------


def test_the_registry_covers_every_validation_artifact() -> None:
    """No artifact may sit in validation/ without a registry entry.

    This is the gate that keeps evidence from accumulating outside the registry
    the way it did before: a campaign that drops a summary in and stops has to
    declare its capability, its command, and what it does not establish.
    """
    registered = {REGISTRY[name].artifact for name in ENTRY_IDS}
    on_disk = {
        f"validation/{path.name}" for path in sorted((ROOT / "validation").glob("*.json"))
    }
    assert on_disk == registered


@pytest.mark.parametrize("entry_id", ENTRY_IDS)
def test_every_registry_entry_is_internally_sound(entry_id: str) -> None:
    """Checksum, claim scope, schema, exclusions, inputs, and capability agree."""
    assert check_entry(REGISTRY[entry_id], REGISTRY, CAPABILITIES) == []


@pytest.mark.parametrize("entry_id", ENTRY_IDS)
def test_every_registered_audit_still_passes(entry_id: str) -> None:
    """Each campaign's own audit recomputes its gates from the sealed artifact."""
    report = run_entry(REGISTRY[entry_id], ROOT)
    assert report["pass"] is True
    assert report.get("errors", []) == []


@pytest.mark.parametrize("entry_id", CORRUPTIBLE_IDS)
def test_every_declared_auditor_rejects_a_tampered_artifact(entry_id: str) -> None:
    """An auditor that accepts a tampered artifact certifies nothing."""
    assert run_corruption_probe(REGISTRY[entry_id], ROOT) == []


@pytest.mark.parametrize("entry_id", ENTRY_IDS)
def test_every_entry_states_what_it_does_not_establish(entry_id: str) -> None:
    """A claim without a boundary is the failure mode this registry exists for."""
    entry = REGISTRY[entry_id]
    assert entry.limitations
    assert entry.claim.strip()
    assert entry.claim_scope.strip()


@pytest.mark.parametrize("entry_id", ENTRY_IDS)
def test_every_entry_names_a_reproducible_command(entry_id: str) -> None:
    entry = REGISTRY[entry_id]
    assert entry.command.startswith("python ")
    assert entry.audit_script
    assert (ROOT / entry.audit_script).is_file()


#: The exclusions each campaign is required to keep stated. These are the exact
#: subsets the per-campaign modules asserted before they were consolidated; a
#: campaign may add exclusions but may never drop one of these.
REQUIRED_LIMITATIONS: dict[str, set[str]] = {
    "ambipolar_pitch_budget": {
        "not_phase_space_convergence_validation",
        "not_speed_or_zeta_convergence_validation",
        "not_independent_cross_code_ambipolar_validation",
        "not_full_fokker_planck_or_phi1_validation",
        "not_experimental_validation",
        "not_cross_code_performance_validation",
    },
    "ambipolar_pitch_combined": {
        "not_phase_space_convergence_validation",
        "not_independent_cross_code_ambipolar_validation",
    },
    "ambipolar_pitch_explicit_groups": {
        "not_phase_space_convergence_validation",
        "not_cross_allocation_runtime_comparison",
    },
    "ambipolar_pitch_speed_groups": {
        "not_phase_space_convergence_validation",
        "not_cross_allocation_runtime_comparison",
    },
    "ambipolar_speed_local_pitch": {
        "not_phase_space_convergence_validation",
        "not_full_profile_root_validation",
    },
    "full_kinetic_sfincs": {
        "not_multispecies_validation",
        "not_stellarator_full_fokker_planck_validation",
        "not_finite_er_validation",
        "not_ambipolar_profile_validation",
        "not_phi1_validation",
        "not_experimental_validation",
        "not_cross_code_performance_validation",
    },
    "full_kinetic_sfincs_finite_er": {
        "not_multispecies_validation",
        "not_stellarator_full_fokker_planck_validation",
        "not_er_scan_validation",
        "not_ambipolar_profile_validation",
        "not_phi1_validation",
        "not_experimental_validation",
        "not_cross_code_performance_validation",
    },
    "full_kinetic_sfincs_stellarator": {
        "not_multispecies_validation",
        "not_finite_er_validation",
        "not_er_scan_validation",
        "not_ambipolar_profile_validation",
        "not_phi1_validation",
        "not_experimental_validation",
    },
    "independent_cross_code": {
        "not_full_fokker_planck",
        "not_ambipolar_profile_validation",
        "not_experimental_validation",
    },
    "native_ambipolar_profile": {
        "not_phase_space_convergence_validation",
        "not_continuous_branch_event_localization",
        "not_experimental_validation",
        "not_full_fokker_planck_ambipolar_validation",
        "not_phi1_validation",
        "not_independent_cross_code_ambipolar_validation",
        "not_second_stellarator_family_validation",
        "not_cross_code_performance_validation",
    },
    "native_physical_flux_sfincs": {
        "whole-profile phase-space convergence",
        "ambipolar root agreement",
        "full Fokker-Planck collisions",
        "Phi1",
        "experimental validation",
        "cross-code performance equivalence",
    },
}

#: The one campaign whose exclusion list is asserted exactly rather than as a
#: subset: the ladder is the reference negative result, so a silently shortened
#: list there would weaken every later admission argument that cites it.
EXACT_LIMITATIONS: dict[str, set[str]] = {
    "ambipolar_phase_space_ladder": {
        "not_phase_space_convergence_validation",
        "not_independent_cross_code_ambipolar_validation",
        "not_full_fokker_planck_ambipolar_validation",
        "not_experimental_validation",
        "not_phi1_validation",
        "not_cross_code_performance_validation",
        "fine_rung_does_not_refine_zeta_or_speed_beyond_reference",
    },
}


@pytest.mark.parametrize("entry_id", sorted(REQUIRED_LIMITATIONS))
def test_required_exclusions_are_still_stated(entry_id: str) -> None:
    assert REQUIRED_LIMITATIONS[entry_id] <= set(REGISTRY[entry_id].limitations)


@pytest.mark.parametrize("entry_id", sorted(EXACT_LIMITATIONS))
def test_exact_exclusion_lists_are_unchanged(entry_id: str) -> None:
    assert EXACT_LIMITATIONS[entry_id] == set(REGISTRY[entry_id].limitations)


def test_negative_results_are_kept_rather_than_deleted() -> None:
    """Ruled-out routes stay registered so they are not retried by accident."""
    negatives = {
        entry.id for entry in REGISTRY.entries if entry.status == "negative_result"
    }
    assert "w7x_admitted_grid_uniform_probe_no_go" in negatives
    assert "ambipolar_phase_space_ladder" in negatives


# --------------------------------------------------------------------------
# Ambipolar phase-space refinement: the axis and ladder negative results
# --------------------------------------------------------------------------


def test_axis_ladder_pins_its_resolutions_and_topology() -> None:
    axes = payload("ambipolar_phase_space_axes")
    assert [rung["resolution"] for rung in axes["rungs"]] == [
        {"theta": 15, "zeta": 37, "pitch": 36, "speed": 6},
        {"theta": 17, "zeta": 37, "pitch": 36, "speed": 6},
        {"theta": 15, "zeta": 37, "pitch": 40, "speed": 6},
        {"theta": 15, "zeta": 37, "pitch": 44, "speed": 6},
    ]
    for rung in axes["rungs"]:
        assert [surface["root_count"] for surface in rung["surfaces"]] == [1, 1, 3, 1, 1]
        assert sum(len(surface["roots"]) for surface in rung["surfaces"]) == 7
    for comparison in axes["comparisons"]:
        assert comparison["summary"]["topology_stable"] is True
        assert len(comparison["root_movements"]) == 7
        assert len(comparison["selected_movements"]) == 5


def test_axis_ladder_identifies_pitch_without_promoting_it() -> None:
    outcome = payload("ambipolar_phase_space_axes")["outcome"]
    diagnosis = outcome["diagnosis"]
    assert outcome["status"] == "refinement_exhausted"
    assert outcome["admission_pass"] is False
    assert outcome["theta17_vs_reference"]["failed_gates"] == [
        "all_root_electric_field_movement"
    ]
    assert outcome["pitch44_vs_pitch40"]["failed_gates"] == [
        "all_root_electric_field_movement",
        "selected_particle_flux_movement",
        "selected_heat_flux_movement",
    ]
    assert diagnosis["dominant_failed_direction"] == "pitch"
    assert diagnosis["theta_max_root_movement_kV_m"] == pytest.approx(0.1611328125)
    assert diagnosis["pitch40_max_root_movement_kV_m"] == pytest.approx(1.7333984375)
    assert diagnosis["pitch40_to_pitch44_approaches_gate"] is False
    assert diagnosis["pitch48_bruteforce_admitted"] is False
    assert outcome["measured_max_accepted_true_residual"] == pytest.approx(
        3.75131427218295e-13
    )


def test_phase_space_ladder_records_every_root_and_selected_observable() -> None:
    ladder = payload("ambipolar_phase_space_ladder")
    assert [rung["resolution"] for rung in ladder["rungs"]] == [
        {"theta": 13, "zeta": 31, "pitch": 32, "speed": 5},
        {"theta": 15, "zeta": 37, "pitch": 36, "speed": 6},
        {"theta": 17, "zeta": 37, "pitch": 40, "speed": 6},
    ]
    assert [
        [surface["root_count"] for surface in rung["surfaces"]]
        for rung in ladder["rungs"]
    ] == [[1, 1, 3, 1, 1]] * 3
    for rung in ladder["rungs"]:
        assert sum(len(surface["roots"]) for surface in rung["surfaces"]) == 7
        for surface in rung["surfaces"]:
            selected = surface["roots"][surface["selected_root_index"]]
            assert selected["selected"] is True
            assert len(selected["particle_flux_m2_s"]) == 2
            assert len(selected["heat_flux_W_m2"]) == 2
    assert all(
        len(comparison["root_movements"]) == 7
        and len(comparison["selected_movements"]) == 5
        for comparison in ladder["comparisons"]
    )


def test_phase_space_ladder_fails_admission_without_relaxing_gates() -> None:
    ladder = payload("ambipolar_phase_space_ladder")
    admission = ladder["admission"]
    latest = ladder["comparisons"][-1]["summary"]
    assert admission["status"] == "refinement_exhausted"
    assert admission["admission_pass"] is False
    assert admission["limits"] == {
        "max_all_root_electric_field_movement_kV_m": 0.005,
        "max_selected_particle_flux_scaled_movement": 0.02,
        "max_selected_heat_flux_scaled_movement": 0.02,
        "max_accepted_true_residual": 1e-12,
    }
    assert latest["topology_stable"] is True
    assert latest["max_all_root_electric_field_movement_kV_m"] == pytest.approx(
        1.6259765625
    )
    assert latest["max_selected_particle_flux_scaled_movement"] == pytest.approx(
        0.040755305441301674
    )
    assert latest["max_selected_heat_flux_scaled_movement"] == pytest.approx(
        0.0781298871962563
    )
    assert admission["measured_max_accepted_true_residual"] == pytest.approx(
        3.917314213777927e-13
    )
    assert admission["failed_gates"] == [
        "all_root_electric_field_movement",
        "selected_particle_flux_movement",
        "selected_heat_flux_movement",
    ]


# --------------------------------------------------------------------------
# Pitch allocation: budget, explicit groups, speed groups, combined
# --------------------------------------------------------------------------


def test_pitch_budget_pins_exact_route_parity() -> None:
    parity = payload("ambipolar_pitch_budget")["route_parity"]
    assert parity["status"] == "resolved"
    assert parity["admission_pass"] is True
    assert parity["metrics"]["roots_and_brackets_exact"] is True
    assert parity["metrics"]["full_executed_routes"] == {"block_tridiagonal": 139}
    assert parity["metrics"]["bounded_executed_routes"] == {
        "block_tridiagonal_truncated": 139
    }
    assert parity["metrics"]["max_evaluation_heat_flux_relative_difference"] < 2.0e-9
    assert parity["footprint_reduction_fraction"] == pytest.approx(0.9082292155037494)
    assert parity["warm_speedup_claim"] is False


def test_uniform_pitch_ladder_retains_every_changed_topology() -> None:
    rungs = payload("ambipolar_pitch_budget")["rungs"]
    assert [rung["resolution"]["pitch"] for rung in rungs] == [22, 26, 30]
    assert [rung["resolution"]["pitch_speed_ramp"] for rung in rungs] == [0, 0, 0]
    assert [
        [surface["root_count"] for surface in rung["surfaces"]] for rung in rungs
    ] == [[3, 1, 1], [1, 1, 1], [1, 3, 1]]
    assert [
        [
            surface["roots"][surface["selected_root_index"]]["electric_field_kV_m"]
            for surface in rung["surfaces"]
        ]
        for rung in rungs
    ] == [
        [-0.361328125, -1.5966796875, -3.4375],
        [9.23828125, -0.9521484375, -3.18359375],
        [10.5615234375, 6.748046875, -2.9736328125],
    ]


def test_uniform_pitch_ladder_fails_unchanged_gates() -> None:
    budget = payload("ambipolar_pitch_budget")
    outcome = budget["outcome"]
    assert outcome["status"] == "refinement_exhausted"
    assert outcome["admission_pass"] is False
    assert outcome["uniform34_or_higher_admitted"] is False
    assert outcome["measured_max_accepted_true_residual"] == pytest.approx(
        5.641144886776424e-14
    )
    for comparison in outcome["comparisons"]:
        assert comparison["failed_gates"] == [
            "topology_stable",
            "selected_electric_field_movement",
            "selected_particle_flux_movement",
            "selected_heat_flux_movement",
        ]
    summaries = [comparison["summary"] for comparison in budget["comparisons"]]
    assert summaries[0]["max_selected_electric_field_movement_kV_m"] == pytest.approx(
        9.599609375
    )
    assert summaries[1]["max_selected_electric_field_movement_kV_m"] == pytest.approx(
        7.7001953125
    )
    assert summaries[0]["max_selected_heat_flux_scaled_movement"] > 0.55
    assert summaries[1]["max_selected_heat_flux_scaled_movement"] > 0.45


def test_explicit_groups_fix_total_and_high_speed_work_exactly() -> None:
    rungs = payload("ambipolar_pitch_explicit_groups")["rungs"]
    assert [rung["allocation"]["active_pitch_mode_sum"] for rung in rungs] == [
        129,
        129,
        129,
    ]
    assert [rung["allocation"]["groups"] for rung in rungs] == [
        {
            "low_speed_nodes_0_1": 13,
            "intermediate_speed_nodes_2_3": 44,
            "high_speed_nodes_4_5": 72,
        },
        {
            "low_speed_nodes_0_1": 24,
            "intermediate_speed_nodes_2_3": 33,
            "high_speed_nodes_4_5": 72,
        },
        {
            "low_speed_nodes_0_1": 8,
            "intermediate_speed_nodes_2_3": 49,
            "high_speed_nodes_4_5": 72,
        },
    ]


def test_explicit_groups_hold_topology_but_fail_movement_gates() -> None:
    groups = payload("ambipolar_pitch_explicit_groups")
    assert [
        [surface["root_count"] for surface in rung["surfaces"]]
        for rung in groups["rungs"]
    ] == [[1, 3], [1, 3], [1, 3]]
    outcome = groups["outcome"]
    assert outcome["phase_space_converged"] is False
    assert outcome["maximum_selected_movements"] == {
        "electric_field_kV_m": 1.064453125,
        "particle_flux_scaled": 0.09894109254425403,
        "heat_flux_scaled": 0.09077156078426843,
    }
    assert groups["intermediate_cold_warm_parity"] == {
        "scientific_arrays_exact": True,
        "ignored_timing_arrays": ["solve_time_s"],
        "mismatches": [],
    }


def test_speed_group_allocations_change_topology_at_fixed_work() -> None:
    speed_groups = payload("ambipolar_pitch_speed_groups")
    rungs = speed_groups["rungs"]
    assert [rung["allocation"]["active_pitch_mode_sum"] for rung in rungs] == [
        132,
        129,
        133,
    ]
    assert [rung["allocation"]["groups"] for rung in rungs] == [
        {
            "low_speed_nodes_0_1": 44,
            "intermediate_speed_nodes_2_3": 44,
            "high_speed_nodes_4_5": 44,
        },
        {
            "low_speed_nodes_0_1": 13,
            "intermediate_speed_nodes_2_3": 44,
            "high_speed_nodes_4_5": 72,
        },
        {
            "low_speed_nodes_0_1": 9,
            "intermediate_speed_nodes_2_3": 36,
            "high_speed_nodes_4_5": 88,
        },
    ]
    assert [
        [surface["root_count"] for surface in rung["surfaces"]] for rung in rungs
    ] == [[3, 1], [1, 3], [1, 1]]
    outcome = speed_groups["outcome"]
    assert outcome["phase_space_converged"] is False
    assert outcome["topology_changing_comparisons"] == [
        "uniform22_to_linear36",
        "linear36_to_quadratic44",
    ]
    assert speed_groups["quadratic_cold_warm_parity"] == {
        "scientific_arrays_exact": True,
        "ignored_timing_arrays": ["solve_time_s"],
        "mismatches": [],
    }


@pytest.mark.parametrize(
    "entry_id", ["ambipolar_pitch_explicit_groups", "ambipolar_pitch_speed_groups"]
)
def test_retained_allocation_rungs_meet_residual_and_memory_gates(entry_id: str) -> None:
    data = payload(entry_id)
    assert (
        max(
            rung["attempts"]["maximum_accepted_true_residual"] for rung in data["rungs"]
        )
        <= 1.0e-12
    )
    assert (
        max(row["peak_footprint_bytes"] for row in data["measurements"].values())
        < 24 * 2**30
    )


def test_combined_pitch_ladder_retains_failed_observable_gates() -> None:
    combined = payload("ambipolar_pitch_combined")
    assert [rung["root_counts"] for rung in combined["rungs"]] == [[1, 3]] * 3
    assert combined["outcome"]["gates"] == {
        "topology_stable": True,
        "maximum_true_residual_below_1e-12": True,
        "all_process_footprints_below_24_gib": True,
        "combined_cold_warm_scientific_arrays_exact": True,
        "electric_field_movement_below_0_005_kV_m": False,
        "particle_flux_movement_below_0_02": False,
        "heat_flux_movement_below_0_02": False,
    }
    assert {row["code"] for row in combined["source"]["inspiration_review"]} == {
        "SFINCS",
        "YANCC",
        "MONKES",
        "STELLOPT/PENTA",
    }


# --------------------------------------------------------------------------
# Speed, zeta, and selected-tail diagnostics
# --------------------------------------------------------------------------


def test_joint_pitch_speed_gates_all_fail_and_the_tail_claim_is_route_aware() -> None:
    joint = payload("ambipolar_joint_pitch_speed")
    gates = joint["outcome"]["gates"]
    assert gates["speed_particle_below_2_percent"] is False
    assert gates["speed_heat_below_2_percent"] is False
    assert gates["pitch_particle_below_2_percent"] is False
    assert gates["pitch_heat_below_2_percent"] is False
    assert joint["analytic_full_state_oracle"]["diagnostic_status"] == (
        "retained_full_state_relative_l2"
    )
    assert all(
        rung["diagnostic_status"] == "unavailable_on_zero_padded_truncated_state"
        for rung in joint["rungs"]
    )


def test_joint_speed_zeta_movements_exceed_the_gate_and_the_tail_is_not_monotone() -> None:
    tail = payload("ambipolar_joint_speed_zeta_tail")
    assert all(
        row["particle_flux_scaled_movement"] > 0.02
        and row["heat_flux_scaled_movement"] > 0.02
        for row in tail["comparisons"]
    )
    tail_maxima = [rung["maximum_selected_tail_bound"] for rung in tail["rungs"]]
    assert max(tail_maxima) > 0.09
    assert tail_maxima[3] < tail_maxima[2]
    assert tail["outcome"]["phase_space_converged"] is False
    assert tail["outcome"]["whole_profile_escalation_admitted"] is False


def test_selected_tail_evidence_is_bounded_sparse_and_unpromoted() -> None:
    import numpy as np

    case = payload("ambipolar_selected_tail_bound")["case"]
    bound = np.asarray(
        case["selected_tail_bound_by_surface_speed_species"], dtype=np.float64
    )
    assert bound.shape == (2, 8, 2)
    assert np.all((bound >= 0.0) & (bound <= 1.0))
    assert np.max(bound) == case["maximum_tail_bound"]
    assert case["finite_tail_values"] == bound.size
    assert case["diagnostic_replays"] == 2
    assert case["diagnostic_status"] == (
        "retained_selected_tail_relative_l2_upper_bound"
    )
    outcome = payload("ambipolar_selected_tail_bound")["outcome"]
    assert outcome["phase_space_converged"] is False
    assert outcome["whole_profile_escalation_admitted"] is False


def test_speed_local_probes_localize_node3_and_retain_failed_ceilings() -> None:
    local = payload("ambipolar_speed_local_pitch")
    shares = local["comparisons"][0]["node3_absolute_delta_share"]
    assert min(shares["particle"] + shares["heat"]) > 0.96
    gates = local["outcome"]["gates"]
    assert gates["node3_dominates_initial_delta"] is True
    assert gates["node3_33_to_36_particle_below_2_percent"] is False
    assert gates["node3_33_to_36_heat_below_2_percent"] is False
    assert gates["pitch36_to_44_particle_below_2_percent"] is False
    assert gates["pitch36_to_44_heat_below_2_percent"] is False


# --------------------------------------------------------------------------
# W7-X seeded roots and the fixed-field referee
# --------------------------------------------------------------------------


def test_admitted_grid_seeded_envelope_keeps_strict_signs_and_scope() -> None:
    envelope = payload("w7x_admitted_grid_seeded_envelope")
    final = envelope["final_replay"]
    assert all(left * right < 0.0 for left, right in final["endpoint_currents_A_m2"])
    assert final["cold_warm_arrays_exact_except_solve_time_s"] is True
    assert final["maximum_primal_residual"] < 1.0e-12
    assert envelope["envelope_run"]["statuses"][1] == "seeded_bracket_failed"
    assert all(
        value < 0.0
        for value in envelope["envelope_run"]["surface_1_endpoint_currents_A_m2"].values()
    )
    assert envelope["case"]["unsampled_crossings_excluded"] is False
    assert envelope["outcome"]["global_all_root_claim"] is False


def test_seeded_bracket_discovery_replays_every_candidate_without_promoting() -> None:
    discovery = payload("w7x_seeded_bracket_discovery")
    endpoints = discovery["seeded_replay"]["endpoints"]
    assert len(endpoints) == 4
    assert all(
        endpoint["left_current_A_m2"] * endpoint["right_current_A_m2"] < 0.0
        for endpoint in endpoints
    )
    assert discovery["outcome"]["discovery_brackets_replayed"] is True
    assert discovery["outcome"]["admitted_grid_promotion_ready"] is False
    assert "admitted-grid ambipolar roots" in discovery["claim_exclusions"]


def test_uniform_admitted_grid_launch_is_a_bounded_no_go() -> None:
    no_go = payload("w7x_admitted_grid_uniform_probe_no_go")
    assert no_go["preflight"] == {
        "hierarchy_points": 33,
        "max_evaluations_per_surface": 825,
        "max_profile_evaluations": 1650,
        "retained_profile_bytes": 1531200,
    }
    measurement = no_go["measurement"]
    assert measurement["completed_surfaces"] == 0
    assert measurement["result_written"] is False
    assert measurement["wall_seconds"] == 2551.33
    assert measurement["maximum_rss_bytes"] == 10205478912
    assert measurement["peak_process_footprint_bytes"] == 21883225584
    route = no_go["route_diagnosis"]
    assert route["reusable_dense_coarse_bands_bytes"] > 3 * 24 * 1024**3
    assert route["schur_lu_factors_bytes"] > 24 * 1024**3
    assert route["checkpointed_dense_factors_per_subsystem_bytes"] < 1024**3
    assert no_go["outcome"]["uniform_high_grid_launch_admitted"] is False
    assert no_go["outcome"]["numerical_failure_claimed"] is False
    assert "no-root evidence" in no_go["claim_exclusions"]
    assert len(no_go["source"]["input_sha256"]) == 64
    assert len(no_go["source"]["log_sha256"]) == 64


def test_fixed_field_referee_retains_the_independent_pitch_ladder() -> None:
    referee = payload("w7x_fixed_field_resolution_referee")
    pitch = referee["comparisons"]["sfincs_pitch"]
    assert [item["label"] for item in pitch] == [
        "sfincs_pitch_52_to_70",
        "sfincs_pitch_70_to_90",
        "sfincs_pitch_90_to_120",
        "sfincs_pitch_120_to_150",
    ]
    assert pitch[0]["heat_flux_max_scaled_movement"] > 0.16
    assert (
        max(
            pitch[-1]["particle_flux_max_scaled_movement"],
            pitch[-1]["heat_flux_max_scaled_movement"],
            pitch[-1]["parallel_current_scaled_movement"],
        )
        < 0.002
    )
    assert (
        referee["comparisons"]["dkx_sfincs_pitch150_zeta37_parity"][
            "maximum_scaled_error"
        ]
        < 0.005
    )


def test_fixed_field_referee_admits_flux_without_promoting_current() -> None:
    referee = payload("w7x_fixed_field_resolution_referee")
    outcome = referee["outcome"]
    assert outcome["sfincs_pitch_converged"] is True
    assert outcome["dkx_pitch_converged_at_zeta85"] is True
    assert outcome["dkx_zeta85_flux_converged_against_zeta109"] is True
    assert outcome["dkx_speed8_flux_converged_against_speed10"] is True
    assert outcome["dkx_theta15_flux_converged_against_theta19"] is True
    assert outcome["transport_flux_fixed_field_admitted"] is True
    assert outcome["parallel_current_theta_converged"] is False
    assert outcome["parallel_current_status"] == "refinement_exhausted"
    assert outcome["whole_profile_admitted"] is False
    theta = referee["comparisons"]["dkx_theta"]
    assert all(item["parallel_current_scaled_movement"] > 0.05 for item in theta)
    assert all(item["particle_flux_max_scaled_movement"] < 0.01 for item in theta)
    assert all(item["heat_flux_max_scaled_movement"] < 0.011 for item in theta)


# --------------------------------------------------------------------------
# Native workflow certificates
# --------------------------------------------------------------------------


def test_native_profile_certificate_pins_its_case() -> None:
    profile = payload("native_ambipolar_profile")
    case = profile["case"]
    assert profile["source"]["geometry_sha256"] == (
        "81c686e5a5bd8f38d8b1f754ebe2910951f20094bab35d73d2827d9875bb6062"
    )
    assert case["case_id"] == (
        "f284407a441b7e06c4f3a24a0b46e80676f60e42d66cf6bce84113c1d1f096bf"
    )
    assert case["physics"] == {
        "collisions": "pitch_angle_scattering",
        "magnetic_drifts": "dkes",
        "model": "full_local",
        "phi1": "off",
    }
    assert case["solver"]["relative_tolerance"] == pytest.approx(1e-9)


def test_native_profile_certificate_retains_all_roots_and_recovery() -> None:
    profile = payload("native_ambipolar_profile")
    acceptance = profile["acceptance"]
    assert acceptance["all_gates_pass"] is True
    assert acceptance["measured_root_counts"] == [1, 1, 3, 1, 1]
    assert acceptance["measured_max_final_bracket_width_kV_m"] == pytest.approx(
        0.0048828125
    )
    assert acceptance["measured_max_selected_primal_residual"] == pytest.approx(
        5.365974684738137e-15
    )
    assert acceptance["measured_max_root_current_bracket_fraction"] < 0.5
    assert acceptance["measured_automatic_recovery_count"] == 1
    assert acceptance["measured_scientific_array_differences"] == []
    attempts = profile["profile"]["attempts"]
    assert attempts["attempt_count"] == 222
    assert attempts["executed_route_counts"] == {
        "block_tridiagonal_truncated": 221,
        "gmres": 1,
    }
    recovery = attempts["recoveries"][0]
    assert recovery["surface_index"] == 4
    assert recovery["electric_field_kV_m"] == 0.0
    assert [attempt["accepted"] for attempt in recovery["attempts"]] == [False, True]
    assert recovery["attempts"][0]["residual"] == pytest.approx(8.234475195278555e-13)
    assert recovery["attempts"][1]["residual"] == pytest.approx(1.9323484429235053e-13)


def test_physical_flux_certificate_pins_the_matched_case_and_factor() -> None:
    flux = payload("native_physical_flux_sfincs")
    case = flux["case"]
    assert case["collision_operator"] == "pitch_angle_scattering"
    assert case["use_dkes_exb_drift"] is True
    assert case["phi1"] is False
    assert case["electric_field_kV_m"] == 8.55
    assert case["resolution"]["pitch_modes_by_speed"] == [6, 11, 19, 30, 42, 52, 52, 52]
    conversion = flux["conversion"]
    assert conversion["correct_d_dr_hat_to_d_dpsi_hat"] == pytest.approx(
        -1.3949598653433055
    )
    assert conversion["previous_inverse_d_dpsi_hat_to_d_dr_hat"] == pytest.approx(
        -0.7168665026458634
    )
    assert conversion["correct_to_previous_ratio"] == pytest.approx(1.9459130259186133)


def test_physical_flux_certificate_supersedes_without_promoting_values() -> None:
    supersession = payload("native_physical_flux_sfincs")["historical_artifact_supersession"]
    assert supersession["absolute_physical_flux_values_promoted"] is False
    assert "validation/native_ambipolar_profile_v1.json" in supersession["artifacts"]
    assert (
        "validation/ambipolar_joint_speed_zeta_tail_v1.json" in supersession["artifacts"]
    )


# --------------------------------------------------------------------------
# Full-kinetic SFINCS parity and the monoenergetic cross-code rung
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("entry_id", "electric_field", "matrix_sizes"),
    [
        ("full_kinetic_sfincs", 0.0, [6887, 12509]),
        ("full_kinetic_sfincs_finite_er", -30.0, [6887, 12509]),
        ("full_kinetic_sfincs_stellarator", 0.0, [54407, 98126]),
    ],
)
def test_full_kinetic_artifacts_pin_matched_equations(
    entry_id: str, electric_field: float, matrix_sizes: list[int]
) -> None:
    """The three matched full-Fokker-Planck rungs share one equation contract.

    Only the field, the geometry, and the matrix sizes differ, which is what
    makes them a family rather than three unrelated comparisons.
    """
    data = payload(entry_id)
    equations = data["equations"]
    assert equations["collision_operator"] == 0
    assert equations["electric_field"] == electric_field
    assert equations["constraint_input"] == -1
    assert equations["constraint_resolved"] == 1
    assert equations["include_x_dot"] is True
    assert equations["include_electric_field_term_in_xi_dot"] is True
    assert equations["use_dkes_exb_drift"] is False

    rungs = data["rungs"]
    assert [rung["id"] for rung in rungs] == ["high", "ultra"]
    sizes = [
        rung["resolution"].get("sfincs_matrix_size", rung["resolution"].get("matrix_size"))
        for rung in rungs
    ]
    assert sizes == matrix_sizes
    assert data["acceptance"]["all_gates_pass"] is True


def test_zero_field_tokamak_parity_reaches_its_recorded_accuracy() -> None:
    acceptance = payload("full_kinetic_sfincs")["acceptance"]
    assert acceptance["measured_max_cross_code_scaled_error"] == pytest.approx(
        2.6843339656885753e-10
    )
    assert acceptance["measured_max_high_to_ultra_scaled_movement"] == pytest.approx(
        0.002797721820022167
    )
    assert acceptance["measured_max_near_zero_absolute_value"] == pytest.approx(
        3.625981422302594e-13
    )
    assert acceptance["measured_max_completed_true_residual"] == pytest.approx(
        1.8153328e-11
    )
    assert (
        acceptance["measured_max_cross_code_scaled_error"]
        < acceptance["max_cross_code_scaled_error"]
    )
    assert (
        acceptance["measured_max_high_to_ultra_scaled_movement"]
        < acceptance["max_high_to_ultra_scaled_movement"]
    )


def test_finite_field_tokamak_parity_reaches_its_recorded_accuracy() -> None:
    data = payload("full_kinetic_sfincs_finite_er")
    assert data["equations"]["solver_tolerance"] == pytest.approx(1e-13)
    assert [rung["resolution"]["dkx_matrix_size"] for rung in data["rungs"]] == [
        10532,
        18614,
    ]
    acceptance = data["acceptance"]
    assert acceptance["measured_max_cross_code_scaled_error"] == pytest.approx(
        1.8756861003267208e-9
    )
    assert acceptance["measured_max_high_to_ultra_scaled_movement"] == pytest.approx(
        0.0032524453653068005
    )
    assert acceptance["measured_max_completed_true_residual"] == pytest.approx(
        5.2475734e-11
    )


def test_stellarator_parity_pins_its_geometry_and_accuracy() -> None:
    data = payload("full_kinetic_sfincs_stellarator")
    equations = data["equations"]
    source = data["source"]
    assert equations["geometry"] == "W7-X SC1 Boozer surface"
    assert equations["geometry_scheme"] == 11
    assert equations["solver_tolerance"] == pytest.approx(1e-12)
    assert source["sfincs_equilibrium"] == "equilibria/w7x-sc1.bc"
    assert source["sfincs_equilibrium_sha256"] == (
        "1d096d5ad8104750fcc787ef226b2fbc8a82bcd3774fbab41a2f87dcb04ce831"
    )
    assert [rung["resolution"]["dkx_matrix_size"] for rung in data["rungs"]] == [
        87887,
        155994,
    ]
    acceptance = data["acceptance"]
    assert acceptance["measured_max_cross_code_scaled_error"] == pytest.approx(
        1.3644269489024203e-8
    )
    assert acceptance["measured_max_high_to_ultra_scaled_movement"] == pytest.approx(
        0.004436383524571227
    )
    assert acceptance["measured_max_near_zero_absolute_value"] == pytest.approx(
        1.8892805190711152e-21
    )
    assert acceptance["measured_max_completed_true_residual"] == pytest.approx(
        1.8182906e-12
    )


def test_nu_over_v_map_includes_the_applied_deflection_shape() -> None:
    value = nu_prime_for_nu_over_v(
        0.01,
        g_hat=2.3204100000000003,
        i_hat=-0.0033862,
        iota=-0.468945,
        b0_over_bbar=1.5451308075,
        nu_d_hat_x0=0.8360276804879032,
    )
    assert value == pytest.approx(0.017975290661104672, rel=2e-15)


def test_dkes_conversion_uses_local_radius_and_orientation() -> None:
    converted = dkes_to_beidler(
        d11=0.09420723079216414,
        d31=0.5671247552808146,
        d13=-0.5689076170400662,
        d33=60.24173801533913,
        nu_over_v=0.01,
        g_hat=2.3204100000000003,
        iota=-0.468945,
        b0_over_bbar=1.5451308075,
        r_hat=0.1600004874677725,
        raw_b0_over_bbar=1.5451292699958306,
        raw_fsab_b2=0.9930199354487245 * 1.5451292699958306**2,
    )
    assert converted["D11_star"] == pytest.approx(0.40334459550951945)
    assert converted["D31_star"] == pytest.approx(-0.13780489737481735)
    assert converted["D13_star"] == pytest.approx(0.13823811260564514)
    assert converted["D33_star"] == pytest.approx(0.9099777738316578)
    assert converted["eps_t"] == pytest.approx(0.10654224141486761)


def test_conversion_rejects_ambiguous_d33_scale() -> None:
    common = dict(
        d11=1.0,
        d31=1.0,
        d13=-1.0,
        d33=1.0,
        nu_over_v=0.1,
        g_hat=1.0,
        iota=0.5,
        b0_over_bbar=1.0,
        r_hat=0.2,
    )
    with pytest.raises(ValueError, match="exactly one"):
        dkes_to_beidler(**common)
    with pytest.raises(ValueError, match="exactly one"):
        dkes_to_beidler(**common, raw_fsab_b2=1.0, d33_spitzer=1.0)


def test_cross_code_rung_recomputes_every_normalization_and_gate() -> None:
    data = payload("independent_cross_code")
    assert {case["family"] for case in data["cases"]} == {
        "axisymmetric_tokamak",
        "ncsx_stellarator",
        "w7x_eim_stellarator",
    }

    for case in data["cases"]:
        raw = case["reference"]["raw_dkes"]
        norm = case["normalization"]
        d33_kwargs = (
            {"d33_spitzer": raw["D33_spitzer"]}
            if "D33_spitzer" in raw
            else {"raw_fsab_b2": raw["fsab_B2"]}
        )
        converted = dkes_to_beidler(
            d11=raw["D11"],
            d31=raw["D31"],
            d13=raw["D13"],
            d33=raw["D33"],
            nu_over_v=case["equations"]["nu_over_v_per_m"],
            g_hat=norm["g_hat_T_m"],
            iota=norm["iota"],
            b0_over_bbar=norm["B0_T"],
            r_hat=norm["local_r_m"],
            raw_b0_over_bbar=norm["raw_B0_T"],
            cross_orientation=norm["cross_orientation"],
            **d33_kwargs,
        )
        for key, expected in case["reference"]["beidler"].items():
            assert converted[key] == pytest.approx(expected, rel=2e-14)

        errors = coefficient_relative_errors(case["dkx"]["beidler"], converted)
        assert errors == pytest.approx(case["comparison"]["relative_error"], rel=2e-13)
        assert max(errors.values()) <= case["comparison"]["relative_tolerance"]


def test_stellarator_audit_rejects_a_wrong_external_equilibrium(tmp_path: Path) -> None:
    """The one probe that reaches outside the artifact.

    The corruption probes in the registry edit the sealed JSON. This campaign
    also pins the SFINCS equilibrium file itself, so the auditor has to notice
    when the file behind the checksum is a different one -- a failure a
    JSON-only tamper cannot express.
    """
    import hashlib

    entry = REGISTRY["full_kinetic_sfincs_stellarator"]
    data = payload(entry.id)
    data["source"]["sfincs_equilibrium_sha256"] = hashlib.sha256(b"expected").hexdigest()
    data["source"]["sfincs_equilibrium_bytes"] = len(b"expected")
    artifact = tmp_path / "stellarator.json"
    artifact.write_text(json.dumps(data), encoding="utf-8")

    results = tmp_path / "results"
    for rung in ("high", "ultra"):
        for code in ("sfincs", "dkx"):
            directory = results / rung / code
            directory.mkdir(parents=True)
            (directory / "w7x-sc1.bc").write_bytes(b"wrong")

    from tools.release.registry import load_audit_callable

    audit = load_audit_callable(entry, ROOT)
    report = audit(artifact, results_root=results)
    assert report["pass"] is False
    assert "high: external sfincs equilibrium checksum mismatch" in report["errors"]
    assert "ultra: external dkx equilibrium checksum mismatch" in report["errors"]


def test_pitch_budget_pins_its_bounded_route_commit() -> None:
    """The registry pins one commit per entry; this campaign compares two."""
    assert payload("ambipolar_pitch_budget")["source"]["bounded_dkx_commit"] == (
        "f08fd7a1c802b0d860a2d694924d33fd2e52cec0"
    )


def test_registry_paths_resolve_from_the_repository_root() -> None:
    """Guard the one assumption every other test here makes."""
    assert (ROOT / "validation" / "registry.toml").is_file()
    assert isinstance(ROOT, Path)


# --------------------------------------------------------------------------
# The runner's own failure behaviour
#
# The registry is only worth having if it refuses bad entries, so each way an
# entry can be wrong is exercised against a small synthetic checkout rather than
# against the real one.
# --------------------------------------------------------------------------

ARTIFACT_BODY = {
    "schema": "dkx.example.v1",
    "claim_scope": "example_scope",
    "source": {"dkx_commit": "a" * 40},
    "outcome": {"value": 2.0},
    "exclusions": ["not_a_real_claim"],
}

AUDIT_SCRIPT = '''
import json
from pathlib import Path


def audit(artifact, *, results_root=None):
    payload = json.loads(Path(artifact).read_text(encoding="utf-8"))
    errors = []
    if payload["outcome"]["value"] != 2.0:
        errors.append("value mismatch")
    return {"pass": not errors, "errors": errors}
'''


def _write_checkout(tmp_path: Path, **overrides: Any) -> Path:
    """Build a one-entry checkout the runner can load, then override fields."""
    import hashlib

    (tmp_path / "validation").mkdir()
    (tmp_path / "tools").mkdir()
    artifact = tmp_path / "validation" / "example_v1.json"
    artifact.write_text(json.dumps(ARTIFACT_BODY), encoding="utf-8")
    (tmp_path / "tools" / "audit_example.py").write_text(AUDIT_SCRIPT, encoding="utf-8")

    entry: dict[str, Any] = {
        "id": "example",
        "capability": "native_case_result",
        "status": "accepted",
        "claim": "An example claim.",
        "claim_scope": "example_scope",
        "artifact": "validation/example_v1.json",
        "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "artifact_schema": "dkx.example.v1",
        "audit_script": "tools/audit_example.py",
        "command": "python tools/audit_example.py --artifact validation/example_v1.json",
        "inputs": [],
        "dkx_commit": "a" * 40,
        "generated_on_host": "official-laptop-cpu",
        "limitations": ["not_a_real_claim"],
    }
    entry.update(overrides)

    def render(value: Any) -> str:
        """JSON is valid TOML for scalars and arrays, but not for tables."""
        if isinstance(value, dict):
            pairs = ", ".join(f"{key} = {render(item)}" for key, item in value.items())
            return "{" + pairs + "}"
        return json.dumps(value)

    lines = [
        "schema_version = 1",
        'recorded_at = "2026-08-30"',
        'registry_commit = "b"',
        'status_values = ["accepted", "negative_result"]',
        "",
        "[[entry]]",
    ]
    lines += [f"{key} = {render(value)}" for key, value in entry.items()]
    (tmp_path / "validation" / "registry.toml").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    (tmp_path / "validation" / "capabilities.toml").write_text(
        '[[capability]]\nid = "native_case_result"\n', encoding="utf-8"
    )
    (tmp_path / "validation" / "hardware.toml").write_text(
        '[[host]]\nid = "official-laptop-cpu"\n', encoding="utf-8"
    )
    return tmp_path


def _problems(tmp_path: Path, **overrides: Any) -> list[str]:
    from tools.release.registry import load_hardware_ids

    root = _write_checkout(tmp_path, **overrides)
    registry = load_registry(root)
    return check_entry(
        registry.entries[0],
        registry,
        load_capability_ids(root),
        load_hardware_ids(root),
    )


def test_a_sound_synthetic_entry_reports_no_problems(tmp_path: Path) -> None:
    assert _problems(tmp_path) == []


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"status": "invented"}, "is not one of"),
        ({"capability": "no_such_capability"}, "not in capabilities.toml"),
        ({"generated_on_host": "no_such_host"}, "not a host in hardware.toml"),
        ({"artifact_sha256": "0" * 64}, "has sha256"),
        ({"claim_scope": "wrong_scope"}, "claim_scope mismatch"),
        ({"artifact_schema": "dkx.other.v1"}, "schema mismatch"),
        ({"limitations": ["something_else"]}, "do not match the artifact"),
        ({"inputs": ["validation/absent.toml"]}, "does not exist"),
        ({"dkx_commit": "b" * 40}, "dkx_commit mismatch"),
        ({"command": "python tools/other.py"}, "does not invoke the declared audit"),
        ({"artifact": "validation/absent.json"}, "does not exist"),
    ],
)
def test_the_runner_names_each_way_an_entry_can_be_wrong(
    tmp_path: Path, overrides: dict[str, Any], expected: str
) -> None:
    problems = _problems(tmp_path, **overrides)
    assert any(expected in problem for problem in problems), problems


def test_a_missing_required_field_is_refused_at_load(tmp_path: Path) -> None:
    root = _write_checkout(tmp_path, claim="")
    with pytest.raises(ValueError, match="missing required fields: claim"):
        load_registry(root)


def test_duplicate_entry_ids_are_refused_at_load(tmp_path: Path) -> None:
    root = _write_checkout(tmp_path)
    path = root / "validation" / "registry.toml"
    body = path.read_text(encoding="utf-8")
    path.write_text(body + "\n" + body.split("[[entry]]", 1)[1].join(["[[entry]]", ""]))
    with pytest.raises(ValueError, match="duplicate registry entry ids: example"):
        load_registry(root)


def test_an_entry_without_an_audit_script_reports_that_it_was_not_audited(
    tmp_path: Path,
) -> None:
    root = _write_checkout(tmp_path, audit_script="", command="python nothing.py")
    registry = load_registry(root)
    report = run_entry(registry.entries[0], root)
    assert report == {
        "pass": True,
        "audited": False,
        "reason": "entry declares no audit script; generic checks only",
    }
    assert run_corruption_probe(registry.entries[0], root) == []


def test_a_declared_probe_without_an_audit_script_is_itself_a_problem(
    tmp_path: Path,
) -> None:
    root = _write_checkout(
        tmp_path,
        audit_script="",
        command="python nothing.py",
        corruption={
            "path": "outcome.value",
            "operation": "set",
            "value": 3.0,
            "expected_errors": ["value mismatch"],
        },
    )
    registry = load_registry(root)
    problems = run_corruption_probe(registry.entries[0], root)
    assert problems == ["example declares a corruption probe but no audit script"]


def test_a_probe_the_auditor_survives_is_reported(tmp_path: Path) -> None:
    """An edit the auditor does not notice must not read as a passing probe."""
    root = _write_checkout(
        tmp_path,
        corruption={
            "path": "claim_scope",
            "operation": "set",
            "value": "irrelevant",
            "expected_errors": [],
        },
    )
    registry = load_registry(root)
    problems = run_corruption_probe(registry.entries[0], root)
    assert problems == ["audit accepted a tampered artifact (set claim_scope)"]


def test_a_probe_that_fails_for_the_wrong_reason_is_reported(tmp_path: Path) -> None:
    root = _write_checkout(
        tmp_path,
        corruption={
            "path": "outcome.value",
            "operation": "scale",
            "value": 2.0,
            "expected_errors": ["some other complaint"],
        },
    )
    registry = load_registry(root)
    problems = run_corruption_probe(registry.entries[0], root)
    assert problems == [
        "tampered audit did not report 'some other complaint'; got 'value mismatch'"
    ]


def test_an_unknown_corruption_operation_is_refused() -> None:
    from tools.release.registry import corrupt_payload

    with pytest.raises(ValueError, match="unknown corruption operation 'invert'"):
        corrupt_payload({"a": 1}, {"path": "a", "operation": "invert", "value": 0})


def test_corruption_paths_index_lists_including_from_the_end() -> None:
    from tools.release.registry import corrupt_payload

    document = {"rows": [{"v": 1.0}, {"v": 2.0}]}
    corrupt_payload(document, {"path": "rows.-1.v", "operation": "scale", "value": 3.0})
    assert document == {"rows": [{"v": 1.0}, {"v": 6.0}]}


def test_an_audit_script_without_an_audit_function_is_refused(tmp_path: Path) -> None:
    from tools.release.registry import load_audit_callable

    root = _write_checkout(tmp_path)
    (root / "tools" / "audit_example.py").write_text("x = 1\n", encoding="utf-8")
    entry = load_registry(root).entries[0]
    with pytest.raises(AttributeError, match="defines no audit"):
        load_audit_callable(entry, root)


def test_a_declared_audit_script_that_is_missing_is_reported(tmp_path: Path) -> None:
    root = _write_checkout(tmp_path)
    (root / "tools" / "audit_example.py").unlink()
    registry = load_registry(root)
    problems = check_entry(registry.entries[0], registry, load_capability_ids(root))
    assert problems == ["audit script tools/audit_example.py does not exist"]


def test_an_audit_returning_a_non_mapping_is_refused(tmp_path: Path) -> None:
    root = _write_checkout(tmp_path)
    (root / "tools" / "audit_example.py").write_text(
        "def audit(artifact, *, results_root=None):\n    return 'fine'\n",
        encoding="utf-8",
    )
    entry = load_registry(root).entries[0]
    with pytest.raises(TypeError, match="audit\\(\\) returned"):
        run_entry(entry, root)


def test_an_entry_whose_audit_fails_is_reported_by_the_registry(tmp_path: Path) -> None:
    """The audit itself failing, with the seal and every other check intact."""
    from tools.release.registry import audit_registry

    root = _write_checkout(tmp_path)
    (root / "tools" / "audit_example.py").write_text(
        "def audit(artifact, *, results_root=None):\n"
        "    return {'pass': False, 'errors': ['deliberate']}\n",
        encoding="utf-8",
    )
    report = audit_registry(root)
    assert report["pass"] is False
    assert report["failed"] == 1
    assert "audit did not pass" in report["results"][0]["problems"][0]


def test_artifact_exclusions_reads_both_recorded_spellings() -> None:
    from tools.release.registry import artifact_exclusions

    assert artifact_exclusions({"exclusions": ["a"]}) == ("a",)
    assert artifact_exclusions({"claim_exclusions": ["b"]}) == ("b",)
    assert artifact_exclusions({}) == ()


def test_the_registry_is_not_available_outside_a_checkout(tmp_path: Path) -> None:
    """A wheel user gets a precise error, not an obscure missing-file traceback."""
    from tools.release.registry import repository_root

    with pytest.raises(FileNotFoundError, match="not from a wheel"):
        repository_root(tmp_path / "nowhere" / "site-packages" / "dkx")


def test_a_non_object_artifact_is_refused(tmp_path: Path) -> None:
    from tools.release.registry import read_artifact

    root = _write_checkout(tmp_path)
    (root / "validation" / "example_v1.json").write_text("[1, 2]", encoding="utf-8")
    entry = load_registry(root).entries[0]
    with pytest.raises(TypeError, match="not a JSON object"):
        read_artifact(entry, root)


def test_an_unknown_entry_id_raises(tmp_path: Path) -> None:
    registry = load_registry(_write_checkout(tmp_path))
    with pytest.raises(KeyError):
        registry["absent"]


def test_the_cli_reports_success_and_failure(tmp_path: Path, capsys) -> None:
    from tools.release.registry import main

    root = _write_checkout(tmp_path)
    assert main(["--root", str(root), "--quiet"]) == 0
    assert "1/1 registry entries pass" in capsys.readouterr().out

    assert main(["--root", str(root), "--entry", "example"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["entries"] == 1
    assert report["results"][0]["id"] == "example"

    (root / "validation" / "example_v1.json").write_text(
        json.dumps({**ARTIFACT_BODY, "outcome": {"value": 9.0}}), encoding="utf-8"
    )
    assert main(["--root", str(root)]) == 1
    failed = json.loads(capsys.readouterr().out)
    assert failed["failed"] == 1

@pytest.mark.parametrize("failure", ["timeout", "environment", "none"])
def test_reference_preflight_is_isolated_and_supervised(tmp_path, monkeypatch, failure):
    import sys
    from tools.benchmarks import parity_performance_matrix as matrix

    binary = tmp_path / "reference"
    body = {
        "timeout": "import time; time.sleep(10)",
        "environment": "print('critical libmamba: prefix does not exist'); print('x' * 10000)",
        "none": "from pathlib import Path; Path('input.namelist').write_text('changed')",
    }[failure]
    binary.write_text(f"#!{sys.executable}\n{body}\n")
    binary.chmod(0o755)
    original = tmp_path / "input.namelist"
    original.write_text("preserve")
    monkeypatch.chdir(tmp_path)
    run = matrix._run_measured
    workdirs = []

    def bounded(command, cwd, timeout_s):
        assert command[-3:] == ["-label", "a b", "-help"]
        workdirs.append(cwd)
        return run(command, cwd, 0.1 if failure == "timeout" else timeout_s)

    monkeypatch.setattr(matrix, "_run_measured", bounded)
    reason = matrix.preflight_fortran(Path("reference"), [], ("-label", "a b"))
    assert original.read_text() == "preserve"
    assert workdirs and all(not path.exists() for path in workdirs)
    if failure == "none":
        assert reason is None
    else:
        assert ("timeout" if failure == "timeout" else "cannot resolve") in reason


def test_fortran_control_inventory_preserves_continuations_and_conditions():
    from tools.parity.output_key_coverage_report import namelist_controls
    source = '''
    #ifdef OPTIONAL_PHYSICS
    namelist / species / density, & ! active continuation
    !! a removed old member must not be counted

       & temperature, &
    #endif
       charge
    #if MODEL_A
    namelist / physics / full_model
    #else
    namelist / physics / reduced_model
    #endif
    '''
    groups = namelist_controls(source)
    assert [item['name'] for item in groups['species']] == ['density', 'temperature', 'charge']
    assert groups['species'][1]['conditions'] == ['#ifdef OPTIONAL_PHYSICS']
    assert groups['species'][2]['conditions'] == []
    assert groups['physics'][1]['conditions'] == ['#if MODEL_A -> #else']
    assert groups['species'][0]['line'] == 3
    for malformed in ('namelist /g/ a, &', 'namelist /g/ a, A', 'namelist /g/ a(1)', '#endif'):
        with pytest.raises(ValueError):
            namelist_controls(malformed)


def test_conditioning_builds_without_solving_and_caps_before_materializing(monkeypatch, capsys):
    import importlib
    from tools.benchmarks import operator_conditioning as probe

    def forbidden(*args, **kwargs):
        raise AssertionError('conditioning inventory must not solve or materialize above its cap')

    monkeypatch.setattr(importlib.import_module('dkx.solve'), 'solve', forbidden)
    monkeypatch.setattr(importlib.import_module('dkx.run'), 'solve', forbidden)
    monkeypatch.setattr(probe, 'materialize_csr', forbidden)
    deck = str(ROOT / 'tests/ref/pas_1species_PAS_noEr_tiny_scheme1.input.namelist')
    op = probe.operator_for(deck)
    assert op.total_size == 111
    probe.main([deck], max_size=1)
    assert 'too large' in capsys.readouterr().out


def test_petsc_arguments_and_observed_backend_are_distinct(tmp_path, monkeypatch):
    from tools.benchmarks import parity_performance_matrix as matrix
    (tmp_path / "input.namelist").write_text("&general\n/\n")
    monkeypatch.setattr(matrix, "deck_metadata", lambda path: {"solverTolerance": 1e-10})
    calls = []

    def measured(command, work, *args, **kwargs):
        calls.append(command)
        (work / "benchmark.stdout.log").write_text(
            "package used to perform factorization: mumps\n" + "x" * 9000
        )
        (work / "benchmark.stderr.log").write_text("")
        return {"returncode": 1}

    monkeypatch.setattr(matrix, "_run_measured", measured)
    options = ("-pc_factor_mat_solver_type", "superlu_dist", "-options_string", "a value with spaces")
    record = matrix.run_case(tmp_path, Path("/reference"), fortran_petsc_opts=options,
                             ranks=[1, 2], reps=1, timeout_s=10, equilibria=None,
                             launcher=["isolated-env"], fortran_residual=True)
    assert calls[0][:6] == ["isolated-env", "/reference", *options]
    assert calls[1][:5] == ["isolated-env", "mpirun", "-n", "2", "/reference"]
    assert calls[1][5:9] == list(options)
    assert record["fortran_petsc_opts"] == list(options)
    assert record["fortran"]["1"]["observed_factor_backends"] == ["mumps"]
    assert not record["algebraic_pair_accepted"]


def test_warm_observable_audit_checks_equations_and_reuse_modes(tmp_path):
    from tools.benchmarks import operator_conditioning as probe

    source = ROOT / 'tests/ref/pas_1species_PAS_noEr_tiny_scheme1.input.namelist'
    target = tmp_path / 'target.namelist'
    target.write_text(source.read_text().replace('epsilon_t = 0.1d+0', 'epsilon_t = 0.12d+0'))
    audit = probe.warm_audit(str(source), str(target), tolerances=(1e-10,))
    assert audit['size'] == 111
    assert len(set(audit['input_sha256'].values())) == 2
    assert audit['dual_relative_residual'] < 1e-12
    assert audit['seed_relative_residual'] < 1e-10
    assert [r['reuse'] for r in audit['records']] == ['cold', 'state', 'recycle', 'state_and_recycle']
    for record in audit['records']:
        assert record['original_relative_residual'] < 1e-10
        # The dual identity includes floating-point residual/moment evaluation.
        assert abs(record['identity_remainder']) < 1e-10 * max(abs(record['observable']), 1e-10)
    with pytest.raises(ValueError, match='size cap'):
        probe.warm_audit(str(source), str(target), max_size=1)
    with pytest.raises(ValueError, match='positive and finite'):
        probe.warm_audit(str(source), str(target), tolerances=(float('nan'),))
    target.write_text(target.read_text().replace('Nxi = 4', 'Nxi = 5'))
    with pytest.raises(ValueError, match='structure'):
        probe.warm_audit(str(source), str(target))


def test_warm_audit_dual_identity_sign_with_deliberately_inaccurate_states(monkeypatch):
    import importlib
    from types import SimpleNamespace
    from tools.benchmarks import operator_conditioning as probe
    import jax.numpy as jnp

    calls = []
    def inaccurate(op, rhs, **kwargs):
        calls.append(None)
        return SimpleNamespace(x=jnp.full((op.total_size, 1), float(len(calls))),
                               recycle=None, method='manufactured', iterations=0, converged=False)
    monkeypatch.setattr(importlib.import_module('dkx.solve'), 'solve', inaccurate)
    deck = str(ROOT / 'tests/ref/pas_1species_PAS_noEr_tiny_scheme1.input.namelist')
    audit = probe.warm_audit(deck, deck, tolerances=(1e-10,))
    for row in audit['records'][1:]:
        assert not row['original_residual_pass'] and not row['solver_converged']
        assert not row['recycle_supplied']
        assert abs(row['observable_difference']) > 1e-5
        assert row['dual_predicted_difference'] == pytest.approx(row['observable_difference'], rel=1e-11)
        assert row['linear_observable_difference'] == pytest.approx(row['observable_difference'], rel=1e-11)



def test_warm_audit_zero_source_and_nonfinite_drives(tmp_path, monkeypatch):
    from tools.benchmarks import operator_conditioning as probe
    from dkx.drift_kinetic import KineticOperator
    import jax.numpy as jnp

    target = ROOT / 'tests/ref/pas_1species_PAS_noEr_tiny_scheme1.input.namelist'
    source = tmp_path / 'zero.namelist'
    source.write_text(target.read_text().replace('dNHatdrHats = -6.0d+0', 'dNHatdrHats = 0.0')
                      .replace('dTHatdrHats = -3.0d+0', 'dTHatdrHats = 0.0'))
    audit = probe.warm_audit(str(source), str(target), tolerances=(1e-10,))
    assert audit['seed_original_residual_pass']
    assert audit['seed_absolute_residual'] == 0.0
    assert audit['seed_relative_residual'] is None
    assert all(r['original_residual_pass'] for r in audit['records'])
    assert not audit['records'][0]['initial_state_supplied']
    assert audit['records'][1]['initial_state_supplied']

    def forbidden(*args, **kwargs):
        raise AssertionError('nonfinite drives must be rejected before materialization')
    monkeypatch.setattr(probe, 'materialize_csr', forbidden)
    for value in (float('nan'), float('inf'), 1e308):
        monkeypatch.setattr(KineticOperator, 'rhs', lambda self: jnp.full(self.total_size, value))
        with pytest.raises(ValueError, match='finite drives and norms'):
            probe.warm_audit(str(target), str(target))
