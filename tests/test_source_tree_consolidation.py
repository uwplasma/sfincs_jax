from __future__ import annotations

import json

import ast
import importlib
from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "src" / "dkx"
SCRIPT_ROOT = REPO_ROOT / "scripts"
PACKAGE_README = PACKAGE_ROOT / "README.md"
SOURCE_MAP_DOC = REPO_ROOT / "docs" / "source_map.rst"
PACKAGE_README_REQUIRED_SECTIONS = (
    "## The Canonical Stack (the architecture)",
    "## Other Root Modules",
    "## Remaining Domain Packages",
    "## Design Rules",
    "## Stability And Compatibility",
    "## Generated Files Policy",
    "## Contributor Workflow",
)
DISALLOWED_TRACKED_PACKAGE_PARTS = {
    "__pycache__",
    ".ipynb_checkpoints",
}
DISALLOWED_TRACKED_PACKAGE_SUFFIXES = {
    ".h5",
    ".hdf5",
    ".nc",
    ".netcdf",
    ".npy",
    ".npz",
    ".pb",
    ".prof",
    ".pyc",
    ".pyo",
}
INVENTORY_CATEGORIES = {
    "core",
    "compat",
    "test-fixture",
    "extract-pr",
    "delete",
}
INVENTORY_ACTIONS = {
    "retain",
    "trim",
    "split",
    "extract",
    "delete",
    "promote",
}
INVENTORY_DECISIONS = {
    "keep",
    "merge",
    "delete",
    "extract-pr",
}
REQUIRED_CORE_SLIM_SOURCE_OWNERS = {
    "src/dkx/drift_kinetic.py",
    "src/dkx/solve.py",
    "src/dkx/writer.py",
    "src/dkx/magnetic_geometry.py",
    "src/dkx/moments.py",
    "tools/release/artifacts.py",
    "tools/release/release.py",
}
REQUIRED_CORE_SLIM_NONPACKAGE_OWNERS = {
    "examples",
    "tests",
    "scripts",
}
REQUIRED_RESEARCH_BRANCHES = {
    "research/parallel-performance",
    "research/publication-audits",
}


def _package_modules() -> list[str]:
    """Every root module in the package, read from the tree rather than a fixture.

    This used to come from tests/fixtures/source_tree_expected.json. A frozen
    inventory makes the documentation check double as a refactor veto -- moving
    or adding a module failed a test about the README -- which plan.md section
    7.2 lists for removal. Walking the tree keeps the useful half, that the
    README documents what is actually there, and drops the half that blocked
    legitimate change.
    """
    return sorted(path.name for path in PACKAGE_ROOT.glob("*.py"))


def _package_packages() -> list[str]:
    """Every domain subpackage in the package tree."""
    return sorted(
        path.name
        for path in PACKAGE_ROOT.iterdir()
        if path.is_dir() and (path / "__init__.py").is_file()
    )


def _package_dirs() -> list[Path]:
    return sorted(
        path
        for path in PACKAGE_ROOT.rglob("*")
        if path.is_dir() and path.name != "__pycache__"
    )


def _relative_dir(path: Path) -> str:
    return path.relative_to(PACKAGE_ROOT).as_posix()


def _tracked_paths() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return sorted(line for line in result.stdout.splitlines() if line)


def _inventory_rule_matches(path: str, rule: dict[str, object]) -> bool:
    for excluded in rule.get("exclude_prefix_any", []):
        if path.startswith(str(excluded)):
            return False

    exact_paths = {str(candidate) for candidate in rule.get("exact_paths", [])}
    if exact_paths:
        return path in exact_paths

    if "prefix_any" in rule:
        return any(path.startswith(str(prefix)) for prefix in rule["prefix_any"])

    if "prefix" in rule:
        return path.startswith(str(rule["prefix"]))

    return False


def test_plan_md_is_the_only_authoritative_planning_file() -> None:
    """Keep one reviewable roadmap without reviving competing plan files."""

    assert (REPO_ROOT / "plan.md").is_file()
    competing_plans = sorted(
        path.name
        for path in REPO_ROOT.glob("*plan*.md")
        if path.name != "plan.md"
    )
    assert competing_plans == [], f"competing planning files: {competing_plans}"
    # 2026-07-17 archived the pytest-split durations snapshot to keep the tree
    # light and accepted a count-only shard split; by 2026-09-06 that split had
    # one shard at 9m53s of a 10-minute budget and timing out on stacked PRs.
    # The snapshot is back, compact and bounded: it must parse and stay small.
    durations = REPO_ROOT / ".test_durations"
    if durations.exists():
        assert durations.stat().st_size <= 512 * 1024, "durations snapshot too large"
        loaded = json.loads(durations.read_text(encoding="utf-8"))
        assert loaded and all(isinstance(v, (int, float)) and v >= 0 for v in loaded.values())


def test_package_readme_describes_current_source_layout() -> None:
    """Every module and subpackage that exists is named in the package README."""
    text = PACKAGE_README.read_text(encoding="utf-8")

    assert "flat, physics-named root modules" in text
    assert "explicitly transitional" in text
    for section in PACKAGE_README_REQUIRED_SECTIONS:
        assert section in text
    for package in _package_packages():
        assert f"`{package}/`" in text, f"{package}/ is undocumented"
    for module in _package_modules():
        assert f"`{module}`" in text or f"`{module.removesuffix('.py')}`" in text, (
            f"{module} is undocumented"
        )
    canonical_phrases = (
        "`drift_kinetic.py` | The `KineticOperator`",
        "frozen-reference loading, Fortran/PETSc fixture readers",
        "must not be reintroduced",
    )
    for phrase in canonical_phrases:
        assert phrase in text


def test_package_readme_explains_public_surface_and_implementation_boundaries() -> None:
    """The source README should be enough to navigate the package during review."""

    text = PACKAGE_README.read_text(encoding="utf-8")
    expected_phrases = (
        "canonical stack of flat, physics-named root modules",
        "transitional interim owners while the vertical slices landed",
        "one folder below `dkx/`, no nested",
        "canonical root modules are the stable import surface",
        "Compatibility aliases may remain",
        "fetched through `validation.data_fetch` from release assets",
    )

    for phrase in expected_phrases:
        assert phrase in text


def test_package_tree_has_no_tracked_generated_or_large_runtime_outputs() -> None:
    """Keep the importable package light and independent of local run artifacts."""

    result = subprocess.run(
        ["git", "ls-files", "src/dkx"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    offenders: list[str] = []
    for line in result.stdout.splitlines():
        path = Path(line)
        if DISALLOWED_TRACKED_PACKAGE_PARTS.intersection(path.parts):
            offenders.append(line)
            continue
        if path.suffix in DISALLOWED_TRACKED_PACKAGE_SUFFIXES:
            offenders.append(line)

    assert offenders == []


def test_source_map_doc_describes_current_one_level_layout() -> None:
    """Keep contributor docs synchronized with the flattened package tree."""

    text = SOURCE_MAP_DOC.read_text(encoding="utf-8")

    assert "one level of domain folders" in text
    for package in _package_packages():
        assert f"``dkx/{package}``" in text, f"{package} missing from the source map"


def test_scripts_do_not_import_missing_sibling_modules() -> None:
    """Keep temporary scripts executable while they are promoted or deleted."""

    if not SCRIPT_ROOT.exists():
        return

    available_modules = {path.stem for path in SCRIPT_ROOT.glob("*.py")}
    offenders: list[tuple[str, str]] = []
    for path in sorted(SCRIPT_ROOT.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root_name = alias.name.split(".", 1)[0]
                    if root_name in available_modules or root_name.startswith("sfincs_"):
                        continue
                    candidate = SCRIPT_ROOT / f"{root_name}.py"
                    if candidate.parent == SCRIPT_ROOT and root_name.startswith(("audit_", "run_", "generate_")):
                        offenders.append((path.relative_to(REPO_ROOT).as_posix(), root_name))
            elif isinstance(node, ast.ImportFrom):
                if node.level != 0 or node.module is None:
                    continue
                root_name = node.module.split(".", 1)[0]
                if root_name in available_modules or root_name.startswith("sfincs_"):
                    continue
                if root_name.startswith(("audit_", "run_", "generate_")):
                    offenders.append((path.relative_to(REPO_ROOT).as_posix(), root_name))

    assert offenders == []


def test_package_sources_do_not_repeat_top_level_defs() -> None:
    """Repeated top-level definitions hide stale helpers in large owner files."""

    offenders: list[tuple[str, str, int, int]] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        seen: dict[str, int] = {}
        for node in tree.body:
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                if node.name in seen:
                    offenders.append(
                        (
                            path.relative_to(REPO_ROOT).as_posix(),
                            node.name,
                            seen[node.name],
                            node.lineno,
                        )
                    )
                seen[node.name] = node.lineno

    assert offenders == []


def test_canonical_root_modules_are_importable() -> None:
    """The canonical flat root modules replace the deleted legacy packages."""

    canonical_modules = (
        "dkx.drift_kinetic",
        "dkx.solve",
        "dkx.run",
        "dkx.writer",
        "dkx.phase_space",
        "dkx.magnetic_geometry",
        "dkx.moments",
        "dkx.collisions",
        "dkx.species",
        "dkx.phi1",
        "dkx.er",
        "dkx.solver_trace",
        "dkx.xgrid",
    )
    for module_name in canonical_modules:
        module = importlib.import_module(module_name)
        assert module.__name__ == module_name
