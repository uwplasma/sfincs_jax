from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_publish_workflow_tests_the_artifact_downloaded_from_pypi() -> None:
    workflow = (ROOT / ".github/workflows/publish-pypi.yml").read_text()
    smoke = workflow.split("  pypi-smoke:", 1)[1]

    assert "needs: build-and-publish" in smoke
    assert "working-directory: ${{ runner.temp }}" in smoke
    assert "--index-url https://pypi.org/simple" in smoke
    assert "--only-binary=:all:" in smoke
    assert "site-packages" in smoke
    assert "result = dkx.run(" in smoke
    assert "np.isfinite(particle_flux)" in smoke
    assert "pip install ." not in smoke


def test_publish_workflow_uses_the_single_version_source() -> None:
    workflow = (ROOT / ".github/workflows/publish-pypi.yml").read_text()

    assert 'Path("src/dkx/_version.py")' in workflow
    assert '["project"]["version"]' not in workflow
    assert 'Path("dkx/__init__.py")' not in workflow


def test_required_ci_covers_sdist_sizes_and_workflow_syntax() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()
    aggregate = workflow.split("  tests:", 1)[1]

    assert "python -m build --outdir dist" in workflow
    assert "python -m twine check dist/*" in workflow
    assert '"${RUNNER_TEMP}/sdistenv/bin/python"' in workflow
    assert "tools/installed_scientific_smoke.py" in workflow
    assert "validation/package_size_contract.toml" in workflow
    assert 'fetch --quiet origin "${GITHUB_REF}"' in workflow
    assert 'checkout --quiet --detach "${GITHUB_SHA}"' in workflow
    assert "name: package-sizes" in workflow
    assert "actionlint_1.7.12_linux_amd64.tar.gz" not in workflow
    assert 'ACTIONLINT_VERSION: "1.7.12"' in workflow
    assert "- workflow-lint" in aggregate
    assert "- wheel-install" in aggregate
