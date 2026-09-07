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
    assert 'fetch --quiet "${GITHUB_WORKSPACE}" "${GITHUB_SHA}"' in workflow
    assert "fetch-depth: 0" in workflow.split("  wheel-install:", 1)[1]
    assert 'checkout --quiet --detach "${GITHUB_SHA}"' in workflow
    assert "name: package-sizes" in workflow
    assert "actionlint_1.7.12_linux_amd64.tar.gz" not in workflow
    assert 'ACTIONLINT_VERSION: "1.7.12"' in workflow
    assert "- workflow-lint" in aggregate
    assert "- wheel-install" in aggregate



def test_release_clone_keeps_tested_tree_after_remote_pr_commit_is_pruned(tmp_path):
    import os
    import subprocess
    import textwrap

    env = dict(os.environ, GIT_AUTHOR_NAME="rogeriojorge", GIT_COMMITTER_NAME="rogeriojorge",
               GIT_AUTHOR_EMAIL="rogerio.jorge@wisc.edu", GIT_COMMITTER_EMAIL="rogerio.jorge@wisc.edu")
    def git(*args):
        return subprocess.check_output(["git", *map(str, args)], env=env, text=True).strip()

    source = tmp_path / "checkout"
    remote = tmp_path / "origin.git"
    git("init", "-q", "-b", "main", source)
    (source / "payload").write_text("base")
    git("-C", source, "add", "payload")
    git("-C", source, "commit", "-qm", "base")
    base = git("-C", source, "rev-parse", "HEAD")
    git("clone", "-q", "--bare", source, remote)
    # The runner retains a full checkout of the old PR tree.
    (source / "payload").write_text("tested PR tree")
    git("-C", source, "commit", "-qam", "tested PR")
    head = git("-C", source, "rev-parse", "HEAD")
    tested = git("-C", source, "commit-tree", "HEAD^{tree}", "-p", base, "-p", head,
                 "-m", "tested synthetic merge")
    git("-C", source, "checkout", "-q", "--detach", tested)
    git("-C", source, "push", "-q", remote, "HEAD:refs/pull/1/merge")
    # GitHub rewrites the ref and eventually removes the old object.
    git("--git-dir", remote, "update-ref", "refs/pull/1/merge", base)
    git("--git-dir", remote, "gc", "--prune=now")
    missing = subprocess.run(["git", "--git-dir", str(remote), "cat-file", "-e", tested],
                             capture_output=True)
    assert missing.returncode != 0
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()
    step = workflow.split("      - name: Measure every DKX-owned release artifact", 1)[1]
    commands = textwrap.dedent(step.split("        run: |", 1)[1].split(
        '"${RUNNER_TEMP}/wheelenv/bin/python"', 1)[0])
    env.update(GITHUB_SERVER_URL=str(tmp_path), GITHUB_REPOSITORY="origin",
               GITHUB_WORKSPACE=str(source), GITHUB_SHA=tested, RUNNER_TEMP=str(tmp_path))
    subprocess.run(["bash", "-eu", "-c", commands], env=env, check=True)
    clone = tmp_path / "full-clone"
    assert git("-C", clone, "rev-parse", "HEAD") == tested
    assert (clone / "payload").read_text() == "tested PR tree"
    assert git("-C", clone, "rev-parse", "--is-shallow-repository") == "false"
