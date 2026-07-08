from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from svla.experiment_manifest import (
    ExperimentManifest,
    MANIFEST_FORMAT,
    build_command_record,
    git_diff_sha256,
    git_untracked_files,
    git_worktree_flags,
    physics_gate_constants,
    pickup_controller_limits,
    sha256_bytes,
    sha256_file,
    sha256_text,
    sidecar_path,
    tracked_source_hashes,
    verify_manifest_identity_consistent,
    verify_manifest_output_hashes,
)
from svla.pickup_task import PICKUP_CONTROLLER_LIMITS


def _init_git_repo(path: Path) -> None:
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=path,
        check=True,
        capture_output=True,
    )


def _commit_all(repo_root: Path, message: str) -> None:
    subprocess.run(["git", "add", "-A"], cwd=repo_root, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", message],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )


def _git_diff_binary_head(repo_root: Path) -> bytes:
    result = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        check=True,
    )
    return result.stdout


def test_sha256_helpers(tmp_path: Path):
    payload = tmp_path / "payload.txt"
    payload.write_text("hello manifest\n", encoding="utf-8")

    assert sha256_text("hello manifest\n") == sha256_file(payload)


def test_tracked_source_hashes_include_manifest_and_scripts(tmp_path: Path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    rel_paths = (
        "assets/pickup_scene.xml",
        "src/svla/experiment_manifest.py",
        "scripts/validate_task_robustness.py",
    )
    for rel_path in rel_paths:
        path = repo_root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"content for {rel_path}\n", encoding="utf-8")

    hashes = tracked_source_hashes(repo_root)

    assert set(hashes) >= set(rel_paths)
    assert hashes["src/svla/experiment_manifest.py"] == sha256_file(
        repo_root / "src/svla/experiment_manifest.py"
    )


def test_pickup_provenance_helpers_match_evaluator_limits():
    limits = pickup_controller_limits()
    gates = physics_gate_constants()

    assert limits["max_step_xyz"] == PICKUP_CONTROLLER_LIMITS.max_step_xyz
    assert limits["orientation_mode"] == "tool_axis"
    assert gates["MAX_GRIPPER_CONTACT_FORCE"] == 22.0
    assert gates["MAX_SUPPORTED_XY_DISPLACEMENT"] == 0.013


def test_build_command_record_includes_executable_cwd_and_pythonpath(
    tmp_path: Path,
    monkeypatch,
):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    monkeypatch.chdir(repo_root)
    monkeypatch.setenv("PYTHONPATH", "src")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(repo_root / ".venv/bin/python"),
            "scripts/validate_task_robustness.py",
            "--domain",
            "readiness",
        ],
    )

    command = build_command_record(sys.argv)

    assert command["argv"] == [
        str(repo_root / ".venv/bin/python"),
        "scripts/validate_task_robustness.py",
        "--domain",
        "readiness",
    ]
    assert command["executable"] == sys.executable
    assert command["working_directory"] == str(repo_root.resolve())
    assert command["environment"] == {"PYTHONPATH": "src"}


def test_manifest_records_clean_git_state_and_output_hashes(tmp_path: Path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _init_git_repo(repo_root)

    (repo_root / ".gitignore").write_text("outputs/\n", encoding="utf-8")
    tracked = repo_root / "src/svla/pickup_task.py"
    tracked.parent.mkdir(parents=True, exist_ok=True)
    tracked.write_text("tracked source\n", encoding="utf-8")
    _commit_all(repo_root, "initial")

    output = repo_root / "outputs" / "summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('{"ok": true}\n', encoding="utf-8")

    manifest = ExperimentManifest.start(
        repo_root=repo_root,
        argv=["python", "scripts/run_pickup_trials.py", "--repeats", "1"],
        seeds={"repeats": 1},
        metadata={"evaluation_config_hash": "abc123"},
    )
    manifest.add_output(output)
    manifest_path = manifest.write_sidecar(output)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert payload["format"] == MANIFEST_FORMAT
    assert payload["command"]["argv"] == [
        "python",
        "scripts/run_pickup_trials.py",
        "--repeats",
        "1",
    ]
    assert payload["command"]["executable"] == sys.executable
    assert payload["command"]["working_directory"] == str(Path.cwd().resolve())
    assert payload["git_dirty"] is False
    assert payload["git_has_staged_changes"] is False
    assert payload["git_has_unstaged_changes"] is False
    assert payload["git_has_untracked_files"] is False
    assert payload["git_diff_sha256"] is None
    assert payload["git_untracked_files"] == []
    assert payload["git_commit_sha"] is not None
    assert payload["versions"]["numpy"] == np.__version__
    assert payload["controller_limits"]["max_joint_step"] == PICKUP_CONTROLLER_LIMITS.max_joint_step
    assert payload["physics_gate_constants"]["MAX_GRIPPER_IMPULSE_BEFORE_LIFT"] == 9.0
    assert payload["seeds"] == {"repeats": 1}
    assert payload["metadata"] == {"evaluation_config_hash": "abc123"}
    assert payload["output_files"] == [
        {
            "path": "outputs/summary.json",
            "sha256": sha256_file(output),
        }
    ]
    assert manifest_path == sidecar_path(output)


def test_git_diff_sha256_hashes_only_binary_head_for_staged_changes(tmp_path: Path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _init_git_repo(repo_root)

    tracked = repo_root / "tracked.txt"
    tracked.write_text("baseline\n", encoding="utf-8")
    _commit_all(repo_root, "initial")

    tracked.write_text("staged edit\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repo_root, check=True, capture_output=True)

    flags = git_worktree_flags(subprocess.check_output(["git", "status", "--porcelain"], cwd=repo_root, text=True))
    assert flags == {
        "has_staged_changes": True,
        "has_unstaged_changes": False,
        "has_untracked_files": False,
    }
    assert git_diff_sha256(repo_root) == sha256_bytes(_git_diff_binary_head(repo_root))
    assert git_untracked_files(repo_root) == []


def test_git_diff_sha256_hashes_only_binary_head_for_unstaged_changes(tmp_path: Path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _init_git_repo(repo_root)

    tracked = repo_root / "tracked.txt"
    tracked.write_text("baseline\n", encoding="utf-8")
    _commit_all(repo_root, "initial")

    tracked.write_text("unstaged edit\n", encoding="utf-8")

    flags = git_worktree_flags(subprocess.check_output(["git", "status", "--porcelain"], cwd=repo_root, text=True))
    assert flags == {
        "has_staged_changes": False,
        "has_unstaged_changes": True,
        "has_untracked_files": False,
    }
    assert git_diff_sha256(repo_root) == sha256_bytes(_git_diff_binary_head(repo_root))
    assert git_untracked_files(repo_root) == []


def test_git_untracked_files_recorded_separately_without_diff_hash(tmp_path: Path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _init_git_repo(repo_root)

    tracked = repo_root / "tracked.txt"
    tracked.write_text("baseline\n", encoding="utf-8")
    _commit_all(repo_root, "initial")

    untracked = repo_root / "scratch.txt"
    untracked.write_text("new file\n", encoding="utf-8")

    flags = git_worktree_flags(subprocess.check_output(["git", "status", "--porcelain"], cwd=repo_root, text=True))
    assert flags == {
        "has_staged_changes": False,
        "has_unstaged_changes": False,
        "has_untracked_files": True,
    }
    assert git_diff_sha256(repo_root) is None
    assert git_untracked_files(repo_root) == [
        {"path": "scratch.txt", "sha256": sha256_file(untracked)}
    ]


def test_manifest_records_staged_unstaged_and_untracked_independently(tmp_path: Path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _init_git_repo(repo_root)

    tracked = repo_root / "tracked.txt"
    tracked.write_text("baseline\n", encoding="utf-8")
    _commit_all(repo_root, "initial")

    tracked.write_text("staged and unstaged\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repo_root, check=True, capture_output=True)
    tracked.write_text("extra unstaged\n", encoding="utf-8")
    untracked = repo_root / "scratch.txt"
    untracked.write_text("new file\n", encoding="utf-8")

    manifest = ExperimentManifest.start(
        repo_root=repo_root,
        argv=["python", "scripts/validate_task_robustness.py"],
        seeds={"random_cases_seed": 7},
    )
    payload = manifest.build()

    assert payload["git_dirty"] is True
    assert payload["git_has_staged_changes"] is True
    assert payload["git_has_unstaged_changes"] is True
    assert payload["git_has_untracked_files"] is True
    assert payload["git_diff_sha256"] == sha256_bytes(_git_diff_binary_head(repo_root))
    assert payload["git_untracked_files"] == [
        {"path": "scratch.txt", "sha256": sha256_file(untracked)}
    ]


def test_build_command_record_accepts_explicit_cwd_and_environment(tmp_path: Path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    command = build_command_record(
        [sys.executable, "-m", "pytest", "-q"],
        working_directory=repo_root,
        environment={"PYTHONPATH": "src"},
    )
    assert command["argv"] == [sys.executable, "-m", "pytest", "-q"]
    assert command["working_directory"] == str(repo_root.resolve())
    assert command["environment"] == {"PYTHONPATH": "src"}


def test_verify_manifest_identity_consistent_detects_mismatch():
    base = {
        "git_commit_sha": "abc",
        "git_diff_sha256": None,
        "git_untracked_files": [],
        "source_hashes": {"src/svla/pickup_task.py": "111"},
        "controller_limits": {"max_step_xyz": 0.019},
        "physics_gate_constants": {"MAX_GRIPPER_CONTACT_FORCE": 22.0},
        "versions": {"python": "3.12.13"},
    }
    ok, issues = verify_manifest_identity_consistent([base, dict(base)])
    assert ok
    assert issues == []

    mismatched = dict(base)
    mismatched["versions"] = {"python": "3.11.0"}
    ok, issues = verify_manifest_identity_consistent([base, mismatched])
    assert not ok
    assert any("versions" in issue for issue in issues)

    untracked_mismatch = dict(base)
    untracked_mismatch["git_untracked_files"] = [
        {"path": "new.py", "sha256": "2" * 64}
    ]
    ok, issues = verify_manifest_identity_consistent([base, untracked_mismatch])
    assert not ok
    assert any("git_untracked_files" in issue for issue in issues)


def test_verify_manifest_output_hashes_detects_missing_and_mismatched(tmp_path: Path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    output = repo_root / "outputs" / "summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('{"ok": true}\n', encoding="utf-8")
    manifest = {
        "output_files": [
            {
                "path": "outputs/summary.json",
                "sha256": sha256_file(output),
            }
        ]
    }
    ok, issues = verify_manifest_output_hashes(repo_root, [manifest])
    assert ok
    assert issues == []

    bad_hash = dict(manifest)
    bad_hash["output_files"] = [
        {"path": "outputs/summary.json", "sha256": "0" * 64}
    ]
    ok, issues = verify_manifest_output_hashes(repo_root, [bad_hash])
    assert not ok
    assert any("hash mismatch" in issue for issue in issues)

    missing = {
        "output_files": [{"path": "outputs/missing.json", "sha256": "0" * 64}]
    }
    ok, issues = verify_manifest_output_hashes(repo_root, [missing])
    assert not ok
    assert any("missing output file" in issue for issue in issues)
