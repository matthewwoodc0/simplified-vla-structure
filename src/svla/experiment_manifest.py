from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import mujoco
import numpy as np
import scipy

from svla.pickup_task import PICKUP_CONTROLLER_LIMITS, pickup_physics_gate_constants
from svla.state_bc import write_json

MANIFEST_FORMAT = "svla_experiment_manifest_v1"

TRACKED_SOURCE_PATHS = (
    "assets/pickup_scene.xml",
    "assets/so101_arm.xml",
    "src/svla/pickup_task.py",
    "src/svla/controller.py",
    "src/svla/action_spaces.py",
    "src/svla/state_bc.py",
    "src/svla/evaluation_protocol.py",
    "src/svla/demo_recorder.py",
    "src/svla/experiment_manifest.py",
    "scripts/run_pickup_trials.py",
    "scripts/run_pick_place_trials.py",
    "scripts/validate_action_replay.py",
    "scripts/validate_task_robustness.py",
    "scripts/train_state_bc.py",
    "src/svla/pick_place_replay.py",
    "scripts/run_phase5_baseline_pytest.py",
    "scripts/build_phase5_baseline_aggregate.py",
    "configs/phase5_evaluation_protocol_v2.json",
)

MANIFEST_IDENTITY_FIELDS = (
    "git_commit_sha",
    "git_diff_sha256",
    "git_untracked_files",
    "source_hashes",
    "controller_limits",
    "physics_gate_constants",
    "versions",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _run_git(args: list[str], repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout


def _run_git_bytes(args: list[str], repo_root: Path) -> bytes | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            capture_output=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout


def git_commit_sha(repo_root: Path) -> str | None:
    output = _run_git(["rev-parse", "HEAD"], repo_root)
    return output.strip() if output else None


def git_porcelain_status(repo_root: Path) -> str | None:
    output = _run_git(["status", "--porcelain"], repo_root)
    if output is None:
        return None
    return output


def git_worktree_flags(porcelain: str) -> dict[str, bool]:
    has_staged = False
    has_unstaged = False
    has_untracked = False
    for line in porcelain.splitlines():
        if len(line) < 2:
            continue
        index_status, worktree_status = line[0], line[1]
        if index_status == "?" and worktree_status == "?":
            has_untracked = True
            continue
        if index_status not in (" ", "?"):
            has_staged = True
        if worktree_status not in (" ", "?"):
            has_unstaged = True
    return {
        "has_staged_changes": has_staged,
        "has_unstaged_changes": has_unstaged,
        "has_untracked_files": has_untracked,
    }


def git_dirty(repo_root: Path) -> bool | None:
    porcelain = git_porcelain_status(repo_root)
    if porcelain is None:
        return None
    return bool(porcelain.strip())


def git_untracked_files(repo_root: Path) -> list[dict[str, str]]:
    output = _run_git(["ls-files", "--others", "--exclude-standard"], repo_root)
    if output is None:
        return []
    records: list[dict[str, str]] = []
    for rel_path in sorted(line for line in output.splitlines() if line):
        file_path = repo_root / rel_path
        if file_path.is_file():
            records.append(
                {
                    "path": rel_path,
                    "sha256": sha256_file(file_path),
                }
            )
    return records


def git_diff_sha256(repo_root: Path) -> str | None:
    porcelain = git_porcelain_status(repo_root)
    if porcelain is None or not porcelain.strip():
        return None
    flags = git_worktree_flags(porcelain)
    if not flags["has_staged_changes"] and not flags["has_unstaged_changes"]:
        return None
    diff = _run_git_bytes(["diff", "--binary", "HEAD"], repo_root)
    if diff is None:
        return None
    return sha256_bytes(diff)


def capture_environment() -> dict[str, str]:
    environment: dict[str, str] = {}
    pythonpath = os.environ.get("PYTHONPATH")
    if pythonpath is not None:
        environment["PYTHONPATH"] = pythonpath
    return environment


def build_command_record(
    argv: list[str] | None = None,
    *,
    working_directory: Path | str | None = None,
    environment: dict[str, str] | None = None,
) -> dict:
    return {
        "argv": list(argv if argv is not None else sys.argv),
        "executable": sys.executable,
        "working_directory": str(
            Path(working_directory).resolve()
            if working_directory is not None
            else Path.cwd().resolve()
        ),
        "environment": capture_environment() if environment is None else dict(environment),
    }


def manifest_identity_slice(manifest: dict) -> dict:
    return {field: manifest.get(field) for field in MANIFEST_IDENTITY_FIELDS}


def verify_manifest_identity_consistent(manifests: list[dict]) -> tuple[bool, list[str]]:
    if not manifests:
        return False, ["no manifests provided"]
    reference = manifest_identity_slice(manifests[0])
    issues: list[str] = []
    for index, manifest in enumerate(manifests[1:], start=1):
        current = manifest_identity_slice(manifest)
        for field in MANIFEST_IDENTITY_FIELDS:
            if current[field] != reference[field]:
                issues.append(f"manifest[{index}] field {field} differs from manifest[0]")
    return not issues, issues


def verify_manifest_output_hashes(
    repo_root: Path,
    manifests: list[dict],
) -> tuple[bool, list[str]]:
    issues: list[str] = []
    repo_root = repo_root.resolve()
    for manifest_index, manifest in enumerate(manifests):
        for record in manifest.get("output_files", []):
            rel_path = record["path"]
            expected = record["sha256"]
            file_path = repo_root / rel_path
            if not file_path.is_file():
                issues.append(f"manifest[{manifest_index}] missing output file {rel_path}")
                continue
            actual = sha256_file(file_path)
            if actual != expected:
                issues.append(
                    f"manifest[{manifest_index}] hash mismatch for {rel_path}"
                )
    return not issues, issues


def runtime_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "mujoco": mujoco.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }


def tracked_source_hashes(repo_root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for rel_path in TRACKED_SOURCE_PATHS:
        path = repo_root / rel_path
        if path.is_file():
            hashes[rel_path] = sha256_file(path)
    return hashes


def pickup_controller_limits() -> dict:
    return asdict(PICKUP_CONTROLLER_LIMITS)


def physics_gate_constants() -> dict[str, float]:
    return pickup_physics_gate_constants()


def sidecar_path(output_path: Path) -> Path:
    return output_path.parent / f"{output_path.stem}.manifest.json"


def output_file_records(
    repo_root: Path,
    paths: Iterable[Path | str],
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_file():
            continue
        resolved = path.resolve()
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        try:
            rel_path = str(resolved.relative_to(repo_root.resolve()))
        except ValueError:
            rel_path = str(resolved)
        records.append(
            {
                "path": rel_path,
                "sha256": sha256_file(resolved),
            }
        )
    records.sort(key=lambda record: record["path"])
    return records


class ExperimentManifest:
    def __init__(
        self,
        *,
        repo_root: Path,
        command: dict,
        seeds: dict | None = None,
        metadata: dict | None = None,
        utc_timestamp: str | None = None,
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.command = dict(command)
        self.seeds = dict(seeds or {})
        self.metadata = dict(metadata or {})
        self.utc_timestamp = utc_timestamp or datetime.now(timezone.utc).isoformat()
        self._output_paths: list[Path] = []

    @classmethod
    def start(
        cls,
        *,
        repo_root: Path,
        argv: list[str] | None = None,
        command: dict | None = None,
        seeds: dict | None = None,
        metadata: dict | None = None,
    ) -> "ExperimentManifest":
        return cls(
            repo_root=repo_root,
            command=command if command is not None else build_command_record(argv),
            seeds=seeds,
            metadata=metadata,
        )

    def add_output(self, path: Path | str) -> None:
        self._output_paths.append(Path(path))

    def add_outputs(self, paths: Iterable[Path | str]) -> None:
        for path in paths:
            self.add_output(path)

    def build(self) -> dict:
        porcelain = git_porcelain_status(self.repo_root) or ""
        worktree = git_worktree_flags(porcelain)
        dirty = bool(porcelain.strip()) if porcelain is not None else None
        return {
            "format": MANIFEST_FORMAT,
            "utc_timestamp": self.utc_timestamp,
            "command": self.command,
            "git_commit_sha": git_commit_sha(self.repo_root),
            "git_dirty": dirty,
            "git_has_staged_changes": worktree["has_staged_changes"],
            "git_has_unstaged_changes": worktree["has_unstaged_changes"],
            "git_has_untracked_files": worktree["has_untracked_files"],
            "git_diff_sha256": git_diff_sha256(self.repo_root),
            "git_untracked_files": git_untracked_files(self.repo_root),
            "versions": runtime_versions(),
            "source_hashes": tracked_source_hashes(self.repo_root),
            "controller_limits": pickup_controller_limits(),
            "physics_gate_constants": physics_gate_constants(),
            "seeds": self.seeds,
            "metadata": self.metadata,
            "output_files": output_file_records(self.repo_root, self._output_paths),
        }

    def write_sidecar(self, anchor_path: Path | str) -> Path:
        manifest_path = sidecar_path(Path(anchor_path))
        write_json(manifest_path, self.build())
        return manifest_path
