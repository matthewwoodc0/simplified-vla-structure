from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from svla.experiment_manifest import ExperimentManifest, build_command_record, sidecar_path
from svla.state_bc import write_json


def run(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    pytest_argv = [sys.executable, "-m", "pytest", "-q"]
    subprocess_env = os.environ.copy()
    subprocess_env["PYTHONPATH"] = "src"
    result = subprocess.run(
        pytest_argv,
        cwd=PROJECT_ROOT,
        env=subprocess_env,
        capture_output=True,
        text=True,
    )
    log_path = output_dir / "pytest_results.txt"
    log_path.write_text(result.stdout + result.stderr, encoding="utf-8")

    passed = None
    match = re.search(r"(\d+) passed", result.stdout + result.stderr)
    if match:
        passed = int(match.group(1))
    failed = None
    fail_match = re.search(r"(\d+) failed", result.stdout + result.stderr)
    if fail_match:
        failed = int(fail_match.group(1))

    command = build_command_record(
        pytest_argv,
        working_directory=PROJECT_ROOT,
        environment={"PYTHONPATH": subprocess_env["PYTHONPATH"]},
    )
    manifest = ExperimentManifest.start(repo_root=PROJECT_ROOT, command=command)
    manifest.add_output(log_path)
    payload = manifest.build()
    payload["pytest_exit_code"] = int(result.returncode)
    payload["pytest_passed"] = result.returncode == 0
    payload["pytest_tests_passed"] = passed
    payload["pytest_tests_failed"] = 0 if result.returncode == 0 else failed
    manifest_path = sidecar_path(log_path)
    write_json(manifest_path, payload)
    print(result.stdout)
    print(result.stderr, file=sys.stderr)
    print(f"wrote {log_path}")
    print(f"wrote {manifest_path}")
    if result.returncode != 0:
        raise SystemExit(result.returncode)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "phase5_baseline_v2",
    )
    args = parser.parse_args()
    run(args.output_dir)


if __name__ == "__main__":
    main()