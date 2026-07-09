from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCRATCH = PROJECT_ROOT / "outputs" / "pick_place_vplan_evidence"
PYTHON = [str(PROJECT_ROOT / ".venv/bin/python")]
ENV = {"PYTHONPATH": str(PROJECT_ROOT / "src")}


def run_logged(command: list[str], log_path: Path) -> None:
    proc = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env={**dict(**__import__("os").environ), **ENV},
        capture_output=True,
        text=True,
    )
    log_path.write_text(
        f"$ {' '.join(command)}\n{proc.stdout}{proc.stderr}",
        encoding="utf-8",
    )
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def write_controller_note(compare_path: Path, note_path: Path) -> None:
    data = json.loads(compare_path.read_text(encoding="utf-8"))
    pickup = data["pickup"]
    place = data["pick_place"]
    pickup_ee = pickup["replay"]["ee_tool_delta"]
    place_ee = place["replay"]["ee_tool_delta"]
    lines = [
        "Controller complexity: pickup vs pick-and-place scripted expert",
        f"Evidence: {compare_path.name} via validate_action_replay.py --task compare",
        f"Pickup phases={pickup['phase_count']} samples={pickup['sample_count']}",
        f"Pick-place phases={place['phase_count']} samples={place['sample_count']}",
        f"EE replay saturation pickup={pickup_ee['saturation_rate']:.1%} pick_place={place_ee['saturation_rate']:.1%}",
        f"Replay event_order_valid pickup={pickup_ee['event_order_valid']} pick_place={place_ee['event_order_valid']}",
        f"Boundary index={place['grasp_segment_finalize_sample_index']} in demo metadata",
        "Scripting stays move-EE + gripper; longer demo composes pickup phases.",
        "Pickup BC failures were learned-policy timing/labels, not scripting difficulty.",
        "Left placement uses separate goal/command markers (test_pick_place_task ablation).",
    ]
    note_path.write_text("\n".join(lines[:15]) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scratch-dir",
        type=Path,
        default=DEFAULT_SCRATCH,
        help="Directory for logs and copied evidence summaries.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scratch = args.scratch_dir.resolve()
    scratch.mkdir(parents=True, exist_ok=True)
    compare_path = PROJECT_ROOT / "outputs" / "action_replay_pick_place_compare.json"

    run_logged(
        PYTHON + ["-m", "pytest", "tests/test_pickup_task.py", "tests/test_demo_recorder.py", "-q"],
        scratch / "pickup_regression.log",
    )
    run_logged(
        PYTHON
        + [
            "-m",
            "pytest",
            "tests/test_pick_place_task.py",
            "tests/test_pick_place_replay.py",
            "-q",
        ],
        scratch / "pick_place_regression.log",
    )
    run_logged(
        PYTHON
        + [
            str(PROJECT_ROOT / "scripts/run_pick_place_trials.py"),
            "--output",
            str(PROJECT_ROOT / "outputs/pick_place_trials.jsonl"),
        ],
        scratch / "pick_place_trials.log",
    )
    run_logged(
        PYTHON
        + [
            str(PROJECT_ROOT / "scripts/record_pick_place_demo.py"),
            "--output",
            str(PROJECT_ROOT / "outputs/pick_place_demo_center_to_right.json"),
        ],
        scratch / "pick_place_demo.log",
    )
    run_logged(
        PYTHON
        + [
            str(PROJECT_ROOT / "scripts/validate_action_replay.py"),
            "--task",
            "compare",
            "--output",
            str(compare_path),
        ],
        scratch / "replay_saturation.log",
    )
    (scratch / "replay_saturation_summary.json").write_text(
        compare_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    run_logged(
        PYTHON
        + [
            str(PROJECT_ROOT / "scripts/validate_task_robustness.py"),
            "--domain",
            "readiness",
            "--output",
            str(PROJECT_ROOT / "outputs/task_robustness_readiness_summary.json"),
        ],
        scratch / "readiness.log",
    )
    write_controller_note(compare_path, scratch / "controller_note.txt")
    print(f"wrote evidence under {scratch}")


if __name__ == "__main__":
    main()
