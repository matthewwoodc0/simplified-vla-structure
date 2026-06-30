from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from svla.demo_recorder import PickupDemoRecorder
from svla.pickup_task import default_trial_specs


def run(output_dir: Path, count: int) -> dict:
    specs = [
        spec
        for spec in default_trial_specs(repeats=1)
        if spec.approach.label == "vertical_pregrasp"
    ][:count]
    recorder = PickupDemoRecorder()
    output_dir.mkdir(parents=True, exist_ok=True)
    demos = []
    for spec in specs:
        path = output_dir / (
            f"pickup_demo_{spec.trial_id:02d}_{spec.orientation.label}_"
            f"{spec.object_pose.label}_{spec.approach.label}.json"
        )
        demo = recorder.write_trial(spec, path)
        demos.append({"path": str(path), "summary": demo["summary"]})
        print(
            f"demo={path.name} success={int(demo['summary']['success'])} "
            f"failure={demo['summary']['failure_category']} samples={len(demo['samples'])}"
        )

    manifest = {
        "format": "svla_pickup_demo_manifest_v1",
        "demo_count": len(demos),
        "demos": demos,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {manifest_path}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "scripted_pickup_demos",
    )
    parser.add_argument("--count", type=int, default=3)
    args = parser.parse_args()
    run(args.output_dir, args.count)


if __name__ == "__main__":
    main()
