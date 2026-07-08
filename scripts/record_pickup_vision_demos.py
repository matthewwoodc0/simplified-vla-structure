from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from svla.experiment_manifest import ExperimentManifest
from svla.pickup_task import default_trial_specs
from svla.vision_dataset import record_pickup_vision_dataset
from svla.vision_observations import FixedCameraConfig


def run(output_dir: Path, count: int, width: int, height: int, camera: str) -> dict:
    specs = [
        spec
        for spec in default_trial_specs(repeats=1)
        if spec.approach.label == "vertical_pregrasp"
    ][:count]
    camera_configs = (FixedCameraConfig(name=camera, width=width, height=height),)
    manifest = record_pickup_vision_dataset(
        specs,
        output_dir,
        camera_configs=camera_configs,
    )
    manifest_recorder = ExperimentManifest.start(
        repo_root=PROJECT_ROOT,
        argv=sys.argv,
        seeds={"trial_ids": [spec.trial_id for spec in specs]},
        metadata={
            "dataset_format": manifest["format"],
            "camera_config": manifest["camera_config"],
            "phase": "6a_vision_infrastructure",
            "no_policy_training": True,
        },
    )
    for record in manifest["output_files"]:
        manifest_recorder.add_output(output_dir / record["path"])
    manifest_recorder.add_output(output_dir / "vision_manifest.json")
    sidecar = manifest_recorder.write_sidecar(output_dir / "vision_manifest.json")
    print(
        f"wrote dataset={output_dir} episodes={manifest['episode_count']} "
        f"sidecar={sidecar}"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "phase6a_vision_sample",
    )
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--camera", default="overview")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    args = parser.parse_args()
    run(args.output_dir, args.count, args.width, args.height, args.camera)


if __name__ == "__main__":
    main()
