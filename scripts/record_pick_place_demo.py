from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from svla.demo_recorder import PickupDemoRecorder
from svla.pickup_task import (
    OBJECT_START_Z,
    ApproachStrategy,
    GraspOrientation,
    ObjectStartPose,
    PickPlaceTrialSpec,
    PickupTaskEvaluator,
    PlacementTarget,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "pick_place_demo_center_to_right.json",
    )
    args = parser.parse_args()

    spec = PickPlaceTrialSpec(
        trial_id=1,
        orientation=GraspOrientation("yaw_0", 0.0),
        object_pose=ObjectStartPose("center", np.array([0.0, -0.235, OBJECT_START_Z])),
        approach=ApproachStrategy("vertical_pregrasp", "world_z"),
        placement_target=PlacementTarget(
            "place_right",
            "place_right_marker",
            "place_right_marker",
        ),
    )
    demo = PickupDemoRecorder(PickupTaskEvaluator()).write_pick_place_trial(spec, args.output)
    sample = demo["samples"][0]
    print(f"demo={args.output} success={demo['summary']['success']} samples={len(demo['samples'])}")
    print(
        "grasp_segment_finalize_sample_index="
        f"{demo['metadata']['grasp_segment_finalize_sample_index']}"
    )
    for field in ("joint_delta", "ee_tool_delta"):
        print(f"labels.{field}=present len={len(sample['labels'][field])}")
        print(f"policy_labels.{field}=present len={len(sample['policy_labels'][field])}")
    print(json.dumps({"label_contract_ok": True}, sort_keys=True))


if __name__ == "__main__":
    main()