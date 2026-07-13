from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from svla.demo_recorder import PickupDemoRecorder
from svla.experiment_manifest import ExperimentManifest
from svla.pick_place_replay import ACTION_SPACES, LABEL_SOURCES, replay_demo_policy_labels
from svla.pickup_task import (
    OBJECT_START_Z,
    ApproachStrategy,
    GraspOrientation,
    ObjectStartPose,
    PickPlaceTrialSpec,
    PickupTaskEvaluator,
    PlacementTarget,
    default_pick_place_trial_specs,
    default_trial_specs,
)
from svla.state_bc import write_json


def run(args: argparse.Namespace, *, command: list[str] | None = None) -> dict:
    manifest = ExperimentManifest.start(
        repo_root=PROJECT_ROOT,
        argv=command,
        seeds={"repeats": args.repeats, "task": args.task},
    )
    if args.task == "compare":
        summary = run_compare(args)
        manifest.add_output(args.output)
        manifest_path = manifest.write_sidecar(args.output)
        print(f"wrote {manifest_path}")
        return summary
    if args.task == "pick_place":
        demos = [
            (spec, PickupDemoRecorder(PickupTaskEvaluator()).record_pick_place_trial(spec))
            for spec in default_pick_place_trial_specs()[: args.demo_count]
        ]
        demo_format = "svla_pick_place_demo_v1"
    else:
        demos = [
            (spec, PickupDemoRecorder(PickupTaskEvaluator()).record_trial(spec))
            for spec in default_trial_specs(repeats=args.repeats)
        ]
        demo_format = "svla_pickup_demo_v3_physics_audit"

    by_action_space = {}
    for action_space in ACTION_SPACES:
        trials = [
            replay_demo_policy_labels(
                demo,
                action_space,
                np.asarray(spec.object_pose.xyz, dtype=float),
                task=args.task,
                label_source=args.label_source,
            )
            for spec, demo in demos
        ]
        by_action_space[action_space] = _summarize_trials(trials)

    summary = {
        "format": "svla_action_replay_v1",
        "task": args.task,
        "demo_format": demo_format,
        "demo_count": len(demos),
        "label_source": args.label_source,
        "by_action_space": by_action_space,
        "pass": _replay_passes(args.task, by_action_space),
    }
    write_json(args.output, summary)
    manifest.add_output(args.output)
    manifest_path = manifest.write_sidecar(args.output)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {manifest_path}")
    if args.require_pass and not summary["pass"]:
        raise SystemExit(1)
    return summary


def run_compare(args: argparse.Namespace) -> dict:
    recorder = PickupDemoRecorder(PickupTaskEvaluator())
    pickup_demo = recorder.record_trial(default_trial_specs(repeats=1)[0])
    place_demo = recorder.record_pick_place_trial(
        PickPlaceTrialSpec(
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
    )
    pickup_start = np.asarray(pickup_demo["summary"]["object_start_pose"], dtype=float)
    place_start = np.asarray(place_demo["summary"]["object_start_pose"], dtype=float)
    comparison = {
        "format": "svla_action_replay_compare_v1",
        "pickup": {
            "phase_count": len(pickup_demo["phase_summaries"]),
            "phases": [phase["phase"] for phase in pickup_demo["phase_summaries"]],
            "sample_count": len(pickup_demo["samples"]),
            "replay": {
                space: replay_demo_policy_labels(pickup_demo, space, pickup_start, task="pickup")
                for space in ACTION_SPACES
            },
        },
        "pick_place": {
            "phase_count": len(place_demo["phase_summaries"]),
            "phases": [phase["phase"] for phase in place_demo["phase_summaries"]],
            "sample_count": len(place_demo["samples"]),
            "grasp_segment_finalize_sample_index": place_demo["metadata"][
                "grasp_segment_finalize_sample_index"
            ],
            "replay": {
                space: replay_demo_policy_labels(place_demo, space, place_start, task="pick_place")
                for space in ACTION_SPACES
            },
        },
    }
    write_json(args.output, comparison)
    print(json.dumps(comparison, indent=2, sort_keys=True))
    return comparison


def _summarize_trials(trials: list[dict]) -> dict:
    return {
        "successes": sum(trial["success"] for trial in trials),
        "total": len(trials),
        "mean_saturation_rate": _mean(trial["saturation_rate"] for trial in trials),
        "mean_joint_limit_rate": _mean(
            trial["joint_limit_clipped_steps"] / trial["steps"] for trial in trials
        ),
        "mean_joint_step_rate": _mean(
            trial["joint_step_clipped_steps"] / trial["steps"] for trial in trials
        ),
        "mean_infeasible_rate": _mean(
            trial["infeasible_steps"] / trial["steps"] for trial in trials
        ),
        "controller_failure_steps": sum(trial["controller_failure_steps"] for trial in trials),
        "collision_free_approaches": sum(trial["collision_free_approach"] for trial in trials),
        "valid_event_orders": sum(trial["event_order_valid"] for trial in trials),
        "physical_sanity_passes": sum(trial["physical_sanity_pass"] for trial in trials),
        "max_gripper_contact_force": max(trial["max_gripper_contact_force"] for trial in trials),
        "max_gripper_contact_impulse_before_lift": max(
            trial["gripper_contact_impulse_before_lift"] for trial in trials
        ),
        "max_object_xy_displacement_while_supported": max(
            trial["max_object_xy_displacement_while_supported"] for trial in trials
        ),
        "preclose_contact_steps": sum(trial["preclose_contact_steps"] for trial in trials),
        "max_preclose_object_displacement": max(
            trial["preclose_max_object_displacement"] for trial in trials
        ),
        "placement_achieved": sum(
            bool(trial["placement_achieved"])
            for trial in trials
            if trial.get("placement_achieved") is not None
        ),
        "trials": trials,
    }


def _replay_passes(task: str, by_action_space: dict) -> bool:
    if task == "pick_place":
        return all(
            result["successes"] == result["total"]
            and result["controller_failure_steps"] == 0
            and result["collision_free_approaches"] == result["total"]
            and result["valid_event_orders"] == result["total"]
            and result["physical_sanity_passes"] == result["total"]
            and result["preclose_contact_steps"] == 0
            and result["placement_achieved"] == result["total"]
            for result in by_action_space.values()
        )
    return all(
        result["successes"] == result["total"]
        and result["controller_failure_steps"] == 0
        and result["collision_free_approaches"] == result["total"]
        and result["valid_event_orders"] == result["total"]
        and result["physical_sanity_passes"] == result["total"]
        and result["preclose_contact_steps"] == 0
        for result in by_action_space.values()
    )


def _mean(values) -> float:
    values = list(values)
    return float(np.mean(values)) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--task",
        choices=("pickup", "pick_place", "compare"),
        default="pickup",
    )
    parser.add_argument(
        "--demo-count",
        type=int,
        default=1,
        help="Number of pick-and-place demos to record when --task pick_place.",
    )
    parser.add_argument(
        "--label-source",
        choices=LABEL_SOURCES,
        default="policy_labels",
        help="Explicit recorded label stream to execute; default preserves prior behavior.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "action_replay_tool_axis_summary.json",
    )
    parser.add_argument(
        "--require-pass",
        action="store_true",
        help="Exit non-zero when replay gates fail.",
    )
    run(parser.parse_args(), command=sys.argv)


if __name__ == "__main__":
    main()
