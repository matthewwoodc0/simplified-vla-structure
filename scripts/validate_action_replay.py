from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from svla.demo_recorder import PickupDemoRecorder
from svla.pickup_task import LIFT_CLEARANCE, RETENTION_CLEARANCE, PickupTaskEvaluator, default_trial_specs
from svla.state_bc import write_json


ACTION_SPACES = ("joint_delta", "ee_tool_delta")


def run(args: argparse.Namespace) -> dict:
    demos = [
        (spec, PickupDemoRecorder(PickupTaskEvaluator()).record_trial(spec))
        for spec in default_trial_specs(repeats=args.repeats)
    ]
    by_action_space = {}
    for action_space in ACTION_SPACES:
        trials = [
            _replay_demo(demo, action_space, np.asarray(spec.object_pose.xyz, dtype=float))
            for spec, demo in demos
        ]
        by_action_space[action_space] = {
            "successes": sum(trial["success"] for trial in trials),
            "total": len(trials),
            "mean_saturation_rate": _mean(
                trial["saturated_steps"] / trial["steps"] for trial in trials
            ),
            "mean_joint_limit_rate": _mean(
                trial["joint_limit_clipped_steps"] / trial["steps"] for trial in trials
            ),
            "mean_joint_step_rate": _mean(
                trial["joint_step_clipped_steps"] / trial["steps"] for trial in trials
            ),
            "mean_infeasible_rate": _mean(
                trial["infeasible_steps"] / trial["steps"] for trial in trials
            ),
            "controller_failure_steps": sum(
                trial["controller_failure_steps"] for trial in trials
            ),
            "collision_free_approaches": sum(
                trial["collision_free_approach"] for trial in trials
            ),
            "preclose_contact_steps": sum(
                trial["preclose_contact_steps"] for trial in trials
            ),
            "max_preclose_object_displacement": max(
                trial["preclose_max_object_displacement"] for trial in trials
            ),
            "trials": trials,
        }
    summary = {
        "format": "svla_action_replay_v1",
        "demo_count": len(demos),
        "by_action_space": by_action_space,
        "pass": all(
            result["successes"] == result["total"]
            and result["controller_failure_steps"] == 0
            and result["collision_free_approaches"] == result["total"]
            and result["preclose_contact_steps"] == 0
            for result in by_action_space.values()
        ),
    }
    write_json(args.output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["pass"]:
        raise SystemExit(1)
    return summary


def _replay_demo(demo: dict, action_space: str, object_start: np.ndarray) -> dict:
    trial = demo["metadata"]["trial_spec"]
    env = PickupTaskEvaluator()
    env.reset(object_start)
    counts = {
        "saturated_steps": 0,
        "joint_limit_clipped_steps": 0,
        "joint_step_clipped_steps": 0,
        "joint_accel_clipped_steps": 0,
        "infeasible_steps": 0,
        "controller_failure_steps": 0,
    }
    for sample in demo["samples"]:
        action = np.asarray(sample["policy_labels"][action_space], dtype=float)
        if action_space == "joint_delta":
            _, _, status = env.step_joint_delta_action(action[:5], action[5])
            get = status.__getitem__
        else:
            _, _, status = env.step_ee_tool_delta_action(action[:3], action[3:5], action[5])
            get = lambda name: getattr(status, name)
        counts["saturated_steps"] += int(get("saturated"))
        counts["joint_limit_clipped_steps"] += int(get("joint_limit_clipped"))
        counts["joint_step_clipped_steps"] += int(get("joint_step_clipped"))
        counts["joint_accel_clipped_steps"] += int(get("joint_accel_clipped"))
        counts["infeasible_steps"] += int(get("infeasible"))
        counts["controller_failure_steps"] += int(get("controller_failed"))

    metrics = env.get_success_metrics()
    success = bool(
        metrics["collision_free_approach"]
        and metrics["contact_achieved"]
        and metrics["max_object_lift"] >= LIFT_CLEARANCE
        and metrics["current_object_lift"] >= RETENTION_CLEARANCE
        and metrics["retained_during_hold"]
    )
    return {
        "trial_id": trial["trial_id"],
        "orientation": trial["orientation"],
        "object_pose": trial["object_pose"],
        "approach": trial["approach"],
        "success": success,
        "collision_free_approach": bool(metrics["collision_free_approach"]),
        "preclose_contact_steps": int(metrics["preclose_contact_steps"]),
        "preclose_max_object_displacement": float(
            metrics["preclose_max_object_displacement"]
        ),
        "steps": len(demo["samples"]),
        **counts,
    }


def _mean(values) -> float:
    values = list(values)
    return float(np.mean(values)) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "action_replay_tool_axis_summary.json",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
