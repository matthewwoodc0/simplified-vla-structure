from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from svla.pickup_task import (
    MAX_GRIPPER_CONTACT_FORCE,
    MAX_GRIPPER_IMPULSE_BEFORE_LIFT,
    MAX_SUPPORTED_ROTATION,
    MAX_SUPPORTED_XY_DISPLACEMENT,
    PRECLOSE_DISPLACEMENT_TOLERANCE,
    PickupTaskEvaluator,
    default_trial_specs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run(args: argparse.Namespace) -> dict:
    trials = []
    for spec in default_trial_specs(repeats=args.repeats):
        env = PickupTaskEvaluator()
        env.reset(spec.object_pose.xyz)
        settled_start = env.object_position.copy()
        commands, grasp_pos, _ = env.scripted_controller_commands(spec, settled_start)
        saturated_steps = 0
        controller_failure_steps = 0

        for command in commands:
            for _ in range(command.max_steps):
                _, metrics, status = env.step_controller_command(
                    command.target_pos,
                    command.target_quat_wxyz,
                    command.gripper_open,
                    substeps=4,
                )
                saturated_steps += int(status.saturated)
                controller_failure_steps += int(status.controller_failed)
                if (
                    command.stop_on_pose_tolerance
                    and status.position_error <= env.controller.limits.position_tolerance
                    and status.rotation_error <= env.controller.limits.rotation_tolerance
                ):
                    break

        grasp_target_error = float(np.linalg.norm(grasp_pos - settled_start))
        success = bool(
            metrics["collision_free_approach"]
            and metrics["event_order_valid"]
            and metrics["physical_sanity_pass"]
            and metrics["contact_achieved"]
            and metrics["object_lifted"]
            and metrics["retained_during_hold"]
            and controller_failure_steps == 0
        )
        trials.append(
            {
                "trial_id": spec.trial_id,
                "orientation": spec.orientation.label,
                "object_pose": spec.object_pose.label,
                "approach": spec.approach.label,
                "repeat": spec.repeat,
                "success": success,
                "collision_free_approach": bool(metrics["collision_free_approach"]),
                "event_order_valid": bool(metrics["event_order_valid"]),
                "early_close": bool(metrics["early_close"]),
                "reopen_events": int(metrics["reopen_events"]),
                "physical_sanity_pass": bool(metrics["physical_sanity_pass"]),
                "max_gripper_contact_force": float(
                    metrics["max_gripper_contact_force"]
                ),
                "gripper_contact_impulse_before_lift": float(
                    metrics["gripper_contact_impulse_before_lift"]
                ),
                "max_object_xy_displacement_while_supported": float(
                    metrics["max_object_xy_displacement_while_supported"]
                ),
                "max_object_rotation_while_supported": float(
                    metrics["max_object_rotation_while_supported"]
                ),
                "preclose_contact_steps": int(metrics["preclose_contact_steps"]),
                "preclose_max_object_displacement": float(
                    metrics["preclose_max_object_displacement"]
                ),
                "grasp_target_error": grasp_target_error,
                "close_contact_achieved": bool(metrics["contact_achieved"]),
                "object_lifted": bool(metrics["object_lifted"]),
                "retained_during_hold": bool(metrics["retained_during_hold"]),
                "saturated_steps": saturated_steps,
                "controller_failure_steps": controller_failure_steps,
            }
        )

    summary = {
        "format": "svla_grasp_geometry_validation_v1",
        "pass": all(trial["success"] for trial in trials),
        "total": len(trials),
        "successes": sum(trial["success"] for trial in trials),
        "collision_free_approaches": sum(
            trial["collision_free_approach"] for trial in trials
        ),
        "valid_event_orders": sum(trial["event_order_valid"] for trial in trials),
        "physical_sanity_passes": sum(
            trial["physical_sanity_pass"] for trial in trials
        ),
        "early_close_trials": sum(trial["early_close"] for trial in trials),
        "reopen_events": sum(trial["reopen_events"] for trial in trials),
        "preclose_contact_steps": sum(
            trial["preclose_contact_steps"] for trial in trials
        ),
        "max_preclose_object_displacement": max(
            trial["preclose_max_object_displacement"] for trial in trials
        ),
        "preclose_displacement_tolerance": PRECLOSE_DISPLACEMENT_TOLERANCE,
        "max_grasp_target_error": max(trial["grasp_target_error"] for trial in trials),
        "max_gripper_contact_force": max(
            trial["max_gripper_contact_force"] for trial in trials
        ),
        "max_gripper_contact_impulse_before_lift": max(
            trial["gripper_contact_impulse_before_lift"] for trial in trials
        ),
        "max_object_xy_displacement_while_supported": max(
            trial["max_object_xy_displacement_while_supported"] for trial in trials
        ),
        "max_object_rotation_while_supported": max(
            trial["max_object_rotation_while_supported"] for trial in trials
        ),
        "audit_limits": {
            "max_gripper_contact_force": MAX_GRIPPER_CONTACT_FORCE,
            "max_gripper_contact_impulse_before_lift": MAX_GRIPPER_IMPULSE_BEFORE_LIFT,
            "max_object_xy_displacement_while_supported": MAX_SUPPORTED_XY_DISPLACEMENT,
            "max_object_rotation_while_supported": MAX_SUPPORTED_ROTATION,
        },
        "close_contacts": sum(trial["close_contact_achieved"] for trial in trials),
        "retained_pickups": sum(trial["retained_during_hold"] for trial in trials),
        "controller_failure_steps": sum(
            trial["controller_failure_steps"] for trial in trials
        ),
        "trials": trials,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"grasp geometry: {summary['successes']}/{summary['total']} pickups, "
        f"{summary['collision_free_approaches']}/{summary['total']} clean approaches, "
        f"preclose_contacts={summary['preclose_contact_steps']}, "
        f"max_preclose_displacement="
        f"{summary['max_preclose_object_displacement']:.9f}m"
    )
    print(f"wrote {args.output}")
    if not summary["pass"]:
        raise SystemExit(1)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate collision-free SO-101 pickup approach and grasp TCP calibration."
    )
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "grasp_geometry_summary.json",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
