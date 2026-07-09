from __future__ import annotations

import numpy as np

from svla.core.action_space import COMPARISON_ACTION_SPACES, get_action_representation
from svla.pickup_task import (
    LIFT_CLEARANCE,
    RETENTION_CLEARANCE,
    PickupTaskEvaluator,
    maybe_finalize_grasp_at_sample,
)


ACTION_SPACES = COMPARISON_ACTION_SPACES


def replay_demo_policy_labels(
    demo: dict,
    action_space: str,
    object_start: np.ndarray,
    *,
    task: str = "pickup",
) -> dict:
    trial = demo["metadata"]["trial_spec"]
    env = PickupTaskEvaluator()
    env.reset(object_start)
    boundary_index = None
    if task == "pick_place":
        if "grasp_segment_finalize_sample_index" not in demo["metadata"]:
            raise ValueError("pick_place demo missing grasp_segment_finalize_sample_index")
        boundary_index = int(demo["metadata"]["grasp_segment_finalize_sample_index"])

    counts = {
        "saturated_steps": 0,
        "joint_limit_clipped_steps": 0,
        "joint_step_clipped_steps": 0,
        "joint_accel_clipped_steps": 0,
        "infeasible_steps": 0,
        "controller_failure_steps": 0,
    }
    contact_during_close = False
    representation = get_action_representation(action_space)
    for sample_index, sample in enumerate(demo["samples"]):
        sample_phase = str(sample.get("phase", ""))
        maybe_finalize_grasp_at_sample(env, sample_index, boundary_index)
        close_contact_steps_before = None
        if task == "pick_place" and sample_phase == "close_gripper":
            close_contact_steps_before = int(
                env.get_success_metrics()["close_contact_steps"]
            )
        action = np.asarray(sample["policy_labels"][action_space], dtype=float)
        _, _, status = representation.execute(env, action)
        telemetry = representation.telemetry(status)
        counts["saturated_steps"] += int(telemetry["saturated"])
        counts["joint_limit_clipped_steps"] += int(telemetry["joint_limit_clipped"])
        counts["joint_step_clipped_steps"] += int(telemetry["joint_step_clipped"])
        counts["joint_accel_clipped_steps"] += int(telemetry["joint_accel_clipped"])
        counts["infeasible_steps"] += int(telemetry["infeasible"])
        counts["controller_failure_steps"] += int(telemetry["controller_failed"])
        if close_contact_steps_before is not None:
            close_contact_steps_after = int(
                env.get_success_metrics()["close_contact_steps"]
            )
            contact_during_close = contact_during_close or (
                close_contact_steps_after > close_contact_steps_before
            )

    return _summarize_replay(
        env,
        demo,
        trial,
        task,
        counts,
        contact_during_close=contact_during_close,
    )


def _summarize_replay(
    env: PickupTaskEvaluator,
    demo: dict,
    trial: dict,
    task: str,
    counts: dict,
    *,
    contact_during_close: bool = False,
) -> dict:
    steps = len(demo["samples"])
    if task == "pick_place":
        grasp_metrics = env._grasp_segment_metrics or env.finalize_grasp_segment()
        placement_goal = np.asarray(demo["summary"]["commanded_placement_pose"], dtype=float)
        placement_ok, placement_metrics = env.evaluate_placement(
            placement_goal,
            goal_xyz=placement_goal,
        )
        success = bool(
            grasp_metrics["collision_free_approach"]
            and grasp_metrics["event_order_valid"]
            and grasp_metrics["physical_sanity_pass"]
            and contact_during_close
            and grasp_metrics["object_lifted"]
            and grasp_metrics["retained_during_hold"]
            and placement_ok
        )
        metrics = grasp_metrics
        placement_achieved = bool(placement_ok)
        placement_xy_error = float(placement_metrics["placement_xy_error"])
    else:
        metrics = env.get_success_metrics()
        success = bool(
            metrics["collision_free_approach"]
            and metrics["event_order_valid"]
            and metrics["physical_sanity_pass"]
            and metrics["contact_achieved"]
            and metrics["max_object_lift"] >= LIFT_CLEARANCE
            and metrics["current_object_lift"] >= RETENTION_CLEARANCE
            and metrics["retained_during_hold"]
        )
        placement_achieved = None
        placement_xy_error = None

    result = {
        "trial_id": trial["trial_id"],
        "orientation": trial["orientation"],
        "object_pose": trial["object_pose"],
        "approach": trial["approach"],
        "success": success,
        "collision_free_approach": bool(metrics["collision_free_approach"]),
        "event_order_valid": bool(metrics["event_order_valid"]),
        "early_close": bool(metrics.get("early_close", False)),
        "reopen_events": int(metrics["reopen_events"]),
        "physical_sanity_pass": bool(metrics["physical_sanity_pass"]),
        "max_gripper_contact_force": float(metrics["max_gripper_contact_force"]),
        "gripper_contact_impulse_before_lift": float(
            metrics["gripper_contact_impulse_before_lift"]
        ),
        "max_object_xy_displacement_while_supported": float(
            metrics["max_object_xy_displacement_while_supported"]
        ),
        "preclose_contact_steps": int(metrics["preclose_contact_steps"]),
        "preclose_max_object_displacement": float(
            metrics["preclose_max_object_displacement"]
        ),
        "steps": steps,
        "saturation_rate": counts["saturated_steps"] / max(1, steps),
        **counts,
    }
    if task == "pick_place":
        result["placement_target"] = trial.get("placement_target")
        result["placement_achieved"] = placement_achieved
        result["placement_xy_error"] = placement_xy_error
        result["contact_during_close"] = bool(contact_during_close)
        result["grasp_segment_finalize_sample_index"] = int(
            demo["metadata"]["grasp_segment_finalize_sample_index"]
        )
    return result
