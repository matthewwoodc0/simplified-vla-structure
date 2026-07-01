from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json

import numpy as np

from svla.action_spaces import TrajectoryState, label_transition_all
from svla.pickup_task import (
    LIFT_CLEARANCE,
    PICK_PLACE_TRANSPORT_PHASE,
    RETENTION_CLEARANCE,
    PickPlaceTrialSpec,
    PickupTaskEvaluator,
    PickupTrialSpec,
    _orientation_error_rotvec,
    _round_list,
    maybe_finalize_grasp_at_sample,
)


class PickupDemoRecorder:
    """Record deterministic controller-only pickup demos with aligned labels."""

    def __init__(self, env: PickupTaskEvaluator | None = None) -> None:
        self.env = env or PickupTaskEvaluator()

    def record_trial(self, spec: PickupTrialSpec) -> dict:
        object_start = np.asarray(spec.object_pose.xyz, dtype=float)
        self.env.reset(object_start)
        settled_start = self.env.object_position.copy()
        commands, grasp_pos, grasp_quat = self.env.scripted_controller_commands(spec, settled_start)

        samples: list[dict] = []
        phase_summaries: list[dict] = []
        clipped_translation = 0
        clipped_rotation = 0
        clipped_joints = 0

        for command in commands:
            phase_start = self.env.get_success_metrics()
            phase_first_sample = len(samples)
            last_status = None
            for phase_step in range(command.max_steps):
                before_obs = self.env.get_observation()
                after_obs, metrics, status = self.env.step_controller_command(
                    command.target_pos,
                    command.target_quat_wxyz,
                    command.gripper_open,
                    substeps=4,
                )
                last_status = status
                clipped_translation += int(status.clipped_translation)
                clipped_rotation += int(status.clipped_rotation)
                clipped_joints += int(status.clipped_joints)

                labels = label_transition_all(
                    TrajectoryState.from_observation(before_obs),
                    TrajectoryState.from_observation(after_obs),
                    command.gripper_open,
                )
                telemetry = self.env.controller.last_telemetry
                policy_labels = _policy_labels(command, telemetry)
                samples.append(
                    {
                        "step_index": len(samples),
                        "phase": command.phase,
                        "phase_step": phase_step,
                        "observation": before_obs,
                        "command": {
                            "target_pos": _round_list(command.target_pos),
                            "target_quat_wxyz": _round_list(command.target_quat_wxyz),
                            "gripper_open": float(command.gripper_open),
                        },
                        "labels": labels,
                        "policy_labels": policy_labels,
                        "next_observation": after_obs,
                        "controller_telemetry": _telemetry_to_dict(telemetry),
                        "success_metrics": metrics,
                    }
                )
                if (
                    command.stop_on_pose_tolerance
                    and status.position_error <= self.env.controller.limits.position_tolerance
                    and status.rotation_error <= self.env.controller.limits.rotation_tolerance
                ):
                    break
            phase_end = self.env.get_success_metrics()
            phase_summaries.append(
                {
                    "phase": command.phase,
                    "samples": len(samples) - phase_first_sample,
                    "contact_steps_delta": phase_end["contact_steps"] - phase_start["contact_steps"],
                    "preclose_contact_steps_delta": (
                        phase_end["preclose_contact_steps"]
                        - phase_start["preclose_contact_steps"]
                    ),
                    "lifted_steps_delta": phase_end["lifted_steps"] - phase_start["lifted_steps"],
                    "final_position_error": float(last_status.position_error)
                    if last_status is not None
                    else float("inf"),
                    "final_rotation_error": float(last_status.rotation_error)
                    if last_status is not None
                    else float("inf"),
                }
            )

        summary = self._summarize_demo(
            spec=spec,
            settled_start=settled_start,
            grasp_pos=grasp_pos,
            grasp_quat=grasp_quat,
            phase_summaries=phase_summaries,
            clipped_translation=clipped_translation,
            clipped_rotation=clipped_rotation,
            clipped_joints=clipped_joints,
        )
        return {
            "format": "svla_pickup_demo_v3_physics_audit",
            "metadata": {
                "trial_spec": {
                    "trial_id": spec.trial_id,
                    "orientation": spec.orientation.label,
                    "object_pose": spec.object_pose.label,
                    "approach": spec.approach.label,
                    "repeat": spec.repeat,
                },
                "action_spaces": {
                    "joint_delta": "arm joint delta followed by gripper_open command",
                    "ee_delta": "end-effector xyz delta, local rotvec delta, gripper_open command",
                    "ee_tool_delta": (
                        "end-effector xyz delta, local X/Y tilt, deterministic local-Z "
                        "posture control, gripper_open command"
                    ),
                },
                "no_ml": True,
                "ee_frame": "calibrated grasp-center TCP",
            },
            "summary": summary,
            "phase_summaries": phase_summaries,
            "samples": samples,
        }

    def write_trial(self, spec: PickupTrialSpec, output_path: Path) -> dict:
        demo = self.record_trial(spec)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(demo, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return demo

    def record_pick_place_trial(self, spec: PickPlaceTrialSpec) -> dict:
        object_start = np.asarray(spec.object_pose.xyz, dtype=float)
        self.env.reset(object_start)
        settled_start = self.env.object_position.copy()
        commands, grasp_pos, grasp_quat, place_goal = self.env.scripted_pick_place_commands(
            spec,
            settled_start,
        )

        samples: list[dict] = []
        phase_summaries: list[dict] = []
        clipped_translation = 0
        clipped_rotation = 0
        clipped_joints = 0

        grasp_boundary_index: int | None = None
        for command in commands:
            phase_start = self.env.get_success_metrics()
            phase_first_sample = len(samples)
            if (
                grasp_boundary_index is None
                and command.phase == PICK_PLACE_TRANSPORT_PHASE
            ):
                grasp_boundary_index = phase_first_sample
            last_status = None
            for phase_step in range(command.max_steps):
                maybe_finalize_grasp_at_sample(
                    self.env,
                    len(samples),
                    grasp_boundary_index,
                )
                before_obs = self.env.get_observation()
                after_obs, metrics, status = self.env.step_controller_command(
                    command.target_pos,
                    command.target_quat_wxyz,
                    command.gripper_open,
                    substeps=4,
                )
                last_status = status
                clipped_translation += int(status.clipped_translation)
                clipped_rotation += int(status.clipped_rotation)
                clipped_joints += int(status.clipped_joints)

                labels = label_transition_all(
                    TrajectoryState.from_observation(before_obs),
                    TrajectoryState.from_observation(after_obs),
                    command.gripper_open,
                )
                telemetry = self.env.controller.last_telemetry
                policy_labels = _policy_labels(command, telemetry)
                samples.append(
                    {
                        "step_index": len(samples),
                        "phase": command.phase,
                        "phase_step": phase_step,
                        "observation": before_obs,
                        "command": {
                            "target_pos": _round_list(command.target_pos),
                            "target_quat_wxyz": _round_list(command.target_quat_wxyz),
                            "gripper_open": float(command.gripper_open),
                        },
                        "labels": labels,
                        "policy_labels": policy_labels,
                        "next_observation": after_obs,
                        "controller_telemetry": _telemetry_to_dict(telemetry),
                        "success_metrics": metrics,
                    }
                )
                if (
                    command.stop_on_pose_tolerance
                    and status.position_error <= self.env.controller.limits.position_tolerance
                    and status.rotation_error <= self.env.controller.limits.rotation_tolerance
                ):
                    break
            phase_end = self.env.get_success_metrics()
            phase_summaries.append(
                {
                    "phase": command.phase,
                    "samples": len(samples) - phase_first_sample,
                    "contact_steps_delta": phase_end["contact_steps"] - phase_start["contact_steps"],
                    "preclose_contact_steps_delta": (
                        phase_end["preclose_contact_steps"]
                        - phase_start["preclose_contact_steps"]
                    ),
                    "lifted_steps_delta": phase_end["lifted_steps"] - phase_start["lifted_steps"],
                    "final_position_error": float(last_status.position_error)
                    if last_status is not None
                    else float("inf"),
                    "final_rotation_error": float(last_status.rotation_error)
                    if last_status is not None
                    else float("inf"),
                }
            )

        placement_achieved, placement_metrics = self.env.evaluate_placement(
            place_goal,
            goal_xyz=place_goal,
        )
        summary = self._summarize_pick_place_demo(
            spec=spec,
            settled_start=settled_start,
            grasp_pos=grasp_pos,
            grasp_quat=grasp_quat,
            place_pos=place_goal,
            phase_summaries=phase_summaries,
            placement_achieved=placement_achieved,
            placement_metrics=placement_metrics,
            clipped_translation=clipped_translation,
            clipped_rotation=clipped_rotation,
            clipped_joints=clipped_joints,
        )
        return {
            "format": "svla_pick_place_demo_v1",
            "metadata": {
                "trial_spec": {
                    "trial_id": spec.trial_id,
                    "orientation": spec.orientation.label,
                    "object_pose": spec.object_pose.label,
                    "approach": spec.approach.label,
                    "placement_target": spec.placement_target.label,
                    "repeat": spec.repeat,
                },
                "action_spaces": {
                    "joint_delta": "arm joint delta followed by gripper_open command",
                    "ee_delta": "end-effector xyz delta, local rotvec delta, gripper_open command",
                    "ee_tool_delta": (
                        "end-effector xyz delta, local X/Y tilt, deterministic local-Z "
                        "posture control, gripper_open command"
                    ),
                },
                "no_ml": True,
                "ee_frame": "calibrated grasp-center TCP",
                "grasp_segment_finalize_sample_index": int(grasp_boundary_index)
                if grasp_boundary_index is not None
                else None,
            },
            "summary": summary,
            "phase_summaries": phase_summaries,
            "samples": samples,
        }

    def write_pick_place_trial(self, spec: PickPlaceTrialSpec, output_path: Path) -> dict:
        demo = self.record_pick_place_trial(spec)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(demo, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return demo

    def _summarize_pick_place_demo(
        self,
        spec: PickPlaceTrialSpec,
        settled_start: np.ndarray,
        grasp_pos: np.ndarray,
        grasp_quat: np.ndarray,
        place_pos: np.ndarray,
        phase_summaries: list[dict],
        placement_achieved: bool,
        placement_metrics: dict,
        clipped_translation: int,
        clipped_rotation: int,
        clipped_joints: int,
    ) -> dict:
        phases = {phase["phase"]: phase for phase in phase_summaries}
        grasp_phase = phases.get("grasp_align", {})
        close_phase = phases.get("close_gripper", {})
        hold_phase = phases.get("hold", {})
        grasp_metrics = self.env._grasp_segment_metrics or self.env.finalize_grasp_segment()
        grasp_position_error = float(grasp_phase.get("final_position_error", float("inf")))
        grasp_rotation_error = float(grasp_phase.get("final_rotation_error", float("inf")))
        reached_grasp = grasp_position_error <= 0.012 and grasp_rotation_error <= 0.22
        contact_during_close = int(close_phase.get("contact_steps_delta", 0)) > 0
        retained = bool(grasp_metrics["retained_during_hold"])
        grasp_success = bool(
            grasp_metrics["collision_free_approach"]
            and grasp_metrics["event_order_valid"]
            and grasp_metrics["physical_sanity_pass"]
            and reached_grasp
            and contact_during_close
            and grasp_metrics["object_lifted"]
            and retained
        )
        success = bool(
            grasp_success
            and placement_achieved
            and placement_metrics["gripper_released"]
        )
        if not grasp_success:
            failure_category, note = self.env._classify_failure(
                collision_free_approach=bool(grasp_metrics["collision_free_approach"]),
                event_order_valid=bool(grasp_metrics["event_order_valid"]),
                physical_sanity_pass=bool(grasp_metrics["physical_sanity_pass"]),
                reached_grasp=reached_grasp,
                contact=contact_during_close,
                lifted=bool(grasp_metrics["object_lifted"]),
                retained=retained,
                final_position_error=grasp_position_error,
                final_rotation_error=grasp_rotation_error,
                clipped_joint_steps=clipped_joints,
            )
        elif not success:
            failure_category, note = (
                "placement_failure",
                "grasp segment passed but placement tolerance or release failed",
            )
        else:
            failure_category, note = (
                "none",
                "pick-and-place met grasp gates and placement criteria",
            )
        return {
            "success": success,
            "failure_category": failure_category,
            "note": note,
            "object_start_pose": _round_list(settled_start),
            "commanded_grasp_pose": _round_list(grasp_pos),
            "commanded_placement_pose": _round_list(place_pos),
            "gripper_orientation_wxyz": _round_list(grasp_quat),
            "final_ee_position_error": grasp_position_error,
            "final_ee_rotation_error": grasp_rotation_error,
            "grasp_ee_position_error": grasp_position_error,
            "grasp_ee_rotation_error": grasp_rotation_error,
            "contact_achieved": bool(contact_during_close),
            "collision_free_approach": bool(grasp_metrics["collision_free_approach"]),
            "event_order_valid": bool(grasp_metrics["event_order_valid"]),
            "early_close": bool(grasp_metrics["early_close"]),
            "reopen_events": int(grasp_metrics["reopen_events"]),
            "physical_sanity_pass": bool(grasp_metrics["physical_sanity_pass"]),
            "max_gripper_contact_force": float(grasp_metrics["max_gripper_contact_force"]),
            "gripper_contact_impulse_before_lift": float(
                grasp_metrics["gripper_contact_impulse_before_lift"]
            ),
            "max_object_xy_displacement_while_supported": float(
                grasp_metrics["max_object_xy_displacement_while_supported"]
            ),
            "max_object_rotation_while_supported": float(
                grasp_metrics["max_object_rotation_while_supported"]
            ),
            "preclose_contact_steps": int(grasp_metrics["preclose_contact_steps"]),
            "preclose_max_object_displacement": float(
                grasp_metrics["preclose_max_object_displacement"]
            ),
            "object_lifted": bool(grasp_metrics["object_lifted"]),
            "retained_during_hold": retained,
            "placement_target": spec.placement_target.label,
            "placement_achieved": bool(placement_achieved),
            "placement_xy_error": float(placement_metrics["placement_xy_error"]),
            "placement_z_error": float(placement_metrics["placement_z_error"]),
            "gripper_released": bool(placement_metrics["gripper_released"]),
            "final_object_pose": _round_list(self.env.object_position),
            "final_object_lift": float(
                self.env.object_position[2] - settled_start[2]
            ),
            "max_object_lift": float(grasp_metrics["max_object_lift"]),
            "gripper_object_distance": float(
                self.env.gripper_object_distance()
            ),
            "clipped_translation_steps": int(clipped_translation),
            "clipped_rotation_steps": int(clipped_rotation),
            "clipped_joint_steps": int(clipped_joints),
            "trial_id": spec.trial_id,
            "orientation": spec.orientation.label,
            "object_pose": spec.object_pose.label,
            "approach": spec.approach.label,
            "repeat": spec.repeat,
        }

    def _summarize_demo(
        self,
        spec: PickupTrialSpec,
        settled_start: np.ndarray,
        grasp_pos: np.ndarray,
        grasp_quat: np.ndarray,
        phase_summaries: list[dict],
        clipped_translation: int,
        clipped_rotation: int,
        clipped_joints: int,
    ) -> dict:
        phases = {phase["phase"]: phase for phase in phase_summaries}
        grasp_phase = phases.get("grasp_align", {})
        close_phase = phases.get("close_gripper", {})
        hold_phase = phases.get("hold", {})
        metrics = self.env.get_success_metrics()
        final_pos, final_quat = self.env.controller.ee_pose(self.env.data)
        post_hold_grasp_offset = float(np.linalg.norm(grasp_pos - final_pos))
        post_hold_rotation_offset = float(
            np.linalg.norm(_orientation_error_rotvec(final_quat, grasp_quat))
        )
        grasp_position_error = float(
            grasp_phase.get("final_position_error", post_hold_grasp_offset)
        )
        grasp_rotation_error = float(
            grasp_phase.get("final_rotation_error", post_hold_rotation_offset)
        )
        reached_grasp = grasp_position_error <= 0.012 and grasp_rotation_error <= 0.22
        collision_free_approach = bool(metrics["collision_free_approach"])
        event_order_valid = bool(metrics["event_order_valid"])
        physical_sanity_pass = bool(metrics["physical_sanity_pass"])
        contact_during_close = int(close_phase.get("contact_steps_delta", 0)) > 0
        object_lifted = bool(metrics["max_object_lift"] >= LIFT_CLEARANCE)
        retained = (
            metrics["current_object_lift"] >= RETENTION_CLEARANCE
            and int(hold_phase.get("lifted_steps_delta", 0)) >= 180
            and int(hold_phase.get("contact_steps_delta", 0)) >= 60
            and metrics["gripper_object_distance"] <= 0.045
        )
        failure_category, note = self.env._classify_failure(
            collision_free_approach=collision_free_approach,
            event_order_valid=event_order_valid,
            physical_sanity_pass=physical_sanity_pass,
            reached_grasp=reached_grasp,
            contact=contact_during_close,
            lifted=object_lifted,
            retained=retained,
            final_position_error=grasp_position_error,
            final_rotation_error=grasp_rotation_error,
            clipped_joint_steps=clipped_joints,
        )
        return {
            "success": bool(
                collision_free_approach
                and event_order_valid
                and physical_sanity_pass
                and reached_grasp
                and contact_during_close
                and object_lifted
                and retained
            ),
            "failure_category": failure_category,
            "note": note,
            "object_start_pose": _round_list(settled_start),
            "commanded_grasp_pose": _round_list(grasp_pos),
            "gripper_orientation_wxyz": _round_list(grasp_quat),
            "final_ee_position_error": grasp_position_error,
            "final_ee_rotation_error": grasp_rotation_error,
            "grasp_ee_position_error": grasp_position_error,
            "grasp_ee_rotation_error": grasp_rotation_error,
            "post_hold_grasp_position_offset": post_hold_grasp_offset,
            "post_hold_grasp_rotation_offset": post_hold_rotation_offset,
            "contact_achieved": bool(contact_during_close),
            "collision_free_approach": collision_free_approach,
            "event_order_valid": event_order_valid,
            "early_close": bool(metrics["early_close"]),
            "reopen_events": int(metrics["reopen_events"]),
            "physical_sanity_pass": physical_sanity_pass,
            "max_gripper_contact_force": float(metrics["max_gripper_contact_force"]),
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
            "object_lifted": object_lifted,
            "retained_during_hold": bool(retained),
            "final_object_pose": _round_list(self.env.object_position),
            "final_object_lift": float(metrics["current_object_lift"]),
            "max_object_lift": float(metrics["max_object_lift"]),
            "gripper_object_distance": float(metrics["gripper_object_distance"]),
            "clipped_translation_steps": int(clipped_translation),
            "clipped_rotation_steps": int(clipped_rotation),
            "clipped_joint_steps": int(clipped_joints),
            "trial_id": spec.trial_id,
            "orientation": spec.orientation.label,
            "object_pose": spec.object_pose.label,
            "approach": spec.approach.label,
            "repeat": spec.repeat,
        }


def _telemetry_to_dict(telemetry) -> dict:
    if telemetry is None:
        return {}
    raw = asdict(telemetry)
    result = {}
    for key, value in raw.items():
        if isinstance(value, np.ndarray):
            result[key] = _round_list(value)
        elif isinstance(value, np.bool_):
            result[key] = bool(value)
        else:
            result[key] = value
    return result


def _policy_labels(
    command,
    telemetry,
) -> dict[str, list[float]]:
    if telemetry is None:
        raise RuntimeError("controller telemetry is required before writing policy labels")
    joint_delta = telemetry.joint_target_error
    # Reconstruct the minimum Cartesian intention whose damped IK map produces
    # the same bounded joint target. This removes unreachable 6-D pose error
    # while preserving controller-boundary equivalence.
    delta_xyz = np.asarray(telemetry.feasible_delta_xyz, dtype=float)
    delta_rotvec = np.asarray(telemetry.feasible_delta_rotvec, dtype=float)
    return {
        "joint_delta": _round_policy(np.concatenate((joint_delta, [command.gripper_open]))),
        "ee_delta": _round_policy(
            np.concatenate((delta_xyz, delta_rotvec, [command.gripper_open]))
        ),
        "ee_tool_delta": _round_policy(
            np.concatenate((delta_xyz, delta_rotvec[:2], [command.gripper_open]))
        ),
    }


def _round_policy(values: np.ndarray) -> list[float]:
    return np.round(np.asarray(values, dtype=float), 9).tolist()
