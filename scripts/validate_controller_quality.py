from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from svla.pickup_task import OBJECT_START_Z, PickupTaskEvaluator
from svla.state_bc import write_json


def action_sequence(length: int) -> list[tuple[np.ndarray, np.ndarray, float]]:
    actions = []
    for index in range(length):
        t = index / max(1, length - 1)
        delta_xyz = np.array(
            [
                0.0035 * np.sin(2.0 * np.pi * t),
                0.0025 * np.cos(1.5 * np.pi * t),
                0.0015 * np.sin(np.pi * t),
            ]
        )
        delta_tilt_xy = np.array(
            [
                0.012 * np.sin(np.pi * t),
                0.008 * np.cos(2.0 * np.pi * t),
            ]
        )
        actions.append((delta_xyz, delta_tilt_xy, 1.0 - 0.4 * t))
    return actions


def rollout(
    object_xyz: np.ndarray,
    actions: list[tuple[np.ndarray, np.ndarray, float]],
    joint_offset: np.ndarray | None = None,
) -> dict:
    env = PickupTaskEvaluator()
    env.reset(object_xyz)
    if joint_offset is not None:
        qpos = env.data.qpos[env.controller.arm_qpos_ids]
        qpos[:] = np.clip(
            qpos + np.asarray(joint_offset, dtype=float),
            env.controller.joint_ranges[:, 0],
            env.controller.joint_ranges[:, 1],
        )
        env.data.ctrl[env.controller.arm_actuator_ids] = qpos
        env.data.qvel[env.controller.arm_dof_ids] = 0.0
        mujoco.mj_forward(env.model, env.data)
        env.controller.reset_target(
            env.data,
            posture_target=env.controller.posture_target,
        )
    targets = []
    ee_positions = []
    joint_targets = []
    clipped_translation = 0
    clipped_rotation = 0
    joint_limit_clipped = 0
    joint_step_clipped = 0
    joint_accel_clipped = 0
    infeasible = 0
    controller_failed = 0
    for delta_xyz, delta_tilt_xy, gripper in actions:
        env.step_ee_tool_delta_action(delta_xyz, delta_tilt_xy, gripper)
        telemetry = env.controller.last_telemetry
        targets.append(telemetry.target_pos)
        ee_positions.append(telemetry.actual_pos)
        joint_targets.append(telemetry.joint_targets)
        clipped_translation += int(telemetry.clipped_translation)
        clipped_rotation += int(telemetry.clipped_rotation)
        joint_limit_clipped += int(telemetry.joint_limit_clipped)
        joint_step_clipped += int(telemetry.joint_step_clipped)
        joint_accel_clipped += int(telemetry.joint_accel_clipped)
        infeasible += int(telemetry.infeasible)
        controller_failed += int(telemetry.controller_failed)

    targets = np.vstack(targets)
    ee_positions = np.vstack(ee_positions)
    joint_targets = np.vstack(joint_targets)
    joint_steps = np.diff(joint_targets, axis=0)
    joint_accels = np.diff(joint_steps, axis=0)
    target_steps = np.diff(targets, axis=0)
    ee_steps = np.diff(ee_positions, axis=0)
    return {
        "final_observation": env.get_observation(),
        "metrics": {
            "max_target_step": _max_norm(target_steps),
            "max_ee_step": _max_norm(ee_steps),
            "max_joint_step": _max_norm(joint_steps),
            "max_joint_accel": _max_norm(joint_accels),
            "mean_tracking_error": float(np.mean(np.linalg.norm(targets - ee_positions, axis=1))),
            "clipped_translation_steps": int(clipped_translation),
            "clipped_rotation_steps": int(clipped_rotation),
            "joint_limit_clipped_steps": int(joint_limit_clipped),
            "joint_step_clipped_steps": int(joint_step_clipped),
            "joint_accel_clipped_steps": int(joint_accel_clipped),
            "infeasible_steps": int(infeasible),
            "controller_failure_steps": int(controller_failed),
        },
    }


def run(args: argparse.Namespace) -> dict:
    actions = action_sequence(args.steps)
    center = np.array([0.0, -0.235, OBJECT_START_Z])
    first = rollout(center, actions)
    repeat = rollout(center, actions)
    shifted = rollout(
        center,
        actions,
        joint_offset=np.array([0.002, -0.003, 0.002, -0.002, 0.001]),
    )

    first_obs = first["final_observation"]
    repeat_obs = repeat["final_observation"]
    shifted_obs = shifted["final_observation"]
    deterministic_error = _observation_error(first_obs, repeat_obs)
    nearby_error = _observation_error(first_obs, shifted_obs)
    summary = {
        "format": "svla_controller_quality_v2_tool_axis",
        "steps": args.steps,
        "deterministic_error": deterministic_error,
        "nearby_start_error": nearby_error,
        "center_metrics": first["metrics"],
        "repeat_metrics": repeat["metrics"],
        "nearby_metrics": shifted["metrics"],
        "pass": bool(
            deterministic_error["joint_positions"] <= 1e-10
            and deterministic_error["ee_position"] <= 1e-10
            and first["metrics"]["max_joint_step"] <= args.max_joint_step_threshold
            and first["metrics"]["max_joint_accel"] <= args.max_joint_accel_threshold
            and nearby_error["ee_position"] <= args.nearby_ee_threshold
            and first["metrics"]["controller_failure_steps"] == 0
        ),
        "thresholds": {
            "max_joint_step": args.max_joint_step_threshold,
            "max_joint_accel": args.max_joint_accel_threshold,
            "nearby_ee_position": args.nearby_ee_threshold,
        },
    }
    write_json(args.output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["pass"]:
        raise SystemExit(1)
    return summary


def _observation_error(first: dict, second: dict) -> dict[str, float]:
    return {
        key: float(
            np.linalg.norm(np.asarray(first[key], dtype=float) - np.asarray(second[key], dtype=float))
        )
        for key in ("joint_positions", "joint_velocities", "ee_position", "ee_quat_wxyz")
    }


def _max_norm(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    return float(np.max(np.linalg.norm(values, axis=1)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "controller_quality_summary.json",
    )
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--max-joint-step-threshold", type=float, default=0.030)
    parser.add_argument("--max-joint-accel-threshold", type=float, default=0.020)
    parser.add_argument("--nearby-ee-threshold", type=float, default=0.020)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
