from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from svla.pickup_task import (
    LIFT_CLEARANCE,
    RETENTION_CLEARANCE,
    PickupTaskEvaluator,
    default_trial_specs,
)
from svla.state_bc import TaskContext, load_policy, observation_to_features


def _overlay(frame: np.ndarray, lines: list[str]) -> None:
    panel_h = min(frame.shape[0], 14 + len(lines) * 18)
    frame[0:panel_h, :] = (frame[0:panel_h, :] * 0.35).astype(np.uint8)
    y = 12
    for line in lines:
        _draw_text(frame, line.upper(), 12, y)
        y += 18


def _draw_text(frame: np.ndarray, text: str, x: int, y: int) -> None:
    scale = 2
    for char in text:
        pattern = _glyph(char)
        for row, row_bits in enumerate(pattern):
            for col, pixel in enumerate(row_bits):
                if pixel == "1":
                    y0 = y + row * scale
                    x0 = x + col * scale
                    frame[y0 : y0 + scale, x0 : x0 + scale] = (220, 245, 235)
        x += 4 * scale


def _glyph(char: str) -> list[str]:
    glyphs = {
        " ": ["000", "000", "000", "000", "000"],
        ".": ["0", "0", "0", "0", "1"],
        "-": ["000", "000", "111", "000", "000"],
        "/": ["001", "001", "010", "100", "100"],
        "0": ["111", "101", "101", "101", "111"],
        "1": ["010", "110", "010", "010", "111"],
        "2": ["111", "001", "111", "100", "111"],
        "3": ["111", "001", "111", "001", "111"],
        "4": ["101", "101", "111", "001", "001"],
        "5": ["111", "100", "111", "001", "111"],
        "6": ["111", "100", "111", "101", "111"],
        "7": ["111", "001", "010", "010", "010"],
        "8": ["111", "101", "111", "101", "111"],
        "9": ["111", "101", "111", "001", "111"],
        "A": ["010", "101", "111", "101", "101"],
        "B": ["110", "101", "110", "101", "110"],
        "C": ["111", "100", "100", "100", "111"],
        "D": ["110", "101", "101", "101", "110"],
        "E": ["111", "100", "110", "100", "111"],
        "F": ["111", "100", "110", "100", "100"],
        "G": ["111", "100", "101", "101", "111"],
        "H": ["101", "101", "111", "101", "101"],
        "I": ["111", "010", "010", "010", "111"],
        "J": ["001", "001", "001", "101", "111"],
        "K": ["101", "101", "110", "101", "101"],
        "L": ["100", "100", "100", "100", "111"],
        "M": ["101", "111", "111", "101", "101"],
        "N": ["101", "111", "111", "111", "101"],
        "O": ["111", "101", "101", "101", "111"],
        "P": ["111", "101", "111", "100", "100"],
        "R": ["110", "101", "110", "101", "101"],
        "S": ["111", "100", "111", "001", "111"],
        "T": ["111", "010", "010", "010", "010"],
        "U": ["101", "101", "101", "101", "111"],
        "V": ["101", "101", "101", "101", "010"],
        "W": ["101", "101", "111", "111", "101"],
        "X": ["101", "101", "010", "101", "101"],
        "Y": ["101", "101", "010", "010", "010"],
        "Z": ["111", "001", "010", "100", "111"],
        "_": ["000", "000", "000", "000", "111"],
    }
    return glyphs.get(char.upper(), glyphs[" "])


def render_rollout(
    output_path: Path,
    policy_path: Path,
    trial_id: int,
    width: int,
    height: int,
    fps: int,
    max_steps: int,
    search_window: int,
) -> dict:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to export MP4 videos.")

    policy = load_policy(policy_path)
    specs = {spec.trial_id: spec for spec in default_trial_specs(repeats=1)}
    if trial_id not in specs:
        raise ValueError(f"Unknown trial_id {trial_id}")
    spec = specs[trial_id]

    env = PickupTaskEvaluator()
    renderer = mujoco.Renderer(env.model, height=height, width=width)
    env.reset(np.asarray(spec.object_pose.xyz, dtype=float))
    settled_start = env.object_position.copy()
    _, grasp_pos, grasp_quat = env.scripted_controller_commands(spec, settled_start)
    context = TaskContext.from_spec(spec)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{width}x{height}",
        "-pix_fmt",
        "rgb24",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]

    clipped_translation = 0
    min_grasp_pos_error = float("inf")
    cursor = 0
    frames = 0
    frame_stride = 4

    with subprocess.Popen(command, stdin=subprocess.PIPE) as process:
        if process.stdin is None:
            raise RuntimeError("Could not open ffmpeg stdin.")

        for step in range(max_steps):
            observation = env.get_observation()
            features = observation_to_features(observation, context, settled_start)
            action, _, nearest_index = policy.predict_with_index(
                features,
                context.key,
                cursor=cursor,
                search_window=search_window,
            )
            cursor = max(cursor + 1, nearest_index + 1)
            executed_action = action.copy()

            if policy.action_space == "joint_delta":
                _, _, status = env.step_joint_delta_action(executed_action[:5], executed_action[5])
            elif policy.action_space == "ee_tool_delta":
                _, _, status = env.step_ee_tool_delta_action(
                    executed_action[:3],
                    executed_action[3:5],
                    executed_action[5],
                )
                clipped_translation += int(status.clipped_translation)
            else:
                raise ValueError(f"unknown action space: {policy.action_space}")

            controller_failed = (
                status["controller_failed"]
                if isinstance(status, dict)
                else status.controller_failed
            )
            if controller_failed:
                break

            ee_pos, ee_quat = env.controller.ee_pose(env.data)
            min_grasp_pos_error = min(
                min_grasp_pos_error,
                float(np.linalg.norm(grasp_pos - ee_pos)),
            )
            metrics = env.get_success_metrics()

            if env.model.nmocap and env.controller.target_pos is not None:
                env.data.mocap_pos[0] = env.controller.target_pos.copy()

            if step % frame_stride == 0:
                renderer.update_scene(env.data, camera="overview")
                frame = renderer.render()
                _overlay(
                    frame,
                    [
                        f"learned policy rollout  {policy.action_space}",
                        f"trial {trial_id}  step {step + 1}",
                        f"contact {int(metrics['contact_achieved'])}  lift {metrics['max_object_lift']:.3f}m",
                        f"grasp err {min_grasp_pos_error * 1000:.1f} mm  clipped steps {clipped_translation}",
                    ],
                )
                process.stdin.write(frame.tobytes())
                frames += 1

            if (
                metrics["current_object_lift"] >= RETENTION_CLEARANCE
                and metrics["lifted_steps"] >= 180
                and metrics["contact_steps"] >= 60
                and env.gripper_object_distance() <= 0.045
            ):
                break

        metrics = env.get_success_metrics()
        reached_grasp = min_grasp_pos_error <= 0.012
        object_lifted = metrics["max_object_lift"] >= LIFT_CLEARANCE
        retained = (
            metrics["current_object_lift"] >= RETENTION_CLEARANCE
            and metrics["lifted_steps"] >= 180
            and metrics["contact_steps"] >= 60
            and env.gripper_object_distance() <= 0.045
        )
        success = bool(
            metrics["collision_free_approach"]
            and reached_grasp
            and metrics["contact_achieved"]
            and object_lifted
            and retained
        )

        renderer.update_scene(env.data, camera="overview")
        frame = renderer.render()
        _overlay(
            frame,
            [
                "rollout result",
                f"success {int(success)}  contact {int(metrics['contact_achieved'])}",
                f"lift {metrics['max_object_lift']:.3f}m  retained {int(retained)}",
            ],
        )
        for _ in range(fps):
            process.stdin.write(frame.tobytes())

        process.stdin.close()
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg failed with exit code {return_code}")

    renderer.close()
    return {
        "policy": str(policy_path),
        "action_space": policy.action_space,
        "trial_id": trial_id,
        "success": success,
        "frames": frames,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a learned BC policy rollout to MP4.")
    parser.add_argument(
        "--policy",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "state_bc" / "models" / "ee_tool_delta_mlp_bc.npz",
    )
    parser.add_argument("--trial-id", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Defaults to outputs/state_bc/<action_space>_rollout_trial<id>.mp4",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=3200)
    parser.add_argument("--search-window", type=int, default=120)
    args = parser.parse_args()

    output = args.output
    if output is None:
        policy = load_policy(args.policy)
        output = (
            PROJECT_ROOT
            / "outputs"
            / "state_bc"
            / f"{policy.action_space}_rollout_trial{args.trial_id:02d}.mp4"
        )

    summary = render_rollout(
        output,
        args.policy,
        args.trial_id,
        args.width,
        args.height,
        args.fps,
        args.max_steps,
        args.search_window,
    )
    print(summary)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
