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

from svla.action_spaces import TrajectoryState, label_transition_all
from svla.pickup_task import (
    BASE_OBJECT_HALF_SIZE,
    PickupTaskEvaluator,
    SUPPORT_TOP_Z,
    default_trial_specs,
)


SHOWCASE_TRIAL_IDS = (1, 8, 18)

GLYPHS = {
    " ": ["000", "000", "000", "000", "000"],
    "-": ["000", "000", "111", "000", "000"],
    "_": ["000", "000", "000", "000", "111"],
    ".": ["0", "0", "0", "0", "1"],
    ":": ["0", "1", "0", "1", "0"],
    "/": ["001", "001", "010", "100", "100"],
    "%": ["101", "001", "010", "100", "101"],
    "+": ["000", "010", "111", "010", "000"],
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
    "Q": ["111", "101", "101", "111", "001"],
    "R": ["110", "101", "110", "101", "101"],
    "S": ["111", "100", "111", "001", "111"],
    "T": ["111", "010", "010", "010", "010"],
    "U": ["101", "101", "101", "101", "111"],
    "V": ["101", "101", "101", "101", "010"],
    "W": ["101", "101", "111", "111", "101"],
    "X": ["101", "101", "010", "101", "101"],
    "Y": ["101", "101", "010", "010", "010"],
    "Z": ["111", "001", "010", "100", "111"],
}


def _render_trial_clip(env, renderer, process, spec, clip_index: int, clip_count: int, fps: int) -> dict:
    xyz = np.asarray(spec.object_pose.xyz, dtype=float).copy()
    xyz[2] = SUPPORT_TOP_Z + float(env.object_half_size[2])
    env.reset(xyz)
    commands, _, _ = env.scripted_controller_commands(spec)
    frame_stride = 5
    frames = 0
    clipped = 0
    final_metrics = env.get_success_metrics()

    _write_title_frames(env, renderer, process, spec, clip_index, clip_count, fps)
    for command in commands:
        for phase_step in range(command.max_steps):
            before = env.get_observation()
            after, metrics, status = env.step_controller_command(
                command.target_pos,
                command.target_quat_wxyz,
                command.gripper_open,
                substeps=4,
            )
            final_metrics = metrics
            clipped += int(status.clipped_translation or status.clipped_rotation or status.clipped_joints)

            if env.model.nmocap:
                env.data.mocap_pos[0] = np.asarray(command.target_pos, dtype=float)
            if phase_step % frame_stride == 0:
                labels = label_transition_all(
                    TrajectoryState.from_observation(before),
                    TrajectoryState.from_observation(after),
                    command.gripper_open,
                )
                renderer.update_scene(env.data, camera="overview")
                frame = renderer.render()
                _overlay(
                    frame,
                    [
                        "SO-101 CONTROLLER PICKUP SHOWCASE",
                        f"CLIP {clip_index}/{clip_count}  {spec.orientation.label}  {spec.object_pose.label}",
                        f"PHASE {command.phase}  GRIPPER {command.gripper_open:.1f}",
                        f"CLEAN APPROACH {int(metrics['collision_free_approach'])}  CONTACT {int(metrics['contact_achieved'])}  LIFT {metrics['max_object_lift']:.3f}M",
                        f"ORDER {int(metrics['event_order_valid'])}  FORCE {metrics['max_gripper_contact_force']:.1f}N  TABLE SHIFT {metrics['max_object_xy_displacement_while_supported'] * 1000:.1f}MM",
                        f"EE ERR {status.position_error:.3f}M  ROT ERR {status.rotation_error:.3f}RAD",
                        f"CLIP STEPS {clipped}  JOINT LABELS {len(labels['joint_delta'])}  EE LABELS {len(labels['ee_delta'])}",
                    ],
                )
                process.stdin.write(frame.tobytes())
                frames += 1
            if (
                command.stop_on_pose_tolerance
                and status.position_error <= env.controller.limits.position_tolerance
                and status.rotation_error <= env.controller.limits.rotation_tolerance
            ):
                break

    for _ in range(fps):
        renderer.update_scene(env.data, camera="overview")
        frame = renderer.render()
        _overlay(
            frame,
            [
                "SCRIPTED CONTROLLER RESULT",
                f"{spec.orientation.label}  {spec.object_pose.label}  {spec.approach.label}",
                f"CONTACT {int(final_metrics['contact_achieved'])}  LIFT {final_metrics['max_object_lift']:.3f}M",
                f"RETAINED HOLD {int(final_metrics['retained_during_hold'])}",
                "NO ML  NO VISION  NO VLA",
            ],
        )
        process.stdin.write(frame.tobytes())
        frames += 1

    return {
        "clip": clip_index,
        "trial_id": spec.trial_id,
        "success": bool(
            final_metrics["collision_free_approach"]
            and final_metrics["event_order_valid"]
            and final_metrics["physical_sanity_pass"]
            and final_metrics["contact_achieved"]
            and final_metrics["object_lifted"]
            and final_metrics["retained_during_hold"]
        ),
        "contact": bool(final_metrics["contact_achieved"]),
        "collision_free_approach": bool(final_metrics["collision_free_approach"]),
        "event_order_valid": bool(final_metrics["event_order_valid"]),
        "physical_sanity_pass": bool(final_metrics["physical_sanity_pass"]),
        "max_lift": float(final_metrics["max_object_lift"]),
        "retained": bool(final_metrics["retained_during_hold"]),
        "frames": frames,
    }


def _write_title_frames(env, renderer, process, spec, clip_index: int, clip_count: int, fps: int) -> None:
    renderer.update_scene(env.data, camera="overview")
    frame = renderer.render()
    _overlay(
        frame,
        [
            "CONTROLLER-ONLY PICKUP",
            f"CLIP {clip_index}/{clip_count}: {spec.orientation.label}  {spec.object_pose.label}",
            "ROTATION-AWARE IK + CONTACT/LIFT/HOLD METRICS",
            "ALIGNED JOINT-DELTA AND EE-DELTA LABELS",
        ],
    )
    for _ in range(fps):
        process.stdin.write(frame.tobytes())


def _overlay(frame: np.ndarray, lines: list[str]) -> None:
    height, width, _ = frame.shape
    panel_h = 18 + len(lines) * 20
    frame[0:panel_h, 0:width] = (frame[0:panel_h, 0:width] * 0.35).astype(np.uint8)
    frame[panel_h - 3 : panel_h, 0:width] = np.array([32, 210, 170], dtype=np.uint8)
    for index, line in enumerate(lines):
        color = (245, 248, 244) if index == 0 else (190, 240, 225)
        _draw_text(frame, line.upper(), 18, 14 + index * 20, color=color, scale=3)


def _draw_text(
    frame: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
    scale: int,
) -> None:
    cursor = x
    max_width = frame.shape[1] - 8
    for char in text:
        glyph = GLYPHS.get(char, GLYPHS[" "])
        glyph_width = len(glyph[0]) * scale
        if cursor + glyph_width >= max_width:
            break
        for row, pattern in enumerate(glyph):
            for col, pixel in enumerate(pattern):
                if pixel == "1":
                    y0 = y + row * scale
                    x0 = cursor + col * scale
                    frame[y0 : y0 + scale, x0 : x0 + scale] = color
        cursor += glyph_width + scale


def render_trials(
    output_path: Path,
    trial_ids: list[int],
    width: int,
    height: int,
    fps: int,
    size_scale: np.ndarray | None = None,
    friction: float | None = None,
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to export MP4 videos.")

    half_size = None
    if size_scale is not None:
        half_size = BASE_OBJECT_HALF_SIZE * np.asarray(size_scale, dtype=float)
    env = PickupTaskEvaluator(
        object_half_size=half_size,
        object_sliding_friction=friction,
    )
    renderer = mujoco.Renderer(env.model, height=height, width=width)
    specs = {spec.trial_id: spec for spec in default_trial_specs(repeats=1)}

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

    clip_summaries = []
    with subprocess.Popen(command, stdin=subprocess.PIPE) as process:
        if process.stdin is None:
            raise RuntimeError("Could not open ffmpeg stdin.")
        for clip_index, trial_id in enumerate(trial_ids, start=1):
            if trial_id not in specs:
                raise ValueError(f"Unknown trial_id {trial_id}; valid ids are 1–18.")
            spec = specs[trial_id]
            summary = _render_trial_clip(
                env=env,
                renderer=renderer,
                process=process,
                spec=spec,
                clip_index=clip_index,
                clip_count=len(trial_ids),
                fps=fps,
            )
            clip_summaries.append(summary)
        process.stdin.close()
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg failed with exit code {return_code}.")

    renderer.close()
    for summary in clip_summaries:
        print(
            f"clip={summary['clip']} trial={summary['trial_id']} success={int(summary['success'])} "
            f"contact={int(summary['contact'])} lift={summary['max_lift']:.3f}m "
            f"hold={int(summary['retained'])} frames={summary['frames']}"
        )
    print(f"wrote {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "pickup_showcase.mp4",
    )
    parser.add_argument(
        "--trial-id",
        type=int,
        nargs="+",
        default=None,
        help="Render only these trial ids (default: showcase set 1, 8, 18).",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--size-scale",
        type=float,
        nargs="+",
        default=None,
        help="Uniform or per-axis object size scale (e.g. 0.95 or 0.85 0.85 0.85).",
    )
    parser.add_argument(
        "--friction",
        type=float,
        default=None,
        help="Object-table sliding friction override.",
    )
    args = parser.parse_args()
    trial_ids = args.trial_id if args.trial_id is not None else list(SHOWCASE_TRIAL_IDS)
    size_scale = None
    if args.size_scale is not None:
        if len(args.size_scale) == 1:
            size_scale = np.full(3, args.size_scale[0])
        elif len(args.size_scale) == 3:
            size_scale = np.asarray(args.size_scale, dtype=float)
        else:
            raise ValueError("--size-scale expects 1 or 3 values.")
    render_trials(
        args.output,
        trial_ids,
        args.width,
        args.height,
        args.fps,
        size_scale=size_scale,
        friction=args.friction,
    )


if __name__ == "__main__":
    main()
