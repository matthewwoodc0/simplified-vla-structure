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

from svla.controller import CartesianCommand, ControllerLimits
from svla.sim import ArmSim


def _overlay(frame: np.ndarray, lines: list[str]) -> None:
    height, width, _ = frame.shape
    panel_h = min(height, 14 + len(lines) * 18)
    frame[0:panel_h, 0:width] = (frame[0:panel_h, 0:width] * 0.35).astype(np.uint8)
    y = 12
    for line in lines:
        _draw_text(frame, line.upper(), 12, y)
        y += 18


def _draw_text(frame: np.ndarray, text: str, x: int, y: int) -> None:
    scale = 2
    glyph_w = 3 * scale + scale
    for char in text:
        pattern = _glyph(char)
        for row, row_bits in enumerate(pattern):
            for col, pixel in enumerate(row_bits):
                if pixel == "1":
                    y0 = y + row * scale
                    x0 = x + col * scale
                    frame[y0 : y0 + scale, x0 : x0 + scale] = (220, 245, 235)
        x += glyph_w


def _glyph(char: str) -> list[str]:
    glyphs = {
        " ": ["000", "000", "000", "000", "000"],
        ".": ["0", "0", "0", "0", "1"],
        "-": ["000", "000", "111", "000", "000"],
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
        "C": ["111", "100", "100", "100", "111"],
        "D": ["110", "101", "101", "101", "110"],
        "E": ["111", "100", "110", "100", "111"],
        "G": ["111", "100", "101", "101", "111"],
        "H": ["101", "101", "111", "101", "101"],
        "I": ["111", "010", "010", "010", "111"],
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
    }
    return glyphs.get(char.upper(), glyphs[" "])


def _write_title(renderer, process, lines: list[str], fps: int) -> None:
    renderer.update_scene(_title_scene_data(), camera="overview")
    frame = renderer.render()
    _overlay(frame, lines)
    for _ in range(fps):
        process.stdin.write(frame.tobytes())


_title_data_holder: dict[str, object] = {}


def _title_scene_data():
    return _title_data_holder["data"]


def _render_clip(
    sim: ArmSim,
    renderer: mujoco.Renderer,
    process,
    *,
    title: str,
    max_step_xyz: float,
    max_target_lag_xyz: float,
    steps: int,
    fps: int,
    frame_stride: int,
) -> None:
    sim.reset()
    sim.controller.limits = ControllerLimits(
        max_step_xyz=max_step_xyz,
        max_target_lag_xyz=max_target_lag_xyz,
    )
    start_ee = sim.ee_position.copy()
    command = CartesianCommand(np.array([1.0, 1.0, 1.0]), np.zeros(3), 0.5)
    _title_data_holder["data"] = sim.data

    _write_title(
        renderer,
        process,
        [
            title,
            f"max_step_xyz={max_step_xyz:.3f}  max_target_lag_xyz={max_target_lag_xyz:.3f}",
            "huge command [1,1,1] m per step — watch target marker vs arm",
        ],
        fps,
    )

    for step in range(steps):
        status = sim.step(command)
        telemetry = sim.controller.last_telemetry
        assert status is not None and telemetry is not None
        if sim.model.nmocap:
            sim.data.mocap_pos[0] = telemetry.integrated_target_pos.copy()

        if step % frame_stride == 0:
            renderer.update_scene(sim.data, camera="overview")
            frame = renderer.render()
            ee_move = float(np.linalg.norm(sim.ee_position - start_ee))
            _overlay(
                frame,
                [
                    title,
                    f"step {step + 1}/{steps}  clipped {int(status.clipped_translation)}",
                    f"target ahead {telemetry.position_error * 1000:.1f} mm",
                    f"ee moved {ee_move * 1000:.1f} mm from start",
                    "green mocap ball = integrated target  blue arm = actual ee",
                ],
            )
            process.stdin.write(frame.tobytes())


def render_clipping_video(
    output_path: Path,
    width: int,
    height: int,
    fps: int,
    steps: int,
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to export MP4 videos.")

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

    sim = ArmSim()
    renderer = mujoco.Renderer(sim.model, height=height, width=width)
    frame_stride = 2

    with subprocess.Popen(command, stdin=subprocess.PIPE) as process:
        if process.stdin is None:
            raise RuntimeError("Could not open ffmpeg stdin.")
        _render_clip(
            sim,
            renderer,
            process,
            title="clip 1/2 default safety rails",
            max_step_xyz=0.025,
            max_target_lag_xyz=0.025,
            steps=steps,
            fps=fps,
            frame_stride=frame_stride,
        )
        _render_clip(
            sim,
            renderer,
            process,
            title="clip 2/2 loose cartesian rails",
            max_step_xyz=0.5,
            max_target_lag_xyz=0.5,
            steps=steps,
            fps=fps,
            frame_stride=frame_stride,
        )
        process.stdin.close()
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg failed with exit code {return_code}.")

    renderer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Render Lab 2 clipping comparison MP4.")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "clipping_demo.mp4",
    )
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--steps", type=int, default=120)
    args = parser.parse_args()
    render_clipping_video(args.output, args.width, args.height, args.fps, args.steps)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()