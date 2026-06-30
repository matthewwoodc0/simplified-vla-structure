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

from svla.sim import ArmSim


TARGET_DELTAS = (
    np.array([-0.04, 0.05, 0.03]),
    np.array([-0.03, 0.09, 0.00]),
    np.array([0.04, -0.07, -0.04]),
    np.array([-0.06, 0.02, 0.01]),
)


def render_reach_video(output_path: Path, width: int, height: int, fps: int) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to export MP4 videos.")

    sim = ArmSim()
    targets = [sim.ee_position + delta for delta in TARGET_DELTAS]
    renderer = mujoco.Renderer(sim.model, height=height, width=width)

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

    with subprocess.Popen(command, stdin=subprocess.PIPE) as process:
        if process.stdin is None:
            raise RuntimeError("Could not open ffmpeg stdin.")
        for target in targets:
            sim.set_target_marker(target)
            for step in range(160):
                sim.move_to(target, max_steps=1)
                if step % 2 == 0:
                    renderer.update_scene(sim.data, camera="overview")
                    frame = renderer.render()
                    process.stdin.write(frame.tobytes())
        process.stdin.close()
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg failed with exit code {return_code}.")

    renderer.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reach_demo.mp4",
        help="Path for the rendered MP4.",
    )
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    render_reach_video(args.output, args.width, args.height, args.fps)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
