from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def render_preview(
    dataset_dir: Path,
    output: Path,
    *,
    camera: str = "overview",
    episode_index: int = 0,
    stride: int = 5,
    fps: int = 24,
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to export MP4 videos.")
    manifest = json.loads((dataset_dir / "vision_manifest.json").read_text(encoding="utf-8"))
    episodes = manifest.get("episodes", [])
    if not episodes:
        raise ValueError("dataset has no episodes")
    episode = episodes[episode_index]
    if camera not in episode.get("cameras", {}):
        raise ValueError(f"episode has no camera {camera!r}")
    record = episode["cameras"][camera]
    frames_path = Path(record["path"])
    if not frames_path.is_absolute():
        frames_path = dataset_dir / frames_path.name
    with np.load(frames_path) as payload:
        frames = payload[record["array"]]
    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("preview expects uint8 RGB frames shaped [T,H,W,3]")

    output.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames.shape[1], frames.shape[2]
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
        str(output),
    ]
    with subprocess.Popen(command, stdin=subprocess.PIPE) as process:
        if process.stdin is None:
            raise RuntimeError("could not open ffmpeg stdin")
        for frame in frames[:: max(1, stride)]:
            process.stdin.write(frame.tobytes())
        process.stdin.close()
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg failed with exit code {return_code}")
    print(
        f"wrote {output} camera={camera} episode={episode_index} "
        f"frames={len(frames[:: max(1, stride)])}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "phase6a_vision_sample",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "phase6a_vision_sample" / "preview.mp4",
    )
    parser.add_argument("--camera", default="overview")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--fps", type=int, default=24)
    args = parser.parse_args()
    render_preview(
        args.dataset_dir,
        args.output,
        camera=args.camera,
        episode_index=args.episode_index,
        stride=args.stride,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
