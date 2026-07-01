#!/usr/bin/env python3
"""Assemble ~2 min midway demo from physics-audit footage. Lives under demos/ only."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT_ROOT / "outputs"

# Physics-audit sources (Jul 2026) — never fall back to pre-audit showcase.
SCRIPTED_CLIP = OUTPUTS / "pickup_showcase_physics_audit.mp4"
JOINT_CLIP = OUTPUTS / "joint_bc_success.mp4"
EE_CLIP = OUTPUTS / "ee_bc_success.mp4"

GLYPHS = {
    " ": ["000", "000", "000", "000", "000"],
    "-": ["000", "000", "111", "000", "000"],
    "/": ["001", "001", "010", "100", "100"],
    "+": ["000", "010", "111", "010", "000"],
    ".": ["0", "0", "0", "0", "1"],
    ":": ["0", "1", "0", "1", "0"],
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


def _draw_text(frame, text, x, y, *, color, scale):
    cursor = x
    for char in text:
        glyph = GLYPHS.get(char.upper(), GLYPHS[" "])
        w = len(glyph[0]) * scale
        for row, pattern in enumerate(glyph):
            for col, pixel in enumerate(pattern):
                if pixel == "1":
                    y0, x0 = y + row * scale, cursor + col * scale
                    frame[y0 : y0 + scale, x0 : x0 + scale] = color
        cursor += w + scale


def _slide_frame(width, height, title, lines):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :] = (14, 18, 24)
    frame[height // 2 - 2 : height // 2 + 4, :] = (32, 210, 170)
    _draw_text(frame, title, 48, 72, color=(245, 248, 244), scale=4)
    y = 160
    for i, line in enumerate(lines):
        color = (190, 240, 225) if i == 0 else (150, 175, 195)
        _draw_text(frame, line, 64, y, color=color, scale=3)
        y += 34
    return frame


def render_slide(path: Path, title: str, lines: list[str], duration: float, w: int, h: int, fps: int):
    ffmpeg = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
    frame = _slide_frame(w, h, title, lines)
    n = max(1, int(round(duration * fps)))
    cmd = [
        ffmpeg, "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{w}x{h}", "-pix_fmt", "rgb24", "-r", str(fps), "-i", "-",
        "-an", "-t", f"{duration:.3f}", "-vcodec", "libx264", "-pix_fmt", "yuv420p", str(path),
    ]
    with subprocess.Popen(cmd, stdin=subprocess.PIPE) as proc:
        assert proc.stdin is not None
        for _ in range(n):
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError(f"slide encode failed: {path}")


def trim_clip(src: Path, dst: Path, duration: float, w: int, h: int, fps: int, start: float = 0.0):
    ffmpeg = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
    cmd = [
        ffmpeg, "-y", "-ss", f"{start:.3f}", "-i", str(src), "-t", f"{duration:.3f}",
        "-vf", f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
               f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps}",
        "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p", str(dst),
    ]
    if subprocess.run(cmd, check=False).returncode != 0:
        raise RuntimeError(f"trim failed: {src}")


def probe_duration(path: Path) -> float:
    ffprobe = shutil.which("ffprobe") or "/opt/homebrew/bin/ffprobe"
    return float(subprocess.check_output(
        [ffprobe, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        text=True,
    ).strip())


def concat(paths: list[Path], out: Path):
    ffmpeg = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        for p in paths:
            f.write(f"file '{p.resolve()}'\n")
        lst = f.name
    try:
        if subprocess.run(
            [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", str(out)],
            check=False,
        ).returncode != 0:
            raise RuntimeError("concat failed")
    finally:
        Path(lst).unlink(missing_ok=True)


def build(output: Path, w: int = 1280, h: int = 720, fps: int = 30) -> None:
    for clip in (SCRIPTED_CLIP, JOINT_CLIP, EE_CLIP):
        if not clip.exists():
            raise FileNotFoundError(f"missing physics-audit clip: {clip}")

    scripted_dur = min(probe_duration(SCRIPTED_CLIP), 26.0)
    slides = {
        "title": ("CONTROLLER-FIRST ROBOT VLA", [
            "MIDWAY TECHNICAL DEMO",
            "PHYSICS-AUDIT FOOTAGE  JUL 2026",
            "SCRIPTED EXPERT + LEARNED POLICIES",
        ]),
        "phases": ("PHASES 1-5 COMPLETE", [
            "[DONE] 1 CONTROLLER IK REACH",
            "[DONE] 2 ACTION-SPACE ADAPTERS",
            "[DONE] 3 TABLE CUBE PICKUP TASK",
            "[DONE] 4 EQUIVALENT DEMO LABELS",
            "[DONE] 5 STATE BC COMPARISON",
            "[NEXT] 6 VISION   [NEXT] 7 LANGUAGE VLA",
        ]),
        "bumper_scripted": ("SCRIPTED EXPERT", [
            "PHYSICS-AUDIT GATES ACTIVE",
            "FORCE + EVENT ORDER + COLLISION FREE",
            "YAW_0 CENTER  TRIALS 7 AND 8",
        ]),
        "gates": ("PHYSICS AUDIT GATES", [
            "FORCE 22N  IMPULSE 9N-S  SHIFT 13MM",
            "READINESS 288/288 PASS",
            "PRE-CLOSE CONTACT STEPS = 0",
            "NUMERICAL != VISUAL  SEE DOCS",
        ]),
        "bumper_joint": ("LEARNED POLICY", [
            "JOINT-DELTA BEHAVIORAL CLONING",
            "47/72 SUCCESS UNDER STRICT GATES",
        ]),
        "bumper_ee": ("LEARNED POLICY", [
            "EE-TOOL-DELTA BEHAVIORAL CLONING",
            "15/72 SUCCESS  IK SATURATION CAVEAT",
        ]),
        "verdict": ("PHASE-6 READINESS", [
            "SCRIPTED EXPERT: READY",
            "JOINT BC: 65%  EE BC: 21%",
            "VISION INFRA: GO  EE PRIMARY: BLOCKED",
        ]),
        "outro": ("RESEARCH QUESTION", [
            "DO CONTROLLER ACTIONS LEARN FASTER",
            "THAN RAW JOINT DELTAS?",
            "SAME TASK  SAME GATES  TWO SPACES",
        ]),
    }
    segments: list[tuple[str, float | None]] = [
        ("title", 8.0),
        ("phases", 18.0),
        ("bumper_scripted", 3.0),
        ("scripted", scripted_dur),
        ("gates", 12.0),
        ("bumper_joint", 3.0),
        ("joint_policy", 11.0),
        ("bumper_ee", 3.0),
        ("ee_policy", 10.0),
        ("verdict", 14.0),
        ("outro", 8.0),
    ]

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="midway_demo_") as tmp:
        tmp_path = Path(tmp)
        clips: list[Path] = []
        for key, duration in segments:
            out = tmp_path / f"{key}.mp4"
            if key in slides:
                title, lines = slides[key]
                render_slide(out, title, lines, float(duration), w, h, fps)
            elif key == "scripted":
                trim_clip(SCRIPTED_CLIP, out, float(duration), w, h, fps)
            elif key == "joint_policy":
                trim_clip(JOINT_CLIP, out, float(duration), w, h, fps)
            elif key == "ee_policy":
                trim_clip(EE_CLIP, out, float(duration), w, h, fps)
            else:
                raise ValueError(key)
            clips.append(out)
            print(f"  {key}: {duration}s  source={SCRIPTED_CLIP.name if key=='scripted' else 'slide' if key in slides else key}")

        concat(clips, output)
        dur = probe_duration(output)
        print(f"wrote {output} ({dur:.1f}s) scripted_source={SCRIPTED_CLIP.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUTS / "midway_technical_demo_silent.mp4")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    build(args.output, args.width, args.height, args.fps)


if __name__ == "__main__":
    main()