#!/usr/bin/env python3
"""One-off: synthesize 8-bit background music and mux onto midway demo MP4.

Not part of the core experiment stack — lives under demos/ only.
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_RATE = 44_100
BPM = 142


def _square(phase: np.ndarray, duty: float = 0.5) -> np.ndarray:
    return np.where(np.mod(phase, 1.0) < duty, 1.0, -1.0)


def _tri(phase: np.ndarray) -> np.ndarray:
    return 2.0 * np.abs(2.0 * (phase - np.floor(phase + 0.5))) - 1.0


def _env(length: int, attack: int, release: int) -> np.ndarray:
    env = np.ones(length, dtype=np.float64)
    if attack > 0:
        env[:attack] = np.linspace(0.0, 1.0, attack)
    if release > 0:
        env[-release:] = np.linspace(1.0, 0.0, release)
    return env


_SEMITONE = {
    "A2": -12,
    "C3": 0,
    "D3": 2,
    "E3": 4,
    "F3": 5,
    "G3": 7,
    "A3": 9,
    "B3": 11,
    "C4": 12,
    "D4": 14,
    "E4": 16,
    "G4": 19,
    "A4": 21,
    "B4": 23,
    "C5": 24,
    "REST": None,
}


def _note_freq(note: str) -> float | None:
    if note == "REST" or _SEMITONE.get(note) is None:
        return None
    return 130.81 * (2.0 ** (_SEMITONE[note] / 12.0))


def _note_samples(
    note: str,
    duration_beats: float,
    *,
    waveform: str = "square",
) -> tuple[np.ndarray, int]:
    beat_samples = int(SAMPLE_RATE * 60.0 / BPM)
    length = max(1, int(beat_samples * duration_beats))
    freq = _note_freq(note)
    if freq is None:
        return np.zeros(length, dtype=np.float64), length

    t = np.arange(length, dtype=np.float64) / SAMPLE_RATE
    phase = t * freq
    if waveform == "tri":
        sample = _tri(phase)
    else:
        sample = _square(phase, duty=0.25)
    return sample, length


def _blit(
    mix: np.ndarray,
    pos: int,
    sample: np.ndarray,
    *,
    gain: float,
    attack: int,
    release: int,
) -> int:
    end = min(len(mix), pos + len(sample))
    seg_len = end - pos
    if seg_len <= 0:
        return pos + len(sample)
    seg = sample[:seg_len] * _env(seg_len, attack, release)
    mix[pos:end] += seg * gain
    return pos + len(sample)


def synthesize_chiptune(duration_s: float) -> np.ndarray:
    """Upbeat retro 8-bit: C-major bounce, lead hook, punchy kit."""
    total = int(math.ceil(duration_s * SAMPLE_RATE))
    mix = np.zeros(total, dtype=np.float64)

    # 2-bar loop — bright I / V / vi / IV (C major family)
    bass_line = [
        ("C3", 0.5), ("C3", 0.5), ("G3", 0.5), ("G3", 0.5),
        ("A3", 0.5), ("A3", 0.5), ("F3", 0.5), ("G3", 0.5),
        ("C3", 0.5), ("E3", 0.5), ("G3", 0.5), ("C4", 0.5),
        ("F3", 0.5), ("A3", 0.5), ("G3", 0.5), ("C4", 0.5),
    ]
    arp_pattern = [
        ("C4", 0.125), ("E4", 0.125), ("G4", 0.125), ("C5", 0.125),
        ("G4", 0.125), ("E4", 0.125), ("C4", 0.125), ("G3", 0.125),
        ("A3", 0.125), ("C4", 0.125), ("E4", 0.125), ("A4", 0.125),
        ("G4", 0.125), ("E4", 0.125), ("C4", 0.125), ("A3", 0.125),
        ("F3", 0.125), ("A3", 0.125), ("C4", 0.125), ("F4", 0.125),
        ("G3", 0.125), ("B3", 0.125), ("D4", 0.125), ("G4", 0.125),
        ("C4", 0.125), ("E4", 0.125), ("G4", 0.125), ("C5", 0.125),
        ("B4", 0.125), ("G4", 0.125), ("E4", 0.125), ("C4", 0.125),
        ("A4", 0.125), ("F4", 0.125), ("D4", 0.125), ("A3", 0.125),
        ("G4", 0.125), ("E4", 0.125), ("C4", 0.125), ("G3", 0.125),
    ]
    lead_hook = [
        ("E4", 0.25), ("G4", 0.25), ("C5", 0.5),
        ("B4", 0.25), ("G4", 0.25), ("E4", 0.5),
        ("F4", 0.25), ("A4", 0.25), ("C5", 0.5),
        ("G4", 0.25), ("E4", 0.25), ("C4", 0.5),
        ("C5", 0.25), ("G4", 0.25), ("E4", 0.25), ("C4", 0.25),
        ("D4", 0.25), ("E4", 0.25), ("G4", 0.5),
        ("A4", 0.25), ("G4", 0.25), ("E4", 0.5),
        ("G4", 0.25), ("A4", 0.25), ("C5", 0.5),
        ("REST", 0.5),
    ]

    cursor = 0
    bar_samples = int(SAMPLE_RATE * 60.0 / BPM * 4)
    loop_samples = bar_samples * 2
    beat = int(SAMPLE_RATE * 60.0 / BPM)
    sixteenth = beat // 4

    while cursor < total:
        loop_start = cursor

        pos = loop_start
        for note, beats in bass_line:
            if pos >= total:
                break
            wave_form, _ = _note_samples(note, beats)
            pos = _blit(mix, pos, wave_form, gain=0.26, attack=30, release=180)

        pos = loop_start
        for note, beats in arp_pattern:
            if pos >= total:
                break
            wave_form, _ = _note_samples(note, beats, waveform="tri")
            pos = _blit(mix, pos, wave_form, gain=0.11, attack=15, release=80)

        pos = loop_start
        for note, beats in lead_hook:
            if pos >= total:
                break
            wave_form, _ = _note_samples(note, beats)
            pos = _blit(mix, pos, wave_form, gain=0.14, attack=10, release=60)

        # Punchy 8-bit kit: kick on 1+3, snare on 2+4, hats on 8ths
        for step in range(32):
            hit = loop_start + step * sixteenth
            if hit >= total:
                break
            if step % 8 == 0 or step % 8 == 4:
                klen = min(420, total - hit)
                t = np.arange(klen, dtype=np.float64) / SAMPLE_RATE
                sweep = np.linspace(130.0, 48.0, klen)
                phase = np.cumsum(sweep / SAMPLE_RATE) % 1.0
                kick = _square(phase, duty=0.55)
                mix[hit : hit + klen] += kick * _env(klen, 4, 280) * 0.28
            if step % 8 == 4 or step % 8 == 12:
                slen = min(700, total - hit)
                noise = np.random.default_rng(step + loop_start).uniform(-1, 1, slen)
                mix[hit : hit + slen] += noise * _env(slen, 6, 220) * 0.11
            if step % 2 == 1:
                hlen = min(500, total - hit)
                noise = np.random.default_rng(step * 17 + loop_start).uniform(-1, 1, hlen)
                mix[hit : hit + hlen] += noise * _env(hlen, 3, 90) * 0.055

        cursor += loop_samples

    fade = int(0.5 * SAMPLE_RATE)
    mix[:fade] *= np.linspace(0.0, 1.0, fade)
    mix[-fade:] *= np.linspace(1.0, 0.0, fade)

    peak = np.max(np.abs(mix)) or 1.0
    return np.clip(mix / peak * 0.88, -1.0, 1.0)


def write_wav(path: Path, samples: np.ndarray) -> None:
    pcm = (samples * 32767.0).astype(np.int16)
    with wave.open(str(path), "w") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm.tobytes())


def mux_audio(video_path: Path, audio_path: Path, output_path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        str(output_path),
    ]
    if subprocess.run(command, check=False).returncode != 0:
        raise RuntimeError("ffmpeg mux failed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Add procedural 8-bit music to midway demo.")
    parser.add_argument(
        "--video",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "midway_technical_demo.mp4",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "midway_technical_demo.mp4",
    )
    args = parser.parse_args()

    video_path = args.video

    if not video_path.exists():
        raise FileNotFoundError(f"missing video: {video_path}")

    probe = shutil.which("ffprobe") or "/opt/homebrew/bin/ffprobe"
    duration = float(
        subprocess.check_output(
            [
                probe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            text=True,
        ).strip()
    )

    audio_path = args.output.with_suffix(".wav")
    print(f"synthesizing {duration:.1f}s chiptune -> {audio_path.name}")
    write_wav(audio_path, synthesize_chiptune(duration + 0.5))

    temp_out = args.output.with_name(args.output.stem + "_with_music.mp4")
    print(f"muxing audio into {temp_out.name}")
    mux_audio(video_path, audio_path, temp_out)

    if args.output.resolve() == args.video.resolve():
        backup = video_path.with_name(video_path.stem + "_no_audio.mp4")
        if video_path.exists():
            shutil.copy2(str(video_path), str(backup))
        if args.output.exists():
            args.output.unlink()
        shutil.move(str(temp_out), str(args.output))
        print(f"wrote {args.output} (silent backup: {backup.name})")
    else:
        shutil.move(str(temp_out), str(args.output))
        print(f"wrote {args.output}")

    audio_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()