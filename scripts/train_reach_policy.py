from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import sys

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from svla.sim import ArmSim


@dataclass(frozen=True)
class LinearPolicy:
    weights: np.ndarray
    max_action_norm: float = 0.025

    def act(self, observation: np.ndarray) -> np.ndarray:
        features = np.append(observation, 1.0)
        action = features @ self.weights
        norm = float(np.linalg.norm(action))
        if norm > self.max_action_norm:
            action = action * (self.max_action_norm / norm)
        return action


def observe(sim: ArmSim, target: np.ndarray) -> np.ndarray:
    ee = sim.ee_position
    return np.concatenate((ee, target, target - ee))


def expert_action(observation: np.ndarray) -> np.ndarray:
    return observation[6:9]


def sample_training_data(seed: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    sim = ArmSim()
    base = sim.ee_position.copy()
    observations = []
    actions = []
    for _ in range(samples):
        ee_offset = rng.uniform([-0.08, -0.10, -0.06], [0.08, 0.10, 0.06])
        target_offset = rng.uniform([-0.08, -0.10, -0.06], [0.08, 0.10, 0.06])
        ee = base + ee_offset
        target = base + target_offset
        obs = np.concatenate((ee, target, target - ee))
        observations.append(obs)
        actions.append(expert_action(obs))
    return np.array(observations), np.array(actions)


def train_policy(seed: int = 7, samples: int = 512) -> LinearPolicy:
    observations, actions = sample_training_data(seed, samples)
    features = np.column_stack((observations, np.ones(len(observations))))
    weights, *_ = np.linalg.lstsq(features, actions, rcond=None)
    train_mae = float(np.mean(np.abs(features @ weights - actions)))
    print(f"trained linear reach policy: samples={samples} train_mae={train_mae:.5f}")
    return LinearPolicy(weights)


def run_episode(policy: LinearPolicy, target: np.ndarray, render: bool = False):
    sim = ArmSim()
    sim.set_target_marker(target)
    frames = []
    renderer = mujoco.Renderer(sim.model, height=720, width=960) if render else None
    for _ in range(400):
        action = policy.act(observe(sim, target))
        sim.controller.move_toward(sim.data, sim.ee_position + action)
        for _ in range(6):
            mujoco.mj_step(sim.model, sim.data)
        if renderer is not None:
            renderer.update_scene(sim.data, camera="overview")
            frames.append(renderer.render())
    if renderer is not None:
        renderer.close()
    return float(np.linalg.norm(target - sim.ee_position)), frames


def write_video(frames: list[np.ndarray], output_path: Path, fps: int = 30) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to export MP4 videos.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width, _ = frames[0].shape
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
        for frame in frames:
            process.stdin.write(frame.tobytes())
        process.stdin.close()
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg failed with exit code {return_code}.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument(
        "--video",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "trained_reach_policy.mp4",
    )
    args = parser.parse_args()

    policy = train_policy(seed=args.seed, samples=args.samples)
    sim = ArmSim()
    target = sim.ee_position + np.array([-0.055, 0.075, 0.025])
    final_error, frames = run_episode(policy, target, render=True)
    write_video(frames, args.video)
    print(f"eval_final_error={final_error:.4f}m")
    print(f"wrote {args.video}")


if __name__ == "__main__":
    main()
