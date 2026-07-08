from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from svla.demo_recorder import PickupDemoRecorder
from svla.experiment_manifest import sha256_file, sha256_text, tracked_source_hashes
from svla.pickup_task import PickupTaskEvaluator, PickupTrialSpec
from svla.vision_observations import (
    DEFAULT_CAMERA_CONFIGS,
    FixedCameraConfig,
    FixedCameraRenderer,
    VISION_OBSERVATION_FORMAT,
    camera_metadata,
)


VISION_DATASET_FORMAT = "svla_pickup_vision_dataset_v1"


def record_pickup_vision_dataset(
    specs: Iterable[PickupTrialSpec],
    output_dir: Path,
    *,
    camera_configs: tuple[FixedCameraConfig, ...] = DEFAULT_CAMERA_CONFIGS,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    env = PickupTaskEvaluator()
    renderer = FixedCameraRenderer(env.model, camera_configs)
    recorder = PickupDemoRecorder(env)
    episodes: list[dict] = []
    output_files: list[Path] = []

    try:
        for episode_index, spec in enumerate(specs):
            demo = recorder.record_trial(spec)
            frames_by_camera = {
                config.name: np.empty(
                    (len(demo["samples"]), config.height, config.width, 3),
                    dtype=np.uint8,
                )
                for config in camera_configs
            }
            frame_index: list[dict] = []

            _replay_frames_for_demo(
                env=env,
                renderer=renderer,
                spec=spec,
                demo=demo,
                frames_by_camera=frames_by_camera,
                frame_index=frame_index,
            )

            episode_slug = _episode_slug(spec)
            demo_path = output_dir / f"{episode_slug}.json"
            demo_path.write_text(
                json.dumps(demo, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            output_files.append(demo_path)

            camera_records: dict[str, dict] = {}
            for camera_name, frames in frames_by_camera.items():
                frames_path = output_dir / f"{episode_slug}_{camera_name}_rgb.npz"
                np.savez_compressed(frames_path, rgb=frames)
                output_files.append(frames_path)
                camera_records[camera_name] = {
                    "path": frames_path.name,
                    "sha256": sha256_file(frames_path),
                    "array": "rgb",
                    "shape": list(frames.shape),
                    "dtype": str(frames.dtype),
                }

            episodes.append(
                {
                    "episode_index": episode_index,
                    "trial_spec": demo["metadata"]["trial_spec"],
                    "demo_path": demo_path.name,
                    "demo_sha256": sha256_file(demo_path),
                    "frame_count": len(demo["samples"]),
                    "cameras": camera_records,
                    "frame_index": frame_index,
                    "summary": demo["summary"],
                    "phase_summaries": demo["phase_summaries"],
                }
            )
    finally:
        renderer.close()

    manifest = {
        "format": VISION_DATASET_FORMAT,
        "vision_observation_format": VISION_OBSERVATION_FORMAT,
        "protocol_sha": _protocol_sha(),
        "source_hashes": tracked_source_hashes(Path(__file__).resolve().parents[2]),
        "source_demo_format": "svla_pickup_demo_v3_physics_audit",
        "action_space_neutral": True,
        "label_sets": ["joint_delta", "ee_delta", "ee_tool_delta"],
        "policy_label_sets": ["joint_delta", "ee_delta", "ee_tool_delta"],
        "camera_config": camera_metadata(camera_configs),
        "episode_count": len(episodes),
        "episodes": episodes,
    }
    manifest_path = output_dir / "vision_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest["output_files"] = [
        {"path": path.name, "sha256": sha256_file(path)}
        for path in sorted(output_files, key=lambda item: item.name)
    ]
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def validate_pickup_vision_dataset(dataset_dir: Path) -> dict:
    manifest_path = dataset_dir / "vision_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"missing dataset manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    issues: list[str] = []

    if manifest.get("format") != VISION_DATASET_FORMAT:
        issues.append("manifest format mismatch")
    if manifest.get("vision_observation_format") != VISION_OBSERVATION_FORMAT:
        issues.append("vision observation format mismatch")
    if manifest.get("action_space_neutral") is not True:
        issues.append("dataset must be action-space-neutral")

    _validate_manifest_output_hashes(dataset_dir, manifest, issues)

    camera_config = manifest.get("camera_config", {}).get("cameras", {})
    if not camera_config:
        issues.append("manifest has no camera config")

    total_frames = 0
    for episode in manifest.get("episodes", []):
        _validate_episode(dataset_dir, episode, camera_config, issues)
        total_frames += int(episode.get("frame_count", 0))

    expected_count = int(manifest.get("episode_count", -1))
    if expected_count != len(manifest.get("episodes", [])):
        issues.append("episode_count does not match episodes length")

    return {
        "format": "svla_pickup_vision_dataset_validation_v1",
        "valid": not issues,
        "issues": issues,
        "episode_count": len(manifest.get("episodes", [])),
        "total_frames": total_frames,
    }


def _validate_episode(
    dataset_dir: Path,
    episode: dict,
    camera_config: dict,
    issues: list[str],
) -> None:
    episode_index = episode.get("episode_index")
    demo_path = Path(episode.get("demo_path", ""))
    if not demo_path.is_absolute():
        demo_path = dataset_dir / demo_path.name
    if not demo_path.is_file():
        issues.append(f"missing demo file for episode {episode_index}")
        return
    _validate_hash_record(
        demo_path,
        episode.get("demo_sha256"),
        f"episode {episode_index} demo",
        issues,
    )
    demo = json.loads(demo_path.read_text(encoding="utf-8"))
    samples = demo.get("samples", [])
    frame_count = int(episode.get("frame_count", -1))
    if frame_count != len(samples):
        issues.append(f"episode {episode_index} frame_count mismatch")
    if demo.get("format") != "svla_pickup_demo_v3_physics_audit":
        issues.append(f"episode {episode_index} demo format mismatch")
    if not demo.get("summary", {}).get("success", False):
        issues.append(f"episode {episode_index} did not pass task gates")

    for sample_index, sample in enumerate(samples):
        if sample.get("step_index") != sample_index:
            issues.append(f"episode {episode_index} sample index mismatch")
            break
        if set(sample.get("policy_labels", {})) != {"joint_delta", "ee_delta", "ee_tool_delta"}:
            issues.append(f"episode {episode_index} missing policy labels")
            break
        if set(sample.get("labels", {})) != {"joint_delta", "ee_delta", "ee_tool_delta"}:
            issues.append(f"episode {episode_index} missing labels")
            break
        if "observation" not in sample or "success_metrics" not in sample:
            issues.append(f"episode {episode_index} missing state/metrics")
            break

    frame_index = episode.get("frame_index", [])
    if len(frame_index) != len(samples):
        issues.append(f"episode {episode_index} frame index length mismatch")
    for index, record in enumerate(frame_index):
        if record.get("step_index") != index:
            issues.append(f"episode {episode_index} frame step mismatch")
            break
        if record.get("demo_sample_index") != index:
            issues.append(f"episode {episode_index} demo sample mismatch")
            break

    for camera_name, record in episode.get("cameras", {}).items():
        if camera_name not in camera_config:
            issues.append(f"episode {episode_index} unknown camera {camera_name}")
            continue
        frames_path = Path(record.get("path", ""))
        if not frames_path.is_absolute():
            frames_path = dataset_dir / frames_path.name
        if not frames_path.is_file():
            issues.append(f"episode {episode_index} missing frame file")
            continue
        _validate_hash_record(
            frames_path,
            record.get("sha256"),
            f"episode {episode_index} camera {camera_name}",
            issues,
        )
        with np.load(frames_path) as payload:
            if record.get("array") not in payload.files:
                issues.append(f"episode {episode_index} missing frame array")
                continue
            frames = payload[record["array"]]
        expected_shape = (
            len(samples),
            int(camera_config[camera_name]["height"]),
            int(camera_config[camera_name]["width"]),
            3,
        )
        if frames.shape != expected_shape:
            issues.append(
                f"episode {episode_index} camera {camera_name} shape mismatch"
            )
        if frames.dtype != np.uint8:
            issues.append(
                f"episode {episode_index} camera {camera_name} dtype mismatch"
            )
        if list(frames.shape) != record.get("shape"):
            issues.append(
                f"episode {episode_index} camera {camera_name} manifest shape mismatch"
            )
        if str(frames.dtype) != record.get("dtype"):
            issues.append(
                f"episode {episode_index} camera {camera_name} manifest dtype mismatch"
            )


def _validate_manifest_output_hashes(
    dataset_dir: Path,
    manifest: dict,
    issues: list[str],
) -> None:
    output_files = manifest.get("output_files", [])
    if not output_files:
        issues.append("manifest has no output_files")
        return
    for record in output_files:
        file_path = Path(record.get("path", ""))
        if not file_path.is_absolute():
            file_path = dataset_dir / file_path.name
        if not file_path.is_file():
            issues.append(f"missing output file {record.get('path')}")
            continue
        _validate_hash_record(
            file_path,
            record.get("sha256"),
            f"output file {record.get('path')}",
            issues,
        )


def _validate_hash_record(
    path: Path,
    expected_sha256: object,
    label: str,
    issues: list[str],
) -> None:
    if not isinstance(expected_sha256, str) or not expected_sha256:
        issues.append(f"{label} missing sha256")
        return
    if sha256_file(path) != expected_sha256:
        issues.append(f"{label} sha256 mismatch")


def _replay_frames_for_demo(
    *,
    env: PickupTaskEvaluator,
    renderer: FixedCameraRenderer,
    spec: PickupTrialSpec,
    demo: dict,
    frames_by_camera: dict[str, np.ndarray],
    frame_index: list[dict],
) -> None:
    object_start = np.asarray(spec.object_pose.xyz, dtype=float)
    env.reset(object_start)
    commands, _, _ = env.scripted_controller_commands(spec, env.object_position.copy())
    sample_index = 0
    for command in commands:
        if sample_index >= len(demo["samples"]):
            break
        for phase_step in range(command.max_steps):
            sample = demo["samples"][sample_index]
            if sample["phase"] != command.phase or sample["phase_step"] != phase_step:
                raise RuntimeError("demo replay diverged from recorded phase sequence")
            for camera_name, frames in frames_by_camera.items():
                frames[sample_index] = renderer.render(env.data, camera_name)
            frame_index.append(
                {
                    "step_index": sample_index,
                    "demo_sample_index": sample_index,
                    "phase": sample["phase"],
                    "phase_step": int(sample["phase_step"]),
                    "camera_frames": {
                        camera_name: {
                            "array_index": sample_index,
                        }
                        for camera_name in frames_by_camera
                    },
                    "observation_time": sample["observation"]["time"],
                    "policy_label_keys": sorted(sample["policy_labels"]),
                }
            )
            _, _, status = env.step_controller_command(
                command.target_pos,
                command.target_quat_wxyz,
                command.gripper_open,
                substeps=4,
            )
            sample_index += 1
            if (
                command.stop_on_pose_tolerance
                and status.position_error <= env.controller.limits.position_tolerance
                and status.rotation_error <= env.controller.limits.rotation_tolerance
            ):
                break
            if sample_index >= len(demo["samples"]):
                break
    if sample_index != len(demo["samples"]):
        raise RuntimeError(
            f"rendered {sample_index} frames for {len(demo['samples'])} samples"
        )


def _episode_slug(spec: PickupTrialSpec) -> str:
    return (
        f"episode_{spec.trial_id:04d}_{spec.orientation.label}_"
        f"{spec.object_pose.label}_{spec.approach.label}"
    )


def _protocol_sha() -> str:
    protocol_path = (
        Path(__file__).resolve().parents[2]
        / "configs/phase5_evaluation_protocol_v2.json"
    )
    if protocol_path.is_file():
        return sha256_file(protocol_path)
    return sha256_text("phase6a_no_protocol_file")
