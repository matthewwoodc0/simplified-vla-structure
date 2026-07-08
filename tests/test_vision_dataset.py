import json
from pathlib import Path

import numpy as np

from svla.experiment_manifest import sha256_file
from svla.pickup_task import default_trial_specs
from svla.state_bc import load_demo_dataset
from svla.vision_dataset import (
    VISION_DATASET_FORMAT,
    _protocol_sha,
    record_pickup_vision_dataset,
    validate_pickup_vision_dataset,
)
from svla.vision_observations import FixedCameraConfig


def test_record_pickup_vision_dataset_manifest_and_frame_alignment(tmp_path):
    spec = default_trial_specs(repeats=1)[0]
    manifest = record_pickup_vision_dataset(
        [spec],
        tmp_path,
        camera_configs=(FixedCameraConfig(name="overview", width=64, height=48),),
    )

    assert manifest["format"] == VISION_DATASET_FORMAT
    assert manifest["action_space_neutral"] is True
    assert manifest["episode_count"] == 1
    assert manifest["protocol_sha"]
    assert "src/svla/vision_dataset.py" in manifest["source_hashes"]

    episode = manifest["episodes"][0]
    demo_path = tmp_path / Path(episode["demo_path"]).name
    demo = json.loads(demo_path.read_text())
    assert episode["frame_count"] == len(demo["samples"])
    assert len(episode["frame_index"]) == len(demo["samples"])
    assert set(demo["samples"][0]["policy_labels"]) == {
        "joint_delta",
        "ee_delta",
        "ee_tool_delta",
    }

    frame_record = episode["cameras"]["overview"]
    frame_path = tmp_path / Path(frame_record["path"]).name
    with np.load(frame_path) as payload:
        frames = payload[frame_record["array"]]
    assert frames.shape == (len(demo["samples"]), 48, 64, 3)
    assert frames.dtype == np.uint8
    assert frame_record["shape"] == list(frames.shape)
    assert episode["frame_index"][0]["observation_time"] == demo["samples"][0]["observation"]["time"]
    assert episode["summary"]["success"]


def test_validate_pickup_vision_dataset_accepts_generated_dataset(tmp_path):
    spec = default_trial_specs(repeats=1)[0]
    record_pickup_vision_dataset(
        [spec],
        tmp_path,
        camera_configs=(FixedCameraConfig(name="overview", width=64, height=48),),
    )

    summary = validate_pickup_vision_dataset(tmp_path)

    assert summary["valid"]
    assert summary["issues"] == []
    assert summary["episode_count"] == 1
    assert summary["total_frames"] > 0


def test_vision_dataset_keeps_non_vision_demo_loader_compatible(tmp_path):
    spec = default_trial_specs(repeats=1)[0]
    record_pickup_vision_dataset(
        [spec],
        tmp_path,
        camera_configs=(FixedCameraConfig(name="overview", width=64, height=48),),
    )

    manifest = json.loads((tmp_path / "vision_manifest.json").read_text())
    demo_path = tmp_path / Path(manifest["episodes"][0]["demo_path"]).name
    dataset = load_demo_dataset([demo_path], action_space="joint_delta", stride=20)

    assert dataset.demo_count == 1
    assert dataset.actions.shape[1] == 6
    assert dataset.features.shape[0] == dataset.actions.shape[0]


def test_validate_pickup_vision_dataset_reports_corrupt_frame_shape(tmp_path):
    spec = default_trial_specs(repeats=1)[0]
    record_pickup_vision_dataset(
        [spec],
        tmp_path,
        camera_configs=(FixedCameraConfig(name="overview", width=64, height=48),),
    )
    manifest = json.loads((tmp_path / "vision_manifest.json").read_text())
    frame_record = manifest["episodes"][0]["cameras"]["overview"]
    np.savez_compressed(
        tmp_path / Path(frame_record["path"]).name,
        rgb=np.zeros((1, 48, 64, 3), dtype=np.uint8),
    )

    summary = validate_pickup_vision_dataset(tmp_path)

    assert not summary["valid"]
    assert any("shape mismatch" in issue for issue in summary["issues"])


def test_validate_pickup_vision_dataset_reports_frame_hash_mismatch(tmp_path):
    spec = default_trial_specs(repeats=1)[0]
    record_pickup_vision_dataset(
        [spec],
        tmp_path,
        camera_configs=(FixedCameraConfig(name="overview", width=64, height=48),),
    )
    manifest = json.loads((tmp_path / "vision_manifest.json").read_text())
    frame_record = manifest["episodes"][0]["cameras"]["overview"]
    frame_path = tmp_path / Path(frame_record["path"]).name
    with np.load(frame_path) as payload:
        frames = payload[frame_record["array"]].copy()
    frames[0, 0, 0, 0] = (int(frames[0, 0, 0, 0]) + 1) % 256
    np.savez_compressed(frame_path, rgb=frames)

    summary = validate_pickup_vision_dataset(tmp_path)

    assert not summary["valid"]
    assert any("output file" in issue and "sha256 mismatch" in issue for issue in summary["issues"])
    assert any("camera overview sha256 mismatch" in issue for issue in summary["issues"])


def test_validate_pickup_vision_dataset_reports_demo_hash_mismatch(tmp_path):
    spec = default_trial_specs(repeats=1)[0]
    record_pickup_vision_dataset(
        [spec],
        tmp_path,
        camera_configs=(FixedCameraConfig(name="overview", width=64, height=48),),
    )
    manifest = json.loads((tmp_path / "vision_manifest.json").read_text())
    demo_path = tmp_path / Path(manifest["episodes"][0]["demo_path"]).name
    demo = json.loads(demo_path.read_text())
    demo["metadata"]["tampered_for_test"] = True
    demo_path.write_text(json.dumps(demo, indent=2, sort_keys=True) + "\n")

    summary = validate_pickup_vision_dataset(tmp_path)

    assert not summary["valid"]
    assert any("output file" in issue and "sha256 mismatch" in issue for issue in summary["issues"])
    assert any("episode 0 demo sha256 mismatch" in issue for issue in summary["issues"])


def test_protocol_sha_uses_repo_root_not_caller_cwd(tmp_path, monkeypatch):
    expected = sha256_file(
        Path(__file__).resolve().parents[1] / "configs/phase5_evaluation_protocol_v2.json"
    )
    monkeypatch.chdir(tmp_path)

    assert _protocol_sha() == expected
