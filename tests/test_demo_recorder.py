import json

import numpy as np

from svla.demo_recorder import PickupDemoRecorder
from svla.pickup_task import (
    LIFT_CLEARANCE,
    RETENTION_CLEARANCE,
    PickupTaskEvaluator,
    default_trial_specs,
)


def test_recorded_demo_contains_aligned_joint_and_ee_labels(tmp_path):
    recorder = PickupDemoRecorder(PickupTaskEvaluator())
    spec = default_trial_specs(repeats=1)[0]

    path = tmp_path / "demo.json"
    demo = recorder.write_trial(spec, path)
    loaded = json.loads(path.read_text(encoding="utf-8"))

    assert loaded["format"] == "svla_pickup_demo_v1"
    assert loaded["summary"]["success"]
    assert loaded["summary"]["failure_category"] == "none"
    assert loaded["metadata"]["no_ml"] is True
    assert loaded["samples"]

    sample = loaded["samples"][0]
    assert set(sample["labels"]) == {"joint_delta", "ee_delta", "ee_tool_delta"}
    assert set(sample["policy_labels"]) == {"joint_delta", "ee_delta", "ee_tool_delta"}
    assert len(sample["labels"]["joint_delta"]) == 6
    assert len(sample["labels"]["ee_delta"]) == 7
    assert len(sample["policy_labels"]["joint_delta"]) == 6
    assert len(sample["policy_labels"]["ee_delta"]) == 7
    assert len(sample["labels"]["ee_tool_delta"]) == 6
    assert len(sample["policy_labels"]["ee_tool_delta"]) == 6
    assert np.linalg.norm(sample["policy_labels"]["ee_delta"][:3]) <= 0.018 + 1e-9
    assert np.linalg.norm(sample["policy_labels"]["ee_delta"][3:6]) <= 0.08 + 1e-9

    before_joints = np.array(sample["observation"]["joint_positions"])
    after_joints = np.array(sample["next_observation"]["joint_positions"])
    before_ee = np.array(sample["observation"]["ee_position"])
    after_ee = np.array(sample["next_observation"]["ee_position"])

    assert np.allclose(sample["labels"]["joint_delta"][:5], after_joints - before_joints)
    assert np.allclose(sample["labels"]["ee_delta"][:3], after_ee - before_ee)
    assert not np.allclose(sample["policy_labels"]["ee_delta"][:3], sample["labels"]["ee_delta"][:3])
    assert "clipped_joints" in sample["controller_telemetry"]
    assert "feasible_delta_xyz" in sample["controller_telemetry"]
    assert "contact_achieved" in sample["success_metrics"]
    assert demo["summary"]["retained_during_hold"]


def test_policy_labels_replay_through_action_space_apis():
    recorder = PickupDemoRecorder(PickupTaskEvaluator())
    spec = default_trial_specs(repeats=1)[0]
    demo = recorder.record_trial(spec)

    for action_space in ("joint_delta", "ee_tool_delta"):
        env = PickupTaskEvaluator()
        env.reset(spec.object_pose.xyz)
        joint_limit_clipped = 0
        for sample in demo["samples"]:
            action = np.asarray(sample["policy_labels"][action_space], dtype=float)
            if action_space == "joint_delta":
                _, _, status = env.step_joint_delta_action(action[:5], action[5])
                joint_limit_clipped += int(status["joint_limit_clipped"])
            else:
                _, _, status = env.step_ee_tool_delta_action(
                    action[:3], action[3:5], action[5]
                )
                joint_limit_clipped += int(status.joint_limit_clipped)

        metrics = env.get_success_metrics()
        assert metrics["contact_achieved"]
        assert metrics["max_object_lift"] >= LIFT_CLEARANCE
        assert metrics["current_object_lift"] >= RETENTION_CLEARANCE
        assert metrics["retained_during_hold"]
        assert joint_limit_clipped < len(demo["samples"])


def test_recorded_demo_is_deterministic():
    spec = default_trial_specs(repeats=1)[0]
    first = PickupDemoRecorder(PickupTaskEvaluator()).record_trial(spec)
    second = PickupDemoRecorder(PickupTaskEvaluator()).record_trial(spec)

    assert first == second
