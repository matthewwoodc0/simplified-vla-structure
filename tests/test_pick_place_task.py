import json

import numpy as np
import pytest

from svla.demo_recorder import PickupDemoRecorder
from svla.pick_place_replay import replay_demo_policy_labels
from svla.pickup_task import (
    MAX_GRIPPER_CONTACT_FORCE,
    MAX_GRIPPER_IMPULSE_BEFORE_LIFT,
    OBJECT_START_Z,
    PLACEMENT_XY_TOLERANCE,
    PLACEMENT_Z_TOLERANCE,
    SUPPORT_TOP_Z,
    ApproachStrategy,
    GraspOrientation,
    ObjectStartPose,
    PickPlaceTrialSpec,
    PickupTaskEvaluator,
    PlacementTarget,
    default_pick_place_trial_specs,
    summarize_pick_place_results,
)


def test_default_pick_place_matrix_covers_required_buckets():
    specs = default_pick_place_trial_specs()
    assert len(specs) >= 6
    assert len({spec.placement_target.label for spec in specs}) >= 2
    assert len({spec.object_pose.label for spec in specs}) >= 2


def test_scripted_pick_place_command_sequence_extends_pickup():
    evaluator = PickupTaskEvaluator()
    spec = default_pick_place_trial_specs()[0]
    evaluator.reset(spec.object_pose.xyz)
    commands, grasp_pos, grasp_quat, place_pos = evaluator.scripted_pick_place_commands(spec)
    phases = [command.phase for command in commands]
    grasp_index = phases.index("grasp_align")
    hold_index = phases.index("hold")
    assert grasp_index > 0
    assert phases[hold_index + 1 : hold_index + 5] == [
        "transport",
        "lower",
        "open_gripper",
        "retreat",
    ]
    assert place_pos[2] == SUPPORT_TOP_Z + evaluator.object_half_size[2]
    assert np.allclose(place_pos, evaluator.placement_goal_xyz(spec.placement_target))
    assert grasp_pos.shape == (3,)
    assert grasp_quat.shape == (4,)


def test_place_left_uses_scene_command_marker_for_transport():
    evaluator = PickupTaskEvaluator()
    target = PlacementTarget(
        "place_left",
        "place_left_marker",
        "place_left_command_marker",
    )
    goal = evaluator.placement_goal_xyz(target)
    command = evaluator.placement_command_xyz(target)
    assert command[0] == -0.067
    assert goal[0] == -0.055
    right = PlacementTarget("place_right", "place_right_marker")
    assert np.allclose(
        evaluator.placement_command_xyz(right),
        evaluator.placement_goal_xyz(right),
    )


def test_evaluate_placement_classification():
    evaluator = PickupTaskEvaluator()
    evaluator.reset(np.array([0.0, -0.235, OBJECT_START_Z]))
    evaluator.data.qpos[evaluator.controller.gripper_qpos_ids] = 0.05
    target = np.array([0.06, -0.235, SUPPORT_TOP_Z + evaluator.object_half_size[2]])
    ok, metrics = evaluator.evaluate_placement(target)
    assert not ok
    assert metrics["placement_xy_error"] > PLACEMENT_XY_TOLERANCE
    assert not metrics["gripper_released"]


def test_representative_pick_place_succeeds():
    evaluator = PickupTaskEvaluator()
    spec = PickPlaceTrialSpec(
        trial_id=99,
        orientation=GraspOrientation("yaw_0", 0.0),
        object_pose=ObjectStartPose("center", np.array([0.0, -0.235, OBJECT_START_Z])),
        approach=ApproachStrategy("vertical_pregrasp", "world_z"),
        placement_target=PlacementTarget("place_right", "place_right_marker", "place_right_marker"),
    )
    result = evaluator.run_pick_place_trial(spec)
    assert result.success
    assert result.placement_achieved
    assert result.gripper_released
    assert result.placement_xy_error <= PLACEMENT_XY_TOLERANCE
    assert result.placement_z_error <= PLACEMENT_Z_TOLERANCE
    assert result.event_order_valid
    assert result.physical_sanity_pass
    assert result.collision_free_approach
    assert result.reopen_events == 0
    assert result.max_gripper_contact_force <= MAX_GRIPPER_CONTACT_FORCE
    assert result.gripper_contact_impulse_before_lift <= MAX_GRIPPER_IMPULSE_BEFORE_LIFT
    assert result.failure_category == "none"


def test_pick_place_summary_reports_rates():
    evaluator = PickupTaskEvaluator()
    results = [
        evaluator.run_pick_place_trial(spec)
        for spec in default_pick_place_trial_specs()[:2]
    ]
    summary = summarize_pick_place_results(results)
    assert summary["total"] == 2
    assert "by_placement_target" in summary
    assert "placement_achieved_rate" in summary


@pytest.mark.parametrize(
    ("goal_marker", "command_marker", "expect_placement"),
    [
        ("place_left_marker", "place_left_command_marker", True),
        ("place_left_marker", "place_left_marker", False),
    ],
)
def test_place_left_command_marker_ablation(goal_marker, command_marker, expect_placement):
    evaluator = PickupTaskEvaluator()
    spec = PickPlaceTrialSpec(
        trial_id=101,
        orientation=GraspOrientation("yaw_0", 0.0),
        object_pose=ObjectStartPose("center", np.array([0.0, -0.235, OBJECT_START_Z])),
        approach=ApproachStrategy("vertical_pregrasp", "world_z"),
        placement_target=PlacementTarget("place_left", goal_marker, command_marker),
    )
    result = evaluator.run_pick_place_trial(spec)
    assert result.placement_achieved is expect_placement
    if not expect_placement:
        assert result.placement_xy_error > PLACEMENT_XY_TOLERANCE


def test_pick_place_policy_label_replay_preserves_grasp_event_order():
    recorder = PickupDemoRecorder(PickupTaskEvaluator())
    spec = PickPlaceTrialSpec(
        trial_id=1,
        orientation=GraspOrientation("yaw_0", 0.0),
        object_pose=ObjectStartPose("center", np.array([0.0, -0.235, OBJECT_START_Z])),
        approach=ApproachStrategy("vertical_pregrasp", "world_z"),
        placement_target=PlacementTarget("place_right", "place_right_marker", "place_right_marker"),
    )
    demo = recorder.record_pick_place_trial(spec)
    assert demo["metadata"]["grasp_segment_finalize_sample_index"] is not None
    replay = replay_demo_policy_labels(
        demo,
        "ee_tool_delta",
        np.asarray(spec.object_pose.xyz, dtype=float),
        task="pick_place",
    )
    assert replay["event_order_valid"]
    assert replay["reopen_events"] == 0
    assert replay["physical_sanity_pass"]
    assert replay["grasp_segment_finalize_sample_index"] == demo["metadata"][
        "grasp_segment_finalize_sample_index"
    ]


def test_recorded_pick_place_demo_contains_aligned_labels(tmp_path):
    recorder = PickupDemoRecorder(PickupTaskEvaluator())
    spec = PickPlaceTrialSpec(
        trial_id=1,
        orientation=GraspOrientation("yaw_0", 0.0),
        object_pose=ObjectStartPose("center", np.array([0.0, -0.235, OBJECT_START_Z])),
        approach=ApproachStrategy("vertical_pregrasp", "world_z"),
        placement_target=PlacementTarget("place_right", "place_right_marker", "place_right_marker"),
    )
    path = tmp_path / "pick_place_demo.json"
    demo = recorder.write_pick_place_trial(spec, path)
    loaded = json.loads(path.read_text(encoding="utf-8"))

    assert loaded["format"] == "svla_pick_place_demo_v1"
    assert loaded["summary"]["success"]
    assert loaded["summary"]["placement_achieved"]
    assert loaded["metadata"]["trial_spec"]["placement_target"] == "place_right"
    assert loaded["samples"]

    sample = loaded["samples"][0]
    assert set(sample["labels"]) == {"joint_delta", "ee_delta", "ee_tool_delta"}
    assert set(sample["policy_labels"]) == {"joint_delta", "ee_delta", "ee_tool_delta"}
    assert len(sample["labels"]["joint_delta"]) == 6
    assert len(sample["labels"]["ee_tool_delta"]) == 6
    assert len(sample["policy_labels"]["joint_delta"]) == 6
    assert len(sample["policy_labels"]["ee_tool_delta"]) == 6
    assert demo["summary"]["event_order_valid"]
    assert demo["summary"]["physical_sanity_pass"]