import numpy as np

from svla.pickup_task import (
    GraspOrientation,
    ObjectStartPose,
    PickupTaskEvaluator,
    PickupTrialSpec,
    default_trial_specs,
    summarize_results,
    OBJECT_START_Z,
)


def test_default_pickup_trial_matrix_covers_required_buckets():
    specs = default_trial_specs(repeats=2)
    assert len(specs) == 36
    assert len({spec.orientation.label for spec in specs}) == 3
    assert len({spec.object_pose.label for spec in specs}) == 3
    assert len({spec.approach.label for spec in specs}) == 2


def test_representative_controller_only_pickup_succeeds():
    evaluator = PickupTaskEvaluator()
    spec = PickupTrialSpec(
        trial_id=1,
        orientation=GraspOrientation("yaw_0", 0.0),
        object_pose=ObjectStartPose("center", np.array([0.0, -0.235, OBJECT_START_Z])),
        approach=default_trial_specs(repeats=1)[0].approach,
    )
    result = evaluator.run_trial(spec)
    assert result.success
    assert result.contact_achieved
    assert result.object_lifted
    assert result.retained_during_hold
    assert result.failure_category == "none"
    assert result.final_ee_position_error < 0.01


def test_pickup_summary_reports_bucket_rates():
    evaluator = PickupTaskEvaluator()
    results = [evaluator.run_trial(spec) for spec in default_trial_specs(repeats=1)[:3]]
    summary = summarize_results(results)
    assert summary["total"] == 3
    assert "by_orientation" in summary
    assert "by_object_pose" in summary
    assert "by_approach" in summary


def test_pickup_task_exposes_environment_style_api():
    evaluator = PickupTaskEvaluator()
    spec = default_trial_specs(repeats=1)[0]

    observation = evaluator.reset(spec.object_pose.xyz)
    commands, grasp_pos, grasp_quat = evaluator.scripted_controller_commands(spec)
    next_observation, metrics, status = evaluator.step_controller_command(
        commands[0].target_pos,
        commands[0].target_quat_wxyz,
        commands[0].gripper_open,
    )

    assert len(commands) >= 5
    assert grasp_pos.shape == (3,)
    assert grasp_quat.shape == (4,)
    assert len(observation["joint_positions"]) == 5
    assert len(next_observation["ee_position"]) == 3
    assert "max_object_lift" in metrics
    assert np.isfinite(status.position_error)


def test_pickup_task_steps_policy_action_spaces():
    evaluator = PickupTaskEvaluator()
    spec = default_trial_specs(repeats=1)[0]

    observation = evaluator.reset(spec.object_pose.xyz)
    next_observation, metrics, joint_status = evaluator.step_joint_delta_action(
        np.zeros(len(observation["joint_positions"])),
        gripper_open=1.0,
    )

    assert len(next_observation["joint_positions"]) == len(observation["joint_positions"])
    assert "max_object_lift" in metrics
    assert joint_status["clipped_joints"] is False

    next_observation, metrics, ee_status = evaluator.step_ee_delta_action(
        np.zeros(3),
        np.zeros(3),
        gripper_open=1.0,
    )

    assert len(next_observation["ee_position"]) == 3
    assert "contact_steps" in metrics
    assert np.isfinite(ee_status.position_error)

    next_observation, metrics, tool_status = evaluator.step_ee_tool_delta_action(
        np.zeros(3),
        np.zeros(2),
        gripper_open=1.0,
    )

    assert len(next_observation["ee_position"]) == 3
    assert np.isfinite(tool_status.rotation_error)
