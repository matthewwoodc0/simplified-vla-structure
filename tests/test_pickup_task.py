import numpy as np

from svla.pickup_task import (
    BASE_OBJECT_HALF_SIZE,
    MAX_GRIPPER_CONTACT_FORCE,
    MAX_GRIPPER_IMPULSE_BEFORE_LIFT,
    MAX_SUPPORTED_XY_DISPLACEMENT,
    SUPPORT_TOP_Z,
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
    assert result.collision_free_approach
    assert result.preclose_contact_steps == 0
    assert result.preclose_max_object_displacement <= 0.001
    assert result.event_order_valid
    assert not result.early_close
    assert result.reopen_events == 0
    assert result.physical_sanity_pass
    assert result.max_gripper_contact_force <= MAX_GRIPPER_CONTACT_FORCE
    assert (
        result.gripper_contact_impulse_before_lift
        <= MAX_GRIPPER_IMPULSE_BEFORE_LIFT
    )
    assert (
        result.max_object_xy_displacement_while_supported
        <= MAX_SUPPORTED_XY_DISPLACEMENT
    )
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
    assert metrics["collision_free_approach"]
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


def test_scripted_open_gripper_approach_does_not_touch_or_move_object():
    evaluator = PickupTaskEvaluator()
    spec = default_trial_specs(repeats=1)[0]
    evaluator.reset(spec.object_pose.xyz)
    settled_start = evaluator.object_position.copy()
    commands, grasp_pos, _ = evaluator.scripted_controller_commands(spec, settled_start)

    assert np.allclose(grasp_pos[:2], settled_start[:2], atol=1e-12)
    assert grasp_pos[2] == OBJECT_START_Z
    for command in commands:
        if command.gripper_open <= 0.5:
            break
        for _ in range(command.max_steps):
            _, metrics, status = evaluator.step_controller_command(
                command.target_pos,
                command.target_quat_wxyz,
                command.gripper_open,
            )
            if (
                command.stop_on_pose_tolerance
                and status.position_error <= evaluator.controller.limits.position_tolerance
                and status.rotation_error <= evaluator.controller.limits.rotation_tolerance
            ):
                break

    assert metrics["collision_free_approach"]
    assert metrics["preclose_contact_steps"] == 0
    assert metrics["preclose_max_object_displacement"] <= 1e-9


def test_early_close_and_reopen_are_reported_as_invalid_event_order():
    evaluator = PickupTaskEvaluator()
    observation = evaluator.reset(np.array([0.0, -0.235, OBJECT_START_Z]))
    zero_joint_delta = np.zeros(len(observation["joint_positions"]))

    evaluator.step_joint_delta_action(zero_joint_delta, gripper_open=0.0)
    _, metrics, _ = evaluator.step_joint_delta_action(
        zero_joint_delta,
        gripper_open=1.0,
    )

    assert metrics["early_close"]
    assert metrics["reopen_events"] == 1
    assert metrics["reopen_command_steps"] == 1
    assert not metrics["event_order_valid"]


def test_object_geometry_and_friction_configuration_updates_dynamics_and_grasp_target():
    half_size = BASE_OBJECT_HALF_SIZE * np.array([0.95, 1.05, 0.95])
    evaluator = PickupTaskEvaluator(
        object_half_size=half_size,
        object_sliding_friction=1.6,
    )
    spec = default_trial_specs(repeats=1)[0]
    xyz = spec.object_pose.xyz.copy()
    xyz[2] = SUPPORT_TOP_Z + half_size[2]
    evaluator.reset(xyz)
    settled = evaluator.object_position.copy()
    _, grasp_pos, _ = evaluator.scripted_controller_commands(spec, settled)

    assert np.allclose(evaluator.model.geom_size[evaluator.object_geom_id], half_size)
    assert evaluator.model.geom_friction[evaluator.object_geom_id, 0] == 1.6
    assert evaluator.object_position[2] > SUPPORT_TOP_Z
    assert grasp_pos[2] == OBJECT_START_Z
    assert not np.allclose(grasp_pos[:2], settled[:2])


def test_contact_model_parameters_are_bounded_and_valid():
    evaluator = PickupTaskEvaluator()
    jaw_actuator = evaluator.controller.gripper_actuator_ids[0]
    pad_ids = list(evaluator.gripper_geom_ids)

    assert np.allclose(evaluator.model.actuator_forcerange[jaw_actuator], [-0.2, 0.2])
    assert np.all(evaluator.model.geom_solimp[pad_ids, 0] <= 1.0)
    assert np.all(evaluator.model.geom_solimp[pad_ids, 0] <= evaluator.model.geom_solimp[pad_ids, 1])
