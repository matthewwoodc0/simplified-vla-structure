import numpy as np

from svla.pickup_task import OBJECT_START_Z, PickupTaskEvaluator
from svla.sim import ArmSim


def test_ee_delta_action_target_is_local_to_current_pose():
    env = PickupTaskEvaluator()
    env.reset(np.array([0.0, -0.235, OBJECT_START_Z]))
    delta = np.array([0.004, -0.002, 0.001])

    for _ in range(3):
        before_pos = env.controller.ee_pose(env.data)[0]
        env.step_ee_delta_action(delta, np.zeros(3), gripper_open=1.0)
        assert np.allclose(
            env.controller.last_telemetry.target_pos,
            before_pos + delta,
            atol=1e-9,
        )


def test_ee_delta_rollout_is_deterministic_for_same_actions():
    actions = [
        (np.array([0.004, 0.0, 0.001]), np.array([0.0, 0.0, 0.006]), 1.0),
        (np.array([0.002, -0.001, 0.0]), np.array([0.0, 0.003, 0.0]), 0.8),
        (np.array([-0.001, 0.002, -0.001]), np.zeros(3), 0.5),
    ]
    finals = []
    for _ in range(2):
        env = PickupTaskEvaluator()
        observation = env.reset(np.array([0.0, -0.235, OBJECT_START_Z]))
        for delta_xyz, delta_rotvec, gripper in actions:
            observation, _, _ = env.step_ee_delta_action(delta_xyz, delta_rotvec, gripper)
        finals.append(observation)

    for key in ("joint_positions", "joint_velocities", "ee_position", "ee_quat_wxyz"):
        assert np.allclose(finals[0][key], finals[1][key], atol=1e-12)


def test_nearby_ee_actions_produce_nearby_targets_and_joint_commands():
    first = PickupTaskEvaluator()
    second = PickupTaskEvaluator()
    object_start = np.array([0.0, -0.235, OBJECT_START_Z])
    first.reset(object_start)
    second.reset(object_start)

    first.step_ee_delta_action(np.array([0.005, 0.0, 0.0]), np.zeros(3), gripper_open=1.0)
    second.step_ee_delta_action(np.array([0.006, 0.0, 0.0]), np.zeros(3), gripper_open=1.0)

    first_telemetry = first.controller.last_telemetry
    second_telemetry = second.controller.last_telemetry
    assert np.linalg.norm(first_telemetry.target_pos - second_telemetry.target_pos) <= 0.0011
    assert (
        np.linalg.norm(first_telemetry.joint_targets - second_telemetry.joint_targets)
        <= 0.004
    )


def test_ee_delta_reports_unreachable_and_saturation_details():
    env = PickupTaskEvaluator()
    env.reset(np.array([0.0, -0.235, OBJECT_START_Z]))

    _, _, status = env.step_ee_delta_action(
        np.array([1.0, 1.0, 1.0]),
        np.array([1.0, 1.0, 1.0]),
        gripper_open=1.0,
    )

    assert status.clipped_translation
    assert status.clipped_rotation
    assert status.clipped_joints
    assert status.joint_step_clipped or status.joint_accel_clipped or status.joint_limit_clipped
    assert status.joint_step_norm <= env.controller.limits.max_joint_step + 1e-9
    assert np.isfinite(status.posture_error)
    assert status.saturated


def test_ee_delta_reports_controller_failure_for_non_finite_target():
    env = PickupTaskEvaluator()
    env.reset(np.array([0.0, -0.235, OBJECT_START_Z]))
    controls_before = env.data.ctrl.copy()

    _, _, status = env.step_ee_delta_action(
        np.array([np.nan, 0.0, 0.0]),
        np.zeros(3),
        gripper_open=1.0,
    )

    assert status.controller_failed
    assert status.infeasible
    assert status.failure_reason == "non_finite_cartesian_target"
    assert np.all(np.isfinite(env.data.ctrl))
    assert np.allclose(env.data.ctrl[env.controller.arm_actuator_ids], controls_before[env.controller.arm_actuator_ids])


def test_ee_delta_target_lag_is_bounded_against_actual_pose():
    env = PickupTaskEvaluator()
    env.reset(np.array([0.0, -0.235, OBJECT_START_Z]))

    for _ in range(10):
        env.step_ee_delta_action(np.array([0.018, 0.0, 0.0]), np.zeros(3), gripper_open=1.0)
        telemetry = env.controller.last_telemetry
        lag = np.linalg.norm(telemetry.target_pos - telemetry.actual_pos)
        assert lag <= env.controller.limits.max_target_lag_xyz + 1e-9


def test_position_only_controller_has_deterministic_posture_bias():
    sim = ArmSim()
    q_start = sim.data.qpos[sim.controller.arm_qpos_ids].copy()
    posture_target = q_start + np.array([0.12, -0.08, 0.06, -0.04, 0.03])
    sim.controller.reset_target(sim.data, posture_target=posture_target)
    ee_start = sim.ee_position.copy()

    status = sim.controller.move_toward(sim.data, ee_start)

    assert status.posture_error > 0.0
    assert np.linalg.norm(sim.data.ctrl[sim.controller.arm_actuator_ids] - q_start) > 1e-6
    assert status.joint_step_norm <= sim.controller.limits.max_joint_step + 1e-9
