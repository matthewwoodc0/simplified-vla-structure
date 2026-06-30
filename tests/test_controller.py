import numpy as np
from scipy.spatial.transform import Rotation

from svla.controller import CartesianCommand
from svla.controller import _compose_wxyz
from svla.sim import ArmSim


def test_sim_loads_and_reports_ee_pose():
    sim = ArmSim()
    assert sim.ee_position.shape == (3,)
    assert np.isfinite(sim.ee_position).all()


def test_controller_reaches_nearby_target():
    sim = ArmSim()
    start = sim.ee_position.copy()
    target = start + np.array([-0.04, 0.05, 0.03])
    error = sim.move_to(target, max_steps=500)
    assert error < 0.02
    telemetry = sim.controller.last_telemetry
    assert telemetry is not None
    assert telemetry.target_pos.shape == (3,)
    assert telemetry.actual_pos.shape == (3,)
    assert telemetry.joint_targets.shape == sim.data.qpos[sim.controller.arm_qpos_ids].shape
    assert np.isfinite(telemetry.position_error)


def test_delta_command_is_clipped_and_keeps_state_finite():
    sim = ArmSim()
    status = sim.step(CartesianCommand(np.array([1.0, 1.0, 1.0]), np.zeros(3), 0.5))
    assert status is not None
    assert status.clipped_translation
    assert np.isfinite(sim.data.qpos).all()
    assert np.isfinite(sim.data.ctrl).all()


def test_cartesian_command_rotvec_is_end_effector_local():
    current = Rotation.from_euler("z", 90, degrees=True)
    current_wxyz = _wxyz_from_rotation(current)
    local_roll = np.array([0.2, 0.0, 0.0])

    actual = _rotation_from_wxyz(_compose_wxyz(current_wxyz, local_roll))
    expected = current * Rotation.from_rotvec(local_roll)
    world_axis_result = Rotation.from_rotvec(local_roll) * current

    assert np.allclose(actual.as_matrix(), expected.as_matrix(), atol=1e-9)
    assert not np.allclose(actual.as_matrix(), world_axis_result.as_matrix(), atol=1e-9)


def _rotation_from_wxyz(quat_wxyz):
    return Rotation.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])


def _wxyz_from_rotation(rotation):
    quat_xyzw = rotation.as_quat()
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
