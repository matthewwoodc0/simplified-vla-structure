import numpy as np
from scipy.spatial.transform import Rotation

from svla.teleop_controller import TeleopIntent, TeleopTargetController
from svla.teleop_workspace import WorkspaceBounds, null_blocked_world_delta


def _controller_at_origin() -> TeleopTargetController:
    workspace = WorkspaceBounds(center=np.zeros(3), half_extents=np.array([0.2, 0.2, 0.2]))
    return TeleopTargetController(
        initial_position=np.zeros(3),
        initial_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        workspace=workspace,
    )


def test_forward_in_tool_frame_moves_diagonally_in_world_when_tilted():
    teleop = _controller_at_origin()
    # Pitch gripper 45 deg down: local +X should have +X and -Z in world.
    teleop.state.quat_wxyz = np.array([0.9238795, 0.0, 0.3826834, 0.0])
    step = teleop.apply_intent(TeleopIntent(local_linear=np.array([0.01, 0.0, 0.0])))
    assert step.applied_world_delta[0] > 0.0
    assert step.applied_world_delta[2] < 0.0
    assert np.isclose(
        np.linalg.norm(step.applied_world_delta),
        0.01,
        atol=1e-6,
    )


def test_workspace_nulls_only_blocked_axis():
    bounds = WorkspaceBounds(center=np.zeros(3), half_extents=np.array([0.1, 0.1, 0.1]))
    clip = null_blocked_world_delta(
        current_target=np.array([0.095, 0.0, 0.0]),
        world_delta=np.array([0.02, 0.01, 0.0]),
        bounds=bounds,
    )
    assert clip.blocked_axes[0] is True
    assert clip.world_delta[0] == 0.0
    assert clip.world_delta[1] == 0.01


def test_space_toggles_gripper():
    teleop = _controller_at_origin()
    teleop.state.gripper_open = 0.9
    step = teleop.apply_intent(TeleopIntent(toggle_gripper=True))
    assert step.gripper_open < 0.5
    step = teleop.apply_intent(TeleopIntent(toggle_gripper=True))
    assert step.gripper_open > 0.5


def test_local_rotation_integrates_in_tool_frame():
    teleop = _controller_at_origin()
    current = Rotation.from_euler("z", 90, degrees=True)
    teleop.state.quat_wxyz = _wxyz_from_rotation(current)

    local_roll = np.array([0.2, 0.0, 0.0])
    step = teleop.apply_intent(TeleopIntent(local_rotvec=local_roll))

    actual = _rotation_from_wxyz(step.target_quat_wxyz)
    expected = current * Rotation.from_rotvec(local_roll)
    world_axis_result = Rotation.from_rotvec(local_roll) * current
    assert np.allclose(actual.as_matrix(), expected.as_matrix(), atol=1e-9)
    assert not np.allclose(actual.as_matrix(), world_axis_result.as_matrix(), atol=1e-9)


def _rotation_from_wxyz(quat_wxyz):
    return Rotation.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])


def _wxyz_from_rotation(rotation):
    quat_xyzw = rotation.as_quat()
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
