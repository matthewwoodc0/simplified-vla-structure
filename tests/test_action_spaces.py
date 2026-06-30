import numpy as np
from scipy.spatial.transform import Rotation

from svla.action_spaces import (
    EndEffectorDeltaActionAdapter,
    JointDeltaActionAdapter,
    TrajectoryState,
    label_transition_all,
)


def test_joint_delta_adapter_labels_transition_with_gripper_command():
    before = TrajectoryState(
        joint_positions=np.array([0.0, 0.1, 0.2, 0.3, 0.4]),
        ee_position=np.zeros(3),
        ee_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        gripper_open=1.0,
    )
    after = TrajectoryState(
        joint_positions=np.array([0.1, 0.0, 0.25, 0.35, 0.2]),
        ee_position=np.zeros(3),
        ee_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        gripper_open=0.0,
    )

    label = JointDeltaActionAdapter().label_transition(before, after, gripper_command=0.25)

    assert label.name == "joint_delta"
    assert np.allclose(label.values, [0.1, -0.1, 0.05, 0.05, -0.2, 0.25])


def test_ee_delta_adapter_uses_local_rotvec_delta():
    before_rotation = Rotation.from_euler("z", 90, degrees=True)
    after_rotation = before_rotation * Rotation.from_rotvec([0.1, 0.0, 0.0])
    before = TrajectoryState(
        joint_positions=np.zeros(5),
        ee_position=np.array([0.1, -0.2, 0.3]),
        ee_quat_wxyz=_wxyz_from_rotation(before_rotation),
        gripper_open=1.0,
    )
    after = TrajectoryState(
        joint_positions=np.zeros(5),
        ee_position=np.array([0.11, -0.22, 0.33]),
        ee_quat_wxyz=_wxyz_from_rotation(after_rotation),
        gripper_open=0.5,
    )

    label = EndEffectorDeltaActionAdapter().label_transition(before, after, gripper_command=0.5)

    assert label.name == "ee_delta"
    assert np.allclose(label.values[:3], [0.01, -0.02, 0.03])
    assert np.allclose(label.values[3:6], [0.1, 0.0, 0.0], atol=1e-9)
    assert label.values[6] == 0.5


def test_same_transition_exports_both_action_space_labels():
    before = TrajectoryState(
        joint_positions=np.zeros(5),
        ee_position=np.zeros(3),
        ee_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        gripper_open=1.0,
    )
    after = TrajectoryState(
        joint_positions=np.ones(5) * 0.01,
        ee_position=np.array([0.01, 0.0, 0.0]),
        ee_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        gripper_open=0.0,
    )

    labels = label_transition_all(before, after, gripper_command=0.0)

    assert set(labels) == {"joint_delta", "ee_delta"}
    assert len(labels["joint_delta"]) == 6
    assert len(labels["ee_delta"]) == 7


def _wxyz_from_rotation(rotation):
    quat_xyzw = rotation.as_quat()
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
