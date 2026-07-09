from __future__ import annotations

import numpy as np
import pytest

from svla.action_spaces import TrajectoryState
from svla.core.action_space import (
    ACTION_REPRESENTATIONS,
    COMPARISON_ACTION_SPACES,
    get_action_representation,
)


def _state(offset: float) -> TrajectoryState:
    return TrajectoryState(
        joint_positions=np.full(5, offset),
        ee_position=np.array([offset, 0.0, 0.1]),
        ee_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        gripper_open=1.0,
    )


@pytest.mark.parametrize("name", tuple(ACTION_REPRESENTATIONS))
def test_action_representation_encode_decode_round_trip(name: str):
    representation = get_action_representation(name)
    label = representation.encoder.label_transition(_state(0.0), _state(0.01), 0.0)
    decoded = representation.decode(label.values)

    assert label.name == name
    assert decoded.shape == (representation.size,)
    assert np.array_equal(decoded, label.values)


def test_action_representation_scales_only_arm_dimensions():
    representation = get_action_representation("ee_tool_delta")
    action = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 0.25])
    scaled = representation.scale_arm(action, 0.5)

    assert np.allclose(scaled[:5], action[:5] * 0.5)
    assert scaled[5] == action[5]
    assert np.array_equal(action, np.array([1.0, 2.0, 3.0, 4.0, 5.0, 0.25]))


def test_action_representation_rejects_wrong_or_nonfinite_vectors():
    representation = get_action_representation("joint_delta")
    with pytest.raises(ValueError, match="shape"):
        representation.decode(np.zeros(5))
    with pytest.raises(ValueError, match="non-finite"):
        representation.decode(np.array([0.0, 0.0, 0.0, 0.0, np.nan, 1.0]))


def test_comparison_action_spaces_are_registered_and_equal_size():
    assert COMPARISON_ACTION_SPACES == ("joint_delta", "ee_tool_delta")
    assert {ACTION_REPRESENTATIONS[name].size for name in COMPARISON_ACTION_SPACES} == {6}
