from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class TrajectoryState:
    """Minimal state needed to label one controller trajectory transition."""

    joint_positions: np.ndarray
    ee_position: np.ndarray
    ee_quat_wxyz: np.ndarray
    gripper_open: float

    @classmethod
    def from_observation(cls, observation: dict) -> "TrajectoryState":
        return cls(
            joint_positions=np.asarray(observation["joint_positions"], dtype=float),
            ee_position=np.asarray(observation["ee_position"], dtype=float),
            ee_quat_wxyz=np.asarray(observation["ee_quat_wxyz"], dtype=float),
            gripper_open=float(observation["gripper_open"]),
        )


@dataclass(frozen=True)
class ActionLabel:
    name: str
    values: np.ndarray

    def to_dict(self) -> dict:
        return {"name": self.name, "values": np.round(self.values, 9).tolist()}


class ActionSpaceAdapter(Protocol):
    name: str
    size: int

    def label_transition(
        self,
        before: TrajectoryState,
        after: TrajectoryState,
        gripper_command: float,
    ) -> ActionLabel:
        """Encode the same observed transition in this action space."""


class JointDeltaActionAdapter:
    """Joint-delta baseline label: arm joint delta plus gripper command."""

    name = "joint_delta"

    def __init__(self, joint_count: int = 5) -> None:
        self.joint_count = joint_count
        self.size = joint_count + 1

    def label_transition(
        self,
        before: TrajectoryState,
        after: TrajectoryState,
        gripper_command: float,
    ) -> ActionLabel:
        before_joints = _require_shape(before.joint_positions, self.joint_count, "before joints")
        after_joints = _require_shape(after.joint_positions, self.joint_count, "after joints")
        values = np.concatenate((after_joints - before_joints, [float(gripper_command)]))
        return ActionLabel(self.name, values)


class EndEffectorDeltaActionAdapter:
    """End-effector delta label: xyz delta, local rotvec delta, gripper command."""

    name = "ee_delta"
    size = 7

    def label_transition(
        self,
        before: TrajectoryState,
        after: TrajectoryState,
        gripper_command: float,
    ) -> ActionLabel:
        delta_xyz = after.ee_position - before.ee_position
        delta_rotvec = _local_rotvec_delta(before.ee_quat_wxyz, after.ee_quat_wxyz)
        values = np.concatenate((delta_xyz, delta_rotvec, [float(gripper_command)]))
        return ActionLabel(self.name, values)


def label_transition_all(
    before: TrajectoryState,
    after: TrajectoryState,
    gripper_command: float,
    adapters: tuple[ActionSpaceAdapter, ...] | None = None,
) -> dict[str, list[float]]:
    """Return aligned labels for the same transition across action spaces."""

    adapters = adapters or (
        JointDeltaActionAdapter(len(before.joint_positions)),
        EndEffectorDeltaActionAdapter(),
    )
    labels = {}
    for adapter in adapters:
        label = adapter.label_transition(before, after, gripper_command)
        labels[label.name] = np.round(label.values, 9).tolist()
    return labels


def _require_shape(values: np.ndarray, expected: int, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.shape != (expected,):
        raise ValueError(f"{name} must have shape ({expected},), got {values.shape}")
    return values


def _local_rotvec_delta(before_wxyz: np.ndarray, after_wxyz: np.ndarray) -> np.ndarray:
    before = _rotation_from_wxyz(before_wxyz)
    after = _rotation_from_wxyz(after_wxyz)
    return (before.inv() * after).as_rotvec()


def _rotation_from_wxyz(quat_wxyz: np.ndarray) -> Rotation:
    quat_wxyz = np.asarray(quat_wxyz, dtype=float)
    return Rotation.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
