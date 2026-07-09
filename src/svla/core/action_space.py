from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from svla.action_spaces import ActionSpaceAdapter, get_action_label_adapter


TELEMETRY_FIELDS = (
    "saturated",
    "clipped_translation",
    "clipped_rotation",
    "clipped_joints",
    "joint_limit_clipped",
    "joint_step_clipped",
    "joint_accel_clipped",
    "infeasible",
    "controller_failed",
    "failure_reason",
)


@dataclass(frozen=True)
class ActionRepresentation:
    """Canonical encode/decode/execute contract for one robot action representation.

    The learner only needs ``name`` and ``size``. Dataset code uses ``encoder`` to
    label an observed transition, and closed-loop evaluation uses ``execute`` to
    decode the policy vector into the corresponding task API. Adding a new physical
    representation therefore does not require editing the training loop.
    """

    name: str
    size: int
    arm_dimensions: int
    encoder: ActionSpaceAdapter

    def decode(self, values: np.ndarray) -> np.ndarray:
        action = np.asarray(values, dtype=float)
        if action.shape != (self.size,):
            raise ValueError(
                f"{self.name} action must have shape ({self.size},), got {action.shape}"
            )
        if not np.isfinite(action).all():
            raise ValueError(f"{self.name} action contains non-finite values")
        return action.copy()

    def scale_arm(self, values: np.ndarray, gain: float) -> np.ndarray:
        action = self.decode(values)
        action[: self.arm_dimensions] *= float(gain)
        return action

    def execute(self, env: Any, values: np.ndarray) -> tuple[Any, Any, Any]:
        action = self.decode(values)
        if self.name == "joint_delta":
            return env.step_joint_delta_action(action[:5], action[5])
        if self.name == "ee_delta":
            return env.step_ee_delta_action(action[:3], action[3:6], action[6])
        if self.name == "ee_tool_delta":
            return env.step_ee_tool_delta_action(action[:3], action[3:5], action[5])
        raise ValueError(f"no executor registered for action space: {self.name}")

    def telemetry(self, status: Any) -> dict[str, Any]:
        """Normalize dict- and dataclass-style controller status objects."""

        result: dict[str, Any] = {}
        for field in TELEMETRY_FIELDS:
            if isinstance(status, Mapping):
                value = status.get(field)
            else:
                value = getattr(status, field, None)
            if field == "failure_reason":
                result[field] = value
            else:
                result[field] = bool(value) if value is not None else False
        return result


def _representation(name: str, size: int, arm_dimensions: int) -> ActionRepresentation:
    return ActionRepresentation(
        name=name,
        size=size,
        arm_dimensions=arm_dimensions,
        encoder=get_action_label_adapter(name),
    )


ACTION_REPRESENTATIONS = {
    "joint_delta": _representation("joint_delta", 6, 5),
    "ee_delta": _representation("ee_delta", 7, 6),
    "ee_tool_delta": _representation("ee_tool_delta", 6, 5),
}

COMPARISON_ACTION_SPACES = ("joint_delta", "ee_tool_delta")


def get_action_representation(name: str) -> ActionRepresentation:
    try:
        return ACTION_REPRESENTATIONS[name]
    except KeyError as exc:
        raise ValueError(f"unknown action space: {name}") from exc
