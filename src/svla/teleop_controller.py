"""Cartesian teleoperation target controller.

This module sits between *raw human input* (keyboard, mouse, gamepad) and the
existing `CartesianIKController`. Its job is to maintain the integrated teleop
target pose and apply the user's motion semantics correctly.

Coordinate frames
-----------------
Inputs arrive in the **gripper-local** frame:

    local +X  = forward  (W / left-stick up)
    local -X  = backward (S / left-stick down)
    local +Y  = left     (A / left-stick left)
    local -Y  = right    (D / left-stick right)
    local +Z  = up       (Q / right-stick up or shoulder buttons)
    local -Z  = down     (E / right-stick down)

The local linear command is a per-frame delta in meters. It is rotated into
world coordinates using the *current target orientation* (not the measured EE
orientation). That way the operator steers a coherent tool frame even when the
physical arm is slightly behind the target due to IK lag.

    world_delta = R(target_quat) @ local_linear_delta

Orientation inputs are also expressed in the gripper-local frame: a small
rotation vector (axis-angle) applied intrinsically on the right:

    R_target_new = R_target_old @ R_local_delta

Workspace clipping
------------------
After converting to world coordinates, translation is passed through
`null_blocked_world_delta` so individual world axes are nulled when the target
would leave the reachable AABB centered at the null-joint FK pose. See
`teleop_workspace.py` for rationale.

Gripper
-------
`toggle_gripper` flips between open and closed fractions. This is intentionally
a discrete toggle (Space / gamepad face button) rather than continuous analog
control, matching the user's spec.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

from svla.teleop_workspace import WorkspaceBounds, null_blocked_world_delta


@dataclass
class TeleopRates:
    linear: float = 0.012
    rotational: float = 0.04
    mouse_rot_scale: float = 0.004


@dataclass
class TeleopIntent:
    """One frame of operator intent before frame conversion and clipping."""

    # Gripper-local linear delta components in meters (see module docstring).
    local_linear: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # Gripper-local incremental rotation as rotvec (radians).
    local_rotvec: np.ndarray = field(default_factory=lambda: np.zeros(3))

    toggle_gripper: bool = False
    reset: bool = False
    pause_toggle: bool = False
    show_help: bool = False
    random_target: bool = False
    preset_target: str | None = None

    def merge(self, other: TeleopIntent) -> None:
        self.local_linear += other.local_linear
        self.local_rotvec += other.local_rotvec
        self.toggle_gripper |= other.toggle_gripper
        self.reset |= other.reset
        self.pause_toggle |= other.pause_toggle
        self.show_help |= other.show_help
        self.random_target |= other.random_target
        if other.preset_target is not None:
            self.preset_target = other.preset_target


@dataclass
class TeleopTargetState:
    position: np.ndarray
    quat_wxyz: np.ndarray
    gripper_open: float = 0.85
    controller_enabled: bool = True


@dataclass(frozen=True)
class TeleopStepResult:
    applied_world_delta: np.ndarray
    blocked_axes: tuple[bool, bool, bool]
    target_position: np.ndarray
    target_quat_wxyz: np.ndarray
    gripper_open: float


class TeleopTargetController:
    """Integrate gripper-local teleop intents into a clipped Cartesian target."""

    GRIPPER_OPEN = 0.92
    GRIPPER_CLOSED = 0.08

    def __init__(
        self,
        initial_position: np.ndarray,
        initial_quat_wxyz: np.ndarray,
        workspace: WorkspaceBounds,
        rates: TeleopRates | None = None,
        gripper_open: float = 0.85,
    ) -> None:
        self.workspace = workspace
        self.rates = rates or TeleopRates()
        self.state = TeleopTargetState(
            position=np.asarray(initial_position, dtype=float).copy(),
            quat_wxyz=np.asarray(initial_quat_wxyz, dtype=float).copy(),
            gripper_open=gripper_open,
        )

    def reset_to(self, position: np.ndarray, quat_wxyz: np.ndarray, gripper_open: float) -> None:
        self.state.position = np.asarray(position, dtype=float).copy()
        self.state.quat_wxyz = np.asarray(quat_wxyz, dtype=float).copy()
        self.state.gripper_open = gripper_open

    def apply_intent(self, intent: TeleopIntent) -> TeleopStepResult:
        if intent.toggle_gripper:
            self.state.gripper_open = (
                self.GRIPPER_CLOSED
                if self.state.gripper_open > 0.5
                else self.GRIPPER_OPEN
            )
        if intent.pause_toggle:
            self.state.controller_enabled = not self.state.controller_enabled

        world_delta = self._local_linear_to_world(intent.local_linear)
        clip = null_blocked_world_delta(self.state.position, world_delta, self.workspace)
        self.state.position = self.state.position + clip.world_delta
        self.state.quat_wxyz = self._integrate_local_rotation(
            self.state.quat_wxyz, intent.local_rotvec
        )

        return TeleopStepResult(
            applied_world_delta=clip.world_delta,
            blocked_axes=clip.blocked_axes,
            target_position=self.state.position.copy(),
            target_quat_wxyz=self.state.quat_wxyz.copy(),
            gripper_open=self.state.gripper_open,
        )

    def _local_linear_to_world(self, local_linear: np.ndarray) -> np.ndarray:
        local_linear = np.asarray(local_linear, dtype=float)
        rotation = _rotation_from_wxyz(self.state.quat_wxyz)
        return rotation.apply(local_linear)

    def _integrate_local_rotation(self, quat_wxyz: np.ndarray, local_rotvec: np.ndarray) -> np.ndarray:
        if np.linalg.norm(local_rotvec) == 0.0:
            return quat_wxyz
        current = _rotation_from_wxyz(quat_wxyz)
        delta = Rotation.from_rotvec(local_rotvec)
        # Intrinsic local-frame rotation: apply delta about the current tool axes.
        updated = current * delta
        return _wxyz_from_rotation(updated)


def _rotation_from_wxyz(quat_wxyz: np.ndarray) -> Rotation:
    quat_wxyz = np.asarray(quat_wxyz, dtype=float)
    return Rotation.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])


def _wxyz_from_rotation(rotation: Rotation) -> np.ndarray:
    quat_xyzw = rotation.as_quat()
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
