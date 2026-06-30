"""Reachable workspace model for Cartesian teleoperation.

Design reference (read this before changing clipping logic)
-----------------------------------------------------------

Teleoperation does not command joint angles directly. The human (or later, a
policy) commands small changes to a *Cartesian target* that the IK controller
tracks. That target lives in world coordinates, but human inputs are expressed
in the *gripper-local* frame so that "forward" always means "where the gripper
is pointing", not "+X in the world".

The workspace model answers one question: *given the current target and a
proposed world-frame delta, which components of that delta should be applied?*

We use a conservative axis-aligned bounding box (AABB) centered on the
end-effector pose when all arm joints are at their **null configuration**
(q = 0 for every arm joint). The user described this as "all motors at null =
0,0,0" — that is the semantic origin of the reachable target cloud, not the
current physical pose of the arm.

Why an AABB instead of exact reachable-set geometry?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Exact reachable workspaces for 6-DOF arms are messy, configuration-dependent,
and expensive to compute every frame. AABB clipping is:

- predictable for operators ("it stops at an invisible box wall"),
- cheap,
- easy to test,
- good enough for bring-up before we add proper IK-feasibility checks.

Per-axis nulling (not projection)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
When W would push the target past the forward bound, we **zero only the world
components of the delta that would violate the bound**, not the entire delta
vector. That means if the gripper points down-right and you press W, the allowed
motion is still the world projection of local forward — but if the X component
of that world delta would cross the X-max plane, the X component is nulled while
Y/Z components may still apply. This matches the user's request to halt the
offending input axis instead of freezing all motion.

Future work
~~~~~~~~~~~
- Replace/augment AABB with Jacobian-based step feasibility.
- Shape workspace from sampled FK over joint limits.
- Separate orientation workspace limits.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from svla.controller import CartesianIKController


# SO-101 null arm joints: all rotation joints at 0 rad.
NULL_ARM_JOINT_POSITIONS = np.zeros(5)


@dataclass(frozen=True)
class WorkspaceBounds:
    """Axis-aligned reachable target region in world coordinates."""

    center: np.ndarray
    half_extents: np.ndarray

    @property
    def min_corner(self) -> np.ndarray:
        return self.center - self.half_extents

    @property
    def max_corner(self) -> np.ndarray:
        return self.center + self.half_extents

    def contains(self, point: np.ndarray) -> bool:
        point = np.asarray(point, dtype=float)
        return bool(np.all(point >= self.min_corner) and np.all(point <= self.max_corner))


@dataclass(frozen=True)
class WorkspaceClipResult:
    world_delta: np.ndarray
    blocked_axes: tuple[bool, bool, bool]


def default_half_extents() -> np.ndarray:
    """Conservative SO-101 bring-up box relative to null-configuration EE center."""
    return np.array([0.20, 0.20, 0.18], dtype=float)


def fk_ee_position(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    controller: CartesianIKController,
    arm_joint_positions: np.ndarray,
) -> np.ndarray:
    """Forward kinematics EE position for a candidate arm joint vector."""
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = data.qpos
    scratch.qvel[:] = 0.0
    scratch.ctrl[:] = data.ctrl
    scratch.qpos[controller.arm_qpos_ids] = np.asarray(arm_joint_positions, dtype=float)
    mujoco.mj_forward(model, scratch)
    return scratch.site_xpos[controller.ee_site_id].copy()


def build_workspace_bounds(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    controller: CartesianIKController,
    half_extents: np.ndarray | None = None,
) -> WorkspaceBounds:
    """Build workspace from null joint configuration FK center."""
    center = fk_ee_position(model, data, controller, NULL_ARM_JOINT_POSITIONS)
    extents = (
        default_half_extents()
        if half_extents is None
        else np.asarray(half_extents, dtype=float)
    )
    return WorkspaceBounds(center=center, half_extents=extents)


def null_blocked_world_delta(
    current_target: np.ndarray,
    world_delta: np.ndarray,
    bounds: WorkspaceBounds,
) -> WorkspaceClipResult:
    """Apply per-axis nulling when a delta would exit the workspace AABB."""
    current_target = np.asarray(current_target, dtype=float)
    world_delta = np.asarray(world_delta, dtype=float)
    allowed = world_delta.copy()
    blocked = [False, False, False]

    for axis in range(3):
        proposed = current_target[axis] + world_delta[axis]
        if world_delta[axis] > 0.0 and proposed > bounds.max_corner[axis]:
            allowed[axis] = 0.0
            blocked[axis] = True
        elif world_delta[axis] < 0.0 and proposed < bounds.min_corner[axis]:
            allowed[axis] = 0.0
            blocked[axis] = True

    return WorkspaceClipResult(world_delta=allowed, blocked_axes=tuple(blocked))
