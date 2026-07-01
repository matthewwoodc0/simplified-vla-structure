from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


ARM_JOINT_NAMES = (
    "Rotation",
    "Pitch",
    "Elbow",
    "Wrist_Pitch",
    "Wrist_Roll",
)

GRIPPER_JOINT_NAMES = ("Jaw",)


@dataclass(frozen=True)
class CartesianCommand:
    """Policy-facing command: end-effector delta plus gripper opening."""

    delta_xyz: np.ndarray
    delta_rotvec: np.ndarray
    gripper_open: float

    @classmethod
    def zero(cls, gripper_open: float = 1.0) -> "CartesianCommand":
        return cls(np.zeros(3), np.zeros(3), gripper_open)


@dataclass(frozen=True)
class ControllerLimits:
    max_step_xyz: float = 0.025
    max_step_rot: float = 0.12
    max_target_lag_xyz: float = 0.025
    max_joint_step: float = 0.035
    max_joint_accel_step: float = 0.018
    posture_gain: float = 0.02
    orientation_mode: str = "full"
    tool_axis_index: int = 0
    position_tolerance: float = 0.008
    rotation_tolerance: float = 0.04
    min_gripper_open_fraction: float = 0.0


@dataclass(frozen=True)
class IKStatus:
    position_error: float
    rotation_error: float
    clipped_translation: bool
    clipped_rotation: bool
    clipped_joints: bool
    joint_limit_clipped: bool = False
    joint_step_clipped: bool = False
    joint_accel_clipped: bool = False
    posture_error: float = 0.0
    joint_step_norm: float = 0.0
    saturated: bool = False
    infeasible: bool = False
    controller_failed: bool = False
    failure_reason: str | None = None

    @property
    def converged(self) -> bool:
        return (
            not self.controller_failed
            and self.position_error <= 0.008
            and self.rotation_error <= 0.04
        )


@dataclass(frozen=True)
class ControllerTelemetry:
    target_pos: np.ndarray
    target_quat_wxyz: np.ndarray
    actual_pos: np.ndarray
    actual_quat_wxyz: np.ndarray
    position_error: float
    rotation_error: float
    clipped_translation: bool
    clipped_rotation: bool
    clipped_joints: bool
    joint_limit_clipped: bool
    joint_step_clipped: bool
    joint_accel_clipped: bool
    posture_error: float
    joint_step_norm: float
    saturated: bool
    infeasible: bool
    controller_failed: bool
    failure_reason: str | None
    feasible_delta_xyz: np.ndarray
    feasible_delta_rotvec: np.ndarray
    joint_targets: np.ndarray
    joint_positions: np.ndarray
    joint_target_error: np.ndarray
    integrated_target_pos: np.ndarray
    integrated_target_quat_wxyz: np.ndarray


class CartesianIKController:
    """Damped-least-squares Cartesian controller for the MuJoCo SO-101 arm.

    This is the low-level tracking layer beneath teleoperation. Teleop maintains
    a *target* pose; this controller converts pose error into joint position
    actuator commands each physics step.

    Important separation of concerns
    --------------------------------
    - `teleop_inputs.py` reads human hardware and produces gripper-local intents.
    - `teleop_controller.py` integrates those intents into a world-frame target
      and applies workspace clipping.
    - *This file* only answers: "given a target pose, what joint commands move
      the EE toward it without violating joint limits?"

    The policy-facing action API remains `CartesianCommand` (delta + gripper),
    which is the shape we expect an eventual learned policy to emit.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        ee_site_name: str = "ee_site",
        damping: float = 0.08,
        limits: ControllerLimits | None = None,
    ) -> None:
        self.model = model
        self.ee_site_id = _require_id(model, mujoco.mjtObj.mjOBJ_SITE, ee_site_name)
        self.arm_joint_ids = [
            _require_id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in ARM_JOINT_NAMES
        ]
        self.gripper_joint_ids = [
            _require_id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in GRIPPER_JOINT_NAMES
        ]
        self.arm_qpos_ids = np.array([model.jnt_qposadr[joint_id] for joint_id in self.arm_joint_ids])
        self.arm_dof_ids = np.array([model.jnt_dofadr[joint_id] for joint_id in self.arm_joint_ids])
        self.gripper_qpos_ids = np.array(
            [model.jnt_qposadr[joint_id] for joint_id in self.gripper_joint_ids]
        )
        self.arm_actuator_ids = np.array(
            [_require_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in ARM_JOINT_NAMES]
        )
        self.gripper_actuator_ids = np.array(
            [_require_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in GRIPPER_JOINT_NAMES]
        )
        self.joint_ranges = model.jnt_range[self.arm_joint_ids].copy()
        self.gripper_ranges = model.jnt_range[self.gripper_joint_ids].copy()
        self.damping = damping
        self.limits = limits or ControllerLimits()
        self.last_telemetry: ControllerTelemetry | None = None
        self.target_pos: np.ndarray | None = None
        self.target_quat_wxyz: np.ndarray | None = None
        self.posture_target: np.ndarray | None = None
        self._last_joint_delta = np.zeros(len(self.arm_qpos_ids), dtype=float)

    def reset_target(
        self,
        data: mujoco.MjData,
        target_pos: np.ndarray | None = None,
        target_quat_wxyz: np.ndarray | None = None,
        posture_target: np.ndarray | None = None,
    ) -> None:
        """Reset the persistent Cartesian intention state used by delta actions."""

        self.target_pos = (
            np.asarray(target_pos, dtype=float).copy()
            if target_pos is not None
            else data.site_xpos[self.ee_site_id].copy()
        )
        self.target_quat_wxyz = (
            np.asarray(target_quat_wxyz, dtype=float).copy()
            if target_quat_wxyz is not None
            else _site_quat_wxyz(data, self.ee_site_id)
        )
        self.posture_target = (
            np.asarray(posture_target, dtype=float).copy()
            if posture_target is not None
            else data.qpos[self.arm_qpos_ids].copy()
        )
        self._last_joint_delta = np.zeros(len(self.arm_qpos_ids), dtype=float)

    def apply_delta(self, data: mujoco.MjData, command: CartesianCommand) -> IKStatus:
        if self.target_pos is None or self.target_quat_wxyz is None:
            self.reset_target(data)
        delta_xyz, clipped_translation = _clip_norm(command.delta_xyz, self.limits.max_step_xyz)
        delta_rotvec, clipped_rotation = _clip_norm(command.delta_rotvec, self.limits.max_step_rot)

        current_pos = data.site_xpos[self.ee_site_id].copy()
        target_pos = self.target_pos + delta_xyz
        target_lag, clipped_target_lag = _clip_norm(
            target_pos - current_pos,
            self.limits.max_target_lag_xyz,
        )
        target_pos = current_pos + target_lag
        target_quat = _compose_wxyz(self.target_quat_wxyz, delta_rotvec)
        self.target_pos = target_pos.copy()
        self.target_quat_wxyz = target_quat.copy()
        status = self.move_toward(data, target_pos, target_quat)
        self.set_gripper(data, command.gripper_open)
        return IKStatus(
            position_error=status.position_error,
            rotation_error=status.rotation_error,
            clipped_translation=clipped_translation or clipped_target_lag or status.clipped_translation,
            clipped_rotation=clipped_rotation or status.clipped_rotation,
            clipped_joints=status.clipped_joints,
            joint_limit_clipped=status.joint_limit_clipped,
            joint_step_clipped=status.joint_step_clipped,
            joint_accel_clipped=status.joint_accel_clipped,
            posture_error=status.posture_error,
            joint_step_norm=status.joint_step_norm,
            saturated=(
                clipped_translation
                or clipped_target_lag
                or clipped_rotation
                or status.saturated
            ),
            infeasible=status.infeasible,
            controller_failed=status.controller_failed,
            failure_reason=status.failure_reason,
        )

    def move_toward(
        self,
        data: mujoco.MjData,
        target_pos: np.ndarray,
        target_quat_wxyz: np.ndarray | None = None,
    ) -> IKStatus:
        """Track a Cartesian target using one damped-least-squares IK step.

        Position-only mode is used by reach tests; full pose mode is used by the
        teleop GUI once operators command orientation as well as translation.
        """
        target_pos = np.asarray(target_pos, dtype=float)
        position_only = target_quat_wxyz is None
        if position_only:
            target_quat_wxyz = _site_quat_wxyz(data, self.ee_site_id)
        else:
            target_quat_wxyz = np.asarray(target_quat_wxyz, dtype=float)

        if not np.isfinite(target_pos).all() or not np.isfinite(target_quat_wxyz).all():
            return self._controller_failure_status(
                data,
                target_pos,
                target_quat_wxyz,
                "non_finite_cartesian_target",
            )

        current_pos = data.site_xpos[self.ee_site_id].copy()
        current_quat = _site_quat_wxyz(data, self.ee_site_id)
        pos_err = target_pos - current_pos
        full_rot_err = _orientation_error_rotvec(current_quat, target_quat_wxyz)
        pos_err, clipped_translation = _clip_norm(pos_err, self.limits.max_step_xyz)

        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, data, jacp, jacr, self.ee_site_id)
        if position_only:
            jac = jacp[:, self.arm_dof_ids]
            task_err = pos_err
            rotation_error = 0.0
            clipped_rotation = False
            orientation_mode = "position_only"
        elif self.limits.orientation_mode == "full":
            rot_err, clipped_rotation = _clip_norm(full_rot_err, self.limits.max_step_rot)
            jac = np.vstack((jacp[:, self.arm_dof_ids], jacr[:, self.arm_dof_ids]))
            task_err = np.concatenate((pos_err, rot_err))
            rotation_error = float(np.linalg.norm(full_rot_err))
            orientation_mode = "full"
        elif self.limits.orientation_mode == "tool_axis":
            tool_axis_index = self.limits.tool_axis_index
            if tool_axis_index not in (0, 1, 2):
                return self._controller_failure_status(
                    data,
                    target_pos,
                    target_quat_wxyz,
                    "invalid_tool_axis_index",
                )
            tangent_indices = [index for index in range(3) if index != tool_axis_index]
            current_rotation = _rotation_from_wxyz(current_quat)
            target_rotation = _rotation_from_wxyz(target_quat_wxyz)
            current_matrix = current_rotation.as_matrix()
            current_axis = current_matrix[:, tool_axis_index]
            target_axis = target_rotation.as_matrix()[:, tool_axis_index]
            tangent_basis = current_matrix[:, tangent_indices].T
            axis_error_world = np.cross(current_axis, target_axis)
            axis_error = tangent_basis @ axis_error_world
            axis_error, clipped_rotation = _clip_norm(
                axis_error,
                self.limits.max_step_rot,
            )
            jac = np.vstack(
                (
                    jacp[:, self.arm_dof_ids],
                    tangent_basis @ jacr[:, self.arm_dof_ids],
                )
            )
            task_err = np.concatenate((pos_err, axis_error))
            rotation_error = float(
                np.arccos(np.clip(np.dot(current_axis, target_axis), -1.0, 1.0))
            )
            orientation_mode = "tool_axis"
        else:
            return self._controller_failure_status(
                data,
                target_pos,
                target_quat_wxyz,
                "invalid_orientation_mode",
            )

        q_current = data.qpos[self.arm_qpos_ids].copy()
        try:
            ik_map = _damped_pseudoinverse(jac, self.damping)
            dq = ik_map @ task_err
        except np.linalg.LinAlgError:
            return self._controller_failure_status(
                data,
                target_pos,
                target_quat_wxyz,
                "damped_ik_solve_failed",
            )
        if not np.isfinite(dq).all():
            return self._controller_failure_status(
                data,
                target_pos,
                target_quat_wxyz,
                "non_finite_ik_solution",
            )
        posture_error = np.zeros_like(q_current)
        posture_component = np.zeros_like(q_current)
        if self.posture_target is not None:
            posture_error = np.asarray(self.posture_target, dtype=float) - q_current
            null_projector = _damped_nullspace_projector(jac, self.damping)
            posture_component = null_projector @ (self.limits.posture_gain * posture_error)
            dq = dq + posture_component
        dq, joint_step_clipped = _clip_norm(dq, self.limits.max_joint_step)
        dq_delta = dq - self._last_joint_delta
        dq_delta, joint_accel_clipped = _clip_norm(dq_delta, self.limits.max_joint_accel_step)
        dq = self._last_joint_delta + dq_delta
        q_target = np.clip(
            q_current + dq,
            self.joint_ranges[:, 0],
            self.joint_ranges[:, 1],
        )
        joint_limit_clipped = not np.allclose(q_target, q_current + dq)
        joint_target_error = q_target - q_current
        try:
            equivalent_task_error = np.linalg.lstsq(
                ik_map,
                joint_target_error - posture_component,
                rcond=None,
            )[0]
        except np.linalg.LinAlgError:
            return self._controller_failure_status(
                data,
                target_pos,
                target_quat_wxyz,
                "cartesian_intention_reconstruction_failed",
            )
        if not np.isfinite(equivalent_task_error).all():
            return self._controller_failure_status(
                data,
                target_pos,
                target_quat_wxyz,
                "non_finite_cartesian_intention",
            )
        feasible_delta_xyz = equivalent_task_error[:3]
        if orientation_mode == "full":
            feasible_delta_rotvec = _rotation_from_wxyz(current_quat).inv().apply(
                equivalent_task_error[3:]
            )
        elif orientation_mode == "tool_axis":
            feasible_delta_rotvec = np.zeros(3)
            feasible_delta_rotvec[tangent_indices] = equivalent_task_error[3:]
        else:
            feasible_delta_rotvec = np.zeros(3)
        clipped_joints = joint_step_clipped or joint_accel_clipped or joint_limit_clipped
        saturated = clipped_translation or clipped_rotation or clipped_joints
        # "Infeasible" means the requested target is blocked at a hard joint
        # limit in this control interval. Step/acceleration limiting alone is a
        # normal temporal saturation and does not imply geometric infeasibility.
        infeasible = joint_limit_clipped and (
            np.linalg.norm(target_pos - current_pos) > self.limits.position_tolerance
            or rotation_error > self.limits.rotation_tolerance
        )
        data.ctrl[self.arm_actuator_ids] = q_target
        self._last_joint_delta = q_target - q_current

        status = IKStatus(
            position_error=float(np.linalg.norm(target_pos - current_pos)),
            rotation_error=rotation_error,
            clipped_translation=clipped_translation,
            clipped_rotation=clipped_rotation,
            clipped_joints=clipped_joints,
            joint_limit_clipped=joint_limit_clipped,
            joint_step_clipped=joint_step_clipped,
            joint_accel_clipped=joint_accel_clipped,
            posture_error=float(np.linalg.norm(posture_error)),
            joint_step_norm=float(np.linalg.norm(q_target - q_current)),
            saturated=saturated,
            infeasible=bool(infeasible),
        )
        self.last_telemetry = ControllerTelemetry(
            target_pos=target_pos.copy(),
            target_quat_wxyz=target_quat_wxyz.copy(),
            actual_pos=current_pos.copy(),
            actual_quat_wxyz=current_quat.copy(),
            position_error=status.position_error,
            rotation_error=status.rotation_error,
            clipped_translation=status.clipped_translation,
            clipped_rotation=status.clipped_rotation,
            clipped_joints=status.clipped_joints,
            joint_limit_clipped=status.joint_limit_clipped,
            joint_step_clipped=status.joint_step_clipped,
            joint_accel_clipped=status.joint_accel_clipped,
            posture_error=status.posture_error,
            joint_step_norm=status.joint_step_norm,
            saturated=status.saturated,
            infeasible=status.infeasible,
            controller_failed=status.controller_failed,
            failure_reason=status.failure_reason,
            feasible_delta_xyz=feasible_delta_xyz.copy(),
            feasible_delta_rotvec=feasible_delta_rotvec.copy(),
            joint_targets=q_target.copy(),
            joint_positions=q_current.copy(),
            joint_target_error=joint_target_error.copy(),
            integrated_target_pos=target_pos.copy(),
            integrated_target_quat_wxyz=target_quat_wxyz.copy(),
        )
        return status

    def _controller_failure_status(
        self,
        data: mujoco.MjData,
        target_pos: np.ndarray,
        target_quat_wxyz: np.ndarray,
        reason: str,
    ) -> IKStatus:
        """Report a numerical/controller failure without writing invalid controls."""

        current_pos = data.site_xpos[self.ee_site_id].copy()
        current_quat = _site_quat_wxyz(data, self.ee_site_id)
        q_current = data.qpos[self.arm_qpos_ids].copy()
        status = IKStatus(
            position_error=float("inf"),
            rotation_error=float("inf"),
            clipped_translation=False,
            clipped_rotation=False,
            clipped_joints=False,
            saturated=False,
            infeasible=True,
            controller_failed=True,
            failure_reason=reason,
        )
        self.last_telemetry = ControllerTelemetry(
            target_pos=np.asarray(target_pos, dtype=float).copy(),
            target_quat_wxyz=np.asarray(target_quat_wxyz, dtype=float).copy(),
            actual_pos=current_pos,
            actual_quat_wxyz=current_quat,
            position_error=status.position_error,
            rotation_error=status.rotation_error,
            clipped_translation=False,
            clipped_rotation=False,
            clipped_joints=False,
            joint_limit_clipped=False,
            joint_step_clipped=False,
            joint_accel_clipped=False,
            posture_error=0.0,
            joint_step_norm=0.0,
            saturated=False,
            infeasible=True,
            controller_failed=True,
            failure_reason=reason,
            feasible_delta_xyz=np.zeros(3),
            feasible_delta_rotvec=np.zeros(3),
            joint_targets=data.ctrl[self.arm_actuator_ids].copy(),
            joint_positions=q_current,
            joint_target_error=data.ctrl[self.arm_actuator_ids].copy() - q_current,
            integrated_target_pos=np.asarray(target_pos, dtype=float).copy(),
            integrated_target_quat_wxyz=np.asarray(target_quat_wxyz, dtype=float).copy(),
        )
        return status

    def set_gripper(self, data: mujoco.MjData, open_fraction: float) -> None:
        open_fraction = float(np.clip(open_fraction, 0.0, 1.0))
        open_fraction = self.limits.min_gripper_open_fraction + open_fraction * (
            1.0 - self.limits.min_gripper_open_fraction
        )
        targets = self.gripper_ranges[:, 0] + open_fraction * (
            self.gripper_ranges[:, 1] - self.gripper_ranges[:, 0]
        )
        data.ctrl[self.gripper_actuator_ids] = targets

    def gripper_open_fraction(self, data: mujoco.MjData) -> float:
        q = data.qpos[self.gripper_qpos_ids]
        lo = self.gripper_ranges[:, 0]
        hi = self.gripper_ranges[:, 1]
        span = hi - lo
        if np.all(span == 0):
            return 0.0
        return float(np.clip(np.mean((q - lo) / span), 0.0, 1.0))

    def ee_pose(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        return data.site_xpos[self.ee_site_id].copy(), _site_quat_wxyz(data, self.ee_site_id)


def _require_id(model: mujoco.MjModel, obj_type: mujoco.mjtObj, name: str) -> int:
    obj_id = mujoco.mj_name2id(model, obj_type, name)
    if obj_id < 0:
        raise ValueError(f"Missing MuJoCo object: {name}")
    return obj_id


def _clip_norm(vector: np.ndarray, max_norm: float) -> tuple[np.ndarray, bool]:
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm <= max_norm or norm == 0.0:
        return vector, False
    return vector * (max_norm / norm), True


def _damped_least_squares(jacobian: np.ndarray, error: np.ndarray, damping: float) -> np.ndarray:
    return _damped_pseudoinverse(jacobian, damping) @ error


def _damped_pseudoinverse(jacobian: np.ndarray, damping: float) -> np.ndarray:
    lhs = jacobian @ jacobian.T + (damping**2) * np.eye(jacobian.shape[0])
    return jacobian.T @ np.linalg.solve(lhs, np.eye(jacobian.shape[0]))


def _damped_nullspace_projector(jacobian: np.ndarray, damping: float) -> np.ndarray:
    pinv = _damped_pseudoinverse(jacobian, damping)
    return np.eye(jacobian.shape[1]) - pinv @ jacobian


def _orientation_error_rotvec(current_wxyz: np.ndarray, target_wxyz: np.ndarray) -> np.ndarray:
    current = _rotation_from_wxyz(current_wxyz)
    target = _rotation_from_wxyz(target_wxyz)
    return (target * current.inv()).as_rotvec()


def _compose_wxyz(current_wxyz: np.ndarray, delta_rotvec: np.ndarray) -> np.ndarray:
    current = _rotation_from_wxyz(current_wxyz)
    delta = Rotation.from_rotvec(delta_rotvec)
    # CartesianCommand deltas are end-effector-local deltas, matching teleop.
    return _wxyz_from_rotation(current * delta)


def _rotation_from_wxyz(quat: np.ndarray) -> Rotation:
    quat = np.asarray(quat, dtype=float)
    return Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])


def _wxyz_from_rotation(rotation: Rotation) -> np.ndarray:
    quat_xyzw = rotation.as_quat()
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])


def _site_quat_wxyz(data: mujoco.MjData, site_id: int) -> np.ndarray:
    site_matrix = data.site_xmat[site_id].reshape(3, 3)
    return _wxyz_from_rotation(Rotation.from_matrix(site_matrix))
