from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from svla.controller import (
    CartesianIKController,
    ControllerLimits,
    IKStatus,
    _clip_norm,
    _compose_wxyz,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PICKUP_MODEL_PATH = PROJECT_ROOT / "assets" / "pickup_scene.xml"

SUPPORT_TOP_Z = 0.056
OBJECT_HALF_HEIGHT = 0.013
OBJECT_START_Z = SUPPORT_TOP_Z + OBJECT_HALF_HEIGHT
BASE_OBJECT_HALF_SIZE = np.array([0.009, 0.007, OBJECT_HALF_HEIGHT])
BASE_OBJECT_MASS = 0.004
BASE_OBJECT_SLIDING_FRICTION = 1.8
LIFT_CLEARANCE = 0.018
RETENTION_CLEARANCE = 0.015
PICK_PLACE_LIFT_HEIGHT = 0.070
PLACEMENT_XY_TOLERANCE = 0.012
PLACEMENT_Z_TOLERANCE = 0.006
GRIPPER_RELEASED_THRESHOLD = 0.50
PLACEMENT_MARKER_BODIES = {
    "place_left": ("place_left_marker", "place_left_command_marker"),
    "place_right": ("place_right_marker", "place_right_marker"),
}
PICK_PLACE_TRANSPORT_PHASE = "transport"


def maybe_finalize_grasp_at_sample(
    env: "PickupTaskEvaluator",
    sample_index: int,
    boundary_index: int | None,
) -> None:
    if boundary_index is not None and int(sample_index) == int(boundary_index):
        env.finalize_grasp_segment()
PRECLOSE_OPEN_THRESHOLD = 0.5
PRECLOSE_DISPLACEMENT_TOLERANCE = 0.001
EARLY_CLOSE_DISTANCE = 0.015
MAX_GRIPPER_CONTACT_FORCE = 22.0
MAX_GRIPPER_IMPULSE_BEFORE_LIFT = 9.0
MAX_SUPPORTED_XY_DISPLACEMENT = 0.013
MAX_SUPPORTED_ROTATION = 0.30
GRASP_WIDTH_SHRINK_COMPENSATION_GAIN = 3.0
GRASP_WIDTH_GROWTH_COMPENSATION_GAIN = 1.0

PICKUP_CONTROLLER_LIMITS = ControllerLimits(
    max_step_xyz=0.019,
    max_step_rot=0.08,
    max_target_lag_xyz=0.019,
    max_joint_step=0.030,
    posture_gain=0.07,
    orientation_mode="tool_axis",
    tool_axis_index=2,
    position_tolerance=0.006,
    rotation_tolerance=0.08,
    min_gripper_open_fraction=0.05,
)


def pickup_physics_gate_constants() -> dict[str, float]:
    return {
        "LIFT_CLEARANCE": LIFT_CLEARANCE,
        "RETENTION_CLEARANCE": RETENTION_CLEARANCE,
        "PRECLOSE_OPEN_THRESHOLD": PRECLOSE_OPEN_THRESHOLD,
        "PRECLOSE_DISPLACEMENT_TOLERANCE": PRECLOSE_DISPLACEMENT_TOLERANCE,
        "EARLY_CLOSE_DISTANCE": EARLY_CLOSE_DISTANCE,
        "MAX_GRIPPER_CONTACT_FORCE": MAX_GRIPPER_CONTACT_FORCE,
        "MAX_GRIPPER_IMPULSE_BEFORE_LIFT": MAX_GRIPPER_IMPULSE_BEFORE_LIFT,
        "MAX_SUPPORTED_XY_DISPLACEMENT": MAX_SUPPORTED_XY_DISPLACEMENT,
        "MAX_SUPPORTED_ROTATION": MAX_SUPPORTED_ROTATION,
    }


@dataclass(frozen=True)
class GraspOrientation:
    label: str
    yaw_degrees: float

    @property
    def quat_wxyz(self) -> np.ndarray:
        yaw = np.deg2rad(self.yaw_degrees)
        local_x = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        local_y = np.array([0.0, 0.0, 1.0])
        local_z = np.cross(local_x, local_y)
        matrix = np.column_stack((local_x, local_y, local_z))
        return _wxyz_from_rotation(Rotation.from_matrix(matrix))

    @property
    def rotation(self) -> Rotation:
        return _rotation_from_wxyz(self.quat_wxyz)


@dataclass(frozen=True)
class ObjectStartPose:
    label: str
    xyz: np.ndarray


@dataclass(frozen=True)
class ApproachStrategy:
    label: str
    pregrasp_axis: str


@dataclass(frozen=True)
class PlacementTarget:
    label: str
    goal_marker_body: str
    command_marker_body: str | None = None

    @property
    def marker_body(self) -> str:
        return self.goal_marker_body


@dataclass(frozen=True)
class PickupTrialSpec:
    trial_id: int
    orientation: GraspOrientation
    object_pose: ObjectStartPose
    approach: ApproachStrategy
    repeat: int = 0


@dataclass(frozen=True)
class PickPlaceTrialSpec:
    trial_id: int
    orientation: GraspOrientation
    object_pose: ObjectStartPose
    approach: ApproachStrategy
    placement_target: PlacementTarget
    repeat: int = 0


@dataclass(frozen=True)
class PickupTrialResult:
    trial_id: int
    orientation: str
    object_pose: str
    approach: str
    repeat: int
    success: bool
    object_start_pose: list[float]
    commanded_grasp_pose: list[float]
    gripper_orientation_wxyz: list[float]
    final_ee_position_error: float
    final_ee_rotation_error: float
    contact_achieved: bool
    object_lifted: bool
    retained_during_hold: bool
    failure_category: str
    final_object_pose: list[float]
    final_object_lift: float
    max_object_lift: float
    gripper_object_distance: float
    collision_free_approach: bool
    preclose_contact_steps: int
    preclose_max_object_displacement: float
    event_order_valid: bool
    early_close: bool
    reopen_events: int
    max_gripper_contact_force: float
    gripper_contact_impulse_before_lift: float
    max_object_xy_displacement_while_supported: float
    max_object_rotation_while_supported: float
    physical_sanity_pass: bool
    clipped_translation_steps: int
    clipped_rotation_steps: int
    clipped_joint_steps: int
    note: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PickPlaceTrialResult:
    trial_id: int
    orientation: str
    object_pose: str
    approach: str
    placement_target: str
    repeat: int
    success: bool
    object_start_pose: list[float]
    commanded_grasp_pose: list[float]
    commanded_placement_pose: list[float]
    gripper_orientation_wxyz: list[float]
    final_ee_position_error: float
    final_ee_rotation_error: float
    contact_achieved: bool
    object_lifted: bool
    retained_during_hold: bool
    placement_achieved: bool
    placement_xy_error: float
    placement_z_error: float
    gripper_released: bool
    failure_category: str
    final_object_pose: list[float]
    final_object_lift: float
    max_object_lift: float
    gripper_object_distance: float
    collision_free_approach: bool
    preclose_contact_steps: int
    preclose_max_object_displacement: float
    event_order_valid: bool
    early_close: bool
    reopen_events: int
    max_gripper_contact_force: float
    gripper_contact_impulse_before_lift: float
    max_object_xy_displacement_while_supported: float
    max_object_rotation_while_supported: float
    physical_sanity_pass: bool
    clipped_translation_steps: int
    clipped_rotation_steps: int
    clipped_joint_steps: int
    note: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ScriptedControllerCommand:
    phase: str
    target_pos: np.ndarray
    target_quat_wxyz: np.ndarray
    gripper_open: float
    max_steps: int
    stop_on_pose_tolerance: bool = True


class PickupTaskEvaluator:
    """Controller-only pickup task API and benchmark evaluator.

    The public API is intentionally environment-like: reset to an object pose,
    step controller commands, read observations/metrics, and reuse the same
    deterministic scripted command sequence for evaluation or demo recording.
    """

    def __init__(
        self,
        model_path: Path | str = PICKUP_MODEL_PATH,
        object_half_size: np.ndarray | None = None,
        object_sliding_friction: float | None = None,
    ) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.controller = CartesianIKController(self.model, limits=PICKUP_CONTROLLER_LIMITS)
        self.object_joint_id = _require_id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "pickup_object_freejoint"
        )
        self.object_qpos_id = self.model.jnt_qposadr[self.object_joint_id]
        self.object_geom_id = _require_id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "pickup_object_geom"
        )
        self.support_geom_id = _require_id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "support_surface"
        )
        self.gripper_geom_ids = {
            _require_id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in (
                "fixed_jaw_pad_1",
                "fixed_jaw_pad_2",
                "fixed_jaw_pad_3",
                "fixed_jaw_pad_4",
                "moving_jaw_pad_1",
                "moving_jaw_pad_2",
                "moving_jaw_pad_3",
                "moving_jaw_pad_4",
            )
        }
        self._placement_marker_body_ids = {
            label: (
                _require_id(self.model, mujoco.mjtObj.mjOBJ_BODY, goal_body),
                _require_id(self.model, mujoco.mjtObj.mjOBJ_BODY, command_body),
            )
            for label, (goal_body, command_body) in PLACEMENT_MARKER_BODIES.items()
        }
        self.configure_object(
            BASE_OBJECT_HALF_SIZE if object_half_size is None else object_half_size,
            BASE_OBJECT_SLIDING_FRICTION
            if object_sliding_friction is None
            else object_sliding_friction,
        )
        self._episode_object_start = np.array([0.0, -0.235, OBJECT_START_Z])
        self._episode_max_lift = 0.0
        self._episode_contact_steps = 0
        self._episode_close_contact_steps = 0
        self._episode_preclose_contact_steps = 0
        self._episode_preclose_max_object_displacement = 0.0
        self._episode_closure_started = False
        self._episode_last_gripper_command = 1.0
        self._episode_reopen_events = 0
        self._episode_reopen_command_steps = 0
        self._episode_first_close_time: float | None = None
        self._episode_first_contact_time: float | None = None
        self._episode_first_lift_time: float | None = None
        self._episode_first_unsupported_time: float | None = None
        self._episode_close_start_distance: float | None = None
        self._episode_max_gripper_contact_force = 0.0
        self._episode_max_gripper_normal_force = 0.0
        self._episode_gripper_contact_impulse = 0.0
        self._episode_gripper_contact_impulse_before_lift = 0.0
        self._episode_max_object_xy_displacement_before_lift = 0.0
        self._episode_max_object_rotation_before_lift = 0.0
        self._episode_max_object_xy_displacement_while_supported = 0.0
        self._episode_max_object_rotation_while_supported = 0.0
        self._episode_lifted_steps = 0
        self._grasp_segment_frozen = False
        self._grasp_segment_metrics: dict | None = None
        self.reset(
            np.array(
                [0.0, -0.235, SUPPORT_TOP_Z + float(self.object_half_size[2])]
            )
        )

    def configure_object(
        self,
        half_size: np.ndarray,
        sliding_friction: float,
    ) -> None:
        half_size = np.asarray(half_size, dtype=float)
        if half_size.shape != (3,) or not np.isfinite(half_size).all() or np.any(half_size <= 0):
            raise ValueError("object half_size must contain three positive finite values")
        if not np.isfinite(sliding_friction) or sliding_friction <= 0:
            raise ValueError("object sliding_friction must be positive and finite")

        self.object_half_size = half_size.copy()
        self.object_sliding_friction = float(sliding_friction)
        self.model.geom_size[self.object_geom_id] = half_size
        self.model.geom_friction[self.object_geom_id, 0] = sliding_friction

        volume_scale = float(np.prod(half_size / BASE_OBJECT_HALF_SIZE))
        mass = BASE_OBJECT_MASS * volume_scale
        body_id = int(self.model.geom_bodyid[self.object_geom_id])
        self.model.body_mass[body_id] = mass
        self.model.body_inertia[body_id] = (mass / 3.0) * np.array(
            [
                half_size[1] ** 2 + half_size[2] ** 2,
                half_size[0] ** 2 + half_size[2] ** 2,
                half_size[0] ** 2 + half_size[1] ** 2,
            ]
        )
        mujoco.mj_setConst(self.model, self.data)

    def reset(self, object_xyz: np.ndarray) -> dict:
        mujoco.mj_resetData(self.model, self.data)
        neutral = np.array([0.0, -2.07, 1.57, 1.57, -1.57])
        self.data.qpos[self.controller.arm_qpos_ids] = neutral
        self.data.ctrl[self.controller.arm_actuator_ids] = neutral
        self.data.qpos[self.controller.gripper_qpos_ids] = 1.2
        self.controller.set_gripper(self.data, 1.0)
        self._set_object_pose(object_xyz)
        mujoco.mj_forward(self.model, self.data)
        self.controller.reset_target(self.data, posture_target=neutral)
        for _ in range(200):
            mujoco.mj_step(self.model, self.data)
        self.controller.reset_target(self.data, posture_target=neutral)
        self._episode_object_start = self.object_position.copy()
        self._episode_object_start_quat = self.object_quat_wxyz.copy()
        self._episode_max_lift = 0.0
        self._episode_contact_steps = 0
        self._episode_close_contact_steps = 0
        self._episode_preclose_contact_steps = 0
        self._episode_preclose_max_object_displacement = 0.0
        self._episode_closure_started = False
        self._episode_last_gripper_command = 1.0
        self._episode_reopen_events = 0
        self._episode_reopen_command_steps = 0
        self._episode_first_close_time = None
        self._episode_first_contact_time = None
        self._episode_first_lift_time = None
        self._episode_first_unsupported_time = None
        self._episode_close_start_distance = None
        self._episode_max_gripper_contact_force = 0.0
        self._episode_max_gripper_normal_force = 0.0
        self._episode_gripper_contact_impulse = 0.0
        self._episode_gripper_contact_impulse_before_lift = 0.0
        self._episode_max_object_xy_displacement_before_lift = 0.0
        self._episode_max_object_rotation_before_lift = 0.0
        self._episode_max_object_xy_displacement_while_supported = 0.0
        self._episode_max_object_rotation_while_supported = 0.0
        self._episode_lifted_steps = 0
        self._grasp_segment_frozen = False
        self._grasp_segment_metrics = None
        return self.get_observation()

    def _body_position(self, body_name: str) -> np.ndarray:
        body_id = _require_id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        return self.model.body_pos[body_id].copy()

    def placement_marker_position(self, placement_target: PlacementTarget) -> np.ndarray:
        return self._body_position(placement_target.goal_marker_body)

    def placement_goal_xyz(self, placement_target: PlacementTarget) -> np.ndarray:
        marker = self.placement_marker_position(placement_target)
        return np.array(
            [
                marker[0],
                marker[1],
                SUPPORT_TOP_Z + float(self.object_half_size[2]),
            ],
            dtype=float,
        )

    def placement_command_xyz(self, placement_target: PlacementTarget) -> np.ndarray:
        command_body = placement_target.command_marker_body or placement_target.goal_marker_body
        marker = self._body_position(command_body)
        return np.array(
            [
                marker[0],
                marker[1],
                SUPPORT_TOP_Z + float(self.object_half_size[2]),
            ],
            dtype=float,
        )

    def placement_target_xyz(self, placement_target: PlacementTarget) -> np.ndarray:
        return self.placement_command_xyz(placement_target)

    def evaluate_placement(
        self,
        placement_xyz: np.ndarray,
        goal_xyz: np.ndarray | None = None,
    ) -> tuple[bool, dict]:
        goal_xyz = np.asarray(
            placement_xyz if goal_xyz is None else goal_xyz,
            dtype=float,
        )
        xy_error = float(np.linalg.norm(self.object_position[:2] - goal_xyz[:2]))
        expected_z = SUPPORT_TOP_Z + float(self.object_half_size[2])
        z_error = float(abs(self.object_position[2] - expected_z))
        on_support = bool(self.object_support_contact)
        gripper_released = (
            float(self.controller.gripper_open_fraction(self.data))
            >= GRIPPER_RELEASED_THRESHOLD
        )
        placement_achieved = bool(
            on_support
            and xy_error <= PLACEMENT_XY_TOLERANCE
            and z_error <= PLACEMENT_Z_TOLERANCE
            and gripper_released
        )
        return placement_achieved, {
            "placement_xy_error": xy_error,
            "placement_z_error": z_error,
            "object_on_support": on_support,
            "gripper_released": gripper_released,
        }

    def finalize_grasp_segment(self) -> dict:
        """Snapshot pickup grasp gates before intentional gripper release.

        Transport/place phases are excluded from supported-object displacement
        and impulse accounting so pick-and-place does not weaken pickup gates.
        """
        if self._grasp_segment_metrics is not None:
            return self._grasp_segment_metrics
        metrics = self.get_success_metrics()
        self._grasp_segment_metrics = {
            "collision_free_approach": bool(metrics["collision_free_approach"]),
            "event_order_valid": bool(metrics["event_order_valid"]),
            "physical_sanity_pass": bool(metrics["physical_sanity_pass"]),
            "early_close": bool(metrics["early_close"]),
            "reopen_events": int(metrics["reopen_events"]),
            "preclose_contact_steps": int(metrics["preclose_contact_steps"]),
            "preclose_max_object_displacement": float(
                metrics["preclose_max_object_displacement"]
            ),
            "max_gripper_contact_force": float(metrics["max_gripper_contact_force"]),
            "gripper_contact_impulse_before_lift": float(
                metrics["gripper_contact_impulse_before_lift"]
            ),
            "max_object_xy_displacement_while_supported": float(
                metrics["max_object_xy_displacement_while_supported"]
            ),
            "max_object_rotation_while_supported": float(
                metrics["max_object_rotation_while_supported"]
            ),
            "contact_achieved": bool(metrics["contact_achieved"]),
            "object_lifted": bool(metrics["object_lifted"]),
            "retained_during_hold": bool(metrics["retained_during_hold"]),
            "max_object_lift": float(metrics["max_object_lift"]),
        }
        self._grasp_segment_frozen = True
        return self._grasp_segment_metrics

    def get_observation(self) -> dict:
        ee_pos, ee_quat = self.controller.ee_pose(self.data)
        target_pos = (
            self.controller.target_pos.copy()
            if self.controller.target_pos is not None
            else ee_pos.copy()
        )
        target_quat = (
            self.controller.target_quat_wxyz.copy()
            if self.controller.target_quat_wxyz is not None
            else ee_quat.copy()
        )
        object_qpos = self.data.qpos[self.object_qpos_id : self.object_qpos_id + 7].copy()
        return {
            "time": float(self.data.time),
            "joint_positions": _round_list(self.data.qpos[self.controller.arm_qpos_ids]),
            "joint_velocities": _round_list(self.data.qvel[self.controller.arm_dof_ids]),
            "ee_position": _round_list(ee_pos),
            "ee_quat_wxyz": _round_list(ee_quat),
            "ee_target_position": _round_list(target_pos),
            "ee_target_quat_wxyz": _round_list(target_quat),
            "ee_target_minus_actual_xyz": _round_list(target_pos - ee_pos),
            "gripper_open": float(self.controller.gripper_open_fraction(self.data)),
            "object_position": _round_list(object_qpos[:3]),
            "object_quat_wxyz": _round_list(object_qpos[3:7]),
            "gripper_object_contact": bool(self.gripper_object_contact),
            "object_support_contact": bool(self.object_support_contact),
        }

    def get_success_metrics(self) -> dict:
        lift = float(self.object_position[2] - self._episode_object_start[2])
        collision_free_approach = (
            self._episode_preclose_contact_steps == 0
            and self._episode_preclose_max_object_displacement
            <= PRECLOSE_DISPLACEMENT_TOLERANCE
        )
        event_order_valid = self._event_order_valid()
        mass_scale = float(np.prod(self.object_half_size / BASE_OBJECT_HALF_SIZE))
        force_limit = MAX_GRIPPER_CONTACT_FORCE * max(1.0, mass_scale)
        impulse_limit = MAX_GRIPPER_IMPULSE_BEFORE_LIFT * max(1.0, mass_scale)
        supported_xy_limit = MAX_SUPPORTED_XY_DISPLACEMENT * max(
            1.0,
            float(np.max(self.object_half_size[:2] / BASE_OBJECT_HALF_SIZE[:2])),
        )
        physical_sanity_pass = bool(
            self._episode_max_gripper_contact_force <= force_limit
            and self._episode_gripper_contact_impulse_before_lift
            <= impulse_limit
            and self._episode_max_object_xy_displacement_while_supported
            <= supported_xy_limit
            and self._episode_max_object_rotation_while_supported
            <= MAX_SUPPORTED_ROTATION
        )
        return {
            "contact_achieved": self._episode_close_contact_steps > 0,
            "collision_free_approach": collision_free_approach,
            "preclose_contact_steps": int(self._episode_preclose_contact_steps),
            "preclose_max_object_displacement": float(
                self._episode_preclose_max_object_displacement
            ),
            "object_lifted": self._episode_max_lift >= LIFT_CLEARANCE,
            "retained_during_hold": (
                lift >= RETENTION_CLEARANCE
                and self._episode_lifted_steps >= 180
                and self._episode_close_contact_steps >= 60
                and self.gripper_object_distance() <= 0.045
            ),
            "current_object_lift": lift,
            "max_object_lift": float(self._episode_max_lift),
            "gripper_object_distance": self.gripper_object_distance(),
            "gripper_object_contact": bool(self.gripper_object_contact),
            "object_support_contact": bool(self.object_support_contact),
            "contact_steps": int(self._episode_contact_steps),
            "close_contact_steps": int(self._episode_close_contact_steps),
            "closure_started": bool(self._episode_closure_started),
            "early_close": bool(
                self._episode_close_start_distance is not None
                and self._episode_close_start_distance > EARLY_CLOSE_DISTANCE
            ),
            "close_start_distance": self._episode_close_start_distance,
            "reopen_events": int(self._episode_reopen_events),
            "reopen_command_steps": int(self._episode_reopen_command_steps),
            "first_close_time": self._episode_first_close_time,
            "first_contact_time": self._episode_first_contact_time,
            "first_lift_time": self._episode_first_lift_time,
            "first_unsupported_time": self._episode_first_unsupported_time,
            "event_order_valid": event_order_valid,
            "physical_sanity_pass": physical_sanity_pass,
            "gripper_contact_force_limit": force_limit,
            "gripper_contact_impulse_before_lift_limit": impulse_limit,
            "supported_xy_displacement_limit": supported_xy_limit,
            "supported_rotation_limit": MAX_SUPPORTED_ROTATION,
            "max_gripper_contact_force": float(
                self._episode_max_gripper_contact_force
            ),
            "max_gripper_normal_force": float(
                self._episode_max_gripper_normal_force
            ),
            "gripper_contact_impulse": float(
                self._episode_gripper_contact_impulse
            ),
            "gripper_contact_impulse_before_lift": float(
                self._episode_gripper_contact_impulse_before_lift
            ),
            "max_object_xy_displacement_before_lift": float(
                self._episode_max_object_xy_displacement_before_lift
            ),
            "max_object_rotation_before_lift": float(
                self._episode_max_object_rotation_before_lift
            ),
            "max_object_xy_displacement_while_supported": float(
                self._episode_max_object_xy_displacement_while_supported
            ),
            "max_object_rotation_while_supported": float(
                self._episode_max_object_rotation_while_supported
            ),
            "lifted_steps": int(self._episode_lifted_steps),
        }

    def step_controller_command(
        self,
        target_pos: np.ndarray,
        target_quat_wxyz: np.ndarray,
        gripper_open: float,
        substeps: int = 4,
    ) -> tuple[dict, dict, object]:
        status = self.controller.move_toward(self.data, target_pos, target_quat_wxyz)
        self.controller.set_gripper(self.data, gripper_open)
        self._advance_episode_physics(substeps, gripper_open)
        return self.get_observation(), self.get_success_metrics(), status

    def step_ee_delta_action(
        self,
        delta_xyz: np.ndarray,
        delta_rotvec: np.ndarray,
        gripper_open: float,
        substeps: int = 4,
    ) -> tuple[dict, dict, object]:
        """Track one bounded EE intention over the policy control interval."""

        delta_xyz, input_translation_clipped = _clip_norm(
            np.asarray(delta_xyz, dtype=float),
            self.controller.limits.max_step_xyz,
        )
        delta_rotvec, input_rotation_clipped = _clip_norm(
            np.asarray(delta_rotvec, dtype=float),
            self.controller.limits.max_step_rot,
        )
        actual_pos, actual_quat = self.controller.ee_pose(self.data)
        target_pos = actual_pos + delta_xyz
        target_quat = _compose_wxyz(actual_quat, delta_rotvec)
        self.controller.target_pos = target_pos.copy()
        self.controller.target_quat_wxyz = target_quat.copy()

        controller_status = self.controller.move_toward(self.data, target_pos, target_quat)
        self.controller.set_gripper(self.data, gripper_open)
        self._advance_episode_physics(substeps, gripper_open)
        status = IKStatus(
            position_error=controller_status.position_error,
            rotation_error=controller_status.rotation_error,
            clipped_translation=input_translation_clipped
            or controller_status.clipped_translation,
            clipped_rotation=input_rotation_clipped
            or controller_status.clipped_rotation,
            clipped_joints=controller_status.clipped_joints,
            joint_limit_clipped=controller_status.joint_limit_clipped,
            joint_step_clipped=controller_status.joint_step_clipped,
            joint_accel_clipped=controller_status.joint_accel_clipped,
            posture_error=controller_status.posture_error,
            joint_step_norm=controller_status.joint_step_norm,
            saturated=(
                input_translation_clipped
                or input_rotation_clipped
                or controller_status.saturated
            ),
            infeasible=controller_status.infeasible,
            controller_failed=controller_status.controller_failed,
            failure_reason=controller_status.failure_reason,
        )
        return self.get_observation(), self.get_success_metrics(), status

    def step_ee_tool_delta_action(
        self,
        delta_xyz: np.ndarray,
        delta_tilt_xy: np.ndarray,
        gripper_open: float,
        substeps: int = 4,
    ) -> tuple[dict, dict, object]:
        """Apply the five-DOF tool-axis action used by the pickup policy."""

        delta_tilt_xy = np.asarray(delta_tilt_xy, dtype=float)
        if delta_tilt_xy.shape != (2,):
            raise ValueError(
                f"delta_tilt_xy must have shape (2,), got {delta_tilt_xy.shape}"
            )
        if (
            self.controller.limits.orientation_mode != "tool_axis"
            or self.controller.limits.tool_axis_index != 2
        ):
            raise RuntimeError("ee_tool_delta requires local-Z tool-axis control")
        delta_rotvec = np.array([delta_tilt_xy[0], delta_tilt_xy[1], 0.0])
        return self.step_ee_delta_action(
            delta_xyz,
            delta_rotvec,
            gripper_open,
            substeps=substeps,
        )

    def step_joint_delta_action(
        self,
        joint_delta: np.ndarray,
        gripper_open: float,
        substeps: int = 4,
    ) -> tuple[dict, dict, dict]:
        """Apply a joint-delta baseline action and advance physics."""

        joint_delta = np.asarray(joint_delta, dtype=float)
        if joint_delta.shape != self.data.qpos[self.controller.arm_qpos_ids].shape:
            raise ValueError(
                f"joint_delta must have shape {self.data.qpos[self.controller.arm_qpos_ids].shape}, "
                f"got {joint_delta.shape}"
            )
        raw_target = self.data.qpos[self.controller.arm_qpos_ids] + joint_delta
        q_target = np.clip(
            raw_target,
            self.controller.joint_ranges[:, 0],
            self.controller.joint_ranges[:, 1],
        )
        clipped_joints = not np.allclose(raw_target, q_target)
        self.data.ctrl[self.controller.arm_actuator_ids] = q_target
        self.controller.set_gripper(self.data, gripper_open)
        self._advance_episode_physics(substeps, gripper_open)
        actual_pos, actual_quat = self.controller.ee_pose(self.data)
        status = {
            "position_error": 0.0,
            "rotation_error": 0.0,
            "clipped_translation": False,
            "clipped_rotation": False,
            "clipped_joints": bool(clipped_joints),
            "joint_limit_clipped": bool(clipped_joints),
            "joint_step_clipped": False,
            "joint_accel_clipped": False,
            "saturated": bool(clipped_joints),
            "infeasible": bool(clipped_joints),
            "controller_failed": False,
            "failure_reason": None,
            "posture_error": 0.0,
            "joint_step_norm": float(np.linalg.norm(q_target - self.data.qpos[self.controller.arm_qpos_ids])),
            "actual_pos": actual_pos,
            "actual_quat_wxyz": actual_quat,
            "joint_targets": q_target.copy(),
            "joint_positions": self.data.qpos[self.controller.arm_qpos_ids].copy(),
        }
        return self.get_observation(), self.get_success_metrics(), status

    def scripted_controller_commands(
        self,
        spec: PickupTrialSpec,
        settled_start: np.ndarray | None = None,
    ) -> tuple[list[ScriptedControllerCommand], np.ndarray, np.ndarray]:
        if settled_start is None:
            settled_start = self.object_position.copy()
        grasp_quat = spec.orientation.quat_wxyz
        grasp_rotation = spec.orientation.rotation
        grasp_pos = self._scripted_grasp_position(
            settled_start,
            spec.orientation,
        )
        commands = [
            ScriptedControllerCommand(
                phase=f"approach_{index}",
                target_pos=waypoint,
                target_quat_wxyz=grasp_quat,
                gripper_open=1.0,
                max_steps=420,
            )
            for index, waypoint in enumerate(
                self._approach_waypoints(grasp_pos, grasp_rotation, spec.approach)
            )
        ]
        commands.append(
            ScriptedControllerCommand("grasp_align", grasp_pos, grasp_quat, 1.0, 520)
        )
        commands.append(
            ScriptedControllerCommand(
                "close_gripper",
                grasp_pos,
                grasp_quat,
                0.0,
                260,
                stop_on_pose_tolerance=False,
            )
        )
        commands.append(
            ScriptedControllerCommand(
                "lift",
                grasp_pos + np.array([0.0, 0.0, 0.070]),
                grasp_quat,
                0.0,
                520,
            )
        )
        commands.append(
            ScriptedControllerCommand(
                "hold",
                grasp_pos + np.array([0.0, 0.0, 0.070]),
                grasp_quat,
                0.0,
                300,
                stop_on_pose_tolerance=False,
            )
        )
        return commands, grasp_pos, grasp_quat

    def scripted_pick_place_commands(
        self,
        spec: PickPlaceTrialSpec,
        settled_start: np.ndarray | None = None,
    ) -> tuple[list[ScriptedControllerCommand], np.ndarray, np.ndarray, np.ndarray]:
        pickup_spec = PickupTrialSpec(
            trial_id=spec.trial_id,
            orientation=spec.orientation,
            object_pose=spec.object_pose,
            approach=spec.approach,
            repeat=spec.repeat,
        )
        commands, grasp_pos, grasp_quat = self.scripted_controller_commands(
            pickup_spec,
            settled_start,
        )
        place_goal = self.placement_goal_xyz(spec.placement_target)
        place_pos = self.placement_command_xyz(spec.placement_target)
        transport_pos = place_pos + np.array([0.0, 0.0, PICK_PLACE_LIFT_HEIGHT])
        commands.extend(
            [
                ScriptedControllerCommand(
                    "transport",
                    transport_pos,
                    grasp_quat,
                    0.0,
                    760,
                ),
                ScriptedControllerCommand(
                    "lower",
                    place_pos,
                    grasp_quat,
                    0.0,
                    520,
                ),
                ScriptedControllerCommand(
                    "open_gripper",
                    place_pos,
                    grasp_quat,
                    1.0,
                    220,
                    stop_on_pose_tolerance=False,
                ),
                ScriptedControllerCommand(
                    "retreat",
                    place_pos + np.array([0.0, 0.0, 0.060]),
                    grasp_quat,
                    1.0,
                    320,
                ),
            ]
        )
        return commands, grasp_pos, grasp_quat, place_goal

    def run_pick_place_trial(self, spec: PickPlaceTrialSpec) -> PickPlaceTrialResult:
        object_start = np.asarray(spec.object_pose.xyz, dtype=float)
        self.reset(object_start)
        settled_start = self.object_position.copy()
        if not np.isfinite(settled_start).all() or settled_start[2] < SUPPORT_TOP_Z:
            return self._pick_place_result(
                spec=spec,
                success=False,
                object_start=settled_start,
                grasp_pos=np.full(3, np.nan),
                place_pos=np.full(3, np.nan),
                grasp_quat=spec.orientation.quat_wxyz,
                final_position_error=float("inf"),
                final_rotation_error=float("inf"),
                placement_xy_error=float("inf"),
                placement_z_error=float("inf"),
                placement_achieved=False,
                gripper_released=False,
                failure_category="task_or_scene_setup_failure",
                max_lift=0.0,
                note="object did not settle on the support surface",
            )

        commands, grasp_pos, grasp_quat, place_goal = self.scripted_pick_place_commands(
            spec,
            settled_start,
        )
        clipped_translation = 0
        clipped_rotation = 0
        clipped_joints = 0
        final_position_error = float("inf")
        final_rotation_error = float("inf")
        reached_grasp = False
        contact_during_close = False
        retained = False
        max_lift = 0.0

        step_index = 0
        grasp_boundary_index = None
        for command in commands:
            if (
                grasp_boundary_index is None
                and command.phase == PICK_PLACE_TRANSPORT_PHASE
            ):
                grasp_boundary_index = step_index
            phase_stats = self._execute_scripted_command(
                command,
                step_index_start=step_index,
                grasp_boundary_index=grasp_boundary_index,
            )
            step_index += int(phase_stats["steps_executed"])
            clipped_translation += int(phase_stats["clipped_translation"])
            clipped_rotation += int(phase_stats["clipped_rotation"])
            clipped_joints += int(phase_stats["clipped_joints"])
            max_lift = max(max_lift, float(phase_stats["max_lift"]))
            if command.phase == "grasp_align":
                final_position_error = float(phase_stats["position_error"])
                final_rotation_error = float(phase_stats["rotation_error"])
                reached_grasp = (
                    final_position_error <= 0.012 and final_rotation_error <= 0.22
                )
            if command.phase == "close_gripper":
                contact_during_close = contact_during_close or bool(
                    int(phase_stats["close_contact_steps_delta"]) > 0
                )
            if command.phase == "hold":
                retained = bool(phase_stats["retained_during_hold"])

        placement_achieved, placement_metrics = self.evaluate_placement(
            place_goal,
            goal_xyz=place_goal,
        )
        grasp_metrics = self._grasp_segment_metrics or self.finalize_grasp_segment()
        collision_free_approach = bool(grasp_metrics["collision_free_approach"])
        event_order_valid = bool(grasp_metrics["event_order_valid"])
        physical_sanity_pass = bool(grasp_metrics["physical_sanity_pass"])
        object_lifted = bool(grasp_metrics["object_lifted"])
        success = bool(
            collision_free_approach
            and event_order_valid
            and physical_sanity_pass
            and reached_grasp
            and contact_during_close
            and object_lifted
            and retained
            and placement_achieved
        )
        failure_category, note = self._classify_pick_place_failure(
            collision_free_approach=collision_free_approach,
            event_order_valid=event_order_valid,
            physical_sanity_pass=physical_sanity_pass,
            reached_grasp=reached_grasp,
            contact=contact_during_close,
            lifted=object_lifted,
            retained=retained,
            placement_achieved=placement_achieved,
            gripper_released=bool(placement_metrics["gripper_released"]),
            final_position_error=final_position_error,
            final_rotation_error=final_rotation_error,
            clipped_joint_steps=clipped_joints,
        )
        return self._pick_place_result(
            spec=spec,
            success=success,
            object_start=settled_start,
            grasp_pos=grasp_pos,
            place_pos=place_goal,
            grasp_quat=grasp_quat,
            final_position_error=final_position_error,
            final_rotation_error=final_rotation_error,
            placement_xy_error=float(placement_metrics["placement_xy_error"]),
            placement_z_error=float(placement_metrics["placement_z_error"]),
            placement_achieved=placement_achieved,
            gripper_released=bool(placement_metrics["gripper_released"]),
            failure_category=failure_category,
            max_lift=max_lift,
            clipped_translation=clipped_translation,
            clipped_rotation=clipped_rotation,
            clipped_joints=clipped_joints,
            note=note,
            grasp_metrics=grasp_metrics,
            contact=contact_during_close,
            lifted=object_lifted,
            retained=retained,
        )

    def run_trial(self, spec: PickupTrialSpec) -> PickupTrialResult:
        object_start = np.asarray(spec.object_pose.xyz, dtype=float)
        self.reset(object_start)
        settled_start = self.object_position.copy()
        if not np.isfinite(settled_start).all() or settled_start[2] < SUPPORT_TOP_Z:
            return self._result(
                spec=spec,
                success=False,
                object_start=settled_start,
                grasp_pos=np.full(3, np.nan),
                grasp_quat=spec.orientation.quat_wxyz,
                final_position_error=float("inf"),
                final_rotation_error=float("inf"),
                contact=False,
                lifted=False,
                retained=False,
                failure_category="task_or_scene_setup_failure",
                max_lift=0.0,
                note="object did not settle on the support surface",
            )

        grasp_quat = spec.orientation.quat_wxyz
        grasp_rotation = spec.orientation.rotation
        grasp_pos = self._scripted_grasp_position(
            settled_start,
            spec.orientation,
        )
        waypoints = self._approach_waypoints(grasp_pos, grasp_rotation, spec.approach)
        clipped_translation = 0
        clipped_rotation = 0
        clipped_joints = 0
        final_position_error = float("inf")
        final_rotation_error = float("inf")

        for waypoint in waypoints:
            stats = self._move_to_pose(waypoint, grasp_quat, gripper_open=1.0, max_steps=420)
            clipped_translation += stats["clipped_translation"]
            clipped_rotation += stats["clipped_rotation"]
            clipped_joints += stats["clipped_joints"]

        stats = self._move_to_pose(grasp_pos, grasp_quat, gripper_open=1.0, max_steps=520)
        clipped_translation += stats["clipped_translation"]
        clipped_rotation += stats["clipped_rotation"]
        clipped_joints += stats["clipped_joints"]
        final_position_error = stats["position_error"]
        final_rotation_error = stats["rotation_error"]

        reached_grasp = final_position_error <= 0.012 and final_rotation_error <= 0.22
        contact_during_close = False
        max_lift = 0.0
        for _ in range(260):
            status = self.controller.move_toward(self.data, grasp_pos, grasp_quat)
            self.controller.set_gripper(self.data, 0.0)
            self._advance_episode_physics(4, gripper_open=0.0)
            contact_during_close = contact_during_close or self.gripper_object_contact
            max_lift = max(max_lift, self.object_position[2] - settled_start[2])
            clipped_translation += int(status.clipped_translation)
            clipped_rotation += int(status.clipped_rotation)
            clipped_joints += int(status.clipped_joints)

        lift_target = grasp_pos + np.array([0.0, 0.0, 0.070])
        stats = self._move_to_pose(lift_target, grasp_quat, gripper_open=0.0, max_steps=520)
        clipped_translation += stats["clipped_translation"]
        clipped_rotation += stats["clipped_rotation"]
        clipped_joints += stats["clipped_joints"]
        max_lift = max(max_lift, stats["max_lift_from_start"] + self.object_position[2] * 0.0)

        hold_contact_steps = 0
        hold_lifted_steps = 0
        for _ in range(300):
            status = self.controller.move_toward(self.data, lift_target, grasp_quat)
            self.controller.set_gripper(self.data, 0.0)
            previous_contact_steps = self._episode_close_contact_steps
            previous_lifted_steps = self._episode_lifted_steps
            self._advance_episode_physics(4, gripper_open=0.0)
            lift = self.object_position[2] - settled_start[2]
            max_lift = max(max_lift, lift)
            hold_contact_steps += self._episode_close_contact_steps - previous_contact_steps
            hold_lifted_steps += self._episode_lifted_steps - previous_lifted_steps
            clipped_translation += int(status.clipped_translation)
            clipped_rotation += int(status.clipped_rotation)
            clipped_joints += int(status.clipped_joints)

        final_object_lift = self.object_position[2] - settled_start[2]
        object_lifted = max_lift >= LIFT_CLEARANCE
        retained = (
            final_object_lift >= RETENTION_CLEARANCE
            and hold_lifted_steps >= 180
            and hold_contact_steps >= 60
            and self.gripper_object_distance() <= 0.045
        )
        final_metrics = self.get_success_metrics()
        collision_free_approach = final_metrics["collision_free_approach"]
        event_order_valid = final_metrics["event_order_valid"]
        physical_sanity_pass = final_metrics["physical_sanity_pass"]
        success = (
            collision_free_approach
            and event_order_valid
            and physical_sanity_pass
            and reached_grasp
            and contact_during_close
            and object_lifted
            and retained
        )
        failure_category, note = self._classify_failure(
            collision_free_approach=collision_free_approach,
            event_order_valid=event_order_valid,
            physical_sanity_pass=physical_sanity_pass,
            reached_grasp=reached_grasp,
            contact=contact_during_close,
            lifted=object_lifted,
            retained=retained,
            final_position_error=final_position_error,
            final_rotation_error=final_rotation_error,
            clipped_joint_steps=clipped_joints,
        )
        return self._result(
            spec=spec,
            success=bool(success),
            object_start=settled_start,
            grasp_pos=grasp_pos,
            grasp_quat=grasp_quat,
            final_position_error=final_position_error,
            final_rotation_error=final_rotation_error,
            contact=contact_during_close,
            lifted=object_lifted,
            retained=retained,
            failure_category=failure_category,
            max_lift=max_lift,
            clipped_translation=clipped_translation,
            clipped_rotation=clipped_rotation,
            clipped_joints=clipped_joints,
            note=note,
        )

    def _execute_scripted_command(
        self,
        command: ScriptedControllerCommand,
        *,
        step_index_start: int = 0,
        grasp_boundary_index: int | None = None,
    ) -> dict[str, float | int | bool]:
        clipped_translation = 0
        clipped_rotation = 0
        clipped_joints = 0
        max_lift = 0.0
        position_error = float("inf")
        rotation_error = float("inf")
        contact_achieved = False
        hold_contact_steps = 0
        hold_lifted_steps = 0
        close_contact_steps_start = self._episode_close_contact_steps
        hold_start_contact = self._episode_close_contact_steps
        hold_start_lifted = self._episode_lifted_steps
        settled_start = self._episode_object_start.copy()
        steps_executed = 0

        for _ in range(command.max_steps):
            maybe_finalize_grasp_at_sample(
                self,
                step_index_start + steps_executed,
                grasp_boundary_index,
            )
            status = self.controller.move_toward(
                self.data,
                command.target_pos,
                command.target_quat_wxyz,
            )
            self.controller.set_gripper(self.data, command.gripper_open)
            self._advance_episode_physics(4, command.gripper_open)
            max_lift = max(max_lift, self._episode_max_lift)
            clipped_translation += int(status.clipped_translation)
            clipped_rotation += int(status.clipped_rotation)
            clipped_joints += int(status.clipped_joints)
            contact_achieved = contact_achieved or self.gripper_object_contact
            if command.phase == "hold":
                hold_contact_steps = self._episode_close_contact_steps - hold_start_contact
                hold_lifted_steps = self._episode_lifted_steps - hold_start_lifted
            position_error = float(np.linalg.norm(command.target_pos - self.controller.ee_pose(self.data)[0]))
            rotation_error = float(
                np.linalg.norm(
                    _orientation_error_rotvec(
                        self.controller.ee_pose(self.data)[1],
                        command.target_quat_wxyz,
                    )
                )
            )
            steps_executed += 1
            if (
                command.stop_on_pose_tolerance
                and status.position_error <= self.controller.limits.position_tolerance
                and status.rotation_error <= self.controller.limits.rotation_tolerance
            ):
                break

        retained_during_hold = False
        if command.phase == "hold":
            lift = float(self.object_position[2] - settled_start[2])
            retained_during_hold = bool(
                lift >= RETENTION_CLEARANCE
                and hold_lifted_steps >= 180
                and hold_contact_steps >= 60
                and self.gripper_object_distance() <= 0.045
            )

        return {
            "position_error": position_error,
            "rotation_error": rotation_error,
            "clipped_translation": clipped_translation,
            "clipped_rotation": clipped_rotation,
            "clipped_joints": clipped_joints,
            "max_lift": max_lift,
            "contact_achieved": contact_achieved,
            "close_contact_steps_delta": (
                self._episode_close_contact_steps - close_contact_steps_start
            ),
            "retained_during_hold": retained_during_hold,
            "steps_executed": steps_executed,
        }

    def _move_to_pose(
        self,
        target_pos: np.ndarray,
        target_quat_wxyz: np.ndarray,
        gripper_open: float,
        max_steps: int,
    ) -> dict[str, float | int]:
        clipped_translation = 0
        clipped_rotation = 0
        clipped_joints = 0
        max_lift = 0.0
        status = None
        for _ in range(max_steps):
            status = self.controller.move_toward(self.data, target_pos, target_quat_wxyz)
            self.controller.set_gripper(self.data, gripper_open)
            self._advance_episode_physics(4, gripper_open)
            max_lift = max(max_lift, self._episode_max_lift)
            clipped_translation += int(status.clipped_translation)
            clipped_rotation += int(status.clipped_rotation)
            clipped_joints += int(status.clipped_joints)
            if status.position_error <= 0.006 and status.rotation_error <= 0.08:
                break
        actual_pos, actual_quat = self.controller.ee_pose(self.data)
        return {
            "position_error": float(np.linalg.norm(target_pos - actual_pos)),
            "rotation_error": float(
                np.linalg.norm(_orientation_error_rotvec(actual_quat, target_quat_wxyz))
            ),
            "clipped_translation": clipped_translation,
            "clipped_rotation": clipped_rotation,
            "clipped_joints": clipped_joints,
            "max_lift_from_start": max_lift,
        }

    def _advance_episode_physics(self, substeps: int, gripper_open: float) -> None:
        if not self._episode_closure_started and gripper_open <= PRECLOSE_OPEN_THRESHOLD:
            self._episode_closure_started = True
            self._episode_first_close_time = float(self.data.time)
            self._episode_close_start_distance = self.gripper_object_distance()
        if not self._grasp_segment_frozen:
            if (
                self._episode_closure_started
                and self._episode_last_gripper_command <= PRECLOSE_OPEN_THRESHOLD
                and gripper_open > PRECLOSE_OPEN_THRESHOLD
            ):
                self._episode_reopen_events += 1
            if self._episode_closure_started and gripper_open > PRECLOSE_OPEN_THRESHOLD:
                self._episode_reopen_command_steps += 1
        self._episode_last_gripper_command = float(gripper_open)

        for _ in range(substeps):
            mujoco.mj_step(self.model, self.data)
            lift = float(self.object_position[2] - self._episode_object_start[2])
            self._episode_max_lift = max(self._episode_max_lift, lift)
            contact = self.gripper_object_contact
            gripper_force, gripper_normal_force = self._object_contact_force(
                self.gripper_geom_ids
            )
            self._episode_max_gripper_contact_force = max(
                self._episode_max_gripper_contact_force,
                gripper_force,
            )
            self._episode_max_gripper_normal_force = max(
                self._episode_max_gripper_normal_force,
                gripper_normal_force,
            )
            contact_impulse = gripper_force * self.model.opt.timestep
            self._episode_gripper_contact_impulse += contact_impulse
            if self._episode_first_lift_time is None:
                self._episode_gripper_contact_impulse_before_lift += contact_impulse
            if contact:
                if self._episode_first_contact_time is None:
                    self._episode_first_contact_time = float(self.data.time)
                self._episode_contact_steps += 1
                if not self._episode_closure_started:
                    self._episode_preclose_contact_steps += 1
                elif gripper_open <= PRECLOSE_OPEN_THRESHOLD:
                    self._episode_close_contact_steps += 1
            if not self._episode_closure_started:
                displacement = float(
                    np.linalg.norm(self.object_position - self._episode_object_start)
                )
                self._episode_preclose_max_object_displacement = max(
                    self._episode_preclose_max_object_displacement,
                    displacement,
                )
            if self._episode_first_lift_time is None:
                xy_displacement = float(
                    np.linalg.norm(
                        self.object_position[:2] - self._episode_object_start[:2]
                    )
                )
                rotation = float(
                    np.linalg.norm(
                        _orientation_error_rotvec(
                            self._episode_object_start_quat,
                            self.object_quat_wxyz,
                        )
                    )
                )
                self._episode_max_object_xy_displacement_before_lift = max(
                    self._episode_max_object_xy_displacement_before_lift,
                    xy_displacement,
                )
                self._episode_max_object_rotation_before_lift = max(
                    self._episode_max_object_rotation_before_lift,
                    rotation,
                )
                if self.object_support_contact:
                    self._episode_max_object_xy_displacement_while_supported = max(
                        self._episode_max_object_xy_displacement_while_supported,
                        xy_displacement,
                    )
                    self._episode_max_object_rotation_while_supported = max(
                        self._episode_max_object_rotation_while_supported,
                        rotation,
                    )
            if lift >= LIFT_CLEARANCE and self._episode_first_lift_time is None:
                self._episode_first_lift_time = float(self.data.time)
            if (
                self._episode_first_contact_time is not None
                and not self.object_support_contact
                and self._episode_first_unsupported_time is None
            ):
                self._episode_first_unsupported_time = float(self.data.time)
            if lift >= LIFT_CLEARANCE and not self.object_support_contact:
                self._episode_lifted_steps += 1

    def _event_order_valid(self) -> bool:
        times = (
            self._episode_first_close_time,
            self._episode_first_contact_time,
            self._episode_first_unsupported_time,
            self._episode_first_lift_time,
        )
        return bool(
            all(time is not None for time in times)
            and times[0] <= times[1] <= times[2] <= times[3]
            and self._episode_preclose_contact_steps == 0
            and self._episode_reopen_events == 0
            and self._episode_close_start_distance is not None
            and self._episode_close_start_distance <= EARLY_CLOSE_DISTANCE
        )

    def _object_contact_force(self, geom_ids: set[int]) -> tuple[float, float]:
        total_force = 0.0
        total_normal_force = 0.0
        force = np.zeros(6)
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            pair = {contact.geom1, contact.geom2}
            if self.object_geom_id not in pair or not pair.intersection(geom_ids):
                continue
            mujoco.mj_contactForce(self.model, self.data, index, force)
            total_force += float(np.linalg.norm(force[:3]))
            total_normal_force += abs(float(force[0]))
        return total_force, total_normal_force

    def _approach_waypoints(
        self,
        grasp_pos: np.ndarray,
        grasp_rotation: Rotation,
        approach: ApproachStrategy,
    ) -> list[np.ndarray]:
        if approach.pregrasp_axis == "tool_z":
            retreat = grasp_rotation.apply(np.array([0.0, 0.0, 0.040]))
            return [grasp_pos + retreat]
        if approach.pregrasp_axis == "world_z":
            return [grasp_pos + np.array([0.0, 0.0, 0.055]), grasp_pos + np.array([0.0, 0.0, 0.020])]
        if approach.pregrasp_axis == "high_world_z":
            return [
                grasp_pos + np.array([0.0, 0.0, 0.085]),
                grasp_pos + np.array([0.0, 0.0, 0.035]),
                grasp_pos + np.array([0.0, 0.0, 0.015]),
            ]
        raise ValueError(f"Unknown approach axis: {approach.pregrasp_axis}")

    def _scripted_grasp_position(
        self,
        object_center: np.ndarray,
        orientation: GraspOrientation,
    ) -> np.ndarray:
        grasp_pos = np.asarray(object_center, dtype=float).copy()
        grasp_pos[2] = SUPPORT_TOP_Z + OBJECT_HALF_HEIGHT
        yaw = np.deg2rad(orientation.yaw_degrees)
        projection = np.array([abs(np.cos(yaw)), abs(np.sin(yaw))])
        object_radius = float(projection @ self.object_half_size[:2])
        baseline_radius = float(projection @ BASE_OBJECT_HALF_SIZE[:2])
        radius_delta = object_radius - baseline_radius
        compensation_gain = (
            GRASP_WIDTH_SHRINK_COMPENSATION_GAIN
            if radius_delta < 0.0
            else GRASP_WIDTH_GROWTH_COMPENSATION_GAIN
        )
        local_x_correction = compensation_gain * radius_delta
        return grasp_pos + orientation.rotation.apply(
            np.array([local_x_correction, 0.0, 0.0])
        )

    def _set_object_pose(self, xyz: np.ndarray) -> None:
        qpos = self.data.qpos
        qpos[self.object_qpos_id : self.object_qpos_id + 3] = xyz
        qpos[self.object_qpos_id + 3 : self.object_qpos_id + 7] = np.array([1.0, 0.0, 0.0, 0.0])
        self.data.qvel[self.model.jnt_dofadr[self.object_joint_id] : self.model.jnt_dofadr[self.object_joint_id] + 6] = 0.0

    @property
    def object_position(self) -> np.ndarray:
        return self.data.qpos[self.object_qpos_id : self.object_qpos_id + 3].copy()

    @property
    def object_quat_wxyz(self) -> np.ndarray:
        return self.data.qpos[self.object_qpos_id + 3 : self.object_qpos_id + 7].copy()

    @property
    def gripper_object_contact(self) -> bool:
        return self._object_contact_with(self.gripper_geom_ids)

    @property
    def object_support_contact(self) -> bool:
        return self._object_contact_with({self.support_geom_id})

    def _object_contact_with(self, geom_ids: set[int]) -> bool:
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            pair = {contact.geom1, contact.geom2}
            if self.object_geom_id in pair and pair.intersection(geom_ids):
                return True
        return False

    def gripper_object_distance(self) -> float:
        grasp_center = self.data.site_xpos[self.controller.ee_site_id]
        return float(np.linalg.norm(self.object_position - grasp_center))

    def _classify_failure(
        self,
        collision_free_approach: bool,
        event_order_valid: bool,
        physical_sanity_pass: bool,
        reached_grasp: bool,
        contact: bool,
        lifted: bool,
        retained: bool,
        final_position_error: float,
        final_rotation_error: float,
        clipped_joint_steps: int,
    ) -> tuple[str, str]:
        if not collision_free_approach:
            return (
                "grasp_geometry_failure",
                "open gripper contacted or displaced the object before closure",
            )
        if not event_order_valid:
            return (
                "event_order_failure",
                "close, contact, unsupported, and lift events were out of order or gripper reopened",
            )
        if not physical_sanity_pass:
            return (
                "contact_dynamics_failure",
                "contact force, impulse, or supported object disturbance exceeded the audit limit",
            )
        if not np.isfinite(final_position_error) or not np.isfinite(final_rotation_error):
            return "evaluation_bug", "non-finite pose error"
        if not reached_grasp:
            if clipped_joint_steps > 100:
                return "controller_or_ik_failure", "IK stayed outside tolerance with joint-step clipping"
            return "controller_or_ik_failure", "EE did not reach commanded grasp pose"
        if not contact:
            return "gripper_or_contact_model_failure", "gripper reached pose but did not contact object"
        if not lifted:
            return "gripper_or_contact_model_failure", "contact occurred but object did not lift clear of support"
        if not retained:
            return "gripper_or_contact_model_failure", "object lifted but was not retained through hold"
        return "none", "pickup met reach, contact, lift, and hold criteria"

    def _classify_pick_place_failure(
        self,
        collision_free_approach: bool,
        event_order_valid: bool,
        physical_sanity_pass: bool,
        reached_grasp: bool,
        contact: bool,
        lifted: bool,
        retained: bool,
        placement_achieved: bool,
        gripper_released: bool,
        final_position_error: float,
        final_rotation_error: float,
        clipped_joint_steps: int,
    ) -> tuple[str, str]:
        pickup_category, pickup_note = self._classify_failure(
            collision_free_approach=collision_free_approach,
            event_order_valid=event_order_valid,
            physical_sanity_pass=physical_sanity_pass,
            reached_grasp=reached_grasp,
            contact=contact,
            lifted=lifted,
            retained=retained,
            final_position_error=final_position_error,
            final_rotation_error=final_rotation_error,
            clipped_joint_steps=clipped_joint_steps,
        )
        if pickup_category != "none":
            return pickup_category, pickup_note
        if not placement_achieved:
            if not gripper_released:
                return "placement_failure", "object reached target region but gripper did not release"
            if not self.object_support_contact:
                return "placement_failure", "gripper released but object is not resting on support"
            return (
                "placement_failure",
                "object released on support but outside placement XY/Z tolerance",
            )
        return "none", "pick-and-place met grasp gates and placement criteria"

    def _pick_place_result(
        self,
        spec: PickPlaceTrialSpec,
        success: bool,
        object_start: np.ndarray,
        grasp_pos: np.ndarray,
        place_pos: np.ndarray,
        grasp_quat: np.ndarray,
        final_position_error: float,
        final_rotation_error: float,
        placement_xy_error: float,
        placement_z_error: float,
        placement_achieved: bool,
        gripper_released: bool,
        failure_category: str,
        max_lift: float,
        note: str,
        contact: bool = False,
        lifted: bool = False,
        retained: bool = False,
        clipped_translation: int = 0,
        clipped_rotation: int = 0,
        clipped_joints: int = 0,
        grasp_metrics: dict | None = None,
    ) -> PickPlaceTrialResult:
        if grasp_metrics is None:
            grasp_metrics = {
                "collision_free_approach": False,
                "event_order_valid": False,
                "physical_sanity_pass": False,
                "early_close": False,
                "reopen_events": 0,
                "preclose_contact_steps": 0,
                "preclose_max_object_displacement": 0.0,
                "max_gripper_contact_force": 0.0,
                "gripper_contact_impulse_before_lift": 0.0,
                "max_object_xy_displacement_while_supported": 0.0,
                "max_object_rotation_while_supported": 0.0,
                "contact_achieved": False,
                "object_lifted": False,
                "retained_during_hold": False,
                "max_object_lift": 0.0,
            }
        return PickPlaceTrialResult(
            trial_id=spec.trial_id,
            orientation=spec.orientation.label,
            object_pose=spec.object_pose.label,
            approach=spec.approach.label,
            placement_target=spec.placement_target.label,
            repeat=spec.repeat,
            success=bool(success),
            object_start_pose=_round_list(object_start),
            commanded_grasp_pose=_round_list(grasp_pos),
            commanded_placement_pose=_round_list(place_pos),
            gripper_orientation_wxyz=_round_list(grasp_quat),
            final_ee_position_error=float(final_position_error),
            final_ee_rotation_error=float(final_rotation_error),
            contact_achieved=bool(contact),
            object_lifted=bool(lifted or grasp_metrics["object_lifted"]),
            retained_during_hold=bool(retained or grasp_metrics["retained_during_hold"]),
            placement_achieved=bool(placement_achieved),
            placement_xy_error=float(placement_xy_error),
            placement_z_error=float(placement_z_error),
            gripper_released=bool(gripper_released),
            failure_category=failure_category,
            final_object_pose=_round_list(self.object_position),
            final_object_lift=float(self.object_position[2] - object_start[2]),
            max_object_lift=float(max_lift),
            gripper_object_distance=self.gripper_object_distance()
            if np.isfinite(grasp_pos).all()
            else float("inf"),
            collision_free_approach=bool(grasp_metrics["collision_free_approach"]),
            preclose_contact_steps=int(grasp_metrics["preclose_contact_steps"]),
            preclose_max_object_displacement=float(
                grasp_metrics["preclose_max_object_displacement"]
            ),
            event_order_valid=bool(grasp_metrics["event_order_valid"]),
            early_close=bool(grasp_metrics["early_close"]),
            reopen_events=int(grasp_metrics["reopen_events"]),
            max_gripper_contact_force=float(grasp_metrics["max_gripper_contact_force"]),
            gripper_contact_impulse_before_lift=float(
                grasp_metrics["gripper_contact_impulse_before_lift"]
            ),
            max_object_xy_displacement_while_supported=float(
                grasp_metrics["max_object_xy_displacement_while_supported"]
            ),
            max_object_rotation_while_supported=float(
                grasp_metrics["max_object_rotation_while_supported"]
            ),
            physical_sanity_pass=bool(grasp_metrics["physical_sanity_pass"]),
            clipped_translation_steps=clipped_translation,
            clipped_rotation_steps=clipped_rotation,
            clipped_joint_steps=clipped_joints,
            note=note,
        )

    def _result(
        self,
        spec: PickupTrialSpec,
        success: bool,
        object_start: np.ndarray,
        grasp_pos: np.ndarray,
        grasp_quat: np.ndarray,
        final_position_error: float,
        final_rotation_error: float,
        contact: bool,
        lifted: bool,
        retained: bool,
        failure_category: str,
        max_lift: float,
        clipped_translation: int = 0,
        clipped_rotation: int = 0,
        clipped_joints: int = 0,
        note: str = "",
    ) -> PickupTrialResult:
        metrics = self.get_success_metrics()
        return PickupTrialResult(
            trial_id=spec.trial_id,
            orientation=spec.orientation.label,
            object_pose=spec.object_pose.label,
            approach=spec.approach.label,
            repeat=spec.repeat,
            success=success,
            object_start_pose=_round_list(object_start),
            commanded_grasp_pose=_round_list(grasp_pos),
            gripper_orientation_wxyz=_round_list(grasp_quat),
            final_ee_position_error=float(final_position_error),
            final_ee_rotation_error=float(final_rotation_error),
            contact_achieved=bool(contact),
            object_lifted=bool(lifted),
            retained_during_hold=bool(retained),
            failure_category=failure_category,
            final_object_pose=_round_list(self.object_position),
            final_object_lift=float(self.object_position[2] - object_start[2]),
            max_object_lift=float(max_lift),
            gripper_object_distance=self.gripper_object_distance()
            if np.isfinite(grasp_pos).all()
            else float("inf"),
            collision_free_approach=bool(
                self.get_success_metrics()["collision_free_approach"]
            ),
            preclose_contact_steps=int(self._episode_preclose_contact_steps),
            preclose_max_object_displacement=float(
                self._episode_preclose_max_object_displacement
            ),
            event_order_valid=bool(metrics["event_order_valid"]),
            early_close=bool(metrics["early_close"]),
            reopen_events=int(metrics["reopen_events"]),
            max_gripper_contact_force=float(metrics["max_gripper_contact_force"]),
            gripper_contact_impulse_before_lift=float(
                metrics["gripper_contact_impulse_before_lift"]
            ),
            max_object_xy_displacement_while_supported=float(
                metrics["max_object_xy_displacement_while_supported"]
            ),
            max_object_rotation_while_supported=float(
                metrics["max_object_rotation_while_supported"]
            ),
            physical_sanity_pass=bool(metrics["physical_sanity_pass"]),
            clipped_translation_steps=clipped_translation,
            clipped_rotation_steps=clipped_rotation,
            clipped_joint_steps=clipped_joints,
            note=note,
        )


def default_pick_place_trial_specs() -> list[PickPlaceTrialSpec]:
    placement_targets = [
        PlacementTarget(
            "place_left",
            PLACEMENT_MARKER_BODIES["place_left"][0],
            PLACEMENT_MARKER_BODIES["place_left"][1],
        ),
        PlacementTarget(
            "place_right",
            PLACEMENT_MARKER_BODIES["place_right"][0],
            PLACEMENT_MARKER_BODIES["place_right"][1],
        ),
    ]
    object_poses = [
        ObjectStartPose("center", np.array([0.000, -0.235, OBJECT_START_Z])),
        ObjectStartPose("left", np.array([-0.018, -0.232, OBJECT_START_Z])),
        ObjectStartPose("right", np.array([0.018, -0.238, OBJECT_START_Z])),
    ]
    approaches = [
        ApproachStrategy("vertical_pregrasp", "world_z"),
        ApproachStrategy("high_staged_vertical_pregrasp", "high_world_z"),
    ]
    matrix = [
        ("place_left", "center", "vertical_pregrasp"),
        ("place_left", "left", "vertical_pregrasp"),
        ("place_right", "center", "vertical_pregrasp"),
        ("place_right", "right", "vertical_pregrasp"),
        ("place_left", "right", "high_staged_vertical_pregrasp"),
        ("place_right", "left", "high_staged_vertical_pregrasp"),
    ]
    placement_by_label = {target.label: target for target in placement_targets}
    pose_by_label = {pose.label: pose for pose in object_poses}
    approach_by_label = {approach.label: approach for approach in approaches}
    specs: list[PickPlaceTrialSpec] = []
    for trial_id, (placement_label, pose_label, approach_label) in enumerate(matrix, start=1):
        specs.append(
            PickPlaceTrialSpec(
                trial_id=trial_id,
                orientation=GraspOrientation("yaw_0", 0.0),
                object_pose=pose_by_label[pose_label],
                approach=approach_by_label[approach_label],
                placement_target=placement_by_label[placement_label],
            )
        )
    return specs


def summarize_pick_place_results(results: Iterable[PickPlaceTrialResult]) -> dict:
    results = list(results)
    return {
        "total": len(results),
        "successes": sum(result.success for result in results),
        "success_rate": _rate(result.success for result in results),
        "by_placement_target": _bucket_rates(results, "placement_target"),
        "by_object_pose": _bucket_rates(results, "object_pose"),
        "by_approach": _bucket_rates(results, "approach"),
        "failure_categories": _failure_counts(results),
        "collision_free_approach_rate": _rate(
            result.collision_free_approach for result in results
        ),
        "event_order_valid_rate": _rate(result.event_order_valid for result in results),
        "physical_sanity_pass_rate": _rate(
            result.physical_sanity_pass for result in results
        ),
        "placement_achieved_rate": _rate(result.placement_achieved for result in results),
        "mean_placement_xy_error": float(
            np.mean([result.placement_xy_error for result in results])
        )
        if results
        else 0.0,
        "reopen_events": sum(result.reopen_events for result in results),
    }


def default_trial_specs(repeats: int = 2) -> list[PickupTrialSpec]:
    orientations = [
        GraspOrientation("yaw_-18", -18.0),
        GraspOrientation("yaw_0", 0.0),
        GraspOrientation("yaw_18", 18.0),
    ]
    object_poses = [
        ObjectStartPose("center", np.array([0.000, -0.235, OBJECT_START_Z])),
        ObjectStartPose("left", np.array([-0.018, -0.232, OBJECT_START_Z])),
        ObjectStartPose("right", np.array([0.018, -0.238, OBJECT_START_Z])),
    ]
    approaches = [
        ApproachStrategy("vertical_pregrasp", "world_z"),
        ApproachStrategy("high_staged_vertical_pregrasp", "high_world_z"),
    ]
    specs: list[PickupTrialSpec] = []
    trial_id = 1
    for repeat in range(repeats):
        for orientation in orientations:
            for object_pose in object_poses:
                for approach in approaches:
                    specs.append(
                        PickupTrialSpec(
                            trial_id=trial_id,
                            orientation=orientation,
                            object_pose=object_pose,
                            approach=approach,
                            repeat=repeat,
                        )
                    )
                    trial_id += 1
    return specs


def summarize_results(results: Iterable[PickupTrialResult]) -> dict:
    results = list(results)
    return {
        "total": len(results),
        "successes": sum(result.success for result in results),
        "success_rate": _rate(result.success for result in results),
        "by_orientation": _bucket_rates(results, "orientation"),
        "by_object_pose": _bucket_rates(results, "object_pose"),
        "by_approach": _bucket_rates(results, "approach"),
        "failure_categories": _failure_counts(results),
        "collision_free_approach_rate": _rate(
            result.collision_free_approach for result in results
        ),
        "event_order_valid_rate": _rate(result.event_order_valid for result in results),
        "physical_sanity_pass_rate": _rate(
            result.physical_sanity_pass for result in results
        ),
        "early_close_trials": sum(result.early_close for result in results),
        "reopen_events": sum(result.reopen_events for result in results),
        "max_gripper_contact_force": max(
            (result.max_gripper_contact_force for result in results), default=0.0
        ),
        "max_gripper_contact_impulse_before_lift": max(
            (result.gripper_contact_impulse_before_lift for result in results),
            default=0.0,
        ),
        "max_object_xy_displacement_while_supported": max(
            (result.max_object_xy_displacement_while_supported for result in results),
            default=0.0,
        ),
        "preclose_contact_steps": sum(result.preclose_contact_steps for result in results),
        "max_preclose_object_displacement": max(
            (result.preclose_max_object_displacement for result in results),
            default=0.0,
        ),
    }


def _bucket_rates(
    results: list[PickupTrialResult] | list[PickPlaceTrialResult],
    attr: str,
) -> dict[str, dict[str, float | int]]:
    buckets = sorted({getattr(result, attr) for result in results})
    return {
        bucket: {
            "total": sum(getattr(result, attr) == bucket for result in results),
            "successes": sum(
                result.success for result in results if getattr(result, attr) == bucket
            ),
            "success_rate": _rate(
                result.success for result in results if getattr(result, attr) == bucket
            ),
        }
        for bucket in buckets
    }


def _failure_counts(
    results: list[PickupTrialResult] | list[PickPlaceTrialResult],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        counts[result.failure_category] = counts.get(result.failure_category, 0) + 1
    return counts


def _rate(values: Iterable[bool]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)


def _require_id(model: mujoco.MjModel, obj_type: mujoco.mjtObj, name: str) -> int:
    obj_id = mujoco.mj_name2id(model, obj_type, name)
    if obj_id < 0:
        raise ValueError(f"Missing MuJoCo object: {name}")
    return obj_id


def _orientation_error_rotvec(current_wxyz: np.ndarray, target_wxyz: np.ndarray) -> np.ndarray:
    current = _rotation_from_wxyz(current_wxyz)
    target = _rotation_from_wxyz(target_wxyz)
    return (target * current.inv()).as_rotvec()


def _rotation_from_wxyz(quat_wxyz: np.ndarray) -> Rotation:
    quat_wxyz = np.asarray(quat_wxyz, dtype=float)
    return Rotation.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])


def _wxyz_from_rotation(rotation: Rotation) -> np.ndarray:
    quat_xyzw = rotation.as_quat()
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])


def _round_list(values: np.ndarray) -> list[float]:
    return np.round(np.asarray(values, dtype=float), 6).tolist()
