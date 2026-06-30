from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from svla.controller import CartesianCommand, CartesianIKController


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "assets" / "so101_scene.xml"


class ArmSim:
    """Small MuJoCo wrapper for controller bring-up."""

    def __init__(self, model_path: Path | str = DEFAULT_MODEL_PATH) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.controller = CartesianIKController(self.model)
        self.reset()

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        neutral = np.array([0.0, -1.57, 1.57, 1.57, -1.57])
        self.data.qpos[self.controller.arm_qpos_ids] = neutral
        self.data.ctrl[self.controller.arm_actuator_ids] = neutral
        self.controller.set_gripper(self.data, 0.85)
        mujoco.mj_forward(self.model, self.data)
        self.controller.reset_target(self.data, posture_target=neutral)

    def step(self, command: CartesianCommand | None = None, substeps: int = 10):
        if command is not None:
            status = self.controller.apply_delta(self.data, command)
        else:
            status = None
        for _ in range(substeps):
            mujoco.mj_step(self.model, self.data)
        return status

    def move_to(self, target_xyz: np.ndarray, max_steps: int = 600, substeps: int = 5) -> float:
        final_error = float("inf")
        for _ in range(max_steps):
            status = self.controller.move_toward(self.data, target_xyz)
            for _ in range(substeps):
                mujoco.mj_step(self.model, self.data)
            final_error = status.position_error
            if final_error <= self.controller.limits.position_tolerance:
                break
        return final_error

    @property
    def ee_position(self) -> np.ndarray:
        return self.controller.ee_pose(self.data)[0]

    @property
    def ee_quat_wxyz(self) -> np.ndarray:
        return self.controller.ee_pose(self.data)[1]

    def set_target_marker(self, xyz: np.ndarray) -> None:
        self.data.mocap_pos[0] = np.asarray(xyz, dtype=float)
