from __future__ import annotations

from pathlib import Path
import sys
import time

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from svla.sim import ArmSim
from svla.teleop_controller import TeleopRates, TeleopTargetController
from svla.teleop_inputs import TeleopInputManager, teleop_help_text
from svla.teleop_workspace import build_workspace_bounds


# Preset reach offsets in world frame from the pose at session start.
TARGET_DELTAS = {
    "1": np.array([0.03, 0.04, 0.02]),
    "2": np.array([0.04, 0.08, 0.00]),
    "3": np.array([-0.03, -0.05, -0.02]),
    "4": np.array([0.05, 0.00, 0.03]),
}


def main() -> None:
    import mujoco.viewer

    rng = np.random.default_rng(11)
    sim = ArmSim()
    workspace = build_workspace_bounds(sim.model, sim.data, sim.controller)
    teleop_inputs = TeleopInputManager(TeleopRates())
    teleop = TeleopTargetController(
        initial_position=sim.ee_position,
        initial_quat_wxyz=sim.ee_quat_wxyz,
        workspace=workspace,
        gripper_open=sim.controller.gripper_open_fraction(sim.data),
    )
    session_base = sim.ee_position.copy()
    sim.set_target_marker(teleop.state.position)
    sim.controller.set_gripper(sim.data, teleop.state.gripper_open)

    print(teleop_help_text())
    print(teleop_inputs.status_line())
    print(
        "workspace center (null joints FK): "
        f"{np.round(workspace.center, 3).tolist()} "
        f"half_extents={np.round(workspace.half_extents, 3).tolist()}"
    )

    def key_callback(key: int) -> None:
        teleop_inputs.on_key(key)

    with mujoco.viewer.launch_passive(
        sim.model,
        sim.data,
        key_callback=key_callback,
        show_left_ui=True,
        show_right_ui=True,
    ) as viewer:
        last_print = time.monotonic()
        while viewer.is_running():
            intent = teleop_inputs.poll()

            if intent.reset:
                sim.reset()
                teleop.reset_to(
                    sim.ee_position,
                    sim.ee_quat_wxyz,
                    sim.controller.gripper_open_fraction(sim.data),
                )
                session_base = sim.ee_position.copy()
                sim.set_target_marker(teleop.state.position)
                print("reset requested")

            if intent.show_help:
                print(teleop_help_text())

            if intent.random_target:
                offset = rng.uniform([-0.06, -0.08, -0.03], [0.06, 0.08, 0.03])
                teleop.state.position = sim.ee_position + offset
                teleop.state.quat_wxyz = sim.ee_quat_wxyz.copy()
                sim.set_target_marker(teleop.state.position)
                teleop.state.controller_enabled = True
                print(f"random target: {np.round(teleop.state.position, 3).tolist()}")

            if intent.preset_target in TARGET_DELTAS:
                teleop.state.position = session_base + TARGET_DELTAS[intent.preset_target]
                teleop.state.quat_wxyz = sim.ee_quat_wxyz.copy()
                sim.set_target_marker(teleop.state.position)
                teleop.state.controller_enabled = True
                print(
                    f"preset {intent.preset_target}: "
                    f"{np.round(teleop.state.position, 3).tolist()}"
                )

            step = teleop.apply_intent(intent)
            sim.set_target_marker(step.target_position)

            if teleop.state.controller_enabled:
                sim.controller.move_toward(
                    sim.data,
                    step.target_position,
                    step.target_quat_wxyz,
                )
                sim.controller.set_gripper(sim.data, step.gripper_open)

            mujoco.mj_step(sim.model, sim.data)
            viewer.sync()

            now = time.monotonic()
            if now - last_print > 2.0:
                error = float(np.linalg.norm(step.target_position - sim.ee_position))
                blocked = step.blocked_axes
                print(
                    f"{teleop_inputs.status_line()} "
                    f"ee={np.round(sim.ee_position, 3).tolist()} "
                    f"target={np.round(step.target_position, 3).tolist()} "
                    f"error={error:.4f}m gripper={step.gripper_open:.2f} "
                    f"blocked_axes={blocked}"
                )
                last_print = now
            time.sleep(sim.model.opt.timestep)

    teleop_inputs.close()


if __name__ == "__main__":
    main()