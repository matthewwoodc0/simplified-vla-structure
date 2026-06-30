from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from svla.sim import ArmSim


TARGET_DELTAS = (
    np.array([-0.04, 0.05, 0.03]),
    np.array([-0.03, 0.09, 0.00]),
    np.array([0.04, -0.07, -0.04]),
    np.array([-0.06, 0.02, 0.01]),
)


def run_headless() -> None:
    sim = ArmSim()
    targets = [sim.ee_position + delta for delta in TARGET_DELTAS]
    for index, target in enumerate(targets, start=1):
        sim.set_target_marker(target)
        error = sim.move_to(target)
        print(
            f"target {index}: target={np.round(target, 3).tolist()} "
            f"ee={np.round(sim.ee_position, 3).tolist()} error={error:.4f}m"
        )


def run_viewer() -> None:
    import mujoco.viewer

    sim = ArmSim()
    targets = [sim.ee_position + delta for delta in TARGET_DELTAS]
    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        while viewer.is_running():
            for target in targets:
                sim.set_target_marker(target)
                for _ in range(240):
                    sim.move_to(target, max_steps=1)
                    viewer.sync()
                    time.sleep(sim.model.opt.timestep * 5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--viewer", action="store_true", help="Open MuJoCo's interactive viewer.")
    args = parser.parse_args()
    if args.viewer:
        run_viewer()
    else:
        run_headless()


if __name__ == "__main__":
    main()
