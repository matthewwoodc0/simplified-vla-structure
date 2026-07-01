from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from svla.controller import CartesianCommand, ControllerLimits
from svla.sim import ArmSim


def _print_policy_path(sim: ArmSim, steps: int = 1) -> None:
    limits = sim.controller.limits
    start_ee = sim.ee_position.copy()
    command = CartesianCommand(np.array([1.0, 1.0, 1.0]), np.zeros(3), 0.5)
    status = None
    for _ in range(steps):
        status = sim.step(command)
    telemetry = sim.controller.last_telemetry
    assert status is not None
    assert telemetry is not None

    print("Policy path (apply_delta) — what learned policies use")
    print(f"  limits: max_step_xyz={limits.max_step_xyz}, max_target_lag_xyz={limits.max_target_lag_xyz}")
    print(f"  raw command norm:        {np.linalg.norm(command.delta_xyz):.4f} m")
    print(f"  clipped_translation:     {status.clipped_translation}")
    print(f"  target ahead of EE:      {telemetry.position_error:.4f} m")
    print(f"  EE moved from start:     {np.linalg.norm(sim.ee_position - start_ee):.4f} m")
    print(f"  joint_step_norm:         {status.joint_step_norm:.4f} rad")
    print(f"  state still finite:      {np.isfinite(sim.data.qpos).all()}")
    print()


def _print_demo_path(sim: ArmSim) -> None:
    start_ee = sim.ee_position.copy()
    target = start_ee + np.array([-0.04, 0.05, 0.03])
    error = sim.move_to(target, max_steps=600)
    print("Demo path (move_to) — what run_reach_demo.py uses")
    print(f"  target offset norm:      {np.linalg.norm(target - start_ee):.4f} m")
    print(f"  final tracking error:    {error:.4f} m")
    print(f"  uses apply_delta:        no")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Contrast policy clipping (apply_delta) with reach-demo tracking (move_to)."
    )
    parser.add_argument(
        "--max-step-xyz",
        type=float,
        default=None,
        help="Override ControllerLimits.max_step_xyz for the policy-path experiment.",
    )
    parser.add_argument(
        "--max-target-lag-xyz",
        type=float,
        default=None,
        help="Override ControllerLimits.max_target_lag_xyz for the policy-path experiment.",
    )
    parser.add_argument(
        "--policy-steps",
        type=int,
        default=1,
        help="How many huge apply_delta commands to chain (default: 1).",
    )
    args = parser.parse_args()

    base_limits = ControllerLimits()
    overrides = {
        "max_step_xyz": args.max_step_xyz if args.max_step_xyz is not None else base_limits.max_step_xyz,
        "max_target_lag_xyz": (
            args.max_target_lag_xyz
            if args.max_target_lag_xyz is not None
            else base_limits.max_target_lag_xyz
        ),
    }
    limits = ControllerLimits(**overrides)

    policy_sim = ArmSim()
    policy_sim.controller.limits = limits
    _print_policy_path(policy_sim, steps=args.policy_steps)

    demo_sim = ArmSim()
    demo_sim.controller.limits = limits
    _print_demo_path(demo_sim)


if __name__ == "__main__":
    main()