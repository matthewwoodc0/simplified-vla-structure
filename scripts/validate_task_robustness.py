from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from svla.pickup_task import (
    BASE_OBJECT_HALF_SIZE,
    PickupTaskEvaluator,
    PickupTrialSpec,
    ObjectStartPose,
    SUPPORT_TOP_Z,
    default_trial_specs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Scenario:
    label: str
    size_scale: np.ndarray
    friction: float


BROAD_SCENARIOS = (
    Scenario("baseline", np.ones(3), 1.8),
    Scenario("uniform_small", np.full(3, 0.85), 1.8),
    Scenario("uniform_large", np.full(3, 1.15), 1.8),
    Scenario("wide_x", np.array([1.15, 1.0, 1.0]), 1.8),
    Scenario("wide_y", np.array([1.0, 1.15, 1.0]), 1.8),
    Scenario("short", np.array([1.0, 1.0, 0.85]), 1.8),
    Scenario("tall", np.array([1.0, 1.0, 1.15]), 1.8),
    Scenario("low_friction", np.ones(3), 0.8),
    Scenario("high_friction", np.ones(3), 2.4),
    Scenario("small_low_friction", np.full(3, 0.85), 0.8),
    Scenario("large_low_friction", np.full(3, 1.15), 0.8),
    Scenario("anisotropic", np.array([1.15, 0.85, 1.10]), 1.2),
)

READINESS_SCENARIOS = (
    Scenario("baseline", np.ones(3), 1.8),
    Scenario("uniform_small", np.full(3, 0.95), 1.8),
    Scenario("uniform_large", np.full(3, 1.05), 1.8),
    Scenario("narrow_x", np.array([0.95, 1.0, 1.0]), 1.8),
    Scenario("wide_x", np.array([1.05, 1.0, 1.0]), 1.8),
    Scenario("narrow_y", np.array([1.0, 0.95, 1.0]), 1.8),
    Scenario("wide_y", np.array([1.0, 1.05, 1.0]), 1.8),
    Scenario("short", np.array([1.0, 1.0, 0.95]), 1.8),
    Scenario("tall", np.array([1.0, 1.0, 1.05]), 1.8),
    Scenario("low_friction", np.ones(3), 1.6),
    Scenario("high_friction", np.ones(3), 2.0),
    Scenario("small_low_friction", np.full(3, 0.95), 1.6),
    Scenario("large_low_friction", np.full(3, 1.05), 1.6),
    Scenario("anisotropic", np.array([1.05, 0.95, 1.05]), 1.7),
)


def _run_trial(
    spec: PickupTrialSpec,
    half_size: np.ndarray,
    friction: float,
    cohort: str,
    scenario: str,
) -> dict:
    env = PickupTaskEvaluator(
        object_half_size=half_size,
        object_sliding_friction=friction,
    )
    xyz = np.asarray(spec.object_pose.xyz, dtype=float).copy()
    xyz[2] = SUPPORT_TOP_Z + half_size[2]
    env.reset(xyz)
    settled_start = env.object_position.copy()
    commands, grasp_pos, _ = env.scripted_controller_commands(spec, settled_start)
    saturated_steps = 0
    controller_failure_steps = 0

    for command in commands:
        for _ in range(command.max_steps):
            _, metrics, status = env.step_controller_command(
                command.target_pos,
                command.target_quat_wxyz,
                command.gripper_open,
                substeps=4,
            )
            saturated_steps += int(status.saturated)
            controller_failure_steps += int(status.controller_failed)
            if (
                command.stop_on_pose_tolerance
                and status.position_error <= env.controller.limits.position_tolerance
                and status.rotation_error <= env.controller.limits.rotation_tolerance
            ):
                break

    success = bool(
        metrics["collision_free_approach"]
        and metrics["event_order_valid"]
        and metrics["physical_sanity_pass"]
        and metrics["contact_achieved"]
        and metrics["object_lifted"]
        and metrics["retained_during_hold"]
        and controller_failure_steps == 0
    )
    return {
        "cohort": cohort,
        "scenario": scenario,
        "trial_id": spec.trial_id,
        "orientation": spec.orientation.label,
        "object_pose": spec.object_pose.label,
        "approach": spec.approach.label,
        "half_size": half_size.tolist(),
        "friction": float(friction),
        "success": success,
        "collision_free_approach": bool(metrics["collision_free_approach"]),
        "event_order_valid": bool(metrics["event_order_valid"]),
        "physical_sanity_pass": bool(metrics["physical_sanity_pass"]),
        "contact_achieved": bool(metrics["contact_achieved"]),
        "object_lifted": bool(metrics["object_lifted"]),
        "retained_during_hold": bool(metrics["retained_during_hold"]),
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
        "grasp_target_error": float(np.linalg.norm(grasp_pos - settled_start)),
        "saturated_steps": saturated_steps,
        "controller_failure_steps": controller_failure_steps,
    }


def _random_specs(
    count: int,
    seed: int,
    scale_low: float,
    scale_high: float,
    friction_low: float,
    friction_high: float,
) -> list[tuple[PickupTrialSpec, np.ndarray, float]]:
    rng = np.random.default_rng(seed)
    templates = default_trial_specs(repeats=1)
    cases = []
    for index in range(count):
        template = templates[index % len(templates)]
        x = float(rng.uniform(-0.018, 0.018))
        y = float(-0.235 - x / 6.0)
        scales = rng.uniform(scale_low, scale_high, size=3)
        half_size = BASE_OBJECT_HALF_SIZE * scales
        friction = float(rng.uniform(friction_low, friction_high))
        spec = PickupTrialSpec(
            trial_id=10_000 + index,
            orientation=template.orientation,
            object_pose=ObjectStartPose(
                f"random_{index:03d}",
                np.array([x, y, SUPPORT_TOP_Z + half_size[2]]),
            ),
            approach=template.approach,
            repeat=0,
        )
        cases.append((spec, half_size, friction))
    return cases


def run(args: argparse.Namespace) -> dict:
    if args.domain == "readiness":
        scenarios = READINESS_SCENARIOS
        random_bounds = (0.95, 1.05, 1.6, 2.0)
    else:
        scenarios = BROAD_SCENARIOS
        random_bounds = (0.85, 1.15, 0.8, 2.4)
    trials = []
    templates = default_trial_specs(repeats=1)
    for scenario in scenarios:
        half_size = BASE_OBJECT_HALF_SIZE * scenario.size_scale
        for template in templates:
            trials.append(
                _run_trial(
                    template,
                    half_size,
                    scenario.friction,
                    cohort="fixed",
                    scenario=scenario.label,
                )
            )

    for spec, half_size, friction in _random_specs(
        args.random_cases,
        args.seed,
        *random_bounds,
    ):
        trials.append(
            _run_trial(
                spec,
                half_size,
                friction,
                cohort="random",
                scenario="seeded_random",
            )
        )

    failures = [trial for trial in trials if not trial["success"]]
    summary = {
        "format": "svla_task_robustness_v2",
        "domain": args.domain,
        "random_bounds": {
            "size_scale": list(random_bounds[:2]),
            "sliding_friction": list(random_bounds[2:]),
        },
        "seed": args.seed,
        "pass": not failures,
        "total": len(trials),
        "successes": len(trials) - len(failures),
        "fixed_total": sum(trial["cohort"] == "fixed" for trial in trials),
        "fixed_successes": sum(
            trial["cohort"] == "fixed" and trial["success"] for trial in trials
        ),
        "random_total": sum(trial["cohort"] == "random" for trial in trials),
        "random_successes": sum(
            trial["cohort"] == "random" and trial["success"] for trial in trials
        ),
        "collision_free_approaches": sum(
            trial["collision_free_approach"] for trial in trials
        ),
        "valid_event_orders": sum(trial["event_order_valid"] for trial in trials),
        "physical_sanity_passes": sum(
            trial["physical_sanity_pass"] for trial in trials
        ),
        "early_close_trials": sum(trial["early_close"] for trial in trials),
        "reopen_events": sum(trial["reopen_events"] for trial in trials),
        "max_gripper_contact_force": max(
            trial["max_gripper_contact_force"] for trial in trials
        ),
        "max_gripper_contact_impulse_before_lift": max(
            trial["gripper_contact_impulse_before_lift"] for trial in trials
        ),
        "max_object_xy_displacement_while_supported": max(
            trial["max_object_xy_displacement_while_supported"] for trial in trials
        ),
        "max_object_rotation_while_supported": max(
            trial["max_object_rotation_while_supported"] for trial in trials
        ),
        "failures": failures,
        "trials": trials,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"task robustness: {summary['successes']}/{summary['total']} total, "
        f"fixed={summary['fixed_successes']}/{summary['fixed_total']}, "
        f"random={summary['random_successes']}/{summary['random_total']}, "
        f"max_force={summary['max_gripper_contact_force']:.3f}N, "
        f"max_supported_shift="
        f"{summary['max_object_xy_displacement_while_supported'] * 1000:.3f}mm"
    )
    print(f"wrote {args.output}")
    if not summary["pass"]:
        raise SystemExit(1)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stress pickup geometry, friction, event order, force, and disturbance."
    )
    parser.add_argument("--random-cases", type=int, default=36)
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--domain", choices=("readiness", "broad"), default="readiness")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "task_robustness_summary.json",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
