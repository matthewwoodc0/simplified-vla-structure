#!/usr/bin/env python3
"""Audit, replay, register, and score the one-off H-EE-007 probe."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from svla.demo_recorder import PickupDemoRecorder  # noqa: E402
from svla.pick_place_replay import replay_demo_policy_labels  # noqa: E402
from svla.pickup_task import (  # noqa: E402
    MAX_GRIPPER_CONTACT_FORCE,
    MAX_GRIPPER_IMPULSE_BEFORE_LIFT,
    MAX_SUPPORTED_ROTATION,
    MAX_SUPPORTED_XY_DISPLACEMENT,
    PICKUP_CONTROLLER_LIMITS,
    PickupTaskEvaluator,
    default_trial_specs,
)
from svla.state_bc import PHASE_LABELS, write_json  # noqa: E402

FORMAT = "svla_h_ee_007_label_contract_probe_v1"
OUT = ROOT / "outputs" / "h_ee_007_label_contract_probe"
BASELINE = ROOT / "outputs" / "h_ee_014_nn_gripper_global_validation"
ARM_NAMES = ("delta_x", "delta_y", "delta_z", "tilt_x", "tilt_y")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dist(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def label_stats(raw: np.ndarray, policy: np.ndarray) -> dict:
    raw_arm, policy_arm = raw[:, :5], policy[:, :5]
    difference = np.abs(raw_arm - policy_arm)
    raw_norm = np.linalg.norm(raw_arm, axis=1)
    policy_norm = np.linalg.norm(policy_arm, axis=1)
    nonzero = (raw_norm > 1e-12) & (policy_norm > 1e-12)
    cosine = np.sum(raw_arm[nonzero] * policy_arm[nonzero], axis=1) / (
        raw_norm[nonzero] * policy_norm[nonzero]
    )

    def magnitudes(actions: np.ndarray) -> dict:
        xyz = np.linalg.norm(actions[:, :3], axis=1)
        rot = np.linalg.norm(actions[:, 3:5], axis=1)
        xyz_limit = PICKUP_CONTROLLER_LIMITS.max_step_xyz
        rot_limit = PICKUP_CONTROLLER_LIMITS.max_step_rot
        return {
            "arm_l2": dist(np.linalg.norm(actions, axis=1)),
            "translation_l2": dist(xyz),
            "rotation_l2": dist(rot),
            "translation_near_clip_fraction": float(np.mean(xyz >= 0.95 * xyz_limit)),
            "translation_at_or_over_clip_fraction": float(np.mean(xyz >= xyz_limit)),
            "rotation_near_clip_fraction": float(np.mean(rot >= 0.95 * rot_limit)),
            "rotation_at_or_over_clip_fraction": float(np.mean(rot >= rot_limit)),
        }

    gripper_diff = np.abs(raw[:, -1] - policy[:, -1])
    return {
        "sample_count": len(raw),
        "per_dimension_absolute_difference": {
            name: dist(difference[:, i]) for i, name in enumerate(ARM_NAMES)
        },
        "arm_vector_l2_difference": dist(np.linalg.norm(raw_arm - policy_arm, axis=1)),
        "cosine_agreement_nonzero": {
            "count": int(nonzero.sum()),
            "mean": float(np.mean(cosine)),
            "median": float(np.median(cosine)),
            "p05": float(np.percentile(cosine, 5)),
            "p95": float(np.percentile(cosine, 95)),
        },
        "action_magnitudes": {"raw": magnitudes(raw_arm), "policy_labels": magnitudes(policy_arm)},
        "gripper_label_equality": {
            "all_equal": bool(np.all(gripper_diff <= 1e-12)),
            "equal_count": int(np.sum(gripper_diff <= 1e-12)),
            "total": len(gripper_diff),
            "max_absolute_difference": float(np.max(gripper_diff)),
        },
    }


def audit(args: argparse.Namespace) -> None:
    demo_dir = args.baseline_dir / "scripted_pickup_demos"
    paths = sorted(demo_dir.glob("pickup_demo_*.json"))
    if len(paths) != 30:
        raise ValueError(f"expected 30 frozen H-EE-014 demos, found {len(paths)}")
    paired = {phase: ([], []) for phase in PHASE_LABELS}
    all_raw, all_policy = [], []
    for path in paths:
        for sample in load_json(path)["samples"]:
            raw = np.asarray(sample["labels"]["ee_tool_delta"], dtype=float)
            policy = np.asarray(sample["policy_labels"]["ee_tool_delta"], dtype=float)
            all_raw.append(raw)
            all_policy.append(policy)
            paired[sample["phase"]][0].append(raw)
            paired[sample["phase"]][1].append(policy)
    result = {
        "format": FORMAT,
        "phase": "A_label_audit",
        "baseline_dir": str(args.baseline_dir),
        "demo_count": len(paths),
        "demo_manifest_sha256": sha256(demo_dir / "manifest.json"),
        "all_samples": label_stats(np.vstack(all_raw), np.vstack(all_policy)),
        "by_phase": {
            phase: label_stats(np.vstack(raw), np.vstack(policy))
            for phase, (raw, policy) in paired.items()
        },
        "candidate_rollout_started": False,
    }
    write_json(args.output_dir / "h_ee_007_label_audit.json", result)


def replay_summary(trials: list[dict]) -> dict:
    result = {
        "total": len(trials),
        "successes": sum(row["success"] for row in trials),
        "valid_event_orders": sum(row["event_order_valid"] for row in trials),
        "physical_sanity_passes": sum(row["physical_sanity_pass"] for row in trials),
        "preclose_contact_steps": sum(row["preclose_contact_steps"] for row in trials),
        "reopen_events": sum(row["reopen_events"] for row in trials),
        "controller_failure_steps": sum(row["controller_failure_steps"] for row in trials),
        "saturated_steps": sum(row["saturated_steps"] for row in trials),
        "joint_limit_clipped_steps": sum(row["joint_limit_clipped_steps"] for row in trials),
        "infeasible_steps": sum(row["infeasible_steps"] for row in trials),
    }
    for name in (
        "max_gripper_contact_force",
        "gripper_contact_impulse_before_lift",
        "max_object_xy_displacement_while_supported",
        "max_object_rotation_while_supported",
    ):
        result[name] = max(row[name] for row in trials)
    result["trials"] = trials
    return result


def replay(args: argparse.Namespace) -> None:
    if not (args.output_dir / "h_ee_007_label_audit.json").is_file():
        raise RuntimeError("run the label audit before candidate replay")
    specs = default_trial_specs(repeats=1)
    demos = [(spec, PickupDemoRecorder(PickupTaskEvaluator()).record_trial(spec)) for spec in specs]
    by_source = {}
    for source in ("policy_labels", "labels"):
        by_source[source] = replay_summary([
            replay_demo_policy_labels(
                demo,
                "ee_tool_delta",
                np.asarray(spec.object_pose.xyz),
                label_source=source,
            )
            for spec, demo in demos
        ])
    raw, control = by_source["labels"], by_source["policy_labels"]
    limits = {
        "max_gripper_contact_force": MAX_GRIPPER_CONTACT_FORCE,
        "gripper_contact_impulse_before_lift": MAX_GRIPPER_IMPULSE_BEFORE_LIFT,
        "max_object_xy_displacement_while_supported": MAX_SUPPORTED_XY_DISPLACEMENT,
        "max_object_rotation_while_supported": MAX_SUPPORTED_ROTATION,
    }
    checks = {
        "raw_success_18_of_18": raw["successes"] == 18,
        "raw_event_order_18_of_18": raw["valid_event_orders"] == 18,
        "raw_physical_sanity_18_of_18": raw["physical_sanity_passes"] == 18,
        "raw_zero_controller_failures": raw["controller_failure_steps"] == 0,
        "raw_zero_preclose_and_reopen": raw["preclose_contact_steps"] == raw["reopen_events"] == 0,
        "control_still_passes": control["successes"] == control["valid_event_orders"] == control["physical_sanity_passes"] == 18,
        "raw_within_physics_limits": all(raw[name] <= limit for name, limit in limits.items()),
    }
    write_json(args.output_dir / "h_ee_007_replay_comparison.json", {
        "format": FORMAT,
        "phase": "B_replay_gate",
        "demo_count": len(demos),
        "same_demos_for_both_sources": True,
        "by_label_source": by_source,
        "gate_checks": checks,
        "replay_gate_pass": all(checks.values()),
        "next": "register" if all(checks.values()) else "stop_rejected",
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("audit", "replay"))
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--baseline-dir", type=Path, default=BASELINE)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.phase == "audit":
        audit(args)
    else:
        replay(args)


if __name__ == "__main__":
    main()
