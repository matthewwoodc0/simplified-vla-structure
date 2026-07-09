#!/usr/bin/env python3
"""H-EE-014 post-run comparison + diagnosis vs H-EE-021 global_gripper MLP baseline.

Reads a completed hybrid validation output dir and writes:
  - h_ee_014_comparison.json  (pre-registered pass bars + deltas)
  - h_ee_014_diagnosis.json   (reopen/flip anatomy, EO failure buckets)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from svla.state_bc import write_json  # noqa: E402

FORMAT_VERSION = "svla_h_ee_014_nn_gripper_v1"

# Frozen H-EE-021 global_gripper pure-MLP baseline (protocol-v2 validation).
# Pass bars are defined against these numbers in the plan.
DEFAULT_BASELINE_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "h_ee_021_loss_decomposition"
    / "h_ee_021_comparison.json"
)
DEFAULT_BASELINE_PROFILE = "global_gripper"

# Pre-registered pass bars (plan §3). Written into comparison before interpreting.
PASS_BARS = {
    "ee_success_delta_min": 10,
    "ee_event_order_delta_min": 12,
    "ee_reopen_relative_reduction_min": 0.20,
    "ee_worst_seed_delta_min": 3,
    "ee_worst_seed_absolute_min": 8,
    "joint_success_floor_delta": -10,
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number) or math.isinf(number):
        return default
    return number


def extract_metrics_from_summary(summary: dict[str, Any], action_space: str) -> dict[str, Any]:
    policy = summary["policies"][action_space]
    seed_summaries = policy.get("seed_summaries") or []
    per_seed = []
    for seed_summary in seed_summaries:
        total = _safe_int(seed_summary.get("total", 0))
        per_seed.append(
            {
                "seed": _safe_int(seed_summary.get("seed", -1)),
                "successes": _safe_int(seed_summary.get("successes", 0)),
                "total": total,
                "event_order_valid": int(
                    round(
                        _safe_float(seed_summary.get("event_order_valid_rate", 0.0)) * total
                    )
                ),
                "physical_sanity_pass": int(
                    round(
                        _safe_float(seed_summary.get("physical_sanity_pass_rate", 0.0))
                        * total
                    )
                ),
            }
        )
    total = _safe_int(policy.get("total", 0))
    successes = _safe_int(policy.get("successes", 0))
    event_order = int(
        round(_safe_float(policy.get("event_order_valid_rate", 0.0)) * total)
    )
    physical = int(
        round(_safe_float(policy.get("physical_sanity_pass_rate", 0.0)) * total)
    )
    worst_seed = min((row["successes"] for row in per_seed), default=0)
    return {
        "successes": successes,
        "total": total,
        "success_rate": _safe_float(policy.get("success_rate", 0.0)),
        "event_order_valid": event_order,
        "event_order_valid_rate": _safe_float(policy.get("event_order_valid_rate", 0.0)),
        "physical_sanity_pass": physical,
        "physical_sanity_pass_rate": _safe_float(
            policy.get("physical_sanity_pass_rate", 0.0)
        ),
        "early_close_trials": _safe_int(policy.get("early_close_trials", 0)),
        "preclose_contact_steps": _safe_int(policy.get("preclose_contact_steps", 0)),
        "reopen_events": _safe_int(policy.get("reopen_events", 0)),
        "controller_failure_steps": _safe_int(policy.get("controller_failure_steps", 0)),
        "mean_joint_limit_clipped_steps": _safe_float(
            policy.get("mean_joint_limit_clipped_steps", 0.0)
        ),
        "mean_infeasible_steps": _safe_float(policy.get("mean_infeasible_steps", 0.0)),
        "failure_categories": dict(policy.get("failure_categories") or {}),
        "per_seed_successes": [row["successes"] for row in per_seed],
        "worst_seed_successes": int(worst_seed),
        "seed_rows": per_seed,
        "shielded_policy": bool(policy.get("shielded_policy", False)),
    }


def extract_baseline_profile(
    comparison: dict[str, Any], profile: str, action_space: str
) -> dict[str, Any]:
    profiles = comparison.get("profiles") or {}
    if profile not in profiles:
        raise KeyError(f"baseline profile {profile!r} missing from comparison")
    block = profiles[profile]
    if action_space not in block:
        raise KeyError(f"action space {action_space!r} missing from baseline profile {profile}")
    return dict(block[action_space])


def load_trial_rows(jsonl_path: Path) -> list[dict[str, Any]]:
    if not jsonl_path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def diagnose_trials(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "trial_count": 0,
            "mean_gripper_command_flips": 0.0,
            "success_mean_flips": 0.0,
            "fail_mean_flips": 0.0,
            "fail_with_reopen": 0,
            "eo_failure_buckets": {},
            "per_seed": {},
        }

    successes = [r for r in rows if r.get("success")]
    fails = [r for r in rows if not r.get("success")]
    flips = [_safe_int(r.get("gripper_command_flips", 0)) for r in rows]
    success_flips = [_safe_int(r.get("gripper_command_flips", 0)) for r in successes]
    fail_flips = [_safe_int(r.get("gripper_command_flips", 0)) for r in fails]
    fail_with_reopen = sum(
        1 for r in fails if _safe_int(r.get("reopen_events", 0)) > 0
    )

    # EO failure anatomy among event_order_valid=False trials.
    eo_fails = [r for r in rows if not r.get("event_order_valid")]
    buckets: Counter[str] = Counter()
    for r in eo_fails:
        early = bool(r.get("early_close"))
        reopen = _safe_int(r.get("reopen_events", 0)) > 0
        contact = bool(r.get("contact_achieved"))
        lifted = bool(r.get("object_lifted"))
        if early and reopen:
            buckets["early_close_and_reopen"] += 1
        elif early:
            buckets["early_close"] += 1
        elif reopen:
            buckets["reopen_only"] += 1
        elif not contact:
            buckets["missing_contact"] += 1
        elif not lifted:
            buckets["missing_lift"] += 1
        else:
            buckets["other_event_order"] += 1

    per_seed: dict[str, dict[str, Any]] = {}
    by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_seed[_safe_int(r.get("seed", -1))].append(r)
    for seed, seed_rows in sorted(by_seed.items()):
        seed_success = [r for r in seed_rows if r.get("success")]
        seed_fail = [r for r in seed_rows if not r.get("success")]
        per_seed[str(seed)] = {
            "total": len(seed_rows),
            "successes": len(seed_success),
            "reopen_events": sum(_safe_int(r.get("reopen_events", 0)) for r in seed_rows),
            "early_close_trials": sum(1 for r in seed_rows if r.get("early_close")),
            "event_order_valid": sum(1 for r in seed_rows if r.get("event_order_valid")),
            "physical_sanity_pass": sum(
                1 for r in seed_rows if r.get("physical_sanity_pass")
            ),
            "mean_gripper_command_flips": (
                sum(_safe_int(r.get("gripper_command_flips", 0)) for r in seed_rows)
                / max(1, len(seed_rows))
            ),
            "success_mean_flips": (
                sum(_safe_int(r.get("gripper_command_flips", 0)) for r in seed_success)
                / max(1, len(seed_success))
                if seed_success
                else 0.0
            ),
            "fail_mean_flips": (
                sum(_safe_int(r.get("gripper_command_flips", 0)) for r in seed_fail)
                / max(1, len(seed_fail))
                if seed_fail
                else 0.0
            ),
            "fail_with_reopen": sum(
                1 for r in seed_fail if _safe_int(r.get("reopen_events", 0)) > 0
            ),
        }

    return {
        "trial_count": len(rows),
        "mean_gripper_command_flips": sum(flips) / len(flips),
        "success_mean_flips": (
            sum(success_flips) / len(success_flips) if success_flips else 0.0
        ),
        "fail_mean_flips": sum(fail_flips) / len(fail_flips) if fail_flips else 0.0,
        "fail_with_reopen": fail_with_reopen,
        "fail_with_reopen_rate": fail_with_reopen / max(1, len(fails)),
        "total_reopen_events": sum(_safe_int(r.get("reopen_events", 0)) for r in rows),
        "eo_failure_buckets": dict(buckets),
        "per_seed": per_seed,
    }


def evaluate_pass_bars(
    hybrid_ee: dict[str, Any],
    baseline_ee: dict[str, Any],
    hybrid_joint: dict[str, Any] | None,
    baseline_joint: dict[str, Any] | None,
) -> dict[str, Any]:
    ee_success_delta = hybrid_ee["successes"] - baseline_ee["successes"]
    ee_eo_delta = hybrid_ee["event_order_valid"] - baseline_ee["event_order_valid"]
    base_reopen = max(1, _safe_int(baseline_ee.get("reopen_events", 0)))
    hybrid_reopen = _safe_int(hybrid_ee.get("reopen_events", 0))
    reopen_relative_reduction = (base_reopen - hybrid_reopen) / float(base_reopen)
    ee_worst_delta = (
        hybrid_ee["worst_seed_successes"] - baseline_ee["worst_seed_successes"]
    )
    worst_seed = hybrid_ee["worst_seed_successes"]

    success_pass = ee_success_delta >= PASS_BARS["ee_success_delta_min"]
    eo_pass = ee_eo_delta >= PASS_BARS["ee_event_order_delta_min"]
    reopen_pass = reopen_relative_reduction >= PASS_BARS["ee_reopen_relative_reduction_min"]
    worst_pass = (
        ee_worst_delta >= PASS_BARS["ee_worst_seed_delta_min"]
        or worst_seed >= PASS_BARS["ee_worst_seed_absolute_min"]
    )

    joint_pass = True
    joint_success_delta = None
    if hybrid_joint is not None and baseline_joint is not None:
        joint_success_delta = hybrid_joint["successes"] - baseline_joint["successes"]
        joint_pass = joint_success_delta >= PASS_BARS["joint_success_floor_delta"]

    confirmed = bool(success_pass and eo_pass and reopen_pass and worst_pass and joint_pass)
    # Selection rule: success + worst-seed improve materially AND reopen falls.
    selectable = bool(success_pass and worst_pass and reopen_pass and joint_pass)

    if confirmed:
        status = "confirmed_validation"
    elif success_pass or eo_pass or reopen_pass or worst_pass:
        status = "partial"
    else:
        status = "rejected"

    return {
        "pass_bars": PASS_BARS,
        "deltas": {
            "ee_success": ee_success_delta,
            "ee_event_order": ee_eo_delta,
            "ee_reopen_relative_reduction": reopen_relative_reduction,
            "ee_worst_seed": ee_worst_delta,
            "joint_success": joint_success_delta,
        },
        "checks": {
            "ee_success": success_pass,
            "ee_event_order": eo_pass,
            "ee_reopen": reopen_pass,
            "ee_worst_seed": worst_pass,
            "joint_non_collapse": joint_pass,
        },
        "confirmed_validation": confirmed,
        "selectable_for_final": selectable,
        "status": status,
        "note": (
            "Final holdout stays closed unless selection rule is met and human "
            "explicitly opens it. Train MSE is not evidence."
        ),
    }


def build_interpretation(pass_eval: dict[str, Any], diagnosis: dict[str, Any]) -> dict[str, Any]:
    status = pass_eval["status"]
    if status == "confirmed_validation":
        next_step = (
            "Freeze hybrid+global_gripper contract; only then consider final once "
            "with explicit human approval."
        )
        suggested = "freeze_and_consider_final"
    elif status == "partial":
        reopen_ok = pass_eval["checks"]["ee_reopen"]
        success_ok = pass_eval["checks"]["ee_success"]
        worst_ok = pass_eval["checks"]["ee_worst_seed"]
        if reopen_ok and not success_ok:
            next_step = (
                "Reopen fell but success barely moved — keep hybrid gripper; "
                "attack arm/lift (A2 or controller-side), not more loss weight."
            )
            suggested = "arm_lift_followup"
        elif success_ok and not worst_ok:
            next_step = (
                "Success up but worst seed still weak — do not open final; "
                "consider H-EE-017 history or seed-robustness."
            )
            suggested = "history_or_seed_robustness"
        else:
            next_step = (
                "Mixed signals — inspect diagnosis; prefer H-EE-017 (history) or "
                "H-EE-015 (FSM gripper) if reopens remain."
            )
            suggested = "inspect_diagnostics"
    else:
        next_step = (
            "State-local NN gripper did not fix reopen-dominated failures. "
            "Next: H-EE-017 (history) or H-EE-015 (FSM gripper + learned arm)."
        )
        suggested = "h_ee_017_or_015"

    ee_diag = diagnosis.get("ee_tool_delta") or {}
    return {
        "status": status,
        "suggested_next": suggested,
        "next_step": next_step,
        "residual_note": (
            f"EO buckets={ee_diag.get('eo_failure_buckets')}; "
            f"success_flips={ee_diag.get('success_mean_flips')}; "
            f"fail_flips={ee_diag.get('fail_mean_flips')}"
        ),
        "deprioritized_still": [
            "H-EE-016 oversampling",
            "H-EE-018 adaptive loss",
            "H-EE-019 transition weighting",
            "Lower EE gain",
            "Vision BC / Phase 6b",
            "Opening final on one seed",
        ],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    summary_path = output_dir / "state_bc_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"missing hybrid summary: {summary_path}")
    summary = _load_json(summary_path)
    baseline_comparison = _load_json(Path(args.baseline_comparison))
    profile = args.baseline_profile

    hybrid_ee = extract_metrics_from_summary(summary, "ee_tool_delta")
    hybrid_joint = extract_metrics_from_summary(summary, "joint_delta")
    baseline_ee = extract_baseline_profile(baseline_comparison, profile, "ee_tool_delta")
    baseline_joint = extract_baseline_profile(baseline_comparison, profile, "joint_delta")

    ee_rows = load_trial_rows(output_dir / "eval" / "ee_tool_delta_policy_trials.jsonl")
    joint_rows = load_trial_rows(output_dir / "eval" / "joint_delta_policy_trials.jsonl")
    diagnosis = {
        "ee_tool_delta": diagnose_trials(ee_rows),
        "joint_delta": diagnose_trials(joint_rows),
    }

    pass_eval = evaluate_pass_bars(hybrid_ee, baseline_ee, hybrid_joint, baseline_joint)
    interpretation = build_interpretation(pass_eval, diagnosis)

    comparison = {
        "format": FORMAT_VERSION,
        "hypothesis": "H-EE-014",
        "recipe": "A1_compositor",
        "loss_profile": summary.get("loss_profile") or profile,
        "loss_profile_contract": summary.get("loss_profile_contract"),
        "hybrid_nn_gripper": True,
        "policy_type": summary.get("policy_type"),
        "match_feature_names": summary.get("match_feature_names"),
        "match_feature_indices": summary.get("match_feature_indices"),
        "nn_k": summary.get("nn_k"),
        "nn_temperature": summary.get("nn_temperature"),
        "baseline": {
            "source": str(Path(args.baseline_comparison)),
            "profile": profile,
            "ee_tool_delta": baseline_ee,
            "joint_delta": baseline_joint,
        },
        "hybrid": {
            "source": str(summary_path),
            "ee_tool_delta": hybrid_ee,
            "joint_delta": hybrid_joint,
            "eval_split": summary.get("eval_split"),
            "temporal_feature_mode": summary.get("temporal_feature_mode"),
            "shielded_policy": summary.get("shielded_policy"),
        },
        "pass_evaluation": pass_eval,
        "interpretation": interpretation,
        "notes": [
            "Raw hybrid policy only; no guard/FSM/threshold/match-set change.",
            "Final holdout not accessed.",
            "Compared against H-EE-021 pure-MLP under the same loss profile.",
            "Constraint exposure is telemetry only.",
        ],
    }

    diagnosis_payload = {
        "format": FORMAT_VERSION + "_diagnosis",
        "hypothesis": "H-EE-014",
        "output_dir": str(output_dir),
        "baseline_profile": profile,
        "pass_evaluation": pass_eval,
        "diagnosis": diagnosis,
        "hybrid_metrics": {
            "ee_tool_delta": hybrid_ee,
            "joint_delta": hybrid_joint,
        },
        "baseline_metrics": {
            "ee_tool_delta": baseline_ee,
            "joint_delta": baseline_joint,
        },
    }

    comparison_path = output_dir / "h_ee_014_comparison.json"
    diagnosis_path = output_dir / "h_ee_014_diagnosis.json"
    write_json(comparison_path, comparison)
    write_json(diagnosis_path, diagnosis_payload)
    print(json.dumps(comparison, indent=2, sort_keys=True))
    print(f"wrote {comparison_path}")
    print(f"wrote {diagnosis_path}")
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "h_ee_014_nn_gripper_global_validation",
    )
    parser.add_argument(
        "--baseline-comparison",
        type=Path,
        default=DEFAULT_BASELINE_PATH,
        help="H-EE-021 comparison JSON providing frozen pure-MLP baselines.",
    )
    parser.add_argument(
        "--baseline-profile",
        default=DEFAULT_BASELINE_PROFILE,
        help="Profile key inside the baseline comparison (default: global_gripper).",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
