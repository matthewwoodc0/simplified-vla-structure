#!/usr/bin/env python3
"""H-EE-021: causal loss-profile decomposition of H-EE-008.

Runs the frozen four-profile matrix under one commit:
  uniform | global_gripper | transition_gripper | combined_h_ee_008

Contract (must stay matched across profiles):
  - protocol-v2 validation
  - legacy_progress_phase temporal features
  - both action spaces
  - five seeds × 24 validation trials
  - raw learned MLP only (no guard / FSM / temporal feature change)
  - final holdout stays closed

Historical H-EE-008 is reused only when its manifest fully matches the frozen
source/data/config; otherwise combined_h_ee_008 is rerun.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from svla.loss_profiles import (  # noqa: E402
    LOSS_PROFILE_NAMES,
    RESEARCH_PARITY_FRONTIER,
    get_loss_profile,
)
from svla.state_bc import write_json  # noqa: E402


FORMAT_VERSION = "svla_h_ee_021_loss_decomposition_v1"

DEFAULT_PROFILES = list(LOSS_PROFILE_NAMES)

# Registered H-EE-008 validation hyperparams (must match report).
DEFAULT_TRAIN_ARGS = [
    "--evaluation-protocol",
    "v2",
    "--eval-split",
    "validation",
    "--temporal-feature-mode",
    "legacy_progress_phase",
    "--policy-type",
    "mlp",
    "--seeds",
    "0",
    "1",
    "2",
    "3",
    "4",
    "--hidden-sizes",
    "128",
    "128",
    "--epochs",
    "300",
    "--batch-size",
    "1024",
    "--learning-rate",
    "0.001",
    "--weight-decay",
    "1e-5",
    "--stride",
    "1",
    "--max-steps",
    "3200",
    "--action-gain",
    "1.0",
    "--label-source",
    "policy_labels",
]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def extract_primary_metrics(summary: dict[str, Any], action_space: str) -> dict[str, Any]:
    policy = summary["policies"][action_space]
    seed_summaries = policy.get("seed_summaries") or []
    per_seed = []
    for seed_summary in seed_summaries:
        per_seed.append(
            {
                "seed": int(seed_summary.get("seed", -1)),
                "successes": int(seed_summary.get("successes", 0)),
                "total": int(seed_summary.get("total", 0)),
                "event_order_valid": int(
                    round(
                        float(seed_summary.get("event_order_valid_rate", 0.0))
                        * float(seed_summary.get("total", 0))
                    )
                ),
                "physical_sanity_pass": int(
                    round(
                        float(seed_summary.get("physical_sanity_pass_rate", 0.0))
                        * float(seed_summary.get("total", 0))
                    )
                ),
            }
        )
    successes = int(policy.get("successes", 0))
    total = int(policy.get("total", 0))
    event_order = int(round(float(policy.get("event_order_valid_rate", 0.0)) * total))
    physical = int(round(float(policy.get("physical_sanity_pass_rate", 0.0)) * total))
    worst_seed = min((row["successes"] for row in per_seed), default=0)
    return {
        "successes": successes,
        "total": total,
        "success_rate": float(policy.get("success_rate", 0.0)),
        "event_order_valid": event_order,
        "event_order_valid_rate": float(policy.get("event_order_valid_rate", 0.0)),
        "physical_sanity_pass": physical,
        "physical_sanity_pass_rate": float(policy.get("physical_sanity_pass_rate", 0.0)),
        "early_close_trials": int(policy.get("early_close_trials", 0)),
        "preclose_contact_steps": int(policy.get("preclose_contact_steps", 0)),
        "reopen_events": int(policy.get("reopen_events", 0)),
        "controller_failure_steps": int(policy.get("controller_failure_steps", 0)),
        "mean_joint_limit_clipped_steps": float(
            policy.get("mean_joint_limit_clipped_steps", 0.0)
        ),
        "mean_infeasible_steps": float(policy.get("mean_infeasible_steps", 0.0)),
        "per_seed_successes": [row["successes"] for row in per_seed],
        "worst_seed_successes": int(worst_seed),
        "seed_rows": per_seed,
        "failure_categories": policy.get("failure_categories", {}),
        "shielded_policy": bool(policy.get("shielded_policy", False)),
    }


def load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def compact_rollout_diagnosis(
    rows: list[dict[str, Any]],
    *,
    action_space: str,
) -> dict[str, Any]:
    """Phase-3 compact diagnosis from enriched trial rows (no re-rollout)."""

    space_rows = [row for row in rows if row.get("action_space") == action_space]
    by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in space_rows:
        by_seed[int(row.get("seed", -1))].append(row)

    seed_diags = []
    for seed in sorted(by_seed):
        seed_rows = by_seed[seed]
        close_dists = [
            d
            for d in (_safe_float(row.get("close_start_distance")) for row in seed_rows)
            if d is not None
        ]
        flips = [int(row.get("gripper_command_flips", 0)) for row in seed_rows]
        reopens = [int(row.get("reopen_events", 0)) for row in seed_rows]
        joint_limits = [int(row.get("joint_limit_clipped_steps", 0)) for row in seed_rows]
        infeasible = [int(row.get("infeasible_steps", 0)) for row in seed_rows]
        steps = [max(1, int(row.get("steps", 1))) for row in seed_rows]
        failure_counts = Counter(str(row.get("failure_category", "unknown")) for row in seed_rows)
        event_times = []
        for row in seed_rows:
            event_times.append(
                {
                    "trial_id": int(row.get("trial_id", -1)),
                    "success": bool(row.get("success", False)),
                    "failure_category": row.get("failure_category"),
                    "close_start_distance": _safe_float(row.get("close_start_distance")),
                    "first_close_time": _safe_float(row.get("first_close_time")),
                    "first_contact_time": _safe_float(row.get("first_contact_time")),
                    "first_unsupported_time": _safe_float(row.get("first_unsupported_time")),
                    "first_lift_time": _safe_float(row.get("first_lift_time")),
                    "early_close": bool(row.get("early_close", False)),
                    "reopen_events": int(row.get("reopen_events", 0)),
                    "gripper_command_flips": int(row.get("gripper_command_flips", 0)),
                    "preclose_contact_steps": int(row.get("preclose_contact_steps", 0)),
                    "joint_limit_rate": float(row.get("joint_limit_clipped_steps", 0))
                    / float(max(1, row.get("steps", 1))),
                    "infeasible_rate": float(row.get("infeasible_steps", 0))
                    / float(max(1, row.get("steps", 1))),
                }
            )
        seed_diags.append(
            {
                "seed": seed,
                "successes": sum(bool(row.get("success")) for row in seed_rows),
                "total": len(seed_rows),
                "event_order_valid": sum(
                    bool(row.get("event_order_valid")) for row in seed_rows
                ),
                "physical_sanity_pass": sum(
                    bool(row.get("physical_sanity_pass")) for row in seed_rows
                ),
                "early_close_trials": sum(bool(row.get("early_close")) for row in seed_rows),
                "mean_close_start_distance": _mean(close_dists),
                "median_close_start_distance": (
                    float(sorted(close_dists)[len(close_dists) // 2]) if close_dists else None
                ),
                "mean_gripper_command_flips": _mean([float(v) for v in flips]),
                "total_reopen_events": int(sum(reopens)),
                "mean_joint_limit_rate": _mean(
                    [jl / st for jl, st in zip(joint_limits, steps)]
                ),
                "mean_infeasible_rate": _mean(
                    [inf / st for inf, st in zip(infeasible, steps)]
                ),
                "failure_categories": dict(sorted(failure_counts.items())),
                "trials": event_times,
            }
        )
    return {
        "action_space": action_space,
        "trial_count": len(space_rows),
        "seeds": seed_diags,
    }


def extract_supervised_phase_loss(summary: dict[str, Any], action_space: str) -> dict[str, Any]:
    """Pull per-phase arm/gripper train residuals from the first seed (same demos)."""

    policy = summary["policies"][action_space]
    trainings = policy.get("training_summaries") or []
    if not trainings:
        return {}
    first = trainings[0]
    return {
        "seed": first.get("seed"),
        "loss_profile": first.get("loss_profile"),
        "phase_sample_counts": first.get("phase_sample_counts"),
        "per_phase_action_loss": first.get("per_phase_action_loss"),
        "train_mse_normalized": first.get("train_mse_normalized"),
        "train_mse_weighted": first.get("train_mse_weighted"),
        "gripper_loss_weight": first.get("gripper_loss_weight"),
        "close_phase_gripper_weight": first.get("close_phase_gripper_weight"),
    }


def profile_output_dir(root: Path, profile: str) -> Path:
    return root / f"profile_{profile}"


def run_profile(
    *,
    profile: str,
    output_root: Path,
    python: Path,
    dry_run: bool,
    epochs: int | None,
    eval_limit: int | None,
) -> dict[str, Any]:
    out_dir = profile_output_dir(output_root, profile)
    summary_path = out_dir / "state_bc_summary.json"
    argv = [
        str(python),
        str(PROJECT_ROOT / "scripts" / "train_state_bc.py"),
        "--output-dir",
        str(out_dir),
        *DEFAULT_TRAIN_ARGS,
        "--loss-profile",
        profile,
    ]
    if epochs is not None:
        # Replace default epochs value after the flag.
        if "--epochs" in argv:
            idx = argv.index("--epochs")
            argv[idx + 1] = str(epochs)
        else:
            argv.extend(["--epochs", str(epochs)])
    if eval_limit is not None:
        argv.extend(["--eval-limit", str(eval_limit)])

    meta = {
        "profile": profile,
        "loss_profile_contract": get_loss_profile(profile).to_dict(),
        "output_dir": str(out_dir),
        "command": argv,
        "summary_path": str(summary_path),
    }
    if dry_run:
        meta["status"] = "dry_run"
        return meta

    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_root / f"profile_{profile}_run.log"
    print(f"[H-EE-021] running profile={profile} -> {out_dir}", flush=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    with log_path.open("w", encoding="utf-8") as log_handle:
        completed = subprocess.run(
            argv,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    meta["log_path"] = str(log_path)
    meta["returncode"] = int(completed.returncode)
    if completed.returncode != 0:
        meta["status"] = "failed"
        return meta
    if not summary_path.exists():
        meta["status"] = "missing_summary"
        return meta
    summary = _load_json(summary_path)
    rows = []
    for action_space in ("ee_tool_delta", "joint_delta"):
        rows.extend(load_jsonl_rows(out_dir / "eval" / f"{action_space}_policy_trials.jsonl"))
    meta["status"] = "ok"
    meta["metrics"] = {
        "ee_tool_delta": extract_primary_metrics(summary, "ee_tool_delta"),
        "joint_delta": extract_primary_metrics(summary, "joint_delta"),
    }
    meta["supervised_phase_loss"] = {
        "ee_tool_delta": extract_supervised_phase_loss(summary, "ee_tool_delta"),
        "joint_delta": extract_supervised_phase_loss(summary, "joint_delta"),
    }
    meta["rollout_diagnosis"] = {
        "ee_tool_delta": compact_rollout_diagnosis(rows, action_space="ee_tool_delta"),
        "joint_delta": compact_rollout_diagnosis(rows, action_space="joint_delta"),
    }
    meta["evaluation_config_hash"] = summary.get("evaluation_config_hash")
    meta["shielded_policy"] = bool(summary.get("shielded_policy", False))
    return meta


def compare_to_frontier(metrics: dict[str, Any]) -> dict[str, Any]:
    frontier = RESEARCH_PARITY_FRONTIER
    return {
        "frontier": frontier,
        "success_gap": int(frontier["successes"]) - int(metrics["successes"]),
        "event_order_gap": int(frontier["event_order_valid"])
        - int(metrics["event_order_valid"]),
        "physical_sanity_gap": int(frontier["physical_sanity_pass"])
        - int(metrics["physical_sanity_pass"]),
        "worst_seed_gap": int(frontier["worst_seed_successes_min"])
        - int(metrics["worst_seed_successes"]),
        "meets_aggregate_success": int(metrics["successes"]) >= int(frontier["successes"]),
        "meets_worst_seed": int(metrics["worst_seed_successes"])
        >= int(frontier["worst_seed_successes_min"]),
    }


def interpret_matrix(profile_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Map Phase-2 evidence to the proposed next hypothesis table."""

    def ee_success(name: str) -> int | None:
        block = profile_results.get(name) or {}
        metrics = block.get("metrics") or {}
        ee = metrics.get("ee_tool_delta") or {}
        if "successes" not in ee:
            return None
        return int(ee["successes"])

    uniform = ee_success("uniform")
    global_g = ee_success("global_gripper")
    transition = ee_success("transition_gripper")
    combined = ee_success("combined_h_ee_008")
    if None in (uniform, global_g, transition, combined):
        return {
            "status": "incomplete",
            "note": "Not all profiles finished; interpretation deferred.",
        }

    global_delta = global_g - uniform
    transition_delta = transition - uniform
    combined_delta = combined - uniform
    interaction = combined - uniform - (global_g - uniform) - (transition - uniform)

    if global_delta >= 10 and transition_delta < 5:
        proposal = "H-EE-018"
        rationale = (
            "Global 5× helps while transition-only does not; test adaptive "
            "gripper-gradient balancing for seed reliability."
        )
    elif transition_delta >= 10 and global_delta < 5:
        proposal = "H-EE-019"
        rationale = (
            "Transition 10× drives most of the gain; test a narrow close-boundary "
            "curriculum without overweighting the long closed/lift segment."
        )
    elif combined_delta > max(global_delta, transition_delta) + 5:
        proposal = "H-EE-008_combined_retains"
        rationale = (
            "Combined 5×/10× is super-additive; keep combined as the loss baseline "
            "and pursue architecture/history fixes rather than scalar reweight alone."
        )
    else:
        proposal = "inspect_diagnostics"
        rationale = (
            "No clean single-factor win; use rollout diagnosis (close distance, "
            "reopen, failure categories) to choose H-EE-014 / H-EE-017 / H-EE-020."
        )

    return {
        "status": "complete",
        "ee_successes": {
            "uniform": uniform,
            "global_gripper": global_g,
            "transition_gripper": transition,
            "combined_h_ee_008": combined,
        },
        "ee_deltas_vs_uniform": {
            "global_gripper": global_delta,
            "transition_gripper": transition_delta,
            "combined_h_ee_008": combined_delta,
            "interaction_combined_minus_main_effects": interaction,
        },
        "suggested_next_proposal": proposal,
        "rationale": rationale,
        "selection_rule": (
            "Select one EE contract only if it materially improves aggregate success "
            "AND raises worst-seed success; then freeze and run final holdout once."
        ),
        "deprioritized": [
            "H-EE-016 (naive close oversampling mostly repeats transition 10×)",
            "Lower EE gain (saturation not supported as the simple cause)",
        ],
    }


def build_comparison(profile_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    table: dict[str, Any] = {}
    for name, block in profile_results.items():
        if block.get("status") != "ok":
            table[name] = {"status": block.get("status"), "returncode": block.get("returncode")}
            continue
        metrics = block["metrics"]
        table[name] = {
            "status": "ok",
            "loss_profile_contract": block.get("loss_profile_contract"),
            "evaluation_config_hash": block.get("evaluation_config_hash"),
            "ee_tool_delta": metrics["ee_tool_delta"],
            "joint_delta": metrics["joint_delta"],
            "ee_vs_frontier": compare_to_frontier(metrics["ee_tool_delta"]),
            "joint_vs_frontier": compare_to_frontier(metrics["joint_delta"]),
            "supervised_phase_loss": block.get("supervised_phase_loss"),
        }
    return {
        "format": FORMAT_VERSION,
        "hypothesis": "H-EE-021",
        "research_parity_frontier": RESEARCH_PARITY_FRONTIER,
        "profiles": table,
        "interpretation": interpret_matrix(profile_results),
        "notes": [
            "Raw learned policy only; no guard/FSM/threshold/temporal change.",
            "Final holdout not accessed.",
            "Constraint exposure is explanatory telemetry, not a substitute for success.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "h_ee_021_loss_decomposition",
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=LOSS_PROFILE_NAMES,
        default=DEFAULT_PROFILES,
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=PROJECT_ROOT / ".venv" / "bin" / "python",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands without training.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override epochs (diagnostic smoke only; full evidence uses 300).",
    )
    parser.add_argument(
        "--eval-limit",
        type=int,
        default=None,
        help="Diagnostic-only eval prefix; recorded and not valid release evidence.",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Rebuild comparison JSON from existing profile summaries without retraining.",
    )
    args = parser.parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    profile_results: dict[str, dict[str, Any]] = {}
    for profile in args.profiles:
        if args.aggregate_only:
            summary_path = profile_output_dir(output_dir, profile) / "state_bc_summary.json"
            if not summary_path.exists():
                profile_results[profile] = {
                    "profile": profile,
                    "status": "missing_summary",
                    "summary_path": str(summary_path),
                }
                continue
            summary = _load_json(summary_path)
            out_dir = summary_path.parent
            rows = []
            for action_space in ("ee_tool_delta", "joint_delta"):
                rows.extend(
                    load_jsonl_rows(out_dir / "eval" / f"{action_space}_policy_trials.jsonl")
                )
            profile_results[profile] = {
                "profile": profile,
                "status": "ok",
                "loss_profile_contract": get_loss_profile(profile).to_dict(),
                "output_dir": str(out_dir),
                "summary_path": str(summary_path),
                "metrics": {
                    "ee_tool_delta": extract_primary_metrics(summary, "ee_tool_delta"),
                    "joint_delta": extract_primary_metrics(summary, "joint_delta"),
                },
                "supervised_phase_loss": {
                    "ee_tool_delta": extract_supervised_phase_loss(summary, "ee_tool_delta"),
                    "joint_delta": extract_supervised_phase_loss(summary, "joint_delta"),
                },
                "rollout_diagnosis": {
                    "ee_tool_delta": compact_rollout_diagnosis(
                        rows, action_space="ee_tool_delta"
                    ),
                    "joint_delta": compact_rollout_diagnosis(
                        rows, action_space="joint_delta"
                    ),
                },
                "evaluation_config_hash": summary.get("evaluation_config_hash"),
                "shielded_policy": bool(summary.get("shielded_policy", False)),
            }
        else:
            profile_results[profile] = run_profile(
                profile=profile,
                output_root=output_dir,
                python=args.python,
                dry_run=args.dry_run,
                epochs=args.epochs,
                eval_limit=args.eval_limit,
            )

    comparison = build_comparison(profile_results)
    comparison_path = output_dir / "h_ee_021_comparison.json"
    diagnosis_path = output_dir / "h_ee_021_rollout_diagnosis.json"
    write_json(comparison_path, comparison)
    write_json(
        diagnosis_path,
        {
            "format": FORMAT_VERSION,
            "hypothesis": "H-EE-021",
            "profiles": {
                name: block.get("rollout_diagnosis")
                for name, block in profile_results.items()
                if block.get("status") == "ok"
            },
        },
    )
    results_path = output_dir / "h_ee_021_profile_results.json"
    write_json(results_path, profile_results)
    print(json.dumps(comparison, indent=2, sort_keys=True))
    print(f"wrote {comparison_path}")
    print(f"wrote {diagnosis_path}")
    print(f"wrote {results_path}")


if __name__ == "__main__":
    main()
