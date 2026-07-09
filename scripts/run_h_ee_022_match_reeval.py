#!/usr/bin/env python3
"""H-EE-022: re-evaluate frozen hybrid A1 models under a named match contract.

One causal change: NN match features only (same MLP/NN weights as H-EE-014).
Protocol-v2 validation, both action spaces, seeds 0-4.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from svla.evaluation_protocol import load_evaluation_protocol
from svla.pickup_task import PickupTaskEvaluator
from svla.state_bc import (
    ACTION_SPACES,
    DEFAULT_MATCH_CONTRACT,
    HYBRID_POLICY_TYPE,
    MATCH_CONTRACT_NAMES,
    MATCH_CONTRACT_RELATIVE_EE,
    HybridNNGripperMLPPolicy,
    TaskContext,
    load_policy,
    resolve_match_contract,
    rollout_policy,
    summarize_policy_results,
    write_json,
)

# Frozen hybrid baseline metrics (H-EE-014 global_gripper validation).
BASELINE_EE = {
    "successes": 62,
    "event_order": 79,
    "phys": 68,
    "reopen": 0,
    "worst_seed": 9,
    "early_close": 11,
    "missing_lift_eo": 29,
}
BASELINE_JOINT = {
    "successes": 97,
    "event_order": 107,
    "phys": 103,
    "worst_seed": 15,
    "early_close": 5,
}


def _count_missing_lift_eo(records: list[dict]) -> int:
    n = 0
    for r in records:
        if r.get("success"):
            continue
        if (
            r.get("failure_category") == "event_order_failure"
            and r.get("contact_achieved")
            and not r.get("object_lifted")
            and not r.get("early_close")
        ):
            n += 1
    return n


def _metrics_from_summary(summary: dict, records: list[dict]) -> dict:
    # Count closed-loop gates from records (summarize exposes rates, not counts).
    per_seed: dict[int, int] = {}
    for r in records:
        seed = int(r["seed"])
        per_seed[seed] = per_seed.get(seed, 0) + int(bool(r.get("success")))
    per_seed_list = [per_seed[s] for s in sorted(per_seed)]
    worst = min(per_seed_list) if per_seed_list else 0
    early = sum(1 for r in records if r.get("early_close"))
    reopen = sum(int(r.get("reopen_events") or 0) for r in records)
    event_order = sum(1 for r in records if r.get("event_order_valid"))
    phys = sum(1 for r in records if r.get("physical_sanity_pass"))
    successes = sum(1 for r in records if r.get("success"))
    return {
        "successes": int(successes),
        "total": int(len(records)),
        "event_order": int(event_order),
        "phys": int(phys),
        "reopen": reopen,
        "worst_seed": int(worst),
        "early_close": int(early),
        "missing_lift_eo": _count_missing_lift_eo(records),
        "per_seed_successes": per_seed_list,
        "failure_categories": summary.get("failure_categories", {}),
        "preclose_contact_steps": int(summary.get("preclose_contact_steps", 0) or 0),
    }


def _pass_bars_ee(ee: dict) -> dict[str, bool]:
    early_bar = ee["early_close"] <= 5 or ee["early_close"] <= 0.5 * BASELINE_EE["early_close"]
    return {
        "early_close": early_bar,
        "reopen": ee["reopen"] <= 5,
        "success": ee["successes"] >= BASELINE_EE["successes"] - 3,
        "worst_seed": ee["worst_seed"] >= 8,
    }


def evaluate_models(
    model_dir: Path,
    match_contract: str,
    *,
    max_steps: int,
    search_window: int,
    action_gain: float,
    eval_limit: int | None,
    seeds: list[int],
) -> dict:
    protocol = load_evaluation_protocol()
    protocol_metadata = protocol.metadata("validation")
    eval_specs = protocol.specs("validation", repeats=1)
    if eval_limit is not None:
        eval_specs = eval_specs[:eval_limit]
    match_indices, match_names = resolve_match_contract(match_contract)

    all_results = []
    per_policy = {}
    for action_space in ACTION_SPACES:
        policy_results = []
        records = []
        for seed in seeds:
            model_path = (
                model_dir / f"{action_space}_{HYBRID_POLICY_TYPE}_seed_{seed}.json"
            )
            if not model_path.exists():
                raise FileNotFoundError(f"missing hybrid model: {model_path}")
            policy = load_policy(model_path)
            if not isinstance(policy, HybridNNGripperMLPPolicy):
                raise TypeError(f"expected hybrid policy at {model_path}")
            policy.set_match_contract(match_contract)
            # Ensure hash matches protocol
            policy.evaluation_config_hash = str(protocol_metadata["config_sha256"])
            policy.evaluation_protocol_version = int(protocol_metadata["version"])

            env = PickupTaskEvaluator()
            available = set(policy.group_keys)
            for spec in eval_specs:
                context = TaskContext.from_spec(spec)
                if context.key not in available:
                    raise RuntimeError(
                        f"missing context {context.key} for {action_space} seed {seed}"
                    )
                result = rollout_policy(
                    env,
                    policy,
                    spec,
                    max_steps=max_steps,
                    search_window=search_window,
                    action_gain=action_gain,
                )
                policy_results.append(result)
                all_results.append(result)
                record = result.to_dict()
                record["seed"] = int(seed)
                record["match_contract"] = match_contract
                record["evaluation_protocol"] = protocol_metadata
                records.append(record)
                print(
                    f"policy={action_space} seed={seed} trial={result.trial_id} "
                    f"success={int(result.success)} fail={result.failure_category} "
                    f"early={int(result.early_close)} lift={result.max_object_lift:.3f}"
                )

        summary = summarize_policy_results(policy_results)
        metrics = _metrics_from_summary(summary, records)
        per_policy[action_space] = {
            "summary": summary,
            "metrics": metrics,
            "records": records,
        }

    return {
        "match_contract": match_contract,
        "match_feature_indices": [int(i) for i in match_indices.tolist()],
        "match_feature_names": list(match_names),
        "protocol": protocol_metadata,
        "evaluated_spec_count": len(eval_specs),
        "seeds": list(seeds),
        "policies": per_policy,
        "combined_summary": summarize_policy_results(all_results),
    }


def build_comparison(eval_payload: dict, output_dir: Path, model_dir: Path) -> dict:
    ee = eval_payload["policies"]["ee_tool_delta"]["metrics"]
    joint = eval_payload["policies"]["joint_delta"]["metrics"]
    bars = _pass_bars_ee(ee)
    status = "confirmed" if all(bars.values()) else "rejected"
    # Partial: early_close improves materially but other bars fail
    if status == "rejected" and bars["early_close"] and bars["reopen"]:
        if ee["early_close"] < BASELINE_EE["early_close"]:
            status = "partial"

    comparison = {
        "format": "svla_h_ee_022_match_relative_ee_v1",
        "hypothesis": "H-EE-022",
        "status": status,
        "match_contract": eval_payload["match_contract"],
        "match_feature_indices": eval_payload["match_feature_indices"],
        "match_feature_names": eval_payload["match_feature_names"],
        "baseline_source": "outputs/h_ee_014_nn_gripper_global_validation/",
        "model_dir": str(model_dir),
        "output_dir": str(output_dir),
        "causal_change": (
            "Named NN match contract only (match_relative_ee); frozen hybrid A1 "
            "weights from H-EE-014; no MLP retrain; no arm-only; no shield/FSM."
        ),
        "baseline_ee": BASELINE_EE,
        "baseline_joint": BASELINE_JOINT,
        "ee": ee,
        "joint": joint,
        "deltas_ee": {
            "successes": ee["successes"] - BASELINE_EE["successes"],
            "event_order": ee["event_order"] - BASELINE_EE["event_order"],
            "phys": ee["phys"] - BASELINE_EE["phys"],
            "reopen": ee["reopen"] - BASELINE_EE["reopen"],
            "worst_seed": ee["worst_seed"] - BASELINE_EE["worst_seed"],
            "early_close": ee["early_close"] - BASELINE_EE["early_close"],
            "missing_lift_eo": ee["missing_lift_eo"] - BASELINE_EE["missing_lift_eo"],
        },
        "pass_bars": bars,
        "bars_all_met": all(bars.values()),
        "historical_match_retained_as_default": True,
        "notes": (
            "Pass bars (EE vs hybrid baseline): early_close ≤5 or ≤−50% rel; "
            "reopen ≤5; success ≥59; worst seed ≥8."
        ),
    }
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "h_ee_014_nn_gripper_global_validation"
        / "models",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "h_ee_022_match_relative_ee_validation",
    )
    parser.add_argument(
        "--match-contract",
        choices=MATCH_CONTRACT_NAMES,
        default=MATCH_CONTRACT_RELATIVE_EE,
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--max-steps", type=int, default=3200)
    parser.add_argument("--search-window", type=int, default=5)
    parser.add_argument("--action-gain", type=float, default=1.0)
    parser.add_argument("--eval-limit", type=int, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = output_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    payload = evaluate_models(
        args.model_dir,
        args.match_contract,
        max_steps=args.max_steps,
        search_window=args.search_window,
        action_gain=args.action_gain,
        eval_limit=args.eval_limit,
        seeds=list(args.seeds),
    )

    for action_space, block in payload["policies"].items():
        jsonl_path = eval_dir / f"{action_space}_policy_trials.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as handle:
            for record in block["records"]:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        write_json(
            eval_dir / f"{action_space}_policy_trials.summary.json",
            {**block["summary"], "metrics": block["metrics"]},
        )
        # Drop records from payload for compact summary dump
        block.pop("records", None)

    comparison = build_comparison(payload, output_dir, args.model_dir)
    write_json(output_dir / "h_ee_022_comparison.json", comparison)
    write_json(output_dir / "state_bc_summary.json", {
        "format": "svla_h_ee_022_reeval_summary_v1",
        "match_contract": payload["match_contract"],
        "policies": {
            name: {"metrics": block["metrics"], "summary": block["summary"]}
            for name, block in payload["policies"].items()
        },
        "comparison_status": comparison["status"],
    })
    print(json.dumps(comparison, indent=2, sort_keys=True))
    print(f"status={comparison['status']} wrote {output_dir / 'h_ee_022_comparison.json'}")


if __name__ == "__main__":
    main()
