#!/usr/bin/env python3
"""Run H-EE-015: frozen hybrid EE arm + oracle gripper FSM upper-bound diagnostic.

Registered order: register -> smoke -> evaluate -> finalize.
No training path. Final holdout closed. Joint is not re-evaluated.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from svla.evaluation_protocol import load_evaluation_protocol
from svla.h_ee_015 import (
    ACTION_SPACE,
    EXPECTED_BASELINE_PRIMARY,
    GRIPPER_SOURCE,
    HYPOTHESIS,
    JOINT_HYBRID_REFERENCE,
    SEEDS,
    TOTAL_TRIALS,
    OracleFsmHybridPolicy,
    align_paired_rows,
    assert_finalize_artifact_hashes,
    build_paired_comparison,
    build_registration,
    classify_verdict,
    load_jsonl,
    model_manifest_name,
    primary_metrics,
    reproduction_check,
    sha256_file,
    summarize_baseline_rows,
    summarize_rows,
    verify_frozen_inputs,
)
from svla.pickup_task import PickupTaskEvaluator
from svla.state_bc import (
    HybridNNGripperMLPPolicy,
    TaskContext,
    load_policy,
    rollout_policy,
    write_json,
)


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "h_ee_015_fsm_upper_bound"
DEFAULT_BASELINE_DIR = (
    PROJECT_ROOT / "outputs" / "h_ee_014_nn_gripper_global_validation"
)


def _registration_path(output_dir: Path) -> Path:
    return output_dir / "h_ee_015_registration.json"


def _load_registration(output_dir: Path) -> dict[str, Any]:
    path = _registration_path(output_dir)
    if not path.is_file():
        raise RuntimeError("registration must be written before any evaluation")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _baseline_jsonl(baseline_dir: Path) -> Path:
    return baseline_dir / "eval" / f"{ACTION_SPACE}_policy_trials.jsonl"


def register(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    baseline_dir = Path(args.baseline_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reg_path = _registration_path(output_dir)
    if reg_path.exists() and not args.force:
        raise RuntimeError(
            f"registration already exists at {reg_path}; "
            "refusing to overwrite (pass --force only before efficacy)"
        )

    protocol = load_evaluation_protocol()
    metadata = protocol.metadata("validation")
    frozen = verify_frozen_inputs(
        baseline_dir,
        protocol_hash=str(metadata["config_sha256"]),
        protocol_version=int(metadata["version"]),
        source_dir=baseline_dir,
    )
    baseline_rows = load_jsonl(_baseline_jsonl(baseline_dir))
    if len(baseline_rows) != TOTAL_TRIALS:
        raise RuntimeError(
            f"expected {TOTAL_TRIALS} baseline rows, got {len(baseline_rows)}"
        )
    baseline_metrics = summarize_baseline_rows(baseline_rows)
    reproduction = reproduction_check(baseline_metrics)
    if not reproduction["exact_primary_counts"]:
        raise RuntimeError(
            f"baseline does not match frozen H-EE-014 primary counts: "
            f"{reproduction['differences']}"
        )

    source_manifest = baseline_dir / "state_bc_summary.manifest.json"
    if not source_manifest.is_file():
        raise FileNotFoundError(source_manifest)

    registration = build_registration(
        protocol_metadata=metadata,
        frozen_verification=frozen,
        baseline_metrics=baseline_metrics,
        paired_keys=baseline_metrics["paired_keys"],
        source_manifest_sha256=sha256_file(source_manifest),
        max_steps=int(args.max_steps),
        search_window=int(args.search_window),
        action_gain=float(args.action_gain),
    )
    write_json(reg_path, registration)
    write_json(
        output_dir / "h_ee_015_frozen_inputs_manifest.json",
        {
            "format": "svla_h_ee_015_frozen_inputs_v1",
            "hypothesis": HYPOTHESIS,
            "baseline_dir": str(baseline_dir.resolve()),
            "verification": frozen,
            "baseline_primary": primary_metrics(baseline_metrics),
            "reproduction": reproduction,
            "final_accessed": False,
        },
    )
    print(json.dumps({"wrote": str(reg_path), "reproduction": reproduction}, indent=2))


def evaluate_rows(
    baseline_dir: Path,
    *,
    seeds: list[int],
    eval_limit: int | None,
    max_steps: int,
    search_window: int,
    action_gain: float,
    protocol_metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    protocol = load_evaluation_protocol()
    eval_specs = protocol.specs("validation", repeats=1)
    if eval_limit is not None:
        eval_specs = eval_specs[: int(eval_limit)]

    rows: list[dict[str, Any]] = []
    for seed in seeds:
        model_path = baseline_dir / "models" / model_manifest_name(seed)
        hybrid = load_policy(model_path)
        if not isinstance(hybrid, HybridNNGripperMLPPolicy):
            raise TypeError(f"expected hybrid policy at {model_path}")
        hybrid.evaluation_config_hash = str(protocol_metadata["config_sha256"])
        hybrid.evaluation_protocol_version = int(protocol_metadata["version"])
        policy = OracleFsmHybridPolicy(hybrid)
        available = set(policy.group_keys)
        env = PickupTaskEvaluator()
        for spec in eval_specs:
            context = TaskContext.from_spec(spec)
            if context.key not in available:
                raise RuntimeError(
                    f"missing context {context.key} for seed {seed}"
                )
            # Fresh FSM per trial.
            policy = OracleFsmHybridPolicy(hybrid)
            result = rollout_policy(
                env,
                policy,
                spec,
                max_steps=max_steps,
                search_window=search_window,
                action_gain=action_gain,
            )
            row = result.to_dict()
            fsm_tel = policy.fsm.telemetry()
            row.update(
                {
                    "seed": int(seed),
                    "action_gain": float(action_gain),
                    "evaluation_protocol": protocol_metadata,
                    "evaluation_config_hash": protocol_metadata["config_sha256"],
                    "hypothesis": HYPOTHESIS,
                    "inference_only": True,
                    "gripper_source": GRIPPER_SOURCE,
                    "oracle_diagnostic": True,
                    "not_learned_policy_performance": True,
                    "final_accessed": False,
                    "training_performed": False,
                    **fsm_tel,
                }
            )
            rows.append(row)
            print(
                f"seed={seed} trial={result.trial_id} "
                f"success={int(result.success)} eo={int(result.event_order_valid)} "
                f"phys={int(result.physical_sanity_pass)} fail={result.failure_category} "
                f"fsm_trans={fsm_tel['fsm_transition_step']} "
                f"never={int(fsm_tel['fsm_never_transitioned'])} "
                f"early={int(result.early_close)} lift={result.max_object_lift:.3f}",
                flush=True,
            )
    return rows


def smoke(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    baseline_dir = Path(args.baseline_dir)
    registration = _load_registration(output_dir)
    protocol = load_evaluation_protocol()
    metadata = protocol.metadata("validation")
    if registration["protocol"]["config_sha256"] != metadata["config_sha256"]:
        raise RuntimeError("protocol hash differs from registration")

    rows = evaluate_rows(
        baseline_dir,
        seeds=[SEEDS[0]],
        eval_limit=1,
        max_steps=int(registration["frozen_baseline"]["max_steps"]),
        search_window=int(registration["frozen_baseline"]["search_window"]),
        action_gain=float(registration["frozen_baseline"]["action_gain"]),
        protocol_metadata=metadata,
    )
    if len(rows) != 1:
        raise RuntimeError(f"smoke expected 1 row, got {len(rows)}")
    row = rows[0]
    if row.get("gripper_source") != GRIPPER_SOURCE:
        raise RuntimeError("smoke missing oracle gripper_source")
    if row.get("oracle_diagnostic") is not True:
        raise RuntimeError("smoke missing oracle_diagnostic")
    payload = {
        "format": "svla_h_ee_015_smoke_v1",
        "hypothesis": HYPOTHESIS,
        "purpose": "plumbing only; one validation trial with oracle FSM",
        "row": row,
        "fsm_transition_step": row.get("fsm_transition_step"),
        "fsm_never_transitioned": row.get("fsm_never_transitioned"),
        "fsm_state_final": row.get("fsm_state_final"),
        "gripper_source": GRIPPER_SOURCE,
        "oracle_diagnostic": True,
        "final_accessed": False,
    }
    write_json(output_dir / "h_ee_015_smoke.json", payload)
    print(json.dumps({k: payload[k] for k in payload if k != "row"}, indent=2))
    print(
        "smoke fsm telemetry: "
        f"state={row.get('fsm_state_final')} "
        f"transition_step={row.get('fsm_transition_step')} "
        f"pos_err={row.get('fsm_transition_pos_error')} "
        f"rot_err={row.get('fsm_transition_rot_error')} "
        f"grip_dist={row.get('fsm_transition_gripper_object_distance')}"
    )


def evaluate(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    baseline_dir = Path(args.baseline_dir)
    registration = _load_registration(output_dir)
    smoke_path = output_dir / "h_ee_015_smoke.json"
    if not smoke_path.is_file():
        raise RuntimeError("smoke must complete before full evaluation")

    trials_path = output_dir / "h_ee_015_trials.jsonl"
    summary_path = output_dir / "h_ee_015_summary.json"
    if (trials_path.exists() or summary_path.exists()) and not args.force:
        raise RuntimeError(
            "refusing to overwrite completed evaluation; pass --force only if intentional"
        )

    protocol = load_evaluation_protocol()
    metadata = protocol.metadata("validation")
    if registration["protocol"]["config_sha256"] != metadata["config_sha256"]:
        raise RuntimeError("protocol hash differs from registration")

    rows = evaluate_rows(
        baseline_dir,
        seeds=list(SEEDS),
        eval_limit=None,
        max_steps=int(registration["frozen_baseline"]["max_steps"]),
        search_window=int(registration["frozen_baseline"]["search_window"]),
        action_gain=float(registration["frozen_baseline"]["action_gain"]),
        protocol_metadata=metadata,
    )
    if len(rows) != TOTAL_TRIALS:
        raise RuntimeError(f"expected {TOTAL_TRIALS} rows, got {len(rows)}")

    # Enforce registered paired keys exactly.
    registered_keys = {
        (int(s), int(t)) for s, t in registration["paired_validation_keys"]
    }
    actual_keys = {(int(r["seed"]), int(r["trial_id"])) for r in rows}
    if actual_keys != registered_keys:
        raise RuntimeError(
            f"evaluated keys differ from registration: "
            f"missing={sorted(registered_keys - actual_keys)[:5]} "
            f"extra={sorted(actual_keys - registered_keys)[:5]}"
        )

    metrics = summarize_rows(rows)
    baseline_metrics = registration["frozen_baseline"]["baseline_metrics"]
    verdict = classify_verdict(metrics, baseline_metrics)

    _write_jsonl(trials_path, rows)
    summary = {
        "format": "svla_h_ee_015_summary_v1",
        "hypothesis": HYPOTHESIS,
        "gripper_source": GRIPPER_SOURCE,
        "oracle_diagnostic": True,
        "not_learned_policy_performance": True,
        "metrics": metrics,
        "baseline_metrics": baseline_metrics,
        "expected_baseline_primary": EXPECTED_BASELINE_PRIMARY,
        "verdict": verdict,
        "registration_sha256": sha256_file(_registration_path(output_dir)),
        "protocol": registration["protocol"],
        "joint_hybrid_reference": JOINT_HYBRID_REFERENCE,
        "training_performed": False,
        "final_accessed": False,
        "phase_6b_started": False,
    }
    write_json(summary_path, summary)
    print(
        json.dumps(
            {
                "metrics": {
                    k: metrics[k]
                    for k in (
                        "successes",
                        "event_order_valid",
                        "physical_sanity_pass",
                        "per_seed_successes",
                        "worst_seed",
                        "missing_lift_eo",
                        "early_close",
                        "reopen_events",
                        "never_transitioned",
                        "controller_failures",
                    )
                },
                "verdict": verdict["status"],
            },
            indent=2,
        )
    )


def finalize(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    baseline_dir = Path(args.baseline_dir)
    registration = _load_registration(output_dir)
    trials_path = output_dir / "h_ee_015_trials.jsonl"
    summary_path = output_dir / "h_ee_015_summary.json"
    if not trials_path.is_file() or not summary_path.is_file():
        raise RuntimeError("evaluate must complete before finalize")

    rows = load_jsonl(trials_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if len(rows) != TOTAL_TRIALS:
        raise RuntimeError(f"expected {TOTAL_TRIALS} trial rows, got {len(rows)}")
    metrics = summarize_rows(rows)
    # Recompute verdict from rows; must match summary.
    baseline_metrics = registration["frozen_baseline"]["baseline_metrics"]
    verdict = classify_verdict(metrics, baseline_metrics)
    if summary["verdict"]["status"] != verdict["status"]:
        raise RuntimeError(
            f"summary verdict {summary['verdict']['status']!r} != recomputed {verdict['status']!r}"
        )

    baseline_rows = load_jsonl(_baseline_jsonl(baseline_dir))
    align_paired_rows(baseline_rows, rows)
    paired = build_paired_comparison(baseline_rows, rows)
    write_json(output_dir / "h_ee_015_paired_comparison.json", paired)

    # Write the final summary BEFORE hashing it into the experiment manifest.
    # Hashing pre-refresh would leave summary_sha256 mismatched with the file.
    summary["paired"] = {
        "new_successes": paired["new_successes"],
        "lost_successes": paired["lost_successes"],
        "net_success_change": paired["net_success_change"],
        "new_successes_from_baseline_missing_lift": paired[
            "new_successes_from_baseline_missing_lift"
        ],
    }
    summary["verdict"] = verdict
    summary["metrics"] = metrics
    write_json(summary_path, summary)

    # Experiment manifest with all hashes and oracle flags (post-final-summary).
    reg_sha = sha256_file(_registration_path(output_dir))
    trials_sha = sha256_file(trials_path)
    summary_sha = sha256_file(summary_path)
    paired_sha = sha256_file(output_dir / "h_ee_015_paired_comparison.json")
    source_manifest = baseline_dir / "state_bc_summary.manifest.json"

    manifest = {
        "format": "svla_h_ee_015_experiment_manifest_v1",
        "hypothesis": HYPOTHESIS,
        "gripper_source": GRIPPER_SOURCE,
        "oracle_diagnostic": True,
        "not_learned_policy_performance": True,
        "final_accessed": False,
        "training_performed": False,
        "joint_reevaluated": False,
        "phase_6b_started": False,
        "protocol": registration["protocol"],
        "registration_sha256": reg_sha,
        "trials_sha256": trials_sha,
        "summary_sha256": summary_sha,
        "paired_comparison_sha256": paired_sha,
        "source_manifest_sha256": sha256_file(source_manifest),
        "frozen_models": registration["frozen_baseline"]["loaded_models"],
        "fsm_contract": registration["fsm_contract"],
        "verdict": verdict,
        "metrics": metrics,
        "baseline_primary": EXPECTED_BASELINE_PRIMARY,
        "paired_summary": {
            "new_successes": paired["new_successes"],
            "lost_successes": paired["lost_successes"],
            "net_success_change": paired["net_success_change"],
            "new_successes_from_baseline_missing_lift": paired[
                "new_successes_from_baseline_missing_lift"
            ],
        },
        "joint_hybrid_reference": JOINT_HYBRID_REFERENCE,
        "artifacts": {
            "registration": "h_ee_015_registration.json",
            "trials": "h_ee_015_trials.jsonl",
            "summary": "h_ee_015_summary.json",
            "paired_comparison": "h_ee_015_paired_comparison.json",
            "smoke": "h_ee_015_smoke.json",
            "frozen_inputs_manifest": "h_ee_015_frozen_inputs_manifest.json",
        },
    }
    write_json(output_dir / "h_ee_015_experiment_manifest.json", manifest)

    # Hard consistency gate: every recorded hash must match the on-disk file.
    assert_finalize_artifact_hashes(output_dir)

    print(
        json.dumps(
            {
                "verdict": verdict["status"],
                "successes": metrics["successes"],
                "event_order_valid": metrics["event_order_valid"],
                "physical_sanity_pass": metrics["physical_sanity_pass"],
                "worst_seed": metrics["worst_seed"],
                "never_transitioned": metrics["never_transitioned"],
                "paired": summary["paired"],
                "final_accessed": False,
                "summary_sha256": summary_sha,
            },
            indent=2,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="experiment output directory",
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=DEFAULT_BASELINE_DIR,
        help="frozen H-EE-014 hybrid validation directory",
    )
    parser.add_argument("--max-steps", type=int, default=3200)
    parser.add_argument("--search-window", type=int, default=120)
    parser.add_argument("--action-gain", type=float, default=1.0)
    parser.add_argument(
        "--force",
        action="store_true",
        help="allow overwrite of registration/eval artifacts (pre-efficacy only)",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("register", help="freeze hashes, keys, FSM, verdict bars")
    sub.add_parser("smoke", help="one-trial non-efficacy plumbing check")
    sub.add_parser("evaluate", help="full 5×24 protocol-v2 validation")
    sub.add_parser("finalize", help="paired comparison + experiment manifest")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if float(args.action_gain) != 1.0:
        raise ValueError("H-EE-015 requires action_gain=1.0 (frozen baseline)")
    command = args.command
    if command == "register":
        register(args)
    elif command == "smoke":
        smoke(args)
    elif command == "evaluate":
        evaluate(args)
    elif command == "finalize":
        finalize(args)
    else:
        raise ValueError(f"unknown command: {command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
