#!/usr/bin/env python3
"""Run H-EE-002: frozen H-EE-014 hybrid EE validation at three fixed gains.

The CLI enforces the registered order: register -> smoke -> gain 1.0 control ->
gain 0.875/0.750 -> finalize.  It contains no training path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from svla.evaluation_protocol import load_evaluation_protocol
from svla.h_ee_002 import (
    ACTION_SPACE,
    EXPECTED_BASELINE_PRIMARY,
    GAINS,
    HYPOTHESIS,
    PASS_BARS,
    SEEDS,
    TOTAL_TRIALS,
    align_paired_rows,
    build_paired_comparison,
    classify_candidate,
    classify_sweep,
    gain_slug,
    load_jsonl,
    model_manifest_name,
    primary_metrics,
    reproduction_check,
    sha256_file,
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


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "h_ee_002_hybrid_gain_sweep"
DEFAULT_BASELINE_DIR = DEFAULT_OUTPUT_DIR / "baseline_inputs"


def _registration_path(output_dir: Path) -> Path:
    return output_dir / "h_ee_002_registration.json"


def _load_registration(output_dir: Path) -> dict[str, Any]:
    path = _registration_path(output_dir)
    if not path.is_file():
        raise RuntimeError("registration must be written before any evaluation")
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_registered_inputs(
    baseline_dir: Path, output_dir: Path, protocol_hash: str
) -> dict[str, Any]:
    registration = _load_registration(output_dir)
    if registration.get("protocol", {}).get("config_sha256") != protocol_hash:
        raise RuntimeError("current protocol hash differs from registration")
    registered = {
        row["path"]: row["sha256"]
        for row in registration["frozen_inputs"]["file_inventory"]
    }
    for relative, expected in registered.items():
        actual = sha256_file(baseline_dir / relative)
        if actual != expected:
            raise RuntimeError(f"registered frozen input changed: {relative}")
    for row in registration["source_hashes"]:
        actual = sha256_file(PROJECT_ROOT / row["path"])
        if actual != row["sha256"]:
            raise RuntimeError(f"registered experiment source changed: {row['path']}")
    return registration


def register(args: argparse.Namespace) -> None:
    if _registration_path(args.output_dir).exists() and not args.overwrite_registration:
        raise RuntimeError("registration already exists; refusing to rewrite it")
    protocol = load_evaluation_protocol()
    metadata = protocol.metadata("validation")
    frozen = verify_frozen_inputs(
        args.baseline_dir,
        protocol_hash=str(metadata["config_sha256"]),
        protocol_version=int(metadata["version"]),
        source_dir=args.source_baseline_dir,
    )
    baseline_rows = load_jsonl(
        args.baseline_dir / "eval" / f"{ACTION_SPACE}_policy_trials.jsonl"
    )
    metrics = summarize_rows(baseline_rows)
    if len(baseline_rows) != TOTAL_TRIALS:
        raise ValueError(f"expected {TOTAL_TRIALS} baseline rows, got {len(baseline_rows)}")
    reproduction = reproduction_check(metrics)
    if not reproduction["exact_primary_counts"]:
        raise ValueError(f"imported H-EE-014 baseline mismatch: {reproduction['differences']}")
    source_files = [
        Path("src/svla/state_bc.py"),
        Path("src/svla/core/action_space.py"),
        Path("src/svla/pickup_task.py"),
        Path("src/svla/h_ee_002.py"),
        Path("scripts/run_h_ee_002_gain_sweep.py"),
        Path("configs/phase5_evaluation_protocol_v2.json"),
    ]
    payload = {
        "format": "svla_h_ee_002_registration_v1",
        "hypothesis": HYPOTHESIS,
        "scope": "inference-only frozen-hybrid EE validation gain sweep",
        "gain_list": list(GAINS),
        "selection_order": [0.875, 0.75],
        "seeds": list(SEEDS),
        "trials_per_seed": 24,
        "total_trials_per_gain": TOTAL_TRIALS,
        "action_space": ACTION_SPACE,
        "frozen_contract": {
            "policy_type": "hybrid_nn_gripper_mlp",
            "recipe": "A1_compositor",
            "loss_profile": "global_gripper",
            "match_contract": "historical_object_contact",
            "temporal_feature_mode": "legacy_progress_phase",
            "search_window": args.search_window,
            "max_steps": args.max_steps,
            "gripper_scaled": False,
            "training_allowed": False,
        },
        "protocol": metadata,
        "baseline_metrics": metrics,
        "baseline_primary": primary_metrics(metrics),
        "expected_baseline_primary": EXPECTED_BASELINE_PRIMARY,
        "pass_bars": PASS_BARS,
        "controller_causality_definition": (
            "Material decline means at least 10% relative reduction in failure-conditioned "
            "joint-limit or infeasible exposure rate. Paired improvement concentration means "
            "at least half of newly successful keys came from the gain-1.0 missing-lift bucket."
        ),
        "partial_definition": (
            "missing_lift_eo <=27 plus material controller-exposure decline and no "
            "physical/event-order/reopen/early-close/controller-failure regression"
        ),
        "frozen_inputs": frozen,
        "source_hashes": [
            {"path": str(path), "sha256": sha256_file(PROJECT_ROOT / path)}
            for path in source_files
        ],
        "final_accessed": False,
        "candidate_evaluation_started": False,
    }
    write_json(_registration_path(args.output_dir), payload)
    write_json(
        args.output_dir / "h_ee_002_frozen_inputs_manifest.json",
        {
            "format": "svla_h_ee_002_frozen_inputs_manifest_v1",
            "protocol": metadata,
            "file_inventory": frozen["file_inventory"],
            "loaded_models": frozen["loaded_models"],
            "final_accessed": False,
        },
    )
    print(json.dumps({"registration": str(_registration_path(args.output_dir)), "baseline_primary": primary_metrics(metrics)}, indent=2))


def evaluate_rows(
    baseline_dir: Path,
    *,
    gain: float,
    seeds: list[int],
    eval_limit: int | None,
    max_steps: int,
    search_window: int,
) -> list[dict[str, Any]]:
    protocol = load_evaluation_protocol()
    metadata = protocol.metadata("validation")
    specs = protocol.specs("validation", repeats=1)
    if eval_limit is not None:
        specs = specs[:eval_limit]
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        path = baseline_dir / "models" / model_manifest_name(seed)
        policy = load_policy(path)
        if not isinstance(policy, HybridNNGripperMLPPolicy):
            raise TypeError(f"expected hybrid policy: {path}")
        if policy.evaluation_config_hash != metadata["config_sha256"]:
            raise RuntimeError(f"policy protocol hash mismatch: {path}")
        available = set(policy.group_keys)
        env = PickupTaskEvaluator()
        for spec in specs:
            context = TaskContext.from_spec(spec)
            if context.key not in available:
                raise RuntimeError(f"missing context {context.key} for seed {seed}")
            result = rollout_policy(
                env,
                policy,
                spec,
                max_steps=max_steps,
                search_window=search_window,
                action_gain=gain,
            )
            row = result.to_dict()
            row.update(
                {
                    "seed": seed,
                    "action_gain": gain,
                    "evaluation_protocol": metadata,
                    "evaluation_config_hash": metadata["config_sha256"],
                    "hypothesis": HYPOTHESIS,
                    "inference_only": True,
                }
            )
            rows.append(row)
            print(
                f"gain={gain:.3f} seed={seed} trial={result.trial_id} "
                f"success={int(result.success)} eo={int(result.event_order_valid)} "
                f"phys={int(result.physical_sanity_pass)} fail={result.failure_category} "
                f"jl={result.joint_limit_clipped_steps} infeasible={result.infeasible_steps}",
                flush=True,
            )
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def smoke(args: argparse.Namespace) -> None:
    protocol = load_evaluation_protocol()
    _verify_registered_inputs(args.baseline_dir, args.output_dir, protocol.sha256)
    smoke_payload: dict[str, Any] = {
        "format": "svla_h_ee_002_gain_smoke_v1",
        "hypothesis": HYPOTHESIS,
        "purpose": "plumbing only; one validation trial at each preregistered gain",
        "gains": {},
        "final_accessed": False,
    }
    for gain in GAINS:
        rows = evaluate_rows(
            args.baseline_dir,
            gain=gain,
            seeds=[SEEDS[0]],
            eval_limit=1,
            max_steps=args.max_steps,
            search_window=args.search_window,
        )
        smoke_payload["gains"][gain_slug(gain)] = rows[0]
    write_json(args.output_dir / "h_ee_002_smoke.json", smoke_payload)
    print(f"wrote {args.output_dir / 'h_ee_002_smoke.json'}")


def evaluate_gain(args: argparse.Namespace) -> None:
    gain = float(args.gain)
    if gain not in GAINS:
        raise ValueError(f"gain must be one of {GAINS}")
    protocol = load_evaluation_protocol()
    registration = _verify_registered_inputs(
        args.baseline_dir, args.output_dir, protocol.sha256
    )
    smoke_path = args.output_dir / "h_ee_002_smoke.json"
    if not smoke_path.is_file():
        raise RuntimeError("all-gain smoke must complete before full evaluation")
    control_summary_path = args.output_dir / f"{gain_slug(1.0)}_summary.json"
    if gain != 1.0:
        if not control_summary_path.is_file():
            raise RuntimeError("gain-1.0 full control must run before candidate gains")
        control = json.loads(control_summary_path.read_text(encoding="utf-8"))
        if not control["reproduction"]["exact_primary_counts"]:
            raise RuntimeError("gain-1.0 did not reproduce; candidate gains are blocked")
    slug = gain_slug(gain)
    jsonl_path = args.output_dir / f"{slug}_policy_trials.jsonl"
    summary_path = args.output_dir / f"{slug}_summary.json"
    if jsonl_path.exists() or summary_path.exists():
        raise RuntimeError(f"refusing to overwrite completed gain output: {slug}")
    rows = evaluate_rows(
        args.baseline_dir,
        gain=gain,
        seeds=list(SEEDS),
        eval_limit=None,
        max_steps=int(registration["frozen_contract"]["max_steps"]),
        search_window=int(registration["frozen_contract"]["search_window"]),
    )
    if len(rows) != TOTAL_TRIALS:
        raise RuntimeError(f"expected {TOTAL_TRIALS} rows, got {len(rows)}")
    metrics = summarize_rows(rows)
    reproduction = reproduction_check(metrics) if gain == 1.0 else None
    _write_jsonl(jsonl_path, rows)
    payload = {
        "format": "svla_h_ee_002_gain_summary_v1",
        "hypothesis": HYPOTHESIS,
        "gain": gain,
        "metrics": metrics,
        "reproduction": reproduction,
        "frozen_input_manifest_sha256": sha256_file(
            args.output_dir / "h_ee_002_frozen_inputs_manifest.json"
        ),
        "protocol": registration["protocol"],
        "training_performed": False,
        "final_accessed": False,
    }
    write_json(summary_path, payload)
    if gain == 1.0 and not reproduction["exact_primary_counts"]:
        raise RuntimeError(
            f"gain-1.0 reproduction failed; STOP before candidates: {reproduction['differences']}"
        )
    print(json.dumps({"gain": gain, "metrics": primary_metrics(metrics), "reproduction": reproduction}, indent=2))


# Immutable experiment JSONL digests. reanalyze-derived refuses to proceed if these drift.
FROZEN_POLICY_TRIAL_SHA256 = {
    gain_slug(1.0): "bf31302e53a4b3b054863d4d849a712f985775d1cef21754777b8a482589aca4",
    gain_slug(0.875): "4a160d8df997816805df9090dc89eb90927eb21d561f20f8800b365763c4044d",
    gain_slug(0.75): "9ee88278aa450f13e090820320c79ec5ea7d213d6acc83523f21f0e6d40880d6",
}


def _assert_frozen_jsonl(output_dir: Path) -> dict[str, list[dict[str, Any]]]:
    rows_by_slug: dict[str, list[dict[str, Any]]] = {}
    for gain in GAINS:
        slug = gain_slug(gain)
        path = output_dir / f"{slug}_policy_trials.jsonl"
        actual = sha256_file(path)
        expected = FROZEN_POLICY_TRIAL_SHA256[slug]
        if actual != expected:
            raise RuntimeError(
                f"refusing reanalyze: immutable JSONL hash mismatch for {path.name}: "
                f"expected {expected}, got {actual}"
            )
        rows = load_jsonl(path)
        if len(rows) != TOTAL_TRIALS:
            raise RuntimeError(f"expected {TOTAL_TRIALS} rows in {path}, got {len(rows)}")
        trial_ids = {int(row["trial_id"]) for row in rows}
        if trial_ids != set(range(6001, 6025)):
            raise RuntimeError(f"unexpected trial_id set in {path.name}: {sorted(trial_ids)[:5]}...")
        rows_by_slug[slug] = rows
    return rows_by_slug


def _write_gain_summary(
    output_dir: Path,
    *,
    gain: float,
    rows: list[dict[str, Any]],
    registration: dict[str, Any],
) -> dict[str, Any]:
    metrics = summarize_rows(rows)
    reproduction = reproduction_check(metrics) if float(gain) == 1.0 else None
    payload = {
        "format": "svla_h_ee_002_gain_summary_v1",
        "hypothesis": HYPOTHESIS,
        "gain": gain,
        "metrics": metrics,
        "reproduction": reproduction,
        "frozen_input_manifest_sha256": sha256_file(
            output_dir / "h_ee_002_frozen_inputs_manifest.json"
        ),
        "protocol": registration["protocol"],
        "training_performed": False,
        "final_accessed": False,
    }
    write_json(output_dir / f"{gain_slug(gain)}_summary.json", payload)
    return payload


def _finalize_from_rows(
    output_dir: Path,
    *,
    baseline_rows: list[dict[str, Any]],
    baseline_metrics: dict[str, Any],
    candidate_rows: dict[str, list[dict[str, Any]]],
    candidate_metrics: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    candidates: dict[str, dict[str, Any]] = {}
    paired_output: dict[str, Any] = {
        "format": "svla_h_ee_002_paired_comparison_v1",
        "baseline_gain": 1.0,
        "candidates": {},
        "final_accessed": False,
    }
    for gain in (0.875, 0.75):
        slug = gain_slug(gain)
        rows = candidate_rows[slug]
        align_paired_rows(baseline_rows, rows)
        paired = build_paired_comparison(baseline_rows, rows)
        classification = classify_candidate(
            candidate_metrics[slug], baseline_metrics, paired
        )
        block = {
            "gain": gain,
            "metrics": candidate_metrics[slug],
            "paired": paired,
            "classification": classification,
        }
        candidates[slug] = block
        paired_output["candidates"][slug] = paired
    sweep = classify_sweep(candidates)
    summary_payload = {
        "format": "svla_h_ee_002_gain_sweep_summary_v1",
        "hypothesis": HYPOTHESIS,
        "status": sweep["status"],
        "selected_gain": sweep["selected_gain"],
        "interpretation": (
            "Lower clipping without successful-lift improvement is not a causal win; "
            "constraint exposure may be a symptom or consequence."
        ),
        "registration_sha256": sha256_file(_registration_path(output_dir)),
        "baseline": {"gain": 1.0, "metrics": baseline_metrics},
        "candidates": candidates,
        "joint_reference": {
            "successes": 97,
            "total": 120,
            "source": "frozen H-EE-014 contract; not re-evaluated",
        },
        "training_performed": False,
        "final_accessed": False,
        "phase_6b_started": False,
    }
    write_json(output_dir / "h_ee_002_paired_comparison.json", paired_output)
    write_json(output_dir / "h_ee_002_gain_sweep_summary.json", summary_payload)
    return {"status": sweep, "candidates": candidates, "paired": paired_output}


def finalize(args: argparse.Namespace) -> None:
    protocol = load_evaluation_protocol()
    _verify_registered_inputs(args.baseline_dir, args.output_dir, protocol.sha256)
    baseline_rows = load_jsonl(
        args.output_dir / f"{gain_slug(1.0)}_policy_trials.jsonl"
    )
    baseline_summary = json.loads(
        (args.output_dir / f"{gain_slug(1.0)}_summary.json").read_text(encoding="utf-8")
    )
    if not baseline_summary["reproduction"]["exact_primary_counts"]:
        raise RuntimeError("cannot finalize: gain-1.0 reproduction failed")
    baseline_metrics = baseline_summary["metrics"]
    candidate_rows: dict[str, list[dict[str, Any]]] = {}
    candidate_metrics: dict[str, dict[str, Any]] = {}
    for gain in (0.875, 0.75):
        slug = gain_slug(gain)
        rows = load_jsonl(args.output_dir / f"{slug}_policy_trials.jsonl")
        summary = json.loads(
            (args.output_dir / f"{slug}_summary.json").read_text(encoding="utf-8")
        )
        candidate_rows[slug] = rows
        candidate_metrics[slug] = summary["metrics"]
    result = _finalize_from_rows(
        args.output_dir,
        baseline_rows=baseline_rows,
        baseline_metrics=baseline_metrics,
        candidate_rows=candidate_rows,
        candidate_metrics=candidate_metrics,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "primary_metrics": {
                    "gain_1_000": primary_metrics(baseline_metrics),
                    **{
                        slug: primary_metrics(block["metrics"])
                        for slug, block in result["candidates"].items()
                    },
                },
            },
            indent=2,
        )
    )


def reanalyze_derived(args: argparse.Namespace) -> None:
    """Rebuild per-gain summaries and paired/sweep artifacts from frozen JSONL only.

    Does not rewrite registration, frozen-input manifest, policy-trial JSONL, or models.
    Safe after post-experiment analysis-schema fixes that change source digests.
    """
    registration = _load_registration(args.output_dir)
    for immutable in (
        "h_ee_002_registration.json",
        "h_ee_002_frozen_inputs_manifest.json",
        f"{gain_slug(1.0)}_policy_trials.jsonl",
        f"{gain_slug(0.875)}_policy_trials.jsonl",
        f"{gain_slug(0.75)}_policy_trials.jsonl",
    ):
        if not (args.output_dir / immutable).is_file():
            raise FileNotFoundError(f"missing immutable artifact: {immutable}")

    rows_by_slug = _assert_frozen_jsonl(args.output_dir)
    summaries: dict[str, dict[str, Any]] = {}
    for gain in GAINS:
        slug = gain_slug(gain)
        summaries[slug] = _write_gain_summary(
            args.output_dir,
            gain=gain,
            rows=rows_by_slug[slug],
            registration=registration,
        )
        exposure = summaries[slug]["metrics"]["constraint_exposure"]
        if "missing_lift_gain_1_bucket" in exposure:
            raise RuntimeError("stale misleading cohort field still present after reanalyze")
        if "current_gain_missing_lift" not in exposure:
            raise RuntimeError("expected current_gain_missing_lift in regenerated summary")

    baseline_metrics = summaries[gain_slug(1.0)]["metrics"]
    if not summaries[gain_slug(1.0)]["reproduction"]["exact_primary_counts"]:
        raise RuntimeError(
            "reanalyze aborted: gain-1.0 primary metrics no longer match frozen baseline"
        )
    result = _finalize_from_rows(
        args.output_dir,
        baseline_rows=rows_by_slug[gain_slug(1.0)],
        baseline_metrics=baseline_metrics,
        candidate_rows={
            gain_slug(0.875): rows_by_slug[gain_slug(0.875)],
            gain_slug(0.75): rows_by_slug[gain_slug(0.75)],
        },
        candidate_metrics={
            gain_slug(0.875): summaries[gain_slug(0.875)]["metrics"],
            gain_slug(0.75): summaries[gain_slug(0.75)]["metrics"],
        },
    )
    if result["status"]["status"] != "rejected" or result["status"]["selected_gain"] is not None:
        raise RuntimeError(
            f"reanalyze changed scientific verdict: {result['status']}"
        )
    for slug, block in result["candidates"].items():
        if block["paired"]["baseline_missing_lift_trial_count"] != 30:
            raise RuntimeError(
                f"{slug} paired baseline missing-lift cohort drifted from 30"
            )
    print(
        json.dumps(
            {
                "reanalyze": "derived_only",
                "status": result["status"],
                "primary_metrics": {
                    slug: primary_metrics(summaries[slug]["metrics"]) for slug in summaries
                },
                "jsonl_sha256": {
                    f"{slug}_policy_trials.jsonl": FROZEN_POLICY_TRIAL_SHA256[slug]
                    for slug in FROZEN_POLICY_TRIAL_SHA256
                },
                "derived_sha256": {
                    name: sha256_file(args.output_dir / name)
                    for name in (
                        f"{gain_slug(1.0)}_summary.json",
                        f"{gain_slug(0.875)}_summary.json",
                        f"{gain_slug(0.75)}_summary.json",
                        "h_ee_002_paired_comparison.json",
                        "h_ee_002_gain_sweep_summary.json",
                    )
                },
            },
            indent=2,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR
    )
    parser.add_argument(
        "--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR
    )
    parser.add_argument(
        "--source-baseline-dir",
        type=Path,
        default=Path(
            "/Users/matthewwoodcock/Documents/Simplified VLA Structure/outputs/"
            "h_ee_014_nn_gripper_global_validation"
        ),
    )
    parser.add_argument("--max-steps", type=int, default=3200)
    parser.add_argument("--search-window", type=int, default=5)
    subparsers = parser.add_subparsers(dest="command", required=True)
    register_parser = subparsers.add_parser("register")
    register_parser.add_argument("--overwrite-registration", action="store_true")
    subparsers.add_parser("smoke")
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--gain", type=float, required=True)
    subparsers.add_parser("finalize")
    subparsers.add_parser(
        "reanalyze-derived",
        help=(
            "Rebuild gain summaries and paired/sweep JSON from immutable policy-trial "
            "JSONL without re-running MuJoCo or rewriting registration/models."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir = args.output_dir.resolve()
    args.baseline_dir = args.baseline_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.command == "register":
        register(args)
    elif args.command == "smoke":
        smoke(args)
    elif args.command == "evaluate":
        evaluate_gain(args)
    elif args.command == "finalize":
        finalize(args)
    elif args.command == "reanalyze-derived":
        reanalyze_derived(args)
    else:  # pragma: no cover
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
