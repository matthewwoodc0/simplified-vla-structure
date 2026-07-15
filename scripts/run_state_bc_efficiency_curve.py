#!/usr/bin/env python3
"""Deterministic, provenance-safe state-BC demonstration-efficiency curve runner.

Produces READY-TO-RUN infrastructure execution modes:
  - dry-run: print/write the full 150-fit matrix (no training)
  - smoke: plumbing-only non-efficacy cell (1 budget × 1 ladder × 1 seed, ≤2 epochs)
  - primary-curve: full development-split matrix (requires review; not part of Goal 02)
  - locked-evaluation: requires explicit --allow-locked-evaluation (never supplied by Goal 02)

Default train_state_bc.py behavior is intentionally unchanged; this script is the
dedicated efficiency entrypoint.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from svla.demo_recorder import PickupDemoRecorder
from svla.efficiency.protocol import (
    ACTION_SPACES,
    EFFICIENCY_PROTOCOL_PATH,
    MatrixCell,
    build_fit_matrix,
    load_efficiency_protocol,
    sha256_json,
    source_demo_paths_for_cell,
    validate_cell_artifact_for_resume,
)
from svla.eval.manifest import ExperimentManifest, tracked_source_hashes
from svla.loss_profiles import resolve_loss_weights
from svla.pickup_task import (
    ApproachStrategy,
    GraspOrientation,
    ObjectStartPose,
    PickupTaskEvaluator,
    PickupTrialSpec,
)
from svla.state_bc import (
    DEFAULT_MATCH_CONTRACT,
    HYBRID_POLICY_TYPE,
    HYBRID_RECIPE_A1,
    HybridNNGripperMLPPolicy,
    TaskContext,
    fit_mlp_policy,
    fit_nearest_neighbor_policy,
    load_demo_dataset,
    resolve_match_contract,
    rollout_policy,
    summarize_policy_results,
    write_json,
)


def _peak_rss_bytes() -> int | None:
    """Best-effort peak RSS without heavyweight dependencies."""
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except Exception:
        return None
    # macOS reports bytes; Linux reports kilobytes.
    if sys.platform == "darwin":
        return int(usage)
    return int(usage) * 1024


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _execution_source_identity() -> tuple[str, dict[str, str]]:
    source_hashes = tracked_source_hashes(PROJECT_ROOT)
    return sha256_json(source_hashes), source_hashes


def _mean_step_rate(summary: dict[str, Any], numerator_key: str) -> float:
    mean_steps = float(summary.get("mean_steps") or 0.0)
    if mean_steps <= 0.0:
        return 0.0
    return float(summary.get(numerator_key) or 0.0) / mean_steps


def demo_specs_from_protocol(protocol) -> list[PickupTrialSpec]:
    specs: list[PickupTrialSpec] = []
    for row in protocol.data["demo_pool"]["demos"]:
        specs.append(
            PickupTrialSpec(
                trial_id=int(row["trial_id"]),
                orientation=GraspOrientation(
                    str(row["orientation"]), float(row["yaw_degrees"])
                ),
                object_pose=ObjectStartPose(
                    str(row["object_pose"]),
                    np.asarray(row["object_xyz"], dtype=float),
                ),
                approach=ApproachStrategy(str(row["approach"]), str(row["axis_mode"])),
                repeat=int(row.get("repeat", 0)),
            )
        )
    return specs


def generate_demo_pool(
    demo_dir: Path,
    specs: list[PickupTrialSpec],
    *,
    protocol_metadata: dict,
) -> dict[int, Path]:
    demo_dir.mkdir(parents=True, exist_ok=True)
    recorder = PickupDemoRecorder(PickupTaskEvaluator())
    by_trial: dict[int, Path] = {}
    records = []
    for spec in specs:
        path = demo_dir / (
            f"pickup_demo_{spec.trial_id:04d}_{spec.orientation.label}_"
            f"{spec.object_pose.label}_{spec.approach.label}_repeat_{spec.repeat}.json"
        )
        if path.is_file():
            demo = json.loads(path.read_text(encoding="utf-8"))
            trial = demo.get("metadata", {}).get("trial_spec", {})
            expected_trial = {
                "trial_id": int(spec.trial_id),
                "orientation": spec.orientation.label,
                "object_pose": spec.object_pose.label,
                "approach": spec.approach.label,
                "repeat": int(spec.repeat),
            }
            existing_protocol = demo.get("metadata", {}).get(
                "evaluation_protocol", {}
            )
            if any(trial.get(key) != value for key, value in expected_trial.items()):
                raise RuntimeError(
                    f"BLOCKED_DATA_POOL: existing demo trial contract mismatch: {path}"
                )
            if existing_protocol.get("config_sha256") != protocol_metadata.get(
                "config_sha256"
            ):
                raise RuntimeError(
                    f"BLOCKED_DATA_POOL: existing demo protocol hash mismatch: {path}"
                )
        else:
            demo = recorder.write_trial(spec, path)
            demo["metadata"]["evaluation_protocol"] = protocol_metadata
            demo["metadata"]["efficiency_study"] = True
            write_json(path, demo)
        if not bool(demo["summary"]["success"]):
            raise RuntimeError(
                f"BLOCKED_DATA_POOL: scripted demo trial_id={spec.trial_id} "
                f"failed success gate ({demo['summary'].get('failure_category')})"
            )
        by_trial[int(spec.trial_id)] = path
        records.append(
            {
                "trial_id": int(spec.trial_id),
                "path": str(path),
                "success": True,
                "sha256": _sha256_file(path),
            }
        )
        print(
            f"demo trial={spec.trial_id} success=1 samples={len(demo['samples'])} "
            f"path={path.name}"
        )
    manifest = {
        "format": "svla_efficiency_demo_pool_manifest_v1",
        "demo_count": len(records),
        "demos": records,
        "protocol_metadata": protocol_metadata,
    }
    write_json(demo_dir / "manifest.json", manifest)
    return by_trial


def matrix_payload(protocol, cells: list[MatrixCell]) -> dict[str, Any]:
    return {
        "format": "svla_state_bc_efficiency_matrix_v1",
        "protocol_path": str(protocol.path),
        "protocol_sha256": protocol.sha256,
        "recipe_hash": protocol.recipe_hash,
        "frozen_recipe": protocol.frozen_recipe,
        "planned_cell_count": len(cells),
        "cell_count": len(cells),
        "cells": [cell.to_dict() for cell in cells],
        "primary_curve_executed": False,
        "locked_evaluation_accessed": False,
    }


def filter_cells(
    cells: list[MatrixCell],
    *,
    budgets: list[int] | None,
    ladders: list[str] | None,
    seeds: list[int] | None,
    action_spaces: list[str] | None,
) -> list[MatrixCell]:
    out = cells
    if budgets is not None:
        allowed = set(int(b) for b in budgets)
        out = [c for c in out if c.budget in allowed]
    if ladders is not None:
        allowed_l = set(ladders)
        out = [c for c in out if c.ladder_id in allowed_l]
    if seeds is not None:
        allowed_s = set(int(s) for s in seeds)
        out = [c for c in out if c.model_seed in allowed_s]
    if action_spaces is not None:
        allowed_a = set(action_spaces)
        out = [c for c in out if c.action_space in allowed_a]
    return out


def cell_dir(output_dir: Path, cell: MatrixCell) -> Path:
    return output_dir / "cells" / cell.cell_id


def load_existing_cell_manifest(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def run_cell(
    *,
    cell: MatrixCell,
    protocol,
    demo_path_by_trial: dict[int, Path],
    eval_specs: list[PickupTrialSpec],
    output_dir: Path,
    epochs: int,
    eval_limit: int | None,
    non_efficacy_smoke: bool,
    resume: bool,
    protocol_metadata: dict,
    execution_mode: str,
    execution_source_identity: str,
) -> dict[str, Any]:
    out = cell_dir(output_dir, cell)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "cell_manifest.json"
    demo_paths = source_demo_paths_for_cell(
        cell=cell, demo_path_by_trial_id=demo_path_by_trial
    )
    demo_path_strings = [str(p.resolve()) for p in demo_paths]
    demo_path_hashes = [_sha256_file(path) for path in demo_paths]

    specs = list(eval_specs)
    registered_eval_total = len(specs)
    if eval_limit is not None:
        specs = specs[: int(eval_limit)]
    evaluation_trial_ids = [int(spec.trial_id) for spec in specs]
    evaluation_identity_hash = sha256_json(
        {
            "eval_split": protocol_metadata["eval_split"],
            "trial_ids": evaluation_trial_ids,
        }
    )
    execution_context = {
        "execution_mode": execution_mode,
        "eval_split": protocol_metadata["eval_split"],
        "epochs_executed": int(epochs),
        "eval_limit": eval_limit,
        "non_efficacy_smoke": bool(non_efficacy_smoke),
        "evaluation_trial_ids": evaluation_trial_ids,
        "evaluation_identity_hash": evaluation_identity_hash,
        "demo_source_path_sha256": demo_path_hashes,
        "execution_source_identity": execution_source_identity,
    }
    existing = load_existing_cell_manifest(manifest_path)
    if existing is not None:
        try:
            validate_cell_artifact_for_resume(
                cell=cell,
                artifact=existing,
                execution_context=execution_context,
            )
        except ValueError as exc:
            raise RuntimeError(
                f"resume rejected stale/mismatched artifact for {cell.cell_id}: {exc}"
            ) from exc
        if resume and existing.get("status") == "completed":
            print(f"resume skip exact-match cell={cell.cell_id}")
            return existing
        if existing.get("status") == "completed":
            raise RuntimeError(
                f"completed cell already exists for {cell.cell_id}; use --resume "
                "or choose a new output directory"
            )

    # Parity guard: both action spaces must receive identical ordered trial ids.
    # Path list identity is enforced at the matrix level; re-check here.
    expected_ids = list(cell.demo_trial_ids)
    actual_ids = []
    for path in demo_paths:
        demo = json.loads(path.read_text(encoding="utf-8"))
        actual_ids.append(int(demo["metadata"]["trial_spec"]["trial_id"]))
    if actual_ids != expected_ids:
        raise RuntimeError(
            "BLOCKED_COMPARISON_PARITY: demo path list trial ids do not match cell "
            f"contract for {cell.cell_id}: {actual_ids} != {expected_ids}"
        )

    recipe = protocol.frozen_recipe
    gripper_loss_weight, close_phase_gripper_weight, resolved_profile = resolve_loss_weights(
        loss_profile=str(recipe["loss"]),
        gripper_loss_weight=None,
        close_phase_gripper_weight=None,
    )
    match_contract = str(recipe["nn_match"])
    match_indices, match_names = resolve_match_contract(match_contract)

    train_t0 = time.perf_counter()
    rss_before = _peak_rss_bytes()
    dataset = load_demo_dataset(
        demo_paths,
        action_space=cell.action_space,
        success_only=True,
        stride=int(recipe.get("stride", 1)),
        label_source=str(recipe["label_source"]),
    )
    if dataset.demo_count != cell.budget:
        raise RuntimeError(
            f"expected {cell.budget} successful demos, got {dataset.demo_count} "
            f"for cell {cell.cell_id}"
        )

    mlp_policy, fit_summary = fit_mlp_policy(
        dataset,
        hidden_sizes=tuple(int(x) for x in recipe["hidden_sizes"]),
        epochs=int(epochs),
        batch_size=int(recipe["batch"]),
        learning_rate=float(recipe["learning_rate"]),
        weight_decay=float(recipe["weight_decay"]),
        seed=int(cell.model_seed),
        temporal_feature_mode=str(recipe["temporal_features"]),
        gripper_loss_weight=gripper_loss_weight,
        close_phase_gripper_weight=close_phase_gripper_weight,
        loss_profile=None if resolved_profile is None else resolved_profile.name,
        arm_only_mlp=False,
    )
    nn_policy = fit_nearest_neighbor_policy(
        dataset,
        k=8,
        temperature=0.75,
        match_contract=match_contract,
    )
    policy = HybridNNGripperMLPPolicy(
        mlp_policy,
        nn_policy,
        match_feature_indices=match_indices,
        match_feature_names=match_names,
        match_contract=match_contract,
        recipe=HYBRID_RECIPE_A1,
        arm_only_mlp=False,
    )
    policy.evaluation_config_hash = protocol.sha256
    policy.evaluation_protocol_version = protocol.version
    model_path = out / f"{cell.action_space}_{HYBRID_POLICY_TYPE}_seed_{cell.model_seed}.json"
    saved = policy.save(model_path)
    if saved is not None:
        model_path = Path(saved)
    train_wall = time.perf_counter() - train_t0

    available_contexts = set(policy.group_keys)
    missing_contexts = sorted(
        {
            TaskContext.from_spec(spec).key
            for spec in specs
            if TaskContext.from_spec(spec).key not in available_contexts
        }
    )
    if missing_contexts:
        raise RuntimeError(
            f"incomplete task contexts for cell {cell.cell_id}: {missing_contexts}"
        )

    env = PickupTaskEvaluator()
    rollout_t0 = time.perf_counter()
    results = []
    for spec in specs:
        result = rollout_policy(
            env,
            policy,
            spec,
            max_steps=int(recipe.get("max_steps", 3200)),
            search_window=24,
            action_gain=float(recipe["action_gain"]),
            gripper_close_guard=False,
        )
        results.append(result)
        print(
            f"cell={cell.cell_id} trial={result.trial_id} "
            f"success={int(result.success)} failure={result.failure_category}"
        )
    rollout_wall = time.perf_counter() - rollout_t0
    rss_after = _peak_rss_bytes()
    summary = summarize_policy_results(results)
    n_trials = int(summary["total"])
    success_count = int(summary["successes"])
    success_rate = float(summary["success_rate"]) if n_trials else 0.0
    joint_limit_rate = _mean_step_rate(summary, "mean_joint_limit_clipped_steps")
    infeasible_rate = _mean_step_rate(summary, "mean_infeasible_steps")

    trials_path = out / "policy_trials.jsonl"
    with trials_path.open("w", encoding="utf-8") as handle:
        for result in results:
            record = result.to_dict()
            record["seed"] = int(cell.model_seed)
            record["cell_id"] = cell.cell_id
            record["budget"] = cell.budget
            record["ladder_id"] = cell.ladder_id
            record["non_efficacy_smoke"] = bool(non_efficacy_smoke)
            record["evaluation_protocol"] = protocol_metadata
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    model_bytes = model_path.stat().st_size if model_path.is_file() else None
    cell_summary = {
        "format": "svla_state_bc_efficiency_cell_v1",
        "status": "completed",
        **cell.to_dict(),
        "non_efficacy_smoke": bool(non_efficacy_smoke),
        "execution_mode": execution_mode,
        "eval_split": protocol_metadata["eval_split"],
        "epochs_executed": int(epochs),
        "eval_limit": eval_limit,
        "registered_eval_total": registered_eval_total,
        "evaluated_spec_count": n_trials,
        "evaluation_trial_ids": evaluation_trial_ids,
        "evaluation_identity_hash": evaluation_identity_hash,
        "success_count": success_count,
        "success_rate": success_rate,
        "n_trials": n_trials,
        "event_order_valid_rate": float(summary.get("event_order_valid_rate", 0.0)),
        "physical_sanity_pass_rate": float(summary.get("physical_sanity_pass_rate", 0.0)),
        "early_close_trials": int(summary.get("early_close_trials", 0)),
        "reopen_events": int(summary.get("reopen_events", 0)),
        "joint_limit_clipped_step_rate": float(joint_limit_rate),
        "infeasible_step_rate": float(infeasible_rate),
        "supervised_timestep_count": int(dataset.features.shape[0]),
        "training_wall_time_s": float(train_wall),
        "rollout_wall_time_s": float(rollout_wall),
        "model_bytes": model_bytes,
        "peak_process_memory_bytes": rss_after,
        "peak_process_memory_delta_bytes": (
            None
            if rss_before is None or rss_after is None
            else int(rss_after - rss_before)
        ),
        "demo_source_paths": demo_path_strings,
        "demo_source_path_sha256": demo_path_hashes,
        "execution_source_identity": execution_source_identity,
        "model_path": str(model_path),
        "model_sha256": _sha256_file(model_path) if model_path.is_file() else None,
        "trials_path": str(trials_path),
        "trials_sha256": _sha256_file(trials_path),
        "fit_summary": {
            **fit_summary,
            "policy_type": HYBRID_POLICY_TYPE,
            "hybrid_recipe": HYBRID_RECIPE_A1,
            "match_contract": match_contract,
            "match_feature_names": list(match_names),
        },
        "protocol_metadata": protocol_metadata,
        "controller": "damped_least_squares_ik",
        "label_source": str(recipe["label_source"]),
    }
    write_json(out / "cell_summary.json", cell_summary)
    write_json(manifest_path, cell_summary)
    return cell_summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="State-BC demonstration-efficiency curve runner"
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=EFFICIENCY_PROTOCOL_PATH,
        help="Path to state_bc_efficiency_protocol_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "state_bc_efficiency_curve",
    )
    parser.add_argument(
        "--mode",
        choices=("dry-run", "smoke", "primary-curve", "locked-evaluation"),
        default="dry-run",
        help="Execution mode. Default dry-run does not train.",
    )
    parser.add_argument(
        "--allow-locked-evaluation",
        action="store_true",
        help="Literal opt-in required for locked-evaluation mode.",
    )
    parser.add_argument("--budgets", type=int, nargs="*", default=None)
    parser.add_argument("--ladders", type=str, nargs="*", default=None)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument(
        "--action-spaces",
        type=str,
        nargs="*",
        default=None,
        choices=list(ACTION_SPACES),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override frozen epoch count (smoke only should use <=2).",
    )
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip cells with exact-matching completed manifests.",
    )
    parser.add_argument(
        "--demo-dir",
        type=Path,
        default=None,
        help="Optional shared demo pool directory (default: <output-dir>/demo_pool).",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace, *, command: list[str] | None = None) -> dict:
    protocol = load_efficiency_protocol(args.protocol)
    cells = build_fit_matrix(protocol)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "locked-evaluation" and not args.allow_locked_evaluation:
        raise ValueError(
            "locked_evaluation requires explicit --allow-locked-evaluation"
        )
    if args.allow_locked_evaluation and args.mode != "locked-evaluation":
        raise ValueError(
            "--allow-locked-evaluation is only valid with --mode locked-evaluation"
        )

    if args.mode in {"primary-curve", "locked-evaluation"}:
        subset_flags = {
            "budgets": args.budgets,
            "ladders": args.ladders,
            "seeds": args.seeds,
            "action_spaces": args.action_spaces,
            "eval_limit": args.eval_limit,
        }
        active = [name for name, value in subset_flags.items() if value is not None]
        if active:
            raise ValueError(
                f"{args.mode} must execute the complete registered matrix; "
                f"subset overrides are forbidden: {active}"
            )

    non_efficacy_smoke = False
    epochs = (
        int(args.epochs)
        if args.epochs is not None
        else int(protocol.frozen_recipe["epochs"])
    )
    eval_limit = args.eval_limit
    selected = cells

    if args.mode == "dry-run":
        payload = matrix_payload(protocol, cells)
        payload["mode"] = "dry-run"
        payload["non_efficacy_smoke"] = False
        matrix_path = output_dir / "efficiency_matrix_dry_run.json"
        write_json(matrix_path, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        print(
            f"dry-run matrix cells={len(cells)} "
            f"unique_ids={len({c.cell_id for c in cells})} wrote={matrix_path}"
        )
        return payload

    if args.mode == "smoke":
        non_efficacy_smoke = True
        # Plumbing-only: one budget, one ladder, one seed, both action spaces allowed
        # but default to both for parity check; epochs <= 2; eval-limit=1; development.
        if args.budgets is not None and len(args.budgets) != 1:
            raise ValueError("smoke mode accepts exactly one budget")
        if args.ladders is not None and len(args.ladders) != 1:
            raise ValueError("smoke mode accepts exactly one ladder")
        if args.seeds is not None and len(args.seeds) != 1:
            raise ValueError("smoke mode accepts exactly one seed")
        if args.action_spaces is not None and set(args.action_spaces) != set(ACTION_SPACES):
            raise ValueError("smoke mode must exercise both registered action spaces")
        smoke_budget = (
            int(args.budgets[0]) if args.budgets else int(protocol.budgets[0])
        )
        smoke_ladder = args.ladders[0] if args.ladders else protocol.ladders[0]["ladder_id"]
        smoke_seed = int(args.seeds[0]) if args.seeds else int(protocol.model_seeds[0])
        if args.epochs is None:
            epochs = 2
        if epochs > 2:
            raise ValueError("smoke mode allows at most 2 epochs")
        if eval_limit is None:
            eval_limit = 1
        if eval_limit > 1:
            raise ValueError("smoke mode allows eval-limit at most 1")
        selected = filter_cells(
            cells,
            budgets=[smoke_budget],
            ladders=[smoke_ladder],
            seeds=[smoke_seed],
            action_spaces=args.action_spaces,
        )
        eval_split = "development"
    elif args.mode == "primary-curve":
        if epochs != int(protocol.frozen_recipe["epochs"]):
            raise ValueError(
                "primary-curve must use frozen epoch count unless this is a "
                "separately registered deviation"
            )
        selected = filter_cells(
            cells,
            budgets=args.budgets,
            ladders=args.ladders,
            seeds=args.seeds,
            action_spaces=args.action_spaces,
        )
        eval_split = "development"
    elif args.mode == "locked-evaluation":
        if epochs != int(protocol.frozen_recipe["epochs"]):
            raise ValueError("locked-evaluation must use the frozen epoch count")
        selected = cells
        eval_split = "locked_evaluation"
    else:
        raise ValueError(f"unknown mode: {args.mode}")

    if not selected:
        raise ValueError("no matrix cells selected")

    protocol_metadata = protocol.metadata(eval_split)
    protocol_metadata["non_efficacy_smoke"] = bool(non_efficacy_smoke)
    protocol_metadata["mode"] = args.mode

    execution_source_identity, execution_source_hashes = _execution_source_identity()
    manifest = ExperimentManifest.start(
        repo_root=PROJECT_ROOT,
        argv=command,
        seeds={
            "model_seeds": sorted({c.model_seed for c in selected}),
            "budgets": sorted({c.budget for c in selected}),
            "ladders": sorted({c.ladder_id for c in selected}),
        },
        metadata={
            "efficiency_protocol": protocol_metadata,
            "frozen_recipe": protocol.frozen_recipe,
            "recipe_hash": protocol.recipe_hash,
            "mode": args.mode,
            "non_efficacy_smoke": bool(non_efficacy_smoke),
            "allow_locked_evaluation": bool(args.allow_locked_evaluation),
            "execution_source_identity": execution_source_identity,
            "execution_source_hashes": execution_source_hashes,
        },
    )

    demo_dir = Path(args.demo_dir) if args.demo_dir else output_dir / "demo_pool"
    run_dir = output_dir / args.mode
    demo_specs = demo_specs_from_protocol(protocol)
    # Only generate demos needed by selected cells (still from the full pool contract).
    needed_ids = sorted({tid for cell in selected for tid in cell.demo_trial_ids})
    needed_specs = [s for s in demo_specs if s.trial_id in set(needed_ids)]
    demo_protocol_metadata = {
        "format": protocol.data["format"],
        "version": protocol.version,
        "config_path": str(protocol.path),
        "config_sha256": protocol.sha256,
        "role": "training_demo_pool",
        "not_protocol_v2_alias": True,
    }
    demo_path_by_trial = generate_demo_pool(
        demo_dir,
        needed_specs,
        protocol_metadata=demo_protocol_metadata,
    )

    eval_specs = protocol.split_specs(eval_split)
    cell_summaries = []
    generated: list[Path] = [demo_dir / "manifest.json"]
    for cell in selected:
        # Ensure sibling action-space cells share identical path lists by trial id order.
        summary = run_cell(
            cell=cell,
            protocol=protocol,
            demo_path_by_trial=demo_path_by_trial,
            eval_specs=eval_specs,
            output_dir=run_dir,
            epochs=epochs,
            eval_limit=eval_limit,
            non_efficacy_smoke=non_efficacy_smoke,
            resume=bool(args.resume),
            protocol_metadata=protocol_metadata,
            execution_mode=args.mode,
            execution_source_identity=execution_source_identity,
        )
        cell_summaries.append(summary)
        generated.extend(
            [
                cell_dir(run_dir, cell) / "cell_manifest.json",
                cell_dir(run_dir, cell) / "cell_summary.json",
            ]
        )

    # Parity audit across action spaces for shared (budget, ladder, seed).
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in cell_summaries:
        key = (row["budget"], row["ladder_id"], row["model_seed"])
        groups[key].append(row)
    for key, rows in groups.items():
        if len(rows) < 2:
            continue
        paths = [tuple(r["demo_source_paths"]) for r in rows]
        if len(set(paths)) != 1:
            raise RuntimeError(
                f"BLOCKED_COMPARISON_PARITY: demo path lists differ for key {key}"
            )

    run_summary = {
        "format": "svla_state_bc_efficiency_run_summary_v1",
        "mode": args.mode,
        "non_efficacy_smoke": bool(non_efficacy_smoke),
        "protocol_path": str(protocol.path),
        "protocol_sha256": protocol.sha256,
        "recipe_hash": protocol.recipe_hash,
        "eval_split": eval_split,
        "allow_locked_evaluation": bool(args.allow_locked_evaluation),
        "primary_curve_executed": args.mode == "primary-curve",
        "locked_evaluation_accessed": args.mode == "locked-evaluation",
        "epochs_executed": int(epochs),
        "eval_limit": eval_limit,
        "selected_cell_count": len(selected),
        "planned_full_matrix_cell_count": 150,
        "cells": cell_summaries,
    }
    summary_path = run_dir / (
        "efficiency_smoke_summary.json"
        if non_efficacy_smoke
        else "efficiency_run_summary.json"
    )
    write_json(summary_path, run_summary)
    generated.append(summary_path)
    manifest.add_outputs(generated)
    sidecar = manifest.write_sidecar(summary_path)
    print(json.dumps(run_summary, indent=2, sort_keys=True))
    print(f"wrote {summary_path}")
    print(f"wrote {sidecar}")
    return run_summary


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    command = list(sys.argv if argv is None else [sys.argv[0], *argv])
    try:
        run(args, command=command)
    except Exception:
        traceback.print_exc()
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
