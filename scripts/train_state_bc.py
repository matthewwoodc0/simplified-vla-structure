from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from svla.demo_recorder import PickupDemoRecorder
from svla.evaluation_protocol import load_evaluation_protocol
from svla.experiment_manifest import ExperimentManifest
import numpy as np

from svla.pickup_task import (
    OBJECT_START_Z,
    ApproachStrategy,
    GraspOrientation,
    ObjectStartPose,
    PickupTaskEvaluator,
    PickupTrialSpec,
    default_trial_specs,
)
from svla.state_bc import (
    ACTION_SPACES,
    TaskContext,
    fit_mlp_policy,
    fit_nearest_neighbor_policy,
    load_demo_dataset,
    rollout_policy,
    summarize_policy_results,
    write_json,
)


def generate_demos(
    output_dir: Path,
    specs: list[PickupTrialSpec],
    evaluation_protocol: dict | None = None,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    recorder = PickupDemoRecorder(PickupTaskEvaluator())
    demos = []
    for spec in specs:
        path = output_dir / (
            f"pickup_demo_{spec.trial_id:03d}_{spec.orientation.label}_"
            f"{spec.object_pose.label}_{spec.approach.label}_repeat_{spec.repeat}.json"
        )
        demo = recorder.write_trial(spec, path)
        if evaluation_protocol is not None:
            demo["metadata"]["evaluation_protocol"] = evaluation_protocol
            write_json(path, demo)
        demos.append({"path": str(path), "summary": demo["summary"]})
        print(
            f"demo={path.name} success={int(demo['summary']['success'])} "
            f"failure={demo['summary']['failure_category']} samples={len(demo['samples'])}"
        )
    manifest = {
        "format": "svla_state_bc_demo_manifest_v1",
        "demo_count": len(demos),
        "demos": demos,
        "evaluation_protocol": evaluation_protocol,
    }
    write_json(output_dir / "manifest.json", manifest)
    return [Path(demo["path"]) for demo in demos]


def grid_object_pose(label: str, x: float) -> ObjectStartPose:
    return ObjectStartPose(label, np.array([x, -0.235 - (x / 6.0), OBJECT_START_Z]))


def trial_specs_for_object_poses(
    object_poses: list[ObjectStartPose],
    start_trial_id: int,
    repeats: int = 1,
) -> list[PickupTrialSpec]:
    orientations = [
        GraspOrientation("yaw_-18", -18.0),
        GraspOrientation("yaw_0", 0.0),
        GraspOrientation("yaw_18", 18.0),
    ]
    approaches = [
        ApproachStrategy("vertical_pregrasp", "world_z"),
        ApproachStrategy("high_staged_vertical_pregrasp", "high_world_z"),
    ]
    specs: list[PickupTrialSpec] = []
    trial_id = start_trial_id
    for repeat in range(repeats):
        for orientation in orientations:
            for object_pose in object_poses:
                for approach in approaches:
                    specs.append(
                        PickupTrialSpec(
                            trial_id=trial_id,
                            orientation=orientation,
                            object_pose=object_pose,
                            approach=approach,
                            repeat=repeat,
                        )
                    )
                    trial_id += 1
    return specs


def dense_training_specs(repeats: int) -> list[PickupTrialSpec]:
    object_poses = [
        grid_object_pose("dense_left", -0.018),
        grid_object_pose("dense_mid_left", -0.009),
        grid_object_pose("dense_center", 0.0),
        grid_object_pose("dense_mid_right", 0.009),
        grid_object_pose("dense_right", 0.018),
    ]
    return trial_specs_for_object_poses(object_poses, start_trial_id=1, repeats=repeats)


def heldout_trial_specs() -> list[PickupTrialSpec]:
    object_poses = [
        grid_object_pose("heldout_left_inner", -0.0135),
        grid_object_pose("heldout_center_left", -0.0045),
        grid_object_pose("heldout_center_right", 0.0045),
        grid_object_pose("heldout_right_inner", 0.0135),
    ]
    return trial_specs_for_object_poses(object_poses, start_trial_id=1001, repeats=1)


def test_trial_specs() -> list[PickupTrialSpec]:
    object_poses = [
        grid_object_pose("test_left_outer", -0.01575),
        grid_object_pose("test_left_inner", -0.00675),
        grid_object_pose("test_right_inner", 0.00675),
        grid_object_pose("test_right_outer", 0.01575),
    ]
    return trial_specs_for_object_poses(object_poses, start_trial_id=2001, repeats=1)


def audit_trial_specs() -> list[PickupTrialSpec]:
    """Final interpolation grid kept separate from train/validation/test tuning."""

    object_poses = [
        grid_object_pose("audit_left_outer", -0.016875),
        grid_object_pose("audit_left_inner", -0.01125),
        grid_object_pose("audit_right_inner", 0.01125),
        grid_object_pose("audit_right_outer", 0.016875),
    ]
    return trial_specs_for_object_poses(object_poses, start_trial_id=3001, repeats=1)


def final_trial_specs() -> list[PickupTrialSpec]:
    """Untouched grid reserved for the reduced tool-axis controller comparison."""

    object_poses = [
        grid_object_pose("final_left_outer", -0.0163125),
        grid_object_pose("final_left_inner", -0.010125),
        grid_object_pose("final_right_inner", 0.010125),
        grid_object_pose("final_right_outer", 0.0163125),
    ]
    return trial_specs_for_object_poses(object_poses, start_trial_id=4001, repeats=1)


def training_specs(mode: str, repeats: int) -> list[PickupTrialSpec]:
    if mode == "default":
        return default_trial_specs(repeats=repeats)
    if mode == "dense":
        return dense_training_specs(repeats=repeats)
    raise ValueError(f"unknown train grid: {mode}")


def evaluation_specs(mode: str, train_grid: str, repeats: int) -> list[PickupTrialSpec]:
    if mode == "train":
        return training_specs(train_grid, repeats=repeats)
    if mode == "heldout":
        return heldout_trial_specs()
    if mode == "test":
        return test_trial_specs()
    if mode == "audit":
        return audit_trial_specs()
    if mode == "final":
        return final_trial_specs()
    if mode == "both":
        return training_specs(train_grid, repeats=repeats) + heldout_trial_specs()
    raise ValueError(f"unknown eval mode: {mode}")


def run(args: argparse.Namespace, *, command: list[str] | None = None) -> dict:
    protocol = None
    protocol_metadata = None
    if args.evaluation_protocol == "v2":
        if args.eval_split is None:
            raise ValueError(
                "v2 evaluation requires explicit --eval-split train, validation, or final"
            )
        protocol = load_evaluation_protocol()
        protocol_metadata = protocol.metadata(args.eval_split)
        seed_values = list(args.seeds) if args.seeds is not None else list(protocol.model_seeds)
        train_specs = protocol.specs("train", repeats=args.demo_repeats)
        eval_specs = protocol.specs(args.eval_split, repeats=args.eval_repeats)
    else:
        if args.eval_split is not None:
            raise ValueError("--eval-split is only valid with --evaluation-protocol v2")
        seed_values = list(args.seeds) if args.seeds is not None else [args.seed]
        train_specs = training_specs(args.train_grid, repeats=args.demo_repeats)
        eval_specs = evaluation_specs(
            args.eval_mode,
            train_grid=args.train_grid,
            repeats=args.eval_repeats,
        )
    registered_eval_total = len(eval_specs)
    if args.eval_limit is not None:
        if args.eval_limit < 1:
            raise ValueError("--eval-limit must be at least one")
        eval_specs = eval_specs[: args.eval_limit]
    if args.policy_type != "mlp" and args.temporal_feature_mode != "legacy_progress_phase":
        raise ValueError("temporal feature modes apply only to MLP policies")
    manifest = ExperimentManifest.start(
        repo_root=PROJECT_ROOT,
        argv=command,
        seeds={
            "training_seeds": [int(seed) for seed in seed_values],
            "demo_repeats": args.demo_repeats,
        },
        metadata={"evaluation_protocol": protocol_metadata},
    )
    output_dir = args.output_dir
    demo_dir = args.demo_dir or output_dir / "scripted_pickup_demos"
    model_dir = output_dir / "models"
    result_dir = output_dir / "eval"
    demo_paths = generate_demos(
        demo_dir,
        train_specs,
        evaluation_protocol=protocol_metadata,
    )

    all_results = []
    per_policy = {}
    generated_outputs: list[Path] = [demo_dir / "manifest.json", *demo_paths]
    for action_space in ACTION_SPACES:
        action_gain = (
            args.joint_action_gain
            if action_space == "joint_delta" and args.joint_action_gain is not None
            else args.ee_action_gain
            if action_space == "ee_tool_delta" and args.ee_action_gain is not None
            else args.action_gain
        )
        dataset = load_demo_dataset(
            demo_paths,
            action_space=action_space,
            success_only=not args.include_failed_demos,
            stride=args.stride,
            label_source=args.label_source,
        )
        policy_results = []
        policy_result_records = []
        seed_summaries = []
        training_summaries = []

        for seed in seed_values:
            seed_suffix = f"_seed_{seed}" if len(seed_values) > 1 else ""
            if args.policy_type == "nearest":
                policy = fit_nearest_neighbor_policy(
                    dataset,
                    k=args.k,
                    temperature=args.temperature,
                )
                fit_summary = {
                    "policy_type": "grouped_nearest_neighbor_bc",
                    "k": args.k,
                    "temperature": args.temperature,
                    "seed": int(seed),
                }
                model_path = model_dir / f"{action_space}_nearest_neighbor_bc{seed_suffix}.npz"
            else:
                policy, fit_summary = fit_mlp_policy(
                    dataset,
                    hidden_sizes=tuple(args.hidden_sizes),
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    weight_decay=args.weight_decay,
                    seed=seed,
                    temporal_feature_mode=args.temporal_feature_mode,
                    gripper_loss_weight=args.gripper_loss_weight,
                    close_phase_gripper_weight=args.close_phase_gripper_weight,
                )
                model_path = model_dir / f"{action_space}_mlp_bc{seed_suffix}.npz"
            policy.evaluation_config_hash = (
                str(protocol_metadata["config_sha256"]) if protocol_metadata else ""
            )
            policy.evaluation_protocol_version = (
                int(protocol_metadata["version"]) if protocol_metadata else 0
            )
            policy.save(model_path)

            train_summary = {
                "action_space": action_space,
                **fit_summary,
                "model_path": str(model_path),
                "demo_count": dataset.demo_count,
                "skipped_demos": dataset.skipped_demos,
                "sample_count": int(dataset.features.shape[0]),
                "feature_count": int(dataset.features.shape[1]),
                "action_size": int(dataset.actions.shape[1]),
                "stride": args.stride,
                "search_window": args.search_window,
                "action_gain": action_gain,
                "eval_mode": args.eval_mode,
                "eval_split": args.eval_split,
                "evaluation_protocol": protocol_metadata,
                "evaluation_config_hash": (
                    protocol_metadata["config_sha256"] if protocol_metadata else None
                ),
                "registered_eval_total": registered_eval_total,
                "evaluated_spec_count": len(eval_specs),
                "explicit_eval_limit": args.eval_limit,
                "train_grid": args.train_grid,
                "label_source": dataset.label_source,
                "feature_names": dataset.feature_names,
                "source_paths": dataset.source_paths,
                "task_context_note": (
                    "Numeric yaw and approach context is included because identical "
                    "robot/object states map to different scripted variants."
                ),
                "policy_limit_note": (
                    "Nearest-neighbor is a replay-style baseline; MLP is a compact "
                    "state BC baseline using its recorded input contract. Both must be judged by held-out "
                    "MuJoCo rollout success, not by supervised loss alone."
                ),
            }
            training_summaries.append(train_summary)
            training_summary_path = output_dir / f"{action_space}{seed_suffix}_training_summary.json"
            write_json(training_summary_path, train_summary)
            generated_outputs.extend([model_path, training_summary_path])

            env = PickupTaskEvaluator()
            seed_results = []
            skipped_specs = []
            available_contexts = set(policy.group_keys)
            missing_contexts = sorted(
                {
                    TaskContext.from_spec(spec).key
                    for spec in eval_specs
                    if TaskContext.from_spec(spec).key not in available_contexts
                }
            )
            if protocol is not None and missing_contexts:
                raise RuntimeError(
                    "v2 evaluation denominator is incomplete; no successful training demo "
                    f"for contexts: {missing_contexts}"
                )
            for spec in eval_specs:
                context = TaskContext.from_spec(spec)
                if context.key not in available_contexts:
                    skipped_specs.append(
                        {
                            "trial_id": spec.trial_id,
                            "orientation": spec.orientation.label,
                            "object_pose": spec.object_pose.label,
                            "approach": spec.approach.label,
                            "repeat": spec.repeat,
                            "reason": "no successful training demo for this context",
                        }
                    )
                    continue
                result = rollout_policy(
                    env,
                    policy,
                    spec,
                    max_steps=args.max_steps,
                    search_window=args.search_window,
                    action_gain=action_gain,
                    gripper_close_guard=args.gripper_close_guard,
                )
                seed_results.append(result)
                policy_results.append(result)
                all_results.append(result)
                record = result.to_dict()
                record["seed"] = int(seed)
                record["evaluation_protocol"] = protocol_metadata
                record["evaluation_config_hash"] = (
                    protocol_metadata["config_sha256"] if protocol_metadata else None
                )
                policy_result_records.append(record)
                print(
                    f"policy={action_space} seed={seed} trial={result.trial_id:02d} "
                    f"success={int(result.success)} failure={result.failure_category} "
                    f"steps={result.steps} lift={result.max_object_lift:.3f} "
                    f"clip_j={result.clipped_joint_steps}"
                )

            seed_summary = summarize_policy_results(seed_results)
            seed_summary["seed"] = int(seed)
            seed_summary["skipped_eval_specs"] = skipped_specs
            seed_summary["evaluation_protocol"] = protocol_metadata
            seed_summary["evaluation_config_hash"] = (
                protocol_metadata["config_sha256"] if protocol_metadata else None
            )
            if protocol is not None and len(seed_results) != len(eval_specs):
                raise RuntimeError(
                    f"v2 evaluation produced {len(seed_results)} results for {len(eval_specs)} specs"
                )
            seed_summary["training_summary"] = train_summary
            seed_summaries.append(seed_summary)

        result_path = result_dir / f"{action_space}_policy_trials.jsonl"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        with result_path.open("w", encoding="utf-8") as handle:
            for record in policy_result_records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        summary = summarize_policy_results(policy_results)
        summary["seeds"] = [int(seed) for seed in seed_values]
        summary["seed_summaries"] = seed_summaries
        summary["training_summaries"] = training_summaries
        summary["evaluation_protocol"] = protocol_metadata
        summary["evaluation_config_hash"] = (
            protocol_metadata["config_sha256"] if protocol_metadata else None
        )
        result_summary_path = result_path.with_suffix(".summary.json")
        write_json(result_summary_path, summary)
        generated_outputs.extend([result_path, result_summary_path])
        per_policy[action_space] = summary

    combined = summarize_policy_results(all_results)
    combined["policies"] = per_policy
    combined["demo_dir"] = str(demo_dir)
    combined["output_dir"] = str(output_dir)
    combined["seeds"] = [int(seed) for seed in seed_values]
    combined["evaluation_protocol"] = protocol_metadata
    combined["evaluation_config_hash"] = (
        protocol_metadata["config_sha256"] if protocol_metadata else None
    )
    combined["eval_split"] = args.eval_split
    combined["eval_spec_count_per_seed"] = len(eval_specs)
    combined["registered_eval_total_per_seed"] = registered_eval_total
    combined["explicit_eval_limit"] = args.eval_limit
    combined["temporal_feature_mode"] = args.temporal_feature_mode
    combined["gripper_loss_weight"] = float(args.gripper_loss_weight)
    combined["close_phase_gripper_weight"] = (
        None
        if args.close_phase_gripper_weight is None
        else float(args.close_phase_gripper_weight)
    )
    combined["shielded_policy"] = bool(args.gripper_close_guard)
    combined["action_gains"] = {
        "joint_delta": args.joint_action_gain
        if args.joint_action_gain is not None
        else args.action_gain,
        "ee_tool_delta": args.ee_action_gain
        if args.ee_action_gain is not None
        else args.action_gain,
    }
    summary_path = output_dir / "state_bc_summary.json"
    write_json(summary_path, combined)
    generated_outputs.append(summary_path)
    manifest.add_outputs(generated_outputs)
    manifest_path = manifest.write_sidecar(summary_path)
    print(json.dumps(combined, indent=2, sort_keys=True))
    print(f"wrote {manifest_path}")
    return combined


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "state_bc",
    )
    parser.add_argument("--demo-dir", type=Path, default=None)
    parser.add_argument("--demo-repeats", type=int, default=1)
    parser.add_argument("--train-grid", choices=("default", "dense"), default="default")
    parser.add_argument("--eval-repeats", type=int, default=1)
    parser.add_argument(
        "--evaluation-protocol",
        choices=("legacy", "v2"),
        default="legacy",
        help="Use the tracked v2 protocol or preserve a historical legacy grid.",
    )
    parser.add_argument(
        "--eval-split",
        choices=("train", "validation", "final"),
        default=None,
        help="Required for v2. The new final holdout is only accessible via this explicit flag.",
    )
    parser.add_argument(
        "--eval-mode",
        choices=("train", "heldout", "test", "audit", "final", "both"),
        default="both",
    )
    parser.add_argument("--max-steps", type=int, default=3200)
    parser.add_argument("--policy-type", choices=("nearest", "mlp"), default="mlp")
    parser.add_argument(
        "--temporal-feature-mode",
        choices=("legacy_progress_phase", "none", "env_derived_phase"),
        default="legacy_progress_phase",
        help=(
            "MLP input contract; none retrains without cursor/progress/phase and adds "
            "distance; env_derived_phase keeps the legacy progress/phase layout but "
            "sources phase from live contact/lift/distance at train and rollout."
        ),
    )
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.75)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Train/evaluate one policy per seed; defaults to the protocol's five seeds "
            "for v2 and --seed for legacy."
        ),
    )
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--search-window", type=int, default=5)
    parser.add_argument(
        "--action-gain",
        type=float,
        default=1.0,
        help="Scale learned delta actions during rollout while leaving gripper commands unchanged.",
    )
    parser.add_argument("--joint-action-gain", type=float, default=None)
    parser.add_argument("--ee-action-gain", type=float, default=None)
    parser.add_argument(
        "--gripper-close-guard",
        action="store_true",
        help="Diagnostic shield: suppress close while farther than EARLY_CLOSE_DISTANCE.",
    )
    parser.add_argument(
        "--eval-limit",
        type=int,
        default=None,
        help="Explicit diagnostic-only prefix limit; recorded in outputs and never implicit.",
    )
    parser.add_argument(
        "--label-source",
        choices=("policy_labels", "labels"),
        default="policy_labels",
        help="Use executable policy labels by default; raw labels are observed transitions.",
    )
    parser.add_argument(
        "--gripper-loss-weight",
        type=float,
        default=1.0,
        help=(
            "H-EE-008: multiply MSE on the gripper action dim. Default 1.0 preserves "
            "uniform MSE; use >1 to upweight gripper timing."
        ),
    )
    parser.add_argument(
        "--close-phase-gripper-weight",
        type=float,
        default=None,
        help=(
            "H-EE-008: optional override gripper MSE weight on demo phases "
            "grasp_align and close_gripper. Default None uses --gripper-loss-weight."
        ),
    )
    parser.add_argument(
        "--include-failed-demos",
        action="store_true",
        help="Include scripted demos that failed the pickup success criteria.",
    )
    args = parser.parse_args()
    run(args, command=sys.argv)


if __name__ == "__main__":
    main()
