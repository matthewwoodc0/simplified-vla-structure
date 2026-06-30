from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from svla.demo_recorder import PickupDemoRecorder
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


def generate_demos(output_dir: Path, specs: list[PickupTrialSpec]) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    recorder = PickupDemoRecorder(PickupTaskEvaluator())
    demos = []
    for spec in specs:
        path = output_dir / (
            f"pickup_demo_{spec.trial_id:03d}_{spec.orientation.label}_"
            f"{spec.object_pose.label}_{spec.approach.label}_repeat_{spec.repeat}.json"
        )
        demo = recorder.write_trial(spec, path)
        demos.append({"path": str(path), "summary": demo["summary"]})
        print(
            f"demo={path.name} success={int(demo['summary']['success'])} "
            f"failure={demo['summary']['failure_category']} samples={len(demo['samples'])}"
        )
    manifest = {
        "format": "svla_state_bc_demo_manifest_v1",
        "demo_count": len(demos),
        "demos": demos,
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
    if mode == "both":
        return training_specs(train_grid, repeats=repeats) + heldout_trial_specs()
    raise ValueError(f"unknown eval mode: {mode}")


def run(args: argparse.Namespace) -> dict:
    output_dir = args.output_dir
    demo_dir = args.demo_dir or output_dir / "scripted_pickup_demos"
    model_dir = output_dir / "models"
    result_dir = output_dir / "eval"
    seed_values = list(args.seeds) if args.seeds is not None else [args.seed]
    demo_paths = generate_demos(
        demo_dir,
        training_specs(args.train_grid, repeats=args.demo_repeats),
    )

    all_results = []
    per_policy = {}
    for action_space in ACTION_SPACES:
        action_gain = (
            args.joint_action_gain
            if action_space == "joint_delta" and args.joint_action_gain is not None
            else args.ee_action_gain
            if action_space == "ee_delta" and args.ee_action_gain is not None
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
                )
                model_path = model_dir / f"{action_space}_mlp_bc{seed_suffix}.npz"
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
                    "state/progress parametric baseline. Both must be judged by held-out "
                    "MuJoCo rollout success, not by supervised loss alone."
                ),
            }
            training_summaries.append(train_summary)
            write_json(output_dir / f"{action_space}{seed_suffix}_training_summary.json", train_summary)

            env = PickupTaskEvaluator()
            seed_results = []
            skipped_specs = []
            available_contexts = set(policy.group_keys)
            for spec in evaluation_specs(
                args.eval_mode,
                train_grid=args.train_grid,
                repeats=args.eval_repeats,
            ):
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
                )
                seed_results.append(result)
                policy_results.append(result)
                all_results.append(result)
                record = result.to_dict()
                record["seed"] = int(seed)
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
        write_json(result_path.with_suffix(".summary.json"), summary)
        per_policy[action_space] = summary

    combined = summarize_policy_results(all_results)
    combined["policies"] = per_policy
    combined["demo_dir"] = str(demo_dir)
    combined["output_dir"] = str(output_dir)
    combined["seeds"] = [int(seed) for seed in seed_values]
    combined["action_gains"] = {
        "joint_delta": args.joint_action_gain
        if args.joint_action_gain is not None
        else args.action_gain,
        "ee_delta": args.ee_action_gain
        if args.ee_action_gain is not None
        else args.action_gain,
    }
    write_json(output_dir / "state_bc_summary.json", combined)
    print(json.dumps(combined, indent=2, sort_keys=True))
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
        "--eval-mode",
        choices=("train", "heldout", "test", "audit", "both"),
        default="both",
    )
    parser.add_argument("--max-steps", type=int, default=3200)
    parser.add_argument("--policy-type", choices=("nearest", "mlp"), default="mlp")
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
        help="Train/evaluate one policy per seed; defaults to --seed.",
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
        "--label-source",
        choices=("policy_labels", "labels"),
        default="policy_labels",
        help="Use executable policy labels by default; raw labels are observed transitions.",
    )
    parser.add_argument(
        "--include-failed-demos",
        action="store_true",
        help="Include scripted demos that failed the pickup success criteria.",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
