from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from svla.experiment_manifest import (
    ExperimentManifest,
    verify_manifest_identity_consistent,
    verify_manifest_output_hashes,
)
from svla.state_bc import write_json


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _max_pickup_rotation(output_dir: Path) -> float:
    jsonl_path = output_dir / "pickup_trials.jsonl"
    max_rotation = 0.0
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        trial = json.loads(line)
        max_rotation = max(
            max_rotation,
            float(trial.get("max_object_rotation_while_supported", 0.0)),
        )
    return max_rotation


def build(output_dir: Path) -> dict:
    output_dir = output_dir.resolve()
    manifest_paths = {
        "pytest": output_dir / "pytest_results.manifest.json",
        "pickup_trials": output_dir / "pickup_trials.manifest.json",
        "pickup_action_replay": output_dir / "pickup_action_replay_summary.manifest.json",
        "pick_place_trials": output_dir / "pick_place_trials.manifest.json",
        "pick_place_action_replay": output_dir / "pick_place_action_replay_summary.manifest.json",
        "readiness_robustness": output_dir / "task_robustness_readiness_summary.manifest.json",
    }
    summaries = {
        "pickup_trials": output_dir / "pickup_trials.summary.json",
        "pickup_action_replay": output_dir / "pickup_action_replay_summary.json",
        "pick_place_trials": output_dir / "pick_place_trials.summary.json",
        "pick_place_action_replay": output_dir / "pick_place_action_replay_summary.json",
        "readiness_robustness": output_dir / "task_robustness_readiness_summary.json",
    }

    component_manifests = [_load_json(path) for path in manifest_paths.values()]
    identity_consistent, identity_issues = verify_manifest_identity_consistent(
        component_manifests
    )
    output_hashes_verified, output_hash_issues = verify_manifest_output_hashes(
        PROJECT_ROOT,
        component_manifests,
    )

    pytest_manifest = component_manifests[0]
    pickup = _load_json(summaries["pickup_trials"])
    pickup_replay = _load_json(summaries["pickup_action_replay"])
    pick_place = _load_json(summaries["pick_place_trials"])
    pick_place_replay = _load_json(summaries["pick_place_action_replay"])
    readiness = _load_json(summaries["readiness_robustness"])

    replay_by_space = pickup_replay["by_action_space"]
    place_replay_by_space = pick_place_replay["by_action_space"]
    reference_identity = component_manifests[0]

    layer_pass = {
        "pytest": bool(pytest_manifest.get("pytest_passed")),
        "scripted_pickup_36": pickup["successes"] == pickup["total"] == 36,
        "pickup_policy_label_replay": bool(pickup_replay["pass"]),
        "scripted_pick_place_6": pick_place["successes"] == pick_place["total"] == 6,
        "pick_place_policy_label_replay": bool(pick_place_replay["pass"]),
        "readiness_robustness_288": bool(readiness["pass"]),
    }

    aggregate = {
        "format": "svla_phase5_baseline_v2_aggregate_v1",
        "scope_statement": (
            "This bundle proves scripted simulator/task readiness under strict Phase 5 "
            "physics gates for the recorded working tree. It does not prove learned-policy "
            "readiness, vision readiness, or hardware-calibrated realism."
        ),
        "evidence_dir": str(output_dir.relative_to(PROJECT_ROOT)),
        "evidence_paths": {
            "pytest_log": str((output_dir / "pytest_results.txt").relative_to(PROJECT_ROOT)),
            "pytest_manifest": str(manifest_paths["pytest"].relative_to(PROJECT_ROOT)),
            "pickup_trials_jsonl": str((output_dir / "pickup_trials.jsonl").relative_to(PROJECT_ROOT)),
            "pickup_trials_summary": str(summaries["pickup_trials"].relative_to(PROJECT_ROOT)),
            "pickup_trials_manifest": str(manifest_paths["pickup_trials"].relative_to(PROJECT_ROOT)),
            "pickup_action_replay_summary": str(summaries["pickup_action_replay"].relative_to(PROJECT_ROOT)),
            "pickup_action_replay_manifest": str(
                manifest_paths["pickup_action_replay"].relative_to(PROJECT_ROOT)
            ),
            "pick_place_trials_jsonl": str((output_dir / "pick_place_trials.jsonl").relative_to(PROJECT_ROOT)),
            "pick_place_trials_summary": str(summaries["pick_place_trials"].relative_to(PROJECT_ROOT)),
            "pick_place_trials_manifest": str(manifest_paths["pick_place_trials"].relative_to(PROJECT_ROOT)),
            "pick_place_action_replay_summary": str(
                summaries["pick_place_action_replay"].relative_to(PROJECT_ROOT)
            ),
            "pick_place_action_replay_manifest": str(
                manifest_paths["pick_place_action_replay"].relative_to(PROJECT_ROOT)
            ),
            "readiness_robustness_summary": str(
                summaries["readiness_robustness"].relative_to(PROJECT_ROOT)
            ),
            "readiness_robustness_manifest": str(
                manifest_paths["readiness_robustness"].relative_to(PROJECT_ROOT)
            ),
            "aggregate_manifest": str(
                (output_dir / "phase5_baseline_v2_aggregate.manifest.json").relative_to(PROJECT_ROOT)
            ),
        },
        "provenance": {
            "identity_consistent": identity_consistent,
            "identity_issues": identity_issues,
            "output_hashes_verified": output_hashes_verified,
            "output_hash_issues": output_hash_issues,
        },
        "git_code_identity": {
            "git_commit_sha": reference_identity["git_commit_sha"],
            "git_dirty": reference_identity["git_dirty"],
            "git_diff_sha256": reference_identity["git_diff_sha256"],
            "git_untracked_files": reference_identity["git_untracked_files"],
            "source_hashes": reference_identity["source_hashes"],
            "controller_limits": reference_identity["controller_limits"],
            "physics_gate_constants": reference_identity["physics_gate_constants"],
            "versions": reference_identity["versions"],
        },
        "layers": {
            "pytest": {
                "pass": layer_pass["pytest"],
                "exit_code": int(pytest_manifest.get("pytest_exit_code", 1)),
                "tests_passed": pytest_manifest.get("pytest_tests_passed"),
                "tests_failed": pytest_manifest.get("pytest_tests_failed"),
            },
            "scripted_pickup_36": {
                "pass": layer_pass["scripted_pickup_36"],
                "successes": pickup["successes"],
                "total": pickup["total"],
            },
            "pickup_policy_label_replay": {
                "pass": layer_pass["pickup_policy_label_replay"],
                "demo_count": pickup_replay["demo_count"],
                "joint_delta": {
                    "successes": replay_by_space["joint_delta"]["successes"],
                    "total": replay_by_space["joint_delta"]["total"],
                    "mean_saturation_rate": replay_by_space["joint_delta"]["mean_saturation_rate"],
                },
                "ee_tool_delta": {
                    "successes": replay_by_space["ee_tool_delta"]["successes"],
                    "total": replay_by_space["ee_tool_delta"]["total"],
                    "mean_saturation_rate": replay_by_space["ee_tool_delta"]["mean_saturation_rate"],
                },
            },
            "scripted_pick_place_6": {
                "pass": layer_pass["scripted_pick_place_6"],
                "successes": pick_place["successes"],
                "total": pick_place["total"],
            },
            "pick_place_policy_label_replay": {
                "pass": layer_pass["pick_place_policy_label_replay"],
                "demo_count": pick_place_replay["demo_count"],
                "joint_delta": {
                    "successes": place_replay_by_space["joint_delta"]["successes"],
                    "total": place_replay_by_space["joint_delta"]["total"],
                    "mean_saturation_rate": place_replay_by_space["joint_delta"]["mean_saturation_rate"],
                    "placement_achieved": place_replay_by_space["joint_delta"]["placement_achieved"],
                },
                "ee_tool_delta": {
                    "successes": place_replay_by_space["ee_tool_delta"]["successes"],
                    "total": place_replay_by_space["ee_tool_delta"]["total"],
                    "mean_saturation_rate": place_replay_by_space["ee_tool_delta"]["mean_saturation_rate"],
                    "placement_achieved": place_replay_by_space["ee_tool_delta"]["placement_achieved"],
                },
            },
            "readiness_robustness_288": {
                "pass": layer_pass["readiness_robustness_288"],
                "successes": readiness["successes"],
                "total": readiness["total"],
                "fixed_successes": readiness["fixed_successes"],
                "fixed_total": readiness["fixed_total"],
                "random_successes": readiness["random_successes"],
                "random_total": readiness["random_total"],
            },
        },
        "telemetry_maxima": {
            "pickup_trials": {
                "max_gripper_contact_force": pickup["max_gripper_contact_force"],
                "max_gripper_contact_impulse_before_lift": pickup[
                    "max_gripper_contact_impulse_before_lift"
                ],
                "max_object_xy_displacement_while_supported": pickup[
                    "max_object_xy_displacement_while_supported"
                ],
                "max_object_rotation_while_supported": _max_pickup_rotation(output_dir),
            },
            "pickup_action_replay": {
                "joint_delta_mean_saturation_rate": replay_by_space["joint_delta"][
                    "mean_saturation_rate"
                ],
                "ee_tool_delta_mean_saturation_rate": replay_by_space["ee_tool_delta"][
                    "mean_saturation_rate"
                ],
                "max_gripper_contact_force": max(
                    replay_by_space[space]["max_gripper_contact_force"] for space in replay_by_space
                ),
                "max_gripper_contact_impulse_before_lift": max(
                    replay_by_space[space]["max_gripper_contact_impulse_before_lift"]
                    for space in replay_by_space
                ),
                "max_object_xy_displacement_while_supported": max(
                    replay_by_space[space]["max_object_xy_displacement_while_supported"]
                    for space in replay_by_space
                ),
            },
            "readiness_robustness": {
                "max_gripper_contact_force": readiness["max_gripper_contact_force"],
                "max_gripper_contact_impulse_before_lift": readiness[
                    "max_gripper_contact_impulse_before_lift"
                ],
                "max_object_xy_displacement_while_supported": readiness[
                    "max_object_xy_displacement_while_supported"
                ],
                "max_object_rotation_while_supported": readiness[
                    "max_object_rotation_while_supported"
                ],
            },
        },
        "overall_pass": (
            identity_consistent
            and output_hashes_verified
            and all(layer_pass.values())
        ),
    }
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "phase5_baseline_v2",
    )
    args = parser.parse_args()
    aggregate = build(args.output_dir.resolve())
    aggregate_path = args.output_dir.resolve() / "phase5_baseline_v2_aggregate.json"
    write_json(aggregate_path, aggregate)
    manifest = ExperimentManifest.start(repo_root=PROJECT_ROOT, argv=sys.argv)
    manifest.add_outputs([aggregate_path])
    manifest.write_sidecar(aggregate_path)
    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()