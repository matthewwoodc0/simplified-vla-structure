import json

import numpy as np

from svla.demo_recorder import PickupDemoRecorder
from svla.pickup_task import PickupTaskEvaluator, default_trial_specs
from svla.state_bc import (
    TaskContext,
    apply_gripper_close_guard,
    fit_mlp_policy,
    fit_nearest_neighbor_policy,
    load_demo_dataset,
    load_policy,
    observation_feature_names,
    observation_to_features,
    rollout_policy,
)


def test_observation_features_include_numeric_task_context():
    env = PickupTaskEvaluator()
    spec = default_trial_specs(repeats=1)[0]
    observation = env.reset(spec.object_pose.xyz)
    context = TaskContext.from_spec(spec)

    features = observation_to_features(observation, context, env.object_position.copy())

    assert features.shape == (len(observation_feature_names()),)
    assert np.isfinite(features).all()
    assert np.allclose(features[-4:-2], [np.sin(np.deg2rad(-18.0)), np.cos(np.deg2rad(-18.0))])


def test_nearest_neighbor_bc_fits_saved_demo_labels(tmp_path):
    recorder = PickupDemoRecorder(PickupTaskEvaluator())
    spec = default_trial_specs(repeats=1)[0]
    demo_path = tmp_path / "demo.json"
    recorder.write_trial(spec, demo_path)

    dataset = load_demo_dataset([demo_path], action_space="ee_tool_delta")
    policy = fit_nearest_neighbor_policy(dataset, k=1)
    reloaded_path = tmp_path / "policy.npz"
    policy.save(reloaded_path)
    policy = load_policy(reloaded_path)

    demo = json.loads(demo_path.read_text(encoding="utf-8"))
    sample = demo["samples"][10]
    context = TaskContext.from_demo(demo)
    features = observation_to_features(
        sample["observation"],
        context,
        np.asarray(demo["summary"]["object_start_pose"], dtype=float),
    )
    action, distance, _ = policy.predict_with_index(
        features,
        context.key,
        cursor=sample["step_index"],
        search_window=1,
    )

    assert distance < 1e-9
    assert action.shape == (6,)
    assert np.allclose(action, sample["policy_labels"]["ee_tool_delta"], atol=1e-9)


def test_mlp_bc_saves_loads_and_predicts(tmp_path):
    recorder = PickupDemoRecorder(PickupTaskEvaluator())
    spec = default_trial_specs(repeats=1)[0]
    demo_path = tmp_path / "demo.json"
    recorder.write_trial(spec, demo_path)

    dataset = load_demo_dataset([demo_path], action_space="joint_delta", stride=25)
    policy, summary = fit_mlp_policy(
        dataset,
        hidden_sizes=(8,),
        epochs=2,
        batch_size=64,
        seed=7,
    )
    policy_path = tmp_path / "mlp_policy.npz"
    policy.save(policy_path)
    policy = load_policy(policy_path)

    demo = json.loads(demo_path.read_text(encoding="utf-8"))
    sample = demo["samples"][0]
    context = TaskContext.from_demo(demo)
    features = observation_to_features(
        sample["observation"],
        context,
        np.asarray(demo["summary"]["object_start_pose"], dtype=float),
    )
    action, distance, index = policy.predict_with_index(features, context.key, cursor=0)

    assert summary["policy_type"] == "mlp_bc"
    assert action.shape == (6,)
    assert np.isfinite(action).all()
    assert distance == 0.0
    assert index == 0


def test_mlp_temporal_modes_round_trip_and_none_is_cursor_invariant(tmp_path):
    recorder = PickupDemoRecorder(PickupTaskEvaluator())
    spec = default_trial_specs(repeats=1)[0]
    demo_path = tmp_path / "demo.json"
    recorder.write_trial(spec, demo_path)
    demo = json.loads(demo_path.read_text(encoding="utf-8"))
    context = TaskContext.from_demo(demo)
    sample = demo["samples"][20]
    features = observation_to_features(
        sample["observation"],
        context,
        np.asarray(demo["summary"]["object_start_pose"], dtype=float),
    )

    for action_space in ("joint_delta", "ee_tool_delta"):
        dataset = load_demo_dataset([demo_path], action_space=action_space, stride=25)
        for mode, expected_dimension in (("legacy_progress_phase", 50), ("none", 36)):
            policy, summary = fit_mlp_policy(
                dataset,
                hidden_sizes=(8,),
                epochs=1,
                batch_size=64,
                seed=3,
                temporal_feature_mode=mode,
            )
            path = tmp_path / f"{action_space}_{mode}.npz"
            policy.save(path)
            reloaded = load_policy(path)
            action_0, _, _ = reloaded.predict_with_index(features, context.key, cursor=0)
            action_99, _, _ = reloaded.predict_with_index(features, context.key, cursor=99)

            assert reloaded.temporal_feature_mode == mode
            assert summary["policy_input_dimension"] == expected_dimension
            assert len(summary["policy_feature_names"]) == expected_dimension
            assert action_0.shape == (6,)
            assert np.isfinite(action_0).all()
            if mode == "none":
                assert np.array_equal(action_0, action_99)


def test_loading_pre_metadata_mlp_defaults_to_legacy_mode(tmp_path):
    recorder = PickupDemoRecorder(PickupTaskEvaluator())
    spec = default_trial_specs(repeats=1)[0]
    demo_path = tmp_path / "demo.json"
    recorder.write_trial(spec, demo_path)
    dataset = load_demo_dataset([demo_path], action_space="joint_delta", stride=40)
    policy, _ = fit_mlp_policy(dataset, hidden_sizes=(4,), epochs=1, batch_size=64)
    modern_path = tmp_path / "modern.npz"
    legacy_path = tmp_path / "legacy.npz"
    policy.save(modern_path)
    with np.load(modern_path, allow_pickle=False) as data:
        removed = {
            "temporal_feature_mode",
            "distance_mean",
            "distance_scale",
            "base_feature_names",
            "policy_feature_names",
        }
        np.savez_compressed(legacy_path, **{key: data[key] for key in data.files if key not in removed})

    reloaded = load_policy(legacy_path)

    assert reloaded.temporal_feature_mode == "legacy_progress_phase"
    assert len(reloaded.policy_feature_names) == 50


def test_gripper_close_guard_is_symmetric_default_off_and_non_mutating():
    for _action_space in ("joint_delta", "ee_tool_delta"):
        raw = np.array([0.1, -0.2, 0.0, 0.0, 0.0, 0.0])
        original = raw.copy()

        unguarded, suppressed, started = apply_gripper_close_guard(
            raw,
            enabled=False,
            closure_started=False,
            gripper_object_distance=1.0,
        )
        guarded, guarded_suppressed, guarded_started = apply_gripper_close_guard(
            raw,
            enabled=True,
            closure_started=False,
            gripper_object_distance=1.0,
        )
        legal, legal_suppressed, legal_started = apply_gripper_close_guard(
            raw,
            enabled=True,
            closure_started=False,
            gripper_object_distance=0.01,
        )

        assert np.array_equal(raw, original)
        assert np.array_equal(unguarded, raw)
        assert not suppressed and not started
        assert guarded[-1] == 1.0
        assert guarded_suppressed and not guarded_started
        assert np.array_equal(legal, raw)
        assert not legal_suppressed and legal_started


def test_gripper_close_guard_does_not_suppress_reopen_after_legal_close():
    reopen = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

    executed, suppressed, closure_started = apply_gripper_close_guard(
        reopen,
        enabled=True,
        closure_started=True,
        gripper_object_distance=1.0,
    )

    assert np.array_equal(executed, reopen)
    assert not suppressed
    assert closure_started


def test_rollout_records_raw_and_executed_guard_metrics_for_both_action_spaces():
    spec = default_trial_specs(repeats=1)[0]

    class AlwaysClosePolicy:
        def __init__(self, action_space):
            self.action_space = action_space
            self.group_keys = [TaskContext.from_spec(spec).key]

        def predict_with_index(self, features, group_key, cursor=None, search_window=None):
            del features, group_key, search_window
            return np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0]), 0.0, int(cursor or 0)

    for action_space in ("joint_delta", "ee_tool_delta"):
        guarded = rollout_policy(
            PickupTaskEvaluator(),
            AlwaysClosePolicy(action_space),
            spec,
            max_steps=1,
            gripper_close_guard=True,
        )
        unguarded = rollout_policy(
            PickupTaskEvaluator(),
            AlwaysClosePolicy(action_space),
            spec,
            max_steps=1,
            gripper_close_guard=False,
        )

        assert guarded.shielded_policy
        assert guarded.suppressed_close_steps == 1
        assert guarded.raw_action_l2_mean != guarded.executed_action_l2_mean
        assert not unguarded.shielded_policy
        assert unguarded.suppressed_close_steps == 0
        assert unguarded.raw_action_l2_mean == unguarded.executed_action_l2_mean
