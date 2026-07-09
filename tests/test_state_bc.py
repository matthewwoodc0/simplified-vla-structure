import json

import numpy as np

from svla.demo_recorder import PickupDemoRecorder
from svla.pickup_task import PickupTaskEvaluator, default_trial_specs
from svla.state_bc import (
    CLOSE_TIMING_PHASES,
    HYBRID_POLICY_TYPE,
    MATCH_FEATURE_INDICES,
    MATCH_FEATURE_NAMES,
    HybridNNGripperMLPPolicy,
    PHASE_LABELS,
    TaskContext,
    action_loss_weights,
    apply_gripper_close_guard,
    estimate_env_phase,
    fit_mlp_policy,
    fit_nearest_neighbor_policy,
    load_demo_dataset,
    load_policy,
    observation_feature_names,
    observation_to_features,
    rollout_policy,
)
from svla.pickup_task import EARLY_CLOSE_DISTANCE


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
        for mode, expected_dimension in (
            ("legacy_progress_phase", 50),
            ("none", 36),
            ("env_derived_phase", 50),
        ):
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
            if mode in ("none", "env_derived_phase"):
                # Phase comes from observation state, not open-loop cursor.
                assert np.array_equal(action_0, action_99)


def test_estimate_env_phase_uses_distance_contact_lift_bins():
    names = observation_feature_names()
    features = np.zeros(len(names), dtype=float)
    features[names.index("gripper_open")] = 1.0
    features[names.index("object_support_contact")] = 1.0

    # Far approach.
    features[names.index("object_minus_ee_z")] = 0.09
    phase_index, progress = estimate_env_phase(features)
    assert PHASE_LABELS[phase_index] == "approach_0"
    assert 0.0 <= progress <= 1.0

    # Grasp-align ball.
    features[names.index("object_minus_ee_z")] = EARLY_CLOSE_DISTANCE * 0.5
    phase_index, progress = estimate_env_phase(features)
    assert PHASE_LABELS[phase_index] == "grasp_align"

    # Close: near object and jaw not fully open.
    features[names.index("gripper_open")] = 0.4
    phase_index, progress = estimate_env_phase(features)
    assert PHASE_LABELS[phase_index] == "close_gripper"

    # Lift: contact + closed grip + rising object.
    features[names.index("gripper_object_contact")] = 1.0
    features[names.index("object_lift_from_start")] = 0.01
    phase_index, progress = estimate_env_phase(features)
    assert PHASE_LABELS[phase_index] == "lift"

    # Hold height.
    features[names.index("object_lift_from_start")] = 0.032
    phase_index, progress = estimate_env_phase(features)
    assert PHASE_LABELS[phase_index] == "hold"


def test_estimate_env_phase_does_not_label_far_opening_jaw_as_close():
    """Home jaw may be partially open while still far from the object."""

    names = observation_feature_names()
    features = np.zeros(len(names), dtype=float)
    features[names.index("gripper_open")] = 0.75
    features[names.index("object_minus_ee_z")] = 0.09
    features[names.index("object_support_contact")] = 1.0

    phase_index, _ = estimate_env_phase(features)
    assert PHASE_LABELS[phase_index] == "approach_0"


def test_action_loss_weights_upweight_gripper_and_close_phases():
    phase_indices = np.array(
        [
            PHASE_LABELS.index("approach_0"),
            PHASE_LABELS.index("grasp_align"),
            PHASE_LABELS.index("close_gripper"),
            PHASE_LABELS.index("lift"),
        ],
        dtype=int,
    )
    weights = action_loss_weights(
        phase_indices,
        action_size=6,
        gripper_loss_weight=5.0,
        close_phase_gripper_weight=10.0,
    )

    assert weights.shape == (4, 6)
    assert np.allclose(weights[:, :5], 1.0)
    assert weights[0, 5] == 5.0
    assert weights[1, 5] == 10.0
    assert weights[2, 5] == 10.0
    assert weights[3, 5] == 5.0
    assert set(CLOSE_TIMING_PHASES) == {"grasp_align", "close_gripper"}


def test_gripper_weighted_mlp_fits_and_reports_weights(tmp_path):
    recorder = PickupDemoRecorder(PickupTaskEvaluator())
    spec = default_trial_specs(repeats=1)[0]
    demo_path = tmp_path / "demo.json"
    recorder.write_trial(spec, demo_path)
    dataset = load_demo_dataset([demo_path], action_space="ee_tool_delta", stride=40)

    uniform, uniform_summary = fit_mlp_policy(
        dataset,
        hidden_sizes=(8,),
        epochs=2,
        batch_size=64,
        seed=1,
        gripper_loss_weight=1.0,
        close_phase_gripper_weight=None,
    )
    weighted, weighted_summary = fit_mlp_policy(
        dataset,
        hidden_sizes=(8,),
        epochs=2,
        batch_size=64,
        seed=1,
        gripper_loss_weight=5.0,
        close_phase_gripper_weight=10.0,
    )

    assert uniform_summary["gripper_loss_weight"] == 1.0
    assert uniform_summary["close_phase_gripper_weight"] is None
    assert weighted_summary["gripper_loss_weight"] == 5.0
    assert weighted_summary["close_phase_gripper_weight"] == 10.0
    assert weighted_summary["close_timing_sample_count"] > 0
    assert "train_mse_weighted" in weighted_summary
    # Same seed + different loss should not produce identical weights.
    assert not np.allclose(uniform.weights[-1], weighted.weights[-1])


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


def test_hybrid_nn_gripper_composes_mlp_arm_and_nn_gripper(tmp_path):
    """H-EE-014: arm dims from MLP, gripper from NN; MLP weights unchanged."""

    recorder = PickupDemoRecorder(PickupTaskEvaluator())
    spec = default_trial_specs(repeats=1)[0]
    demo_path = tmp_path / "demo.json"
    recorder.write_trial(spec, demo_path)
    dataset = load_demo_dataset([demo_path], action_space="ee_tool_delta", stride=20)
    mlp, _ = fit_mlp_policy(dataset, hidden_sizes=(8,), epochs=2, batch_size=64, seed=2)
    nn = fit_nearest_neighbor_policy(dataset, k=1)
    mlp_weight_snapshot = [w.copy() for w in mlp.weights]
    mlp_bias_snapshot = [b.copy() for b in mlp.biases]

    hybrid = HybridNNGripperMLPPolicy(mlp, nn)
    demo = json.loads(demo_path.read_text(encoding="utf-8"))
    sample = demo["samples"][15]
    context = TaskContext.from_demo(demo)
    features = observation_to_features(
        sample["observation"],
        context,
        np.asarray(demo["summary"]["object_start_pose"], dtype=float),
    )
    cursor = int(sample["step_index"])
    a_mlp, _, mlp_idx = mlp.predict_with_index(features, context.key, cursor=cursor)
    a_nn, nn_dist, _ = nn.predict_with_index(
        features, context.key, cursor=cursor, search_window=1
    )
    a_hybrid, hybrid_dist, hybrid_idx = hybrid.predict_with_index(
        features, context.key, cursor=cursor, search_window=1
    )

    assert a_hybrid.shape == (6,)
    assert np.allclose(a_hybrid[:5], a_mlp[:5])
    assert np.isclose(a_hybrid[-1], a_nn[-1])
    assert np.isclose(hybrid_dist, nn_dist)
    assert hybrid_idx == mlp_idx
    # Compositor must not mutate trained MLP parameters.
    for original, current in zip(mlp_weight_snapshot, mlp.weights):
        assert np.array_equal(original, current)
    for original, current in zip(mlp_bias_snapshot, mlp.biases):
        assert np.array_equal(original, current)

    manifest_path = hybrid.save(tmp_path / "hybrid_policy.json")
    reloaded = load_policy(manifest_path)
    assert isinstance(reloaded, HybridNNGripperMLPPolicy)
    a_reload, _, _ = reloaded.predict_with_index(
        features, context.key, cursor=cursor, search_window=1
    )
    assert np.allclose(a_reload, a_hybrid)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["policy_type"] == HYBRID_POLICY_TYPE
    assert manifest["match_feature_names"] == list(MATCH_FEATURE_NAMES)
    assert manifest["match_feature_indices"] == [int(i) for i in MATCH_FEATURE_INDICES]
    assert manifest["recipe"] == "A1_compositor"


def test_hybrid_nn_gripper_rollout_both_action_spaces(tmp_path):
    recorder = PickupDemoRecorder(PickupTaskEvaluator())
    spec = default_trial_specs(repeats=1)[0]
    demo_path = tmp_path / "demo.json"
    recorder.write_trial(spec, demo_path)

    for action_space in ("joint_delta", "ee_tool_delta"):
        dataset = load_demo_dataset([demo_path], action_space=action_space, stride=30)
        mlp, _ = fit_mlp_policy(
            dataset, hidden_sizes=(8,), epochs=1, batch_size=64, seed=0
        )
        nn = fit_nearest_neighbor_policy(dataset, k=1)
        hybrid = HybridNNGripperMLPPolicy(mlp, nn)
        result = rollout_policy(
            PickupTaskEvaluator(),
            hybrid,
            spec,
            max_steps=8,
            search_window=5,
        )
        assert result.action_space == action_space
        assert result.steps >= 1
        assert np.isfinite(result.nearest_distance_mean)


def test_match_relative_ee_contract_is_named_and_does_not_change_default():
    from svla.state_bc import (
        DEFAULT_MATCH_CONTRACT,
        MATCH_CONTRACT_RELATIVE_EE,
        MATCH_FEATURE_INDICES,
        resolve_match_contract,
    )

    hist_idx, hist_names = resolve_match_contract(DEFAULT_MATCH_CONTRACT)
    assert np.array_equal(hist_idx, MATCH_FEATURE_INDICES)
    rel_idx, rel_names = resolve_match_contract(MATCH_CONTRACT_RELATIVE_EE)
    assert "object_minus_ee_x" in rel_names
    assert "gripper_open" in rel_names
    assert "object_x" not in rel_names
    assert not np.array_equal(rel_idx, hist_idx)


def test_hybrid_set_match_contract_switches_nn_retrieval(tmp_path):
    from svla.state_bc import MATCH_CONTRACT_RELATIVE_EE

    recorder = PickupDemoRecorder(PickupTaskEvaluator())
    spec = default_trial_specs(repeats=1)[0]
    demo_path = tmp_path / "demo.json"
    recorder.write_trial(spec, demo_path)
    dataset = load_demo_dataset([demo_path], action_space="ee_tool_delta", stride=20)
    mlp, _ = fit_mlp_policy(dataset, hidden_sizes=(8,), epochs=1, batch_size=64, seed=0)
    nn = fit_nearest_neighbor_policy(dataset, k=1)
    hybrid = HybridNNGripperMLPPolicy(mlp, nn)
    assert hybrid.match_contract == "historical"
    hybrid.set_match_contract(MATCH_CONTRACT_RELATIVE_EE)
    assert hybrid.match_contract == MATCH_CONTRACT_RELATIVE_EE
    assert hybrid.nn.match_contract == MATCH_CONTRACT_RELATIVE_EE
    assert "object_minus_ee_x" in hybrid.match_feature_names
    # Rollout still produces finite gripper action under the secondary contract.
    result = rollout_policy(
        PickupTaskEvaluator(), hybrid, spec, max_steps=4, search_window=5
    )
    assert result.steps >= 1


def test_arm_only_mlp_zeros_gripper_loss_weight():
    from svla.state_bc import action_loss_weights

    phases = np.array([0, 3, 4, 5], dtype=int)  # includes grasp_align/close
    weights = action_loss_weights(phases, action_size=6, arm_only_mlp=True)
    assert weights.shape == (4, 6)
    assert np.allclose(weights[:, :5], 1.0)
    assert np.allclose(weights[:, 5], 0.0)
    # Non-arm-only still enforces positive gripper weight.
    uniform = action_loss_weights(phases, action_size=6, gripper_loss_weight=5.0)
    assert np.allclose(uniform[:, 5], 5.0)
