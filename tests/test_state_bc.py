import json

import numpy as np

from svla.demo_recorder import PickupDemoRecorder
from svla.pickup_task import PickupTaskEvaluator, default_trial_specs
from svla.state_bc import (
    TaskContext,
    fit_mlp_policy,
    fit_nearest_neighbor_policy,
    load_demo_dataset,
    load_policy,
    observation_feature_names,
    observation_to_features,
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

    dataset = load_demo_dataset([demo_path], action_space="ee_delta")
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
    assert action.shape == (7,)
    assert np.allclose(action, sample["policy_labels"]["ee_delta"], atol=1e-9)


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
