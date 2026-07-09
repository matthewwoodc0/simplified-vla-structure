from __future__ import annotations

from svla.demo_recorder import PickupDemoRecorder
from svla.pickup_task import PickupTaskEvaluator, default_trial_specs
from svla.state_bc import fit_mlp_policy, load_demo_dataset, rollout_policy


def test_tiny_train_and_closed_loop_eval_step(tmp_path):
    """One tiny train + eval step catches wiring drift without claiming efficacy."""

    spec = default_trial_specs(repeats=1)[0]
    demo_path = tmp_path / "demo.json"
    PickupDemoRecorder(PickupTaskEvaluator()).write_trial(spec, demo_path)
    dataset = load_demo_dataset(
        [demo_path],
        action_space="ee_tool_delta",
        stride=40,
    )
    policy, training = fit_mlp_policy(
        dataset,
        hidden_sizes=(8,),
        epochs=1,
        batch_size=32,
        seed=7,
    )
    result = rollout_policy(PickupTaskEvaluator(), policy, spec, max_steps=2)

    assert training["epochs"] == 1
    assert result.steps >= 1
    assert result.action_space == "ee_tool_delta"
