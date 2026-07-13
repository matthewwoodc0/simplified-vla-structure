from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from svla.h_ee_002 import (
    EXPECTED_BASELINE_PRIMARY,
    SEEDS,
    align_paired_rows,
    classify_candidate,
    required_frozen_relative_paths,
    reproduction_check,
    verify_frozen_inputs,
)
from svla.state_bc import HybridNNGripperMLPPolicy


def _row(seed: int, trial_id: int, *, success: bool = False) -> dict:
    return {
        "seed": seed,
        "trial_id": trial_id,
        "success": success,
        "event_order_valid": success,
        "physical_sanity_pass": True,
        "early_close": False,
        "reopen_events": 0,
        "contact_achieved": True,
        "object_lifted": success,
        "failure_category": "none" if success else "event_order_failure",
        "joint_limit_clipped_steps": 10,
        "infeasible_steps": 10,
    }


def _constraint(rate: float) -> dict:
    return {
        "failure": {
            "joint_limit": {"exposure_rate": rate},
            "infeasible": {"exposure_rate": rate},
        }
    }


def test_paired_alignment_rejects_duplicate_and_missing_keys():
    baseline = [_row(0, 1), _row(0, 2)]
    with pytest.raises(ValueError, match="duplicate"):
        align_paired_rows(baseline, [_row(0, 1), _row(0, 1)])
    with pytest.raises(ValueError, match="paired key mismatch"):
        align_paired_rows(baseline, [_row(0, 1), _row(0, 3)])


def test_reproduction_requires_exact_primary_counts():
    exact = dict(EXPECTED_BASELINE_PRIMARY)
    assert reproduction_check(exact)["exact_primary_counts"]
    drifted = dict(exact)
    drifted["successes"] += 1
    result = reproduction_check(drifted)
    assert not result["exact_primary_counts"]
    assert result["differences"]["successes"] == {"expected": 62, "actual": 63}


def test_candidate_classification_confirmed_only_when_every_bar_passes():
    baseline = {"constraint_exposure": _constraint(0.5)}
    metrics = {
        "successes": 72,
        "missing_lift_eo": 21,
        "worst_seed": 11,
        "physical_sanity_pass": 68,
        "event_order_valid": 79,
        "reopen_events": 0,
        "early_close": 11,
        "controller_failures": 0,
        "constraint_exposure": _constraint(0.4),
    }
    paired = {
        "new_successes": 10,
        "new_success_from_missing_lift_fraction": 0.6,
    }

    result = classify_candidate(metrics, baseline, paired)

    assert result["status"] == "confirmed"
    assert result["all_confirmation_bars_met"]


def test_candidate_classification_partial_and_rejected_rules():
    baseline = {"constraint_exposure": _constraint(0.5)}
    metrics = {
        "successes": 68,
        "missing_lift_eo": 27,
        "worst_seed": 10,
        "physical_sanity_pass": 68,
        "event_order_valid": 79,
        "reopen_events": 0,
        "early_close": 11,
        "controller_failures": 0,
        "constraint_exposure": _constraint(0.4),
    }
    paired = {
        "new_successes": 6,
        "new_success_from_missing_lift_fraction": 0.5,
    }
    assert classify_candidate(metrics, baseline, paired)["status"] == "partial"

    regressed = copy.deepcopy(metrics)
    regressed["physical_sanity_pass"] = 67
    assert classify_candidate(regressed, baseline, paired)["status"] == "rejected"


class _FrozenSubpolicy:
    evaluation_config_hash = "protocol-hash"
    evaluation_protocol_version = 2


def _fake_hybrid() -> HybridNNGripperMLPPolicy:
    policy = object.__new__(HybridNNGripperMLPPolicy)
    policy.mlp = _FrozenSubpolicy()
    policy.nn = _FrozenSubpolicy()
    policy.action_space = "ee_tool_delta"
    return policy


def _write_fake_frozen_inputs(root: Path) -> None:
    for relative in required_frozen_relative_paths():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"{}\n")
    for seed in SEEDS:
        stem = f"ee_tool_delta_hybrid_nn_gripper_mlp_seed_{seed}"
        manifest = {
            "format": "svla_hybrid_nn_gripper_mlp_manifest_v1",
            "policy_type": "hybrid_nn_gripper_mlp",
            "action_space": "ee_tool_delta",
            "recipe": "A1_compositor",
            "match_contract": "historical_object_contact",
            "match_feature_indices": [18, 19, 20, 28, 29, 30],
            "mlp_temporal_feature_mode": "legacy_progress_phase",
            "evaluation_config_hash": "protocol-hash",
            "evaluation_protocol_version": 2,
            "mlp_path": f"{stem}_mlp_component.npz",
            "nn_path": f"{stem}_nn_component.npz",
        }
        (root / "models" / f"{stem}.json").write_text(json.dumps(manifest))


def test_frozen_model_loading_and_protocol_hash_checks(tmp_path: Path):
    _write_fake_frozen_inputs(tmp_path)
    calls = []

    def loader(path: Path):
        calls.append(path.name)
        return _fake_hybrid()

    payload = verify_frozen_inputs(
        tmp_path,
        protocol_hash="protocol-hash",
        protocol_version=2,
        policy_loader=loader,
    )
    assert len(payload["loaded_models"]) == 5
    assert len(calls) == 5

    first = tmp_path / "models" / "ee_tool_delta_hybrid_nn_gripper_mlp_seed_0.json"
    manifest = json.loads(first.read_text())
    manifest["evaluation_config_hash"] = "wrong"
    first.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="protocol hash mismatch"):
        verify_frozen_inputs(
            tmp_path,
            protocol_hash="protocol-hash",
            protocol_version=2,
            policy_loader=loader,
        )
