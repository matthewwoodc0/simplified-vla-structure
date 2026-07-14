from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from svla.h_ee_002 import (
    EXPECTED_BASELINE_PRIMARY,
    SEEDS,
    align_paired_rows,
    build_paired_comparison,
    classify_candidate,
    classify_sweep,
    is_missing_lift_eo,
    load_jsonl,
    required_frozen_relative_paths,
    reproduction_check,
    summarize_rows,
    verify_frozen_inputs,
)
from svla.state_bc import HybridNNGripperMLPPolicy

# Frozen H-EE-002 JSONL digests (immutable experiment rows; never rewrite).
FROZEN_JSONL_SHA256 = {
    "gain_1_000_policy_trials.jsonl": (
        "bf31302e53a4b3b054863d4d849a712f985775d1cef21754777b8a482589aca4"
    ),
    "gain_0_875_policy_trials.jsonl": (
        "4a160d8df997816805df9090dc89eb90927eb21d561f20f8800b365763c4044d"
    ),
    "gain_0_750_policy_trials.jsonl": (
        "9ee88278aa450f13e090820320c79ec5ea7d213d6acc83523f21f0e6d40880d6"
    ),
}
H_EE_002_OUTPUT_DIR = (
    Path(__file__).resolve().parents[1] / "outputs" / "h_ee_002_hybrid_gain_sweep"
)
# Headline metrics frozen at experiment close (schema rename must not change these).
FROZEN_PRIMARY = {
    "gain_1_000": {
        "successes": 62,
        "event_order_valid": 79,
        "physical_sanity_pass": 68,
        "per_seed_successes": [20, 14, 9, 9, 10],
        "worst_seed": 9,
        "early_close": 11,
        "reopen_events": 0,
        "missing_lift_eo": 30,
        "controller_failures": 0,
    },
    "gain_0_875": {
        "successes": 5,
        "event_order_valid": 9,
        "physical_sanity_pass": 26,
        "per_seed_successes": [5, 0, 0, 0, 0],
        "worst_seed": 0,
        "early_close": 25,
        "reopen_events": 0,
        "missing_lift_eo": 86,
        "controller_failures": 0,
    },
    "gain_0_750": {
        "successes": 0,
        "event_order_valid": 0,
        "physical_sanity_pass": 37,
        "per_seed_successes": [0, 0, 0, 0, 0],
        "worst_seed": 0,
        "early_close": 48,
        "reopen_events": 0,
        "missing_lift_eo": 72,
        "controller_failures": 0,
    },
}


def _row(
    seed: int,
    trial_id: int,
    *,
    success: bool = False,
    missing_lift: bool = False,
    steps: int = 100,
    joint_limit_clipped_steps: int = 10,
    infeasible_steps: int = 10,
) -> dict:
    if missing_lift and success:
        raise ValueError("success and missing_lift are mutually exclusive")
    if missing_lift:
        return {
            "seed": seed,
            "trial_id": trial_id,
            "success": False,
            "event_order_valid": False,
            "physical_sanity_pass": True,
            "early_close": False,
            "reopen_events": 0,
            "contact_achieved": True,
            "object_lifted": False,
            "failure_category": "event_order_failure",
            "joint_limit_clipped_steps": joint_limit_clipped_steps,
            "infeasible_steps": infeasible_steps,
            "controller_failure_steps": 0,
            "steps": steps,
            "gripper_contact_impulse_before_lift": 0.0,
        }
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
        "joint_limit_clipped_steps": joint_limit_clipped_steps,
        "infeasible_steps": infeasible_steps,
        "controller_failure_steps": 0,
        "steps": steps,
        "gripper_contact_impulse_before_lift": 0.0,
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


def test_summarize_rows_uses_current_gain_missing_lift_not_gain1_bucket():
    """Per-gain constraint block is this gain's missing-lift rows, not gain-1.0."""
    # Baseline-like set: 1 success + 2 missing-lift failures.
    baseline_like = [
        _row(0, 1, success=True, joint_limit_clipped_steps=5),
        _row(0, 2, missing_lift=True, joint_limit_clipped_steps=100),
        _row(0, 3, missing_lift=True, joint_limit_clipped_steps=200),
    ]
    # Candidate-like set: more missing-lift than baseline (simulates 0.875 collapse).
    candidate_like = [
        _row(0, 1, missing_lift=True, joint_limit_clipped_steps=50),
        _row(0, 2, missing_lift=True, joint_limit_clipped_steps=60),
        _row(0, 3, missing_lift=True, joint_limit_clipped_steps=70),
        _row(0, 4, missing_lift=True, joint_limit_clipped_steps=80),
    ]

    baseline_metrics = summarize_rows(baseline_like)
    candidate_metrics = summarize_rows(candidate_like)

    for metrics, expected_count in (
        (baseline_metrics, 2),
        (candidate_metrics, 4),
    ):
        exposure = metrics["constraint_exposure"]
        assert "current_gain_missing_lift" in exposure
        assert "missing_lift_gain_1_bucket" not in exposure
        assert exposure["current_gain_missing_lift"]["trial_count"] == expected_count
        assert metrics["missing_lift_eo"] == expected_count

    # Misnamed key would incorrectly imply a fixed gain-1.0 cohort of size 2 for both.
    assert (
        candidate_metrics["constraint_exposure"]["current_gain_missing_lift"]["trial_count"]
        != baseline_metrics["constraint_exposure"]["current_gain_missing_lift"]["trial_count"]
    )


def test_paired_baseline_missing_lift_cohort_is_fixed_from_gain1():
    """Paired analysis must use the original gain-1.0 missing-lift keys only."""
    early_close_failure = _row(1, 1, success=False)
    early_close_failure["early_close"] = True  # EO fail path that is not missing-lift
    baseline = [
        _row(0, 1, success=True),
        _row(0, 2, missing_lift=True, joint_limit_clipped_steps=900),
        _row(0, 3, missing_lift=True, joint_limit_clipped_steps=1000),
        early_close_failure,
    ]
    # Candidate invents many more missing-lift outcomes; paired cohort stays size 2.
    candidate = [
        _row(0, 1, missing_lift=True, joint_limit_clipped_steps=10),
        _row(0, 2, missing_lift=True, joint_limit_clipped_steps=20),
        _row(0, 3, success=True, joint_limit_clipped_steps=5),
        _row(1, 1, missing_lift=True, joint_limit_clipped_steps=30),
    ]

    paired = build_paired_comparison(baseline, candidate)

    assert paired["baseline_missing_lift_trial_count"] == 2
    cohort = paired["constraint_deltas"]["baseline_missing_lift_trials"]
    assert cohort["joint_limit"]["trial_count"] == 2
    assert cohort["infeasible"]["trial_count"] == 2
    # One of the two original missing-lift trials recovered to success.
    assert paired["new_successes_from_baseline_missing_lift"] == 1
    assert paired["baseline_missing_lift_candidate_outcomes"]["recovered_to_success"] == 1
    assert paired["baseline_missing_lift_candidate_outcomes"]["still_missing_lift_eo"] == 1


def test_classification_unchanged_by_missing_lift_cohort_field_rename():
    """Bars and verdict logic use failure exposure + primary metrics, not the cohort name."""
    baseline_rows = [
        _row(seed, trial_id, success=(trial_id % 2 == 0), missing_lift=(trial_id % 2 == 1))
        for seed in range(5)
        for trial_id in range(4)
    ]
    # Force primary counts toward a rejected candidate shape.
    candidate_rows = [
        _row(seed, trial_id, missing_lift=True)
        for seed in range(5)
        for trial_id in range(4)
    ]
    baseline_metrics = summarize_rows(baseline_rows)
    candidate_metrics = summarize_rows(candidate_rows)
    assert "current_gain_missing_lift" in candidate_metrics["constraint_exposure"]
    assert "missing_lift_gain_1_bucket" not in candidate_metrics["constraint_exposure"]

    paired = build_paired_comparison(baseline_rows, candidate_rows)
    result = classify_candidate(candidate_metrics, baseline_metrics, paired)
    assert result["status"] == "rejected"
    assert not result["all_confirmation_bars_met"]

    sweep = classify_sweep(
        {
            "gain_0_875": {"classification": result},
            "gain_0_750": {"classification": result},
        }
    )
    assert sweep == {"status": "rejected", "selected_gain": None}


@pytest.mark.skipif(
    not (H_EE_002_OUTPUT_DIR / "gain_1_000_policy_trials.jsonl").is_file(),
    reason="H-EE-002 frozen experiment artifacts not present in this checkout",
)
def test_frozen_h_ee_002_artifacts_cohort_labeling_and_immutability():
    """Regression on the real experiment: rename only; rows/metrics/verdict fixed."""
    import hashlib

    def sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    rows_by_gain: dict[str, list[dict]] = {}
    for name, expected in FROZEN_JSONL_SHA256.items():
        path = H_EE_002_OUTPUT_DIR / name
        assert sha256(path) == expected, f"JSONL mutated: {name}"
        rows = load_jsonl(path)
        assert len(rows) == 120
        trial_ids = {int(row["trial_id"]) for row in rows}
        assert trial_ids == set(range(6001, 6025))
        slug = name.replace("_policy_trials.jsonl", "")
        rows_by_gain[slug] = rows

    for slug, expected_primary in FROZEN_PRIMARY.items():
        metrics = summarize_rows(rows_by_gain[slug])
        exposure = metrics["constraint_exposure"]
        assert "current_gain_missing_lift" in exposure
        assert "missing_lift_gain_1_bucket" not in exposure
        assert exposure["current_gain_missing_lift"]["trial_count"] == metrics["missing_lift_eo"]
        for field, value in expected_primary.items():
            assert metrics[field] == value, f"{slug}.{field}"

        summary_path = H_EE_002_OUTPUT_DIR / f"{slug}_summary.json"
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary_exposure = summary["metrics"]["constraint_exposure"]
            assert "current_gain_missing_lift" in summary_exposure
            assert "missing_lift_gain_1_bucket" not in summary_exposure
            assert summary["metrics"]["successes"] == expected_primary["successes"]
            assert summary["metrics"]["missing_lift_eo"] == expected_primary["missing_lift_eo"]

    baseline = rows_by_gain["gain_1_000"]
    assert sum(is_missing_lift_eo(row) for row in baseline) == 30

    classifications = {}
    baseline_metrics = summarize_rows(baseline)
    for slug in ("gain_0_875", "gain_0_750"):
        candidate = rows_by_gain[slug]
        paired = build_paired_comparison(baseline, candidate)
        assert paired["baseline_missing_lift_trial_count"] == 30
        assert (
            paired["constraint_deltas"]["baseline_missing_lift_trials"]["joint_limit"][
                "trial_count"
            ]
            == 30
        )
        classification = classify_candidate(
            summarize_rows(candidate), baseline_metrics, paired
        )
        classifications[slug] = {"classification": classification}
        assert classification["status"] == "rejected"
        assert paired["new_successes"] == 0

    assert classify_sweep(classifications) == {
        "status": "rejected",
        "selected_gain": None,
    }

    # Registration is immutable and may still embed the historical field name.
    registration = json.loads(
        (H_EE_002_OUTPUT_DIR / "h_ee_002_registration.json").read_text(encoding="utf-8")
    )
    assert registration["baseline_primary"]["successes"] == 62
    assert registration["final_accessed"] is False
