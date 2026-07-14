"""Unit tests for H-EE-015 oracle FSM arm upper-bound diagnostic."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from svla.h_ee_015 import (
    EFFICACY_ARTIFACT_NAMES,
    EXPECTED_BASELINE_PRIMARY,
    FSM_CLOSE_COMMAND,
    FSM_GRIPPER_OBJECT_DISTANCE_MAX_M,
    FSM_OPEN_COMMAND,
    FSM_POS_ERROR_MAX_M,
    FSM_ROT_ERROR_MAX_RAD,
    FSM_STATE_CLOSE,
    FSM_STATE_OPEN,
    GRIPPER_SOURCE,
    PARTIAL_BARS,
    STRONG_POSITIVE_BARS,
    TOTAL_TRIALS,
    OracleFsmHybridPolicy,
    OracleGripperFSM,
    align_paired_rows,
    assert_finalize_artifact_hashes,
    assert_oracle_flags,
    assert_registration_mutable,
    build_paired_comparison,
    build_registration,
    classify_verdict,
    existing_efficacy_artifacts,
    is_missing_lift_eo,
    sha256_file,
    summarize_baseline_rows,
    summarize_rows,
)
from svla.state_bc import HybridNNGripperMLPPolicy, write_json


# ---------------------------------------------------------------------------
# Pure FSM threshold / latch tests
# ---------------------------------------------------------------------------


def test_fsm_threshold_position_both_sides_and_equality():
    fsm = OracleGripperFSM()
    # Below threshold: eligible (with other conditions met).
    assert fsm.transition_conditions_met(
        pos_error_m=FSM_POS_ERROR_MAX_M - 1e-9,
        rot_error_rad=0.0,
        gripper_object_distance_m=0.0,
    )
    # Equality is inclusive.
    assert fsm.transition_conditions_met(
        pos_error_m=FSM_POS_ERROR_MAX_M,
        rot_error_rad=0.0,
        gripper_object_distance_m=0.0,
    )
    # Above: not met.
    assert not fsm.transition_conditions_met(
        pos_error_m=FSM_POS_ERROR_MAX_M + 1e-9,
        rot_error_rad=0.0,
        gripper_object_distance_m=0.0,
    )


def test_fsm_threshold_rotation_both_sides_and_equality():
    fsm = OracleGripperFSM()
    assert fsm.transition_conditions_met(
        pos_error_m=0.0,
        rot_error_rad=FSM_ROT_ERROR_MAX_RAD - 1e-9,
        gripper_object_distance_m=0.0,
    )
    assert fsm.transition_conditions_met(
        pos_error_m=0.0,
        rot_error_rad=FSM_ROT_ERROR_MAX_RAD,
        gripper_object_distance_m=0.0,
    )
    assert not fsm.transition_conditions_met(
        pos_error_m=0.0,
        rot_error_rad=FSM_ROT_ERROR_MAX_RAD + 1e-9,
        gripper_object_distance_m=0.0,
    )


def test_fsm_threshold_distance_both_sides_and_equality():
    fsm = OracleGripperFSM()
    assert fsm.transition_conditions_met(
        pos_error_m=0.0,
        rot_error_rad=0.0,
        gripper_object_distance_m=FSM_GRIPPER_OBJECT_DISTANCE_MAX_M - 1e-9,
    )
    assert fsm.transition_conditions_met(
        pos_error_m=0.0,
        rot_error_rad=0.0,
        gripper_object_distance_m=FSM_GRIPPER_OBJECT_DISTANCE_MAX_M,
    )
    assert not fsm.transition_conditions_met(
        pos_error_m=0.0,
        rot_error_rad=0.0,
        gripper_object_distance_m=FSM_GRIPPER_OBJECT_DISTANCE_MAX_M + 1e-9,
    )


def test_fsm_requires_all_three_conditions_simultaneously():
    fsm = OracleGripperFSM()
    # Only position ok.
    cmd = fsm.step(
        pos_error_m=0.0,
        rot_error_rad=FSM_ROT_ERROR_MAX_RAD + 0.01,
        gripper_object_distance_m=0.0,
    )
    assert cmd == FSM_OPEN_COMMAND
    assert fsm.state == FSM_STATE_OPEN

    # Only rotation ok (reset).
    fsm = OracleGripperFSM()
    cmd = fsm.step(
        pos_error_m=FSM_POS_ERROR_MAX_M + 0.01,
        rot_error_rad=0.0,
        gripper_object_distance_m=0.0,
    )
    assert cmd == FSM_OPEN_COMMAND

    # Only distance ok.
    fsm = OracleGripperFSM()
    cmd = fsm.step(
        pos_error_m=0.0,
        rot_error_rad=0.0,
        gripper_object_distance_m=FSM_GRIPPER_OBJECT_DISTANCE_MAX_M + 0.01,
    )
    assert cmd == FSM_OPEN_COMMAND

    # All three: transition.
    fsm = OracleGripperFSM()
    cmd = fsm.step(
        pos_error_m=FSM_POS_ERROR_MAX_M,
        rot_error_rad=FSM_ROT_ERROR_MAX_RAD,
        gripper_object_distance_m=FSM_GRIPPER_OBJECT_DISTANCE_MAX_M,
    )
    assert cmd == FSM_CLOSE_COMMAND
    assert fsm.state == FSM_STATE_CLOSE
    assert fsm.transition_step == 1


def test_fsm_permanent_latch_never_reopens():
    fsm = OracleGripperFSM()
    # Transition.
    assert (
        fsm.step(pos_error_m=0.0, rot_error_rad=0.0, gripper_object_distance_m=0.0)
        == FSM_CLOSE_COMMAND
    )
    # Even when all conditions become false, remain closed.
    for _ in range(20):
        cmd = fsm.step(
            pos_error_m=1.0,
            rot_error_rad=1.0,
            gripper_object_distance_m=1.0,
        )
        assert cmd == FSM_CLOSE_COMMAND
        assert fsm.state == FSM_STATE_CLOSE
    assert fsm.transition_step == 1
    assert not fsm.never_transitioned


def test_fsm_never_transitioned_stays_open():
    fsm = OracleGripperFSM()
    for _ in range(10):
        cmd = fsm.step(
            pos_error_m=0.05,
            rot_error_rad=0.5,
            gripper_object_distance_m=0.05,
        )
        assert cmd == FSM_OPEN_COMMAND
    assert fsm.never_transitioned
    assert fsm.transition_step is None
    assert fsm.state == FSM_STATE_OPEN


# ---------------------------------------------------------------------------
# Policy wrapper: arm identity + default path untouched
# ---------------------------------------------------------------------------


class _StubHybrid:
    """Minimal hybrid stand-in with a fixed action."""

    def __init__(self, action: np.ndarray):
        self._action = np.asarray(action, dtype=float)
        self.action_space = "ee_tool_delta"
        self.gripper_dim = -1
        self.group_keys = ["k"]
        self.evaluation_config_hash = "hash"
        self.evaluation_protocol_version = 2

    def predict_with_index(
        self, observation_features, group_key, cursor=None, search_window=None
    ):
        return self._action.copy(), 0.0, int(cursor or 0)


def test_oracle_wrapper_requires_hybrid_type():
    with pytest.raises(TypeError):
        OracleFsmHybridPolicy(object())  # type: ignore[arg-type]


def test_arm_dims_byte_identical_with_and_without_fsm():
    base_action = np.array([0.1, -0.2, 0.3, 0.01, -0.02, 0.55], dtype=float)
    hybrid = _StubHybrid(base_action)
    # Bypass type check by constructing via __new__ and assigning.
    policy = object.__new__(OracleFsmHybridPolicy)
    policy.hybrid = hybrid
    policy.fsm = OracleGripperFSM()
    policy._signals = None
    policy.action_space = hybrid.action_space
    policy.gripper_dim = hybrid.gripper_dim
    policy.group_keys = list(hybrid.group_keys)
    policy.gripper_source = GRIPPER_SOURCE
    policy.oracle_diagnostic = True

    features = np.zeros(32)
    # Hybrid baseline action.
    a_hybrid, _, _ = hybrid.predict_with_index(features, "k", cursor=0)

    # FSM open (far from target).
    policy.set_oracle_signals(
        pos_error_m=0.05, rot_error_rad=0.5, gripper_object_distance_m=0.05
    )
    a_open, _, _ = policy.predict_with_index(features, "k", cursor=0)
    np.testing.assert_array_equal(a_open[:-1], a_hybrid[:-1])
    assert a_open[-1] == FSM_OPEN_COMMAND
    assert a_open[-1] != a_hybrid[-1] or a_hybrid[-1] == FSM_OPEN_COMMAND

    # FSM close (all thresholds).
    policy.set_oracle_signals(
        pos_error_m=0.0, rot_error_rad=0.0, gripper_object_distance_m=0.0
    )
    a_close, _, _ = policy.predict_with_index(features, "k", cursor=1)
    np.testing.assert_array_equal(a_close[:-1], a_hybrid[:-1])
    assert a_close[-1] == FSM_CLOSE_COMMAND


def test_default_hybrid_has_no_set_oracle_signals():
    hybrid = object.__new__(HybridNNGripperMLPPolicy)
    assert not callable(getattr(hybrid, "set_oracle_signals", None))


def test_oracle_wrapper_requires_signals_before_predict():
    base_action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.5], dtype=float)
    hybrid = _StubHybrid(base_action)
    policy = object.__new__(OracleFsmHybridPolicy)
    policy.hybrid = hybrid
    policy.fsm = OracleGripperFSM()
    policy._signals = None
    policy.action_space = hybrid.action_space
    policy.gripper_dim = -1
    policy.group_keys = ["k"]
    with pytest.raises(RuntimeError, match="set_oracle_signals"):
        policy.predict_with_index(np.zeros(4), "k", cursor=0)


# ---------------------------------------------------------------------------
# Oracle flags, pairing, verdict boundaries
# ---------------------------------------------------------------------------


def _oracle_row(
    seed: int,
    trial_id: int,
    *,
    success: bool = False,
    event_order_valid: bool = False,
    physical_sanity_pass: bool = True,
    early_close: bool = False,
    reopen_events: int = 0,
    contact_achieved: bool = True,
    object_lifted: bool = False,
    controller_failure_steps: int = 0,
    failure_category: str = "event_order_failure",
    fsm_never_transitioned: bool = False,
    steps: int = 100,
) -> dict:
    return {
        "seed": seed,
        "trial_id": trial_id,
        "success": success,
        "event_order_valid": event_order_valid,
        "physical_sanity_pass": physical_sanity_pass,
        "early_close": early_close,
        "reopen_events": reopen_events,
        "contact_achieved": contact_achieved,
        "object_lifted": object_lifted,
        "controller_failure_steps": controller_failure_steps,
        "failure_category": failure_category,
        "gripper_source": GRIPPER_SOURCE,
        "oracle_diagnostic": True,
        "fsm_never_transitioned": fsm_never_transitioned,
        "fsm_transition_step": None if fsm_never_transitioned else 50,
        "fsm_transition_pos_error": None if fsm_never_transitioned else 0.01,
        "fsm_transition_rot_error": None if fsm_never_transitioned else 0.1,
        "fsm_transition_gripper_object_distance": (
            None if fsm_never_transitioned else 0.01
        ),
        "steps": steps,
        "joint_limit_clipped_steps": 0,
        "infeasible_steps": 0,
        "max_gripper_contact_force": 1.0,
        "gripper_contact_impulse_before_lift": 1.0,
        "max_object_xy_displacement_while_supported": 0.001,
        "max_object_rotation_while_supported": 0.01,
        "retained_during_hold": success,
    }


def test_assert_oracle_flags_mandatory():
    row = _oracle_row(0, 6001)
    assert_oracle_flags(row)
    bad = dict(row)
    bad["gripper_source"] = "nn_gripper"
    with pytest.raises(ValueError, match="gripper_source"):
        assert_oracle_flags(bad)
    bad2 = dict(row)
    bad2["oracle_diagnostic"] = False
    with pytest.raises(ValueError, match="oracle_diagnostic"):
        assert_oracle_flags(bad2)


def test_summarize_rows_rejects_missing_oracle_flags():
    rows = [_oracle_row(0, 6001 + i) for i in range(5)]
    # Pad to full seed coverage with dummy keys across seeds for Counter.
    for seed in range(5):
        for i in range(24):
            if seed == 0 and i < 5:
                continue
            rows.append(_oracle_row(seed, 6001 + i, success=False))
    # Break one flag.
    rows[0]["oracle_diagnostic"] = False
    with pytest.raises(ValueError, match="oracle_diagnostic"):
        summarize_rows(rows)


def test_paired_key_missing_and_duplicate_rejection():
    baseline = [_oracle_row(0, 6001), _oracle_row(0, 6002)]
    # Remove oracle flags for baseline-style rows.
    for r in baseline:
        r.pop("gripper_source")
        r.pop("oracle_diagnostic")
    candidate = [_oracle_row(0, 6001)]
    with pytest.raises(ValueError, match="paired key mismatch"):
        align_paired_rows(baseline, candidate)

    dup = [_oracle_row(0, 6001), _oracle_row(0, 6001)]
    with pytest.raises(ValueError, match="duplicate"):
        align_paired_rows(baseline[:1], dup)


def test_verdict_strong_positive_at_boundary():
    metrics = {
        "successes": STRONG_POSITIVE_BARS["successes_min"],
        "event_order_valid": STRONG_POSITIVE_BARS["event_order_valid_min"],
        "physical_sanity_pass": STRONG_POSITIVE_BARS["physical_sanity_pass_min"],
        "worst_seed": STRONG_POSITIVE_BARS["worst_seed_min"],
        "controller_failures": 0,
        "missing_lift_eo": 0,
        "early_close": 0,
    }
    v = classify_verdict(metrics, EXPECTED_BASELINE_PRIMARY)
    assert v["status"] == "strong_positive_arm_upper_bound"


def test_verdict_strong_positive_fails_one_below_boundary():
    metrics = {
        "successes": STRONG_POSITIVE_BARS["successes_min"] - 1,
        "event_order_valid": STRONG_POSITIVE_BARS["event_order_valid_min"],
        "physical_sanity_pass": STRONG_POSITIVE_BARS["physical_sanity_pass_min"],
        "worst_seed": STRONG_POSITIVE_BARS["worst_seed_min"],
        "controller_failures": 0,
        "missing_lift_eo": 0,
        "early_close": 0,
    }
    v = classify_verdict(metrics, EXPECTED_BASELINE_PRIMARY)
    # May be partial if partial bars met.
    assert v["status"] in ("partial", "negative_arm_ceiling")
    assert v["status"] != "strong_positive_arm_upper_bound"


def test_verdict_partial_at_boundary():
    # successes >= 72, material residual drop, phys >= 68, worst >= 10, ctrl 0
    metrics = {
        "successes": PARTIAL_BARS["successes_min"],
        "event_order_valid": 80,
        "physical_sanity_pass": PARTIAL_BARS["physical_sanity_pass_min"],
        "worst_seed": PARTIAL_BARS["worst_seed_min"],
        "controller_failures": 0,
        # baseline early_close=11; drop by >= 5 → early_close <= 6
        "missing_lift_eo": EXPECTED_BASELINE_PRIMARY["missing_lift_eo"],
        "early_close": EXPECTED_BASELINE_PRIMARY["early_close"]
        - PARTIAL_BARS["material_residual_drop_min"],
    }
    v = classify_verdict(metrics, EXPECTED_BASELINE_PRIMARY)
    assert v["status"] == "partial"
    assert v["partial_bars"]["material_residual_drop"] is True


def test_verdict_partial_fails_without_material_residual_drop():
    metrics = {
        "successes": 75,
        "event_order_valid": 85,
        "physical_sanity_pass": 70,
        "worst_seed": 12,
        "controller_failures": 0,
        "missing_lift_eo": EXPECTED_BASELINE_PRIMARY["missing_lift_eo"],
        "early_close": EXPECTED_BASELINE_PRIMARY["early_close"],  # no drop
    }
    v = classify_verdict(metrics, EXPECTED_BASELINE_PRIMARY)
    assert v["status"] == "negative_arm_ceiling"
    assert v["partial_bars"]["material_residual_drop"] is False


def test_verdict_negative_when_controller_failures():
    metrics = {
        "successes": 90,
        "event_order_valid": 95,
        "physical_sanity_pass": 90,
        "worst_seed": 15,
        "controller_failures": 1,
        "missing_lift_eo": 0,
        "early_close": 0,
    }
    v = classify_verdict(metrics, EXPECTED_BASELINE_PRIMARY)
    assert v["status"] == "negative_arm_ceiling"


def test_verdict_partial_just_below_success_boundary():
    metrics = {
        "successes": PARTIAL_BARS["successes_min"] - 1,
        "event_order_valid": 85,
        "physical_sanity_pass": 70,
        "worst_seed": 12,
        "controller_failures": 0,
        "missing_lift_eo": 0,
        "early_close": 0,
    }
    v = classify_verdict(metrics, EXPECTED_BASELINE_PRIMARY)
    assert v["status"] == "negative_arm_ceiling"


def test_missing_lift_eo_predicate():
    row = _oracle_row(
        0,
        6001,
        success=False,
        event_order_valid=False,
        early_close=False,
        reopen_events=0,
        contact_achieved=True,
        object_lifted=False,
    )
    assert is_missing_lift_eo(row)
    row["early_close"] = True
    assert not is_missing_lift_eo(row)


def test_build_paired_comparison_recoveries_and_regressions():
    baseline = [
        {
            "seed": 0,
            "trial_id": 6001,
            "success": False,
            "event_order_valid": False,
            "failure_category": "event_order_failure",
            "early_close": False,
            "reopen_events": 0,
            "contact_achieved": True,
            "object_lifted": False,
            "joint_limit_clipped_steps": 10,
            "infeasible_steps": 5,
        },
        {
            "seed": 0,
            "trial_id": 6002,
            "success": True,
            "event_order_valid": True,
            "failure_category": "none",
            "early_close": False,
            "reopen_events": 0,
            "contact_achieved": True,
            "object_lifted": True,
            "joint_limit_clipped_steps": 2,
            "infeasible_steps": 1,
        },
    ]
    candidate = [
        _oracle_row(0, 6001, success=True, event_order_valid=True, failure_category="none"),
        _oracle_row(
            0, 6002, success=False, event_order_valid=False, failure_category="event_order_failure"
        ),
    ]
    paired = build_paired_comparison(baseline, candidate)
    assert paired["new_successes"] == 1
    assert paired["lost_successes"] == 1
    assert paired["net_success_change"] == 0
    assert paired["oracle_diagnostic"] is True
    assert paired["gripper_source"] == GRIPPER_SOURCE


def test_build_registration_rejects_bad_key_count():
    with pytest.raises(ValueError, match="paired keys"):
        build_registration(
            protocol_metadata={
                "version": 2,
                "config_sha256": "abc",
                "format": "x",
            },
            frozen_verification={"loaded_models": [], "file_inventory": []},
            baseline_metrics=dict(EXPECTED_BASELINE_PRIMARY),
            paired_keys=[[0, 6001]],
            source_manifest_sha256="deadbeef",
        )


def test_build_registration_rejects_baseline_mismatch():
    bad = dict(EXPECTED_BASELINE_PRIMARY)
    bad["successes"] = 0
    keys = [[s, 6001 + t] for s in range(5) for t in range(24)]
    with pytest.raises(ValueError, match="baseline primary"):
        build_registration(
            protocol_metadata={
                "version": 2,
                "config_sha256": "abc",
                "format": "x",
            },
            frozen_verification={"loaded_models": [], "file_inventory": []},
            baseline_metrics=bad,
            paired_keys=keys,
            source_manifest_sha256="deadbeef",
        )


def test_summarize_baseline_rows_primary_fields():
    # Minimal 120 rows matching expected seed successes pattern is heavy;
    # instead verify counting mechanics on a small set and that keys form.
    rows = []
    for seed in range(5):
        for i in range(24):
            rows.append(
                {
                    "seed": seed,
                    "trial_id": 6001 + i,
                    "success": seed == 0 and i < 20,
                    "event_order_valid": seed == 0 and i < 20,
                    "physical_sanity_pass": True,
                    "early_close": False,
                    "reopen_events": 0,
                    "contact_achieved": True,
                    "object_lifted": seed == 0 and i < 20,
                    "controller_failure_steps": 0,
                    "failure_category": "none" if (seed == 0 and i < 20) else "event_order_failure",
                }
            )
    metrics = summarize_baseline_rows(rows)
    assert metrics["total"] == TOTAL_TRIALS
    assert metrics["successes"] == 20
    assert metrics["per_seed_successes"][0] == 20
    assert len(metrics["paired_keys"]) == TOTAL_TRIALS


def test_registration_mutable_before_efficacy_artifacts(tmp_path: Path):
    assert existing_efficacy_artifacts(tmp_path) == []
    assert_registration_mutable(tmp_path)  # no raise


@pytest.mark.parametrize("artifact_name", list(EFFICACY_ARTIFACT_NAMES))
def test_registration_immutable_once_any_efficacy_artifact_exists(
    tmp_path: Path, artifact_name: str
):
    (tmp_path / artifact_name).write_text("x\n", encoding="utf-8")
    found = existing_efficacy_artifacts(tmp_path)
    assert artifact_name in found
    with pytest.raises(RuntimeError, match="immutable after efficacy"):
        assert_registration_mutable(tmp_path)


def test_registration_immutable_even_when_registration_json_also_present(tmp_path: Path):
    # --force may rewrite registration only when no efficacy artifacts exist.
    (tmp_path / "h_ee_015_registration.json").write_text("{}", encoding="utf-8")
    (tmp_path / "h_ee_015_trials.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="immutable after efficacy"):
        assert_registration_mutable(tmp_path)


def test_verify_frozen_inputs_same_source_is_inventory_only(tmp_path: Path, monkeypatch):
    """Same baseline_dir and source_dir must not claim independent copy match."""

    from svla import h_ee_015 as mod

    # Build a minimal fake inventory tree so verify can list relative paths.
    # We only test the source_comparison branch; short-circuit missing files.
    def fake_required():
        return [Path("models") / "dummy.json"]

    def fake_sha(path: Path) -> str:
        return "abc"

    class _FakeHybrid(HybridNNGripperMLPPolicy):
        def __init__(self) -> None:
            # Bypass HybridNNGripperMLPPolicy.__init__; only attrs used by verify.
            self.action_space = "ee_tool_delta"
            self._evaluation_config_hash = "proto"
            self._evaluation_protocol_version = 2

        @property
        def evaluation_config_hash(self) -> str:
            return self._evaluation_config_hash

        @evaluation_config_hash.setter
        def evaluation_config_hash(self, value: str) -> None:
            self._evaluation_config_hash = str(value)

        @property
        def evaluation_protocol_version(self) -> int:
            return self._evaluation_protocol_version

        @evaluation_protocol_version.setter
        def evaluation_protocol_version(self, value: int) -> None:
            self._evaluation_protocol_version = int(value)

    def fake_load(path: Path):
        return _FakeHybrid()

    monkeypatch.setattr(mod, "required_frozen_relative_paths", fake_required)
    monkeypatch.setattr(mod, "sha256_file", fake_sha)
    monkeypatch.setattr(mod, "SEEDS", (0,))
    monkeypatch.setattr(mod, "model_manifest_name", lambda seed: "dummy.json")

    models = tmp_path / "models"
    models.mkdir()
    manifest = {
        "format": "svla_hybrid_nn_gripper_mlp_manifest_v1",
        "action_space": "ee_tool_delta",
        "policy_type": "hybrid_nn_gripper_mlp",
        "recipe": "A1_compositor",
        "mlp_temporal_feature_mode": "legacy_progress_phase",
        "match_contract": "historical_object_contact",
        "match_feature_indices": [18, 19, 20, 28, 29, 30],
        "evaluation_config_hash": "proto",
        "evaluation_protocol_version": 2,
        "mlp_path": "dummy_mlp.npz",
        "nn_path": "dummy_nn.npz",
    }
    (models / "dummy.json").write_text(json.dumps(manifest), encoding="utf-8")
    (models / "dummy_mlp.npz").write_bytes(b"mlp")
    (models / "dummy_nn.npz").write_bytes(b"nn")

    result = mod.verify_frozen_inputs(
        tmp_path,
        protocol_hash="proto",
        protocol_version=2,
        source_dir=tmp_path,  # tautological path
        policy_loader=fake_load,
    )
    assert result["source_comparison"] == "inventory_only"
    assert result["source_dir"] is None
    assert result["all_copies_match_primary"] is False
    assert "copy_matches_primary" not in result["file_inventory"][0]


def test_assert_finalize_artifact_hashes_detects_stale_summary_sha(tmp_path: Path):
    """Manifest must hash the *final* summary file, not a pre-refresh copy."""

    reg = tmp_path / "h_ee_015_registration.json"
    trials = tmp_path / "h_ee_015_trials.jsonl"
    summary = tmp_path / "h_ee_015_summary.json"
    paired = tmp_path / "h_ee_015_paired_comparison.json"
    manifest = tmp_path / "h_ee_015_experiment_manifest.json"

    write_json(reg, {"format": "registration", "final_accessed": False})
    trials.write_text('{"seed":0,"trial_id":6001}\n', encoding="utf-8")
    write_json(paired, {"format": "paired", "new_successes": 0})

    # Pre-refresh summary (missing paired block).
    write_json(summary, {"format": "summary", "metrics": {"successes": 47}})
    stale_summary_sha = sha256_file(summary)

    # Final summary rewrite (what finalize must hash).
    write_json(
        summary,
        {
            "format": "summary",
            "metrics": {"successes": 47},
            "paired": {"new_successes": 5, "lost_successes": 20},
        },
    )
    final_summary_sha = sha256_file(summary)
    assert final_summary_sha != stale_summary_sha

    # Bug pattern: manifest recorded the stale pre-refresh hash.
    write_json(
        manifest,
        {
            "format": "svla_h_ee_015_experiment_manifest_v1",
            "registration_sha256": sha256_file(reg),
            "trials_sha256": sha256_file(trials),
            "summary_sha256": stale_summary_sha,
            "paired_comparison_sha256": sha256_file(paired),
        },
    )
    with pytest.raises(ValueError, match="summary_sha256"):
        assert_finalize_artifact_hashes(tmp_path)

    # Correct order: hash after final summary write.
    write_json(
        manifest,
        {
            "format": "svla_h_ee_015_experiment_manifest_v1",
            "registration_sha256": sha256_file(reg),
            "trials_sha256": sha256_file(trials),
            "summary_sha256": final_summary_sha,
            "paired_comparison_sha256": sha256_file(paired),
        },
    )
    actual = assert_finalize_artifact_hashes(tmp_path)
    assert actual["summary_sha256"] == final_summary_sha
