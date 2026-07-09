"""Unit tests for the frozen H-EE-021 loss-profile contract."""

from __future__ import annotations

import numpy as np
import pytest

from svla.loss_profiles import (
    CLOSE_TIMING_PHASES,
    LOSS_PROFILE_NAMES,
    LOSS_PROFILES,
    RESEARCH_PARITY_FRONTIER,
    expected_gripper_weight_for_phase,
    get_loss_profile,
    resolve_loss_weights,
)
from svla.state_bc import (
    PHASE_LABELS,
    action_loss_weights,
    phase_action_loss_report,
    phase_sample_counts,
)


def test_frozen_profiles_match_h_ee_008_factorial_matrix():
    assert LOSS_PROFILE_NAMES == (
        "uniform",
        "global_gripper",
        "transition_gripper",
        "combined_h_ee_008",
    )
    assert LOSS_PROFILES["uniform"].global_gripper_weight == 1.0
    assert LOSS_PROFILES["uniform"].transition_gripper_weight == 1.0
    assert LOSS_PROFILES["global_gripper"].global_gripper_weight == 5.0
    assert LOSS_PROFILES["global_gripper"].transition_gripper_weight == 5.0
    assert LOSS_PROFILES["transition_gripper"].global_gripper_weight == 1.0
    assert LOSS_PROFILES["transition_gripper"].transition_gripper_weight == 10.0
    assert LOSS_PROFILES["combined_h_ee_008"].global_gripper_weight == 5.0
    assert LOSS_PROFILES["combined_h_ee_008"].transition_gripper_weight == 10.0
    assert set(CLOSE_TIMING_PHASES) == {"grasp_align", "close_gripper"}


def test_research_parity_frontier_is_weighted_joint_validation():
    frontier = RESEARCH_PARITY_FRONTIER
    assert frontier["source_action_space"] == "joint_delta"
    assert frontier["successes"] == 84
    assert frontier["event_order_valid"] == 90
    assert frontier["physical_sanity_pass"] == 100
    assert frontier["worst_seed_successes_min"] == 12
    assert frontier["total_trials"] == 120


@pytest.mark.parametrize("profile_name", LOSS_PROFILE_NAMES)
def test_each_profile_builds_intended_weight_matrix_and_leaves_arm_unchanged(profile_name):
    profile = get_loss_profile(profile_name)
    phase_indices = np.array(
        [
            PHASE_LABELS.index("approach_0"),
            PHASE_LABELS.index("approach_1"),
            PHASE_LABELS.index("grasp_align"),
            PHASE_LABELS.index("close_gripper"),
            PHASE_LABELS.index("lift"),
            PHASE_LABELS.index("hold"),
        ],
        dtype=int,
    )
    weights = action_loss_weights(
        phase_indices,
        action_size=6,
        loss_profile=profile_name,
    )

    assert weights.shape == (6, 6)
    # Arm dims are never reweighted by loss profiles.
    assert np.allclose(weights[:, :5], 1.0)

    for row, phase in enumerate(
        ("approach_0", "approach_1", "grasp_align", "close_gripper", "lift", "hold")
    ):
        expected = expected_gripper_weight_for_phase(profile, phase)
        assert weights[row, 5] == expected, f"{profile_name}/{phase}"


def test_named_profile_overrides_free_form_weights():
    grip, close, profile = resolve_loss_weights(
        loss_profile="combined_h_ee_008",
        gripper_loss_weight=99.0,
        close_phase_gripper_weight=3.0,
    )
    assert profile is not None
    assert profile.name == "combined_h_ee_008"
    assert grip == 5.0
    assert close == 10.0


def test_phase_sample_counts_and_action_loss_report():
    phase_indices = np.array(
        [
            PHASE_LABELS.index("approach_0"),
            PHASE_LABELS.index("grasp_align"),
            PHASE_LABELS.index("close_gripper"),
            PHASE_LABELS.index("lift"),
        ],
        dtype=int,
    )
    residual = np.zeros((4, 6), dtype=float)
    residual[:, 5] = np.array([0.1, 0.2, 0.3, 0.4])
    residual[:, 0] = np.array([1.0, 1.0, 1.0, 1.0])
    loss_weights = action_loss_weights(
        phase_indices,
        action_size=6,
        loss_profile="transition_gripper",
    )
    counts = phase_sample_counts(phase_indices)
    report = phase_action_loss_report(residual, phase_indices, loss_weights)

    assert counts["total"] == 4
    assert counts["close_timing_total"] == 2
    assert counts["grasp_align"] == 1
    assert report["approach_0"]["sample_count"] == 1
    assert report["approach_0"]["gripper_mse"] == pytest.approx(0.01)
    assert report["approach_0"]["arm_mse"] == pytest.approx(0.2)  # 1^2 / 5 dims
    assert report["grasp_align"]["mean_gripper_weight"] == 10.0
    assert report["approach_0"]["mean_gripper_weight"] == 1.0


def test_unknown_profile_raises():
    with pytest.raises(ValueError, match="unknown loss profile"):
        get_loss_profile("not_a_real_profile")
