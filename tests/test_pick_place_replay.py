import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from svla.demo_recorder import PickupDemoRecorder
from svla.pick_place_replay import replay_demo_policy_labels
from svla.pickup_task import (
    OBJECT_START_Z,
    ApproachStrategy,
    GraspOrientation,
    ObjectStartPose,
    PickPlaceTrialSpec,
    PickupTaskEvaluator,
    PlacementTarget,
    maybe_finalize_grasp_at_sample,
)


def _load_validate_action_replay():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "validate_action_replay.py"
    spec = importlib.util.spec_from_file_location("validate_action_replay", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _FinalizeTracker:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def finalize_grasp_segment(self) -> dict:
        self.calls.append(1)
        return {
            "collision_free_approach": True,
            "event_order_valid": True,
            "physical_sanity_pass": True,
            "early_close": False,
            "reopen_events": 0,
            "preclose_contact_steps": 0,
            "preclose_max_object_displacement": 0.0,
            "max_gripper_contact_force": 0.0,
            "gripper_contact_impulse_before_lift": 0.0,
            "max_object_xy_displacement_while_supported": 0.0,
            "max_object_rotation_while_supported": 0.0,
            "contact_achieved": True,
            "object_lifted": True,
            "retained_during_hold": True,
            "max_object_lift": 0.03,
        }


def _pick_place_demo():
    recorder = PickupDemoRecorder(PickupTaskEvaluator())
    spec = PickPlaceTrialSpec(
        trial_id=1,
        orientation=GraspOrientation("yaw_0", 0.0),
        object_pose=ObjectStartPose("center", np.array([0.0, -0.235, OBJECT_START_Z])),
        approach=ApproachStrategy("vertical_pregrasp", "world_z"),
        placement_target=PlacementTarget("place_right", "place_right_marker", "place_right_marker"),
    )
    return spec, recorder.record_pick_place_trial(spec)


def test_maybe_finalize_grasp_at_sample_only_at_boundary():
    tracker = _FinalizeTracker()
    for sample_index in range(5):
        maybe_finalize_grasp_at_sample(tracker, sample_index, boundary_index=2)
    assert len(tracker.calls) == 1


def test_replay_requires_grasp_boundary_metadata():
    demo = {
        "metadata": {"trial_spec": {"trial_id": 1, "orientation": "yaw_0", "object_pose": "center", "approach": "vertical_pregrasp"}},
        "summary": {"object_start_pose": [0, -0.235, 0.069], "commanded_placement_pose": [0.06, -0.235, 0.069]},
        "samples": [{"policy_labels": {"ee_tool_delta": [0.0] * 6}}],
    }
    try:
        replay_demo_policy_labels(
            demo,
            "ee_tool_delta",
            np.array([0.0, -0.235, 0.069]),
            task="pick_place",
        )
    except ValueError as exc:
        assert "grasp_segment_finalize_sample_index" in str(exc)
    else:
        raise AssertionError("expected ValueError for missing boundary metadata")


def test_pick_place_replay_fails_without_close_phase_contact(monkeypatch):
    spec, demo = _pick_place_demo()

    def contact_only_during_lift(self) -> bool:
        return getattr(self, "_replay_sample_phase", "") == "lift"

    original_step_joint = PickupTaskEvaluator.step_joint_delta_action
    original_step_ee = PickupTaskEvaluator.step_ee_tool_delta_action

    def step_joint(self, *args, **kwargs):
        self._replay_sample_phase = getattr(self, "_replay_sample_phase", "")
        return original_step_joint(self, *args, **kwargs)

    def step_ee(self, *args, **kwargs):
        self._replay_sample_phase = getattr(self, "_replay_sample_phase", "")
        return original_step_ee(self, *args, **kwargs)

    def track_phase(env, sample_index, boundary_index):
        sample = demo["samples"][sample_index]
        env._replay_sample_phase = str(sample.get("phase", ""))
        return maybe_finalize_grasp_at_sample(env, sample_index, boundary_index)

    monkeypatch.setattr(PickupTaskEvaluator, "gripper_object_contact", property(contact_only_during_lift))
    monkeypatch.setattr(PickupTaskEvaluator, "step_joint_delta_action", step_joint)
    monkeypatch.setattr(PickupTaskEvaluator, "step_ee_tool_delta_action", step_ee)
    monkeypatch.setattr("svla.pick_place_replay.maybe_finalize_grasp_at_sample", track_phase)

    replay = replay_demo_policy_labels(
        demo,
        "ee_tool_delta",
        np.asarray(spec.object_pose.xyz, dtype=float),
        task="pick_place",
    )

    assert not replay["contact_during_close"]
    assert not replay["success"]


def test_pick_place_replay_captures_transient_close_contact_from_counter(monkeypatch):
    spec, demo = _pick_place_demo()
    original_step_ee = PickupTaskEvaluator.step_ee_tool_delta_action
    injected = False

    def no_endpoint_contact(self) -> bool:
        return False

    def step_ee(self, *args, **kwargs):
        nonlocal injected
        result = original_step_ee(self, *args, **kwargs)
        if getattr(self, "_replay_sample_phase", "") == "close_gripper" and not injected:
            self._episode_close_contact_steps += 1
            injected = True
        return result

    def track_phase(env, sample_index, boundary_index):
        sample = demo["samples"][sample_index]
        env._replay_sample_phase = str(sample.get("phase", ""))
        return maybe_finalize_grasp_at_sample(env, sample_index, boundary_index)

    monkeypatch.setattr(
        PickupTaskEvaluator,
        "gripper_object_contact",
        property(no_endpoint_contact),
    )
    monkeypatch.setattr(PickupTaskEvaluator, "step_ee_tool_delta_action", step_ee)
    monkeypatch.setattr("svla.pick_place_replay.maybe_finalize_grasp_at_sample", track_phase)

    replay = replay_demo_policy_labels(
        demo,
        "ee_tool_delta",
        np.asarray(spec.object_pose.xyz, dtype=float),
        task="pick_place",
    )

    assert injected
    assert replay["contact_during_close"]


def test_pick_place_replay_fails_when_placement_does_not_succeed(monkeypatch):
    spec, demo = _pick_place_demo()

    def placement_fails(self, *args, **kwargs):
        return False, {
            "placement_xy_error": 0.05,
            "placement_z_error": 0.01,
            "gripper_released": False,
        }

    monkeypatch.setattr(PickupTaskEvaluator, "evaluate_placement", placement_fails)

    replay = replay_demo_policy_labels(
        demo,
        "joint_delta",
        np.asarray(spec.object_pose.xyz, dtype=float),
        task="pick_place",
    )

    assert not replay["placement_achieved"]
    assert not replay["success"]


def test_pick_place_replay_pass_requires_success_and_placement():
    validate_action_replay = _load_validate_action_replay()

    failing = {
        "joint_delta": {
            "successes": 0,
            "total": 1,
            "controller_failure_steps": 0,
            "collision_free_approaches": 1,
            "valid_event_orders": 1,
            "physical_sanity_passes": 1,
            "preclose_contact_steps": 0,
            "placement_achieved": 0,
        }
    }
    passing = {
        "joint_delta": {
            "successes": 1,
            "total": 1,
            "controller_failure_steps": 0,
            "collision_free_approaches": 1,
            "valid_event_orders": 1,
            "physical_sanity_passes": 1,
            "preclose_contact_steps": 0,
            "placement_achieved": 1,
        }
    }

    assert not validate_action_replay._replay_passes("pick_place", failing)
    assert validate_action_replay._replay_passes("pick_place", passing)
