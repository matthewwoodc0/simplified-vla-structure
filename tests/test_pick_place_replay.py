import numpy as np

from svla.pick_place_replay import replay_demo_policy_labels
from svla.pickup_task import maybe_finalize_grasp_at_sample


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