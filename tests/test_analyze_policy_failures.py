from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from analysis.policy_failures import analyze_rows, load_jsonl, render_markdown


def _row(
    *,
    seed: int,
    trial_id: int,
    success: bool,
    event: bool,
    physical: bool,
    clipped: int,
    infeasible: int = 0,
    action_space: str = "ee_tool_delta",
) -> dict[str, object]:
    return {
        "action_space": action_space,
        "seed": seed,
        "trial_id": trial_id,
        "steps": 100,
        "success": success,
        "event_order_valid": event,
        "physical_sanity_pass": physical,
        "collision_free_approach": True,
        "early_close": not event and trial_id == 1,
        "reopen_events": 2 if not event else 0,
        "failure_category": (
            "none"
            if success
            else "event_order_failure"
            if not event
            else "contact_dynamics_failure"
        ),
        "orientation": "yaw_0" if trial_id % 2 else "yaw_18",
        "approach": "vertical_pregrasp",
        "object_pose": "center",
        "clipped_translation_steps": 0,
        "clipped_rotation_steps": 0,
        "clipped_joint_steps": clipped,
        "joint_limit_clipped_steps": clipped // 2,
        "joint_step_clipped_steps": clipped,
        "joint_accel_clipped_steps": 1 if clipped else 0,
        "infeasible_steps": infeasible,
        "controller_failure_steps": 0,
    }


def test_analyzer_preserves_overlapping_gate_failures_and_step_metrics():
    rows = [
        _row(seed=0, trial_id=1, success=False, event=False, physical=False, clipped=40, infeasible=10),
        _row(seed=0, trial_id=2, success=False, event=False, physical=True, clipped=30),
        _row(seed=1, trial_id=3, success=False, event=True, physical=False, clipped=20),
        _row(seed=1, trial_id=4, success=True, event=True, physical=True, clipped=0),
    ]

    summary = analyze_rows(rows)["by_action_space"]["ee_tool_delta"]

    assert summary["total"] == 4
    assert summary["successes"] == 1
    assert summary["event_order_valid_count"] == 2
    assert summary["physical_sanity_pass_count"] == 2
    assert summary["event_and_physical_pass_count"] == 1
    assert summary["event_and_physical_fail_count"] == 1
    assert summary["early_close_trials"] == 1
    assert summary["reopen_trials"] == 2
    assert summary["reopen_events"] == 4
    assert summary["constraints"]["total_rollout_steps"] == 400
    assert summary["constraints"]["step_totals"]["clipped_joint_steps"] == 90
    assert summary["constraints"]["step_rates"]["clipped_joint_steps"] == pytest.approx(0.225)
    assert summary["constraints"]["trials_with_any"]["any_saturation"] == 3
    assert summary["constraints"]["trials_with_any"]["any_hard_limit_or_infeasible"] == 3
    assert [quartile["total"] for quartile in summary["saturation_rate_quartiles"]] == [1, 1, 1, 1]


def test_analyzer_groups_action_space_seed_and_task_buckets():
    rows = [
        _row(seed=0, trial_id=1, success=False, event=False, physical=True, clipped=10),
        _row(seed=1, trial_id=2, success=True, event=True, physical=True, clipped=0),
        _row(
            seed=0,
            trial_id=3,
            success=True,
            event=True,
            physical=True,
            clipped=0,
            action_space="joint_delta",
        ),
    ]

    analysis = analyze_rows(rows)

    assert set(analysis["by_action_space"]) == {"ee_tool_delta", "joint_delta"}
    ee = analysis["by_action_space"]["ee_tool_delta"]
    assert set(ee["by_seed"]) == {"0", "1"}
    assert set(ee["by_orientation"]) == {"yaw_0", "yaw_18"}
    assert ee["seed_variability"]["success_rate"] == {
        "min": 0.0,
        "max": 1.0,
        "mean": 0.5,
        "population_stddev": 0.5,
    }


def test_jsonl_validation_and_cli_outputs(tmp_path: Path):
    input_path = tmp_path / "trials.jsonl"
    rows = [
        _row(seed=0, trial_id=1, success=False, event=False, physical=False, clipped=10),
        _row(seed=0, trial_id=2, success=True, event=True, physical=True, clipped=0),
    ]
    input_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    output_json = tmp_path / "analysis.json"
    output_markdown = tmp_path / "analysis.md"

    subprocess.run(
        [
            sys.executable,
            "scripts/analyze_policy_failures.py",
            str(input_path),
            "--output-json",
            str(output_json),
            "--output-markdown",
            str(output_markdown),
        ],
        check=True,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["format"] == "svla_policy_failure_analysis_v1"
    assert payload["source_paths"] == [str(input_path)]
    assert "# Policy failure analysis" in output_markdown.read_text(encoding="utf-8")
    assert "`clipped_joint_steps`" in render_markdown(payload)


def test_jsonl_loader_reports_missing_fields(tmp_path: Path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"action_space": "ee_tool_delta"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="missing required fields"):
        load_jsonl([path])
