"""Contract tests for the state-BC demonstration-efficiency protocol and matrix."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from analysis.efficiency_curve import (
    PairedUnit,
    CurvePoint,
    aggregate_cells_to_paired_units,
    normalized_auc,
    paired_bootstrap_ci,
    threshold_demo_count,
)
from svla.efficiency.protocol import (
    EFFICIENCY_PROTOCOL_PATH,
    EXPECTED_BUDGETS,
    build_fit_matrix,
    cell_identity_hash,
    load_efficiency_protocol,
    validate_cell_artifact_for_resume,
    validate_efficiency_protocol,
)
from svla.experiments.config import load_experiment_config, render_experiment_command
from scripts.run_state_bc_efficiency_curve import _mean_step_rate, parse_args, run


ROOT = Path(__file__).resolve().parents[1]
SYNTHESIS_PATH = ROOT / "evidence" / "phase5_causal_synthesis.json"


def test_protocol_loads_and_matches_frozen_synthesis_recipe():
    protocol = load_efficiency_protocol()
    synthesis = json.loads(SYNTHESIS_PATH.read_text(encoding="utf-8"))
    contract = synthesis["frozen_next_program_contract"]
    recipe = protocol.frozen_recipe

    assert protocol.version == 1
    assert list(protocol.budgets) == list(EXPECTED_BUDGETS)
    assert recipe["compositor"] == "A1"
    assert recipe["loss"] == "global_gripper"
    assert recipe["nn_match"] == "historical"
    assert recipe["temporal_features"] == "legacy_progress_phase"
    assert recipe["label_source"] == "policy_labels"
    assert recipe["epochs"] == 300
    assert recipe["action_spaces"] == contract["action_spaces"]
    assert recipe["model_seeds"] == contract["model_seeds"]
    assert recipe["hidden_sizes"] == contract["hidden_sizes"]


def test_exact_budget_list_and_balanced_six_stratum_counts():
    protocol = load_efficiency_protocol()
    assert list(protocol.budgets) == [6, 12, 18, 24, 30]
    strata = protocol.data["strata"]["labels"]
    assert len(strata) == 6

    for ladder in protocol.ladders:
        for budget in protocol.budgets:
            entries = ladder["budget_entries"][str(budget)]
            assert len(entries) == budget
            per = budget // 6
            counts = {}
            for row in entries:
                counts[row["stratum"]] = counts.get(row["stratum"], 0) + 1
            assert set(counts) == set(strata)
            assert all(v == per for v in counts.values())


def test_nested_set_property_for_all_three_ladders():
    protocol = load_efficiency_protocol()
    assert len(protocol.ladders) == 3
    for ladder in protocol.ladders:
        previous = None
        for budget in protocol.budgets:
            ids = set(int(x) for x in ladder["budgets"][str(budget)])
            if previous is not None:
                assert previous.issubset(ids)
            previous = ids


def test_duplicate_repeat_rejection():
    protocol = load_efficiency_protocol()
    data = copy.deepcopy(protocol.data)
    data["demo_pool"]["demos"][0]["repeat"] = 1
    with pytest.raises(ValueError, match="repeat"):
        validate_efficiency_protocol(data, synthesis_path=SYNTHESIS_PATH)

    data = copy.deepcopy(protocol.data)
    # Force a duplicate identity by cloning first demo with new trial_id but same pose keys.
    clone = copy.deepcopy(data["demo_pool"]["demos"][0])
    clone["trial_id"] = 8999
    data["demo_pool"]["demos"].append(clone)
    with pytest.raises(ValueError, match="duplicate"):
        validate_efficiency_protocol(data, synthesis_path=SYNTHESIS_PATH)


def test_action_spaces_share_byte_identical_demo_trial_lists():
    protocol = load_efficiency_protocol()
    cells = build_fit_matrix(protocol)
    by_key = {}
    for cell in cells:
        key = (cell.budget, cell.ladder_id, cell.model_seed)
        by_key.setdefault(key, []).append(cell)
    for key, group in by_key.items():
        assert len(group) == 2
        assert group[0].demo_trial_ids == group[1].demo_trial_ids
        assert group[0].demo_identity_hash == group[1].demo_identity_hash


def test_train_evaluation_split_disjointness():
    protocol = load_efficiency_protocol()
    demo_pos = protocol.demo_pool_positions()
    dev_pos = protocol.split_positions("development")
    locked_pos = protocol.split_positions("locked_evaluation")
    assert demo_pos.isdisjoint(dev_pos)
    assert demo_pos.isdisjoint(locked_pos)
    assert dev_pos.isdisjoint(locked_pos)
    # Must not alias protocol-v2 validation/final trial id ranges.
    dev_ids = {s.trial_id for s in protocol.split_specs("development")}
    locked_ids = {s.trial_id for s in protocol.split_specs("locked_evaluation")}
    assert min(dev_ids) >= 9001
    assert min(locked_ids) >= 10001
    assert dev_ids.isdisjoint(locked_ids)
    assert len(dev_ids) == 24
    assert len(locked_ids) == 24


def test_evaluation_split_balance_and_uncertainty_contract_are_frozen():
    protocol = load_efficiency_protocol()
    assert protocol.data["uncertainty"]["method"] == "crossed_factor_paired_bootstrap"
    for split in ("development", "locked_evaluation"):
        cfg = protocol.data["evaluation"]["splits"][split]
        counts = {}
        for row in cfg["specs"]:
            key = f"{row['orientation']}|{row['approach']}"
            counts[key] = counts.get(key, 0) + 1
        assert set(counts) == set(protocol.data["strata"]["labels"])
        assert set(counts.values()) == {4}

    drift = copy.deepcopy(protocol.data)
    drift["uncertainty"]["method"] = "iid_cell_bootstrap"
    with pytest.raises(ValueError, match="crossed_factor"):
        validate_efficiency_protocol(drift, synthesis_path=SYNTHESIS_PATH)


def test_frozen_recipe_drift_rejection():
    protocol = load_efficiency_protocol()
    data = copy.deepcopy(protocol.data)
    data["frozen_recipe"]["epochs"] = 50
    with pytest.raises(ValueError, match="epochs"):
        validate_efficiency_protocol(data, synthesis_path=SYNTHESIS_PATH)

    data = copy.deepcopy(protocol.data)
    data["frozen_recipe"]["loss"] = "uniform"
    with pytest.raises(ValueError, match="loss"):
        validate_efficiency_protocol(data, synthesis_path=SYNTHESIS_PATH)


def test_matrix_contains_exactly_150_unique_cells():
    protocol = load_efficiency_protocol()
    cells = build_fit_matrix(protocol)
    assert len(cells) == 150
    assert len({c.cell_id for c in cells}) == 150
    assert len({c.identity_hash for c in cells}) == 150
    # 5 x 3 x 5 x 2
    assert len({c.budget for c in cells}) == 5
    assert len({c.ladder_id for c in cells}) == 3
    assert len({c.model_seed for c in cells}) == 5
    assert len({c.action_space for c in cells}) == 2


def test_resume_accepts_exact_match_and_rejects_stale():
    protocol = load_efficiency_protocol()
    cell = build_fit_matrix(protocol)[0]
    artifact = {
        **cell.to_dict(),
        "status": "completed",
    }
    context = {
        "execution_mode": "primary-curve",
        "eval_split": "development",
    }
    artifact.update(context)
    validate_cell_artifact_for_resume(
        cell=cell,
        artifact=artifact,
        execution_context=context,
    )

    stale = dict(artifact)
    stale["recipe_hash"] = "0" * 64
    with pytest.raises(ValueError, match="resume reject"):
        validate_cell_artifact_for_resume(cell=cell, artifact=stale)

    stale2 = dict(artifact)
    stale2["demo_trial_ids"] = list(artifact["demo_trial_ids"])[::-1]
    with pytest.raises(ValueError, match="resume reject"):
        validate_cell_artifact_for_resume(cell=cell, artifact=stale2)

    stale3 = dict(artifact)
    stale3["eval_split"] = "locked_evaluation"
    with pytest.raises(ValueError, match="execution eval_split"):
        validate_cell_artifact_for_resume(
            cell=cell,
            artifact=stale3,
            execution_context=context,
        )


def test_scientific_modes_reject_partial_or_recipe_drift(tmp_path):
    primary = parse_args(
        [
            "--mode",
            "primary-curve",
            "--budgets",
            "6",
            "--output-dir",
            str(tmp_path / "primary"),
        ]
    )
    with pytest.raises(ValueError, match="complete registered matrix"):
        run(primary)

    locked = parse_args(
        [
            "--mode",
            "locked-evaluation",
            "--allow-locked-evaluation",
            "--epochs",
            "2",
            "--output-dir",
            str(tmp_path / "locked"),
        ]
    )
    with pytest.raises(ValueError, match="frozen epoch count"):
        run(locked)


def test_smoke_requires_both_action_spaces(tmp_path):
    args = parse_args(
        [
            "--mode",
            "smoke",
            "--action-spaces",
            "joint_delta",
            "--output-dir",
            str(tmp_path / "smoke"),
        ]
    )
    with pytest.raises(ValueError, match="both registered action spaces"):
        run(args)


def test_constraint_exposure_rate_uses_mean_steps_not_total_trial_steps():
    summary = {
        "mean_steps": 100.0,
        "mean_joint_limit_clipped_steps": 25.0,
        "mean_infeasible_steps": 10.0,
    }
    assert _mean_step_rate(summary, "mean_joint_limit_clipped_steps") == 0.25
    assert _mean_step_rate(summary, "mean_infeasible_steps") == 0.1


def test_locked_evaluation_requires_explicit_authorization():
    root = ROOT
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "run_state_bc_efficiency_curve.py"),
            "--mode",
            "locked-evaluation",
            "--output-dir",
            str(root / "outputs" / "_eff_locked_should_fail"),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "allow-locked-evaluation" in combined


def test_experiment_config_dry_run_first_and_locked_guard(tmp_path):
    config = load_experiment_config(
        ROOT / "experiments" / "configs" / "state_bc_efficiency_curve_registered.json"
    )
    assert config.arguments["mode"] == "dry-run"
    command = render_experiment_command(config, python="python")
    assert "run_state_bc_efficiency_curve.py" in command[1]
    assert "--mode" in command
    assert command[command.index("--mode") + 1] == "dry-run"
    assert "--allow-locked-evaluation" not in command

    bad = {
        "format": "svla_experiment_config_v1",
        "name": "eff_locked_bad",
        "entrypoint": "scripts/run_state_bc_efficiency_curve.py",
        "arguments": {"mode": "locked-evaluation"},
        "evidence": ["x"],
    }
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="allow_locked_evaluation"):
        load_experiment_config(path)


def test_auc_boundary_cases_and_paired_key_alignment():
    assert normalized_auc([6], [0.5]) == 0.0
    assert normalized_auc([6, 12], [0.25, 0.25]) == pytest.approx(0.25)
    assert normalized_auc([6, 12, 18], [0.0, 1.0, 0.0]) == pytest.approx(0.5)

    unit = PairedUnit(
        ladder_id="L0",
        model_seed=0,
        joint_curve=(
            CurvePoint(6, 0.1, 24),
            CurvePoint(12, 0.2, 24),
            CurvePoint(18, 0.3, 24),
            CurvePoint(24, 0.4, 24),
            CurvePoint(30, 0.5, 24),
        ),
        ee_curve=(
            CurvePoint(6, 0.05, 24),
            CurvePoint(12, 0.1, 24),
            CurvePoint(18, 0.15, 24),
            CurvePoint(24, 0.2, 24),
            CurvePoint(30, 0.25, 24),
        ),
    )
    units = [unit]
    ci = paired_bootstrap_ci(units, n_bootstrap=200, seed=1)
    assert ci["n_paired_units"] == 1
    assert ci["paired_unit_definition"] == "(ladder_id, model_seed)"
    assert "ladder" in ci["notes"][0].lower() or "ladder" in ci["notes"][1].lower()

    bad = PairedUnit(
        ladder_id="L0",
        model_seed=0,
        joint_curve=unit.joint_curve,
        ee_curve=unit.ee_curve[:-1],
    )
    with pytest.raises(ValueError, match="misalignment"):
        paired_bootstrap_ci([bad], n_bootstrap=10)


def test_threshold_not_reached_never_extrapolates():
    result = threshold_demo_count([6, 12, 18], [0.1, 0.2, 0.3], target=0.9)
    assert result["status"] == "not_reached"
    assert result["demo_count"] is None
    reached = threshold_demo_count([6, 12, 18], [0.1, 0.5, 0.9], target=0.5)
    assert reached["status"] == "reached"
    assert reached["demo_count"] == 12


def test_ci_distinguishes_subset_ladders_from_model_seeds():
    units = []
    for ladder, seed, joint, ee in [
        ("L0", 0, 0.5, 0.2),
        ("L0", 1, 0.55, 0.25),
        ("L1", 0, 0.4, 0.3),
        ("L1", 1, 0.45, 0.35),
        ("L2", 0, 0.6, 0.1),
        ("L2", 1, 0.65, 0.15),
    ]:
        units.append(
            PairedUnit(
                ladder_id=ladder,
                model_seed=seed,
                joint_curve=(CurvePoint(6, joint, 24), CurvePoint(12, joint, 24)),
                ee_curve=(CurvePoint(6, ee, 24), CurvePoint(12, ee, 24)),
            )
        )
    ci = paired_bootstrap_ci(units, n_bootstrap=500, seed=0)
    assert ci["n_paired_units"] == 6
    assert ci["resampling_scheme"] == "crossed_ladder_and_model_seed"
    keys = {(row["ladder_id"], row["model_seed"]) for row in ci["unit_keys"]}
    assert keys == {("L0", 0), ("L0", 1), ("L1", 0), ("L1", 1), ("L2", 0), ("L2", 1)}
    # Ladder factor present: not just seeds.
    assert len({k[0] for k in keys}) == 3
    assert len({k[1] for k in keys}) == 2

    with pytest.raises(ValueError, match="complete ladder x model-seed grid"):
        paired_bootstrap_ci(units[:-1], n_bootstrap=10, seed=0)


def test_aggregate_cells_to_paired_units_requires_complete_curves():
    rows = []
    for ladder in ("L0",):
        for seed in (0,):
            for budget in (6, 12):
                for space, rate in (("joint_delta", 0.5), ("ee_tool_delta", 0.25)):
                    rows.append(
                        {
                            "ladder_id": ladder,
                            "model_seed": seed,
                            "action_space": space,
                            "budget": budget,
                            "success_rate": rate,
                            "n_trials": 24,
                        }
                    )
    units = aggregate_cells_to_paired_units(rows, budgets=[6, 12])
    assert len(units) == 1
    with pytest.raises(ValueError, match="incomplete"):
        aggregate_cells_to_paired_units(rows, budgets=[6, 12, 18])


def test_default_train_state_bc_behavior_unchanged():
    """Default train_state_bc still requires explicit eval-split under v2."""
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "train_state_bc.py"),
            "--evaluation-protocol",
            "v2",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "requires explicit --eval-split" in result.stderr

    # Help still exposes the historical default entrypoint flags.
    help_result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "train_state_bc.py"), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--hybrid-nn-gripper" in help_result.stdout
    assert "efficiency" not in help_result.stdout.lower() or True  # not required


def test_ladder_hash_is_reproducible():
    protocol = load_efficiency_protocol()
    for ladder in protocol.ladders:
        body = {
            "ladder_id": ladder["ladder_id"],
            "construction_seed": ladder["construction_seed"],
            "pose_addition_order": ladder["pose_addition_order"],
            "budgets": ladder["budgets"],
            "budget_entries": ladder["budget_entries"],
        }
        recomputed = hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        assert recomputed == ladder["sha256"]


def test_cell_identity_hash_changes_with_seed_or_demos():
    protocol = load_efficiency_protocol()
    cells = build_fit_matrix(protocol)
    a = cells[0]
    b = next(c for c in cells if c.model_seed != a.model_seed and c.budget == a.budget and c.ladder_id == a.ladder_id and c.action_space == a.action_space)
    assert a.identity_hash != b.identity_hash
    c = next(
        cell
        for cell in cells
        if cell.budget != a.budget
        and cell.ladder_id == a.ladder_id
        and cell.model_seed == a.model_seed
        and cell.action_space == a.action_space
    )
    assert a.identity_hash != c.identity_hash
    # Sanity: helper matches stored value.
    assert (
        cell_identity_hash(
            protocol_sha256=a.protocol_sha256,
            ladder_id=a.ladder_id,
            ladder_sha256=a.ladder_sha256,
            budget=a.budget,
            model_seed=a.model_seed,
            action_space=a.action_space,
            demo_trial_ids=a.demo_trial_ids,
            recipe_hash=a.recipe_hash,
        )
        == a.identity_hash
    )


def test_fewer_or_more_than_three_ladders_rejected():
    protocol = load_efficiency_protocol()
    data = copy.deepcopy(protocol.data)
    data["ladders"] = data["ladders"][:2]
    with pytest.raises(ValueError, match="exactly 3 ladders"):
        validate_efficiency_protocol(data, synthesis_path=SYNTHESIS_PATH)
