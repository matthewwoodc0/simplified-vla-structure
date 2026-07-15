"""Loader, validator, and fit-matrix builder for the efficiency-curve protocol."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from svla.pickup_task import (
    OBJECT_START_Z,
    ApproachStrategy,
    GraspOrientation,
    ObjectStartPose,
    PickupTrialSpec,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EFFICIENCY_PROTOCOL_FORMAT = "svla_state_bc_efficiency_protocol_v1"
EFFICIENCY_PROTOCOL_PATH = (
    PROJECT_ROOT / "configs" / "state_bc_efficiency_protocol_v1.json"
)
SYNTHESIS_PATH = PROJECT_ROOT / "evidence" / "phase5_causal_synthesis.json"
EXPECTED_BUDGETS = (6, 12, 18, 24, 30)
EXPECTED_LADDER_COUNT = 3
EXPECTED_SEED_COUNT = 5
EXPECTED_STRATUM_COUNT = 6
EXPECTED_EVAL_TRIAL_COUNT = 24
ACTION_SPACES = ("joint_delta", "ee_tool_delta")
EVAL_SPLITS = ("development", "locked_evaluation")
FORBIDDEN_SPLIT_ALIASES = ("validation", "final", "train")

# Canonical frozen recipe fields required to match phase5_causal_synthesis.
FROZEN_RECIPE_REQUIRED = {
    "action_spaces": ["joint_delta", "ee_tool_delta"],
    "policy_family": "hybrid_nn_gripper_mlp",
    "compositor": "A1",
    "loss": "global_gripper",
    "nn_match": "historical",
    "temporal_features": "legacy_progress_phase",
    "label_source": "policy_labels",
    "hidden_sizes": [128, 128],
    "epochs": 300,
    "batch": 1024,
    "learning_rate": 0.001,
    "weight_decay": 1e-05,
    "action_gain": 1.0,
    "model_seeds": [0, 1, 2, 3, 4],
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(raw)


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


@dataclass(frozen=True)
class MatrixCell:
    cell_id: str
    budget: int
    ladder_id: str
    model_seed: int
    action_space: str
    demo_trial_ids: tuple[int, ...]
    demo_identity_hash: str
    recipe_hash: str
    protocol_sha256: str
    ladder_sha256: str
    identity_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "budget": self.budget,
            "ladder_id": self.ladder_id,
            "model_seed": self.model_seed,
            "action_space": self.action_space,
            "demo_trial_ids": list(self.demo_trial_ids),
            "demo_identity_hash": self.demo_identity_hash,
            "recipe_hash": self.recipe_hash,
            "protocol_sha256": self.protocol_sha256,
            "ladder_sha256": self.ladder_sha256,
            "identity_hash": self.identity_hash,
        }


@dataclass(frozen=True)
class EfficiencyProtocol:
    path: Path
    data: dict
    sha256: str

    @property
    def version(self) -> int:
        return int(self.data["version"])

    @property
    def budgets(self) -> tuple[int, ...]:
        return tuple(int(b) for b in self.data["data_budgets"])

    @property
    def model_seeds(self) -> tuple[int, ...]:
        return tuple(int(s) for s in self.data["frozen_recipe"]["model_seeds"])

    @property
    def action_spaces(self) -> tuple[str, ...]:
        return tuple(str(s) for s in self.data["frozen_recipe"]["action_spaces"])

    @property
    def ladders(self) -> list[dict]:
        return list(self.data["ladders"])

    @property
    def frozen_recipe(self) -> dict:
        return dict(self.data["frozen_recipe"])

    @property
    def recipe_hash(self) -> str:
        return sha256_json(self.frozen_recipe)

    def ladder_by_id(self, ladder_id: str) -> dict:
        for ladder in self.ladders:
            if ladder["ladder_id"] == ladder_id:
                return ladder
        raise KeyError(f"unknown ladder_id: {ladder_id}")

    def demo_trial_ids_for(self, ladder_id: str, budget: int) -> tuple[int, ...]:
        ladder = self.ladder_by_id(ladder_id)
        ids = ladder["budgets"][str(int(budget))]
        return tuple(int(x) for x in ids)

    def demo_entries_for(self, ladder_id: str, budget: int) -> list[dict]:
        ladder = self.ladder_by_id(ladder_id)
        return list(ladder["budget_entries"][str(int(budget))])

    def demo_pool_by_trial_id(self) -> dict[int, dict]:
        return {
            int(row["trial_id"]): dict(row)
            for row in self.data["demo_pool"]["demos"]
        }

    def split_specs(self, split: str) -> list[PickupTrialSpec]:
        if split not in EVAL_SPLITS:
            raise ValueError(
                f"efficiency protocol split must be one of {EVAL_SPLITS}, got {split!r}"
            )
        if split in FORBIDDEN_SPLIT_ALIASES:
            raise ValueError(f"refusing protocol-v2 split alias: {split}")
        split_cfg = self.data["evaluation"]["splits"][split]
        specs: list[PickupTrialSpec] = []
        for row in split_cfg["specs"]:
            specs.append(
                PickupTrialSpec(
                    trial_id=int(row["trial_id"]),
                    orientation=GraspOrientation(
                        str(row["orientation"]), float(row["yaw_degrees"])
                    ),
                    object_pose=ObjectStartPose(
                        str(row["object_pose"]),
                        np.asarray(row["object_xyz"], dtype=float),
                    ),
                    approach=ApproachStrategy(
                        str(row["approach"]), str(row["axis_mode"])
                    ),
                    repeat=int(row.get("repeat", 0)),
                )
            )
        return specs

    def split_positions(self, split: str) -> set[tuple[float, float, float]]:
        specs = self.split_specs(split)
        return {
            tuple(float(v) for v in spec.object_pose.xyz.tolist()) for spec in specs
        }

    def demo_pool_positions(self) -> set[tuple[float, float, float]]:
        return {
            tuple(float(v) for v in row["object_xyz"])
            for row in self.data["demo_pool"]["demos"]
        }

    def metadata(self, split: str) -> dict[str, Any]:
        self.split_specs(split)  # validate
        return {
            "format": self.data["format"],
            "version": self.version,
            "config_path": str(self.path),
            "config_sha256": self.sha256,
            "eval_split": split,
            "not_protocol_v2_alias": True,
        }


def load_efficiency_protocol(
    path: Path = EFFICIENCY_PROTOCOL_PATH,
    *,
    synthesis_path: Path = SYNTHESIS_PATH,
) -> EfficiencyProtocol:
    path = Path(path)
    raw = path.read_bytes()
    data = json.loads(raw)
    validate_efficiency_protocol(data, synthesis_path=synthesis_path)
    return EfficiencyProtocol(path=path.resolve(), data=data, sha256=sha256_bytes(raw))


def validate_efficiency_protocol(
    data: dict,
    *,
    synthesis_path: Path = SYNTHESIS_PATH,
) -> None:
    if data.get("format") != EFFICIENCY_PROTOCOL_FORMAT:
        raise ValueError(
            f"efficiency protocol must use {EFFICIENCY_PROTOCOL_FORMAT}, "
            f"got {data.get('format')!r}"
        )
    if int(data.get("version", -1)) != 1:
        raise ValueError("efficiency protocol version must be 1")

    budgets = [int(b) for b in data.get("data_budgets", [])]
    if budgets != list(EXPECTED_BUDGETS):
        raise ValueError(
            f"data_budgets must be exactly {list(EXPECTED_BUDGETS)}, got {budgets}"
        )

    recipe = data.get("frozen_recipe", {})
    _validate_frozen_recipe(recipe, synthesis_path=synthesis_path)

    strata_labels = list(data.get("strata", {}).get("labels", []))
    if len(strata_labels) != EXPECTED_STRATUM_COUNT:
        raise ValueError(
            f"expected {EXPECTED_STRATUM_COUNT} strata, got {len(strata_labels)}"
        )
    if len(set(strata_labels)) != len(strata_labels):
        raise ValueError("stratum labels must be unique")

    demos = list(data.get("demo_pool", {}).get("demos", []))
    _validate_demo_pool(demos, strata_labels=strata_labels)

    ladders = list(data.get("ladders", []))
    if len(ladders) != EXPECTED_LADDER_COUNT:
        raise ValueError(
            f"expected exactly {EXPECTED_LADDER_COUNT} ladders, got {len(ladders)}"
        )
    ladder_ids = [str(row["ladder_id"]) for row in ladders]
    if len(set(ladder_ids)) != len(ladder_ids):
        raise ValueError("ladder_id values must be unique")
    for ladder in ladders:
        _validate_ladder(ladder, demos=demos, budgets=budgets, strata_labels=strata_labels)

    evaluation = data.get("evaluation", {})
    splits = evaluation.get("splits", {})
    if set(splits) != set(EVAL_SPLITS):
        raise ValueError(f"evaluation splits must be exactly {EVAL_SPLITS}")
    for forbidden in FORBIDDEN_SPLIT_ALIASES:
        if forbidden in splits:
            raise ValueError(
                f"efficiency protocol must not alias protocol-v2 split {forbidden!r}"
            )

    positions_by_split: dict[str, set[tuple[float, float, float]]] = {}
    all_trial_ids: set[int] = set()
    for split_name, split_cfg in splits.items():
        positions = set()
        for row in split_cfg["specs"]:
            trial_id = int(row["trial_id"])
            if trial_id in all_trial_ids:
                raise ValueError(f"duplicate trial_id across splits: {trial_id}")
            all_trial_ids.add(trial_id)
            xyz = tuple(float(v) for v in row["object_xyz"])
            if len(xyz) != 3 or not np.isfinite(xyz).all():
                raise ValueError(f"invalid object_xyz in split {split_name}")
            if not np.isclose(xyz[2], OBJECT_START_Z, atol=1e-12):
                raise ValueError(
                    f"split {split_name} pose z={xyz[2]} != OBJECT_START_Z={OBJECT_START_Z}"
                )
            positions.add(xyz)
        positions_by_split[split_name] = positions
        _validate_split_specs(split_cfg, strata_labels=strata_labels)

    demo_positions = {
        tuple(float(v) for v in row["object_xyz"]) for row in demos
    }
    for split_name, positions in positions_by_split.items():
        overlap = positions & demo_positions
        if overlap:
            raise ValueError(
                f"train/evaluation position overlap between demo_pool and "
                f"{split_name}: {sorted(overlap)[:3]}"
            )
    dev_locked = positions_by_split["development"] & positions_by_split["locked_evaluation"]
    if dev_locked:
        raise ValueError(
            f"development and locked_evaluation positions overlap: {sorted(dev_locked)[:3]}"
        )

    matrix = data.get("matrix", {})
    planned = int(matrix.get("planned_cell_count", -1))
    if planned != 150:
        raise ValueError(f"planned_cell_count must be 150, got {planned}")

    uncertainty = data.get("uncertainty", {})
    if uncertainty.get("method") != "crossed_factor_paired_bootstrap":
        raise ValueError("uncertainty method must be crossed_factor_paired_bootstrap")
    if uncertainty.get("paired_unit") != "(ladder_id, model_seed)":
        raise ValueError("uncertainty paired_unit contract drift")
    if int(uncertainty.get("paired_unit_count", -1)) != 15:
        raise ValueError("uncertainty paired_unit_count must be 15")


def _validate_frozen_recipe(recipe: dict, *, synthesis_path: Path) -> None:
    for key, expected in FROZEN_RECIPE_REQUIRED.items():
        if key not in recipe:
            raise ValueError(f"frozen_recipe missing required field: {key}")
        actual = recipe[key]
        if key in {"learning_rate", "weight_decay", "action_gain"}:
            if not np.isclose(float(actual), float(expected), rtol=0.0, atol=1e-15):
                raise ValueError(
                    f"frozen_recipe.{key} drift: expected {expected}, got {actual}"
                )
        elif actual != expected:
            raise ValueError(
                f"frozen_recipe.{key} drift: expected {expected}, got {actual}"
            )

    if not synthesis_path.is_file():
        raise ValueError(f"missing synthesis freeze file: {synthesis_path}")
    synthesis = json.loads(synthesis_path.read_text(encoding="utf-8"))
    contract = synthesis.get("frozen_next_program_contract", {})
    mapping = {
        "action_spaces": "action_spaces",
        "policy_family": "policy_family",
        "compositor": "compositor",
        "loss": "loss",
        "nn_match": "nn_match",
        "temporal_features": "temporal_features",
        "label_source": "label_source",
        "hidden_sizes": "hidden_sizes",
        "epochs": "epochs",
        "batch": "batch",
        "learning_rate": "learning_rate",
        "weight_decay": "weight_decay",
        "action_gain": "action_gain",
        "model_seeds": "model_seeds",
    }
    for recipe_key, contract_key in mapping.items():
        if contract_key not in contract:
            raise ValueError(
                f"synthesis freeze missing frozen_next_program_contract.{contract_key}"
            )
        left = recipe[recipe_key]
        right = contract[contract_key]
        if recipe_key in {"learning_rate", "weight_decay", "action_gain"}:
            if not np.isclose(float(left), float(right), rtol=0.0, atol=1e-15):
                raise ValueError(
                    f"frozen recipe drift vs synthesis for {recipe_key}: "
                    f"{left} != {right}"
                )
        elif left != right:
            raise ValueError(
                f"frozen recipe drift vs synthesis for {recipe_key}: {left} != {right}"
            )


def _validate_demo_pool(demos: list[dict], *, strata_labels: list[str]) -> None:
    if not demos:
        raise ValueError("demo_pool is empty")
    trial_ids: list[int] = []
    identity_keys: list[tuple] = []
    for row in demos:
        trial_id = int(row["trial_id"])
        trial_ids.append(trial_id)
        repeat = int(row.get("repeat", 0))
        if repeat != 0:
            raise ValueError(
                f"demo trial_id={trial_id} has repeat={repeat}; "
                "deterministic duplicates do not count as distinct demos"
            )
        key = (
            str(row["orientation"]),
            str(row["object_pose"]),
            tuple(float(v) for v in row["object_xyz"]),
            str(row["approach"]),
            repeat,
        )
        identity_keys.append(key)
        stratum = str(row["stratum"])
        expected_stratum = f"{row['orientation']}|{row['approach']}"
        if stratum != expected_stratum:
            raise ValueError(
                f"demo trial_id={trial_id} stratum mismatch: {stratum} != {expected_stratum}"
            )
        if stratum not in strata_labels:
            raise ValueError(f"demo trial_id={trial_id} unknown stratum {stratum}")
        xyz = tuple(float(v) for v in row["object_xyz"])
        if not np.isclose(xyz[2], OBJECT_START_Z, atol=1e-12):
            raise ValueError(f"demo trial_id={trial_id} z != OBJECT_START_Z")

    if len(trial_ids) != len(set(trial_ids)):
        raise ValueError("demo_pool contains duplicate trial_id values")
    if len(identity_keys) != len(set(identity_keys)):
        raise ValueError(
            "demo_pool contains duplicate (orientation, pose, xyz, approach, repeat) "
            "entries counted as distinct"
        )
    # Balanced full pool: equal demos per stratum.
    counts = Counter(str(row["stratum"]) for row in demos)
    if set(counts) != set(strata_labels):
        raise ValueError("demo_pool strata do not match protocol stratum labels")
    if len(set(counts.values())) != 1:
        raise ValueError(f"demo_pool stratum imbalance: {dict(counts)}")


def _validate_ladder(
    ladder: dict,
    *,
    demos: list[dict],
    budgets: list[int],
    strata_labels: list[str],
) -> None:
    ladder_id = str(ladder.get("ladder_id", ""))
    if not ladder_id:
        raise ValueError("ladder missing ladder_id")
    demo_by_id = {int(row["trial_id"]): row for row in demos}
    budget_map = ladder.get("budgets", {})
    entries_map = ladder.get("budget_entries", {})
    if set(str(b) for b in budgets) != set(budget_map) or set(str(b) for b in budgets) != set(
        entries_map
    ):
        raise ValueError(f"ladder {ladder_id} must define all budgets {budgets}")

    previous_ids: set[int] | None = None
    for budget in budgets:
        key = str(budget)
        ids = [int(x) for x in budget_map[key]]
        entries = list(entries_map[key])
        if len(ids) != budget:
            raise ValueError(
                f"ladder {ladder_id} budget {budget} has {len(ids)} ids, expected {budget}"
            )
        if len(ids) != len(set(ids)):
            raise ValueError(
                f"ladder {ladder_id} budget {budget} has duplicate trial ids"
            )
        if len(entries) != budget:
            raise ValueError(
                f"ladder {ladder_id} budget {budget} entries length {len(entries)} "
                f"!= budget {budget}"
            )
        entry_ids = [int(row["trial_id"]) for row in entries]
        if entry_ids != ids:
            raise ValueError(
                f"ladder {ladder_id} budget {budget} budgets list != budget_entries trial_ids"
            )
        # Distinctness and membership.
        for trial_id in ids:
            if trial_id not in demo_by_id:
                raise ValueError(
                    f"ladder {ladder_id} budget {budget} unknown trial_id {trial_id}"
                )
            if int(demo_by_id[trial_id].get("repeat", 0)) != 0:
                raise ValueError(
                    f"ladder {ladder_id} includes repeat demo trial_id={trial_id}"
                )
        # Balanced strata: budget / 6 demos per stratum.
        if budget % EXPECTED_STRATUM_COUNT != 0:
            raise ValueError(
                f"budget {budget} is not divisible by stratum count {EXPECTED_STRATUM_COUNT}"
            )
        per_stratum = budget // EXPECTED_STRATUM_COUNT
        stratum_counts = Counter(str(row["stratum"]) for row in entries)
        if set(stratum_counts) != set(strata_labels):
            raise ValueError(
                f"ladder {ladder_id} budget {budget} missing strata: "
                f"{set(strata_labels) - set(stratum_counts)}"
            )
        if any(count != per_stratum for count in stratum_counts.values()):
            raise ValueError(
                f"ladder {ladder_id} budget {budget} stratum imbalance: "
                f"{dict(stratum_counts)} (expected {per_stratum} each)"
            )
        # Nested property.
        id_set = set(ids)
        if previous_ids is not None and not previous_ids.issubset(id_set):
            missing = sorted(previous_ids - id_set)
            raise ValueError(
                f"ladder {ladder_id} is not nested at budget {budget}; "
                f"missing prior ids {missing[:5]}"
            )
        previous_ids = id_set

    # Recompute ladder hash over explicit contract body (without sha256 field).
    body = {
        "ladder_id": ladder["ladder_id"],
        "construction_seed": ladder["construction_seed"],
        "pose_addition_order": ladder["pose_addition_order"],
        "budgets": ladder["budgets"],
        "budget_entries": ladder["budget_entries"],
    }
    expected_hash = sha256_json(body)
    actual_hash = str(ladder.get("sha256", ""))
    if actual_hash != expected_hash:
        raise ValueError(
            f"ladder {ladder_id} sha256 mismatch: stored={actual_hash} recomputed={expected_hash}"
        )


def _validate_split_specs(split_cfg: dict, *, strata_labels: list[str]) -> None:
    specs = list(split_cfg.get("specs", []))
    if len(specs) != EXPECTED_EVAL_TRIAL_COUNT:
        raise ValueError(
            f"evaluation split must contain {EXPECTED_EVAL_TRIAL_COUNT} specs, "
            f"got {len(specs)}"
        )
    if int(split_cfg.get("trial_count", -1)) != len(specs):
        raise ValueError("evaluation split trial_count does not match specs")
    trial_ids = [int(row["trial_id"]) for row in specs]
    if len(trial_ids) != len(set(trial_ids)):
        raise ValueError("evaluation split has duplicate trial_ids")
    # Each (orientation, approach) stratum must appear; positions may vary.
    counts = Counter(f"{row['orientation']}|{row['approach']}" for row in specs)
    if set(counts) != set(strata_labels):
        raise ValueError(
            f"evaluation split strata mismatch: {set(strata_labels) - set(counts)}"
        )
    expected_per_stratum = EXPECTED_EVAL_TRIAL_COUNT // EXPECTED_STRATUM_COUNT
    if any(count != expected_per_stratum for count in counts.values()):
        raise ValueError(f"evaluation split stratum imbalance: {dict(counts)}")


def demo_identity_hash(trial_ids: Iterable[int]) -> str:
    payload = [int(x) for x in trial_ids]
    return sha256_json(payload)


def cell_identity_hash(
    *,
    protocol_sha256: str,
    ladder_id: str,
    ladder_sha256: str,
    budget: int,
    model_seed: int,
    action_space: str,
    demo_trial_ids: Iterable[int],
    recipe_hash: str,
) -> str:
    payload = {
        "protocol_sha256": protocol_sha256,
        "ladder_id": ladder_id,
        "ladder_sha256": ladder_sha256,
        "budget": int(budget),
        "model_seed": int(model_seed),
        "action_space": action_space,
        "demo_trial_ids": [int(x) for x in demo_trial_ids],
        "recipe_hash": recipe_hash,
    }
    return sha256_json(payload)


def build_fit_matrix(protocol: EfficiencyProtocol) -> list[MatrixCell]:
    cells: list[MatrixCell] = []
    recipe_hash = protocol.recipe_hash
    for budget in protocol.budgets:
        for ladder in protocol.ladders:
            ladder_id = str(ladder["ladder_id"])
            ladder_sha = str(ladder["sha256"])
            demo_ids = protocol.demo_trial_ids_for(ladder_id, budget)
            demo_hash = demo_identity_hash(demo_ids)
            for seed in protocol.model_seeds:
                for action_space in protocol.action_spaces:
                    cell_id = (
                        f"b{budget:02d}_{ladder_id}_seed{seed}_{action_space}"
                    )
                    identity = cell_identity_hash(
                        protocol_sha256=protocol.sha256,
                        ladder_id=ladder_id,
                        ladder_sha256=ladder_sha,
                        budget=budget,
                        model_seed=seed,
                        action_space=action_space,
                        demo_trial_ids=demo_ids,
                        recipe_hash=recipe_hash,
                    )
                    cells.append(
                        MatrixCell(
                            cell_id=cell_id,
                            budget=int(budget),
                            ladder_id=ladder_id,
                            model_seed=int(seed),
                            action_space=str(action_space),
                            demo_trial_ids=demo_ids,
                            demo_identity_hash=demo_hash,
                            recipe_hash=recipe_hash,
                            protocol_sha256=protocol.sha256,
                            ladder_sha256=ladder_sha,
                            identity_hash=identity,
                        )
                    )
    cell_ids = [cell.cell_id for cell in cells]
    if len(cells) != 150:
        raise ValueError(f"matrix must contain 150 cells, got {len(cells)}")
    if len(set(cell_ids)) != len(cell_ids):
        raise ValueError("matrix cell_id values are not unique")
    identity_hashes = [cell.identity_hash for cell in cells]
    if len(set(identity_hashes)) != len(identity_hashes):
        raise ValueError("matrix identity_hash values are not unique")
    # Parity: for each (budget, ladder, seed), both action spaces share demo ids.
    by_key: dict[tuple, list[MatrixCell]] = {}
    for cell in cells:
        key = (cell.budget, cell.ladder_id, cell.model_seed)
        by_key.setdefault(key, []).append(cell)
    for key, group in by_key.items():
        if len(group) != 2:
            raise ValueError(f"expected 2 action spaces for matrix key {key}")
        left_ids = group[0].demo_trial_ids
        right_ids = group[1].demo_trial_ids
        if left_ids != right_ids:
            raise ValueError(
                f"demo parity failure for {key}: {left_ids} != {right_ids}"
            )
        if group[0].demo_identity_hash != group[1].demo_identity_hash:
            raise ValueError(f"demo identity hash parity failure for {key}")
    return cells


def validate_cell_artifact_for_resume(
    *,
    cell: MatrixCell,
    artifact: dict,
    execution_context: dict[str, Any] | None = None,
) -> None:
    """Accept exact-match artifacts; reject any stale/mismatched provenance."""
    required = {
        "cell_id": cell.cell_id,
        "identity_hash": cell.identity_hash,
        "protocol_sha256": cell.protocol_sha256,
        "ladder_id": cell.ladder_id,
        "ladder_sha256": cell.ladder_sha256,
        "budget": cell.budget,
        "model_seed": cell.model_seed,
        "action_space": cell.action_space,
        "demo_identity_hash": cell.demo_identity_hash,
        "recipe_hash": cell.recipe_hash,
        "demo_trial_ids": list(cell.demo_trial_ids),
    }
    for key, expected in required.items():
        if key not in artifact:
            raise ValueError(f"resume artifact missing field {key}")
        actual = artifact[key]
        if key == "demo_trial_ids":
            actual = [int(x) for x in actual]
            expected = [int(x) for x in expected]
        if actual != expected:
            raise ValueError(
                f"resume reject: mismatched {key}: artifact={actual!r} cell={expected!r}"
            )
    for key, expected in (execution_context or {}).items():
        if key not in artifact:
            raise ValueError(f"resume artifact missing execution field {key}")
        actual = artifact[key]
        if actual != expected:
            raise ValueError(
                f"resume reject: mismatched execution {key}: "
                f"artifact={actual!r} expected={expected!r}"
            )


def source_demo_paths_for_cell(
    *,
    cell: MatrixCell,
    demo_path_by_trial_id: dict[int, Path],
) -> list[Path]:
    paths: list[Path] = []
    for trial_id in cell.demo_trial_ids:
        if trial_id not in demo_path_by_trial_id:
            raise KeyError(f"missing demo path for trial_id={trial_id}")
        paths.append(Path(demo_path_by_trial_id[trial_id]))
    return paths
