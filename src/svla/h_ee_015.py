"""H-EE-015: oracle FSM gripper + frozen hybrid EE arm upper-bound diagnostic.

Inference-only. Replaces the hybrid NN gripper with a fixed two-state FSM that
uses privileged scripted grasp-target information. Never a fair learned-policy
result and never a substitute for the raw EE-vs-joint comparison.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from svla.state_bc import (
    HYBRID_POLICY_TYPE,
    HybridNNGripperMLPPolicy,
    load_policy,
)


HYPOTHESIS = "H-EE-015"
ACTION_SPACE = "ee_tool_delta"
SEEDS = (0, 1, 2, 3, 4)
TRIALS_PER_SEED = 24
TOTAL_TRIALS = len(SEEDS) * TRIALS_PER_SEED

GRIPPER_SOURCE = "oracle_fsm_h_ee_015"
DEFAULT_GRIPPER_SOURCE = "nn_gripper"

FSM_STATE_OPEN = "OPEN_APPROACH"
FSM_STATE_CLOSE = "CLOSE_LATCHED"

# Fixed before any efficacy evaluation. Do not tune after seeing validation.
FSM_POS_ERROR_MAX_M = 0.012
FSM_ROT_ERROR_MAX_RAD = 0.22
FSM_GRIPPER_OBJECT_DISTANCE_MAX_M = 0.015
FSM_OPEN_COMMAND = 1.0
FSM_CLOSE_COMMAND = 0.0

EXPECTED_BASELINE_PRIMARY = {
    "successes": 62,
    "event_order_valid": 79,
    "physical_sanity_pass": 68,
    "per_seed_successes": [20, 14, 9, 9, 10],
    "worst_seed": 9,
    "early_close": 11,
    "reopen_events": 0,
    "missing_lift_eo": 30,
    "controller_failures": 0,
}

# Pre-registered verdict bars (immutable after registration).
STRONG_POSITIVE_BARS = {
    "successes_min": 84,
    "event_order_valid_min": 90,
    "physical_sanity_pass_min": 80,
    "worst_seed_min": 12,
    "controller_failures_max": 0,
}

PARTIAL_BARS = {
    "successes_min": 72,
    "physical_sanity_pass_min": 68,
    "worst_seed_min": 10,
    "controller_failures_max": 0,
    # Material reduction: absolute drop of at least 5 in either residual bucket.
    "material_residual_drop_min": 5,
}

JOINT_HYBRID_REFERENCE = {
    "successes": 97,
    "total": 120,
    "source": "frozen H-EE-014 hybrid joint; not re-evaluated under H-EE-015",
}

PRIMARY_REPRODUCTION_FIELDS = tuple(EXPECTED_BASELINE_PRIMARY.keys())


class OracleGripperFSM:
    """Two-state latched gripper FSM for H-EE-015.

    OPEN_APPROACH commands fully open until all three inclusive thresholds hold
    on the same step, then CLOSE_LATCHED permanently commands closed.
    """

    def __init__(
        self,
        *,
        pos_error_max_m: float = FSM_POS_ERROR_MAX_M,
        rot_error_max_rad: float = FSM_ROT_ERROR_MAX_RAD,
        gripper_object_distance_max_m: float = FSM_GRIPPER_OBJECT_DISTANCE_MAX_M,
    ) -> None:
        self.pos_error_max_m = float(pos_error_max_m)
        self.rot_error_max_rad = float(rot_error_max_rad)
        self.gripper_object_distance_max_m = float(gripper_object_distance_max_m)
        self.state = FSM_STATE_OPEN
        self.step_count = 0
        self.transition_step: int | None = None
        self.transition_pos_error: float | None = None
        self.transition_rot_error: float | None = None
        self.transition_gripper_object_distance: float | None = None

    def transition_conditions_met(
        self,
        *,
        pos_error_m: float,
        rot_error_rad: float,
        gripper_object_distance_m: float,
    ) -> bool:
        return (
            float(pos_error_m) <= self.pos_error_max_m
            and float(rot_error_rad) <= self.rot_error_max_rad
            and float(gripper_object_distance_m) <= self.gripper_object_distance_max_m
        )

    def step(
        self,
        *,
        pos_error_m: float,
        rot_error_rad: float,
        gripper_object_distance_m: float,
    ) -> float:
        """Advance one simulator step and return the gripper command in [0, 1]."""

        self.step_count += 1
        if self.state == FSM_STATE_CLOSE:
            return FSM_CLOSE_COMMAND

        if self.transition_conditions_met(
            pos_error_m=pos_error_m,
            rot_error_rad=rot_error_rad,
            gripper_object_distance_m=gripper_object_distance_m,
        ):
            self.state = FSM_STATE_CLOSE
            self.transition_step = int(self.step_count)
            self.transition_pos_error = float(pos_error_m)
            self.transition_rot_error = float(rot_error_rad)
            self.transition_gripper_object_distance = float(gripper_object_distance_m)
            return FSM_CLOSE_COMMAND
        return FSM_OPEN_COMMAND

    @property
    def never_transitioned(self) -> bool:
        return self.transition_step is None

    def telemetry(self) -> dict[str, Any]:
        return {
            "fsm_state_final": self.state,
            "fsm_transition_step": self.transition_step,
            "fsm_transition_pos_error": self.transition_pos_error,
            "fsm_transition_rot_error": self.transition_rot_error,
            "fsm_transition_gripper_object_distance": (
                self.transition_gripper_object_distance
            ),
            "fsm_never_transitioned": self.never_transitioned,
            "fsm_step_count": int(self.step_count),
        }


class OracleFsmHybridPolicy:
    """Wrap frozen hybrid A1: MLP arm dims unchanged; gripper from oracle FSM.

    Call :meth:`set_oracle_signals` with current pose/distance errors before
    each :meth:`predict_with_index`. Default hybrid policies do not expose this
    method, so ordinary rollouts never activate the FSM.
    """

    def __init__(self, hybrid: HybridNNGripperMLPPolicy) -> None:
        if not isinstance(hybrid, HybridNNGripperMLPPolicy):
            raise TypeError(
                f"OracleFsmHybridPolicy requires HybridNNGripperMLPPolicy, got {type(hybrid)}"
            )
        self.hybrid = hybrid
        self.fsm = OracleGripperFSM()
        self._signals: tuple[float, float, float] | None = None
        self.action_space = hybrid.action_space
        self.gripper_dim = int(hybrid.gripper_dim)
        self.group_keys = list(hybrid.group_keys)
        self.gripper_source = GRIPPER_SOURCE
        self.oracle_diagnostic = True

    def set_oracle_signals(
        self,
        *,
        pos_error_m: float,
        rot_error_rad: float,
        gripper_object_distance_m: float,
    ) -> None:
        self._signals = (
            float(pos_error_m),
            float(rot_error_rad),
            float(gripper_object_distance_m),
        )

    @property
    def evaluation_config_hash(self) -> str:
        return str(self.hybrid.evaluation_config_hash)

    @evaluation_config_hash.setter
    def evaluation_config_hash(self, value: str) -> None:
        self.hybrid.evaluation_config_hash = str(value)

    @property
    def evaluation_protocol_version(self) -> int:
        return int(self.hybrid.evaluation_protocol_version)

    @evaluation_protocol_version.setter
    def evaluation_protocol_version(self, value: int) -> None:
        self.hybrid.evaluation_protocol_version = int(value)

    def predict(
        self, observation_features: np.ndarray, group_key: str
    ) -> tuple[np.ndarray, float]:
        action, distance, _ = self.predict_with_index(observation_features, group_key)
        return action, distance

    def predict_with_index(
        self,
        observation_features: np.ndarray,
        group_key: str,
        cursor: int | None = None,
        search_window: int | None = None,
    ) -> tuple[np.ndarray, float, int]:
        if self._signals is None:
            raise RuntimeError(
                "OracleFsmHybridPolicy requires set_oracle_signals before predict"
            )
        action, distance, index = self.hybrid.predict_with_index(
            observation_features,
            group_key,
            cursor=cursor,
            search_window=search_window,
        )
        action = np.asarray(action, dtype=float).copy()
        pos_err, rot_err, grip_dist = self._signals
        gripper_cmd = self.fsm.step(
            pos_error_m=pos_err,
            rot_error_rad=rot_err,
            gripper_object_distance_m=grip_dist,
        )
        action[self.gripper_dim] = float(gripper_cmd)
        # Consume signals so a missed update cannot silently reuse stale state.
        self._signals = None
        return action, float(distance), int(index)

    def arm_action_only(
        self,
        observation_features: np.ndarray,
        group_key: str,
        cursor: int | None = None,
        search_window: int | None = None,
    ) -> np.ndarray:
        """Return the five-dimensional arm slice from the frozen hybrid (no FSM)."""

        action, _, _ = self.hybrid.predict_with_index(
            observation_features,
            group_key,
            cursor=cursor,
            search_window=search_window,
        )
        action = np.asarray(action, dtype=float)
        grip = self.gripper_dim if self.gripper_dim >= 0 else action.shape[0] + self.gripper_dim
        mask = np.ones(action.shape[0], dtype=bool)
        mask[grip] = False
        return action[mask].copy()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_manifest_name(seed: int) -> str:
    return f"{ACTION_SPACE}_{HYBRID_POLICY_TYPE}_seed_{seed}.json"


def required_frozen_relative_paths() -> list[Path]:
    paths: list[Path] = []
    for seed in SEEDS:
        stem = f"{ACTION_SPACE}_{HYBRID_POLICY_TYPE}_seed_{seed}"
        paths.extend(
            [
                Path("models") / f"{stem}.json",
                Path("models") / f"{stem}_mlp_component.npz",
                Path("models") / f"{stem}_nn_component.npz",
            ]
        )
    paths.extend(
        [
            Path("eval") / f"{ACTION_SPACE}_policy_trials.jsonl",
            Path("eval") / f"{ACTION_SPACE}_policy_trials.summary.json",
            Path("h_ee_014_comparison.json"),
            Path("h_ee_014_diagnosis.json"),
            Path("state_bc_summary.json"),
            Path("state_bc_summary.manifest.json"),
        ]
    )
    return paths


EFFICACY_ARTIFACT_NAMES = (
    "h_ee_015_trials.jsonl",
    "h_ee_015_summary.json",
    "h_ee_015_paired_comparison.json",
    "h_ee_015_experiment_manifest.json",
)


def existing_efficacy_artifacts(output_dir: Path) -> list[str]:
    """Return efficacy artifact basenames present under ``output_dir``."""

    output_dir = Path(output_dir)
    return [
        name
        for name in EFFICACY_ARTIFACT_NAMES
        if (output_dir / name).is_file()
    ]


def assert_registration_mutable(output_dir: Path) -> None:
    """Refuse registration rewrite once any efficacy artifact exists.

    This is unconditional: ``--force`` must not bypass the preregistration
    immutability contract after trials/summary/paired/manifest exist.
    """

    found = existing_efficacy_artifacts(output_dir)
    if found:
        raise RuntimeError(
            "registration is immutable after efficacy evaluation begins; "
            f"found efficacy artifacts: {found}"
        )


def verify_frozen_inputs(
    baseline_dir: Path,
    *,
    protocol_hash: str,
    protocol_version: int,
    source_dir: Path | None = None,
    policy_loader: Callable[[Path], Any] = load_policy,
) -> dict[str, Any]:
    """Verify every required frozen H-EE-014 EE hybrid file and load policies.

    ``source_dir`` enables an *independent* copy comparison only when it resolves
    to a different path than ``baseline_dir``. Passing the same directory is treated
    as inventory-only hashing (not an independent source-copy verification).
    """

    baseline_dir = baseline_dir.resolve()
    independent_source = (
        source_dir is not None and Path(source_dir).resolve() != baseline_dir
    )
    inventory: list[dict[str, Any]] = []
    for relative in required_frozen_relative_paths():
        local = baseline_dir / relative
        if not local.is_file():
            raise FileNotFoundError(f"missing frozen input: {local}")
        digest = sha256_file(local)
        row: dict[str, Any] = {
            "path": str(relative),
            "sha256": digest,
            "size_bytes": local.stat().st_size,
        }
        if independent_source:
            source = Path(source_dir).resolve() / relative
            if not source.is_file():
                raise FileNotFoundError(f"missing primary source input: {source}")
            source_digest = sha256_file(source)
            row["primary_source_sha256"] = source_digest
            row["copy_matches_primary"] = source_digest == digest
            if source_digest != digest:
                raise ValueError(f"frozen copy hash mismatch: {relative}")
        inventory.append(row)

    loaded_models: list[dict[str, Any]] = []
    for seed in SEEDS:
        manifest_path = baseline_dir / "models" / model_manifest_name(seed)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format") != "svla_hybrid_nn_gripper_mlp_manifest_v1":
            raise ValueError(f"unexpected hybrid manifest format: {manifest_path}")
        if manifest.get("action_space") != ACTION_SPACE:
            raise ValueError(f"unexpected action space in {manifest_path}")
        if manifest.get("policy_type") != HYBRID_POLICY_TYPE:
            raise ValueError(f"unexpected policy type in {manifest_path}")
        if manifest.get("recipe") != "A1_compositor":
            raise ValueError(f"unexpected hybrid recipe in {manifest_path}")
        if manifest.get("mlp_temporal_feature_mode") != "legacy_progress_phase":
            raise ValueError(f"unexpected temporal mode in {manifest_path}")
        if (
            manifest.get("match_contract", "historical_object_contact")
            != "historical_object_contact"
        ):
            raise ValueError(f"unexpected match contract in {manifest_path}")
        if manifest.get("match_feature_indices") != [18, 19, 20, 28, 29, 30]:
            raise ValueError(f"unexpected historical match indices in {manifest_path}")
        if str(manifest.get("evaluation_config_hash")) != str(protocol_hash):
            raise ValueError(f"protocol hash mismatch in {manifest_path}")
        if int(manifest.get("evaluation_protocol_version", -1)) != int(protocol_version):
            raise ValueError(f"protocol version mismatch in {manifest_path}")
        for component_field in ("mlp_path", "nn_path"):
            component = manifest_path.parent / str(manifest[component_field])
            if not component.is_file():
                raise FileNotFoundError(f"missing {component_field}: {component}")

        policy = policy_loader(manifest_path)
        if not isinstance(policy, HybridNNGripperMLPPolicy):
            raise TypeError(f"expected frozen hybrid policy: {manifest_path}")
        if policy.action_space != ACTION_SPACE:
            raise ValueError(f"loaded policy action-space mismatch: {manifest_path}")
        if str(policy.evaluation_config_hash) != str(protocol_hash):
            raise ValueError(f"loaded policy protocol hash mismatch: {manifest_path}")
        if int(policy.evaluation_protocol_version) != int(protocol_version):
            raise ValueError(f"loaded policy protocol version mismatch: {manifest_path}")
        loaded_models.append(
            {
                "seed": seed,
                "manifest": str(Path("models") / manifest_path.name),
                "manifest_sha256": sha256_file(manifest_path),
                "mlp_component_sha256": sha256_file(
                    manifest_path.parent / str(manifest["mlp_path"])
                ),
                "nn_component_sha256": sha256_file(
                    manifest_path.parent / str(manifest["nn_path"])
                ),
            }
        )

    return {
        "baseline_dir": str(baseline_dir),
        "source_dir": (
            str(Path(source_dir).resolve()) if independent_source else None
        ),
        "source_comparison": (
            "independent_copy" if independent_source else "inventory_only"
        ),
        "all_copies_match_primary": bool(
            independent_source
            and all(row.get("copy_matches_primary") for row in inventory)
        ),
        "file_inventory": inventory,
        "loaded_models": loaded_models,
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
    _keyed_rows(rows, label=str(path))
    return rows


def _key(row: dict[str, Any]) -> tuple[int, int]:
    return int(row["seed"]), int(row["trial_id"])


def _keyed_rows(
    rows: Iterable[dict[str, Any]], *, label: str
) -> dict[tuple[int, int], dict[str, Any]]:
    keyed: dict[tuple[int, int], dict[str, Any]] = {}
    for row in rows:
        key = _key(row)
        if key in keyed:
            raise ValueError(f"duplicate (seed, trial_id) key in {label}: {key}")
        keyed[key] = row
    return keyed


def align_paired_rows(
    baseline_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]]
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    baseline = _keyed_rows(baseline_rows, label="baseline")
    candidate = _keyed_rows(candidate_rows, label="candidate")
    if set(baseline) != set(candidate):
        missing = sorted(set(baseline) - set(candidate))
        extra = sorted(set(candidate) - set(baseline))
        raise ValueError(f"paired key mismatch: missing={missing}, extra={extra}")
    return [(baseline[key], candidate[key]) for key in sorted(baseline)]


def assert_oracle_flags(row: dict[str, Any], *, label: str = "row") -> None:
    if row.get("gripper_source") != GRIPPER_SOURCE:
        raise ValueError(
            f"{label}: gripper_source must be {GRIPPER_SOURCE!r}, got {row.get('gripper_source')!r}"
        )
    if row.get("oracle_diagnostic") is not True:
        raise ValueError(
            f"{label}: oracle_diagnostic must be True, got {row.get('oracle_diagnostic')!r}"
        )


def is_missing_lift_eo(row: dict[str, Any]) -> bool:
    return bool(
        not row.get("event_order_valid")
        and not row.get("early_close")
        and int(row.get("reopen_events") or 0) == 0
        and row.get("contact_achieved")
        and not row.get("object_lifted")
    )


def is_impulse_almost_win(row: dict[str, Any]) -> bool:
    return bool(
        not row.get("success")
        and row.get("failure_category") == "contact_dynamics_failure"
        and row.get("event_order_valid")
        and row.get("object_lifted")
        and row.get("retained_during_hold")
    )


def _distribution(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def _constraint_block(rows: list[dict[str, Any]]) -> dict[str, Any]:
    steps = sum(int(row.get("steps") or 0) for row in rows)
    block: dict[str, Any] = {"trial_count": len(rows), "rollout_steps": steps}
    for name, field in (
        ("joint_limit", "joint_limit_clipped_steps"),
        ("infeasible", "infeasible_steps"),
        ("controller_failure", "controller_failure_steps"),
    ):
        counts = [int(row.get(field) or 0) for row in rows]
        total = sum(counts)
        block[name] = {
            **_distribution(counts),
            "total_steps": total,
            "exposure_rate": float(total / steps) if steps else 0.0,
        }
    return block


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    _keyed_rows(rows, label="summary rows")
    for index, row in enumerate(rows):
        assert_oracle_flags(row, label=f"summary row {index}")

    per_seed: dict[int, int] = Counter()
    for row in rows:
        per_seed[int(row["seed"])] += int(bool(row.get("success")))
    per_seed_vector = [int(per_seed[seed]) for seed in SEEDS]
    failures = [row for row in rows if not row.get("success")]
    successes = [row for row in rows if row.get("success")]
    missing = [row for row in rows if is_missing_lift_eo(row)]
    almost = [row for row in rows if is_impulse_almost_win(row)]
    never_transitioned = [row for row in rows if row.get("fsm_never_transitioned")]
    transitioned = [row for row in rows if not row.get("fsm_never_transitioned")]
    impulses = [
        float(row.get("gripper_contact_impulse_before_lift") or 0.0) for row in rows
    ]
    forces = [float(row.get("max_gripper_contact_force") or 0.0) for row in rows]
    displacements = [
        float(row.get("max_object_xy_displacement_while_supported") or 0.0)
        for row in rows
    ]
    rotations = [
        float(row.get("max_object_rotation_while_supported") or 0.0) for row in rows
    ]
    failure_categories = Counter(str(row.get("failure_category")) for row in rows)

    metrics = {
        "total": len(rows),
        "successes": sum(bool(row.get("success")) for row in rows),
        "event_order_valid": sum(bool(row.get("event_order_valid")) for row in rows),
        "physical_sanity_pass": sum(
            bool(row.get("physical_sanity_pass")) for row in rows
        ),
        "per_seed_successes": per_seed_vector,
        "worst_seed": min(per_seed_vector) if per_seed_vector else 0,
        "missing_lift_eo": len(missing),
        "early_close": sum(bool(row.get("early_close")) for row in rows),
        "reopen_events": sum(int(row.get("reopen_events") or 0) for row in rows),
        "contact_dynamics_failures": failure_categories.get(
            "contact_dynamics_failure", 0
        ),
        "impulse_almost_wins": len(almost),
        "never_transitioned": len(never_transitioned),
        "transitioned": len(transitioned),
        "fsm_transition_step": _distribution(
            float(row["fsm_transition_step"])
            for row in transitioned
            if row.get("fsm_transition_step") is not None
        ),
        "fsm_transition_pos_error": _distribution(
            float(row["fsm_transition_pos_error"])
            for row in transitioned
            if row.get("fsm_transition_pos_error") is not None
        ),
        "fsm_transition_rot_error": _distribution(
            float(row["fsm_transition_rot_error"])
            for row in transitioned
            if row.get("fsm_transition_rot_error") is not None
        ),
        "fsm_transition_gripper_object_distance": _distribution(
            float(row["fsm_transition_gripper_object_distance"])
            for row in transitioned
            if row.get("fsm_transition_gripper_object_distance") is not None
        ),
        "max_gripper_contact_force": _distribution(forces),
        "gripper_contact_impulse_before_lift": _distribution(impulses),
        "max_object_xy_displacement_while_supported": _distribution(displacements),
        "max_object_rotation_while_supported": _distribution(rotations),
        "controller_failures": sum(
            int(row.get("controller_failure_steps") or 0) for row in rows
        ),
        "failure_categories": dict(sorted(failure_categories.items())),
        "rollout_steps": {
            "total": sum(int(row.get("steps") or 0) for row in rows),
            **_distribution(int(row.get("steps") or 0) for row in rows),
        },
        "constraint_exposure": {
            "all": _constraint_block(rows),
            "success": _constraint_block(successes),
            "failure": _constraint_block(failures),
        },
        "gripper_source": GRIPPER_SOURCE,
        "oracle_diagnostic": True,
    }
    return metrics


def primary_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {field: metrics[field] for field in PRIMARY_REPRODUCTION_FIELDS}


def reproduction_check(metrics: dict[str, Any]) -> dict[str, Any]:
    """Compare recomputed baseline counts to the frozen H-EE-014 primary vector."""

    actual = {
        field: metrics[field]
        for field in PRIMARY_REPRODUCTION_FIELDS
        if field in metrics
    }
    differences = {
        field: {"expected": EXPECTED_BASELINE_PRIMARY[field], "actual": actual.get(field)}
        for field in PRIMARY_REPRODUCTION_FIELDS
        if actual.get(field) != EXPECTED_BASELINE_PRIMARY[field]
    }
    return {
        "exact_primary_counts": not differences,
        "expected": EXPECTED_BASELINE_PRIMARY,
        "actual": actual,
        "differences": differences,
    }


def summarize_baseline_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize raw H-EE-014 baseline rows (no oracle flags required)."""

    _keyed_rows(rows, label="baseline rows")
    per_seed: dict[int, int] = Counter()
    for row in rows:
        per_seed[int(row["seed"])] += int(bool(row.get("success")))
    per_seed_vector = [int(per_seed[seed]) for seed in SEEDS]
    missing = [row for row in rows if is_missing_lift_eo(row)]
    failure_categories = Counter(str(row.get("failure_category")) for row in rows)
    return {
        "total": len(rows),
        "successes": sum(bool(row.get("success")) for row in rows),
        "event_order_valid": sum(bool(row.get("event_order_valid")) for row in rows),
        "physical_sanity_pass": sum(
            bool(row.get("physical_sanity_pass")) for row in rows
        ),
        "per_seed_successes": per_seed_vector,
        "worst_seed": min(per_seed_vector) if per_seed_vector else 0,
        "missing_lift_eo": len(missing),
        "early_close": sum(bool(row.get("early_close")) for row in rows),
        "reopen_events": sum(int(row.get("reopen_events") or 0) for row in rows),
        "controller_failures": sum(
            int(row.get("controller_failure_steps") or 0) for row in rows
        ),
        "failure_categories": dict(sorted(failure_categories.items())),
        "paired_keys": [[int(s), int(t)] for s, t in sorted(_keyed_rows(rows, label="keys"))],
    }


def build_paired_comparison(
    baseline_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    for index, row in enumerate(candidate_rows):
        assert_oracle_flags(row, label=f"candidate row {index}")
    pairs = align_paired_rows(baseline_rows, candidate_rows)
    new_successes = [
        pair for pair in pairs if not pair[0].get("success") and pair[1].get("success")
    ]
    lost_successes = [
        pair for pair in pairs if pair[0].get("success") and not pair[1].get("success")
    ]
    new_from_missing = [pair for pair in new_successes if is_missing_lift_eo(pair[0])]
    changed_trials: list[dict[str, Any]] = []
    for baseline, candidate in pairs:
        if (
            bool(baseline.get("success")) != bool(candidate.get("success"))
            or bool(baseline.get("event_order_valid"))
            != bool(candidate.get("event_order_valid"))
            or str(baseline.get("failure_category"))
            != str(candidate.get("failure_category"))
        ):
            changed_trials.append(
                {
                    "seed": int(baseline["seed"]),
                    "trial_id": int(baseline["trial_id"]),
                    "baseline_success": bool(baseline.get("success")),
                    "candidate_success": bool(candidate.get("success")),
                    "baseline_event_order_valid": bool(
                        baseline.get("event_order_valid")
                    ),
                    "candidate_event_order_valid": bool(
                        candidate.get("event_order_valid")
                    ),
                    "baseline_failure_category": str(baseline.get("failure_category")),
                    "candidate_failure_category": str(
                        candidate.get("failure_category")
                    ),
                    "baseline_missing_lift_eo": is_missing_lift_eo(baseline),
                    "candidate_missing_lift_eo": is_missing_lift_eo(candidate),
                    "candidate_fsm_never_transitioned": bool(
                        candidate.get("fsm_never_transitioned")
                    ),
                    "candidate_fsm_transition_step": candidate.get(
                        "fsm_transition_step"
                    ),
                    "joint_limit_clipped_steps_delta": int(
                        candidate.get("joint_limit_clipped_steps") or 0
                    )
                    - int(baseline.get("joint_limit_clipped_steps") or 0),
                    "infeasible_steps_delta": int(
                        candidate.get("infeasible_steps") or 0
                    )
                    - int(baseline.get("infeasible_steps") or 0),
                }
            )

    recovery_keys = [
        {"seed": int(b["seed"]), "trial_id": int(b["trial_id"])}
        for b, _ in new_successes
    ]
    regression_keys = [
        {"seed": int(b["seed"]), "trial_id": int(b["trial_id"])}
        for b, _ in lost_successes
    ]

    return {
        "format": "svla_h_ee_015_paired_comparison_v1",
        "hypothesis": HYPOTHESIS,
        "gripper_source": GRIPPER_SOURCE,
        "oracle_diagnostic": True,
        "paired_trial_count": len(pairs),
        "new_successes": len(new_successes),
        "lost_successes": len(lost_successes),
        "net_success_change": len(new_successes) - len(lost_successes),
        "new_successes_from_baseline_missing_lift": len(new_from_missing),
        "recovery_keys": recovery_keys,
        "regression_keys": regression_keys,
        "changed_trials": changed_trials,
        "final_accessed": False,
    }


def residual_materially_improved(
    metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
    *,
    min_drop: int = PARTIAL_BARS["material_residual_drop_min"],
) -> dict[str, Any]:
    missing_drop = int(baseline_metrics["missing_lift_eo"]) - int(
        metrics["missing_lift_eo"]
    )
    early_drop = int(baseline_metrics["early_close"]) - int(metrics["early_close"])
    return {
        "missing_lift_eo_drop": missing_drop,
        "early_close_drop": early_drop,
        "material": bool(missing_drop >= min_drop or early_drop >= min_drop),
        "min_drop": int(min_drop),
    }


def classify_verdict(
    metrics: dict[str, Any],
    baseline_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply pre-registered H-EE-015 verdict bars.

    Returns one of:
    - strong_positive_arm_upper_bound
    - partial
    - negative_arm_ceiling
    """

    if baseline_metrics is None:
        baseline_metrics = EXPECTED_BASELINE_PRIMARY

    strong_bars = {
        "successes": metrics["successes"] >= STRONG_POSITIVE_BARS["successes_min"],
        "event_order": (
            metrics["event_order_valid"] >= STRONG_POSITIVE_BARS["event_order_valid_min"]
        ),
        "physical_sanity": (
            metrics["physical_sanity_pass"]
            >= STRONG_POSITIVE_BARS["physical_sanity_pass_min"]
        ),
        "worst_seed": metrics["worst_seed"] >= STRONG_POSITIVE_BARS["worst_seed_min"],
        "controller_failures": (
            metrics["controller_failures"]
            <= STRONG_POSITIVE_BARS["controller_failures_max"]
        ),
    }
    residual = residual_materially_improved(metrics, baseline_metrics)
    partial_bars = {
        "successes": metrics["successes"] >= PARTIAL_BARS["successes_min"],
        "material_residual_drop": residual["material"],
        "physical_sanity": (
            metrics["physical_sanity_pass"] >= PARTIAL_BARS["physical_sanity_pass_min"]
        ),
        "worst_seed": metrics["worst_seed"] >= PARTIAL_BARS["worst_seed_min"],
        "controller_failures": (
            metrics["controller_failures"] <= PARTIAL_BARS["controller_failures_max"]
        ),
    }

    if all(strong_bars.values()):
        status = "strong_positive_arm_upper_bound"
    elif all(partial_bars.values()):
        status = "partial"
    else:
        status = "negative_arm_ceiling"

    return {
        "status": status,
        "strong_positive_bars": strong_bars,
        "partial_bars": partial_bars,
        "residual_improvement": residual,
        "strong_positive_thresholds": STRONG_POSITIVE_BARS,
        "partial_thresholds": PARTIAL_BARS,
        "interpretation": {
            "strong_positive_arm_upper_bound": (
                "Privileged gripper sequencing revealed a substantially stronger "
                "frozen arm policy. Not learned-policy performance; next learned "
                "work must imitate the FSM information without oracle task state."
            ),
            "partial": (
                "Gripper timing contributes, but does not explain the full "
                "action-space gap. Arm/controller interaction remains material."
            ),
            "negative_arm_ceiling": (
                "The current learned arm trajectory / label / controller "
                "interaction is the nearer ceiling; stop treating gripper logic "
                "as the primary fix."
            ),
        }[status],
        "oracle_diagnostic": True,
        "not_learned_policy_performance": True,
    }


def assert_finalize_artifact_hashes(output_dir: Path) -> dict[str, str]:
    """Verify experiment-manifest SHA-256 fields match the on-disk artifact files.

    Finalize must write the final summary *before* hashing it into the
    experiment manifest. This gate catches that ordering bug.
    """

    output_dir = Path(output_dir)
    manifest_path = output_dir / "h_ee_015_experiment_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing experiment manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks = {
        "registration_sha256": output_dir / "h_ee_015_registration.json",
        "trials_sha256": output_dir / "h_ee_015_trials.jsonl",
        "summary_sha256": output_dir / "h_ee_015_summary.json",
        "paired_comparison_sha256": output_dir / "h_ee_015_paired_comparison.json",
    }
    actual: dict[str, str] = {}
    mismatches: dict[str, dict[str, str]] = {}
    for field, path in checks.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing finalize artifact: {path}")
        digest = sha256_file(path)
        actual[field] = digest
        recorded = str(manifest.get(field) or "")
        if recorded != digest:
            mismatches[field] = {"recorded": recorded, "actual": digest, "path": str(path)}
    if mismatches:
        raise ValueError(
            "finalize artifact hash mismatch (write final summary before hashing): "
            f"{mismatches}"
        )
    return actual


def build_registration(
    *,
    protocol_metadata: dict[str, Any],
    frozen_verification: dict[str, Any],
    baseline_metrics: dict[str, Any],
    paired_keys: Sequence[Sequence[int]],
    source_manifest_sha256: str,
    max_steps: int = 3200,
    search_window: int = 120,
    action_gain: float = 1.0,
) -> dict[str, Any]:
    if len(paired_keys) != TOTAL_TRIALS:
        raise ValueError(
            f"registration requires {TOTAL_TRIALS} paired keys, got {len(paired_keys)}"
        )
    keys = [[int(s), int(t)] for s, t in paired_keys]
    if len({(s, t) for s, t in keys}) != TOTAL_TRIALS:
        raise ValueError("registration paired keys contain duplicates")
    reproduction = reproduction_check(baseline_metrics)
    if not reproduction["exact_primary_counts"]:
        raise ValueError(
            f"baseline primary counts do not match frozen H-EE-014: {reproduction['differences']}"
        )

    return {
        "format": "svla_h_ee_015_registration_v1",
        "hypothesis": HYPOTHESIS,
        "status": "registered",
        "immutable_after_efficacy": True,
        "final_accessed": False,
        "training_performed": False,
        "joint_reevaluated": False,
        "oracle_diagnostic": True,
        "gripper_source": GRIPPER_SOURCE,
        "not_learned_policy_performance": True,
        "protocol": {
            "version": int(protocol_metadata["version"]),
            "config_sha256": str(protocol_metadata["config_sha256"]),
            "eval_split": "validation",
            "format": protocol_metadata.get("format"),
        },
        "frozen_baseline": {
            "source": "outputs/h_ee_014_nn_gripper_global_validation/",
            "action_space": ACTION_SPACE,
            "policy_type": HYBRID_POLICY_TYPE,
            "recipe": "A1_compositor",
            "loss_profile": "global_gripper",
            "temporal_feature_mode": "legacy_progress_phase",
            "match_contract": "historical_object_contact",
            "match_feature_indices": [18, 19, 20, 28, 29, 30],
            "action_gain": float(action_gain),
            "max_steps": int(max_steps),
            "search_window": int(search_window),
            "expected_primary": EXPECTED_BASELINE_PRIMARY,
            "baseline_metrics": baseline_metrics,
            "reproduction": reproduction,
            "source_manifest_sha256": source_manifest_sha256,
            "loaded_models": frozen_verification["loaded_models"],
            "file_inventory": frozen_verification["file_inventory"],
        },
        "fsm_contract": {
            "states": [FSM_STATE_OPEN, FSM_STATE_CLOSE],
            "open_command": FSM_OPEN_COMMAND,
            "close_command": FSM_CLOSE_COMMAND,
            "thresholds": {
                "pos_error_max_m": FSM_POS_ERROR_MAX_M,
                "rot_error_max_rad": FSM_ROT_ERROR_MAX_RAD,
                "gripper_object_distance_max_m": FSM_GRIPPER_OBJECT_DISTANCE_MAX_M,
                "inclusive": True,
            },
            "latch": True,
            "never_reopen": True,
            "privileged_grasp_target": True,
            "forbidden_inputs": [
                "contact_state",
                "future_state",
                "outcome",
                "success_gate",
                "phase_clock",
                "trial_specific_logic",
            ],
        },
        "verdict_thresholds": {
            "strong_positive_arm_upper_bound": STRONG_POSITIVE_BARS,
            "partial": PARTIAL_BARS,
            "otherwise": "negative_arm_ceiling",
        },
        "paired_validation_keys": keys,
        "paired_key_count": len(keys),
        "seeds": list(SEEDS),
        "trials_per_seed": TRIALS_PER_SEED,
        "total_trials": TOTAL_TRIALS,
        "joint_hybrid_reference": JOINT_HYBRID_REFERENCE,
        "scientific_change": (
            "Replace only the hybrid NN gripper output with the fixed oracle FSM; "
            "five EE arm dimensions remain byte-identical frozen MLP arm outputs."
        ),
    }
