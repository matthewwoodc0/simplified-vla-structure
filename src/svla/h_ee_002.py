"""Analysis and provenance helpers for the H-EE-002 frozen gain sweep.

This module contains no training path.  It validates frozen H-EE-014 hybrid
policies, summarizes validation rows, aligns paired trials, and applies the
pre-registered H-EE-002 decision bars.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from svla.state_bc import HYBRID_POLICY_TYPE, HybridNNGripperMLPPolicy, load_policy


HYPOTHESIS = "H-EE-002"
ACTION_SPACE = "ee_tool_delta"
SEEDS = (0, 1, 2, 3, 4)
GAINS = (1.0, 0.875, 0.75)
SELECTION_ORDER = (0.875, 0.75)
TRIALS_PER_SEED = 24
TOTAL_TRIALS = len(SEEDS) * TRIALS_PER_SEED

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

PASS_BARS = {
    "successes_min": 72,
    "missing_lift_eo_max": 21,
    "worst_seed_min": 11,
    "physical_sanity_pass_min": 68,
    "event_order_valid_min": 79,
    "reopen_events_max": 5,
    "early_close_max": 11,
    "controller_failures_max": 0,
    # "Material" is fixed before candidate evaluation as a >=10% relative
    # reduction in failure-conditioned exposure rate for either measure.
    "failure_constraint_relative_reduction_min": 0.10,
    # At least half of newly successful paired trials must originate in the
    # gain-1.0 missing-lift/thrash bucket.
    "new_success_from_missing_lift_fraction_min": 0.50,
    # Partial requires a clear >=10% missing-lift reduction (30 -> <=27),
    # controller support, and no safety/event-order regression.
    "partial_missing_lift_eo_max": 27,
}

PRIMARY_REPRODUCTION_FIELDS = tuple(EXPECTED_BASELINE_PRIMARY)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gain_slug(gain: float) -> str:
    if float(gain) not in GAINS:
        raise ValueError(f"gain must be one of {GAINS}, got {gain}")
    return f"gain_{float(gain):.3f}".replace(".", "_")


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


def verify_frozen_inputs(
    baseline_dir: Path,
    *,
    protocol_hash: str,
    protocol_version: int,
    source_dir: Path | None = None,
    policy_loader: Callable[[Path], Any] = load_policy,
) -> dict[str, Any]:
    """Verify every required frozen file and load all five EE policies."""

    baseline_dir = baseline_dir.resolve()
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
        if source_dir is not None:
            source = source_dir.resolve() / relative
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
        if manifest.get("match_contract", "historical_object_contact") != "historical_object_contact":
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
        "source_dir": None if source_dir is None else str(source_dir.resolve()),
        "all_copies_match_primary": bool(source_dir is not None),
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
    per_seed: dict[int, int] = Counter()
    for row in rows:
        per_seed[int(row["seed"])] += int(bool(row.get("success")))
    per_seed_vector = [int(per_seed[seed]) for seed in SEEDS]
    failures = [row for row in rows if not row.get("success")]
    successes = [row for row in rows if row.get("success")]
    missing = [row for row in rows if is_missing_lift_eo(row)]
    almost = [row for row in rows if is_impulse_almost_win(row)]
    impulses = [float(row.get("gripper_contact_impulse_before_lift") or 0.0) for row in rows]
    almost_impulses = [
        float(row.get("gripper_contact_impulse_before_lift") or 0.0) for row in almost
    ]
    failure_categories = Counter(str(row.get("failure_category")) for row in rows)
    metrics = {
        "total": len(rows),
        "successes": sum(bool(row.get("success")) for row in rows),
        "event_order_valid": sum(bool(row.get("event_order_valid")) for row in rows),
        "physical_sanity_pass": sum(bool(row.get("physical_sanity_pass")) for row in rows),
        "per_seed_successes": per_seed_vector,
        "worst_seed": min(per_seed_vector) if per_seed_vector else 0,
        "missing_lift_eo": len(missing),
        "early_close": sum(bool(row.get("early_close")) for row in rows),
        "reopen_events": sum(int(row.get("reopen_events") or 0) for row in rows),
        "contact_dynamics_failures": failure_categories.get("contact_dynamics_failure", 0),
        "impulse_almost_wins": len(almost),
        "impulse_almost_wins_over_9_ns": sum(value > 9.0 for value in almost_impulses),
        "impulse_all_trials": _distribution(impulses),
        "impulse_almost_win_trials": _distribution(almost_impulses),
        "controller_failures": sum(int(row.get("controller_failure_steps") or 0) for row in rows),
        "failure_categories": dict(sorted(failure_categories.items())),
        "rollout_steps": {
            "total": sum(int(row.get("steps") or 0) for row in rows),
            **_distribution(int(row.get("steps") or 0) for row in rows),
        },
        "constraint_exposure": {
            "all": _constraint_block(rows),
            "success": _constraint_block(successes),
            "failure": _constraint_block(failures),
            # Current-gain missing-lift rows only. Not the fixed gain-1.0 paired
            # cohort (see build_paired_comparison.baseline_missing_lift_trials).
            "current_gain_missing_lift": _constraint_block(missing),
        },
    }
    return metrics


def primary_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {field: metrics[field] for field in PRIMARY_REPRODUCTION_FIELDS}


def reproduction_check(metrics: dict[str, Any]) -> dict[str, Any]:
    actual = primary_metrics(metrics)
    differences = {
        field: {"expected": EXPECTED_BASELINE_PRIMARY[field], "actual": actual[field]}
        for field in PRIMARY_REPRODUCTION_FIELDS
        if actual[field] != EXPECTED_BASELINE_PRIMARY[field]
    }
    return {
        "exact_primary_counts": not differences,
        "expected": EXPECTED_BASELINE_PRIMARY,
        "actual": actual,
        "differences": differences,
    }


def _paired_constraint_stats(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]], field: str
) -> dict[str, Any]:
    baseline = np.asarray([int(left.get(field) or 0) for left, _ in pairs], dtype=float)
    candidate = np.asarray([int(right.get(field) or 0) for _, right in pairs], dtype=float)
    delta = candidate - baseline
    return {
        "trial_count": len(pairs),
        "baseline_mean": float(np.mean(baseline)) if len(baseline) else 0.0,
        "candidate_mean": float(np.mean(candidate)) if len(candidate) else 0.0,
        "mean_delta": float(np.mean(delta)) if len(delta) else 0.0,
        "candidate_lower_trials": int(np.sum(delta < 0)),
        "equal_trials": int(np.sum(delta == 0)),
        "candidate_higher_trials": int(np.sum(delta > 0)),
    }


def build_paired_comparison(
    baseline_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    pairs = align_paired_rows(baseline_rows, candidate_rows)
    baseline_missing_pairs = [pair for pair in pairs if is_missing_lift_eo(pair[0])]
    baseline_failure_pairs = [pair for pair in pairs if not pair[0].get("success")]
    candidate_failure_pairs = [pair for pair in pairs if not pair[1].get("success")]
    new_successes = [pair for pair in pairs if not pair[0].get("success") and pair[1].get("success")]
    lost_successes = [pair for pair in pairs if pair[0].get("success") and not pair[1].get("success")]
    new_from_missing = [pair for pair in new_successes if is_missing_lift_eo(pair[0])]
    changed_trials: list[dict[str, Any]] = []
    for baseline, candidate in pairs:
        if (
            bool(baseline.get("success")) != bool(candidate.get("success"))
            or bool(baseline.get("event_order_valid"))
            != bool(candidate.get("event_order_valid"))
            or str(baseline.get("failure_category")) != str(candidate.get("failure_category"))
        ):
            changed_trials.append(
                {
                    "seed": int(baseline["seed"]),
                    "trial_id": int(baseline["trial_id"]),
                    "baseline_success": bool(baseline.get("success")),
                    "candidate_success": bool(candidate.get("success")),
                    "baseline_event_order_valid": bool(baseline.get("event_order_valid")),
                    "candidate_event_order_valid": bool(candidate.get("event_order_valid")),
                    "baseline_failure_category": str(baseline.get("failure_category")),
                    "candidate_failure_category": str(candidate.get("failure_category")),
                    "baseline_missing_lift_eo": is_missing_lift_eo(baseline),
                    "candidate_missing_lift_eo": is_missing_lift_eo(candidate),
                    "joint_limit_clipped_steps_delta": int(
                        candidate.get("joint_limit_clipped_steps") or 0
                    )
                    - int(baseline.get("joint_limit_clipped_steps") or 0),
                    "infeasible_steps_delta": int(candidate.get("infeasible_steps") or 0)
                    - int(baseline.get("infeasible_steps") or 0),
                }
            )

    def constraints(cohort: list[tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, Any]:
        return {
            "joint_limit": _paired_constraint_stats(cohort, "joint_limit_clipped_steps"),
            "infeasible": _paired_constraint_stats(cohort, "infeasible_steps"),
        }

    baseline_missing_outcomes = Counter()
    for _, candidate in baseline_missing_pairs:
        if candidate.get("success"):
            label = "recovered_to_success"
        elif is_missing_lift_eo(candidate):
            label = "still_missing_lift_eo"
        elif candidate.get("event_order_valid"):
            label = f"event_order_recovered_{candidate.get('failure_category')}"
        else:
            label = f"changed_to_{candidate.get('failure_category')}"
        baseline_missing_outcomes[label] += 1

    return {
        "paired_trial_count": len(pairs),
        "new_successes": len(new_successes),
        "lost_successes": len(lost_successes),
        "net_success_change": len(new_successes) - len(lost_successes),
        "new_successes_from_baseline_missing_lift": len(new_from_missing),
        "new_success_from_missing_lift_fraction": (
            float(len(new_from_missing) / len(new_successes)) if new_successes else 0.0
        ),
        "baseline_missing_lift_trial_count": len(baseline_missing_pairs),
        "baseline_missing_lift_candidate_outcomes": dict(sorted(baseline_missing_outcomes.items())),
        "constraint_deltas": {
            "all_trials": constraints(pairs),
            "baseline_failure_trials": constraints(baseline_failure_pairs),
            "baseline_missing_lift_trials": constraints(baseline_missing_pairs),
            "candidate_failure_trials": constraints(candidate_failure_pairs),
        },
        "changed_trials": changed_trials,
    }


def _relative_reduction(baseline: float, candidate: float) -> float:
    if baseline <= 0:
        return 0.0
    return float((baseline - candidate) / baseline)


def classify_candidate(
    metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
    paired: dict[str, Any],
) -> dict[str, Any]:
    baseline_failure = baseline_metrics["constraint_exposure"]["failure"]
    candidate_failure = metrics["constraint_exposure"]["failure"]
    joint_limit_reduction = _relative_reduction(
        baseline_failure["joint_limit"]["exposure_rate"],
        candidate_failure["joint_limit"]["exposure_rate"],
    )
    infeasible_reduction = _relative_reduction(
        baseline_failure["infeasible"]["exposure_rate"],
        candidate_failure["infeasible"]["exposure_rate"],
    )
    controller_decline = max(joint_limit_reduction, infeasible_reduction) >= PASS_BARS[
        "failure_constraint_relative_reduction_min"
    ]
    concentration = (
        paired["new_success_from_missing_lift_fraction"]
        >= PASS_BARS["new_success_from_missing_lift_fraction_min"]
        and paired["new_successes"] > 0
    )
    bars = {
        "primary_efficacy": (
            metrics["successes"] >= PASS_BARS["successes_min"]
            or metrics["missing_lift_eo"] <= PASS_BARS["missing_lift_eo_max"]
        ),
        "worst_seed": metrics["worst_seed"] >= PASS_BARS["worst_seed_min"],
        "physical_sanity": (
            metrics["physical_sanity_pass"] >= PASS_BARS["physical_sanity_pass_min"]
        ),
        "event_order": metrics["event_order_valid"] >= PASS_BARS["event_order_valid_min"],
        "reopen": metrics["reopen_events"] <= PASS_BARS["reopen_events_max"],
        "early_close": metrics["early_close"] <= PASS_BARS["early_close_max"],
        "controller_failures": (
            metrics["controller_failures"] <= PASS_BARS["controller_failures_max"]
        ),
        "failure_constraint_exposure_decline": controller_decline,
        "paired_improvements_concentrated_in_missing_lift": concentration,
    }
    confirmed = all(bars.values())
    no_safety_or_event_regression = all(
        bars[name]
        for name in (
            "physical_sanity",
            "event_order",
            "reopen",
            "early_close",
            "controller_failures",
        )
    )
    partial = bool(
        not confirmed
        and metrics["missing_lift_eo"] <= PASS_BARS["partial_missing_lift_eo_max"]
        and controller_decline
        and no_safety_or_event_regression
    )
    status = "confirmed" if confirmed else "partial" if partial else "rejected"
    return {
        "status": status,
        "bars": bars,
        "all_confirmation_bars_met": confirmed,
        "failure_constraint_relative_reduction": {
            "joint_limit": joint_limit_reduction,
            "infeasible": infeasible_reduction,
        },
        "partial_rule_met": partial,
    }


def classify_sweep(candidates: dict[str, dict[str, Any]]) -> dict[str, Any]:
    for gain in SELECTION_ORDER:
        slug = gain_slug(gain)
        if candidates[slug]["classification"]["status"] == "confirmed":
            return {"status": "confirmed", "selected_gain": gain}
    if any(
        payload["classification"]["status"] == "partial"
        for payload in candidates.values()
    ):
        return {"status": "partial", "selected_gain": None}
    return {"status": "rejected", "selected_gain": None}
