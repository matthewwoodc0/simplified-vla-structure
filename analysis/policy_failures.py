#!/usr/bin/env python3
"""Analyze closed-loop policy failures without changing rollout behavior.

The input JSONL rows are the records written by ``scripts/train_state_bc.py``.
This tool intentionally reports gate overlap separately from ``failure_category``:
the latter is a priority classifier and therefore hides lower-priority failures.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


FORMAT_VERSION = "svla_policy_failure_analysis_v1"

STEP_FIELDS = (
    "clipped_translation_steps",
    "clipped_rotation_steps",
    "clipped_joint_steps",
    "joint_limit_clipped_steps",
    "joint_step_clipped_steps",
    "joint_accel_clipped_steps",
    "infeasible_steps",
    "controller_failure_steps",
)

REQUIRED_FIELDS = (
    "action_space",
    "seed",
    "trial_id",
    "steps",
    "success",
    "event_order_valid",
    "physical_sanity_pass",
    "collision_free_approach",
    "early_close",
    "reopen_events",
    "failure_category",
    "orientation",
    "approach",
    "object_pose",
    *STEP_FIELDS,
)

FAILURE_CLASSIFIER_PRIORITY = (
    "collision_free_approach",
    "event_order_valid",
    "physical_sanity_pass",
    "finite_pose_error",
    "reached_grasp",
    "contact_achieved",
    "object_lifted",
    "retained_during_hold",
)


def load_jsonl(paths: Sequence[Path]) -> list[dict[str, Any]]:
    """Load and validate policy-trial JSONL files."""

    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_number}: expected a JSON object")
                missing = [field for field in REQUIRED_FIELDS if field not in row]
                if missing:
                    raise ValueError(
                        f"{path}:{line_number}: missing required fields: {', '.join(missing)}"
                    )
                rows.append(row)
    if not rows:
        raise ValueError("No policy trial rows were loaded")
    return rows


def _rate(count: int | float, total: int | float) -> float:
    return float(count) / float(total) if total else 0.0


def _row_saturation_rate(row: dict[str, Any]) -> float:
    """Primary saturation measure: steps with any joint clipping / rollout steps.

    ``clipped_joint_steps`` is already the rollout's per-step aggregate joint-clipping
    flag. The translation, rotation, limit, step, and acceleration counters overlap,
    so summing them would double-count constrained steps.
    """

    return _rate(int(row["clipped_joint_steps"]), int(row["steps"]))


def _row_infeasible_rate(row: dict[str, Any]) -> float:
    return _rate(int(row["infeasible_steps"]), int(row["steps"]))


def _gate_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    successes = sum(bool(row["success"]) for row in rows)
    event_valid = sum(bool(row["event_order_valid"]) for row in rows)
    physical_valid = sum(bool(row["physical_sanity_pass"]) for row in rows)
    both_pass = sum(
        bool(row["event_order_valid"]) and bool(row["physical_sanity_pass"])
        for row in rows
    )
    both_fail = sum(
        not bool(row["event_order_valid"]) and not bool(row["physical_sanity_pass"])
        for row in rows
    )
    collision_free = sum(bool(row["collision_free_approach"]) for row in rows)
    return {
        "total": total,
        "successes": successes,
        "success_rate": _rate(successes, total),
        "event_order_valid_count": event_valid,
        "event_order_valid_rate": _rate(event_valid, total),
        "physical_sanity_pass_count": physical_valid,
        "physical_sanity_pass_rate": _rate(physical_valid, total),
        "event_and_physical_pass_count": both_pass,
        "event_and_physical_pass_rate": _rate(both_pass, total),
        "event_and_physical_fail_count": both_fail,
        "event_and_physical_fail_rate": _rate(both_fail, total),
        "collision_free_approach_count": collision_free,
        "collision_free_approach_rate": _rate(collision_free, total),
    }


def _bucket_summary(
    rows: Sequence[dict[str, Any]], field: str
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[field])].append(row)
    return {key: _gate_summary(groups[key]) for key in sorted(groups)}


def _constraint_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    total_steps = sum(int(row["steps"]) for row in rows)
    step_totals = {
        field: sum(int(row[field]) for row in rows) for field in STEP_FIELDS
    }
    step_rates = {
        field: _rate(total, total_steps) for field, total in step_totals.items()
    }
    trials_with_any = {
        field: sum(int(row[field]) > 0 for row in rows) for field in STEP_FIELDS
    }
    trials_with_any.update(
        {
            "any_saturation": sum(
                any(
                    int(row[field]) > 0
                    for field in (
                        "clipped_translation_steps",
                        "clipped_rotation_steps",
                        "clipped_joint_steps",
                    )
                )
                for row in rows
            ),
            "any_hard_limit_or_infeasible": sum(
                int(row["joint_limit_clipped_steps"]) > 0
                or int(row["infeasible_steps"]) > 0
                for row in rows
            ),
            "any_controller_failure": sum(
                int(row["controller_failure_steps"]) > 0 for row in rows
            ),
        }
    )
    return {
        "total_rollout_steps": total_steps,
        "step_totals": step_totals,
        "step_rates": step_rates,
        "trials_with_any": trials_with_any,
        "trial_rates_with_any": {
            key: _rate(count, len(rows)) for key, count in trials_with_any.items()
        },
    }


def _rank_quartiles(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Split rows into deterministic, near-equal rank quartiles.

    Equal saturation rates are ordered by seed and trial id. This keeps denominators
    balanced and reproducible; boundaries are included so tied splits are visible.
    """

    ranked = sorted(
        rows,
        key=lambda row: (
            _row_saturation_rate(row),
            int(row["seed"]),
            int(row["trial_id"]),
        ),
    )
    quartiles = []
    for index in range(4):
        start = math.floor(index * len(ranked) / 4)
        end = math.floor((index + 1) * len(ranked) / 4)
        bucket = ranked[start:end]
        gate = _gate_summary(bucket)
        rates = [_row_saturation_rate(row) for row in bucket]
        gate.update(
            {
                "quartile": index + 1,
                "min_saturation_rate": min(rates) if rates else None,
                "max_saturation_rate": max(rates) if rates else None,
                "mean_saturation_rate": statistics.fmean(rates) if rates else None,
            }
        )
        quartiles.append(gate)
    return quartiles


def _pearson(values: Sequence[float], indicators: Sequence[bool]) -> float | None:
    if len(values) < 2 or len(values) != len(indicators):
        return None
    x_mean = statistics.fmean(values)
    numeric_indicators = [float(value) for value in indicators]
    y_mean = statistics.fmean(numeric_indicators)
    x_variance = sum((value - x_mean) ** 2 for value in values)
    y_variance = sum((value - y_mean) ** 2 for value in numeric_indicators)
    if x_variance == 0.0 or y_variance == 0.0:
        return None
    covariance = sum(
        (x - x_mean) * (y - y_mean)
        for x, y in zip(values, numeric_indicators, strict=True)
    )
    return covariance / math.sqrt(x_variance * y_variance)


def _outcome_rate_comparison(
    rows: Sequence[dict[str, Any]], predicate: Any
) -> dict[str, Any]:
    positive = [row for row in rows if predicate(row)]
    negative = [row for row in rows if not predicate(row)]

    def rates(group: Sequence[dict[str, Any]]) -> dict[str, Any]:
        saturation = [_row_saturation_rate(row) for row in group]
        infeasible = [_row_infeasible_rate(row) for row in group]
        return {
            "count": len(group),
            "mean_saturation_rate": statistics.fmean(saturation) if saturation else None,
            "mean_infeasible_rate": statistics.fmean(infeasible) if infeasible else None,
        }

    return {"true": rates(positive), "false": rates(negative)}


def _associations(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    saturation = [_row_saturation_rate(row) for row in rows]
    infeasible = [_row_infeasible_rate(row) for row in rows]
    outcomes = {
        "event_order_failure": [not bool(row["event_order_valid"]) for row in rows],
        "physical_sanity_failure": [
            not bool(row["physical_sanity_pass"]) for row in rows
        ],
        "rollout_failure": [not bool(row["success"]) for row in rows],
        "early_close": [bool(row["early_close"]) for row in rows],
    }
    return {
        "point_biserial_correlations": {
            name: {
                "saturation_rate": _pearson(saturation, indicators),
                "infeasible_rate": _pearson(infeasible, indicators),
            }
            for name, indicators in outcomes.items()
        },
        "rates_by_outcome": {
            name: _outcome_rate_comparison(
                rows, lambda row, name=name: {
                    "event_order_failure": not bool(row["event_order_valid"]),
                    "physical_sanity_failure": not bool(row["physical_sanity_pass"]),
                    "rollout_failure": not bool(row["success"]),
                    "early_close": bool(row["early_close"]),
                }[name]
            )
            for name in outcomes
        },
        "caution": (
            "These are descriptive associations on reused evaluation rollouts. "
            "They do not establish that saturation caused a gate failure."
        ),
    }


def _seed_variability(seed_summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    metric_names = (
        "success_rate",
        "event_order_valid_rate",
        "physical_sanity_pass_rate",
        "event_and_physical_pass_rate",
    )
    result: dict[str, Any] = {}
    for metric in metric_names:
        values = [float(summary[metric]) for summary in seed_summaries.values()]
        result[metric] = {
            "min": min(values),
            "max": max(values),
            "mean": statistics.fmean(values),
            "population_stddev": statistics.pstdev(values),
        }
    return result


def _analyze_group(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    seed_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        seed_groups[str(row["seed"])].append(row)
    seed_summaries = {
        seed: {
            **_gate_summary(seed_groups[seed]),
            "constraints": _constraint_summary(seed_groups[seed]),
        }
        for seed in sorted(seed_groups, key=lambda value: int(value))
    }
    return {
        **_gate_summary(rows),
        "early_close_trials": sum(bool(row["early_close"]) for row in rows),
        "reopen_trials": sum(int(row["reopen_events"]) > 0 for row in rows),
        "reopen_events": sum(int(row["reopen_events"]) for row in rows),
        "failure_categories": dict(
            sorted(Counter(str(row["failure_category"]) for row in rows).items())
        ),
        "constraints": _constraint_summary(rows),
        "saturation_rate_quartiles": _rank_quartiles(rows),
        "by_orientation": _bucket_summary(rows, "orientation"),
        "by_approach": _bucket_summary(rows, "approach"),
        "by_object_pose": _bucket_summary(rows, "object_pose"),
        "by_seed": seed_summaries,
        "seed_variability": _seed_variability(seed_summaries),
        "associations": _associations(rows),
    }


def analyze_rows(
    rows: Sequence[dict[str, Any]], source_paths: Iterable[str] = ()
) -> dict[str, Any]:
    """Return analysis grouped by action space."""

    if not rows:
        raise ValueError("At least one policy trial row is required")
    action_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        missing = [field for field in REQUIRED_FIELDS if field not in row]
        if missing:
            raise ValueError(f"Row missing required fields: {', '.join(missing)}")
        if int(row["steps"]) < 0:
            raise ValueError("steps must be non-negative")
        action_groups[str(row["action_space"])].append(row)
    return {
        "format": FORMAT_VERSION,
        "source_paths": list(source_paths),
        "metric_definitions": {
            "saturation_rate": "clipped_joint_steps / total rollout steps",
            "infeasible_rate": "infeasible_steps / total rollout steps",
            "counter_overlap": (
                "Clipping/infeasibility counters may overlap on one rollout step; "
                "do not sum category rates as a union."
            ),
            "failure_category_priority": list(FAILURE_CLASSIFIER_PRIORITY),
            "failure_category_warning": (
                "failure_category is hierarchical; gate-overlap counts are the "
                "authoritative view of simultaneous event and physical failures."
            ),
            "quartiles": (
                "Near-equal rank groups sorted by saturation_rate, seed, and trial_id; "
                "ties can span adjacent quartiles."
            ),
        },
        "by_action_space": {
            action_space: _analyze_group(action_groups[action_space])
            for action_space in sorted(action_groups)
        },
    }


def render_markdown(analysis: dict[str, Any]) -> str:
    lines = [
        "# Policy failure analysis",
        "",
        "Failure categories are priority-classified. Gate overlaps below are reported "
        "independently, and clipping counters must not be summed because they can overlap.",
        "",
    ]
    for action_space, summary in analysis["by_action_space"].items():
        lines.extend(
            [
                f"## `{action_space}`",
                "",
                "| Metric | Count | Rate |",
                "|---|---:|---:|",
                f"| Success | {summary['successes']} / {summary['total']} | {summary['success_rate']:.3f} |",
                f"| Event order valid | {summary['event_order_valid_count']} / {summary['total']} | {summary['event_order_valid_rate']:.3f} |",
                f"| Physical sanity pass | {summary['physical_sanity_pass_count']} / {summary['total']} | {summary['physical_sanity_pass_rate']:.3f} |",
                f"| Both pass | {summary['event_and_physical_pass_count']} / {summary['total']} | {summary['event_and_physical_pass_rate']:.3f} |",
                f"| Both fail | {summary['event_and_physical_fail_count']} / {summary['total']} | {summary['event_and_physical_fail_rate']:.3f} |",
                f"| Collision-free approach | {summary['collision_free_approach_count']} / {summary['total']} | {summary['collision_free_approach_rate']:.3f} |",
                "",
                f"Early-close trials: **{summary['early_close_trials']}**. "
                f"Trials with reopen: **{summary['reopen_trials']}**; total reopen events: "
                f"**{summary['reopen_events']}**.",
                "",
                "### Controller constraints",
                "",
                f"Total rollout steps: **{summary['constraints']['total_rollout_steps']}**.",
                "",
                "| Counter | Steps | Per-step rate | Trials with any |",
                "|---|---:|---:|---:|",
            ]
        )
        for field in STEP_FIELDS:
            lines.append(
                f"| `{field}` | {summary['constraints']['step_totals'][field]} | "
                f"{summary['constraints']['step_rates'][field]:.4f} | "
                f"{summary['constraints']['trials_with_any'][field]} |"
            )
        lines.extend(
            [
                "",
                "### Saturation-rate quartiles",
                "",
                "| Quartile | Trials | Saturation range | Success | Event valid | Physical pass |",
                "|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for quartile in summary["saturation_rate_quartiles"]:
            minimum = quartile["min_saturation_rate"]
            maximum = quartile["max_saturation_rate"]
            range_text = "n/a" if minimum is None else f"{minimum:.4f}–{maximum:.4f}"
            lines.append(
                f"| {quartile['quartile']} | {quartile['total']} | {range_text} | "
                f"{quartile['success_rate']:.3f} | {quartile['event_order_valid_rate']:.3f} | "
                f"{quartile['physical_sanity_pass_rate']:.3f} |"
            )
        lines.extend(["", "### Per-seed results", ""])
        lines.extend(
            [
                "| Seed | Trials | Success | Event valid | Physical pass | Both pass |",
                "|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for seed, seed_summary in summary["by_seed"].items():
            lines.append(
                f"| {seed} | {seed_summary['total']} | {seed_summary['success_rate']:.3f} | "
                f"{seed_summary['event_order_valid_rate']:.3f} | "
                f"{seed_summary['physical_sanity_pass_rate']:.3f} | "
                f"{seed_summary['event_and_physical_pass_rate']:.3f} |"
            )
        lines.extend(["", "### Task buckets", ""])
        for heading, field in (
            ("Orientation", "by_orientation"),
            ("Approach", "by_approach"),
            ("Object pose", "by_object_pose"),
        ):
            lines.extend(
                [
                    f"#### {heading}",
                    "",
                    "| Bucket | Trials | Success | Event valid | Physical pass |",
                    "|---|---:|---:|---:|---:|",
                ]
            )
            for bucket, bucket_summary in summary[field].items():
                lines.append(
                    f"| `{bucket}` | {bucket_summary['total']} | "
                    f"{bucket_summary['success_rate']:.3f} | "
                    f"{bucket_summary['event_order_valid_rate']:.3f} | "
                    f"{bucket_summary['physical_sanity_pass_rate']:.3f} |"
                )
            lines.append("")
        lines.extend(
            [
                "### Priority-classified failures",
                "",
                "| Category | Trials |",
                "|---|---:|",
            ]
        )
        for category, count in summary["failure_categories"].items():
            lines.append(f"| `{category}` | {count} |")
        correlations = summary["associations"]["point_biserial_correlations"]
        lines.extend(
            [
                "",
                "### Descriptive associations",
                "",
                "Point-biserial correlations use a true failure/condition indicator. "
                "They are descriptive and do not establish causality.",
                "",
                "| Outcome indicator | Saturation rate | Infeasible rate |",
                "|---|---:|---:|",
            ]
        )
        for outcome, values in correlations.items():
            saturation = values["saturation_rate"]
            infeasible = values["infeasible_rate"]
            saturation_text = "n/a" if saturation is None else f"{saturation:.3f}"
            infeasible_text = "n/a" if infeasible is None else f"{infeasible:.3f}"
            lines.append(
                f"| `{outcome}` | {saturation_text} | {infeasible_text} |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Policy-trial JSONL paths")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.inputs)
    analysis = analyze_rows(rows, source_paths=(str(path) for path in args.inputs))
    rendered_json = json.dumps(analysis, indent=2, sort_keys=True) + "\n"
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered_json, encoding="utf-8")
    if args.output_markdown:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(render_markdown(analysis), encoding="utf-8")
    if not args.output_json and not args.output_markdown:
        print(rendered_json, end="")


if __name__ == "__main__":
    main()
