"""Read-only efficiency-curve aggregation and paired bootstrap CIs.

This module never trains, rolls out, or mutates experiment artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class CurvePoint:
    budget: int
    success_rate: float
    n_trials: int


@dataclass(frozen=True)
class PairedUnit:
    """One preregistered paired unit: (ladder_id, model_seed)."""

    ladder_id: str
    model_seed: int
    joint_curve: tuple[CurvePoint, ...]
    ee_curve: tuple[CurvePoint, ...]


def normalized_auc(
    budgets: Sequence[int],
    success_rates: Sequence[float],
) -> float:
    """Trapezoidal AUC of success rate vs demo count, normalized by budget span.

    Boundary cases:
    - empty -> raises
    - single budget -> 0.0 (no span; not extrapolated)
    - constant rates -> rate itself (normalized rectangle)
    """
    if len(budgets) == 0:
        raise ValueError("budgets must be non-empty")
    if len(budgets) != len(success_rates):
        raise ValueError("budgets and success_rates length mismatch")
    x = np.asarray([int(b) for b in budgets], dtype=float)
    y = np.asarray([float(r) for r in success_rates], dtype=float)
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
        raise ValueError("non-finite budget or success_rate")
    if np.any((y < 0.0) | (y > 1.0)):
        raise ValueError("success_rates must be in [0, 1]")
    if len(x) == 1:
        return 0.0
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    if np.any(np.diff(x) <= 0):
        raise ValueError("budgets must be strictly increasing after sort")
    span = float(x[-1] - x[0])
    if span <= 0:
        return 0.0
    # NumPy 2.0 renamed trapz -> trapezoid; support both.
    trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    area = float(trapz(y, x))
    return area / span


def threshold_demo_count(
    budgets: Sequence[int],
    success_rates: Sequence[float],
    *,
    target: float,
) -> dict[str, Any]:
    """Smallest observed budget with success_rate >= target, else not_reached.

    Never extrapolates beyond observed budgets.
    """
    if not (0.0 <= float(target) <= 1.0):
        raise ValueError("target must be in [0, 1]")
    pairs = sorted(
        ((int(b), float(r)) for b, r in zip(budgets, success_rates)),
        key=lambda item: item[0],
    )
    for budget, rate in pairs:
        if rate + 1e-15 >= float(target):
            return {
                "status": "reached",
                "target": float(target),
                "demo_count": int(budget),
                "success_rate": float(rate),
            }
    return {
        "status": "not_reached",
        "target": float(target),
        "demo_count": None,
        "success_rate": None,
        "observed_budgets": [int(b) for b, _ in pairs],
    }


def assert_paired_key_alignment(units: Sequence[PairedUnit]) -> None:
    """Require identical budget grids and paired keys across joint/EE curves."""
    if not units:
        raise ValueError("no paired units")
    keys = [(u.ladder_id, int(u.model_seed)) for u in units]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate paired unit keys (ladder_id, model_seed)")
    reference_budgets: list[int] | None = None
    for unit in units:
        joint_budgets = [p.budget for p in unit.joint_curve]
        ee_budgets = [p.budget for p in unit.ee_curve]
        if joint_budgets != ee_budgets:
            raise ValueError(
                f"paired budget misalignment for {unit.ladder_id}/seed"
                f"{unit.model_seed}: joint={joint_budgets} ee={ee_budgets}"
            )
        if len(joint_budgets) != len(set(joint_budgets)):
            raise ValueError("duplicate budgets within a curve")
        if reference_budgets is None:
            reference_budgets = joint_budgets
        elif joint_budgets != reference_budgets:
            raise ValueError(
                "paired budget grids differ across ladder/model-seed units"
            )


def unit_auc_pair(unit: PairedUnit) -> tuple[float, float, float]:
    """Return (joint_auc, ee_auc, joint_minus_ee)."""
    joint_auc = normalized_auc(
        [p.budget for p in unit.joint_curve],
        [p.success_rate for p in unit.joint_curve],
    )
    ee_auc = normalized_auc(
        [p.budget for p in unit.ee_curve],
        [p.success_rate for p in unit.ee_curve],
    )
    return joint_auc, ee_auc, joint_auc - ee_auc


def paired_bootstrap_ci(
    units: Sequence[PairedUnit],
    *,
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    seed: int = 0,
    statistic: str = "joint_minus_ee_auc",
) -> dict[str, Any]:
    """Crossed-factor paired bootstrap over ladders and model seeds.

    Each observed cell remains a paired joint-minus-EE unit, but ladders and model
    seeds are resampled independently as crossed factors. Treating all 15
    ladder-by-seed cells as independent would be pseudoreplication because cells
    sharing a ladder or seed are correlated.
    """
    assert_paired_key_alignment(units)
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be in (0, 1)")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be >= 1")

    values = []
    value_by_key: dict[tuple[str, int], float] = {}
    for unit in units:
        joint_auc, ee_auc, delta = unit_auc_pair(unit)
        if statistic == "joint_minus_ee_auc":
            values.append(delta)
        elif statistic == "joint_auc":
            values.append(joint_auc)
        elif statistic == "ee_auc":
            values.append(ee_auc)
        else:
            raise ValueError(f"unknown statistic: {statistic}")
        value_by_key[(unit.ladder_id, int(unit.model_seed))] = values[-1]
    ladder_ids = sorted({unit.ladder_id for unit in units})
    model_seeds = sorted({int(unit.model_seed) for unit in units})
    expected_keys = {(ladder, seed) for ladder in ladder_ids for seed in model_seeds}
    actual_keys = set(value_by_key)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        raise ValueError(
            "crossed bootstrap requires the complete ladder x model-seed grid; "
            f"missing={missing}"
        )
    observed = float(np.mean(values))
    rng = np.random.default_rng(int(seed))
    boots = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        sampled_ladders = rng.choice(ladder_ids, size=len(ladder_ids), replace=True)
        sampled_seeds = rng.choice(model_seeds, size=len(model_seeds), replace=True)
        sample = [
            value_by_key[(str(ladder), int(model_seed))]
            for ladder in sampled_ladders
            for model_seed in sampled_seeds
        ]
        boots[i] = float(np.mean(sample))
    alpha = (1.0 - confidence_level) / 2.0
    low = float(np.quantile(boots, alpha))
    high = float(np.quantile(boots, 1.0 - alpha))
    return {
        "statistic": statistic,
        "observed_mean": observed,
        "confidence_level": float(confidence_level),
        "n_bootstrap": int(n_bootstrap),
        "n_paired_units": len(values),
        "paired_unit_definition": "(ladder_id, model_seed)",
        "resampling_scheme": "crossed_ladder_and_model_seed",
        "ladder_count": len(ladder_ids),
        "model_seed_count": len(model_seeds),
        "ci_low": low,
        "ci_high": high,
        "unit_values": [float(v) for v in values],
        "unit_keys": [
            {"ladder_id": u.ladder_id, "model_seed": int(u.model_seed)} for u in units
        ],
        "notes": [
            "Bootstrap resamples ladders and model seeds independently as crossed factors.",
            "Cells sharing a ladder or seed are not treated as independent replicates.",
        ],
    }


def success_differences_by_budget(units: Sequence[PairedUnit]) -> dict[str, Any]:
    """Mean joint-minus-EE success difference at each shared budget."""
    assert_paired_key_alignment(units)
    budgets = [p.budget for p in units[0].joint_curve]
    out: dict[str, Any] = {"budgets": budgets, "mean_joint_minus_ee": [], "per_unit": []}
    for budget_index, budget in enumerate(budgets):
        deltas = []
        for unit in units:
            j = unit.joint_curve[budget_index].success_rate
            e = unit.ee_curve[budget_index].success_rate
            deltas.append(float(j - e))
        out["mean_joint_minus_ee"].append(float(np.mean(deltas)))
    for unit in units:
        out["per_unit"].append(
            {
                "ladder_id": unit.ladder_id,
                "model_seed": int(unit.model_seed),
                "joint_minus_ee": [
                    float(j.success_rate - e.success_rate)
                    for j, e in zip(unit.joint_curve, unit.ee_curve)
                ],
            }
        )
    return out


def aggregate_cells_to_paired_units(
    cell_summaries: Sequence[Mapping[str, Any]],
    *,
    budgets: Sequence[int],
) -> list[PairedUnit]:
    """Build paired units from completed cell summaries.

    Each summary must include: ladder_id, model_seed, action_space, budget,
    success_rate, n_trials.
    """
    required = {
        "ladder_id",
        "model_seed",
        "action_space",
        "budget",
        "success_rate",
        "n_trials",
    }
    indexed: dict[tuple[str, int, str, int], Mapping[str, Any]] = {}
    for row in cell_summaries:
        missing = required - set(row)
        if missing:
            raise ValueError(f"cell summary missing fields: {sorted(missing)}")
        key = (
            str(row["ladder_id"]),
            int(row["model_seed"]),
            str(row["action_space"]),
            int(row["budget"]),
        )
        if key in indexed:
            raise ValueError(f"duplicate cell summary key: {key}")
        indexed[key] = row

    ladders_seeds = sorted(
        {(str(r["ladder_id"]), int(r["model_seed"])) for r in cell_summaries}
    )
    units: list[PairedUnit] = []
    budget_list = [int(b) for b in budgets]
    for ladder_id, seed in ladders_seeds:
        joint_points: list[CurvePoint] = []
        ee_points: list[CurvePoint] = []
        for budget in budget_list:
            j = indexed.get((ladder_id, seed, "joint_delta", budget))
            e = indexed.get((ladder_id, seed, "ee_tool_delta", budget))
            if j is None or e is None:
                raise ValueError(
                    f"incomplete curve for ladder={ladder_id} seed={seed} budget={budget}"
                )
            joint_points.append(
                CurvePoint(
                    budget=budget,
                    success_rate=float(j["success_rate"]),
                    n_trials=int(j["n_trials"]),
                )
            )
            ee_points.append(
                CurvePoint(
                    budget=budget,
                    success_rate=float(e["success_rate"]),
                    n_trials=int(e["n_trials"]),
                )
            )
        units.append(
            PairedUnit(
                ladder_id=ladder_id,
                model_seed=seed,
                joint_curve=tuple(joint_points),
                ee_curve=tuple(ee_points),
            )
        )
    assert_paired_key_alignment(units)
    return units
