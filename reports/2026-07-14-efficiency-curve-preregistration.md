# 2026-07-14 - Efficiency Curve Preregistration (EFF-001)

## Plain-English Summary

This change builds **ready-to-run infrastructure** for the first post-synthesis comparative
study: a demonstration-efficiency curve comparing `joint_delta` vs `ee_tool_delta` under the
frozen hybrid A1 fair contract. Exact nested demo ladders, a 150-fit matrix, a new
evaluation protocol with development and locked splits, dry-run mode, resume hashing,
read-only AUC/CI analysis, and contract tests are in place.

**The scientific primary curve was not run. Locked evaluation was not accessed.** This stop
is intentional so Codex (or another reviewer) can audit the protocol before any 300-epoch
matrix execution.

## What To Review

- [ ] `configs/state_bc_efficiency_protocol_v1.json` — budgets, three ladders, frozen recipe, development + locked splits, endpoints.
- [ ] `evidence/state_bc_efficiency_curve_registration.json` — tracked registration with hashes and `primary_curve_executed: false`.
- [ ] `evidence/state_bc_efficiency_curve_matrix_dry_run.json` — 150 unique cell IDs and identity hashes.
- [ ] `scripts/run_state_bc_efficiency_curve.py` — dry-run / smoke / primary / locked modes; locked requires `--allow-locked-evaluation`.
- [ ] `analysis/efficiency_curve.py` — read-only AUC and paired bootstrap (ladder × seed units).
- [ ] `tests/test_efficiency_protocol.py` — contract coverage.
- [ ] Smoke only under `outputs/state_bc_efficiency_curve_smoke/` with `non_efficacy_smoke=true` (not scientific).

## Implementation Details

### Frozen scientific design

| Item | Value |
|------|--------|
| Action spaces | `joint_delta`, `ee_tool_delta` |
| Policy | hybrid NN gripper + MLP arm, compositor **A1** |
| Loss / match / temporal | `global_gripper` / `historical` / `legacy_progress_phase` |
| Labels / gain | `policy_labels` / 1.0 |
| MLP | 128×128, 300 epochs, batch 1024, lr 1e-3, wd 1e-5 |
| Model seeds | 0, 1, 2, 3, 4 |
| Budgets | 6, 12, 18, 24, 30 distinct successful demos |
| Strata | 6 = 3 orientations × 2 approaches (balanced at every budget) |
| Ladders | L0, L1, L2 (immutable nested; construction seeds 101/202/303) |
| Planned cells | **150** = 5 × 3 × 5 × 2 |

Demo pool trial IDs **8001–8030** use the same nominal XY envelope as protocol-v2 train
positions but are **not** aliased to the v2 train split identity. Evaluation:

| Split | Trial IDs | Count | Role |
|-------|-----------|-------|------|
| `development` | 9001+ | 24 | Primary-curve eval split (after review) |
| `locked_evaluation` | 10001+ | 24 | Locked; requires `--allow-locked-evaluation` |

Positions are pairwise disjoint from the demo pool and from each other, and are not the
protocol-v2 validation or final grids.

### Code layout

- `src/svla/efficiency/protocol.py` — load/validate protocol; build matrix; resume identity checks; reject non-nested/unbalanced/duplicate/recipe-drift/overlap.
- `scripts/run_state_bc_efficiency_curve.py` — dedicated runner (does **not** change default `train_state_bc.py` behavior).
- `analysis/efficiency_curve.py` — normalized AUC, `not_reached` thresholds, paired bootstrap over `(ladder_id, model_seed)`.
- `experiments/configs/state_bc_efficiency_curve_registered.json` — dry-run-first experiment config; `allow_locked_evaluation: false`.

### Endpoints (preregistered)

**Primary:** normalized area under success-rate vs distinct-demo-count curve; paired
joint−EE AUC difference on identical (ladder, seed, eval specs).

**Secondary:** per-budget success difference, event-order, physical-sanity, worst seed,
hard-limit/infeasible exposure, early close/reopen, supervised timestep count, train/rollout
wall time, model bytes, peak RSS when available.

**Uncertainty:** paired bootstrap over 15 preregistered units `(ladder, seed)`. Five model
seeds alone do **not** measure demonstration-selection variance. No extrapolation past
observed budgets; unmet targets report `not_reached`.

## Evidence And Verification

```bash
# Contract tests
.venv/bin/pytest tests/test_efficiency_protocol.py tests/test_experiment_config.py -q

# Dry-run 150-cell matrix
PYTHONPATH=src .venv/bin/python scripts/run_state_bc_efficiency_curve.py --mode dry-run \
  --output-dir outputs/state_bc_efficiency_curve

# Plumbing-only smoke (not efficacy)
PYTHONPATH=src .venv/bin/python scripts/run_state_bc_efficiency_curve.py --mode smoke \
  --output-dir outputs/state_bc_efficiency_curve_smoke \
  --budgets 6 --ladders L0 --seeds 0 --epochs 2 --eval-limit 1
```

- Result: dry-run wrote 150 unique cells; smoke completed 2 cells (joint + EE) with
  `non_efficacy_smoke=true`, development split, eval_limit=1, epochs=2.
- Output artifacts:
  - `outputs/state_bc_efficiency_curve/efficiency_matrix_dry_run.json` (gitignored full matrix)
  - `evidence/state_bc_efficiency_curve_matrix_dry_run.json` (tracked compact matrix)
  - `outputs/state_bc_efficiency_curve_smoke/efficiency_smoke_summary.json`
- What this proves: protocol validation, matrix identity, locked-access guard, demo
  generation, hybrid train/rollout plumbing, resume identity fields.
- What it does **not** prove: sample efficiency of either action space; any ranking of
  joint vs EE; any demo-count threshold.

Locked evaluation without the opt-in flag fails:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_state_bc_efficiency_curve.py --mode locked-evaluation
# → requires explicit --allow-locked-evaluation
```

## Demo Videos / Visual Artifacts

None for this goal. Smoke is plumbing-only and not for visual efficacy review.

## Decisions Made

- Decision: New efficiency protocol v1 rather than mutating protocol-v2.
  Reason: Keep v2 validation/final frozen; never alias locked splits.
- Decision: Explicit nested ladders with stored trial IDs + SHA-256, not seed-only construction.
  Reason: Construction seed is convenience; explicit lists are the scientific contract.
- Decision: Same ladder and exact demo trial IDs for both action spaces per matrix cell key.
  Reason: Comparison parity; both spaces read labels from the same scripted demos.
- Decision: Goal 02 never supplies `--allow-locked-evaluation`.
  Reason: Review stop before locked access.
- Decision: Smoke labeled `non_efficacy_smoke=true` everywhere.
  Reason: Prevent selection/tuning from under-trained outcomes.

## Risks And Limitations

- Risk: Primary curve is expensive (150 full 300-epoch fits).
  Why it matters: Review should confirm matrix and endpoints before spend.
- Risk: Demo pool uses five nominal poses (30 distinct demos); max budget is the full pool.
  Why it matters: Efficiency curve cannot exceed 30 distinct demos without a new registered pool.
- Risk: Smoke success rates are meaningless (2 epochs, 1 eval trial).
  Why it matters: Must not be cited as efficiency evidence.
- Limitation: Peak memory is best-effort `resource.ru_maxrss`, not a full profiler.
  Why it matters: Secondary endpoint only when available without heavyweight deps.

## Action Items

- [ ] Independent review of protocol, ladders, splits, and runner (Codex).
- [ ] After approval only: run `--mode primary-curve` on development split.
- [ ] After primary analysis and separate authorization only: locked evaluation with
      `--allow-locked-evaluation`.
- [ ] Do not open protocol-v2 final from this study.
- [ ] Do not tune recipe/ladders from smoke.

## Files Changed

- `configs/state_bc_efficiency_protocol_v1.json` — versioned efficiency protocol.
- `src/svla/efficiency/protocol.py` — loader/validator/matrix/resume helpers.
- `src/svla/efficiency/__init__.py` — package exports.
- `scripts/run_state_bc_efficiency_curve.py` — dedicated runner.
- `analysis/efficiency_curve.py` — read-only AUC/CI aggregation.
- `experiments/configs/state_bc_efficiency_curve_registered.json` — dry-run-first config.
- `src/svla/experiments/config.py` — allow efficiency entrypoint; locked-access guard.
- `src/svla/eval/manifest.py` — track efficiency sources in manifests.
- `tests/test_efficiency_protocol.py` — contract tests.
- `evidence/state_bc_efficiency_curve_registration.json` — tracked registration.
- `evidence/state_bc_efficiency_curve_matrix_dry_run.json` — compact 150-cell dry-run matrix.
- `evidence/README.md`, `researchnotes.md`, `AGENTS.md` — operator/research docs.
- `pyproject.toml` — pytest pythonpath includes repo root for `analysis/`.
- `reports/2026-07-14-efficiency-curve-preregistration.md` — this report.

## Current Verdict

**READY_FOR_EFFICIENCY_REVIEW**

Infrastructure is ready to run. The primary efficiency curve and locked evaluation remain
**unexecuted** and require independent review before any scientific matrix launch.
`primary_curve_executed: false`, `locked_evaluation_accessed: false`.
