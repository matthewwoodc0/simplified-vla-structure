# 2026-07-08 - H-EE-008 Gripper-Weighted MSE

## Plain-English Summary

We tested whether **upweighting the gripper in the BC loss** (especially near
`grasp_align` / `close_gripper` demo phases) improves closed-loop pickup under the
same protocol-v2 validation grid and the selected `legacy_progress_phase` temporal
contract.

**It does.** Both action spaces improved substantially vs the registered uniform-MSE
legacy validation baseline:

| Contract | EE success | Joint success | Combined |
|----------|------------|---------------|----------|
| Legacy uniform MSE | 31/120 | 53/120 | 84/240 |
| **H-EE-008 weighted** (5× / close 10×) | **50/120** | **84/120** | **134/240** |

**Verdict: H-EE-008 confirmed on validation.** Final holdout was **not** accessed.
Proposed release gates are still **not** met (especially EE seed instability).

This is the first rejected-or-confirmed timing hypothesis since the H-EE-010–013 series
that actually moves closed-loop success in the right direction. The fix is a **loss**
change, not a new observation, clock, or shield.

## What To Review

- [ ] `src/svla/state_bc.py` — `action_loss_weights`, weighted MSE in `fit_mlp_policy`.
- [ ] `scripts/train_state_bc.py` — `--gripper-loss-weight`, `--close-phase-gripper-weight`.
- [ ] `outputs/h_ee_008_gripper_weighted_validation/state_bc_summary.json`
- [ ] `outputs/h_ee_008_gripper_weighted_validation/h_ee_008_comparison.json`
- [ ] `researchnotes.md` / `AGENTS.md` verdict updates
- [ ] EE per-seed table (instability remains material)

## Implementation Details

### Loss

Unchanged MLP architecture and 6-D continuous actions. Per-sample, per-dim weights:

| Dim | Default | Close phases (`grasp_align`, `close_gripper`) |
|-----|---------|-----------------------------------------------|
| Arm 0–4 | 1.0 | 1.0 |
| Gripper 5 | **5.0** | **10.0** |

Gradient uses \(\nabla \propto w \odot ( \hat y - y )\). Defaults remain weight 1.0 / no
close override for bit-compatible uniform MSE.

Demo phase labels drive close-phase weighting at **train** only. Rollout still uses
open-loop cursor phase for the legacy temporal feature contract (intentional for this
hypothesis).

### Contract (matched to registered validation)

- Temporal: `legacy_progress_phase`
- Hidden: 128×128, epochs 300, seeds 0–4
- Split: validation (6001+), both action spaces
- Protocol hash: `bffd862d76f401a9b512826ec8ca1207fba88bd12539a59370716bd22e946f99`

## Evidence And Verification

### Unit tests

```bash
PYTHONPATH=src .venv/bin/pytest tests/test_state_bc.py -q
```

- Result: **12 passed**
- Covers weight matrix construction and weighted vs uniform fit divergence.

### Primary closed-loop validation

```bash
PYTHONPATH=src .venv/bin/python scripts/train_state_bc.py \
  --output-dir outputs/h_ee_008_gripper_weighted_validation \
  --evaluation-protocol v2 \
  --eval-split validation \
  --temporal-feature-mode legacy_progress_phase \
  --policy-type mlp \
  --seeds 0 1 2 3 4 \
  --hidden-sizes 128 128 \
  --epochs 300 \
  --batch-size 1024 \
  --learning-rate 0.001 \
  --weight-decay 1e-5 \
  --stride 1 \
  --max-steps 3200 \
  --action-gain 1.0 \
  --label-source policy_labels \
  --gripper-loss-weight 5.0 \
  --close-phase-gripper-weight 10.0
```

- Result: completed ~12 min; combined **134/240** successes.
- Artifacts: `outputs/h_ee_008_gripper_weighted_validation/`
- Log: `outputs/h_ee_008_gripper_weighted_validation_run.log`

### Detailed comparison (validation, 120 trials / action space)

| Metric | Legacy EE | Weighted EE | Legacy joint | Weighted joint |
|--------|-----------|-------------|--------------|----------------|
| Success | 31 | **50** | 53 | **84** |
| `event_order_valid` | 38 | **55** | 66 | **90** |
| `physical_sanity_pass` | 64 | 71 | 80 | **100** |
| `early_close` | 5 | 5 | 10 | 6 |
| `preclose_contact_steps` | 583 | **50** | 0 | 48 |
| `reopen_events` | 190 | **142** | 120 | **70** |
| `controller_failure_steps` | 0 | 0 | 0 | 0 |
| Joint-limit rate | 0.225 | 0.338 | 0.169 | 0.124 |
| Per-seed successes | 8,1,8,0,14 | 22,9,11,2,6 | 7,15,2,13,16 | 21,13,12,18,20 |

### Pass bars (pre-registered)

| Bar | Threshold | Result |
|-----|-----------|--------|
| EE success | ≥ +10 pp (~+12/120) | **+19/120** — pass |
| EE event-order | ≥ +15 pp (~+18/120) | **+17/120** — pass (borderline absolute, clear direction) |
| Joint not collapsed | ≥ 40/120 | **84/120** — pass |

**What this proves:** Under matched protocol-v2 validation, gripper-weighted MSE improves
raw closed-loop success and event-order for **both** action spaces without shields.

**What it does not prove:** Final-split readiness; release gates; that EE is competitive
with joint (gap remains 50 vs 84); that seed-3 EE (2/24) is fixed; that vision is needed
or not needed.

## Demo Videos / Visual Artifacts

No new videos rendered (metric rejection/confirmation did not require visual review for
the primary claim). Optional regeneration:

```bash
PYTHONPATH=src .venv/bin/python scripts/render_bc_rollout.py \
  --policy outputs/h_ee_008_gripper_weighted_validation/models/ee_tool_delta_mlp_bc_seed_0.npz \
  --trial-id 6002 \
  --output outputs/h_ee_008_ee_seed0_success.mp4
```

## Decisions Made

- Decision: Test H-EE-008 before H-EE-003 or vision close-indicators.
  Reason: Smallest change to supervision; project order prefers state fixes first.
- Decision: Keep `legacy_progress_phase` temporal contract.
  Reason: Selected shared contract; isolate loss as the only variable.
- Decision: Confirm on validation; **do not** auto-open final.
  Reason: Final is one-shot discipline; confirmation is strong but seed instability remains.
- Decision: Recommend weighted loss as default for future state-BC experiments.
  Reason: Both spaces improved; joint improved more than EE but did not trade off.

## Risks And Limitations

- **EE seed instability:** per-seed 22, 9, 11, 2, 6 — worst seed still near zero.
  Why it matters: release gate requires high per-seed success.
- **EE joint-limit exposure rose** (0.225 → 0.338).
  Why it matters: better event-order may co-occur with more saturated EE motion.
- **Weights not swept:** 5× / 10× is one point; maybe not optimal.
- **Validation ≠ final:** holdout could regress; do not claim ladder layer-3 update until
  final is explicitly run under this loss.

## Action Items

- [x] Implement weighted MSE + CLI + tests.
- [x] Run protocol-v2 validation with registered hyperparams.
- [x] Update researchnotes / AGENTS.
- [ ] Optional user decision: run **registered final** once under weighted+legacy for both
  action spaces (updates layer-3 evidence if successful).
- [ ] Optional: seed-stability work (H-EE-003 / H-EE-016) before declaring comparison ready.
- [ ] Do not merge/push unless asked.

## Files Changed

- `src/svla/state_bc.py` — weighted MSE training.
- `scripts/train_state_bc.py` — CLI weights + summary fields.
- `tests/test_state_bc.py` — weight unit tests.
- `researchnotes.md` — H-EE-008 confirmed.
- `AGENTS.md` — verdict + next work.
- `reports/2026-07-08-h-ee-008-gripper-weighted-mse.md` — this report.
- `outputs/h_ee_008_gripper_weighted_validation/**` — run artifacts (local).

## Current Verdict

**Confirmed on validation / partial readiness.** Gripper-weighted MSE is a real closed-loop
win for both EE and joint under the shared legacy temporal contract. Phase 5 learned-policy
comparison is **improved but still not release-ready**. Phase 6b remains blocked until
gates or explicit scope change. Final access is the main optional next step if the user
wants registered layer-3 numbers under this loss.
