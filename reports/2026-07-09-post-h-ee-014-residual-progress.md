# 2026-07-09 - Post-H-EE-014 Residual Progress

## Plain-English Summary

The post-H-EE-014 residual program ran under the frozen hybrid A1 baseline (EE 62/120).
**SP0** froze three residual stories with videos. **SP1 (H-EE-022)** and **SP2 (H-EE-023)**
were both **rejected** against pre-registered closed-loop bars. **SP3 (H-EE-024)** was
**diagnosed** as an impulse/path residual with **no train yet**. Best EE contract is still
hybrid A1 + `global_gripper` + historical match at **62/120**. Final holdout was not
opened. Phase 6b was not started.

Rejection with evidence is progress: match-set and arm-only MLP are crossed off.

## What To Review

- [ ] Scoreboard: `outputs/post_h_ee_014_residual_scoreboard.json`
- [ ] SP0 visual review: `outputs/h_ee_014_residual_visual_review.md` + three MP4 clips
- [ ] SP1 comparison: `outputs/h_ee_022_match_relative_ee_validation/h_ee_022_comparison.json`
- [ ] SP2 comparison: `outputs/h_ee_023_arm_only_mlp_validation/h_ee_023_comparison.json`
- [ ] SP3 diagnosis: `outputs/h_ee_024_impulse_diagnosis/h_ee_024_impulse_diagnosis.md`
- [ ] researchnotes + AGENTS residual program status updates

## Implementation Details

### Plumbing (named contracts)

- **Named match contracts** in `src/svla/state_bc.py`: `historical` (default) and
  `match_relative_ee` (gripper_open + object_minus_ee_* + contact/lift). NN retrieval
  now uses per-policy match indices (previously hardcoded historical).
- **CLI:** `--match-contract`, `--arm-only-mlp` on `scripts/train_state_bc.py`.
- **A2 arm-only:** `action_loss_weights(..., arm_only_mlp=True)` zeros gripper MSE weight
  while hybrid rollout still uses NN gripper.
- **SP1 re-eval script:** `scripts/run_h_ee_022_match_reeval.py` — one causal change
  (match contract) on frozen H-EE-014 hybrid weights.
- **Render:** `scripts/render_bc_rollout.py` supports `--eval-mode validation` (protocol-v2
  trial IDs 6001+).

### SP0 — Visual freeze

Clips under `outputs/h_ee_014_residual_clips/`:

| Class | Clip | trial / seed |
|-------|------|--------------|
| missing_lift thrash | `missing_lift_thrash_t6022_s1.mp4` | 6022 / 1 |
| impulse almost-win | `impulse_almost_win_t6014_s2.mp4` | 6014 / 2 |
| early_close vertical | `early_close_vertical_t6003_s3.mp4` | 6003 / 3 |

Re-count: missing_lift 29 (~30), impulse almost-win 15, early_close 11.

### SP1 — H-EE-022 rejected

Frozen hybrid models + `match_relative_ee` only:

| Metric | Baseline | H-EE-022 | Bar |
|--------|--------:|---------:|-----|
| early_close | 11 | **11** | ≤5 or −50% |
| success | 62 | 63 | ≥59 |
| reopen | 0 | 1 | ≤5 |
| worst seed | 9 | 8 | ≥8 |

**early_close bar failed.** Historical match retained.

### SP2 — H-EE-023 rejected

Full 5-seed retrain, A2 arm-only + historical match + hybrid NN gripper:

| Metric | Baseline hybrid A1 | A2 | Bar |
|--------|-------------------:|---:|-----|
| EE success | 62 | **67** | ≥72 or missing_lift ≤~21 |
| missing_lift_eo | ~29–30 | **32** | ≤−30% rel |
| worst seed | 9 | **6** | ≥11 |
| reopen | 0 | 0 | ≤5 |
| joint success | 97 | **89** | ≥87 |

Success +5 is noise vs seed variance; worst seed and missing_lift got worse. **Do not freeze A2.**

Note: A2 EE seed vector `[14,6,8,15,24]` includes a perfect seed 4 and a 6/24 seed —
instability increased.

### SP3 — H-EE-024 diagnosed

15 EO+lift+retain contact_dynamics fails:

- 13/15 over impulse thr (mean 11.44 vs thr 9; success mean 6.55)
- Gate breakdown: 10 impulse-only, 3 impulse+xy, 1 force-only, 1 xy-only
- Looks like prolonged contact force integral, not kN impact
- **Decision: `no_train_yet`** (no softer path design registered yet)

## Evidence And Verification

```bash
# SP0 renders (examples)
PYTHONPATH=src .venv/bin/python scripts/render_bc_rollout.py \
  --policy outputs/h_ee_014_nn_gripper_global_validation/models/ee_tool_delta_hybrid_nn_gripper_mlp_seed_1.json \
  --trial-id 6022 --eval-mode validation \
  --output outputs/h_ee_014_residual_clips/missing_lift_thrash_t6022_s1.mp4

# SP1 full re-eval
PYTHONPATH=src .venv/bin/python scripts/run_h_ee_022_match_reeval.py \
  --output-dir outputs/h_ee_022_match_relative_ee_validation --seeds 0 1 2 3 4

# SP2 full train
PYTHONPATH=src .venv/bin/python scripts/train_state_bc.py \
  --output-dir outputs/h_ee_023_arm_only_mlp_validation \
  --evaluation-protocol v2 --eval-split validation \
  --temporal-feature-mode legacy_progress_phase \
  --policy-type mlp --hybrid-nn-gripper --arm-only-mlp \
  --match-contract historical --loss-profile global_gripper \
  --seeds 0 1 2 3 4 --hidden-sizes 128 128 --epochs 300 \
  --batch-size 1024 --learning-rate 0.001 --weight-decay 1e-5 \
  --stride 1 --max-steps 3200 --action-gain 1.0 --label-source policy_labels

# Unit tests
PYTHONPATH=src .venv/bin/python -m pytest tests/test_state_bc.py tests/test_loss_profiles.py -q
```

- Result: 26 tests passed; SP1/SP2 full 5×24×2 validations completed; SP3 diagnosis written.
- What this proves: named match-set does not fix early-close; arm-only MLP does not fix
  missing_lift thrash under these bars; impulse almost-wins are force/path.
- What it does not prove: that no future match variant or arm architecture can work;
  max 1 retry per SP was observed (no infinite thrash). FSM (SP2b) not run.

## Demo Videos / Visual Artifacts

- Open missing-lift thrash: [`missing_lift_thrash_t6022_s1.mp4`](/Users/matthewwoodcock/Documents/Simplified%20VLA%20Structure/outputs/h_ee_014_residual_clips/missing_lift_thrash_t6022_s1.mp4)
- Open impulse almost-win: [`impulse_almost_win_t6014_s2.mp4`](/Users/matthewwoodcock/Documents/Simplified%20VLA%20Structure/outputs/h_ee_014_residual_clips/impulse_almost_win_t6014_s2.mp4)
- Open early-close vertical: [`early_close_vertical_t6003_s3.mp4`](/Users/matthewwoodcock/Documents/Simplified%20VLA%20Structure/outputs/h_ee_014_residual_clips/early_close_vertical_t6003_s3.mp4)

Videos are local/gitignored-style artifacts; regenerate with commands above.

## Decisions Made

- Decision: Reject H-EE-022; keep historical match default.
  Reason: early_close unchanged at 11.
- Decision: Reject H-EE-023 A2 as default contract.
  Reason: missing_lift and worst seed failed bars; slight aggregate +5 not enough.
- Decision: H-EE-024 diagnosis only; no train.
  Reason: mechanism clear; softer path not yet designed; do not relax gates.
- Decision: Best EE remains hybrid A1 62/120 (not A2 67).
  Reason: seed reliability and residual class targets matter more than +5 raw success.

## Risks And Limitations

- Risk: A2 seed 4 hit 24/24 while seed 1 hit 6/24 — high variance; do not cherry-pick A2.
- Risk: SP1 used weight re-eval not full retrain; plan preferred fit-NN-only, so this is
  the purer causal test. A full retrain with match_relative_ee is still possible as a
  single retry but was not taken after clean failure.
- Limitation: SP2b FSM not executed (optional after C0–C5).
- Limitation: Missing_lift count ~29 vs plan ~30 — recount difference of 1 only.

## Action Items

- [ ] Optional SP2b / H-EE-015 FSM gripper diagnostic (early_close still 11).
- [ ] SP4: H-EE-007 labels / H-EE-002 gain under hybrid if thrash remains primary.
- [ ] Consider H-EE-017 non-Markov arm history after SP1/SP2 reject.
- [ ] Keep final closed until named frontier + worst-seed/phys move.
- [ ] Do not start Phase 6b as residual workaround.

## Files Changed

- `src/svla/state_bc.py` — match contracts, NN match indices, arm-only loss, hybrid recipe A2
- `scripts/train_state_bc.py` — `--match-contract`, `--arm-only-mlp`
- `scripts/run_h_ee_022_match_reeval.py` — SP1 re-eval
- `scripts/render_bc_rollout.py` — validation eval-mode
- `tests/test_state_bc.py` — match + arm-only tests
- `researchnotes.md`, `AGENTS.md` — statuses and verdict
- `outputs/post_h_ee_014_residual_scoreboard.json` — scoreboard
- residual clips / diagnosis / comparison JSONs under `outputs/`

## Current Verdict

**Residual program complete with rejections.** Best EE: **62/120** hybrid A1. Best joint:
**97/120** hybrid A1. Meets legacy frontier 84: **false**. Final: **closed**. Phase 6b:
**blocked**. Next open SP: **SP2b / SP4** survivors only.
