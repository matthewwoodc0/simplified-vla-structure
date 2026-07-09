# 2026-07-09 - H-EE-014 NN Gripper + MLP Arm

## Plain-English Summary

H-EE-014 asked whether **gripper timing should be a state-local nearest-neighbor lookup** while the **arm stays a learned MLP**. This is different from the rejected H-EE-003 (learned binary gripper head): here the gripper command is copied from nearby demo states, not re-learned as a classifier.

Under the same protocol-v2 **validation** contract and the H-EE-021-preferred **`global_gripper` (5×/5×)** loss, the **A1 compositor** (train MLP as usual, fit NN on the same demos, replace only gripper at rollout) **passed every pre-registered bar** against the pure-MLP global baseline:

| Metric (EE) | Pure MLP global | Hybrid NN+MLP | Δ |
|-------------|----------------:|--------------:|--:|
| Success | 49/120 | **62/120** | **+13** |
| Event order | 60/120 | **79/120** | **+19** |
| Reopen events | 155 | **0** | **−100%** |
| Worst seed | 4/24 | **9/24** | **+5** |
| Physical sanity | 68/120 | 68/120 | 0 |
| Early-close trials | 2 | 11 | +9 |

Joint did not collapse — it improved strongly (76→**97**/120, EO 84→107, worst 10→15).

**Main causal claim supported:** state-local NN gripper stops the reopen/oscillation residual that pure MLP could not fix with loss reweight alone. **Residual after 014** is **missing lift** and some **early-close**, not reopen.

Final holdout was **not** accessed. Research parity frontier (84/90/100, worst ≥12) is still not met for EE.

## What To Review

- [ ] `src/svla/state_bc.py` — `HybridNNGripperMLPPolicy` compositor, save/load manifest.
- [ ] `scripts/train_state_bc.py` — `--hybrid-nn-gripper` A1 wiring.
- [ ] `scripts/run_h_ee_014_diagnosis.py` — comparison vs H-EE-021 baseline + diagnosis.
- [ ] `outputs/h_ee_014_nn_gripper_global_validation/h_ee_014_comparison.json` — pass bars.
- [ ] `outputs/h_ee_014_nn_gripper_global_validation/h_ee_014_diagnosis.json` — EO anatomy.
- [ ] `outputs/h_ee_014_nn_gripper_global_validation/state_bc_summary.json` — raw aggregate.
- [ ] Research notes / AGENTS verdict updates.

## Implementation Details

### Hybrid policy (A1 compositor)

At each step:

```text
action[:5] = MLP arm deltas
action[5]  = NN gripper command
```

- MLP trained under `--loss-profile global_gripper` exactly as pure MLP (no arm-only loss; A2 not run).
- NN fit on the **same demos** with historical `MATCH_FEATURE_INDICES` (object x/y/z, lift, contact, support) — **not** silently changed to relative EE–object features.
- Cursor advancement follows **MLP open-loop** index so the hybrid does not introduce a second temporal contract; NN still uses the same `cursor` + `search_window` for candidate windowing.
- Save contract: hybrid JSON manifest + `_mlp_component.npz` + `_nn_component.npz`.
- `policy_type` recorded as `hybrid_nn_gripper_mlp`.

### Frozen experimental contract

| Knob | Value |
|------|-------|
| Protocol | v2 validation only |
| Temporal | `legacy_progress_phase` |
| MLP | 128×128, 300 epochs, batch 1024, lr 1e-3, wd 1e-5 |
| Seeds | 0–4 |
| Action spaces | both |
| Loss profile | `global_gripper` |
| Shield/FSM | off |
| Final | not accessed |
| NN | k=1, temperature=0.75 (CLI defaults) |

### Pre-registered pass bars (vs pure-MLP global)

| Bar | Required | Observed | Pass? |
|-----|----------|----------|:-----:|
| EE success | ≥ +10 | +13 | yes |
| EE event-order | ≥ +12 | +19 | yes |
| EE reopen | ≤ −20% relative | −100% | yes |
| EE worst seed | +3 abs or ≥8/24 | +5 and 9/24 | yes |
| Joint non-collapse | ≥ baseline −10 | +21 | yes |

## Evidence And Verification

### Unit tests

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_state_bc.py tests/test_loss_profiles.py -q
```

- Result: **23 passed**
- What this proves: hybrid composes MLP arm + NN gripper, save/load round-trips, rollout path works for both action spaces, MLP weights unchanged by compositor, match feature names align with indices.
- What it does not prove: closed-loop pickup success.

### Smoke

```bash
PYTHONPATH=src .venv/bin/python scripts/train_state_bc.py \
  --output-dir outputs/h_ee_014_nn_gripper_smoke \
  --evaluation-protocol v2 --eval-split validation \
  --temporal-feature-mode legacy_progress_phase \
  --policy-type mlp --hybrid-nn-gripper --loss-profile global_gripper \
  --seeds 0 --hidden-sizes 128 128 --epochs 2 --eval-limit 1 --max-steps 100 ...
```

- Result: completed; hybrid models + summary written.
- What this proves: end-to-end train/eval path under protocol-v2.
- What it does not prove: efficacy (undertrained, 1 trial).

### Full validation

```bash
PYTHONPATH=src .venv/bin/python scripts/train_state_bc.py \
  --output-dir outputs/h_ee_014_nn_gripper_global_validation \
  --evaluation-protocol v2 --eval-split validation \
  --temporal-feature-mode legacy_progress_phase \
  --policy-type mlp --hybrid-nn-gripper --loss-profile global_gripper \
  --seeds 0 1 2 3 4 --hidden-sizes 128 128 --epochs 300 \
  --batch-size 1024 --learning-rate 0.001 --weight-decay 1e-5 \
  --stride 1 --max-steps 3200 --action-gain 1.0 --label-source policy_labels
```

```bash
PYTHONPATH=src .venv/bin/python scripts/run_h_ee_014_diagnosis.py \
  --output-dir outputs/h_ee_014_nn_gripper_global_validation
```

- Result: **confirmed_validation**; all pass checks true; `selectable_for_final=true` by plan bars only.
- Output artifacts:
  - `outputs/h_ee_014_nn_gripper_global_validation/state_bc_summary.json`
  - `outputs/h_ee_014_nn_gripper_global_validation/h_ee_014_comparison.json`
  - `outputs/h_ee_014_nn_gripper_global_validation/h_ee_014_diagnosis.json`
  - `outputs/h_ee_014_nn_gripper_global_validation_run.log`
- What this proves: hybrid gripper source improves EE success/EO/worst-seed **and** eliminates reopens under the frozen validation contract; joint improves under the same compositor.
- What it does not prove: research-parity frontier readiness, final holdout performance, hardware transfer, or that MLP arm is optimal (A2 not tested).

### Detailed EE / joint tables

**EE per-seed successes:** hybrid `[20, 14, 9, 9, 10]` vs baseline global `[20, 12, 7, 4, 6]`.

**EE EO failure anatomy (hybrid):** `missing_lift=30`, `early_close=11`, **reopen_only=0**. Mean gripper flips = **1.0** on successes **and** failures (hold is stable; failures are not oscillation).

**Joint:** hybrid 97/120 (EO 107, phys 103, worst 15) vs baseline global 76/120 (EO 84, phys 92, worst 10). Note: this joint result **exceeds** the old H-EE-008 combined frontier joint (84/90/100) under a different loss profile; do not silently redefine the EE frontier without an explicit decision.

## Demo Videos / Visual Artifacts

- No new MP4s rendered for this run (gated metrics only).
- To render representative hybrid rollouts later:

```bash
# Use saved hybrid manifests under outputs/h_ee_014_nn_gripper_global_validation/models/
# with scripts/render_bc_rollout.py (adapt if script still expects pure MLP npz paths).
```

Generated models/jsonl are local outputs (often gitignored). Regenerate with the full validation command above.

## Decisions Made

- Decision: Primary contract is **`global_gripper` + A1 hybrid compositor**, not `combined_h_ee_008`.
  Reason: H-EE-021 showed global is the main EE reliability driver; plan recommended global as primary.
- Decision: Keep historical `MATCH_FEATURE_INDICES` unchanged.
  Reason: Comparability with existing NN baseline; silent match-set changes forbidden.
- Decision: Preserve MLP open-loop cursor advancement.
  Reason: Isolate gripper source from a second temporal-mode change.
- Decision: **Do not open final** despite plan selection bars being met.
  Reason: EE still short of research parity frontier (62 vs 84 success; worst 9 vs ≥12; phys 68 vs 100). Bars select a contract for *consideration*, not automatic holdout access. Human must approve final.
- Decision: Mark H-EE-014 **confirmed on validation**.
  Reason: All pre-registered pass bars met with closed-loop gates, not train loss.

## Risks And Limitations

- Risk: **Early-close rose** (2→11). Reopen fixed, but NN match features may still close too early in some states.
  Why it matters: next work should not claim “event-order solved”; EO failures shifted to early-close + missing lift.
- Risk: **Physical sanity flat** at 68/120 for EE.
  Why it matters: frontier needs phys ~100; hybrid did not fix contact dynamics failures.
- Risk: **Missing lift (30)** is now the largest EO bucket.
  Why it matters: arm trajectory / grasp geometry may be the new bottleneck once gripper hold is stable.
- Risk: Joint 97/120 under hybrid+global may tempt redefining the EE frontier using hybrid joint as target.
  Why it matters: frontier source was `combined_h_ee_008` joint; redefining requires explicit researchnotes decision.
- Risk: Match features lack relative EE–object geometry.
  Why it matters: early-close residual might need a **named** match-set ablation, not more MSE weight.
- Limitation: A2 arm-only MLP loss not run; pure compositor only.
- Limitation: No visual review clips for hybrid successes/failures yet.

## Action Items

- [ ] Freeze hybrid + `global_gripper` as the current best EE validation recipe for follow-ups.
- [ ] Inspect missing-lift failures (arm path / grasp distance / force) under hybrid — optional render clips.
- [ ] Optional named match-set ablation (relative EE–object + gripper_open) **only as a labeled secondary contract**.
- [ ] Consider H-EE-015 (FSM legal close window) or H-EE-017 (history) for residual early-close — not more loss reweight.
- [ ] Do **not** open final until worst-seed/phys/success approach frontier **or** human explicitly lowers the bar.
- [ ] Do not start Phase 6b vision BC as a substitute.

## Files Changed

- `src/svla/state_bc.py` — `HybridNNGripperMLPPolicy`, match feature name constants, load_policy JSON hybrid path.
- `scripts/train_state_bc.py` — `--hybrid-nn-gripper` A1 train/eval wiring and summary fields.
- `scripts/run_h_ee_014_diagnosis.py` — comparison + diagnosis writer vs H-EE-021 baseline.
- `tests/test_state_bc.py` — hybrid compose / save-load / rollout tests.
- `researchnotes.md` — H-EE-014 confirmed; priority order; results log; H-EE-005 residual update.
- `AGENTS.md` — verdict bullet + next-work guidance.
- `prompts/h-ee-014-nn-gripper-plan.md` — (pre-existing plan; executed as written).
- `reports/2026-07-09-h-ee-014-nn-gripper.md` — this report.

## Current Verdict

**Confirmed on validation (diagnostic+selection progress), not release-ready.**

H-EE-014 works as claimed for the reopen residual: EE success/EO/worst-seed all improve past pass bars, reopens go to zero, joint improves. The learned-policy comparison is **less blocked on gripper oscillation**, but EE is still **not** at the research parity frontier, physical sanity is unchanged, and early-close ticked up. Final holdout stays **closed**. Phase 6b remains **blocked** until the action-space comparison is closer to viable or scope is explicitly changed.

Honest one-liner for notes:

> H-EE-014 confirmed on validation: hybrid NN gripper + MLP arm under `global_gripper` improved EE from 49→62/120, event-order 60→79, reopen 155→0, worst seed 4→9. Final not accessed.
