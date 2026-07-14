# 2026-07-14 - H-EE-015 Oracle FSM Arm Upper Bound

## Plain-English Summary

H-EE-015 asked whether the frozen H-EE-014 hybrid EE arm is substantially better than its
62/120 aggregate suggests once gripper sequencing is supplied by a privileged, task-aware
FSM. The answer is **no**.

Replacing only the NN gripper with a fixed two-state latched FSM
(`OPEN_APPROACH` → `CLOSE_LATCHED` on inclusive pose/distance thresholds) **reduced** EE
success from **62/120 to 47/120**. Event order was roughly flat (79→77). Physical sanity
fell (68→56). Worst seed fell (9→5). Missing-lift event-order failures rose (30→42).
Impulse almost-wins rose (15→24).

Pre-registered verdict: **`negative_arm_ceiling`**.

**This is an oracle diagnostic, not learned-policy performance.** The gripper used
privileged scripted grasp-target information. Early-close and reopen went to zero by
construction; that must not be credited as a policy improvement. The decisive evidence is
physically sane success, lift, worst-seed reliability, and paired recovery — all of which
failed to improve.

## What To Review

- [ ] `outputs/h_ee_015_fsm_upper_bound/h_ee_015_summary.json` — metrics and verdict
- [ ] `outputs/h_ee_015_fsm_upper_bound/h_ee_015_paired_comparison.json` — 5 recoveries / 20 regressions
- [ ] `outputs/h_ee_015_fsm_upper_bound/h_ee_015_registration.json` — frozen hashes and bars before efficacy
- [ ] `outputs/h_ee_015_fsm_upper_bound/h_ee_015_experiment_manifest.json` — provenance
- [ ] `tests/test_h_ee_015.py` — FSM thresholds, latch, arm identity, verdict boundaries

## Implementation Details

### Scientific change (only allowed variable)

- Five EE arm dimensions: byte-identical frozen H-EE-014 MLP arm output (hybrid A1).
- Gripper: oracle FSM only.
  - `OPEN_APPROACH`: command `1.0`.
  - Transition when **all three** hold on the same step (inclusive):
    - position error to scripted grasp target ≤ 0.012 m
    - rotation error ≤ 0.22 rad
    - `gripper_object_distance` ≤ 0.015 m
  - `CLOSE_LATCHED`: command `0.0` for the rest of the episode; never reopen.

### Code

- `src/svla/h_ee_015.py` — pure FSM, policy wrapper, registration/summary/verdict helpers.
- `src/svla/state_bc.py` — opt-in: if policy exposes `set_oracle_signals`, feed privileged
  grasp errors before each predict. Default hybrid/MLP/NN paths unchanged.
- `scripts/run_h_ee_015_fsm_upper_bound.py` — register → smoke → evaluate → finalize.
- `tests/test_h_ee_015.py` — focused unit tests.

### Oracle labeling

Every experimental row/summary/manifest carries:

- `"gripper_source": "oracle_fsm_h_ee_015"`
- `"oracle_diagnostic": true`
- `"final_accessed": false`

## Evidence And Verification

### Registration (pre-efficacy)

```bash
PYTHONPATH=src .venv/bin/python scripts/run_h_ee_015_fsm_upper_bound.py \
  --baseline-dir outputs/h_ee_014_nn_gripper_global_validation \
  --output-dir outputs/h_ee_015_fsm_upper_bound \
  register
```

- Result: exact H-EE-014 primary reproduction (62/79/68, seeds [20,14,9,9,10], missing_lift 30).
- 120 paired `(seed, trial_id)` keys frozen; FSM thresholds and verdict bars frozen.
- Output: `outputs/h_ee_015_fsm_upper_bound/h_ee_015_registration.json`

### Focused tests

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_h_ee_015.py -v
```

- Result: 24 passed (threshold sides + equality, simultaneous transition, latch/no-reopen,
  arm 5-D identity, default hybrid has no FSM hook, oracle flags, pairing, verdict bounds).

### Smoke

```bash
PYTHONPATH=src .venv/bin/python scripts/run_h_ee_015_fsm_upper_bound.py \
  --baseline-dir outputs/h_ee_014_nn_gripper_global_validation \
  --output-dir outputs/h_ee_015_fsm_upper_bound \
  smoke
```

- Result: trial 6001 seed 0 transitioned at step 245 with pos/rot/dist under thresholds.
- Telemetry inspected; oracle flags present.

### Full validation (5 seeds × 24 = 120)

```bash
PYTHONPATH=src .venv/bin/python scripts/run_h_ee_015_fsm_upper_bound.py \
  --baseline-dir outputs/h_ee_014_nn_gripper_global_validation \
  --output-dir outputs/h_ee_015_fsm_upper_bound \
  evaluate
PYTHONPATH=src .venv/bin/python scripts/run_h_ee_015_fsm_upper_bound.py \
  --baseline-dir outputs/h_ee_014_nn_gripper_global_validation \
  --output-dir outputs/h_ee_015_fsm_upper_bound \
  finalize
```

### Headline metrics

| Metric | H-EE-014 hybrid A1 (raw) | H-EE-015 oracle FSM |
|--------|--------------------------:|--------------------:|
| Success | 62/120 | **47/120** |
| Event order | 79/120 | 77/120 |
| Physical sanity | 68/120 | **56/120** |
| Worst seed | 9/24 | **5/24** |
| Per-seed success | [20, 14, 9, 9, 10] | [16, 12, 7, 7, 5] |
| Missing-lift EO | 30 | **42** |
| Early-close | 11 | 0 (hard-coded) |
| Reopen events | 0 | 0 (hard-coded) |
| Never transitioned | n/a | **0** |
| Impulse almost-wins | 15 | **24** |
| Contact-dynamics fails | 15 | **27** |
| Controller failures | 0 | 0 |

### Paired comparison vs H-EE-014

- Recoveries (fail→success): **5**
- Regressions (success→fail): **20**
- Net success change: **−15**
- Recoveries from baseline missing-lift: **1**

### What this proves

- Privileged, structurally correct gripper sequencing does **not** unlock a substantially
  stronger frozen arm policy under this fixed FSM.
- The FSM always closed (0 never-transitioned), so failures are not “gripper never closed.”
- Residual mass shifted toward missing-lift EO and contact-dynamics / impulse almost-wins.
- The nearer ceiling is the learned arm trajectory / label / controller interaction, not
  NN gripper timing alone.

### What this does not prove

- That no other gripper schedule could help (only this one fixed FSM was tested).
- That learned EE is hopeless (joint hybrid remains 97/120; raw EE comparison unchanged).
- Any fair learned-policy gain (oracle task state).

## Demo Videos / Visual Artifacts

No new videos were generated for this diagnostic. Residual visual freeze from SP0 remains at:

- `outputs/h_ee_014_residual_visual_review.md`
- `outputs/h_ee_014_residual_clips/`

## Decisions Made

- Decision: Assign `negative_arm_ceiling` under pre-registered bars.
  Reason: success 47 < 72 partial bar; phys 56 < 68; worst seed 5 < 10; missing-lift rose.
- Decision: Do not credit early_close=0 / reopen=0 as improvement.
  Reason: hard-coded by latched FSM; not learned behavior.
- Decision: Do not open final, retrain, retune thresholds, or start Phase 6b.
  Reason: contract forbids rescue and final access; negative result is complete.
- Decision: Best learned EE baseline remains hybrid A1 62/120.
  Reason: oracle FSM is not a substitute for raw EE performance.

## Risks And Limitations

- Risk: FSM uses privileged scripted grasp targets.
  Why it matters: not deployable; must stay labeled oracle diagnostic forever.
- Risk: Only one fixed threshold set was tested.
  Why it matters: a different schedule might behave differently — but post-hoc tuning is
  forbidden for this registered experiment.
- Risk: Closing at geometric legality may be earlier/later than the NN’s state-local timing
  that co-evolved with the arm labels.
  Why it matters: explains why success can fall even when early-close is eliminated.

## Action Items

- [x] Register, implement, test, run full 120, document, commit on isolated branch.
- [ ] Next scientific step: stop treating gripper logic as the primary remaining fix.
      Prefer arm/path/force residuals (H-EE-024 train path only if registered softer close
      design), optional H-EE-017 if a non-Markov arm residual is argued carefully, or SP6
      joint pick-place track. Final remains closed; Phase 6b remains blocked.
- [ ] Do not report 47/120 (or any FSM success count) as learned EE BC performance.

## Files Changed

- `src/svla/h_ee_015.py` — experiment library (FSM, wrapper, metrics, verdict).
- `src/svla/state_bc.py` — opt-in oracle signal hook in `rollout_policy`.
- `scripts/run_h_ee_015_fsm_upper_bound.py` — dedicated runner; finalize writes the
  final summary before hashing it into the experiment manifest, then runs
  `assert_finalize_artifact_hashes`.
- `tests/test_h_ee_015.py` — focused unit tests, including stale-summary-hash detection.
- `outputs/h_ee_015_fsm_upper_bound/*` — registration, trials, summary, paired, manifest.
- `reports/2026-07-14-h-ee-015-fsm-upper-bound.md` — this report.
- `researchnotes.md`, `RESULTS.md`, `AGENTS.md` — status / next-step updates.

## Current Verdict

**`negative_arm_ceiling`** — oracle diagnostic complete. The frozen learned EE arm under this
fixed privileged gripper FSM is **worse**, not better, than the raw hybrid A1 baseline.
Learned-policy comparison remains NOT READY; best EE stays hybrid A1 **62/120**; Phase 6b
blocked; final closed.
