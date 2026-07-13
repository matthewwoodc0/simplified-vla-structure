# 2026-07-13 - H-EE-002 Frozen-Hybrid EE Gain Causality Test

## Plain-English Summary

H-EE-002 is **rejected**. The exact frozen H-EE-014 hybrid models did not fail because arm
gain 1.0 was simply too aggressive. Reducing the five EE arm dimensions to 0.875 or 0.750
made the robot move more slowly but destroyed its learned timing and lift behavior.

The gain-1.0 same-code control reproduced the stored baseline exactly at 62/120 success.
Gain 0.875 collapsed to 5/120, and gain 0.750 produced 0/120. Constraint exposure did fall
on failure trials, but neither lower gain recovered a single paired missing-lift trial.
Instead, they lost 57 and 62 baseline successes. This is the critical causal result:
constraint telemetry is real, but reducing global arm gain is not the fix and the exposure
is likely a symptom, consequence, or more complex interaction.

This was validation-only and inference-only. There was no training, no cap rescue, no extra
gain, no joint re-evaluation, no final access, and no Phase 6b or Phase 7 work.

Worktree: `/Users/matthewwoodcock/.codex/worktrees/56c6/Simplified VLA Structure`

Branch: `codex/h-ee-002-hybrid-gain`

## What To Review

- [ ] `evidence/h_ee_002_hybrid_gain_sweep.json`: committed durable verdict, metrics,
  paired evidence, and hashes for the ignored local artifacts.
- [ ] `outputs/h_ee_002_hybrid_gain_sweep/h_ee_002_registration.json`: preregistered gains,
  bars, source hashes, baseline metrics, and 21-file frozen-input hash inventory.
- [ ] `outputs/h_ee_002_hybrid_gain_sweep/h_ee_002_gain_sweep_summary.json`: complete metrics,
  classifications, and paired trial details for both candidate gains.
- [ ] `outputs/h_ee_002_hybrid_gain_sweep/h_ee_002_paired_comparison.json`: exact
  `(seed, trial_id)` outcome and constraint changes.

## Implementation Details

The experiment changes one scalar at inference: `ActionRepresentation.scale_arm` multiplies
the first five `ee_tool_delta` dimensions by the fixed gain and leaves the gripper dimension
unchanged. The models, gripper commands, NN match contract, MLP temporal features, search
window, task, controller, physics gates, and 120 validation keys stayed frozen.

The dedicated runner enforces this order:

1. Import/hash/load the five frozen EE hybrid A1 policies and original baseline rows.
2. Write registration before any candidate evaluation.
3. Smoke one validation trial at gains 1.0, 0.875, and 0.750.
4. Require exact gain-1.0 reproduction before unlocking candidates.
5. Run 0.875 and 0.750 on the same five seeds × 24 trial keys.
6. Align paired keys and apply the registered pass/partial/reject bars.

The imported baseline directory contains only the required EE model components and H-EE-014
provenance/evaluation files. All copies matched the primary checkout byte-for-byte. The
primary checkout was read only and was never modified.

## Evidence And Verification

### Registration and control

```bash
PYTHONPATH=src '/Users/matthewwoodcock/Documents/Simplified VLA Structure/.venv/bin/python' \
  scripts/run_h_ee_002_gain_sweep.py register

PYTHONPATH=src '/Users/matthewwoodcock/Documents/Simplified VLA Structure/.venv/bin/python' \
  scripts/run_h_ee_002_gain_sweep.py smoke

PYTHONPATH=src '/Users/matthewwoodcock/Documents/Simplified VLA Structure/.venv/bin/python' \
  scripts/run_h_ee_002_gain_sweep.py evaluate --gain 1.0
```

- Result: all 21 imported inputs matched the primary source; all five hybrid policies loaded;
  protocol SHA-256 was `bffd862d...e946f99`.
- Result: gain 1.0 reproduced exact primary counts: success 62, EO 79, phys 68,
  seeds 20/14/9/9/10, missing-lift 30, early-close 11, reopen 0, controller failures 0.
- Output artifact: `outputs/h_ee_002_hybrid_gain_sweep/h_ee_002_registration.json`.
- What this proves: the candidate comparison uses the intended frozen models and deterministic
  validation contract.
- What it does not prove: whether a lower gain improves behavior; that required the sweep.

### Full registered sweep

```bash
PYTHONPATH=src '/Users/matthewwoodcock/Documents/Simplified VLA Structure/.venv/bin/python' \
  scripts/run_h_ee_002_gain_sweep.py evaluate --gain 0.875

PYTHONPATH=src '/Users/matthewwoodcock/Documents/Simplified VLA Structure/.venv/bin/python' \
  scripts/run_h_ee_002_gain_sweep.py evaluate --gain 0.75

PYTHONPATH=src '/Users/matthewwoodcock/Documents/Simplified VLA Structure/.venv/bin/python' \
  scripts/run_h_ee_002_gain_sweep.py finalize
```

| Gain | Success | EO | Phys | Per seed | Worst | Missing lift | Early close | Reopen | Controller failures |
|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 1.000 | 62/120 | 79 | 68 | 20, 14, 9, 9, 10 | 9 | 30 | 11 | 0 | 0 |
| 0.875 | 5/120 | 9 | 26 | 5, 0, 0, 0, 0 | 0 | 86 | 25 | 0 | 0 |
| 0.750 | 0/120 | 0 | 37 | 0, 0, 0, 0, 0 | 0 | 72 | 48 | 0 | 0 |

Constraint counts are per rollout; rates normalize by total rollout steps.

| Gain | Rollout steps total / mean | JL mean / median / p95 | Infeasible mean / median / p95 | Failure JL rate | Failure infeasible rate |
|---:|---:|---:|---:|---:|---:|
| 1.000 | 212,699 / 1,772.49 | 404.69 / 119.5 / 1,899.1 | 228.57 / 73.5 / 1,565.35 | 28.24% | 15.43% |
| 0.875 | 362,856 / 3,023.80 | 466.09 / 72 / 2,170.55 | 83.48 / 29.5 / 302.65 | 15.38% | 2.78% |
| 0.750 | 384,000 / 3,200.00 | 332.59 / 0 / 2,265.65 | 5.72 / 0 / 41.30 | 10.39% | 0.18% |

The lower gains ran much longer because almost every rollout exhausted the 3,200-step limit.
That is why raw count means and normalized exposure rates must both be shown.

Success-conditioned joint-limit/infeasible rates were 9.33%/6.58% at gain 1.0 and
17.46%/1.75% across the five successes at gain 0.875. Gain 0.750 has no success-conditioned
cohort because it produced no successes. Complete success/failure distributions are retained
in each per-gain summary JSON.

### Paired missing-lift and constraint evidence

- Gain 0.875: zero new successes, 57 lost successes. In the 30 original missing-lift trials,
  mean joint-limit steps fell 964.63→516.80 and infeasible steps fell 490.30→84.47, but
  25 remained missing-lift, one recovered event order only to fail contact dynamics, four
  changed to other event-order failures, and zero became successful.
- Gain 0.750: zero new successes, all 62 baseline successes lost. In the same 30 original
  missing-lift trials, mean joint-limit steps fell 964.63→364.30 and infeasible steps fell
  490.30→1.60, but 19 remained missing-lift, 11 changed to other event-order failures, and
  zero became successful.
- What this proves: lower global gain reduces much of the measured constraint exposure but
  does not convert constraint-heavy failures into successful lifts.
- What it does not prove: controller constraints never matter. It rejects this monotonic
  global-gain intervention, not every possible controller/path interaction.

### Contact-dynamics almost-wins

- Gain 1.0: 15 contact-dynamics almost-wins; 13 exceeded 9 N s; mean impulse 11.44 N s,
  maximum 20.67 N s.
- Gain 0.875: 4 almost-wins; all exceeded 9 N s; mean impulse 13.70 N s, maximum 19.42 N s.
- Gain 0.750: 0 almost-wins because no rollout achieved a valid event order/lift/retention.
- Interpretation: fewer almost-wins at lower gain are not progress; the policies stopped
  reaching the successful-lift neighborhood.

### Tests

```bash
PYTHONPATH=src '/Users/matthewwoodcock/Documents/Simplified VLA Structure/.venv/bin/python' \
  -m pytest -q tests/test_action_representation.py tests/test_h_ee_002.py
```

- Result before registration: **12 passed**.
- Sandboxed full-suite attempt: **124 passed, 8 failed**. Every failure was a vision test
  raising `CGLError: invalid CoreGraphics connection`; this is the known macOS renderer
  sandbox limitation, not an H-EE-002 assertion failure.
- Full repository rerun with graphics permission: **132 passed in 105.25 s**.

## Demo Videos / Visual Artifacts

No new video was generated. The registered causal decision is fully determined by paired
validation rows and physics/controller telemetry. Existing H-EE-014 residual videos were not
modified. Generated `outputs/` artifacts are ignored by Git; rerun the commands above to
regenerate them from the separately retained frozen inputs.

## Decisions Made

- Decision: reject H-EE-002 and select no lower gain.
  Reason: both candidates failed primary efficacy, reliability, and paired-causality bars.
- Decision: treat lower constraint telemetry as non-causal under this intervention.
  Reason: exposure fell without any paired missing-lift recovery and task behavior collapsed.
- Decision: do not test another gain or cap.
  Reason: that would violate the preregistered sweep and create a post-hoc rescue variant.
- Decision: retain hybrid A1 gain 1.0 as the frozen baseline.
  Reason: it remains 57–62 successes better than the tested lower gains.

## Risks And Limitations

- Risk or limitation: this rejects fixed global arm-gain reduction, not all forms of
  controller-aware policy design.
  Why it matters: a separately preregistered path/controller hypothesis could still be valid,
  but H-EE-002 provides no license for post-hoc cap tuning.
- Risk or limitation: the evaluation is MuJoCo validation only.
  Why it matters: it is neither final-holdout evidence nor hardware-calibrated evidence.
- Risk or limitation: `outputs/` is ignored by Git.
  Why it matters: the committed evidence file stores exact artifact hashes, but the full JSONL
  rows must remain with the local experiment directory or be regenerated.

## Action Items

- [ ] Keep gain 1.0 and the H-EE-014 hybrid A1 models frozen as the best EE baseline.
- [ ] Do not lower gain further or add cap tuning as an H-EE-002 rescue.
- [ ] If research continues, H-EE-015 is the next registered diagnostic; H-EE-017 follows
  only if the arm residual remains plausibly non-Markov.
- [ ] Keep the final holdout and Phase 6b closed.

## Files Changed

- `src/svla/h_ee_002.py` - frozen-input validation, metrics, paired analysis, and verdict bars.
- `scripts/run_h_ee_002_gain_sweep.py` - ordered inference-only experiment runner.
- `tests/test_action_representation.py` - byte-identity test for gain 1.0.
- `tests/test_h_ee_002.py` - frozen model/protocol, paired-key, and classification tests.
- `evidence/h_ee_002_hybrid_gain_sweep.json` - committed durable evidence and artifact hashes.
- `researchnotes.md` - rejected verdict, SP4 status, priority, and Results log.
- `AGENTS.md` - durable verdict and next-work boundary.
- `RESULTS.md` - concise negative-result index.
- `reports/2026-07-13-h-ee-002-hybrid-gain.md` - this plain-language audit report.

## Current Verdict

**Rejected on validation.** The experiment is complete, not blocked. No fixed lower gain was
selected. H-EE-014 hybrid A1 at gain 1.0 remains the EE baseline; final and Phase 6b remain
closed.
