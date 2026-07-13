# H-EE-002 Prompt — Frozen-Hybrid EE Gain Causality Test

**Target agent:** Grok Build 4.5
**Status:** ready to execute
**Hypothesis:** the hybrid EE policy fails to lift because action gain 1.0 lets target error
accumulate faster than the constrained controller can track, producing limit thrash and weak lift.
**Scope:** inference-only evaluation of frozen H-EE-014 models at preregistered EE gains.
**Final holdout:** closed.
**Phase 6b / Phase 7:** out of scope.

Copy the block below into a fresh Grok task. This file is the complete experiment contract.

---

## PASTE INTO GROK

```text
You are the research engineer responsible for H-EE-002 in the repository:
/Users/matthewwoodcock/Documents/Simplified VLA Structure

GOAL
Test whether the remaining H-EE-014 EE missing-lift/constraint-thrash failures are caused by
deployment action gain. Re-evaluate the exact frozen hybrid A1 models at a preregistered set of
EE arm gains while keeping weights, gripper commands, task, validation trials, controller,
gates, temporal behavior, and joint reference unchanged.

This is an inference-only causal sweep of one scalar. Do not retrain and do not turn it into a
general controller-tuning exercise.

READ FIRST — THESE FILES ARE AUTHORITY
1. AGENTS.md
2. researchnotes.md — H-EE-002 and post-H-EE-014 residual anatomy
3. prompts/h-ee-002-hybrid-gain-sweep.md — this entire contract
4. prompts/h-ee-014-nn-gripper-plan.md
5. src/svla/state_bc.py — `rollout_policy(action_gain=...)`
6. src/svla/core/action_space.py — arm-only scaling contract
7. scripts/run_h_ee_022_match_reeval.py — example of frozen-model re-evaluation
8. outputs/h_ee_014_nn_gripper_global_validation/h_ee_014_comparison.json
9. outputs/post_h_ee_014_residual_scoreboard.json

STARTING BRANCH
- Start from current `main`.
- Create and work on: `research/h-ee-002-hybrid-gain`
- Never rewrite history, delete outputs, or force-push.

FROZEN BASELINE
- Models: exact H-EE-014 hybrid A1 model files for EE seeds 0-4.
- Model directory: outputs/h_ee_014_nn_gripper_global_validation/models/
- Policy/loss/match: hybrid NN gripper + MLP arm, `global_gripper`, historical match.
- Evaluation: protocol-v2 validation only, 24 trials per seed.
- Temporal mode, search window, k, temperature, observations, action labels: frozen.
- Controller/task/physics/event gates: frozen.
- Gripper command must never be scaled.
- EE baseline at gain 1.0: success 62/120, EO 79, phys 68, reopen 0,
  worst seed 9, early-close 11, missing-lift EO about 30.
- Joint reference: 97/120 at its frozen contract. Do not retrain or re-evaluate joint as part
  of this EE-only causal test.

PRE-REGISTERED GAIN VALUES
- 1.000 — same-code control
- 0.875 — primary moderate reduction
- 0.750 — stronger reduction

Do not add 0.9, 0.8, per-axis gains, clipping changes, or adaptive gain after seeing results.
If this sweep suggests a cap-specific follow-up, propose it later as a separate hypothesis.

ALLOWED CODE CHANGES
- Add a dedicated script such as `scripts/run_h_ee_002_gain_sweep.py` that loads frozen hybrid
  models and performs validation re-evaluation without training.
- Reuse existing policy loading, protocol, rollout, summary, and manifest utilities.
- Add analysis/tests/config/report files required by this experiment.
- Add telemetry aggregation for the residual metrics listed below if absent.

FORBIDDEN CHANGES
- No training, new demos, label change, loss change, architecture change, history, NN match
  change, gripper override, FSM, distance guard, controller limit/cap change, or task change.
- No per-axis or phase-dependent scaling.
- No gate relaxation.
- No final split.
- No vision, language, VLA, diffusion, tokenization, or action chunking.
- Do not select the best gain separately per seed or per trial.

MANDATORY EXECUTION ORDER

PHASE A — REGISTRATION BEFORE CANDIDATE EVALUATION
1. Verify SHA-256 identities of all frozen hybrid manifests/components for seeds 0-4.
2. Load the original baseline trial rows and compute exact baseline metrics, including the
   missing-lift bucket and per-seed vector.
3. Write before evaluating 0.875 or 0.750:
   outputs/h_ee_002_hybrid_gain_sweep/h_ee_002_registration.json
4. Registration must include gain list, model hashes, protocol hash, baseline metrics,
   pass/kill bars, and `final_accessed=false`.

PHASE B — TESTS AND SAME-CODE CONTROL
1. Unit test that `ActionRepresentation.scale_arm` scales exactly the five EE arm dimensions
   and leaves the gripper dimension byte-identical.
2. Unit test that gain 1.0 leaves the entire action vector unchanged.
3. Run a tiny one-trial smoke at each gain to prove plumbing only.
4. Run the full gain-1.0 control first (5 seeds × 24 validation trials).
5. Control must reproduce the stored baseline within exact deterministic equality for primary
   counts. If it does not, STOP and diagnose code/source drift before candidate gains.

PHASE C — FULL FROZEN-MODEL SWEEP
Run gain 0.875 and 0.750, each on the same five seeds and same 24 validation specs.
No training and no eval-limit.

FOR EACH GAIN RECORD
- success / 120
- event-order-valid / 120
- physical-sanity-pass / 120
- per-seed successes and worst seed
- missing-lift EO count
- early-close count
- reopen events
- contact-dynamics failures
- impulse almost-wins and mean/max impulse
- mean/median/p95 joint-limit-clipped steps
- mean/median/p95 infeasible steps
- controller failures
- rollout steps and success-conditioned/failure-conditioned constraint exposure
- paired trial outcome changes versus gain 1.0 using identical `(seed, trial_id)` keys

PASS / SELECTION BARS
A lower gain is a meaningful causal win only if all are true:

Primary efficacy — at least one:
- success >=72/120 (+10), OR
- missing-lift EO <=21/120 (>=30% reduction)

Reliability/safety:
- worst seed >=11/24
- physical sanity >=68/120
- event order >=79/120
- reopen <=5
- early-close <=11
- zero controller failures

Controller-causality support:
- joint-limit or infeasible exposure must materially decline on failure trials, and
- paired improvements must be concentrated in the prior missing-lift/thrash bucket rather than
  created by unrelated gate changes.

OUTCOME LABELS
- `confirmed` if one fixed gain passes every bar and the paired evidence supports controller
  causality.
- `partial` if constraints and missing-lift improve clearly but aggregate/worst-seed bars miss.
- `rejected` if lower gain reduces telemetry exposure without meaningful success/lift gains,
  or if task/physics/event results regress.

CRITICAL INTERPRETATION RULE
If lower gain reduces clipping but does not improve successful lift, conclude that constraint
exposure is likely a symptom or consequence—not the root cause. Do not continue lowering gain.

REQUIRED ARTIFACTS
- outputs/h_ee_002_hybrid_gain_sweep/h_ee_002_registration.json
- one JSONL file per gain with full trial rows
- one summary JSON per gain
- outputs/h_ee_002_hybrid_gain_sweep/h_ee_002_paired_comparison.json
- outputs/h_ee_002_hybrid_gain_sweep/h_ee_002_gain_sweep_summary.json
- manifest(s) with frozen model/protocol/source hashes
- reports/YYYY-MM-DD-h-ee-002-hybrid-gain.md

REQUIRED TESTS
- scaling affects arm only
- gain 1.0 is identity
- frozen model loading and protocol hash checks
- paired-key alignment rejects missing/duplicate `(seed, trial_id)` rows
- summary/pass-bar classification tests
- full repository pytest passes

DOCUMENTATION
- Update H-EE-002 and Results log in researchnotes.md.
- Add an AGENTS.md verdict bullet only if the causal verdict or next phase changes.
- Update RESULTS.md with durable confirmed/partial/rejected evidence.
- The report must follow the AGENTS.md template and explicitly state this is validation-only,
  inference-only, and not a final or hardware result.

STOP CONDITIONS
- A clean rejection is completion.
- No cap retry inside this experiment.
- No gain values beyond the preregistered three.
- No final, Phase 6b, or Phase 7.

FINAL RESPONSE TO USER
Lead with H-EE-002 confirmed / partial / rejected. State the selected fixed gain if any, exact
metrics at all three gains, paired missing-lift changes, constraint evidence, artifact/report
paths, tests, and the next decision. Do not call lower clipping alone a win.
```
