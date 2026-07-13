# H-EE-015 Prompt — Oracle-Gripper Arm Upper-Bound Diagnostic

**Target agent:** Grok Build 4.5
**Status:** ready to execute
**Hypothesis:** the frozen hybrid EE arm policy is substantially better than its aggregate
result suggests, but residual gripper timing prevents legal grasp and lift execution.
**Scope:** inference-only pickup diagnostic using an explicitly oracle, latched gripper FSM.
**Final holdout:** closed.
**Phase 6b / Phase 7:** out of scope.

Copy the block below into a fresh Grok task. This file is the complete experiment contract.

---

## PASTE INTO GROK

```text
You are the research engineer responsible for H-EE-015 in the repository:
/Users/matthewwoodcock/Documents/Simplified VLA Structure

GOAL
Measure the upper bound of the frozen H-EE-014 EE arm policy when gripper sequencing is supplied
by an explicit task-aware finite-state machine. This experiment asks whether the remaining gap
is still primarily gripper timing, or whether the arm labels/controller/path are now the ceiling.

This is an oracle diagnostic, not a fair learned-policy result. Never report its success count as
EE behavioral cloning performance and never use it to replace the raw EE-vs-joint comparison.

READ FIRST — THESE FILES ARE AUTHORITY
1. AGENTS.md
2. researchnotes.md — H-EE-015 and the post-H-EE-014 residual program
3. prompts/h-ee-015-fsm-arm-upper-bound.md — this entire contract
4. prompts/h-ee-014-nn-gripper-plan.md — frozen hybrid baseline context
5. src/svla/state_bc.py — hybrid rollout and action composition
6. src/svla/pickup_task.py — event-order gates and grasp geometry
7. scripts/run_h_ee_022_match_reeval.py — frozen-model re-evaluation example
8. outputs/h_ee_014_nn_gripper_global_validation/h_ee_014_comparison.json
9. outputs/post_h_ee_014_residual_scoreboard.json

STARTING BRANCH
- Start from current `main`.
- Create and work on: `research/h-ee-015-fsm-upper-bound`
- Never rewrite history, delete outputs, or force-push.

FROZEN BASELINE
- Models: exact H-EE-014 hybrid A1 EE model files, seeds 0-4.
- Model directory: outputs/h_ee_014_nn_gripper_global_validation/models/
- Evaluation: protocol-v2 validation only, 24 pickup trials per seed.
- Arm output: frozen MLP arm component, `global_gripper`, historical temporal contract.
- Action gain 1.0; task/controller/physics/event gates unchanged.
- No retraining and no new demonstrations.
- Raw hybrid EE baseline: success 62/120, EO 79/120, phys 68/120,
  per-seed [20,14,9,9,10], worst 9, early-close 11, reopen 0,
  missing-lift EO 30, impulse almost-wins 15.
- Frozen hybrid joint reference: 97/120. Do not re-evaluate or modify joint.

THE ONLY CHANGED SCIENTIFIC VARIABLE
Replace the frozen hybrid NN gripper output at rollout with the oracle FSM below. The five EE arm
dimensions must still come from the same frozen MLP component and must not be scaled, filtered,
shielded, retrained, or replaced.

FIXED FSM CONTRACT
States:
- `OPEN_APPROACH`
- `CLOSE_LATCHED`

In `OPEN_APPROACH`, command fully open (`gripper = 1.0`). Transition once, and only once, to
`CLOSE_LATCHED` when every condition below is true on the same simulator step:
1. Cartesian position error between current tool pose and the task's scripted grasp target
   is <= 0.012 m.
2. Rotation error to that target is <= 0.22 rad.
3. `gripper_object_distance <= 0.015 m`, the existing early-close legality threshold.

In `CLOSE_LATCHED`, command fully closed (`gripper = 0.0`) for the remainder of the pickup
episode. Never reopen. If the transition conditions are never met, remain open and let the trial
fail naturally.

The scripted grasp target is privileged task information. Mark every affected row and summary:
- `gripper_source = "oracle_fsm_h_ee_015"`
- `oracle_diagnostic = true`

Do not add contact, phase clock, trial outcome, future state, or success-gate information to the
transition. Do not tune thresholds after observing validation results.

ALLOWED CODE CHANGES
- Add a named, opt-in rollout gripper source for this exact FSM; default rollout behavior must
  remain unchanged.
- Add a dedicated frozen-model evaluation script such as
  `scripts/run_h_ee_015_fsm_upper_bound.py`.
- Add transition telemetry, summaries, tests, manifests, and the required report.

FORBIDDEN CHANGES
- No training, new demos, raw/policy label change, gain/cap change, controller change, task or
  gate change, loss change, temporal/history change, NN match change, or arm action change.
- No retreat/open state; this experiment is pickup only.
- No trial-specific thresholds or post-hoc FSM variants.
- No final split.
- No vision, language, VLA, Transformer, diffusion, or action chunking.
- Never call the FSM learned, autonomous, deployable, or a policy improvement.

MANDATORY EXECUTION ORDER

PHASE A — REGISTRATION
1. Verify SHA-256 identities of the five frozen hybrid models/components and source manifest.
2. Recompute baseline counts and paired `(seed, trial_id)` keys from stored H-EE-014 rows.
3. Freeze the FSM thresholds, pass bars, model/protocol hashes, and `final_accessed=false` in:
   outputs/h_ee_015_fsm_upper_bound/h_ee_015_registration.json

PHASE B — IMPLEMENTATION TESTS AND SMOKE
1. Unit test every FSM condition at both sides of each threshold.
2. Unit test that the transition latches and can never reopen.
3. Unit test that, for an identical observation, the five arm outputs are exactly equal with and
   without the FSM wrapper; only the gripper scalar may differ.
4. Unit test that default rollouts do not activate the oracle.
5. Run a tiny non-efficacy smoke and inspect transition telemetry.

PHASE C — FULL FROZEN-MODEL VALIDATION DIAGNOSTIC
Run the same seeds 0-4 and the same 24 protocol-v2 validation trials per seed with the FSM.
Do not train, use eval-limit, access final, or select a subset.

RECORD
- success / 120
- event-order-valid / 120
- physical-sanity-pass / 120
- per-seed successes and worst seed
- missing-lift EO count
- early-close count and reopen events
- never-transitioned trials
- FSM transition step and the three transition errors/values
- contact-dynamics failures and impulse almost-wins
- force, impulse, supported displacement, and supported rotation
- joint-limit-clipped, infeasible, and controller-failure steps
- paired outcome transitions versus raw hybrid baseline by `(seed, trial_id)`

PRE-REGISTERED INTERPRETATION
`strong_positive_arm_upper_bound` only if all are true:
- success >=84/120
- event order >=90/120
- physical sanity >=80/120
- worst seed >=12/24
- zero controller failures

Interpretation: gripper decisions were still hiding a substantially stronger arm policy. This does
not validate a learned EE policy; the next learned experiment must imitate the information used by
the FSM without oracle task state.

`partial` if all are true:
- success >=72/120 (+10)
- missing-lift EO or early-close falls materially
- physical sanity >=68/120
- worst seed >=10/24
- zero controller failures

Interpretation: gripper timing contributes, but it does not explain the full action-space gap.

`negative_arm_ceiling` if the partial bars fail, or if event order improves structurally without
successful, physically sane lift. Interpretation: the current learned arm trajectory / label /
controller interaction is the nearer ceiling; stop treating gripper logic as the primary fix.

Do not credit expected `early_close = 0` or `reopen = 0` as learned-policy gains. Those outcomes are
hard-coded by the oracle FSM. The decisive measures are physically sane success, lift, worst-seed
reliability, and paired recovery of previously failed trials.

REQUIRED ARTIFACTS
- outputs/h_ee_015_fsm_upper_bound/h_ee_015_registration.json
- outputs/h_ee_015_fsm_upper_bound/h_ee_015_trials.jsonl
- outputs/h_ee_015_fsm_upper_bound/h_ee_015_summary.json
- outputs/h_ee_015_fsm_upper_bound/h_ee_015_paired_comparison.json
- experiment manifest with model/protocol/source hashes and oracle flags
- reports/YYYY-MM-DD-h-ee-015-fsm-upper-bound.md

REQUIRED TESTS
- exact threshold behavior and latch semantics
- arm-action equality under the wrapper
- default behavior remains NN-gripper hybrid
- oracle flags are mandatory in rows, summary, and manifest
- paired-key alignment rejects missing/duplicate `(seed, trial_id)` rows
- outcome classification at boundary values
- full repository pytest passes

DOCUMENTATION
- Update H-EE-015 and the Results log in researchnotes.md.
- Add an AGENTS.md verdict bullet only if the causal verdict or recommended next step moves.
- Update RESULTS.md with the diagnostic outcome, always labeled oracle/non-learned.
- The report must follow the AGENTS.md large-change template and clearly separate what the FSM
  proves from what it bypasses.

STOP CONDITIONS
- One fixed FSM only. Do not tune or rescue it after validation.
- A negative arm ceiling is a completed, useful experiment.
- Never open final or start Phase 6b/7.

FINAL RESPONSE TO USER
Lead with `strong_positive_arm_upper_bound`, `partial`, or `negative_arm_ceiling`. Give exact raw
versus FSM metrics, paired recoveries/regressions, never-transitioned count, physical-sanity and
constraint evidence, artifact/report paths, tests, and the next decision. State prominently that
the gripper used privileged task information and the result is not learned-policy performance.
```
