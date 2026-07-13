# H-EE-007 Prompt — EE Label-Contract Causal Probe

**Target agent:** Grok Build 4.5
**Status:** ready to execute
**Hypothesis:** the reconstructed executable EE labels introduce enough bias/noise to make
`ee_tool_delta` harder to learn than the raw observed EE transition labels.
**Scope:** replay audit → one-seed screen → full validation only if the screen passes.
**Final holdout:** closed.
**Phase 6b / Phase 7:** out of scope.

Copy the block below into a fresh Grok task. This file is the complete experiment contract.

---

## PASTE INTO GROK

```text
You are the research engineer responsible for H-EE-007 in the repository:
/Users/matthewwoodcock/Documents/Simplified VLA Structure

GOAL
Determine whether the EE policy is disadvantaged by its training-label construction.
Compare raw observed transition labels (`labels.ee_tool_delta`) against reconstructed
controller-feasible labels (`policy_labels.ee_tool_delta`) without changing the task,
controller, demonstrations, model, loss, temporal features, gripper strategy, validation
starts, gates, or joint baseline.

This is a causal experiment, not a general cleanup. One scientific variable may change:
the EE arm training/replay label source. Do not stack other improvements.

READ FIRST — THESE FILES ARE AUTHORITY
1. AGENTS.md
2. researchnotes.md — H-EE-007 and the post-H-EE-014 residual program
3. prompts/h-ee-007-label-contract-probe.md — this entire contract
4. prompts/h-ee-014-nn-gripper-plan.md — frozen hybrid baseline context
5. src/svla/demo_recorder.py — `_policy_labels`
6. src/svla/state_bc.py and scripts/train_state_bc.py
7. src/svla/pick_place_replay.py and scripts/validate_action_replay.py
8. outputs/h_ee_014_nn_gripper_global_validation/h_ee_014_comparison.json

STARTING BRANCH
- Start from current `main`.
- Create and work on: `research/h-ee-007-label-contract`
- Never rewrite history, delete outputs, or force-push.

FROZEN RESEARCH BASELINE
- Task: MuJoCo pickup only.
- Evaluation: protocol-v2 `validation`; final is forbidden.
- Policy: H-EE-014 hybrid A1 (`hybrid_nn_gripper_mlp`).
- Loss: `global_gripper`.
- Temporal mode: `legacy_progress_phase`.
- Architecture: MLP 128 128, 300 epochs, batch 1024, lr 1e-3, wd 1e-5.
- Full seeds: 0,1,2,3,4; 24 validation trials per seed.
- Historical NN match contract; k/temperature unchanged.
- Action gain 1.0; shield/FSM off.
- Physics/event/success gates unchanged.
- Frozen hybrid EE baseline: success 62/120, EO 79/120, phys 68/120,
  reopen 0, worst seed 9/24, early-close 11, missing-lift EO about 30.
- Frozen hybrid joint reference: success 97/120. Do not retrain or modify joint.

THE PRECISE QUESTION
`demo_recorder._policy_labels` uses controller telemetry:
- joint arm label = `joint_target_error`
- EE arm label = reconstructed `feasible_delta_xyz` + `feasible_delta_rotvec[:2]`

Raw `labels.ee_tool_delta` instead records the observed before→after EE transition.
Test whether that distinction causes the EE learned-policy gap.

ALLOWED CODE CHANGES
- Add an explicit label-source argument to replay helpers, defaulting to the current
  `policy_labels` behavior for compatibility.
- Add a narrow way to train/evaluate only `ee_tool_delta` while leaving the existing joint
  baseline untouched (for example `--action-spaces ee_tool_delta`). Default CLI behavior must
  remain both action spaces.
- Add an EE-specific label-source override if needed. Do not change joint label source.
- Add analysis/output scripts and tests required for this experiment.
- Add manifests/config records and the required report.

FORBIDDEN CHANGES
- No controller gains, caps, IK, task geometry, demonstrations, observations, temporal mode,
  model size, loss weights, NN match features, gripper override, shield, or success-gate change.
- No extra demonstrations.
- No final split.
- No vision policy, language, VLA, Transformer, diffusion, or action chunking.
- Do not silently use raw joint labels as part of this experiment.
- Do not claim success from train loss or label reconstruction error alone.

MANDATORY EXECUTION ORDER

PHASE A — READ-ONLY LABEL AUDIT
1. Load the exact scripted demos used by the H-EE-014 baseline.
2. For every sample, compare raw and policy EE arm labels by phase:
   - per-dimension mean/median/p95 absolute difference
   - vector L2 mean/median/p95/max
   - cosine agreement where norms are nonzero
   - action magnitude distributions
   - fraction at/near controller translation or rotation clipping
   - gripper-label equality (must be checked explicitly)
3. Break results down by `approach_0/1/2`, `grasp_align`, `close_gripper`, `lift`, `hold`.
4. Write before any candidate rollout:
   outputs/h_ee_007_label_contract_probe/h_ee_007_label_audit.json

PHASE B — EXECUTABILITY / REPLAY GATE
1. Extend replay so the label source is explicit and recorded.
2. Replay the same 18 pickup demonstrations using:
   A. `policy_labels.ee_tool_delta` (control)
   B. `labels.ee_tool_delta` (candidate)
3. Record success, event order, physical sanity, preclose contact, reopen, force, impulse,
   supported displacement, controller failures, clipping, and saturation.
4. Write:
   outputs/h_ee_007_label_contract_probe/h_ee_007_replay_comparison.json

REPLAY GO/NO-GO
- GO only if raw-label replay is 18/18 successful, event-order-valid, and physical-sanity-pass,
  with zero controller failures and no material gate regression relative to policy-label replay.
- If raw replay fails this gate, mark H-EE-007 `rejected` or `diagnosed` with exact causes,
  update docs/report, and STOP. Do not train.

PHASE C — PRE-REGISTERED ONE-SEED SCREEN
Before training, compute seed-2 baseline metrics from the frozen H-EE-014 trial rows and write:
outputs/h_ee_007_label_contract_probe/h_ee_007_registration.json

The registration must freeze:
- seed 2
- all baseline metrics and source hashes
- raw EE labels as the only changed scientific variable
- pass/kill bars below

Train EE only, seed 2, under the frozen H-EE-014 contract using raw EE labels. The hybrid NN
gripper remains; because gripper commands should be identical across label sources, verify and
record that fact rather than assuming it.

SCREEN PASS — proceed to full validation only if all are true:
- success improves by at least +5/24 versus frozen seed-2 baseline OR seed-2 missing-lift EO
  falls by at least 30%
- physical-sanity count is no more than 2 below seed-2 baseline
- reopen events remain <=2
- no controller failures

If the screen fails, mark H-EE-007 rejected on the screen and STOP. Do not try multiple seeds,
different gains, history, or loss changes to rescue it.

PHASE D — FULL VALIDATION (ONLY AFTER SCREEN PASS)
Train raw-label EE policies for seeds 0-4 and run 24 protocol-v2 validation trials per seed.
Do not access final. Do not retrain joint; use the frozen joint reference only.

FULL CONFIRMATION BARS
Primary efficacy — at least one:
- EE success >=72/120 (+10 over hybrid A1), OR
- missing-lift EO <=21/120 (>=30% reduction from about 30)

All safety/reliability bars:
- worst seed >=11/24
- physical sanity >=68/120
- reopen <=5 total
- early-close <=11
- zero controller failures

Outcome labels:
- `confirmed` only if a primary efficacy bar and every safety/reliability bar pass.
- `partial` if label audit/replay shows a real contract difference and rollout improves, but
  full confirmation bars are not all met.
- `rejected` if the replay gate, screen, or full bars fail without a meaningful causal gain.

REQUIRED TESTS
- Existing `policy_labels` replay behavior remains byte/metric compatible by default.
- Raw vs policy label-source selection is explicit and unit tested.
- Joint label source remains unchanged.
- CLI defaults still train both action spaces with `policy_labels`.
- Tiny raw-label EE smoke can train, save, load, and roll out.
- Full repository pytest passes. MuJoCo rendering tests may need the macOS graphics context.

REQUIRED ARTIFACTS
- outputs/h_ee_007_label_contract_probe/h_ee_007_label_audit.json
- outputs/h_ee_007_label_contract_probe/h_ee_007_replay_comparison.json
- outputs/h_ee_007_label_contract_probe/h_ee_007_registration.json
- one-seed screen summary/manifest
- full validation summary/manifest only if screen passed
- outputs/h_ee_007_label_contract_probe/h_ee_007_comparison.json
- reports/YYYY-MM-DD-h-ee-007-label-contract.md

DOCUMENTATION
- Update the H-EE-007 row and Results log in researchnotes.md.
- Add an AGENTS.md verdict bullet only if the research verdict/next step moves.
- Update RESULTS.md if H-EE-007 becomes confirmed/partial/rejected with durable evidence.
- The report must follow the AGENTS.md large-change template and state what was not proven.

STOP CONDITIONS
- Respect every gate above.
- A clean rejection is a completed experiment.
- Maximum one implementation retry for a genuine code bug; no scientific rescue variants.
- Never open final or start Phase 6b/7.

FINAL RESPONSE TO USER
Lead with: H-EE-007 confirmed / partial / rejected.
Then give exact replay, seed-screen, and (if run) five-seed metrics; evidence paths; tests; files;
and the next decision. Do not call lower train loss a research win.
```
