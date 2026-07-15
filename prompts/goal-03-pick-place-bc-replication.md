# Goal 03 — Learned Pick-Place Replication For Both Action Spaces

**Target agent:** Grok 4.5
**Goal type:** task protocol, BC integration, and registered validation experiment
**Locked holdout:** closed
**Prerequisite:** Goal 01 and Goal 02 preregistration reviewed; the separately authorized
efficiency curve has then been executed, reviewed, and merged

Copy the block below into a fresh Grok task. This file is the complete contract.

---

## PASTE INTO GROK

```text
You are the research engineer responsible for the second learned manipulation task in:
/Users/matthewwoodcock/Documents/Simplified VLA Structure

GOAL
Turn the existing scripted/replay-only pick-place extension into a fair, goal-conditioned,
validation-only behavioral-cloning comparison for both `joint_delta` and `ee_tool_delta` under the
frozen Phase 5 contract. Determine whether the pickup action-space ordering replicates on the
longer task without opening a locked holdout or rescuing either policy after results.

READ FIRST — AUTHORITY
1. AGENTS.md
2. RESULTS.md
3. researchnotes.md
4. evidence/phase5_causal_synthesis.json
5. evidence/state_bc_efficiency_curve_registration.json
6. the reviewed durable efficiency-curve result evidence and report
7. prompts/goal-03-pick-place-bc-replication.md — this full contract
8. src/svla/pickup_task.py — pick-place task, gates, and placement metrics
9. src/svla/demo_recorder.py — `svla_pick_place_demo_v1`
10. src/svla/pick_place_replay.py
11. src/svla/state_bc.py
12. scripts/train_state_bc.py
13. scripts/run_pick_place_trials.py
14. scripts/record_pick_place_demo.py
15. outputs/phase5_baseline_final/phase5_baseline_v2_aggregate.json

STARTING STATE
- Start only from reviewed `main` after Goal 01, Goal 02 preregistration, and the separately
  authorized/independently reviewed efficiency-curve execution.
- Create `codex/pick-place-bc-replication`.
- Confirm the synthesis and efficiency registrations have matching frozen recipe identities.
- Keep unrelated work out of the branch.

CRITICAL DESIGN PROBLEM TO SOLVE
Pick-place is goal-conditioned. Existing pickup features do not identify whether the requested
placement is left or right. Training divergent post-lift actions from identical observations
without a goal feature would create label ambiguity and invalidate the experiment.

Add an explicit, inspectable placement-goal representation for pick-place only, preferably target
XYZ relative to the object and/or EE plus a named placement-target feature. Preserve byte-identical
pickup features and model loading for existing pickup artifacts.

FROZEN POLICY CONTRACT
- compare both `joint_delta` and `ee_tool_delta`
- hybrid NN gripper + MLP arm A1 compositor
- `global_gripper`, historical NN match, `legacy_progress_phase`
- `policy_labels`, action gain 1.0
- hidden sizes 128 128, 300 epochs, batch 1024, lr 1e-3, wd 1e-5
- seeds 0 1 2 3 4
- no shield, oracle FSM, gain/cap change, history model, or hyperparameter variant

TASK PROTOCOL
Create a new versioned pick-place protocol; do not mutate pickup protocol v2.

It must define disjoint train, validation, and locked-holdout object positions and include:
- 3 grasp yaw orientations
- 2 approach strategies
- both left and right placement targets
- exactly 5 distinct training object positions: 60 train cells total
- exactly 4 distinct validation object positions: 48 validation cells total
- exactly 4 distinct locked-holdout object positions: 48 locked cells total
- every split is the full orientation x approach x placement-target factorial at each position
- validation cells balanced across all factors
- exact trial IDs, positions, target coordinates, controller ID, gates, and hashes
- locked holdout requiring a literal opt-in flag that this goal never supplies

Do not use the existing six-case yaw-zero matrix as the learned validation protocol. It remains a
scripted smoke/baseline only.

PHASE/TEMPORAL CONTRACT
Extend task phase handling explicitly for transport, lower, open, and retreat. Do not silently map
them into pickup phases. Record phase vocabulary and feature names in datasets/models/manifests.
Default pickup phase behavior and old model loading must remain unchanged.

MANDATORY GATES AND EXECUTION ORDER

PHASE A — PROTOCOL AND SCRIPTED TASK GATE
1. Freeze protocol, goal features, phase vocabulary, metrics, and interpretation before ML.
2. Run scripted expert across the complete train and validation matrices.
3. Require 100% scripted success, event order, physical sanity, and placement success.
4. If this fails, stop at the task/controller layer. Do not train.

PHASE B — LABEL REPLAY GATE
1. Record one shared scripted trajectory source per task spec with both action-space labels.
2. Replay all train and validation demonstration labels in both action spaces.
3. Require 100% replay success and zero missing contexts for both spaces.
4. Verify `grasp_segment_finalize_sample_index` alignment and post-grasp placement metrics.
5. If either space fails, stop and classify the label/controller cause. Do not train.

PHASE C — IMPLEMENTATION SMOKE
- one context, one seed, at most two epochs, eval-limit=1
- verify goal features, extended phases, model save/load, rollout, placement metrics, and manifests
- smoke is plumbing only

PHASE D — REGISTERED VALIDATION TRAIN
- train exactly one frozen configuration for both action spaces, seeds 0-4
- evaluate the complete validation split with identical specs and gates
- no eval-limit, no retry architecture, no post-result threshold changes
- do not access locked holdout or pickup final

PRIMARY AND SECONDARY ENDPOINTS
Primary:
- end-to-end pick-place success

Required decomposition:
- valid grasp-segment rate
- placement success conditional on a valid grasp segment
- unconditional placement-achieved rate
- event order and physical sanity
- target release/open and retreat completion
- placement XY and Z error
- force/impulse/disturbance
- controller constraint exposure and failures
- per-seed outcomes and worst seed
- paired joint/EE result by identical `(seed, trial_id)`

INTERPRETATION — PRE-REGISTER BEFORE VALIDATION
Use paired bootstrap confidence intervals for the joint-minus-EE success difference.

Outcome labels:
- `TASK_FLOOR`: both policies <=10% end-to-end success or too few valid grasps to interpret
  conditional placement. Do not claim action-space replication.
- `REPLICATED_JOINT_ADVANTAGE`: joint-minus-EE 95% paired CI is entirely above zero.
- `EE_ADVANTAGE_ON_PICK_PLACE`: joint-minus-EE 95% paired CI is entirely below zero.
- `NO_CLEAR_ACTION_SPACE_DIFFERENCE`: CI crosses zero and the task is not at floor.
- `PIPELINE_BLOCKED`: scripted or replay gate fails, or validation denominator is incomplete.

Report conditional placement separately even if the primary task is at floor. Do not redefine
success to hide inherited grasp failures.

REQUIRED IMPLEMENTATION
- task-aware protocol loader and validation
- goal-conditioned feature contract for pick-place only
- explicit extended phase support
- pick-place dataset loading/training/rollout using shared action representation registry
- complete result rows and summaries
- paired analysis in `analysis/`
- experiment config and manifest support
- render representative scripted and learned outcomes if graphics access works; rendering is
  review evidence, not an efficacy gate

REQUIRED TESTS
- goal-left and goal-right produce distinct explicit task features
- pickup feature names/values remain backward-compatible
- all pick-place phases are encoded and save/load round-trips
- protocol split disjointness and balanced factors
- scripted/replay gate classification
- grasp-segment boundary finalization occurs exactly once
- conditional placement denominator is correct
- paired-key alignment rejects missing/duplicate rows
- locked holdout authorization guard
- old pickup tests and frozen model-loading tests pass
- full non-render suite passes

REQUIRED ARTIFACTS
- versioned pick-place protocol config
- tracked experiment config/registration
- scripted train/validation summaries and manifests
- replay summaries for both action spaces and manifests
- validation models, JSONL rows, summaries, and comparison under `outputs/`
- durable tracked evidence JSON with hashes and exact verdict
- representative visual review if rendering is available
- `reports/YYYY-MM-DD-pick-place-bc-replication.md`

DOCUMENTATION
- Add a registered hypothesis/task-replication row and final validation outcome to
  `researchnotes.md`.
- Update `RESULTS.md` experiment matrix and task result.
- Add an AGENTS.md dated verdict because this changes the research evidence ladder.
- Follow the AGENTS.md large-change report template.

FORBIDDEN
- joint-only result presented as task replication
- modifying pickup final or new pick-place locked holdout
- tuning goal features, phases, loss, gain, or model after validation results
- weakening grasp, physics, or placement gates
- counting conditional placement as end-to-end success
- vision, language, VLA, second robot, or hardware

STOP CONDITIONS
- A scripted/replay failure is completed diagnostic work; stop without ML.
- A task floor or no-difference result is a completed experiment; do not rescue it.
- One registered validation train only.

VERIFICATION
- targeted protocol/task/replay/BC tests
- full non-render pytest
- rendering checks separately when graphics access permits
- `git diff --check`, config dry runs, manifest/hash verification

FINAL RESPONSE
Lead with one outcome label. Give scripted and replay gates, exact joint/EE validation metrics,
conditional placement decomposition, paired CI, artifacts/report, tests, locked-holdout status,
branch, and commit. State any task-floor limitation prominently.
```
