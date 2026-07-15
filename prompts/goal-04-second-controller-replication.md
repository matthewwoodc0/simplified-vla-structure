# Goal 04 — Second Controller-Integration Replication

**Target agent:** Grok 4.5
**Goal type:** controller integration + pickup and pick-place validation replication
**Locked holdouts:** closed
**Prerequisite:** Goals 01–03 reviewed and merged

Copy the block below into a fresh Grok task. This file is the complete contract.

---

## PASTE INTO GROK

```text
You are the research engineer responsible for testing controller sensitivity in:
/Users/matthewwoodcock/Documents/Simplified VLA Structure

GOAL
Add a second explicit EE controller-integration contract on the same SO-101 and determine whether
the joint-versus-EE learned ordering replicates across controller integration on pickup and learned
pick-place. Preserve the current controller as the default and keep all locked holdouts closed.

This is a controller-integration replication, not a gain rescue and not yet a claim about an
independent IK algorithm.

READ FIRST — AUTHORITY
1. AGENTS.md
2. RESULTS.md
3. researchnotes.md
4. evidence/phase5_causal_synthesis.json
5. evidence/state_bc_efficiency_curve_registration.json
6. the reviewed pick-place replication evidence/report from Goal 03
7. prompts/goal-04-second-controller-replication.md — this full contract
8. src/svla/controller.py
9. src/svla/pickup_task.py — especially `step_ee_delta_action`
10. src/svla/core/action_space.py
11. src/svla/demo_recorder.py
12. src/svla/state_bc.py
13. scripts/validate_controller_quality.py
14. scripts/validate_action_replay.py
15. configs/phase5_evaluation_protocol_v2.json

STARTING STATE
- Start only from reviewed `main` after Goals 01–03.
- Create `codex/second-controller-integration`.
- Verify current pickup and pick-place evidence identities before edits.
- Keep unrelated changes out of the branch.

CONTROLLER CONTRACTS

Controller A — frozen current default:
- ID: `stateless_current_pose_dls_v1`
- every learned EE action creates a bounded target from measured current EE pose + delta
- current DLS solver, posture/null-space behavior, limits, and telemetry
- must remain byte-for-byte/default behavior when no controller ID is supplied

Controller B — new replication:
- ID: `persistent_target_lag_dls_v1`
- integrate learned EE deltas into a persistent Cartesian target
- bound target lag using the existing controller target-lag limit
- use the same DLS solver, posture/null-space behavior, joint limits, step/accel limits,
  tool-axis orientation convention, gripper interface, substeps, and physics gates
- reset persistent target deterministically at episode reset

Do not describe B as a different IK algorithm. The scientific variable is target integration.

WHY THIS IS NOT AN INFERENCE RESCUE
Controller integration changes the executable action contract and may change demonstration
trajectories/labels. Do not load frozen Controller-A models under Controller B and call the result
causal. Regenerate controller-specific demonstrations and train under a preregistered factorial
comparison.

FACTORIAL COMPARISON
- controllers: A and B
- action spaces: `joint_delta`, `ee_tool_delta`
- tasks: pickup and pick-place
- same task specs, object positions, orientations, approaches, placement targets, observations,
  model recipe, seeds, gates, and evaluation denominators within each task
- shared scripted command-target source across controllers
- within each controller/task dataset, both action spaces must come from the exact same realized
  demonstrations

The joint policy may differ between controller datasets because the realized expert trajectory may
differ. Report that explicitly. The controller factor therefore covers integration plus its induced
demonstration distribution; do not call it a solver-only effect.

FROZEN LEARNING RECIPE
- hybrid NN gripper + MLP arm A1
- `global_gripper`, historical match, `legacy_progress_phase`
- `policy_labels`, gain 1.0
- 128 128, 300 epochs, batch 1024, lr 1e-3, wd 1e-5
- seeds 0 1 2 3 4
- no controller-specific hyperparameter changes

MANDATORY EXECUTION ORDER

PHASE A — IMPLEMENTATION AND DEFAULT-COMPATIBILITY GATE
1. Add a small named controller-integration selection boundary. Do not fork the whole task stack.
2. Preserve Controller A as default in every existing script and model path.
3. Add controller ID to demos, models, result rows, configs, and manifests.
4. Prove existing Controller-A deterministic tests/artifacts still reproduce where test fixtures
   support exact comparison.
5. Unit test B reset, target accumulation, target-lag clipping, orientation composition, telemetry,
   and no state leakage across episodes.

PHASE B — CONTROLLER-ONLY GATES FOR B
Run controller-quality, scripted pickup, scripted pick-place, readiness-domain, and visual checks
under B before ML.

Required:
- all scripted pickup and pick-place protocol cells succeed under unchanged strict gates
- readiness-domain gate passes at the same declared standard used for A
- zero numerical controller failures
- target-lag, limit, infeasible, step, and acceleration telemetry is explicit

If B fails, classify it as a controller/task integration failure and stop. Do not tune learning.
One pre-result implementation correction is allowed for a clear bug; no parameter sweep.

PHASE C — CONTROLLER-SPECIFIC LABEL REPLAY
1. Record B demonstrations from the same scripted target/spec sources.
2. Audit label magnitude, direction, and controller telemetry versus A without changing labels.
3. Replay pickup and pick-place labels for both action spaces under B.
4. Require 100% replay success and complete denominators.

If replay fails, stop at the label/controller contract. Do not train or rescale labels.

PHASE D — REGISTER BEFORE LEARNED RESULTS
Write a tracked registration containing:
- exact controller definitions and source hashes
- task protocols and hashes
- dataset/demo hashes per controller
- frozen learning recipe
- model seeds
- metrics, paired keys, and outcome classification
- `locked_holdouts_accessed: false`

PHASE E — VALIDATION-ONLY FACTORIAL TRAIN
Train and evaluate exactly one recipe for every controller x action-space cell:
- complete pickup validation
- complete pick-place validation
- seeds 0-4
- no eval-limit, no retry recipe, no final/locked holdout

If Controller-A results already exist under byte-identical task/recipe/protocol hashes, exact
artifacts may be referenced instead of rerun. If any identity differs, rerun and label the new
comparison; never mix incompatible rows.

PRIMARY ANALYSIS
For each task and controller:
- joint-minus-EE success difference and paired 95% CI
- event order, physical sanity, worst seed, constraint exposure
- for pick-place: valid grasp and conditional placement decomposition

Controller sensitivity:
- difference-in-differences:
  `(joint - EE under B) - (joint - EE under A)`
- report uncertainty and exact paired unit construction
- separately report whether B changes EE success and whether it changes the action-space gap

OUTCOME LABELS
- `ORDERING_REPLICATES_ACROSS_CONTROLLERS`: joint-minus-EE CI is above zero under both A and B
  for the interpretable task(s), with no controller/task floor.
- `CONTROLLER_SENSITIVE_ACTION_SPACE_GAP`: the gap changes materially and the registered
  difference-in-differences CI excludes zero.
- `NO_CLEAR_CONTROLLER_INTERACTION`: controller interaction CI crosses zero with adequate task
  performance.
- `TASK_OR_CONTROLLER_FLOOR`: a task/controller cell is too weak for action-space interpretation.
- `BLOCKED_CONTROLLER_GATE`: B fails controller/task readiness.
- `BLOCKED_LABEL_REPLAY`: B labels are not executable.

Do not force one global label if pickup and pick-place differ. Report per-task outcomes and then a
carefully limited cross-task synthesis.

REQUIRED TESTS
- A remains default and unchanged
- controller ID validation and manifest propagation
- B accumulation/lag/reset boundary behavior
- no cross-episode target leakage
- tool-axis orientation behavior under both integrations
- controller-specific label provenance
- replay gate and denominator checks
- incompatible artifact/hash mixing rejected
- difference-in-differences and paired-CI boundary tests
- old pickup/pick-place/model-loading tests pass
- full non-render suite passes

REQUIRED ARTIFACTS
- named controller-integration config/registry
- tracked factorial registration and durable evidence JSON
- controller quality, scripted, readiness, and replay outputs/manifests
- validation rows/summaries/models for required new cells
- paired task analyses and cross-controller comparison
- visual review for representative B scripted and learned cases if graphics works
- `reports/YYYY-MM-DD-second-controller-replication.md`

DOCUMENTATION
- update `researchnotes.md` registration and outcome
- update `RESULTS.md` with per-controller/per-task comparison
- add an AGENTS.md dated verdict and current next step
- document exact commands and what the replication does and does not prove
- follow the AGENTS.md large-change report template

FORBIDDEN
- changing controller gains/limits after validation begins
- inference-only use of A models under B as the main result
- label rescaling rescue
- controller-specific model hyperparameters
- weakened task/physics/placement gates
- locked pickup efficiency evaluation, pickup final, or pick-place locked holdout
- vision, language, VLA, hardware, or second robot
- calling B an independent IK solver

STOP CONDITIONS
- Controller/task or replay gate failure is completed diagnostic work; do not train.
- A null interaction or failed replication is a completed experiment; do not rescue it.
- One registered learned recipe only.

VERIFICATION
- targeted controller/task/replay/factorial tests
- full non-render pytest
- rendering checks separately when graphics access permits
- `git diff --check`, config dry runs, artifact/hash verification

FINAL RESPONSE
Lead with the applicable outcome label(s). Give Controller-B readiness and replay verdicts, exact
pickup and pick-place joint/EE metrics under A and B, interaction estimate/CI, artifacts/report,
tests, locked-holdout status, branch, and commit. State prominently that this tests controller
integration, not a different IK algorithm or hardware generalization.
```
