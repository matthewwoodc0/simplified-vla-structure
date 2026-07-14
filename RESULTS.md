# Results

This file is the concise index of research outcomes. It separates scripted feasibility,
label executability, raw learned policies, validation-only follow-ups, and infrastructure.
The detailed hypotheses and in-flight reasoning remain in `researchnotes.md`.

## Evidence Ladder

| Layer | Joint delta | EE tool delta | Interpretation |
|---|---:|---:|---|
| Scripted pickup expert | 36/36 | 36/36 shared source | Task/controller feasible in the declared envelope |
| Pickup policy-label replay | 18/18 | 18/18 | Recorded executable labels replay correctly |
| Raw MLP BC, protocol-v2 final | 51/120 | 28/120 | Both fail proposed learned-policy gates |
| Distance-guard diagnostic | 55/120 | 28/120 | Rejected as a fix; shielded evidence is not raw BC |
| Best hybrid validation (H-EE-014) | 97/120 | 62/120 | Strong validation gain; final still closed |

Primary tracked records:

- `evidence/phase5_v2_model_selection.json`
- `evidence/phase5_v2_final_results.json`
- `evidence/phase5_causal_synthesis.json` — durable rescue-chapter freeze (2026-07-14)
- `outputs/phase5_baseline_final/phase5_baseline_v2_aggregate.json`
- `outputs/h_ee_014_nn_gripper_global_validation/`

## Confirmed or Useful Results

- H-EE-008: gripper-weighted MSE improved validation from joint 53→84 and EE 31→50.
- H-EE-021: the EE gain came mainly from global gripper 5× weighting (31→49), not the
  transition-only 10× intervention (31→38). Joint performed best with the combined profile.
- H-EE-014: replacing only the gripper output with state-local nearest-neighbor retrieval
  improved EE 49→62, event order 60→79, worst seed 4→9, and reopen 155→0. Joint improved
  76→97. This isolates gripper sequencing as important, but does not solve EE lift/control.
- Phase 6a: deterministic fixed-camera RGB capture, NPZ frame datasets, hash validation,
  and preview rendering are implemented without training a vision policy.

## Negative and Inconclusive Results

- H-EE-003 separate gripper classifier: rejected (EE 34/120, joint 41/120).
- H-EE-010 inference-only cursor removal: rejected (EE 0/72).
- H-EE-012 cursor-free retraining: rejected (EE 32/120, joint 18/120).
- H-EE-013 env-derived phase: rejected (EE 19/120, joint 23/120).
- H-EE-011 / H-JNT-001 distance guard: rejected as readiness fixes.
- H-EE-022 match-relative NN retrieval: rejected; early close did not improve.
- H-EE-023 arm-only MLP under NN gripper: rejected; EE 67/120 did not meet the bar and
  missing-lift failures worsened.
- H-EE-024 impulse-dominant failures are diagnosed; train path is optional mechanism backlog
  only (not mainline).
- H-EE-007 raw observed EE labels: rejected at replay (0/18 success and 0/18 event order,
  versus executable-label control 18/18). The raw transition vectors are not command-scale
  executable labels, so no seed screen or validation training was run. This does **not**
  prove reconstructed `policy_labels` are optimal.
- H-EE-002 frozen hybrid gain sweep: rejected. Gain 1.0 reproduced EE 62/120 exactly;
  0.875 collapsed to 5/120 and 0.750 to 0/120. Lower failure-conditioned constraint
  exposure recovered zero paired missing-lift successes and lost 57/62 baseline successes.
  Rules out simple monotonic gain rescue; does **not** prove controllers are irrelevant.
  Evidence: `evidence/h_ee_002_hybrid_gain_sweep.json`.
- H-EE-015 oracle FSM arm diagnostic: **`negative_arm_ceiling`** (oracle diagnostic, **not**
  learned-policy performance, **not** a literal arm upper bound). Frozen hybrid A1 arm +
  privileged latched gripper FSM: success 62→**47**/120, EO 79→77, phys 68→**56**, worst
  seed 9→**5**, missing-lift EO 30→**42**, never-transitioned 0, paired recoveries 5 /
  regressions 20. Early-close and reopen are zero by construction and must not be credited
  as policy gains. Evidence: `outputs/h_ee_015_fsm_upper_bound/`,
  `evidence/h_ee_015_fsm_upper_bound.json`,
  `reports/2026-07-14-h-ee-015-fsm-upper-bound.md`.

## Pickup rescue chapter (closed 2026-07-14)

**Status:** `rescue_program_status: closed` — see `evidence/phase5_causal_synthesis.json`
and `reports/2026-07-14-phase5-causal-synthesis.md`.

| Decision | Meaning |
|----------|---------|
| End default gripper/gain/FSM/match/loss rescue tuning | Decisive probes closed the mainline residual rescue queue |
| Freeze fair comparison on hybrid A1 + `global_gripper` | Strongest symmetric validation family; not “EE ready” |
| Do not claim EE is universally worse than joint | One sim, one task family, one controller, one BC family |
| Final holdout | Still closed |
| Phase 6b | Still not started |

**Frozen fair contract (next comparative work):** hybrid NN gripper + MLP arm A1,
`global_gripper`, historical NN match, `legacy_progress_phase`, `policy_labels`,
hidden 128×128, 300 epochs, batch 1024, lr 0.001, weight decay 1e-5, action gain 1.0,
seeds 0–4, strict event-order and physical-sanity gates unchanged.

## Experiment Matrix

| Representation | Pickup scripted/replay | Pickup state BC | Pick-place | Vision | Language |
|---|---|---|---|---|---|
| `joint_delta` | complete | raw final + hybrid validation | scripted/replay only | data labels only | untested |
| `ee_tool_delta` | complete | raw final + hybrid validation | scripted/replay only | data labels only | untested |
| full `ee_delta` | labels/controller smoke | no fair BC comparison | labels only | data labels only | untested |
| discrete/tokenized actions | untested | untested | untested | untested | untested |
| chunked/latent actions | untested | untested | untested | untested | untested |

The highest-value open cells are **not** “train a VLA” and **not** more pickup
gripper/gain rescue. Ordered next program under the frozen fair contract:

1. **Demonstration efficiency** — preregistered nested/stratified demo-count curve, both spaces.
2. **Learned pick-place BC** — second manipulation task for both action spaces (scripted/replay
   already exist); not a joint-only detour framed as residual rescue.
3. **Controller-integration replication** — Controller A (stateless
   current-measured-pose-plus-delta DLS; current learned EE rollout) vs Controller B
   (persistent-target-lag DLS, same underlying DLS solver). Not an independent IK
   algorithm. Require identical task specs, observation schema, evaluation trials, and
   gates; controller-specific executable demos/labels with exact joint/EE demo parity
   within each controller; do **not** require byte-identical realized demos across
   controllers.

Optional mechanism backlog only: H-EE-024/SP3 impulse path if registered; H-EE-017 history
only with a careful non-Markov arm argument. Phase 6b remains blocked. Best fair validation
EE remains hybrid A1 **62/120** (joint **97/120**); raw final remains joint **51**/EE **28**.
