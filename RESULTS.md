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
- H-EE-024 impulse-dominant failures are diagnosed but not yet a trained intervention.
- H-EE-007 raw observed EE labels: rejected at replay (0/18 success and 0/18 event order,
  versus executable-label control 18/18). The raw transition vectors are not command-scale
  executable labels, so no seed screen or validation training was run.
- H-EE-002 frozen hybrid gain sweep: rejected. Gain 1.0 reproduced EE 62/120 exactly;
  0.875 collapsed to 5/120 and 0.750 to 0/120. Lower failure-conditioned constraint
  exposure recovered zero paired missing-lift successes and lost 57/62 baseline successes.
  Evidence: `evidence/h_ee_002_hybrid_gain_sweep.json`.
- H-EE-015 oracle FSM arm upper bound: **`negative_arm_ceiling`** (oracle diagnostic, not
  learned-policy performance). Frozen hybrid A1 arm + privileged latched gripper FSM:
  success 62→**47**/120, EO 79→77, phys 68→**56**, worst seed 9→**5**, missing-lift EO
  30→**42**, never-transitioned 0, paired recoveries 5 / regressions 20. Early-close and
  reopen are zero by construction and must not be credited as policy gains. Evidence:
  `outputs/h_ee_015_fsm_upper_bound/`, `reports/2026-07-14-h-ee-015-fsm-upper-bound.md`.

## Experiment Matrix

| Representation | Pickup scripted/replay | Pickup state BC | Pick-place | Vision | Language |
|---|---|---|---|---|---|
| `joint_delta` | complete | raw final + hybrid validation | scripted/replay only | data labels only | untested |
| `ee_tool_delta` | complete | raw final + hybrid validation | scripted/replay only | data labels only | untested |
| full `ee_delta` | labels/controller smoke | no fair BC comparison | labels only | data labels only | untested |
| discrete/tokenized actions | untested | untested | untested | untested | untested |
| chunked/latent actions | untested | untested | untested | untested | untested |

The highest-value open cells are not “train a VLA.” They are closed-loop residual fixes for
EE pickup arm/path/force (H-EE-015 showed gripper oracle does not raise the arm ceiling),
optional H-EE-017 only if a non-Markov arm residual is justified, followed by a
deliberately scoped joint-only pick-place BC benchmark if the team accepts that detour.
Phase 6b remains blocked. Best learned EE remains hybrid A1 62/120.
