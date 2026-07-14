# Agent Notes

Be critical of weak assumptions in this repo. The user wants a thinking partner, not a
rubber stamp. If a plan mixes controller bugs, simulator limitations, and ML failure modes,
separate them before implementing.

## Project Intent

This repo is for a controller-first Robot VLA experiment. The core question is whether a
small VLA or imitation policy learns more efficiently from controller-level actions than
from low-level joint actions.

Keep the order of operations strict:

1. Controller works.
2. Task environment works.
3. Scripted demonstrations work.
4. State-based behavioral cloning works.
5. Vision is added.
6. Language/VLA complexity is added.

Do not jump straight to VLA training. That would hide whether the controller/action-space
idea is actually working.

## Research notes (`researchnotes.md`)

Use **`researchnotes.md`** for hypotheses, experiment design, and in-flight results.
This file (`AGENTS.md`) stays the stable operator summary; update it only when a tested
hypothesis changes a verdict, blocker, or recommended next step.

**Workflow when testing a hypothesis:**

1. Add or pick a row in `researchnotes.md` (ID, status `untested` → `testing`).
2. Run the minimal closed-loop test (rollout + strict gates, not train loss alone).
3. Record outcome in the **Results log** with an `outputs/...` or scratch log path.
4. Set hypothesis status to `confirmed` / `rejected` / `partial`.
5. If the phase verdict moves (e.g. EE BC unblocked, Phase 6b policy training GO), add one
   dated bullet under **Research verdict updates** below.

## Large-change review reports

After every **big or large implementation/change**, generate a Markdown review report for
the user to read in Obsidian. This is required when the work changes architecture, research
verdicts, experiment protocol, data format, task/controller behavior, training/evaluation
logic, or multiple files in a way that future agents need to understand.

Small bug fixes or one-line cleanups do not need a report unless they change a gate,
verdict, artifact contract, or user-facing workflow.

Write reports under `reports/` using:

```text
reports/YYYY-MM-DD-short-change-name.md
```

The report must be detailed enough to audit the change later, but written in plain language
so the user can understand what happened without reading every diff. Link exact files,
commands, outputs, evidence artifacts, and unresolved risks. Be direct about weak evidence,
failed tests, skipped checks, and anything that still needs human review.

If the change creates, modifies, or depends on demo videos, rendered previews, screenshots,
or other visual artifacts, include a dedicated section with openable Markdown links to those
local files. Generated videos are usually ignored by Git; say that clearly and include the
exact command to regenerate them. If embedding video is unreliable in Obsidian, provide a
plain "Open this video:" link or absolute path that the user can click or paste into Finder.

Use this template:

````markdown
# YYYY-MM-DD - Short Change Name

## Plain-English Summary

What changed, why it changed, and what the user should understand first.

## What To Review

- [ ] File or artifact: what changed and why it matters.
- [ ] File or artifact: what changed and why it matters.
- [ ] Evidence output, video, dataset, or report the user should inspect.

## Implementation Details

Describe the important technical changes. Keep this explainable: name the modules,
contracts, data shapes, controller/task behavior, or experiment logic that changed.

## Evidence And Verification

List the commands that were run and their outcomes.

```bash
command that was run
```

- Result:
- Output artifact:
- What this proves:
- What it does not prove:

## Demo Videos / Visual Artifacts

- Open this video or visual artifact: `[artifact-name](</absolute/path/to/artifact.mp4>)`.
- If the artifact is ignored or not committed, include the regeneration command.

## Decisions Made

- Decision:
  Reason:
- Decision:
  Reason:

## Risks And Limitations

- Risk or limitation:
  Why it matters:
- Risk or limitation:
  Why it matters:

## Action Items

- [ ] Next concrete action for the user or future agent.
- [ ] Follow-up test, visual review, experiment, or cleanup.
- [ ] Documentation/researchnotes update if needed.

## Files Changed

- `path/to/file.py` - short reason.
- `path/to/other_file.md` - short reason.

## Current Verdict

State the honest status after the change: ready, blocked, partial, diagnostic only, or
needs review. Do not overclaim from passing tests alone.
````

### Research verdict updates

- **2026-07-01:** Historical legacy-grid EE event misordering documented (15/72 vs joint
  47/72); this was superseded by the registered protocol-v2 result below.
- **2026-07-01:** **H-EE-010 rejected** — inference-only no-cursor ablation made EE **worse**
  (0/72 success, 24 early_close vs 14/72 baseline). Cursor/progress is required for the
  *current* MLP weights; fix needs retrain or env-derived phase, not rollout ablation alone.
  Evidence: `outputs/h_ee_010_no_cursor_ablation.json`. `state_bc.py` changes reverted.
- **2026-07-02:** Protocol-v2 validation selected one shared `legacy_progress_phase` contract:
  joint 53/120 and EE 31/120, versus cursor-free joint 18/120 and EE 32/120. **H-EE-012
  rejected.** Evidence: `evidence/phase5_v2_model_selection.json`.
- **2026-07-02:** Registered raw final remains below the proposed policy gates: joint 51/120
  and EE 28/120. Event order, physical sanity, hard-limit exposure, and seed instability all
  remain material; this is not a timing-only result.
- **2026-07-02:** The separately labeled distance-guard diagnostic left EE unchanged at
  28/120 and moved joint only from 51/120 to 55/120. **H-EE-011 and H-JNT-001 rejected.**
  Evidence: `evidence/phase5_v2_final_results.json`.
- **2026-07-08:** **Phase 6a vision infrastructure implemented** — fixed-camera RGB
  capture, scripted pickup RGB dataset export, manifest validation, and MP4 preview
  plumbing are present. This does not change the Phase 5 evidence ladder and does not open
  Phase 6b policy/VLA work.
- **2026-07-08:** **H-EE-013 rejected** — env-derived phase (train+rollout, same protocol-v2
  validation contract as H-EE-012) made EE worse (19/120 vs legacy 31/120) and joint worse
  (23/120 vs 53/120). Early-close and reopen rose sharply; do not access final under this
  contract. Evidence: `outputs/h_ee_013_env_phase_validation/state_bc_summary.json`.
- **2026-07-08:** **H-EE-008 confirmed on validation** — gripper-weighted MSE (5× gripper,
  10× on grasp_align/close_gripper) under legacy temporal contract: EE 50/120 (+19 vs 31),
  event-order 55/120 (+17); joint 84/120 (+31 vs 53). Preclose EE contact 583→50. Seed
  instability remains (EE per-seed 22,9,11,2,6). Final not accessed; learned-policy release
  gates still not met. Evidence: `outputs/h_ee_008_gripper_weighted_validation/`.
- **2026-07-09:** **H-EE-003 rejected** — a separate binary gripper classifier (same
  weighted legacy protocol-v2 validation contract) regressed EE to 34/120 and joint to
  41/120. EE early-close rose 5→29; the lower preclose-contact and constraint exposure did
  not improve raw event order or seed reliability. Final not accessed. Evidence:
  `outputs/h_ee_003_separate_gripper_head_validation/h_ee_003_comparison.json`.
- **2026-07-09:** **H-EE-021 confirmed** — H-EE-008 gain is mostly **global gripper 5×**
  (EE 31→49), not transition 10× (EE 38). Combined EE 50 ≈ global; joint still wants
  combined (84 vs global 76). Residual under all profiles is **reopen/gripper flips**,
  not early-close. No EE profile meets frontier; final closed. Next: H-EE-014 (NN
  gripper + MLP arm). Evidence: `outputs/h_ee_021_loss_decomposition/`.
- **2026-07-09:** **H-EE-014 confirmed on validation** — hybrid NN gripper + MLP arm
  under `global_gripper` (A1 compositor): EE 49→**62**/120, EO 60→**79**, reopen
  **155→0**, worst seed 4→**9**/24; joint 76→**97**/120. All pre-registered pass bars
  met. Residual EO is **missing_lift + early_close** (not reopen; flips=1.0). Still
  short of research parity frontier; final closed. Evidence:
  `outputs/h_ee_014_nn_gripper_global_validation/`.
- **2026-07-09:** **Post-H-EE-014 residual program complete (SP0–SP3)** — SP0 visual freeze
  done. **H-EE-022 rejected** (match_relative_ee early_close 11→11). **H-EE-023 rejected**
  (A2 arm-only EE 67/120, missing_lift worse, worst seed 6). **H-EE-024 diagnosed**
  (impulse-dominant almost-wins; no train yet). Best EE remains hybrid A1 **62/120**;
  final closed; Phase 6b blocked. Scoreboard:
  `outputs/post_h_ee_014_residual_scoreboard.json`. Report:
  `reports/2026-07-09-post-h-ee-014-residual-progress.md`.
- **2026-07-13:** **H-EE-007 rejected at the replay gate** — raw observed EE transitions
  were ~30× smaller than executable policy labels and replayed at **0/18** success / **0/18**
  event order, versus policy-label control **18/18**. Gripper labels matched exactly; no
  training, validation, or final access occurred. Evidence:
  `outputs/h_ee_007_label_contract_probe/`.
- **2026-07-13:** **H-EE-002 rejected on validation** — byte-identical frozen H-EE-014
  models reproduced gain 1.0 exactly at **62/120**, then collapsed to **5/120** at 0.875
  and **0/120** at 0.750. Failure-conditioned joint-limit/infeasible exposure declined,
  but no gain recovered a paired missing-lift success; 57/62 baseline successes were lost
  and early-close rose 11→25→48. Do not lower gain further or run an unregistered cap rescue.
  Final remains closed. Evidence: `outputs/h_ee_002_hybrid_gain_sweep/` and
  `evidence/h_ee_002_hybrid_gain_sweep.json`.
- **2026-07-14:** **H-EE-015 `negative_arm_ceiling`** — frozen hybrid A1 EE arm + fixed
  oracle gripper FSM (privileged scripted grasp-target thresholds) regressed EE success
  **62→47**/120, EO 79→77, phys 68→**56**, worst seed 9→**5**, missing-lift EO 30→**42**.
  FSM always transitioned (0 never-transitioned); early_close/reopen 0 by construction and
  must not be credited as learned gains. Paired +5 recoveries / −20 regressions. Oracle
  diagnostic only — not learned-policy performance; does not replace raw EE-vs-joint.
  Best EE remains hybrid A1 **62/120**. Stop treating gripper logic as the primary remaining
  fix. Evidence: `outputs/h_ee_015_fsm_upper_bound/`. Report:
  `reports/2026-07-14-h-ee-015-fsm-upper-bound.md`.
- **2026-07-14:** **Phase 5 causal synthesis frozen** — pickup rescue mainline **closed**.
  Decisive probes: H-EE-007 (raw labels 0/18 vs policy 18/18), H-EE-002 (gain 1.0→0.875→0.750
  = 62→5→0/120), H-EE-015 (oracle FSM 47 vs hybrid 62). Fair contract freeze: hybrid A1 +
  `global_gripper` + historical match + `legacy_progress_phase` + `policy_labels` + gain 1.0
  + seeds 0–4 (validation family joint **97**/EE **62**; raw final still joint **51**/EE
  **28**). Next program: demonstration efficiency → learned pick-place →
  controller-integration replication. Do not claim EE is universally worse. Final not
  accessed; Phase 6b not started. Evidence: `evidence/phase5_causal_synthesis.json`. Report:
  `reports/2026-07-14-phase5-causal-synthesis.md`.

## YOU ARE HERE

**Integration target:** `main` contains the reviewed Phase 5 causal synthesis freeze plus
the audited Phase 6a infrastructure and post-H-EE-014 residual evidence. Create a focused
research branch for each later experiment; do not run new hypotheses directly on `main`.

**Current phase:** Phase 5 pickup **rescue program closed** (synthesis freeze). Phase 6a
vision infrastructure remains plumbing-only. The learned-policy comparison uses the frozen
fair hybrid contract for the **next comparative program** (efficiency, pick-place,
controller-integration replication). The final holdout remains **closed**, and Phase 6b
vision-conditioned policy/VLA work is **not started**.

Phases 1–5 plus Phase 6a infrastructure are built: MuJoCo SO-101 arm, damped-least-squares
IK controller, action-space adapters, table/cube pickup task, scripted demonstrations,
state-based BC comparison, and fixed-camera RGB dataset plumbing.
A pre-Phase-6 physics audit fixed contact model bugs, added force/impulse/disturbance
telemetry and conservative success gates, stress-tested geometry/friction, and retrained BC
under corrected physics.

Vision-only infrastructure is now present as data/render/validation plumbing. Do not start
vision-conditioned policy training or VLA work until the research comparison action spaces
are in an acceptable state or the scope is explicitly changed (see verdict).

## Current Implementation State

Core modules:

- `assets/so101_arm.xml`, `assets/pickup_scene.xml`: SO-101 arm + table/cube scene.
- `src/svla/controller.py`: damped-least-squares Cartesian IK controller.
- `src/svla/action_spaces.py`: joint-delta and EE-tool-delta adapters.
- `src/svla/pickup_task.py`: pickup + pick-place evaluator with physics-audit telemetry and gates.
- `src/svla/pick_place_replay.py`: action replay with grasp-segment boundary from demo metadata.
- `src/svla/state_bc.py`, `src/svla/demo_recorder.py`: demonstration recording and BC.
- `src/svla/vision_observations.py`, `src/svla/vision_dataset.py`: fixed-camera RGB
  observation capture, compact NPZ frame datasets, and manifest validation.
- `scripts/validate_task_robustness.py`: readiness vs broad domain stress tests.
- `scripts/run_pick_place_trials.py`, `scripts/record_pick_place_demo.py`: pick-place matrix and demo export.
- `scripts/render_pickup_showcase.py`, `scripts/render_bc_rollout.py`: MP4 visual review.
- `scripts/record_pickup_vision_demos.py`, `scripts/validate_vision_dataset.py`,
  `scripts/render_vision_dataset_preview.py`: Phase 6a dataset, validation, and preview tools.

Pickup success requires all of: collision-free approach, valid event order, physical sanity
(force/impulse/disturbance limits), contact, lift, and retention — not geometry alone.
Pick-place adds transport → lower → open → retreat and placement XY/Z tolerance on top of
the same grasp-segment gates.

### Physics-audit gate constants (MuJoCo sanity limits, not hardware-calibrated)

| Gate | Value |
|------|-------|
| `MAX_GRIPPER_CONTACT_FORCE` | 22.0 N |
| `MAX_GRIPPER_IMPULSE_BEFORE_LIFT` | 9.0 N·s |
| `MAX_SUPPORTED_XY_DISPLACEMENT` | 0.013 m |
| `MAX_SUPPORTED_ROTATION` | 0.30 rad |

Contact fixes: valid finger-pad `solimp`, jaw `forcerange="-0.20 0.20"`, table-relative
grasp height, asymmetric width compensation (shrink gain 3.0, growth gain 1.0).

## Physics-Audit Lessons Learned

1. Boolean contact + lift success is insufficient; force/impulse/disturbance telemetry is required.
2. Invalid MuJoCo contact params (`solimp="2 1 0.01"`) and uncapped jaw torque can produce
   kN-scale forces while trajectories look geometrically correct.
3. Metrics can be satisfied through unintended trajectories (shove-then-recover, late contact
   during lift phase for smaller objects).
4. TCP/grasp calibration is domain-specific: fixed TCP works only inside a declared envelope.
5. Asymmetric pivoting jaw causes ~13–20 mm supported translation during closure; lateral
   compensation is direction-dependent (shrink vs growth gains differ).
6. Numerical success ≠ visually/physically acceptable behavior.
7. Current action replay retains controller saturation: pickup EE 7.76% vs joint 0%;
   pick-place EE 8.49% vs joint 0.09%.
8. Registered raw learned policies remain poor: EE 28/120 (23.3%), joint 51/120 (42.5%).
9. Separate "Phase-6 readiness domain" (±5% geometry, friction 1.6–2.0) from "broad stress
   report" (±15%, friction 0.8–2.4). Do not relabel broad failures as passes.
10. Controller-level scripting composes cleanly (pickup 36/36, pick-place 6/6), but learned
    failures span event order, physical sanity, controller-constraint exposure, and seed
    instability. Zero numerical controller failures does not make saturation irrelevant.

## End of Phase 5: Evidence Ladder (strict physics gates)

Separate **scripting**, **label replay**, **raw learned BC**, and **shielded diagnostics**.
They answer different questions and must not be collapsed into one readiness claim.

| Layer | What it measures | EE (`ee_tool_delta`) | Joint (`joint_delta`) |
|-------|------------------|----------------------|------------------------|
| 1. Scripted pickup expert | One shared expert trajectory source, before policy adaptation | 36/36 shared scripted trials | 36/36 shared scripted trials |
| 2. Pickup policy-label replay | Executability of recorded action-space labels | 18/18 | 18/18 |
| 3. Raw learned MLP BC | Protocol-v2 final, 5 seeds × 24 trials | 28/120 (23.3%) | 51/120 (42.5%) |
| 4. Distance-guard diagnostic | Same byte-identical models with a shield; not raw BC | 28/120 (23.3%) | 55/120 (45.8%) |

Current learned evidence: `evidence/phase5_v2_model_selection.json`,
`evidence/phase5_v2_final_results.json`, and the manifests referenced there. Source-matched
scripted/replay evidence is
`outputs/phase5_baseline_final/phase5_baseline_v2_aggregate.json`.

**Interpretation:** scripting and replay rule out a basic task-infeasibility or
non-executable-label failure. They do not prove that learned rollouts are free of controller
constraint interactions. Both policies fail the proposed learned-policy gates; EE remains
worse, but timing is not established as the sole cause.

### Learned-policy failure breakdown (final eval, pickup)

| Metric | Raw EE BC | Raw joint BC |
|--------|-------|----------|
| Success | 28/120 (23.3%) | 51/120 (42.5%) |
| `event_order_valid` | 38/120 (31.7%) | 65/120 (54.2%) |
| `physical_sanity_pass` | 68/120 (56.7%) | 80/120 (66.7%) |
| `early_close` trials | 2 | 11 |
| `preclose_contact_steps` | 741 | 0 |
| `reopen_events` | 196 | 128 |
| `controller_failure_steps` | 0 | 0 |
| Per-seed successes | 9, 2, 3, 6, 8 | 8, 14, 2, 12, 15 |

The 4001+ grids under `state_bc_grasp_tcp_final` and `state_bc_physics_audit_final` are
historical. The current registered final uses protocol-v2 trial IDs 7001+ and five seeds.

### Pick-and-place extension (scripted validation only)

Post-lift phases (transport → lower → open → retreat) extend pickup without new controller
primitives. The scripted matrix is **6/6**, and six demos carry the
`svla_pick_place_demo_v1` label contract plus the grasp-boundary metadata. The latest
source-matched baseline replayed all six demos successfully in both action spaces. **No
pick-place BC yet.**

Left placement currently uses separate goal and command markers to compensate asymmetric-jaw
transport slip. A nominal regression test fails when the goal marker is used directly, but
there is no artifact supporting a general “50%” ablation claim; robustness beyond the
scripted matrix is unknown.

## Declared Domains

### Phase-6 readiness envelope (gate domain)

- Object size scale: 0.95–1.05 per axis (±5%)
- Sliding friction: 1.6–2.0
- Scripted expert must pass here before Phase 6 policy work.

### Broad stress boundary (diagnostic only)

- Object size scale: 0.85–1.15 per axis (±15%)
- Sliding friction: 0.8–2.4
- Failures here are expected OOD signal, not gate failures.

## Required Pre-Phase-6 Checks

Run before advancing past the physics-audit gate:

1. **Contact force/impulse gates** — `max_gripper_contact_force`, `gripper_contact_impulse_before_lift`.
2. **Event order** — close start → finger contact → object unsupported → lift clearance;
   no pre-close contact, no reopen, and close begins within 15 mm of the object.
3. **Disturbance limits** — supported-object XY displacement and rotation while grasped.
4. **Randomized geometry/friction** — readiness domain stress (`validate_task_robustness.py --domain readiness`).
5. **Visual review** — render representative scripted and BC cases; confirm overlays match gate telemetry.

```bash
PYTHONPATH=src .venv/bin/python scripts/validate_task_robustness.py --domain readiness \
  --output outputs/task_robustness_readiness_summary.json

PYTHONPATH=src .venv/bin/python scripts/render_pickup_showcase.py \
  --output outputs/pickup_showcase_physics_audit.mp4
```

Readiness robustness is a **manual release gate**, not part of default `pytest` CI. The
readiness suite runs 288 physics simulations (~minutes) and belongs in pre-Phase-6 /
pre-release checklists. Unit tests in `tests/test_pickup_task.py` cover gate logic on
synthetic metrics; they do not replace domain stress.

## Known Limitations

- **Controller constraints are distinct from numerical failure:** raw final has zero
  `controller_failure_steps`, but joint-limit exposure is 21.3% of EE rollout steps and
  19.7% of joint rollout steps. Do not use zero numerical failures to dismiss saturation.
- **Asymmetric jaw:** pivoting gripper causes direction-dependent lateral shift during closure;
  compensation gains are tuned for the readiness envelope only.
- **Simulation-only force gates:** limits are MuJoCo sanity checks, not hardware-calibrated.
- **BC eval grids:** protocol-v2 final uses trial IDs 7001+; 4001+ clips and artifacts are
  historical legacy-grid examples.

## Historical Visual Review Findings (2026-07-01 physics audit)

These clips align with their historical physics-audit rows, but they are not visual evidence
for the current protocol-v2 models. New v2 visual review remains separate future work.

**BC cross-check rule:** eval jsonl rows are keyed by `(trial_id, seed)`. Always match
`models/*_seed_N.npz` to the jsonl row with the same `seed`. Trial 4005 illustrates why:
seed 0 → `contact_dynamics_failure` (early_close=false, reopen=0); seed 1 →
`event_order_failure` with early_close=true and reopen_events=2 (the rendered exemplar).

| Clip | Path | Render stdout | Eval jsonl (matched seed) | Gate alignment |
|------|------|---------------|---------------------------|----------------|
| Scripted baseline yaw_0 center | `outputs/pickup_showcase_physics_audit.mp4` (trials 7, 8) | success=1, contact=1, lift≈0.031–0.032 m, hold=1 | n/a (scripted) | Matches 36/36 scripted audit; overlays ORDER=1, force <22 N |
| Readiness edge (uniform_small 0.95, μ=1.6) | `outputs/scripted_readiness_edge.mp4` (trial 7) | success=1, contact=1, lift≈0.032 m, hold=1 | n/a | Matches readiness 288/288 at envelope boundary |
| Broad OOD failure (uniform_small 0.85, yaw_-18 right) | `outputs/scripted_broad_failure_uniform_small.mp4` (trial 5) | success=0, contact=1, lift≈0.029 m, hold=1 | `task_robustness_broad_summary.json` trial 5 | Lifts visually; `collision_free_approach` / `event_order_valid` fail — correctly gated |
| Joint BC success | `outputs/joint_bc_success.mp4` | success=1 | seed 1, trial 4002: success=true, event_order_valid=true, physical_sanity_pass=true | Clean approach-close-lift |
| Joint BC failure | `outputs/joint_bc_failure.mp4` | success=0 | seed 1, trial 4001: success=false, reopen_events=3, event_order_valid=false, failure_category=event_order_failure | Long rollout; gripper reopens before valid sequence |
| EE BC success | `outputs/ee_bc_success.mp4` | success=1 | seed 1, trial 4016: success=true, event_order_valid=true, physical_sanity_pass=true | Works on easy yaw_0 pose only |
| EE BC failure (early-close / reopen) | `outputs/ee_bc_failure.mp4` | success=0 | seed 1, trial 4005: success=false, **early_close=true**, reopen_events=2, event_order_valid=false, failure_category=event_order_failure | Policy closes early, reopens, violates event order; not seed-0 contact_dynamics row |

## Readiness Verdict by Layer

Learned-policy evidence is frozen in `evidence/phase5_v2_model_selection.json` and
`evidence/phase5_v2_final_results.json`. Use
`outputs/phase5_baseline_final/phase5_baseline_v2_aggregate.json` for
scripted/replay/readiness claims.

| Decision | Verdict |
|----------|---------|
| Scripted simulator/task readiness | **READY in the declared MuJoCo envelope** — pickup 36/36, pick-place 6/6, readiness 288/288 |
| Policy-label replay | **READY as an executability check** — pickup 18/18 per action space and pick-place 6/6 per action space; not learned-policy evidence |
| Raw learned-policy comparison | **NOT READY** — joint 51/120, EE 28/120; both fail the proposed success, event-order, physical-sanity, hard-limit, and per-seed gates |
| Shielded distance-guard diagnostic | **REJECTED as a fix** — EE unchanged; joint +4/120. Shielded numbers never replace the raw ladder |
| Hardware realism | **NOT ASSESSED** — force/impulse thresholds are MuJoCo sanity limits, not calibrated hardware limits |
| Phase 6a vision infrastructure | **IMPLEMENTED AS PLUMBING ONLY** — fixed-camera RGB capture, scripted dataset export, validation, and preview are available |
| Phase 6b vision-conditioned BC / VLA | **BLOCKED** until the action-space comparison is viable or scope is explicitly changed |

**Controller vs simulator vs ML:** kN forces were a **simulator/contact-model bug** (invalid
`solimp`, uncapped jaw). Broad-domain shove-and-recover failures are **task/controller
envelope** limits. Current BC failures occur in learned closed loop and include event-order,
physical-sanity, and constraint-exposure symptoms; the evidence does not isolate one cause.

## Commands

Use the existing venv:

```bash
source .venv/bin/activate
pytest
bash scripts/run_mujoco_gui.sh
PYTHONPATH=src .venv/bin/python scripts/run_pickup_trials.py
PYTHONPATH=src .venv/bin/python scripts/validate_task_robustness.py --domain readiness
PYTHONPATH=src .venv/bin/python scripts/train_state_bc.py --output-dir outputs/state_bc
```

For ad hoc Python commands from the repo without installing the package, use:

```bash
PYTHONPATH=src .venv/bin/python ...
```

The current venv has `mujoco`, `numpy`, `scipy`, and `pytest`. Editable install previously
failed because `setuptools` was not installed and the sandbox had no PyPI network access.
Do not make editable install a required path unless you also fix that dependency issue.
Do not tell the user to run `mjpython` from this venv. `.venv/bin/mjpython` has a broken
shebang because the repo path contains spaces. Use `bash scripts/run_mujoco_gui.sh`, which
runs `.venv/bin/python .venv/bin/mjpython scripts/open_mujoco_gui.py`.
`ffmpeg` is available at `/opt/homebrew/bin/ffmpeg` in the current environment and is used
for MP4 export.

## MuJoCo Gotchas

- MuJoCo Python exposes site orientation through `data.site_xmat`, not `data.site_xquat`.
- Use `scipy.spatial.transform.Rotation` to convert the site matrix to quaternion/rotvec.
- Keep position-only reaching separate from full pose reaching. Full pose IK is stricter and
  can make simple reach tests look broken.
- The controller should report clipping and infeasibility instead of silently masking them.
- Contact `solimp` first value must be ≥1; values like `2 1 0.01` are invalid and produce
  non-physical forces.

## Design Constraints

- MuJoCo first on the local Mac.
- Isaac Sim later only when Linux/NVIDIA hardware or cloud GPU is available and justified.
- Unity ML-Agents may be a secondary learnability benchmark, not the main VLA/controller stack.
- Prefer small, inspectable Python modules over a large framework.
- Do not overclaim results from reaching tests. Reaching is only the controller smoke test.
- Do not jump to VLA/vision training as a workaround for grasp physics failures.

## Next Useful Work

**Pickup rescue mainline is closed.** Do not default to more gripper/gain/FSM/match/loss
tuning. Durable freeze: `evidence/phase5_causal_synthesis.json`.

### Ordered next research program (state-based, frozen fair contract)

1. **Demonstration efficiency** — preregister nested/stratified demo-count curve for both
   action spaces under the frozen hybrid A1 + `global_gripper` contract.
2. **Learned pick-place BC** — second manipulation task for **both** action spaces under the
   same contract (scripted/replay plumbing already exists).
3. **Controller-integration replication** — Controller A (stateless
   current-measured-pose-plus-delta DLS; current learned EE rollout) vs Controller B
   (persistent-target-lag DLS, same underlying DLS solver). Not an independent IK
   algorithm. Require identical task specs, observation schema, evaluation trials, and
   gates; controller-specific executable demos/labels with exact joint/EE demo parity
   within each controller; do **not** require byte-identical realized demos across
   controllers.

### Frozen fair contract (do not silently change)

- Action spaces: `joint_delta`, `ee_tool_delta`
- Policy: hybrid NN gripper + MLP arm, A1 compositor
- Loss: `global_gripper`; NN match: historical; temporal: `legacy_progress_phase`
- Labels: `policy_labels`; gain 1.0; hidden 128×128; 300 epochs; batch 1024; lr 0.001;
  weight decay 1e-5; seeds 0–4
- Strict event-order and physical-sanity gates unchanged
- Does **not** authorize final access, Phase 6b, or deployment claims

### Still closed / blocked / optional only

- Final holdout: **closed**
- Phase 6b vision-conditioned BC / language / VLA: **blocked / not started**
- H-EE-024/SP3 impulse train and H-EE-017 history: **optional mechanism backlog only**
- Do not re-run H-EE-003/007/010/011/012/013/015/002/022/023 or H-JNT-001 as defaults
- Do not rescale raw labels, lower EE gain further, add a cap rescue, or retune the FSM

### Phase 6a vision infrastructure (plumbing only)

- Keep fixed camera observations and scripted RGB datasets in the readiness domain only.
- Do not store large RGB arrays in JSON demo rows; keep NPZ frames plus manifests.
- Do not start vision-conditioned policy training as a residual workaround.

When evaluating policies, keep observations, demonstrations, task initialization, and
success metrics identical across action spaces. Otherwise the result will not answer the
research question.
