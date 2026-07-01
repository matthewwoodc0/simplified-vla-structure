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
This file (`Agents.md`) stays the stable operator summary; update it only when a tested
hypothesis changes a verdict, blocker, or recommended next step.

**Workflow when testing a hypothesis:**

1. Add or pick a row in `researchnotes.md` (ID, status `untested` → `testing`).
2. Run the minimal closed-loop test (rollout + strict gates, not train loss alone).
3. Record outcome in the **Results log** with an `outputs/...` or scratch log path.
4. Set hypothesis status to `confirmed` / `rejected` / `partial`.
5. If the phase verdict moves (e.g. EE BC unblocked, Phase 6b policy training GO), add one
   dated bullet under **Research verdict updates** below.

### Research verdict updates

- **2026-07-01:** EE BC event misordering documented as primary ML blocker (15/72 vs joint
  47/72); hypotheses logged in `researchnotes.md`.
- **2026-07-01:** **H-EE-010 rejected** — inference-only no-cursor ablation made EE **worse**
  (0/72 success, 24 early_close vs 14/72 baseline). Cursor/progress is required for the
  *current* MLP weights; fix needs retrain or env-derived phase, not rollout ablation alone.
  Evidence: `outputs/h_ee_010_no_cursor_ablation.json`. `state_bc.py` changes reverted.

## YOU ARE HERE

**Current phase:** End of Phase 5 / pre-Phase-6 physics-audit gate (gate **closed** as of
2026-07-01).

Phases 1–5 are built: MuJoCo SO-101 arm, damped-least-squares IK controller, action-space
adapters, table/cube pickup task, scripted demonstrations, and state-based BC comparison.
A pre-Phase-6 physics audit fixed contact model bugs, added force/impulse/disturbance
telemetry and conservative success gates, stress-tested geometry/friction, and retrained BC
under corrected physics.

**Phase 6 (vision)** may proceed for vision-only infrastructure (camera observations,
rendering pipeline, dataset format). Do not start vision-conditioned policy training or VLA
work until the research comparison action spaces are in an acceptable state (see verdict).

## Current Implementation State

Core modules:

- `assets/so101_arm.xml`, `assets/pickup_scene.xml`: SO-101 arm + table/cube scene.
- `src/svla/controller.py`: damped-least-squares Cartesian IK controller.
- `src/svla/action_spaces.py`: joint-delta and EE-tool-delta adapters.
- `src/svla/pickup_task.py`: pickup + pick-place evaluator with physics-audit telemetry and gates.
- `src/svla/pick_place_replay.py`: action replay with grasp-segment boundary from demo metadata.
- `src/svla/state_bc.py`, `src/svla/demo_recorder.py`: demonstration recording and BC.
- `scripts/validate_task_robustness.py`: readiness vs broad domain stress tests.
- `scripts/run_pick_place_trials.py`, `scripts/record_pick_place_demo.py`: pick-place matrix and demo export.
- `scripts/render_pickup_showcase.py`, `scripts/render_bc_rollout.py`: MP4 visual review.

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
7. EE action replay retains ~9% controller saturation; joint replay does not.
8. Learned EE policies fail stricter gates badly (21% success); joint is better but not clean (65%).
9. Separate "Phase-6 readiness domain" (±5% geometry, friction 1.6–2.0) from "broad stress
   report" (±15%, friction 0.8–2.4). Do not relabel broad failures as passes.
10. Controller-level scripting composes cleanly (pickup 36/36, pick-place 6/6); BC gaps are
    learning/timing/saturation, not IK or scripting difficulty. See success-rate ladder below.

## End of Phase 5: Success-Rate Ladder (pickup, strict physics gates)

Separate **scripting**, **label replay**, and **learned BC**. All three use the same gates;
they answer different questions.

| Layer | What it measures | EE (`ee_tool_delta`) | Joint (`joint_delta`) |
|-------|------------------|----------------------|------------------------|
| 1. Scripted controller | Expert runs `scripted_*_commands` directly | 36/36 (100%) | 36/36 (100%) |
| 2. Action replay | Replays recorded `policy_labels` from demos | 18/18 (100%) | 18/18 (100%) |
| 3. Learned MLP BC | State BC on final eval grid (3 seeds × 24 trials) | 15/72 (20.8%) | 47/72 (65.3%) |

Evidence: `outputs/pickup_trials_physics_audit.summary.json`,
`outputs/action_replay_physics_audit_summary.json`,
`outputs/state_bc_physics_audit_final/state_bc_summary.json`.

**Interpretation:** the controller and demo labels are not the bottleneck. EE BC fails mostly
on timing/sequencing, not IK infeasibility. Replay saturation: EE ~9% (pickup), ~7% (pick-place);
joint ~0%.

### Learned-policy failure breakdown (final eval, pickup)

| Metric | EE BC | Joint BC |
|--------|-------|----------|
| `event_order_valid` rate | 41.7% | 72.2% |
| `early_close` trials | 3 | 0 |
| Top failure among unsuccessful rollouts | `event_order_failure` (41) | `event_order_failure` (18) |

Pre–physics-audit runs (`state_bc_grasp_tcp_final`, 63/72 each) used looser success criteria;
do not cite them as current readiness evidence.

### Pick-and-place extension (scripted validation only)

Post-lift phases (transport → lower → open → retreat) extend pickup without new controller
primitives. Scripted matrix: **6/6** (`outputs/pick_place_trials.summary.json`). One recorded
demo with full label contract (`svla_pick_place_demo_v1`, boundary index in metadata).
Action-replay compare: both action spaces succeed on the recorded demo
(`outputs/action_replay_pick_place_compare.json`). **No pick-place BC yet.**

Left placement needs separate goal vs command markers (`place_left_command_marker` offset) due
to ~12 mm asymmetric-jaw transport slip; ablation without offset drops to 50% on left trials.

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
2. **Event order** — contact before close, close before lift, no pre-close contact, no reopen.
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

- **EE controller saturation:** scripted EE action replay clips ~9% of steps; joint replay 0%.
  Learned EE policies inherit this and show high `clipped_translation` counts.
- **Asymmetric jaw:** pivoting gripper causes direction-dependent lateral shift during closure;
  compensation gains are tuned for the readiness envelope only.
- **Simulation-only force gates:** limits are MuJoCo sanity checks, not hardware-calibrated.
- **BC eval grid:** final eval uses trial IDs 4001+ (`final_trial_specs`); render with
  `--eval-mode final` on `render_bc_rollout.py`.

## Visual Review Findings (2026-07-01 physics audit)

Render stdout metrics align with gate telemetry. No visual/numerical mismatch that would
block Phase 6 vision infrastructure.

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

## Phase-6 Readiness Verdict

Evidence paths: `outputs/task_robustness_readiness_summary.json`,
`outputs/task_robustness_broad_summary.json`, `outputs/pickup_trials_physics_audit.summary.json`,
`outputs/action_replay_physics_audit_summary.json`, `outputs/state_bc_physics_audit_final/state_bc_summary.json`,
`outputs/pick_place_trials.summary.json`, `outputs/action_replay_pick_place_compare.json`,
visual clips listed above.

| Decision | Verdict |
|----------|---------|
| Scripted expert in readiness domain | **READY** — 288/288 pass; baseline 36/36; max force 21.2 N, impulse 8.9 N·s (within MuJoCo sanity gates) |
| Learned policies for research comparison | **Partially viable** — joint-delta 47/72 (65%) under strict gates; not clean enough for final claims. EE 15/72 (21%) **not viable** as primary comparison |
| EE action space | **Blocked** for primary research claims under strict gates. Secondary use allowed only with documented ~9% scripted replay saturation and 21% BC success caveat |
| Phase 6 start (vision-only infrastructure) | **GO** — physics gate closed for scripted expert in readiness domain. Vision pipeline (observations, rendering, dataset format) may proceed. Vision-conditioned training and VLA must wait for EE policy fixes or accept joint-only comparison |
| Out of domain | Broad stress 241/252 — 11 expected OOD failures; do not relabel as readiness passes |

**Controller vs simulator vs ML:** kN forces were a **simulator/contact-model bug** (invalid
`solimp`, uncapped jaw). Broad-domain shove-and-recover failures are **task/controller envelope**
limits. EE BC failures (early close, reopen, event-order) are **ML failure modes** distinct
from corrected physics.

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

Phase 6 vision infrastructure (after gate):

- Add fixed camera observations to `pickup_task.py` / demo recorder.
- Render vision datasets from scripted demos in readiness domain only.
- Keep BC comparison on joint-delta until EE event-order failures are addressed.

Phase 5 follow-up (not blocking vision infra):

- H-EE-010 inference ablation **rejected** — next: H-EE-001/009 env-derived phase or retrain without progress features.
- Reduce EE controller saturation or widen training distribution (H-EE-002).
- Improve joint BC from 65% toward stable readiness-domain pass rate.
- Run joint-only pick-place BC first; defer EE pick-place until pickup EE event-order improves.

When evaluating policies, keep observations, demonstrations, task initialization, and
success metrics identical across action spaces. Otherwise the result will not answer the
research question.