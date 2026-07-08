# Research Notes

Living document for hypotheses, open questions, and experiment results. **`AGENTS.md`**
holds stable phase verdicts and commands; this file holds reasoning that is still moving.

When a hypothesis is tested and changes what we believe, update **both** files:
1. Set status + result here (with evidence path).
2. Add a dated one-liner under **Research verdict updates** in `AGENTS.md` if the phase
   verdict, blocker list, or success-rate ladder changes.

## How to log a hypothesis

| Field | What to write |
|-------|----------------|
| **ID** | `H-EE-001` style |
| **Status** | `untested` → `testing` → `confirmed` / `rejected` / `partial` |
| **Claim** | One sentence, falsifiable |
| **Why we think it** | Mechanism + code/telemetry pointer |
| **Test** | Minimal experiment (script, metric, pass/fail bar) |
| **Result** | Fill after run; link `outputs/...` or scratch log |

Do not mark `confirmed` from supervised loss alone — closed-loop rollout under strict gates
is the arbiter for BC hypotheses.

---

## Experiment provenance (manifests)

**Problem:** `outputs/` is gitignored. Prior artifacts did not record enough information to
prove which code, assets, controller limits, and seeds produced a given JSON/JSONL/MP4.

**Fix (2026-07-01):** `src/svla/experiment_manifest.py` writes sidecar manifests beside
primary outputs. Format: `svla_experiment_manifest_v1`. Audit workflow is documented in
`evidence/README.md`.

| Script | Primary output | Sidecar manifest |
|--------|----------------|------------------|
| `scripts/run_pickup_trials.py` | `outputs/pickup_trials.jsonl` | `outputs/pickup_trials.manifest.json` |
| `scripts/run_pick_place_trials.py` | `outputs/pick_place_trials.jsonl` | `outputs/pick_place_trials.manifest.json` |
| `scripts/validate_action_replay.py` | `outputs/action_replay_tool_axis_summary.json` | `outputs/action_replay_tool_axis_summary.manifest.json` |
| `scripts/validate_task_robustness.py` | `outputs/task_robustness_summary.json` | `outputs/task_robustness_summary.manifest.json` |
| `scripts/train_state_bc.py` | `outputs/state_bc/state_bc_summary.json` | `outputs/state_bc/state_bc_summary.manifest.json` |
| `scripts/record_pickup_vision_demos.py` | `outputs/phase6a_vision_sample/vision_manifest.json` | `outputs/phase6a_vision_sample/vision_manifest.manifest.json` |

Each manifest records: UTC timestamp, exact command, git SHA, dirty flag, diff SHA-256 when
dirty, untracked-file identities when present, Python/MuJoCo/NumPy/SciPy versions, SHA-256
of source/asset files, `PICKUP_CONTROLLER_LIMITS`, physics gate constants, seeds, and
SHA-256 of listed output files.

**Rule:** Before comparing runs, match the manifest source hashes, commit, dirty diff, and
untracked-file identities. The 4001+ `state_bc_physics_audit_final` grid is historical;
do not relabel it as protocol-v2 evidence.

---

## Phase 5 evaluation protocol v2 (registered and executed)

`configs/phase5_evaluation_protocol_v2.json` froze the state-BC comparison before model
selection. It contains exact nominal-physics train, validation, and final object
positions; three orientations; two approaches; and five model seeds. The split positions
are pairwise disjoint. The runner records the config SHA-256 in demos, model artifacts,
trial rows, summaries, and provenance source hashes.

The old `heldout` / `test` / `audit` / `final` modes remain available only as historical
legacy grids. Protocol v2 requires `--evaluation-protocol v2 --eval-split ...`; in
particular, the new final holdout cannot be selected without the literal
`--eval-split final` flag. Missing orientation/approach contexts are fatal instead of being
silently removed from the denominator. `--eval-limit` is an explicit diagnostic-only slice
and is recorded in every summary; it is not valid evidence for a release decision.

The registered gates cover success, event order, physical sanity, pre-close contact,
numerical controller failure, hard-limit/infeasible step rate, and per-seed stability. Their
config status is **`proposed_awaiting_approval`**. They are not a settled verdict and must
not be retroactively tuned after the final split is opened. Geometry and friction remain
nominal because they are not policy observations in this state-only experiment; randomizing
hidden dynamics would confound the action-space comparison.

Implementation support now exists for two MLP input contracts:

- `legacy_progress_phase`: the historical observation + cursor/progress/phase contract.
- `none`: observation + explicit gripper-object distance, trained and evaluated without
  progress, phase one-hot, or phase-progress inputs (H-EE-012).

An optional `--gripper-close-guard` implements the symmetric H-EE-011 / H-JNT-001 diagnostic.
It is off by default, applies the same distance rule to both action spaces, records raw and
executed action metrics separately, and never suppresses reopen commands after legal closure.
Guarded results remain labeled `shielded_policy=true` and separate from raw BC.

Validation selected one shared `legacy_progress_phase` contract for both action spaces:
joint 53/120 and EE 31/120, versus cursor-free joint 18/120 and EE 32/120. The cursor-free
contract changed EE by only +1/120 while lowering EE event validity and increasing
early-close trials; **H-EE-012 is rejected**. The selection was frozen before final access in
`evidence/phase5_v2_model_selection.json`.

The registered raw final produced joint 51/120 and EE 28/120. Both fail the proposed
aggregate success, event-order, physical-sanity, hard-limit, and per-seed thresholds. The
separate byte-identical-model distance-guard diagnostic left EE unchanged and moved joint
only to 55/120, so **H-EE-011 and H-JNT-001 are rejected**. Evidence:
`evidence/phase5_v2_final_results.json`.

---

## Open question: Why does EE BC mis-order grasp events?

**Current symptom:** On the registered raw final, `ee_tool_delta` succeeds **28/120
(23.3%)** versus joint **51/120 (42.5%)**. EE is event-order valid on 38/120 and physically
sane on 68/120; joint is event-order valid on 65/120 and physically sane on 80/120. EE also
records 741 pre-close contact steps and 196 reopen events. This is a multi-gate failure, not
a demonstrated timing-only mechanism.

**Event-order gate (grasp segment):** required timestamps (all present):
close command starts → finger–object contact → object unsupported → lift clearance. No
**pre-close** contact (`preclose_contact_steps == 0`), no **reopen**, and close must begin
within **15 mm** of the object (`EARLY_CLOSE_DISTANCE`). See `_event_order_valid()` in
`pickup_task.py`.

**What is already ruled out:**

| Ruled out | Evidence |
|-----------|----------|
| Basic scripted task infeasibility | Scripted pickup and readiness bundles pass in the declared simulator envelope |
| Non-executable policy labels | Pickup and pick-place label replay pass for both action spaces |
| Numerical controller breakdown | Both raw final policies have zero `controller_failure_steps` |

These checks do **not** rule out data insufficiency or controller-constraint interaction.
Raw final joint-limit exposure is 21.3% of EE rollout steps and 19.7% of joint rollout steps;
`controller_failed` is a numerical-failure flag, not a synonym for saturation. Recording more
demos without a falsifiable coverage hypothesis is still low-value, but it has not been
causally ruled out. See the ladder in `AGENTS.md`.

---

## Hypotheses (EE event misordering)

| ID | Status | Claim | Why we think it | Test | Result |
|----|--------|-------|-----------------|------|--------|
| H-EE-001 | untested | **Open-loop phase clock drift:** rollout advances `cursor` each step and maps it to demo phase lengths; when motion lags demos, the policy may index into `close_gripper` at the wrong physical state. | `state_bc.py` derives MLP phase from `_phase_at_cursor`, not live contact. Constraint exposure is substantial, but the registered results do not establish a causal link to event failure. | Train/evaluate an env-derived phase contract on protocol-v2 validation before any new final access. | |
| H-EE-002 | partial | **EE saturation lag:** controller constraints interact with the learned EE trajectory and event sequence. | Raw final EE has 21.3% joint-limit exposure and 33.5% clipped-joint exposure; labels use feasible deltas from telemetry. | Causal lower-gain or cap ablation on protocol-v2 validation. | **Simple monotonic causality is not supported.** In registered raw final EE rollouts, saturation is weakly inversely associated with event failure (`r=-0.118`) and rollout failure (`r=-0.166`). Constraint exposure remains real, but may be consequence, exposure time, or an interaction. Evidence: `outputs/phase5_v2_final_selected_legacy/eval/policy_failure_analysis.json`. |
| H-EE-003 | untested | **Gripper coupled in one MLP head:** a single 6-D output mixes Cartesian and gripper; arm error states map to multiple gripper label modes in training, so the network closes/opens at wrong states. | Gripper is dim 5 of `ee_tool_delta`; MSE treats all dims equally. Joint 6-D may factor gripper timing with smoother arm coords. | Train gripper-only head (or thresholded schedule) on same demos; compare event-order rate. | |
| H-EE-004 | partial | **Early-close geometry threshold:** gate requires close command within 15 mm; some policies trigger close from the wrong state. | Raw final has 2 EE and 11 joint early-close trials. The guard removes both counts without materially improving success. | Analyze close-start distance jointly with pre-close contact and reopen behavior; do not relax the release gate. | Early-close alone is not the main blocker: guarded EE remains 28/120 and guarded joint reaches only 55/120. |
| H-EE-005 | untested | **Gripper oscillation → reopen:** policy outputs open commands after partial close when phase clock or state mismatch; `_episode_reopen_events` increments. | Failure clips (`ee_bc_failure.mp4`, trial 4005 seed 1) show reopen before valid sequence. | Plot gripper command vs `gripper_open` state on failure rollouts; count sign changes near close phase. | |
| H-EE-006 | rejected | **Yaw-specific held-out difficulty:** the historical `yaw_-18` gap is the main EE failure cluster. | The legacy 4001+ grid suggested a yaw gap. | Check the registered final bucket breakdown before collecting targeted yaw data. | **Not reproduced.** Registered raw EE success is 9/40 at `yaw_-18`, 9/40 at `yaw_0`, and 10/40 at `yaw_18`; yaw-targeted data is not justified by this result. |
| H-EE-007 | untested | **Label asymmetry:** `policy_labels.ee_tool_delta` are reconstructed feasible Cartesian deltas; joint labels are direct joint-target errors. Extra label noise targets EE timing. | `demo_recorder._policy_labels` uses `feasible_delta_xyz/rotvec`; joint uses `joint_target_error`. | Compare NN replay on raw `labels.ee_tool_delta` vs `policy_labels.ee_tool_delta`; if only policy_labels replay clean, noise is in reconstruction path. | |
| H-EE-008 | untested | **Loss underweights sequencing:** uniform MSE on all action dimensions does not penalize gripper timing errors that cause gate failures. | BC training has no event-order-aware loss. | Weight gripper dim higher near `close_gripper` phase in training, or auxiliary classifier for "should close"; re-eval final grid. | |
| H-EE-009 | untested | **Progress features train on demo time but rollout time diverges:** MLP gets progress/phase one-hot at train time from demo `step_index`; at rollout, progress is cursor/demo-length ratio while physical phase lags. | `fit_mlp_policy` stacks progress + phase from demos; rollout `_phase_at_cursor` uses training `group_phase_lengths`. | Train with **normalized phase_step only** (no global cursor) and/or add `gripper_object_distance` bins; compare EE event order. | |
| H-EE-010 | rejected | **No-cursor rollout ablation:** removing open-loop `cursor` (and MLP progress/phase inputs at inference) lets proprioception drive gripper timing closed-loop, improving EE `event_order_valid` without more demos or vision. | State obs already includes contact, gripper, object−EE offset; cursor forces demo-time indexing that desyncs when EE saturates (see “Why a step counter” below). | Inference-only ablation: zero progress/phase features, do not advance `cursor`. EE MLP seeds 0–2, final grid 72 trials. Pass bar: ≥15 pp `event_order_valid` or ≥10 pp success. | **Failed badly.** Baseline rerun 14/72 success (37.5% event-order). Ablation **0/72** success, **1/72** event-order, **24** `early_close`. Policy trained *with* progress/phase; zeroing them at inference is OOD — not a fair closed-loop fix without retraining. Evidence: `outputs/h_ee_010_no_cursor_ablation.json`. Code reverted. |
| H-EE-011 | rejected | **Gripper distance gate at rollout:** suppress close commands until `gripper_object_distance` ≤15 mm. | Proprioception exposes distance and the guard can remove structurally early close commands without retraining. | Symmetric guarded diagnostic on the selected byte-identical models. Pass bar: ≥10 pp success or ≥15 pp event-order vs raw. | **Rejected.** The guard suppressed 507 EE close steps and removed 2 early-close trials, but success stayed 28/120 and event order stayed 38/120; pre-close contact stayed 741 and reopen events rose 196→198. |
| H-EE-012 | rejected | **Retrain without progress/phase + explicit distance:** cursor-free state BC improves the shared action-space comparison. | H-EE-010 showed inference-only stripping was unfair, so the model was retrained for the new input contract. | Registered validation, five seeds, same architecture/data for both action spaces. | **Rejected.** EE changed 31/120→32/120 while event validity declined and early-close rose 5→24; joint collapsed 53/120→18/120. The shared legacy contract was selected before final access. |
| H-EE-013 | untested | **Env-derived phase at rollout:** feed MLP progress/phase from live contact, lift, and distance bins instead of `cursor` / demo phase lengths. | Extends H-EE-001/009 without zeroing phase features (H-EE-010 lesson). Maps env state → phase index + phase_progress. | Implement env phase estimator in `rollout_policy`; optional light finetune if OOD. Compare EE event-order on final grid. | |
| H-EE-014 | untested | **Hybrid NN gripper + MLP arm:** nearest-neighbor on `MATCH_FEATURE_INDICES` for gripper dim only; MLP for Cartesian dims. | NN matches state-local demo timing on object-relative features (`state_bc.py` `MATCH_FEATURE_INDICES`); timing may be more replay-stable than global MLP phase clock. | Rollout compositor or two-head export; no full retrain if NN policy exists for same demos. | |
| H-EE-015 | untested | **Scripted gripper schedule + learned arm:** expert/task defines when close is legal; policy outputs arm deltas only (or gripper overridden by FSM). | Shrinks ML problem to motion; controller-first ethos. Event order becomes structural. | Phase FSM from env distance/contact; BC on arm only; eval gates unchanged. | |
| H-EE-016 | untested | **Close-phase demo oversampling:** weight loss or duplicate samples around `grasp_align` → `close_gripper` boundary. | Failures cluster at phase transitions; uniform stride under-represents close timing. | 2–5× sample weight on close phases; one seed EE retrain; event-order rate vs baseline. | |
| H-EE-017 | untested | **Short observation history (stack or tiny GRU):** gripper timing needs temporal context; Markovian single-step state is insufficient. | Policies reopen or close early despite contact flags — may need velocity/trend of approach, not just instant state. | 3–5 step joint/EE/gripper history or 1-layer GRU; same demos; compare EE event-order. | |
| H-JNT-001 | rejected | **The same distance guard materially improves joint readiness.** | Raw joint has 11 early-close trials, so the symmetric guard has a plausible target. | Same registered guarded diagnostic and byte-identical selected models. | **Rejected as a readiness fix.** Success moved 51/120→55/120 and event order 65/120→70/120; physical sanity fell 80/120→76/120 and the worst seed remained 2/24. |

---

## Improvement ideas backlog (tiered)

Ideas below complement the hypothesis table. H-EE-010, H-EE-011, and H-EE-012 rule out
inference-only feature stripping, a distance-only shield, and the tested cursor-free MLP
contract as sufficient fixes. Prefer mechanisms that change temporal supervision or the
gripper objective, and select them on validation.

### Tier 1 — Gripper timing (highest leverage)

| ID | Idea | Retrain? | Effort |
|----|------|----------|--------|
| H-EE-003 | **Separate gripper head** (classifier or 1-dim head) | Yes | Medium |
| H-EE-013 | **Env-derived phase** feeds existing MLP phase inputs | Finetune optional | Medium |

### Tier 2 — Labels / controller / loss

| ID | Idea | Retrain? | Effort |
|----|------|----------|--------|
| H-EE-002 | Lower EE **gain** or saturation in labels/rollout | Yes | Medium |
| H-EE-008 | **Gripper-weighted** or phase-aware MSE | Yes | Low–medium |
| H-EE-007 | Compare raw `labels` vs `policy_labels` replay | No | Low |
| H-EE-014 | **NN gripper + MLP arm** hybrid | Maybe | Medium |

### Tier 3 — Data & coverage (after timing fixes)

| ID | Idea | Retrain? | Effort |
|----|------|----------|--------|
| H-EE-016 | Oversample **close-phase** transitions | Yes | Low |

### Tier 4 — Architecture & Phase 6

| ID | Idea | Retrain? | Effort |
|----|------|----------|--------|
| H-EE-017 | Short **history** or tiny GRU | Yes | Medium–high |
| H-VIS-001 | RGB + proprio with a newly justified temporal contract | Yes | High |
| H-EE-015 | **FSM gripper** + learned arm | Partial | Medium |

### Deprioritized / ruled out

| Item | Why |
|------|-----|
| H-EE-010 inference-only no-cursor | **Rejected** — 0/72 success |
| H-EE-011 distance guard | **Rejected** — EE success/event order unchanged on registered diagnostic |
| H-EE-012 cursor-free state MLP | **Rejected** — EE unchanged in practice; joint validation collapsed |
| H-JNT-001 joint distance guard | **Rejected** — +4/120 success, worse physical-sanity rate, unstable seeds |
| H-EE-006 yaw-targeted data | **Rejected for current evidence** — registered yaw buckets are nearly equal |
| 10× more demos without a coverage hypothesis | Costly and non-falsifiable; registered pose buckets should guide any targeted data test |
| Phase 6 vision + same cursor MLP | Cameras don't fix open-loop clock (see vision section) |
| Loosen gates for “better” numbers | Diagnostic only; not a behavior fix |

---

## Hypothesis priority (suggested order)

1. **H-EE-013** — train/evaluate env-derived phase on validation; do not retrofit it only at inference.
2. **H-EE-003 / H-EE-008** — separate gripper head or weighted/auxiliary sequencing loss.
3. **H-EE-014** — hybrid state-local gripper decision with learned arm output.
4. **H-EE-002** — causal saturation/gain ablation; descriptive correlation alone is insufficient.
5. **H-EE-016** — close-transition oversampling under the same validation contract.
6. **H-EE-017 / H-EE-015** — temporal history or an explicitly hybrid FSM-gripper baseline.
7. **H-VIS-001** — only after Phase 5 policy scope is resolved; vision work was not started here.

---

## Results log

| Date | Hypothesis | Experiment | Outcome | Evidence |
|------|------------|------------|---------|----------|
| 2026-07-01 | — | Historical physics-audit legacy-grid eval | EE 15/72, joint 47/72; superseded by registered protocol-v2 evidence | `outputs/state_bc_physics_audit_final/` |
| 2026-07-01 | — | Historical action replay compare | EE/joint replay 18/18 pickup | `outputs/action_replay_physics_audit_summary.json` |
| 2026-07-01 | — | Historical scripted pick-place | 6/6 scripted; later baseline expanded replay to six demos per space | `outputs/pick_place_trials.summary.json` |
| 2026-07-01 | H-EE-010 | No-cursor inference ablation (EE MLP, 3 seeds × 24 trials) | **Rejected** — 0/72 vs baseline 14/72 success; 1/72 vs 27/72 event-order; 24 early_close | `outputs/h_ee_010_no_cursor_ablation.json` |
| 2026-07-01 | — | Experiment manifest utility | Sidecar provenance on five experiment scripts; unit tests pass | `src/svla/experiment_manifest.py`, `evidence/README.md`, `tests/test_experiment_manifest.py` |
| 2026-07-01 | H-EE-011 / H-EE-012 | One-seed implementation smoke (1 epoch; not efficacy evidence) | Both code paths trained/loaded and shield telemetry remained separate; later registered tests determined the verdicts | `outputs/scratch_h_ee_012_v2_validation_smoke_retry/`, `outputs/scratch_h_ee_011_guard_v2_validation_smoke/` |
| 2026-07-01 | H-EE-002 | Existing-rollout failure correlation | **Partial** — higher saturation is associated with better, not worse, EE outcomes; 3 early-close trials show higher infeasible rate; causal ablation still needed | `outputs/state_bc_physics_audit_final/eval/policy_failure_analysis.json` |
| 2026-07-02 | H-EE-012 | Registered legacy-vs-cursor-free validation, 5 seeds × 24 | **Rejected** — EE 31→32/120 with worse event order and more early close; joint 53→18/120. Legacy selected once for both spaces | `evidence/phase5_v2_model_selection.json` |
| 2026-07-02 | — | Registered raw final, selected legacy contract | EE 28/120; joint 51/120. Both fail proposed learned-policy gates; zero numerical controller failures does not erase hard-limit exposure | `evidence/phase5_v2_final_results.json` |
| 2026-07-02 | H-EE-011 / H-JNT-001 | Shielded distance-guard final diagnostic using byte-identical models | **Rejected** — EE unchanged at 28/120 and 38/120 event order; joint 55/120, only +4 successes, with physical sanity down 80→76 | `evidence/phase5_v2_final_results.json` |
| 2026-07-02 | H-EE-002 | Registered raw-final correlation analysis | **Partial** — EE saturation weakly inversely associated with event and rollout failure; causal gain/cap ablation remains untested | `outputs/phase5_v2_final_selected_legacy/eval/policy_failure_analysis.json` |
| 2026-07-02 | — | Final source-matched scripted/replay/readiness bundle | **Pass** — 85 tests; pickup 36/36; pickup replay 18/18 per space; pick-place 6/6; pick-place replay 6/6 per space; readiness 288/288 | `outputs/phase5_baseline_final/phase5_baseline_v2_aggregate.json` |
| 2026-07-08 | — | Phase 6a scripted RGB dataset infrastructure | **Implemented, not policy evidence** — fixed-camera RGB capture, NPZ frame arrays, action-space-neutral manifests, validator, MP4 preview path, and compatibility tests pass; no vision policy or VLA training started | `outputs/phase6a_vision_sample/vision_manifest.json` |

*(Add a row when you run a hypothesis test — do not infer from loss curves alone.)*

---

## Why a step counter (`cursor`) at all?

**Not because we lack vision.** Phase 5 state BC already has rich proprioception (EE pose,
object pose, `gripper_object_contact`, `gripper_open`, relative vectors). The counter is a
**demo-alignment prior**:

- **Nearest-neighbor BC** needs “which part of the demonstration am I imitating?” —
  `cursor` + `progress_indices` restrict the search window along the recorded timeline
  (`state_bc.py`, `NearestNeighborBCPolicy.predict_with_index`).
- **MLP BC** inherited the same rollout driver: each step increments `cursor`, and
  progress/phase one-hot are computed from **training demo lengths** for that task context,
  not from live env phase (`MLPBCPolicy._phase_at_cursor`).

Training stacks `(observation, progress, phase)` → action from demos where `step_index` and
`phase` are known. At rollout, `cursor` substitutes for “true” demo time when the physical
trajectory drifts — that is the suspected failure mode (H-EE-001), independent of cameras.

Tested state-only alternatives did not solve the problem: inference-only cursor drop
(**H-EE-010**), the distance guard (**H-EE-011**), and cursor-free retraining
(**H-EE-012**) are **rejected**. Remaining state-only candidates include env-derived phase
(**H-EE-013**) and a separate/weighted gripper objective (**H-EE-003/008**).

---

## Could vision fix the EE event-order gap?

**Potentially, but not automatically.** Vision changes what the policy can see, not the
supervision contract.

| Mechanism | Vision might help | Vision might not help |
|-----------|-------------------|----------------------|
| Close when visually aligned over cube | Yes — closed-loop cue for gripper timing | — |
| Compensate EE saturation / slower motion | Indirectly, if policy adapts path | If rollout still advances open-loop `cursor`, same desync |
| Pose-dependent failures | Possibly — appearance may add alignment information | Registered yaw buckets are nearly equal, so the old `yaw_-18` story did not reproduce |
| Event-order gates | Only if learned behavior respects contact sequence | Pretty trajectories that still close early / reopen |

Phase 6 risk: bolting RGB onto the **same** `(state, cursor, phase) → action` MLP keeps the
clock problem; cameras become extra dims while gripper timing stays tied to demo step count.

H-EE-012 shows that simply retraining the current MLP without cursor features is not enough.
Any future vision-policy experiment needs a newly justified temporal/gripper contract rather
than assuming RGB plus cursor removal is a fix. **Phase 6a data/render/validation plumbing is
implemented; Phase 6b learned vision policy and VLA work are not started.**

| ID | Status | Claim | Test | Result |
|----|--------|-------|------|--------|
| H-VIS-001 | untested | RGB + proprio with a newly selected temporal/gripper contract improves event order over the registered state baseline | First select the non-visual temporal design on validation; then add RGB without opening the final split for architecture tuning | Not started |
| H-VIS-002 | untested | RGB adds useful alignment information beyond proprioception | Compare matched-capacity state and vision policies by pose/approach buckets; do not anchor the test to the rejected yaw or distance-guard hypotheses | Not started |

---

## Related open questions (backlog)

- What causes the remaining joint event-order and physical failures after the distance guard
  failed as a readiness fix?
- Will pick-place BC inherit grasp-boundary mis-timing even with `grasp_segment_finalize_sample_index`?
- Does H-EE-015 (FSM gripper + learned arm) answer the research question, or sidestep the action-space comparison?
