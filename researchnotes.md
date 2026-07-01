# Research Notes

Living document for hypotheses, open questions, and experiment results. **`Agents.md`**
holds stable phase verdicts and commands; this file holds reasoning that is still moving.

When a hypothesis is tested and changes what we believe, update **both** files:
1. Set status + result here (with evidence path).
2. Add a dated one-liner under **Research verdict updates** in `Agents.md` if the phase
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

## Open question: Why does EE BC mis-order grasp events?

**Symptom:** Learned `ee_tool_delta` policies fail strict pickup gates at **15/72 (20.8%)**
vs joint **47/72 (65.3%)** (`outputs/state_bc_physics_audit_final`). Dominant failure:
**`event_order_failure`** (41 EE rollouts). Smaller counts: `early_close` (3 trials),
`reopen_events` > 0 on failure clips.

**Event-order gate (grasp segment):** required timestamps (all present):
close command starts → finger–object contact → object unsupported → lift clearance. No
**pre-close** contact (`preclose_contact_steps == 0`), no **reopen**, and close must begin
within **15 mm** of the object (`EARLY_CLOSE_DISTANCE`). See `_event_order_valid()` in
`pickup_task.py`.

**What is already ruled out:**

| Ruled out | Evidence |
|-----------|----------|
| Controller cannot do the task | Scripted 36/36; demo label replay EE 18/18 |
| Pure demo shortage at current scale | Same 30 demos; joint reaches 65%; replay 100% |
| IK systematically broken on EE | Zero controller-failure steps in BC eval summary |

**Not the main fix:** "Record more demos" without changing timing supervision or
rollout–phase coupling. More data may help joint coverage; EE gap is larger than a sample-count
story (see ladder in `Agents.md`).

---

## Hypotheses (EE event misordering)

| ID | Status | Claim | Why we think it | Test | Result |
|----|--------|-------|-----------------|------|--------|
| H-EE-001 | untested | **Open-loop phase clock drift:** rollout advances `cursor` each step and maps it to demo phase lengths; when EE motion lags demos (saturation, infeasible), the policy still indexes into `close_gripper` while the arm is too far away → `early_close` / wrong contact order. | `state_bc.py` `rollout_policy` uses `cursor = max(cursor+1, nearest_index+1)`; MLP phase comes from `_phase_at_cursor`, not live contact. EE rollouts show higher `clipped_joint_steps` / infeasible counts than joint. | Rollout with phase derived from **env** (sample `phase` string or contact flags) instead of cursor; compare EE `event_order_valid` rate on 24-trial slice. | |
| H-EE-002 | untested | **EE saturation lag:** ~9% scripted EE replay steps saturate; learned EE hits bounded joint path more often, stretching approach so close-phase labels fire late or gripper opens to recover. | `action_replay_physics_audit_summary.json` mean EE saturation 9%; EE BC jsonl shows high `joint_step_clipped_steps`. Labels use `feasible_delta_*` from telemetry (`demo_recorder._policy_labels`). | Correlate per-rollout saturation/infeasible step rate with `event_order_failure`; ablate with lower EE gain or stricter delta caps. | |
| H-EE-003 | untested | **Gripper coupled in one MLP head:** a single 6-D output mixes Cartesian and gripper; arm error states map to multiple gripper label modes in training, so the network closes/opens at wrong states. | Gripper is dim 5 of `ee_tool_delta`; MSE treats all dims equally. Joint 6-D may factor gripper timing with smoother arm coords. | Train gripper-only head (or thresholded schedule) on same demos; compare event-order rate. | |
| H-EE-004 | untested | **Early-close geometry threshold:** gate requires close command within 15 mm; EE policies approach on slightly different paths and trigger close from demo phase clock while still >15 mm. | `early_close` flag uses `_episode_close_start_distance > EARLY_CLOSE_DISTANCE`. 3 EE trials with `early_close=true` in eval. | Log close-start distance on all EE failures; relax threshold on **eval only** to see if failures are purely geometric vs ordering. | |
| H-EE-005 | untested | **Gripper oscillation → reopen:** policy outputs open commands after partial close when phase clock or state mismatch; `_episode_reopen_events` increments. | Failure clips (`ee_bc_failure.mp4`, trial 4005 seed 1) show reopen before valid sequence. | Plot gripper command vs `gripper_open` state on failure rollouts; count sign changes near close phase. | |
| H-EE-006 | partial | **Held-out pose difficulty (not random noise):** EE success is lowest on `yaw_-18` (~25%) vs `yaw_0` (~52%) in final eval — misordering clusters on hard buckets. | `state_bc_physics_audit_final` eval summaries by orientation. | Add demos only for `yaw_-18` bucket (+10 demos) with **same** MLP; if success stays flat, data volume is not the driver. | |
| H-EE-007 | untested | **Label asymmetry:** `policy_labels.ee_tool_delta` are reconstructed feasible Cartesian deltas; joint labels are direct joint-target errors. Extra label noise targets EE timing. | `demo_recorder._policy_labels` uses `feasible_delta_xyz/rotvec`; joint uses `joint_target_error`. | Compare NN replay on raw `labels.ee_tool_delta` vs `policy_labels.ee_tool_delta`; if only policy_labels replay clean, noise is in reconstruction path. | |
| H-EE-008 | untested | **Loss underweights sequencing:** uniform MSE on all action dimensions does not penalize gripper timing errors that cause gate failures. | BC training has no event-order-aware loss. | Weight gripper dim higher near `close_gripper` phase in training, or auxiliary classifier for "should close"; re-eval final grid. | |
| H-EE-009 | untested | **Progress features train on demo time but rollout time diverges:** MLP gets progress/phase one-hot at train time from demo `step_index`; at rollout, progress is cursor/demo-length ratio while physical phase lags. | `fit_mlp_policy` stacks progress + phase from demos; rollout `_phase_at_cursor` uses training `group_phase_lengths`. | Train with **normalized phase_step only** (no global cursor) and/or add `gripper_object_distance` bins; compare EE event order. | |
| H-EE-010 | rejected | **No-cursor rollout ablation:** removing open-loop `cursor` (and MLP progress/phase inputs at inference) lets proprioception drive gripper timing closed-loop, improving EE `event_order_valid` without more demos or vision. | State obs already includes contact, gripper, object−EE offset; cursor forces demo-time indexing that desyncs when EE saturates (see “Why a step counter” below). | Inference-only ablation: zero progress/phase features, do not advance `cursor`. EE MLP seeds 0–2, final grid 72 trials. Pass bar: ≥15 pp `event_order_valid` or ≥10 pp success. | **Failed badly.** Baseline rerun 14/72 success (37.5% event-order). Ablation **0/72** success, **1/72** event-order, **24** `early_close`. Policy trained *with* progress/phase; zeroing them at inference is OOD — not a fair closed-loop fix without retraining. Evidence: `outputs/h_ee_010_no_cursor_ablation.json`. Code reverted. |
| H-EE-011 | untested | **Gripper distance gate at rollout:** suppress close commands until `gripper_object_distance` ≤ threshold (and optionally after approach); arm stays learned, gripper becomes state-gated. | H-EE-010 showed 24 `early_close` when phase signal removed — policy wants to close too early. Proprio already has distance/contact; gate enforces closed-loop timing without retrain. | `rollout_policy` wrapper: if distance > 15 mm, force gripper command open/hold. EE MLP seeds 0–2, final grid. Pass bar: ≥10 pp success or ≥15 pp `event_order_valid` vs baseline; `early_close` count drops. | |
| H-EE-012 | untested | **Retrain without progress/phase + explicit distance features:** MLP trained on proprio only (plus `gripper_object_distance` or bins); no cursor at train or eval. | H-EE-010 rejected inference-only strip; model must be trained for closed-loop state. Distance is the natural close trigger missing from obs vector today (`observation_to_features` has relative XYZ but gate uses distance explicitly). | New `fit_mlp_policy(use_progress=False)` path; add distance feature; same 30 demos; 3 seeds; final eval. Pass bar: beat 15/72 EE baseline on success or event-order. | |
| H-EE-013 | untested | **Env-derived phase at rollout:** feed MLP progress/phase from live contact, lift, and distance bins instead of `cursor` / demo phase lengths. | Extends H-EE-001/009 without zeroing phase features (H-EE-010 lesson). Maps env state → phase index + phase_progress. | Implement env phase estimator in `rollout_policy`; optional light finetune if OOD. Compare EE event-order on final grid. | |
| H-EE-014 | untested | **Hybrid NN gripper + MLP arm:** nearest-neighbor on `MATCH_FEATURE_INDICES` for gripper dim only; MLP for Cartesian dims. | NN matches state-local demo timing on object-relative features (`state_bc.py` `MATCH_FEATURE_INDICES`); timing may be more replay-stable than global MLP phase clock. | Rollout compositor or two-head export; no full retrain if NN policy exists for same demos. | |
| H-EE-015 | untested | **Scripted gripper schedule + learned arm:** expert/task defines when close is legal; policy outputs arm deltas only (or gripper overridden by FSM). | Shrinks ML problem to motion; controller-first ethos. Event order becomes structural. | Phase FSM from env distance/contact; BC on arm only; eval gates unchanged. | |
| H-EE-016 | untested | **Close-phase demo oversampling:** weight loss or duplicate samples around `grasp_align` → `close_gripper` boundary. | Failures cluster at phase transitions; uniform stride under-represents close timing. | 2–5× sample weight on close phases; one seed EE retrain; event-order rate vs baseline. | |
| H-EE-017 | untested | **Short observation history (stack or tiny GRU):** gripper timing needs temporal context; Markovian single-step state is insufficient. | Policies reopen or close early despite contact flags — may need velocity/trend of approach, not just instant state. | 3–5 step joint/EE/gripper history or 1-layer GRU; same demos; compare EE event-order. | |
| H-JNT-001 | untested | **Gripper gate helps joint too:** same distance gate as H-EE-011 pushes joint 47/72 toward 80%+ with little retrain. | Joint already 72% event-order valid; failures may be contact dynamics not early-close. | Joint MLP + H-EE-011 gate; final grid; success and event-order vs 47/72. | |

---

## Improvement ideas backlog (tiered)

Ideas below complement the hypothesis table. **H-EE-010** showed progress/phase are load-bearing
for the *current* weights — prefer **retrain** or **rollout wrappers** over inference-only
feature stripping.

### Tier 1 — Gripper timing (highest leverage)

| ID | Idea | Retrain? | Effort |
|----|------|----------|--------|
| H-EE-011 | Distance/contact **gripper gate** at rollout | No | Low |
| H-EE-003 | **Separate gripper head** (classifier or 1-dim head) | Yes | Medium |
| H-EE-012 | **Retrain** proprio-only + distance features (no cursor/progress) | Yes | Medium |
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
| H-EE-006 | Targeted demos for `yaw_-18` bucket | Yes | Medium |
| H-EE-016 | Oversample **close-phase** transitions | Yes | Low |

### Tier 4 — Architecture & Phase 6

| ID | Idea | Retrain? | Effort |
|----|------|----------|--------|
| H-EE-017 | Short **history** or tiny GRU | Yes | Medium–high |
| H-VIS-001 | RGB + proprio, **no cursor** (retrain) | Yes | High |
| H-EE-015 | **FSM gripper** + learned arm | Partial | Medium |

### Deprioritized / ruled out

| Item | Why |
|------|-----|
| H-EE-010 inference-only no-cursor | **Rejected** — 0/72 success |
| 10× more demos alone | Same failure mode without timing fix (H-EE-006 is *targeted* only) |
| Phase 6 vision + same cursor MLP | Cameras don't fix open-loop clock (see vision section) |
| Loosen gates for “better” numbers | Diagnostic only; not a behavior fix |

---

## Hypothesis priority (suggested order)

1. **H-EE-011** — gripper distance gate at rollout (cheap; no retrain).
2. **H-EE-012** — retrain without progress/phase + distance features (fair test of closed-loop proprio).
3. **H-EE-003 / H-EE-008** — separate gripper head or weighted loss.
4. **H-EE-013** — env-derived phase (if H-EE-011 helps but full retrain is deferred).
5. **H-EE-002** — saturation/gain ablation.
6. **H-EE-014** — hybrid NN gripper + MLP arm.
7. **H-JNT-001** — same gate on joint (quick win check).
8. **H-EE-006 / H-EE-016** — targeted data only after timing interventions disappoint.
9. **H-EE-017 / H-VIS-001 / H-EE-015** — larger architecture or FSM changes.

---

## Results log

| Date | Hypothesis | Experiment | Outcome | Evidence |
|------|------------|------------|---------|----------|
| 2026-07-01 | — | Physics-audit BC final eval | EE 15/72, joint 47/72; EE 41× `event_order_failure` | `outputs/state_bc_physics_audit_final/` |
| 2026-07-01 | — | Action replay compare | EE/joint replay 18/18 pickup; EE saturation ~9% | `outputs/action_replay_physics_audit_summary.json` |
| 2026-07-01 | — | Scripted pick-place | 6/6 scripted; replay 1/1 both spaces | `outputs/pick_place_trials.summary.json` |
| 2026-07-01 | H-EE-010 | No-cursor inference ablation (EE MLP, 3 seeds × 24 trials) | **Rejected** — 0/72 vs baseline 14/72 success; 1/72 vs 27/72 event-order; 24 early_close | `outputs/h_ee_010_no_cursor_ablation.json` |

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

**Alternatives that do not require vision:** gripper gate (**H-EE-011**); retrain without
cursor (**H-EE-012**); env-derived phase (**H-EE-013**); separate gripper head (**H-EE-003**).
Inference-only cursor drop (**H-EE-010**) is **rejected**.

---

## Could vision fix the EE event-order gap?

**Potentially, but not automatically.** Vision changes what the policy can see, not the
supervision contract.

| Mechanism | Vision might help | Vision might not help |
|-----------|-------------------|----------------------|
| Close when visually aligned over cube | Yes — closed-loop cue for gripper timing | — |
| Compensate EE saturation / slower motion | Indirectly, if policy adapts path | If rollout still advances open-loop `cursor`, same desync |
| Hard poses (`yaw_-18`) | Yes — appearance of cube corner / approach angle | If model is small and data sparse, same errors |
| Event-order gates | Only if learned behavior respects contact sequence | Pretty trajectories that still close early / reopen |

Phase 6 risk: bolting RGB onto the **same** `(state, cursor, phase) → action` MLP keeps the
clock problem; cameras become extra dims while gripper timing stays tied to demo step count.

Phase 6 opportunity: vision-conditioned policy **without** cursor — e.g. gripper from visual
proximity / alignment — is a distinct experiment (see H-VIS-001 below).

| ID | Status | Claim | Test | Result |
|----|--------|-------|------|--------|
| H-VIS-001 | untested | RGB + proprio BC **retrained without** cursor/progress fixes EE event-order vs state+cursor baseline | Render readiness-domain RGB on scripted demos; train vision MLP **without** progress/phase; compare `event_order_valid` on final grid. Do not bolt RGB onto cursor MLP (Phase 6 risk). | |
| H-VIS-002 | untested | RGB helps **alignment-based close** (visual proximity over cube) even when proprio gate alone is insufficient | Compare H-EE-011 gate (proprio only) vs gate + RGB finetune on hard `yaw_-18` bucket | |

---

## Related open questions (backlog)

- Does joint BC event-order failure (18 rollouts) share H-EE-001 clock drift or is it mostly contact dynamics? (**H-JNT-001**)
- Will pick-place BC inherit grasp-boundary mis-timing even with `grasp_segment_finalize_sample_index`?
- Does H-EE-015 (FSM gripper + learned arm) answer the research question, or sidestep the action-space comparison?