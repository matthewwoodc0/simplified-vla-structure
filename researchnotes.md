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
| H-EE-002 | rejected | **EE saturation lag:** controller constraints interact with the learned EE trajectory and event sequence. | Raw final EE has 21.3% joint-limit exposure and 33.5% clipped-joint exposure; labels use feasible deltas from telemetry. | Frozen-hybrid inference-only gain test under protocol-v2 validation. Execution prompt: `prompts/h-ee-002-hybrid-gain-sweep.md`. | **Rejected on validation.** The exact gain-1.0 control reproduced 62/120, EO 79, phys 68, missing-lift 30, seeds 20/14/9/9/10. Gain 0.875 collapsed to **5/120**, EO 9, phys 26, missing-lift 86, early-close 25, seeds 5/0/0/0/0. Gain 0.750 reached **0/120**, EO 0, phys 37, missing-lift 72, early-close 48. Lower gains reduced failure-conditioned constraint exposure, including the original 30 missing-lift trials, but recovered **zero** paired successes and lost 57/62 prior successes. Constraint exposure is not the root cause under this monotonic gain intervention; do not lower gain further or add an unregistered cap rescue. Final not accessed. Evidence: `outputs/h_ee_002_hybrid_gain_sweep/`, `evidence/h_ee_002_hybrid_gain_sweep.json`. |
| H-EE-003 | rejected | **Gripper coupled in one MLP head:** a single 6-D output mixes Cartesian and gripper; arm error states map to multiple gripper label modes in training, so the network closes/opens at wrong states. | The protocol-v2 training labels are exactly binary for both spaces (open `1`: 15,712 / 48,112; close `0`: 32,400 / 48,112; no intermediate values). H-EE-008 improved weighted MSE but EE seeds remained unstable. | Same H-EE-008 legacy temporal/data/128×128/5-seed validation contract, but split only the output: 5-D arm MSE head plus sigmoid binary gripper head (weighted BCE 5× / close 10×), raw rollout only. | **Rejected on validation.** EE success fell 50→34/120 and event order 55→50/120; early-close rose 5→29. EE per-seed successes were 9,1,7,6,11: lower variance only because all seeds were poor, not improved reliability. Joint also collapsed 84→41/120, event order 90→48, and physical sanity 100→73. No controller failures; final was not accessed. Evidence: `outputs/h_ee_003_separate_gripper_head_validation/h_ee_003_comparison.json`. |
| H-EE-004 | partial | **Early-close geometry threshold:** gate requires close command within 15 mm; some policies trigger close from the wrong state. | Raw final has 2 EE and 11 joint early-close trials. The guard removes both counts without materially improving success. | Analyze close-start distance jointly with pre-close contact and reopen behavior; do not relax the release gate. | Early-close alone is not the main blocker: guarded EE remains 28/120 and guarded joint reaches only 55/120. |
| H-EE-005 | partial | **Gripper oscillation → reopen:** policy outputs open commands after partial close when phase clock or state mismatch; `_episode_reopen_events` increments. | Failure clips (`ee_bc_failure.mp4`, trial 4005 seed 1) show reopen before valid sequence. | Plot gripper command vs `gripper_open` state on failure rollouts; count sign changes near close phase. | **Confirmed as the dominant residual under pure-MLP H-EE-021** (EO fails mostly reopen; fail flips ~5). **H-EE-014 hybrid eliminated reopens** (155→0 EE; flips=1.0 on success and fail). Residual under hybrid is missing_lift + early_close, not oscillation. |
| H-EE-006 | rejected | **Yaw-specific held-out difficulty:** the historical `yaw_-18` gap is the main EE failure cluster. | The legacy 4001+ grid suggested a yaw gap. | Check the registered final bucket breakdown before collecting targeted yaw data. | **Not reproduced.** Registered raw EE success is 9/40 at `yaw_-18`, 9/40 at `yaw_0`, and 10/40 at `yaw_18`; yaw-targeted data is not justified by this result. |
| H-EE-007 | rejected | **Label asymmetry:** `policy_labels.ee_tool_delta` are reconstructed feasible Cartesian deltas; joint labels are direct joint-target errors. Extra label noise targets EE timing. | `demo_recorder._policy_labels` uses `feasible_delta_xyz/rotvec`; joint uses `joint_target_error`. | Compare raw `labels.ee_tool_delta` vs `policy_labels.ee_tool_delta` through audit, replay, one-seed screen, then gated validation. Execution prompt: `prompts/h-ee-007-label-contract-probe.md`. | **Rejected at replay gate; no training.** Across 48,112 H-EE-014 samples, raw observed EE actions were ~30× smaller in mean arm L2 (0.00052 vs 0.01542), with mean cosine agreement 0.383; gripper labels matched 48,112/48,112. Policy-label control replay passed 18/18, while raw-label replay was 0/18 success and 0/18 event-order (physical sanity 18/18, zero controller failures, but zero force/impulse). Raw transitions are not executable command-scale labels. Evidence: `outputs/h_ee_007_label_contract_probe/`. |
| H-EE-008 | confirmed | **Loss underweights sequencing:** uniform MSE on all action dimensions does not penalize gripper timing errors that cause gate failures. | BC training has no event-order-aware loss. | Weight gripper dim (5×) and close phases grasp_align/close_gripper (10×) under protocol-v2 validation with legacy temporal contract; compare EE event-order/success vs legacy 31/120. | **Confirmed on validation.** Gripper-weighted MSE (5× / close 10×), same legacy temporal contract + registered hyperparams. EE 50/120 (+19 vs 31) and event-order 55/120 (+17 vs 38); early_close flat at 5; preclose 583→50; reopen 190→142. Joint 84/120 (+31 vs 53), event-order 90/120, physical-sanity 100/120. Per-seed EE still unstable (22,9,11,2,6). Pass bar met on EE success; joint did not collapse. Final not accessed. Evidence: `outputs/h_ee_008_gripper_weighted_validation/`. Report: `reports/2026-07-08-h-ee-008-gripper-weighted-mse.md`. **Causal caveat:** 008 changes global 5× and transition 10× simultaneously — see H-EE-021. |
| H-EE-009 | untested | **Progress features train on demo time but rollout time diverges:** MLP gets progress/phase one-hot at train time from demo `step_index`; at rollout, progress is cursor/demo-length ratio while physical phase lags. | `fit_mlp_policy` stacks progress + phase from demos; rollout `_phase_at_cursor` uses training `group_phase_lengths`. | Train with **normalized phase_step only** (no global cursor) and/or add `gripper_object_distance` bins; compare EE event order. | |
| H-EE-010 | rejected | **No-cursor rollout ablation:** removing open-loop `cursor` (and MLP progress/phase inputs at inference) lets proprioception drive gripper timing closed-loop, improving EE `event_order_valid` without more demos or vision. | State obs already includes contact, gripper, object−EE offset; cursor forces demo-time indexing that desyncs when EE saturates (see “Why a step counter” below). | Inference-only ablation: zero progress/phase features, do not advance `cursor`. EE MLP seeds 0–2, final grid 72 trials. Pass bar: ≥15 pp `event_order_valid` or ≥10 pp success. | **Failed badly.** Baseline rerun 14/72 success (37.5% event-order). Ablation **0/72** success, **1/72** event-order, **24** `early_close`. Policy trained *with* progress/phase; zeroing them at inference is OOD — not a fair closed-loop fix without retraining. Evidence: `outputs/h_ee_010_no_cursor_ablation.json`. Code reverted. |
| H-EE-011 | rejected | **Gripper distance gate at rollout:** suppress close commands until `gripper_object_distance` ≤15 mm. | Proprioception exposes distance and the guard can remove structurally early close commands without retraining. | Symmetric guarded diagnostic on the selected byte-identical models. Pass bar: ≥10 pp success or ≥15 pp event-order vs raw. | **Rejected.** The guard suppressed 507 EE close steps and removed 2 early-close trials, but success stayed 28/120 and event order stayed 38/120; pre-close contact stayed 741 and reopen events rose 196→198. |
| H-EE-012 | rejected | **Retrain without progress/phase + explicit distance:** cursor-free state BC improves the shared action-space comparison. | H-EE-010 showed inference-only stripping was unfair, so the model was retrained for the new input contract. | Registered validation, five seeds, same architecture/data for both action spaces. | **Rejected.** EE changed 31/120→32/120 while event validity declined and early-close rose 5→24; joint collapsed 53/120→18/120. The shared legacy contract was selected before final access. |
| H-EE-013 | rejected | **Env-derived phase at train+rollout:** feed MLP progress/phase from live contact, lift, and distance bins instead of `cursor` / demo phase lengths. | Extends H-EE-001/009 without zeroing phase features (H-EE-010 lesson). Maps env state → phase index + phase_progress with the same contract at train and eval. | Implement `env_derived_phase` temporal mode; train/evaluate under protocol-v2 validation (5 seeds × 24). Compare success, event-order, physical-sanity vs registered legacy. | **Rejected.** Same registered hyperparams (128×128, 300 ep, 5 seeds). EE 19/120 vs legacy 31/120; event-order 20/120 vs 38/120; early_close 40 vs 5; reopen 375 vs 190. Joint 23/120 vs legacy 53/120 (better than cursor-free 18/120 but still far below legacy). Combined 42/240 vs legacy 84 and cursor-free 50. Preclose contact fell (EE 15 vs 583) but early-close/reopen worsened — not a timing fix. Evidence: `outputs/h_ee_013_env_phase_validation/state_bc_summary.json`. Report: `reports/2026-07-08-h-ee-013-env-phase.md`. |
| H-EE-014 | confirmed | **Hybrid NN gripper + MLP arm:** nearest-neighbor on `MATCH_FEATURE_INDICES` for gripper dim only; MLP for Cartesian dims. | NN matches state-local demo timing; residual under H-EE-021 is reopen/hold after close, not early-close. Note: current match indices are object pose + contact/lift, not object_minus_ee. | Compositor A1 under `global_gripper` (primary); protocol-v2 validation; raw policy; compare EE success, EO, reopen, worst seed vs H-EE-021 MLP-only global baseline. Full plan: `prompts/h-ee-014-nn-gripper-plan.md`. | **Confirmed on validation.** Hybrid A1 under `global_gripper` vs pure-MLP global baseline: EE 49→**62**/120 (+13), EO 60→**79** (+19), reopen **155→0**, worst seed 4→**9**/24 (+5). Joint 76→**97**/120 (no collapse; +21). All pre-registered pass bars met. Residual EO failures: **missing_lift (30) + early_close (11)** — reopens eliminated (flips=1.0 success and fail). Early-close rose 2→11 (secondary). Phys flat at 68. Still short of research parity frontier (84/90/100, worst≥12). Final not accessed. Evidence: `outputs/h_ee_014_nn_gripper_global_validation/`. |
| H-EE-015 | untested | **Scripted gripper schedule + learned arm:** expert/task defines when close is legal; policy outputs arm deltas only (or gripper overridden by FSM). | Shrinks ML problem to motion; controller-first ethos. Event order becomes structural. Under hybrid, early-close is only 11/120 — FSM would mostly be an **arm upper-bound diagnostic**, not the main path to learned EE parity. | Frozen-arm inference-only oracle FSM diagnostic; eval gates unchanged. Execution prompt: `prompts/h-ee-015-fsm-arm-upper-bound.md`. | **Sub-phase SP2b fallback** if named match-set fails. Sidesteps learned gripper timing — do not redefine the EE vs joint learned comparison without explicit scope change. |
| H-EE-016 | deprioritized | **Close-phase demo oversampling:** weight loss or duplicate samples around `grasp_align` → `close_gripper` boundary. | Failures cluster at phase transitions; uniform stride under-represents close timing. | 2–5× sample weight on close phases; one seed EE retrain; event-order rate vs baseline. | **Deprioritized.** H-EE-008 already applies 10× loss at the transition; naïve oversampling mostly repeats that intervention. Prefer H-EE-019 if transition is the true causal factor. |
| H-EE-017 | untested | **Short observation history (stack or tiny GRU):** approach→close or lift needs temporal context beyond single-step state. | Reopen residual is **gone** under hybrid (H-EE-014). History is only still plausible for early-close approach velocity or arm thrash context — not as a reopen fix. | 3–5 step joint/EE/gripper history or 1-layer GRU under frozen hybrid gripper; protocol-v2 validation. | **Deprioritized vs SP1/SP2.** Select only if match-set fails and early-close remains, or A2 fails and thrash looks non-Markov. |
| H-EE-018 | deprioritized | **Adaptive gripper-gradient balancing:** keep gripper importance stable relative to arm error rather than fixed scalars. | If global 5× helps but seed reliability remains weak, fixed scalar weighting may over/under-penalize as arm residual shrinks. | Train with adaptive per-batch gripper scale under protocol-v2 validation; compare worst-seed success. | **Deprioritized.** Gripper is now NN under hybrid; adaptive gripper MSE is mostly obsolete for A1. |
| H-EE-019 | deprioritized | **Narrow close-boundary curriculum:** focus learning on approach→close boundary without overweighting the long closed/lift segment. | If transition 10× is the main H-EE-008 driver, broad close-phase overweight may waste capacity on already-closed steps. | Curriculum / boundary-only weight schedule; same raw validation contract. | **Deprioritized.** Transition-only is weak on EE (+7/120); not the main driver. |
| H-EE-020 | untested | **Targeted boundary demonstrations:** add demos only around measured ambiguous close states. | Off-support at the transition is a data coverage problem, not a generic “more demos” problem. | Collect/script demos only for diagnosed ambiguous close states; retrain selected loss contract. | **Low priority post-014.** Residual is missing_lift thrash + impulse almost-wins + vertical early-close — not a clear off-support reopen pocket. |
| H-EE-021 | confirmed | **H-EE-008 is a causal clue, not the answer:** isolate whether 008 gains come from global gripper 5×, transition 10×, or both. | 008 changed two weights at once; EE 50/120 still far behind weighted joint 84/120. | Frozen four-profile matrix under one commit: `uniform` (1/1), `global_gripper` (5/5), `transition_gripper` (1/10), `combined_h_ee_008` (5/10). Same demos, obs, gains, MLP, schedule, seeds, protocol-v2 validation; both action spaces; raw policy only. | **Confirmed causal split.** EE success: uniform 31, global **49**, transition 38, combined **50** (deltas +18 / +7 / +19). **Global 5× is the main EE driver; transition alone is weak; combined is not super-additive for EE.** Global beats combined on EO (60 vs 55), worst seed (4 vs 2), early_close (2 vs 5). Joint still wants combined (84/90/100 vs global 76/84/92). Residual under all profiles: **reopen/gripper flips**, not early-close. No EE profile meets frontier. Final not accessed. Evidence: `outputs/h_ee_021_loss_decomposition/`. Diagnosis: `h_ee_021_global_vs_combined_diagnosis.json`. |
| H-EE-022 | rejected | **Named NN match-set with relative EE–object features** reduces hybrid early-close without reintroducing reopens. | Hybrid A1 match set is absolute object pose + contact/lift only — not `object_minus_ee_*`. Early-close trials close at ~18 mm vs ~7 mm legal; all 11 early-closes are `vertical_pregrasp`. | Secondary match contract (e.g. `match_relative_ee`) under frozen hybrid A1 + `global_gripper`; protocol-v2 validation; do **not** silently change the historical default match set. Full residual plan: `prompts/post-h-ee-014-residual-plan.md` SP1. | **Rejected.** Frozen hybrid weights re-eval under `match_relative_ee`: early_close **11→11** (bar ≤5); success 63; reopen 1; worst 8. Historical match retained as default. Evidence: `outputs/h_ee_022_match_relative_ee_validation/h_ee_022_comparison.json`. |
| H-EE-023 | rejected | **A2 arm-only MLP loss under frozen NN gripper** improves lift/path after hold is solved. | Hybrid A1 still trains MLP on gripper MSE even though rollout gripper is NN. Missing-lift EO fails (30) show thrash (joint-limit ~965 vs ~92 on success) and weak lift (~5.8 mm mean). | Mask gripper residual / weight 0 in MLP train; NN supplies gripper; same hybrid rollout; compare missing_lift bucket, success, worst seed, reopen stays ~0. Plan SP2. | **Rejected.** Full retrain A2 under historical match: EE 67/120 (+5, bar ≥72 or missing_lift ≤~21); missing_lift_eo **32** (worse); worst seed **6**; reopen 0; joint 89 (≥87). Do not freeze A2. Keep A1 hybrid baseline. Evidence: `outputs/h_ee_023_arm_only_mlp_validation/h_ee_023_comparison.json`. |
| H-EE-024 | diagnosed | **Impulse almost-wins are a path/force residual, not a gripper-timing residual.** | Under hybrid EE, 15 contact_dynamics fails are EO-valid **and** lifted **and** retained; 13/15 exceed impulse gate (mean ~11.4 vs thr 9.0). | Visual + telemetry diagnosis first (SP0); only then consider softer close / approach path changes or expert label changes — never relax the impulse gate as “progress.” Plan SP3. | **Diagnosed; no train yet.** 10 impulse-only, 3 impulse+xy, 1 force-only, 1 xy-only; mean impulse 11.44 vs success 6.55. Prolonged contact integral, not kN impact. Decision `no_train_yet`. Evidence: `outputs/h_ee_024_impulse_diagnosis/`. |
| H-JNT-001 | rejected | **The same distance guard materially improves joint readiness.** | Raw joint has 11 early-close trials, so the symmetric guard has a plausible target. | Same registered guarded diagnostic and byte-identical selected models. | **Rejected as a readiness fix.** Success moved 51/120→55/120 and event order 65/120→70/120; physical sanity fell 80/120→76/120 and the worst seed remained 2/24. |

---

## Post-H-EE-014 residual program (sub-phases)

**Frozen baseline contract:** hybrid A1 (`hybrid_nn_gripper_mlp`) + `global_gripper` +
protocol-v2 **validation** + legacy_progress_phase + raw policy (no shield/FSM).
**Baseline numbers:** EE 62/120, EO 79, phys 68, reopen 0, worst 9/24; joint 97/120.
**Evidence:** `outputs/h_ee_014_nn_gripper_global_validation/`.
**Full execution plan:** `prompts/post-h-ee-014-residual-plan.md`.
**Goal-mode contract (pasteable, with real bars):** `prompts/goal-post-h-ee-014-residual.md`.

H-EE-014 closed the **reopen/hold** chapter. Residuals are **three different mechanisms**
— do not treat them as one “more gripper MSE” problem.

### Residual anatomy (EE hybrid, 120 trials)

| Residual class | n | Mechanism (telemetry) | Primary sub-phase |
|----------------|--:|-----------------------|-------------------|
| Missing lift (EO, contact, no lift) | **30** | Mean max lift ~5.8 mm; joint-limit thrash ~965 vs ~92 on success | **SP2** H-EE-023 A2 arm-only |
| Contact dynamics “almost-win” | **15** | EO+lift+retain true; **13/15** over impulse thr (~11.4 vs 9) | **SP0/SP3** H-EE-024 |
| Early close | **11** | Close ~18 mm; **all** `vertical_pregrasp` | **SP1** H-EE-022 match-set |
| Other (grasp geo / jaw model) | 2+2 | Small | diagnose in SP0 |

Also: EE success **high_staged 39/60** vs **vertical 23/60** — approach bias, not the old yaw story.

### Sub-phase order

| SP | Name | Hypothesis / work | Status | What success looks like | What it does **not** fix |
|----|------|-------------------|--------|-------------------------|--------------------------|
| **SP0** | Visual + residual freeze | Render missing_lift thrash, impulse almost-win, early_close vertical; optionally declare frontier dual-bar | **complete** | Clips + one-page residual freeze in notes | Any metric alone |
| **SP1** | Named relative match-set | **H-EE-022** | **rejected** | early_close ↓, reopen stays ~0, EE success not worse | missing_lift thrash, impulse |
| **SP2** | Arm-only MLP under NN gripper | **H-EE-023** (A2) | **rejected** | missing_lift bucket ↓ or EE +10; worst seed ↑; reopen ~0 | early-close (unless side effect), impulse style |
| **SP2b** | FSM gripper diagnostic | **H-EE-015** | untested fallback (early_close still 11) | Arm upper bound if SP1 fails | Learned gripper comparison purity |
| **SP3** | Impulse almost-wins | **H-EE-024** | **diagnosed** (no train) | Mechanism confirmed; only then path/demo change | Timing metrics as substitute |
| **SP4** | Parallel cheap probes | **H-EE-007** labels; H-EE-002 frozen gain | **complete; both rejected** | Label/gain causality yes/no | Raw labels failed replay; lower EE gains reduced exposure but collapsed success/lift |
| **SP5** | Selection / final | only if bars + worst-seed/phys approach frontier | blocked | Human-approved single final access | Auto-open because validation improved |
| **SP6** | Joint pick-place track | joint hybrid is strong (97/120) | optional | Task expansion without claiming EE fixed | EE parity |
| **SP7** | Phase 6b vision | **H-VIS-001** | blocked | New temporal/gripper contract required | Fixing residual thrash by RGB alone |

### Do not do next

| Anti-pattern | Why |
|--------------|-----|
| More transition 10× / H-EE-016 / H-EE-019 | Reopen solved; transition was weak driver |
| Pure gripper MSE reweight under hybrid | Gripper hold is already NN |
| Open final on current EE 62 / worst 9 / phys 68 | Short of frontier and seed bar |
| Phase 6b vision BC now | Hides arm/force residual behind RGB |
| Distance-guard shield (H-EE-011 style) | Rejected; early-close is small pocket |
| Relax impulse/phys gates | 15 almost-wins are real physics-audit signal |
| Lower EE gain again or add a cap rescue under H-EE-002 | Registered 0.875/0.750 collapsed success; a cap is a new hypothesis, not a rescue |
| Combine SP1+SP2 in one first train | Confounds match-set vs arm-only causal claims |
| Redefine EE vs joint learned comparison via FSM without labeling diagnostic | H-EE-015 purity risk |

---

## Improvement ideas backlog (tiered, post-H-EE-014)

Ideas below complement the hypothesis table. Prefer **residual-matched** experiments
under the frozen hybrid contract. Full sub-phase program above.

### Active — residual-matched (do these)

| Tier | ID / SP | Idea | Retrain? | Effort | Targets residual |
|------|---------|------|----------|--------|------------------|
| 0 | SP0 | Visual review of 3 residual classes | No | Low | **done** |
| 1 | H-EE-022 / SP1 | Named match-set + relative EE–object | Fit NN only | Low–medium | **rejected** (early_close 11) |
| 1 | H-EE-023 / SP2 | A2 arm-only MLP under frozen NN gripper | Yes | Medium | **rejected** (missing_lift worse) |
| 1 | H-EE-024 / SP3 | Impulse almost-win mechanism | Maybe later | Low then medium | **diagnosed; no train** |
| 2 | H-EE-007 / SP4 | raw `labels` vs `policy_labels` probe | No | **rejected at replay** | Raw transitions are not command-scale executable labels |
| 2 | H-EE-015 / SP2b | FSM gripper + learned arm (diagnostic) | Partial | Medium | early-close upper bound (still open after SP1 reject) |
| 2 | H-EE-002 | EE gain under **hybrid** (causal) | No | **rejected** | 0.875/0.750 collapsed success despite lower exposure; no cap rescue |
| 3 | H-EE-017 | History / tiny GRU (arm or approach only) | Yes | Medium–high | non-Markov after SP1/SP2 reject |
| 3 | SP6 | Joint-only pick-place BC track | Yes | Medium | task expansion (joint hybrid 97 still best) |
| 4 | H-VIS-001 / SP7 | Vision BC with new gripper/temporal contract | Yes | High | blocked until EE comparison viable |

### Confirmed (keep as foundations)

| ID | Idea | Note |
|----|------|------|
| H-EE-008 | Gripper-weighted MSE | Confirmed; causal split H-EE-021 |
| H-EE-021 | Global 5× main EE loss driver | Prefer `global_gripper` for EE hybrid |
| H-EE-014 | Hybrid NN gripper + MLP arm | Confirmed; freeze as baseline |

### Deprioritized / ruled out

| Item | Why |
|------|-----|
| H-EE-003 separate learned gripper head | **Rejected** — EE 50→34 under weighted contract |
| H-EE-010 inference-only no-cursor | **Rejected** — 0/72 success |
| H-EE-011 distance guard | **Rejected** — EE success/event order unchanged on registered diagnostic |
| H-EE-012 cursor-free state MLP | **Rejected** — EE unchanged in practice; joint validation collapsed |
| H-EE-013 env-derived phase | **Rejected** — EE 19/120 and joint 23/120 on validation |
| H-JNT-001 joint distance guard | **Rejected** — +4/120 success, worse physical-sanity rate |
| H-EE-006 yaw-targeted data | **Rejected for current evidence** — registered yaw buckets nearly equal |
| H-EE-016 / H-EE-019 transition oversampling / curriculum | Deprioritized — not main driver; reopen solved |
| H-EE-018 adaptive gripper MSE | Deprioritized — hybrid NN gripper makes this obsolete for A1 |
| More pure gripper MSE reweight | Reopen residual solved by H-EE-014 |
| 10× more demos without a coverage hypothesis | Costly and non-falsifiable |
| Phase 6 vision + same pure MLP gripper / cursor clock | Cameras don't fix hold or thrash by themselves |
| Loosen gates for “better” numbers | Diagnostic only; not a behavior fix |

---

## Research parity frontier (EE catch-up target)

### Legacy frontier (still cited; H-EE-008 combined pure-MLP joint)

| Metric | Target |
|--------|--------|
| Success | 84/120 |
| Event order | 90/120 |
| Physical sanity | 100/120 |
| Worst seed | ≥12/24 |

### Stretched frontier (honest post-014; hybrid joint under global)

Hybrid joint under the same A1+`global_gripper` contract is **97/120, EO 107, phys 103,
worst 15**. If EE “catch-up” means match **current best joint**, the bar is **higher** than
84. Do **not** silently mix the two bars in reports — name which frontier you claim against.

| Metric | Stretched target (hybrid joint) | Current hybrid EE |
|--------|--------------------------------:|------------------:|
| Success | 97/120 | 62/120 |
| Event order | 107/120 | 79/120 |
| Physical sanity | 103/120 | 68/120 |
| Worst seed | ≥15/24 | 9/24 |

Final holdout remains **closed** until one EE configuration is selected on validation
because it improves **aggregate** success **and** worst-seed success (and preferably phys),
under an explicitly named frontier bar.

---

## Hypothesis priority (suggested order)

**Post residual program plus SP4 complete:** best EE remains hybrid A1 62/120.
H-EE-002/007/022/023 rejected; H-EE-024 diagnosed no-train. Next survivors:

1. **SP2b / H-EE-015** — oracle FSM arm upper bound; prompt: `prompts/h-ee-015-fsm-arm-upper-bound.md`.
2. **H-EE-017** — history / tiny GRU only if H-EE-015 leaves a non-Markov residual.
3. **SP3 train path** — only if softer close/approach design is registered (not gate relaxation).
4. **SP6** joint pick-place track optional (joint hybrid 97/120 still best).
5. **SP5 final** — still closed; EE 62/worst 9/phys 68 short of legacy_84.
6. **SP7** Phase 6b blocked; Phase 7 language/VLA is not the next step.
7. **Crossed off:** H-EE-002 lower gain, H-EE-007 raw labels, H-EE-022 match-set, H-EE-023 A2 arm-only, H-EE-016/018/019, pure gripper MSE, distance guards.
8. **Keep frozen:** hybrid A1 + `global_gripper` + historical match + H-EE-008/021 foundations.

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
| 2026-07-08 | H-EE-013 | Env-derived phase temporal mode on protocol-v2 validation (128×128, 300 ep, 5 seeds × 24) | **Rejected** — EE 19/120 (event-order 20, early_close 40) vs legacy 31/120 (38, 5); joint 23/120 vs 53/120; combined 42/240 vs 84. Not a fair open-loop-clock fix under this estimator | `outputs/h_ee_013_env_phase_validation/state_bc_summary.json`, `reports/2026-07-08-h-ee-013-env-phase.md` |
| 2026-07-08 | H-EE-008 | Gripper-weighted / close-phase MSE (5× / 10×) on protocol-v2 validation, legacy temporal | **Confirmed** — EE 50/120 (+19), event-order 55/120 (+17); joint 84/120 (+31). Preclose EE 583→50. Seed instability remains. Final not accessed | `outputs/h_ee_008_gripper_weighted_validation/`, `reports/2026-07-08-h-ee-008-gripper-weighted-mse.md` |
| 2026-07-09 | H-EE-003 | Separate 5-D arm MSE + 1-D binary gripper classifier under the H-EE-008 weighted legacy contract, protocol-v2 validation | **Rejected** — binary labels did justify classification, but raw EE fell 50→34/120 with early-close 5→29; joint fell 84→41/120. Lower EE preclose contact (50→0) and constraint exposure did not translate into event-order or success gains. Final not accessed | `outputs/h_ee_003_separate_gripper_head_validation/h_ee_003_comparison.json`, `reports/2026-07-09-h-ee-003-separate-gripper-head.md` |
| 2026-07-09 | H-EE-021 | Four-profile loss decomposition of H-EE-008 (uniform / global 5× / transition 10× / combined 5×10×), protocol-v2 validation, both spaces, 5 seeds | **Confirmed causal split** — EE gains almost entirely from global 5× (31→49); transition alone 38; combined 50. Residual is reopen/flips not early-close. Joint still needs combined for 84/120. No EE frontier hit. Final closed | `outputs/h_ee_021_loss_decomposition/h_ee_021_comparison.json`, `h_ee_021_global_vs_combined_diagnosis.json`, `reports/2026-07-09-h-ee-021-loss-decomposition.md` |
| 2026-07-09 | H-EE-014 | Hybrid NN gripper + MLP arm A1 compositor under `global_gripper`, protocol-v2 validation, 5 seeds × 24, both spaces | **Confirmed** — EE 49→62/120, EO 60→79, reopen 155→0, worst 4→9; joint 76→97. Residual is missing_lift + early_close (not reopen). Final not accessed | `outputs/h_ee_014_nn_gripper_global_validation/h_ee_014_comparison.json`, `h_ee_014_diagnosis.json`, `reports/2026-07-09-h-ee-014-nn-gripper.md` |
| 2026-07-09 | SP0 residual visual | Three residual class clips from hybrid EE models | **Complete** — missing_lift thrash, impulse almost-win, early_close vertical frozen with telemetry | `outputs/h_ee_014_residual_visual_review.md`, `outputs/h_ee_014_residual_clips/` |
| 2026-07-09 | H-EE-022 | Named `match_relative_ee` on frozen hybrid A1 weights (no MLP retrain), protocol-v2 validation | **Rejected** — early_close 11→11 (bar ≤5); success 63; reopen 1; worst 8. Historical match retained | `outputs/h_ee_022_match_relative_ee_validation/h_ee_022_comparison.json` |
| 2026-07-09 | H-EE-023 | A2 arm-only MLP under frozen NN gripper + historical match + `global_gripper` name, full 5-seed train | **Rejected** — EE 67/120 (+5, not ≥72); missing_lift_eo 32 (worse); worst 6; reopen 0; joint 89. Keep A1 baseline | `outputs/h_ee_023_arm_only_mlp_validation/h_ee_023_comparison.json` |
| 2026-07-09 | H-EE-024 | Impulse almost-win telemetry + visual diagnosis | **Diagnosed; no train** — 13/15 impulse over thr; mean 11.44 vs success 6.55; path/force residual | `outputs/h_ee_024_impulse_diagnosis/h_ee_024_impulse_diagnosis.json` |
| 2026-07-13 | H-EE-007 | Raw observed EE labels vs reconstructed executable EE labels; 48,112-sample audit then 18-demo replay gate | **Rejected at replay; no train** — control 18/18, raw 0/18 success and 0/18 event-order. Raw arm labels were ~30× smaller on mean L2; gripper labels were identical. Final not accessed | `outputs/h_ee_007_label_contract_probe/h_ee_007_comparison.json`, `reports/2026-07-13-h-ee-007-label-contract.md` |
| 2026-07-13 | H-EE-002 | Frozen H-EE-014 hybrid A1, inference-only EE arm gain sweep (1.0/0.875/0.750), protocol-v2 validation, 5 seeds × 24 | **Rejected** — 1.0 reproduced 62/120 exactly; 0.875 fell to 5/120 and 0.750 to 0/120. Lower failure-conditioned constraint exposure did not recover a single prior missing-lift trial; it lost 57/62 baseline successes and increased missing-lift/early-close. No training, cap rescue, final, or Phase 6b access | `outputs/h_ee_002_hybrid_gain_sweep/h_ee_002_gain_sweep_summary.json`, `evidence/h_ee_002_hybrid_gain_sweep.json`, `reports/2026-07-13-h-ee-002-hybrid-gain.md` |

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
(**H-EE-010**), the distance guard (**H-EE-011**), cursor-free retraining
(**H-EE-012**), and env-derived phase (**H-EE-013**) are **rejected**. Remaining
state-only candidates that advanced the ladder: weighted gripper objective (**H-EE-008**,
causal split **H-EE-021**) and hybrid NN gripper (**H-EE-014 confirmed** — reopens gone).
Remaining residuals under hybrid are **missing_lift thrash**, **impulse almost-wins**, and
**early-close (vertical)** — see **Post-H-EE-014 residual program** sub-phases SP0–SP7 and
`prompts/post-h-ee-014-residual-plan.md` (H-EE-022/023/024).

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
