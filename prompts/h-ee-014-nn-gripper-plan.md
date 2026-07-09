# H-EE-014 Plan — NN Gripper + MLP Arm (Fresh-Thread Brief)

**Status:** executed — **confirmed on validation** (2026-07-09)
**Branch:** `research/h-ee-014-nn-gripper`
**Evidence:** `outputs/h_ee_014_nn_gripper_global_validation/`, `reports/2026-07-09-h-ee-014-nn-gripper.md`
**Do not open final holdout** (frontier still not met: EE 62 vs 84).
**Do not start Phase 6b / vision BC.**

This document is self-contained. A new agent should be able to execute from it plus `AGENTS.md` / `researchnotes.md`.

---

## 0. One-paragraph intent

We are testing whether **gripper timing should be state-local lookup (nearest neighbor)** while the **arm stays a learned MLP**. H-EE-008 improved pickup by reweighting gripper MSE, but H-EE-021 showed that almost all EE gain was from **global 5× gripper weight**, not transition 10× — and the residual failure under every loss profile is **reopen / gripper oscillation after close**, not early-close. A separate learned binary gripper head (H-EE-003) already failed. H-EE-014 asks a different question: does **demo-matched local state** for the gripper command stop reopens better than a global MLP phase-clock prediction?

---

## 1. Where we are (facts only)

### Research ladder (do not collapse layers)

| Layer | EE | Joint | Meaning |
|-------|---:|------:|---------|
| Scripted expert | 36/36 | 36/36 | Task feasible |
| Label replay | 18/18 | 18/18 | Labels executable |
| Raw uniform MSE validation | 31/120 | 53/120 | Registered legacy baseline |
| H-EE-008 combined 5×/10× validation | 50/120 | **84/120** | Confirmed loss reweight helps |
| H-EE-021 global 5× only | **49/120** | 76/120 | Main EE driver of 008 |
| H-EE-021 transition 10× only | 38/120 | 68/120 | Weak alone |
| Research parity frontier (target, not release) | want EE → | **84/120 success, 90 EO, 100 phys, worst seed ≥12/24** | Defined as weighted joint under 008 |

Final holdout (protocol-v2 `final`) remains **closed**.

### H-EE-021 causal result (why 014 is next)

Frozen profiles, same commit, protocol-v2 validation, 5 seeds × 24, raw MLP, no shields:

| Profile | EE | EO | Worst seed | Joint |
|---------|---:|---:|-----------:|------:|
| uniform | 31 | 38 | 0 | 53 |
| global_gripper (5/5) | **49** | **60** | **4** | 76 |
| transition_gripper (1/10) | 38 | 41 | 2 | 68 |
| combined_h_ee_008 (5/10) | **50** | 55 | 2 | **84** |

**Interpretation:**

1. **Global 5× is the main EE driver** of H-EE-008 (+18 success). Transition alone is weak (+7). Combined ≈ global on EE success (+19), not super-additive.
2. For EE reliability metrics, **global slightly beats combined** (EO 60 vs 55, worst seed 4 vs 2, early_close 2 vs 5).
3. For joint frontier, **combined still wins** (84/90/100).
4. Residual under all profiles: **reopen-dominated event-order failure**. Successes have ~1 gripper flip; failures ~5. Early-close is rare.
5. H-EE-005 is **partially confirmed** as the dominant residual.

Evidence:

- `outputs/h_ee_021_loss_decomposition/h_ee_021_comparison.json`
- `outputs/h_ee_021_loss_decomposition/h_ee_021_global_vs_combined_diagnosis.json`
- `reports/2026-07-09-h-ee-021-loss-decomposition.md`
- `researchnotes.md` (H-EE-021 confirmed; H-EE-014 next)

### Explicitly rejected / deprioritized (do not redo as default)

| ID | Status | Note |
|----|--------|------|
| H-EE-003 | rejected | Separate **learned** binary gripper head under 5×/10× made EE worse (50→34) |
| H-EE-010–013 | rejected | Cursor ablations / env phase / distance guard not fixes |
| H-EE-016 | deprioritized | Oversampling mostly repeats transition 10× |
| H-EE-018 | deprioritized for now | Adaptive loss scalars after global already works; residual is reopen structure |
| H-EE-019 | deprioritized | Transition is not the main driver |
| Lower EE gain | deprioritized | Saturation not supported as simple cause |

### Why H-EE-014 is not H-EE-003

| | H-EE-003 (rejected) | H-EE-014 (this plan) |
|--|---------------------|----------------------|
| Gripper source | Learned sigmoid head, BCE | **Nearest-neighbor lookup** on demo features |
| What it tests | Classification vs regression loss | **State-local label matching** vs global MLP timing |
| Arm | Same MLP | Same MLP (arm dims) |
| Failure mode targeted | Mode mixing in one head | **Reopen / wrong hold after close** |

---

## 2. Hypothesis

**H-EE-014 claim (falsifiable):**

> Under the same protocol-v2 validation contract and a fixed loss profile, replacing only the gripper action dimension with a nearest-neighbor prediction (matched on `MATCH_FEATURE_INDICES`) while keeping the MLP arm deltas will **reduce reopen events and raise EE event-order / success / worst-seed** relative to the pure MLP baseline with the same arm training.

**Mechanism:**

- Residual failures reopen after close: MLP gripper is open-loop / phase-clock correlated and not reliably state-local.
- NN gripper copies demo gripper commands from nearby object/contact state, which should favor **hold-closed** when demos are closed in similar contact geometry.
- Arm motion still needs generalization → keep MLP.

**If false:** reopens persist or success does not improve → gripper problem is not “wrong local label match” (try H-EE-017 history / H-EE-015 FSM gripper next).

---

## 3. Frozen experimental contract

### Must match H-EE-021 / registered validation

| Knob | Value |
|------|-------|
| Protocol | v2 (`configs/phase5_evaluation_protocol_v2.json`) |
| Split | **validation only** |
| Temporal features | `legacy_progress_phase` |
| Policy base | MLP `128 128`, 300 epochs, batch 1024, lr 1e-3, wd 1e-5 |
| Seeds | 0 1 2 3 4 |
| Action spaces | **both** `ee_tool_delta` and `joint_delta` |
| Action gain | 1.0 |
| Label source | `policy_labels` |
| Shield / guard / FSM | **off** |
| Eval limit | none (full 24 trials/seed) |
| Final | **not accessed** |

### Loss profile choice (important)

Two legitimate options — pick **one primary**, optionally one diagnostic:

| Mode | Loss profile | When to use |
|------|--------------|-------------|
| **Primary (recommended)** | `global_gripper` (5×/5×) | Best current EE reliability baseline from H-EE-021 |
| Shared-parity diagnostic | `combined_h_ee_008` (5×/10×) | Only if you need joint comparison against the 84/120 frontier in the same run |

**Do not** invent a new loss profile for 014.
**Do not** use uniform MSE as the only arm of the test (that confounds architecture with the already-confirmed loss fix).

Primary baseline for comparison:

- EE: H-EE-021 `profile_global_gripper` → 49/120, EO 60, worst 4, reopen 155
  Path: `outputs/h_ee_021_loss_decomposition/profile_global_gripper/`
- If using combined: H-EE-021 `profile_combined_h_ee_008` → EE 50 / joint 84
  Path: `outputs/h_ee_021_loss_decomposition/profile_combined_h_ee_008/`

You may **reuse byte-identical MLP weights** from those dirs for a pure compositor diagnostic, but a full train+eval under the hybrid is still required for the registered claim (because training dynamics may change if arm-only loss is used — see implementation note below).

### Research selection metrics (primary)

1. EE success / 120
2. EE `event_order_valid` / 120
3. EE total `reopen_events` and mean `gripper_command_flips`
4. EE worst-seed successes / 24
5. EE `physical_sanity_pass` / 120
6. Early-close, preclose contact (secondary)
7. Joint metrics (must not collapse if shared comparison is claimed)
8. Constraint exposure (joint-limit / infeasible rates) as **telemetry only**, never a substitute for success

### Pre-registered pass bars (write into comparison JSON before interpreting)

Against the matched pure-MLP baseline under the same loss profile:

| Bar | Pass condition |
|-----|----------------|
| EE success | ≥ **+10 / 120** vs baseline |
| EE event-order | ≥ **+12 / 120** vs baseline |
| EE reopen | ≤ **−20%** relative total reopen events **or** clear drop in fail-with-reopen rate |
| EE worst seed | **≥ +3 / 24** absolute on the worst seed (or worst seed ≥ 8/24) |
| Joint non-collapse | joint success ≥ baseline − 10/120 (if both spaces claimed) |

**Select for final only if:** EE success and worst-seed both improve materially **and** reopen falls.
Meeting train MSE alone is **not** evidence.

---

## 4. Implementation plan

### Stage A — Hybrid policy object (no new training algorithm required for compositor)

**Goal:** one policy that at each step outputs:

```text
action[:5]  = MLP arm deltas   (joint or EE tool)
action[5]   = NN gripper command
```

Suggested design (keep modules small):

1. Add `HybridNNGripperMLPPolicy` in `src/svla/state_bc.py` (or thin `src/svla/hybrid_bc.py` if cleaner).
2. Holds:
   - `mlp: MLPBCPolicy`
   - `nn: NearestNeighborBCPolicy` (same demos / action space)
   - `gripper_dim = -1`
3. `predict_with_index(...)`:
   - call MLP → `a_mlp`
   - call NN → `a_nn` (existing NN path already uses `MATCH_FEATURE_INDICES` + optional cursor window)
   - return `concat(a_mlp[:-1], [a_nn[-1]])`
   - nearest distance / index: report NN’s for diagnosis (or both in metrics later)
4. Save/load: save MLP npz + NN npz + small JSON sidecar `hybrid_manifest.json` listing paths and contract.
5. Wire `load_policy` / train script to produce and roll out hybrid.

**Existing NN match features** (`MATCH_FEATURE_INDICES` in `state_bc.py`):

| Index | Name |
|------:|------|
| 18 | `object_x` |
| 19 | `object_y` |
| 20 | `object_z` |
| 28 | `object_lift_from_start` |
| 29 | `gripper_object_contact` |
| 30 | `object_support_contact` |

**Critical honesty note for implementers:**
Current match features are **object pose + contact/lift**, not `object_minus_ee_*`. That is intentional historical NN design. For H-EE-014:

- **Primary run:** keep existing `MATCH_FEATURE_INDICES` unchanged (comparability with existing NN baseline).
- **Optional ablation (only if primary fails or as secondary):** add relative EE–object features and/or `gripper_open` to the match set under a named match-contract string. Do **not** silently change the default.

Also note NN already uses **cursor search window** when `cursor` is passed from `rollout_policy`. Hybrid must pass the same cursor/window as today so it is not an accidental second temporal contract change.

### Stage B — Training recipe

Two acceptable recipes; pick **A1** as default:

#### A1 — Compositor (preferred first)

1. Train MLP exactly as today under `--loss-profile global_gripper` (or combined).
2. Fit NN on the **same demos** (`fit_nearest_neighbor_policy`, default k/temperature unless smoke shows need).
3. At rollout only, replace gripper dim with NN.
4. **No change to MLP loss.** This isolates “gripper source” from “retrain dynamics.”

This is the cleanest causal test of the hypothesis.

#### A2 — Arm-only MLP loss (optional follow-up, not first)

1. Train MLP with gripper weight 0 or freeze gripper labels to constant / mask gripper residual.
2. NN supplies all gripper commands.
3. Only run if A1 improves EO/reopen but arm still fails lift often.

Do **not** start with A2; it confounds two changes.

### Stage C — CLI / runner

Add either:

- flags on `scripts/train_state_bc.py`:
  - `--hybrid-nn-gripper`
  - `--loss-profile global_gripper`
  - existing protocol flags
- or `scripts/run_h_ee_014_hybrid.py` that calls the train pipeline twice (MLP fit + NN fit) and evaluates hybrid.

Recommended output dir:

```text
outputs/h_ee_014_nn_gripper_global_validation/
```

Write:

- `state_bc_summary.json` + manifest
- `h_ee_014_comparison.json` vs frozen baseline numbers
- per-seed diagnosis fields already on `PolicyTrialResult` (close distance, event times, flips, reopens)

### Stage D — Tests (required before full matrix)

1. Hybrid action: arm equals MLP arm; gripper equals NN gripper on a fixed toy/demo batch.
2. Hybrid does not alter MLP weights.
3. Rollout path works for both action spaces (1 seed, `--eval-limit 1` smoke).
4. Manifest records `policy_type=hybrid_nn_gripper_mlp`, loss profile, k, temperature, match feature names.
5. Existing state_bc / loss profile tests still pass.

### Stage E — Full validation matrix

```bash
# Example shape — exact flags should match train_state_bc after wiring
PYTHONPATH=src .venv/bin/python scripts/train_state_bc.py \
  --output-dir outputs/h_ee_014_nn_gripper_global_validation \
  --evaluation-protocol v2 \
  --eval-split validation \
  --temporal-feature-mode legacy_progress_phase \
  --policy-type mlp \
  --hybrid-nn-gripper \
  --loss-profile global_gripper \
  --seeds 0 1 2 3 4 \
  --hidden-sizes 128 128 \
  --epochs 300 \
  --batch-size 1024 \
  --learning-rate 0.001 \
  --weight-decay 1e-5 \
  --stride 1 \
  --max-steps 3200 \
  --action-gain 1.0 \
  --label-source policy_labels
```

Run **both** action spaces (default loop already does).
Expect ~10–20+ minutes (similar to one H-EE-021 profile).

### Stage F — Diagnosis (same style as H-EE-021 Phase 3)

For each seed and aggregate:

1. success / EO / phys / early / preclose / reopen / flips
2. EO failure anatomy buckets: reopen vs early vs missing lift vs missing contact
3. success flips ≈ 1? fail flips still high?
4. worst seed vs best seed
5. pairwise agreement vs pure MLP baseline if trial rows aligned

Write:

```text
outputs/h_ee_014_nn_gripper_global_validation/h_ee_014_diagnosis.json
reports/YYYY-MM-DD-h-ee-014-nn-gripper.md
```

Update `researchnotes.md` + one AGENTS.md verdict bullet.

---

## 5. Decision tree after results

| Outcome | Verdict | Next |
|---------|---------|------|
| EE success + EO + reopen + worst-seed all improve past pass bars; joint not collapsed | **H-EE-014 confirmed (validation)** | Freeze hybrid+loss contract; only then consider final once |
| Reopen falls a lot but success barely moves (lift/physics still fail) | **partial** | Keep hybrid gripper; attack arm/lift (A2 or controller-side), not more loss weight |
| Success up but worst seed still ≤4/24 | **partial** | Do **not** open final; consider H-EE-017 history or seed-robustness next |
| Reopen unchanged / worse | **rejected** | Go to **H-EE-017** (history) or **H-EE-015** (FSM gripper + learned arm) |
| EE improves, joint collapses | **not selectable as shared contract** | Report EE-only diagnostic; do not redefine joint baseline without explicit scope change |

### Still not next by default

- More transition weighting / H-EE-019
- H-EE-016 oversampling
- Lower EE gain
- Vision BC / VLA
- Opening final because “numbers look better on one seed”

---

## 6. Code map (where to touch)

| Path | Role |
|------|------|
| `src/svla/state_bc.py` | `MLPBCPolicy`, `NearestNeighborBCPolicy`, `MATCH_FEATURE_INDICES`, `rollout_policy`, `fit_mlp_policy`, `fit_nearest_neighbor_policy` |
| `src/svla/loss_profiles.py` | Frozen `global_gripper` / `combined_h_ee_008` |
| `scripts/train_state_bc.py` | Multi-seed protocol-v2 train/eval loop, manifests |
| `scripts/run_loss_decomposition.py` | Pattern for comparison JSON (reuse style, not required to extend) |
| `tests/test_state_bc.py` | Extend with hybrid unit tests |
| `configs/phase5_evaluation_protocol_v2.json` | Frozen trial grid — **do not edit** |
| `researchnotes.md` / `AGENTS.md` | Hypothesis status + verdict |
| `reports/` | Large-change review report required |

---

## 7. Implementation order checklist

- [x] Create branch
- [x] Implement hybrid policy + save/load + tests
- [x] Wire CLI (`--hybrid-nn-gripper` or dedicated script)
- [x] Smoke: 1 seed, 1 epoch or full epochs with `--eval-limit 1`
- [x] Full validation under `--loss-profile global_gripper`
- [x] Comparison JSON + diagnosis vs H-EE-021 global baseline
- [x] Update researchnotes / AGENTS / report
- [x] Stop. Do not open final unless selection rule is met (bars met; frontier not — final left closed)

---

## 8. Critical risks (read before coding)

1. **H-EE-003 lesson:** a different gripper *head* is not automatically better. 014 must stay raw closed-loop gated; no success claim from imitation loss.
2. **Cursor coupling:** NN and MLP both see cursor today. Hybrid is not “closed-loop gripper free of open-loop time” unless you explicitly ablate that later. Primary 014 is **compositor**, not a new temporal mode.
3. **Match features may be weak for gripper hold:** object absolute pose + contact may not separate “should stay closed” from “should open.” If primary fails, a named match-set ablation is allowed; silent feature changes are not.
4. **Seed instability is the release blocker:** one good seed (20+/24) already exists under global MLP. 014 must move the **worst** seed, not only the mean.
5. **Joint parity tension:** global loss is better for EE reliability; combined is better for joint frontier. State which contract you froze in the report.
6. **Do not treat constraint rate as success.** Bad seeds show high joint-limit rates co-occurring with reopen; that is telemetry.

---

## 9. Suggested commit / report language

**If confirmed:**

> H-EE-014 confirmed on validation: hybrid NN gripper + MLP arm under `global_gripper` improved EE from 49→X/120, event-order Y→Z, reopen A→B, worst seed C→D. Final not accessed.

**If rejected:**

> H-EE-014 rejected: state-local NN gripper did not reduce reopen-dominated failures (or did not improve success/worst-seed). Residual likely needs history (H-EE-017) or structural gripper FSM (H-EE-015), not further loss reweighting.

---

## 10. Fresh-thread starter prompt (pasteable)

```text
Execute H-EE-014 from prompts/h-ee-014-nn-gripper-plan.md.

Context: controller-first VLA repo; Phase 5 state BC; final holdout closed; Phase 6b blocked.
H-EE-021 showed EE gain from H-EE-008 is mostly global gripper 5×; residual failure is reopen/gripper flips.
Do NOT open final. Do NOT start vision training.

Implement hybrid NN-gripper + MLP-arm compositor under --loss-profile global_gripper,
protocol-v2 validation, 5 seeds, both action spaces, raw policy only.
Compare to outputs/h_ee_021_loss_decomposition/profile_global_gripper/.
Write comparison + diagnosis + researchnotes/AGENTS/report updates.
Be critical: no claims from train loss alone; worst-seed and reopen matter.
```

---

## 11. Bottom line

| Question | Answer |
|----------|--------|
| What are we testing? | State-local NN gripper vs MLP gripper, arm fixed as MLP |
| Why now? | Loss reweight plateaued; residual is reopen after close |
| Baseline | H-EE-021 `global_gripper` MLP (EE 49/120) |
| Success looks like | Higher EE success **and** EO **and** lower reopen **and** better worst seed |
| Failure looks like | Reopen unchanged → go history/FSM, not more MSE tricks |
| Final holdout | Closed until selection rule met |
