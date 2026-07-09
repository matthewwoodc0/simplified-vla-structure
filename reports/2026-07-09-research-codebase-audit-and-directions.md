# 2026-07-09 - Research Codebase Audit and Directions

## 1. Executive Summary

- **Health verdict: sound research foundation, moderate maintainability risk.** The
  controller/task/evidence stack is unusually disciplined, but `pickup_task.py` and
  `state_bc.py` are each about 1,900 lines and active experiments were reconstructed mostly
  from CLI history.
- The actual action-space question is continuous **5-joint delta + gripper** versus continuous
  **3D translation + 2D tool tilt + gripper**. There was no existing tokenizer,
  discretizer, diffusion head, or VLA training loop to audit.
- Baseline verification was **114/114 tests passing outside the sandbox**. The sandbox-only
  run produced eight known CoreGraphics failures in render-backed tests and 106 passes.
- Registered raw final evidence remains negative: joint **51/120**, EE **28/120**. Both fail
  the proposed gates; this does not support the thesis that the simpler EE action space is
  easier to learn.
- The best later validation result is H-EE-014: hybrid NN gripper + MLP arm, joint **97/120**
  and EE **62/120**. EE reopen events fell **155→0**, but missing-lift/constraint thrash,
  impulse failures, early close, and seed stability remain unresolved. Final was not accessed.
- Safe fixes corrected obsolete 61/72-vs-60/72 documentation and removed a hardcoded deleted
  `/var/folders/...` scratch path. No controller, task gate, loss, split, or result was
  silently changed.
- The codebase now has a canonical action registry, versioned experiment configs, a dedicated
  evaluation/provenance package, offline analysis separation, a current `RESULTS.md`, and
  explicit tiny-train/closed-loop smoke tests.
- **Top recommendation: run H-EE-007 first**—a raw-label vs executable-policy-label causal
  probe. It is cheaper and more diagnostic than H-EE-015, and it directly tests whether the
  EE disadvantage comes from the reconstructed label contract rather than representation
  dimensionality.
- Do **not** start FAST/VQ tokens, diffusion/flow heads, vision BC, or language yet. Those
  approaches are meaningful after a stable continuous state-policy baseline, not as a way to
  bury the current label/control failure.

## 2. Immediate Fixes Made

| File | Fix | Why |
|---|---|---|
| `scripts/capture_pick_place_vplan_evidence.py` | Replaced a machine-specific temporary path with ignored `outputs/pick_place_vplan_evidence/` and `--scratch-dir` | The old path referred to a deleted one-off agent directory and was not portable |
| `LEARNING_GUIDE.md` | Replaced obsolete 61/72 vs 60/72 and “vision not started” claims | Those claims contradict registered protocol-v2 evidence and Phase 6a state |
| `README.md` | Corrected the joint action vector from six arm joints to the actual five; added current status and structure | Prevents the project thesis from describing the wrong embodiment/action dimension |
| `src/svla/core/action_space.py` | Centralized decode validation, arm scaling, execution, and status normalization | Removed three duplicated joint-vs-EE execution branches without changing commands |
| `src/svla/eval/*` | Separated protocol and manifest implementation with compatibility imports | Clarifies ownership while preserving every existing import/CLI |
| `analysis/policy_failures.py` | Moved offline analysis behind a compatibility CLI | Analysis can no longer be confused with experiment execution |
| `experiments/configs/*.json` | Captured maintained experiment commands as hashable configs | Reduces silent CLI/config drift and records the exact validation contract |
| `RESULTS.md` | Added a concise result ladder and experiment matrix | Separates final, validation, replay, and infrastructure claims |

## 3. Proposed Behavior-Changing Fixes Awaiting Approval

No behavior-changing research fix was applied during the audit.

| Proposal | Evidence | Why approval is required |
|---|---|---|
| H-EE-007 raw `labels.ee_tool_delta` vs executable `policy_labels.ee_tool_delta` | EE policy labels reconstruct feasible Cartesian motion while joint labels use direct joint-target error; best EE still trails joint 62 vs 97 | Changes the training target and therefore the scientific result |
| H-EE-002 gain/cap sweep under frozen hybrid A1 | EE missing-lift failures show roughly 965 joint-limit steps vs roughly 92 for successes | Changes executed motion and may trade lift for event/physics failures |
| H-EE-015 environment FSM gripper diagnostic | Early close remains 11/120 after H-EE-022; NN already eliminated reopen | Makes gripper timing structural, so it is an arm upper-bound diagnostic—not the same learned comparison |
| Short history or 2–4 step arm chunk | H-EE-023 worsened missing lift despite arm-only loss; residual may be non-Markov | Changes policy input/output contract and must be preregistered |
| Physical split of `state_bc.py` and `pickup_task.py` | Both are ~1,900 lines | High diff risk on top of active uncommitted research; should happen only after freezing the branch |

## 4. Reorganization Summary

### Before

```text
src/svla/                 controller, task, action labels, policies, eval, provenance
scripts/                  launches plus offline analysis
configs/                  one protocol JSON
outputs/                  many ignored runs reconstructed from manifests/CLI history
```

### After

```text
src/svla/core/            canonical action representation registry
src/svla/                 stable controller/task/data/policy implementation
src/svla/eval/            frozen split and provenance implementation
src/svla/experiments/     experiment-config schema and deterministic command rendering
experiments/configs/      versioned maintained launches + honest historical-only records
analysis/                 offline evidence analysis
RESULTS.md                current evidence ladder and untested matrix
scripts/                  execution entry points and compatibility wrappers
```

The action representation is now pluggable at the execution boundary: a representation owns
its vector size, transition encoder, shape/finite checks, arm dimensions, executor, and status
normalization. Adding a new physical representation no longer requires editing the rollout,
replay, and rendering loops separately.

The config system records the config hash in experiment-launch environment metadata. It also
rejects unapproved final-split configs. H-EE-003 is `runnable: false`: its command is preserved,
but the rejected classifier head was removed previously, so claiming current reproducibility
would be dishonest.

## 5. Our Success Stories

| Result | Evidence | What it proves | What it does not prove |
|---|---|---|---|
| Scripted pickup 36/36; pick-place 6/6 | `outputs/phase5_baseline_final/phase5_baseline_v2_aggregate.json` | Controller/task scripting works in the declared MuJoCo envelope | Learned policy readiness or hardware realism |
| Policy-label replay: pickup 18/18 and pick-place 6/6 per action space | Same aggregate | Saved executable labels can drive both action APIs | Closed-loop generalization |
| H-EE-008 weighted MSE: EE 31→50; joint 53→84 | `outputs/h_ee_008_gripper_weighted_validation/` | Loss weighting materially affects gripper/event learning | Which of the two weights caused the gain |
| H-EE-021 decomposition: EE global 49, transition 38, combined 50; joint combined 84 | `outputs/h_ee_021_loss_decomposition/` | Global gripper 5× is the main EE driver; transition 10× is weak | A complete EE solution |
| H-EE-014 hybrid: EE 49→62, event order 60→79, reopen 155→0, worst 4→9; joint 76→97 | `outputs/h_ee_014_nn_gripper_global_validation/` | State-local gripper retrieval solves reopen/hold errors | EE parity, lift/path quality, final readiness |
| Phase 6a RGB data plumbing | `outputs/phase6a_vision_sample/` and its manifest | Deterministic capture, NPZ storage, hashes, validation, preview | Any vision-conditioned policy result |

The most important negative result is also a success in research hygiene: the raw final result
joint 51/120 vs EE 28/120 was not hidden by replay, shields, legacy grids, or attractive videos.

## 6. Literature Landscape

| Direction | Primary-source finding | Relation to this repo | Implication |
|---|---|---|---|
| Redundancy-aware action spaces | [Mazzaglia et al.](https://arxiv.org/abs/2406.04144) argue that task space can be data-efficient but under-controls the full configuration, and introduce EE-plus-redundancy variants | The repo’s EE failures include joint-limit thrash and missing lift; “lower-dimensional is better” may be too crude | Test label/control causality before inventing tokens; a small redundancy variable is a later candidate |
| Action chunking | [ACT](https://arxiv.org/abs/2304.13705) predicts action sequences to reduce compounding error and reports 80–90% on six real tasks with short demonstrations | EE missing-lift behavior may need short temporal context, but current data is single expert mode | Screen a 2–4 step state-action chunk only after label/gain probes |
| Diffusion actions | [Diffusion Policy](https://roboticsproceedings.org/rss19/p026.html) models multimodal action sequences and reported a 46.9% average improvement over prior methods across its benchmark | Current scripted data is not meaningfully multimodal and the repo is NumPy-only | Diffusion is unjustified compute/complexity now; reconsider with varied demonstrations |
| Flow matching | [π0](https://arxiv.org/abs/2410.24164) uses a VLM-backed flow-matching action expert across diverse robot platforms | Demonstrates a scalable continuous alternative to discrete tokens | Not a substitute for fixing a one-task state-BC contract; keep as Phase 7 context |
| Frequency action tokens | [FAST](https://arxiv.org/abs/2501.09747) compresses 1-second action chunks in DCT space; naïve per-dimension/time binning degrades at high frequency | Directly relevant only after this repo has stable chunks and explicit action normalization | If tokenization is tested later, compare reconstruction error and closed-loop gates; do not start with naïve bins |
| Learned discrete latent actions | [VQ-BeT](https://arxiv.org/abs/2403.03181) uses hierarchical vector quantization for multimodal continuous behavior and reports gains across seven environments | Could model multiple motion modes, which this deterministic demo set does not contain | Collect/control real modes first; otherwise codebook learning is decoration |
| Video latent actions | [LAPA](https://arxiv.org/abs/2410.11758) learns VQ latent actions between frames, then maps them to robot actions with limited labeled data | Phase 6a provides frames, but only one task and a small scripted distribution | Defer until multiple behaviors/views exist and state policies are stable |
| Language action hierarchy | [RT-H](https://arxiv.org/abs/2403.01823) predicts intermediate language motions before low-level actions | Supports a future phase/skill hierarchy | Language is currently non-identifying because there is one pickup behavior |
| Cross-embodiment data | [Open X-Embodiment](https://arxiv.org/abs/2310.08864) standardizes data from 22 robots and shows positive transfer with RT-X | Highlights the importance of explicit embodiment/action contracts | Useful later; current single SO-101 simulator cannot test cross-embodiment claims |

## 7. Ranked Research Proposals

### 1. H-EE-007 — executable-label contract probe (recommended)

- **Hypothesis:** EE underperformance is materially caused by reconstruction noise/bias in
  `policy_labels.ee_tool_delta`, not by the EE representation alone.
- **Why now:** H-EE-014 solved gripper reopen but EE remains 35 successes behind joint; the
  redundancy-aware literature warns that task-space mappings can lose configuration-relevant
  information. This probe directly tests the mapping before adding model capacity.
- **Minimal experiment:** (a) replay raw vs policy EE labels on the same 18 scripted pickup
  trajectories; (b) measure per-phase vector disagreement, clipping, and gate outcomes; (c) only
  if replay is clean, train one seed under raw EE labels on protocol-v2 validation. Joint stays
  unchanged as a control. Expected compute: low for replay; moderate for one seed.
- **Success:** raw-label replay remains executable and one-seed EE success/event order improves
  by at least 5/24 without worse physical sanity or constraint exposure.
- **Kill:** raw labels fail replay/gates, or one seed changes by fewer than 3 successes with no
  clear constraint reduction.
- **Priority / effort:** P0 / low–medium.

### 2. H-EE-002 — hybrid EE gain/cap causal sweep

- **Hypothesis:** missing-lift EE failures are caused by closed-loop target accumulation and
  constraint thrash; a preregistered smaller gain/cap improves lift and worst-seed success.
- **Why now:** missing lift is the largest residual bucket, while H-EE-023 arm-only loss made it
  worse. Redundancy-aware control work suggests the mapping/controller boundary—not just loss—can
  dominate task-space behavior.
- **Minimal experiment:** frozen hybrid A1 weights; inference-only gains 0.75/0.9/1.0 on one seed,
  with success, missing lift, joint-limit exposure, impulse, and event-order gates. Promote one
  setting to five seeds only if the preregistered screen passes. Low compute for screen; medium
  for confirmation.
- **Success:** missing-lift failures fall at least 30%, success rises at least 5/24, and physical
  sanity/early-close do not regress.
- **Kill:** constraint exposure falls but lift/success do not improve, showing saturation is a
  symptom rather than cause.
- **Priority / effort:** P1 / medium.

### 3. H-EE-015 — FSM gripper as an arm upper-bound diagnostic

- **Hypothesis:** with legal close/open transitions made structural, the learned EE arm reaches
  at least the legacy 84/120 frontier; if not, gripper timing is no longer the main blocker.
- **Why now:** H-EE-014 removed reopen, H-EE-022 did not reduce early close, and RT-H-style
  hierarchies support separating phase intent from low-level execution. This is diagnostic, not
  a fair learned-gripper comparison.
- **Minimal experiment:** frozen arm policy plus environment-derived legal-transition FSM on
  validation; no gate relaxation and no final access. Moderate compute, no new large model.
- **Success:** EE ≥84/120, physical sanity materially improves, and missing-lift falls.
- **Kill:** EE remains below 75/120 or impulse/missing-lift dominates; stop working on gripper.
- **Priority / effort:** P2 / medium.

### 4. Short-history / short-chunk state policy

- **Hypothesis:** EE lift control is partially non-Markov under the current single-frame state;
  3–5 frame history or a 2–4 step arm chunk reduces thrash and seed variance.
- **Why now:** H-EE-023’s static arm loss failed, while ACT shows action sequences can reduce
  compounding error. This is the smallest legitimate bridge toward chunk-based methods.
- **Minimal experiment:** tiny MLP with stacked recent state/action features or fixed 2–4 step
  arm output; keep hybrid gripper frozen; one seed before any five-seed run. Medium compute.
- **Success:** at least +5/24 success and ≥30% missing-lift reduction with no physics regression.
- **Kill:** train loss improves without closed-loop gates, or worse action smoothness/impulse.
- **Priority / effort:** P3 / medium–high.

### 5. Deferred tokenizer benchmark (FAST vs VQ vs naïve bins)

- **Hypothesis:** once action chunks and multiple behavior modes exist, compression-aware or
  learned tokenization preserves closed-loop behavior better than per-dimension bins.
- **Why now:** FAST and VQ-BeT make this the relevant future comparison; internal evidence does
  not yet justify executing it.
- **Minimal experiment:** reconstruction-only study on fixed action chunks, then closed-loop
  only if reconstruction and event-boundary fidelity pass. High effort relative to current repo.
- **Success:** lower reconstruction error at matched token budget and no gate loss after decode.
- **Kill:** codebook collapse, boundary/gripper corruption, or no advantage over continuous chunks.
- **Priority / effort:** P4 deferred / high.

## 8. Open Questions for the User

1. Approve moving H-EE-007 ahead of the existing H-EE-015 priority? I recommend yes because it
   is cheaper and attacks the original action-representation question more directly.
2. Should historical rejected implementations such as H-EE-003 be restored in a frozen legacy
   module for exact reruns, or is a command/config/evidence record sufficient? I recommend the
   latter unless replication is specifically needed.
3. **Resolved after the audit:** the user approved integrating the audited residual-program and
   reorganization changes into `main` together after artifact cleanup and final verification.

## What To Review

- [ ] `src/svla/core/action_space.py` — verify the canonical action contract is the right
  extension point.
- [ ] `experiments/configs/` — inspect the exact commands and the historical-only H-EE-003
  decision.
- [ ] `RESULTS.md` — verify the evidence ladder and untested matrix.
- [ ] This report’s H-EE-007 recommendation and kill criteria.

## Implementation Details

The reorganization preserves public imports. Existing calls to
`svla.evaluation_protocol`, `svla.experiment_manifest`, and
`scripts/analyze_policy_failures.py` still work. The new action registry replaced duplicated
rollout/replay/render dispatch but uses the same task API methods and vector slices.

Experiment configs render deterministic CLI commands. `experiments/run.py --dry-run` prints the
command and config SHA. When executed, the config path/SHA are added to provenance environment
metadata. Config validation refuses final-split launches unless the JSON explicitly opts in.

## Evidence And Verification

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
```

- Baseline result outside sandbox: 114 passed.
- Baseline sandbox result: 106 passed, 8 CoreGraphics render failures.
- What this proves: inherited checkout behavior was green before refactoring.
- What it does not prove: policy efficacy or hardware realism.

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q \
  tests/test_action_representation.py tests/test_experiment_config.py \
  tests/test_research_smoke.py tests/test_state_bc.py \
  tests/test_pick_place_replay.py tests/test_analyze_policy_failures.py \
  tests/test_evaluation_protocol.py tests/test_experiment_manifest.py
```

- Result: 54 passed.
- What this proves: action dispatch, configs, compatibility paths, tiny training, and closed-loop
  evaluation work together after the change.

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
```

- Final merge-gate result outside sandbox: **125 passed in 112.62s**.
- What this proves: the complete inherited and new test suite passes with the macOS graphics
  context available.
- What it does not prove: research efficacy beyond the frozen evidence artifacts.

```bash
PYTHONPATH=src .venv/bin/python experiments/run.py \
  experiments/configs/h_ee_014_nn_gripper_global_validation.json --dry-run
```

- Result: rendered the exact five-seed validation command and config SHA
  `9c475d8859caca38dda11c0c6838e7131b64e22f67f25256ace304f2186a0f23`.
- What this proves: the config is executable and deterministic.
- What it does not prove: that the expensive experiment was rerun in this audit.

## Demo Videos / Visual Artifacts

- Open Phase 6a preview: `[preview.mp4](</Users/matthewwoodcock/Documents/Simplified VLA Structure/outputs/phase6a_vision_sample/preview.mp4>)`.
- Open H-EE-014 residual clips directory: `[residual clips](</Users/matthewwoodcock/Documents/Simplified VLA Structure/outputs/h_ee_014_residual_clips/>)`.
- These outputs are gitignored and were not regenerated by this audit. Regenerate the Phase 6a
  preview with:

```bash
PYTHONPATH=src .venv/bin/python scripts/render_vision_dataset_preview.py \
  --dataset-dir outputs/phase6a_vision_sample \
  --output outputs/phase6a_vision_sample/preview.mp4
```

## Decisions Made

- **Decision:** preserve controller/task/training behavior during audit.
  **Reason:** no evidence justified changing a scientific variable under “cleanup.”
- **Decision:** keep monoliths physically in place for now.
  **Reason:** active uncommitted residual work makes a mass move high-risk and hard to review.
- **Decision:** mark H-EE-003 historical-only.
  **Reason:** current code cannot execute its removed output head; false reproducibility is worse
  than an explicit limitation.
- **Decision:** rank H-EE-007 above tokenization and VLA work.
  **Reason:** it has the highest information gain per unit compute for the current discrepancy.

## Risks And Limitations

- The worktree was dirty before this audit. A single commit would absorb unrelated research
  changes unless hunks are separated carefully.
- The new config catalog covers maintained Phase 5/H-EE experiments, not every one-off scratch
  or obsolete legacy grid under 5.7 GB of `outputs/`.
- `state_bc.py` and `pickup_task.py` remain large; logical separation improved, physical
  maintainability remains moderate risk.
- Literature methods were mapped conceptually; no external model was installed or benchmarked.
- No final holdout, Phase 6b policy, VLA, or language experiment was run.

## Action Items

- [ ] User approval: H-EE-007 first versus H-EE-015 first.
- [x] User approved integrating the audited residual work and reorganization into `main`.
- [ ] If H-EE-007 passes its replay gate, preregister the one-seed train and exact bars.
- [ ] Only promote to five seeds after the one-seed causal screen passes.
- [ ] Keep final and Phase 6b closed until the existing frontier/worst-seed/physics rules are met.

## Files Changed

- `src/svla/core/action_space.py` — canonical representation contract.
- `src/svla/state_bc.py` — registry-based rollout dispatch.
- `src/svla/pick_place_replay.py` — registry-based replay dispatch.
- `scripts/render_bc_rollout.py` — registry-based render dispatch.
- `src/svla/eval/` — canonical protocol/provenance package.
- `src/svla/experiments/config.py`, `experiments/` — config system and launches.
- `analysis/policy_failures.py` — offline analysis location.
- `README.md`, `LEARNING_GUIDE.md`, `RESULTS.md`, `CHANGELOG.md` — current documentation.
- `tests/test_action_representation.py`, `tests/test_experiment_config.py`,
  `tests/test_research_smoke.py` — new smoke/contract coverage.

## Current Verdict

**Implementation complete, 125 tests passing, scientifically conservative, and ready for code
review.** The
reorganization improves extension and reproducibility without changing the research verdict.
The learned-policy comparison remains **blocked**, Phase 6b remains **not started**, and the
recommended next experiment still requires explicit approval because it changes labels.
