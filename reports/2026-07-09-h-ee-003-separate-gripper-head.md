# 2026-07-09 - H-EE-003 Separate Gripper Head

## Plain-English Summary

H-EE-003 tested whether a separate gripper prediction head improves state-BC pickup
reliability. The data audit found that every recorded gripper label is binary, so the
experiment used a classifier rather than pretending the command is continuous: 15,712 open
labels (`1`) and 32,400 close labels (`0`) across 48,112 training samples, with zero
intermediate values.

The model kept H-EE-008's successful training and rollout contract: the same scripted demos,
observations, legacy progress/phase features, 128x128 trunk capacity, seeds, action gains,
five-by-24 protocol-v2 validation grid, and raw rollout evaluator. The only intended model
change was splitting the old six-output linear head into a five-output arm MSE head and a
one-logit gripper classifier. The classifier used the same 5x gripper and 10x close-phase
weighting as H-EE-008, now applied to binary cross-entropy.

This did not improve closed-loop reliability. EE success fell from 50/120 to 34/120 and
event-order validity from 55/120 to 50/120. Joint performance collapsed from 84/120 to
41/120. **H-EE-003 is rejected.** The final holdout was not accessed.

After this rejection, the classifier implementation and its experimental tests were removed
rather than merged. This report, the validation artifacts, and the research verdict are kept
as the durable record of the negative result.

## What To Review

- [ ] [H-EE-003 comparison](</Users/matthewwoodcock/Documents/Simplified VLA Structure/outputs/h_ee_003_separate_gripper_head_validation/h_ee_003_comparison.json>) — exact raw validation comparison to H-EE-008.
- [ ] [Combined validation summary](</Users/matthewwoodcock/Documents/Simplified VLA Structure/outputs/h_ee_003_separate_gripper_head_validation/state_bc_summary.json>) — all 240 raw rollout rows summarized across both action spaces.
- [ ] `researchnotes.md` and `AGENTS.md` — rejection and next-step updates.

## Implementation Details

The discarded experiment implemented two explicit output contracts:

- `single_head` preserves the original serialized 6-D linear MSE policy unchanged.
- `separate_gripper_classifier` keeps the shared MLP trunk but adds a 5-D normalized arm
  regression head and a single sigmoid `P(open)` head. The classifier decodes at 0.5 into the
  existing action-adapter convention: `0=close`, `1=open`.

The new model artifacts include `output_head`, trunk parameters, arm-head parameters, and
gripper-head parameters. Loading an older `.npz` with no `output_head` still selects
`single_head`, so existing saved policies remain usable.

The experiment exposed `--mlp-output-head`; H-EE-003 selected
`separate_gripper_classifier` and retained H-EE-008's
`--gripper-loss-weight 5.0` and `--close-phase-gripper-weight 10.0` values. This is not a
claim that BCE and MSE are numerically identical; the weighting is held constant while the
binary label-compatible head/loss changes as required by the label audit.

## Evidence And Verification

### Focused unit tests

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_state_bc.py -q
```

- Result: **16 passed**.
- What this proves: both output contracts construct, save/load, return six-element actions,
  execute through joint and EE adapters, enforce binary-label encoding, preserve legacy model
  loading, and use a classifier loss distinct from the single-head MSE path.
- What this does not prove: closed-loop pickup reliability.

### Full regression suite

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
```

- Result: **96 passed, 8 failed**.
- The eight failures are confined to `tests/test_vision_dataset.py` and
  `tests/test_vision_observations.py`, all failing while MuJoCo creates a macOS CGL renderer
  with `CGLError: invalid CoreGraphics connection` in this headless terminal session.
- What this proves: non-rendering regression coverage, including the full state-BC path,
  passed; the H-EE-003 focused suite itself was 16/16.
- What this does not prove: the fixed-camera rendering tests need a GUI-capable macOS
  CoreGraphics context. This is unrelated to H-EE-003 and no vision code was changed here.

### CLI smoke test

```bash
PYTHONPATH=src .venv/bin/python scripts/train_state_bc.py \
  --output-dir outputs/scratch_h_ee_003_cli_smoke \
  --evaluation-protocol v2 --eval-split validation --eval-limit 1 --seeds 0 \
  --hidden-sizes 8 8 --epochs 1 \
  --mlp-output-head separate_gripper_classifier \
  --gripper-loss-weight 5 --close-phase-gripper-weight 10
```

- Result: both action spaces trained, loaded, and executed with the classifier head.
- What this proves: command-line plumbing and artifact contract work end to end.
- What this does not prove: the registered matrix; it deliberately used `--eval-limit 1`.

### Registered validation

```bash
PYTHONPATH=src .venv/bin/python scripts/train_state_bc.py \
  --output-dir outputs/h_ee_003_separate_gripper_head_validation \
  --evaluation-protocol v2 --eval-split validation \
  --temporal-feature-mode legacy_progress_phase --policy-type mlp \
  --mlp-output-head separate_gripper_classifier \
  --seeds 0 1 2 3 4 --hidden-sizes 128 128 --epochs 300 \
  --batch-size 1024 --learning-rate 0.001 --weight-decay 1e-5 \
  --stride 1 --max-steps 3200 --action-gain 1.0 \
  --label-source policy_labels \
  --gripper-loss-weight 5.0 --close-phase-gripper-weight 10.0
```

- Result: completed protocol-v2 **validation only**: five seeds x 24 trials x two action
  spaces, raw learned-policy evaluation with the guard disabled.
- Output artifacts: [comparison JSON](</Users/matthewwoodcock/Documents/Simplified VLA Structure/outputs/h_ee_003_separate_gripper_head_validation/h_ee_003_comparison.json>), [EE trial rows](</Users/matthewwoodcock/Documents/Simplified VLA Structure/outputs/h_ee_003_separate_gripper_head_validation/eval/ee_tool_delta_policy_trials.jsonl>), and [joint trial rows](</Users/matthewwoodcock/Documents/Simplified VLA Structure/outputs/h_ee_003_separate_gripper_head_validation/eval/joint_delta_policy_trials.jsonl>).
- What this proves: under the registered validation contract, the separate classifier is worse
  than H-EE-008's weighted single head on the key raw closed-loop outcomes.
- What this does not prove: final-holdout behavior or a general claim that all possible
  multi-head designs fail. The final split was intentionally not selected.

### Raw validation comparison

| Metric | H-EE-008 weighted single head | H-EE-003 separate head |
|---|---:|---:|
| EE success | 50/120 | 34/120 |
| EE event order valid | 55/120 | 50/120 |
| EE physical sanity pass | 71/120 | 70/120 |
| EE early-close trials | 5 | 29 |
| EE pre-close contact steps | 50 | 0 |
| EE reopen events | 142 | 93 |
| EE per-seed successes | 22, 9, 11, 2, 6 | 9, 1, 7, 6, 11 |
| Joint success | 84/120 | 41/120 |
| Joint event order valid | 90/120 | 48/120 |
| Joint physical sanity pass | 100/120 | 73/120 |
| Joint early-close trials | 6 | 36 |

Neither run had numerical controller failures. EE joint-limit and infeasible exposure fell
(33.8% -> 16.4% and 16.0% -> 9.2% of rollout steps), but that reduction did not produce a
valid sequence: early closes increased sharply. The narrower EE seed range is not meaningful
seed-stability progress because all five seeds are uniformly weak and aggregate success is
lower.

## Demo Videos / Visual Artifacts

No new video was generated: this hypothesis is rejected directly by the registered raw
closed-loop gate metrics, not by a visual-only judgment. The JSONL files above are generated
artifacts and are ignored by Git. Regenerate them with the registered validation command in
the previous section; do **not** substitute `--eval-split final`.

## Decisions Made

- Decision: use a classifier rather than 1-D gripper regression.
  Reason: all observed labels were exactly `0` or `1`; a continuous head would misrepresent
  the dataset's output contract.
- Decision: retain H-EE-008's legacy temporal contract and 5x/10x weighting.
  Reason: this makes H-EE-003 a narrow head/output-loss test rather than another temporal or
  data-distribution ablation.
- Decision: reject the hypothesis without final evaluation.
  Reason: EE success and event order both regressed on validation, early close increased, and
  joint performance degraded severely.

## Risks And Limitations

- Risk: switching MSE to BCE changes loss geometry as well as head separation.
  Why it matters: the label audit requires classification, so the result rejects this
  classifier-head design, not every conceivable separately parameterized gripper design.
- Risk: lower EE pre-close contact and constraint exposure may look superficially better.
  Why it matters: strict success and event-order gates show the model instead closes too
  early; telemetry improvements cannot replace closed-loop success.
- Risk: H-EE-008 itself remains validation-only.
  Why it matters: this rejection does not authorize access to the final holdout; final access
  remains a separate explicit decision.
- Risk: the full test suite cannot render fixed-camera RGB in this headless CGL session.
  Why it matters: eight unrelated vision tests need a GUI-capable macOS renderer before the
  suite can be fully green; the state-BC test suite passed independently.

## Action Items

- [ ] Keep H-EE-008 weighted single-head MSE as the current validated state-BC baseline.
- [ ] Do not rerun H-EE-003 as the default next experiment.
- [ ] If more EE work is authorized, prefer H-EE-016 close-transition oversampling or a
  separately justified causal experiment; retain the same validation-only discipline.
- [ ] Do not begin vision-conditioned BC, VLA, or language work from this result.

## Files Changed

- `researchnotes.md` - H-EE-003 testing-to-rejected result and evidence log.
- `AGENTS.md` - stable rejection and next-work guidance.
- `reports/2026-07-09-h-ee-003-separate-gripper-head.md` - this audit report.

## Current Verdict

**Rejected.** The separate binary gripper classifier is not a viable H-EE-008 replacement
under the current protocol-v2 validation contract. Phase 6b vision-conditioned BC/VLA remains
blocked, and the final holdout remains untouched.
