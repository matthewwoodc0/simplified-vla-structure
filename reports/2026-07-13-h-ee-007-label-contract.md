# 2026-07-13 - H-EE-007 Label Contract Probe

## Plain-English Summary

H-EE-007 is **rejected at the replay gate**. The raw observed EE transitions are not a
usable alternative training-label contract: they are roughly 30 times smaller than the
controller-feasible policy labels and fail to execute the pickup at all.

The policy-label control replay passed 18/18. Raw-label replay produced 0/18 successes and
0/18 valid event orders. It remained physically sane and had zero controller failures
because the arm barely moved: it generated zero gripper force and zero pre-lift impulse.

The mandatory stop condition was followed. No model was trained, no seed screen or full
validation was run, and the final holdout was not accessed.

Worktree: `/Users/matthewwoodcock/.codex/worktrees/cbc7/Simplified VLA Structure`

Branch: `codex/h-ee-007-label-contract` (the isolated worktree started detached; the source
checkout's existing `research/h-ee-007-label-contract` branch was not moved or modified).

## What To Review

- [ ] `outputs/h_ee_007_label_contract_probe/h_ee_007_label_audit.json`: 48,112-sample
  raw-versus-policy label audit, including phase breakdowns.
- [ ] `outputs/h_ee_007_label_contract_probe/h_ee_007_replay_comparison.json`: exact 18-demo
  control/candidate replay rows and strict gate checks.
- [ ] `outputs/h_ee_007_label_contract_probe/h_ee_007_comparison.json`: compact rejection
  verdict and confirmation that later phases did not run.
- [ ] `src/svla/pick_place_replay.py`: the small durable code change making replay label
  source explicit while keeping `policy_labels` as the default.

## Implementation Details

Only the replay label source changed scientifically:

- `policy_labels.ee_tool_delta` remained the control and compatibility default.
- `labels.ee_tool_delta` was selected explicitly for the raw candidate.
- Joint replay stayed on its existing default and was not part of the candidate.
- Task, controller, demos, action gain, physics gates, and gripper commands were unchanged.

The one-off script performs only the two phases that actually ran: the required label audit
and the 18-demo replay gate. Unused training/registration plumbing was removed after the
gate rejected the hypothesis.

## Evidence And Verification

### Phase A - label audit

```bash
PYTHONPATH=src '/Users/matthewwoodcock/Documents/Simplified VLA Structure/.venv/bin/python' \
  scripts/run_h_ee_007_label_contract.py audit \
  --baseline-dir outputs/h_ee_007_label_contract_probe/baseline_inputs \
  --output-dir outputs/h_ee_007_label_contract_probe
```

- Result: 30 exact H-EE-014 scripted demos, 48,112 samples.
- Raw mean arm L2: `0.0005204255`.
- Policy-label mean arm L2: `0.0154210354` (about 29.6 times larger).
- Mean raw/policy arm cosine agreement: `0.3830471`.
- Mean arm-label L2 difference: `0.0150831075`.
- Gripper equality: 48,112/48,112, maximum difference `0.0`.
- Output artifact: `outputs/h_ee_007_label_contract_probe/h_ee_007_label_audit.json`.
- What this proves: the contracts differ materially in scale and direction while the
  gripper dimension is identical.
- What this does not prove: which contract learns better in closed loop.

### Phase B - executability replay gate

```bash
PYTHONPATH=src '/Users/matthewwoodcock/Documents/Simplified VLA Structure/.venv/bin/python' \
  scripts/run_h_ee_007_label_contract.py replay \
  --baseline-dir outputs/h_ee_007_label_contract_probe/baseline_inputs \
  --output-dir outputs/h_ee_007_label_contract_probe
```

- Policy-label control: success 18/18, event order 18/18, physical sanity 18/18,
  controller failures 0.
- Raw-label candidate: success 0/18, event order 0/18, physical sanity 18/18,
  controller failures 0.
- Raw candidate force/impulse: maximum gripper force `0.0 N`; pre-lift impulse `0.0 N s`.
- Raw candidate preclose/reopen: 0/0.
- Output artifact: `outputs/h_ee_007_label_contract_probe/h_ee_007_replay_comparison.json`.
- What this proves: raw observed transitions are not executable command-scale labels for
  this action representation.
- What this does not prove: that reconstructed labels are globally optimal, or why the
  remaining hybrid EE arm/lift residual exists.

### Tests

```bash
PYTHONPATH=src '/Users/matthewwoodcock/Documents/Simplified VLA Structure/.venv/bin/python' \
  -m pytest -q tests/test_pick_place_replay.py
```

- Result: **7 passed in 309.94 s**.

```bash
PYTHONPATH=src '/Users/matthewwoodcock/Documents/Simplified VLA Structure/.venv/bin/python' \
  -m pytest -q
```

- Result: **126 passed in 479.42 s**.

## Demo Videos / Visual Artifacts

No video or new visual artifact was needed. This experiment stopped on a numerical
executability gate before training.

## Decisions Made

- Decision: reject H-EE-007 at replay and do not train.
  Reason: the candidate missed both mandatory 18/18 success and event-order gates.
- Decision: do not rescale the raw labels as a rescue.
  Reason: that would be a second scientific variant forbidden by the prompt.
- Decision: leave the final holdout closed.
  Reason: the experiment did not reach the one-seed screen, much less full validation.

## Risks And Limitations

- The result rejects raw observed transitions as a drop-in label source. It does not prove
  that every alternative EE label construction is bad.
- The raw replay is physically sane because it does almost nothing. Physical sanity alone
  must not be misread as task progress.
- `outputs/` is gitignored. The exact artifacts and copied frozen demo inputs remain local;
  this report and research documents carry their durable references.

## Action Items

- [ ] Do not train or rescale H-EE-007 raw labels as an unregistered rescue variant.
- [ ] Keep H-EE-014 hybrid A1 + `global_gripper` as the frozen learned baseline.
- [ ] If continuing SP4, H-EE-002 is a separate causal gain probe; H-EE-015 remains the
  structural arm upper-bound diagnostic.

## Files Changed

- `src/svla/pick_place_replay.py` - explicit replay label-source selection; default unchanged.
- `scripts/validate_action_replay.py` - exposes the explicit selector in the existing CLI.
- `scripts/run_h_ee_007_label_contract.py` - compact audit and replay-gate runner.
- `tests/test_pick_place_replay.py` - focused selection/default/joint-preservation test.
- `researchnotes.md` - H-EE-007 verdict and Results log.
- `AGENTS.md` - durable verdict and next-step update.
- `RESULTS.md` - concise negative-result index.

## Current Verdict

**Rejected at replay.** This is a completed negative experiment, not a blocked or partial
training run. The final holdout remains closed and Phase 6b remains blocked.
