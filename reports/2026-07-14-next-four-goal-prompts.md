# 2026-07-14 - Next Four Goal Prompts

## Plain-English Summary

Four sequential Grok 4.5 goal contracts now cover the post-probe research program: causal
synthesis, efficiency-curve preregistration, learned pick-place replication, and a second
controller-integration replication. Each prompt preserves strict state-based boundaries and stops
before locked holdout access.

The efficiency prompt intentionally builds and validates the runner without executing its primary
150-fit curve. That creates a real independent-review gate before a new locked evaluation is
exposed. Pick-place and controller replication may run their registered validation experiments,
but not their locked holdouts.

## What To Review

- [ ] `prompts/goal-01-phase5-synthesis-freeze.md` - closes rescue tuning and freezes the fair contract.
- [ ] `prompts/goal-02-efficiency-curve-preregistration.md` - freezes nested data budgets, three subset ladders, endpoints, runner, and locked-access guard.
- [ ] `prompts/goal-03-pick-place-bc-replication.md` - adds goal-conditioned learned pick-place for both action spaces with task-floor handling.
- [ ] `prompts/goal-04-second-controller-replication.md` - tests stateless versus persistent-target DLS integration across pickup and pick-place.

## Implementation Details

The contracts are deliberately sequential. Every goal starts from reviewed `main`, creates a
focused branch, identifies authority artifacts, specifies allowed changes, defines mandatory gates,
requires provenance and tests, and treats a clean negative result as completion.

Goal 02 uses budgets 6, 12, 18, 24, and 30 across six balanced orientation/approach strata, three
immutable nested subset ladders, five model seeds, and two action spaces. That implies 150 fits in
the eventual primary curve. This prompt stops at preregistration and smoke validation.

Goal 03 explicitly requires placement-goal features because left/right pick-place trajectories are
not learnable fairly if divergent post-lift actions share indistinguishable task observations. It
also separates end-to-end success from placement conditional on valid grasp.

Goal 04 accurately labels the current learned EE path as stateless current-pose-plus-delta and the
new path as persistent-target integration. Both use the same DLS solver, so the prompt forbids
calling this an independent IK algorithm.

## Evidence And Verification

```bash
git diff --check
rg -n '[[:blank:]]+$' prompts/goal-0*.md reports/2026-07-14-next-four-goal-prompts.md
```

- Result: passed with no whitespace errors; every prompt has exactly one complete paste block.
- Output artifact: the four prompt files listed above.
- What this proves: the Markdown patches have no whitespace errors once the command passes.
- What it does not prove: none of the four research goals has been executed.

Pytest was not run because this change adds Markdown execution contracts only and does not modify
Python, configs, assets, controller/task behavior, or experiment results.

## Demo Videos / Visual Artifacts

No demo videos or visual artifacts were created. Future validation prompts require visual review
when graphics access is available, while keeping rendering separate from numerical efficacy gates.

## Decisions Made

- Decision: stop Goal 02 before primary curve execution.
  Reason: independent review should occur before accessing a newly locked evaluation.
- Decision: require both action spaces for learned pick-place.
  Reason: joint-only training would not constitute task replication of the research question.
- Decision: require explicit placement-goal features.
  Reason: otherwise left/right task commands create ambiguous labels from indistinguishable states.
- Decision: call Goal 04 controller-integration replication.
  Reason: both variants retain the same damped-least-squares IK solver.

## Risks And Limitations

- Risk or limitation: the eventual efficiency curve is 150 model fits plus rollout evaluation.
  Why it matters: it should not run until protocol, resume behavior, and locked-access guards pass review.
- Risk or limitation: pick-place extends pickup rather than providing a fully independent task.
  Why it matters: grasp-segment and conditional-placement metrics must remain separate.
- Risk or limitation: persistent-target DLS is not a distinct solver.
  Why it matters: controller generality claims must be limited to integration sensitivity.

## Action Items

- [x] Goal 01 completed, reviewed, and merged as the Phase 5 causal synthesis freeze.
- [x] Goal 02 preregistration completed; independent review passed on 2026-07-15.
- [ ] Run and review the complete EFF-001 development primary curve on a focused branch.
- [ ] Review and merge the resulting efficiency evidence before starting Goal 03.
- [ ] Keep every final/locked holdout closed until its implementation and registration are reviewed.

## Files Changed

- `prompts/goal-01-phase5-synthesis-freeze.md` - synthesis and contract-freeze execution contract.
- `prompts/goal-02-efficiency-curve-preregistration.md` - curve infrastructure and preregistration contract.
- `prompts/goal-03-pick-place-bc-replication.md` - second learned-task contract.
- `prompts/goal-04-second-controller-replication.md` - controller-integration replication contract.
- `reports/2026-07-14-next-four-goal-prompts.md` - audit report for this prompt set.

## Current Verdict

**Prompt pack preserved; Goals 01 and 02 are complete through preregistration review.** The
EFF-001 primary curve is ready but not run. Goal 03 remains blocked until that curve is executed,
reviewed, and merged. No final/locked evaluation, vision policy, or VLA work is authorized.
