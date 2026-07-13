# 2026-07-09 - Three Decisive Tests And Paper Strategy

## Plain-English Summary

The broad claim that robot learning changes when actions are expressed in joint space versus
end-effector space is not novel by itself. It has been studied repeatedly in reinforcement
learning, imitation learning, simulation-to-real transfer, and large recent real-robot studies.
The current project is more interesting as a narrow, carefully instrumented case study: identical
SO-101 demonstrations can be executable in both action spaces while the learned closed-loop
policies separate sharply, and the gap can be decomposed into label construction, controller
tracking, gripper sequencing, event order, and physical-sanity failures.

That is enough for a useful mini writeup now, especially as a transparent negative result. It is
not yet a strong conventional research paper because the evidence is still one simulated robot,
one main learned task, one controller integration, and one small state-BC family.

Phase 6a is complete only as infrastructure: fixed-camera RGB capture, scripted image datasets,
validation, and previews. Phase 6b, a learned vision-conditioned policy, has not started. Phase 7
language/VLA work is therefore not the right next step. It would add ambiguity before the simpler
action-space mechanism is understood.

## What To Review

- [ ] [`prompts/h-ee-007-label-contract-probe.md`](../prompts/h-ee-007-label-contract-probe.md): tests whether the EE training-label contract is itself the disadvantage.
- [ ] [`prompts/h-ee-002-hybrid-gain-sweep.md`](../prompts/h-ee-002-hybrid-gain-sweep.md): tests whether deployment gain causes constraint thrash and missing lift.
- [ ] [`prompts/h-ee-015-fsm-arm-upper-bound.md`](../prompts/h-ee-015-fsm-arm-upper-bound.md): measures the learned arm ceiling with an explicitly oracle gripper FSM.
- [ ] [`researchnotes.md`](../researchnotes.md): review the reordered next-test priority and frozen baselines.
- [ ] Existing evidence: `outputs/h_ee_014_nn_gripper_global_validation/` and
  `outputs/post_h_ee_014_residual_scoreboard.json`.

## Implementation Details

Three self-contained execution contracts were prepared for Grok Build 4.5. Every prompt freezes
the H-EE-014 hybrid baseline, keeps the final holdout closed, permits one causal variable, defines
pass and stop bars before execution, requires exact artifacts, and forbids Phase 6b/7 expansion.

Recommended execution order:

1. **H-EE-007 label contract.** Highest scientific value because it tests whether the comparison
   itself gives EE a noisier or more indirect target. It includes an executability gate and a
   one-seed kill screen before five-seed training.
2. **H-EE-002 frozen gain.** Cheapest clean controller-causality test. It reuses byte-identical
   models and changes only arm action gain at inference.
3. **H-EE-015 oracle FSM.** A ceiling diagnostic, not a fair learned-policy result. Run it last so
   a structurally enforced gripper sequence cannot distract from the cleaner label/controller
   tests.

The stronger-paper route after those tests should remain state-based:

1. Freeze the best fair contract from the three tests.
2. Run a preregistered demonstration-count efficiency curve using nested, stratified subsets and
   common seeds for both action spaces.
3. Add learned pick-place as a second manipulation task for both action spaces; scripted and replay
   plumbing already exists, but no pick-place BC result exists yet.
4. Add a genuinely second controller integration on SO-101, such as current accumulated-target DLS
   versus a current-state one-step/velocity-style EE tracking contract. Keep observations, demos,
   trials, and gates identical.
5. Only then consider a second simulated robot or hardware replication. Hardware can remain a
   distant validation target rather than the next milestone.

This route supports a defensible paper question: **when does a controller-level action space fail
to deliver the expected learning advantage, and how much of that failure comes from labels,
controller integration, versus task sequencing?** It does not claim to discover that action-space
choice matters.

## Evidence And Verification

The local evidence used to freeze the prompts was checked with:

```bash
jq '.' outputs/h_ee_014_nn_gripper_global_validation/h_ee_014_comparison.json
jq '.' outputs/post_h_ee_014_residual_scoreboard.json
rg -n "H-EE-002|H-EE-007|H-EE-015" researchnotes.md AGENTS.md RESULTS.md
git diff --check
```

- Result: the prompt baselines match the frozen hybrid evidence: EE 62/120, event order 79,
  physical sanity 68, worst seed 9, early-close 11, reopen 0, and missing-lift EO 30; joint is
  97/120 under the same hybrid comparison family.
- Output artifact: the three prompt files listed above.
- What this proves: the contracts are grounded in the current evidence and have explicit causal
  boundaries and stopping rules.
- What this does not prove: none of the three hypotheses has been run by this documentation change.
  No runtime code or research verdict changed, so pytest was not required for this prompt-only work.

Literature positioning was checked against primary or official sources:

- [Varin et al., *A Comparison of Action Spaces for Learning Manipulation Tasks* (2019)](https://arxiv.org/abs/1908.08659)
- [Aljalbout et al., *On the Role of the Action Space in Robot Manipulation Learning and Sim-to-Real Transfer* (2023)](https://arxiv.org/abs/2312.03673)
- [Sontakke et al., *Redundancy-aware Action Spaces for Robot Learning* (2024)](https://arxiv.org/abs/2406.04144)
- [*Demystifying Action Space Design for Robotic Manipulation Policies* (2026)](https://arxiv.org/abs/2602.23408)
- [Lin et al., *Data Scaling Laws in Imitation Learning for Robotic Manipulation* (ICLR 2025)](https://arxiv.org/abs/2410.18647)
- [LeRobot action-representation documentation](https://huggingface.co/docs/lerobot/action_representations)

These sources show that broad action-space comparisons and sample-efficiency questions are active,
well-established research areas. The potential novelty here must come from the controlled failure
decomposition and reproducible low-cost platform case, not from merely comparing joint and EE deltas.

## Demo Videos / Visual Artifacts

No new videos or visual artifacts were created. Existing post-H-EE-014 clips remain under
`outputs/h_ee_014_residual_clips/` and are documented in
`outputs/h_ee_014_residual_visual_review.md`.

## Decisions Made

- Decision: do not advance to Phase 7.
  Reason: Phase 6a plumbing is complete, but Phase 6b learned vision behavior does not exist; language
  would multiply confounds without strengthening the action-space result.
- Decision: write the mini paper even if all three hypotheses are rejected.
  Reason: a registered, physically gated negative result with explicit failure decomposition is more
  valuable than an inflated success claim.
- Decision: prioritize label contract, then frozen gain, then oracle FSM.
  Reason: this orders the tests from cleanest comparison threat to controller deployment cause to
  intentionally unfair upper-bound diagnosis.
- Decision: make the efficiency curve and second controller/task the route to a stronger paper.
  Reason: they test generality and learning efficiency without requiring near-term hardware or VLA.

## Risks And Limitations

- Risk or limitation: H-EE-007 changes the training target and may require nontrivial replay plumbing.
  Why it matters: default behavior and joint labels must remain unchanged or the causal claim breaks.
- Risk or limitation: H-EE-002 tests action gain, not every possible controller design.
  Why it matters: rejection rules out simple monotonic gain causality, not controller interaction broadly.
- Risk or limitation: H-EE-015 uses privileged scripted grasp information.
  Why it matters: its score is an arm upper bound and cannot be reported as learned policy performance.
- Risk or limitation: a single SO-101 simulation/task/model family remains weak generalization evidence.
  Why it matters: a conventional paper needs the efficiency curve plus task/controller replication.
- Risk or limitation: simulation force thresholds are sanity gates, not hardware calibration.
  Why it matters: physically sane simulation is necessary but does not establish real-world safety.

## Action Items

- [ ] Run H-EE-007 and stop at its replay or one-seed kill gate if it fails.
- [ ] Run H-EE-002 on frozen models with no rescue gains.
- [ ] Run H-EE-015 once with its preregistered oracle FSM and label it diagnostic.
- [ ] Freeze the selected fair contract before designing the efficiency curve.
- [ ] Preregister nested demo counts, common seeds, second task, and second controller.
- [ ] Draft the mini writeup around the evidence ladder and negative/causal findings now; update its
  result tables after the three tests.

## Files Changed

- `prompts/h-ee-007-label-contract-probe.md` - execution contract for label-source causality.
- `prompts/h-ee-002-hybrid-gain-sweep.md` - execution contract for frozen-model gain causality.
- `prompts/h-ee-015-fsm-arm-upper-bound.md` - execution contract for the oracle arm upper bound.
- `researchnotes.md` - prompt links and decisive-test execution order.
- `reports/2026-07-09-three-decisive-tests-and-paper-strategy.md` - novelty and paper roadmap.

## Current Verdict

**Ready to run the three validation-only tests; not ready for Phase 7 or a broad conventional-paper
claim.** A mini writeup is already worthwhile. A stronger paper should be earned through the three
causal tests, an efficiency curve, a second learned manipulation task, and a second controller
integration before distant hardware replication.
