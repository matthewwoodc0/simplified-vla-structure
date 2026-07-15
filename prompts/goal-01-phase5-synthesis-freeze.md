# Goal 01 — Phase 5 Causal Synthesis And Contract Freeze

**Target agent:** Grok 4.5
**Goal type:** evidence synthesis and research-program transition
**Experiment execution:** forbidden
**Final holdout:** closed
**Prerequisite:** current `main` includes H-EE-007, H-EE-002, and H-EE-015 results

Copy the block below into a fresh Grok task. This file is the complete contract.

---

## PASTE INTO GROK

```text
You are the research engineer responsible for closing the pickup rescue chapter in:
/Users/matthewwoodcock/Documents/Simplified VLA Structure

GOAL
Produce the durable Phase 5 causal synthesis, freeze the fair comparison contract for the next
research program, and update the live roadmap so it points to demonstration efficiency, learned
pick-place, and second-controller replication—not more gripper/gain rescue tuning.

This is a read/analyze/document goal. Do not run a new hypothesis, train a model, re-evaluate a
policy, open a holdout, or change controller/task behavior.

READ FIRST — AUTHORITY
1. AGENTS.md
2. RESULTS.md
3. researchnotes.md
4. reports/2026-07-09-three-decisive-tests-and-paper-strategy.md
5. reports/2026-07-13-h-ee-007-label-contract.md
6. reports/2026-07-13-h-ee-002-hybrid-gain.md
7. reports/2026-07-14-h-ee-015-fsm-upper-bound.md
8. reports/2026-07-09-h-ee-014-nn-gripper.md
9. evidence/h_ee_002_hybrid_gain_sweep.json
10. evidence/h_ee_015_fsm_upper_bound.json
11. outputs/h_ee_007_label_contract_probe/h_ee_007_comparison.json
12. outputs/h_ee_014_nn_gripper_global_validation/h_ee_014_comparison.json

STARTING STATE
- Start from the reviewed current `main`.
- Create and work on `codex/phase5-causal-synthesis`.
- Working tree must be clean before edits. Do not absorb unrelated changes.
- Never delete or rewrite historical reports or evidence.

FROZEN FACTS TO VERIFY, NOT BLINDLY COPY
- Raw protocol-v2 final: joint 51/120, EE 28/120.
- Best fair validation family: hybrid NN gripper + MLP arm A1, `global_gripper`, historical
  match, `legacy_progress_phase`, action gain 1.0, seeds 0-4.
- Best fair validation result: joint 97/120, EE 62/120.
- H-EE-007: raw observed EE labels failed replay at 0/18 versus policy-label control 18/18.
- H-EE-002: gain 1.0 reproduced 62/120; 0.875 produced 5/120; 0.750 produced 0/120.
- H-EE-015: oracle FSM produced 47/120 versus 62/120, with paired +5/-20.

Verify every number from the named artifact. If an artifact is missing or disagrees, stop and
report the exact mismatch. Do not repair evidence by guessing.

REQUIRED INTERPRETATION
Write the synthesis around what was ruled out and what was not ruled out:

1. H-EE-007 rules out raw observed transitions as executable command-scale replacement labels.
   It does not prove the current reconstructed policy-label contract is optimal.
2. H-EE-002 rules out simple monotonic reduction of frozen EE arm gain as a rescue. Lower
   constraint exposure without recovered success is not causal improvement. It does not rule out
   controller integration as a broader moderator.
3. H-EE-015 shows that structurally legal gripper timing does not improve the frozen arm rollout.
   It is not a learned result and is not a literal arm upper bound because the oracle changes the
   closed-loop trajectory and caused paired regressions.
4. H-EE-014 remains the selected fair comparison contract because it was applied symmetrically
   and is the strongest validation family, not because it made EE ready.
5. The evidence supports ending gripper/gain rescue tuning. It does not support claiming that EE
   actions are universally worse than joint actions.

FROZEN NEXT-PROGRAM CONTRACT
Record the following as the primary contract for the next state-based comparisons:
- action spaces: `joint_delta` and `ee_tool_delta`
- policy family: hybrid NN gripper + MLP arm, A1 compositor
- loss: `global_gripper`
- NN match: historical
- temporal features: `legacy_progress_phase`
- label source: `policy_labels`
- hidden sizes: 128 128
- epochs: 300
- batch: 1024
- learning rate: 0.001
- weight decay: 0.00001
- action gain: 1.0
- model seeds: 0 1 2 3 4
- strict event-order and physical-sanity gates unchanged

This freeze is for fair comparative work. It does not authorize final access, Phase 6b, or a
claim that the policy is deployment-ready.

REQUIRED ARTIFACTS
1. `evidence/phase5_causal_synthesis.json`
   Must include:
   - format/version
   - source artifact paths and SHA-256 hashes
   - raw and hybrid comparison metrics
   - one record per decisive probe with `claim`, `outcome`, `ruled_out`, `not_ruled_out`
   - frozen contract fields above
   - `rescue_program_status: "closed"`
   - `final_accessed_by_this_goal: false`
   - `phase6b_started: false`
   - ordered next program: efficiency, pick-place, controller replication
2. `reports/YYYY-MM-DD-phase5-causal-synthesis.md`
   Follow the AGENTS.md report template. Include a plain-language claim table and exact links.
3. Update `RESULTS.md`
   - replace stale language that recommends residual rescue or joint-only pick-place
   - make the next program explicit
4. Update `researchnotes.md`
   - close the mainline rescue queue
   - move H-EE-024/SP3 and H-EE-017 to optional mechanism backlog
   - freeze the selected fair contract
5. Update `AGENTS.md`
   - add one dated synthesis verdict bullet
   - update `YOU ARE HERE` and `Next Useful Work`

DO NOT
- edit historical experiment reports to make them look current
- mark H-EE-015 as learned-policy evidence
- call H-EE-007 proof that reconstructed labels are optimal
- call H-EE-002 proof that controllers do not matter
- promote H-EE-024, H-EE-017, SP3, gain/cap, FSM, loss, or match tuning as the default next step
- access `eval-split final`
- start vision-conditioned BC, language, VLA, hardware, or a second robot
- change Python/runtime behavior

VERIFICATION
- Parse every JSON file changed or created.
- Recompute/check source hashes recorded in the synthesis evidence.
- Run `git diff --check`.
- Use targeted text searches to prove stale mainline recommendations were removed from live
  documents while historical reports remain unchanged.
- If only Markdown/JSON documentation changed, full pytest is optional; state why it was skipped.
- Working tree should contain only the intended synthesis changes.

COMPLETION LABELS
- `SYNTHESIS_FROZEN` if all artifacts, hashes, and live-document updates are complete.
- `BLOCKED_EVIDENCE_MISMATCH` if required source evidence is missing or contradictory.

FINAL RESPONSE
Lead with the completion label. State the frozen contract, the three decisive conclusions, files
changed, verification performed, branch, and commit. Explicitly state that no new experiment,
final access, or Phase 6b work occurred.
```
