# Goal 02 — Demonstration-Efficiency Curve Infrastructure And Preregistration

**Target agent:** Grok 4.5
**Goal type:** protocol + runner implementation, with a review stop before locked evaluation
**Primary result execution:** forbidden until Codex review
**Final holdout:** closed
**Prerequisite:** Goal 01 reviewed, merged, and `phase5_causal_synthesis.json` present

Copy the block below into a fresh Grok task. This file is the complete contract.

---

## PASTE INTO GROK

```text
You are the research engineer responsible for preregistering the state-BC demonstration-efficiency
study in:
/Users/matthewwoodcock/Documents/Simplified VLA Structure

GOAL
Implement and validate a deterministic, provenance-safe efficiency-curve runner for the frozen
joint-delta versus EE-tool-delta comparison. Freeze exact nested demonstration subsets, analysis
endpoints, model seeds, and a new locked evaluation split. Stop for independent review before any
primary curve or locked evaluation is run.

This goal produces READY-TO-RUN infrastructure, not the scientific curve result.

READ FIRST — AUTHORITY
1. AGENTS.md
2. RESULTS.md
3. researchnotes.md
4. evidence/phase5_causal_synthesis.json
5. reports/*phase5-causal-synthesis.md
6. configs/phase5_evaluation_protocol_v2.json
7. src/svla/eval/protocol.py
8. src/svla/experiments/config.py
9. src/svla/state_bc.py
10. scripts/train_state_bc.py
11. experiments/configs/h_ee_014_nn_gripper_global_validation.json
12. evidence/README.md
13. prompts/goal-02-efficiency-curve-preregistration.md — this full contract

STARTING STATE
- Start only from reviewed `main` after Goal 01.
- Create `codex/state-bc-efficiency-curve`.
- Confirm the frozen contract in `evidence/phase5_causal_synthesis.json` before coding.
- Do not absorb unrelated changes.

SCIENTIFIC DESIGN — FREEZE BEFORE RESULTS

Primary comparison:
- action spaces: `joint_delta`, `ee_tool_delta`
- frozen hybrid A1 + `global_gripper` + historical match
- `legacy_progress_phase`, `policy_labels`, action gain 1.0
- MLP 128 128, 300 epochs, batch 1024, lr 1e-3, wd 1e-5
- model seeds 0 1 2 3 4

Data budgets:
- 6, 12, 18, 24, 30 distinct successful scripted pickup demonstrations
- six strata = 3 orientations x 2 approaches
- every budget must contribute exactly the same number of demos to every stratum
- one new distinct object-pose demo per stratum is added at each budget step
- subsets must be nested: budget N is an exact subset of every larger budget in its ladder
- deterministic duplicate `repeat` trajectories do not count as new demonstrations

Subset-selection variance:
- preregister exactly three immutable nested ladders
- each ladder must store explicit demo/trial IDs, pose labels, strata, order, and a SHA-256 hash
- ladder construction may use fixed seeds, but the resulting explicit lists are the contract
- the same ladder and exact demos must be used for both action spaces
- no ladder selection after observing rollout results

Evaluation:
- add a new versioned efficiency-study protocol; do not mutate protocol v2
- include development/smoke and `locked_evaluation` splits with disjoint object positions
- locked positions must remain inside the declared nominal/readiness task envelope
- store exact trial IDs, positions, orientations, approaches, gates, and protocol hash
- never alias the new locked split to the existing v2 validation or final split
- the runner must require a literal opt-in flag such as `--allow-locked-evaluation`
- this goal must never supply that flag

Primary endpoint:
- normalized area under success rate versus distinct-demo-count curve, per action space
- paired joint-minus-EE AUC difference using the same ladder, model seed, and evaluation specs

Secondary endpoints:
- success difference at every budget
- event-order-valid rate
- physical-sanity-pass rate
- worst-model-seed success
- hard-limit/infeasible exposure
- early close and reopen
- supervised timestep count
- training wall time, rollout wall time, model bytes, and peak process memory if available without
  adding a heavyweight dependency

Uncertainty:
- analysis must keep model seed and subset ladder distinct
- provide paired bootstrap confidence intervals over preregistered paired units
- do not pretend five model seeds alone measure demonstration-selection variance
- never extrapolate a demo threshold beyond observed budgets
- if a target performance level is not crossed, report `not_reached`, not an estimate

REQUIRED IMPLEMENTATION
1. A versioned config such as `configs/state_bc_efficiency_protocol_v1.json`.
2. A loader/validator that rejects:
   - non-nested subsets
   - stratum imbalance
   - duplicate demos counted as distinct
   - train/evaluation position overlap
   - changed frozen-contract fields
   - fewer/more than three ladders or five budgets
3. A dedicated runner such as `scripts/run_state_bc_efficiency_curve.py`.
4. Read-only analysis code under `analysis/`, not rollout code, for curve aggregation and CIs.
5. A dry-run mode that prints the complete 150-fit matrix:
   5 budgets x 3 ladders x 5 seeds x 2 action spaces.
6. Resume behavior based on immutable cell identity and artifact hashes; never silently reuse an
   output whose protocol, demos, source, model recipe, or seed differs.
7. Config support under `experiments/` with dry-run-first behavior and explicit locked-access guard.
8. Manifests that record protocol, ladder, demo, source, controller, recipe, and output hashes.

REQUIRED TESTS
- exact budget list and balanced six-stratum counts
- nested-set property for all three ladders
- duplicate/repeat rejection
- action spaces receive byte-identical source demo path lists per matrix cell
- train/evaluation split disjointness
- frozen recipe drift rejection
- matrix contains exactly 150 fits with unique cell IDs
- resume accepts exact matches and rejects stale/mismatched artifacts
- locked evaluation cannot run without explicit authorization
- AUC boundary cases and paired-key alignment
- CI code distinguishes subset ladders from model seeds
- default `scripts/train_state_bc.py` behavior remains unchanged

ALLOWED EXECUTION IN THIS GOAL
- unit and integration tests
- experiment-config dry runs
- one plumbing-only smoke using one budget, one ladder, one model seed, at most two training epochs,
  and `eval-limit=1` on the development split
- the smoke must be labeled `non_efficacy_smoke=true` everywhere

FORBIDDEN EXECUTION
- no 150-fit curve
- no 300-epoch matrix cell
- no complete development curve
- no `locked_evaluation`
- no protocol-v2 `final`
- no selection or tuning from smoke outcomes

REQUIRED ARTIFACTS
- `configs/state_bc_efficiency_protocol_v1.json`
- `experiments/configs/state_bc_efficiency_curve_registered.json`
- `evidence/state_bc_efficiency_curve_registration.json`
- dry-run matrix JSON with 150 unique cells and hashes
- smoke artifacts under `outputs/` clearly labeled non-efficacy
- `reports/YYYY-MM-DD-efficiency-curve-preregistration.md`
- tests for every contract above

The tracked registration must include:
- exact frozen recipe and source synthesis hash
- five budgets and three explicit ladders
- model seeds
- evaluation split identities and hashes
- endpoints and uncertainty method
- planned cell count
- `primary_curve_executed: false`
- `locked_evaluation_accessed: false`
- exact future command that would execute the registered study after review

DOCUMENTATION
- Add the registered study to `researchnotes.md` as `registered_not_run`.
- Update `AGENTS.md` only if commands/operator guidance changed.
- Do not add results language to `RESULTS.md`; there is no curve result yet.
- Follow the AGENTS.md large-change report template.

STOP CONDITIONS
- If balanced nested subsets cannot be created from distinct successful demos, stop with
  `BLOCKED_DATA_POOL`; do not count repeats as data.
- If the runner cannot preserve exact demo parity between action spaces, stop with
  `BLOCKED_COMPARISON_PARITY`.
- If the smoke fails, diagnose plumbing only. Do not weaken the registered design.
- Stop before the primary execution so Codex can review the protocol and code.

VERIFICATION
- Run targeted tests, then the full non-render test suite.
- Render-backed vision failures from a missing macOS graphics connection must be reported
  separately; they are not permission to ignore other failures.
- Run `git diff --check` and config dry runs.
- Verify tracked registration hashes after all edits are final.

COMPLETION LABELS
- `READY_FOR_EFFICIENCY_REVIEW`
- `BLOCKED_DATA_POOL`
- `BLOCKED_COMPARISON_PARITY`
- `BLOCKED_IMPLEMENTATION`

FINAL RESPONSE
Lead with the completion label. State budgets, ladder count, cell count, frozen recipe, locked
split status, smoke scope, tests, artifacts, branch, and commit. Prominently state that the primary
curve and locked evaluation were not run and require independent review first.
```
