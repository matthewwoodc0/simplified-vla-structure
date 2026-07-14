# 2026-07-14 - Phase 5 Causal Synthesis

## Plain-English Summary

This change freezes the pickup rescue chapter as a **documentation synthesis**. It does not
train a model, re-evaluate a policy, open the final holdout, or start Phase 6b.

Three decisive probes after the hybrid H-EE-014 baseline closed the mainline rescue path:

| Probe | One-line result |
|-------|-----------------|
| **H-EE-007** | Raw observed EE labels failed replay **0/18** vs policy-label control **18/18**. |
| **H-EE-002** | Frozen arm gain 1.0 = **62/120**; 0.875 = **5/120**; 0.750 = **0/120**. |
| **H-EE-015** | Oracle gripper FSM = **47/120** vs hybrid baseline **62/120** (paired +5/−20). |

**What to believe now:** stop default gripper/gain/FSM/match/loss rescue tuning. Keep
hybrid A1 + `global_gripper` as the **fair comparative contract** for the next program
(demonstration efficiency → learned pick-place → controller-integration replication). Do
**not** claim that EE actions are universally worse than joint actions — the evidence is
one sim, one task family, one controller integration, and one BC family.

Durable record: [`evidence/phase5_causal_synthesis.json`](../evidence/phase5_causal_synthesis.json).

## What To Review

- [ ] [`evidence/phase5_causal_synthesis.json`](../evidence/phase5_causal_synthesis.json):
  hashes, metrics, probe records, frozen contract, next program order.
- [ ] [`RESULTS.md`](../RESULTS.md): next program is efficiency / pick-place / controller
  replication, not residual rescue.
- [ ] [`researchnotes.md`](../researchnotes.md): mainline rescue queue closed; optional
  mechanism backlog only for H-EE-024/SP3 and H-EE-017.
- [ ] [`AGENTS.md`](../AGENTS.md): dated synthesis verdict, YOU ARE HERE, Next Useful Work.
- [ ] Source evidence still present and hashed (do not rewrite historical reports).

## Implementation Details

### Scope of this change

- Read/analyze/document only.
- Branch: `codex/phase5-causal-synthesis`.
- No Python/runtime, controller, task, gate, or training-code edits.
- No new hypothesis run, no re-eval, no final access, no Phase 6b.

### Verified headline facts (from named artifacts)

| Fact | Verified value | Source |
|------|----------------|--------|
| Raw protocol-v2 final joint | **51/120** | `evidence/phase5_v2_final_results.json` |
| Raw protocol-v2 final EE | **28/120** | same |
| Best fair validation joint (hybrid A1) | **97/120** | `outputs/h_ee_014_nn_gripper_global_validation/h_ee_014_comparison.json` |
| Best fair validation EE (hybrid A1) | **62/120** | same |
| H-EE-007 raw-label replay | **0/18** success, **0/18** event order | `outputs/h_ee_007_label_contract_probe/h_ee_007_comparison.json` |
| H-EE-007 policy-label control | **18/18** success and event order | same |
| H-EE-002 gain 1.0 / 0.875 / 0.750 | **62 / 5 / 0** per 120 | `evidence/h_ee_002_hybrid_gain_sweep.json` |
| H-EE-015 oracle FSM vs baseline | **47** vs **62**; paired **+5 / −20** | `evidence/h_ee_015_fsm_upper_bound.json` |

All values matched the pre-registered frozen facts. No evidence mismatch.

### Claim table (ruled out vs not ruled out)

| ID | Claim tested | Outcome | Ruled out | Not ruled out |
|----|--------------|---------|-----------|---------------|
| **H-EE-007** | Raw observed EE transitions are executable command-scale replacement labels | **Rejected at replay** | Drop-in use of raw observed EE transitions as training/command labels under the current record-and-replay contract | That reconstructed `policy_labels` are *optimal* among all possible constructions |
| **H-EE-002** | Simple monotonic reduction of frozen EE arm gain rescues the hybrid residual | **Rejected on validation** | Gain 0.875/0.750 as a rescue; treating lower constraint exposure alone as improvement | Controller integration as a *broader* moderator; non-gain controller redesigns |
| **H-EE-015** | Structurally legal oracle gripper timing raises the frozen arm ceiling | **`negative_arm_ceiling`** | Gripper timing as the primary remaining fix for the frozen hybrid arm residual; crediting oracle early_close/reopen zeros as learned gains | Co-trained arm+gripper learning; that a different *learned* gripper could help a re-trained arm. **Not** a literal arm upper bound (oracle changes closed loop; paired regressions) |
| **H-EE-014** | Hybrid NN gripper + MLP arm under `global_gripper` is the best fair validation family | **Confirmed on validation**; selected as freeze | Treating reopen as the still-dominant residual under pure-MLP global | That EE is final-ready or universally competitive with joint |

### Frozen next-program contract

Primary contract for the next state-based comparisons (not a readiness claim):

| Field | Value |
|-------|-------|
| Action spaces | `joint_delta`, `ee_tool_delta` |
| Policy family | hybrid NN gripper + MLP arm, **A1** compositor |
| Loss | `global_gripper` |
| NN match | **historical** |
| Temporal features | `legacy_progress_phase` |
| Label source | `policy_labels` |
| Hidden sizes | 128 128 |
| Epochs | 300 |
| Batch | 1024 |
| Learning rate | 0.001 |
| Weight decay | 0.00001 |
| Action gain | 1.0 |
| Model seeds | 0 1 2 3 4 |
| Gates | strict event-order and physical-sanity **unchanged** |

This freeze does **not** authorize final access, Phase 6b, or deployment claims.

### Ordered next program

1. **Demonstration efficiency** — preregistered nested/stratified demo-count curve, common seeds, both spaces.
2. **Learned pick-place** — second manipulation task under the same fair contract (scripted/replay already exist).
3. **Controller-integration replication** — Controller A (stateless
   current-measured-pose-plus-delta DLS; current learned EE rollout) vs Controller B
   (persistent-target-lag DLS, same underlying DLS solver). Not an independent IK
   algorithm. Require identical task specs, observation schema, evaluation trials, and
   gates; controller-specific executable demos/labels with exact joint/EE demo parity
   within each controller; do **not** require byte-identical realized demos across
   controllers.

Optional mechanism backlog only (not mainline): H-EE-024/SP3 impulse train path if separately registered; H-EE-017 history/GRU only with a careful non-Markov arm argument.

### Historical reports left untouched

These remain the experiment-of-record writeups and must not be rewritten to look “current”:

- [`reports/2026-07-09-h-ee-014-nn-gripper.md`](2026-07-09-h-ee-014-nn-gripper.md)
- [`reports/2026-07-09-three-decisive-tests-and-paper-strategy.md`](2026-07-09-three-decisive-tests-and-paper-strategy.md)
- [`reports/2026-07-13-h-ee-007-label-contract.md`](2026-07-13-h-ee-007-label-contract.md)
- [`reports/2026-07-13-h-ee-002-hybrid-gain.md`](2026-07-13-h-ee-002-hybrid-gain.md)
- [`reports/2026-07-14-h-ee-015-fsm-upper-bound.md`](2026-07-14-h-ee-015-fsm-upper-bound.md)

## Evidence And Verification

```bash
# Parse synthesis + recompute source hashes
python3 - <<'PY'
import json, hashlib
from pathlib import Path
doc = json.loads(Path('evidence/phase5_causal_synthesis.json').read_text())
assert doc['rescue_program_status'] == 'closed'
assert doc['final_accessed_by_this_goal'] is False
assert doc['phase6b_started'] is False
for path, meta in doc['source_artifacts'].items():
    h = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    assert h == meta['sha256'], (path, h, meta['sha256'])
print('hash_check_ok', len(doc['source_artifacts']))
PY

git diff --check
rg -n "residual fix|joint-only pick-place|Next open:|SP3 train path only if" RESULTS.md researchnotes.md AGENTS.md || true
rg -n "demonstration efficiency|second controller|rescue_program_status|SYNTHESIS" RESULTS.md researchnotes.md AGENTS.md evidence/phase5_causal_synthesis.json reports/2026-07-14-phase5-causal-synthesis.md
```

- Result: all required source artifacts present; SHA-256 digests recorded in the synthesis
  file match recomputed digests; frozen headline metrics match the named JSON sources.
- Output artifact: `evidence/phase5_causal_synthesis.json`, this report, and live-doc updates.
- What this proves: the pickup rescue chapter can be closed as a research program decision
  with auditable provenance and explicit non-claims.
- What it does not prove: any new closed-loop policy improvement; universality of EE vs
  joint; readiness for final access or Phase 6b.

**Pytest:** skipped. Only Markdown/JSON documentation changed; no runtime behavior.

## Demo Videos / Visual Artifacts

No new videos. Existing residual review clips remain under
`outputs/h_ee_014_residual_clips/` (gitignored) and are documented in
`outputs/h_ee_014_residual_visual_review.md`.

## Decisions Made

- Decision: close the mainline pickup rescue program.
  Reason: H-EE-007, H-EE-002, and H-EE-015 exhaust the registered label, gain, and oracle-gripper
  residual routes; SP1/SP2 already rejected match-set and A2 arm-only.
- Decision: freeze H-EE-014 hybrid A1 + `global_gripper` as the fair comparative contract.
  Reason: strongest symmetric validation family; not a claim that EE is ready.
- Decision: point live docs at efficiency → pick-place → controller-integration replication.
  Reason: matches the paper strategy after decisive tests
  (`reports/2026-07-09-three-decisive-tests-and-paper-strategy.md`), with the controller
  comparison scoped as integration replication (A vs B DLS wrappers), not a new IK algorithm.
- Decision: move H-EE-024/SP3 and H-EE-017 to optional mechanism backlog.
  Reason: mechanism interest remains, but they are not the default next comparative step.
- Decision: do not rewrite historical experiment reports.
  Reason: synthesis is additive; historical reports stay audit trails.

## Risks And Limitations

- Risk: readers may treat hybrid 62/120 EE as “fixed enough.”
  Why it matters: both raw final and hybrid validation still fail readiness/frontier bars;
  final remains closed.
- Risk: readers may over-read H-EE-007 as proof that policy labels are optimal.
  Why it matters: the probe only shows raw observed transitions are not executable at command scale.
- Risk: readers may over-read H-EE-002 as proof that controllers do not matter.
  Why it matters: only simple monotonic gain reduction on frozen models was rejected.
- Risk: readers may treat H-EE-015 47/120 as learned-policy evidence or a literal arm upper bound.
  Why it matters: oracle changes the closed-loop trajectory; paired regressions make the number
  diagnostic, not an upper bound and not a fair EE-vs-joint score.
- Risk: single SO-101 sim/task/controller/model family.
  Why it matters: the next program exists precisely because universality is unproven.

## Action Items

- [ ] Preregister the demonstration-efficiency curve under the frozen contract.
- [ ] Design learned pick-place BC for both action spaces under the same contract.
- [ ] Specify controller-integration replication (A: stateless measured-pose+delta DLS vs B:
      persistent-target-lag DLS) with identical task/obs/trials/gates and controller-specific
      demos (joint/EE parity within controller).
- [ ] Keep final holdout and Phase 6b closed unless scope is explicitly reopened.
- [ ] Do not promote gain/cap, FSM retune, match-set, or loss reweight as default next work.

## Files Changed

- `evidence/phase5_causal_synthesis.json` — durable synthesis evidence with hashes and freeze.
- `reports/2026-07-14-phase5-causal-synthesis.md` — this review report.
- `RESULTS.md` — replace stale residual-rescue next-step language; point to next program.
- `researchnotes.md` — close mainline rescue queue; freeze fair contract; optional backlog.
- `AGENTS.md` — synthesis verdict bullet; YOU ARE HERE; Next Useful Work.

## Current Verdict

**`SYNTHESIS_FROZEN`.** Pickup rescue mainline is closed. Fair comparison contract is frozen
for the next state-based research program. Final holdout not accessed. Phase 6b not started.
No new experiment was run.
