# Goal Mode Prompt — Post-H-EE-014 Residual Program

**How to launch (copy the block under “PASTE INTO `/goal`” below).**
This file is the full contract. The paste block is the goal objective; this file is the law.

---

## Can goal mode finish this?

**Yes, with a bounded definition of “done.”**

| Realistic for one goal run | Not realistic as goal completion |
|----------------------------|----------------------------------|
| SP0 visual freeze | Hitting EE research frontier 84/120 |
| SP1 H-EE-022 full validation → **pass or reject** | Opening final holdout |
| SP2 H-EE-023 full validation → **pass or reject** | Phase 6b vision BC |
| SP3 impulse **diagnosis** written (train optional) | Joint pick-place full productization |
| researchnotes + AGENTS + report updates | “EE fixed forever” |

**Done = residual program advanced with hard evidence**, not “numbers look nice.”
A rejected hypothesis with a comparison JSON and a `rejected` status in `researchnotes.md` **counts as progress** and must be crossable.

---

## PASTE INTO `/goal`

```text
GOAL: Execute the post-H-EE-014 residual program under a frozen hybrid baseline until SP0, SP1, and SP2 are each COMPLETE (pass OR reject with evidence), SP3 diagnosis is written, and researchnotes/AGENTS/report reflect outcomes. Prefer real closed-loop metric gains; equally accept rigorous rejection of ideas that fail pre-registered bars.

Authority files (read first, obey):
- prompts/post-h-ee-014-residual-plan.md
- prompts/goal-post-h-ee-014-residual.md  (this contract’s full file in repo)
- researchnotes.md → “Post-H-EE-014 residual program”
- AGENTS.md (do not open final; do not start Phase 6b)
- Baseline: outputs/h_ee_014_nn_gripper_global_validation/

FROZEN BASELINE (do not drift):
- policy: hybrid_nn_gripper_mlp A1 compositor
- loss: global_gripper
- protocol-v2 validation only; legacy_progress_phase
- MLP 128 128, 300 epochs, batch 1024, lr 1e-3, wd 1e-5, seeds 0-4
- both action spaces unless a phase is labeled EE-only diagnostic
- shield/FSM off except SP2b labeled diagnostic
- historical MATCH_FEATURE_INDICES unless SP1 uses a *named* secondary match contract
- compare every train against hybrid EE 62/120 EO 79 phys 68 reopen 0 worst 9 early_close 11
  (joint 97/120) from h_ee_014_nn_gripper_global_validation — NOT against pure-MLP-only

HARD RULES:
1. One causal change per full validation train. Never combine SP1 match-set + SP2 A2 arm-only in the same first run.
2. Registered claims require full 5 seeds × 24 trials (no eval-limit). Smoke may use eval-limit=1 only to prove plumbing.
3. Success from train MSE alone is forbidden. Closed-loop gates only.
4. Final holdout (eval-split final) is CLOSED. Phase 6b vision BC is BLOCKED.
5. Do not relax physics gates. Do not redefine success by dropping impulse/force limits.
6. Rejection is a valid outcome: if pass bars fail, mark hypothesis rejected/partial, write comparison JSON, update researchnotes, and proceed — do not thrash infinite variants.
7. Max 1 retry variant per SP after a clean failure (e.g. one match-set tweak). Then reject and move on.
8. Branch: research/post-h-ee-014-residual (create from current research branch if needed).
9. Use existing venv: PYTHONPATH=src .venv/bin/python …
10. After each completed SP: update researchnotes status + results log; AGENTS bullet only if verdict moves; report under reports/YYYY-MM-DD-*.md for large changes.

════════════════════════════════════════
COMPLETION CRITERIA (ALL must be true)
════════════════════════════════════════

C0. SP0 COMPLETE — Visual residual freeze
  Required artifacts:
  - ≥1 MP4 or rendered review path for each class:
    (a) missing_lift thrash  (b) impulse almost-win  (c) early_close vertical
  - outputs/h_ee_014_residual_visual_review.md listing absolute/openable paths,
    trial_id, seed, failure_category, and 2–4 telemetry numbers per clip
  - Residual counts still consistent with baseline jsonl anatomy
    (missing_lift~30, impulse almost-win~15, early_close=11) or documented if re-counted
  Pass/fail: DONE when artifacts exist. No success-rate bar.

C1. SP1 COMPLETE — H-EE-022 named relative match-set (early-close)
  Required:
  - Implementation of a *named* match contract (e.g. match_relative_ee) without
    silently changing the historical default
  - Full validation under frozen hybrid+global_gripper (5 seeds, both spaces)
  - outputs/.../h_ee_022_comparison.json with pre-registered bars and deltas vs hybrid baseline
  Pass bars (EE vs hybrid baseline):
    - early_close ≤ 5/120  OR  ≤ −50% relative vs 11
    - reopen ≤ 5 total
    - success ≥ 59/120  (baseline 62 − 3)
    - worst seed ≥ 8/24
  Outcomes:
    - PASS → status confirmed (or partial if only early_close improves); freeze match contract name
    - FAIL → status rejected; keep historical MATCH_FEATURE_INDICES; do not loop forever
  Either PASS or FAIL with evidence counts as C1 complete.

C2. SP2 COMPLETE — H-EE-023 A2 arm-only MLP under frozen NN gripper
  Required:
  - MLP train masks gripper residual (weight 0); rollout gripper still NN hybrid
  - Same frozen match set as selected after SP1 (historical if SP1 rejected)
  - Full validation 5 seeds both spaces
  - outputs/.../h_ee_023_comparison.json vs hybrid A1 baseline
  Pass bars (EE vs hybrid baseline):
    - success ≥ +10/120 (→ ≥72)  OR  missing_lift EO bucket ≤ −30% relative vs ~30
    - reopen ≤ 5
    - worst seed ≥ +2 absolute (→ ≥11)  OR  ≥ 11/24
    - joint success ≥ baseline − 10 (→ ≥87)
  Outcomes:
    - PASS / PARTIAL / REJECT with evidence all complete C2
  Do NOT combine with a new match-set in this first A2 run.

C3. SP3 DIAGNOSIS COMPLETE — H-EE-024 impulse almost-wins (train optional)
  Required (diagnosis floor):
  - Written diagnosis in outputs/.../h_ee_024_impulse_diagnosis.json or .md
  - For the ~15 EO+lift+retain contact_dynamics fails: which gate fails
    (impulse vs force vs xy), mean impulse, and whether path looks “hard shove”
  - Explicit decision: “no train yet” OR one registered path/demo change with bars
  Optional train only if diagnosis justifies it; if trained:
    - phys ≥ 85/120 and contact_dynamics fails ≤ 6/120 without gate relaxation
  C3 is complete with diagnosis alone if no train is justified.

C4. DOCUMENTATION COMPLETE
  - researchnotes.md: H-EE-022/023/024 status filled; residual program SP rows updated;
    results log rows for each run; priority list reflects survivors
  - AGENTS.md: one dated verdict bullet if any SP passed or residual story changed
  - reports/YYYY-MM-DD-post-h-ee-014-residual-progress.md summarizing:
      what improved, what was rejected, current best EE/joint numbers, next open SP
  - Scoreboard file: outputs/post_h_ee_014_residual_scoreboard.json (schema below)

C5. SCOREBOARD HONESTY
  outputs/post_h_ee_014_residual_scoreboard.json must include:
  {
    "baseline_ee": {"successes": 62, "event_order": 79, "phys": 68, "reopen": 0,
                    "worst_seed": 9, "early_close": 11, "missing_lift_eo": 30},
    "best_ee_after_goal": { ... actual numbers ... },
    "sp0": "complete",
    "sp1": "passed"|"rejected"|"partial",
    "sp2": "passed"|"rejected"|"partial",
    "sp3": "diagnosed"|"passed"|"rejected"|"skipped_justified",
    "hypotheses_crossed_off": ["..."],
    "hypotheses_confirmed": ["..."],
    "final_accessed": false,
    "phase6b_started": false,
    "frontier_named": "legacy_84" | "stretched_hybrid_joint_97",
    "meets_legacy_frontier": false/true,
    "notes": "..."
  }
  Goal may complete with meets_legacy_frontier=false if C0–C5 satisfied.

════════════════════════════════════════
EXPLICIT NON-GOALS (do not pursue to “finish”)
════════════════════════════════════════
- eval-split final / final holdout
- Phase 6b / vision-conditioned BC / VLA
- H-EE-016/018/019, pure gripper MSE reweight, distance-guard shields
- Infinite hyperparam search; max 1 retry variant per SP
- Claiming success from supervised loss or smoke eval-limit runs
- Relaxing MAX_GRIPPER_* / physical sanity gates

════════════════════════════════════════
EXECUTION ORDER (mandatory)
════════════════════════════════════════
1) SP0 visual freeze → write visual review
2) SP1 H-EE-022 implement + smoke + full validation → comparison → pass/reject
3) SP2 H-EE-023 implement + smoke + full validation → comparison → pass/reject
   (use historical match if SP1 rejected; use SP1 contract only if SP1 passed and frozen)
4) SP3 impulse diagnosis (train only if justified)
5) Scoreboard + researchnotes + AGENTS + progress report
6) Call update_goal completed only when C0–C5 all true

Optional if time remains AFTER C0–C5:
- SP4 H-EE-007 one-seed/label probe
- SP2b H-EE-015 only if SP1 rejected AND early_close still ≥8
- SP6 joint pick-place is OUT OF SCOPE for this goal unless user expands scope

════════════════════════════════════════
EVIDENCE COMMANDS (shape; adjust output dirs)
════════════════════════════════════════
# Full hybrid-compatible train (after wiring flags for match-set / arm-only):
PYTHONPATH=src .venv/bin/python scripts/train_state_bc.py \
  --output-dir outputs/<run_name> \
  --evaluation-protocol v2 --eval-split validation \
  --temporal-feature-mode legacy_progress_phase \
  --policy-type mlp --hybrid-nn-gripper --loss-profile global_gripper \
  --seeds 0 1 2 3 4 --hidden-sizes 128 128 --epochs 300 \
  --batch-size 1024 --learning-rate 0.001 --weight-decay 1e-5 \
  --stride 1 --max-steps 3200 --action-gain 1.0 --label-source policy_labels

# Smoke only (plumbing; not evidence for pass bars):
# add --eval-limit 1 --epochs 2  (must not be cited as validation)

# Tests before long runs:
PYTHONPATH=src .venv/bin/python -m pytest tests/test_state_bc.py tests/test_loss_profiles.py -q

════════════════════════════════════════
PROGRESS REPORTING
════════════════════════════════════════
Use update_goal messages at SP boundaries, e.g.:
- “SP0 complete: 3 clips + visual review written”
- “SP1 full val done: early_close 11→X, success 62→Y → REJECTED/PASSED”
- “SP2 full val done: missing_lift 30→X, success 62→Y → …”
When blocked after 3+ failed attempts on the same bug, set blocked_reason.
When C0–C5 true, set completed=true with a summary that lists pass/reject per SP
and best EE numbers vs baseline 62/79/68/9.

Be a critical research partner: if an idea fails, cross it off loudly with paths.
If it passes, freeze it and do not silently stack untested changes.
```

---

## Scoreboard schema (canonical)

Write `outputs/post_h_ee_014_residual_scoreboard.json` before claiming goal complete:

```json
{
  "format": "svla_post_h_ee_014_residual_scoreboard_v1",
  "goal": "post-h-ee-014-residual",
  "baseline_source": "outputs/h_ee_014_nn_gripper_global_validation/",
  "baseline_ee": {
    "successes": 62,
    "event_order": 79,
    "phys": 68,
    "reopen": 0,
    "worst_seed": 9,
    "early_close": 11,
    "missing_lift_eo": 30,
    "impulse_almost_wins": 15
  },
  "baseline_joint": {
    "successes": 97,
    "event_order": 107,
    "phys": 103,
    "worst_seed": 15
  },
  "sp0": { "status": "complete", "artifact": "outputs/h_ee_014_residual_visual_review.md" },
  "sp1": {
    "status": "passed|rejected|partial",
    "hypothesis": "H-EE-022",
    "output_dir": "outputs/...",
    "comparison": "outputs/.../h_ee_022_comparison.json",
    "ee": {},
    "bars_met": {}
  },
  "sp2": {
    "status": "passed|rejected|partial",
    "hypothesis": "H-EE-023",
    "output_dir": "outputs/...",
    "comparison": "outputs/.../h_ee_023_comparison.json",
    "ee": {},
    "bars_met": {}
  },
  "sp3": {
    "status": "diagnosed|passed|rejected|skipped_justified",
    "hypothesis": "H-EE-024",
    "artifact": "outputs/..."
  },
  "best_ee_after_goal": {},
  "best_joint_after_goal": {},
  "hypotheses_confirmed": [],
  "hypotheses_crossed_off": [],
  "final_accessed": false,
  "phase6b_started": false,
  "frontier_named": "legacy_84",
  "meets_legacy_frontier": false,
  "meets_stretched_frontier": false,
  "notes": ""
}
```

---

## Real targets (what “improved outputs” means)

### Primary (must measure every full val)

| Metric | Baseline hybrid EE | Stretch nice-to-have | Goal-completion requirement |
|--------|-------------------:|---------------------:|----------------------------|
| Success | 62/120 | ≥72 (SP2 bar) or ≥84 frontier | SP bars only; frontier **not** required |
| Event order | 79/120 | ≥90 | report only |
| Phys | 68/120 | ≥85 if SP3 trains | report; SP3 optional train bar |
| Worst seed | 9/24 | ≥11 (SP2) or ≥12 frontier | SP bars |
| Reopen | 0 | stay ≤5 | hard floor both SP1/SP2 |
| Early close | 11 | ≤5 (SP1) | SP1 bar |
| Missing-lift EO | ~30 | ≤21 (−30%) | SP2 alternate bar |

### Secondary (telemetry, never substitute for success)

- mean joint-limit steps on fails vs success
- mean gripper flips (should stay ~1.0)
- impulse distribution on almost-wins
- per-seed success vector

---

## Time budget (honest)

| Work | Approx wall time (this machine class) |
|------|----------------------------------------|
| SP0 render 3 clips | ~10–30 min |
| SP1 implement + tests + smoke | ~30–60 min |
| SP1 full 5-seed both spaces | ~12–25 min train/eval (similar to H-EE-014) |
| SP2 implement + tests + smoke | ~30–60 min |
| SP2 full validation | ~12–25 min |
| SP3 diagnosis | ~30–90 min |
| Docs/scoreboard | ~20–40 min |
| **Total** | **~3–8 hours** depending on thrash |

If the session is shorter: complete SP0+SP1 fully (pass or reject), mark goal **not** complete, leave scoreboard partial — do not claim C0–C5.

---

## Failure handling (how to cross ideas off)

When a SP fails bars:

1. Write comparison JSON with `"status": "rejected"` and exact deltas.
2. Set hypothesis status in `researchnotes.md` to `rejected` or `partial`.
3. Append results log row with evidence path.
4. Add ID to `hypotheses_crossed_off` in scoreboard.
5. **Stop variants** after at most one retry.
6. Proceed to next SP with frozen survivors only.

Example rejection language (good):

> H-EE-022 rejected: relative match-set early_close 11→9 (bar ≤5); reopen 0; success 62→58. Historical match retained. Evidence: `outputs/.../h_ee_022_comparison.json`.

---

## Why this is a good goal-mode shape

1. **Falsifiable** — each SP has numeric bars.
2. **Rejection counts** — skeptics can verify “rejected” as success of the *process*.
3. **No final/vision trap** — hard non-goals prevent scope explosion.
4. **One-causal-change** — avoids confounded “we changed three things and numbers moved.”
5. **Scoreboard** — single artifact for you to scan without reading every log.
6. **Baseline is hybrid 62**, not pure MLP 49 — so improvements are real increments on the current best.

---

## Optional: shorter goal (if you only want one experiment night)

Paste this instead for a smaller goal:

```text
GOAL: Complete SP0 visual freeze + SP1 H-EE-022 full validation (pass OR reject) under frozen hybrid+global_gripper baseline (EE 62/120). Write h_ee_022_comparison.json, update researchnotes, and outputs/post_h_ee_014_residual_scoreboard.json for SP0+SP1 only. Do not open final. Do not start SP2 unless SP1 finishes early with time remaining. Full contract: prompts/goal-post-h-ee-014-residual.md
```

---

## Launch checklist

1. Commit or stash WIP so manifests are clean if possible.
2. Ensure baseline still at `outputs/h_ee_014_nn_gripper_global_validation/`.
3. `/goal` + paste the main block (or short variant).
4. Let it run; check `/goal status` periodically.
5. When complete, open:
   - `outputs/post_h_ee_014_residual_scoreboard.json`
   - `reports/*post-h-ee-014-residual*`
   - any new `outputs/h_ee_022_*` / `h_ee_023_*`
