# Post-H-EE-014 Residual Program — Sub-Phase Plan

**Status:** SP0–SP3 complete (2026-07-09) — SP1/SP2 rejected; SP3 diagnosed no-train
**Parent result:** H-EE-014 confirmed (`reports/2026-07-09-h-ee-014-nn-gripper.md`)
**Living notes:** `researchnotes.md` → “Post-H-EE-014 residual program”
**Scoreboard:** `outputs/post_h_ee_014_residual_scoreboard.json`
**Progress report:** `reports/2026-07-09-post-h-ee-014-residual-progress.md`
**Do not open final holdout.**
**Do not start Phase 6b / vision BC.**
**Best EE remains hybrid A1 62/120** (do not freeze A2).

This plan turns the post-014 residual analysis into **ordered sub-phases** with
falsifiable bars. Prefer one causal change per train run.

---

## 0. Frozen baseline (do not drift)

| Knob | Value |
|------|-------|
| Policy | `hybrid_nn_gripper_mlp` (A1 compositor) |
| Loss profile | `global_gripper` |
| Protocol | v2 **validation** only |
| Temporal | `legacy_progress_phase` |
| MLP | 128 128, 300 ep, batch 1024, lr 1e-3, wd 1e-5 |
| Seeds | 0 1 2 3 4 |
| Spaces | both (unless a phase is explicitly EE-only diagnostic) |
| Shield / FSM | **off** (except SP2b labeled diagnostic) |
| Match set default | historical `MATCH_FEATURE_INDICES` unless SP1 named contract |

**Baseline metrics (H-EE-014 hybrid):**

| | EE | Joint |
|--|---:|------:|
| Success | 62/120 | 97/120 |
| Event order | 79 | 107 |
| Physical sanity | 68 | 103 |
| Reopen | 0 | 0 |
| Worst seed | 9/24 | 15/24 |
| Early close | 11 | 5 |

Evidence dir: `outputs/h_ee_014_nn_gripper_global_validation/`.

---

## 1. What H-EE-014 actually solved vs what remains

**Solved:** gripper reopen/oscillation (155→0; flips=1.0).
**Not solved:** three residual classes (EE):

| Class | n | Telemetry snapshot | SP |
|-------|--:|--------------------|----|
| Missing lift | 30 | contact yes; mean lift ~5.8 mm; joint-limit thrash ~965 vs ~92 success | SP2 |
| Impulse almost-win | 15 | EO+lift+retain; 13/15 over impulse thr ~11.4 vs 9 | SP0/SP3 |
| Early close | 11 | close ~18 mm; all `vertical_pregrasp` | SP1 |

---

## 2. Sub-phases

### SP0 — Visual residual freeze (do first)

**Goal:** Confirm the three residual stories with clips before more training.

**Actions:**

1. Render ≥1 trial per class (missing_lift thrash, impulse almost-win, early_close vertical).
2. Optionally write `outputs/h_ee_014_residual_visual_review.md` with paths.
3. Decide whether reports use **legacy frontier (84)** or also cite **stretched hybrid-joint frontier (97)** — name it explicitly.

**Done when:** clips exist + residual table still matches telemetry.
**No pass bar on success rate.**

---

### SP1 — H-EE-022 named relative match-set (early-close)

**Claim:** Adding relative EE–object (and optionally `gripper_open`) to the NN match set
reduces early-close without reintroducing reopens.

**Rules:**

- New contract must be a **named** string (e.g. `match_relative_ee`).
- Historical `MATCH_FEATURE_INDICES` remains default for comparability.
- Prefer fit-NN-only compositor first (same MLP weights if possible) so arm train is not confounded; full retrain only if needed.

**Pass bars (vs hybrid baseline EE):**

| Metric | Bar |
|--------|-----|
| early_close | ≤ 5/120 (or ≤ −50% relative) |
| reopen | ≤ 5 total (must not return to 50+) |
| success | ≥ baseline − 3 (no large regression) |
| worst seed | ≥ 8/24 |

**If fail:** keep historical match; go SP2b (FSM) only as diagnostic, or SP2 arm-only.

---

### SP2 — H-EE-023 A2 arm-only MLP under frozen NN gripper (main gap)

**Claim:** Masking gripper from MLP training improves post-close lift/path because MLP
capacity/gradients are no longer spent on a dim that NN overwrites.

**Rules:**

- Gripper at rollout still NN (hybrid A1 compositor).
- MLP train: gripper residual weight 0 / mask (not a new loss profile name unless registered).
- **Do not** combine with SP1 match-set change in the first causal run.
- Historical match set unless SP1 already selected as frozen.

**Pass bars (vs hybrid baseline EE):**

| Metric | Bar |
|--------|-----|
| success | ≥ +10 / 120 **or** missing_lift EO bucket ≤ −30% relative |
| reopen | ≤ 5 |
| worst seed | ≥ +2 absolute or ≥ 11/24 |
| joint | ≥ baseline − 10 / 120 |

**If partial (reopen still 0, lift better, success flat):** keep A2; attack SP3 impulse.
**If fail:** do not stack more loss tricks; diagnose thrash (SP0/SP4 gain/labels).

---

### SP2b — H-EE-015 FSM gripper (diagnostic only)

**When:** SP1 fails and early-close remains material.
**Label:** diagnostic arm upper bound — **not** the learned EE vs joint comparison.
**Pass bar:** early_close → 0; report arm-limited success separately from hybrid learned gripper.

---

### SP3 — H-EE-024 impulse almost-wins

**Claim:** The 15 EO+lift+retain fails are a force/path residual, not gripper timing.

**Phase A (required first):** telemetry + visual — which motion produces excess impulse.
**Phase B (only if A clear):** softer close / approach path / expert change — **never** gate relaxation.

**Pass bar for “understood”:** mechanism written with evidence paths.
**Pass bar for “fixed” (only after B):** contact_dynamics fails ≤ 6/120 and phys ≥ 85 without gate change.

---

### SP4 — Parallel cheap probes

| Probe | ID | Cost | Question |
|-------|-----|------|----------|
| Label asymmetry | H-EE-007 | low | EE policy_labels reconstruction noise? |
| Gain/cap under hybrid | H-EE-002 | medium | thrash causal vs symptom? |

Do not block SP1/SP2 on these; run in parallel if capacity allows.

---

### SP5 — Selection / final

Open final **only if**:

1. Named EE contract frozen (hybrid + loss + match + A1/A2).
2. Aggregate success **and** worst-seed clear the **named** frontier bar.
3. Phys not left collapsed.
4. Human explicitly approves.

Default: final stays closed at current 62/9/68.

---

### SP6 — Joint pick-place track (optional)

Joint hybrid **97/120** makes joint-only pick-place BC more defensible as a **parallel track**.
Must not be sold as “EE fixed.” Keep pickup hybrid EE residual program primary for the research comparison.

---

### SP7 — Phase 6b vision (blocked)

Requires a newly justified temporal **and** gripper contract (hybrid or better).
Do not bolt RGB onto pure MLP gripper + open-loop clock.

---

## 3. Suggested experiment sequence

```text
SP0  visual freeze
  ├─ SP1  H-EE-022 match-set (early-close)
  ├─ SP2  H-EE-023 A2 arm-only (missing_lift)   [not same train as SP1]
  ├─ SP4  H-EE-007 / optional H-EE-002
  └─ SP3  H-EE-024 impulse (after SP0; train only if mechanism clear)
       └─ SP2b only if SP1 fails
SP5 final only if bars + human
SP6 joint pick-place optional parallel
SP7 vision blocked
```

---

## 4. Reporting contract

After each SP that trains/evaluates:

1. `outputs/.../state_bc_summary.json` (+ manifest)
2. Comparison vs **H-EE-014 hybrid baseline** (not only vs pure MLP global)
3. Residual bucket counts: missing_lift, early_close, impulse almost-wins, reopen, flips
4. `researchnotes.md` status + results log row
5. `AGENTS.md` verdict bullet only if phase status moves
6. `reports/YYYY-MM-DD-...md` for large changes

---

## 5. Bottom line

| Question | Answer |
|----------|--------|
| What did 014 buy? | Reopen/hold solved; hybrid is frozen baseline |
| What is the main EE gap? | Missing-lift thrash + impulse almost-wins (+ small early-close) |
| First train? | Prefer SP1 **or** SP2 alone after SP0 — not both at once |
| Final / vision? | Still closed / blocked |
| Frontier? | Name legacy 84 vs stretched hybrid-joint 97 |

---

## 6. Fresh-thread starter

```text
Execute the next open sub-phase from prompts/post-h-ee-014-residual-plan.md
and researchnotes.md “Post-H-EE-014 residual program”.

Frozen baseline: hybrid A1 + global_gripper, protocol-v2 validation,
outputs/h_ee_014_nn_gripper_global_validation/ as comparison floor.
Do NOT open final. Do NOT start Phase 6b.
One causal change per train. Report residual buckets, not train loss.
```
