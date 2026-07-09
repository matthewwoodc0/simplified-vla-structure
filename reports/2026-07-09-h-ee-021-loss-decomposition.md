# 2026-07-09 - H-EE-021 Loss-Profile Decomposition

## Plain-English Summary

H-EE-008 improved both action spaces by reweighting gripper MSE, but it changed **two
things at once**:

1. Global gripper errors became 5× more important everywhere.
2. Transition phases (`grasp_align` / `close_gripper`) became 10× more important.

So 008 is a **causal clue**, not the answer. We still do not know whether the gain came
from global emphasis, transition emphasis, or both — and EE at 50/120 remains far behind
the weighted joint research frontier of **84/120**.

This change builds the frozen loss-decomposition harness and matrix runner so those four
conditions can be compared under one commit, with no rollout shields and no final holdout
access.

## What To Review

- [ ] `src/svla/loss_profiles.py` — frozen profile contract + research parity frontier.
- [ ] `src/svla/state_bc.py` — profile-aware weights, phase loss report, event-timing fields.
- [ ] `scripts/train_state_bc.py` — `--loss-profile` CLI.
- [ ] `scripts/run_loss_decomposition.py` — four-profile causal matrix + diagnosis aggregate.
- [ ] `tests/test_loss_profiles.py` — proves weight matrices and arm dims unchanged.
- [ ] `researchnotes.md` / `AGENTS.md` — H-EE-021 status and next-proposal table.

## Implementation Details

### Frozen profiles (no rollout changes)

| Profile | Global gripper weight | Transition weight |
|---------|----------------------:|------------------:|
| `uniform` | 1× | 1× |
| `global_gripper` | 5× | 5× |
| `transition_gripper` | 1× | 10× |
| `combined_h_ee_008` | 5× | 10× |

Arm action dims always stay at weight **1.0**. Transition phases remain
`grasp_align` and `close_gripper`.

### Harness records

Each profile run records:

- exact named profile + weight contract
- phase sample counts
- per-phase arm vs gripper train residual MSE
- seeds, protocol hash, source hashes (via existing experiment manifests)
- raw closed-loop metrics
- compact rollout diagnosis (close distance, event times, flips/reopens, constraint rates)

### Research parity frontier

Not a release claim. Target for EE catch-up is H-EE-008 weighted joint validation:

- success 84/120
- event order 90/120
- physical sanity 100/120
- worst seed ≥12/24

Selection rule: choose one EE contract only if aggregate **and** worst-seed improve; then
freeze and open final once.

### Phase-2 → next hypothesis map

| Evidence | Proposal |
|----------|----------|
| Global helps, transition does not | H-EE-018 adaptive gripper-gradient balancing |
| Transition helps most | H-EE-019 narrow close-boundary curriculum |
| All profiles still state-local timing errors | H-EE-014 NN gripper + MLP arm |
| Similar states → different gripper labels | H-EE-017 short history / tiny recurrent |
| Failures off-support at transition | H-EE-020 targeted boundary demos |

Explicitly deprioritized as immediate next: **H-EE-016** (mostly repeats transition 10×)
and lower EE gain (saturation not supported as the simple cause).

## Evidence And Verification

### Unit tests

```bash
PYTHONPATH=src .venv/bin/pytest tests/test_loss_profiles.py tests/test_state_bc.py -q
```

- Result: **21 passed**
- What this proves: every profile builds the intended weight matrix; arm dims stay 1.0;
  phase sample/loss reporting works; legacy H-EE-008 weight API still fits.
- What it does not prove: which profile wins closed-loop.

### Dry-run / matrix command

```bash
PYTHONPATH=src .venv/bin/python scripts/run_loss_decomposition.py --dry-run

PYTHONPATH=src .venv/bin/python scripts/run_loss_decomposition.py \
  --output-dir outputs/h_ee_021_loss_decomposition
```

Full matrix: 4 profiles × both action spaces × 5 seeds × 24 validation trials, epochs 300.
Expected wall time is on the order of the historical H-EE-008 run × 4 (~45–60+ minutes).

Historical H-EE-008 is **not** reused as a causal cell: its manifest was dirty/untracked
and source hashes will not match the frozen H-EE-021 commit. `combined_h_ee_008` is
rerun under the same commit as the other three profiles.

## Demo Videos / Visual Artifacts

None yet. After the matrix finishes, optional per-seed renders can be produced with
`scripts/render_bc_rollout.py` for the winning EE profile. Generated videos are gitignored.

## Decisions Made

- Decision: Treat H-EE-008 as a factorial clue and decompose before more architecture work.
  Reason: two simultaneous interventions make the “why it worked” claim underdetermined.
- Decision: Rerun `combined_h_ee_008` rather than reuse historical 008 artifacts.
  Reason: causal comparisons require one frozen source/data/config commit.
- Decision: Keep final holdout closed.
  Reason: selection must happen on validation first; EE still not at the joint frontier.
- Decision: Deprioritize H-EE-016 and lower EE gain as immediate next tests.
  Reason: 008 already overweights the transition; saturation is not the simple story.

## Risks And Limitations

- Risk: seed instability can make one-cell “wins” look larger than they are.
  Why it matters: selection requires worst-seed improvement, not only aggregate.
- Risk: train residual reports are supervised, not closed-loop causality.
  Why it matters: Phase-3 diagnosis uses rollout event timing to interpret the winner.
- Risk: full matrix is expensive.
  Why it matters: do not partial-slice (`--eval-limit`) for selection evidence.

## Action Items

- [ ] Run full H-EE-021 matrix on this branch and freeze comparison JSON.
- [ ] Inspect EE success / event-order / worst-seed across four profiles.
- [ ] Use rollout diagnosis to choose H-EE-018 / 019 / 014 / 017 / 020.
- [ ] Do not open final until one EE contract is selected.

## Files Changed

- `src/svla/loss_profiles.py` — frozen profiles + frontier constants.
- `src/svla/state_bc.py` — profile weights, phase losses, event timing fields.
- `scripts/train_state_bc.py` — `--loss-profile` integration.
- `scripts/run_loss_decomposition.py` — matrix runner + comparison/diagnosis.
- `tests/test_loss_profiles.py` — contract tests.
- `researchnotes.md` — H-EE-021 + conditional proposals + frontier.
- `AGENTS.md` — verdict update and next-work redirect.
- `reports/2026-07-09-h-ee-021-loss-decomposition.md` — this report.

## Phase 2 Results (completed)

| Profile | EE | EO | Phys | early | preclose | reopen | EE worst | Joint |
|---------|---:|---:|---:|-----:|---------:|-------:|---------:|------:|
| uniform | 31 | 38 | 64 | 5 | 583 | 190 | 0 | 53 |
| global_gripper | **49** | **60** | 68 | **2** | 70 | 155 | **4** | 76 |
| transition_gripper | 38 | 41 | 63 | 5 | 128 | 187 | 2 | 68 |
| combined_h_ee_008 | **50** | 55 | 71 | 5 | **50** | **142** | 2 | **84** |

EE deltas vs uniform: global **+18**, transition **+7**, combined **+19** (interaction −6).

Replication: uniform matches registered legacy (31/53); combined matches historical H-EE-008 (50/84).

## Phase 3 Diagnosis (global vs combined)

Artifact: `outputs/h_ee_021_loss_decomposition/h_ee_021_global_vs_combined_diagnosis.json`

- Pairwise EE: both_ok 30, both_fail 51, global_only 19, combined_only 20 — high seed lottery, not a clean dominance.
- Event-order failure anatomy is **reopen-dominated** (global 55/60 EO fails; combined 52/65).
- Successes: ~**1 gripper flip**. Failures: ~**5 flips**.
- Early-close is rare and not the residual story.
- Best seeds (global/combined seed0: 20–22/24) have low flips/reopen; worst seeds (2–4/24) have high flips/reopen and high joint-limit exposure.
- Supervised residuals fit all profiles well; train loss does not select the winner.

## Current Verdict

**H-EE-021 confirmed as a causal split.** Global 5× is the main EE driver; transition 10×
is weak alone; combined is not super-additive for EE. Joint still needs combined for the
research frontier (84/90/100). **No EE profile is selectable for final.**

**Next:** H-EE-014 (NN gripper + MLP arm), preferably under `global_gripper` loss for EE
reliability metrics; use `combined_h_ee_008` only if the comparison must keep joint parity
in the same training contract. Alternate: H-EE-017 if reopens persist. Deprioritize
H-EE-016 / H-EE-018 / H-EE-019 as immediate next tests.
