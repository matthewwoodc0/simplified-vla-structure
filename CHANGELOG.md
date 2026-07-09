# Changelog

## 2026-07-09 - Research Codebase Audit

### Safe fixes

- `scripts/capture_pick_place_vplan_evidence.py` — replaced a deleted,
  machine-specific temporary directory with a repo-local ignored output directory and an
  explicit `--scratch-dir` override. This changes only where diagnostic logs are written.
- `LEARNING_GUIDE.md` — replaced obsolete Phase 5 results and the outdated “vision not
  started” roadmap with the registered raw-final, H-EE-014 validation, and Phase 6a/6b
  status. This is documentation-only and does not change experiment behavior.

### Audit baseline

- Full suite before audit edits: `114 passed` outside the sandbox. The sandbox-only run
  produced eight expected MuJoCo/CoreGraphics failures and `106 passed`.
- Full suite after the audit/reorganization: `125 passed` outside the sandbox.

### Research-codebase organization

- `src/svla/core/action_space.py` — added the canonical encode/decode/execute registry and
  routed state rollout, policy-label replay, and BC video rendering through it.
- `src/svla/eval/` — moved protocol and provenance implementation into a dedicated package;
  retained compatibility imports at the original module paths.
- `src/svla/experiments/config.py`, `experiments/run.py`, and `experiments/configs/` — added
  validated, hashable, dry-run-first experiment configs for the maintained Phase 5/H-EE
  experiments. H-EE-003 is honestly marked historical-only because its rejected output-head
  implementation no longer exists.
- `analysis/policy_failures.py` — moved offline failure analysis out of launch scripts and
  retained the original CLI as a compatibility wrapper.
- `RESULTS.md` and `README.md` — added a current evidence index, experiment matrix, and the
  new repository map.
- `tests/test_action_representation.py`, `tests/test_experiment_config.py`, and
  `tests/test_research_smoke.py` — added representation round-trip, config/final-access, tiny
  training, and closed-loop smoke coverage.
