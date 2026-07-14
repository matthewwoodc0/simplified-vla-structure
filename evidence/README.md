# Experiment Evidence and Manifests

Large experiment artifacts live under `outputs/`, which is gitignored. Each major
experiment script now writes a small sidecar manifest next to its primary output so
ignored artifacts remain auditable without checking models, videos, or JSONL into git.

## Sidecar naming

For a primary output like `outputs/pickup_trials.jsonl`, the manifest is:

```text
outputs/pickup_trials.manifest.json
```

Scripts that write a single summary JSON use the same pattern, for example
`outputs/action_replay_tool_axis_summary.manifest.json`.

## What a manifest records

Each manifest uses format `svla_experiment_manifest_v1` and includes:

- UTC timestamp of the run
- Exact command and arguments
- Git commit SHA and whether the working tree was dirty
- SHA-256 of the working-tree diff when dirty
- Paths and SHA-256 identities of untracked files when present
- Python, MuJoCo, NumPy, and SciPy versions
- SHA-256 hashes of tracked source and asset files
- Pickup controller limits and physics gate constants
- Random or training seeds used by the script
- SHA-256 hashes of generated output files listed in the manifest

## Covered scripts

- `scripts/run_pickup_trials.py`
- `scripts/run_pick_place_trials.py`
- `scripts/validate_action_replay.py`
- `scripts/validate_task_robustness.py`
- `scripts/train_state_bc.py`

## How to audit an ignored output

1. Locate the sidecar manifest beside the artifact, for example
   `outputs/state_bc/state_bc_summary.manifest.json`.
2. Confirm `output_files` contains the artifact path and matching SHA-256.
3. Compare `source_hashes`, `controller_limits`, and `physics_gate_constants` against
   the code revision you expect.
4. Use `git_commit_sha`, `git_diff_sha256`, and `git_untracked_files` together. A matching
   tracked diff is not sufficient when experiment code or config is untracked.

## Phase-5 protocol-v2 records

Small tracked evidence records point to the ignored raw artifacts and freeze the conclusions:

- `evidence/phase5_v2_model_selection.json` records validation candidates, the shared
  selected temporal contract, and the final-access policy.
- `evidence/phase5_v2_final_results.json` records raw and guarded final metrics, verifies
  matching source identity and byte-identical models, and keeps shielded evidence separate.
- `evidence/phase5_causal_synthesis.json` freezes the 2026-07-14 pickup rescue chapter
  closeout: source artifact SHA-256 hashes, decisive probe ruled-out/not-ruled-out claims,
  the selected fair hybrid comparison contract, and the ordered next program
  (efficiency → pick-place → second controller). Documentation-only; no final access.
- `evidence/state_bc_efficiency_curve_registration.json` preregisters EFF-001 (nested
  demo ladders, 150-fit matrix, development + locked_evaluation splits, endpoints, and
  future primary command). Status `registered_not_run`: primary curve and locked
  evaluation were **not** executed. Companion compact dry-run matrix:
  `evidence/state_bc_efficiency_curve_matrix_dry_run.json`.

The raw selected-policy result is the learned-policy ladder. The guarded result is a
`shielded_policy=true` diagnostic and must never replace or be blended with raw BC. The
4001+ `state_bc_physics_audit_final` grid is historical; protocol-v2 final uses trial IDs
7001+ and five model seeds.

The current source-matched scripted/replay/readiness bundle is
`outputs/phase5_baseline_final/phase5_baseline_v2_aggregate.json`. Its component manifests
must report consistent working-tree identity and verified output hashes; it does not prove
learned-policy, vision, or hardware readiness.

Re-hash any file with:

```bash
shasum -a 256 path/to/file
```

Compare that digest to the manifest entry.
