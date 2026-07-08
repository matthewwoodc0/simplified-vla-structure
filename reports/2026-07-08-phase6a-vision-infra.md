# 2026-07-08 - Phase 6a Vision Infrastructure

## Plain-English Summary

This change adds Phase 6a vision infrastructure only: fixed-camera RGB rendering, scripted
pickup RGB dataset export, dataset validation, and MP4 preview tooling. It does not train a
vision policy, does not add language conditioning, and does not change the Phase 5 learned
policy verdict.

The important boundary is that this records images alongside the existing scripted pickup
demonstration contract. The current state-based BC evidence still says the raw learned
policy comparison is not ready, especially for EE event-order behavior.

## What To Review

- [ ] `src/svla/vision_observations.py`: new fixed-camera RGB renderer and camera metadata
  contract.
- [ ] `src/svla/vision_dataset.py`: scripted demo plus NPZ frame export, manifest format,
  dataset validator, hash checks, and protocol identity.
- [ ] `scripts/record_pickup_vision_demos.py`: CLI for exporting a small scripted RGB
  dataset.
- [ ] `scripts/validate_vision_dataset.py`: CLI wrapper around the validator.
- [ ] `scripts/render_vision_dataset_preview.py`: MP4 preview generation from stored RGB
  frames.
- [ ] `tests/test_vision_dataset.py` and `tests/test_vision_observations.py`: coverage for
  frame alignment, schema validation, hash mismatch detection, protocol hashing, rendering,
  and non-vision demo-loader compatibility.
- [ ] This report and the README/AGENTS/researchnotes updates, because they define the
  current research interpretation.

## Implementation Details

`FixedCameraConfig` defines named RGB camera outputs as uint8 arrays shaped
`[height, width, 3]`. `FixedCameraRenderer` owns MuJoCo renderer objects and returns copied
frames so callers do not retain mutable renderer-owned buffers. `PickupTaskEvaluator` now
has an opt-in `get_rgb_observation()` helper; ordinary state observations are not changed.

`record_pickup_vision_dataset()` records the existing scripted pickup demo, replays the same
scripted controller sequence to capture fixed-camera frames, and stores the frames in
compressed NPZ files rather than embedding RGB arrays in JSON. Each episode stores the demo
path, demo SHA-256, frame count, camera records, frame index, task summary, and phase
summaries. The top-level `vision_manifest.json` records source hashes, protocol SHA,
camera metadata, output files, and label-set metadata.

`validate_pickup_vision_dataset()` checks the manifest format, action-space neutrality,
camera config, episode count, demo schema, task success gates, sample index alignment,
policy/state label presence, frame-index alignment, frame array shape/dtype, and recorded
SHA-256 values for demos, frame NPZ files, and manifest `output_files`. The protocol hash
is resolved from the repository root, not the caller working directory.

The scripts expose the workflow as three explicit steps: record a dataset, validate it, and
render a preview MP4 from stored frames. Preview rendering assumes the dataset is already
valid; validation remains a separate gate.

## Evidence And Verification

```bash
git diff --check
```

- Result: passed.
- Output artifact: none.
- What this proves: the tracked diff does not introduce whitespace errors.
- What it does not prove: runtime behavior, renderer availability, or research validity.

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_vision_dataset.py
```

- Result: passed outside the sandbox with macOS graphics access, `7 passed in 47.04s`.
- Output artifact: pytest output only.
- What this proves: generated datasets validate, tampered demo/frame hashes are detected,
  frame alignment holds for a small scripted RGB dataset, and protocol SHA does not depend
  on caller cwd.
- What it does not prove: full repository behavior or hardware realism.

```bash
PYTHONPATH=src .venv/bin/python -m pytest
```

- Result: passed outside the sandbox with macOS graphics access, `96 passed in 100.15s`.
- Output artifact: pytest output only.
- What this proves: the full repository test suite passes with the Phase 6a modules,
  scripts, documentation exports, and existing Phase 5 behavior in place.
- What it does not prove: readiness-domain robustness, visual acceptability of every
  generated clip, or learned-policy readiness.

```bash
PYTHONPATH=src .venv/bin/python scripts/record_pickup_vision_demos.py \
  --output-dir /private/tmp/phase6a_vision_smoke_20260708 \
  --count 1 --width 64 --height 48
```

- Result: passed; wrote a one-episode dataset and sidecar manifest.
- Output artifact: `/private/tmp/phase6a_vision_smoke_20260708/vision_manifest.json`.
- What this proves: the dataset recording CLI can generate the scripted RGB artifact path.
- What it does not prove: large dataset performance or learned-policy readiness.

```bash
PYTHONPATH=src .venv/bin/python scripts/validate_vision_dataset.py \
  --dataset-dir /private/tmp/phase6a_vision_smoke_20260708 \
  --output /private/tmp/phase6a_vision_smoke_20260708/validation_summary.json
```

- Result: passed; `valid=true`, `issues=[]`, `episode_count=1`, `total_frames=1582`.
- Output artifact: `/private/tmp/phase6a_vision_smoke_20260708/validation_summary.json`.
- What this proves: the CLI validator accepts the generated dataset and verifies the
  recorded artifact contract for this smoke case.
- What it does not prove: visual quality or downstream training readiness.

```bash
PYTHONPATH=src .venv/bin/python scripts/render_vision_dataset_preview.py \
  --dataset-dir /private/tmp/phase6a_vision_smoke_20260708 \
  --output /private/tmp/phase6a_vision_smoke_20260708/preview.mp4 \
  --stride 20 --fps 24
```

- Result: passed; wrote an 80-frame MP4 preview.
- Output artifact: `/private/tmp/phase6a_vision_smoke_20260708/preview.mp4`.
- What this proves: the preview script can encode stored RGB frames through ffmpeg.
- What it does not prove: human visual acceptance; the preview should still be inspected.

## Decisions Made

- Decision: keep RGB frames in NPZ files and JSON metadata in `vision_manifest.json`.
  Reason: storing large image arrays directly in demo rows would make demos harder to diff,
  inspect, validate, and reuse with non-vision loaders.
- Decision: keep Phase 6a action-space-neutral.
  Reason: the dataset should preserve the same observations, trial starts, task summaries,
  and label sets across `joint_delta`, `ee_delta`, and `ee_tool_delta`; otherwise it would
  not support a fair later comparison.
- Decision: validate recorded SHA-256 values.
  Reason: evidence artifacts are only useful if the validator checks artifact identity, not
  just schema-compatible shape and dtype.
- Decision: document Phase 6a as plumbing only.
  Reason: passing renderer/dataset tests does not change the Phase 5 policy verdict or open
  Phase 6b policy/VLA work.

## Risks And Limitations

- Risk or limitation: MuJoCo RGB rendering requires a usable macOS graphics context.
  Why it matters: renderer tests can fail inside the sandbox with `invalid CoreGraphics
  connection` before exercising project code.
- Risk or limitation: preview rendering does not independently call the dataset validator.
  Why it matters: malformed or tampered datasets should be rejected by
  `validate_vision_dataset.py` before preview generation.
- Risk or limitation: this records scripted pickup RGB data only.
  Why it matters: it is not learned-policy evidence, not pick-place BC, and not VLA
  readiness.
- Risk or limitation: force/impulse thresholds remain MuJoCo sanity checks.
  Why it matters: hardware realism is still not assessed.

## Action Items

- [ ] User should review this report before any merge to `main`.
- [ ] Review a generated MP4 preview from a representative dataset before treating Phase 6a
  artifacts as visually acceptable.
- [ ] Keep Phase 6b blocked until the action-space comparison is viable or scope is
  explicitly changed.
- [ ] If vision policy work is later authorized, define the temporal/gripper contract before
  training rather than assuming RGB fixes current BC event-order failures.

## Files Changed

- `.gitignore` - ignores local Obsidian workspace metadata.
- `AGENTS.md` - records the large-change report requirement and Phase 6a status.
- `README.md` - documents the new dataset export, validation, and preview workflow.
- `researchnotes.md` - logs Phase 6a as infrastructure, not policy evidence.
- `src/svla/__init__.py` - exports fixed-camera configuration and renderer types.
- `src/svla/experiment_manifest.py` - includes Phase 6a files in tracked source hashes.
- `src/svla/pickup_task.py` - adds opt-in RGB observation rendering.
- `src/svla/vision_observations.py` - adds fixed-camera renderer and metadata.
- `src/svla/vision_dataset.py` - adds scripted RGB dataset export and validation.
- `scripts/record_pickup_vision_demos.py` - records scripted RGB datasets.
- `scripts/validate_vision_dataset.py` - validates dataset manifests and frame artifacts.
- `scripts/render_vision_dataset_preview.py` - renders MP4 previews from stored RGB frames.
- `tests/test_vision_dataset.py` - covers dataset export, validation, hashes, and loader
  compatibility.
- `tests/test_vision_observations.py` - covers camera metadata, opt-in rendering, and
  renderer output.

## Current Verdict

Needs review before merge. Phase 6a infrastructure is implemented as data/render/validation
plumbing, but it is not policy evidence and does not make Phase 6b vision-conditioned BC or
VLA work ready.
