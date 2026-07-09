# Evaluation

This package owns frozen split definitions and provenance manifests. Compatibility modules
remain at `svla.evaluation_protocol` and `svla.experiment_manifest`, but new code should
import from `svla.eval.protocol` and `svla.eval.manifest`.

Validation, final, shielded diagnostics, scripted replay, and visual review are distinct
evidence layers. Never promote a diagnostic result into the raw learned-policy ladder.
