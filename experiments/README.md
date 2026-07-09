# Experiments

Every maintained experiment launch is represented by a versioned JSON file in
`experiments/configs/`. The config records the hypothesis, exact CLI arguments, status,
and evidence paths. Run one with:

```bash
PYTHONPATH=src .venv/bin/python experiments/run.py \
  experiments/configs/h_ee_014_nn_gripper_global_validation.json --dry-run
```

Remove `--dry-run` only after checking the printed command. Configs never grant implicit
access to the final holdout. Historical experiments whose implementation was intentionally
removed remain documented as `runnable: false` rather than pretending the current code can
reproduce them byte-for-byte.
