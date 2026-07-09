# Analysis

Offline analysis belongs here. These tools may read JSON/JSONL evidence and write derived
tables or reports; they must not execute a policy, change a task gate, or overwrite source
evidence. `policy_failures.py` contains the failure/gate-overlap analysis formerly embedded
under `scripts/`; the old path remains a small CLI compatibility wrapper.
