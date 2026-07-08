from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from svla.experiment_manifest import ExperimentManifest
from svla.pickup_task import (
    PickupTaskEvaluator,
    default_pick_place_trial_specs,
    summarize_pick_place_results,
)


def run(output: Path, *, command: list[str] | None = None) -> dict:
    manifest = ExperimentManifest.start(repo_root=PROJECT_ROOT, argv=command)
    evaluator = PickupTaskEvaluator()
    specs = default_pick_place_trial_specs()
    output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    with output.open("w", encoding="utf-8") as handle:
        for spec in specs:
            result = evaluator.run_pick_place_trial(spec)
            results.append(result)
            handle.write(json.dumps(result.to_dict(), sort_keys=True) + "\n")
            print(
                f"trial={result.trial_id:02d} success={int(result.success)} "
                f"placement={result.placement_target} start={result.object_pose} "
                f"approach={result.approach} place_xy={result.placement_xy_error:.4f} "
                f"place_z={result.placement_z_error:.4f} released={int(result.gripper_released)} "
                f"order={int(result.event_order_valid)} phys={int(result.physical_sanity_pass)} "
                f"failure={result.failure_category}"
            )

    summary = summarize_pick_place_results(results)
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {output}")
    print(f"wrote {summary_path}")
    manifest.add_outputs([output, summary_path])
    manifest_path = manifest.write_sidecar(output)
    print(f"wrote {manifest_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "pick_place_trials.jsonl",
    )
    args = parser.parse_args()
    run(args.output, command=sys.argv)


if __name__ == "__main__":
    main()