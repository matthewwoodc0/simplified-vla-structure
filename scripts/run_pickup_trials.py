from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from svla.pickup_task import PickupTaskEvaluator, default_trial_specs, summarize_results


def run(output: Path, repeats: int, stop_after: int | None = None) -> dict:
    evaluator = PickupTaskEvaluator()
    specs = default_trial_specs(repeats=repeats)
    if stop_after is not None:
        specs = specs[:stop_after]

    output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    with output.open("w", encoding="utf-8") as handle:
        for spec in specs:
            result = evaluator.run_trial(spec)
            results.append(result)
            handle.write(json.dumps(result.to_dict(), sort_keys=True) + "\n")
            print(
                f"trial={result.trial_id:02d} success={int(result.success)} "
                f"orientation={result.orientation} pose={result.object_pose} "
                f"approach={result.approach} pos_err={result.final_ee_position_error:.4f} "
                f"rot_err={result.final_ee_rotation_error:.3f} contact={int(result.contact_achieved)} "
                f"lifted={int(result.object_lifted)} retained={int(result.retained_during_hold)} "
                f"failure={result.failure_category}"
            )

    summary = summarize_results(results)
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {output}")
    print(f"wrote {summary_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "pickup_trials.jsonl",
    )
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--stop-after", type=int, default=None)
    args = parser.parse_args()
    run(args.output, repeats=args.repeats, stop_after=args.stop_after)


if __name__ == "__main__":
    main()
