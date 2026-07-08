from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from svla.vision_dataset import validate_pickup_vision_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "phase6a_vision_sample",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON validation summary path.",
    )
    args = parser.parse_args()
    summary = validate_pickup_vision_dataset(args.dataset_dir)
    text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"wrote {args.output}")
    print(text, end="")
    if not summary["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
