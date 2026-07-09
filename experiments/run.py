#!/usr/bin/env python3
"""Run a versioned SVLA experiment config or print the exact command."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from svla.experiments.config import load_experiment_config, render_experiment_command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--python",
        default=str(PROJECT_ROOT / ".venv" / "bin" / "python"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    command = render_experiment_command(
        config,
        python=args.python,
        output_dir=args.output_dir,
    )
    print(" ".join(command))
    print(f"config_sha256={config.sha256}")
    if args.dry_run:
        return
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    environment["SVLA_EXPERIMENT_CONFIG_PATH"] = str(config.path)
    environment["SVLA_EXPERIMENT_CONFIG_SHA256"] = config.sha256
    subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=True)


if __name__ == "__main__":
    main()
