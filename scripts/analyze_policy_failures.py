#!/usr/bin/env python3
"""Compatibility entry point for :mod:`analysis.policy_failures`."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.policy_failures import *  # noqa: F401,F403
from analysis.policy_failures import main


if __name__ == "__main__":
    main()
