#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

exec "$ROOT/.venv/bin/python" "$ROOT/.venv/bin/mjpython" "$ROOT/scripts/open_mujoco_gui.py"
