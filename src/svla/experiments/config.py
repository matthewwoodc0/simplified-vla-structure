from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any


EXPERIMENT_CONFIG_FORMAT = "svla_experiment_config_v1"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
ALLOWED_ENTRYPOINTS = {
    "scripts/train_state_bc.py",
    "scripts/run_loss_decomposition.py",
    "scripts/run_h_ee_014_diagnosis.py",
    "scripts/run_h_ee_022_match_reeval.py",
}


@dataclass(frozen=True)
class ExperimentConfig:
    path: Path
    sha256: str
    name: str
    hypothesis: str | None
    status: str
    entrypoint: str
    arguments: dict[str, Any]
    runnable: bool
    non_runnable_reason: str | None
    evidence: tuple[str, ...]


def load_experiment_config(path: Path) -> ExperimentConfig:
    path = Path(path)
    raw = path.read_bytes()
    payload = json.loads(raw)
    if payload.get("format") != EXPERIMENT_CONFIG_FORMAT:
        raise ValueError(f"experiment config must use {EXPERIMENT_CONFIG_FORMAT}")
    name = str(payload.get("name", ""))
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", name):
        raise ValueError(f"invalid experiment config name: {name!r}")
    entrypoint = str(payload.get("entrypoint", ""))
    if entrypoint not in ALLOWED_ENTRYPOINTS:
        raise ValueError(f"unsupported experiment entrypoint: {entrypoint!r}")
    arguments = payload.get("arguments", {})
    if not isinstance(arguments, dict):
        raise ValueError("experiment arguments must be a JSON object")
    runnable = bool(payload.get("runnable", True))
    reason = payload.get("non_runnable_reason")
    if not runnable and not reason:
        raise ValueError("non-runnable experiment configs require non_runnable_reason")
    if arguments.get("eval-split") == "final" and not payload.get("allow_final_access", False):
        raise ValueError("final split configs require explicit allow_final_access=true")
    return ExperimentConfig(
        path=path.resolve(),
        sha256=hashlib.sha256(raw).hexdigest(),
        name=name,
        hypothesis=payload.get("hypothesis"),
        status=str(payload.get("status", "unknown")),
        entrypoint=entrypoint,
        arguments=dict(arguments),
        runnable=runnable,
        non_runnable_reason=None if reason is None else str(reason),
        evidence=tuple(str(value) for value in payload.get("evidence", [])),
    )


def render_experiment_command(
    config: ExperimentConfig,
    *,
    python: str,
    output_dir: Path | None = None,
) -> list[str]:
    if not config.runnable:
        raise ValueError(
            f"experiment {config.name!r} is historical-only: "
            f"{config.non_runnable_reason}"
        )
    arguments = dict(config.arguments)
    if output_dir is not None:
        arguments["output-dir"] = str(Path(output_dir))
    command = [python, str(PROJECT_ROOT / config.entrypoint)]
    for name, value in arguments.items():
        flag = f"--{name}"
        if isinstance(value, bool):
            if value:
                command.append(flag)
            continue
        if value is None:
            continue
        command.append(flag)
        if isinstance(value, list):
            command.extend(str(item) for item in value)
        else:
            command.append(str(value))
    return command
