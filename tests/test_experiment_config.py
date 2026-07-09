from __future__ import annotations

import json
from pathlib import Path

import pytest

from svla.experiments.config import load_experiment_config, render_experiment_command


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "experiments" / "configs"


def test_all_versioned_experiment_configs_validate():
    configs = [load_experiment_config(path) for path in sorted(CONFIG_DIR.glob("*.json"))]

    assert len(configs) >= 8
    assert len({config.name for config in configs}) == len(configs)
    assert all(config.sha256 for config in configs)
    assert all(config.evidence for config in configs)


def test_h_ee_014_config_renders_exact_validation_contract(tmp_path):
    config = load_experiment_config(
        CONFIG_DIR / "h_ee_014_nn_gripper_global_validation.json"
    )
    command = render_experiment_command(
        config,
        python="python",
        output_dir=tmp_path / "result",
    )

    assert command[:2] == ["python", str(ROOT / "scripts" / "train_state_bc.py")]
    assert "--hybrid-nn-gripper" in command
    assert command[command.index("--eval-split") + 1] == "validation"
    assert command[command.index("--loss-profile") + 1] == "global_gripper"
    assert command[command.index("--output-dir") + 1] == str(tmp_path / "result")


def test_historical_removed_experiment_cannot_be_launched():
    config = load_experiment_config(
        CONFIG_DIR / "h_ee_003_separate_gripper_head_historical.json"
    )
    with pytest.raises(ValueError, match="historical-only"):
        render_experiment_command(config, python="python")


def test_final_split_requires_explicit_access_flag(tmp_path):
    path = tmp_path / "unsafe-final.json"
    path.write_text(
        json.dumps(
            {
                "format": "svla_experiment_config_v1",
                "name": "unsafe_final",
                "entrypoint": "scripts/train_state_bc.py",
                "arguments": {"eval-split": "final"},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="allow_final_access"):
        load_experiment_config(path)
