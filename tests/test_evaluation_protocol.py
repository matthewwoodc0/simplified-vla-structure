import json
from pathlib import Path
import subprocess
import sys

from svla.evaluation_protocol import (
    PROTOCOL_PATH,
    SPLIT_NAMES,
    load_evaluation_protocol,
)
from svla.pickup_task import OBJECT_START_Z


def _positions(specs):
    return {tuple(float(value) for value in spec.object_pose.xyz) for spec in specs}


def test_v2_protocol_is_deterministic_disjoint_and_uses_five_seeds():
    first = load_evaluation_protocol()
    second = load_evaluation_protocol()

    assert first.sha256 == second.sha256
    assert len(first.model_seeds) >= 5
    split_positions = {split: _positions(first.specs(split)) for split in SPLIT_NAMES}
    assert split_positions["train"].isdisjoint(split_positions["validation"])
    assert split_positions["train"].isdisjoint(split_positions["final"])
    assert split_positions["validation"].isdisjoint(split_positions["final"])

    for split in SPLIT_NAMES:
        first_rows = [spec.object_pose.xyz.tolist() for spec in first.specs(split)]
        second_rows = [spec.object_pose.xyz.tolist() for spec in second.specs(split)]
        assert first_rows == second_rows
        assert {row[2] for row in first_rows} == {OBJECT_START_Z}


def test_v2_protocol_materializes_nominal_domain_and_proposed_gates():
    protocol = load_evaluation_protocol()
    raw = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))

    assert raw["domain"]["object_size_scale"] == [1.0, 1.0, 1.0]
    assert raw["domain"]["sliding_friction"] == 1.8
    assert raw["proposed_release_gates"]["status"] == "proposed_awaiting_approval"
    assert protocol.metadata("final")["config_sha256"] == protocol.sha256


def test_v2_protocol_has_complete_context_denominators():
    protocol = load_evaluation_protocol()

    assert len(protocol.specs("train")) == 30
    assert len(protocol.specs("validation")) == 24
    assert len(protocol.specs("final")) == 24


def test_v2_runner_rejects_implicit_split_access():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "train_state_bc.py"),
            "--evaluation-protocol",
            "v2",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "requires explicit --eval-split" in result.stderr
