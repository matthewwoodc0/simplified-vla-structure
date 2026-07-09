from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from svla.pickup_task import (
    OBJECT_START_Z,
    ApproachStrategy,
    GraspOrientation,
    ObjectStartPose,
    PickupTrialSpec,
)


PROTOCOL_FORMAT = "svla_phase5_evaluation_protocol_v2"
PROTOCOL_PATH = Path(__file__).resolve().parents[3] / "configs" / "phase5_evaluation_protocol_v2.json"
SPLIT_NAMES = ("train", "validation", "final")


@dataclass(frozen=True)
class EvaluationProtocol:
    path: Path
    data: dict
    sha256: str

    @property
    def version(self) -> int:
        return int(self.data["version"])

    @property
    def model_seeds(self) -> tuple[int, ...]:
        return tuple(int(seed) for seed in self.data["model_seeds"])

    def metadata(self, split: str) -> dict[str, str | int]:
        self._validate_split(split)
        return {
            "format": str(self.data["format"]),
            "version": self.version,
            "config_path": str(self.path),
            "config_sha256": self.sha256,
            "eval_split": split,
        }

    def specs(self, split: str, *, repeats: int = 1) -> list[PickupTrialSpec]:
        self._validate_split(split)
        if repeats < 1:
            raise ValueError("repeats must be at least one")
        split_config = self.data["splits"][split]
        orientations = [
            GraspOrientation(str(row["label"]), float(row["yaw_degrees"]))
            for row in self.data["orientations"]
        ]
        approaches = [
            ApproachStrategy(str(row["label"]), str(row["axis_mode"]))
            for row in self.data["approaches"]
        ]
        object_poses = [
            ObjectStartPose(str(row["label"]), np.asarray(row["xyz"], dtype=float))
            for row in split_config["object_poses"]
        ]
        specs: list[PickupTrialSpec] = []
        trial_id = int(split_config["start_trial_id"])
        for repeat in range(repeats):
            for orientation in orientations:
                for object_pose in object_poses:
                    for approach in approaches:
                        specs.append(
                            PickupTrialSpec(
                                trial_id=trial_id,
                                orientation=orientation,
                                object_pose=object_pose,
                                approach=approach,
                                repeat=repeat,
                            )
                        )
                        trial_id += 1
        return specs

    def _validate_split(self, split: str) -> None:
        if split not in SPLIT_NAMES:
            raise ValueError(f"unknown evaluation split: {split}")


def load_evaluation_protocol(path: Path = PROTOCOL_PATH) -> EvaluationProtocol:
    raw = path.read_bytes()
    data = json.loads(raw)
    _validate_protocol(data)
    return EvaluationProtocol(
        path=path.resolve(),
        data=data,
        sha256=hashlib.sha256(raw).hexdigest(),
    )


def _validate_protocol(data: dict) -> None:
    if data.get("format") != PROTOCOL_FORMAT or int(data.get("version", -1)) != 2:
        raise ValueError("evaluation protocol must use the v2 format")
    seeds = [int(seed) for seed in data.get("model_seeds", [])]
    if len(seeds) < 5 or len(seeds) != len(set(seeds)):
        raise ValueError("evaluation protocol requires at least five unique model seeds")
    splits = data.get("splits", {})
    if set(splits) != set(SPLIT_NAMES):
        raise ValueError(f"evaluation protocol splits must be {SPLIT_NAMES}")
    positions_by_split: dict[str, set[tuple[float, float, float]]] = {}
    labels: set[str] = set()
    trial_starts: set[int] = set()
    for split in SPLIT_NAMES:
        split_config = splits[split]
        start = int(split_config["start_trial_id"])
        if start in trial_starts:
            raise ValueError("split trial-id ranges must have distinct starts")
        trial_starts.add(start)
        positions: set[tuple[float, float, float]] = set()
        for row in split_config["object_poses"]:
            label = str(row["label"])
            if label in labels:
                raise ValueError(f"duplicate object-pose label: {label}")
            labels.add(label)
            xyz = tuple(float(value) for value in row["xyz"])
            if len(xyz) != 3 or not np.isfinite(xyz).all():
                raise ValueError(f"invalid object pose for {label}")
            if not np.isclose(xyz[2], OBJECT_START_Z, atol=1e-12):
                raise ValueError(
                    f"object pose {label} z={xyz[2]} does not match nominal "
                    f"OBJECT_START_Z={OBJECT_START_Z}"
                )
            if xyz in positions:
                raise ValueError(f"duplicate object position in split {split}: {xyz}")
            positions.add(xyz)
        positions_by_split[split] = positions
    for index, left in enumerate(SPLIT_NAMES):
        for right in SPLIT_NAMES[index + 1 :]:
            overlap = positions_by_split[left] & positions_by_split[right]
            if overlap:
                raise ValueError(f"object positions overlap between {left} and {right}: {overlap}")
    gates = data.get("proposed_release_gates", {})
    if gates.get("status") != "proposed_awaiting_approval":
        raise ValueError("v2 release gates must remain explicitly proposed")
