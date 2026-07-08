from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json

import numpy as np

from svla.pickup_task import (
    EARLY_CLOSE_DISTANCE,
    LIFT_CLEARANCE,
    PRECLOSE_OPEN_THRESHOLD,
    RETENTION_CLEARANCE,
    PickupTaskEvaluator,
    PickupTrialSpec,
    _orientation_error_rotvec,
)


ACTION_SPACE_SIZES = {"joint_delta": 6, "ee_delta": 7, "ee_tool_delta": 6}
ACTION_SPACES = {name: ACTION_SPACE_SIZES[name] for name in ("joint_delta", "ee_tool_delta")}
APPROACH_LABELS = ("vertical_pregrasp", "high_staged_vertical_pregrasp")
PHASE_LABELS = (
    "approach_0",
    "approach_1",
    "approach_2",
    "grasp_align",
    "close_gripper",
    "lift",
    "hold",
)
TEMPORAL_FEATURE_MODES = ("legacy_progress_phase", "none")
MATCH_FEATURE_INDICES = np.array([18, 19, 20, 28, 29, 30], dtype=int)


@dataclass(frozen=True)
class TaskContext:
    orientation_label: str
    yaw_degrees: float
    object_pose_label: str
    approach_label: str

    @classmethod
    def from_spec(cls, spec: PickupTrialSpec) -> "TaskContext":
        return cls(
            orientation_label=spec.orientation.label,
            yaw_degrees=spec.orientation.yaw_degrees,
            object_pose_label=spec.object_pose.label,
            approach_label=spec.approach.label,
        )

    @classmethod
    def from_demo(cls, demo: dict) -> "TaskContext":
        trial = demo["metadata"]["trial_spec"]
        return cls(
            orientation_label=trial["orientation"],
            yaw_degrees=float(trial["orientation"].replace("yaw_", "")),
            object_pose_label=trial["object_pose"],
            approach_label=trial["approach"],
        )

    @property
    def key(self) -> str:
        # Object pose stays in the observation; grouping by pose would prevent
        # held-out object-position evaluation from using neighboring demos.
        return "|".join((self.orientation_label, self.approach_label))

    def features(self) -> np.ndarray:
        yaw = np.deg2rad(self.yaw_degrees)
        approach = np.zeros(len(APPROACH_LABELS), dtype=float)
        if self.approach_label in APPROACH_LABELS:
            approach[APPROACH_LABELS.index(self.approach_label)] = 1.0
        else:
            raise ValueError(f"unknown approach label: {self.approach_label}")
        return np.concatenate(([np.sin(yaw), np.cos(yaw)], approach))


@dataclass(frozen=True)
class Dataset:
    action_space: str
    features: np.ndarray
    actions: np.ndarray
    group_keys: np.ndarray
    progress_indices: np.ndarray
    phase_indices: np.ndarray
    phase_progress: np.ndarray
    phase_lengths: np.ndarray
    feature_names: list[str]
    source_paths: list[str]
    demo_count: int
    skipped_demos: int
    label_source: str


@dataclass(frozen=True)
class PolicyTrialResult:
    trial_id: int
    action_space: str
    orientation: str
    object_pose: str
    approach: str
    repeat: int
    success: bool
    failure_category: str
    note: str
    steps: int
    contact_achieved: bool
    collision_free_approach: bool
    preclose_contact_steps: int
    preclose_max_object_displacement: float
    event_order_valid: bool
    early_close: bool
    reopen_events: int
    reopen_command_steps: int
    max_gripper_contact_force: float
    gripper_contact_impulse_before_lift: float
    max_object_xy_displacement_while_supported: float
    max_object_rotation_while_supported: float
    physical_sanity_pass: bool
    object_lifted: bool
    retained_during_hold: bool
    min_grasp_position_error: float
    min_grasp_rotation_error: float
    final_object_lift: float
    max_object_lift: float
    gripper_object_distance: float
    clipped_translation_steps: int
    clipped_rotation_steps: int
    clipped_joint_steps: int
    joint_limit_clipped_steps: int
    joint_step_clipped_steps: int
    joint_accel_clipped_steps: int
    infeasible_steps: int
    controller_failure_steps: int
    controller_failure_reason: str | None
    shielded_policy: bool
    suppressed_close_steps: int
    raw_action_l2_mean: float
    raw_action_l2_max: float
    raw_action_delta_l2_mean: float
    executed_action_l2_mean: float
    executed_action_l2_max: float
    executed_action_delta_l2_mean: float
    action_l2_mean: float
    action_l2_max: float
    action_delta_l2_mean: float
    nearest_distance_mean: float
    nearest_distance_max: float

    def to_dict(self) -> dict:
        return asdict(self)


class NearestNeighborBCPolicy:
    """State-conditioned nearest-neighbor behavioral cloning baseline."""

    def __init__(
        self,
        action_space: str,
        feature_mean: np.ndarray,
        feature_scale: np.ndarray,
        group_keys: list[str],
        group_starts: np.ndarray,
        group_ends: np.ndarray,
        progress_indices: np.ndarray,
        phase_indices: np.ndarray,
        phase_progress: np.ndarray,
        features: np.ndarray,
        actions: np.ndarray,
        k: int = 8,
        temperature: float = 0.75,
        evaluation_config_hash: str = "",
        evaluation_protocol_version: int = 0,
    ) -> None:
        if action_space not in ACTION_SPACE_SIZES:
            raise ValueError(f"unknown action space: {action_space}")
        self.action_space = action_space
        self.feature_mean = feature_mean
        self.feature_scale = np.where(feature_scale <= 1e-9, 1.0, feature_scale)
        self.group_keys = group_keys
        self.group_starts = group_starts
        self.group_ends = group_ends
        self.progress_indices = progress_indices
        self.phase_indices = phase_indices
        self.phase_progress = phase_progress
        self.features = features
        self.actions = actions
        self.k = int(k)
        self.temperature = float(temperature)
        self.evaluation_config_hash = str(evaluation_config_hash)
        self.evaluation_protocol_version = int(evaluation_protocol_version)
        self._group_index = {key: index for index, key in enumerate(group_keys)}

    def predict(self, observation_features: np.ndarray, group_key: str) -> tuple[np.ndarray, float]:
        action, distance, _ = self.predict_with_index(observation_features, group_key)
        return action, distance

    def predict_with_index(
        self,
        observation_features: np.ndarray,
        group_key: str,
        cursor: int | None = None,
        search_window: int | None = None,
    ) -> tuple[np.ndarray, float, int]:
        if group_key not in self._group_index:
            raise KeyError(f"policy has no demonstrations for task context {group_key!r}")
        index = self._group_index[group_key]
        start = int(self.group_starts[index])
        end = int(self.group_ends[index])
        if cursor is not None:
            cursor = max(0, int(cursor))
            group_progress = self.progress_indices[start:end]
            if search_window is not None:
                window_end = cursor + max(1, int(search_window))
                local_candidates = np.flatnonzero(
                    (group_progress >= cursor) & (group_progress < window_end)
                )
            else:
                local_candidates = np.flatnonzero(group_progress >= cursor)
            if len(local_candidates) == 0:
                local_candidates = np.array([int(np.argmax(group_progress))])
            candidate_indices = start + local_candidates
        else:
            candidate_indices = np.arange(start, end)
        query = self._standardize(observation_features)[MATCH_FEATURE_INDICES]
        group_features = self.features[candidate_indices][:, MATCH_FEATURE_INDICES]
        distances = np.linalg.norm(group_features - query, axis=1)
        k = min(self.k, len(distances))
        nearest = np.argpartition(distances, k - 1)[:k]
        nearest_distances = distances[nearest]
        scale = max(float(np.median(nearest_distances)), 1e-6) * self.temperature
        weights = np.exp(-(nearest_distances**2) / (2.0 * scale**2))
        if float(weights.sum()) <= 1e-12:
            weights = np.ones_like(weights)
        nearest_indices = candidate_indices[nearest]
        action = np.average(self.actions[nearest_indices], axis=0, weights=weights)
        best_index = int(nearest_indices[np.argmin(nearest_distances)])
        return action, float(nearest_distances.min()), int(self.progress_indices[best_index])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            action_space=np.array(self.action_space),
            feature_mean=self.feature_mean,
            feature_scale=self.feature_scale,
            group_keys=np.array(self.group_keys),
            group_starts=self.group_starts,
            group_ends=self.group_ends,
            progress_indices=self.progress_indices,
            phase_indices=self.phase_indices,
            phase_progress=self.phase_progress,
            features=self.features,
            actions=self.actions,
            k=np.array(self.k),
            temperature=np.array(self.temperature),
            evaluation_config_hash=np.array(self.evaluation_config_hash),
            evaluation_protocol_version=np.array(self.evaluation_protocol_version),
        )

    def _standardize(self, features: np.ndarray) -> np.ndarray:
        return (np.asarray(features, dtype=float) - self.feature_mean) / self.feature_scale


class MLPBCPolicy:
    """Small numpy MLP BC policy with an explicit temporal-feature contract."""

    def __init__(
        self,
        action_space: str,
        feature_mean: np.ndarray,
        feature_scale: np.ndarray,
        action_mean: np.ndarray,
        action_scale: np.ndarray,
        weights: list[np.ndarray],
        biases: list[np.ndarray],
        group_keys: list[str],
        group_max_progress: np.ndarray,
        group_phase_lengths: np.ndarray,
        temporal_feature_mode: str = "legacy_progress_phase",
        distance_mean: float = 0.0,
        distance_scale: float = 1.0,
        base_feature_names: list[str] | None = None,
        policy_feature_names: list[str] | None = None,
        evaluation_config_hash: str = "",
        evaluation_protocol_version: int = 0,
    ) -> None:
        if action_space not in ACTION_SPACE_SIZES:
            raise ValueError(f"unknown action space: {action_space}")
        self.action_space = action_space
        self.feature_mean = feature_mean
        self.feature_scale = np.where(feature_scale <= 1e-9, 1.0, feature_scale)
        self.action_mean = action_mean
        self.action_scale = np.where(action_scale <= 1e-9, 1.0, action_scale)
        self.weights = weights
        self.biases = biases
        self.group_keys = group_keys
        self.group_max_progress = group_max_progress
        self.group_phase_lengths = group_phase_lengths
        if temporal_feature_mode not in TEMPORAL_FEATURE_MODES:
            raise ValueError(f"unknown temporal feature mode: {temporal_feature_mode}")
        self.temporal_feature_mode = temporal_feature_mode
        self.distance_mean = float(distance_mean)
        self.distance_scale = max(float(distance_scale), 1e-9)
        self.base_feature_names = list(base_feature_names or observation_feature_names())
        self.policy_feature_names = list(
            policy_feature_names or mlp_policy_feature_names(temporal_feature_mode)
        )
        self.evaluation_config_hash = str(evaluation_config_hash)
        self.evaluation_protocol_version = int(evaluation_protocol_version)
        self._group_index = {key: index for index, key in enumerate(group_keys)}

    def predict_with_index(
        self,
        observation_features: np.ndarray,
        group_key: str,
        cursor: int | None = None,
        search_window: int | None = None,
    ) -> tuple[np.ndarray, float, int]:
        del search_window
        cursor = 0 if cursor is None else max(0, int(cursor))
        features = self._policy_features(observation_features, group_key, cursor)
        hidden = features
        for weights, bias in zip(self.weights[:-1], self.biases[:-1]):
            hidden = np.tanh(hidden @ weights + bias)
        normalized_action = hidden @ self.weights[-1] + self.biases[-1]
        action = normalized_action * self.action_scale + self.action_mean
        return action, 0.0, cursor

    def predict(self, observation_features: np.ndarray, group_key: str) -> tuple[np.ndarray, float]:
        action, distance, _ = self.predict_with_index(observation_features, group_key)
        return action, distance

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "policy_type": np.array("mlp_bc"),
            "action_space": np.array(self.action_space),
            "feature_mean": self.feature_mean,
            "feature_scale": self.feature_scale,
            "action_mean": self.action_mean,
            "action_scale": self.action_scale,
            "group_keys": np.array(self.group_keys),
            "group_max_progress": self.group_max_progress,
            "group_phase_lengths": self.group_phase_lengths,
            "temporal_feature_mode": np.array(self.temporal_feature_mode),
            "distance_mean": np.array(self.distance_mean),
            "distance_scale": np.array(self.distance_scale),
            "base_feature_names": np.asarray(self.base_feature_names),
            "policy_feature_names": np.asarray(self.policy_feature_names),
            "evaluation_config_hash": np.array(self.evaluation_config_hash),
            "evaluation_protocol_version": np.array(self.evaluation_protocol_version),
            "layer_count": np.array(len(self.weights)),
        }
        for index, (weights, bias) in enumerate(zip(self.weights, self.biases)):
            payload[f"weights_{index}"] = weights
            payload[f"bias_{index}"] = bias
        np.savez_compressed(path, **payload)

    def _policy_features(
        self,
        observation_features: np.ndarray,
        group_key: str,
        cursor: int,
    ) -> np.ndarray:
        if group_key not in self._group_index:
            raise KeyError(f"policy has no demonstrations for task context {group_key!r}")
        raw = np.asarray(observation_features, dtype=float)
        base = (raw - self.feature_mean) / self.feature_scale
        if self.temporal_feature_mode == "none":
            distance = _gripper_object_distance_from_features(raw)
            normalized_distance = (distance - self.distance_mean) / self.distance_scale
            return np.concatenate((base, np.array([normalized_distance], dtype=float)))
        max_progress = max(1.0, float(self.group_max_progress[self._group_index[group_key]]))
        phase_index, phase_progress = self._phase_at_cursor(group_key, cursor)
        phase_one_hot = np.zeros(len(PHASE_LABELS), dtype=float)
        phase_one_hot[phase_index] = 1.0
        progress = np.clip(float(cursor) / max_progress, 0.0, 1.5)
        return np.concatenate(
            (
                base,
                np.array(
                    [
                        progress,
                        progress * progress,
                        np.sin(np.pi * progress),
                        np.cos(np.pi * progress),
                    ],
                    dtype=float,
                ),
                phase_one_hot,
                np.array(
                    [
                        phase_progress,
                        phase_progress * phase_progress,
                        np.sin(np.pi * phase_progress),
                        np.cos(np.pi * phase_progress),
                    ],
                    dtype=float,
                ),
            )
        )

    def _phase_at_cursor(self, group_key: str, cursor: int) -> tuple[int, float]:
        lengths = self.group_phase_lengths[self._group_index[group_key]]
        cursor = max(0, int(cursor))
        elapsed = 0
        for phase_index, length in enumerate(lengths):
            length = max(1, int(round(float(length))))
            if cursor < elapsed + length or phase_index == len(lengths) - 1:
                phase_step = min(length - 1, max(0, cursor - elapsed))
                return phase_index, float(phase_step / max(1, length - 1))
            elapsed += length
        return len(PHASE_LABELS) - 1, 1.0


def load_policy(path: Path) -> NearestNeighborBCPolicy | MLPBCPolicy:
    data = np.load(path, allow_pickle=False)
    if "policy_type" in data.files and str(data["policy_type"]) == "mlp_bc":
        layer_count = int(data["layer_count"])
        return MLPBCPolicy(
            action_space=str(data["action_space"]),
            feature_mean=data["feature_mean"],
            feature_scale=data["feature_scale"],
            action_mean=data["action_mean"],
            action_scale=data["action_scale"],
            weights=[data[f"weights_{index}"] for index in range(layer_count)],
            biases=[data[f"bias_{index}"] for index in range(layer_count)],
            group_keys=[str(key) for key in data["group_keys"].tolist()],
            group_max_progress=data["group_max_progress"],
            group_phase_lengths=data["group_phase_lengths"],
            temporal_feature_mode=(
                str(data["temporal_feature_mode"])
                if "temporal_feature_mode" in data.files
                else "legacy_progress_phase"
            ),
            distance_mean=(
                float(data["distance_mean"]) if "distance_mean" in data.files else 0.0
            ),
            distance_scale=(
                float(data["distance_scale"]) if "distance_scale" in data.files else 1.0
            ),
            base_feature_names=(
                [str(name) for name in data["base_feature_names"].tolist()]
                if "base_feature_names" in data.files
                else observation_feature_names()
            ),
            policy_feature_names=(
                [str(name) for name in data["policy_feature_names"].tolist()]
                if "policy_feature_names" in data.files
                else mlp_policy_feature_names("legacy_progress_phase")
            ),
            evaluation_config_hash=(
                str(data["evaluation_config_hash"])
                if "evaluation_config_hash" in data.files
                else ""
            ),
            evaluation_protocol_version=(
                int(data["evaluation_protocol_version"])
                if "evaluation_protocol_version" in data.files
                else 0
            ),
        )
    return NearestNeighborBCPolicy(
        action_space=str(data["action_space"]),
        feature_mean=data["feature_mean"],
        feature_scale=data["feature_scale"],
        group_keys=[str(key) for key in data["group_keys"].tolist()],
        group_starts=data["group_starts"],
        group_ends=data["group_ends"],
        progress_indices=data["progress_indices"],
        phase_indices=data["phase_indices"],
        phase_progress=data["phase_progress"],
        features=data["features"],
        actions=data["actions"],
        k=int(data["k"]),
        temperature=float(data["temperature"]),
        evaluation_config_hash=(
            str(data["evaluation_config_hash"])
            if "evaluation_config_hash" in data.files
            else ""
        ),
        evaluation_protocol_version=(
            int(data["evaluation_protocol_version"])
            if "evaluation_protocol_version" in data.files
            else 0
        ),
    )


def observation_feature_names() -> list[str]:
    return [
        *(f"joint_position_{index}" for index in range(5)),
        *(f"joint_velocity_{index}" for index in range(5)),
        "ee_x",
        "ee_y",
        "ee_z",
        "ee_quat_w",
        "ee_quat_x",
        "ee_quat_y",
        "ee_quat_z",
        "gripper_open",
        "object_x",
        "object_y",
        "object_z",
        "object_quat_w",
        "object_quat_x",
        "object_quat_y",
        "object_quat_z",
        "object_minus_ee_x",
        "object_minus_ee_y",
        "object_minus_ee_z",
        "object_lift_from_start",
        "gripper_object_contact",
        "object_support_contact",
        "task_yaw_sin",
        "task_yaw_cos",
        *(f"approach_{label}" for label in APPROACH_LABELS),
    ]


def mlp_policy_feature_names(temporal_feature_mode: str) -> list[str]:
    if temporal_feature_mode not in TEMPORAL_FEATURE_MODES:
        raise ValueError(f"unknown temporal feature mode: {temporal_feature_mode}")
    base = observation_feature_names()
    if temporal_feature_mode == "none":
        return [*base, "gripper_object_distance"]
    return [
        *base,
        "progress",
        "progress_squared",
        "progress_sin_pi",
        "progress_cos_pi",
        *(f"phase_{phase}" for phase in PHASE_LABELS),
        "phase_progress",
        "phase_progress_squared",
        "phase_progress_sin_pi",
        "phase_progress_cos_pi",
    ]


def observation_to_features(
    observation: dict,
    context: TaskContext,
    object_start_pose: np.ndarray,
) -> np.ndarray:
    joints = np.asarray(observation["joint_positions"], dtype=float)
    velocities = np.asarray(observation["joint_velocities"], dtype=float)
    ee_position = np.asarray(observation["ee_position"], dtype=float)
    ee_quat = np.asarray(observation["ee_quat_wxyz"], dtype=float)
    object_position = np.asarray(observation["object_position"], dtype=float)
    object_quat = np.asarray(observation["object_quat_wxyz"], dtype=float)
    object_start_pose = np.asarray(object_start_pose, dtype=float)
    return np.concatenate(
        (
            joints,
            velocities,
            ee_position,
            ee_quat,
            [float(observation["gripper_open"])],
            object_position,
            object_quat,
            object_position - ee_position,
            [float(object_position[2] - object_start_pose[2])],
            [float(observation["gripper_object_contact"])],
            [float(observation["object_support_contact"])],
            context.features(),
        )
    )


def load_demo_dataset(
    demo_paths: list[Path],
    action_space: str,
    success_only: bool = True,
    stride: int = 1,
    label_source: str = "policy_labels",
) -> Dataset:
    if action_space not in ACTION_SPACE_SIZES:
        raise ValueError(f"unknown action space: {action_space}")
    feature_rows: list[np.ndarray] = []
    action_rows: list[np.ndarray] = []
    group_keys: list[str] = []
    progress_indices: list[int] = []
    phase_indices: list[int] = []
    phase_progress_rows: list[float] = []
    phase_length_rows: list[int] = []
    used_paths: list[str] = []
    skipped = 0
    for path in demo_paths:
        demo = json.loads(path.read_text(encoding="utf-8"))
        if success_only and not bool(demo["summary"]["success"]):
            skipped += 1
            continue
        context = TaskContext.from_demo(demo)
        object_start = np.asarray(demo["summary"]["object_start_pose"], dtype=float)
        phase_lengths = _phase_lengths(demo["samples"])
        for sample in demo["samples"][::stride]:
            labels = sample.get(label_source)
            if labels is None:
                labels = sample["labels"]
            phase = sample["phase"]
            phase_index = _phase_index(phase)
            phase_length = max(1, phase_lengths[phase])
            feature_rows.append(
                observation_to_features(sample["observation"], context, object_start)
            )
            action_rows.append(np.asarray(labels[action_space], dtype=float))
            group_keys.append(context.key)
            progress_indices.append(int(sample["step_index"]))
            phase_indices.append(phase_index)
            phase_progress_rows.append(float(sample["phase_step"] / max(1, phase_length - 1)))
            phase_length_rows.append(int(phase_length))
        used_paths.append(str(path))
    if not feature_rows:
        raise ValueError("no usable demo samples found")
    return Dataset(
        action_space=action_space,
        features=np.vstack(feature_rows),
        actions=np.vstack(action_rows),
        group_keys=np.asarray(group_keys),
        progress_indices=np.asarray(progress_indices, dtype=int),
        phase_indices=np.asarray(phase_indices, dtype=int),
        phase_progress=np.asarray(phase_progress_rows, dtype=float),
        phase_lengths=np.asarray(phase_length_rows, dtype=int),
        feature_names=observation_feature_names(),
        source_paths=used_paths,
        demo_count=len(used_paths),
        skipped_demos=skipped,
        label_source=label_source,
    )


def fit_nearest_neighbor_policy(
    dataset: Dataset,
    k: int = 8,
    temperature: float = 0.75,
) -> NearestNeighborBCPolicy:
    order = np.lexsort(
        (np.arange(len(dataset.group_keys)), dataset.progress_indices, dataset.group_keys)
    )
    raw_features = dataset.features[order]
    actions = dataset.actions[order]
    sorted_keys = dataset.group_keys[order]
    progress_indices = dataset.progress_indices[order]
    phase_indices = dataset.phase_indices[order]
    phase_progress = dataset.phase_progress[order]
    feature_mean = raw_features.mean(axis=0)
    feature_scale = raw_features.std(axis=0)
    standardized = (raw_features - feature_mean) / np.where(feature_scale <= 1e-9, 1.0, feature_scale)

    unique_keys: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    cursor = 0
    while cursor < len(sorted_keys):
        key = str(sorted_keys[cursor])
        end = cursor + 1
        while end < len(sorted_keys) and sorted_keys[end] == key:
            end += 1
        unique_keys.append(key)
        starts.append(cursor)
        ends.append(end)
        cursor = end

    return NearestNeighborBCPolicy(
        action_space=dataset.action_space,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        group_keys=unique_keys,
        group_starts=np.asarray(starts, dtype=int),
        group_ends=np.asarray(ends, dtype=int),
        progress_indices=progress_indices,
        phase_indices=phase_indices,
        phase_progress=phase_progress,
        features=standardized,
        actions=actions,
        k=k,
        temperature=temperature,
    )


def fit_mlp_policy(
    dataset: Dataset,
    hidden_sizes: tuple[int, ...] = (64, 64),
    epochs: int = 120,
    batch_size: int = 1024,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-5,
    seed: int = 0,
    temporal_feature_mode: str = "legacy_progress_phase",
) -> tuple[MLPBCPolicy, dict]:
    if temporal_feature_mode not in TEMPORAL_FEATURE_MODES:
        raise ValueError(f"unknown temporal feature mode: {temporal_feature_mode}")
    feature_mean = dataset.features.mean(axis=0)
    feature_scale = dataset.features.std(axis=0)
    feature_scale = np.where(feature_scale <= 1e-9, 1.0, feature_scale)
    group_keys = sorted({str(key) for key in dataset.group_keys})
    group_max_progress = np.array(
        [
            max(1, int(dataset.progress_indices[dataset.group_keys == key].max()))
            for key in group_keys
        ],
        dtype=float,
    )
    group_phase_lengths = _group_phase_lengths(dataset, group_keys)
    max_progress_by_key = {key: group_max_progress[index] for index, key in enumerate(group_keys)}
    base = (dataset.features - feature_mean) / feature_scale
    distance_mean = 0.0
    distance_scale = 1.0
    if temporal_feature_mode == "none":
        distances = np.linalg.norm(dataset.features[:, 25:28], axis=1)
        distance_mean = float(distances.mean())
        distance_scale = max(float(distances.std()), 1e-9)
        x = np.column_stack((base, (distances - distance_mean) / distance_scale))
    else:
        progress = np.array(
            [
                min(1.5, float(index) / max_progress_by_key[str(key)])
                for index, key in zip(dataset.progress_indices, dataset.group_keys)
            ],
            dtype=float,
        )
        phase_one_hot = np.zeros((len(dataset.phase_indices), len(PHASE_LABELS)), dtype=float)
        phase_one_hot[np.arange(len(dataset.phase_indices)), dataset.phase_indices] = 1.0
        phase_progress = dataset.phase_progress
        x = np.column_stack(
            (
                base,
                progress,
                progress * progress,
                np.sin(np.pi * progress),
                np.cos(np.pi * progress),
                phase_one_hot,
                phase_progress,
                phase_progress * phase_progress,
                np.sin(np.pi * phase_progress),
                np.cos(np.pi * phase_progress),
            )
        )
    action_mean = dataset.actions.mean(axis=0)
    action_scale = dataset.actions.std(axis=0)
    action_scale = np.where(action_scale <= 1e-9, 1.0, action_scale)
    y = (dataset.actions - action_mean) / action_scale

    rng = np.random.default_rng(seed)
    layer_sizes = (x.shape[1], *hidden_sizes, y.shape[1])
    weights = [
        rng.normal(0.0, np.sqrt(2.0 / (layer_sizes[index] + layer_sizes[index + 1])), size=(layer_sizes[index], layer_sizes[index + 1]))
        for index in range(len(layer_sizes) - 1)
    ]
    biases = [np.zeros(size, dtype=float) for size in layer_sizes[1:]]
    weight_m = [np.zeros_like(weight) for weight in weights]
    weight_v = [np.zeros_like(weight) for weight in weights]
    bias_m = [np.zeros_like(bias) for bias in biases]
    bias_v = [np.zeros_like(bias) for bias in biases]

    step = 0
    last_loss = float("inf")
    for _ in range(epochs):
        order = rng.permutation(len(x))
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            activations = [x[batch]]
            preactivations = []
            for weight, bias in zip(weights[:-1], biases[:-1]):
                z = activations[-1] @ weight + bias
                preactivations.append(z)
                activations.append(np.tanh(z))
            prediction = activations[-1] @ weights[-1] + biases[-1]
            activations.append(prediction)
            error = prediction - y[batch]
            grad = (2.0 / len(batch)) * error
            grad_weights: list[np.ndarray] = []
            grad_biases: list[np.ndarray] = []
            for layer_index in reversed(range(len(weights))):
                grad_weights.append(activations[layer_index].T @ grad + weight_decay * weights[layer_index])
                grad_biases.append(grad.sum(axis=0))
                if layer_index > 0:
                    grad = (grad @ weights[layer_index].T) * (1.0 - np.tanh(preactivations[layer_index - 1]) ** 2)
            grad_weights.reverse()
            grad_biases.reverse()
            step += 1
            _adam_update(weights, grad_weights, weight_m, weight_v, step, learning_rate)
            _adam_update(biases, grad_biases, bias_m, bias_v, step, learning_rate)
        last_loss = float(np.mean((forward_mlp(x, weights, biases) - y) ** 2))

    policy = MLPBCPolicy(
        action_space=dataset.action_space,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        action_mean=action_mean,
        action_scale=action_scale,
        weights=weights,
        biases=biases,
        group_keys=group_keys,
        group_max_progress=group_max_progress,
        group_phase_lengths=group_phase_lengths,
        temporal_feature_mode=temporal_feature_mode,
        distance_mean=distance_mean,
        distance_scale=distance_scale,
        base_feature_names=dataset.feature_names,
        policy_feature_names=mlp_policy_feature_names(temporal_feature_mode),
    )
    return policy, {
        "policy_type": "mlp_bc",
        "hidden_sizes": list(hidden_sizes),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "seed": int(seed),
        "temporal_feature_mode": temporal_feature_mode,
        "base_feature_names": list(dataset.feature_names),
        "policy_feature_names": mlp_policy_feature_names(temporal_feature_mode),
        "policy_input_dimension": int(x.shape[1]),
        "train_mse_normalized": last_loss,
    }


def forward_mlp(x: np.ndarray, weights: list[np.ndarray], biases: list[np.ndarray]) -> np.ndarray:
    hidden = x
    for weight, bias in zip(weights[:-1], biases[:-1]):
        hidden = np.tanh(hidden @ weight + bias)
    return hidden @ weights[-1] + biases[-1]


def _adam_update(
    values: list[np.ndarray],
    grads: list[np.ndarray],
    first_moments: list[np.ndarray],
    second_moments: list[np.ndarray],
    step: int,
    learning_rate: float,
) -> None:
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    for index, grad in enumerate(grads):
        first_moments[index] = beta1 * first_moments[index] + (1.0 - beta1) * grad
        second_moments[index] = beta2 * second_moments[index] + (1.0 - beta2) * (grad * grad)
        m_hat = first_moments[index] / (1.0 - beta1**step)
        v_hat = second_moments[index] / (1.0 - beta2**step)
        values[index] -= learning_rate * m_hat / (np.sqrt(v_hat) + eps)


def rollout_policy(
    env: PickupTaskEvaluator,
    policy: NearestNeighborBCPolicy | MLPBCPolicy,
    spec: PickupTrialSpec,
    max_steps: int = 3200,
    search_window: int = 120,
    action_gain: float = 1.0,
    gripper_close_guard: bool = False,
) -> PolicyTrialResult:
    env.reset(np.asarray(spec.object_pose.xyz, dtype=float))
    settled_start = env.object_position.copy()
    _, grasp_pos, grasp_quat = env.scripted_controller_commands(spec, settled_start)
    context = TaskContext.from_spec(spec)
    clipped_translation = 0
    clipped_rotation = 0
    clipped_joints = 0
    joint_limit_clipped = 0
    joint_step_clipped = 0
    joint_accel_clipped = 0
    infeasible = 0
    controller_failures = 0
    controller_failure_reason: str | None = None
    raw_action_norms: list[float] = []
    raw_action_delta_norms: list[float] = []
    executed_action_norms: list[float] = []
    executed_action_delta_norms: list[float] = []
    nearest_distances: list[float] = []
    previous_raw_action: np.ndarray | None = None
    previous_executed_action: np.ndarray | None = None
    min_grasp_pos_error = float("inf")
    min_grasp_rot_error = float("inf")
    cursor = 0
    suppressed_close_steps = 0
    guard_closure_started = False

    for _ in range(max_steps):
        observation = env.get_observation()
        features = observation_to_features(observation, context, settled_start)
        action, nearest_distance, nearest_index = policy.predict_with_index(
            features,
            context.key,
            cursor=cursor,
            search_window=search_window,
        )
        cursor = max(cursor + 1, nearest_index + 1)
        nearest_distances.append(nearest_distance)
        raw_action = action.copy()
        executed_action = raw_action.copy()
        if policy.action_space == "joint_delta":
            executed_action[:5] *= action_gain
        elif policy.action_space == "ee_tool_delta":
            executed_action[:5] *= action_gain
        raw_action_norms.append(float(np.linalg.norm(raw_action)))
        if previous_raw_action is not None:
            raw_action_delta_norms.append(float(np.linalg.norm(raw_action - previous_raw_action)))
        previous_raw_action = raw_action.copy()

        executed_action, suppressed, guard_closure_started = apply_gripper_close_guard(
            executed_action,
            enabled=gripper_close_guard,
            closure_started=guard_closure_started,
            gripper_object_distance=env.gripper_object_distance(),
        )
        suppressed_close_steps += int(suppressed)

        executed_action_norms.append(float(np.linalg.norm(executed_action)))
        if previous_executed_action is not None:
            executed_action_delta_norms.append(
                float(np.linalg.norm(executed_action - previous_executed_action))
            )
        previous_executed_action = executed_action.copy()

        if policy.action_space == "joint_delta":
            _, _, status = env.step_joint_delta_action(executed_action[:5], executed_action[5])
            clipped_joints += int(status["clipped_joints"])
            joint_limit_clipped += int(status["joint_limit_clipped"])
            joint_step_clipped += int(status["joint_step_clipped"])
            joint_accel_clipped += int(status["joint_accel_clipped"])
            infeasible += int(status["infeasible"])
            controller_failures += int(status["controller_failed"])
            controller_failure_reason = status["failure_reason"] or controller_failure_reason
        elif policy.action_space == "ee_tool_delta":
            _, _, status = env.step_ee_tool_delta_action(
                executed_action[:3],
                executed_action[3:5],
                executed_action[5],
            )
            clipped_translation += int(status.clipped_translation)
            clipped_rotation += int(status.clipped_rotation)
            clipped_joints += int(status.clipped_joints)
            joint_limit_clipped += int(status.joint_limit_clipped)
            joint_step_clipped += int(status.joint_step_clipped)
            joint_accel_clipped += int(status.joint_accel_clipped)
            infeasible += int(status.infeasible)
            controller_failures += int(status.controller_failed)
            controller_failure_reason = status.failure_reason or controller_failure_reason
        else:
            raise ValueError(f"unknown action space: {policy.action_space}")

        if controller_failures:
            break

        ee_pos, ee_quat = env.controller.ee_pose(env.data)
        min_grasp_pos_error = min(min_grasp_pos_error, float(np.linalg.norm(grasp_pos - ee_pos)))
        min_grasp_rot_error = min(
            min_grasp_rot_error,
            float(np.linalg.norm(_orientation_error_rotvec(ee_quat, grasp_quat))),
        )

        metrics = env.get_success_metrics()
        if (
            metrics["current_object_lift"] >= RETENTION_CLEARANCE
            and metrics["lifted_steps"] >= 180
            and metrics["close_contact_steps"] >= 60
            and env.gripper_object_distance() <= 0.045
        ):
            break

    metrics = env.get_success_metrics()
    reached_grasp = min_grasp_pos_error <= 0.012 and min_grasp_rot_error <= 0.22
    object_lifted = bool(metrics["max_object_lift"] >= LIFT_CLEARANCE)
    retained = (
        metrics["current_object_lift"] >= RETENTION_CLEARANCE
        and metrics["lifted_steps"] >= 180
        and metrics["close_contact_steps"] >= 60
        and metrics["gripper_object_distance"] <= 0.045
    )
    success = bool(
        metrics["collision_free_approach"]
        and metrics["event_order_valid"]
        and metrics["physical_sanity_pass"]
        and reached_grasp
        and metrics["contact_achieved"]
        and object_lifted
        and retained
    )
    failure_category, note = env._classify_failure(
        collision_free_approach=bool(metrics["collision_free_approach"]),
        event_order_valid=bool(metrics["event_order_valid"]),
        physical_sanity_pass=bool(metrics["physical_sanity_pass"]),
        reached_grasp=reached_grasp,
        contact=bool(metrics["contact_achieved"]),
        lifted=object_lifted,
        retained=bool(retained),
        final_position_error=min_grasp_pos_error,
        final_rotation_error=min_grasp_rot_error,
        clipped_joint_steps=clipped_joints,
    )
    if not success and failure_category == "none":
        failure_category = "policy_rollout_failure"
        note = "policy did not satisfy all rollout success criteria"
    if controller_failures:
        success = False
        failure_category = "controller_or_ik_failure"
        note = f"controller failure: {controller_failure_reason}"

    return PolicyTrialResult(
        trial_id=spec.trial_id,
        action_space=policy.action_space,
        orientation=spec.orientation.label,
        object_pose=spec.object_pose.label,
        approach=spec.approach.label,
        repeat=spec.repeat,
        success=success,
        failure_category=failure_category,
        note=note,
        steps=len(executed_action_norms),
        contact_achieved=bool(metrics["contact_achieved"]),
        collision_free_approach=bool(metrics["collision_free_approach"]),
        preclose_contact_steps=int(metrics["preclose_contact_steps"]),
        preclose_max_object_displacement=float(
            metrics["preclose_max_object_displacement"]
        ),
        event_order_valid=bool(metrics["event_order_valid"]),
        early_close=bool(metrics["early_close"]),
        reopen_events=int(metrics["reopen_events"]),
        reopen_command_steps=int(metrics["reopen_command_steps"]),
        max_gripper_contact_force=float(metrics["max_gripper_contact_force"]),
        gripper_contact_impulse_before_lift=float(
            metrics["gripper_contact_impulse_before_lift"]
        ),
        max_object_xy_displacement_while_supported=float(
            metrics["max_object_xy_displacement_while_supported"]
        ),
        max_object_rotation_while_supported=float(
            metrics["max_object_rotation_while_supported"]
        ),
        physical_sanity_pass=bool(metrics["physical_sanity_pass"]),
        object_lifted=object_lifted,
        retained_during_hold=bool(retained),
        min_grasp_position_error=float(min_grasp_pos_error),
        min_grasp_rotation_error=float(min_grasp_rot_error),
        final_object_lift=float(metrics["current_object_lift"]),
        max_object_lift=float(metrics["max_object_lift"]),
        gripper_object_distance=float(metrics["gripper_object_distance"]),
        clipped_translation_steps=clipped_translation,
        clipped_rotation_steps=clipped_rotation,
        clipped_joint_steps=clipped_joints,
        joint_limit_clipped_steps=joint_limit_clipped,
        joint_step_clipped_steps=joint_step_clipped,
        joint_accel_clipped_steps=joint_accel_clipped,
        infeasible_steps=infeasible,
        controller_failure_steps=controller_failures,
        controller_failure_reason=controller_failure_reason,
        shielded_policy=bool(gripper_close_guard),
        suppressed_close_steps=suppressed_close_steps,
        raw_action_l2_mean=_mean(raw_action_norms),
        raw_action_l2_max=max(raw_action_norms) if raw_action_norms else 0.0,
        raw_action_delta_l2_mean=_mean(raw_action_delta_norms),
        executed_action_l2_mean=_mean(executed_action_norms),
        executed_action_l2_max=max(executed_action_norms) if executed_action_norms else 0.0,
        executed_action_delta_l2_mean=_mean(executed_action_delta_norms),
        action_l2_mean=_mean(executed_action_norms),
        action_l2_max=max(executed_action_norms) if executed_action_norms else 0.0,
        action_delta_l2_mean=_mean(executed_action_delta_norms),
        nearest_distance_mean=_mean(nearest_distances),
        nearest_distance_max=max(nearest_distances) if nearest_distances else 0.0,
    )


def summarize_policy_results(results: list[PolicyTrialResult]) -> dict:
    return {
        "total": len(results),
        "successes": sum(result.success for result in results),
        "success_rate": _rate(result.success for result in results),
        "by_action_space": _bucket_rates(results, "action_space"),
        "by_approach": _bucket_rates(results, "approach"),
        "by_object_pose": _bucket_rates(results, "object_pose"),
        "by_orientation": _bucket_rates(results, "orientation"),
        "failure_categories": _failure_counts(results),
        "collision_free_approach_rate": _rate(
            result.collision_free_approach for result in results
        ),
        "event_order_valid_rate": _rate(result.event_order_valid for result in results),
        "physical_sanity_pass_rate": _rate(
            result.physical_sanity_pass for result in results
        ),
        "early_close_trials": sum(result.early_close for result in results),
        "reopen_events": sum(result.reopen_events for result in results),
        "reopen_command_steps": sum(result.reopen_command_steps for result in results),
        "shielded_policy": bool(results) and all(result.shielded_policy for result in results),
        "suppressed_close_steps": sum(result.suppressed_close_steps for result in results),
        "max_gripper_contact_force": max(
            (result.max_gripper_contact_force for result in results), default=0.0
        ),
        "max_gripper_contact_impulse_before_lift": max(
            (result.gripper_contact_impulse_before_lift for result in results),
            default=0.0,
        ),
        "max_object_xy_displacement_while_supported": max(
            (result.max_object_xy_displacement_while_supported for result in results),
            default=0.0,
        ),
        "preclose_contact_steps": sum(result.preclose_contact_steps for result in results),
        "max_preclose_object_displacement": max(
            (result.preclose_max_object_displacement for result in results),
            default=0.0,
        ),
        "mean_steps": _mean([result.steps for result in results]),
        "mean_action_l2": _mean([result.action_l2_mean for result in results]),
        "mean_action_delta_l2": _mean([result.action_delta_l2_mean for result in results]),
        "mean_raw_action_l2": _mean([result.raw_action_l2_mean for result in results]),
        "mean_raw_action_delta_l2": _mean(
            [result.raw_action_delta_l2_mean for result in results]
        ),
        "mean_executed_action_l2": _mean(
            [result.executed_action_l2_mean for result in results]
        ),
        "mean_executed_action_delta_l2": _mean(
            [result.executed_action_delta_l2_mean for result in results]
        ),
        "mean_nearest_distance": _mean([result.nearest_distance_mean for result in results]),
        "mean_clipped_joint_steps": _mean([result.clipped_joint_steps for result in results]),
        "mean_joint_limit_clipped_steps": _mean(
            [result.joint_limit_clipped_steps for result in results]
        ),
        "mean_joint_step_clipped_steps": _mean(
            [result.joint_step_clipped_steps for result in results]
        ),
        "mean_joint_accel_clipped_steps": _mean(
            [result.joint_accel_clipped_steps for result in results]
        ),
        "mean_infeasible_steps": _mean([result.infeasible_steps for result in results]),
        "controller_failure_steps": sum(result.controller_failure_steps for result in results),
    }


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _bucket_rates(results: list[PolicyTrialResult], attr: str) -> dict[str, dict[str, float | int]]:
    buckets = sorted({getattr(result, attr) for result in results})
    return {
        bucket: {
            "total": sum(getattr(result, attr) == bucket for result in results),
            "successes": sum(
                result.success for result in results if getattr(result, attr) == bucket
            ),
            "success_rate": _rate(
                result.success for result in results if getattr(result, attr) == bucket
            ),
        }
        for bucket in buckets
    }


def _failure_counts(results: list[PolicyTrialResult]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        counts[result.failure_category] = counts.get(result.failure_category, 0) + 1
    return counts


def _rate(values) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


def _mean(values) -> float:
    values = list(values)
    return float(np.mean(values)) if values else 0.0


def _gripper_object_distance_from_features(features: np.ndarray) -> float:
    values = np.asarray(features, dtype=float)
    indices = [observation_feature_names().index(f"object_minus_ee_{axis}") for axis in "xyz"]
    return float(np.linalg.norm(values[indices]))


def apply_gripper_close_guard(
    action: np.ndarray,
    *,
    enabled: bool,
    closure_started: bool,
    gripper_object_distance: float,
) -> tuple[np.ndarray, bool, bool]:
    """Optionally suppress premature close without altering later reopen commands."""

    executed = np.asarray(action, dtype=float).copy()
    if not enabled or closure_started:
        return executed, False, closure_started
    raw_close_command = bool(executed[-1] <= PRECLOSE_OPEN_THRESHOLD)
    if not raw_close_command:
        return executed, False, False
    if gripper_object_distance > EARLY_CLOSE_DISTANCE:
        executed[-1] = 1.0
        return executed, True, False
    return executed, False, True


def _phase_index(phase: str) -> int:
    if phase not in PHASE_LABELS:
        raise ValueError(f"unknown phase label: {phase}")
    return PHASE_LABELS.index(phase)


def _phase_lengths(samples: list[dict]) -> dict[str, int]:
    lengths = {phase: 0 for phase in PHASE_LABELS}
    for sample in samples:
        phase = sample["phase"]
        if phase not in lengths:
            raise ValueError(f"unknown phase label: {phase}")
        lengths[phase] += 1
    return lengths


def _group_phase_lengths(dataset: Dataset, group_keys: list[str]) -> np.ndarray:
    lengths = np.ones((len(group_keys), len(PHASE_LABELS)), dtype=float)
    for group_index, key in enumerate(group_keys):
        group_mask = dataset.group_keys == key
        for phase_index in range(len(PHASE_LABELS)):
            phase_mask = group_mask & (dataset.phase_indices == phase_index)
            if np.any(phase_mask):
                lengths[group_index, phase_index] = max(
                    1.0,
                    float(np.median(dataset.phase_lengths[phase_mask])),
                )
    return lengths
