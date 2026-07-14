from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import json

import numpy as np

from svla.core.action_space import (
    ACTION_REPRESENTATIONS,
    COMPARISON_ACTION_SPACES,
    get_action_representation,
)
from svla.loss_profiles import (
    CLOSE_TIMING_PHASES as PROFILE_CLOSE_TIMING_PHASES,
    resolve_loss_weights,
)
from svla.pickup_task import (
    EARLY_CLOSE_DISTANCE,
    LIFT_CLEARANCE,
    PRECLOSE_OPEN_THRESHOLD,
    RETENTION_CLEARANCE,
    PickupTaskEvaluator,
    PickupTrialSpec,
    _orientation_error_rotvec,
)


ACTION_SPACE_SIZES = {
    name: representation.size for name, representation in ACTION_REPRESENTATIONS.items()
}
ACTION_SPACES = {name: ACTION_SPACE_SIZES[name] for name in COMPARISON_ACTION_SPACES}
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
TEMPORAL_FEATURE_MODES = ("legacy_progress_phase", "none", "env_derived_phase")
# Object pose + contact/lift only (historical NN match contract). Not object_minus_ee_*.
MATCH_FEATURE_INDICES = np.array([18, 19, 20, 28, 29, 30], dtype=int)
MATCH_FEATURE_NAMES = (
    "object_x",
    "object_y",
    "object_z",
    "object_lift_from_start",
    "gripper_object_contact",
    "object_support_contact",
)
# Named secondary match contracts (H-EE-022). Default remains historical.
# Indices refer to observation_feature_names() base features.
MATCH_CONTRACT_HISTORICAL = "historical"
MATCH_CONTRACT_RELATIVE_EE = "match_relative_ee"
MATCH_CONTRACTS: dict[str, dict[str, list[str] | list[int]]] = {
    MATCH_CONTRACT_HISTORICAL: {
        "indices": [18, 19, 20, 28, 29, 30],
        "names": list(MATCH_FEATURE_NAMES),
    },
    # Relative EE–object + gripper_open + contact/lift (no absolute object xyz).
    MATCH_CONTRACT_RELATIVE_EE: {
        "indices": [17, 25, 26, 27, 28, 29, 30],
        "names": [
            "gripper_open",
            "object_minus_ee_x",
            "object_minus_ee_y",
            "object_minus_ee_z",
            "object_lift_from_start",
            "gripper_object_contact",
            "object_support_contact",
        ],
    },
}
MATCH_CONTRACT_NAMES = tuple(MATCH_CONTRACTS.keys())
DEFAULT_MATCH_CONTRACT = MATCH_CONTRACT_HISTORICAL
HYBRID_POLICY_TYPE = "hybrid_nn_gripper_mlp"
HYBRID_RECIPE_A1 = "A1_compositor"
HYBRID_RECIPE_A2 = "A2_arm_only_mlp"


def resolve_match_contract(
    match_contract: str = DEFAULT_MATCH_CONTRACT,
) -> tuple[np.ndarray, list[str]]:
    """Resolve a named NN match contract to indices + names.

    Historical remains the default for comparability. Secondary contracts (e.g.
    match_relative_ee for H-EE-022) must be selected explicitly by name.
    """

    if match_contract not in MATCH_CONTRACTS:
        raise ValueError(
            f"unknown match contract {match_contract!r}; "
            f"known: {list(MATCH_CONTRACTS)}"
        )
    contract = MATCH_CONTRACTS[match_contract]
    indices = np.asarray(contract["indices"], dtype=int)
    names = [str(name) for name in contract["names"]]
    if len(indices) != len(names):
        raise ValueError(
            f"match contract {match_contract!r} has len(indices)={len(indices)} "
            f"!= len(names)={len(names)}"
        )
    return indices, names

# Env-derived phase thresholds (meters / gripper-open fraction). Tuned against
# scripted pickup demos so distance/contact/lift bins recover a monotonic
# phase sequence without using demo step counters.
ENV_PHASE_HOLD_LIFT = 0.028
ENV_PHASE_LIFT_START = 0.002
ENV_PHASE_CLOSE_DISTANCE = 0.025
ENV_PHASE_GRASP_DISTANCE = 0.022
ENV_PHASE_APPROACH2_DISTANCE = 0.040
ENV_PHASE_APPROACH1_DISTANCE = 0.058
ENV_PHASE_CLOSE_OPEN_THRESHOLD = 0.97


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
    # Compact rollout diagnosis (event timing; does not change control).
    close_start_distance: float | None
    first_close_time: float | None
    first_contact_time: float | None
    first_unsupported_time: float | None
    first_lift_time: float | None
    gripper_command_flips: int

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
        match_feature_indices: np.ndarray | None = None,
        match_feature_names: list[str] | tuple[str, ...] | None = None,
        match_contract: str = DEFAULT_MATCH_CONTRACT,
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
        self.match_contract = str(match_contract)
        if match_feature_indices is None:
            default_indices, default_names = resolve_match_contract(DEFAULT_MATCH_CONTRACT)
            self.match_feature_indices = default_indices.copy()
            self.match_feature_names = list(default_names)
        else:
            self.match_feature_indices = np.asarray(match_feature_indices, dtype=int).copy()
            if match_feature_names is None:
                # Best-effort names from observation features when possible.
                base_names = observation_feature_names()
                self.match_feature_names = [
                    base_names[i] if 0 <= int(i) < len(base_names) else f"feature_{int(i)}"
                    for i in self.match_feature_indices
                ]
            else:
                self.match_feature_names = [str(name) for name in match_feature_names]
        self._group_index = {key: index for index, key in enumerate(group_keys)}

    def set_match_contract(
        self,
        match_contract: str,
        *,
        match_feature_indices: np.ndarray | None = None,
        match_feature_names: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        """Switch NN retrieval features without re-fitting the bank."""

        if match_feature_indices is None:
            indices, names = resolve_match_contract(match_contract)
        else:
            indices = np.asarray(match_feature_indices, dtype=int)
            if match_feature_names is None:
                base_names = observation_feature_names()
                names = [
                    base_names[i] if 0 <= int(i) < len(base_names) else f"feature_{int(i)}"
                    for i in indices
                ]
            else:
                names = [str(name) for name in match_feature_names]
        self.match_contract = str(match_contract)
        self.match_feature_indices = np.asarray(indices, dtype=int).copy()
        self.match_feature_names = list(names)

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
        match_idx = self.match_feature_indices
        query = self._standardize(observation_features)[match_idx]
        group_features = self.features[candidate_indices][:, match_idx]
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
            match_feature_indices=np.asarray(self.match_feature_indices, dtype=int),
            match_feature_names=np.asarray(self.match_feature_names),
            match_contract=np.array(self.match_contract),
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
        group_index = self._group_index[group_key]
        max_progress = max(1.0, float(self.group_max_progress[group_index]))
        if self.temporal_feature_mode == "env_derived_phase":
            phase_index, phase_progress = estimate_env_phase(raw)
            progress = env_phase_global_progress(
                phase_index,
                phase_progress,
                self.group_phase_lengths[group_index],
                max_progress,
            )
        else:
            phase_index, phase_progress = self._phase_at_cursor(group_key, cursor)
            progress = np.clip(float(cursor) / max_progress, 0.0, 1.5)
        phase_one_hot = np.zeros(len(PHASE_LABELS), dtype=float)
        phase_one_hot[phase_index] = 1.0
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


class HybridNNGripperMLPPolicy:
    """H-EE-014 compositor: MLP arm deltas + nearest-neighbor gripper command.

    Training is unchanged (MLP fit + NN fit on the same demos) unless A2
    arm-only MLP loss is selected (H-EE-023). At prediction time only the
    gripper dimension is taken from the NN policy so the arm retains MLP
    generalization while gripper timing uses state-local demo labels. Cursor
    advancement follows the MLP open-loop index so the hybrid does not
    silently change the temporal feature contract.
    """

    def __init__(
        self,
        mlp: MLPBCPolicy,
        nn: NearestNeighborBCPolicy,
        *,
        gripper_dim: int = -1,
        match_feature_indices: np.ndarray | None = None,
        match_feature_names: list[str] | tuple[str, ...] | None = None,
        match_contract: str = DEFAULT_MATCH_CONTRACT,
        recipe: str = HYBRID_RECIPE_A1,
        arm_only_mlp: bool = False,
    ) -> None:
        if mlp.action_space != nn.action_space:
            raise ValueError(
                "hybrid policy requires matching action spaces; "
                f"got mlp={mlp.action_space!r} nn={nn.action_space!r}"
            )
        self.mlp = mlp
        self.nn = nn
        self.action_space = mlp.action_space
        self.gripper_dim = int(gripper_dim)
        self.match_contract = str(match_contract)
        self.recipe = str(recipe)
        self.arm_only_mlp = bool(arm_only_mlp)
        if match_feature_indices is None:
            indices, names = resolve_match_contract(self.match_contract)
            self.match_feature_indices = indices.copy()
            self.match_feature_names = list(names)
        else:
            self.match_feature_indices = np.asarray(match_feature_indices, dtype=int).copy()
            if match_feature_names is None:
                base_names = observation_feature_names()
                self.match_feature_names = [
                    base_names[i] if 0 <= int(i) < len(base_names) else f"feature_{int(i)}"
                    for i in self.match_feature_indices
                ]
            else:
                self.match_feature_names = [str(name) for name in match_feature_names]
        # Keep NN retrieval aligned with the hybrid match contract at predict time.
        self.nn.set_match_contract(
            self.match_contract,
            match_feature_indices=self.match_feature_indices,
            match_feature_names=self.match_feature_names,
        )
        # Availability for eval uses the MLP group keys (same demos as NN fit).
        self.group_keys = list(mlp.group_keys)

    def set_match_contract(self, match_contract: str) -> None:
        """Apply a named secondary match contract without retraining."""

        indices, names = resolve_match_contract(match_contract)
        self.match_contract = str(match_contract)
        self.match_feature_indices = indices.copy()
        self.match_feature_names = list(names)
        self.nn.set_match_contract(
            self.match_contract,
            match_feature_indices=self.match_feature_indices,
            match_feature_names=self.match_feature_names,
        )

    @property
    def evaluation_config_hash(self) -> str:
        return str(self.mlp.evaluation_config_hash)

    @evaluation_config_hash.setter
    def evaluation_config_hash(self, value: str) -> None:
        self.mlp.evaluation_config_hash = str(value)
        self.nn.evaluation_config_hash = str(value)

    @property
    def evaluation_protocol_version(self) -> int:
        return int(self.mlp.evaluation_protocol_version)

    @evaluation_protocol_version.setter
    def evaluation_protocol_version(self, value: int) -> None:
        self.mlp.evaluation_protocol_version = int(value)
        self.nn.evaluation_protocol_version = int(value)

    def predict(
        self, observation_features: np.ndarray, group_key: str
    ) -> tuple[np.ndarray, float]:
        action, distance, _ = self.predict_with_index(observation_features, group_key)
        return action, distance

    def predict_with_index(
        self,
        observation_features: np.ndarray,
        group_key: str,
        cursor: int | None = None,
        search_window: int | None = None,
    ) -> tuple[np.ndarray, float, int]:
        a_mlp, _, mlp_index = self.mlp.predict_with_index(
            observation_features,
            group_key,
            cursor=cursor,
            search_window=search_window,
        )
        a_nn, nn_distance, _nn_index = self.nn.predict_with_index(
            observation_features,
            group_key,
            cursor=cursor,
            search_window=search_window,
        )
        action = np.asarray(a_mlp, dtype=float).copy()
        nn_action = np.asarray(a_nn, dtype=float)
        if action.ndim != 1 or nn_action.ndim != 1:
            raise ValueError("hybrid actions must be 1-D vectors")
        if action.shape != nn_action.shape:
            raise ValueError(
                f"hybrid action shape mismatch: mlp={action.shape} nn={nn_action.shape}"
            )
        action[self.gripper_dim] = float(nn_action[self.gripper_dim])
        # Report NN distance for diagnosis; keep MLP cursor for open-loop advance.
        return action, float(nn_distance), int(mlp_index)

    def save(self, path: Path) -> Path:
        """Save MLP npz, NN npz, and a JSON hybrid manifest. Returns manifest path."""

        path = Path(path)
        if path.suffix == ".npz":
            path = path.with_suffix(".json")
        elif path.suffix != ".json":
            path = path.with_name(path.name + ".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        mlp_path = path.parent / f"{path.stem}_mlp_component.npz"
        nn_path = path.parent / f"{path.stem}_nn_component.npz"
        self.mlp.save(mlp_path)
        self.nn.save(nn_path)
        manifest = {
            "format": "svla_hybrid_nn_gripper_mlp_manifest_v1",
            "policy_type": HYBRID_POLICY_TYPE,
            "action_space": self.action_space,
            "gripper_dim": self.gripper_dim,
            "mlp_path": mlp_path.name,
            "nn_path": nn_path.name,
            "nn_k": int(self.nn.k),
            "nn_temperature": float(self.nn.temperature),
            "match_contract": self.match_contract,
            "match_feature_indices": [int(i) for i in self.match_feature_indices.tolist()],
            "match_feature_names": list(self.match_feature_names),
            "mlp_temporal_feature_mode": self.mlp.temporal_feature_mode,
            "evaluation_config_hash": self.evaluation_config_hash,
            "evaluation_protocol_version": self.evaluation_protocol_version,
            "recipe": self.recipe,
            "arm_only_mlp": bool(self.arm_only_mlp),
            "note": (
                "MLP arm + NN gripper at rollout only. "
                f"Match contract={self.match_contract}; recipe={self.recipe}."
            ),
        }
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "HybridNNGripperMLPPolicy":
        path = Path(path)
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("policy_type") != HYBRID_POLICY_TYPE:
            raise ValueError(
                f"not a hybrid manifest at {path}: policy_type={manifest.get('policy_type')!r}"
            )
        mlp_path = path.parent / str(manifest["mlp_path"])
        nn_path = path.parent / str(manifest["nn_path"])
        mlp = load_policy(mlp_path)
        nn = load_policy(nn_path)
        if not isinstance(mlp, MLPBCPolicy):
            raise TypeError(f"hybrid mlp_path did not load an MLPBCPolicy: {mlp_path}")
        if not isinstance(nn, NearestNeighborBCPolicy):
            raise TypeError(
                f"hybrid nn_path did not load a NearestNeighborBCPolicy: {nn_path}"
            )
        match_contract = str(manifest.get("match_contract", DEFAULT_MATCH_CONTRACT))
        return cls(
            mlp,
            nn,
            gripper_dim=int(manifest.get("gripper_dim", -1)),
            match_feature_indices=np.asarray(
                manifest.get("match_feature_indices", MATCH_FEATURE_INDICES),
                dtype=int,
            ),
            match_feature_names=manifest.get("match_feature_names"),
            match_contract=match_contract,
            recipe=str(manifest.get("recipe", HYBRID_RECIPE_A1)),
            arm_only_mlp=bool(manifest.get("arm_only_mlp", False)),
        )


def load_policy(
    path: Path,
) -> NearestNeighborBCPolicy | MLPBCPolicy | HybridNNGripperMLPPolicy:
    path = Path(path)
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("policy_type") == HYBRID_POLICY_TYPE:
            return HybridNNGripperMLPPolicy.load(path)
        raise ValueError(f"unsupported policy JSON at {path}: {payload.get('policy_type')!r}")
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
    match_indices = (
        data["match_feature_indices"]
        if "match_feature_indices" in data.files
        else MATCH_FEATURE_INDICES
    )
    match_names = (
        [str(name) for name in data["match_feature_names"].tolist()]
        if "match_feature_names" in data.files
        else None
    )
    match_contract = (
        str(data["match_contract"])
        if "match_contract" in data.files
        else DEFAULT_MATCH_CONTRACT
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
        match_feature_indices=np.asarray(match_indices, dtype=int),
        match_feature_names=match_names,
        match_contract=match_contract,
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
    # legacy_progress_phase and env_derived_phase share the same input layout;
    # only the source of progress/phase values differs (cursor vs env state).
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


def estimate_env_phase(observation_features: np.ndarray) -> tuple[int, float]:
    """Map live observation features to (phase_index, phase_progress).

    Uses contact, lift, gripper opening, and gripper–object distance bins only.
    Does not use demo cursor, step index, or demo phase lengths. Same function
    is applied at train time and rollout so the temporal contract is matched.
    """

    names = observation_feature_names()
    values = np.asarray(observation_features, dtype=float)
    if values.shape[-1] < len(names):
        raise ValueError(
            f"observation features have length {values.shape[-1]}, expected >= {len(names)}"
        )
    grip = float(values[names.index("gripper_open")])
    dist = float(
        np.linalg.norm(
            [
                values[names.index("object_minus_ee_x")],
                values[names.index("object_minus_ee_y")],
                values[names.index("object_minus_ee_z")],
            ]
        )
    )
    lift = float(values[names.index("object_lift_from_start")])
    contact = float(values[names.index("gripper_object_contact")]) > 0.5

    if contact and lift >= ENV_PHASE_HOLD_LIFT:
        phase = "hold"
        progress = float(np.clip((lift - ENV_PHASE_HOLD_LIFT) / 0.01, 0.0, 1.0))
    elif contact and grip <= PRECLOSE_OPEN_THRESHOLD and lift >= ENV_PHASE_LIFT_START:
        phase = "lift"
        progress = float(np.clip(lift / max(ENV_PHASE_HOLD_LIFT, 1e-6), 0.0, 1.0))
    elif contact and grip <= PRECLOSE_OPEN_THRESHOLD and lift < ENV_PHASE_LIFT_START:
        phase = "close_gripper"
        progress = 1.0
    elif dist <= ENV_PHASE_CLOSE_DISTANCE and grip < ENV_PHASE_CLOSE_OPEN_THRESHOLD:
        # Near the object and jaw not fully open → close sequence.
        # Gating on distance avoids mislabeling far approach steps where the
        # jaw is still opening from the home pose.
        phase = "close_gripper"
        progress = float(
            np.clip(0.6 * (1.0 - grip) + 0.4 * float(contact), 0.0, 1.0)
        )
    elif dist <= ENV_PHASE_GRASP_DISTANCE:
        phase = "grasp_align"
        progress = float(np.clip(1.0 - dist / ENV_PHASE_GRASP_DISTANCE, 0.0, 1.0))
    elif dist <= ENV_PHASE_APPROACH2_DISTANCE:
        phase = "approach_2"
        progress = float(
            np.clip(
                1.0
                - (dist - ENV_PHASE_GRASP_DISTANCE)
                / max(1e-6, ENV_PHASE_APPROACH2_DISTANCE - ENV_PHASE_GRASP_DISTANCE),
                0.0,
                1.0,
            )
        )
    elif dist <= ENV_PHASE_APPROACH1_DISTANCE:
        phase = "approach_1"
        progress = float(
            np.clip(
                1.0
                - (dist - ENV_PHASE_APPROACH2_DISTANCE)
                / max(1e-6, ENV_PHASE_APPROACH1_DISTANCE - ENV_PHASE_APPROACH2_DISTANCE),
                0.0,
                1.0,
            )
        )
    else:
        phase = "approach_0"
        progress = float(
            np.clip(
                1.0 - (min(dist, 0.12) - ENV_PHASE_APPROACH1_DISTANCE) / 0.05,
                0.0,
                1.0,
            )
        )
    return _phase_index(phase), progress


def env_phase_global_progress(
    phase_index: int,
    phase_progress: float,
    phase_lengths: np.ndarray,
    max_progress: float,
) -> float:
    """Map env phase to a global progress scalar on the demo length scale."""

    lengths = np.asarray(phase_lengths, dtype=float)
    if lengths.shape != (len(PHASE_LABELS),):
        raise ValueError(
            f"phase_lengths must have shape ({len(PHASE_LABELS)},), got {lengths.shape}"
        )
    phase_index = int(np.clip(phase_index, 0, len(PHASE_LABELS) - 1))
    phase_progress = float(np.clip(phase_progress, 0.0, 1.0))
    elapsed = float(np.sum(lengths[:phase_index])) + phase_progress * max(
        1.0, float(lengths[phase_index])
    )
    return float(np.clip(elapsed / max(1.0, float(max_progress)), 0.0, 1.5))


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
    match_contract: str = DEFAULT_MATCH_CONTRACT,
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

    match_indices, match_names = resolve_match_contract(match_contract)
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
        match_feature_indices=match_indices,
        match_feature_names=match_names,
        match_contract=match_contract,
    )


# Demo phases where gripper timing is critical for event-order gates.
# Kept as a local alias so existing imports/tests remain stable.
CLOSE_TIMING_PHASES = PROFILE_CLOSE_TIMING_PHASES


def action_loss_weights(
    phase_indices: np.ndarray,
    action_size: int,
    *,
    gripper_loss_weight: float = 1.0,
    close_phase_gripper_weight: float | None = None,
    loss_profile: str | None = None,
    arm_only_mlp: bool = False,
) -> np.ndarray:
    """Per-sample, per-dim MSE weights. Arm dims stay 1.0; gripper dim is reweighted.

    When arm_only_mlp=True (H-EE-023 A2), gripper residual weight is 0 for all
    samples so MLP capacity is not spent on a dim the hybrid NN overwrites.
    """

    if action_size < 1:
        raise ValueError("action_size must be at least 1")
    n = len(phase_indices)
    weights = np.ones((n, action_size), dtype=float)
    gripper_dim = action_size - 1
    if arm_only_mlp:
        weights[:, gripper_dim] = 0.0
        return weights

    gripper_loss_weight, close_phase_gripper_weight, _ = resolve_loss_weights(
        loss_profile=loss_profile,
        gripper_loss_weight=gripper_loss_weight,
        close_phase_gripper_weight=close_phase_gripper_weight,
    )
    if gripper_loss_weight <= 0.0:
        raise ValueError("gripper_loss_weight must be positive")
    if close_phase_gripper_weight is not None and close_phase_gripper_weight <= 0.0:
        raise ValueError("close_phase_gripper_weight must be positive when set")

    base_gripper = float(gripper_loss_weight)
    weights[:, gripper_dim] = base_gripper
    if close_phase_gripper_weight is not None:
        close_indices = {_phase_index(phase) for phase in CLOSE_TIMING_PHASES}
        close_mask = np.isin(np.asarray(phase_indices, dtype=int), list(close_indices))
        weights[close_mask, gripper_dim] = float(close_phase_gripper_weight)
    return weights


def phase_sample_counts(phase_indices: np.ndarray) -> dict[str, int]:
    """Count training samples in each demo phase."""

    indices = np.asarray(phase_indices, dtype=int)
    counts = {phase: 0 for phase in PHASE_LABELS}
    for phase in PHASE_LABELS:
        counts[phase] = int(np.sum(indices == _phase_index(phase)))
    counts["close_timing_total"] = int(
        sum(counts[phase] for phase in CLOSE_TIMING_PHASES)
    )
    counts["total"] = int(len(indices))
    return counts


def phase_action_loss_report(
    residual: np.ndarray,
    phase_indices: np.ndarray,
    loss_weights: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    """Unweighted arm/gripper MSE by demo phase, plus mean applied gripper weight."""

    residual = np.asarray(residual, dtype=float)
    phase_indices = np.asarray(phase_indices, dtype=int)
    loss_weights = np.asarray(loss_weights, dtype=float)
    if residual.ndim != 2:
        raise ValueError("residual must be (N, action_dim)")
    action_size = residual.shape[1]
    gripper_dim = action_size - 1
    report: dict[str, dict[str, float | int]] = {}
    for phase in PHASE_LABELS:
        mask = phase_indices == _phase_index(phase)
        count = int(mask.sum())
        if count == 0:
            report[phase] = {
                "sample_count": 0,
                "arm_mse": 0.0,
                "gripper_mse": 0.0,
                "mean_gripper_weight": 0.0,
            }
            continue
        arm_residual = residual[mask, :gripper_dim]
        grip_residual = residual[mask, gripper_dim]
        report[phase] = {
            "sample_count": count,
            "arm_mse": float(np.mean(arm_residual**2)),
            "gripper_mse": float(np.mean(grip_residual**2)),
            "mean_gripper_weight": float(np.mean(loss_weights[mask, gripper_dim])),
        }
    return report


def fit_mlp_policy(
    dataset: Dataset,
    hidden_sizes: tuple[int, ...] = (64, 64),
    epochs: int = 120,
    batch_size: int = 1024,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-5,
    seed: int = 0,
    temporal_feature_mode: str = "legacy_progress_phase",
    gripper_loss_weight: float = 1.0,
    close_phase_gripper_weight: float | None = None,
    loss_profile: str | None = None,
    arm_only_mlp: bool = False,
) -> tuple[MLPBCPolicy, dict]:
    if temporal_feature_mode not in TEMPORAL_FEATURE_MODES:
        raise ValueError(f"unknown temporal feature mode: {temporal_feature_mode}")
    if arm_only_mlp:
        # A2: skip profile resolution for gripper weights; arm dims stay 1.0.
        resolved_profile = None
        if loss_profile is not None:
            # Still record the requested profile name for manifests, but arm-only
            # zeros gripper regardless of named profile weights.
            _, _, resolved_profile = resolve_loss_weights(
                loss_profile=loss_profile,
                gripper_loss_weight=gripper_loss_weight,
                close_phase_gripper_weight=close_phase_gripper_weight,
            )
    else:
        gripper_loss_weight, close_phase_gripper_weight, resolved_profile = resolve_loss_weights(
            loss_profile=loss_profile,
            gripper_loss_weight=gripper_loss_weight,
            close_phase_gripper_weight=close_phase_gripper_weight,
        )
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
        if temporal_feature_mode == "env_derived_phase":
            env_phases = np.asarray(
                [estimate_env_phase(row) for row in dataset.features],
                dtype=float,
            )
            phase_indices = env_phases[:, 0].astype(int)
            phase_progress = env_phases[:, 1]
            phase_lengths_by_key = {
                key: group_phase_lengths[index] for index, key in enumerate(group_keys)
            }
            progress = np.array(
                [
                    env_phase_global_progress(
                        int(phase_index),
                        float(phase_prog),
                        phase_lengths_by_key[str(key)],
                        max_progress_by_key[str(key)],
                    )
                    for phase_index, phase_prog, key in zip(
                        phase_indices, phase_progress, dataset.group_keys
                    )
                ],
                dtype=float,
            )
        else:
            progress = np.array(
                [
                    min(1.5, float(index) / max_progress_by_key[str(key)])
                    for index, key in zip(dataset.progress_indices, dataset.group_keys)
                ],
                dtype=float,
            )
            phase_indices = dataset.phase_indices
            phase_progress = dataset.phase_progress
        phase_one_hot = np.zeros((len(phase_indices), len(PHASE_LABELS)), dtype=float)
        phase_one_hot[np.arange(len(phase_indices)), phase_indices] = 1.0
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
    # Loss weights use demo phase labels (not env-derived phase / cursor).
    loss_weights = action_loss_weights(
        dataset.phase_indices,
        y.shape[1],
        arm_only_mlp=arm_only_mlp,
        gripper_loss_weight=gripper_loss_weight,
        close_phase_gripper_weight=close_phase_gripper_weight,
        loss_profile=None,  # already resolved above
    )

    rng = np.random.default_rng(seed)
    layer_sizes = (x.shape[1], *hidden_sizes, y.shape[1])
    weights = [
        rng.normal(
            0.0,
            np.sqrt(2.0 / (layer_sizes[index] + layer_sizes[index + 1])),
            size=(layer_sizes[index], layer_sizes[index + 1]),
        )
        for index in range(len(layer_sizes) - 1)
    ]
    biases = [np.zeros(size, dtype=float) for size in layer_sizes[1:]]
    weight_m = [np.zeros_like(weight) for weight in weights]
    weight_v = [np.zeros_like(weight) for weight in weights]
    bias_m = [np.zeros_like(bias) for bias in biases]
    bias_v = [np.zeros_like(bias) for bias in biases]

    step = 0
    last_loss = float("inf")
    last_weighted_loss = float("inf")
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
            # Weighted MSE: d/de mean(w * e^2) = (2/B) * w * e.
            grad = (2.0 / len(batch)) * (loss_weights[batch] * error)
            grad_weights: list[np.ndarray] = []
            grad_biases: list[np.ndarray] = []
            for layer_index in reversed(range(len(weights))):
                grad_weights.append(
                    activations[layer_index].T @ grad + weight_decay * weights[layer_index]
                )
                grad_biases.append(grad.sum(axis=0))
                if layer_index > 0:
                    grad = (grad @ weights[layer_index].T) * (
                        1.0 - np.tanh(preactivations[layer_index - 1]) ** 2
                    )
            grad_weights.reverse()
            grad_biases.reverse()
            step += 1
            _adam_update(weights, grad_weights, weight_m, weight_v, step, learning_rate)
            _adam_update(biases, grad_biases, bias_m, bias_v, step, learning_rate)
        residual = forward_mlp(x, weights, biases) - y
        last_loss = float(np.mean(residual**2))
        last_weighted_loss = float(np.mean(loss_weights * residual**2))
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
    sample_counts = phase_sample_counts(dataset.phase_indices)
    final_residual = forward_mlp(x, weights, biases) - y
    per_phase_losses = phase_action_loss_report(
        final_residual,
        dataset.phase_indices,
        loss_weights,
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
        "train_mse_weighted": last_weighted_loss,
        "gripper_loss_weight": float(gripper_loss_weight),
        "close_phase_gripper_weight": (
            None
            if close_phase_gripper_weight is None
            else float(close_phase_gripper_weight)
        ),
        "arm_only_mlp": bool(arm_only_mlp),
        "loss_profile": None if resolved_profile is None else resolved_profile.name,
        "loss_profile_contract": (
            None if resolved_profile is None else resolved_profile.to_dict()
        ),
        "close_timing_phases": list(CLOSE_TIMING_PHASES),
        "close_timing_sample_count": int(sample_counts["close_timing_total"]),
        "phase_sample_counts": sample_counts,
        "per_phase_action_loss": per_phase_losses,
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
    policy: NearestNeighborBCPolicy | MLPBCPolicy | HybridNNGripperMLPPolicy | Any,
    spec: PickupTrialSpec,
    max_steps: int = 3200,
    search_window: int = 120,
    action_gain: float = 1.0,
    gripper_close_guard: bool = False,
) -> PolicyTrialResult:
    # Optional oracle-FSM policies (H-EE-015) expose set_oracle_signals; default
    # hybrid / MLP / NN policies do not, so this path is a pure no-op for them.
    representation = get_action_representation(policy.action_space)
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
    gripper_command_flips = 0
    previous_gripper_closed: bool | None = None
    uses_oracle_fsm = callable(getattr(policy, "set_oracle_signals", None))

    for _ in range(max_steps):
        observation = env.get_observation()
        features = observation_to_features(observation, context, settled_start)
        # Privileged scripted-grasp signals for H-EE-015 oracle FSM only.
        if uses_oracle_fsm:
            ee_pos_pre, ee_quat_pre = env.controller.ee_pose(env.data)
            pos_err_pre = float(np.linalg.norm(grasp_pos - ee_pos_pre))
            rot_err_pre = float(
                np.linalg.norm(_orientation_error_rotvec(ee_quat_pre, grasp_quat))
            )
            policy.set_oracle_signals(
                pos_error_m=pos_err_pre,
                rot_error_rad=rot_err_pre,
                gripper_object_distance_m=env.gripper_object_distance(),
            )
        action, nearest_distance, nearest_index = policy.predict_with_index(
            features,
            context.key,
            cursor=cursor,
            search_window=search_window,
        )
        cursor = max(cursor + 1, nearest_index + 1)
        nearest_distances.append(nearest_distance)
        raw_action = action.copy()
        executed_action = representation.scale_arm(raw_action, action_gain)
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

        gripper_closed = bool(executed_action[-1] <= PRECLOSE_OPEN_THRESHOLD)
        if previous_gripper_closed is not None and gripper_closed != previous_gripper_closed:
            gripper_command_flips += 1
        previous_gripper_closed = gripper_closed

        executed_action_norms.append(float(np.linalg.norm(executed_action)))
        if previous_executed_action is not None:
            executed_action_delta_norms.append(
                float(np.linalg.norm(executed_action - previous_executed_action))
            )
        previous_executed_action = executed_action.copy()

        _, _, status = representation.execute(env, executed_action)
        telemetry = representation.telemetry(status)
        clipped_translation += int(telemetry["clipped_translation"])
        clipped_rotation += int(telemetry["clipped_rotation"])
        clipped_joints += int(telemetry["clipped_joints"])
        joint_limit_clipped += int(telemetry["joint_limit_clipped"])
        joint_step_clipped += int(telemetry["joint_step_clipped"])
        joint_accel_clipped += int(telemetry["joint_accel_clipped"])
        infeasible += int(telemetry["infeasible"])
        controller_failures += int(telemetry["controller_failed"])
        controller_failure_reason = (
            telemetry["failure_reason"] or controller_failure_reason
        )

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
        close_start_distance=(
            None
            if metrics["close_start_distance"] is None
            else float(metrics["close_start_distance"])
        ),
        first_close_time=(
            None if metrics["first_close_time"] is None else float(metrics["first_close_time"])
        ),
        first_contact_time=(
            None
            if metrics["first_contact_time"] is None
            else float(metrics["first_contact_time"])
        ),
        first_unsupported_time=(
            None
            if metrics["first_unsupported_time"] is None
            else float(metrics["first_unsupported_time"])
        ),
        first_lift_time=(
            None if metrics["first_lift_time"] is None else float(metrics["first_lift_time"])
        ),
        gripper_command_flips=int(gripper_command_flips),
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
