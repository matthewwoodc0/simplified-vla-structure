"""Frozen loss-profile contracts for H-EE-008 causal decomposition (H-EE-021).

H-EE-008 applied two interventions at once:
  1. global gripper MSE weight 5×
  2. transition (grasp_align / close_gripper) gripper weight 10×

These named profiles isolate those factors under an otherwise identical training
and protocol-v2 validation contract. Profiles are intentional constants — do not
edit weights in place; add a new named profile if a later experiment needs
different scalars.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping


FORMAT_VERSION = "svla_loss_profile_contract_v1"

# Demo phases where close timing is critical for event-order gates.
# Must stay aligned with state_bc.CLOSE_TIMING_PHASES.
CLOSE_TIMING_PHASES: tuple[str, ...] = ("grasp_align", "close_gripper")


@dataclass(frozen=True)
class LossProfile:
    """Named MSE reweighting contract. Arm action dims always stay at weight 1.0."""

    name: str
    global_gripper_weight: float
    transition_gripper_weight: float
    description: str

    def __post_init__(self) -> None:
        if self.global_gripper_weight <= 0.0:
            raise ValueError(f"{self.name}: global_gripper_weight must be positive")
        if self.transition_gripper_weight <= 0.0:
            raise ValueError(f"{self.name}: transition_gripper_weight must be positive")

    @property
    def gripper_loss_weight(self) -> float:
        """Alias used by fit_mlp_policy / train_state_bc CLI fields."""

        return float(self.global_gripper_weight)

    @property
    def close_phase_gripper_weight(self) -> float:
        """Alias used by fit_mlp_policy / train_state_bc CLI fields."""

        return float(self.transition_gripper_weight)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["close_timing_phases"] = list(CLOSE_TIMING_PHASES)
        payload["arm_weight"] = 1.0
        payload["format"] = FORMAT_VERSION
        return payload


# Frozen H-EE-021 factorial matrix. Do not mutate.
LOSS_PROFILES: Mapping[str, LossProfile] = {
    "uniform": LossProfile(
        name="uniform",
        global_gripper_weight=1.0,
        transition_gripper_weight=1.0,
        description="Uniform MSE on all action dims (registered Phase-5 baseline loss).",
    ),
    "global_gripper": LossProfile(
        name="global_gripper",
        global_gripper_weight=5.0,
        transition_gripper_weight=5.0,
        description="Global 5× gripper emphasis with no extra transition boost.",
    ),
    "transition_gripper": LossProfile(
        name="transition_gripper",
        global_gripper_weight=1.0,
        transition_gripper_weight=10.0,
        description="Transition-only 10× gripper emphasis on grasp_align/close_gripper.",
    ),
    "combined_h_ee_008": LossProfile(
        name="combined_h_ee_008",
        global_gripper_weight=5.0,
        transition_gripper_weight=10.0,
        description="H-EE-008 combined contract: global 5× and transition 10×.",
    ),
}

LOSS_PROFILE_NAMES: tuple[str, ...] = tuple(LOSS_PROFILES.keys())

# Research parity frontier: H-EE-008 weighted joint validation (not a release claim).
RESEARCH_PARITY_FRONTIER: dict = {
    "source_hypothesis": "H-EE-008",
    "source_action_space": "joint_delta",
    "source_contract": "combined_h_ee_008",
    "split": "validation",
    "protocol": "v2",
    "total_trials": 120,
    "successes": 84,
    "event_order_valid": 90,
    "physical_sanity_pass": 100,
    "worst_seed_successes_min": 12,
    "worst_seed_trials": 24,
    "note": (
        "Research target for EE catch-up under protocol-v2 validation. "
        "Final holdout stays closed until one EE configuration is selected."
    ),
}


def get_loss_profile(name: str) -> LossProfile:
    key = str(name).strip()
    if key not in LOSS_PROFILES:
        known = ", ".join(LOSS_PROFILE_NAMES)
        raise ValueError(f"unknown loss profile {name!r}; expected one of: {known}")
    return LOSS_PROFILES[key]


def resolve_loss_weights(
    *,
    loss_profile: str | None = None,
    gripper_loss_weight: float = 1.0,
    close_phase_gripper_weight: float | None = None,
) -> tuple[float, float | None, LossProfile | None]:
    """Resolve CLI weights. Named profile overrides free-form weight flags."""

    if loss_profile is not None:
        profile = get_loss_profile(loss_profile)
        return (
            profile.gripper_loss_weight,
            profile.close_phase_gripper_weight,
            profile,
        )
    return float(gripper_loss_weight), close_phase_gripper_weight, None


def expected_gripper_weight_for_phase(profile: LossProfile, phase: str) -> float:
    """Gripper dim weight for a demo phase under a frozen profile."""

    if phase in CLOSE_TIMING_PHASES:
        return float(profile.transition_gripper_weight)
    return float(profile.global_gripper_weight)
