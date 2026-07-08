from __future__ import annotations

from dataclasses import asdict, dataclass

import mujoco
import numpy as np


VISION_OBSERVATION_FORMAT = "svla_fixed_camera_rgb_v1"


@dataclass(frozen=True)
class FixedCameraConfig:
    name: str = "overview"
    width: int = 320
    height: int = 240
    dtype: str = "uint8"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("camera name must be non-empty")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("camera width and height must be positive")
        if self.dtype != "uint8":
            raise ValueError("only uint8 RGB camera frames are supported")

    @property
    def shape(self) -> tuple[int, int, int]:
        return (int(self.height), int(self.width), 3)

    def to_metadata(self) -> dict:
        return {
            **asdict(self),
            "shape": list(self.shape),
            "format": VISION_OBSERVATION_FORMAT,
            "color_space": "rgb",
            "renderer": "mujoco.Renderer",
            "deterministic_settings": {
                "camera": self.name,
                "segmentation": False,
                "depth": False,
            },
        }


DEFAULT_CAMERA_CONFIGS = (
    FixedCameraConfig(name="overview", width=320, height=240),
)


class FixedCameraRenderer:
    """Small MuJoCo RGB renderer that keeps camera capture opt-in."""

    def __init__(
        self,
        model: mujoco.MjModel,
        camera_configs: tuple[FixedCameraConfig, ...] = DEFAULT_CAMERA_CONFIGS,
    ) -> None:
        if not camera_configs:
            raise ValueError("at least one camera config is required")
        self.model = model
        self.camera_configs = tuple(camera_configs)
        self._renderers = {
            config.name: mujoco.Renderer(
                model,
                height=int(config.height),
                width=int(config.width),
            )
            for config in self.camera_configs
        }

    @property
    def metadata(self) -> dict:
        return {
            "format": VISION_OBSERVATION_FORMAT,
            "cameras": {
                config.name: config.to_metadata()
                for config in self.camera_configs
            },
        }

    def render(self, data: mujoco.MjData, camera_name: str) -> np.ndarray:
        if camera_name not in self._renderers:
            raise KeyError(f"unknown camera {camera_name!r}")
        renderer = self._renderers[camera_name]
        renderer.update_scene(data, camera=camera_name)
        frame = renderer.render()
        if frame.dtype != np.uint8:
            frame = np.asarray(frame, dtype=np.uint8)
        expected_shape = self._config_by_name(camera_name).shape
        if frame.shape != expected_shape:
            raise RuntimeError(
                f"camera {camera_name!r} rendered shape {frame.shape}, expected {expected_shape}"
            )
        return frame.copy()

    def render_all(self, data: mujoco.MjData) -> dict[str, np.ndarray]:
        return {
            config.name: self.render(data, config.name)
            for config in self.camera_configs
        }

    def close(self) -> None:
        for renderer in self._renderers.values():
            renderer.close()
        self._renderers.clear()

    def _config_by_name(self, camera_name: str) -> FixedCameraConfig:
        for config in self.camera_configs:
            if config.name == camera_name:
                return config
        raise KeyError(f"unknown camera {camera_name!r}")


def camera_metadata(
    camera_configs: tuple[FixedCameraConfig, ...] = DEFAULT_CAMERA_CONFIGS,
) -> dict:
    return {
        "format": VISION_OBSERVATION_FORMAT,
        "cameras": {
            config.name: config.to_metadata()
            for config in camera_configs
        },
    }
