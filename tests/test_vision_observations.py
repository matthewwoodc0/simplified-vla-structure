import numpy as np
import pytest

from svla.pickup_task import PickupTaskEvaluator
from svla.vision_observations import FixedCameraConfig, FixedCameraRenderer, camera_metadata


def test_fixed_camera_config_reports_rgb_uint8_metadata():
    config = FixedCameraConfig(name="overview", width=64, height=48)
    metadata = config.to_metadata()

    assert metadata["shape"] == [48, 64, 3]
    assert metadata["dtype"] == "uint8"
    assert metadata["color_space"] == "rgb"
    assert metadata["renderer"] == "mujoco.Renderer"


def test_fixed_camera_config_rejects_non_uint8():
    with pytest.raises(ValueError, match="uint8"):
        FixedCameraConfig(name="overview", width=64, height=48, dtype="float32")


def test_pickup_task_rgb_observation_is_opt_in_and_does_not_mutate_state_observation():
    env = PickupTaskEvaluator()
    before = env.get_observation()

    frame = env.get_rgb_observation(width=64, height=48)
    after = env.get_observation()

    assert frame.shape == (48, 64, 3)
    assert frame.dtype == np.uint8
    assert before == after


def test_fixed_camera_renderer_renders_named_camera():
    env = PickupTaskEvaluator()
    renderer = FixedCameraRenderer(
        env.model,
        (FixedCameraConfig(name="overview", width=64, height=48),),
    )
    try:
        frame = renderer.render(env.data, "overview")
    finally:
        renderer.close()

    assert frame.shape == (48, 64, 3)
    assert frame.dtype == np.uint8
    assert frame.max() > frame.min()
    assert "overview" in camera_metadata()["cameras"]
