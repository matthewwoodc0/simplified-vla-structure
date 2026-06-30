import numpy as np

from svla.teleop_controller import TeleopIntent
from svla.teleop_inputs import KeyboardTeleop, TeleopInputManager


def test_keyboard_poll_returns_tool_frame_intent():
    teleop = KeyboardTeleop()
    frame = teleop.poll()
    assert isinstance(frame, TeleopIntent)
    assert frame.local_linear.shape == (3,)
    assert frame.local_rotvec.shape == (3,)


def test_mujoco_key_callback_edges_drive_one_frame_motion(monkeypatch):
    teleop = KeyboardTeleop()
    monkeypatch.setattr(teleop, "_pressed_names", lambda: set())

    teleop.on_key(ord("w"))
    frame = teleop.poll()
    assert frame.local_linear[0] > 0.0

    next_frame = teleop.poll()
    assert np.allclose(next_frame.local_linear, np.zeros(3))


def test_mujoco_key_callback_edges_drive_backup_rotation(monkeypatch):
    teleop = KeyboardTeleop()
    monkeypatch.setattr(teleop, "_pressed_names", lambda: set())

    teleop.on_key(ord("i"))
    frame = teleop.poll()
    assert frame.local_rotvec[1] > 0.0


def test_teleop_manager_poll_returns_finite_intent():
    manager = TeleopInputManager()
    try:
        frame = manager.poll()
        assert np.isfinite(frame.local_linear).all()
        assert np.isfinite(frame.local_rotvec).all()
    finally:
        manager.close()
