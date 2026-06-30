"""Human input adapters for SO-101 Cartesian teleoperation.

All devices emit a shared `TeleopIntent` in the **gripper-local** frame. See
`teleop_controller.py` for how those intents become world-frame target updates.

Binding map (keyboard baseline)
-------------------------------
Linear (gripper-local):
    W / S  -> forward / backward  (+/- local X)
    A / D  -> left / right        (+/- local Y)
    Q / E  -> up / down           (+/- local Z)

Orientation (gripper-local rotvec):
    Mouse / trackpad drag         -> pitch (mouse Y) and yaw (mouse X)
    I / K                         -> pitch manual backup
    J / L                         -> yaw manual backup
    U / O                         -> roll manual backup

Gripper:
    Space                         -> toggle open/closed

Session:
    R  reset arm + target
    P  pause/resume IK tracking
    H  print help

Gamepad mapping
-----------------
We support Xbox-style and PlayStation-style layouts via `GamepadProfile`. Axis
indices differ between vendors; profiles are easy to tune in code without
changing the integrator.

    Left stick        -> local X/Y translation (forward/back, left/right)
    Right stick Y     -> local Z translation (up/down)
    Right stick X     -> local roll
    Right stick + modifiers / hat -> pitch & yaw assist
    A / Cross         -> gripper toggle
    B / Circle        -> reset
    X / Square        -> pause
    Y / Triangle      -> help

Research notes baked into this design
-------------------------------------
- LeRobot SO-101 teleop uses a physical leader arm, not gamepads.
- gym-so100-c maps gamepad axes to *joint* deltas; we intentionally map to
  *Cartesian tool-frame* deltas because that matches the project's controller-
  first hypothesis and future policy action space.
- pygame is used instead of raw HID because Xbox and DualSense controllers are
  reliably enumerated on macOS through SDL. HID byte offsets are vendor-specific
  and brittle (see gym-so100-c `GamepadControllerHID`).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from svla.teleop_controller import TeleopIntent, TeleopRates


def _ensure_pygame() -> None:
    import pygame

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    if not pygame.get_init():
        pygame.init()
    if pygame.display.get_surface() is None:
        pygame.display.set_mode((1, 1))


@dataclass(frozen=True)
class GamepadProfile:
    """Axis/button indices for a controller family."""

    name: str
    left_x: int = 0
    left_y: int = 1
    right_x: int = 2
    right_y: int = 3
    invert_left_y: bool = True
    invert_right_y: bool = True
    gripper_button: int = 0
    reset_button: int = 1
    pause_button: int = 2
    help_button: int = 3


XBOX_PROFILE = GamepadProfile(name="xbox")
PLAYSTATION_PROFILE = GamepadProfile(
    name="playstation",
    # Many DualSense mappings match Xbox in SDL, but we keep a separate profile
    # so per-platform tuning does not leak into gameplay semantics.
    gripper_button=0,
    reset_button=1,
    pause_button=2,
    help_button=3,
)


def _detect_gamepad_profile(device_name: str) -> GamepadProfile:
    lowered = device_name.lower()
    if any(token in lowered for token in ("playstation", "dualsense", "dualshock", "ps5", "ps4", "wireless controller")):
        return PLAYSTATION_PROFILE
    return XBOX_PROFILE


class KeyboardTeleop:
    """WASD/QE translation + mouse rotation + Space gripper toggle."""

    _PYGAME_KEYMAP: dict[int, str] | None = None

    def __init__(self, rates: TeleopRates | None = None) -> None:
        self.rates = rates or TeleopRates()
        self._edge: set[str] = set()

    @classmethod
    def _pygame_keys(cls) -> dict[int, str]:
        if cls._PYGAME_KEYMAP is None:
            import pygame

            cls._PYGAME_KEYMAP = {
                pygame.K_w: "w",
                pygame.K_s: "s",
                pygame.K_a: "a",
                pygame.K_d: "d",
                pygame.K_q: "q",
                pygame.K_e: "e",
                pygame.K_i: "i",
                pygame.K_k: "k",
                pygame.K_j: "j",
                pygame.K_l: "l",
                pygame.K_u: "u",
                pygame.K_o: "o",
                pygame.K_SPACE: "space",
                pygame.K_r: "r",
                pygame.K_p: "p",
                pygame.K_h: "h",
                pygame.K_n: "n",
                pygame.K_1: "1",
                pygame.K_2: "2",
                pygame.K_3: "3",
                pygame.K_4: "4",
            }
        return cls._PYGAME_KEYMAP

    def on_key(self, key: int) -> None:
        # MuJoCo viewer callback only gives edge events; mirror into pygame edge set.
        if key < 0 or key >= 256:
            return
        char = chr(key).lower()
        if char in {
            "w",
            "s",
            "a",
            "d",
            "q",
            "e",
            "i",
            "k",
            "j",
            "l",
            "u",
            "o",
            "r",
            "p",
            "h",
            "n",
            "1",
            "2",
            "3",
            "4",
        }:
            self._edge.add(char)
        if key == 32:
            self._edge.add("space")

    def _pressed_names(self) -> set[str]:
        import pygame

        pressed: set[str] = set()
        keys = pygame.key.get_pressed()
        for key_code, name in self._pygame_keys().items():
            if keys[key_code]:
                pressed.add(name)
        return pressed

    def poll(self) -> TeleopIntent:
        import pygame

        _ensure_pygame()
        pygame.event.pump()
        for event in pygame.event.get(pygame.KEYDOWN):
            name = self._pygame_keys().get(event.key)
            if name is not None:
                self._edge.add(name)

        intent = TeleopIntent()
        linear = self.rates.linear
        rot = self.rates.rotational
        pressed = self._pressed_names()
        active = pressed | self._edge

        # Tool-frame translation ------------------------------------------------
        if "w" in active:
            intent.local_linear[0] += linear
        if "s" in active:
            intent.local_linear[0] -= linear
        if "a" in active:
            intent.local_linear[1] += linear
        if "d" in active:
            intent.local_linear[1] -= linear
        if "q" in active:
            intent.local_linear[2] += linear
        if "e" in active:
            intent.local_linear[2] -= linear

        # Tool-frame rotation backups -------------------------------------------
        if "i" in active:
            intent.local_rotvec[1] += rot
        if "k" in active:
            intent.local_rotvec[1] -= rot
        if "j" in active:
            intent.local_rotvec[2] += rot
        if "l" in active:
            intent.local_rotvec[2] -= rot
        if "u" in active:
            intent.local_rotvec[0] += rot
        if "o" in active:
            intent.local_rotvec[0] -= rot

        if "space" in self._edge:
            intent.toggle_gripper = True
        if "r" in self._edge:
            intent.reset = True
        if "p" in self._edge:
            intent.pause_toggle = True
        if "h" in self._edge:
            intent.show_help = True
        if "n" in self._edge:
            intent.random_target = True
        for preset in ("1", "2", "3", "4"):
            if preset in self._edge:
                intent.preset_target = preset

        self._edge.clear()
        return intent


class MouseTeleop:
    """Trackpad/mouse drag for tool-frame pitch/yaw."""

    def __init__(self, rates: TeleopRates | None = None) -> None:
        self.rates = rates or TeleopRates()

    def poll(self) -> TeleopIntent:
        import pygame

        _ensure_pygame()
        pygame.event.pump()
        rel_x, rel_y = pygame.mouse.get_rel()
        if rel_x == 0 and rel_y == 0:
            return TeleopIntent()

        intent = TeleopIntent()
        scale = self.rates.mouse_rot_scale
        # Mouse right -> positive local yaw; mouse up -> positive local pitch.
        intent.local_rotvec[2] += rel_x * scale
        intent.local_rotvec[1] -= rel_y * scale
        return intent


class GamepadTeleop:
    """Xbox / PlayStation compatible gamepad adapter via pygame."""

    def __init__(self, rates: TeleopRates | None = None, deadzone: float = 0.12) -> None:
        self.rates = rates or TeleopRates()
        self.deadzone = deadzone
        self._joystick = None
        self._available = False
        self._profile = XBOX_PROFILE
        self._edge_buttons: set[int] = set()

    def start(self) -> bool:
        import pygame

        _ensure_pygame()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            return False
        self._joystick = pygame.joystick.Joystick(0)
        self._joystick.init()
        self._profile = _detect_gamepad_profile(self._joystick.get_name())
        self._available = True
        return True

    @property
    def name(self) -> str | None:
        return self._joystick.get_name() if self._joystick is not None else None

    @property
    def profile_name(self) -> str | None:
        return self._profile.name if self._available else None

    @property
    def available(self) -> bool:
        return self._available

    def stop(self) -> None:
        import pygame

        if self._joystick is not None:
            self._joystick.quit()
            self._joystick = None
        self._available = False
        if pygame.joystick.get_count() == 0:
            pygame.joystick.quit()

    def _axis(self, index: int) -> float:
        if self._joystick is None:
            return 0.0
        value = float(self._joystick.get_axis(index))
        if abs(value) < self.deadzone:
            return 0.0
        return value

    def poll(self) -> TeleopIntent:
        import pygame

        _ensure_pygame()
        intent = TeleopIntent()
        if not self._available or self._joystick is None:
            return intent

        pygame.event.pump()
        for event in pygame.event.get():
            if event.type == pygame.JOYBUTTONDOWN:
                self._edge_buttons.add(event.button)

        linear = self.rates.linear
        rot = self.rates.rotational
        profile = self._profile

        left_x = self._axis(profile.left_x)
        left_y = self._axis(profile.left_y)
        right_x = self._axis(profile.right_x)
        right_y = self._axis(profile.right_y)
        if profile.invert_left_y:
            left_y = -left_y
        if profile.invert_right_y:
            right_y = -right_y

        intent.local_linear[0] += left_y * linear
        intent.local_linear[1] -= left_x * linear
        intent.local_linear[2] += right_y * linear
        intent.local_rotvec[0] += right_x * rot

        if self._joystick.get_numhats() > 0:
            hat_x, hat_y = self._joystick.get_hat(0)
            intent.local_rotvec[1] += hat_y * rot
            intent.local_rotvec[2] += hat_x * rot

        if profile.gripper_button in self._edge_buttons:
            intent.toggle_gripper = True
        if profile.reset_button in self._edge_buttons:
            intent.reset = True
        if profile.pause_button in self._edge_buttons:
            intent.pause_toggle = True
        if profile.help_button in self._edge_buttons:
            intent.show_help = True

        self._edge_buttons.clear()
        return intent


class TeleopInputManager:
    """Merge keyboard, mouse, and gamepad into one gripper-local intent stream."""

    def __init__(self, rates: TeleopRates | None = None) -> None:
        self.rates = rates or TeleopRates()
        _ensure_pygame()
        self.keyboard = KeyboardTeleop(self.rates)
        self.mouse = MouseTeleop(self.rates)
        self.gamepad = GamepadTeleop(self.rates)
        self.gamepad_connected = self.gamepad.start()

    def on_key(self, key: int) -> None:
        self.keyboard.on_key(key)

    def poll(self) -> TeleopIntent:
        intent = TeleopIntent()
        intent.merge(self.keyboard.poll())
        intent.merge(self.mouse.poll())
        if self.gamepad_connected:
            intent.merge(self.gamepad.poll())
        return intent

    def close(self) -> None:
        self.gamepad.stop()

    def status_line(self) -> str:
        if self.gamepad_connected:
            return f"gamepad={self.gamepad.name} profile={self.gamepad.profile_name}"
        return "gamepad=not connected (keyboard + mouse/trackpad active)"


def teleop_help_text() -> str:
    return """
SO-101 tool-frame teleop
  Movement is relative to the gripper, not the world:
    W/S = forward / backward along where the gripper points
    A/D = left / right relative to the gripper
    Q/E = up / down relative to the gripper
  If W would cross the reachable workspace boundary, only the blocked world
  axis is nulled; other components of the motion still apply.

  Orientation
    Mouse / trackpad drag = pitch + yaw
    I/K/J/L/U/O = manual pitch/yaw/roll backup
    Gamepad right stick X = roll, hat = pitch/yaw assist

  Gripper
    Space = toggle open/closed
    Gamepad A/Cross = toggle

  Session
    R = reset, P = pause IK tracking, H = help
"""
