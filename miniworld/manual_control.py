import math
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pyglet
from gymnasium import spaces
from pyglet.window import key, mouse

from miniworld.lerobot_writer import DatasetManager, EpisodeWriter, build_state_vector


class ManualControl:
    def __init__(
        self,
        env,
        no_time_limit: bool,
        domain_rand: bool,
        mouse_sensitivity: float = 0.0025,
        mouse_rotation_deadzone: float = 0.05,
        fullscreen: bool = False,
        window_size: Optional[str] = None,
        task_description: str = "Center and zoom on the target.",
        append: bool = False,
        automatic_recording: bool = False,
        max_chunk_size_mb: Optional[int] = None,
        show_controls: bool = True,
    ):
        self.env = env.unwrapped
        self._box_action_space = self._get_box_action_space()
        self._discrete_actions = self._get_discrete_actions()
        self._discrete_action_map = self._build_discrete_action_lookup()

        self.key_handler = key.KeyStateHandler()
        self.mouse_dx = 0.0
        self.mouse_dy = 0.0
        self.pickup_requested = False
        self.drop_requested = False
        self._ignore_mouse_motion = False

        # Mouse sensitivity factors for yaw/pitch updates
        self.turn_sensitivity = mouse_sensitivity
        self.pitch_sensitivity = mouse_sensitivity
        self.mouse_rotation_deadzone = mouse_rotation_deadzone

        self._last_mouse_turn_delta = 0.0
        self._last_mouse_pitch_delta = 0.0

        self._fullscreen = fullscreen
        # When the on-screen controls are enabled, allow the mouse cursor to move
        # freely so the user can click the buttons. Mouse-look should only be
        # active when the controls are hidden.
        self._mouse_exclusive = not show_controls
        self._window_size: Optional[Tuple[int, int]] = None
        if window_size is not None:
            self._window_size = self._parse_window_size(window_size)

        self._windowed_size: Optional[Tuple[int, int]] = None

        if no_time_limit:
            self.env.max_episode_steps = math.inf
        if domain_rand:
            self.env.domain_rand = True

        self._episode_writer: Optional[EpisodeWriter] = None
        self._episode_index: int = 0
        self._frame_index: int = 0
        self._recordings_dir = Path.cwd() / "episode_recordings"
        self._task_description = task_description
        max_chunk_size_bytes = (
            200_000_000 if max_chunk_size_mb is None else int(max_chunk_size_mb * 1_000_000)
        )
        self._dataset_manager = DatasetManager(
            self._recordings_dir,
            default_task=self._task_description,
            append=append,
            max_chunk_size_bytes=max_chunk_size_bytes,
        )
        if append:
            self._episode_index = self._dataset_manager.num_episodes
            if self._episode_index:
                print(
                    "[Recorder] Append mode enabled; "
                    f"resuming at episode index {self._episode_index}"
                )
        self._exit_requested = False
        self._automatic_recording = automatic_recording
        self._pressed_controls = set()
        self._hovered_control: Optional[str] = None
        self._show_controls = show_controls

    @staticmethod
    def _parse_window_size(window_size: str):
        try:
            width_str, height_str = window_size.lower().split("x", maxsplit=1)
            width, height = int(width_str), int(height_str)
        except Exception as exc:  # pragma: no cover - user input parsing
            raise ValueError(
                "Window size must be provided as WIDTHxHEIGHT (e.g., 1920x1080)."
            ) from exc

        if width <= 0 or height <= 0:
            raise ValueError("Window dimensions must be positive integers.")

        return width, height

    def run(self):
        print("============")
        print("Instructions")
        print("============")
        print(
            "move: arrow keys (mouse to look)\n"
            "strafe: A/D\npickup: P\ndrop: B\n"
            "toggle fullscreen: F11\nquit: ESC"
        )
        if self._show_controls:
            print(
                "Tip: click the on-screen buttons to move, strafe, and look if you prefer"
                " not to use the keyboard/mouse."
            )
        print("============")

        self.env.reset()

        # Create the display window
        self.env.render()

        env = self.env

        window = env.unwrapped.window
        window.push_handlers(self.key_handler)
        self._ensure_mouse_capture(window)

        if self._fullscreen:
            window.set_fullscreen(True)
        elif self._window_size is not None:
            width, height = self._window_size
            window.set_size(width, height)

        window.set_exclusive_mouse(self._mouse_exclusive)
        self._windowed_size = (window.width, window.height)
        self._recenter_mouse_cursor(window)

        if self._automatic_recording:
            self._start_episode_writer_if_needed()

        @env.unwrapped.window.event
        def on_key_press(symbol, modifiers):
            """
            This handler processes keyboard commands that
            control the simulation
            """

            # Update key state tracking so held keys are captured.
            self.key_handler.on_key_press(symbol, modifiers)

            if symbol == key.BACKSPACE or symbol == key.SLASH:
                print("RESET")
                self.env.reset()
                self.env.render()
                return

            if symbol == key.ESCAPE:
                self._stop_episode_writer()
                self._request_exit()
                return

            if symbol == key.ENTER:
                self._request_exit()
                return

            if symbol == key.PAGEUP or symbol == key.P:
                self.pickup_requested = True
            elif symbol == key.PAGEDOWN or symbol == key.B:
                self.drop_requested = True
            elif symbol == key.SPACE:
                self._toggle_episode_writer()
            elif symbol == key.F11:
                self._toggle_fullscreen()

        @env.unwrapped.window.event
        def on_key_release(symbol, modifiers):
            # Keep KeyStateHandler in sync with released keys.
            self.key_handler.on_key_release(symbol, modifiers)

        @env.unwrapped.window.event
        def on_mouse_press(x, y, button, modifiers):
            control = self._control_at_position(x, y)
            if control is not None and button == mouse.LEFT:
                self._pressed_controls.add(control)
                self._sync_control_pressed_states()
                return

        @env.unwrapped.window.event
        def on_mouse_release(x, y, button, modifiers):
            control = self._control_at_position(x, y)
            if button == mouse.LEFT:
                if control is None:
                    self._pressed_controls.clear()
                else:
                    self._pressed_controls.discard(control)
                self._sync_control_pressed_states()

            self._update_hover_state(x, y)

        @env.unwrapped.window.event
        def on_mouse_motion(x, y, dx, dy):
            self._update_hover_state(x, y)
            if self._ignore_mouse_motion:
                self._ignore_mouse_motion = False
                return
            self.mouse_dx += dx
            self.mouse_dy += dy
            self._recenter_mouse_cursor(window)

        @env.unwrapped.window.event
        def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
            self._update_hover_state(x, y)
            if self._ignore_mouse_motion:
                self._ignore_mouse_motion = False
                return
            self.mouse_dx += dx
            self.mouse_dy += dy
            self._recenter_mouse_cursor(window)

        @env.unwrapped.window.event
        def on_draw():
            self.env.render()

        @env.unwrapped.window.event
        def on_close():
            # Defer actual window destruction until after the pyglet event loop
            # exits to avoid losing the GL context while ``idle`` is still
            # running (which causes ``AttributeError: 'NoneType' object has no
            # attribute 'flip'`` inside pyglet on Windows).
            self._stop_episode_writer()
            self._request_exit()
            return pyglet.event.EVENT_HANDLED

        @env.unwrapped.window.event
        def on_activate():
            # Re-enable mouse capture when the window regains focus to avoid
            # scenarios where the cursor remains free until a fullscreen toggle
            # occurs (common on some platforms when starting in fullscreen).
            self._ensure_mouse_capture(window)

        def update(dt):
            if self._exit_requested:
                return

            action = np.zeros(len(self.env.actions), dtype=np.float32)

            # Keyboard state-based movement
            if not self._show_controls:
                action[self.env.actions.forward_speed] = (
                    float(self.key_handler[key.UP])
                    + float(self.key_handler[key.W])
                    - float(self.key_handler[key.DOWN])
                    - float(self.key_handler[key.S])
                )
                action[self.env.actions.strafe_speed] = (
                    float(self.key_handler[key.D]) - float(self.key_handler[key.A])
                )

            action[self.env.actions.forward_speed] += (
                float("forward" in self._pressed_controls)
                - float("backward" in self._pressed_controls)
            )
            action[self.env.actions.strafe_speed] += (
                float("strafe_right" in self._pressed_controls)
                - float("strafe_left" in self._pressed_controls)
            )

            turn_input = 0.0
            if not self._show_controls:
                turn_input = float(self.key_handler[key.RIGHT]) - float(
                    self.key_handler[key.LEFT]
                )
            turn_input += (
                float("turn_left" in self._pressed_controls)
                - float("turn_right" in self._pressed_controls)
            )
            mouse_turn_delta = (
                -self.mouse_dx * self.turn_sensitivity if not self._show_controls else 0.0
            )
            mouse_pitch_delta = (
                self.mouse_dy * self.pitch_sensitivity if not self._show_controls else 0.0
            )

            pitch_input = (
                float("pitch_up" in self._pressed_controls)
                - float("pitch_down" in self._pressed_controls)
            )

            self._last_mouse_turn_delta = mouse_turn_delta
            self._last_mouse_pitch_delta = mouse_pitch_delta

            turn_input += mouse_turn_delta
            pitch_input += mouse_pitch_delta

            action[self.env.actions.turn_delta] = turn_input
            action[self.env.actions.pitch_delta] = pitch_input

            if self.pickup_requested:
                action[self.env.actions.pickup] = 1.0
            if self.drop_requested:
                action[self.env.actions.drop] = 1.0

            self.mouse_dx = 0.0
            self.mouse_dy = 0.0
            self.pickup_requested = False
            self.drop_requested = False

            action_to_take = self._map_controls_to_action(action)

            if action_to_take is None:
                self.env.render()
                idle_info = {"agent": self.env.unwrapped._get_agent_state()}
                idle_frame = self.env.render_obs()
                self._record_frame_if_needed(
                    action,
                    frame=idle_frame,
                    info=idle_info,
                    timestamp=time.time(),
                )
                return

            if isinstance(action_to_take, np.ndarray) and self._box_action_space is not None:
                action_to_take = np.clip(
                    action_to_take,
                    self._box_action_space.low,
                    self._box_action_space.high,
                )

            self.step(action_to_take)

        pyglet.clock.schedule_interval(update, 1.0 / 60.0)

        # Enter main event loop
        pyglet.app.run()

        window.set_visible(False)
        self._stop_episode_writer()
        self.env.close()
        self._dataset_manager.finalize()

    def _request_exit(self):
        if self._exit_requested:
            return

        self._exit_requested = True
        pyglet.app.exit()

    def step(self, action):
        if isinstance(action, np.ndarray):
            action_desc = self._describe_action_vector(action)
        elif isinstance(self.env.action_space, spaces.Discrete):
            action_vector = self._discrete_actions[int(action)]
            action_desc = self._describe_action_vector(action_vector)
        else:
            action_desc = str(action)

        print(
            f"step {self.env.unwrapped.step_count + 1}/{self.env.unwrapped.max_episode_steps}: {action_desc}"
        )

        obs, reward, termination, truncation, info = self.env.step(action)

        self._record_frame_if_needed(
            action,
            frame=obs,
            info=info,
        )

        if reward > 0:
            print(f"reward={reward:.2f}")

        if termination or truncation:
            print("done!")
            if self._automatic_recording:
                self._stop_episode_writer()
            self.env.reset()
            if self._automatic_recording:
                self._start_episode_writer_if_needed()

        self.env.render()

    def _toggle_fullscreen(self):
        window = self.env.unwrapped.window

        if window is None:
            return

        target_fullscreen = not window.fullscreen

        if target_fullscreen:
            self._windowed_size = (window.width, window.height)
            window.set_fullscreen(True)
        else:
            window.set_fullscreen(False)

            restore_size = self._window_size or self._windowed_size
            if restore_size is not None:
                width, height = restore_size
                window.set_size(width, height)

        self._ensure_mouse_capture(window)
        self._fullscreen = window.fullscreen
        
    def _toggle_episode_writer(self):
        if self._episode_writer is None:
            self._start_episode_writer()
        else:
            self._stop_episode_writer()

    def _start_episode_writer_if_needed(self):
        if self._episode_writer is None:
            self._start_episode_writer()

    def _start_episode_writer(self):
        self._frame_index = 0
        self._episode_writer = self._dataset_manager.create_episode_writer(
            episode_index=self._episode_index, tasks=[self._task_description]
        )
        print(
            f"[Recorder] Started recording episode {self._episode_index}"
        )

    def _stop_episode_writer(self):
        if self._episode_writer is None:
            return

        episode_dir = self._episode_writer.close()
        print(
            f"[Recorder] Saved {self._episode_writer.num_frames} frames to {episode_dir}"
        )
        self._episode_index += 1
        self._episode_writer = None

    def _record_frame_if_needed(self, action, *, frame=None, info=None, timestamp=None):
        if self._episode_writer is None:
            return

        action_vector = self._normalize_action(action)
        frame = frame if frame is not None else self.env.render_obs()
        info = info or {"agent": self.env.unwrapped._get_agent_state()}
        state = build_state_vector(info)

        self._episode_writer.add_sample(
            frame=frame,
            action=action_vector,
            state=state,
            frame_index=self._frame_index,
            timestamp=time.time() if timestamp is None else timestamp,
        )
        self._frame_index += 1

    def _normalize_action(self, action):
        if isinstance(action, np.ndarray):
            return np.array(action, dtype=np.float32)
        if isinstance(self.env.action_space, spaces.Discrete):
            return np.array(self._discrete_actions[int(action)], dtype=np.float32)
        return np.array([action], dtype=np.float32)

    def _control_at_position(self, x, y) -> Optional[str]:
        if not self._show_controls:
            return None

        control_boxes = getattr(self.env.unwrapped, "control_boxes", {})

        for name, data in control_boxes.items():
            bounds = data.get("bounds") if isinstance(data, dict) else data
            if bounds is None:
                continue

            box_x, box_y, box_w, box_h = bounds

            if box_x <= x <= box_x + box_w and box_y <= y <= box_y + box_h:
                return name

        return None

    def _update_hover_state(self, x, y):
        if not self._show_controls:
            return

        hovered = self._control_at_position(x, y)

        if hovered == self._hovered_control:
            return

        self._hovered_control = hovered

        env = getattr(self.env, "unwrapped", self.env)

        if hasattr(env, "set_control_hover"):
            env.set_control_hover(hovered)

    def _sync_control_pressed_states(self):
        if not self._show_controls:
            return

        env = getattr(self.env, "unwrapped", self.env)

        if hasattr(env, "set_control_pressed"):
            env.set_control_pressed(self._pressed_controls)

    def _recenter_mouse_cursor(self, window):
        if not self._mouse_exclusive or window is None:
            return

        self._ignore_mouse_motion = True
        center_x = window.width // 2
        center_y = window.height // 2
        window.set_mouse_position(center_x, center_y)

    def _ensure_mouse_capture(self, window):
        if not self._mouse_exclusive or window is None:
            return

        window.set_exclusive_mouse(True)
        self._recenter_mouse_cursor(window)

    def _get_box_action_space(self):
        action_space = getattr(self.env, "action_space", None)
        if isinstance(action_space, spaces.Box):
            return action_space

        try:
            low = np.array([-1.0, -1.0, -1.0, -1.0, 0.0, 0.0], dtype=np.float32)
            high = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
            return spaces.Box(low=low, high=high, dtype=np.float32)
        except Exception:
            return None

    def _get_discrete_actions(self):
        discrete_actions = getattr(self.env, "_discrete_actions", None)
        if discrete_actions is None and hasattr(self.env, "_default_discrete_actions"):
            discrete_actions = self.env._default_discrete_actions()

        if discrete_actions is None:
            return []

        return [np.asarray(action, dtype=np.float32) for action in discrete_actions]

    def _build_discrete_action_lookup(self):
        action_map = {}

        has_pitch = False

        for idx, action in enumerate(self._discrete_actions):
            if action[self.env.actions.forward_speed] > 0:
                action_map.setdefault("forward", idx)
            if action[self.env.actions.forward_speed] < 0:
                action_map.setdefault("backward", idx)
            if action[self.env.actions.strafe_speed] > 0:
                action_map.setdefault("strafe_right", idx)
            if action[self.env.actions.strafe_speed] < 0:
                action_map.setdefault("strafe_left", idx)
            if action[self.env.actions.turn_delta] > 0:
                action_map.setdefault("turn_left", idx)
            if action[self.env.actions.turn_delta] < 0:
                action_map.setdefault("turn_right", idx)
            if action[self.env.actions.pitch_delta] > 0:
                action_map.setdefault("pitch_up", idx)
                has_pitch = True
            if action[self.env.actions.pitch_delta] < 0:
                action_map.setdefault("pitch_down", idx)
                has_pitch = True
            if action[self.env.actions.pickup] > 0.5:
                action_map.setdefault("pickup", idx)
            if action[self.env.actions.drop] > 0.5:
                action_map.setdefault("drop", idx)

        # Remember whether the discrete space can represent pitch updates.
        self._has_discrete_pitch = has_pitch

        return action_map

    def _map_controls_to_action(self, action):
        # Check if environment provides a custom control-to-action mapping
        # (used by environments like CameraControl with non-standard actions)
        env_control_map = getattr(self.env.unwrapped, "control_action_map", None)
        if env_control_map and self._pressed_controls:
            for control_name in self._pressed_controls:
                if control_name in env_control_map:
                    return env_control_map[control_name]

        if isinstance(self.env.action_space, spaces.Discrete):
            if not self._discrete_action_map:
                return None

            return self._map_to_discrete_index(action)

        if np.any(action != 0.0):
            return action

        return None

    def _map_to_discrete_index(self, action):
        action = np.array(action, copy=True)
        inputs = []

        self._apply_mouse_rotation_updates(action)

        pitch = action[self.env.actions.pitch_delta]
        if pitch != 0 and not self._has_discrete_pitch:
            self._apply_discrete_pitch_update(pitch)
            action[self.env.actions.pitch_delta] = 0.0

        if action[self.env.actions.pickup] > 0.5 and "pickup" in self._discrete_action_map:
            inputs.append(("pickup", action[self.env.actions.pickup]))
        if action[self.env.actions.drop] > 0.5 and "drop" in self._discrete_action_map:
            inputs.append(("drop", action[self.env.actions.drop]))

        forward = action[self.env.actions.forward_speed]
        if forward > 0 and "forward" in self._discrete_action_map:
            inputs.append(("forward", forward))
        if forward < 0 and "backward" in self._discrete_action_map:
            inputs.append(("backward", forward))

        strafe = action[self.env.actions.strafe_speed]
        if strafe > 0 and "strafe_right" in self._discrete_action_map:
            inputs.append(("strafe_right", strafe))
        if strafe < 0 and "strafe_left" in self._discrete_action_map:
            inputs.append(("strafe_left", strafe))

        turn = action[self.env.actions.turn_delta]
        if turn > 0 and "turn_right" in self._discrete_action_map:
            inputs.append(("turn_right", turn))
        if turn < 0 and "turn_left" in self._discrete_action_map:
            inputs.append(("turn_left", turn))

        pitch = action[self.env.actions.pitch_delta]
        if pitch > 0 and "pitch_up" in self._discrete_action_map:
            inputs.append(("pitch_up", pitch))
        if pitch < 0 and "pitch_down" in self._discrete_action_map:
            inputs.append(("pitch_down", pitch))

        if not inputs:
            return None

        action_name, _ = max(inputs, key=lambda entry: abs(entry[1]))
        return self._discrete_action_map[action_name]

    def _apply_mouse_rotation_updates(self, action):
        def apply_with_deadzone(delta, mouse_delta, update_fn):
            if mouse_delta == 0.0:
                return delta

            # Remove the mouse-derived portion from downstream discrete mapping.
            remaining_delta = delta - mouse_delta

            if abs(mouse_delta) < self.mouse_rotation_deadzone:
                return remaining_delta

            update_fn(mouse_delta)
            return remaining_delta

        turn_idx = self.env.actions.turn_delta
        pitch_idx = self.env.actions.pitch_delta

        action[turn_idx] = apply_with_deadzone(
            action[turn_idx], self._last_mouse_turn_delta, self._apply_fractional_turn_update
        )

        action[pitch_idx] = apply_with_deadzone(
            action[pitch_idx],
            self._last_mouse_pitch_delta,
            self._apply_discrete_pitch_update,
        )

    def _apply_fractional_turn_update(self, turn_delta):
        rand = self.env.np_random if self.env.domain_rand else None
        turn_step = self.env.params.sample(rand, "turn_step")
        yaw_delta = turn_delta * turn_step * math.pi / 180
        self.env.unwrapped._update_agent_orientation(yaw_delta, 0.0)

    def _apply_discrete_pitch_update(self, pitch_delta):
        rand = self.env.np_random if self.env.domain_rand else None
        turn_step = self.env.params.sample(rand, "turn_step")
        self.env.unwrapped._update_agent_orientation(0.0, pitch_delta * turn_step)

    def _describe_action_vector(self, action):
        return (
            f"forward={action[self.env.actions.forward_speed]:.2f} "
            f"strafe={action[self.env.actions.strafe_speed]:.2f} "
            f"turn={action[self.env.actions.turn_delta]:.2f} "
            f"pitch={action[self.env.actions.pitch_delta]:.2f} "
            f"pickup={action[self.env.actions.pickup]:.0f} "
            f"drop={action[self.env.actions.drop]:.0f}"
        )
