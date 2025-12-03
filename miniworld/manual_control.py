import math

import numpy as np
import pyglet
from gymnasium import spaces
from pyglet.window import key


class ManualControl:
    def __init__(
        self,
        env,
        no_time_limit: bool,
        domain_rand: bool,
        mouse_sensitivity: float = 0.0025,
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

        # Mouse sensitivity factors for yaw/pitch updates
        self.turn_sensitivity = mouse_sensitivity
        self.pitch_sensitivity = mouse_sensitivity

        if no_time_limit:
            self.env.max_episode_steps = math.inf
        if domain_rand:
            self.env.domain_rand = True

    def run(self):
        print("============")
        print("Instructions")
        print("============")
        print(
            "move: arrow keys (mouse to look)\nstrafe: A/D\npickup: P\ndrop: B\nquit: ESC"
        )
        print("============")

        self.env.reset()

        # Create the display window
        self.env.render()

        env = self.env

        window = env.unwrapped.window
        window.push_handlers(self.key_handler)
        window.set_exclusive_mouse(True)

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
                self.env.close()

            if symbol == key.PAGEUP or symbol == key.P:
                self.pickup_requested = True
            elif symbol == key.PAGEDOWN or symbol == key.B:
                self.drop_requested = True
            elif symbol == key.ENTER:
                pyglet.app.exit()

        @env.unwrapped.window.event
        def on_key_release(symbol, modifiers):
            # Keep KeyStateHandler in sync with released keys.
            self.key_handler.on_key_release(symbol, modifiers)

        @env.unwrapped.window.event
        def on_mouse_motion(x, y, dx, dy):
            self.mouse_dx += dx
            self.mouse_dy += dy

        @env.unwrapped.window.event
        def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
            self.mouse_dx += dx
            self.mouse_dy += dy

        @env.unwrapped.window.event
        def on_draw():
            self.env.render()

        @env.unwrapped.window.event
        def on_close():
            pyglet.app.exit()

        def update(dt):
            action = np.zeros(len(self.env.actions), dtype=np.float32)

            # Keyboard state-based movement
            action[self.env.actions.forward_speed] = (
                float(self.key_handler[key.UP])
                + float(self.key_handler[key.W])
                - float(self.key_handler[key.DOWN])
                - float(self.key_handler[key.S])
            )
            action[self.env.actions.strafe_speed] = (
                float(self.key_handler[key.D]) - float(self.key_handler[key.A])
            )

            turn_input = float(self.key_handler[key.RIGHT]) - float(
                self.key_handler[key.LEFT]
            )
            turn_input += -self.mouse_dx * self.turn_sensitivity
            pitch_input = self.mouse_dy * self.pitch_sensitivity

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

        self.env.close()

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

        if reward > 0:
            print(f"reward={reward:.2f}")

        if termination or truncation:
            print("done!")
            self.env.reset()

        self.env.render()

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
                action_map.setdefault("turn_right", idx)
            if action[self.env.actions.turn_delta] < 0:
                action_map.setdefault("turn_left", idx)
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
        if isinstance(self.env.action_space, spaces.Discrete):
            if not self._discrete_action_map:
                return None

            return self._map_to_discrete_index(action)

        if np.any(action != 0.0):
            return action

        return None

    def _map_to_discrete_index(self, action):
        inputs = []

        pitch = action[self.env.actions.pitch_delta]
        if pitch != 0 and not self._has_discrete_pitch:
            self._apply_discrete_pitch_update(pitch)
            action = np.array(action, copy=True)
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
