import math

import numpy as np
import pyglet
from pyglet.window import key


class ManualControl:
    def __init__(self, env, no_time_limit: bool, domain_rand: bool):
        self.env = env.unwrapped

        self.key_handler = key.KeyStateHandler()
        self.mouse_dx = 0.0
        self.mouse_dy = 0.0
        self.pickup_requested = False
        self.drop_requested = False

        # Mouse sensitivity factors for yaw/pitch updates
        self.turn_sensitivity = 0.0025
        self.pitch_sensitivity = 0.0025

        if no_time_limit:
            self.env.max_episode_steps = math.inf
        if domain_rand:
            self.env.domain_rand = True

    def run(self):
        print("============")
        print("Instructions")
        print("============")
        print(
            "move: arrow keys (mouse to look)\nstrafe: A/D\npickup: P\ndrop: D\nquit: ESC"
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

            if symbol == key.BACKSPACE or symbol == key.SLASH:
                print("RESET")
                self.env.reset()
                self.env.render()
                return

            if symbol == key.ESCAPE:
                self.env.close()

            if symbol == key.PAGEUP or symbol == key.P:
                self.pickup_requested = True
            elif symbol == key.PAGEDOWN or symbol == key.D:
                self.drop_requested = True
            elif symbol == key.ENTER:
                pyglet.app.exit()

        @env.unwrapped.window.event
        def on_key_release(symbol, modifiers):
            pass

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
                float(self.key_handler[key.UP]) - float(self.key_handler[key.DOWN])
            )
            action[self.env.actions.strafe_speed] = (
                float(self.key_handler[key.D]) - float(self.key_handler[key.A])
            )

            turn_input = float(self.key_handler[key.RIGHT]) - float(
                self.key_handler[key.LEFT]
            )
            turn_input += self.mouse_dx * self.turn_sensitivity
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

            action = np.clip(action, -1.0, 1.0)

            if np.any(action != 0.0):
                self.step(action)
            else:
                self.env.render()

        pyglet.clock.schedule_interval(update, 1.0 / 60.0)

        # Enter main event loop
        pyglet.app.run()

        self.env.close()

    def step(self, action):
        if isinstance(action, np.ndarray):
            action_desc = (
                f"forward={action[self.env.actions.forward_speed]:.2f} "
                f"strafe={action[self.env.actions.strafe_speed]:.2f} "
                f"turn={action[self.env.actions.turn_delta]:.2f} "
                f"pitch={action[self.env.actions.pitch_delta]:.2f} "
                f"pickup={action[self.env.actions.pickup]:.0f} "
                f"drop={action[self.env.actions.drop]:.0f}"
            )
        else:
            action_desc = self.env.unwrapped.actions(action).name

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
