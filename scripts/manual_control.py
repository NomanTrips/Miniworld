#!/usr/bin/env python3

"""
This script allows you to manually control the simulator
using the keyboard arrows.
"""

import argparse

import gymnasium as gym

import miniworld
from miniworld.manual_control import ManualControl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", default="MiniWorld-Hallway-v0")
    parser.add_argument(
        "--domain-rand", action="store_true", help="enable domain randomization"
    )
    parser.add_argument(
        "--no-time-limit", action="store_true", help="ignore time step limits"
    )
    parser.add_argument(
        "--top_view",
        action="store_true",
        help="show the top view instead of the agent view",
    )
    parser.add_argument(
        "--mouse-sensitivity",
        type=float,
        default=0.0025,
        help="mouse sensitivity for yaw and pitch updates",
    )
    parser.add_argument(
        "--fullscreen",
        action="store_true",
        help="start the viewer in fullscreen mode",
    )
    parser.add_argument(
        "--window-size",
        type=str,
        default=None,
        help=(
            "set the initial window size as WIDTHxHEIGHT (e.g., 1920x1080); "
            "ignored when --fullscreen is set"
        ),
    )
    parser.add_argument(
        "--hide-hud",
        action="store_true",
        help="run the viewer without the HUD overlay (cleaner recordings)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="Center and zoom on the target.",
        help="short task description to record in tasks.parquet",
    )
    parser.add_argument(
        "--obs-width",
        type=int,
        default=512,
        help=(
            "horizontal resolution for recorded observations; divisible by 16 to "
            "avoid codec padding (e.g., 512, 1280)"
        ),
    )
    parser.add_argument(
        "--obs-height",
        type=int,
        default=512,
        help=(
            "vertical resolution for recorded observations; divisible by 16 to "
            "avoid codec padding (e.g., 512, 704)"
        ),
    )
    args = parser.parse_args()
    view_mode = "top" if args.top_view else "agent"

    env = gym.make(
        args.env_name,
        view=view_mode,
        render_mode="human",
        show_hud=not args.hide_hud,
        obs_width=args.obs_width,
        obs_height=args.obs_height,
    )
    miniworld_version = miniworld.__version__

    print(f"Miniworld v{miniworld_version}, Env: {args.env_name}")

    manual_control = ManualControl(
        env,
        args.no_time_limit,
        args.domain_rand,
        mouse_sensitivity=args.mouse_sensitivity,
        fullscreen=args.fullscreen,
        window_size=args.window_size,
        task_description=args.task,
    )
    manual_control.run()


if __name__ == "__main__":
    main()
