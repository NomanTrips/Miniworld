<p align="center">
    <img src="https://raw.githubusercontent.com/Farama-Foundation/Miniworld/master/miniworld-text.png" width="500px"/>
</p>

**Aug 11, 2025: This project has been deprecated due to a lack of wide spread community use, and is no longer planned to receive any additional updates or support.**

[![Build Status](https://travis-ci.org/maximecb/gym-miniworld.svg?branch=master)](https://travis-ci.org/maximecb/gym-miniworld)

Contents:
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Environments](https://miniworld.farama.org/content/env_list/)
- [Design and Customization](https://miniworld.farama.org/content/design/)
- [Troubleshooting](https://miniworld.farama.org/content/troubleshooting/)

## Introduction

MiniWorld is a minimalistic 3D interior environment simulator for reinforcement
learning &amp; robotics research. It can be used to simulate environments with
rooms, doors, hallways and various objects (eg: office and home environments, mazes).
MiniWorld can be seen as a simpler alternative to VizDoom or DMLab. It is written
100% in Python and designed to be easily modified or extended by students.

<p align="center">
    <img src="images/maze_top_view.jpg" width=260 alt="Figure of Maze environment from top view">
    <img src="images/sidewalk_0.jpg" width=260 alt="Figure of Sidewalk environment">
    <img src="images/collecthealth_0.jpg" width=260 alt="Figure of Collect Health environment">
</p>

Features:
- Few dependencies, less likely to break, easy to install
- Easy to create your own levels, or modify existing ones
- Good performance, high frame rate, support for multiple processes
- Lightweight, small download, low memory requirements
- Provided under a permissive MIT license
- Comes with a variety of free 3D models and textures
- Fully observable [top-down/overhead view](images/maze_top_view.jpg) available
- [Domain randomization](https://blog.openai.com/generalizing-from-simulation/) support, for sim-to-real transfer
- Ability to [display alphanumeric strings](images/textframe.jpg) on walls
- Ability to produce depth maps matching camera images (RGB-D)

Limitations:
- Graphics are basic, nowhere near photorealism
- Physics are very basic, not sufficient for robot arms or manipulation

List of publications & submissions using MiniWorld (please open a pull request to add missing entries):
- [Towards real-world navigation with deep differentiable planners](https://arxiv.org/abs/2108.05713) (VGG, Oxford, CVPR 2022)
- [Decoupling Exploration and Exploitation for Meta-Reinforcement Learning without Sacrifices](https://arxiv.org/abs/2008.02790) (Stanford University, ICML 2021)
- [Rank the Episodes: A Simple Approach for Exploration in Procedurally-Generated Environments](https://openreview.net/forum?id=MtEE0CktZht) (Texas A&M University, Kuai Inc., ICLR 2021)
- [DeepAveragers: Offline Reinforcement Learning by Solving Derived Non-Parametric MDPs](https://arxiv.org/abs/2010.08891) (NeurIPS Offline RL Workshop, Oct 2020)
- [Pre-trained Word Embeddings for Goal-conditional Transfer Learning in Reinforcement Learning](https://arxiv.org/abs/2007.05196) (University of Antwerp, Jul 2020, ICML 2020 LaReL Workshop)
- [Temporal Abstraction with Interest Functions](https://arxiv.org/abs/2001.00271) (Mila, Feb 2020, AAAI 2020)
- [Addressing Sample Complexity in Visual Tasks Using Hindsight Experience Replay and Hallucinatory GANs](https://openreview.net/forum?id=H1xSXdV0i4) (Offworld Inc, Georgia Tech, UC Berkeley, ICML 2019 Workshop RL4RealLife)
- [Avoidance Learning Using Observational Reinforcement Learning](https://arxiv.org/abs/1909.11228) (Mila, McGill, Sept 2019)
- [Visual Hindsight Experience Replay](https://arxiv.org/pdf/1901.11529.pdf) (Georgia Tech, UC Berkeley, Jan 2019)

This simulator was created as part of work done at [Mila](https://mila.quebec/).

## Installation

Requirements:
- Python 3.7+
- Gymnasium
- NumPy
- Pyglet (OpenGL 3D graphics)
- GPU for 3D graphics acceleration (optional)

You can install it from `PyPI` using:

```console
python3 -m pip install miniworld
```

You can also install from source:

```console
git clone https://github.com/Farama-Foundation/Miniworld.git
cd Miniworld
python3 -m pip install -e .
```

If you run into any problems, please take a look at the [troubleshooting guide](docs/content/troubleshooting.md).

## Usage

There is a simple UI application which allows you to control the simulation or real robot manually.
The `manual_control.py` application will launch the Gym environment, display camera images and send actions
(keyboard commands) back to the simulator or robot. The `--env-name` argument specifies which environment to load.
See the list of [available environments](docs/environments.md) for more information.

```
./manual_control.py --env-name MiniWorld-Hallway-v0

# Display an overhead view of the environment
./manual_control.py --env-name MiniWorld-Hallway-v0 --top_view

# Launch in fullscreen or target a specific window size
./manual_control.py --env-name MiniWorld-Hallway-v0 --fullscreen
./manual_control.py --env-name MiniWorld-Hallway-v0 --window-size 1920x1080

# Hide the HUD overlay for clean recordings
./manual_control.py --env-name MiniWorld-Hallway-v0 --hide-hud

# Disable the clickable on-screen controls (enabled by default)
./manual_control.py --env-name MiniWorld-Hallway-v0 --no-show-controls
```

### Action space and controls

All MiniWorld environments expose a 6-D continuous action vector `[forward_speed, strafe_speed, yaw_delta, pitch_delta, pickup, drop]`.
Values are expected to be in the range `[-1, 1]` and are scaled internally using the environment's motion parameters.
The camera pitch is clamped to `[-89, 89]` degrees, and the default manual controller applies mouse input with a sensitivity of `0.0025` per pixel to both yaw and pitch updates before the environment scaling is applied.

```python
import gymnasium as gym
import numpy as np

env = gym.make("MiniWorld-Hallway-v0", render_mode="rgb_array")
action = np.zeros(len(env.unwrapped.actions), dtype=np.float32)
action[env.unwrapped.actions.forward_speed] = 0.75  # move forward
action[env.unwrapped.actions.turn_delta] = 0.25     # gentle yaw change
action[env.unwrapped.actions.pitch_delta] = -0.1    # look slightly down

obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)

env.close()
```

The snippet above is the minimal end-to-end example for continuous movement and mouse-look: it creates an environment, issues a single normalized action that moves forward while turning and pitching the camera, and then closes the environment. You can plug the same action structure into rollout loops or adapt it for `scripts/manual_control.py`, which binds the keyboard arrows/WASD for movement and uses the mouse for yaw/pitch while respecting the same sensitivity and pitch limits.

The manual controller also draws a HUD overlay with clickable buttons for moving, strafing, turning, and pitching the camera if `--show-controls` is left enabled (default). Each button mirrors the same WASD/arrow and mouse-look controls, so you can drive the agent entirely with the overlay when recording demos or using a trackpad. Disable the overlay with `--no-show-controls` if you prefer a keyboard-only experience. To regenerate the illustration locally without storing the binary in version control, run `python scripts/generate_manual_control_overlay.py` (requires Pillow) to write `images/manual_control_overlay.png`.

![HUD movement and look buttons](images/manual_control_overlay.png)

While running `scripts/manual_control.py` you can toggle fullscreen at any time with **F11**; the viewer will remember the previous windowed size and keep the mouse cursor captured in both modes.

There is also a script to run automated tests (`run_tests.py`) and a script to gather performance metrics (`benchmark.py`).

### Offscreen Rendering (Clusters and Colab)

When running MiniWorld on a cluster or in a Colab environment, you need to render to an offscreen display. You can
run `gym-miniworld` offscreen by setting the environment variable `PYOPENGL_PLATFORM` to `egl` before running MiniWorld, e.g.

```
PYOPENGL_PLATFORM=egl python3 your_script.py
```

Alternatively, if this doesn't work, you can also try running MiniWorld with `xvfb`, e.g.

```
xvfb-run -a -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset" python3 your_script.py
```

# Citation

To cite this project please use:

```bibtex
@article{MinigridMiniworld23,
  author       = {Maxime Chevalier-Boisvert and Bolun Dai and Mark Towers and Rodrigo de Lazcano and Lucas Willems and Salem Lahlou and Suman Pal and Pablo Samuel Castro and Jordan Terry},
  title        = {Minigrid \& Miniworld: Modular \& Customizable Reinforcement Learning Environments for Goal-Oriented Tasks},
  journal      = {CoRR},
  volume       = {abs/2306.13831},
  year         = {2023},
}
```
