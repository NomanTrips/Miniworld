import math
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import utils
from gymnasium.core import ObsType
from gymnasium.spaces import Dict, Discrete

from miniworld.entity import COLOR_NAMES, Box, Key, MeshEnt, TextFrame
from miniworld.miniworld import MiniWorldEnv
from miniworld.params import DEFAULT_PARAMS


class BigKey(Key):
    """A key with a bigger size for better visibility."""

    def __init__(self, color, size=0.6):
        assert color in COLOR_NAMES
        MeshEnt.__init__(self, mesh_name=f"key_{color}", height=size, static=False)


class Sign(MiniWorldEnv, utils.EzPickle):
    """
    ## Description

    Sign environment from https://arxiv.org/abs/2008.02790.

    If you use this environment, please cite the above paper (Liu et al., 2020).

    Small U-shaped maze with 6 objects: (blue, red, green) x (key, box).
    A sign on the wall displays a random color ("BLUE", "GREEN", or "RED") at the
    start of each episode. The agent's goal is to navigate to any object matching
    that color.

    In addition to the normal state, accessible under state["obs"], the state also
    includes a goal under state["goal"] that specifies box or key.

    The episode ends when any object is touched.

    Includes an action to end the episode.

    ## Action Space

    | Num | Action                      |
    |-----|-----------------------------|
    | 0   | turn left                   |
    | 1   | turn right                  |
    | 2   | move forward                |
    | 3   | end episode                 |

    ## Observation Space

    The observation space is an `ndarray` with shape `(obs_height, obs_width, 3)`
    representing an RGB image of what the agents see.

    ## Rewards

    +1 for touching any object whose color matches the sign
    Touching objects with non-matching colors has no effect

    ## Arguments

    * `size`:  size of the square room.
    * `max_episode_steps`: number of steps before the episode ends.
    * `goal`: specifies box (0) or key (1) for the observation space goal field.

    ```python
    env = gymnasium.make("Miniworld-Sign-v0", size=10, max_episode_steps=20)
    ```
    """

    def __init__(self, size=10, max_episode_steps=20, color_index=0, goal=0, **kwargs):
        if color_index not in [0, 1, 2]:
            raise ValueError("Only supported values for color_index are 0, 1, 2.")

        if goal not in [0, 1]:
            raise ValueError("Only supported values for goal are 0, 1.")

        params = DEFAULT_PARAMS.no_random()
        params.set("forward_step", 0.7)  # larger steps
        params.set("turn_step", 45)  # 45 degree rotation

        self._size = size
        self._goal = goal
        self._color_index = color_index

        MiniWorldEnv.__init__(
            self,
            params=params,
            max_episode_steps=max_episode_steps,
            domain_rand=False,
            **kwargs,
        )
        utils.EzPickle.__init__(
            self, size, max_episode_steps, color_index, goal, **kwargs
        )

        self.observation_space = Dict(obs=self.observation_space, goal=Discrete(2))

        # Allow for left / right / forward + custom end episode
        self._end_action_index = 3
        self.set_discrete_actions(
            [
                self._action_from_components(turn=-1.0),
                self._action_from_components(turn=1.0),
                self._action_from_components(forward=1.0),
                self._action_from_components(),
            ]
        )

    def set_color_index(self, color_index):
        self._color_index = color_index

    def _gen_world(self):
        # Randomize the color shown on the sign for each episode
        self._color_index = self.np_random.integers(0, 3)

        gap_size = 0.25
        top_room = self.add_rect_room(
            min_x=0, max_x=self._size, min_z=0, max_z=self._size * 0.65
        )
        left_room = self.add_rect_room(
            min_x=0,
            max_x=self._size * 3 / 5,
            min_z=self._size * 0.65 + gap_size,
            max_z=self._size * 1.3,
        )
        right_room = self.add_rect_room(
            min_x=self._size * 3 / 5,
            max_x=self._size,
            min_z=self._size * 0.65 + gap_size,
            max_z=self._size * 1.3,
        )
        self.connect_rooms(top_room, left_room, min_x=0, max_x=self._size * 3 / 5)
        self.connect_rooms(
            left_room,
            right_room,
            min_z=self._size * 0.65 + gap_size,
            max_z=self._size * 1.3,
        )

        self._objects = [
            # Boxes
            (
                self.place_entity(Box(color="blue"), pos=(1, 0, 1)),
                self.place_entity(Box(color="red"), pos=(9, 0, 1)),
                self.place_entity(Box(color="green"), pos=(9, 0, 5)),
            ),
            # Keys
            (
                self.place_entity(BigKey(color="blue"), pos=(5, 0, 1)),
                self.place_entity(BigKey(color="red"), pos=(1, 0, 5)),
                self.place_entity(BigKey(color="green"), pos=(1, 0, 9)),
            ),
        ]

        text = ["BLUE", "RED", "GREEN"][self._color_index]
        sign = TextFrame(
            pos=[self._size, 1.35, self._size + gap_size],
            dir=math.pi,
            str=text,
            height=1,
        )
        self.entities.append(sign)
        self.place_agent(min_x=4, max_x=5, min_z=4, max_z=6)

    def step(self, action):
        end_requested = np.isscalar(action) and int(action) == self._end_action_index

        obs, reward, termination, truncation, info = super().step(action)

        if end_requested:
            termination = True

        for obj_index, object_pair in enumerate(self._objects):
            for color_index, obj in enumerate(object_pair):
                if self.near(obj) and color_index == self._color_index:
                    # Only end episode when touching an object with the correct color
                    termination = True
                    reward = 1.0

        state = {"obs": obs, "goal": self._goal}
        return state, reward, termination, truncation, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[ObsType, dict]:
        obs, info = super().reset(seed=seed, options=options)
        return {"obs": obs, "goal": self._goal}, info
