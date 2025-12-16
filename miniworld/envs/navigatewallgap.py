import math

import numpy as np
from gymnasium import utils

from miniworld.entity import Box, MeshEnt
from miniworld.miniworld import MiniWorldEnv


class NavigateWallGap(MiniWorldEnv, utils.EzPickle):
    """
    ## Description

    Outside environment with two rooms connected by a gap in a wall. The
    goal is simply to move through the gap between the rooms.

    ## Action Space

    | Num | Action                      |
    |-----|-----------------------------|
    | 0   | turn left                   |
    | 1   | turn right                  |
    | 2   | move forward                |

    ## Observation Space

    The observation space is an `ndarray` with shape `(obs_height, obs_width, 3)`
    representing an RGB image of what the agents see.

    ## Rewards

    +1 when the agent reaches the room on the other side of the wall gap

    ## Arguments

    ```python
    env = gymnasium.make("MiniWorld-NavigateWallGap-v0")
    ```
    """

    def __init__(self, **kwargs):
        MiniWorldEnv.__init__(self, max_episode_steps=300, **kwargs)
        utils.EzPickle.__init__(self, **kwargs)

        # Allow only the movement actions
        self.set_discrete_actions()

    def _gen_world(self):
        # Top
        self.top_room = self.add_rect_room(
            min_x=-7,
            max_x=7,
            min_z=0.5,
            max_z=8,
            wall_tex="brick_wall",
            floor_tex="asphalt",
            no_ceiling=True,
        )
        # Bottom
        self.bottom_room = self.add_rect_room(
            min_x=-7,
            max_x=7,
            min_z=-8,
            max_z=-0.5,
            wall_tex="brick_wall",
            floor_tex="asphalt",
            no_ceiling=True,
        )
        self.connect_rooms(self.top_room, self.bottom_room, min_x=-1.5, max_x=1.5)

        # Keep the red box for a familiar visual cue
        self.box = self.place_entity(Box(color="red"), room=self.bottom_room)

        # Decorative building in the background
        self.place_entity(
            MeshEnt(mesh_name="building", height=30),
            pos=np.array([30, 0, 30]),
            dir=-math.pi,
        )

        self.place_agent(room=self.top_room)

        self.passed_gap = False

    def _agent_in_bottom_room(self):
        pos = self.agent.pos
        return (
            self.bottom_room.min_x <= pos[0] <= self.bottom_room.max_x
            and self.bottom_room.min_z <= pos[2] <= self.bottom_room.max_z
        )

    def step(self, action):
        obs, reward, termination, truncation, info = super().step(action)

        if not self.passed_gap and self._agent_in_bottom_room():
            reward = 1.0
            termination = True
            self.passed_gap = True

        return obs, reward, termination, truncation, info
