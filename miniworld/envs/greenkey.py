from gymnasium import utils

from miniworld.entity import Ball, Box, Key
from miniworld.miniworld import MiniWorldEnv


class GreenKey(MiniWorldEnv, utils.EzPickle):
    """
    ## Description

    A single-room environment where the goal is to reach a green key placed
    randomly in the room. A pink ball and a blue box are also placed in the
    room as distractor objects. The episode ends with a reward when the agent
    reaches the green key.

    ## Action Space

    | Num | Action        |
    |-----|---------------|
    | 0   | turn left     |
    | 1   | turn right    |
    | 2   | move forward  |

    ## Observation Space

    The observation space is an `ndarray` with shape `(obs_height, obs_width, 3)`
    representing an RGB image of what the agent sees.

    ## Rewards

    +(1 - 0.2 * (step_count / max_episode_steps)) when the green key is
    reached and zero otherwise.

    ## Arguments

    ```python
    env = gymnasium.make("MiniWorld-GreenKey-v0")
    ```
    """

    def __init__(self, size=8, max_episode_steps=2000, **kwargs):
        assert size >= 2
        self.size = size

        MiniWorldEnv.__init__(self, max_episode_steps=max_episode_steps, **kwargs)
        utils.EzPickle.__init__(self, size=size, max_episode_steps=max_episode_steps, **kwargs)

        # Allow only movement actions (left/right/forward)
        self.set_discrete_actions()

    def _gen_world(self):
        self.add_rect_room(min_x=0, max_x=self.size, min_z=0, max_z=self.size)

        self.key = self.place_entity(Key(color="green"))
        self.place_entity(Ball(color="red"))
        self.place_entity(Box(color="blue"))
        self.place_agent()

    def step(self, action):
        obs, reward, termination, truncation, info = super().step(action)

        if self.near(self.key):
            reward += self._reward()
            termination = True

        return obs, reward, termination, truncation, info
