# Tutorial on Creating Environments

In this tutorial, we will go through the process of creating a new environment.

## Boilerplate Code

```python
def __init__(self, size=10, **kwargs):
    # Size of environment
    self.size = size

    super().__init__(self, **kwargs)

    # Continuous actions for forward/strafe speeds, yaw/pitch deltas, pickup, drop
    # Each component is expected to be in the range [-1, 1].
    self.action_space = spaces.Box(
        low=np.array([-1.0, -1.0, -1.0, -1.0, 0.0, 0.0], dtype=np.float32),
        high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        dtype=np.float32,
    )
```

First, we need to create a class that inherits from `MiniWorldEnv`; we call our class `SimpleEnv`. The default MiniWorld action space is a 6-D vector: forward speed, strafe speed, yaw delta, pitch delta, pickup flag, and drop flag. Movement and orientation values are normalized to `[-1, 1]`, and pitch is clamped internally to `[-89, 89]` degrees to keep the camera stable.

## Generate the walls

To generate the walls, we override the function `_gen_world`.

```python
def _gen_world(self):
    self.add_rect_room(min_x=0, max_x=self.size, min_z=0, max_z=self.size)
    self.place_agent()
```

The function `_gen_world` takes the arguments: `min_x`, `max_x`, `min_z`, `max_z`. Note that instead of using the X-Y plane, we use the X-Z plane for movement. After doing this, the environment should look like this:

```{figure} ../../images/tutorial_imgs/first_step.png
:alt: env after first step
:width: 500px
```

### Place Goal

To place a goal in the environment, we use the function

```python
self.box = self.place_entity(Box(color=COLOR_NAMES[0]), pos=np.array([4.5, 0.5, 4.5]), dir=0.0)
```

which places the goal in the middle. Now the environment should look like this:

```{figure} ../../images/tutorial_imgs/second_step.png
:alt: env after second step
:width: 500px
```

### Add reward

To add a reward when the agent gets close to the box, we can do the following:

```python
def step(self, action):
    obs, reward, termination, truncation, info = super().step(action)

    if self.near(self.box):
        reward += self._reward()
        termination = True

    return obs, reward, termination, truncation, info
```
