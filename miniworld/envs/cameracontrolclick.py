"""Camera Control Click Environment.

A variant of CameraControl where the agent clicks on objects in the scene
to move the camera crosshair toward the clicked position. No button UI is shown.

The agent clicks on a target (e.g., the green key), and the camera pans/tilts
to move the crosshair toward that clicked position.
"""

import math
from typing import Optional, Tuple

import numpy as np
from gymnasium import spaces, utils

from miniworld.entity import Ball, Box, Key
from miniworld.miniworld import MiniWorldEnv


class CameraControlClick(MiniWorldEnv, utils.EzPickle):
    """
    ## Description

    A single-room environment where a camera is mounted on a wall.
    The agent clicks on objects in the scene, and the camera pans/tilts
    to move the crosshair toward the clicked position.

    ## Action Space

    Continuous Box(2) representing normalized (x, y) click coordinates:
    - x: 0.0 (left edge) to 1.0 (right edge)
    - y: 0.0 (top edge) to 1.0 (bottom edge)

    ## Observation Space

    RGB image of what the camera sees, with a red crosshair overlay in the center.

    ## Rewards

    Reward based on how close the crosshair is to the green key after the action.
    Episode ends when the key is centered in the crosshair.
    """

    def __init__(
        self,
        size: int = 8,
        max_episode_steps: int = 500,
        pan_speed: float = 5.0,
        tilt_speed: float = 5.0,
        center_threshold: float = 0.15,
        movement_scale: float = 0.5,
        **kwargs,
    ):
        """Initialize the CameraControlClick environment.

        Args:
            size: Room size (size x size).
            max_episode_steps: Maximum steps before truncation.
            pan_speed: Base degrees to pan per unit offset.
            tilt_speed: Base degrees to tilt per unit offset.
            center_threshold: How close to center the key must be (0-1 normalized).
            movement_scale: Scale factor for click-to-movement conversion.
        """
        assert size >= 2
        self.size = size
        self.pan_speed = pan_speed
        self.tilt_speed = tilt_speed
        self.center_threshold = center_threshold
        self.movement_scale = movement_scale

        # Camera state (will be set in _gen_world)
        self.camera_yaw = 0.0
        self.camera_pitch = 0.0
        self.camera_fov = 60.0
        self.camera_pos = None
        self.camera_wall = None

        # Store the target entity
        self.key = None

        # Force show_controls to False - we don't want button UI
        kwargs['show_controls'] = False

        MiniWorldEnv.__init__(self, max_episode_steps=max_episode_steps, **kwargs)
        utils.EzPickle.__init__(
            self,
            size=size,
            max_episode_steps=max_episode_steps,
            pan_speed=pan_speed,
            tilt_speed=tilt_speed,
            center_threshold=center_threshold,
            movement_scale=movement_scale,
            **kwargs,
        )

        # Continuous action space for (x, y) click coordinates, normalized 0-1
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32,
        )

    def _gen_world(self):
        """Generate the world with room, objects, and wall-mounted camera."""
        # Create the room
        self.add_rect_room(min_x=0, max_x=self.size, min_z=0, max_z=self.size)

        # Place objects in the room
        self.key = self.place_entity(Key(color="green"))
        self.place_entity(Ball(color="red"))
        self.place_entity(Box(color="blue"))

        # Place agent somewhere (won't be used for viewing, but required)
        self.agent.pos = np.array([0.5, 0, 0.5])
        self.agent.dir = 0

        # Set up camera on a random wall
        self._setup_wall_camera()

    def _setup_wall_camera(self):
        """Position the camera on a random wall looking into the room."""
        self.camera_wall = self.np_random.integers(0, 4)

        room_center_x = self.size / 2
        room_center_z = self.size / 2
        camera_height = 1.5

        wall_offset = 0.1

        if self.camera_wall == 0:  # East wall
            self.camera_pos = np.array([self.size - wall_offset, camera_height, room_center_z])
            self.camera_yaw = math.pi
        elif self.camera_wall == 1:  # North wall
            self.camera_pos = np.array([room_center_x, camera_height, wall_offset])
            self.camera_yaw = -math.pi / 2
        elif self.camera_wall == 2:  # West wall
            self.camera_pos = np.array([wall_offset, camera_height, room_center_z])
            self.camera_yaw = 0
        else:  # South wall
            self.camera_pos = np.array([room_center_x, camera_height, self.size - wall_offset])
            self.camera_yaw = math.pi / 2

        self.camera_pitch = 0.0
        self.camera_fov = 60.0
        self._sync_agent_to_camera()

    def _sync_agent_to_camera(self):
        """Sync the agent's camera properties to our wall camera."""
        self.agent.pos = self.camera_pos.copy()
        self.agent.pos[1] = 0
        self.agent.dir = self.camera_yaw
        self.agent.cam_height = self.camera_pos[1]
        self.agent.cam_pitch = self.camera_pitch
        self.agent.cam_fov_y = self.camera_fov
        self.agent.cam_fwd_disp = 0

    def step(self, action):
        """Process click action and move camera toward clicked position.

        Args:
            action: numpy array [x, y] with normalized coordinates (0-1).
                    x=0 is left edge, x=1 is right edge.
                    y=0 is top edge, y=1 is bottom edge.
        """
        self.step_count += 1

        # Extract click coordinates
        click_x, click_y = float(action[0]), float(action[1])

        # Calculate offset from center (center is at 0.5, 0.5)
        # dx > 0 means click is to the right of center
        # dy > 0 means click is below center (in image coords, y increases downward)
        dx = click_x - 0.5
        dy = click_y - 0.5

        # Convert click offset to camera movement
        # Scale by FOV to make movement proportional to what's visible
        fov_scale = self.camera_fov / 60.0  # Normalize to default FOV

        # Pan: if click is to the right (dx > 0), pan right (decrease yaw)
        # The pan_speed is in degrees, dx is -0.5 to 0.5
        pan_amount = -dx * self.pan_speed * self.movement_scale * fov_scale
        self.camera_yaw += pan_amount * math.pi / 180

        # Tilt: if click is above center (dy < 0), tilt up (increase pitch)
        # if click is below center (dy > 0), tilt down (decrease pitch)
        tilt_amount = -dy * self.tilt_speed * self.movement_scale * fov_scale
        self.camera_pitch = np.clip(self.camera_pitch + tilt_amount, -89.0, 89.0)

        # Sync agent to camera for rendering
        self._sync_agent_to_camera()

        # Render observation
        obs = self.render_obs()

        # Check if we've reached max steps
        if self.step_count >= self.max_episode_steps:
            return obs, 0, False, True, self._get_info()

        # Check goal condition
        reward = 0
        termination = False

        is_centered, distance_from_center = self._check_key_centered()
        if is_centered:
            reward = self._reward()
            termination = True

        return obs, reward, termination, False, self._get_info()

    def _get_info(self):
        """Get info dict with camera state."""
        is_centered, distance = self._check_key_centered()
        return {
            "camera_yaw": self.camera_yaw,
            "camera_pitch": self.camera_pitch,
            "camera_fov": self.camera_fov,
            "camera_wall": self.camera_wall,
            "key_centered": is_centered,
            "distance_from_center": distance,
        }

    def _check_key_centered(self) -> Tuple[bool, float]:
        """Check if the green key is centered in the camera view."""
        if self.key is None:
            return False, 1.0

        key_pos = self.key.pos.copy()
        key_pos[1] = self.key.height / 2

        cam_pos = self.camera_pos.copy()
        to_key = key_pos - cam_pos
        distance_to_key = np.linalg.norm(to_key)

        if distance_to_key < 0.01:
            return True, 0.0

        to_key_normalized = to_key / distance_to_key

        pitch_rad = self.camera_pitch * math.pi / 180
        cam_dir = np.array([
            math.cos(pitch_rad) * math.cos(self.camera_yaw),
            math.sin(pitch_rad),
            -math.cos(pitch_rad) * math.sin(self.camera_yaw),
        ])

        dot_product = np.clip(np.dot(cam_dir, to_key_normalized), -1.0, 1.0)
        angle_to_key = math.acos(dot_product)

        half_fov_rad = (self.camera_fov / 2) * math.pi / 180
        normalized_distance = angle_to_key / half_fov_rad

        is_centered = normalized_distance <= self.center_threshold

        return is_centered, min(normalized_distance, 1.0)

    def render_obs(self, frame_buffer=None):
        """Render observation with crosshair overlay."""
        obs = super().render_obs(frame_buffer)
        obs = self._draw_crosshair(obs)
        return obs

    def _draw_crosshair(self, img: np.ndarray) -> np.ndarray:
        """Draw a red crosshair in the center of the image."""
        import cv2

        img = img.copy()
        h, w = img.shape[:2]
        center_x, center_y = w // 2, h // 2

        # Crosshair parameters - red and bold
        color = (255, 0, 0)  # Red in RGB
        thickness = 2
        gap = 4
        length = 20

        # Draw horizontal lines
        cv2.line(img, (center_x - length - gap, center_y),
                 (center_x - gap, center_y), color, thickness)
        cv2.line(img, (center_x + gap, center_y),
                 (center_x + length + gap, center_y), color, thickness)

        # Draw vertical lines
        cv2.line(img, (center_x, center_y - length - gap),
                 (center_x, center_y - gap), color, thickness)
        cv2.line(img, (center_x, center_y + gap),
                 (center_x, center_y + length + gap), color, thickness)

        # Draw small center dot
        cv2.circle(img, (center_x, center_y), 3, color, -1)

        return img
