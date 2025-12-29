"""Camera Control Environment.

A single-room environment where the goal is to control a wall-mounted camera
to find and center a crosshair on a green key. The camera can pan left/right,
tilt up/down, and zoom in/out.

The episode ends with a reward when the crosshair is centered on the green key
and the zoom level is appropriate.
"""

import math
from typing import Optional

import numpy as np
from gymnasium import spaces, utils

from miniworld.entity import Ball, Box, Key
from miniworld.miniworld import MiniWorldEnv


class CameraControl(MiniWorldEnv, utils.EzPickle):
    """
    ## Description

    A single-room environment where a camera is mounted on a randomly chosen
    wall. The agent controls the camera using pan, tilt, and zoom to find
    and center the crosshair on a green key. A red ball and blue box serve
    as distractor objects.

    ## Action Space

    | Num | Action      |
    |-----|-------------|
    | 0   | pan left    |
    | 1   | pan right   |
    | 2   | tilt up     |
    | 3   | tilt down   |
    | 4   | zoom in     |
    | 5   | zoom out    |

    ## Observation Space

    The observation space is an `ndarray` with shape `(obs_height, obs_width, 3)`
    representing an RGB image of what the camera sees, with a crosshair overlay
    in the center.

    ## Rewards

    +(1 - 0.2 * (step_count / max_episode_steps)) when the green key is
    centered in the crosshair and appropriately zoomed, zero otherwise.
    """

    def __init__(
        self,
        size: int = 8,
        max_episode_steps: int = 500,
        pan_speed: float = 5.0,
        tilt_speed: float = 5.0,
        zoom_speed: float = 2.0,
        center_threshold: float = 0.15,
        min_fov: float = 20.0,
        max_fov: float = 90.0,
        **kwargs,
    ):
        """Initialize the CameraControl environment.

        Args:
            size: Room size (size x size).
            max_episode_steps: Maximum steps before truncation.
            pan_speed: Degrees to pan per action.
            tilt_speed: Degrees to tilt per action.
            zoom_speed: FOV change per zoom action.
            center_threshold: How close to center the key must be (0-1 normalized).
            min_fov: Minimum field of view (maximum zoom in).
            max_fov: Maximum field of view (maximum zoom out).
        """
        assert size >= 2
        self.size = size
        self.pan_speed = pan_speed
        self.tilt_speed = tilt_speed
        self.zoom_speed = zoom_speed
        self.center_threshold = center_threshold
        self.min_fov = min_fov
        self.max_fov = max_fov

        # Camera state (will be set in _gen_world)
        self.camera_yaw = 0.0  # Pan angle in radians
        self.camera_pitch = 0.0  # Tilt angle in degrees
        self.camera_fov = 60.0  # Current field of view
        self.camera_pos = None  # Camera position on wall
        self.camera_wall = None  # Which wall the camera is on

        # Store the target entity
        self.key = None

        # Custom control button configuration
        self._camera_control_buttons = [
            ("pan_left", "Pan Left"),
            ("pan_right", "Pan Right"),
            ("tilt_up", "Tilt Up"),
            ("tilt_down", "Tilt Down"),
            ("zoom_in", "Zoom In"),
            ("zoom_out", "Zoom Out"),
        ]

        MiniWorldEnv.__init__(self, max_episode_steps=max_episode_steps, **kwargs)
        utils.EzPickle.__init__(
            self,
            size=size,
            max_episode_steps=max_episode_steps,
            pan_speed=pan_speed,
            tilt_speed=tilt_speed,
            zoom_speed=zoom_speed,
            center_threshold=center_threshold,
            min_fov=min_fov,
            max_fov=max_fov,
            **kwargs,
        )

        # Set up discrete action space for camera controls
        self.action_space = spaces.Discrete(6)

        # Provide direct mapping from control button names to action indices
        # This allows manual_control.py to handle our custom buttons
        self.control_action_map = {
            "pan_left": 0,
            "pan_right": 1,
            "tilt_up": 2,
            "tilt_down": 3,
            "zoom_in": 4,
            "zoom_out": 5,
        }

    def _gen_world(self):
        """Generate the world with room, objects, and wall-mounted camera."""
        # Create the room
        self.add_rect_room(min_x=0, max_x=self.size, min_z=0, max_z=self.size)

        # Place objects in the room
        self.key = self.place_entity(Key(color="green"))
        self.place_entity(Ball(color="red"))
        self.place_entity(Box(color="blue"))

        # Place agent somewhere (won't be used for viewing, but required)
        # Put the agent in a corner out of the way
        self.agent.pos = np.array([0.5, 0, 0.5])
        self.agent.dir = 0

        # Set up camera on a random wall
        self._setup_wall_camera()

    def _setup_wall_camera(self):
        """Position the camera on a random wall looking into the room."""
        # Choose a random wall: 0=East, 1=North, 2=West, 3=South
        self.camera_wall = self.np_random.integers(0, 4)

        room_center_x = self.size / 2
        room_center_z = self.size / 2
        camera_height = 1.5  # Height of camera on wall

        # Wall offset from edge
        wall_offset = 0.1

        if self.camera_wall == 0:  # East wall (max_x)
            self.camera_pos = np.array([self.size - wall_offset, camera_height, room_center_z])
            self.camera_yaw = math.pi  # Looking west (into room)
        elif self.camera_wall == 1:  # North wall (min_z)
            self.camera_pos = np.array([room_center_x, camera_height, wall_offset])
            self.camera_yaw = -math.pi / 2  # Looking south (into room)
        elif self.camera_wall == 2:  # West wall (min_x)
            self.camera_pos = np.array([wall_offset, camera_height, room_center_z])
            self.camera_yaw = 0  # Looking east (into room)
        else:  # South wall (max_z)
            self.camera_pos = np.array([room_center_x, camera_height, self.size - wall_offset])
            self.camera_yaw = math.pi / 2  # Looking north (into room)

        # Reset camera orientation
        self.camera_pitch = 0.0
        self.camera_fov = 60.0

        # Update agent to use camera position/orientation for rendering
        self._sync_agent_to_camera()

    def _sync_agent_to_camera(self):
        """Sync the agent's camera properties to our wall camera."""
        self.agent.pos = self.camera_pos.copy()
        self.agent.pos[1] = 0  # Agent pos is at floor level
        self.agent.dir = self.camera_yaw
        self.agent.cam_height = self.camera_pos[1]
        self.agent.cam_pitch = self.camera_pitch
        self.agent.cam_fov_y = self.camera_fov

    def step(self, action):
        """Process camera control action and check for goal completion."""
        self.step_count += 1

        # Process discrete camera control actions
        if action == 0:  # Pan left
            self.camera_yaw += self.pan_speed * math.pi / 180
        elif action == 1:  # Pan right
            self.camera_yaw -= self.pan_speed * math.pi / 180
        elif action == 2:  # Tilt up
            self.camera_pitch = min(89.0, self.camera_pitch + self.tilt_speed)
        elif action == 3:  # Tilt down
            self.camera_pitch = max(-89.0, self.camera_pitch - self.tilt_speed)
        elif action == 4:  # Zoom in (decrease FOV)
            self.camera_fov = max(self.min_fov, self.camera_fov - self.zoom_speed)
        elif action == 5:  # Zoom out (increase FOV)
            self.camera_fov = min(self.max_fov, self.camera_fov + self.zoom_speed)

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

    def _check_key_centered(self) -> tuple[bool, float]:
        """Check if the green key is centered in the camera view.

        Returns:
            (is_centered, distance_from_center) where distance is normalized 0-1.
        """
        if self.key is None:
            return False, 1.0

        # Get key position in world coordinates
        key_pos = self.key.pos.copy()
        key_pos[1] = self.key.height / 2  # Use center of key

        # Get camera position and direction
        cam_pos = self.camera_pos.copy()

        # Calculate direction from camera to key
        to_key = key_pos - cam_pos
        distance_to_key = np.linalg.norm(to_key)

        if distance_to_key < 0.01:
            return True, 0.0

        to_key_normalized = to_key / distance_to_key

        # Calculate camera direction vector
        pitch_rad = self.camera_pitch * math.pi / 180
        cam_dir = np.array([
            math.cos(pitch_rad) * math.cos(self.camera_yaw),
            math.sin(pitch_rad),
            -math.cos(pitch_rad) * math.sin(self.camera_yaw),
        ])

        # Calculate angle between camera direction and direction to key
        dot_product = np.clip(np.dot(cam_dir, to_key_normalized), -1.0, 1.0)
        angle_to_key = math.acos(dot_product)

        # Normalize by half the FOV to get a 0-1 distance from center
        half_fov_rad = (self.camera_fov / 2) * math.pi / 180
        normalized_distance = angle_to_key / half_fov_rad

        # Key is centered if within threshold
        is_centered = normalized_distance <= self.center_threshold

        return is_centered, min(normalized_distance, 1.0)

    def render_obs(self, frame_buffer=None):
        """Render observation with crosshair overlay."""
        # Get the base observation
        obs = super().render_obs(frame_buffer)

        # Add crosshair to observation
        obs = self._draw_crosshair(obs)

        return obs

    def _draw_crosshair(self, img: np.ndarray) -> np.ndarray:
        """Draw a crosshair in the center of the image."""
        import cv2

        img = img.copy()
        h, w = img.shape[:2]
        center_x, center_y = w // 2, h // 2

        # Crosshair parameters
        color = (0, 255, 0)  # Green in RGB
        thickness = 1
        gap = 5  # Gap in center
        length = 15  # Length of crosshair lines

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
        cv2.circle(img, (center_x, center_y), 2, color, -1)

        return img

    def _draw_control_overlay_rgb(self, img):
        """Draw camera control buttons onto an RGB numpy array."""
        import cv2

        if not self.show_controls:
            self.control_boxes = {}
            return img

        img = img.copy()
        img_height, img_width = img.shape[:2]

        # Panel dimensions
        panel_width = max(img_width // 4, 200)
        panel_height = 160
        panel_x = img_width - panel_width - 10
        panel_y = img_height - panel_height - 20

        padding = 6
        button_height = 32

        # Draw dark background panel
        overlay = img.copy()
        cv2.rectangle(
            overlay,
            (panel_x, panel_y),
            (panel_x + panel_width, panel_y + panel_height),
            (20, 20, 20),
            -1,
        )
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

        self.control_boxes = {}

        def add_button(name, label, x, y, w, h):
            is_pressed = name in self._pressed_control_names
            is_hovered = name == self._hovered_control_name

            # Button colors (BGR for cv2)
            color = (180, 104, 60)  # Blue in BGR
            if is_pressed:
                color = (156, 88, 45)
            elif is_hovered:
                color = (212, 140, 82)

            x, y, w, h = int(x), int(y), int(w), int(h)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)
            cv2.rectangle(img, (x, y), (x + w, y + h), (40, 40, 40), 1)

            # Draw text
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            thickness = 1
            text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
            text_x = x + (w - text_size[0]) // 2
            text_y = y + (h + text_size[1]) // 2

            cv2.putText(img, label, (text_x + 1, text_y + 1), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
            cv2.putText(img, label, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

            self.control_boxes[name] = {"bounds": (x, y, w, h)}

        # Button layout: 2 columns, 3 rows
        button_width = (panel_width - padding * 3) // 2

        # Row 1: Pan Left, Pan Right
        row_y = panel_y + padding
        add_button("pan_left", "Pan Left", panel_x + padding, row_y, button_width, button_height)
        add_button("pan_right", "Pan Right", panel_x + padding * 2 + button_width, row_y, button_width, button_height)

        # Row 2: Tilt Up, Tilt Down
        row_y += button_height + padding
        add_button("tilt_up", "Tilt Up", panel_x + padding, row_y, button_width, button_height)
        add_button("tilt_down", "Tilt Down", panel_x + padding * 2 + button_width, row_y, button_width, button_height)

        # Row 3: Zoom In, Zoom Out
        row_y += button_height + padding
        add_button("zoom_in", "Zoom In", panel_x + padding, row_y, button_width, button_height)
        add_button("zoom_out", "Zoom Out", panel_x + padding * 2 + button_width, row_y, button_width, button_height)

        return img

    def _draw_control_overlay(self, window_width, window_height, img_width, img_height):
        """Draw camera control buttons for pyglet window mode."""
        import pyglet
        from pyglet import shapes

        if not self.show_controls:
            self.control_boxes = {}
            return

        panel_width = max(self.obs_disp_width - 20, 200)
        panel_height = 160
        panel_x = window_width - panel_width - 10
        panel_y = 20

        padding = 6
        button_height = 32

        batch = pyglet.graphics.Batch()
        background_group = pyglet.graphics.OrderedGroup(0)
        shadow_group = pyglet.graphics.OrderedGroup(1)
        label_group = pyglet.graphics.OrderedGroup(2)

        # Dark background
        background = shapes.Rectangle(
            panel_x, panel_y, panel_width, panel_height,
            color=(20, 20, 20),
            batch=batch,
            group=background_group,
        )
        background.opacity = 180

        self.control_boxes = {}

        def add_button(name, label, x, y, w, h):
            is_pressed = name in self._pressed_control_names
            is_hovered = name == self._hovered_control_name

            color = (60, 104, 180)
            if is_pressed:
                color = (45, 88, 156)
            elif is_hovered:
                color = (82, 140, 212)

            text_color = (255, 255, 255, 255)
            if is_pressed:
                text_color = (235, 255, 205, 255)
            elif is_hovered:
                text_color = (255, 255, 225, 255)

            rect = shapes.Rectangle(x, y, w, h, color=color, batch=batch, group=background_group)
            rect.opacity = 210

            shadow_offset = 1.5
            pyglet.text.Label(
                label,
                font_name="Arial",
                font_size=11,
                bold=True,
                x=x + w / 2 + shadow_offset,
                y=y + h / 2 - shadow_offset,
                anchor_x="center",
                anchor_y="center",
                color=(0, 0, 0, 200),
                batch=batch,
                group=shadow_group,
            )
            label_obj = pyglet.text.Label(
                label,
                font_name="Arial",
                font_size=11,
                bold=True,
                x=x + w / 2,
                y=y + h / 2,
                anchor_x="center",
                anchor_y="center",
                color=text_color,
                batch=batch,
                group=label_group,
            )
            self.control_boxes[name] = {
                "bounds": (x, y, w, h),
                "rect": rect,
                "label": label_obj,
            }

        button_width = (panel_width - padding * 3) / 2

        # Row 1 (top in OpenGL coords = bottom visually): Zoom In, Zoom Out
        row_y = panel_y + padding
        add_button("zoom_in", "Zoom In", panel_x + padding, row_y, button_width, button_height)
        add_button("zoom_out", "Zoom Out", panel_x + padding * 2 + button_width, row_y, button_width, button_height)

        # Row 2: Tilt Up, Tilt Down
        row_y += button_height + padding
        add_button("tilt_up", "Tilt Up", panel_x + padding, row_y, button_width, button_height)
        add_button("tilt_down", "Tilt Down", panel_x + padding * 2 + button_width, row_y, button_width, button_height)

        # Row 3 (bottom visually = top in OpenGL): Pan Left, Pan Right
        row_y += button_height + padding
        add_button("pan_left", "Pan Left", panel_x + padding, row_y, button_width, button_height)
        add_button("pan_right", "Pan Right", panel_x + padding * 2 + button_width, row_y, button_width, button_height)

        batch.draw()

    def render(self):
        """Override render to include crosshair in rgb_array mode."""
        mode = self.render_mode

        if mode == "rgb_array":
            # Get the standard render
            img = super().render()

            # The parent render already calls _draw_control_overlay_rgb
            # Just need to ensure crosshair is on the observation part
            # Actually, the parent render combines obs and top view
            # We already add crosshair in render_obs, so it should be there

            return img

        return super().render()
