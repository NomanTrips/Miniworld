"""Utilities for writing Miniworld rollouts in the LeRobot format.

This module provides a lightweight dataset manager that handles chunked
video + Parquet outputs, basic statistics, and metadata emission so
scripts like :mod:`miniworld.manual_control` do not have to reimplement
recording logic.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import imageio
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


FEATURE_SCHEMA = {
    "timestamp": {
        "dtype": "float64",
        "description": "Seconds elapsed since the beginning of the episode.",
    },
    "episode_index": {"dtype": "int64"},
    "frame_index": {"dtype": "int64"},
    "video_frame_index": {"dtype": "int64"},
    "video_path": {"dtype": "string"},
    "action": {"dtype": "float32", "shape": ["action_dim"]},
    "state": {
        "dtype": "float32",
        "shape": ["state_dim"],
        "optional": True,
        "description": (
            "[pos_x, pos_y, pos_z, yaw, pitch] followed by any task-specific "
            "scalars or vectors sorted by their info dict keys and flattened in "
            "C-order to keep observation.state deterministic."
        ),
    },
    "image": {
        "dtype": "uint8",
        "description": "RGB frames stored in the accompanying chunked video files.",
    },
}


def build_state_vector(info: Optional[Dict[str, object]]) -> Optional[np.ndarray]:
    """
    Flatten ``info`` into a deterministic observation.state vector.

    The output always begins with the agent pose from ``info["agent"]`` in the order
    ``[pos_x, pos_y, pos_z, yaw, pitch]``. Any additional scalar or vector entries in
    ``info`` (excluding the ``"agent"`` key) are appended next, sorted by key and
    flattened in C-order. If ``info`` is ``None`` or lacks the ``"agent"`` entry,
    ``None`` is returned so the caller can skip optional state logging.
    """

    if info is None:
        return None

    agent_info = info.get("agent")
    if agent_info is None:
        return None

    pos = np.asarray(agent_info.get("pos"), dtype=np.float32).reshape(-1)
    if pos.size < 3:
        return None

    yaw = float(np.asarray(agent_info.get("dir"), dtype=np.float32).reshape(-1)[0])
    pitch = float(
        np.asarray(agent_info.get("cam_pitch"), dtype=np.float32).reshape(-1)[0]
    )

    state_parts: List[float] = [
        float(pos[0]),
        float(pos[1]),
        float(pos[2]),
        yaw,
        pitch,
    ]

    extras = {key: value for key, value in info.items() if key != "agent"}
    for key in sorted(extras):
        state_parts.extend(np.asarray(extras[key], dtype=np.float32).ravel().tolist())

    return np.asarray(state_parts, dtype=np.float32)


@dataclass
class _RunningStats:
    count: int = 0
    sum: float = 0.0
    sum_of_squares: float = 0.0
    minimum: float = field(default_factory=lambda: float("inf"))
    maximum: float = field(default_factory=lambda: float("-inf"))

    def update(self, values: np.ndarray) -> None:
        flat = values.astype(np.float64).ravel()
        if flat.size == 0:
            return

        self.count += flat.size
        self.sum += float(np.sum(flat))
        self.sum_of_squares += float(np.sum(flat ** 2))
        self.minimum = min(self.minimum, float(np.min(flat)))
        self.maximum = max(self.maximum, float(np.max(flat)))

    def as_dict(self) -> Dict[str, float]:
        if self.count == 0:
            return {
                "count": 0,
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
            }

        mean = self.sum / self.count
        variance = max(self.sum_of_squares / self.count - mean**2, 0.0)
        std = float(np.sqrt(variance))
        return {
            "count": self.count,
            "mean": float(mean),
            "std": std,
            "min": float(self.minimum),
            "max": float(self.maximum),
        }


class _StatsAggregator:
    def __init__(self) -> None:
        self._stats: Dict[str, _RunningStats] = {}
        self.shapes: Dict[str, Sequence[int]] = {}

    def update_numeric(self, name: str, array: np.ndarray) -> None:
        stats = self._stats.setdefault(name, _RunningStats())
        stats.update(np.asarray(array))
        if name not in self.shapes:
            self.shapes[name] = tuple(np.asarray(array).shape)

    def update_bool(self, name: str, array: np.ndarray) -> None:
        self.update_numeric(name, np.asarray(array, dtype=np.int8))

    def update_image(self, name: str, image: np.ndarray) -> None:
        self.update_numeric(name, image)
        if name not in self.shapes:
            self.shapes[name] = tuple(image.shape)

    def summary(self) -> Dict[str, Dict[str, float]]:
        summary: Dict[str, Dict[str, float]] = {}
        for name, stats in self._stats.items():
            summary[name] = stats.as_dict()
            if name in self.shapes:
                summary[name]["shape"] = list(self.shapes[name])
        return summary


class EpisodeWriter:
    """Collects per-episode samples before handing them to the dataset manager."""

    def __init__(self, manager: "DatasetManager", episode_index: int) -> None:
        self.manager = manager
        self.episode_index = episode_index

        self._frames: List[np.ndarray] = []
        self._actions: List[np.ndarray] = []
        self._states: List[Optional[np.ndarray]] = []
        self._timestamps: List[float] = []
        self._frame_indices: List[int] = []

        self._start_time: Optional[float] = None
        self._closed = False

    @property
    def num_frames(self) -> int:
        return len(self._frames)

    def add_sample(
        self,
        *,
        frame: np.ndarray,
        action: np.ndarray,
        state: Optional[np.ndarray] = None,
        timestamp: Optional[float] = None,
        frame_index: Optional[int] = None,
    ) -> None:
        """Append a new observation/action pair to the episode buffer.

        The provided timestamp is normalized so the first sample starts at ``0.0``
        seconds. If no timestamp is provided, wall-clock time is used.
        """

        if self._closed:
            return

        ts = time.time() if timestamp is None else float(timestamp)
        if self._start_time is None:
            self._start_time = ts
        normalized_timestamp = ts - self._start_time

        self._timestamps.append(normalized_timestamp)
        self._frame_indices.append(
            len(self._frames) if frame_index is None else int(frame_index)
        )
        self._frames.append(np.array(frame, copy=True))
        self._actions.append(np.array(action, dtype=np.float32, copy=True))
        self._states.append(None if state is None else np.array(state, dtype=np.float32, copy=True))

    def close(self) -> Path:
        if self._closed:
            return self.manager.dataset_dir

        self._closed = True
        self.manager.append_episode(
            episode_index=self.episode_index,
            frames=self._frames,
            frame_indices=self._frame_indices,
            actions=self._actions,
            states=self._states,
            timestamps=self._timestamps,
        )
        return self.manager.dataset_dir

    def __enter__(self) -> "EpisodeWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class DatasetManager:
    """Manage chunked LeRobot-style dataset creation for Miniworld rollouts."""

    def __init__(
        self,
        dataset_dir: Path,
        *,
        chunk_size: int = 512,
        fps: int = 30,
        data_template: str = "data/chunk_{chunk:06d}.parquet",
        video_template: str = "videos/chunk_{chunk:06d}.mp4",
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

        self.chunk_size = chunk_size
        self.fps = fps
        self.data_template = data_template
        self.video_template = video_template

        self.data_dir = self.dataset_dir / Path(data_template).parent
        self.video_dir = self.dataset_dir / Path(video_template).parent
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.video_dir.mkdir(parents=True, exist_ok=True)

        self._rows: List[Dict[str, object]] = []
        self._frames: List[np.ndarray] = []
        self._chunk_index: int = 0
        self._num_episodes: int = 0
        self._num_samples: int = 0

        self._stats = _StatsAggregator()
        self._action_dim: Optional[int] = None
        self._state_dim: Optional[int] = None
        self._frame_shape: Optional[Sequence[int]] = None

    def create_episode_writer(self, episode_index: Optional[int] = None) -> EpisodeWriter:
        if episode_index is None:
            episode_index = self._num_episodes
        self._num_episodes = max(self._num_episodes, episode_index + 1)
        return EpisodeWriter(self, episode_index)

    def append_episode(
        self,
        *,
        episode_index: int,
        frames: Sequence[np.ndarray],
        frame_indices: Sequence[int],
        actions: Sequence[np.ndarray],
        states: Sequence[Optional[np.ndarray]],
        timestamps: Sequence[float],
    ) -> None:
        if not frames:
            return

        if self._frame_shape is None:
            self._frame_shape = tuple(np.asarray(frames[0]).shape)
        if self._action_dim is None:
            self._action_dim = int(np.asarray(actions[0]).size)
        if any(state is not None for state in states) and self._state_dim is None:
            first_state = next(state for state in states if state is not None)
            self._state_dim = int(np.asarray(first_state).size)

        for frame_idx, frame, action, state, timestamp in zip(
            frame_indices, frames, actions, states, timestamps
        ):
            self._rows.append(
                {
                    "episode_index": int(episode_index),
                    "frame_index": int(frame_idx),
                    "timestamp": float(timestamp),
                    "action": np.asarray(action, dtype=np.float32),
                    "state": None if state is None else np.asarray(state, dtype=np.float32),
                }
            )
            self._frames.append(np.asarray(frame))

            self._stats.update_numeric("timestamp", np.asarray(timestamp))
            self._stats.update_numeric("action", np.asarray(action))
            if state is not None:
                self._stats.update_numeric("state", np.asarray(state))
            self._stats.update_image("image", np.asarray(frame))

        self._num_samples += len(frames)
        self._flush_if_needed()

    def finalize(self) -> Path:
        if self._rows:
            self._flush_chunk(len(self._rows))

        metadata = {
            "feature_schema": FEATURE_SCHEMA,
            "paths": {
                "data_path": self.data_template,
                "video_path": self.video_template,
            },
            "stats": self._stats.summary(),
            "num_episodes": self._num_episodes,
            "num_samples": self._num_samples,
            "chunk_size": self.chunk_size,
            "fps": self.fps,
            "frame_shape": list(self._frame_shape) if self._frame_shape is not None else None,
            "action_dim": self._action_dim,
            "state_dim": self._state_dim,
        }
        metadata_path = self.dataset_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2))
        return metadata_path

    # Internal helpers -------------------------------------------------

    def _flush_if_needed(self) -> None:
        while len(self._rows) >= self.chunk_size:
            self._flush_chunk(self.chunk_size)

    def _flush_chunk(self, size: int) -> None:
        rows_chunk = self._rows[:size]
        frames_chunk = self._frames[:size]

        del self._rows[:size]
        del self._frames[:size]

        video_rel_path = Path(self.video_template.format(chunk=self._chunk_index))
        video_path = self.dataset_dir / video_rel_path
        self._write_video(video_path, frames_chunk)

        data_rel_path = Path(self.data_template.format(chunk=self._chunk_index))
        data_path = self.dataset_dir / data_rel_path
        self._write_parquet(data_path, rows_chunk, video_rel_path)

        self._chunk_index += 1

    def _write_video(self, video_path: Path, frames: Iterable[np.ndarray]) -> None:
        video_path.parent.mkdir(parents=True, exist_ok=True)
        with imageio.get_writer(
            video_path,
            fps=self.fps,
            format="FFMPEG",
            codec="libx264",
            quality=8,
        ) as writer:
            for frame in frames:
                writer.append_data(np.asarray(frame))

    def _write_parquet(
        self, data_path: Path, rows: Sequence[Dict[str, object]], video_rel_path: Path
    ) -> None:
        data_path.parent.mkdir(parents=True, exist_ok=True)

        episode_indices = pa.array([row["episode_index"] for row in rows], type=pa.int64())
        frame_indices = pa.array([row["frame_index"] for row in rows], type=pa.int64())
        timestamps = pa.array([row["timestamp"] for row in rows], type=pa.float64())
        video_frame_indices = pa.array(range(len(rows)), type=pa.int64())
        video_paths = pa.array([str(video_rel_path)] * len(rows))

        action_array = pa.array(
            [np.asarray(row["action"]).tolist() for row in rows],
            type=pa.list_(pa.float32()),
        )

        state_values = [
            None if row["state"] is None else np.asarray(row["state"]).tolist()
            for row in rows
        ]
        state_array = pa.array(state_values, type=pa.list_(pa.float32()))

        table = pa.Table.from_arrays(
            [
                episode_indices,
                frame_indices,
                timestamps,
                video_frame_indices,
                video_paths,
                action_array,
                state_array,
            ],
            names=[
                "episode_index",
                "frame_index",
                "timestamp",
                "video_frame_index",
                "video_path",
                "action",
                "state",
            ],
        )
        pq.write_table(table, data_path)
