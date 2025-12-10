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
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from miniworld import __version__


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

    def __init__(
        self,
        manager: "DatasetManager",
        episode_index: int,
        *,
        tasks: Optional[Sequence[str]] = None,
    ) -> None:
        self.manager = manager
        self.episode_index = episode_index
        self.tasks = list(tasks) if tasks is not None else []

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
            tasks=self.tasks,
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
        chunk_size: int = 1000,
        max_chunk_size_bytes: int = 200_000_000,
        fps: int = 10,
        data_template: str = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        video_template: str = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        default_task: str = "Center and zoom on the target.",
        append: bool = False,
        force_single_chunk: bool = True,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

        self.chunk_size = chunk_size
        self.max_chunk_size_bytes = max_chunk_size_bytes
        self.fps = fps
        self.data_template = data_template
        self.video_template = video_template
        self.default_task = default_task
        self.force_single_chunk = force_single_chunk
        self._append_mode = append
        self.video_key = "observation.image"

        sample_data_path = Path(data_template.format(chunk_index=0, file_index=0))
        sample_video_path = Path(
            video_template.format(video_key=self.video_key, chunk_index=0, file_index=0)
        )

        self.data_dir = self.dataset_dir / sample_data_path.parent
        self.video_dir = self.dataset_dir / sample_video_path.parent
        self.meta_dir = self.dataset_dir / "meta"
        self.episodes_dir = self.meta_dir / "episodes"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.episodes_dir.mkdir(parents=True, exist_ok=True)

        self._rows: List[Dict[str, object]] = []
        self._frames: List[np.ndarray] = []
        self._chunk_index: int = 0
        self._file_index: int = 0
        self._current_chunk_bytes: int = 0
        self._num_episodes: int = 0
        self._num_samples: int = 0
        self._data_files: List[Path] = []
        self._video_files: List[Path] = []

        self._stats = _StatsAggregator()
        self._action_dim: Optional[int] = None
        self._state_dim: Optional[int] = None
        self._frame_shape: Optional[Sequence[int]] = None
        self._episodes_rows: List[Dict[str, object]] = []
        self._tasks: Dict[str, int] = {}

        if append:
            self._load_existing_dataset()

    def _load_existing_dataset(self) -> None:
        info_path = self.meta_dir / "info.json"
        if info_path.exists():
            try:
                info = json.loads(info_path.read_text())
            except json.JSONDecodeError:
                info = {}
            self._num_samples = int(info.get("total_frames", 0))
            self._num_episodes = int(info.get("total_episodes", 0))

            features = info.get("features", {})
            image_schema = features.get("observation.image", {})
            self._frame_shape = tuple(image_schema.get("shape", [])) or None
            state_schema = features.get("observation.state", {})
            state_shape = state_schema.get("shape") or []
            if state_shape:
                self._state_dim = int(state_shape[0])
            action_schema = features.get("action", {})
            action_shape = action_schema.get("shape") or []
            if action_shape:
                self._action_dim = int(action_shape[0])

        tasks_parquet_path = self.meta_dir / "tasks.parquet"
        tasks_jsonl_path = self.meta_dir / "tasks.jsonl"
        if tasks_parquet_path.exists():
            table = pq.read_table(tasks_parquet_path)
            descriptions = table.column("description") if "description" in table.column_names else []
            task_ids = table.column("task_id") if "task_id" in table.column_names else []
            for description, task_id in zip(descriptions, task_ids):
                self._tasks[str(description.as_py())] = int(task_id.as_py())
        elif tasks_jsonl_path.exists():
            for line in tasks_jsonl_path.read_text().splitlines():
                try:
                    task = json.loads(line)
                except json.JSONDecodeError:
                    continue
                description = task.get("description")
                task_id = task.get("task_id")
                if description is not None and task_id is not None:
                    self._tasks[str(description)] = int(task_id)

        episodes_path = self.episodes_dir / "chunk-000" / "episodes-000.parquet"
        if episodes_path.exists():
            table = pq.read_table(episodes_path)
            for row in table.to_pylist():
                self._episodes_rows.append(
                    {
                        "episode_index": int(row.get("episode_index", 0)),
                        "data_chunk_index": int(row.get("data/chunk_index", 0)),
                        "data_file_index": int(row.get("data/file_index", 0)),
                        "dataset_from_index": int(row.get("dataset_from_index", 0)),
                        "dataset_to_index": int(row.get("dataset_to_index", 0)),
                        "video_chunk_index": int(
                            row.get("videos/observation.image/chunk_index", 0)
                        ),
                        "video_file_index": int(
                            row.get("videos/observation.image/file_index", 0)
                        ),
                        "video_from_timestamp": float(
                            row.get("videos/observation.image/from_timestamp", 0.0)
                        ),
                        "video_to_timestamp": float(
                            row.get("videos/observation.image/to_timestamp", 0.0)
                        ),
                        "tasks": list(row.get("tasks", [])),
                        "length": int(row.get("length", 0)),
                    }
                )

        self._data_files = sorted(
            self.dataset_dir.glob("data/chunk-*/file-*.parquet")
        )
        self._video_files = sorted(
            self.dataset_dir.glob(f"videos/{self.video_key}/chunk-*/file-*.mp4")
        )

        existing_indices = [
            int(path.stem.split("-")[1]) for path in self._data_files if "-" in path.stem
        ]
        last_chunk_index = max(existing_indices, default=-1)

        if last_chunk_index >= 0:
            last_data = [
                p for p in self._data_files if f"chunk-{last_chunk_index:03d}" in str(p)
            ]
            last_videos = [
                p for p in self._video_files if f"chunk-{last_chunk_index:03d}" in str(p)
            ]
            self._current_chunk_bytes = sum(p.stat().st_size for p in [*last_data, *last_videos])

        if self._append_mode and last_chunk_index >= 0:
            self._chunk_index = last_chunk_index
            self._file_index = last_chunk_index
        else:
            self._chunk_index = last_chunk_index + 1
            self._file_index = self._chunk_index

    def create_episode_writer(
        self, episode_index: Optional[int] = None, *, tasks: Optional[Sequence[str]] = None
    ) -> EpisodeWriter:
        if episode_index is None:
            episode_index = self._num_episodes
        self._num_episodes = max(self._num_episodes, episode_index + 1)
        resolved_tasks = list(tasks) if tasks is not None else []
        if not resolved_tasks and self.default_task:
            resolved_tasks = [self.default_task]
        return EpisodeWriter(self, episode_index, tasks=resolved_tasks)

    def append_episode(
        self,
        *,
        episode_index: int,
        frames: Sequence[np.ndarray],
        frame_indices: Sequence[int],
        actions: Sequence[np.ndarray],
        states: Sequence[Optional[np.ndarray]],
        timestamps: Sequence[float],
        tasks: Sequence[str],
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

        dataset_from_index = self._num_samples
        dataset_to_index = dataset_from_index + len(frames)

        resolved_tasks = list(tasks) if tasks else [self.default_task]
        task_indices = [self._register_task(task) for task in resolved_tasks]
        task_index = task_indices[0]

        for offset, (frame_idx, frame, action, state, timestamp) in enumerate(
            zip(frame_indices, frames, actions, states, timestamps)
        ):
            is_last_in_episode = offset == len(frames) - 1
            dataset_index = dataset_from_index + offset
            self._rows.append(
                {
                    "index": int(dataset_index),
                    "episode_index": int(episode_index),
                    "frame_index": int(frame_idx),
                    "timestamp": float(timestamp),
                    "action": np.asarray(action, dtype=np.float32),
                    "state": None if state is None else np.asarray(state, dtype=np.float32),
                    "next.reward": 0.0,
                    "next.done": bool(is_last_in_episode),
                    "next.success": False,
                    "task_index": int(task_index),
                }
            )
            self._frames.append(np.asarray(frame))

            self._stats.update_numeric("timestamp", np.asarray(timestamp))
            self._stats.update_numeric("action", np.asarray(action))
            if state is not None:
                self._stats.update_numeric("observation.state", np.asarray(state))
            self._stats.update_image("observation.image", np.asarray(frame))
            self._stats.update_numeric("next.reward", np.asarray(0.0))
            self._stats.update_bool("next.done", np.asarray(is_last_in_episode))
            self._stats.update_bool("next.success", np.asarray(False))

        self._num_samples += len(frames)
        self._record_episode_metadata(
            episode_index=episode_index,
            dataset_from_index=dataset_from_index,
            dataset_to_index=dataset_to_index,
            tasks=resolved_tasks,
        )
        self._flush_if_needed()

    def finalize(self) -> Path:
        if self._rows:
            self._flush_chunk(len(self._rows))

        episodes_path = self._write_episodes_metadata()
        tasks_path = self._write_tasks_metadata()
        stats_path = self._write_stats_metadata()
        info_path = self._write_info_json()

        return info_path

    # Internal helpers -------------------------------------------------

    def _flush_if_needed(self) -> None:
        if self.force_single_chunk:
            return

        while len(self._rows) >= self.chunk_size:
            self._flush_chunk(self.chunk_size)

    def _flush_chunk(self, size: int) -> None:
        rows_chunk = self._rows[:size]
        frames_chunk = self._frames[:size]

        del self._rows[:size]
        del self._frames[:size]

        video_rel_path = Path(
            self.video_template.format(
                video_key=self.video_key, chunk_index=self._chunk_index, file_index=self._file_index
            )
        )
        video_path = self.dataset_dir / video_rel_path

        data_rel_path = Path(
            self.data_template.format(chunk_index=self._chunk_index, file_index=self._file_index)
        )
        data_path = self.dataset_dir / data_rel_path
        append_to_existing = (
            self._append_mode
            and self._current_chunk_bytes < self.max_chunk_size_bytes
            and video_path.exists()
            and data_path.exists()
        )

        if not append_to_existing and self._append_mode and self._current_chunk_bytes >= self.max_chunk_size_bytes:
            self._chunk_index += 1
            self._file_index += 1
            self._current_chunk_bytes = 0

            video_rel_path = Path(
                self.video_template.format(
                    video_key=self.video_key, chunk_index=self._chunk_index, file_index=self._file_index
                )
            )
            video_path = self.dataset_dir / video_rel_path

            data_rel_path = Path(
                self.data_template.format(chunk_index=self._chunk_index, file_index=self._file_index)
            )
            data_path = self.dataset_dir / data_rel_path
            append_to_existing = False

        self._write_video(video_path, frames_chunk, append=append_to_existing)
        self._write_parquet(data_path, rows_chunk, append=append_to_existing)

        self._current_chunk_bytes = sum(
            path.stat().st_size for path in (video_path, data_path) if path.exists()
        )

        if not append_to_existing:
            self._chunk_index += 1
            self._file_index += 1

    def _write_video(
        self, video_path: Path, frames: Iterable[np.ndarray], *, append: bool = False
    ) -> None:
        video_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = None

        if append and video_path.exists():
            # ``imageio`` dispatches writers based on the file suffix. Using a
            # generic ``.tmp`` extension prevents the FFMPEG writer from being
            # selected, causing append mode to fail. Keep the ``.mp4`` suffix so
            # FFMPEG can handle the temporary file correctly before replacing
            # the original.
            temp_path = video_path.with_name(f"{video_path.stem}.tmp{video_path.suffix}")
            with imageio.get_writer(
                temp_path,
                fps=self.fps,
                format="FFMPEG",
                codec="libaom-av1",
                quality=8,
                ffmpeg_params=["-pix_fmt", "yuv420p"],
            ) as writer:
                with imageio.get_reader(video_path) as reader:
                    for frame in reader:
                        writer.append_data(frame)
                for frame in frames:
                    writer.append_data(np.asarray(frame))
            temp_path.replace(video_path)
        else:
            with imageio.get_writer(
                video_path,
                fps=self.fps,
                format="FFMPEG",
                codec="libaom-av1",
                quality=8,
                ffmpeg_params=["-pix_fmt", "yuv420p"],
            ) as writer:
                for frame in frames:
                    writer.append_data(np.asarray(frame))

        if video_path not in self._video_files:
            self._video_files.append(video_path)

    def _write_parquet(
        self, data_path: Path, rows: Sequence[Dict[str, object]], *, append: bool = False
    ) -> None:
        data_path.parent.mkdir(parents=True, exist_ok=True)

        dataset_indices = pa.array([row["index"] for row in rows], type=pa.int64())
        episode_indices = pa.array([row["episode_index"] for row in rows], type=pa.int64())
        frame_indices = pa.array([row["frame_index"] for row in rows], type=pa.int64())
        task_indices = pa.array([row["task_index"] for row in rows], type=pa.int64())
        timestamps = pa.array([row["timestamp"] for row in rows], type=pa.float32())

        action_array = pa.array(
            [np.asarray(row["action"]).tolist() for row in rows],
            type=pa.list_(pa.float32()),
        )

        state_values = [
            None if row["state"] is None else np.asarray(row["state"]).tolist()
            for row in rows
        ]
        state_array = pa.array(state_values, type=pa.list_(pa.float32()))

        rewards = pa.array([row["next.reward"] for row in rows], type=pa.float32())
        dones = pa.array([row["next.done"] for row in rows], type=pa.bool_())
        successes = pa.array([row["next.success"] for row in rows], type=pa.bool_())

        new_table = pa.Table.from_arrays(
            [
                dataset_indices,
                episode_indices,
                frame_indices,
                timestamps,
                task_indices,
                action_array,
                state_array,
                rewards,
                dones,
                successes,
            ],
            names=[
                "index",
                "episode_index",
                "frame_index",
                "timestamp",
                "task_index",
                "action",
                "observation.state",
                "next.reward",
                "next.done",
                "next.success",
            ],
        )
        if append and data_path.exists():
            existing = pq.read_table(data_path)
            combined = pa.concat_tables([existing, new_table])
            pq.write_table(combined, data_path)
        else:
            pq.write_table(new_table, data_path)

        if data_path not in self._data_files:
            self._data_files.append(data_path)

    # Metadata writers -------------------------------------------------

    def _register_task(self, task: str) -> int:
        if task not in self._tasks:
            self._tasks[task] = len(self._tasks)
        return self._tasks[task]

    def _record_episode_metadata(
        self,
        *,
        episode_index: int,
        dataset_from_index: int,
        dataset_to_index: int,
        tasks: Sequence[str],
    ) -> None:
        data_chunk_index = 0 if self.force_single_chunk else dataset_from_index // self.chunk_size
        start_in_chunk = dataset_from_index if self.force_single_chunk else dataset_from_index - data_chunk_index * self.chunk_size
        length = dataset_to_index - dataset_from_index
        from_timestamp = start_in_chunk / float(self.fps)
        to_timestamp = (start_in_chunk + length) / float(self.fps)

        self._episodes_rows.append(
            {
                "episode_index": int(episode_index),
                "data_chunk_index": int(
                    self._chunk_index if self.force_single_chunk else data_chunk_index
                ),
                "data_file_index": int(
                    self._chunk_index if self.force_single_chunk else data_chunk_index
                ),
                "dataset_from_index": int(dataset_from_index),
                "dataset_to_index": int(dataset_to_index),
                "video_chunk_index": int(
                    self._chunk_index if self.force_single_chunk else data_chunk_index
                ),
                "video_file_index": int(
                    self._chunk_index if self.force_single_chunk else data_chunk_index
                ),
                "video_from_timestamp": float(from_timestamp),
                "video_to_timestamp": float(to_timestamp),
                "tasks": list(tasks) if tasks else [self.default_task],
                "length": int(length),
            }
        )

    def _write_tasks_metadata(self) -> Path:
        if not self._tasks:
            self._register_task(self.default_task)

        self.meta_dir.mkdir(parents=True, exist_ok=True)
        tasks_path = self.meta_dir / "tasks.parquet"
        sorted_tasks = sorted(self._tasks.items(), key=lambda item: item[1])
        df = pd.DataFrame(
            {"task_index": [index for _, index in sorted_tasks]},
            index=[task for task, _ in sorted_tasks],
        )
        df.to_parquet(tasks_path, index=True)
        return tasks_path

    def _write_stats_metadata(self) -> Path:
        stats = self._stats.summary()
        stats_path = self.meta_dir / "stats.json"
        stats_path.write_text(json.dumps(stats, indent=2))
        return stats_path

    def _write_episodes_metadata(self) -> Path:
        episodes_chunk_dir = self.episodes_dir / f"chunk-{0:03d}"
        episodes_chunk_dir.mkdir(parents=True, exist_ok=True)
        episodes_path = episodes_chunk_dir / "episodes-000.parquet"

        table = pa.Table.from_arrays(
            [
                pa.array([row["episode_index"] for row in self._episodes_rows], type=pa.int64()),
                pa.array([row["data_chunk_index"] for row in self._episodes_rows], type=pa.int64()),
                pa.array([row["data_file_index"] for row in self._episodes_rows], type=pa.int64()),
                pa.array([row["dataset_from_index"] for row in self._episodes_rows], type=pa.int64()),
                pa.array([row["dataset_to_index"] for row in self._episodes_rows], type=pa.int64()),
                pa.array([row["video_chunk_index"] for row in self._episodes_rows], type=pa.int64()),
                pa.array([row["video_file_index"] for row in self._episodes_rows], type=pa.int64()),
                pa.array(
                    [row["video_from_timestamp"] for row in self._episodes_rows],
                    type=pa.float32(),
                ),
                pa.array(
                    [row["video_to_timestamp"] for row in self._episodes_rows], type=pa.float32()
                ),
                pa.array([row["tasks"] for row in self._episodes_rows], type=pa.list_(pa.string())),
                pa.array([row["length"] for row in self._episodes_rows], type=pa.int64()),
            ],
            names=[
                "episode_index",
                "data/chunk_index",
                "data/file_index",
                "dataset_from_index",
                "dataset_to_index",
                "videos/observation.image/chunk_index",
                "videos/observation.image/file_index",
                "videos/observation.image/from_timestamp",
                "videos/observation.image/to_timestamp",
                "tasks",
                "length",
            ],
        )

        pq.write_table(table, episodes_path)
        return episodes_path

    def _write_info_json(self) -> Path:
        info = {
            "codebase_version": "v3.0",
            "robot_type": "unknown",
            "total_episodes": len(self._episodes_rows),
            "total_frames": self._num_samples,
            "total_tasks": len(self._tasks) if self._tasks else 1,
            "chunks_size": self.chunk_size,
            "fps": self.fps,
            "splits": {"train": f"0:{len(self._episodes_rows)}"},
            "data_path": self.data_template,
            "video_path": self.video_template,
            "features": self._feature_schema(),
            "data_files_size_in_mb": self._total_size_in_mb(self._data_files),
            "video_files_size_in_mb": self._total_size_in_mb(self._video_files),
        }

        self.meta_dir.mkdir(parents=True, exist_ok=True)
        info_path = self.meta_dir / "info.json"
        info_path.write_text(json.dumps(info, indent=2))
        return info_path

    def _total_size_in_mb(self, files: Sequence[Path]) -> float:
        total_bytes = sum(path.stat().st_size for path in files if path.exists())
        return total_bytes / 1_000_000 if total_bytes else 0.0

    def _feature_schema(self) -> Dict[str, object]:
        image_shape = list(self._frame_shape) if self._frame_shape is not None else []
        state_shape = [self._state_dim] if self._state_dim is not None else []
        action_shape = [self._action_dim] if self._action_dim is not None else []

        return {
            "observation.image": {
                "dtype": "video",
                "shape": image_shape,
                "names": ["height", "width", "channel"],
                "video_info": {
                    "video.fps": float(self.fps),
                    "video.codec": "av1",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            },
            "observation.state": {
                "dtype": "float32",
                "shape": state_shape,
                "names": None,
                "fps": self.fps,
            },
            "action": {
                "dtype": "float32",
                "shape": action_shape,
                "names": None,
                "fps": self.fps,
            },
            "episode_index": {"dtype": "int64", "shape": [1], "names": None, "fps": self.fps},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None, "fps": self.fps},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None, "fps": self.fps},
            "next.reward": {"dtype": "float32", "shape": [1], "names": None, "fps": self.fps},
            "next.done": {"dtype": "bool", "shape": [1], "names": None, "fps": self.fps},
            "next.success": {"dtype": "bool", "shape": [1], "names": None, "fps": self.fps},
            "index": {"dtype": "int64", "shape": [1], "names": None, "fps": self.fps},
            "task_index": {"dtype": "int64", "shape": [1], "names": None, "fps": self.fps},
        }
