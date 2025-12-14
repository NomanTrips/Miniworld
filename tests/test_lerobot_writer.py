import importlib.util
import json
from pathlib import Path

import pytest


def _load_dataset_manager():
    pytest.importorskip("imageio")
    pytest.importorskip("pyarrow")
    module_path = Path(__file__).parent.parent / "miniworld" / "lerobot_writer.py"
    spec = importlib.util.spec_from_file_location("miniworld.lerobot_writer", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.DatasetManager


def test_create_episode_writer_respects_existing_count(tmp_path):
    meta_dir = tmp_path / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "info.json").write_text(
        json.dumps({"total_episodes": 2, "total_frames": 10})
    )

    DatasetManager = _load_dataset_manager()
    manager = DatasetManager(tmp_path, append=True)

    writer = manager.create_episode_writer()

    assert writer.episode_index == 2
    assert manager.num_episodes == 3
