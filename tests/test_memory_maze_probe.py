from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_probe():
    path = ROOT / "scripts" / "data" / "probe_memory_maze.py"
    spec = importlib.util.spec_from_file_location("probe_memory_maze", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def valid_observation(module):
    return {
        key: np.zeros(shape, dtype=dtype)
        for key, (shape, dtype) in module.REQUIRED_OBSERVATIONS.items()
    }


def test_expected_discrete_actions_match_published_order() -> None:
    module = load_probe()
    assert module.ACTION_NAMES == (
        "noop",
        "forward",
        "left",
        "right",
        "forward_left",
        "forward_right",
    )
    np.testing.assert_array_equal(
        module.EXPECTED_ACTIONS,
        [
            [0.0, 0.0],
            [-1.0, 0.0],
            [0.0, -1.0],
            [0.0, 1.0],
            [-1.0, -1.0],
            [-1.0, 1.0],
        ],
    )


def test_validate_observation_accepts_expected_global_fields() -> None:
    module = load_probe()
    observation = valid_observation(module)
    observation["agent_dir"] = np.asarray([1.0, 0.0])
    observation["maze_layout"][1:8, 1:8] = 1
    module.validate_observation(observation)


def test_validate_observation_rejects_missing_field() -> None:
    module = load_probe()
    observation = valid_observation(module)
    observation["agent_dir"] = np.asarray([1.0, 0.0])
    del observation["agent_pos"]
    with pytest.raises(AssertionError, match="Unexpected observation keys"):
        module.validate_observation(observation)


def test_validate_observation_rejects_non_unit_heading() -> None:
    module = load_probe()
    observation = valid_observation(module)
    observation["agent_dir"] = np.asarray([2.0, 0.0])
    with pytest.raises(AssertionError, match="not a unit vector"):
        module.validate_observation(observation)
