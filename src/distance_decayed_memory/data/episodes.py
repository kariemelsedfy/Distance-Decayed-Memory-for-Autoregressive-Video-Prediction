"""Generate one scripted Memory Maze episode and describe it.

Memory Maze is imported lazily: set ``MUJOCO_GL`` (``egl`` on the cluster,
``glfw`` on a desktop) before the first call, because the renderer is chosen
at import time. Everything else here is importable without Memory Maze.
"""

from __future__ import annotations

import pathlib
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from distance_decayed_memory.data.navigation import ACTION_NAMES, heading_of
from distance_decayed_memory.data.revisit_script import (
    RevisitEvent,
    RevisitScript,
    ScriptConfig,
)

CONTROL_FREQUENCY = 4.0
IMAGE_SIZE = 64
MAZE_SETTINGS: dict[int, tuple[int, dict[str, int]]] = {
    9: (3, {}),
    11: (4, {}),
    13: (5, {}),
    15: (6, {"max_rooms": 9, "room_max_size": 3}),
}
ACTION_CONVENTION = "actions[t] is taken after frames[t] and produces frames[t + 1]"


@dataclass
class Episode:
    episode_id: str
    seed: int
    maze_size: int
    frames: np.ndarray
    actions: np.ndarray
    poses: np.ndarray
    layout: np.ndarray
    phases: list[str]
    events: list[RevisitEvent]
    revisits_enabled: bool
    seconds: float
    config: ScriptConfig = field(repr=False)

    @property
    def length(self) -> int:
        return len(self.frames)


def episode_id(maze_size: int, seed: int) -> str:
    return f"memmaze{maze_size}-seed{seed:07d}"


def git_state(root: pathlib.Path | None = None) -> dict[str, Any]:
    root = root or pathlib.Path(__file__).resolve().parents[3]

    def run(*command: str) -> str:
        return subprocess.run(
            ["git", *command], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        return {
            "sha": run("rev-parse", "HEAD"),
            "dirty": bool(run("status", "--porcelain")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"sha": "unknown", "dirty": None}


def make_env(maze_size: int, seed: int, frames: int):
    from memory_maze import tasks

    n_targets, room_settings = MAZE_SETTINGS[maze_size]
    # The public constructors fix a 250-1000 s time limit (1000-4000 steps);
    # the pinned private builder lets episodes run to the requested length (D-008).
    return tasks._memory_maze(
        maze_size,
        n_targets,
        frames / CONTROL_FREQUENCY + 10.0,
        control_freq=CONTROL_FREQUENCY,
        discrete_actions=True,
        image_only_obs=False,
        target_color_in_image=False,
        global_observables=True,
        top_camera=False,
        camera_resolution=IMAGE_SIZE,
        seed=seed,
        **room_settings,
    )


def run_length(phases: list[str]) -> list[list[Any]]:
    """Compress per-frame phase names into ``[name, start, stop)`` segments."""
    segments: list[list[Any]] = []
    for t, phase in enumerate(phases):
        if segments and segments[-1][0] == phase:
            segments[-1][2] = t + 1
        else:
            segments.append([phase, t, t + 1])
    return segments


def generate_episode(seed: int, frames: int, maze_size: int = 9) -> Episode:
    env = make_env(maze_size, seed, frames)
    try:
        timestep = env.reset()
        layout = np.asarray(timestep.observation["maze_layout"]).copy()
        script = RevisitScript(
            layout,
            np.random.default_rng([seed, 1]),
            ScriptConfig(episode_frames=frames),
        )
        images = np.empty((frames, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        actions = np.empty(frames, dtype=np.int8)
        poses = np.empty((frames, 3), dtype=np.float32)
        started = time.perf_counter()
        for t in range(frames):
            observation = timestep.observation
            position = np.asarray(observation["agent_pos"], dtype=np.float64)
            heading = heading_of(observation["agent_dir"])
            images[t] = observation["image"]
            poses[t] = (position[0], position[1], heading)
            actions[t] = script.act(t, position, heading)
            timestep = env.step(int(actions[t]))
            if timestep.last() and t < frames - 1:
                raise RuntimeError(f"Episode ended early at frame {t + 1}")
        seconds = time.perf_counter() - started
    finally:
        env.close()
    return Episode(
        episode_id=episode_id(maze_size, seed),
        seed=seed,
        maze_size=maze_size,
        frames=images,
        actions=actions,
        poses=poses,
        layout=layout,
        phases=script.phases,
        events=script.finish(),
        revisits_enabled=script.revisits_enabled,
        seconds=seconds,
        config=script.config,
    )


def episode_meta(episode: Episode, git: dict[str, Any]) -> dict[str, Any]:
    config = episode.config
    return {
        "episode_id": episode.episode_id,
        "environment_seed": episode.seed,
        "script_seed": [episode.seed, 1],
        "maze_size": episode.maze_size,
        "frames": episode.length,
        "control_frequency_hz": CONTROL_FREQUENCY,
        "target_color_in_image": False,
        "action_names": list(ACTION_NAMES),
        "action_convention": ACTION_CONVENTION,
        "pose_columns": ["x", "y", "heading_radians"],
        "maze_layout": episode.layout.tolist(),
        "revisits_enabled": episode.revisits_enabled,
        "events": [event.to_dict() for event in episode.events],
        "phases": run_length(episode.phases),
        "script_config": {
            key: value for key, value in vars(config).items() if key != "controller"
        },
        "controller_config": vars(config.controller),
        "generator_git": git,
        "generation_seconds": episode.seconds,
    }


def write_previews(directory: pathlib.Path, episode: Episode) -> None:
    from distance_decayed_memory.data import preview

    directory.mkdir(parents=True, exist_ok=True)
    preview.write_map(
        directory / "map.png",
        episode.layout,
        episode.poses,
        episode.phases,
        episode.events,
    )
    preview.write_revisit_pairs(
        directory / "revisit_pairs.png", episode.frames, episode.events
    )
    preview.write_gif(
        directory / "preview.gif",
        episode.frames,
        episode.layout,
        episode.poses,
        episode.phases,
        episode.events,
    )
