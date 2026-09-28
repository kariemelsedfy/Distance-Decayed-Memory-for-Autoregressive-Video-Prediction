#!/usr/bin/env python3
"""Generate Memory Maze episodes with scripted revisits (TRACK_A_PLAN.md §3.2).

Each episode is written to ``<output-dir>/<episode_id>/`` as ``frames.npy``
(uint8 [T,64,64,3]), ``actions.npy`` (int8 [T]), ``pose.npy`` (float32 [T,3]:
x, y, heading) and ``meta.json``. ``actions[t]`` is the action taken after
observing ``frames[t]``; it produced ``frames[t + 1]``. Sharding and loading
are issue #11; this script writes one directory per episode.

Set ``MUJOCO_GL`` before starting (``egl`` on the cluster, ``glfw`` on a
desktop); Memory Maze selects its renderer at import time.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import pathlib
import shutil
import statistics
import subprocess
import time
from typing import Any

import numpy as np

from distance_decayed_memory.data.navigation import ACTION_NAMES, heading_of
from distance_decayed_memory.data.revisit_script import RevisitScript, ScriptConfig

CONTROL_FREQUENCY = 4.0
MAZE_SETTINGS: dict[int, tuple[int, dict[str, int]]] = {
    9: (3, {}),
    11: (4, {}),
    13: (5, {}),
    15: (6, {"max_rooms": 9, "room_max_size": 3}),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--frames", type=int, default=2048)
    parser.add_argument(
        "--maze-size", type=int, choices=sorted(MAZE_SETTINGS), default=9
    )
    parser.add_argument(
        "--preview-episodes",
        type=int,
        default=0,
        help="write preview.gif, map.png and revisit_pairs.png for the first N",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.episodes < 1 or args.frames < 64:
        parser.error("need at least one episode of at least 64 frames")
    if "MUJOCO_GL" not in os.environ:
        parser.error("set MUJOCO_GL (egl on the cluster, glfw on a desktop)")
    return args


def git_state() -> dict[str, Any]:
    root = pathlib.Path(__file__).resolve().parents[2]

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
    # the pinned private builder lets episodes run to the requested length.
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
        camera_resolution=64,
        seed=seed,
        **room_settings,
    )


def run_length(phases: list[str]) -> list[list[Any]]:
    segments: list[list[Any]] = []
    for t, phase in enumerate(phases):
        if segments and segments[-1][0] == phase:
            segments[-1][2] = t + 1
        else:
            segments.append([phase, t, t + 1])
    return segments


def generate(seed: int, frames: int, maze_size: int) -> dict[str, Any]:
    env = make_env(maze_size, seed, frames)
    try:
        timestep = env.reset()
        layout = np.asarray(timestep.observation["maze_layout"]).copy()
        script = RevisitScript(
            layout,
            np.random.default_rng([seed, 1]),
            ScriptConfig(episode_frames=frames),
        )
        images = np.empty((frames, 64, 64, 3), dtype=np.uint8)
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
    events = script.finish()
    return {
        "frames": images,
        "actions": actions,
        "poses": poses,
        "layout": layout,
        "phases": script.phases,
        "events": events,
        "revisits_enabled": script.revisits_enabled,
        "seconds": seconds,
        "config": script.config,
    }


def write_episode(
    directory: pathlib.Path, episode: dict[str, Any], meta: dict[str, Any]
) -> None:
    temporary = directory.with_name(directory.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    np.save(temporary / "frames.npy", episode["frames"])
    np.save(temporary / "actions.npy", episode["actions"])
    np.save(temporary / "pose.npy", episode["poses"])
    (temporary / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    if directory.exists():
        shutil.rmtree(directory)
    temporary.rename(directory)


def main() -> int:
    args = parse_args()
    import memory_maze  # noqa: F401 -- import establishes the selected renderer

    from distance_decayed_memory.data import preview

    args.output_dir.mkdir(parents=True, exist_ok=True)
    git = git_state()
    summary_events: collections.Counter[str] = collections.Counter()
    gaps: list[int] = []
    rates: list[float] = []
    for index in range(args.episodes):
        seed = args.first_seed + index
        episode_id = f"memmaze{args.maze_size}-seed{seed:07d}"
        directory = args.output_dir / episode_id
        if directory.exists() and not args.overwrite:
            print(f"skip {episode_id}: exists")
            continue
        episode = generate(seed, args.frames, args.maze_size)
        events = episode["events"]
        config = episode["config"]
        meta = {
            "episode_id": episode_id,
            "environment_seed": seed,
            "script_seed": [seed, 1],
            "maze_size": args.maze_size,
            "frames": args.frames,
            "control_frequency_hz": CONTROL_FREQUENCY,
            "target_color_in_image": False,
            "action_names": list(ACTION_NAMES),
            "action_convention": "actions[t] is taken after frames[t] and produces "
            "frames[t + 1]",
            "pose_columns": ["x", "y", "heading_radians"],
            "maze_layout": episode["layout"].tolist(),
            "revisits_enabled": episode["revisits_enabled"],
            "events": [event.to_dict() for event in events],
            "phases": run_length(episode["phases"]),
            "script_config": {
                key: value for key, value in vars(config).items() if key != "controller"
            },
            "controller_config": vars(config.controller),
            "generator_git": git,
            "generation_seconds": episode["seconds"],
        }
        write_episode(directory, episode, meta)
        if index < args.preview_episodes:
            poses = episode["poses"]
            preview.write_map(
                directory / "map.png",
                episode["layout"],
                poses,
                episode["phases"],
                events,
            )
            preview.write_revisit_pairs(
                directory / "revisit_pairs.png", episode["frames"], events
            )
            preview.write_gif(
                directory / "preview.gif",
                episode["frames"],
                episode["layout"],
                poses,
                episode["phases"],
                events,
            )
        counts = collections.Counter(event.status for event in events)
        summary_events.update(counts)
        gaps.extend(e.gap for e in events if e.status == "completed")
        rates.append(args.frames / episode["seconds"])
        print(
            f"{episode_id}: {args.frames / episode['seconds']:.1f} frames/s, "
            f"events {dict(counts)}"
        )

    summary = {
        "episodes": args.episodes,
        "first_seed": args.first_seed,
        "frames_per_episode": args.frames,
        "maze_size": args.maze_size,
        "mujoco_gl": os.environ.get("MUJOCO_GL"),
        "generator_git": git,
        "event_status_counts": dict(summary_events),
        "completed_gaps": sorted(gaps),
        "log2_gap_histogram": dict(
            sorted(collections.Counter(int(math.log2(g)) for g in gaps).items())
        ),
        "frames_per_second_median": statistics.median(rates) if rates else None,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k != "completed_gaps"}, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
