#!/usr/bin/env python3
"""Generate Memory Maze episodes with scripted revisits (TRACK_A_PLAN.md §3.2).

Each episode is written to ``<output-dir>/<episode_id>/`` as ``frames.npy``
(uint8 [T,64,64,3]), ``actions.npy`` (int8 [T]), ``pose.npy`` (float32 [T,3]:
x, y, heading) and ``meta.json``. ``actions[t]`` is the action taken after
observing ``frames[t]``; it produced ``frames[t + 1]``. Use this for small
inspections; ``generate_shard.py`` writes the sharded training format.

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

import numpy as np

from distance_decayed_memory.data.episodes import (
    MAZE_SETTINGS,
    Episode,
    episode_id,
    episode_meta,
    generate_episode,
    git_state,
    write_previews,
)

ROOT = pathlib.Path(__file__).resolve().parents[2]


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


def write_episode(directory: pathlib.Path, episode: Episode, meta: dict) -> None:
    temporary = directory.with_name(directory.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    np.save(temporary / "frames.npy", episode.frames)
    np.save(temporary / "actions.npy", episode.actions)
    np.save(temporary / "pose.npy", episode.poses)
    (temporary / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    if directory.exists():
        shutil.rmtree(directory)
    temporary.rename(directory)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    git = git_state(ROOT)
    summary_events: collections.Counter[str] = collections.Counter()
    gaps: list[int] = []
    rates: list[float] = []
    for index in range(args.episodes):
        seed = args.first_seed + index
        directory = args.output_dir / episode_id(args.maze_size, seed)
        if directory.exists() and not args.overwrite:
            print(f"skip {directory.name}: exists")
            continue
        episode = generate_episode(seed, args.frames, args.maze_size)
        write_episode(directory, episode, episode_meta(episode, git))
        if index < args.preview_episodes:
            write_previews(directory, episode)
        counts = collections.Counter(event.status for event in episode.events)
        summary_events.update(counts)
        gaps.extend(e.gap for e in episode.events if e.status == "completed")
        rates.append(args.frames / episode.seconds)
        print(f"{directory.name}: {rates[-1]:.1f} frames/s, events {dict(counts)}")

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
    print(json.dumps({k: v for k, v in summary.items() if k != "completed_gaps"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
