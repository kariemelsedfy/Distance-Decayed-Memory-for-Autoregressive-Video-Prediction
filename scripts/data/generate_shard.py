#!/usr/bin/env python3
"""Generate one shard of scripted Memory Maze episodes (issue #11).

Shard ``i`` of a split holds episodes with environment seeds
``first_seed + i * episodes_per_shard + j`` for ``j < episodes_per_shard``.
Splits use disjoint seed ranges (``docs/DATASETS.md``). A finished shard is
skipped, so a resubmitted array task only redoes missing shards.

Under Slurm the shard index defaults to ``SLURM_ARRAY_TASK_ID``. Set
``MUJOCO_GL`` before starting (``egl`` on the cluster, ``glfw`` on a desktop).
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import pathlib
import statistics
import time

from distance_decayed_memory.data.episodes import (
    MAZE_SETTINGS,
    episode_meta,
    generate_episode,
    git_state,
    write_previews,
)
from distance_decayed_memory.data.shards import (
    ShardWriter,
    is_complete_shard,
    shard_name,
)

ROOT = pathlib.Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", required=True, type=pathlib.Path)
    parser.add_argument("--split", required=True)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--episodes-per-shard", type=int, required=True)
    parser.add_argument("--first-seed", type=int, required=True)
    parser.add_argument("--frames", type=int, default=2048)
    parser.add_argument(
        "--maze-size", type=int, choices=sorted(MAZE_SETTINGS), default=9
    )
    parser.add_argument(
        "--preview-episodes",
        type=int,
        default=0,
        help="write previews for the first N episodes of this shard",
    )
    args = parser.parse_args()
    if args.shard_index is None:
        task = os.environ.get("SLURM_ARRAY_TASK_ID")
        if task is None:
            parser.error("pass --shard-index or run as a Slurm array task")
        args.shard_index = int(task)
    if "MUJOCO_GL" not in os.environ:
        parser.error("set MUJOCO_GL (egl on the cluster, glfw on a desktop)")
    return args


def main() -> int:
    args = parse_args()
    directory = args.split_dir / shard_name(args.shard_index)
    if is_complete_shard(directory):
        print(f"{directory} is complete; nothing to do")
        return 0
    args.split_dir.mkdir(parents=True, exist_ok=True)
    lock = (args.split_dir / f".{directory.name}.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"another task is already writing {directory}") from None
    git = git_state(ROOT)
    first = args.first_seed + args.shard_index * args.episodes_per_shard
    seeds = list(range(first, first + args.episodes_per_shard))
    writer = ShardWriter(
        directory,
        total_frames=args.episodes_per_shard * args.frames,
        info={
            "split": args.split,
            "shard_index": args.shard_index,
            "seeds": [seeds[0], seeds[-1]],
            "maze_size": args.maze_size,
            "frames_per_episode": args.frames,
            "generator_git": git,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "hostname": os.uname().nodename,
        },
    )
    started = time.perf_counter()
    rates = []
    for number, seed in enumerate(seeds):
        episode = generate_episode(seed, args.frames, args.maze_size)
        writer.add(
            episode.frames, episode.actions, episode.poses, episode_meta(episode, git)
        )
        if number < args.preview_episodes:
            write_previews(args.split_dir / "previews" / episode.episode_id, episode)
        rates.append(args.frames / episode.seconds)
        print(
            f"{episode.episode_id}: {rates[-1]:.1f} frames/s "
            f"({number + 1}/{len(seeds)})",
            flush=True,
        )
    writer.info["environment_frames_per_second_median"] = statistics.median(rates)
    writer.info["shard_wall_seconds"] = time.perf_counter() - started
    record = writer.finish()
    print(json.dumps({k: v for k, v in record.items() if k != "episode_ids"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
