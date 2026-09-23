#!/usr/bin/env python3
"""Validate the pose-based revisit detector on generated episodes (issue #10).

For each episode directory (``frames.npy``, ``pose.npy``, ``meta.json``) this
labels every frame with the detector, then writes to ``--output-dir``:

- ``report.json``: label counts, frames per gap bucket, whether every scripted
  return is found, the pixel difference of matched pairs per bucket against
  random pairs, and a sweep over position and heading tolerances;
- ``spot_check.png``: random (source, revisit) frame pairs for every gap bucket,
  plus near misses just outside the tolerances, for visual inspection.

It needs only NumPy and Pillow, not Memory Maze.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import pathlib
from typing import Any

import numpy as np

from distance_decayed_memory.data import preview
from distance_decayed_memory.data.revisit_detector import (
    KIND_NAMES,
    REVISIT,
    DetectorConfig,
    bucket_label,
    detect_revisits,
    gap_bucket,
)

SWEEP_POSITIONS = (0.15, 0.3, 0.5)
SWEEP_HEADINGS = (7.5, 15.0, 30.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episodes", nargs="+", type=pathlib.Path)
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    parser.add_argument("--pairs-per-bucket", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--position-tolerance", type=float, default=0.3)
    parser.add_argument("--heading-tolerance-degrees", type=float, default=15.0)
    parser.add_argument("--min-gap", type=int, default=16)
    parser.add_argument("--no-sweep", action="store_true")
    return parser.parse_args()


def pixel_difference(frames: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Mean absolute RGB difference (0-255) between frames ``a[i]`` and ``b[i]``."""
    if not len(a):
        return np.zeros(0)
    return np.abs(frames[a].astype(np.int16) - frames[b].astype(np.int16)).mean(
        axis=(1, 2, 3)
    )


def median_or_none(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def scripted_agreement(labels, meta: dict[str, Any]) -> collections.Counter[str]:
    """How the detector labels each completed scripted return frame."""
    agreement: collections.Counter[str] = collections.Counter()
    for event in meta.get("events", []):
        if event["status"] != "completed":
            continue
        frame = event["return_frame"]
        agreement[KIND_NAMES[labels.kind[frame]]] += 1
        if (
            labels.kind[frame] == REVISIT
            and labels.source[frame] < event["anchor_frame"]
        ):
            agreement["source_before_anchor"] += 1
    return agreement


def near_misses(
    poses: np.ndarray, config: DetectorConfig, rng: np.random.Generator, count: int
) -> list[tuple[int, int, str]]:
    """Pairs at least ``min_gap`` apart that fall just outside one tolerance."""
    found: list[tuple[int, int, str]] = []
    frames = len(poses)
    for _ in range(4000):
        if len(found) >= count:
            break
        t = int(rng.integers(config.min_gap, frames))
        earlier = poses[: t - config.min_gap + 1]
        distance = np.linalg.norm(earlier[:, :2] - poses[t, :2], axis=1)
        turn = np.abs((earlier[:, 2] - poses[t, 2] + np.pi) % (2 * np.pi) - np.pi)
        position_miss = (distance > config.position_tolerance) & (
            distance <= 2 * config.position_tolerance
        )
        heading_miss = (turn > config.heading_tolerance) & (
            turn <= 2 * config.heading_tolerance
        )
        in_position = distance <= config.position_tolerance
        in_heading = turn <= config.heading_tolerance
        candidates = np.flatnonzero(
            (position_miss & in_heading) | (heading_miss & in_position)
        )
        if len(candidates):
            source = int(candidates[-1])
            found.append(
                (
                    source,
                    t,
                    f"d={distance[source]:.2f} "
                    f"turn={math.degrees(turn[source]):.0f}°",
                )
            )
    return found


def main() -> int:
    args = parse_args()
    config = DetectorConfig(
        position_tolerance=args.position_tolerance,
        heading_tolerance=math.radians(args.heading_tolerance_degrees),
        min_gap=args.min_gap,
    )
    rng = np.random.default_rng(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    counts: collections.Counter[str] = collections.Counter()
    bucket_frames: collections.Counter[int] = collections.Counter()
    bucket_difference: dict[int, list[float]] = collections.defaultdict(list)
    random_difference: list[float] = []
    agreement: collections.Counter[str] = collections.Counter()
    visit_ages: list[int] = []
    sweep: dict[str, dict[str, Any]] = {}
    examples: dict[int, list[tuple[np.ndarray, np.ndarray, str]]] = (
        collections.defaultdict(list)
    )
    misses: list[tuple[np.ndarray, np.ndarray, str]] = []
    total_frames = 0

    for directory in args.episodes:
        poses = np.load(directory / "pose.npy")
        frames = np.load(directory / "frames.npy")
        meta = json.loads((directory / "meta.json").read_text())
        total_frames += len(poses)
        labels = detect_revisits(poses, config)
        counts.update(labels.counts())
        agreement.update(scripted_agreement(labels, meta))

        revisits = np.flatnonzero(labels.kind == REVISIT)
        buckets = gap_bucket(labels.gap[revisits], config.min_gap)
        differences = pixel_difference(frames, revisits, labels.source[revisits])
        visit_ages.extend(labels.visit_age[revisits].tolist())
        for bucket, difference in zip(buckets, differences, strict=True):
            bucket_frames[int(bucket)] += 1
            bucket_difference[int(bucket)].append(float(difference))
        pairs = rng.integers(0, len(frames), size=(200, 2))
        random_difference.extend(
            pixel_difference(frames, pairs[:, 0], pairs[:, 1]).tolist()
        )

        for bucket in np.unique(buckets):
            chosen = revisits[buckets == bucket]
            for t in rng.choice(chosen, size=min(2, len(chosen)), replace=False):
                source = labels.source[t]
                examples[int(bucket)].append(
                    (frames[source], frames[t], f"gap {labels.gap[t]}")
                )
        for source, t, caption in near_misses(poses, config, rng, 1):
            misses.append((frames[source], frames[t], caption))

        if not args.no_sweep:
            for position in SWEEP_POSITIONS:
                for heading in SWEEP_HEADINGS:
                    key = f"pos{position}_head{heading}"
                    swept = detect_revisits(
                        poses,
                        DetectorConfig(position, math.radians(heading), config.min_gap),
                    )
                    hits = np.flatnonzero(swept.kind == REVISIT)
                    entry = sweep.setdefault(key, {"revisit_frames": 0, "diffs": []})
                    entry["revisit_frames"] += len(hits)
                    entry["diffs"].extend(
                        pixel_difference(frames, hits, swept.source[hits]).tolist()
                    )

    rows = []
    for bucket in sorted(examples):
        picks = examples[bucket]
        order = rng.permutation(len(picks))[: args.pairs_per_bucket]
        rows.append((bucket_label(bucket, config.min_gap), [picks[i] for i in order]))
    rows.append(("near misses", misses[: args.pairs_per_bucket]))
    preview.write_pair_grid(args.output_dir / "spot_check.png", rows)

    report = {
        "episodes": [str(path) for path in args.episodes],
        "frames": total_frames,
        "config": {
            "position_tolerance": config.position_tolerance,
            "heading_tolerance_degrees": math.degrees(config.heading_tolerance),
            "min_gap": config.min_gap,
        },
        "label_counts": dict(counts),
        "label_fractions": {k: v / total_frames for k, v in counts.items()},
        "revisit_frames_per_bucket": {
            bucket_label(b, config.min_gap): bucket_frames[b]
            for b in sorted(bucket_frames)
        },
        "median_pixel_difference": {
            "revisit_pairs_by_bucket": {
                bucket_label(b, config.min_gap): median_or_none(bucket_difference[b])
                for b in sorted(bucket_difference)
            },
            "random_pairs": median_or_none(random_difference),
        },
        "scripted_returns": dict(agreement),
        "revisit_visit_age_percentiles": {
            str(p): float(np.percentile(visit_ages, p)) if visit_ages else None
            for p in (50, 90, 99)
        },
        "tolerance_sweep": {
            key: {
                "revisit_fraction": entry["revisit_frames"] / total_frames,
                "median_pixel_difference": median_or_none(entry["diffs"]),
                "p90_pixel_difference": (
                    float(np.percentile(entry["diffs"], 90)) if entry["diffs"] else None
                ),
            }
            for key, entry in sweep.items()
        },
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
