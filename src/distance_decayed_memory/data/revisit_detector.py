"""Pose-based revisit detection and gap bucketing (TRACK_A_PLAN.md §3.3).

The detector works from poses alone, independently of the script that made
the episode, so it also finds natural revisits.

Two poses *match* when their positions are within ``position_tolerance`` cells
and their headings within ``heading_tolerance``. For frame ``t``, the *current
visit* is the unbroken run of matching frames that ends at ``t - 1``: the view
has stayed on screen through it. The source frame ``t'`` is the most recent
match **before** the current visit, so the agent must have left the view and
come back. Then:

- no earlier match outside the current visit: ``NOVEL`` (the control
  condition);
- ``t - t' < min_gap``: ``RECENT`` (left only briefly);
- otherwise ``REVISIT`` with gap ``t - t'``.

``visit_age = t - start of current visit`` counts how long the view has been
continuously visible. It lets evaluation drop frames that are trivial because
the agent has been looking at the same view for a while (for example, a long
pause at a revisited spot).

This refines the plan's wording ("most recent ``t' < t - w_min``"), which
would label a 20-frame pause as a 16-frame revisit although the view never
left the screen.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

NOVEL, RECENT, REVISIT = 0, 1, 2
KIND_NAMES = ("novel", "recent", "revisit")


@dataclass(frozen=True)
class DetectorConfig:
    position_tolerance: float = 0.3
    heading_tolerance: float = math.radians(15.0)
    min_gap: int = 16


@dataclass(frozen=True)
class RevisitLabels:
    """Per-frame labels; ``gap`` and ``source`` are -1 unless ``kind == REVISIT``."""

    kind: np.ndarray
    gap: np.ndarray
    source: np.ndarray
    visit_age: np.ndarray

    def counts(self) -> dict[str, int]:
        return {
            name: int(np.sum(self.kind == value))
            for value, name in enumerate(KIND_NAMES)
        }


def pose_matches(
    poses: np.ndarray, t: int, config: DetectorConfig, stop: int | None = None
) -> np.ndarray:
    """Boolean mask over frames ``[0, stop)`` whose pose matches frame ``t``."""
    stop = t if stop is None else stop
    earlier = poses[:stop]
    delta = earlier[:, :2] - poses[t, :2]
    close = np.einsum("ij,ij->i", delta, delta) <= config.position_tolerance**2
    turn = np.abs((earlier[:, 2] - poses[t, 2] + np.pi) % (2 * np.pi) - np.pi)
    return close & (turn <= config.heading_tolerance)


def detect_revisits(
    poses: np.ndarray, config: DetectorConfig | None = None
) -> RevisitLabels:
    """Label every frame of an episode from its ``[T, 3]`` (x, y, heading) poses."""
    config = config or DetectorConfig()
    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != 3:
        raise ValueError(f"poses must be [T, 3], got {poses.shape}")
    frames = len(poses)
    kind = np.full(frames, NOVEL, dtype=np.int8)
    gap = np.full(frames, -1, dtype=np.int32)
    source = np.full(frames, -1, dtype=np.int32)
    visit_age = np.zeros(frames, dtype=np.int32)
    for t in range(1, frames):
        matches = pose_matches(poses, t, config)
        misses = np.flatnonzero(~matches)
        visit_start = int(misses[-1]) + 1 if len(misses) else 0
        visit_age[t] = t - visit_start
        earlier = np.flatnonzero(matches[:visit_start])
        if not len(earlier):
            continue
        previous = int(earlier[-1])
        if t - previous < config.min_gap:
            kind[t] = RECENT
            continue
        kind[t] = REVISIT
        gap[t] = t - previous
        source[t] = previous
    return RevisitLabels(kind=kind, gap=gap, source=source, visit_age=visit_age)


def gap_bucket(gap: np.ndarray | int, min_gap: int = 16) -> np.ndarray:
    """Power-of-two bucket index: ``[16, 32) -> 0``, ``[32, 64) -> 1``, ...

    Non-positive gaps (non-revisits) map to -1.
    """
    gap = np.asarray(gap)
    bucket = np.full(gap.shape, -1, dtype=np.int32)
    valid = gap >= min_gap
    bucket[valid] = np.floor(np.log2(gap[valid] / min_gap)).astype(np.int32)
    return bucket


def bucket_label(index: int, min_gap: int = 16) -> str:
    low = min_gap * 2**index
    return f"[{low},{2 * low})"
