from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from distance_decayed_memory.data.revisit_detector import (
    NOVEL,
    RECENT,
    REVISIT,
    DetectorConfig,
    bucket_label,
    detect_revisits,
    gap_bucket,
)
from distance_decayed_memory.data.revisit_script import RevisitScript, ScriptConfig
from distance_decayed_memory.data.toy_maze import TOY_LAYOUT_9X9, ToyMaze

ROOT = Path(__file__).resolve().parents[1]


def still(x: float, y: float, heading: float, frames: int) -> np.ndarray:
    return np.tile([x, y, heading], (frames, 1))


def walk(start, end, heading: float, frames: int) -> np.ndarray:
    xy = np.linspace(start, end, frames)
    return np.column_stack([xy, np.full(frames, heading)])


def reference(poses: np.ndarray, config: DetectorConfig) -> tuple[list, list]:
    """Direct transcription of the detector's definition, for comparison."""

    def match(i: int, j: int) -> bool:
        close = math.dist(poses[i, :2], poses[j, :2]) <= config.position_tolerance
        turn = abs((poses[i, 2] - poses[j, 2] + math.pi) % (2 * math.pi) - math.pi)
        return close and turn <= config.heading_tolerance

    kinds, gaps = [], []
    for t in range(len(poses)):
        start = t
        while start > 0 and match(t, start - 1):
            start -= 1
        earlier = [j for j in range(start) if match(t, j)]
        if not earlier:
            kinds.append(NOVEL)
            gaps.append(-1)
        elif t - earlier[-1] < config.min_gap:
            kinds.append(RECENT)
            gaps.append(-1)
        else:
            kinds.append(REVISIT)
            gaps.append(t - earlier[-1])
    return kinds, gaps


def test_standing_still_is_never_a_revisit() -> None:
    labels = detect_revisits(still(1.5, 1.5, 0.0, 100))
    assert np.all(labels.kind == NOVEL)
    np.testing.assert_array_equal(labels.visit_age, np.arange(100))


def test_leaving_and_returning_is_a_revisit_with_exact_gap() -> None:
    poses = np.concatenate(
        [
            still(1.5, 1.5, 0.0, 5),
            walk((1.5, 1.5), (5.5, 1.5), 0.0, 20),
            still(5.5, 1.5, math.pi, 30),
            walk((5.5, 1.5), (1.5, 1.5), math.pi, 20),
            still(1.5, 1.5, 0.0, 3),
        ]
    )
    labels = detect_revisits(poses)
    back = 75
    assert labels.kind[back] == REVISIT
    # Walk frames are 0.21 cell apart, so the latest match is the outbound
    # walk's second frame (frame 6), not its first.
    assert labels.source[back] == 6
    assert labels.gap[back] == back - 6
    assert labels.visit_age[back] == 0
    assert labels.kind[back + 1] == REVISIT
    assert labels.visit_age[back + 1] == 1


def test_brief_departure_is_recent_not_revisit() -> None:
    poses = np.concatenate(
        [
            still(1.5, 1.5, 0.0, 10),
            still(1.5, 1.5, math.radians(40), 4),
            still(1.5, 1.5, 0.0, 2),
        ]
    )
    labels = detect_revisits(poses)
    assert labels.kind[14] == RECENT
    assert labels.gap[14] == -1


def test_same_place_facing_another_way_is_novel() -> None:
    poses = np.concatenate(
        [
            still(1.5, 1.5, 0.0, 5),
            walk((1.5, 1.5), (4.5, 1.5), 0.0, 20),
            walk((4.5, 1.5), (1.5, 1.5), math.pi, 20),
        ]
    )
    labels = detect_revisits(poses)
    assert labels.kind[-1] == NOVEL


def test_heading_match_wraps_around_pi() -> None:
    poses = np.concatenate(
        [
            still(1.5, 1.5, math.radians(179), 1),
            walk((1.5, 1.5), (5.5, 1.5), 0.0, 30),
            still(1.5, 1.5, math.radians(-179), 1),
        ]
    )
    labels = detect_revisits(poses)
    assert labels.kind[-1] == REVISIT
    assert labels.source[-1] == 0


def test_corridor_rewalk_keeps_a_steady_gap() -> None:
    poses = np.concatenate(
        [
            walk((0.5, 0.5), (8.5, 0.5), 0.0, 33),
            walk((8.5, 0.5), (8.5, 6.5), math.pi / 2, 40),
            walk((8.5, 6.5), (0.5, 0.5), math.pi, 40),
            walk((0.5, 0.5), (8.5, 0.5), 0.0, 33),
        ]
    )
    labels = detect_revisits(poses)
    second = slice(113 + 2, 113 + 31)
    assert np.all(labels.kind[second] == REVISIT)
    # Frame k of the first pass is 0.25 cell behind frame 113 + k, so the latest
    # match is frame k + 1 and the gap is 112.
    assert np.all(labels.gap[second] == 112)


@pytest.mark.parametrize("seed", range(5))
def test_matches_reference_on_random_walks(seed: int) -> None:
    rng = np.random.default_rng(seed)
    steps = rng.normal(0, 0.2, size=(300, 2))
    xy = np.clip(np.cumsum(steps, axis=0), -1.5, 1.5)
    heading = np.cumsum(rng.normal(0, 0.4, size=300))
    poses = np.column_stack([xy, (heading + np.pi) % (2 * np.pi) - np.pi])
    config = DetectorConfig()
    labels = detect_revisits(poses, config)
    kinds, gaps = reference(poses, config)
    np.testing.assert_array_equal(labels.kind, kinds)
    np.testing.assert_array_equal(labels.gap, gaps)
    assert {NOVEL, REVISIT} <= set(labels.kind.tolist())


def test_scripted_returns_are_detected_in_toy_episodes() -> None:
    found = 0
    for seed in range(4):
        config = ScriptConfig(no_revisit_fraction=0.0)
        sim = ToyMaze(TOY_LAYOUT_9X9, (6.5, 7.5))
        script = RevisitScript(TOY_LAYOUT_9X9, np.random.default_rng(seed), config)
        poses = np.empty((config.episode_frames, 3))
        for t in range(config.episode_frames):
            poses[t] = (*sim.position, sim.heading)
            sim.step(script.act(t, sim.position, sim.heading))
        labels = detect_revisits(poses)
        for event in script.finish():
            if event.status != "completed":
                continue
            frame = event.return_frame
            assert labels.kind[frame] in (REVISIT, RECENT)
            if labels.kind[frame] == REVISIT:
                assert labels.source[frame] >= event.anchor_frame
                found += 1
    assert found > 0


def test_gap_buckets_are_powers_of_two() -> None:
    gaps = np.asarray([-1, 0, 15, 16, 31, 32, 63, 64, 2047, 2048, 4095])
    np.testing.assert_array_equal(
        gap_bucket(gaps), [-1, -1, -1, 0, 0, 1, 1, 2, 6, 7, 7]
    )
    assert bucket_label(0) == "[16,32)"
    assert bucket_label(7) == "[2048,4096)"


def test_rejects_malformed_poses() -> None:
    with pytest.raises(ValueError, match="poses must be"):
        detect_revisits(np.zeros((10, 2)))


def test_spot_check_script_writes_report(tmp_path: Path, monkeypatch) -> None:
    episode = tmp_path / "episode"
    episode.mkdir()
    poses = np.concatenate(
        [
            walk((0.5, 0.5), (8.5, 0.5), 0.0, 33),
            walk((8.5, 0.5), (0.5, 0.5), math.pi, 33),
            walk((0.5, 0.5), (8.5, 0.5), 0.0, 33),
        ]
    ).astype(np.float32)
    frames = np.random.default_rng(0).integers(
        0, 256, (len(poses), 64, 64, 3), dtype=np.uint8
    )
    np.save(episode / "pose.npy", poses)
    np.save(episode / "frames.npy", frames)
    (episode / "meta.json").write_text(json.dumps({"events": []}))

    path = ROOT / "scripts" / "data" / "spot_check_revisits.py"
    spec = importlib.util.spec_from_file_location("spot_check_revisits", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path / "out"
    monkeypatch.setattr(
        sys, "argv", ["spot_check", str(episode), "--output-dir", str(output)]
    )
    assert module.main() == 0
    report = json.loads((output / "report.json").read_text())
    assert report["frames"] == len(poses)
    assert report["label_counts"]["revisit"] > 0
    assert "[64,128)" in report["revisit_frames_per_bucket"]
    assert (output / "spot_check.png").is_file()
