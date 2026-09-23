from __future__ import annotations

import collections
import math

import numpy as np
import pytest

from distance_decayed_memory.data.navigation import NUM_ACTIONS, wrap_angle
from distance_decayed_memory.data.revisit_script import (
    RevisitScript,
    ScriptConfig,
    sample_log_uniform_gap,
)
from distance_decayed_memory.data.toy_maze import TOY_LAYOUT_9X9, ToyMaze


def run_episode(seed: int, config: ScriptConfig | None = None):
    config = config or ScriptConfig()
    rng = np.random.default_rng(seed)
    sim = ToyMaze(TOY_LAYOUT_9X9, (6.5, 7.5), heading=float(rng.uniform(-3, 3)))
    script = RevisitScript(TOY_LAYOUT_9X9, rng, config)
    actions = []
    for t in range(config.episode_frames):
        actions.append(script.act(t, sim.position, sim.heading))
        sim.step(actions[-1])
    return script, script.finish(), actions


def test_log_uniform_gap_sampler_is_uniform_in_log_space() -> None:
    rng = np.random.default_rng(0)
    gaps = np.asarray([sample_log_uniform_gap(rng, 16, 2048) for _ in range(20_000)])
    assert gaps.min() >= 16 and gaps.max() <= 2048
    octaves = np.bincount(np.log2(gaps).astype(int))[4:]
    assert len(octaves) == 8
    assert octaves[:7].min() > 0.8 * octaves[:7].mean()
    assert octaves[:7].max() < 1.2 * octaves[:7].mean()


@pytest.mark.parametrize("seed", range(8))
def test_completed_revisits_return_to_anchor_pose(seed: int) -> None:
    config = ScriptConfig(no_revisit_fraction=0.0)
    script, events, actions = run_episode(seed, config)
    assert all(0 <= a < NUM_ACTIONS for a in actions)
    assert len(script.phases) == config.episode_frames
    completed = [e for e in events if e.status == "completed"]
    assert completed, collections.Counter(e.status for e in events)
    for event in completed:
        assert config.min_gap <= event.gap <= config.episode_frames
        assert event.anchor_frame < event.depart_frame <= event.return_frame
        np.testing.assert_allclose(
            event.anchor_position, script.positions[event.anchor_frame]
        )
        assert math.dist(event.return_position, event.anchor_position) <= 0.3
        error = abs(wrap_angle(event.return_heading - event.anchor_heading))
        assert error <= math.radians(15)
    assert {e.status for e in events} <= {
        "completed",
        "early",
        "partial",
        "timeout",
        "unfinished",
    }


def test_events_do_not_overlap() -> None:
    _, events, _ = run_episode(3, ScriptConfig(no_revisit_fraction=0.0))
    closing = [e.return_frame or e.depart_frame for e in events[:-1]]
    for previous_end, event in zip(closing, events[1:], strict=True):
        assert event.scheduled_frame >= (previous_end or 0)


def test_no_revisit_episodes_schedule_nothing() -> None:
    script, events, _ = run_episode(1, ScriptConfig(no_revisit_fraction=1.0))
    assert not script.revisits_enabled
    assert events == []
    assert "return" not in script.phases


def test_script_is_deterministic_for_a_seed() -> None:
    first = run_episode(5)
    second = run_episode(5)
    assert first[2] == second[2]
    assert [e.to_dict() for e in first[1]] == [e.to_dict() for e in second[1]]


def test_exploration_covers_the_maze() -> None:
    script, _, _ = run_episode(2, ScriptConfig(no_revisit_fraction=1.0))
    free = int(TOY_LAYOUT_9X9.sum())
    assert len(script.visited) == free


def test_about_thirty_percent_of_episodes_have_no_scripted_revisits() -> None:
    config = ScriptConfig()
    enabled = [
        RevisitScript(TOY_LAYOUT_9X9, np.random.default_rng(s), config).revisits_enabled
        for s in range(2_000)
    ]
    assert 0.27 < 1 - np.mean(enabled) < 0.33


def test_action_noise_rate_matches_config() -> None:
    config = ScriptConfig(
        no_revisit_fraction=1.0, action_noise=0.0, pause_probability=0.0
    )
    quiet = run_episode(4, config)[2]
    noisy = run_episode(
        4,
        ScriptConfig(no_revisit_fraction=1.0, action_noise=1.0, pause_probability=0),
    )[2]
    assert len(set(quiet)) < NUM_ACTIONS or quiet != noisy
    counts = np.bincount(noisy, minlength=NUM_ACTIONS)
    assert counts.min() > 0.1 * len(noisy)


def test_frames_must_arrive_in_order() -> None:
    script = RevisitScript(TOY_LAYOUT_9X9, np.random.default_rng(0))
    script.act(0, (6.5, 7.5), 0.0)
    with pytest.raises(ValueError, match="Expected frame 1"):
        script.act(2, (6.5, 7.5), 0.0)
