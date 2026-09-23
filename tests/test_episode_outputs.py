from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image

from distance_decayed_memory.data import preview
from distance_decayed_memory.data.revisit_script import RevisitScript, ScriptConfig
from distance_decayed_memory.data.toy_maze import TOY_LAYOUT_9X9, ToyMaze

ROOT = Path(__file__).resolve().parents[1]


def load_generator():
    path = ROOT / "scripts" / "data" / "generate_revisit_episodes.py"
    spec = importlib.util.spec_from_file_location("generate_revisit_episodes", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_phase_run_length_encoding_round_trips() -> None:
    module = load_generator()
    phases = ["explore"] * 3 + ["pause"] * 2 + ["explore"]
    segments = module.run_length(phases)
    assert segments == [["explore", 0, 3], ["pause", 3, 5], ["explore", 5, 6]]
    decoded = [name for name, start, end in segments for _ in range(start, end)]
    assert decoded == phases


def test_previews_render_for_a_toy_episode(tmp_path: Path) -> None:
    frames_count = 400
    config = ScriptConfig(episode_frames=frames_count, no_revisit_fraction=0.0)
    sim = ToyMaze(TOY_LAYOUT_9X9, (6.5, 7.5))
    script = RevisitScript(TOY_LAYOUT_9X9, np.random.default_rng(0), config)
    poses = np.empty((frames_count, 3), dtype=np.float32)
    for t in range(frames_count):
        poses[t] = (*sim.position, sim.heading)
        sim.step(script.act(t, sim.position, sim.heading))
    events = script.finish()
    frames = np.random.default_rng(0).integers(
        0, 256, (frames_count, 64, 64, 3), dtype=np.uint8
    )

    preview.write_map(
        tmp_path / "map.png", TOY_LAYOUT_9X9, poses, script.phases, events
    )
    preview.write_revisit_pairs(tmp_path / "pairs.png", frames, events)
    preview.write_gif(
        tmp_path / "preview.gif",
        frames,
        TOY_LAYOUT_9X9,
        poses,
        script.phases,
        events,
        stride=20,
    )
    assert Image.open(tmp_path / "map.png").size == (9 * 48, 9 * 48)
    assert Image.open(tmp_path / "pairs.png").width > 2 * 64
    with Image.open(tmp_path / "preview.gif") as gif:
        assert gif.n_frames == frames_count // 20
