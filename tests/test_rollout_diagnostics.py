from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from distance_decayed_memory.eval import diagnostics as dx  # noqa: E402
from distance_decayed_memory.train import a1  # noqa: E402
from tests.test_evaluation import TINY, make_split, run_main  # noqa: E402


def textured(size: int = 64, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, (size // 4, size // 4, 3)).astype(np.uint8)
    return np.kron(base, np.ones((4, 4, 1), dtype=np.uint8))


@pytest.mark.parametrize("pixels", [-5, -2, 3, 7])
def test_horizontal_shift_recovers_a_roll(pixels) -> None:
    gray = dx.grayscale(textured())
    moved = np.roll(gray, pixels, axis=1)
    assert dx.horizontal_shift(gray, moved) == pytest.approx(pixels, abs=0.3)


def test_shifts_detail_and_motion_shapes() -> None:
    clip = np.stack([np.roll(textured(), 2 * t, axis=1) for t in range(5)])
    values = dx.shifts(clip[None])
    assert values.shape == (1, 5) and np.isnan(values[0, 0])
    assert np.allclose(values[0, 1:], 2, atol=0.3)
    assert dx.motion(clip)[1:].min() > 0 and np.isnan(dx.motion(clip)[0])
    flat = np.full_like(clip, 90)
    assert dx.detail(flat).max() == 0 and dx.detail(clip).min() > 10


def test_stuck_flags_a_static_flat_rollout() -> None:
    moving = np.stack([np.roll(textured(), 3 * t, axis=1) for t in range(16)])
    flat = np.full_like(moving, 120)
    assert dx.stuck(flat, moving, 8)[0] is True
    assert dx.stuck(moving, moving, 8)[0] is False
    still = np.repeat(moving[:1], 16, axis=0)
    assert dx.stuck(flat, still, 8)[0] is None  # a pause cannot be classified


def test_override_actions_keeps_the_context() -> None:
    prev = np.arange(12).reshape(2, 6)
    out = dx.override_actions(prev, 4, 2)
    assert np.array_equal(out[:, :4], prev[:, :4]) and np.all(out[:, 4:] == 2)
    assert np.array_equal(dx.override_actions(prev, 4, None), prev)
    groups = dx.shift_by_action(np.array([1.0, 3.0, np.nan]), np.array([2, 2, 3]), 6)
    assert groups == {2: (2.0, 1.0, 2)}


def test_rollout_diagnostics_end_to_end(tmp_path, monkeypatch) -> None:
    val = make_split(tmp_path, "val", episodes=3)
    config = a1.A1Config(
        run_dir=str(tmp_path / "a1"),
        train_split=str(val),
        val_split=str(val),
        model_overrides=TINY,
        clip_frames=8,
        batch_size=2,
        warmup_steps=1,
        max_steps=1,
        num_workers=0,
        precision="fp32",
        val_every=100,
        sample_every=100,
        log_every=1,
        checkpoint_every=1,
        keep_every=100,
    )
    assert a1.train(config) == 0
    out = tmp_path / "diagnostics"
    assert (
        run_main(
            monkeypatch,
            "scripts/eval/rollout_diagnostics.py",
            "--checkpoint",
            str(tmp_path / "a1" / "checkpoints" / "latest.pt"),
            "--split-dir",
            str(val),
            "--output-dir",
            str(out),
            "--clips",
            "3",
            "--seeds",
            "2",
            "--action-clips",
            "2",
            "--context",
            "8",
            "--frames",
            "16",
            "--tail",
            "4",
            "--steps",
            "2",
            "--batch",
            "2",
            "--sheets",
            "2",
            "--precision",
            "fp32",
        )
        == 0
    )
    summary = json.loads((out / "diagnostics.json").read_text())
    assert len(summary["psnr_by_generated_frame"]["median"]) == 8
    assert set(summary["shift_generated_by_override"]) == {
        "true",
        "noop",
        "forward",
        "left",
        "right",
    }
    arrays = np.load(out / "diagnostics.npz")
    assert arrays["psnr"].shape == (2, 3, 16)
    # Context frames are copied through, so they match the truth exactly.
    assert np.all(arrays["psnr"][:, :, :8] > 60)
    assert arrays["shift_left"].shape == (2, 8)
    assert len(list((out / "sheets").glob("actions-*.png"))) == 2
    assert len(list((out / "sheets").glob("*.png"))) == 4
