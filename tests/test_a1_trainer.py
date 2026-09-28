from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from distance_decayed_memory.data.shards import (  # noqa: E402
    ClipDataset,
    ShardWriter,
    build_manifest,
    shard_name,
)
from distance_decayed_memory.train import a1  # noqa: E402

TINY = {"layers": 1, "width": 48, "heads": 2, "patch": 16}


def make_split(root: pathlib.Path, name: str, episodes: int = 2, frames: int = 24):
    split = root / name
    rng = np.random.default_rng(0)
    writer = ShardWriter(split / shard_name(0), episodes * frames)
    for index in range(episodes):
        pose = np.column_stack(
            [np.linspace(0.5, 6.5, frames), np.full(frames, 0.5), np.zeros(frames)]
        ).astype(np.float32)
        writer.add(
            rng.integers(0, 256, (frames, 64, 64, 3), dtype=np.uint8),
            rng.integers(0, 6, frames).astype(np.int8),
            pose,
            {"episode_id": f"{name}-{index}"},
        )
    writer.finish()
    build_manifest(split, {"split": name})
    return split


def config(tmp_path: pathlib.Path, **overrides) -> a1.A1Config:
    train_split = make_split(tmp_path, "train")
    val_split = make_split(tmp_path, "val")
    values = dict(
        run_dir=str(tmp_path / "run"),
        train_split=str(train_split),
        val_split=str(val_split),
        model_overrides=TINY,
        clip_frames=8,
        batch_size=2,
        warmup_steps=2,
        max_steps=6,
        num_workers=0,
        precision="fp32",
        log_every=2,
        val_every=3,
        val_clips=2,
        sample_every=6,
        sample_context=4,
        sample_steps=2,
        checkpoint_every=3,
        keep_every=6,
    )
    values.update(overrides)
    return a1.A1Config(**values)


def read_jsonl(path: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_clip_dataset_reports_preceding_action(tmp_path: pathlib.Path) -> None:
    split = make_split(tmp_path, "clips")
    dataset = ClipDataset(split, clip_frames=8, start_stride=4)
    first, later = dataset[0], dataset[1]
    assert int(first["start"]) == 0 and int(first["prev_action"]) == 6
    episode = dataset.reader.episode(0)
    assert int(later["prev_action"]) == int(episode["actions"][3])


def test_a1_trains_logs_checkpoints_and_resumes(tmp_path: pathlib.Path) -> None:
    run = pathlib.Path(config(tmp_path).run_dir)
    assert a1.train(config(tmp_path)) == 0
    metrics = read_jsonl(run / "metrics.jsonl")
    assert [m["step"] for m in metrics] == [2, 4, 6]
    assert all(np.isfinite(m["loss"]) for m in metrics)
    validation = read_jsonl(run / "validation.jsonl")
    assert {"val_loss", "val_loss_ema"} <= validation[0].keys()
    assert any("sample_psnr_mean" in record for record in validation)
    assert (run / "samples" / "step-0000006.gif").is_file()
    assert (run / "checkpoints" / "step-0000006.pt").is_file()

    assert a1.train(config(tmp_path, max_steps=8)) == 0
    attempts = read_jsonl(run / "attempts.jsonl")
    assert attempts[-1]["resumed_from_step"] == 6
    state = torch.load(run / "checkpoints" / "latest.pt", weights_only=False)
    assert state["step"] == 8


def test_stop_signal_checkpoints_and_requests_requeue(tmp_path: pathlib.Path) -> None:
    run = pathlib.Path(config(tmp_path).run_dir)
    a1.request_stop(0, None)
    try:
        assert a1.train(config(tmp_path)) == a1.REQUEUE_EXIT_CODE
    finally:
        a1._STOP = False
    state = torch.load(run / "checkpoints" / "latest.pt", weights_only=False)
    assert state["step"] == 1


def test_learning_rate_schedule() -> None:
    schedule = a1.A1Config("r", "t", "v", warmup_steps=10, max_steps=110)
    assert a1.learning_rate(schedule, 0) == pytest.approx(2e-5)
    assert a1.learning_rate(schedule, 9) == pytest.approx(2e-4)
    assert a1.learning_rate(schedule, 110) == pytest.approx(2e-5)
