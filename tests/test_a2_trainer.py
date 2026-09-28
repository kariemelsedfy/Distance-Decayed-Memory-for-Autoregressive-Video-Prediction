from __future__ import annotations

import json
import pathlib

import pytest

torch = pytest.importorskip("torch")

from distance_decayed_memory.train import a1, a2  # noqa: E402
from tests.test_a1_trainer import config as a1_config  # noqa: E402

PER_FRAME = 16  # tiny model: 64×64 frames, 16×16 patches


@pytest.fixture(scope="module")
def base(tmp_path_factory) -> tuple[pathlib.Path, pathlib.Path]:
    root = tmp_path_factory.mktemp("a2")
    config = a1_config(root, max_steps=2, sample_every=100, val_every=100)
    assert a1.train(config) == 0
    return pathlib.Path(config.run_dir) / "checkpoints" / "latest.pt", pathlib.Path(
        config.train_split
    )


def a2_config(tmp_path, base, policy, **overrides) -> a2.A2Config:
    checkpoint, split = base
    values = dict(
        run_dir=str(tmp_path / f"a2-{policy}"),
        train_split=str(split),
        base_checkpoint=str(checkpoint),
        policy=policy,
        policy_config={"budget_tokens": 3 * PER_FRAME, "horizon": 64},
        streams=3,
        max_steps=12,
        warmup_steps=2,
        precision="fp32",
        log_every=4,
        checkpoint_every=6,
        staleness_every=4,
    )
    if policy == "full":
        values["policy_config"] = {}
    if policy in {"decay_continuous", "relic_discrete"}:
        values["policy_config"] = {**values["policy_config"], "window": 1}
    values.update(overrides)
    return a2.A2Config(**values)


@pytest.mark.parametrize("policy", ["window", "decay_continuous", "full"])
def test_a2_streams_train_and_respect_budget(tmp_path, base, policy) -> None:
    config = a2_config(tmp_path, base, policy)
    assert a2.train(config) == 0
    run = pathlib.Path(config.run_dir)
    records = [
        json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()
    ]
    losses = [r for r in records if "loss" in r]
    assert [r["step"] for r in losses] == [4, 8, 12]
    assert all(r["loss"] == r["loss"] for r in losses)  # finite, not NaN
    if policy != "full":
        assert all(r["max_cache_tokens"] <= 3 * PER_FRAME for r in losses)
    assert losses[-1]["episodes_completed"] >= 1
    state = torch.load(run / "checkpoints" / "latest.pt", weights_only=False)
    assert state["step"] == 12 and state["policy"] == policy


def test_a2_resumes_and_stops_for_requeue(tmp_path, base) -> None:
    config = a2_config(tmp_path, base, "window", max_steps=6)
    assert a2.train(config) == 0
    a2.request_stop(0, None)
    try:
        assert a2.train(a2_config(tmp_path, base, "window", max_steps=10)) == 99
    finally:
        a2._STOP = False
    run = pathlib.Path(config.run_dir)
    attempts = [
        json.loads(x) for x in (run / "attempts.jsonl").read_text().splitlines()
    ]
    assert attempts[-1]["resumed_from_step"] == 6
    state = torch.load(run / "checkpoints" / "latest.pt", weights_only=False)
    assert state["step"] == 7


def test_rollout_local_window_matches_training_structure(base) -> None:
    from distance_decayed_memory.memory import Geometry, make_policy
    from distance_decayed_memory.models.dit import DiTConfig, PixelDiT
    from distance_decayed_memory.models.flow import StreamingCache

    checkpoint, _ = base
    state = torch.load(checkpoint, weights_only=False)
    model = PixelDiT(DiTConfig(**state["model_config"]))
    geometry = Geometry(model.config.grid)
    cache = StreamingCache(
        [make_policy("window", budget_tokens=2 * PER_FRAME, geometry=geometry)], 2
    )
    chunk = model.config.chunk_frames * PER_FRAME
    shape = (1, model.config.layers, model.config.heads, chunk, model.config.head_dim)
    for start in range(0, 16, 4):
        cache.push(torch.randn(shape), torch.randn(shape), start)
    # Two local chunks at full fidelity plus the policy's two-frame window.
    assert cache.model_cache().k.shape[-2] == 2 * chunk + 2 * PER_FRAME
    assert cache.policies[0].now == 7
