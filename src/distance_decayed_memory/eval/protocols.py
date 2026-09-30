"""Evaluation protocols P1 and P2 (TRACK_A_PLAN.md §7).

**P1 — observe then predict.** The cache is built from *true* frames up to
shortly before each scripted return; the model then generates the next frames
from actions alone. Each episode is encoded once, and the cache is forked at
every return point, so the prefix is never re-encoded.

**P2 — full rollout.** From the first true frames and the action sequence, the
model generates the rest of the episode.

Both use :class:`~distance_decayed_memory.models.flow.StreamingCache` with the
shared local window (D-011), so evaluation reads the same cache structure as
A2 training. Every scored frame is returned with its revisit label from the
pose-based detector (D-009), so results can be bucketed by gap.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
import torch

from distance_decayed_memory.data.revisit_detector import gap_bucket
from distance_decayed_memory.eval.metrics import FrameMetrics
from distance_decayed_memory.models.dit import PixelDiT, conditioning_actions
from distance_decayed_memory.models.flow import (
    StreamingCache,
    encode_chunk,
    sample_chunk,
    to_model_range,
    to_uint8,
)

RECORD_FIELDS = (
    "episode",
    "frame",
    "kind",
    "gap",
    "bucket",
    "visit_age",
    "psnr",
    "ssim",
    "lpips",
    "window_start",
)


@dataclass(frozen=True)
class ProtocolConfig:
    steps: int = 16  # Euler steps per generated chunk
    local_chunks: int = 2  # D-011
    context_before: int = 16  # P1: true frames observed before a return
    generate_frames: int = 32  # P1: frames generated per return
    context_frames: int = 16  # P2: true frames given at the start
    ablation: str | None = None  # sanity check 5: None, "drop", or "shuffle"
    seed: int = 0


class Records:
    """Column store of per-frame results."""

    def __init__(self) -> None:
        self.columns: dict[str, list] = {name: [] for name in RECORD_FIELDS}

    def add(self, **values: np.ndarray) -> None:
        length = len(values["frame"])
        for name in RECORD_FIELDS:
            value = values.get(name)
            if value is None:
                value = np.full(length, np.nan)
            self.columns[name].append(np.asarray(value))

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            name: (np.concatenate(parts) if parts else np.zeros(0))
            for name, parts in self.columns.items()
        }


def _labels(episode: dict, frames: np.ndarray) -> dict[str, np.ndarray]:
    gap = np.asarray(episode["revisit_gap"])[frames]
    return {
        "kind": np.asarray(episode["revisit_kind"])[frames],
        "gap": gap,
        "bucket": gap_bucket(gap),
        "visit_age": np.asarray(episode["visit_age"])[frames],
    }


def _device_generator(seed: int, device) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(int(seed))


class _Tracker:
    """Cache size statistics across an evaluation."""

    def __init__(self) -> None:
        self.tokens: list[int] = []

    def observe(self, cache: StreamingCache) -> None:
        self.tokens.extend(policy.total_tokens() for policy in cache.policies)

    def summary(self) -> dict[str, float]:
        if not self.tokens:
            return {"mean_policy_tokens": 0.0, "max_policy_tokens": 0}
        return {
            "mean_policy_tokens": float(np.mean(self.tokens)),
            "max_policy_tokens": int(max(self.tokens)),
        }


def p1_windows(
    episode: dict, chunk: int, config: ProtocolConfig
) -> list[tuple[int, int]]:
    """Chunk-aligned ``(start, return_frame)`` generation windows, one per return.

    Only completed scripted returns count. A window starts ``context_before``
    frames before the return (rounded down to a chunk) and never before the
    local window has filled. Returns whose window would run past the episode
    are skipped, as are duplicates of an earlier window's start.
    """
    length = len(episode["frames"])
    earliest = config.local_chunks * chunk
    windows, seen = [], set()
    for event in episode["meta"].get("events", []):
        if event.get("status") != "completed":
            continue
        start = (event["return_frame"] - config.context_before) // chunk * chunk
        start = max(start, earliest)
        if start + config.generate_frames > length or start in seen:
            continue
        seen.add(start)
        windows.append((start, event["return_frame"]))
    return sorted(windows)


@torch.no_grad()
def evaluate_p1(
    model: PixelDiT,
    episodes: Iterable[tuple[int, dict]],
    make_policy: Callable[[], object],
    config: ProtocolConfig,
    metrics: FrameMetrics,
    device,
    autocast=None,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Score generated frames after each scripted return point, per episode."""
    chunk = model.config.chunk_frames
    no_action = model.config.no_action
    records, tracker = Records(), _Tracker()
    autocast = autocast or torch.autocast(device.type, enabled=False)
    generated_frames = 0
    for episode_index, episode in episodes:
        windows = p1_windows(episode, chunk, config)
        if not windows:
            continue
        actions = torch.from_numpy(np.asarray(episode["actions"]).astype(np.int64))
        prev = conditioning_actions(actions[None], torch.tensor([no_action])).to(device)
        frames = episode["frames"]
        cache = StreamingCache([make_policy()], config.local_chunks)
        position = 0
        for start, _ in windows:
            while position < start:
                truth = to_model_range(
                    torch.from_numpy(np.asarray(frames[position : position + chunk]))[
                        None
                    ].to(device)
                )
                with autocast:
                    k, v = encode_chunk(
                        model,
                        truth,
                        prev[:, position : position + chunk],
                        position,
                        cache.model_cache(),
                    )
                cache.push(k, v, position)
                tracker.observe(cache)
                position += chunk
            branch = cache.fork()
            generator = _device_generator(
                config.seed * 1_000_003 + episode_index * 10_007 + start, device
            )
            outputs = []
            for offset in range(start, start + config.generate_frames, chunk):
                actions_chunk = prev[:, offset : offset + chunk]
                current = branch.model_cache(config.ablation, None)
                with autocast:
                    sample = sample_chunk(
                        model, actions_chunk, offset, current, config.steps, generator
                    )
                    k, v = encode_chunk(model, sample, actions_chunk, offset, current)
                branch.push(k, v, offset)
                outputs.append(to_uint8(sample.float())[0].cpu())
            generated = torch.cat(outputs)
            index = np.arange(start, start + config.generate_frames)
            truth = torch.from_numpy(np.asarray(frames[start : start + len(index)]))
            values = metrics(generated.to(device), truth.to(device))
            records.add(
                episode=np.full(len(index), episode_index),
                frame=index,
                window_start=np.full(len(index), start),
                **_labels(episode, index),
                **{name: value.numpy() for name, value in values.items()},
            )
            generated_frames += len(index)
    return records.arrays(), {**tracker.summary(), "generated_frames": generated_frames}


@torch.no_grad()
def evaluate_p2(
    model: PixelDiT,
    episodes: list[tuple[int, dict]],
    make_policy: Callable[[], object],
    config: ProtocolConfig,
    metrics: FrameMetrics,
    device,
    autocast=None,
    total_frames: int | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Roll out a batch of equal-length episodes from their first true frames."""
    chunk = model.config.chunk_frames
    no_action = model.config.no_action
    autocast = autocast or torch.autocast(device.type, enabled=False)
    lengths = {len(episode["frames"]) for _, episode in episodes}
    if len(lengths) != 1:
        raise ValueError("P2 batches need equal-length episodes")
    length = total_frames or lengths.pop()
    known = config.context_frames
    if known % chunk or length % chunk:
        raise ValueError("context and length must be multiples of the chunk")
    actions = torch.from_numpy(
        np.stack([np.asarray(e["actions"][:length]) for _, e in episodes]).astype(
            np.int64
        )
    )
    prev = conditioning_actions(actions, torch.full((len(episodes),), no_action)).to(
        device
    )
    cache = StreamingCache([make_policy() for _ in episodes], config.local_chunks)
    tracker = _Tracker()
    generator = _device_generator(config.seed * 1_000_003 + episodes[0][0], device)
    generated = []
    for start in range(0, length, chunk):
        span = slice(start, start + chunk)
        current = cache.model_cache(config.ablation if start >= known else None)
        with autocast:
            if start < known:
                block = to_model_range(
                    torch.from_numpy(
                        np.stack([np.asarray(e["frames"][span]) for _, e in episodes])
                    ).to(device)
                )
            else:
                block = sample_chunk(
                    model, prev[:, span], start, current, config.steps, generator
                )
                generated.append(to_uint8(block.float()).cpu())
            k, v = encode_chunk(model, block, prev[:, span], start, current)
        cache.push(k, v, start)
        tracker.observe(cache)
    video = torch.cat(generated, dim=1)
    records = Records()
    index = np.arange(known, length)
    for row, (episode_index, episode) in enumerate(episodes):
        truth = torch.from_numpy(np.asarray(episode["frames"][known:length]))
        values = metrics(video[row].to(device), truth.to(device))
        records.add(
            episode=np.full(len(index), episode_index),
            frame=index,
            window_start=np.full(len(index), known),
            **_labels(episode, index),
            **{name: value.numpy() for name, value in values.items()},
        )
    stats = {**tracker.summary(), "generated_frames": len(index) * len(episodes)}
    return records.arrays(), stats
