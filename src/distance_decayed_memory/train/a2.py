"""Stage A2: streaming memory fine-tuning, one run per policy × budget × seed.

Follows TRACK_A_PLAN.md §5 (A2) and decisions D-004 and D-011:

- ``streams`` episodes run in parallel, each with its own memory policy.
  First episodes end at random lengths, so streams desynchronize and every
  batch mixes short and long histories.
- Each optimizer step advances every stream by one chunk with a single
  forward pass over ``[k clean chunks | 1 noisy chunk]``. The clean chunks are
  the ``k = grad_chunks`` most recent ones (the local window); they get light
  noise augmentation and are recomputed with gradients, so the loss on the
  noisy chunk trains the model both to read the cache and to write useful keys
  and values for the near future. Everything older sits in the policy's cache,
  detached.
- After the step, the oldest local chunk's keys and values (detached) are
  handed to the policy, which compacts under its budget.
- The first ``k`` chunks of each episode are context only (never targets).

Single GPU per run. The run directory mirrors A1: ``config.json``,
``checkpoints/latest.pt``, ``metrics.jsonl``; SIGUSR1 checkpoints and exits 99
for requeue. Stream caches are not checkpointed: after a resume every stream
starts a fresh episode.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import math
import os
import pathlib
import signal
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from distance_decayed_memory.data.shards import SplitReader, stage_split
from distance_decayed_memory.memory import Geometry, make_policy
from distance_decayed_memory.models.dit import (
    DiTConfig,
    ModelCache,
    PixelDiT,
    conditioning_actions,
)
from distance_decayed_memory.models.flow import chunk_noise_levels, to_model_range

REQUEUE_EXIT_CODE = 99
_STOP = False


def request_stop(_signal: int, _frame: Any) -> None:
    global _STOP
    _STOP = True


@dataclass
class A2Config:
    run_dir: str
    train_split: str
    base_checkpoint: str
    policy: str
    policy_config: dict[str, Any] = field(default_factory=dict)
    streams: int = 8
    grad_chunks: int = 2
    learning_rate: float = 5e-5
    betas: tuple[float, float] = (0.9, 0.95)
    warmup_steps: int = 200
    max_steps: int = 20_000
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    noise_schedule: str = "logit_normal"
    augment_max: float = 0.05
    precision: str = "bf16"
    seed: int = 0
    stage_root: str | None = None
    log_every: int = 50
    checkpoint_every: int = 500
    staleness_every: int = 1_000


def _log(path: pathlib.Path, record: dict[str, Any]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(record) + "\n")


def _save(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def learning_rate(config: A2Config, step: int) -> float:
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / config.warmup_steps
    progress = (step - config.warmup_steps) / max(
        1, config.max_steps - config.warmup_steps
    )
    return config.learning_rate * (0.1 + 0.45 * (1 + math.cos(math.pi * progress)))


class Streams:
    """Parallel episodes, their positions, and their streaming caches."""

    def __init__(
        self,
        reader: SplitReader,
        config: A2Config,
        model_config: DiTConfig,
        rng: np.random.Generator,
    ) -> None:
        self.reader = reader
        self.config = config
        self.model_config = model_config
        self.rng = rng
        self.geometry = Geometry(model_config.grid)
        self.chunk = model_config.chunk_frames
        self.episodes: list[dict] = [{} for _ in range(config.streams)]
        self.position = [0] * config.streams  # index of the chunk to predict
        self.end = [0] * config.streams  # first chunk index past the end
        self.policies: list = [None] * config.streams
        self.completed = 0
        for index in range(config.streams):
            self._start(index, first=True)

    def _policy(self):
        return make_policy(
            self.config.policy, geometry=self.geometry, **self.config.policy_config
        )

    def _start(self, index: int, first: bool = False) -> None:
        episode_index = int(self.rng.integers(len(self.reader)))
        episode = self.reader.episode(episode_index, ("frames", "actions"))
        chunks = len(episode["frames"]) // self.chunk
        minimum = self.config.grad_chunks + 1
        end = int(self.rng.integers(minimum, chunks + 1)) if first else chunks
        self.episodes[index] = episode
        self.position[index] = self.config.grad_chunks
        self.end[index] = end
        self.policies[index] = self._policy()

    def batch(self, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Frames ``[S, (k+1)·chunk, …]``, conditioning actions, and start frames."""
        k, chunk = self.config.grad_chunks, self.chunk
        frames, actions, starts = [], [], []
        for index, episode in enumerate(self.episodes):
            first = (self.position[index] - k) * chunk
            stop = (self.position[index] + 1) * chunk
            frames.append(torch.from_numpy(np.asarray(episode["frames"][first:stop])))
            clip_actions = torch.from_numpy(
                np.asarray(episode["actions"][first:stop]).astype(np.int64)
            )
            before = (
                int(episode["actions"][first - 1])
                if first
                else self.model_config.no_action
            )
            actions.append(
                conditioning_actions(clip_actions[None], torch.tensor([before]))[0]
            )
            starts.append(first)
        return (
            to_model_range(torch.stack(frames).to(device)),
            torch.stack(actions).to(device),
            torch.tensor(starts, device=device),
        )

    def advance(self, k_chunk: torch.Tensor, v_chunk: torch.Tensor) -> None:
        """Hand each stream's oldest local chunk to its policy and move on."""
        k, chunk = self.config.grad_chunks, self.chunk
        for index in range(len(self.episodes)):
            start = (self.position[index] - k) * chunk
            policy = self.policies[index]
            policy.append(k_chunk[index], v_chunk[index], None, start)
            policy.compact()
            budget = getattr(policy, "budget_tokens", None)
            if budget is not None and policy.total_tokens() > budget:
                raise AssertionError(f"{policy.name} exceeded its budget")
            self.position[index] += 1
            if self.position[index] >= self.end[index]:
                self.completed += 1
                self._start(index)

    def policy_cache(self) -> ModelCache | None:
        return ModelCache.from_policies(self.policies)

    def history_frames(self) -> list[int]:
        return [p * self.chunk for p in self.position]


def train(config: A2Config) -> int:
    signal.signal(signal.SIGUSR1, request_stop)
    device = (
        torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    torch.manual_seed(config.seed)
    run_dir = pathlib.Path(config.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    split = pathlib.Path(config.train_split)
    if config.stage_root:
        split = stage_split(split, pathlib.Path(config.stage_root))
    reader = SplitReader(split)

    base = torch.load(config.base_checkpoint, map_location="cpu", weights_only=False)
    model_config = DiTConfig(**base["model_config"])
    model = PixelDiT(model_config)
    model.load_state_dict(base["ema"])
    model.to(device).train()
    ema = copy.deepcopy(model).requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, betas=config.betas
    )
    step = 0
    checkpoint_path = run_dir / "checkpoints" / "latest.pt"
    if checkpoint_path.is_file():
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        ema.load_state_dict(state["ema"])
        optimizer.load_state_dict(state["optimizer"])
        step = int(state["step"])
    record = {
        "config": dataclasses.asdict(config),
        "model_config": dataclasses.asdict(model_config),
        "git_sha": os.environ.get("DD_MEMORY_GIT_SHA", "unknown"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "restart_count": int(os.environ.get("SLURM_RESTART_COUNT", "0")),
        "resumed_from_step": step,
    }
    _log(run_dir / "attempts.jsonl", record)
    (run_dir / "config.json").write_text(json.dumps(record, indent=2) + "\n")

    rng = np.random.default_rng([config.seed, step])
    streams = Streams(reader, config, model_config, rng)
    chunk, k = model_config.chunk_frames, config.grad_chunks
    autocast = torch.autocast(
        device.type, dtype=torch.bfloat16, enabled=config.precision == "bf16"
    )
    window_start = time.perf_counter()

    def checkpoint() -> None:
        _save(
            checkpoint_path,
            {
                "step": step,
                "model": model.state_dict(),
                "ema": ema.state_dict(),
                "optimizer": optimizer.state_dict(),
                "model_config": dataclasses.asdict(model_config),
                "policy": config.policy,
                "policy_config": config.policy_config,
            },
        )

    while step < config.max_steps:
        frames, prev, starts = streams.batch(device)
        s = frames.shape[0]
        augment = torch.rand(s, k, device=device) * config.augment_max
        target = chunk_noise_levels(
            s, chunk, chunk, config.noise_schedule, device=device
        )[:, :1]
        levels = torch.cat((augment, target), dim=1).repeat_interleave(chunk, dim=1)
        noise = torch.randn_like(frames)
        tau = levels[..., None, None, None]
        noisy = (1 - tau) * frames + tau * noise
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(config, step)
        cache = streams.policy_cache()
        with autocast:
            velocity, keys, values = model(
                noisy, levels, prev, starts, cache, return_kv=True
            )
            loss = F.mse_loss(
                velocity[:, -chunk:].float(), (noise - frames)[:, -chunk:].float()
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        with torch.no_grad():
            for target_p, source in zip(
                ema.parameters(), model.parameters(), strict=True
            ):
                target_p.lerp_(source, 1 - config.ema_decay)
        per_chunk = chunk * model_config.tokens_per_frame
        history = streams.history_frames()
        streams.advance(
            keys[..., :per_chunk, :].detach(), values[..., :per_chunk, :].detach()
        )
        step += 1

        if step % config.log_every == 0:
            elapsed = time.perf_counter() - window_start
            tokens = [policy.total_tokens() for policy in streams.policies]
            record = {
                "step": step,
                "loss": loss.item(),
                "lr": learning_rate(config, step - 1),
                "grad_norm": float(grad_norm),
                "steps_per_second": config.log_every / elapsed,
                "mean_history_frames": float(np.mean(history)),
                "mean_cache_tokens": float(np.mean(tokens)),
                "max_cache_tokens": int(max(tokens)),
                "episodes_completed": streams.completed,
            }
            if device.type == "cuda":
                record["max_memory_gb"] = torch.cuda.max_memory_allocated() / 2**30
            _log(run_dir / "metrics.jsonl", record)
            print(json.dumps(record), flush=True)
            window_start = time.perf_counter()
        if step % config.staleness_every == 0:
            value = staleness(model, streams, device, autocast)
            if value is not None:
                _log(run_dir / "metrics.jsonl", {"step": step, **value})
        if step % config.checkpoint_every == 0:
            checkpoint()
        if _STOP:
            checkpoint()
            print(f"Stopped at step {step} for requeue", flush=True)
            return REQUEUE_EXIT_CODE

    checkpoint()
    _save(
        run_dir / "checkpoints" / f"step-{step:07d}.pt",
        torch.load(checkpoint_path, map_location="cpu", weights_only=False),
    )
    return 0


@torch.no_grad()
def staleness(model, streams: Streams, device, autocast) -> dict[str, float] | None:
    """How much stream 0's newest full-fidelity cached chunk changes if re-encoded.

    The stored keys were written by an earlier step (older weights, noise
    augmentation); re-encoding the same frames cleanly with the current
    weights against the older part of the cache shows how stale they are.
    """
    policy = streams.policies[0]
    chunk = streams.chunk
    fresh = [b for b in policy.blocks if b.level == 0 and not b.sink]
    if len(fresh) < chunk:
        return None
    newest = fresh[-chunk:]
    start = newest[0].t0
    if [b.t0 for b in newest] != list(range(start, start + chunk)):
        return None
    older = [b for b in policy.blocks if b.newest < start]
    part = None
    if older:
        weight = torch.cat(
            [
                torch.full((b.tokens,), float(b.volume), dtype=torch.float64)
                for b in older
            ]
        )
        part = (
            torch.cat([b.k for b in older], dim=-2),
            torch.cat([b.v for b in older], dim=-2),
            torch.cat([b.pos for b in older]),
            weight,
        )
    cache = ModelCache.from_parts([part]) if part is not None else None
    episode = streams.episodes[0]
    frames = to_model_range(
        torch.from_numpy(np.asarray(episode["frames"][start : start + chunk]))[None].to(
            device
        )
    )
    actions = torch.from_numpy(
        np.asarray(episode["actions"][start : start + chunk]).astype(np.int64)
    )[None]
    before = int(episode["actions"][start - 1]) if start else model.config.no_action
    prev = conditioning_actions(actions, torch.tensor([before])).to(device)
    zero = torch.zeros(1, chunk, device=device)
    with autocast:
        _, keys, _ = model(frames, zero, prev, start, cache, return_kv=True)
    stored = torch.cat([b.k for b in newest], dim=-2).float()
    difference = (keys[0].float() - stored).norm() / stored.norm().clamp_min(1e-8)
    return {
        "staleness_relative_key_change": float(difference),
        "staleness_chunk_age_frames": float(policy.now - start),
    }
