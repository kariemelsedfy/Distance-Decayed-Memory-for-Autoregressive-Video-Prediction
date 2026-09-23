"""Stage A1: train the shared base model on short clips (TRACK_A_PLAN.md §5).

Single process or DDP (launched with ``scripts/hpc/launch_distributed.sh``).
The run directory holds everything needed to audit or resume a run:

- ``config.json`` (resolved config, git SHA, Slurm job ID, world size);
- ``checkpoints/latest.pt`` (model, EMA, optimizer, step), written atomically,
  plus a kept copy every ``keep_every`` steps;
- ``metrics.jsonl`` (training), ``validation.jsonl`` (EMA and raw model), and
  ``samples/step-NNNNNNN.gif`` (ground truth beside a rollout).

A run resumes automatically from ``latest.pt``. SIGUSR1 (Slurm's pre-timeout
signal) makes every rank stop after the current step; rank 0 writes a
checkpoint and the process exits with code 99 so the batch script requeues.
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
import torch.distributed as dist

from distance_decayed_memory.data.shards import ClipDataset, stage_split
from distance_decayed_memory.memory import Geometry, make_policy
from distance_decayed_memory.models.dit import (
    SIZES,
    DiTConfig,
    PixelDiT,
    conditioning_actions,
)
from distance_decayed_memory.models.flow import (
    flow_matching_loss,
    rollout,
    to_model_range,
    to_uint8,
)

REQUEUE_EXIT_CODE = 99
_STOP = False


def request_stop(_signal: int, _frame: Any) -> None:
    global _STOP
    _STOP = True


@dataclass
class A1Config:
    run_dir: str
    train_split: str
    val_split: str
    size: str = "M"
    model_overrides: dict[str, Any] = field(default_factory=dict)
    clip_frames: int = 64
    batch_size: int = 8  # per GPU
    learning_rate: float = 2e-4
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.0
    warmup_steps: int = 1_000
    max_steps: int = 100_000
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    noise_schedule: str = "logit_normal"
    precision: str = "bf16"
    attention: str = "flex"
    compile: bool = False
    seed: int = 0
    num_workers: int = 6
    stage_root: str | None = None
    log_every: int = 50
    val_every: int = 2_000
    val_clips: int = 64
    sample_every: int = 10_000
    sample_context: int = 16
    sample_steps: int = 16
    checkpoint_every: int = 1_000
    keep_every: int = 20_000

    def model_config(self) -> DiTConfig:
        return dataclasses.replace(SIZES[self.size], **self.model_overrides)


def _rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def _world() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def _log(path: pathlib.Path, record: dict[str, Any]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(record) + "\n")


def _save(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def learning_rate(config: A1Config, step: int) -> float:
    """Linear warmup, then cosine decay to 10% of the peak."""
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / config.warmup_steps
    progress = (step - config.warmup_steps) / max(
        1, config.max_steps - config.warmup_steps
    )
    return config.learning_rate * (0.1 + 0.45 * (1 + math.cos(math.pi * progress)))


def _batches(dataset: ClipDataset, config: A1Config, rank: int, start_step: int):
    """Endless shuffled batches; each rank and resume point gets its own stream."""
    seed = np.random.SeedSequence([config.seed, rank, start_step]).generate_state(1)
    generator = torch.Generator().manual_seed(int(seed[0]))
    sampler = torch.utils.data.RandomSampler(
        dataset, replacement=True, num_samples=2**31 - 1, generator=generator
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0,
        drop_last=True,
    )
    yield from loader


def _prepare(batch: dict[str, torch.Tensor], device, no_action: int):
    frames = to_model_range(batch["frames"].to(device, non_blocking=True))
    actions = batch["actions"].to(device, non_blocking=True).long()
    prev = conditioning_actions(actions, batch["prev_action"].to(device).long())
    return frames, prev, batch["start"].to(device)


@torch.no_grad()
def _ema_update(ema: PixelDiT, model: PixelDiT, decay: float) -> None:
    for target, source in zip(ema.parameters(), model.parameters(), strict=True):
        target.lerp_(source.detach(), 1 - decay)


@torch.no_grad()
def validate(model, dataset, config, device, autocast) -> float:
    model.eval()
    indices = np.linspace(0, len(dataset) - 1, config.val_clips).astype(int)
    losses = []
    for number, index in enumerate(indices):
        item = dataset[int(index)]
        batch = {k: torch.as_tensor(v)[None] for k, v in item.items()}
        frames, prev, start = _prepare(batch, device, model.config.no_action)
        generator = torch.Generator(device=device).manual_seed(number)
        with autocast:
            loss = flow_matching_loss(
                model, frames, prev, start, config.noise_schedule, generator
            )
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


@torch.no_grad()
def write_sample(model, dataset, config, device, autocast, path) -> dict[str, float]:
    """Roll out the clip from ``sample_context`` true frames with a full cache."""
    from PIL import Image

    model.eval()
    item = dataset[0]
    batch = {k: torch.as_tensor(v)[None] for k, v in item.items()}
    frames, prev, _ = _prepare(batch, device, model.config.no_action)
    grid = model.config.grid

    def full():
        return make_policy("full", geometry=Geometry(grid))

    with autocast:
        video = rollout(
            model,
            full,
            frames[:, : config.sample_context],
            prev,
            frames.shape[1],
            steps=config.sample_steps,
            generator=torch.Generator(device=device).manual_seed(0),
        )
    truth, generated = to_uint8(frames[0]).cpu(), to_uint8(video[0].float()).cpu()
    error = (truth.float() - generated.float()).pow(2).mean(dim=(1, 2, 3))
    psnr = (10 * torch.log10(255.0**2 / error.clamp_min(1e-8)))[config.sample_context :]
    pictures = [
        Image.fromarray(np.concatenate([a.numpy(), b.numpy()], axis=1)).resize(
            (256, 128), Image.NEAREST
        )
        for a, b in zip(truth, generated, strict=True)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    pictures[0].save(
        path, save_all=True, append_images=pictures[1:], duration=125, loop=0
    )
    model.train()
    return {"sample_psnr_mean": float(psnr.mean())}


def train(config: A1Config) -> int:
    signal.signal(signal.SIGUSR1, request_stop)
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        dist.init_process_group("nccl")
    rank, world = _rank(), _world()
    local = int(os.environ.get("LOCAL_RANK", "0"))
    device = (
        torch.device("cuda", local)
        if torch.cuda.is_available()
        else (torch.device("cpu"))
    )
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(config.seed + rank)

    run_dir = pathlib.Path(config.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    train_dir = pathlib.Path(config.train_split)
    val_dir = pathlib.Path(config.val_split)
    if config.stage_root:
        train_dir = stage_split(train_dir, pathlib.Path(config.stage_root))
        val_dir = stage_split(val_dir, pathlib.Path(config.stage_root))
    model_config = config.model_config()
    train_set = ClipDataset(
        train_dir, config.clip_frames, no_action=model_config.no_action
    )
    val_set = ClipDataset(
        val_dir,
        config.clip_frames,
        start_stride=config.clip_frames,
        no_action=model_config.no_action,
    )

    model = PixelDiT(model_config).to(device)
    model.attention_backend = config.attention if device.type == "cuda" else "sdpa"
    ema = copy.deepcopy(model).requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=config.betas,
        weight_decay=config.weight_decay,
    )
    step = 0
    checkpoint_path = run_dir / "checkpoints" / "latest.pt"
    if checkpoint_path.is_file():
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        ema.load_state_dict(state["ema"])
        optimizer.load_state_dict(state["optimizer"])
        step = int(state["step"])
    if rank == 0:
        record = {
            "config": dataclasses.asdict(config),
            "model_config": dataclasses.asdict(model_config),
            "parameters": sum(p.numel() for p in model.parameters()),
            "world_size": world,
            "git_sha": os.environ.get("DD_MEMORY_GIT_SHA", "unknown"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "restart_count": int(os.environ.get("SLURM_RESTART_COUNT", "0")),
            "resumed_from_step": step,
            "train_clips": len(train_set),
        }
        _log(run_dir / "attempts.jsonl", record)
        (run_dir / "config.json").write_text(json.dumps(record, indent=2) + "\n")

    trained = model
    if distributed:
        trained = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local])
    forward = torch.compile(trained) if config.compile else trained
    autocast = torch.autocast(
        device.type, dtype=torch.bfloat16, enabled=config.precision == "bf16"
    )
    batches = _batches(train_set, config, rank, step)
    window_start, window_frames = time.perf_counter(), 0
    stop_flag = torch.zeros(1, device=device)

    def checkpoint(path: pathlib.Path) -> None:
        if rank == 0:
            _save(
                path,
                {
                    "step": step,
                    "model": model.state_dict(),
                    "ema": ema.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_config": dataclasses.asdict(model_config),
                },
            )

    while step < config.max_steps:
        batch = next(batches)
        frames, prev, start = _prepare(batch, device, model_config.no_action)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(config, step)
        with autocast:
            loss = flow_matching_loss(
                forward, frames, prev, start, config.noise_schedule
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        _ema_update(ema, model, config.ema_decay)
        step += 1
        window_frames += frames.shape[0] * frames.shape[1] * world

        if step % config.log_every == 0:
            losses = loss.detach().reshape(1)
            if distributed:
                dist.all_reduce(losses)
                losses /= world
            elapsed = time.perf_counter() - window_start
            if rank == 0:
                record = {
                    "step": step,
                    "loss": losses.item(),
                    "lr": learning_rate(config, step - 1),
                    "grad_norm": float(grad_norm),
                    "frames_per_second": window_frames / elapsed,
                    "time": time.time(),
                }
                if device.type == "cuda":
                    record["max_memory_gb"] = torch.cuda.max_memory_allocated() / 2**30
                _log(run_dir / "metrics.jsonl", record)
                print(json.dumps(record), flush=True)
            window_start, window_frames = time.perf_counter(), 0
        if step % config.val_every == 0 and rank == 0:
            record = {
                "step": step,
                "val_loss_ema": validate(ema, val_set, config, device, autocast),
                "val_loss": validate(model, val_set, config, device, autocast),
            }
            _log(run_dir / "validation.jsonl", record)
            print(json.dumps(record), flush=True)
        if step % config.sample_every == 0 and rank == 0:
            path = run_dir / "samples" / f"step-{step:07d}.gif"
            record = {
                "step": step,
                **write_sample(ema, val_set, config, device, autocast, path),
            }
            _log(run_dir / "validation.jsonl", record)
        if step % config.checkpoint_every == 0:
            checkpoint(checkpoint_path)
        if step % config.keep_every == 0:
            checkpoint(run_dir / "checkpoints" / f"step-{step:07d}.pt")

        stop_flag.fill_(1.0 if _STOP else 0.0)
        if distributed:
            dist.all_reduce(stop_flag)
        if stop_flag.item() > 0:
            checkpoint(checkpoint_path)
            if distributed:
                dist.barrier()
                dist.destroy_process_group()
            if rank == 0:
                print(f"Stopped at step {step} for requeue", flush=True)
            return REQUEUE_EXIT_CODE
        if distributed and step % config.val_every == 0:
            dist.barrier()

    checkpoint(checkpoint_path)
    checkpoint(run_dir / "checkpoints" / f"step-{step:07d}.pt")
    if distributed:
        dist.barrier()
        dist.destroy_process_group()
    return 0
