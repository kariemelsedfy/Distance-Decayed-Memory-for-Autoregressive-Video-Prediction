"""Flow-matching loss, Euler sampling, and cached chunk-by-chunk rollout.

Rectified flow: ``x_τ = (1 - τ)·x + τ·ε`` with ``τ ∈ [0, 1]`` (0 is clean) and
velocity target ``ε - x``. Training draws one ``τ`` per chunk independently
(Diffusion Forcing), from a logit-normal distribution by default. Sampling
integrates from ``τ = 1`` to ``0`` with Euler steps while attending to a
memory policy's cache; each finished chunk is then re-encoded clean and its
keys and values are appended to the cache (TRACK_A_PLAN.md §5).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
import torch.nn.functional as F

from distance_decayed_memory.models.dit import ModelCache, PixelDiT


def to_model_range(frames_uint8: torch.Tensor) -> torch.Tensor:
    return frames_uint8.float() / 127.5 - 1.0


def to_uint8(frames: torch.Tensor) -> torch.Tensor:
    return ((frames.clamp(-1, 1) + 1.0) * 127.5).round().to(torch.uint8)


def chunk_noise_levels(
    batch: int,
    frames: int,
    chunk_frames: int,
    schedule: str = "logit_normal",
    generator: torch.Generator | None = None,
    device=None,
) -> torch.Tensor:
    """One noise level per chunk, repeated over its frames: ``[batch, frames]``."""
    chunks = frames // chunk_frames
    if schedule == "logit_normal":
        draws = torch.randn(batch, chunks, generator=generator, device=device)
        levels = torch.sigmoid(draws)
    elif schedule == "uniform":
        levels = torch.rand(batch, chunks, generator=generator, device=device)
    else:
        raise ValueError(f"unknown schedule {schedule!r}")
    return levels.repeat_interleave(chunk_frames, dim=1)


def flow_matching_loss(
    model: PixelDiT,
    frames: torch.Tensor,
    prev_actions: torch.Tensor,
    frame_start: int | torch.Tensor = 0,
    schedule: str = "logit_normal",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Mean squared velocity error on a clip ``[B, T, H, W, C]`` in ``[-1, 1]``."""
    b, t = frames.shape[:2]
    levels = chunk_noise_levels(
        b, t, model.config.chunk_frames, schedule, generator, frames.device
    )
    noise = torch.randn(frames.shape, generator=generator, device=frames.device)
    tau = levels[..., None, None, None].to(frames)
    noisy = (1 - tau) * frames + tau * noise
    velocity = model(noisy, levels, prev_actions, frame_start)
    return F.mse_loss(velocity, noise - frames)


@torch.no_grad()
def sample_chunk(
    model: PixelDiT,
    prev_actions: torch.Tensor,
    frame_start: int | torch.Tensor,
    cache: ModelCache | None,
    steps: int = 16,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Generate one chunk ``[B, chunk_frames, H, W, C]`` by Euler integration."""
    config = model.config
    b = prev_actions.shape[0]
    shape = (
        b,
        config.chunk_frames,
        config.image_size,
        config.image_size,
        config.channels,
    )
    parameter = next(model.parameters())
    x = torch.randn(
        shape, generator=generator, device=parameter.device, dtype=parameter.dtype
    )
    levels = torch.linspace(1.0, 0.0, steps + 1, device=x.device, dtype=x.dtype)
    for current, following in zip(levels[:-1], levels[1:], strict=True):
        tau = current.expand(b, config.chunk_frames)
        velocity = model(x, tau, prev_actions, frame_start, cache)
        x = x + (following - current) * velocity
    return x


@torch.no_grad()
def encode_chunk(
    model: PixelDiT,
    frames: torch.Tensor,
    prev_actions: torch.Tensor,
    frame_start: int | torch.Tensor,
    cache: ModelCache | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Clean pass over a chunk; returns its pre-RoPE ``(k, v)`` for the cache."""
    zero = torch.zeros(frames.shape[:2], device=frames.device, dtype=frames.dtype)
    _, k, v = model(frames, zero, prev_actions, frame_start, cache, return_kv=True)
    return k, v


def _write(policies: Sequence, k: torch.Tensor, v: torch.Tensor, frame: int) -> None:
    for index, policy in enumerate(policies):
        policy.append(k[index], v[index], None, frame)
        policy.compact()


@torch.no_grad()
def rollout(
    model: PixelDiT,
    make_policy: Callable[[], object],
    context: torch.Tensor,
    prev_actions: torch.Tensor,
    total_frames: int,
    steps: int = 16,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Autoregressive generation from ``context`` true frames (protocol P2).

    ``context`` is ``[B, C0, H, W, C]`` with ``C0`` a multiple of the chunk;
    ``prev_actions`` is ``[B, total_frames]``. Each episode gets its own policy
    from ``make_policy``. Returns all ``total_frames`` frames (context first).
    """
    config = model.config
    chunk = config.chunk_frames
    b, known = context.shape[:2]
    policies = [make_policy() for _ in range(b)]
    output = [context]
    for start in range(0, known, chunk):
        cache = ModelCache.from_policies(policies) if start else None
        k, v = encode_chunk(
            model,
            context[:, start : start + chunk],
            prev_actions[:, start : start + chunk],
            start,
            cache,
        )
        _write(policies, k, v, start)
    for start in range(known, total_frames, chunk):
        cache = ModelCache.from_policies(policies)
        actions = prev_actions[:, start : start + chunk]
        frames = sample_chunk(model, actions, start, cache, steps, generator)
        k, v = encode_chunk(model, frames, actions, start, cache)
        _write(policies, k, v, start)
        output.append(frames)
    return torch.cat(output, dim=1)
