"""Flow-matching loss, Euler sampling, and cached chunk-by-chunk rollout.

Rectified flow: ``x_τ = (1 - τ)·x + τ·ε`` with ``τ ∈ [0, 1]`` (0 is clean) and
velocity target ``ε - x``. Training draws one ``τ`` per chunk independently
(Diffusion Forcing), from a logit-normal distribution by default. Sampling
integrates from ``τ = 1`` to ``0`` with Euler steps while attending to a
memory policy's cache; each finished chunk is then re-encoded clean and its
keys and values are appended to the cache (TRACK_A_PLAN.md §5).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence

import torch
import torch.nn.functional as F

from distance_decayed_memory.models.dit import ModelCache, PixelDiT


def unwrap(model) -> PixelDiT:
    """The :class:`PixelDiT` inside DDP and/or ``torch.compile`` wrappers."""
    while not isinstance(model, PixelDiT):
        model = getattr(model, "module", None) or model._orig_mod
    return model


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
    model,
    frames: torch.Tensor,
    prev_actions: torch.Tensor,
    frame_start: int | torch.Tensor = 0,
    schedule: str = "logit_normal",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Mean squared velocity error on a clip ``[B, T, H, W, C]`` in ``[-1, 1]``.

    ``model`` may be wrapped (DDP, ``torch.compile``); its config is read from
    the underlying :class:`PixelDiT`.
    """
    b, t = frames.shape[:2]
    config = unwrap(model).config
    levels = chunk_noise_levels(
        b, t, config.chunk_frames, schedule, generator, frames.device
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


class StreamingCache:
    """Per-episode memory policies plus a full-fidelity local window (D-011).

    The newest ``local_chunks`` encoded chunks stay outside the policy at full
    fidelity for every policy alike; older chunks are handed to the policy,
    which compacts them under its budget. Training (A2) and inference use the
    same structure, so the cache a model reads is the same in both.
    """

    def __init__(self, policies: Sequence, local_chunks: int = 0) -> None:
        self.policies = list(policies)
        self.local_chunks = local_chunks
        self.pending: deque[tuple[torch.Tensor, torch.Tensor, int]] = deque()

    def model_cache(self) -> ModelCache | None:
        parts = []
        for index, policy in enumerate(self.policies):
            pieces = [policy.kv()] if policy.blocks else []
            for k, v, start in self.pending:
                geometry = policy.geometry
                frames = k.shape[-2] // geometry.tokens_per_frame
                pos = torch.cat(
                    [geometry.frame_positions(start + f) for f in range(frames)]
                )
                pieces.append(
                    (
                        k[index],
                        v[index],
                        pos,
                        torch.ones(pos.shape[0], dtype=torch.float64),
                    )
                )
            if not pieces:
                parts.append(None)
                continue
            parts.append(
                (
                    torch.cat([piece[0] for piece in pieces], dim=-2),
                    torch.cat([piece[1] for piece in pieces], dim=-2),
                    torch.cat([piece[2].to(torch.float64) for piece in pieces]),
                    torch.cat([piece[3].to(torch.float64) for piece in pieces]),
                )
            )
        return ModelCache.from_parts(parts)

    def push(self, k: torch.Tensor, v: torch.Tensor, start: int) -> None:
        """Add a chunk's ``[B, layers, heads, n, dim]`` K/V starting at ``start``."""
        self.pending.append((k, v, start))
        while len(self.pending) > self.local_chunks:
            old_k, old_v, old_start = self.pending.popleft()
            for index, policy in enumerate(self.policies):
                policy.append(old_k[index], old_v[index], None, old_start)
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
    local_chunks: int = 0,
) -> torch.Tensor:
    """Autoregressive generation from ``context`` true frames (protocol P2).

    ``context`` is ``[B, C0, H, W, C]`` with ``C0`` a multiple of the chunk;
    ``prev_actions`` is ``[B, total_frames]``. Each episode gets its own policy
    from ``make_policy``; the newest ``local_chunks`` chunks are kept at full
    fidelity outside it (D-011). Returns all ``total_frames`` frames.
    """
    config = model.config
    chunk = config.chunk_frames
    b, known = context.shape[:2]
    cache = StreamingCache([make_policy() for _ in range(b)], local_chunks)
    output = [context]
    for start in range(0, known, chunk):
        k, v = encode_chunk(
            model,
            context[:, start : start + chunk],
            prev_actions[:, start : start + chunk],
            start,
            cache.model_cache(),
        )
        cache.push(k, v, start)
    for start in range(known, total_frames, chunk):
        current = cache.model_cache()
        actions = prev_actions[:, start : start + chunk]
        frames = sample_chunk(model, actions, start, current, steps, generator)
        k, v = encode_chunk(model, frames, actions, start, current)
        cache.push(k, v, start)
        output.append(frames.to(context.dtype))
    return torch.cat(output, dim=1)
