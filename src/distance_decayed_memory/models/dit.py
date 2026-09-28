"""Frame-causal pixel diffusion transformer for Track A (TRACK_A_PLAN.md §4).

- 64×64×3 frames, 4×4 patches: 256 tokens per frame, 3D RoPE over (t, y, x).
- Frames are generated in chunks of ``chunk_frames``. Tokens attend
  bidirectionally inside their chunk and causally to earlier chunks.
- Each frame is conditioned, through adaLN-Zero, on the action that led into
  it and on its diffusion noise level (Diffusion-Forcing style: one level per
  chunk).
- The network predicts flow-matching velocity (``flow.py``).

Two attention paths share every weight:

- **Clip mode** (``cache=None``): a whole clip with a block-causal mask, used
  by A1 training.
- **Cache mode**: the current chunk(s) also attend to a memory policy's cache
  of pre-RoPE keys and values, with an optional ``log v`` bias for pooled
  cells. Used by A2 streaming and by sampling. Passing ``return_kv=True``
  returns the chunk's own pre-RoPE keys and values for writing into the cache.

Frames are channels-last floats in ``[-1, 1]``, shaped ``[B, T, H, W, C]``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from distance_decayed_memory.memory.rope import apply_rope


@dataclass(frozen=True)
class DiTConfig:
    layers: int = 16
    width: int = 768
    heads: int = 12
    image_size: int = 64
    patch: int = 4
    channels: int = 3
    chunk_frames: int = 4
    actions: int = 6
    mlp_ratio: float = 4.0
    rope_base: float = 10_000.0
    proportional_attention: bool = True

    def __post_init__(self) -> None:
        if self.width % self.heads:
            raise ValueError("width must be divisible by heads")
        if self.image_size % self.patch:
            raise ValueError("image_size must be divisible by patch")

    @property
    def grid(self) -> int:
        return self.image_size // self.patch

    @property
    def tokens_per_frame(self) -> int:
        return self.grid * self.grid

    @property
    def head_dim(self) -> int:
        return self.width // self.heads

    @property
    def patch_dim(self) -> int:
        return self.patch * self.patch * self.channels

    @property
    def no_action(self) -> int:
        """Action index for "no previous action" (the first frame)."""
        return self.actions


SIZES = {
    "S": DiTConfig(layers=12, width=512, heads=8),
    "M": DiTConfig(layers=16, width=768, heads=12),
    "L": DiTConfig(layers=24, width=1024, heads=16),
}


@dataclass
class ModelCache:
    """A batch of memory caches for every layer, padded to a common length.

    ``k``/``v``: ``[B, layers, heads, N, head_dim]`` pre-RoPE; ``pos``:
    ``[B, N, 3]``; ``weight``: ``[B, N]`` cell volumes; ``valid``: ``[B, N]``.
    """

    k: torch.Tensor
    v: torch.Tensor
    pos: torch.Tensor
    weight: torch.Tensor
    valid: torch.Tensor

    @classmethod
    def from_policies(cls, policies, dtype=None, device=None) -> ModelCache:
        """Stack per-episode ``MemoryPolicy.kv()`` results; pad with invalid slots."""
        parts = [policy.kv() for policy in policies]
        length = max(part[0].shape[-2] for part in parts)
        layers, heads, _, dim = parts[0][0].shape
        dtype = dtype or parts[0][0].dtype
        batch = len(parts)
        k = torch.zeros(batch, layers, heads, length, dim, dtype=dtype, device=device)
        v = torch.zeros_like(k)
        pos = torch.zeros(batch, length, 3, dtype=torch.float64, device=device)
        weight = torch.ones(batch, length, dtype=torch.float64, device=device)
        valid = torch.zeros(batch, length, dtype=torch.bool, device=device)
        for index, (pk, pv, ppos, pweight) in enumerate(parts):
            n = pk.shape[-2]
            k[index, ..., :n, :] = pk.to(k)
            v[index, ..., :n, :] = pv.to(v)
            pos[index, :n] = ppos.to(pos)
            weight[index, :n] = pweight.to(weight)
            valid[index, :n] = True
        return cls(k, v, pos, weight, valid)


def patchify(frames: torch.Tensor, patch: int) -> torch.Tensor:
    """``[B, T, H, W, C]`` → ``[B, T·(H/p)·(W/p), p·p·C]``, frame-major, row-major."""
    b, t, h, w, c = frames.shape
    x = frames.reshape(b, t, h // patch, patch, w // patch, patch, c)
    x = x.permute(0, 1, 2, 4, 3, 5, 6)
    return x.reshape(b, t * (h // patch) * (w // patch), patch * patch * c)


def unpatchify(tokens: torch.Tensor, frames: int, size: int, patch: int, channels: int):
    b = tokens.shape[0]
    grid = size // patch
    x = tokens.reshape(b, frames, grid, grid, patch, patch, channels)
    x = x.permute(0, 1, 2, 4, 3, 5, 6)
    return x.reshape(b, frames, size, size, channels)


def token_positions(frame_start: torch.Tensor, frames: int, grid: int) -> torch.Tensor:
    """``[B, frames·grid², 3]`` (t, y, x) positions, t offset per batch element."""
    t = torch.arange(frames, dtype=torch.float64, device=frame_start.device)
    y = torch.arange(grid, dtype=torch.float64, device=frame_start.device)
    tt, yy, xx = torch.meshgrid(t, y, y, indexing="ij")
    base = torch.stack((tt, yy, xx), dim=-1).reshape(-1, 3)
    offset = torch.zeros(
        frame_start.shape[0], 1, 3, dtype=torch.float64, device=frame_start.device
    )
    offset[..., 0] = frame_start.to(torch.float64)[:, None]
    return base[None] + offset


def chunk_causal_mask(
    frames: int, tokens_per_frame: int, chunk_frames: int, device=None
):
    """Boolean ``[n, n]``: query token may attend key token (same or earlier chunk)."""
    chunk = torch.arange(frames, device=device) // chunk_frames
    chunk = chunk.repeat_interleave(tokens_per_frame)
    return chunk[:, None] >= chunk[None, :]


def timestep_embedding(levels: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding of noise levels in ``[0, 1]`` (scaled by 1000)."""
    half = dim // 2
    frequencies = torch.exp(
        -math.log(10_000) * torch.arange(half, device=levels.device) / half
    )
    angles = levels.float()[..., None] * 1000.0 * frequencies
    return torch.cat((angles.cos(), angles.sin()), dim=-1)


class RMSNorm(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (x.shape[-1],)) * self.scale


class Block(nn.Module):
    def __init__(self, config: DiTConfig) -> None:
        super().__init__()
        width, heads = config.width, config.heads
        self.config = config
        self.norm1 = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(width, 3 * width)
        self.q_norm = RMSNorm(config.head_dim)
        self.k_norm = RMSNorm(config.head_dim)
        self.proj = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        hidden = int(width * config.mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(width, hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden, width),
        )
        self.modulation = nn.Linear(width, 6 * width)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)
        self.heads = heads

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
        positions: torch.Tensor,
        mask: torch.Tensor | None,
        cache: tuple | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, n, width = x.shape
        modulation = self.modulation(condition).repeat_interleave(
            self.config.tokens_per_frame, dim=1
        )
        shift1, scale1, gate1, shift2, scale2, gate2 = modulation.chunk(6, dim=-1)
        h = self.norm1(x) * (1 + scale1) + shift1
        q, k, v = self.qkv(h).reshape(b, n, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
        q, k = self.q_norm(q), self.k_norm(k)
        fresh_k, fresh_v = k, v
        q_rot = apply_rope(q, positions[:, None], self.config.rope_base)
        k_rot = apply_rope(k, positions[:, None], self.config.rope_base)
        bias = None
        if cache is not None:
            cache_k, cache_v, cache_pos, cache_bias = cache
            cache_rot = apply_rope(cache_k, cache_pos[:, None], self.config.rope_base)
            k_rot = torch.cat((cache_rot.to(k_rot), k_rot), dim=-2)
            v = torch.cat((cache_v.to(v), v), dim=-2)
            bias = cache_bias
        if mask is not None:
            blocked = torch.zeros(mask.shape, dtype=q.dtype, device=q.device)
            blocked = blocked.masked_fill(~mask, float("-inf"))
            if bias is not None:
                blocked = torch.cat(
                    (
                        bias.to(q.dtype)[:, None, None, :].expand(b, 1, n, -1),
                        blocked.expand(b, 1, n, n),
                    ),
                    dim=-1,
                )
            bias = blocked
        elif bias is not None:
            zeros = torch.zeros(b, n, dtype=bias.dtype, device=bias.device)
            bias = torch.cat((bias, zeros), dim=-1).to(q.dtype)[:, None, None, :]
        attended = F.scaled_dot_product_attention(q_rot, k_rot, v, attn_mask=bias)
        attended = attended.transpose(1, 2).reshape(b, n, width)
        x = x + gate1 * self.proj(attended)
        h = self.norm2(x) * (1 + scale2) + shift2
        x = x + gate2 * self.mlp(h)
        return x, fresh_k, fresh_v


class PixelDiT(nn.Module):
    def __init__(self, config: DiTConfig | None = None) -> None:
        super().__init__()
        self.config = config = config or SIZES["M"]
        width = config.width
        self.patch_embed = nn.Linear(config.patch_dim, width)
        self.time_embed = nn.Sequential(
            nn.Linear(256, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.action_embed = nn.Embedding(config.actions + 1, width)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.layers))
        self.final_norm = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.final_modulation = nn.Linear(width, 2 * width)
        self.head = nn.Linear(width, config.patch_dim)
        for layer in (self.final_modulation, self.head):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(
        self,
        frames: torch.Tensor,
        noise_level: torch.Tensor,
        prev_actions: torch.Tensor,
        frame_start: int | torch.Tensor = 0,
        cache: ModelCache | None = None,
        return_kv: bool = False,
    ):
        """Predict velocity for ``frames`` ``[B, T, H, W, C]``.

        ``noise_level`` and ``prev_actions`` are ``[B, T]``; ``prev_actions[:, i]``
        is the action that produced frame ``i`` (``config.no_action`` for the
        first frame of an episode). ``frame_start`` is the absolute index of the
        first frame, per batch element or shared. With ``return_kv`` also returns
        this input's pre-RoPE keys and values ``[B, layers, heads, n, head_dim]``.
        """
        config = self.config
        b, t = frames.shape[:2]
        if t % config.chunk_frames:
            raise ValueError("frame count must be a multiple of chunk_frames")
        start = torch.as_tensor(frame_start, device=frames.device).reshape(-1)
        start = start.expand(b) if start.numel() == 1 else start
        x = self.patch_embed(patchify(frames, config.patch))
        # One conditioning vector per frame; blocks expand it over the frame's tokens.
        dtype = self.patch_embed.weight.dtype
        condition = self.time_embed(
            timestep_embedding(noise_level, 256).to(dtype)
        ) + self.action_embed(prev_actions)
        positions = token_positions(start, t, config.grid)
        mask = None
        if t > config.chunk_frames:
            mask = chunk_causal_mask(
                t, config.tokens_per_frame, config.chunk_frames, frames.device
            )
        cache_bias = None
        if cache is not None:
            cache_bias = torch.where(
                cache.valid,
                (
                    cache.weight.log()
                    if config.proportional_attention
                    else torch.zeros_like(cache.weight)
                ),
                torch.full_like(cache.weight, float("-inf")),
            )
        keys, values = [], []
        for index, block in enumerate(self.blocks):
            layer_cache = None
            if cache is not None:
                layer_cache = (
                    cache.k[:, index],
                    cache.v[:, index],
                    cache.pos,
                    cache_bias,
                )
            x, k, v = block(x, condition, positions, mask, layer_cache)
            if return_kv:
                keys.append(k)
                values.append(v)
        final = self.final_modulation(condition).repeat_interleave(
            config.tokens_per_frame, dim=1
        )
        shift, scale = final.chunk(2, dim=-1)
        x = self.head(self.final_norm(x) * (1 + scale) + shift)
        velocity = unpatchify(x, t, config.image_size, config.patch, config.channels)
        if return_kv:
            return velocity, torch.stack(keys, dim=1), torch.stack(values, dim=1)
        return velocity


def conditioning_actions(
    actions: torch.Tensor, preceding: torch.Tensor
) -> torch.Tensor:
    """Actions that led into each frame of a clip.

    ``actions[:, i]`` is taken after frame ``i`` (D-008), so frame ``i`` is
    conditioned on ``actions[:, i - 1]``. ``preceding[b]`` is the action before
    the clip's first frame, or ``config.no_action`` at the start of an episode.
    """
    return torch.cat((preceding.reshape(-1, 1).to(actions), actions[:, :-1]), dim=1)


def count_parameters(config: DiTConfig) -> int:
    with torch.device("meta"):
        model = PixelDiT(config)
    return sum(p.numel() for p in model.parameters())
