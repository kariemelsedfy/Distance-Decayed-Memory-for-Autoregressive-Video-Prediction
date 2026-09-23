"""3D rotary position embedding at fractional positions.

Keys are cached *before* RoPE and rotated at attention time using each cached
token's position. A pooled cell's position is the centroid of the tokens it
summarizes, which is generally fractional (PROJECT_PLAN.md §4.2, decision 2).
Queries and keys share one function, so ``q · k`` depends only on the relative
offset of their positions.

The head dimension is split into three equal parts for the (t, y, x) axes; each
part must be even.
"""

from __future__ import annotations

import torch


def rope_frequencies(
    head_dim: int, base: float = 10_000.0, device=None
) -> list[torch.Tensor]:
    """Per-axis inverse frequencies for a head dimension split into (t, y, x)."""
    if head_dim % 6:
        raise ValueError("head_dim must be divisible by 6 for 3D RoPE")
    part = head_dim // 3
    exponents = torch.arange(0, part, 2, dtype=torch.float64, device=device) / part
    inverse = base**-exponents
    return [inverse, inverse.clone(), inverse.clone()]


def apply_rope(x: torch.Tensor, positions: torch.Tensor, base: float = 10_000.0):
    """Rotate ``x[..., n, head_dim]`` by ``positions[n, 3]`` (t, y, x), any real values.

    Computed in float64 for the angles and returned in ``x``'s dtype.
    """
    head_dim = x.shape[-1]
    if positions.shape[-1] != 3 or positions.shape[-2] != x.shape[-2]:
        raise ValueError("positions must be [n, 3] matching x's token axis")
    part = head_dim // 3
    pieces = []
    for axis, inverse in enumerate(rope_frequencies(head_dim, base, x.device)):
        chunk = x[..., axis * part : (axis + 1) * part].to(torch.float64)
        angle = positions[..., axis].to(torch.float64)[:, None] * inverse[None, :]
        cos, sin = angle.cos(), angle.sin()
        even, odd = chunk[..., 0::2], chunk[..., 1::2]
        rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), -1)
        pieces.append(rotated.flatten(-2))
    return torch.cat(pieces, dim=-1).to(x.dtype)
