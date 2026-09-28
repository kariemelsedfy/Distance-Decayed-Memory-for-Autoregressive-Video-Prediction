"""3D rotary position embedding at fractional positions.

Keys are cached *before* RoPE and rotated at attention time using each cached
token's position. A pooled cell's position is the centroid of the tokens it
summarizes, which is generally fractional (PROJECT_PLAN.md §4.2, decision 2).
Queries and keys share one function, so ``q · k`` depends only on the relative
offset of their positions.

The head dimension is split into three even parts for the (t, y, x) axes: the
two spatial axes get equal parts and time gets the rest (24/20/20 for a
64-dimensional head).
"""

from __future__ import annotations

import torch


def axis_sizes(head_dim: int) -> tuple[int, int, int]:
    """Even (t, y, x) split of ``head_dim``; spatial axes are equal."""
    if head_dim % 2 or head_dim < 6:
        raise ValueError("head_dim must be even and at least 6 for 3D RoPE")
    spatial = (head_dim // 3) // 2 * 2
    return head_dim - 2 * spatial, spatial, spatial


def rope_frequencies(
    head_dim: int, base: float = 10_000.0, device=None
) -> list[torch.Tensor]:
    """Per-axis inverse frequencies for a head dimension split into (t, y, x)."""
    return [
        base ** -(torch.arange(0, size, 2, dtype=torch.float64, device=device) / size)
        for size in axis_sizes(head_dim)
    ]


def rope_tables(
    positions: torch.Tensor,
    head_dim: int,
    base: float = 10_000.0,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(cos, sin)`` of shape ``[..., n, head_dim / 2]`` for ``positions[..., n, 3]``.

    Angles are computed in float64 (time positions reach thousands of radians)
    and the tables are returned in ``dtype``. Build them once per forward pass
    and reuse them across layers with :func:`rotate`.
    """
    if positions.shape[-1] != 3:
        raise ValueError("positions must be [..., n, 3]")
    angles = [
        positions[..., axis].to(torch.float64)[..., None] * inverse
        for axis, inverse in enumerate(
            rope_frequencies(head_dim, base, positions.device)
        )
    ]
    angle = torch.cat(angles, dim=-1)
    return angle.cos().to(dtype), angle.sin().to(dtype)


def rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply precomputed RoPE tables to ``x[..., n, head_dim]`` (pairs 2i, 2i+1)."""
    values = x.to(cos.dtype)
    even, odd = values[..., 0::2], values[..., 1::2]
    rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
    return rotated.flatten(-2).to(x.dtype)


def apply_rope(x: torch.Tensor, positions: torch.Tensor, base: float = 10_000.0):
    """Rotate ``x[..., n, head_dim]`` by ``positions[..., n, 3]`` (t, y, x).

    Positions may be any real values and may carry leading batch dimensions
    that broadcast against ``x``'s (for example ``[batch, 1, n, 3]``). Tables
    are float64 for float64 inputs and float32 otherwise.
    """
    if positions.shape[-2] != x.shape[-2]:
        raise ValueError("positions must be [..., n, 3] matching x's token axis")
    dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    cos, sin = rope_tables(positions, x.shape[-1], base, dtype)
    return rotate(x, cos, sin)
