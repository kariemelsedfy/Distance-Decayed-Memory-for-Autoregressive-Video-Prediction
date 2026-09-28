"""Spacetime cells: the unit every memory policy stores, pools, and merges.

A frame arrives as a ``grid × grid`` block of patch tokens (level 0). Each
aging step halves the number of tokens by averaging adjacent pairs, so a
level-``ℓ`` token summarizes ``2**ℓ`` original tokens (its *volume*):

- levels ``1 … S`` pool spatially, alternating rows then columns, until one
  token covers the whole frame (``S = 2·log2(grid)``, 8 for a 16×16 grid);
- levels above ``S`` merge temporally aligned pairs of frames, so a block at
  level ``S + j`` is one token covering ``2**j`` frames whose first frame is a
  multiple of ``2**j``.

Every step is an exact average of equal-volume children, so a cell is always
the mean of the original tokens it covers, and a cell never becomes finer.
Keys are stored before RoPE; positions are centroids ``(t, y, x)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Geometry:
    grid: int = 16

    def __post_init__(self) -> None:
        if self.grid < 1 or self.grid & (self.grid - 1):
            raise ValueError("grid must be a power of two")

    @property
    def tokens_per_frame(self) -> int:
        return self.grid * self.grid

    @property
    def spatial_levels(self) -> int:
        return 2 * int(math.log2(self.grid))

    def shape(self, level: int) -> tuple[int, int]:
        """Token grid (rows, columns) of one block at ``level``."""
        spatial = min(level, self.spatial_levels)
        rows = self.grid >> ((spatial + 1) // 2)
        columns = self.grid >> (spatial // 2)
        return rows, columns

    def frames(self, level: int) -> int:
        return 1 << max(level - self.spatial_levels, 0)

    def tokens_per_frame_at(self, level: int) -> float:
        return self.tokens_per_frame / 2.0**level

    def frame_positions(self, t: int) -> torch.Tensor:
        """Level-0 positions ``[grid², 3]`` for frame ``t`` in row-major order."""
        y, x = torch.meshgrid(
            torch.arange(self.grid, dtype=torch.float64),
            torch.arange(self.grid, dtype=torch.float64),
            indexing="ij",
        )
        t_column = torch.full_like(y, float(t))
        return torch.stack((t_column, y, x), dim=-1).reshape(-1, 3)


@dataclass
class Block:
    """Cached tokens for ``frames`` consecutive frames starting at ``t0``.

    ``k`` and ``v`` are ``[layers, heads, tokens, head_dim]``; ``pos`` is
    ``[tokens, 3]`` in float64.
    """

    t0: int
    frames: int
    level: int
    k: torch.Tensor
    v: torch.Tensor
    pos: torch.Tensor
    salience: float = 1.0
    sink: bool = False

    @property
    def newest(self) -> int:
        return self.t0 + self.frames - 1

    @property
    def tokens(self) -> int:
        return self.k.shape[-2]

    @property
    def volume(self) -> int:
        return 1 << self.level

    def detach(self) -> None:
        self.k = self.k.detach()
        self.v = self.v.detach()


def _pool_pairs(x: torch.Tensor, rows: int, columns: int, along_rows: bool):
    """Average adjacent token pairs of ``x[..., rows*columns, d]`` on its grid."""
    grid = x.reshape(*x.shape[:-2], rows, columns, x.shape[-1])
    if along_rows:
        pooled = grid.reshape(*x.shape[:-2], rows // 2, 2, columns, x.shape[-1])
        pooled = pooled.mean(dim=-3)
    else:
        pooled = grid.reshape(*x.shape[:-2], rows, columns // 2, 2, x.shape[-1])
        pooled = pooled.mean(dim=-2)
    return pooled.reshape(*x.shape[:-2], -1, x.shape[-1])


def pool_spatial(block: Block, geometry: Geometry) -> None:
    """Coarsen ``block`` by one spatial level in place."""
    if block.level >= geometry.spatial_levels:
        raise ValueError("block is already one token per frame")
    rows, columns = geometry.shape(block.level)
    along_rows = block.level % 2 == 0
    block.k = _pool_pairs(block.k, rows, columns, along_rows)
    block.v = _pool_pairs(block.v, rows, columns, along_rows)
    block.pos = _pool_pairs(block.pos, rows, columns, along_rows)
    block.level += 1


def can_merge(older: Block, newer: Block, geometry: Geometry) -> bool:
    return (
        older.level == newer.level >= geometry.spatial_levels
        and not (older.sink or newer.sink)
        and older.t0 % (2 * older.frames) == 0
        and newer.t0 == older.t0 + older.frames
    )


def merge_temporal(older: Block, newer: Block, geometry: Geometry) -> Block:
    """One block covering both aligned, equal-level, single-token blocks."""
    if not can_merge(older, newer, geometry):
        raise ValueError("blocks are not an aligned equal-level pair")
    return Block(
        t0=older.t0,
        frames=older.frames * 2,
        level=older.level + 1,
        k=(older.k + newer.k) / 2,
        v=(older.v + newer.v) / 2,
        pos=(older.pos + newer.pos) / 2,
        salience=(older.salience + newer.salience) / 2,
    )
