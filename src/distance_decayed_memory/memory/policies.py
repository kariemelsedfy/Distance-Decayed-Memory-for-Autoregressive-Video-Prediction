"""Memory policies (PROJECT_PLAN.md §4.1–4.2) over one shared cell mechanism.

Every policy stores blocks of cells (:mod:`cells`) and differs only in
:meth:`MemoryPolicy.target_level`: how coarse a block should be at its current
distance, or ``None`` to drop it. :meth:`MemoryPolicy.compact` then applies the
same steps for every policy:

1. drop blocks whose target is ``None`` (schedule or horizon);
2. pool each block spatially up to its target;
3. merge aligned single-token blocks temporally while their target allows;
4. enforce the budget: coarsen, merge, or drop the oldest non-sink blocks
   until the token count fits.

Distance is measured from a block's newest frame: ``d = now - newest``.
A policy therefore never keeps more than ``budget_tokens`` after
:meth:`compact`, never keeps a block whose newest frame is beyond
``horizon``, and never makes a cell finer.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch

from distance_decayed_memory.memory.cells import (
    Block,
    Geometry,
    can_merge,
    merge_temporal,
    pool_spatial,
)


class MemoryPolicy:
    name = "base"

    def __init__(
        self,
        budget_tokens: int | None,
        horizon: int | None,
        geometry: Geometry | None = None,
    ) -> None:
        self.budget_tokens = budget_tokens
        self.horizon = horizon
        self.geometry = geometry or Geometry()
        self.blocks: list[Block] = []
        self.now = -1

    # -- schedule ---------------------------------------------------------------

    def target_level(self, block: Block, now: int) -> int | None:
        raise NotImplementedError

    def is_sink(self, t: int) -> bool:
        return False

    def salience(self, k: torch.Tensor, t: int) -> float:
        return 1.0

    # -- interface ------------------------------------------------------------

    def append(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        pos: torch.Tensor | None,
        t: int,
    ) -> None:
        """Add frames ``t, t+1, …`` as level-0 blocks.

        ``k`` and ``v`` are ``[layers, heads, frames·grid², head_dim]`` pre-RoPE
        keys and values in frame-major, row-major order. ``pos`` may be ``None``
        (the standard grid positions are used) or must equal them.
        """
        per_frame = self.geometry.tokens_per_frame
        if k.shape != v.shape or k.shape[-2] % per_frame:
            raise ValueError("k and v must match and hold whole frames")
        if t <= self.now:
            raise ValueError(f"frame {t} is not after the current frame {self.now}")
        frames = k.shape[-2] // per_frame
        for offset in range(frames):
            frame = t + offset
            span = slice(offset * per_frame, (offset + 1) * per_frame)
            expected = self.geometry.frame_positions(frame)
            if pos is not None and not torch.equal(pos[span].to(expected), expected):
                raise ValueError("pos must be the standard (t, y, x) grid")
            frame_k = k[..., span, :]
            self.blocks.append(
                Block(
                    t0=frame,
                    frames=1,
                    level=0,
                    k=frame_k,
                    v=v[..., span, :],
                    pos=expected,
                    salience=self.salience(frame_k, frame),
                    sink=self.is_sink(frame),
                )
            )
        self.now = t + frames - 1

    def compact(self, t: int | None = None) -> None:
        now = self.now if t is None else t
        if now < self.now:
            raise ValueError("compaction time cannot precede the newest frame")
        self.now = now
        geometry = self.geometry
        self.blocks = [
            b for b in self.blocks if b.sink or self.target_level(b, now) is not None
        ]
        for block in self.blocks:
            if block.sink:
                continue
            target = min(self.target_level(block, now), geometry.spatial_levels)
            while block.level < target:
                pool_spatial(block, geometry)
        self._merge_ready(now)
        self._enforce_budget()

    def kv(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(K, V, pos, weight)``: ``[L, H, N, D]`` twice, ``[N, 3]``, ``[N]``.

        ``weight`` is each token's volume (original tokens it summarizes), for
        the optional ``log v`` attention bias.
        """
        if not self.blocks:
            raise ValueError("cache is empty")
        k = torch.cat([b.k for b in self.blocks], dim=-2)
        v = torch.cat([b.v for b in self.blocks], dim=-2)
        pos = torch.cat([b.pos for b in self.blocks], dim=0)
        weight = torch.cat(
            [
                torch.full((b.tokens,), float(b.volume), dtype=torch.float64)
                for b in self.blocks
            ]
        )
        return k, v, pos, weight

    def total_tokens(self) -> int:
        return sum(b.tokens for b in self.blocks)

    def density(self, length: int | None = None) -> torch.Tensor:
        """Cached tokens per frame at each distance ``0 … length-1`` from now."""
        length = length or self.now + 1
        values = torch.zeros(length, dtype=torch.float64)
        for block in self.blocks:
            share = block.tokens / block.frames
            for frame in range(block.t0, block.t0 + block.frames):
                distance = self.now - frame
                if 0 <= distance < length:
                    values[distance] += share
        return values

    def stats(self) -> dict:
        buckets: dict[str, int] = {}
        levels: dict[int, int] = {}
        for block in self.blocks:
            distance = self.now - block.newest
            low = 0 if distance < 1 else 1 << int(math.log2(distance))
            label = f"[{low},{max(1, 2 * low)})"
            buckets[label] = buckets.get(label, 0) + block.tokens
            levels[block.level] = levels.get(block.level, 0) + block.tokens
        size = sum(
            b.k.numel() * b.k.element_size() + b.v.numel() * b.v.element_size()
            for b in self.blocks
        )
        return {
            "policy": self.name,
            "now": self.now,
            "budget_tokens": self.budget_tokens,
            "horizon": self.horizon,
            "total_tokens": self.total_tokens(),
            "blocks": len(self.blocks),
            "tokens_by_distance": buckets,
            "tokens_by_level": dict(sorted(levels.items())),
            "kv_bytes": size,
        }

    def detach_before(self, t: int) -> None:
        """Cut autograd history for blocks whose newest frame precedes ``t``."""
        for block in self.blocks:
            if block.newest < t:
                block.detach()

    # -- internals ------------------------------------------------------------

    def _beyond_horizon(self, block: Block, now: int) -> bool:
        return self.horizon is not None and now - block.newest >= self.horizon

    def _merge_ready(self, now: int) -> None:
        changed = True
        while changed:
            changed = False
            for index in range(len(self.blocks) - 1):
                older, newer = self.blocks[index], self.blocks[index + 1]
                if not can_merge(older, newer, self.geometry):
                    continue
                probe = Block(
                    older.t0,
                    older.frames * 2,
                    older.level + 1,
                    older.k,
                    older.v,
                    older.pos,
                    (older.salience + newer.salience) / 2,
                )
                target = self.target_level(probe, now)
                if target is not None and target > older.level:
                    merged = merge_temporal(older, newer, self.geometry)
                    self.blocks[index : index + 2] = [merged]
                    changed = True
                    break

    def _enforce_budget(self) -> None:
        if self.budget_tokens is None:
            return
        while self.total_tokens() > self.budget_tokens:
            candidates = [i for i, b in enumerate(self.blocks) if not b.sink]
            if not candidates:
                raise RuntimeError("sink frames alone exceed the budget")
            index = candidates[0]
            block = self.blocks[index]
            following = self.blocks[index + 1] if index + 1 < len(self.blocks) else None
            if block.level < self.geometry.spatial_levels:
                pool_spatial(block, self.geometry)
            elif following is not None and can_merge(block, following, self.geometry):
                self.blocks[index : index + 2] = [
                    merge_temporal(block, following, self.geometry)
                ]
            else:
                del self.blocks[index]


class FullPolicy(MemoryPolicy):
    """Exact cache, no budget or horizon: the oracle."""

    name = "full"

    def __init__(self, geometry: Geometry | None = None, **_: object) -> None:
        super().__init__(None, None, geometry)

    def target_level(self, block: Block, now: int) -> int | None:
        return 0


class WindowPolicy(MemoryPolicy):
    """The last ``window`` frames at full fidelity (all of the budget by default)."""

    name = "window"

    def __init__(
        self,
        budget_tokens: int,
        horizon: int | None = None,
        window: int | None = None,
        geometry: Geometry | None = None,
    ) -> None:
        super().__init__(budget_tokens, horizon, geometry)
        self.window = window or budget_tokens // self.geometry.tokens_per_frame

    def target_level(self, block: Block, now: int) -> int | None:
        return 0 if now - block.newest < self.window else None


class WindowSinkPolicy(MemoryPolicy):
    """LongLive-style: the first ``sinks`` frames forever plus a recent window."""

    name = "window_sink"

    def __init__(
        self,
        budget_tokens: int,
        horizon: int | None = None,
        sinks: int = 1,
        geometry: Geometry | None = None,
    ) -> None:
        super().__init__(budget_tokens, horizon, geometry)
        self.sinks = sinks
        self.window = budget_tokens // self.geometry.tokens_per_frame - sinks
        if self.window < 1:
            raise ValueError("budget leaves no room for a window after the sinks")

    def is_sink(self, t: int) -> bool:
        return t < self.sinks

    def target_level(self, block: Block, now: int) -> int | None:
        return 0 if now - block.newest < self.window else None


class UniformSubsamplePolicy(MemoryPolicy):
    """A recent window plus every ``stride``-th older frame at full fidelity."""

    name = "uniform_subsample"

    def __init__(
        self,
        budget_tokens: int,
        horizon: int,
        window: int = 4,
        geometry: Geometry | None = None,
    ) -> None:
        super().__init__(budget_tokens, horizon, geometry)
        per_frame = self.geometry.tokens_per_frame
        spare_frames = budget_tokens // per_frame - window
        if spare_frames < 1:
            raise ValueError("budget leaves no room beyond the window")
        self.window = window
        self.stride = math.ceil((horizon - window) / spare_frames)

    def target_level(self, block: Block, now: int) -> int | None:
        if self._beyond_horizon(block, now):
            return None
        if now - block.newest < self.window or block.t0 % self.stride == 0:
            return 0
        return None


class RelicDiscretePolicy(MemoryPolicy):
    """A recent window plus a fixed, distance-independent compression pattern.

    Following RELIC's discrete schedule, each older frame keeps a spatial
    downsampling factor chosen by a repeating pattern over frame index
    (default 1×, 4×, 2×, 4× per side, i.e. levels 0, 4, 2, 4). To reach the
    horizon at equal budget, only every ``stride``-th older frame is kept.
    """

    name = "relic_discrete"

    def __init__(
        self,
        budget_tokens: int,
        horizon: int,
        window: int = 4,
        pattern: tuple[int, ...] = (0, 4, 2, 4),
        geometry: Geometry | None = None,
    ) -> None:
        super().__init__(budget_tokens, horizon, geometry)
        per_frame = self.geometry.tokens_per_frame
        spare = budget_tokens - window * per_frame
        if spare <= 0:
            raise ValueError("budget leaves no room beyond the window")
        mean_tokens = sum(self.geometry.tokens_per_frame_at(p) for p in pattern)
        mean_tokens /= len(pattern)
        self.window = window
        self.pattern = pattern
        self.stride = max(1, math.ceil((horizon - window) * mean_tokens / spare))

    def target_level(self, block: Block, now: int) -> int | None:
        if self._beyond_horizon(block, now):
            return None
        if now - block.newest < self.window:
            return 0
        if block.t0 % self.stride:
            return None
        return self.pattern[(block.t0 // self.stride) % len(self.pattern)]


DECAY_SHAPES: dict[str, Callable[[float, float], float]] = {
    "exp": lambda x, scale: math.exp(-x / scale),
    "power": lambda x, scale: (x + 1.0) ** -scale,
    "linear": lambda x, scale: max(1.0 - x / scale, 1e-9),
}


class DecayContinuousPolicy(MemoryPolicy):
    """The proposal: token density decays smoothly with distance.

    Density ``ρ(d)`` (tokens per frame) is ``grid²`` inside the window and
    ``ρ0 · shape(d - window)`` beyond it. A block's level is the smallest whose
    tokens per frame, ``grid² / 2**level``, do not exceed ``ρ(d)``; because each
    level halves the density, the cache steps down in factors of two at
    distances set by the continuous curve. ``ρ0`` is fitted so the steady-state
    cache fills the budget over the horizon.

    ``scale`` is ``τ`` for ``exp``, ``α`` for ``power``, and the zero-crossing
    distance for ``linear`` (default: the rest of the horizon).
    """

    name = "decay_continuous"

    def __init__(
        self,
        budget_tokens: int,
        horizon: int,
        window: int = 4,
        shape: str = "exp",
        scale: float | None = None,
        fill: float = 0.97,
        geometry: Geometry | None = None,
    ) -> None:
        super().__init__(budget_tokens, horizon, geometry)
        if shape not in DECAY_SHAPES:
            raise ValueError(f"unknown shape {shape!r}; choose {sorted(DECAY_SHAPES)}")
        defaults = {"exp": 256.0, "power": 1.0, "linear": float(horizon - window)}
        self.window = window
        self.shape = shape
        self.scale = defaults[shape] if scale is None else float(scale)
        self.max_level = self.geometry.spatial_levels + math.ceil(math.log2(horizon))
        self.rho0 = self._fit(budget_tokens, fill)

    def level_at(self, distance: float, rho0: float | None = None) -> int:
        if distance < self.window:
            return 0
        rho0 = self.rho0 if rho0 is None else rho0
        rho = rho0 * DECAY_SHAPES[self.shape](distance - self.window, self.scale)
        full = self.geometry.tokens_per_frame
        if rho >= full:
            return 0
        if rho <= 0:
            return self.max_level
        return min(math.ceil(math.log2(full / rho) - 1e-12), self.max_level)

    def steady_tokens(self, rho0: float) -> float:
        return sum(
            self.geometry.tokens_per_frame_at(self.level_at(d, rho0))
            for d in range(self.horizon)
        )

    def _fit(self, budget: float, fill: float) -> float:
        """Largest ``ρ0`` whose steady state fits; keep everything if it all fits.

        Below that, aim for ``fill · budget`` so aligned merges that are still
        waiting for a partner rarely trip the hard budget.
        """
        full = float(self.geometry.tokens_per_frame)
        if self.steady_tokens(full * 1e12) <= budget:
            return full * 1e12
        target = fill * budget
        low, high = -60.0, math.log(full) + 60.0
        for _ in range(100):
            middle = (low + high) / 2
            if self.steady_tokens(math.exp(middle)) <= target:
                low = middle
            else:
                high = middle
        return math.exp(low)

    def effective_distance(self, block: Block, now: int) -> float:
        return float(now - block.newest)

    def target_level(self, block: Block, now: int) -> int | None:
        if self._beyond_horizon(block, now):
            return None
        return self.level_at(self.effective_distance(block, now))


class DecayContentPolicy(DecayContinuousPolicy):
    """Continuous decay whose effective distance shrinks for salient frames (H4).

    Salience is feature novelty: one minus the cosine similarity between a
    frame's mean key and the previous frame's, relative to the running mean.
    A frame twice as novel as average ages half as fast, clipped to
    ``[1 / max_ratio, max_ratio]``. The budget is enforced as for every policy.
    """

    name = "decay_content"

    def __init__(self, *args, max_ratio: float = 2.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.max_ratio = max_ratio
        self._previous_key: torch.Tensor | None = None
        self._novelty_sum = 0.0
        self._novelty_count = 0

    def salience(self, k: torch.Tensor, t: int) -> float:
        mean_key = k.detach().to(torch.float64).mean(dim=-2).flatten()
        previous, self._previous_key = self._previous_key, mean_key
        if previous is None:
            return 1.0
        similarity = torch.nn.functional.cosine_similarity(mean_key, previous, dim=0)
        novelty = float(1.0 - similarity)
        self._novelty_sum += novelty
        self._novelty_count += 1
        average = self._novelty_sum / self._novelty_count
        if average <= 0:
            return 1.0
        return min(max(novelty / average, 1 / self.max_ratio), self.max_ratio)

    def effective_distance(self, block: Block, now: int) -> float:
        return (now - block.newest) / block.salience


POLICIES: dict[str, type[MemoryPolicy]] = {
    cls.name: cls
    for cls in (
        FullPolicy,
        WindowPolicy,
        WindowSinkPolicy,
        UniformSubsamplePolicy,
        RelicDiscretePolicy,
        DecayContinuousPolicy,
        DecayContentPolicy,
    )
}


def make_policy(name: str, **config) -> MemoryPolicy:
    if name not in POLICIES:
        raise ValueError(f"unknown policy {name!r}; choose from {sorted(POLICIES)}")
    return POLICIES[name](**config)
