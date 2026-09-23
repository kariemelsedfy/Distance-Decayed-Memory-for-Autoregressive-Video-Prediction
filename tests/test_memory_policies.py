from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from distance_decayed_memory.memory import Geometry, make_policy  # noqa: E402
from distance_decayed_memory.memory.attention import cache_attention  # noqa: E402
from distance_decayed_memory.memory.cells import (  # noqa: E402
    Block,
    merge_temporal,
    pool_spatial,
)
from distance_decayed_memory.memory.policies import (  # noqa: E402
    POLICIES,
    DecayContinuousPolicy,
)
from distance_decayed_memory.memory.rope import apply_rope  # noqa: E402

GRID = 4  # 16 tokens per frame keeps the tests fast; the logic is grid-agnostic
PER_FRAME = GRID * GRID
HORIZON = 256
LAYERS, HEADS, DIM = 2, 2, 6
BUDGETED = sorted(set(POLICIES) - {"full"})


def policy(name: str, budget: int, horizon: int = HORIZON, **config):
    """Build a policy on the small test grid, with a 2-frame window by default."""
    if name == "full":
        return make_policy(name, geometry=Geometry(GRID))
    if name in {"uniform_subsample", "relic_discrete", "decay_continuous"} | {
        "decay_content"
    }:
        config = {"window": 2, **config}
    return make_policy(
        name, geometry=Geometry(GRID), budget_tokens=budget, horizon=horizon, **config
    )


def frames(count: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    shape = (LAYERS, HEADS, count * PER_FRAME, DIM)
    return (
        torch.randn(shape, generator=generator, dtype=torch.float64),
        torch.randn(shape, generator=generator, dtype=torch.float64),
    )


def run(p, total: int, chunk: int = 4, seed: int = 0, each_step=None):
    k_all, v_all = frames(total, seed)
    for t in range(0, total, chunk):
        span = slice(t * PER_FRAME, (t + chunk) * PER_FRAME)
        p.append(k_all[..., span, :], v_all[..., span, :], None, t)
        p.compact()
        if each_step is not None:
            each_step(p)
    return k_all, v_all


def originals_for(block: Block, geometry: Geometry, source: torch.Tensor):
    """Mean of the original tokens each of ``block``'s cells covers."""
    rows, columns = geometry.shape(block.level)
    cell_rows, cell_columns = geometry.grid // rows, geometry.grid // columns
    frames_ = source[..., block.t0 * PER_FRAME : (block.newest + 1) * PER_FRAME, :]
    grid = frames_.reshape(*frames_.shape[:-2], block.frames, rows, cell_rows, -1)
    grid = grid.reshape(
        *frames_.shape[:-2], block.frames, rows, cell_rows, columns, cell_columns, DIM
    )
    return grid.mean(dim=(-6, -4, -2)).reshape(*frames_.shape[:-2], -1, DIM)


@pytest.mark.parametrize("name", BUDGETED)
@pytest.mark.parametrize("budget", [4 * PER_FRAME + 8, 8 * PER_FRAME, 24 * PER_FRAME])
def test_budget_is_never_exceeded(name: str, budget: int) -> None:
    p = policy(name, budget)

    def check(p) -> None:
        assert p.total_tokens() <= budget
        assert p.kv()[0].shape[-2] == p.total_tokens()

    run(p, 3 * HORIZON, chunk=3, each_step=check)


@pytest.mark.parametrize(
    "name", ["uniform_subsample", "relic_discrete", "decay_continuous", "decay_content"]
)
def test_horizon_cutoff(name: str) -> None:
    p = policy(name, 16 * PER_FRAME)

    def check(p) -> None:
        assert all(p.now - block.newest < HORIZON for block in p.blocks)

    run(p, 3 * HORIZON, each_step=check)
    assert min(block.t0 for block in p.blocks) > HORIZON


@pytest.mark.parametrize("name", sorted(POLICIES))
def test_aging_is_monotone_and_dropped_frames_stay_dropped(name: str) -> None:
    p = policy(name, 10 * PER_FRAME)
    level: dict[int, int] = {}
    dropped: set[int] = set()

    def check(p) -> None:
        present: dict[int, int] = {}
        for block in p.blocks:
            for frame in range(block.t0, block.newest + 1):
                present[frame] = block.level
        for frame, previous in level.items():
            if frame in present:
                assert frame not in dropped
                assert present[frame] >= previous
            else:
                dropped.add(frame)
        level.update(present)

    run(p, 2 * HORIZON, each_step=check)


@pytest.mark.parametrize("name", ["decay_continuous", "relic_discrete", "window_sink"])
def test_every_cell_is_the_mean_of_the_tokens_it_covers(name: str) -> None:
    p = policy(name, 12 * PER_FRAME)
    k_all, v_all = run(p, 2 * HORIZON)
    for block in p.blocks:
        torch.testing.assert_close(block.k, originals_for(block, p.geometry, k_all))
        torch.testing.assert_close(block.v, originals_for(block, p.geometry, v_all))


def test_positions_are_centroids() -> None:
    geometry = Geometry(GRID)
    k, v = frames(2, seed=1)
    older = Block(
        0,
        1,
        0,
        k[..., :PER_FRAME, :],
        v[..., :PER_FRAME, :],
        geometry.frame_positions(0),
    )
    newer = Block(
        1,
        1,
        0,
        k[..., PER_FRAME:, :],
        v[..., PER_FRAME:, :],
        geometry.frame_positions(1),
    )
    for block in (older, newer):
        while block.level < geometry.spatial_levels:
            pool_spatial(block, geometry)
    torch.testing.assert_close(
        older.pos,
        torch.tensor([[0.0, (GRID - 1) / 2, (GRID - 1) / 2]], dtype=torch.float64),
    )
    merged = merge_temporal(older, newer, geometry)
    assert merged.frames == 2 and merged.volume == 2 * PER_FRAME
    torch.testing.assert_close(merged.pos[0, 0], torch.tensor(0.5, dtype=torch.float64))


def test_full_policy_equals_exact_attention() -> None:
    p = policy("full", 0)
    k_all, v_all = run(p, 64)
    k, v, pos, weight = p.kv()
    assert torch.equal(k, k_all) and torch.equal(v, v_all)
    assert torch.all(weight == 1)
    q = torch.randn(LAYERS, HEADS, PER_FRAME, DIM, dtype=torch.float64)
    q_pos = Geometry(GRID).frame_positions(64)
    exact_pos = torch.cat([Geometry(GRID).frame_positions(t) for t in range(64)])
    expected = cache_attention(q, q_pos, k_all, v_all, exact_pos, proportional=False)
    actual = cache_attention(q, q_pos, k, v, pos, weight)
    torch.testing.assert_close(actual, expected)


def test_decay_with_ample_budget_reproduces_full() -> None:
    total = 128
    full, decay = policy("full", 0), policy(
        "decay_continuous", (total + 1) * PER_FRAME, horizon=total + 1
    )
    run(full, total)
    run(decay, total)
    for mine, theirs in zip(decay.kv(), full.kv(), strict=True):
        torch.testing.assert_close(mine, theirs)


def test_rope_depends_only_on_relative_fractional_offset() -> None:
    generator = torch.Generator().manual_seed(3)
    q = torch.randn(1, 1, DIM, generator=generator, dtype=torch.float64)
    k = torch.randn(1, 1, DIM, generator=generator, dtype=torch.float64)
    q_pos = torch.tensor([[5.25, 1.5, 2.75]], dtype=torch.float64)
    k_pos = torch.tensor([[1.5, 0.25, 3.0]], dtype=torch.float64)
    shift = torch.tensor([[17.125, -3.5, 0.625]], dtype=torch.float64)
    before = (apply_rope(q, q_pos) * apply_rope(k, k_pos)).sum()
    after = (apply_rope(q, q_pos + shift) * apply_rope(k, k_pos + shift)).sum()
    torch.testing.assert_close(before, after)
    zero = torch.zeros(1, 3, dtype=torch.float64)
    torch.testing.assert_close(apply_rope(k, zero), k)


def test_rope_at_centroid_differs_from_averaging_rotated_keys() -> None:
    geometry = Geometry(GRID)
    key = torch.randn(1, 1, 1, DIM, dtype=torch.float64).expand(1, 1, PER_FRAME, DIM)
    block = Block(0, 1, 0, key.clone(), key.clone(), geometry.frame_positions(0))
    while block.level < geometry.spatial_levels:
        pool_spatial(block, geometry)
    at_centroid = apply_rope(block.k, block.pos)
    torch.testing.assert_close(at_centroid, apply_rope(key[..., :1, :], block.pos))
    averaged = apply_rope(key, geometry.frame_positions(0)).mean(dim=-2, keepdim=True)
    assert not torch.allclose(at_centroid, averaged)


def test_stats_and_density_are_consistent() -> None:
    p = policy("decay_continuous", 12 * PER_FRAME)
    run(p, 2 * HORIZON)
    stats = p.stats()
    assert stats["total_tokens"] == p.kv()[0].shape[-2]
    assert sum(stats["tokens_by_distance"].values()) == stats["total_tokens"]
    assert sum(stats["tokens_by_level"].values()) == stats["total_tokens"]
    assert math.isclose(float(p.density(HORIZON).sum()), stats["total_tokens"])
    assert stats["kv_bytes"] == 2 * stats["total_tokens"] * LAYERS * HEADS * DIM * 8


def test_window_keeps_exactly_the_recent_frames() -> None:
    p = policy("window", 8 * PER_FRAME)
    run(p, 100)
    density = p.density(20)
    assert torch.all(density[:8] == PER_FRAME) and torch.all(density[8:] == 0)


def test_window_sink_keeps_the_first_frame() -> None:
    p = policy("window_sink", 8 * PER_FRAME, sinks=1)
    run(p, 200)
    assert p.blocks[0].t0 == 0 and p.blocks[0].level == 0
    assert p.total_tokens() == 8 * PER_FRAME


def test_relic_levels_do_not_depend_on_distance() -> None:
    p = policy("relic_discrete", 16 * PER_FRAME)
    seen: dict[int, int] = {}

    def check(p) -> None:
        for block in p.blocks:
            if p.now - block.newest >= p.window and block.level <= 4:
                seen.setdefault(block.t0, block.level)
                assert block.level == seen[block.t0]

    run(p, HORIZON, each_step=check)
    assert {block.level for block in p.blocks} >= {0, 2, 4}


def test_decay_schedule_coarsens_with_distance() -> None:
    p = DecayContinuousPolicy(
        16 * PER_FRAME, HORIZON, window=2, geometry=Geometry(GRID)
    )
    levels = [p.level_at(d) for d in range(HORIZON)]
    assert levels[:2] == [0, 0]
    assert all(a <= b for a, b in zip(levels, levels[1:], strict=False))
    assert p.steady_tokens(p.rho0) <= 16 * PER_FRAME


@pytest.mark.parametrize("shape", ["exp", "power", "linear"])
def test_decay_shapes_fit_the_budget(shape: str) -> None:
    p = policy("decay_continuous", 16 * PER_FRAME, shape=shape)
    assert p.steady_tokens(p.rho0) <= 16 * PER_FRAME
    run(p, 2 * HORIZON)
    assert p.total_tokens() <= 16 * PER_FRAME


def test_detach_before_cuts_old_gradients() -> None:
    p = policy("window", 8 * PER_FRAME)
    k, v = frames(8, seed=2)
    k.requires_grad_(True)
    p.append(k, v, None, 0)
    p.detach_before(6)
    assert [b.k.requires_grad for b in p.blocks] == [False] * 6 + [True] * 2


def test_input_validation() -> None:
    p = policy("window", 8 * PER_FRAME)
    k, v = frames(1, seed=0)
    with pytest.raises(ValueError, match="whole frames"):
        p.append(k[..., :-1, :], v[..., :-1, :], None, 0)
    p.append(k, v, None, 0)
    with pytest.raises(ValueError, match="not after"):
        p.append(k, v, None, 0)
    with pytest.raises(ValueError, match="standard"):
        p.append(k, v, torch.zeros(PER_FRAME, 3), 1)
    with pytest.raises(ValueError, match="unknown policy"):
        make_policy("nope")


def test_rope_splits_model_head_dimensions_evenly() -> None:
    from distance_decayed_memory.memory.rope import axis_sizes

    assert axis_sizes(64) == (24, 20, 20)
    assert axis_sizes(6) == (2, 2, 2)
    q = torch.randn(2, 3, 5, 64, dtype=torch.float64)
    k = torch.randn(2, 3, 5, 64, dtype=torch.float64)
    pos = torch.rand(2, 1, 5, 3, dtype=torch.float64) * 100
    shift = torch.tensor([3.5, -1.25, 7.0], dtype=torch.float64)
    before = apply_rope(q, pos) @ apply_rope(k, pos).transpose(-1, -2)
    after = apply_rope(q, pos + shift) @ apply_rope(k, pos + shift).transpose(-1, -2)
    torch.testing.assert_close(before, after)
    with pytest.raises(ValueError, match="even"):
        axis_sizes(63)
