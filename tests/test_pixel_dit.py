from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from distance_decayed_memory.memory import Geometry, make_policy  # noqa: E402
from distance_decayed_memory.models.dit import (  # noqa: E402
    SIZES,
    DiTConfig,
    ModelCache,
    PixelDiT,
    conditioning_actions,
    count_parameters,
    patchify,
    unpatchify,
)
from distance_decayed_memory.models.flow import (  # noqa: E402
    chunk_noise_levels,
    encode_chunk,
    flow_matching_loss,
    rollout,
    sample_chunk,
    to_model_range,
    to_uint8,
)

TINY = DiTConfig(layers=2, width=48, heads=2, image_size=16, patch=4, chunk_frames=2)
FRAMES = 8


def tiny_model(seed: int = 0, randomize: bool = True) -> PixelDiT:
    torch.manual_seed(seed)
    model = PixelDiT(TINY).double()
    if randomize:
        # adaLN-Zero starts every block as the identity; perturb so tests see signal.
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(0.05 * torch.randn_like(parameter))
    return model.eval()


def clip(seed: int = 1, frames: int = FRAMES):
    generator = torch.Generator().manual_seed(seed)
    x = torch.rand(2, frames, 16, 16, 3, generator=generator, dtype=torch.float64)
    actions = torch.randint(0, 6, (2, frames), generator=generator)
    prev = conditioning_actions(actions, torch.full((2,), TINY.no_action))
    levels = chunk_noise_levels(2, frames, TINY.chunk_frames, generator=generator)
    return 2 * x - 1, prev, levels.double()


def test_patchify_round_trip() -> None:
    x = torch.randn(2, 3, 16, 16, 3)
    tokens = patchify(x, 4)
    assert tokens.shape == (2, 3 * 16, 48)
    torch.testing.assert_close(unpatchify(tokens, 3, 16, 4, 3), x)
    # The first token is the top-left 4×4 patch of the first frame.
    torch.testing.assert_close(tokens[0, 0], x[0, 0, :4, :4].reshape(-1))


def test_conditioning_actions_shift_by_one() -> None:
    actions = torch.tensor([[1, 2, 3]])
    out = conditioning_actions(actions, torch.tensor([TINY.no_action]))
    assert out.tolist() == [[TINY.no_action, 1, 2]]


def test_zero_initialised_model_predicts_zero() -> None:
    model = tiny_model(randomize=False)
    x, prev, levels = clip()
    assert torch.count_nonzero(model(x, levels, prev)) == 0


def test_chunks_are_causal_and_bidirectional_within() -> None:
    model = tiny_model()
    x, prev, levels = clip()
    base = model(x, levels, prev)
    changed = x.clone()
    changed[:, 5] += 1.0  # frame 5 belongs to chunk 2 (frames 4-5)
    out = model(changed, levels, prev)
    torch.testing.assert_close(out[:, :4], base[:, :4])
    assert not torch.allclose(out[:, 4], base[:, 4])  # same chunk sees it
    assert not torch.allclose(out[:, 6:], base[:, 6:])  # later chunks see it


def test_streaming_with_full_cache_equals_clip_forward() -> None:
    model = tiny_model()
    x, prev, levels = clip()
    expected = model(x, levels, prev)
    policies = [make_policy("full", geometry=Geometry(TINY.grid)) for _ in range(2)]
    chunk = TINY.chunk_frames
    for start in range(0, FRAMES, chunk):
        span = slice(start, start + chunk)
        cache = ModelCache.from_policies(policies) if start else None
        out, k, v = model(
            x[:, span], levels[:, span], prev[:, span], start, cache, return_kv=True
        )
        torch.testing.assert_close(out, expected[:, span])
        for index, policy in enumerate(policies):
            policy.append(k[index], v[index], None, start)
            policy.compact()


def test_unit_weights_match_no_proportional_bias() -> None:
    model = tiny_model()
    x, prev, levels = clip()
    policies = [make_policy("full", geometry=Geometry(TINY.grid)) for _ in range(2)]
    _, k, v = model(x[:, :2], levels[:, :2], prev[:, :2], 0, None, return_kv=True)
    for index, policy in enumerate(policies):
        policy.append(k[index], v[index], None, 0)
    cache = ModelCache.from_policies(policies)
    with_bias = model(x[:, 2:4], levels[:, 2:4], prev[:, 2:4], 2, cache)
    model.config = DiTConfig(**{**TINY.__dict__, "proportional_attention": False})
    without = model(x[:, 2:4], levels[:, 2:4], prev[:, 2:4], 2, cache)
    torch.testing.assert_close(with_bias, without)


def test_padded_cache_slots_are_ignored() -> None:
    model = tiny_model()
    x, prev, levels = clip()
    short = make_policy("full", geometry=Geometry(TINY.grid))
    long = make_policy("full", geometry=Geometry(TINY.grid))
    _, k, v = model(x[:, :4], levels[:, :4], prev[:, :4], 0, None, return_kv=True)
    short.append(
        k[0, ..., : 2 * TINY.tokens_per_frame, :],
        v[0, ..., : 2 * TINY.tokens_per_frame, :],
        None,
        0,
    )
    long.append(k[1], v[1], None, 0)
    batch = ModelCache.from_policies([short, long])
    alone = ModelCache.from_policies([short])
    span = slice(4, 6)
    out = model(x[:, span], levels[:, span], prev[:, span], 4, batch)
    solo = model(x[:1, span], levels[:1, span], prev[:1, span], 4, alone)
    torch.testing.assert_close(out[:1], solo)


def test_training_reduces_loss_on_a_fixed_batch() -> None:
    torch.manual_seed(0)
    model = PixelDiT(TINY)
    x, prev, _ = clip()
    x = x.float()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    losses = []
    for step in range(60):
        generator = torch.Generator().manual_seed(step % 4)
        loss = flow_matching_loss(model, x, prev, generator=generator)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert sum(losses[-4:]) < 0.6 * sum(losses[:4])


def test_sampling_and_rollout_shapes_and_determinism() -> None:
    model = tiny_model().float()
    x, prev, _ = clip()
    x = x.float()
    first = sample_chunk(
        model, prev[:, :2], 0, None, steps=4, generator=torch.Generator().manual_seed(7)
    )
    again = sample_chunk(
        model, prev[:, :2], 0, None, steps=4, generator=torch.Generator().manual_seed(7)
    )
    assert first.shape == (2, 2, 16, 16, 3)
    torch.testing.assert_close(first, again)
    k, v = encode_chunk(model, first, prev[:, :2], 0, None)
    assert k.shape == (
        2,
        TINY.layers,
        TINY.heads,
        2 * TINY.tokens_per_frame,
        TINY.head_dim,
    )

    budget = 3 * TINY.tokens_per_frame

    def window():
        return make_policy("window", budget_tokens=budget, geometry=Geometry(TINY.grid))

    video = rollout(
        model,
        window,
        x[:, :2],
        prev,
        FRAMES,
        steps=2,
        generator=torch.Generator().manual_seed(0),
    )
    assert video.shape == (2, FRAMES, 16, 16, 3)
    torch.testing.assert_close(video[:, :2], x[:, :2])
    assert to_uint8(video).dtype == torch.uint8


def test_pixel_range_conversion_round_trips() -> None:
    values = torch.arange(256, dtype=torch.uint8)
    assert torch.equal(to_uint8(to_model_range(values)), values)


def test_size_ladder_parameter_counts() -> None:
    counts = {name: count_parameters(config) for name, config in SIZES.items()}
    assert counts["S"] < counts["M"] < counts["L"]
    # Transformer trunk (attention + MLP) matches the plan's ~40/115/300M;
    # per-block adaLN adds about half again (TRACK_A_PLAN.md §4 note).
    for name, trunk in (("S", 38e6), ("M", 113e6), ("L", 302e6)):
        assert 1.3 * trunk < counts[name] < 1.6 * trunk
    assert all(config.head_dim == 64 for config in SIZES.values())


def test_flex_block_mask_matches_dense_mask() -> None:
    pytest.importorskip("torch.nn.attention.flex_attention")
    model = tiny_model().float()
    x, prev, levels = clip()
    with torch.no_grad():
        dense = model(x.float(), levels.float(), prev)
        model.attention_backend = "flex"
        flex = model(x.float(), levels.float(), prev)
    torch.testing.assert_close(flex, dense, atol=1e-5, rtol=1e-4)


def test_activation_checkpointing_preserves_gradients() -> None:
    x, prev, levels = clip()
    grads = []
    for checkpointing in (False, True):
        model = tiny_model().train()
        model.activation_checkpointing = checkpointing
        model(x, levels, prev).square().mean().backward()
        grads.append(torch.cat([p.grad.flatten() for p in model.parameters()]))
    torch.testing.assert_close(grads[0], grads[1])
