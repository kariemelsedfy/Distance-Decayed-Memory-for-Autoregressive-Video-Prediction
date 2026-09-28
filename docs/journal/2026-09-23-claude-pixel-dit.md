# 2026-09-23 — Claude — Pixel DiT, flow matching, and sampler (issue #13)

## What

- `src/distance_decayed_memory/models/dit.py`: the frame-causal pixel
  diffusion transformer (4×4 patches, 3D RoPE, adaLN-Zero conditioning on
  action and noise level, chunk-causal attention) with a cache path that
  reads a memory policy's keys and values.
- `src/distance_decayed_memory/models/flow.py`: the flow-matching loss with
  one noise level per chunk, Euler sampling, clean re-encoding of each chunk
  into the cache, and a full autoregressive rollout (protocol P2).

## Why

This is the video model for both training stages. The cache path is how every
memory policy plugs in, so it must behave exactly like normal attention when
nothing is compressed.

## Evidence

11 CPU tests: patchify round trip, action shifting, zero-initialized output,
chunk causality with bidirectional attention inside a chunk, exact equality
between chunk-by-chunk generation with a full cache and the one-pass clip
forward, padded cache slots ignored, unit cell weights equal to no bias, loss
falls when fitting a small batch, deterministic sampling and rollout, and the
size ladder.

## Worth knowing

- The model sizes are larger than the plan's table (M is 172M, not 115M)
  because each block has its own adaLN layer. The attention/MLP trunk matches
  the table, and compute is barely affected. The A1 scaling check can still
  pick S if M is too slow.
- Training attention uses a dense mask for now; FlexAttention (already
  verified on the pro6000s) is the obvious speed-up in the A1 trainer.

## Plain-language explanation

This is the video model itself. It looks at the maze frames so far and the
agent's actions and paints the next four frames. It can read its memory from
any of the memory managers built in issue #12, and a test proves that, with
perfect memory, it gives exactly the same answer as when it sees everything
directly.
