# 2026-09-23 — Claude — Memory policy library (issue #12)

## What

`src/distance_decayed_memory/memory/` implements every policy in the plan
(`full`, `window`, `window_sink`, `uniform_subsample`, `relic_discrete`,
`decay_continuous`, `decay_content`), 3D RoPE at fractional positions, and a
reference attention with the optional `log v` bias.
`scripts/figures/memory_policy_density.py` draws the Gate 1 figure.

## Why

The policies are the project's contribution, and the plan requires them to be
tested before any experiment uses them.

## How it works

Every policy stores the same kind of cells. Old memory is shrunk by averaging
neighbouring pairs of tokens, so each stored token is exactly the average of
the originals it replaces. A policy is only a rule for how coarse memory
should be at each distance. See D-010.

## Evidence

- 47 policy tests: the budget is never exceeded, the horizon cutoff holds, aging is
  monotone and dropped frames never return, every cell equals the mean of the
  tokens it covers, positions are centroids, `full` matches exact attention,
  `decay_continuous` with an ample budget reproduces `full`, RoPE depends only
  on relative (fractional) offsets, and `stats()` is consistent.
- The Gate 1 figure at 4,096 tokens shows the contrast the paper tests: the
  proposal keeps about 8–16 tokens per frame for ~250 frames and then fades
  smoothly; RELIC and uniform subsampling spend 1–2 tokens per frame evenly
  to the horizon; the windows spend everything on the last 16 frames.

## Open points

- RELIC's exact downsampling pattern needs checking against the paper
  (marked *(verify)* in D-010).
- `decay_content` uses key novelty as salience; attention-mass salience is an
  alternative for the H4 ablation.

## Plain-language explanation

We built the different "memory managers" the project compares. Each gets the
same small memory allowance. The picture shows how each one spends it: ours
keeps recent history in medium detail and older history in fading detail,
while the others either forget everything old or keep old frames equally
blurry no matter how old they are.
