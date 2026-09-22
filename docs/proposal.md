
# Bounded-Buffer Distance-Decayed Memory for Autoregressive Video Prediction

## 1. Problem statement

Causal, frame-by-frame video generation (predicting latent frame `t+1` from frame `t` and history) needs some form of memory of the past. The two standard extremes are:

- **Full KV cache** — exact memory of every past frame, but cost grows linearly with video length, which is intractable for long videos.
- **Plain sliding window** — only the last `w` frames are kept; anything older is dropped with zero trace. Cheap, but content that leaves the window is gone permanently, causing drift and loss of long-range consistency (e.g. a character leaving and re-entering frame).

This document describes a proposed middle point between these extremes.

## 2. Proposed design

A single latent buffer of fixed size, corresponding to a bounded time horizon (e.g. ~10 minutes of video), used for causal next-frame prediction:

- **Recent frames (small window near `t`)** are kept at full fidelity — exact latent tokens, dense attention.
- **Older frames within the buffer** are kept at reduced fidelity, with the degree of compression **decaying continuously as a function of distance from the current frame `t`** — closer frames are compressed less, more distant frames are compressed more.
- **Anything older than the buffer horizon is not stored at all** — a hard cutoff, not compressed further.
- The current frame's prediction of `t+1` is computed by attending across the whole buffer, with attention weight/fidelity naturally decaying toward the older end (a "radial"-style decay, but applied to the causal cache rather than a bidirectional clip).

This differs from a hard sliding window (single fidelity level, then a cliff) by degrading gracefully rather than dropping abruptly, and it differs from unbounded KV caching by capping total memory and compute at a fixed size regardless of video length.

### Open parameters
- Buffer horizon (proposed: ~10 minutes)
- Shape of the decay function (linear, exponential, log-spaced steps, etc.)
- Whether decay is applied per-frame, per-spatial-token, or both
- Whether compression is spatial-only (fewer tokens per frame), temporal-only (fewer frames sampled, e.g. every t-2, t-4, t-8...), or both

## 3. Relationship to existing work

No system found combines all of the following properties at once — this is the specific gap the design sits in:

| Property | Found in | Not found combined with the others |
|---|---|---|
| Continuous, distance-based decay of attention/fidelity | **Radial Attention** (exponential decay by spatial+temporal distance) | Built for bidirectional inference speedup on an already-generated clip, not a causal generation cache |
| Bounded buffer with degraded (not dropped) far history | **RELIC** (rolling-window cache + compressed long-horizon cache) | Compression is a **fixed discrete schedule** (1x/2x/4x spatial downsampling, interleaved by a preset pattern) — not a continuous function of distance, and it currently scales to ~20 seconds, not minutes |
| Hard window + minimal historical anchors | **LongLive** (short window + frame-level attention sinks) | No graded degradation — it's window-plus-anchor, not distance-decayed |
| Distance-based empirical attention decay (evidence, not architecture) | **Sparse Forcing**, calibrated sparse attention studies | Describe the phenomenon in existing bidirectional/short-context models; don't propose a causal buffer built around it |
| Compressed running state instead of exact cache for distant history | Linear-attention hybrid streaming paper | Uses a single accumulated state, not a graded multi-level buffer |
| Minute-scale causal generation | **LongLive** (240s), **VideoMLA** (targeting minute-scale) | Neither uses distance-graded compression; LongLive uses window+sinks, VideoMLA compresses per-token representation (low-rank), not per-distance |

**Summary:** the pieces (bounded windows, compression instead of dropping, distance-based decay) are each independently established. The specific combination — a continuous, distance-scaled compression gradient inside a single bounded causal buffer, purpose-built for plain next-frame prediction at multi-minute scale — has not been demonstrated as of this writing (per available search, mid-2026).

## 4. Known risks, from analogous systems

- **Empirical evidence against pure distance-based retention:** a KV-cache attention study (PackForcing) found that attention does *not* concentrate predictably on "recent + very start" — content far back in the sequence kept mattering unpredictably. A purely distance-based decay schedule may discard information that turns out to matter, regardless of how gradual the decay is.
- **Content-independent decay vs. content-dependent relevance:** Radial Attention's decay is an empirical average over many generations; RELIC and Sparse Forcing both found that some form of *content-aware* selection (pose-tagging, salience/anchor detection) outperforms pure recency/distance heuristics. A purely geometric decay schedule (no content awareness) may underperform a learned or salience-based policy at the same compute budget.
- **Training, not just architecture, is required to make this work.** RELIC only functions because its teacher was trained on long (20s) clips with an explicit "time-reverse" augmentation forcing viewpoint revisitation. A distance-decayed buffer would likely need equivalent training pressure — plain short-clip fine-tuning is unlikely to teach a model to actually use compressed distant memory.
- **Scale gap:** the frontier for causal video memory as of mid-2026 sits around 20 seconds (RELIC) to 4 minutes (LongLive). A 10-minute horizon is beyond what's been publicly demonstrated with any memory mechanism, graded or otherwise.

## 5. Suggested minimal experiment

To test whether continuous distance-decay outperforms RELIC's discrete schedule, at small scale:

1. Start from a small pretrained causal video model (or distill one, per CausVid/Self-Forcing recipes) rather than training from scratch.
2. Implement two cache variants on top of the same base model:
   - **Baseline:** RELIC-style fixed discrete schedule (rolling window + 1x/2x/4x downsampled remainder).
   - **Proposed:** same total budget, but with a smooth decay function (e.g. exponential) determining per-frame compression level continuously by distance.
3. Train both with the same recipe, including revisitation-style augmentation (time-reversal or looped camera paths) to give the model a reason to use distant memory at all.
4. Evaluate at a horizon well beyond the training window (e.g. train on 20–60s clips, evaluate consistency at 2–5 minutes) — long-horizon identity/scene consistency after a revisit is the metric that would actually distinguish these two designs, not short-clip quality.

