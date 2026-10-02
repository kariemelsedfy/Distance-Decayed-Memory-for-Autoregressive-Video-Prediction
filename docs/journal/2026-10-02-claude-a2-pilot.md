# 2026-10-02 — Claude — A1 rollout diagnostics and the A2 pilot

## What

- **A1 rollout diagnostics** (`scripts/eval/rollout_diagnostics.py`, job
  `69372`): 64 val clips × 3 seeds, plus forced-action rollouts on 24 clips.
- **A2 pilot**: five policies (`window`, `window_sink`, `uniform_subsample`,
  `relic_discrete`, `decay_continuous`) at B-mid, seed 0, 5,000 steps from
  A1's EMA weights. All completed.
- **Pilot evaluation**: P1 on the first 100 val episodes for each policy,
  the drop ablation, the streaming check, the equivalence pair, and the
  `full` oracle (still running at the time of writing).
- Fixes along the way: A2 activation checkpointing; a memory leak in the
  policies; expandable CUDA segments for evaluation; `--script` for evaluation
  jobs; npz records in `fetch.sh`.

## Why

The A2 pilot is the last step before the A3 gate (`TRACK_A_PLAN.md` §11): the
sanity checks must pass, the cost per run must be measured, and the A2 step
count fixed.

## What I learned

- **A1 follows actions.** Forced left/right turns shift the view by about
  ±11.7 px per frame (real frames: +9.8 / −8.9); noop holds still. Errors
  after about ten generated frames look like sampling, not bias: two seeds
  differ from each other as much as from the truth. Many rollouts lock onto a
  wall close-up; my automatic `stuck` rule (motion and detail ratios) missed
  all of them, so that number in `diagnostics.json` is wrong.
- **The A2 memory leak.** Policies stored per-frame *views* of the step's
  stacked keys and values, so any frame kept at full fidelity pinned the
  whole ~4.5 GB step buffer. `decay_continuous` pools everything and was
  unaffected; `relic_discrete` ran out of memory. Blocks now hold copies.
- **Cost.** ≈0.65 steps/s with L and activation checkpointing, so ≈8.5
  GPU-hours per 20k-step run (D-012's estimate holds). Loss levels off by
  about 2,500 steps.
- **The oracle is expensive.** Evaluation keeps the cache in fp32 and holds
  about four copies of it while assembling the model's cache, so `full` at
  H 2,048 needs about 200 GB+; even H 512 failed. The pilot oracle uses
  H 256. In A2 training, `full` cannot fit 8 streams at all.
- **The main result: no memory benefit yet.** On P1, `window` is best in every
  gap bucket and on novel views (LPIPS 0.118 novel, 0.09 at gaps 16–31,
  0.14–0.19 beyond); `decay_continuous`, `relic_discrete`, and
  `uniform_subsample` are 0.03–0.09 worse everywhere. Dropping
  `decay_continuous`'s memory hurts revisits and novel views by about the
  same amount (+0.02 to +0.06), so check 5 fails: the model uses memory as
  general context, not to recall revisited places.
- Checks that pass: equivalence (exact, zero difference), budgets, and
  streaming = inference for `full` (6e-7). Compressing policies differ by
  3.7–6.0%, the D-011 structural difference.

## How to verify

```
python scripts/eval/summarize.py outputs/eval-pilot/eval-pilot-*/ \
  --output outputs/eval-pilot/summary.json
```
after fetching each run with `scripts/hpc/fetch.sh <run-id>`; run IDs are in
`docs/EXPERIMENTS.md`.

## Plain-language explanation

We taught five versions of the model to use five different kinds of memory,
each allowed the same amount of storage, and tested how well each predicts
what it sees when it comes back to a place. The simplest one, "remember only
the last couple of seconds in full detail", won everywhere, even on views
nobody had seen before. That means the model is not yet using old memories to
recall places; it mostly benefits from having more recent frames in full
detail. So the pilot cannot yet say whether our smoothly fading memory is
better, because no memory is helping with recall at all. The scoring code
itself checks out.

## Next

The oracle (`full`, 256 frames) decides how to read this: if perfect memory
beats `window` on revisits, recall is possible and the compressed policies
fail to use it; if not, the training setup has to change before A3. Either
way, A3 should not launch as planned; options are in `STATUS.md`.
