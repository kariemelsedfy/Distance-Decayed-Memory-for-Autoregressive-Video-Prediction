# Decision log

This file is append-only. Each decision records the constraint it establishes so
future work does not silently change experimental meaning.

## D-001 — Track A is the only active track

- **Status:** accepted (2026-09-22)
- **Decision:** Track A only; Track B is deferred until the owner reopens it after Gate 2.
- **Consequence:** Do not download Wan checkpoints, prepare Track B data, or use GPU time for Track B.

## D-002 — Start with Memory Maze pixels

- **Status:** accepted (2026-09-22)
- **Decision:** Use Memory Maze 9×9 first, with 64×64 pixels, 4×4 patches, and no VAE.
- **Consequence:** The controlled test remains small and avoids representation-learning confounds from a VAE.

## D-003 — Separate base learning from memory learning

- **Status:** accepted (2026-09-22)
- **Decision:** Train one shared A1 base model, then start every policy-specific A2 memory fine-tune from that checkpoint.
- **Consequence:** Policy comparisons share the same visual and dynamics model and use less compute.

## D-004 — Use streaming A2 training

- **Status:** accepted (2026-09-22)
- **Decision:** A2 uses live caches and a two-chunk gradient window instead of single-pass teacher forcing.
- **Consequence:** Training matches inference cache behavior with linear cost, while older K/V entries are detached and may become stale.

## D-005 — Match budgets as fractions of the horizon

- **Status:** accepted (2026-09-22)
- **Decision:** Use 0.4%, 0.8%, and 2.3% of horizon tokens; always include the full-cache oracle.
- **Consequence:** Track A mirrors large-model scarcity while retaining a true upper bound.

## D-006 — Equal tuning and a frozen test set

- **Status:** accepted (2026-09-22)
- **Decision:** Give every policy equal tuning effort and freeze and hash the test split before final evaluation.
- **Consequence:** The proposed policy cannot receive an unfair search advantage, and the final comparison remains auditable.

## D-007 — Plan Track A around four concurrent GPUs

- **Status:** accepted (2026-09-23)
- **Decision:** Build the compute plan on the measured Slurm ceiling rather than
  on the seven installed cards: at most 2 pro6000 GPUs in one job, and at most 4
  concurrently per user (2 jobs on `mixed`, 2 on `gpu`). A1 becomes a 2-GPU DDP
  run; the A2 sweep runs as single-GPU jobs, four at a time, in the staged
  priority order of `TRACK_A_PLAN.md` §9.3. Do not request a raised QOS limit
  for now.
- **Consequence:** A2 plus evaluation is roughly 8-17 days of wall clock instead
  of 5-10, so the sweep needs a queue runner with auto-resume, and per-run cost
  must be measured in the A2 pilot before the sweep is launched. Stopping the
  sweep early still yields the headline result, because the protected core runs
  first.

## D-008 — Scripted-revisit episode conventions

- **Status:** accepted (2026-09-23); owner may revisit at the A0 check-in
- **Decision:** Generate episodes with Memory Maze's pinned private builder
  `tasks._memory_maze` so episodes can exceed the public 1,000-step 9×9 time
  limit; disable `target_color_in_image` so the frame border does not encode
  task state the scripted agent ignores; store `actions[t]` as the action taken
  after `frames[t]` (it produces `frames[t + 1]`); and score revisits by the
  **realized** gap (`return_frame − anchor_frame`), keeping the sampled target
  gap only as metadata. Returns that arrive before `min_gap` are labelled
  `early` and are not scripted revisits.
- **Consequence:** Frames show only the maze, the agent's view, and objects;
  the private builder is tied to `memory-maze==1.0.3` and must be rechecked on
  any upgrade. Gap buckets come from what actually happened, not from the
  schedule.

## D-009 — A revisit requires leaving the view first

- **Status:** accepted (2026-09-23); tolerances to be confirmed at the A0 pilot
- **Decision:** Refine `TRACK_A_PLAN.md` §3.3. The source of a revisit is the
  most recent matching pose **before the current unbroken run of matching
  frames**, not simply the most recent match older than `w_min`. Frames with
  no such source are novel; sources fewer than 16 frames back are labelled
  `recent`; the rest are revisits with gap `t − t'`. Every frame also records
  `visit_age`, the length of that unbroken run. Tolerances stay at 0.3 cell
  and 15°.
- **Consequence:** Standing still or lingering can no longer create fake
  short-gap revisits, so the `[16, 32)` bucket measures memory rather than
  continuity. Evaluation can additionally drop frames with a large
  `visit_age`.

## D-010 — One cell mechanism for every memory policy

- **Status:** accepted (2026-09-23); RELIC pattern marked *(verify)*
- **Decision:** All policies share one mechanism (`src/distance_decayed_memory/memory/`):
  a frame enters as 16×16 pre-RoPE tokens; each aging step averages adjacent
  pairs (rows, then columns, down to one token per frame, then temporally
  aligned pairs of frames), so a level-ℓ token is the exact mean of `2**ℓ`
  original tokens and cells never become finer. Policies differ only in the
  target level at distance `d` (measured from a block's newest frame), and a
  shared budget step coarsens or drops the oldest blocks if needed.
  `decay_continuous` fits `ρ0` so its steady state uses 97% of the budget
  (or keeps everything when that fits). `relic_discrete` keeps a window plus
  a repeating, distance-independent per-frame pattern of 1×/4×/2×/4× spatial
  downsampling on every `k`-th older frame, `k` chosen to fit the budget; the
  pattern must be checked against the RELIC paper *(verify)*. `decay_content`
  divides distance by a clipped novelty ratio of each frame's mean key.
- **Consequence:** Continuous decay is realized as many factor-of-two steps
  whose positions follow the continuous curve, rather than arbitrary
  fractional pooling; this keeps every cell an exact average and makes the
  comparison differ only in allocation, not in pooling arithmetic.

## D-011 — A shared full-fidelity local window of two chunks

- **Status:** accepted by the owner (2026-09-23), option 1: the window is added
  on top of the D-005 budgets, which stay unchanged
- **Decision:** The two most recent chunks (8 frames) are kept at full
  fidelity outside every policy's budget, identically for all policies, in A2
  training and at inference (`StreamingCache`). A2 recomputes those two
  chunks with gradients in the same forward pass as the noisy target chunk,
  which realizes D-004's two-chunk gradient window at the cost of one pass per
  step. The first two chunks of each episode are context only.
- **Consequence:** A policy's budget is its cache *beyond* the last 8 frames,
  so every policy effectively sees 8 more recent frames than its budget alone
  (for example, `window` at B-low sees 16 frames). The comparison stays fair
  because the local window is identical for all policies, and training and
  inference read the same cache structure (sanity check 6).
- **Reporting:** every table and figure states budgets as "policy budget +
  shared 8-frame window" and gives both fractions of horizon tokens: B-low
  2,048 + 2,048 (0.4% policy, 0.8% total), B-mid 4,096 + 2,048 (0.8%, 1.2%),
  B-high 12,288 + 2,048 (2.3%, 2.7%). Considered and declined: subtracting the
  window from each budget (would redefine B-low), counting it inside the
  budget (about 3× A2 compute), and a one-chunk window (weaker write signal).
