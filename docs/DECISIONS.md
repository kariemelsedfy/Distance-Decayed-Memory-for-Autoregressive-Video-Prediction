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
