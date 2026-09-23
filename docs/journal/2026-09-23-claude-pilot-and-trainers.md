# 2026-09-23 — Claude — Pilot split, trainers, and GPU smoke runs

## What

- **Pilot split on the cluster** (issue #11): 200 episodes, 409,600 frames,
  8 shards, verified manifest. Generation runs at 25.2 frames/s per core.
- **A1 trainer** (issue #14) and **A2 streaming trainer** (issue #15), both
  auto-resuming and requeue-safe, with CPU tests.
- **GPU smoke runs:** A1 on 2 GPUs (511 frames/s) and A2 on 1 GPU (1.8
  steps/s with 8 parallel episodes), both completed end to end.

## Why

The A0 check-in needs the pilot. Knowing the real cost of A1 and A2 before
asking to launch them turns the plan's guesses into measurements.

## What I learned

- **The data looks right at scale.** 47% of frames are revisits, every gap
  bucket from 16 to 2,047 frames has thousands of frames, and matched views
  differ by a median 8–9 (0–255) against 26 for random pairs.
- **Compute is cheaper than planned.** A1 at 100k steps is about 28 hours on
  2 GPUs (plan: 3–5 days). One A2 run at 20k steps is about 3 GPU-hours
  (plan: 8–16), so the full sweep fits in about 3 days on 4 GPUs.
- **Both conda environments had lost library files**, which broke pip and
  then `torch.compile`. The GPU environment was rebuilt as `dd-memory-gpu`
  with copied files and a link check (`docs/hpc/environments.md`).
- **A2 needs one new decision (D-011):** the last two chunks are kept at full
  detail for every policy, which makes the two-chunk gradient window cost a
  single forward pass.
- **To watch:** re-encoding a 3-frame-old cached chunk changes its keys by
  30–50% on the barely trained smoke model. This must be re-measured with a
  trained A1 model before the sweep.
- Tooling fixes: carriage returns from SSH broke `fetch.sh` and a watcher; a
  submission that reported failure had actually succeeded, so the data submit
  script now refuses duplicates and shards take a lock.

## How to verify

- `pytest` (155 tests), `ruff check`, `black --check`.
- `docs/EXPERIMENTS.md` rows `memmaze9-pilot-…`, `a1-smoke`, `a2-smoke`.
- `scripts/hpc/fetch.sh memmaze9-pilot-20260923T125213Z outputs/pilot`, then
  open `outputs/pilot/spot-check/spot_check.png` and the preview GIFs.

## Plain-language explanation

The cluster made a first batch of 200 practice maze videos, and the checks say
the revisits in them are real and cover every delay we need. The two training
programs — one teaching the model what the maze looks like, one teaching it to
use a limited memory — both run on the cluster's graphics cards, and both are
faster than the plan assumed.

## Next

Owner approvals: the full dataset, then the A1 size check and A1 run.
Meanwhile the evaluation harness (#16) and sanity checks (#17) can be built
without GPUs.
