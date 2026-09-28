# 2026-09-28 — Claude — Evaluation harness and sanity checks (issues #16, #17)

## What

- Merged all six open PRs (#22–#27) into `main`, resolving decision-log
  conflicts; closed issues #9–#15 and #18.
- `src/distance_decayed_memory/eval/`: protocols P1 and P2, PSNR/SSIM/LPIPS,
  aggregation with bootstrap intervals, and the six sanity checks.
- Scripts to evaluate a checkpoint, summarize runs (with checks 1–5), check
  streaming against inference (check 6), draw the headline figure, and submit
  evaluation jobs.

## Why

The A2 pilot cannot be judged without them, and the A3 sweep's results are
only trusted once all six sanity checks pass.

## What I learned

- **The perfect-memory oracle does not fit with the L model at full length**
  (about 100 GB of keys and values for 4,096 frames). It is now capped at the
  2,048-frame horizon like every other policy (D-013).
- **Sanity check 6 exposes a small structural gap in D-011.** With perfect
  memory, the training and inference caches match exactly. With compressing
  policies they differ a little, because at inference a chunk is encoded while
  its predecessors are still at full detail. It will be measured on the
  trained model at the A2 pilot.
- LPIPS needs torchvision; the environment build now pins PyTorch so adding
  torchvision cannot upgrade it. The cluster environment will get the LPIPS
  packages after A1 finishes, so the running job is not disturbed.

## How to verify

`pytest` (171 tests; the new end-to-end test trains a tiny A1 model and runs
P1, P2, an ablation, the summary with its checks, and the headline figure).

## Plain-language explanation

This is the scoring machinery. It replays the maze episodes, lets each memory
policy's model predict what it sees when it comes back to a place, compares
the prediction with the real frame, and draws the final comparison plot. It
also runs six automatic sanity checks that must pass before any result is
believed — for example, a policy with a 16-frame memory must fail right at 16
frames.

## Next

A1 finishes around 2026-10-01. Then: add the LPIPS packages to the cluster
environment, run the A2 pilot and the sanity checks on it, and bring the
results to the owner for the sweep gate.
