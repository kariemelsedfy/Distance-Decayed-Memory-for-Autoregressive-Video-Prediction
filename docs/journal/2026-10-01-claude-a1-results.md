# 2026-10-01 — Claude — A1 base model results and LPIPS environment

## What

- A1 job `69049` (run `a1-L-20260928T084605Z`) completed 100k steps at
  09:26 EDT after 3 d 4 h 40 min: one attempt, no requeue, no non-finite loss,
  189 frames/s median, peak 23.4 GB per GPU.
- Fetched its metrics, validation log, config, and the ten sample GIFs to
  `outputs/a1-L-20260928T084605Z/`, and added
  `scripts/figures/training_curves.py` to plot them.
- Added torchvision 0.26.0 and LPIPS 0.1.4 to `dd-memory-gpu-20260928` by
  re-running `scripts/hpc/build_env.sh` from `main` at `26cca17`; torch stayed
  at 2.11.0+cu128 and the import check passed. Committed the lock file.
- Updated `EXPERIMENTS.md`, `STATUS.md`, and `docs/hpc/environments.md`.

## Why

Every A2 run starts from A1's EMA weights, and the evaluation harness needs
LPIPS on the cluster. Both were the first step before the A2 pilot.

## Results

| Step | Val loss (raw) | Val loss (EMA) | Sample PSNR (dB) |
|---|---|---|---|
| 10k | 0.00501 | 0.00414 | 17.0 |
| 50k | 0.00203 | 0.00164 | 19.3 |
| 90k | 0.00137 | 0.00127 | 23.2 |
| 100k | 0.00133 | 0.00124 | 20.9 |

- Validation loss is still falling at 100k, slowly (about 2% over the last
  10k steps). Train and validation losses track each other: no sign of
  overfitting.
- Sample PSNR is measured on a single validation clip, so it is noisy (the
  90k → 100k drop is within that noise). The P1 evaluation in the A2 pilot is
  the real measurement.
- The size check's L model reached val 0.0102 at 4k steps on one GPU; the
  full run is about 8× lower at 100k.

## How to verify

`python scripts/figures/training_curves.py outputs/a1-L-20260928T084605Z`
after `scripts/hpc/fetch.sh a1-L-20260928T084605Z outputs/a1-L-20260928T084605Z`.

## Plain-language explanation

The base video model has finished training. It learned to predict the next
frames of maze videos steadily, and its error is still creeping down, but
slowly enough that more training would not change the picture much. It has
not yet been tested on remembering places it saw long ago: that is what the
next step (the A2 pilot, with each memory policy) measures. The cluster
environment can now also compute LPIPS, the perceptual image-quality score
used in evaluation.

## Next

The A2 pilot (STATUS.md, Next 1). It needs owner approval before submission.
The A1 checkpoints sit on scratch, which is purged by file age; ask HPC staff
for a persistent location before early November.
