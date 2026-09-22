# 2026-09-22 — Claude — DDP and requeue smoke tests (issue #6)

## What
- Found the first one-GPU job (`68178`) had failed in one second: the submit
  script wrapped the checkout path in escaped quotes, so Slurm received a path
  that literally began with `"` and `cd` failed. Removed the quotes (`8fe148f`).
- Re-ran the one-GPU smoke test as job `68179` on moose68. It passed: the DDP
  step left parameters identical across ranks and the all-reduce was exact.
- Submitted the four-GPU single-node test as job `68180` on moose69. The VPN
  dropped before the result could be read.

## Why
Track A training needs working multi-GPU gradient sync and safe restarts before
any trainer is built.

## How to verify
`scripts/hpc/monitor.sh 68179` and the `results.json` under
`runs/phase0-one-20260922T170125Z/` on scratch.

## Plain-language explanation
The single-GPU plumbing works. The first attempt failed only because of a typo
in how the job was told where the code lives. The four-GPU test was sent but
its answer is still waiting on the VPN; the cross-node and restart tests are
next.
