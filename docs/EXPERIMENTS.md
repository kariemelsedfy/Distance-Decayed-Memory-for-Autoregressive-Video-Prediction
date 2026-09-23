# Experiment registry

No research experiment has been launched. The Phase 0 infrastructure smoke test
is recorded below so its GPU use and result remain auditable.

Register every run before submission when its requested wall time exceeds 30
minutes. Do not record credentials or private tokens.

| Run ID | Date | Git SHA | Config | Slurm job | Resources | Output | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| `blackwell-kernels-20260922` | 2026-09-22 | `e97171c` | bf16; B=1, H=4, T=128, D=64 | `68171` | 1× pro6000, 4 CPU, 16 GB, 15 min | `runs/blackwell-check-68171/results.json` on scratch | completed | All forced SDPA backends, compiled FlexAttention, and bf16 matmul passed with finite outputs. Environment refresh job `68170` also completed. |
| `ddp-one-20260922` | 2026-09-22 | `8fe148f` | 1 rank, 128 MiB all-reduce | `68179` | 1× pro6000, 4 CPU, 16 GB, 10 min | `runs/phase0-one-20260922T170125Z/results.json` | completed | DDP parameter delta 0.0; all-reduce exact. |
| `ddp-pair-20260922` | 2026-09-22 | `e248bc3` | 2 ranks, one node (moose69) | `68222` | 2× pro6000, 8 CPU, 16 GB, 10 min | `runs/phase0-pair-20260922T195844Z/results.json` | completed | Delta 0.0; 24.3 GB/s bus bandwidth. Partition `mixed`. |
| `ddp-multi-20260922` | 2026-09-22 | `e248bc3` | 2 ranks, one per node (moose68+69) | `68224` | 2× pro6000, 8 CPU, 32 GB, 10 min | `runs/phase0-multi-20260922T195934Z/results.json` | completed | Delta 0.0; 11.0 GB/s over `bond0`. Partition `mixed`. |
| `requeue-check-20260922` | 2026-09-22 | `dcd60a2` | 20 steps, SIGUSR1 at 25 s | `68228` | 1× pro6000, 4 CPU, 16 GB, 15 min | `runs/phase0-requeue-20260922T200600Z/results.json` | completed | Checkpointed step 4, resumed step 4, finished step 20. |
| `memory-maze-probe-20260923T084614Z` | 2026-09-23 | `6794684` | 9×9, 64×64, software EGL, 200 warmup + 3×2,000 timed steps | `68321` | `main`, 1 CPU, 8 GB, 30 min cap | `runs/memory-maze-probe-20260923T084614Z/results.json` | completed | Mesa llvmpipe confirmed; median 23.85 frames/s; required observations and six actions passed. Failed setup attempts `68314` and `68315` are explained in `docs/hpc/memory-maze.md`. |

Jobs `68178`, `68180`, `68208`, and `68226` were failed or cancelled attempts at
the same four probes; their causes are in
[docs/hpc/ddp-and-requeue.md](hpc/ddp-and-requeue.md).
