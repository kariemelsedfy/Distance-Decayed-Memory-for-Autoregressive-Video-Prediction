# Multi-GPU and requeue validation (Phase 0, issue #6)

**Date:** 2026-09-22 · each run's commit is in the table and in its
`git-sha.txt` on scratch
**Scripts:** `scripts/hpc/ddp_smoke.py`, `scripts/hpc/checkpoint_smoke.py`,
`scripts/hpc/launch_distributed.sh`, `scripts/hpc/submit_phase0_smoke.sh`,
`slurm/ddp-smoke.sbatch`, `slurm/requeue-check.sbatch`

## 1. The per-user GPU ceiling (plan-level constraint)

Both partitions that contain the pro6000 nodes enforce a per-user QOS cap.
`scontrol show partition` and `sacctmgr show qos` give:

| Partition | Nodes with pro6000 | QOS | Per-user ceiling |
|---|---|---|---|
| `gpu` | `moose68` (3), `moose69` (4) | `maxgpu` | **2 pro6000**, 3 GPUs total, 4 CPUs, 40G |
| `mixed` | `moose68`, `moose69` | `qosmixed` | **2 pro6000**, 2 GPUs total, 80 CPUs, 500G |

So one user can hold **at most two pro6000 GPUs at a time**, whichever
partition they use, and `gpu` additionally limits them to 4 CPUs and 40G across
all their jobs. A job asking for more does not fail; it pends forever with
`Reason=QOSMaxCpuPerUserLimit` (observed on cancelled job `68180`).

This caps Track A training at 2 pro6000 GPUs unless the owner obtains a raised
limit or a reservation from HPC staff. Use `mixed` for anything needing more
than 4 CPUs or 40G.

## 2. What was validated

| Mode | Job | Layout | Result |
|---|---|---|---|
| `one` | `68179` (`8fe148f`) | 1 rank, `moose68` | DDP delta 0.0; all-reduce exact |
| `pair` | `68222` (`e248bc3`) | 2 ranks, `moose69`, 1 node | delta 0.0; error 0.0; **24.3 GB/s** bus bandwidth |
| `multi` | `68224` (`e248bc3`) | 2 ranks, 1 per node, `moose68`+`moose69` | delta 0.0; error 0.0; **11.0 GB/s** bus bandwidth over `bond0` |
| `requeue` | `68228` (`dcd60a2`) | 1 rank, `moose68`, `scontrol requeue` | checkpointed at step 4, resumed at step 4, finished step 20 |

`max_parameter_delta = 0.0` means every rank held bit-identical parameters
after one optimizer step. `max_absolute_error = 0.0` means the 128 MiB float32
all-reduce summed exactly. Bandwidth is a floor, not a benchmark: 128 MiB, 10
iterations, no NCCL tuning.

Cross-node is roughly half of intra-node here, which is the expected shape; do
not read it as a tuned network number.

## 3. Three failures worth remembering

1. **Literal quotes in `--export`** (`68178`): the submit script wrapped the
   checkout path in escaped quotes, so Slurm exported a path starting with `"`
   and `cd` failed in one second.
2. **`invalid device ordinal` in NCCL shm** (`68208`): with
   `--gpus-per-task` + `--gpu-bind=single:1`, the cgroup hid each rank's peer
   GPU, so NCCL could not open it for the shared-memory transport. Fix:
   allocate GPUs per node with `--gres=gpu:pro6000:N`, keep all node GPUs
   visible to every rank, and select the device by `SLURM_LOCALID`.
3. **Preemption signal during CUDA init** (`68226`): the checkpoint probe
   registered its `SIGUSR1` handler only after importing torch, so the signal
   killed it outright and the restart resumed from step 0. Register the handler
   before any heavy import. This applies to the real trainer too.

## 4. How to reproduce

```
scripts/hpc/checkout.sh --git-ref <ref> --repo <url>
scripts/hpc/submit_phase0_smoke.sh --mode {one|pair|multi|requeue} --workdir <scratch checkout>
scripts/hpc/monitor.sh <job id>
```

Results land in `/mnt/hpc/tmp/$USER/dd-memory/runs/<run id>/results.json`.
