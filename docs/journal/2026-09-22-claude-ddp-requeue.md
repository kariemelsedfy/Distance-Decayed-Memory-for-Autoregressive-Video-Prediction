# 2026-09-22 — Claude — DDP and requeue smoke tests (issue #6)

## What
All four Phase 0 infrastructure probes now pass on the Blackwell nodes:

- one GPU (job `68179`),
- two GPUs on one node (`68222`, 24.3 GB/s all-reduce),
- one GPU on each of the two nodes (`68224`, 11.0 GB/s over the bonded link),
- checkpoint, forced requeue, and resume (`68228`).

Getting there meant finding one hard constraint and fixing three defects.

**The constraint.** Both Slurm partitions that hold the pro6000 nodes cap a
single user at **two pro6000 GPUs**. The `gpu` partition also caps a user at 4
CPUs and 40 GB of memory across all their jobs. A job that asks for more never
starts; it simply waits forever. The original plan for this issue — a 4-GPU
test and a 7-GPU test — could not have run. I resized the tests to 2 GPUs, and
moved the multi-GPU ones to the `mixed` partition, which has the same GPU cap
but far more CPU and memory headroom.

**The defects.**
1. The submit script passed the code's location to Slurm wrapped in quote
   characters, so the job looked for a directory whose name began with `"`.
2. Slurm's per-task GPU binding hid each process's partner GPU, so the
   communication library could not open it and failed with `invalid device
   ordinal`. Fixed by giving every process on a node the whole node's GPUs and
   letting each pick the one matching its local index.
3. The checkpoint probe set up its "please save and stop" signal handler only
   after loading PyTorch. The stop signal arrived during CUDA startup, killed
   the process outright, and the restart began from scratch. The handler is now
   installed before any heavy import — this matters for the real trainer too.

## Why
Track A training needs three guarantees before any trainer is written: GPUs
must agree on model updates, the two nodes must be able to talk, and an
interrupted job must resume where it stopped rather than restart. All three now
hold, within a GPU budget that is smaller than the plan assumed.

## How to verify
`docs/hpc/ddp-and-requeue.md` lists each job, its commit, and its result file.
Re-run any probe with
`scripts/hpc/submit_phase0_smoke.sh --mode {one|pair|multi|requeue} --workdir <scratch checkout>`.
Results are in `/mnt/hpc/tmp/$USER/dd-memory/runs/<run id>/results.json`.

## Plain-language explanation
The plumbing works: two graphics cards stay perfectly in step with each other,
cards in different machines can talk across the network, and a job that gets
interrupted picks up exactly where it left off instead of starting over. The
surprise is a house rule on the cluster: one person may use only two of these
cards at a time, not seven. That is a planning question for the owner — either
we design the experiments around two cards, or we ask the cluster staff for
more.

## Owner decision needed
Plan Track A around 2 pro6000 GPUs, or request a raised QOS limit or a
reservation from Bowdoin HPC staff.
