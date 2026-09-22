# Scratch environment and Blackwell kernels — 2026-09-22

## What changed

I created `scripts/hpc/build_env.sh` so environment creation runs as a CPU
Slurm job rather than doing package work on the login node. The environment,
package caches, build temporaries, and compiler caches all live under
`/mnt/hpc/tmp/$USER`. I also restricted setuptools discovery to `src/`, added
NumPy as a runtime dependency, and committed the final package freeze in
`environment/hpc-dd-memory.txt`.

I added `scripts/hpc/check_blackwell.py` and
`slurm/blackwell-check.sbatch`. The probe forces each PyTorch SDPA backend,
compiles FlexAttention, exercises bf16 matrix multiplication, checks every
output for finite values, and fails the job unless every named check passes.
The Slurm wrapper requests one pro6000 for at most 15 minutes and places its log,
result, and runtime caches on scratch.

This work is on `phase0/blackwell-env` in draft PR #19 and closes issue #5.
Track B remains deferred.

## What the cluster proved

Final job `68171` completed on `moose68` in 45 seconds from commit `e97171c`.
The RTX PRO 6000 Blackwell Server Edition reported 97,887 MiB, driver 615.71.09,
Default compute mode, and compute capability 12.0. Python 3.11.16 and PyTorch
2.11.0+cu128 recognized CUDA 12.8 and native bf16 support.

All six checks returned finite bf16 tensors:

- matrix multiplication;
- forced math SDPA;
- forced Flash Attention SDPA;
- forced efficient-attention SDPA;
- forced cuDNN-attention SDPA; and
- FlexAttention through `torch.compile`.

This establishes compatibility for the planned Track A attention work. It does
not establish which kernel is fastest: first-use initialization and compilation
are included in the recorded call times, so those numbers are not benchmarks.

## Failures and lessons

The first environment attempt exposed three useful bootstrap problems:

- Sourcing `/etc/profile` after `set -u` failed because the cluster profile
  reads unset interactive-shell variables. The generated job now sources the
  profile before enabling strict mode.
- Editable installation initially treated the scratch-linked `data/`,
  `weights/`, and `slurm/` directories as top-level packages. Explicit `src/`
  package discovery fixed this and will remain important as the repository
  grows.
- The first successful PyTorch install warned that NumPy was absent. NumPy is
  now an explicit project dependency, and the final environment has 2.4.6.

The initial GPU probe `68169` passed, but its pass/fail rule did not require
finite values or every SDPA backend. I tightened the rule, committed it, created
a new exact checkout, refreshed the editable install as CPU job `68170`, and
reran the final evidence as job `68171`. Both final jobs completed. No GPU job
requested more than 15 minutes, and no jobs remain running.

## How to verify

- Read `docs/hpc/blackwell-kernels.md` for the result, interpretation, and
  remote evidence paths.
- Compare the cluster package set with `environment/hpc-dd-memory.txt`.
- Run `scripts/hpc/build_env.sh --workdir <remote-checkout>` to refresh the
  environment through Slurm.
- Submit `slurm/blackwell-check.sbatch` from an exact scratch checkout and
  inspect it with `scripts/hpc/monitor.sh <job-id>`.
- Run the repository pre-commit suite and pytest locally.

## Handoff

After PR #19 is reviewed and merged, the next infrastructure task is issue #6:
DDP smoke tests at 1 GPU, the maximum GPUs on each node, and across both nodes;
all-reduce measurement; then checkpoint, resume, and requeue validation. That
work should remain a separate branch and PR. The home directory is at its hard
quota, so issue #6 must continue the scratch-only policy used here.
