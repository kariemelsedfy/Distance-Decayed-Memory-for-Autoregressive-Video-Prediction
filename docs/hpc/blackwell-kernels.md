# Blackwell kernel compatibility

- **Verified:** 2026-09-22
- **Git commit:** `e97171c06eded2509af65ca090a06c275cb41d06`
- **Slurm job:** `68171` on `moose68`
- **Result:** all required checks passed with finite outputs

## Environment

The reusable conda environment is at
`/mnt/hpc/tmp/$USER/envs/dd-memory`. It was created and updated on the CPU-only
`main` partition by `scripts/hpc/build_env.sh`; package downloads, conda files,
pip caches, build temporaries, and the environment itself stay in scratch.

The exact final package freeze is committed as
`environment/hpc-dd-memory.txt`. Key versions are:

- Python 3.11.16
- PyTorch 2.11.0+cu128
- PyTorch CUDA 12.8 and `cuda-toolkit` 12.8.1
- Triton 3.6.0
- cuDNN 9.19.0.56
- NCCL 2.28.9
- NumPy 2.4.6

## Hardware and result

The job requested one `gpu:pro6000` for 15 minutes and completed in 45 seconds.
The allocated card was an NVIDIA RTX PRO 6000 Blackwell Server Edition with
97,887 MiB reported VRAM, compute capability 12.0 (`sm_120`), driver 615.71.09,
and Default compute mode.

All tensors used bf16 with batch 1, 4 heads, sequence length 128, and head
dimension 64.

| Check | Output shape | Finite | Result |
|---|---:|---:|---:|
| bf16 matrix multiplication | 1×4×128×128 | yes | pass |
| SDPA math backend | 1×4×128×64 | yes | pass |
| SDPA Flash Attention backend | 1×4×128×64 | yes | pass |
| SDPA efficient-attention backend | 1×4×128×64 | yes | pass |
| SDPA cuDNN-attention backend | 1×4×128×64 | yes | pass |
| compiled FlexAttention | 1×4×128×64 | yes | pass |

Each SDPA implementation was explicitly selected through PyTorch's
`sdpa_kernel` context, so an unavailable forced backend would have failed the
check instead of silently choosing the ordinary math path. FlexAttention was
compiled with `torch.compile` before execution.

This is a compatibility smoke test, not a speed comparison. The per-call
timings include first-use initialization or compilation and must not be read as
kernel benchmarks.

## Reproduce

1. Create a fresh scratch checkout with `scripts/hpc/checkout.sh`.
2. Run `scripts/hpc/build_env.sh --workdir <remote-checkout>`.
3. Submit `slurm/blackwell-check.sbatch` with `DD_MEMORY_CHECKOUT` set to that
   checkout. The job is capped at 15 minutes and writes its log and JSON result
   below `/mnt/hpc/tmp/$USER/dd-memory/`.
4. Inspect it with `scripts/hpc/monitor.sh <job-id>`.

The final evidence is stored remotely at:

- log: `/mnt/hpc/tmp/$USER/dd-memory/logs/blackwell-check-68171.out`
- JSON: `/mnt/hpc/tmp/$USER/dd-memory/runs/blackwell-check-68171/results.json`

No credential is needed inside the job, and no credential is present in either
artifact.
