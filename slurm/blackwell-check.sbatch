#!/bin/bash
#SBATCH --job-name=blackwell-check
#SBATCH --partition=gpu
#SBATCH --gres=gpu:pro6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/mnt/hpc/tmp/%u/dd-memory/logs/%x-%j.out
#SBATCH --requeue
#SBATCH --signal=B:USR1@300

set -euo pipefail

: "${DD_MEMORY_CHECKOUT:?Set DD_MEMORY_CHECKOUT to the scratch checkout path}"

export TORCH_HOME=/mnt/hpc/tmp/$USER/model_cache/torch
export HF_HOME=/mnt/hpc/tmp/$USER/model_cache/hf
export HUGGINGFACE_HUB_CACHE=/mnt/hpc/tmp/$USER/model_cache/hf
export PIP_CACHE_DIR=/mnt/hpc/tmp/$USER/cache/pip
export XDG_CACHE_HOME=/mnt/hpc/tmp/$USER/cache
export TRITON_CACHE_DIR=/mnt/hpc/tmp/$USER/cache/triton
export TMPDIR=/mnt/hpc/tmp/$USER/tmp
export PYTHONNOUSERSITE=1

run_dir="/mnt/hpc/tmp/$USER/dd-memory/runs/blackwell-check-$SLURM_JOB_ID"
mkdir -p "$run_dir" "$TRITON_CACHE_DIR" "$TMPDIR"
cd "$DD_MEMORY_CHECKOUT"

echo "GPU inventory:"
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_mode --format=csv
echo "Git SHA: $(git rev-parse HEAD)"
echo "Resolved probe config: dtype=bf16,batch=1,heads=4,sequence=128,head_dim=64"

trap 'echo "USR1 received; diagnostic stopped before wall time"; exit 99' USR1

/mnt/hpc/tmp/$USER/envs/dd-memory/bin/python \
  scripts/hpc/check_blackwell.py \
  --output "$run_dir/results.json"
