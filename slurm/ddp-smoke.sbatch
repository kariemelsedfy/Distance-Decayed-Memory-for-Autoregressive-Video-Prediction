#!/bin/bash
#SBATCH --job-name=ddp-smoke
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=pro6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/mnt/hpc/tmp/%u/dd-memory/logs/%x-%j.out

set -euo pipefail

: "${DD_MEMORY_CHECKOUT:?Set DD_MEMORY_CHECKOUT to the scratch checkout path}"
: "${DD_MEMORY_RUN_ID:?Set DD_MEMORY_RUN_ID to a unique run identifier}"

export TORCH_HOME=/mnt/hpc/tmp/$USER/model_cache/torch
export HF_HOME=/mnt/hpc/tmp/$USER/model_cache/hf
export HUGGINGFACE_HUB_CACHE=/mnt/hpc/tmp/$USER/model_cache/hf
export PIP_CACHE_DIR=/mnt/hpc/tmp/$USER/cache/pip
export XDG_CACHE_HOME=/mnt/hpc/tmp/$USER/cache
export TRITON_CACHE_DIR=/mnt/hpc/tmp/$USER/cache/triton
export TMPDIR=/mnt/hpc/tmp/$USER/tmp
export PYTHONNOUSERSITE=1
export DD_MEMORY_RUN_DIR=/mnt/hpc/tmp/$USER/dd-memory/runs/$DD_MEMORY_RUN_ID

mkdir -p "$DD_MEMORY_RUN_DIR" "$TRITON_CACHE_DIR" "$TMPDIR"
cd "$DD_MEMORY_CHECKOUT"

echo "GPU inventory:"
nvidia-smi --query-gpu=name,pci.bus_id,driver_version,compute_mode --format=csv
echo "Git SHA: $(git rev-parse HEAD)"
echo "Slurm allocation:"
env | grep -E '^SLURM_(HET_SIZE|JOB_NODELIST|NTASKS)' | sort
scontrol show job "$SLURM_JOB_ID"
git rev-parse HEAD > "$DD_MEMORY_RUN_DIR/git-sha.txt"

launch=(
  --kill-on-bad-exit=1
  --gpus-per-task=pro6000:1
  --gpu-bind=single:1
)
if [[ "${SLURM_HET_SIZE:-1}" -gt 1 ]]; then
  launch+=(--het-group=0,1)
else
  launch+=(--ntasks="$SLURM_NTASKS")
fi

srun "${launch[@]}" \
    "$DD_MEMORY_CHECKOUT/scripts/hpc/launch_distributed.sh" \
    "/mnt/hpc/tmp/$USER/envs/dd-memory/bin/python" \
    "$DD_MEMORY_CHECKOUT/scripts/hpc/ddp_smoke.py" \
    --output "$DD_MEMORY_RUN_DIR/results.json"
