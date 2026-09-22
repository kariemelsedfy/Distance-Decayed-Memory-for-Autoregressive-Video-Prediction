#!/bin/bash
#SBATCH --job-name=requeue-check
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=pro6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/mnt/hpc/tmp/%u/dd-memory/logs/%x-%j.out
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --signal=B:USR1@300

set -euo pipefail

: "${DD_MEMORY_CHECKOUT:?Set DD_MEMORY_CHECKOUT to the scratch checkout path}"
: "${DD_MEMORY_RUN_ID:?Set DD_MEMORY_RUN_ID to a unique run identifier}"

export TORCH_HOME=/mnt/hpc/tmp/$USER/model_cache/torch
export PIP_CACHE_DIR=/mnt/hpc/tmp/$USER/cache/pip
export XDG_CACHE_HOME=/mnt/hpc/tmp/$USER/cache
export TMPDIR=/mnt/hpc/tmp/$USER/tmp
export PYTHONNOUSERSITE=1
export DD_MEMORY_RUN_DIR=/mnt/hpc/tmp/$USER/dd-memory/runs/$DD_MEMORY_RUN_ID

mkdir -p "$DD_MEMORY_RUN_DIR" "$TMPDIR"
cd "$DD_MEMORY_CHECKOUT"

echo "Attempt ${SLURM_RESTART_COUNT:-0}"
echo "GPU inventory:"
nvidia-smi --query-gpu=name,pci.bus_id,driver_version,compute_mode --format=csv
echo "Git SHA: $(git rev-parse HEAD)"
git rev-parse HEAD > "$DD_MEMORY_RUN_DIR/git-sha.txt"

child_pid=""
timer_pid=""
handle_preemption() {
  trap - USR1
  echo "USR1 received; asking the child to checkpoint."
  if [[ -n "$child_pid" ]] && kill -0 "$child_pid" 2>/dev/null; then
    kill -USR1 "$child_pid"
    wait "$child_pid" || true
  fi
  echo "Checkpoint complete; requeueing Slurm job $SLURM_JOB_ID."
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
}
cleanup() {
  if [[ -n "$timer_pid" ]] && kill -0 "$timer_pid" 2>/dev/null; then
    kill "$timer_pid" 2>/dev/null || true
  fi
}
trap handle_preemption USR1
trap cleanup EXIT

if [[ "${SLURM_RESTART_COUNT:-0}" -eq 0 ]]; then
  (
    sleep "${DD_MEMORY_REQUEUE_AFTER_SECONDS:-8}"
    kill -USR1 "$$"
  ) &
  timer_pid=$!
fi

"/mnt/hpc/tmp/$USER/envs/dd-memory/bin/python" \
  "$DD_MEMORY_CHECKOUT/scripts/hpc/checkpoint_smoke.py" \
  --run-dir "$DD_MEMORY_RUN_DIR" &
child_pid=$!
wait "$child_pid"
