#!/bin/bash
#SBATCH --job-name=a2-train
#SBATCH --partition=mixed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:pro6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/mnt/hpc/tmp/%u/dd-memory/logs/%x-%j.out
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --signal=B:USR1@300

# A2 memory fine-tune (TRACK_A_PLAN.md §5), one GPU. Submitted like A1 with
# DD_A2_CONFIG and DD_BASE_CHECKPOINT; --partition may be gpu or mixed.
# On the pre-timeout USR1 the trainer checkpoints and the job requeues itself;
# the trainer resumes from checkpoints/latest.pt.

source /etc/profile
set -euo pipefail

: "${DD_MEMORY_CHECKOUT:?}" "${DD_MEMORY_RUN_ID:?}" "${DD_A2_CONFIG:?}" "${DD_BASE_CHECKPOINT:?}"
: "${DD_TRAIN_SPLIT:?}"

env_path="/mnt/hpc/tmp/$USER/envs/${DD_MEMORY_ENV:-dd-memory-gpu}"
run_dir="/mnt/hpc/tmp/$USER/dd-memory/runs/$DD_MEMORY_RUN_ID"
export PIP_CACHE_DIR="/mnt/hpc/tmp/$USER/cache/pip"
export XDG_CACHE_HOME="/mnt/hpc/tmp/$USER/cache"
export TORCHINDUCTOR_CACHE_DIR="/mnt/hpc/tmp/$USER/cache/inductor"
export TRITON_CACHE_DIR="/mnt/hpc/tmp/$USER/cache/triton"
export TMPDIR="/mnt/hpc/tmp/$USER/tmp"
export PYTHONNOUSERSITE=1
# The environment holds dependencies only; the code comes from this checkout
# (checked below).
export PYTHONPATH="$DD_MEMORY_CHECKOUT/src"
export DD_MEMORY_GIT_SHA
DD_MEMORY_GIT_SHA=$(git -C "$DD_MEMORY_CHECKOUT" rev-parse HEAD)
export OMP_NUM_THREADS=4
mkdir -p "$run_dir" "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
cd "$DD_MEMORY_CHECKOUT"
python="$env_path/bin/python"

echo "Attempt ${SLURM_RESTART_COUNT:-0}; git $DD_MEMORY_GIT_SHA; nodes $SLURM_JOB_NODELIST"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$python" -c 'import PIL' 2>/dev/null || "$python" -m pip install "pillow>=10.0"
"$python" -c 'import distance_decayed_memory as d, sys; p=d.__path__[0]; print("package:", p); sys.exit(0 if p.startswith(sys.argv[1]) else 1)' "$DD_MEMORY_CHECKOUT"
echo "$DD_MEMORY_GIT_SHA" > "$run_dir/git-sha.txt"

stage_args=()
[[ -n "${DD_STAGE_ROOT:-}" ]] && stage_args=(--set "stage_root=$DD_STAGE_ROOT")

child_pid=""
requeue=0
forward_usr1() {
  echo "USR1 received; asking the trainer to checkpoint."
  requeue=1
  [[ -n "$child_pid" ]] && kill -USR1 "$child_pid" 2>/dev/null || true
}
trap forward_usr1 USR1

set +e
srun --kill-on-bad-exit=1 \
  "$DD_MEMORY_CHECKOUT/scripts/hpc/launch_distributed.sh" \
  "$python" scripts/train/train_a2.py \
    --config "$DD_A2_CONFIG" \
    --set "run_dir=$run_dir" \
    --set "train_split=$DD_TRAIN_SPLIT" \
    --set "base_checkpoint=$DD_BASE_CHECKPOINT" \
    "${stage_args[@]}" &
child_pid=$!
wait "$child_pid"
status=$?
while kill -0 "$child_pid" 2>/dev/null; do
  wait "$child_pid"
  status=$?
done
set -e

echo "Trainer exited with status $status"
if [[ $status -eq 99 || $requeue -eq 1 ]]; then
  echo "Requeueing Slurm job $SLURM_JOB_ID"
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
fi
exit "$status"
