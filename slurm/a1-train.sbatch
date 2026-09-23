#!/bin/bash
#SBATCH --job-name=a1-train
#SBATCH --partition=mixed
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:pro6000:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=/mnt/hpc/tmp/%u/dd-memory/logs/%x-%j.out
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --signal=B:USR1@300

# A1 training (TRACK_A_PLAN.md §5). Submitted by scripts/hpc/submit_a1.sh,
# which sets --ntasks/--gres together (1 or 2 GPUs) and the wall time.
# On the pre-timeout USR1 the trainer checkpoints and the job requeues itself;
# the trainer resumes from checkpoints/latest.pt.

source /etc/profile
set -euo pipefail

: "${DD_MEMORY_CHECKOUT:?}" "${DD_MEMORY_RUN_ID:?}" "${DD_A1_CONFIG:?}"
: "${DD_TRAIN_SPLIT:?}" "${DD_VAL_SPLIT:?}"

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

# Splits are read from scratch by default (A1 needs only ~6 MB/s); set
# DD_STAGE_ROOT to copy them to node-local storage first.
stage_args=()
[[ -n "${DD_STAGE_ROOT:-}" ]] && stage_args=(--set "stage_root=$DD_STAGE_ROOT")
workers=$(( ${SLURM_CPUS_PER_TASK:-4} > 3 ? SLURM_CPUS_PER_TASK - 2 : 1 ))

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
  "$python" scripts/train/train_a1.py \
    --config "$DD_A1_CONFIG" \
    --set "run_dir=$run_dir" \
    --set "train_split=$DD_TRAIN_SPLIT" \
    --set "val_split=$DD_VAL_SPLIT" \
    --set "num_workers=$workers" \
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
