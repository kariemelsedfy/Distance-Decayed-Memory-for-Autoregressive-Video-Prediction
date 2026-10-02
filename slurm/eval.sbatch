#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --partition=mixed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:pro6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=/mnt/hpc/tmp/%u/dd-memory/logs/%x-%j.out

# One evaluation run (scripts/eval/evaluate.py, or another scripts/eval/ script
# named by DD_MEMORY_EVAL_SCRIPT). Submitted by scripts/hpc/submit_eval.sh,
# which writes the arguments to $run_dir/eval-args.json (JSON policy configs
# cannot pass through --export).

source /etc/profile
set -euo pipefail
: "${DD_MEMORY_CHECKOUT:?}" "${DD_MEMORY_RUN_ID:?}"

env_path="/mnt/hpc/tmp/$USER/envs/${DD_MEMORY_ENV:-dd-memory-gpu-20260928}"
run_dir="/mnt/hpc/tmp/$USER/dd-memory/runs/$DD_MEMORY_RUN_ID"
export XDG_CACHE_HOME="/mnt/hpc/tmp/$USER/cache"
export TORCH_HOME="/mnt/hpc/tmp/$USER/model_cache/torch"
export TORCHINDUCTOR_CACHE_DIR="/mnt/hpc/tmp/$USER/cache/inductor"
export TRITON_CACHE_DIR="/mnt/hpc/tmp/$USER/cache/triton"
export TMPDIR="/mnt/hpc/tmp/$USER/tmp"
export PYTHONNOUSERSITE=1
# Large, growing caches fragment the default allocator (job 69454 failed with
# 34 GB reserved but unused); expandable segments only change the layout.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$DD_MEMORY_CHECKOUT/src"
export DD_MEMORY_GIT_SHA
DD_MEMORY_GIT_SHA=$(git -C "$DD_MEMORY_CHECKOUT" rev-parse HEAD)
mkdir -p "$TORCH_HOME" "$TMPDIR"
cd "$DD_MEMORY_CHECKOUT"
echo "git $DD_MEMORY_GIT_SHA; node $(hostname); run $DD_MEMORY_RUN_ID"
"$env_path/bin/python" "${DD_MEMORY_EVAL_SCRIPT:-scripts/eval/evaluate.py}" --args-file "$run_dir/eval-args.json"
