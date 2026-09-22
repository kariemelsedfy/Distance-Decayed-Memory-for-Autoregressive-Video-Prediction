#!/usr/bin/env bash

set -euo pipefail

: "${SLURM_JOB_ID:?This launcher must run inside a Slurm job}"
: "${SLURM_JOB_NODELIST:?SLURM_JOB_NODELIST is required}"
: "${SLURM_NTASKS:?SLURM_NTASKS is required}"
: "${SLURM_PROCID:?SLURM_PROCID is required}"
: "${SLURM_LOCALID:?SLURM_LOCALID is required}"
[[ $# -gt 0 ]] || { echo "Usage: launch_distributed.sh COMMAND [ARGS...]" >&2; exit 2; }

master_addr="${DD_MEMORY_MASTER_ADDR:-}"
if [[ -z "$master_addr" ]]; then
  master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sed -n '1p')
fi
job_number="${SLURM_JOB_ID%%_*}"

export MASTER_ADDR="$master_addr"
export MASTER_PORT="${DD_MEMORY_MASTER_PORT:-$((20000 + job_number % 20000))}"
export WORLD_SIZE="$SLURM_NTASKS"
export RANK="$SLURM_PROCID"
# Slurm exposes one bound GPU per process, so its visible CUDA index is zero.
export LOCAL_RANK=0
export DD_MEMORY_NODE_LOCAL_RANK="$SLURM_LOCALID"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond0}"

exec "$@"
