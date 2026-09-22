#!/usr/bin/env bash

set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mode=""
remote_checkout=""
wall_time="00:10:00"

usage() {
  cat <<'EOF'
Usage: scripts/hpc/submit_phase0_smoke.sh --mode MODE --workdir REMOTE_PATH [options]

Modes:
  one       One rank on one pro6000 on moose68 (partition gpu)
  pair      Two ranks on two pro6000 on moose69 (partition mixed)
  multi     Two ranks, one per node, on moose68 and moose69 (partition mixed)
  requeue   One-GPU checkpoint and automatic Slurm requeue test

The per-user QOS ceiling on both pro6000 partitions is 2 pro6000 GPUs, so no
mode may request more. Partition gpu (QOS maxgpu) additionally caps a user at
4 CPUs and 40G; partition mixed (QOS qosmixed) allows 80 CPUs and 500G.

Options:
  --time HH:MM:SS   Wall time, at most 00:30:00 (default: 00:10:00)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) mode="${2:?missing mode}"; shift 2 ;;
    --workdir) remote_checkout="${2:?missing remote path}"; shift 2 ;;
    --time) wall_time="${2:?missing wall time}"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$mode" ]] || { echo "--mode is required" >&2; exit 2; }
[[ "$remote_checkout" == /mnt/hpc/tmp/* ]] || {
  echo "--workdir must be an absolute scratch path under /mnt/hpc/tmp" >&2
  exit 2
}
if [[ ! "$wall_time" =~ ^([0-9]{2}):([0-5][0-9]):([0-5][0-9])$ ]]; then
  echo "Invalid --time; expected HH:MM:SS" >&2
  exit 2
fi
wall_seconds=$((10#${BASH_REMATCH[1]} * 3600 + 10#${BASH_REMATCH[2]} * 60 + 10#${BASH_REMATCH[3]}))
if ((wall_seconds > 1800)); then
  echo "Phase 0 smoke jobs may not exceed 00:30:00" >&2
  exit 2
fi

timestamp=$(date -u +%Y%m%dT%H%M%SZ)
run_id="phase0-${mode}-${timestamp}"
case "$mode" in
  one)
    job_name="ddp-one"
    partition="gpu"
    batch_script="slurm/ddp-smoke.sbatch"
    allocation="--nodes=1 --ntasks=1 --nodelist=moose68 --gres=gpu:pro6000:1"
    ;;
  pair)
    job_name="ddp-pair"
    partition="mixed"
    batch_script="slurm/ddp-smoke.sbatch"
    allocation="--nodes=1 --ntasks=2 --nodelist=moose69 --gres=gpu:pro6000:2"
    ;;
  multi)
    job_name="ddp-multi"
    partition="mixed"
    batch_script="slurm/ddp-smoke.sbatch"
    allocation="--nodes=2 --ntasks=2 --ntasks-per-node=1 --nodelist=moose68,moose69 --gres=gpu:pro6000:1"
    ;;
  requeue)
    job_name="requeue-check"
    partition="gpu"
    batch_script="slurm/requeue-check.sbatch"
    allocation="--nodes=1 --ntasks=1 --nodelist=moose68 --gres=gpu:pro6000:1"
    ;;
  *) echo "Unknown mode: $mode" >&2; usage >&2; exit 2 ;;
esac

printf -v quoted_partition '%q' "$partition"
printf -v quoted_checkout '%q' "$remote_checkout"
printf -v quoted_run_id '%q' "$run_id"
printf -v quoted_time '%q' "$wall_time"
printf -v quoted_job_name '%q' "$job_name"
printf -v quoted_batch '%q' "$remote_checkout/$batch_script"
read -r -d '' remote_command <<EOF || true
set -euo pipefail
checkout=$quoted_checkout
test -f $quoted_batch
mkdir -p "/mnt/hpc/tmp/\$USER/dd-memory/logs"
sbatch --parsable \
  --partition=$quoted_partition \
  --time=$quoted_time \
  --job-name=$quoted_job_name \
  --cpus-per-task=4 \
  --mem=16G \
  --gpus-per-task=pro6000:1 \
  --export=ALL,DD_MEMORY_CHECKOUT=\$checkout,DD_MEMORY_RUN_ID=$quoted_run_id \
  $allocation \
  $quoted_batch
EOF

set +e
response=$("$script_dir/remote.sh" --timeout 60 "$remote_command")
remote_status=$?
set -e
if [[ $remote_status -ne 0 ]]; then
  printf '%s\n' "$response" >&2
  exit "$remote_status"
fi
job_id=$(printf '%s\n' "$response" | tr -d '\r' | tail -n 1)
[[ "$job_id" =~ ^[0-9]+$ ]] || { echo "Unexpected sbatch response: $job_id" >&2; exit 1; }
printf '%s\n' "$job_id"
printf 'run-id: %s\n' "$run_id" >&2
