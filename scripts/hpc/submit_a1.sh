#!/usr/bin/env bash

set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
remote_checkout=""
config=""
train_split=""
val_split=""
gpus=2
wall_time="12:00:00"
name="a1"

usage() {
  cat <<'EOF_USAGE'
Usage: scripts/hpc/submit_a1.sh --workdir REMOTE_PATH --config CONFIG_JSON \
  --train-split SPLIT_DIR --val-split SPLIT_DIR [options]

Submits A1 training on the mixed partition. The job checkpoints on the
pre-timeout signal and requeues itself, so a long run spans several wall
times. CONFIG_JSON and split paths are relative to the checkout or absolute.

Options:
  --gpus 1|2       GPUs on one node (default 2; the per-job QOS ceiling)
  --time HH:MM:SS  wall time per attempt (default 12:00:00)
  --name NAME      run-id prefix (default a1)
EOF_USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workdir) remote_checkout="${2:?}"; shift 2 ;;
    --config) config="${2:?}"; shift 2 ;;
    --train-split) train_split="${2:?}"; shift 2 ;;
    --val-split) val_split="${2:?}"; shift 2 ;;
    --gpus) gpus="${2:?}"; shift 2 ;;
    --time) wall_time="${2:?}"; shift 2 ;;
    --name) name="${2:?}"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ "$remote_checkout" == /mnt/hpc/tmp/* ]] || { echo "--workdir must be under /mnt/hpc/tmp" >&2; exit 2; }
[[ -n "$config" && -n "$train_split" && -n "$val_split" ]] || { usage >&2; exit 2; }
[[ "$gpus" == 1 || "$gpus" == 2 ]] || { echo "--gpus must be 1 or 2" >&2; exit 2; }
[[ "$wall_time" =~ ^[0-9]{1,2}:[0-5][0-9]:[0-5][0-9]$ ]] || { echo "invalid --time" >&2; exit 2; }
[[ "$name" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "invalid --name" >&2; exit 2; }

absolute() {
  [[ "$1" == /* ]] && printf '%s' "$1" || printf '%s/%s' "$remote_checkout" "$1"
}
run_id="$name-$(date -u +%Y%m%dT%H%M%SZ)"
exports="ALL,DD_MEMORY_CHECKOUT=$remote_checkout,DD_MEMORY_RUN_ID=$run_id"
exports+=",DD_A1_CONFIG=$(absolute "$config")"
exports+=",DD_TRAIN_SPLIT=$(absolute "$train_split"),DD_VAL_SPLIT=$(absolute "$val_split")"
printf -v q_exports '%q' "$exports"
printf -v q_batch '%q' "$remote_checkout/slurm/a1-train.sbatch"

read -r -d '' remote_command <<EOF_REMOTE || true
set -euo pipefail
test -f $q_batch
mkdir -p "/mnt/hpc/tmp/\$USER/dd-memory/logs"
sbatch --parsable --time=$wall_time --ntasks=$gpus --gres=gpu:pro6000:$gpus \
  --job-name=$name --export=$q_exports $q_batch
EOF_REMOTE

set +e
response=$("$script_dir/remote.sh" --timeout 60 "$remote_command" 2>&1)
remote_status=$?
set -e
if [[ $remote_status -ne 0 ]]; then
  printf 'Remote submission failed (exit %s):\n%s\n' "$remote_status" "$response" >&2
  echo "Check squeue before retrying: the job may have been submitted." >&2
  exit "$remote_status"
fi
job_id=$(printf '%s\n' "$response" | tr -d '\r' | tail -n 1)
[[ "$job_id" =~ ^[0-9]+$ ]] || { echo "Unexpected sbatch response: $response" >&2; exit 1; }
printf 'job: %s\nrun-id: %s\n' "$job_id" "$run_id"
