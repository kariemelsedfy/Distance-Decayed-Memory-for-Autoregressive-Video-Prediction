#!/usr/bin/env bash

set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
remote_checkout=""
split=""
first_seed=""
episodes_per_shard=""
shards=""
frames=2048
shard_time="02:00:00"
preview_episodes=2
after=""

usage() {
  cat <<'EOF_USAGE'
Usage: scripts/hpc/submit_memmaze_data.sh --workdir REMOTE_PATH --split NAME \
  --first-seed N --episodes-per-shard N --shards N [options]

Submits three chained CPU jobs on the main partition: environment setup, a
shard array (one CPU per shard), and a finalize job that writes the manifest,
verifies it, and runs the revisit spot check. Prints the three job IDs.

Options:
  --frames N          frames per episode (default 2048; 4096 for test)
  --shard-time T      wall time per shard task, HH:MM:SS (default 02:00:00)
  --preview-episodes N  previews for the first N episodes of shard 0 (default 2)
  --after JOBID       skip the setup job and start shards after JOBID succeeds
                      (use the setup job of a split submitted just before)
EOF_USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workdir) remote_checkout="${2:?}"; shift 2 ;;
    --split) split="${2:?}"; shift 2 ;;
    --first-seed) first_seed="${2:?}"; shift 2 ;;
    --episodes-per-shard) episodes_per_shard="${2:?}"; shift 2 ;;
    --shards) shards="${2:?}"; shift 2 ;;
    --frames) frames="${2:?}"; shift 2 ;;
    --shard-time) shard_time="${2:?}"; shift 2 ;;
    --preview-episodes) preview_episodes="${2:?}"; shift 2 ;;
    --after) after="${2:?}"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ "$remote_checkout" == /mnt/hpc/tmp/* ]] || {
  echo "--workdir must be an absolute scratch path under /mnt/hpc/tmp" >&2
  exit 2
}
[[ "$split" =~ ^[a-z0-9_-]+$ ]] || { echo "invalid --split" >&2; exit 2; }
for value in "$first_seed" "$episodes_per_shard" "$shards" "$frames" "$preview_episodes"; do
  [[ "$value" =~ ^[0-9]+$ ]] || { echo "numeric options are required" >&2; exit 2; }
done
((shards >= 1 && shards <= 1000)) || { echo "--shards must be 1-1000" >&2; exit 2; }
[[ -z "$after" || "$after" =~ ^[0-9]+$ ]] || { echo "invalid --after" >&2; exit 2; }
[[ "$shard_time" =~ ^[0-9]{2}:[0-5][0-9]:[0-5][0-9]$ ]] || {
  echo "invalid --shard-time" >&2
  exit 2
}

run_id="memmaze9-$split-$(date -u +%Y%m%dT%H%M%SZ)"
printf -v q_checkout '%q' "$remote_checkout"
printf -v q_batch '%q' "$remote_checkout/slurm/memmaze-data.sbatch"
exports="ALL,DD_MEMORY_CHECKOUT=$remote_checkout,DD_MEMORY_RUN_ID=$run_id"
exports+=",DD_SPLIT=$split,DD_FIRST_SEED=$first_seed"
exports+=",DD_EPISODES_PER_SHARD=$episodes_per_shard,DD_SHARDS=$shards"
exports+=",DD_FRAMES=$frames,DD_PREVIEW_EPISODES=$preview_episodes"
printf -v q_exports '%q' "$exports"
last_shard=$((shards - 1))

read -r -d '' remote_command <<EOF_REMOTE || true
set -euo pipefail
test -f $q_batch
active=\$(squeue -u "\$USER" -h -o '%j' | grep -E '^memmaze-(setup|final)?-?$split\$' || true)
if [[ -n "\$active" ]]; then
  echo "Refusing to submit: jobs for split $split are already queued:" >&2
  echo "\$active" >&2
  exit 3
fi
if [[ -n "$after" ]]; then
  setup=$after
else
  setup=\$(sbatch --parsable --time=00:30:00 --job-name=memmaze-setup-$split \
    --export=$q_exports,DD_MEMORY_MODE=setup $q_batch)
fi
array=\$(sbatch --parsable --time=$shard_time --array=0-$last_shard \
  --dependency=afterok:\$setup --job-name=memmaze-$split \
  --export=$q_exports,DD_MEMORY_MODE=shard $q_batch)
final=\$(sbatch --parsable --time=02:00:00 --mem=16G --job-name=memmaze-final-$split \
  --dependency=afterok:\$array \
  --export=$q_exports,DD_MEMORY_MODE=finalize $q_batch)
printf '%s %s %s\n' "\$setup" "\$array" "\$final"
EOF_REMOTE

set +e
response=$("$script_dir/remote.sh" --timeout 60 "$remote_command" 2>&1)
remote_status=$?
set -e
if [[ $remote_status -ne 0 ]]; then
  printf 'Remote submission failed (exit %s):\n%s\n' "$remote_status" "$response" >&2
  exit "$remote_status"
fi
ids=$(printf '%s\n' "$response" | tr -d '\r' | tail -n 1)
[[ "$ids" =~ ^[0-9]+\ [0-9]+\ [0-9]+$ ]] || {
  echo "Unexpected sbatch response: $response" >&2
  exit 1
}
printf 'setup array finalize: %s\n' "$ids"
printf 'run-id: %s\n' "$run_id"
