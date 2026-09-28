#!/usr/bin/env bash

set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
remote_checkout=""
partition="mixed"
wall_time="12:00:00"
name="eval"
after=""

usage() {
  cat <<'EOF_USAGE'
Usage: scripts/hpc/submit_eval.sh --workdir REMOTE_PATH [options] -- EVALUATE_ARGS...

Submits one single-GPU evaluation. Everything after "--" is passed to
scripts/eval/evaluate.py (relative paths resolve from the checkout); the
script adds --output-dir as the run directory. Example:

  scripts/hpc/submit_eval.sh --workdir /mnt/hpc/tmp/$USER/dd-memory/checkouts/X \
    --name eval-decay -- --checkpoint /path/latest.pt \
    --split-dir /mnt/hpc/tmp/$USER/dd-memory/data/memmaze9/val --protocol p1

Options:
  --partition mixed|gpu  (default mixed)
  --time HH:MM:SS        wall time (default 12:00:00)
  --name NAME            run-id prefix (default eval)
  --after JOBID          start after JOBID succeeds
EOF_USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workdir) remote_checkout="${2:?}"; shift 2 ;;
    --partition) partition="${2:?}"; shift 2 ;;
    --time) wall_time="${2:?}"; shift 2 ;;
    --name) name="${2:?}"; shift 2 ;;
    --after) after="${2:?}"; shift 2 ;;
    --) shift; break ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ "$remote_checkout" == /mnt/hpc/tmp/* ]] || { echo "--workdir must be under /mnt/hpc/tmp" >&2; exit 2; }
[[ $# -gt 0 ]] || { echo "pass evaluate.py arguments after --" >&2; exit 2; }
[[ "$partition" == mixed || "$partition" == gpu ]] || { echo "invalid --partition" >&2; exit 2; }
[[ "$wall_time" =~ ^[0-9]{1,3}:[0-5][0-9]:[0-5][0-9]$ ]] || { echo "invalid --time" >&2; exit 2; }
[[ "$name" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "invalid --name" >&2; exit 2; }
[[ -z "$after" || "$after" =~ ^[0-9:]+$ ]] || { echo "invalid --after" >&2; exit 2; }

run_id="$name-$(date -u +%Y%m%dT%H%M%SZ)"
run_dir="/mnt/hpc/tmp/\$USER/dd-memory/runs/$run_id"
args_json=$(python3 -c 'import json, sys; print(json.dumps(sys.argv[1:]))' \
  "$@" --output-dir "__RUN_DIR__")
encoded=$(printf '%s' "$args_json" | base64 | tr -d '\n')
resources="--partition=$partition"
[[ "$partition" == gpu ]] && resources+=" --cpus-per-task=2 --mem=20G"
[[ -n "$after" ]] && resources+=" --dependency=afterok:$after"
printf -v q_batch '%q' "$remote_checkout/slurm/eval.sbatch"
printf -v q_exports '%q' "ALL,DD_MEMORY_CHECKOUT=$remote_checkout,DD_MEMORY_RUN_ID=$run_id"

read -r -d '' remote_command <<EOF_REMOTE || true
set -euo pipefail
test -f $q_batch
mkdir -p "$run_dir" "/mnt/hpc/tmp/\$USER/dd-memory/logs"
printf '%s' '$encoded' | base64 -d | sed "s|__RUN_DIR__|$run_dir|" > "$run_dir/eval-args.json"
cd $(printf '%q' "$remote_checkout")
sbatch --parsable --time=$wall_time $resources --job-name=$name \
  --export=$q_exports $q_batch
EOF_REMOTE

set +e
response=$("$script_dir/remote.sh" --timeout 60 "$remote_command" 2>&1)
status=$?
set -e
if [[ $status -ne 0 ]]; then
  printf 'Remote submission failed (exit %s):\n%s\n' "$status" "$response" >&2
  echo "Check squeue before retrying: the job may have been submitted." >&2
  exit "$status"
fi
job_id=$(printf '%s\n' "$response" | tr -d '\r' | tail -n 1)
[[ "$job_id" =~ ^[0-9]+$ ]] || { echo "Unexpected sbatch response: $response" >&2; exit 1; }
printf 'job: %s\nrun-id: %s\n' "$job_id" "$run_id"
