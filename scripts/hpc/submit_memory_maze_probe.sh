#!/usr/bin/env bash

set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
remote_checkout=""
wall_time="00:30:00"

usage() {
  cat <<'EOF'
Usage: scripts/hpc/submit_memory_maze_probe.sh --workdir REMOTE_PATH [options]

Installs the pinned Memory Maze stack into a scratch conda environment and
runs the one-core software-EGL probe on the CPU-only main partition.

Options:
  --time HH:MM:SS   Wall time, at most 00:30:00 (default: 00:30:00)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workdir) remote_checkout="${2:?missing remote path}"; shift 2 ;;
    --time) wall_time="${2:?missing wall time}"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

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
  echo "Memory Maze probe jobs may not exceed 00:30:00" >&2
  exit 2
fi

timestamp=$(date -u +%Y%m%dT%H%M%SZ)
run_id="memory-maze-probe-$timestamp"
printf -v quoted_checkout '%q' "$remote_checkout"
printf -v quoted_run_id '%q' "$run_id"
printf -v quoted_time '%q' "$wall_time"
printf -v quoted_batch '%q' "$remote_checkout/slurm/memory-maze-probe.sbatch"
read -r -d '' remote_command <<EOF || true
set -euo pipefail
checkout=$quoted_checkout
test -f $quoted_batch
mkdir -p "/mnt/hpc/tmp/\$USER/dd-memory/logs"
sbatch --parsable \
  --time=$quoted_time \
  --export=ALL,DD_MEMORY_CHECKOUT=\$checkout,DD_MEMORY_RUN_ID=$quoted_run_id \
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
[[ "$job_id" =~ ^[0-9]+$ ]] || {
  echo "Unexpected sbatch response: $job_id" >&2
  exit 1
}
printf '%s\n' "$job_id"
printf 'run-id: %s\n' "$run_id" >&2
