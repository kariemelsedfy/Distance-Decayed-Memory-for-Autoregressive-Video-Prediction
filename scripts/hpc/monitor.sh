#!/usr/bin/env bash

set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
job_id="${1:-}"
poll_seconds="${HPC_POLL_SECONDS:-30}"
timeout_seconds="${HPC_MONITOR_TIMEOUT:-0}"

if [[ ! "$job_id" =~ ^[0-9]+([_.][0-9]+)?$ ]]; then
  echo "Usage: scripts/hpc/monitor.sh JOB_ID" >&2
  exit 2
fi

started_at=$(date +%s)
empty_replies=0
while true; do
  printf -v quoted_job '%q' "$job_id"
  read -r -d '' remote_command <<EOF || true
job_id=$quoted_job
sacct -X -j "\$job_id" -n -P -o JobIDRaw,JobName,Elapsed,State,AllocTRES,NodeList
log_path=\$(scontrol show job "\$job_id" 2>/dev/null | sed -n 's/.*StdOut=\([^ ]*\).*/\1/p' | head -n 1)
if [[ -n "\$log_path" && -r "\$log_path" ]]; then
  echo '--- log tail ---'
  tail -n 25 "\$log_path"
fi
EOF

  set +e
  reply=$("$script_dir/remote.sh" "$remote_command")
  status=$?
  set -e
  if [[ $status -ne 0 ]]; then
    echo "Status query failed; retrying." >&2
  elif [[ -z "$reply" ]]; then
    empty_replies=$((empty_replies + 1))
    echo "No sacct reply yet (attempt $empty_replies); retrying."
  else
    printf '%s\n' "$reply"
    state=$(printf '%s\n' "$reply" | awk -F'|' 'NF >= 4 && $1 !~ /\./ {print $4; exit}')
    case "$state" in
      COMPLETED*|FAILED*|CANCELLED*|TIMEOUT*|OUT_OF_MEMORY*|NODE_FAIL*|PREEMPTED*) exit 0 ;;
    esac
  fi

  if [[ $timeout_seconds -gt 0 && $(($(date +%s) - started_at)) -ge $timeout_seconds ]]; then
    echo "Monitor timeout reached; the job may still be running." >&2
    exit 124
  fi
  sleep "$poll_seconds"
done
