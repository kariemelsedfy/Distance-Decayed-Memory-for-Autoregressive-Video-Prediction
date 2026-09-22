#!/usr/bin/env bash

set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
run_id="${1:-}"
destination="${2:-artifacts}"

if [[ ! "$run_id" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "Usage: scripts/hpc/fetch.sh RUN_ID [LOCAL_DIRECTORY]" >&2
  exit 2
fi

archive_name="${run_id}-artifacts.tgz"
read -r -d '' remote_command <<EOF || true
set -euo pipefail
run_dir="/mnt/hpc/tmp/\$USER/dd-memory/runs/$run_id"
transfer_dir="/mnt/hpc/tmp/\$USER/dd-memory/transfers"
archive="\$transfer_dir/$archive_name"
[[ -d "\$run_dir" ]] || { echo "Run not found: $run_id" >&2; exit 1; }
mkdir -p "\$transfer_dir"
cd "\$run_dir"
find . -maxdepth 4 -type f \( -name 'metrics*' -o -name '*.csv' -o -name '*.json' -o -name '*.png' -o -name '*.pdf' -o -name '*.gif' -o -name '*.mp4' \) -print0 \
  | tar --null -czf "\$archive" --files-from=-
printf '%s\n' "\$archive"
EOF
remote_archive=$("$script_dir/remote.sh" "$remote_command" | tail -n 1)

set -a
# shellcheck disable=SC1090
source .env.hpc.local
set +a
mkdir -p "$destination"
local_archive="$destination/$archive_name"

if [[ -n "${BOWDOIN_HPC_PASSWORD:-}" ]]; then
  export BOWDOIN_HPC_REMOTE_ARCHIVE="$remote_archive"
  export BOWDOIN_HPC_LOCAL_ARCHIVE="$local_archive"
  expect <<'EOF'
set timeout 120
log_user 0
set target "$env(BOWDOIN_HPC_USER)@$env(BOWDOIN_HPC_HOST):$env(BOWDOIN_HPC_REMOTE_ARCHIVE)"
eval spawn -noecho [list scp -o StrictHostKeyChecking=accept-new -o PreferredAuthentications=password -o PubkeyAuthentication=no -o NumberOfPasswordPrompts=1 $target $env(BOWDOIN_HPC_LOCAL_ARCHIVE)]
expect {
  -re "(?i)password:" { send "$env(BOWDOIN_HPC_PASSWORD)\r"; log_user 1; exp_continue }
  -re {Permission denied|[Aa]uthentication failed} { exit 2 }
  timeout { exit 124 }
  eof { catch wait result; exit [lindex $result 3] }
}
EOF
else
  scp -o BatchMode=yes -o StrictHostKeyChecking=accept-new \
    "${BOWDOIN_HPC_USER}@${BOWDOIN_HPC_HOST}:$remote_archive" "$local_archive"
fi

tar -xzf "$local_archive" -C "$destination"
echo "Fetched small artifacts to $destination"
