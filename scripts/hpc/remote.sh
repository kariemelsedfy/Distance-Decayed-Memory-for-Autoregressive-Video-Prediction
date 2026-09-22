#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/hpc/remote.sh [--env-file PATH] "REMOTE COMMAND"

Runs one non-interactive command on Bowdoin HPC. Password output is suppressed.
If BOWDOIN_HPC_PASSWORD is absent, SSH key authentication is attempted.
EOF
}

env_file=".env.hpc.local"
if [[ "${1:-}" == "--env-file" ]]; then
  [[ $# -ge 3 ]] || { usage >&2; exit 2; }
  env_file="$2"
  shift 2
fi

if [[ $# -ne 1 || "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  [[ $# -eq 1 ]] && exit 0
  exit 2
fi

remote_command="$1"
if [[ ! -f "$env_file" ]]; then
  echo "HPC env file not found: $env_file" >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "$env_file"
set +a

: "${BOWDOIN_HPC_HOST:?Missing BOWDOIN_HPC_HOST in $env_file}"
: "${BOWDOIN_HPC_USER:?Missing BOWDOIN_HPC_USER in $env_file}"

export BOWDOIN_HPC_REMOTE_COMMAND="$remote_command"

if [[ -n "${BOWDOIN_HPC_PASSWORD:-}" ]]; then
  command -v expect >/dev/null 2>&1 || {
    echo "The 'expect' command is required for password authentication." >&2
    exit 1
  }

  set +e
  expect <<'EOF'
set timeout 60
log_user 0
set sent_password 0
set target "$env(BOWDOIN_HPC_USER)@$env(BOWDOIN_HPC_HOST)"
set ssh_cmd [list ssh \
  -o StrictHostKeyChecking=accept-new \
  -o PreferredAuthentications=password \
  -o PubkeyAuthentication=no \
  -o NumberOfPasswordPrompts=1 \
  -o ConnectTimeout=20 \
  $target \
  $env(BOWDOIN_HPC_REMOTE_COMMAND)]

eval spawn -noecho $ssh_cmd
expect {
  -re "(?i)yes/no" {
    send "yes\r"
    exp_continue
  }
  -re "(?i)password:" {
    send "$env(BOWDOIN_HPC_PASSWORD)\r"
    set sent_password 1
    log_user 1
    exp_continue
  }
  -re {Permission denied|[Aa]uthentication failed} {
    puts stderr "Bowdoin HPC authentication failed."
    exit 2
  }
  timeout {
    puts stderr "Bowdoin HPC connection timed out."
    exit 124
  }
  eof {
    if {!$sent_password} { log_user 1 }
    catch wait result
    exit [lindex $result 3]
  }
}
EOF
  status=$?
  set -e
else
  set +e
  ssh \
    -o BatchMode=yes \
    -o StrictHostKeyChecking=accept-new \
    -o ConnectTimeout=20 \
    "${BOWDOIN_HPC_USER}@${BOWDOIN_HPC_HOST}" \
    "$remote_command"
  status=$?
  set -e
fi

if [[ $status -eq 255 ]]; then
  echo "SSH exited 255. Are you connected to the Bowdoin VPN?" >&2
fi
exit "$status"
