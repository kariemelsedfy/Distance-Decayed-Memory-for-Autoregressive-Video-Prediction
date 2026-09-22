#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/hpc/remote.sh [--env-file PATH] [--timeout SECONDS] "REMOTE COMMAND"

Runs one non-interactive command on Bowdoin HPC. Password output is suppressed.
If BOWDOIN_HPC_PASSWORD is absent, SSH key authentication is attempted.
EOF
}

env_file=".env.hpc.local"
ssh_timeout=60

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-file)
      env_file="${2:?missing value for --env-file}"
      shift 2
      ;;
    --timeout)
      ssh_timeout="${2:?missing value for --timeout}"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      break
      ;;
  esac
done

if [[ $# -ne 1 ]]; then
  usage >&2
  exit 2
fi
if [[ ! "$ssh_timeout" =~ ^[1-9][0-9]*$ ]]; then
  echo "--timeout must be a positive integer number of seconds." >&2
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
export BOWDOIN_HPC_SSH_TIMEOUT="$ssh_timeout"

if [[ -n "${BOWDOIN_HPC_PASSWORD:-}" ]]; then
  command -v expect >/dev/null 2>&1 || {
    echo "The 'expect' command is required for password authentication." >&2
    exit 1
  }

  set +e
  expect <<'EOF'
set timeout $env(BOWDOIN_HPC_SSH_TIMEOUT)
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
