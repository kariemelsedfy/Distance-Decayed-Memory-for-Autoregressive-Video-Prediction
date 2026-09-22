#!/usr/bin/env bash

set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

read -r -d '' remote_command <<'EOF' || true
set -u
echo 'GPU nodes (configured and allocated GRES):'
sinfo -p gpu -N -h -O NodeHost:16,Gres:58,GresUsed:58,StateCompact:14
echo
echo 'My jobs:'
squeue -u "$USER" -o '%.18i %.24j %.10T %.10M %.4D %R'
echo
echo 'Home quota:'
quota -s 2>&1 || true
echo
echo 'Scratch filesystem:'
df -h /mnt/hpc/tmp 2>&1 || true
echo
echo 'My scratch usage (20 second limit):'
timeout 20s du -sh "/mnt/hpc/tmp/$USER" 2>&1 || echo 'scratch usage timed out or is unavailable'
EOF

"$script_dir/remote.sh" "$remote_command"
