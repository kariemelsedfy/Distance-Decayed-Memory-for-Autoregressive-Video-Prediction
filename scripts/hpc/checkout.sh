#!/usr/bin/env bash

set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
git_ref=""
repo_url="${DD_MEMORY_REPO_URL:-}"

usage() {
  cat <<'EOF'
Usage: scripts/hpc/checkout.sh --git-ref REF [--repo URL]

Clones an exact ref into a new scratch checkout and links persistent data,
weights, environments, logs, and checkpoints from /mnt/hpc/tmp/$USER/dd-memory.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --git-ref) git_ref="${2:?missing ref}"; shift 2 ;;
    --repo) repo_url="${2:?missing repository URL}"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$git_ref" ]] || { echo "--git-ref is required" >&2; exit 2; }
[[ -n "$repo_url" ]] || {
  echo "Provide --repo or set DD_MEMORY_REPO_URL." >&2
  exit 2
}

safe_ref="${git_ref//[^A-Za-z0-9._-]/-}"
checkout_name="${safe_ref}-$(date -u +%Y%m%dT%H%M%SZ)"
printf -v quoted_repo '%q' "$repo_url"
printf -v quoted_ref '%q' "$git_ref"
printf -v quoted_name '%q' "$checkout_name"

read -r -d '' remote_command <<EOF || true
set -euo pipefail
project_root="/mnt/hpc/tmp/\$USER/dd-memory"
destination="\$project_root/checkouts/$quoted_name"
mkdir -p "\$project_root"/{checkouts,data,weights,envs,logs,ckpt,runs,transfers}
git clone --filter=blob:none --no-checkout $quoted_repo "\$destination"
git -C "\$destination" fetch --depth=1 origin $quoted_ref
git -C "\$destination" checkout --detach FETCH_HEAD
ln -sfn "\$project_root/data" "\$destination/data"
ln -sfn "\$project_root/weights" "\$destination/weights"
mkdir -p "\$destination/.hpc"
ln -sfn "\$project_root/envs" "\$destination/.hpc/envs"
printf '%s\n' "\$destination"
EOF

"$script_dir/remote.sh" "$remote_command"
