#!/usr/bin/env bash

set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
env_name="dd-memory"
python_version="3.11"
remote_workdir=""

usage() {
  cat <<'EOF'
Usage: scripts/hpc/build_env.sh [--name NAME] [--python VERSION] --workdir REMOTE_PATH

Creates/updates a conda environment in scratch, installs a PyTorch cu128 build
and the checked-out project, then writes pip freeze under environment/.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name) env_name="${2:?missing name}"; shift 2 ;;
    --python) python_version="${2:?missing version}"; shift 2 ;;
    --workdir) remote_workdir="${2:?missing path}"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ "$env_name" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "Invalid env name" >&2; exit 2; }
[[ "$python_version" =~ ^[0-9]+\.[0-9]+$ ]] || { echo "Invalid Python version" >&2; exit 2; }
[[ -n "$remote_workdir" ]] || { echo "--workdir is required" >&2; exit 2; }
printf -v quoted_workdir '%q' "$remote_workdir"

read -r -d '' remote_command <<EOF || true
set -euo pipefail
module load miniconda3
env_path="/mnt/hpc/tmp/\$USER/envs/$env_name"
export PIP_CACHE_DIR="/mnt/hpc/tmp/\$USER/cache/pip"
export XDG_CACHE_HOME="/mnt/hpc/tmp/\$USER/cache"
mkdir -p "\$PIP_CACHE_DIR" "\$XDG_CACHE_HOME"
if [[ ! -x "\$env_path/bin/python" ]]; then
  conda create -y -p "\$env_path" python=$python_version pip
fi
"\$env_path/bin/python" -m pip install --upgrade pip
"\$env_path/bin/python" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
"\$env_path/bin/python" -m pip install -e "$quoted_workdir[dev]"
mkdir -p $quoted_workdir/environment
"\$env_path/bin/python" -m pip freeze > $quoted_workdir/environment/hpc-$env_name.txt
"\$env_path/bin/python" -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
EOF

"$script_dir/remote.sh" "$remote_command"
