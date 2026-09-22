#!/usr/bin/env bash

set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
env_name="dd-memory"
python_version="3.11"
remote_workdir=""
wall_time="00:30:00"
cpus=4
memory="16G"

usage() {
  cat <<'EOF'
Usage: scripts/hpc/build_env.sh [options] --workdir REMOTE_PATH

Creates/updates a conda environment in scratch, installs a PyTorch cu128 build
and the checked-out project, then writes pip freeze under environment/. The
build itself runs as a CPU job on the main partition, never on the login node.

Options:
  --name NAME       Environment name (default: dd-memory)
  --python VERSION  Python version (default: 3.11)
  --time HH:MM:SS   Slurm wall time (default: 00:30:00)
  --cpus N          CPU cores (default: 4)
  --mem SIZE        Slurm memory (default: 16G)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name) env_name="${2:?missing name}"; shift 2 ;;
    --python) python_version="${2:?missing version}"; shift 2 ;;
    --workdir) remote_workdir="${2:?missing path}"; shift 2 ;;
    --time) wall_time="${2:?missing wall time}"; shift 2 ;;
    --cpus) cpus="${2:?missing CPU count}"; shift 2 ;;
    --mem) memory="${2:?missing memory}"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ "$env_name" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "Invalid env name" >&2; exit 2; }
[[ "$python_version" =~ ^[0-9]+\.[0-9]+$ ]] || { echo "Invalid Python version" >&2; exit 2; }
[[ "$wall_time" =~ ^[0-9]{2}:[0-9]{2}:[0-9]{2}$ ]] || { echo "Invalid wall time" >&2; exit 2; }
[[ "$cpus" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid CPU count" >&2; exit 2; }
[[ "$memory" =~ ^[1-9][0-9]*[KMGTP]?$ ]] || { echo "Invalid memory" >&2; exit 2; }
[[ -n "$remote_workdir" ]] || { echo "--workdir is required" >&2; exit 2; }
printf -v quoted_workdir '%q' "$remote_workdir"

read -r -d '' build_script <<EOF || true
#!/usr/bin/env bash
set -euo pipefail
source /etc/profile >/dev/null 2>&1 || true
module load miniconda3
env_path="/mnt/hpc/tmp/\$USER/envs/$env_name"
workdir=$quoted_workdir
export CONDA_ENVS_PATH="/mnt/hpc/tmp/\$USER/envs"
export CONDA_PKGS_DIRS="/mnt/hpc/tmp/\$USER/cache/conda/pkgs"
export PIP_CACHE_DIR="/mnt/hpc/tmp/\$USER/cache/pip"
export XDG_CACHE_HOME="/mnt/hpc/tmp/\$USER/cache"
export TMPDIR="/mnt/hpc/tmp/\$USER/tmp"
export PYTHONNOUSERSITE=1
mkdir -p "\$CONDA_ENVS_PATH" "\$CONDA_PKGS_DIRS" "\$PIP_CACHE_DIR" "\$XDG_CACHE_HOME" "\$TMPDIR"
if [[ ! -x "\$env_path/bin/python" ]]; then
  conda create -y -p "\$env_path" python=$python_version pip
fi
"\$env_path/bin/python" -m pip install --upgrade pip
"\$env_path/bin/python" -m pip install torch --index-url https://download.pytorch.org/whl/cu128
"\$env_path/bin/python" -m pip install -e "\${workdir}[dev]"
mkdir -p "\$workdir/environment"
"\$env_path/bin/python" -m pip freeze > "\$workdir/environment/hpc-$env_name.txt"
"\$env_path/bin/python" -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
EOF

encoded_script=$(printf '%s' "$build_script" | base64 | tr -d '\n')
printf -v quoted_payload '%q' "$encoded_script"
read -r -d '' remote_command <<EOF || true
set -euo pipefail
job_dir="/mnt/hpc/tmp/\$USER/dd-memory/jobs"
job_script="\$job_dir/build-env-$env_name.sh"
mkdir -p "\$job_dir"
printf '%s' $quoted_payload | base64 -d > "\$job_script"
chmod 700 "\$job_script"
srun --partition=main --job-name=build-$env_name --cpus-per-task=$cpus --mem=$memory --time=$wall_time --kill-on-bad-exit=1 bash "\$job_script"
EOF

"$script_dir/remote.sh" --timeout 3600 "$remote_command"
