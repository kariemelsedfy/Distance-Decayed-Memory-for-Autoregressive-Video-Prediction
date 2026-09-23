#!/bin/bash
#SBATCH --job-name=memmaze-probe
#SBATCH --partition=main
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/hpc/tmp/%u/dd-memory/logs/%x-%j.out

source /etc/profile
set -euo pipefail

: "${DD_MEMORY_CHECKOUT:?Set DD_MEMORY_CHECKOUT to the scratch checkout path}"
: "${DD_MEMORY_RUN_ID:?Set DD_MEMORY_RUN_ID to the probe run ID}"

module load miniconda3

env_path="/mnt/hpc/tmp/$USER/envs/dd-memory-memmaze-egl"
run_dir="/mnt/hpc/tmp/$USER/dd-memory/runs/$DD_MEMORY_RUN_ID"
export CONDA_ENVS_PATH="/mnt/hpc/tmp/$USER/envs"
export CONDA_PKGS_DIRS="/mnt/hpc/tmp/$USER/cache/conda/pkgs"
export PIP_CACHE_DIR="/mnt/hpc/tmp/$USER/cache/pip"
export XDG_CACHE_HOME="/mnt/hpc/tmp/$USER/cache"
export TMPDIR="/mnt/hpc/tmp/$USER/tmp"
export PYTHONNOUSERSITE=1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export LIBGL_ALWAYS_SOFTWARE=1
export MESA_SHADER_CACHE_DIR="/mnt/hpc/tmp/$USER/cache/mesa"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p \
  "$CONDA_ENVS_PATH" \
  "$CONDA_PKGS_DIRS" \
  "$PIP_CACHE_DIR" \
  "$XDG_CACHE_HOME" \
  "$MESA_SHADER_CACHE_DIR" \
  "$TMPDIR" \
  "$run_dir" \
  /mnt/hpc/tmp/$USER/dd-memory/logs
cd "$DD_MEMORY_CHECKOUT"

echo "Git SHA: $(git rev-parse HEAD)"
echo "Node: $(hostname)"
echo "Renderer: MUJOCO_GL=$MUJOCO_GL, software=$LIBGL_ALWAYS_SOFTWARE"
echo "CPU allocation: ${SLURM_CPUS_PER_TASK:-unknown}"

if [[ ! -x "$env_path/bin/python" ]]; then
  conda create -y -p "$env_path" python=3.11 pip
fi
"$env_path/bin/python" -m pip install --upgrade pip
"$env_path/bin/python" -m pip install ".[track-a]"

git rev-parse HEAD > "$run_dir/git-sha.txt"
"$env_path/bin/python" -m pip freeze > "$run_dir/pip-freeze.txt"
conda list -p "$env_path" --explicit > "$run_dir/conda-explicit.txt"

trap 'echo "USR1 received; CPU probe stopped before wall time"; exit 99' USR1

"$env_path/bin/python" scripts/data/probe_memory_maze.py \
  --steps 2000 \
  --warmup-steps 200 \
  --repeats 3 \
  --seed 0 \
  --expected-renderer egl \
  --output "$run_dir/results.json"
