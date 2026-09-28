#!/bin/bash
#SBATCH --job-name=memmaze-data
#SBATCH --partition=main
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --output=/mnt/hpc/tmp/%u/dd-memory/logs/%x-%A_%a.out

# One script, three modes, chained by scripts/hpc/submit_memmaze_data.sh:
#   setup    install the checkout into the scratch environment (once)
#   shard    generate shard $SLURM_ARRAY_TASK_ID of the split
#   finalize write and verify the manifest, then run the spot check

source /etc/profile
set -euo pipefail

: "${DD_MEMORY_CHECKOUT:?}" "${DD_MEMORY_RUN_ID:?}" "${DD_MEMORY_MODE:?}"
: "${DD_SPLIT:?}" "${DD_FIRST_SEED:?}" "${DD_EPISODES_PER_SHARD:?}"
: "${DD_SHARDS:?}" "${DD_FRAMES:?}"

module load miniconda3

env_path="/mnt/hpc/tmp/$USER/envs/dd-memory-memmaze-egl"
run_dir="/mnt/hpc/tmp/$USER/dd-memory/runs/$DD_MEMORY_RUN_ID"
split_dir="/mnt/hpc/tmp/$USER/dd-memory/data/memmaze9/$DD_SPLIT"
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

mkdir -p "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$MESA_SHADER_CACHE_DIR" "$TMPDIR" \
  "$run_dir" "$split_dir"
cd "$DD_MEMORY_CHECKOUT"
python="$env_path/bin/python"

echo "Mode: $DD_MEMORY_MODE; git SHA: $(git rev-parse HEAD); node: $(hostname)"

case "$DD_MEMORY_MODE" in
  setup)
    [[ -x "$python" ]] || conda create -y -p "$env_path" python=3.11 pip
    # An empty setuptools metadata directory (seen 2026-09-23, job 68334)
    # makes pip crash before installing anything; remove it and reinstall.
    site=$("$python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
    for info in "$site"/setuptools-*.egg-info "$site"/setuptools-*.dist-info; do
      if [[ -d "$info" && -z "$(ls -A "$info")" ]]; then
        echo "Removing empty metadata directory $info"
        rmdir "$info"
        "$python" -m pip install --force-reinstall "setuptools>=75"
      fi
    done
    "$python" -m pip install ".[track-a]"
    git rev-parse HEAD > "$run_dir/git-sha.txt"
    "$python" -m pip freeze > "$run_dir/pip-freeze.txt"
    ;;
  shard)
    preview_args=()
    if [[ "${SLURM_ARRAY_TASK_ID:-}" == "0" && -n "${DD_PREVIEW_EPISODES:-}" ]]; then
      preview_args=(--preview-episodes "$DD_PREVIEW_EPISODES")
    fi
    "$python" scripts/data/generate_shard.py \
      --split-dir "$split_dir" \
      --split "$DD_SPLIT" \
      --episodes-per-shard "$DD_EPISODES_PER_SHARD" \
      --first-seed "$DD_FIRST_SEED" \
      --frames "$DD_FRAMES" \
      "${preview_args[@]}"
    ;;
  finalize)
    "$python" scripts/data/finalize_split.py \
      --split-dir "$split_dir" \
      --split "$DD_SPLIT" \
      --expected-shards "$DD_SHARDS" \
      --first-seed "$DD_FIRST_SEED" \
      --episodes-per-shard "$DD_EPISODES_PER_SHARD" | tee "$run_dir/finalize.json"
    cp "$split_dir/manifest.json" "$split_dir/manifest.sha256" "$run_dir/"
    "$python" - "$split_dir" > "$run_dir/generation-metrics.json" <<'PY'
import json, pathlib, statistics, sys
split = pathlib.Path(sys.argv[1])
shards = [json.loads(p.read_text()) for p in sorted(split.glob("shard-*[0-9]/shard.json"))]
print(json.dumps({
    "shards": len(shards),
    "environment_frames_per_second_median": statistics.median(
        s["environment_frames_per_second_median"] for s in shards),
    "shard_wall_seconds": [round(s["shard_wall_seconds"]) for s in shards],
    "hosts": sorted({s["hostname"] for s in shards}),
}, indent=2))
PY
    "$python" scripts/data/spot_check_revisits.py "$split_dir" \
      --output-dir "$run_dir/spot-check" > /dev/null
    if [[ -d "$split_dir/previews" ]]; then
      cp -r "$split_dir/previews" "$run_dir/previews"
    fi
    ;;
  *)
    echo "Unknown DD_MEMORY_MODE=$DD_MEMORY_MODE" >&2
    exit 2
    ;;
esac
