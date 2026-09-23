# Cluster Python environments

## Current environments

| Name | Purpose | Built by | Notes |
|---|---|---|---|
| `dd-memory-gpu` | GPU training (PyTorch cu128) | `scripts/hpc/build_env.sh --name dd-memory-gpu` | Copy-based, dependencies only; jobs set `PYTHONPATH=<checkout>/src`. |
| `dd-memory-memmaze-egl` | Memory Maze data generation (CPU, software EGL) | `slurm/memmaze-data.sbatch` setup mode | Has broken library links (below) but everything data generation imports works; rebuild with `--copy` before relying on it further. |
| `dd-memory` | Phase 0 GPU environment | original `build_env.sh` | **Do not use:** broken `libbz2`, `libffi`, and `libstdc++` links. |

## 2026-09-23: environments lost library files

A1 smoke job `68368` failed at import: `torch.compile`'s backward partitioner
imports `networkx`, which imports `bz2`, whose `libbz2.so.1.0` symlink pointed
to a missing `libbz2.so.1.0.8`. Checking both environments found 86 dangling
links in `dd-memory` (libbz2, libffi, libstdc++, libxcb, …) and many in
`dd-memory-memmaze-egl`. The same files are also missing from the conda
package cache (`cache/conda/pkgs/bzip2-1.0.8-h5eee18b_6/lib/`), while that
directory was last modified at 04:39 on 2026-09-23, during the Memory Maze
setup attempts. The earlier empty `setuptools` metadata (job `68334`) is
probably the same event. The exact cause is unknown; data splits are not
affected (the pilot manifest verified afterwards).

Mitigation: `build_env.sh` now creates environments with `conda create --copy`
(no hard links into the shared package cache), installs dependencies only (no
editable install tied to one checkout), fails if any link is broken, and
imports `bz2`, `ctypes`, `lzma`, `sqlite3`, and torch's partitioner before
declaring success.
