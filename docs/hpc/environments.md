# Cluster Python environments

## Current environments

| Name | Purpose | Built by | Notes |
|---|---|---|---|
| `dd-memory-gpu-20260928` | GPU training (PyTorch cu128) | `scripts/hpc/build_env.sh --name dd-memory-gpu-20260928` | Current. Fresh package cache, file mtimes stamped at build; dependencies only (jobs set `PYTHONPATH=<checkout>/src`). Rebuild within about six weeks of its build date. |
| `dd-memory-gpu` | Former GPU environment | 2026-09-23 build | **Do not use:** lost 86 library files to the scratch purge on 2026-09-24. |
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

## 2026-09-28: the cause is a scratch purge by file age

A1 job `69047` failed with the same missing `libbz2` in the environment rebuilt
on 2026-09-23. Its `lib/` directory was modified at 07:00:26 on 2026-09-24,
when no job of ours ran. Comparing conda's own file lists with the disk shows
whole packages removed (bzip2, libffi, libgcc/libstdc++, readline, tk,
libxcb, libuuid, …), and the oldest file that remains is dated 2026-08-13.

**`/mnt/hpc/tmp` deletes files whose modification time is older than some
threshold, apparently in a daily sweep around 07:00.** Conda extracts package
files with their original (often years-old) mtimes, so they are deleted the
day after a build; pip-installed files get the install time and survive. The
threshold is at least 42 days (files dated 2026-08-13 survived the
2026-09-24 sweep); the exact value must come from HPC staff.

Consequences:

- `build_env.sh` now uses a fresh package cache per build and stamps every
  installed file with the build time.
- **Data and checkpoints are at risk too.** The full splits were written on
  2026-09-23 and would be deleted in the first sweep after their mtime passes
  the threshold (early November at the earliest, if the threshold is about six
  weeks). Regenerating all splits takes about five hours, but checkpoints and
  results cannot be regenerated cheaply. Ask HPC staff for the policy and a
  persistent location before then; do not work around the purge without
  their agreement.
