# Dataset registry

Generated splits are listed under Memory Maze below.

## Memory Maze

- **Role:** primary Track A environment and dataset source.
- **Planned data:** generated 9×9 trajectories at 64×64, beginning with the 200-episode pilot.
- **Canonical source:** <https://github.com/jurgisp/memory-maze> and the
  `memory-maze` 1.0.3 package from PyPI.
- **License and terms:** Memory Maze is MIT licensed with no access gate. Its
  simulator dependencies are `dm-control` and MuJoCo (both Apache-2.0); Gym is
  MIT licensed. We generate trajectories ourselves rather than downloading the
  authors' approximately 100 GB offline dataset. Keep the source/version record
  with every generated split; confirm any publication-specific asset credit
  before redistributing rendered frames.
- **Pinned environment:** `memory-maze==1.0.3`, `dm-control==1.0.47`,
  `mujoco==3.14.0`, and `gym==0.26.2`. On Bowdoin `main`, set
  `MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl`, and
  `LIBGL_ALWAYS_SOFTWARE=1`; job `68321` identified the renderer as Mesa
  llvmpipe. OSMesa and Xvfb are not installed on the CPU nodes.
- **Cluster location:** `/mnt/hpc/tmp/$USER/dd-memory/data/memmaze9/<split>/`
  (sharded format, `src/distance_decayed_memory/data/shards.py`).
- **Seed ranges (disjoint):** train 0–19,999; val 1,000,000–1,000,499; test
  2,000,000–2,000,999 (4,096 frames); pilot 3,000,000–3,000,199.
- **Pilot split (2026-09-23):** 200 episodes × 2,048 frames in 8 shards of 25,
  generated at `9c4bece` by jobs `68357`–`68359`. Manifest SHA-256
  `cd259cda0e053052e067be3ced0b5ccaea9ec636ba90c6c98be3018ef07c17a6`.
  About 5 GB. Revisit labels use the D-009 detector at 0.3 cell / 15°.
- **Integrity:** freeze the final test manifest and record its SHA-256 before final evaluation.

## Dataset checklist

For every source, record its canonical URL, version, license, download terms,
generation code SHA, split seeds, storage location, and integrity hashes.
