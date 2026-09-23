# Dataset registry

No datasets have been downloaded or generated.

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
  `mujoco==3.14.0`, and `gym==0.26.2`; the CPU renderer is recorded by the
  environment probe.
- **Cluster location:** planned under `/mnt/hpc/tmp/$USER/dd-memory/data/memmaze9/`.
- **Integrity:** freeze the final test manifest and record its SHA-256 before final evaluation.

## Dataset checklist

For every source, record its canonical URL, version, license, download terms,
generation code SHA, split seeds, storage location, and integrity hashes.
