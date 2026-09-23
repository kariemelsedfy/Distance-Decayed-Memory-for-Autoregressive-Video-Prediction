# 2026-09-23 — Codex — Headless Memory Maze on CPU

## What

Issue #8 is implemented and measured. I added a reproducible Memory Maze probe,
a bounded Slurm launcher for the CPU-only `main` partition, CPU tests for the
expected observations and actions, pinned dependencies, and license notes.

The successful run was job `68321` on `moose17`, using one CPU and no GPU. It
rendered 6,000 timed 64×64 frames in three repeats at a median 23.85 frames/s.
It also checked every observation shape and dtype and the six discrete actions.

## Why

This is the first link in the A0 data chain. The trajectory generator needs a
stable renderer plus exact layout and pose information before A* navigation,
revisit detection, sharding, or the 200-episode pilot can be trusted.

The speed result changes the compute estimate. The full 41M-frame training set
is about 480 core-hours before navigation and I/O, or roughly 15 hours at ideal
32-core scaling—not the earlier 45-minute estimate. The pilot must measure
end-to-end scaling before the full data job is scheduled.

## What I learned

The planned OSMesa backend is not available on Bowdoin's CPU nodes: there is no
system OSMesa or Xvfb, and conda-forge's current `mesalib` package did not ship
`libOSMesa.so`. Headless EGL does work on `main` when software rendering is
forced. The runtime identified itself as Mesa llvmpipe, so this did not borrow
a GPU invisibly.

The environment exposes `maze_layout`, `agent_pos`, and `agent_dir`, but not a
scalar heading or an action observation. Heading is derived from the unit
direction vector with `atan2`; actions are a separate six-valued input.

## How to verify

- Run the 14 CPU tests, Ruff, and Black.
- Use `scripts/hpc/submit_memory_maze_probe.sh` from a clean scratch checkout.
- Inspect
  `/mnt/hpc/tmp/$USER/dd-memory/runs/memory-maze-probe-20260923T084614Z/results.json`.
- The exact package set is in `environment/hpc-memory-maze.txt`; detailed
  results and failed setup attempts are in `docs/hpc/memory-maze.md`.

## Plain-language explanation

Memory Maze can now generate pictures on the cluster's ordinary processors
without reserving an expensive graphics card. One processor makes about 24
frames each second. The simulator also gives us the maze map, the agent's exact
position, and the direction it is facing, which is everything the next step
needs to deliberately walk away from a place and return later.

## Next

Merge the issue #8 PR, then implement issue #9: the A* scripted revisit
trajectory generator and toy-grid tests. Issues #12 and #13 can proceed in
parallel, and #7 remains independent. The expensive full dataset is not ready
to launch; A0 still requires the detector, writer/loader, and 200-episode pilot.
