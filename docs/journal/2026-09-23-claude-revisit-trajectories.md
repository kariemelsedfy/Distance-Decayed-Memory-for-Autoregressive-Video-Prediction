# 2026-09-23 — Claude — Scripted revisit trajectories (issue #9)

## What

The Memory Maze agent can now be scripted to explore a maze and then
deliberately come back to an earlier view after a chosen delay. New code:

- `src/distance_decayed_memory/data/navigation.py`: A* on the maze grid, a
  closed-loop controller that turns Memory Maze's six discrete actions into
  "drive to this point and face this way", and a travel-time estimate.
- `src/distance_decayed_memory/data/revisit_script.py`: the planner. It
  explores, samples a revisit gap log-uniformly in [16, 2048] frames, leaves in
  time to arrive near the target, and records each event.
- `src/distance_decayed_memory/data/toy_maze.py`: a pixel-free simulator with
  the measured dynamics, used by the tests.
- `src/distance_decayed_memory/data/preview.py` and
  `scripts/data/generate_revisit_episodes.py`: write episodes in the §3.4
  per-episode format plus `preview.gif`, `map.png`, and `revisit_pairs.png`.

## Why

A0 needs episodes whose revisit gaps we control. The detector (#10), the
writer/loader (#11), and the pilot split (#18) all build on this generator.

## What I learned

- The shipped 9×9 task ends after 1,000 steps, shorter than our 2,048-frame
  episodes. The generator uses the pinned private builder with a longer limit
  (D-008).
- One turn command rotates about 17–18° after momentum dies out. The first
  heading tolerance (8°) made the controller oscillate forever in a toy case;
  it is now 10° with a 15° fallback.
- Very short gaps are hard to script: the anchor must be only a few frames old,
  and in 20 episodes only 3 of 81 completed returns fell in `[16, 32)`. The
  natural revisits that #10 detects should fill this bucket. The pilot must
  check.

## How to verify

- `pytest` (69 tests), `ruff check`, and `black --check`.
- On a desktop with Memory Maze installed:
  `MUJOCO_GL=glfw python scripts/data/generate_revisit_episodes.py
  --output-dir outputs/revisit-preview --episodes 2 --first-seed 4
  --preview-episodes 2`, then open the `revisit_pairs.png` and `preview.gif`
  files it writes. On the cluster use `MUJOCO_GL=egl` with the variables in
  `docs/hpc/memory-maze.md`.

## Plain-language explanation

We now have a robot script for the maze. It wanders around, remembers where it
was at some moment, and a chosen number of frames later walks back to that
exact spot and faces the same way. The pictures from the two moments match
almost exactly. Those matched pairs are what we will use to test whether a
video model remembers what it saw.

## Next

Issue #10 (revisit detector and gap bucketing), then #11 and the pilot (#18).
No cluster jobs were run for this issue.
