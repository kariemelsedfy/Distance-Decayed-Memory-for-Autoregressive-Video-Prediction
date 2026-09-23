# Project status

**Updated:** 2026-09-23
**Active phase:** Track A milestone A0 — data generation
**Active scope:** Track A only; Track B is deferred.
**Active branch:** `phase2/revisit-trajectories`
**PR:** draft for issue #9

## Done

- Read `PROJECT_PLAN.md` and `TRACK_A_PLAN.md` in full.
- Moved both plans into `docs/`.
- Extracted Appendices C and D verbatim and replaced them with links.
- Added the initial repository documentation and quality scaffold.
- Added canonical HPC run/status/sync/submit/monitor/fetch/environment scripts
  and matching Claude skills.
- Tested `hpc-run` and `hpc-status` successfully against `moosehead`.
- Verified the seven pro6000 cards (3 on `moose68`, 4 on `moose69`), 30-day
  partition maximum, outbound compute-node internet, bonded network links, and
  scratch capacity. Short Slurm probes `68124`–`68126` completed.
- Added and dry-ran `scripts/github/create_track_a_issues.py`, containing all 17
  issues, labels, and dependency links from `TRACK_A_PLAN.md` §12.
- Created and verified all 17 GitHub issues as #2–#18, with `phase-*`,
  `track-a`, `infra`, and `paper` labels and explicit dependency references.
- Initialized the empty GitHub repository with the original three-file baseline,
  then pushed the bootstrap work in small commits on `phase0/bootstrap`.
- Opened draft PR #1 and linked issues #2, #3, and #4 to the work.
- Marked PR #1 ready for review after its `quality` CI job passed.
- Passed 3 repository tests, Ruff, Black, shell syntax checks, TOML/YAML parsing,
  appendix verification, and a dry render of the Slurm template.
- Merged the initial bootstrap as PR #1.
- Built the Python 3.11 conda environment entirely in scratch through a CPU
  Slurm job, with conda, pip, build, and compiler caches also on scratch.
- Recorded the exact environment in `environment/hpc-dd-memory.txt`: PyTorch
  2.11.0+cu128, CUDA runtime 12.8, Triton 3.6.0, cuDNN 9.19.0.56, NCCL 2.28.9,
  and NumPy 2.4.6.
- Added a reproducible Blackwell probe and 15-minute Slurm job. Final job
  `68171` ran on `moose68` at commit `e97171c` and returned finite bf16 results
  from matrix multiplication, all four forced PyTorch SDPA backends, and
  compiled FlexAttention on `sm_120`.
- Documented the result and its limits in `docs/hpc/blackwell-kernels.md` and
  updated the general Bowdoin HPC reference and experiment registry.
- Merged PR #19 (closes issue #5).
- On `phase0/ddp-requeue` (draft PR #20): added DDP/NCCL all-reduce and
  checkpoint/resume probes, a distributed launcher, bounded Slurm scripts, and
  `scripts/hpc/submit_phase0_smoke.sh` (modes `one`, `node`, `multi`, `requeue`).
- Fixed the submit script passing literal quotes into `DD_MEMORY_CHECKOUT`
  (cause of failed job `68178`); commit `8fe148f`.
- Discovered the per-user QOS ceiling on both pro6000 partitions: **2 pro6000
  GPUs per user**, and on `gpu` also 4 CPUs and 40G. Resized the smoke modes to
  fit it and documented it in `docs/hpc/ddp-and-requeue.md`.
- All four Phase 0 probes passed: `68179` (1 GPU), `68222` (2 GPUs on moose69,
  24.3 GB/s), `68224` (1 GPU on each node, 11.0 GB/s over `bond0`), `68228`
  (checkpoint at step 4, requeue, resume, finish). Every DDP parameter delta and
  all-reduce error was exactly zero.
- Fixed three real defects found by those runs: literal quotes in the Slurm
  `--export` path, per-task GPU binding breaking NCCL's shared-memory transport,
  and a `SIGUSR1` handler registered after the torch import.
- Merged PR #20 (closes issue #6), then recorded the four-concurrent-GPU plan
  on the follow-up branch.
- Installed the pinned Memory Maze stack in scratch and completed CPU job
  `68321` on `main`. Headless EGL was forced to Mesa llvmpipe, and the verified
  median was 23.85 rendered 64×64 frames/s on one core.
- Confirmed the nine global observation keys, including binary 9×9
  `maze_layout`, 2D `agent_pos`, and unit-vector `agent_dir`; confirmed all six
  discrete actions and their order. Exact versions are in
  `environment/hpc-memory-maze.txt`.
- Merged PR #21 (closes issue #8).
- Implemented issue #9: A* grid navigation, a closed-loop controller for the
  six discrete actions, the scripted revisit planner (log-uniform gaps,
  pauses, detours, partial returns, 30% no-revisit episodes, 5% action noise),
  a toy maze simulator, the episode writer, and viewable previews (GIF, map,
  anchor/return pairs). 69 CPU tests pass. A local 20-episode run on the real
  environment completed 81 returns, all within 0.20 cell and 7.9° of the
  anchor pose. Conventions recorded as D-008.

## In progress

- Draft PR for issue #9 on `phase2/revisit-trajectories`; awaiting CI and
  owner review.

## Next

1. Merge the issue #9 PR, then start issue #10: the independent revisit
   detector and gap bucketing, using the new previews for spot checks.
2. Continue through #10 (revisit detector), #11 (writer/loader), and #18 (pilot
   split) to complete A0; the full split is not ready to launch yet.
3. In parallel, issue #12 (`MemoryPolicy` library and tests) and issue #13
   (pixel DiT, flow loss, sampler) need no cluster GPUs beyond short checks.
4. Add a sweep runner (keeps four single-GPU slots full, auto-resumes) before
   A2 — proposed as a new issue, not yet opened.
5. Issue #7 (literature refresh) remains independent.

## Blockers

None. The GPU ceiling is now a recorded plan constraint (D-007), not a blocker.
HPC home remains at its 25,600 MB hard limit; keep everything on scratch.

## Running jobs

None. Memory Maze probe `68321` completed. Setup attempts `68314` and `68315`
failed before the successful renderer configuration; diagnostic jobs
`68316`–`68320` are also finished. Queue-policy probes `68310`–`68313` were
cancelled after confirming four concurrent pro6000 GPUs.
