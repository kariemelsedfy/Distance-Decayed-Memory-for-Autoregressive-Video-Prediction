# Project status

**Updated:** 2026-09-23
**Active phase:** Track A — A0 check-in with the owner (pilot done); A1/A2
trainers smoke-tested
**Active scope:** Track A only; Track B is deferred.
**Active branch:** `phase2/a1-trainer` (merges the data stack and the model
stack)
**PRs (all draft, merge in this order):** #22 (issue #9) → #23 (#10) → #26
(#11); #24 (#12) → #25 (#13); then the trainer PR for #14/#15 on
`phase2/a1-trainer`.

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
- Implemented issue #10: the pose-based revisit detector (refined so a pause
  cannot count as a revisit, D-009), power-of-two gap buckets, and
  `scripts/data/spot_check_revisits.py` (report, tolerance sweep, and a
  pair grid). On 20 local episodes, 46% of frames are revisits and all seven
  buckets up to 2,048 are populated. 84 CPU tests pass.
- Issue #11: sharded writer/loader with manifests, verification, and
  node-local staging; chained CPU data jobs. **Pilot split** (jobs
  `68357`–`68359`): 200 episodes × 2,048 frames, manifest SHA-256
  `cd259cda…17a6`, 25.2 frames/s per core, 47% revisit frames, every gap bucket
  16–2,047 populated, all 944 scripted returns detected.
- Issue #12: all seven memory policies on one exact cell mechanism (D-010), 3D
  RoPE at fractional positions, and the Gate 1 density figure.
- Issue #13: pixel DiT (S/M/L = 58/172/457M parameters), flow matching,
  sampler, and cached rollout; streaming with a full cache equals the clip
  forward exactly (test).
- Issue #14 (trainer part): auto-resuming A1 DDP trainer with EMA, validation,
  samples, and requeue. **A1 smoke** `68374`: 511 frames/s on 2 GPUs at batch
  4/GPU (61.5 GB/GPU); 100k steps ≈ 28 h.
- Issue #15: streaming A2 trainer with a shared two-chunk local window
  (D-011). **A2 smoke** `68377`: 1.8 steps/s on 1 GPU with 8 streams, budget
  held; 20k steps ≈ 3 GPU-hours.
- Rebuilt the GPU environment as `dd-memory-gpu` after both conda environments
  lost library files (`docs/hpc/environments.md`); `build_env.sh` now copies
  files and verifies links.

## In progress

- Owner review of the draft PRs and the A0 check-in (pilot results, spot-check
  grid at `outputs/pilot/spot-check/spot_check.png` after fetching).

## Next (needs owner approval where marked)

1. **Approve** generating the full splits (#18): train 20,000 × 2,048 frames
   (~450 core-hours, ≈4.5 h on `main`), val 500, test 1,000 × 4,096; freeze and
   hash the test manifest.
2. **Approve** the A1 size check (S/M/L, short runs on 2 GPUs) and then the A1
   run (≈28 h for 100k steps on 2 GPUs).
3. Issue #16 evaluation harness (P1/P2, LPIPS/PSNR/SSIM, bootstrap CIs,
   headline plot) and #17 sanity-check suite — CPU work that can start now.
4. A2 pilot with a trained A1 model: fix the A2 step count and re-measure
   staleness (0.30–0.49 relative key change on the barely trained smoke model).
   **Gate:** owner approves the A3 sweep.
5. Verify RELIC's discrete pattern against the paper (D-010 *(verify)*); issue
   #7 (literature refresh) is independent.

## Blockers

None. The GPU ceiling is now a recorded plan constraint (D-007), not a blocker.
HPC home remains at its 25,600 MB hard limit; keep everything on scratch.

## Running jobs

Submitted 2026-09-23 ~15:47 UTC from checkout `eda4394` (owner-approved):

- Data (CPU, `main`): train `68388` → finalize `68389`; val `68390` →
  `68391`; test `68493` → `68494` (setup `68387`). Expected about 4.5 h.
- A1 size check (GPU): `68544` S and `68545` M on `mixed`, `68546` L on `gpu`,
  1 GPU each, 4,000 steps; start automatically after `68389` and `68391`.
  **Report to the owner before the full A1 run.**

After the test split finalizes, record its manifest SHA-256 in
`docs/DATASETS.md` (frozen).
