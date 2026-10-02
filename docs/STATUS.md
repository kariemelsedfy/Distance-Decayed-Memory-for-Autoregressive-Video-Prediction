# Project status

**Updated:** 2026-10-02
**Active phase:** Track A — A2 pilot trained and mostly evaluated; A3 gate pending
**Active scope:** Track A only; Track B is deferred.
**Active branch:** `phase2/a1-rollout-diagnostics` (stacked on `phase2/a1-results`)
**PRs:** #22–#28 merged into `main`; #29 (A1 results) is a draft.

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

- Merged PRs #22–#27; closed issues #9–#15 and #18.
- Issues #16 and #17: evaluation protocols P1/P2, PSNR/SSIM/LPIPS, bootstrap
  aggregation, the headline figure, and the six sanity checks, with cluster
  job scripts. The oracle is capped at the horizon (D-013). Merged as PR #28.
- **A1 base model complete** (job `69049`, run `a1-L-20260928T084605Z`):
  100k steps in 3 d 4 h 40 min, one attempt, no requeue. Val loss 0.00124
  (EMA), still falling slowly; one-clip sample PSNR 17.0 → 20.9 dB. Curves
  and samples fetched to `outputs/a1-L-20260928T084605Z/`
  (`scripts/figures/training_curves.py`).
- Added torchvision and LPIPS to `dd-memory-gpu-20260928` (torch unchanged at
  2.11.0+cu128); lock in `environment/hpc-dd-memory-gpu-20260928.txt`.

- A1 rollout diagnostics (job `69372`): rollouts follow actions (turning
  shift ±11.7 px/frame vs ±9–10 in real frames; noop holds still) and drift
  like a sampler (truth-vs-generated PSNR tracks seed-vs-seed). Wall-filled
  views lock in for many rollouts; the automatic `stuck` flag missed them.
- A2 trainer: activation checkpointing option; fixed a memory leak (policies
  stored *views* of each step's whole K/V buffer, `5747538`).
- **A2 pilot** (B-mid, seed 0, 5,000 steps, five policies, jobs
  `69392`–`69395`, `69430`, `69431`): all completed; ≈0.65 steps/s, ≈8.5
  GPU-hours per 20k-step run; loss levels off by ~2,500 steps; budgets held.
- **Pilot P1 on val (100 episodes):** `window` beats every memory-keeping
  policy in every gap bucket *and* on novel views (LPIPS 0.12 novel vs
  0.16–0.17); no policy shows a long-gap revisit benefit. Check 3
  (equivalence) exact; check 4 (budgets) passes; check 5 (memory used) fails:
  dropping `decay_continuous`'s memory hurts revisits and novel views alike;
  check 6 exact for `full`, 3.7% / 6.0% for decay / relic (D-011 by design).
  Details: `docs/EXPERIMENTS.md`, journal 2026-10-02.

## In progress

- Pilot oracle (`full`, H 256): jobs `69458` (100 episodes, ≈6.7 h) and
  `69460` (30 episodes, insurance). Gives checks 1 (window cliff) and 2.
- Owner decisions before A3 (below).

## Next

1. Finish the pilot: oracle results → checks 1 and 2 → write-up for the A3
   gate. **Recommendation so far: do not launch A3 as planned** until memory
   measurably helps revisits (options: equal full-fidelity recent window for
   all policies, longer A2, revisit-weighted loss, longer A1 clips).
2. **Owner decisions:** (a) the `full` oracle cannot be trained in A2 with 8
   streams (≈50 GB cache per stream) and cannot be evaluated at H 2,048
   (≈4 fp32 copies of the cache, ≈200 GB+) without reworking cache
   attention; (b) whether a 4–6% check-6 difference is acceptable.
3. Verify RELIC's discrete pattern against the paper (D-010 *(verify)*); issue
   #7 (literature refresh) remains independent.

## Blockers

- **Scratch purge (found 2026-09-28):** `/mnt/hpc/tmp` deletes files older
  than an unknown threshold (at least 42 days) by modification time. It broke
  two conda environments; the dataset (written 2026-09-23) and the A1
  checkpoints (41 GB, written 2026-09-28 to 2026-10-01) are exposed from
  early November. **Owner action:** ask Bowdoin
  HPC staff for the exact policy and a persistent location for data and
  checkpoints. Details: `docs/hpc/environments.md`.
- The working copy is inside OneDrive, which corrupted `.git` on 2026-10-01
  (renamed `refs/remotes/origin/phase2` to `phase2 2`, rolled a branch back
  one commit; nothing lost). Move the working copy out of OneDrive.
- The GPU ceiling is a recorded plan constraint (D-007). HPC home remains at
  its 25,600 MB hard limit; keep everything on scratch.

## Running jobs

- `69458`, `69460`: pilot oracle evaluations (above).
- Completed: A1 base model (`69049`), full splits (test frozen,
  `docs/DATASETS.md`), and the S/M/L size check.
