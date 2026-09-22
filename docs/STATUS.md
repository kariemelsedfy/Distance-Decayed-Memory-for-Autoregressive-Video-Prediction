# Project status

**Updated:** 2026-09-22
**Active phase:** Phase 0 — foundations
**Active scope:** Track A only; Track B is deferred.
**Active branch:** `phase0/ddp-requeue`
**PR:** #20 (draft; issue #6)

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

## In progress

- PR #20 covers issue #6 and is ready to mark for review once CI is green.

## Next

1. **Owner decision:** Track A can use at most 2 pro6000 GPUs per user under the
   current QOS. Either plan the training budget around 2 GPUs or ask HPC staff
   for a raised limit or a reservation on `moose68`/`moose69`.
2. Mark PR #20 ready, review, merge, and close issue #6.
3. Move to the next Phase 0 item once #6 is merged; issue #7 (literature
   refresh) remains independent.

## Blockers

- The 2-pro6000 per-user ceiling constrains every Track A scaling assumption
  that expected up to 7 GPUs. Needs an owner decision (see Next, item 1).
- HPC home remains at its 25,600 MB hard limit; keep everything on scratch.

## Running jobs

None. Jobs `68179`, `68222`, `68224`, and `68228` completed; `68180` was
cancelled as unrunnable under the QOS cap.
