# Project status

**Updated:** 2026-09-22
**Active phase:** Phase 0 — foundations
**Active scope:** Track A only; Track B is deferred.
**Active branch:** `phase0/blackwell-env`
**PR:** #19 (ready for review; CI green)

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

## In progress

- PR #19 is awaiting owner review and merge. Its local repository-wide checks
  and GitHub `quality` workflow passed.

## Next

1. Review and merge PR #19, which closes issue #5.
2. Continue Phase 0 with issue #6: run the 1-GPU, per-node, and multi-node DDP
   smoke tests, measure all-reduce bandwidth, and validate checkpoint/requeue.
3. Address issue #7 (literature refresh) independently when useful.

## Blockers

No code blocker. The HPC home directory now reports 25,600 MB used against its
25,600 MB hard limit. Keep caches, environments, logs, data, checkpoints, and
temporary build files on scratch; do not add anything to home.

## Running jobs

None. Final environment refresh job `68170` and Blackwell probe `68171`
completed; `scripts/hpc/status.sh` showed an empty personal queue afterward.
